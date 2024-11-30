import requests
from bs4 import BeautifulSoup
import csv
import urllib.parse
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
import sys
import re
from pyedhrec import EDHRec

# Configure logging
logging.basicConfig(
    filename='decklist_scraper_skyler.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_soup_with_requests(url):
    """
    Fetches the content of a URL using requests and returns a BeautifulSoup object.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return BeautifulSoup(response.text, 'html.parser')
    except requests.RequestException as e:
        logging.error(f"Requests error for URL {url}: {e}")
        return None

def get_soup_with_selenium(url, driver):
    """
    Fetches the content of a URL using Selenium and returns a BeautifulSoup object.
    """
    try:
        driver.get(url)
        time.sleep(3)  # Wait for JavaScript to load content
        html = driver.page_source
        return BeautifulSoup(html, 'html.parser')
    except Exception as e:
        logging.error(f"Selenium error for URL {url}: {e}")
        return None

def get_total_decks(main_page_soup):
    """
    Extracts the total number of decks from the page.
    """
    total_decks_span = main_page_soup.find('span', class_='react-bootstrap-table-pagination-total')
    if total_decks_span:
        total_text = total_decks_span.text.strip()
        # The text is something like "Showing 1 to 10 of 35211 decks"
        match = re.search(r'of\s+([\d,]+)\s+decks', total_text)
        if match:
            total_decks_str = match.group(1)
            total_decks = int(total_decks_str.replace(',', ''))
            return total_decks
    return None

def extract_decklist_links(main_page_soup):
    """
    Extracts and returns decklist URLs, along with their price, tag, and date.
    """
    deck_data = []
    if not main_page_soup:
        logging.error("Main page soup is None. Cannot extract deck links.")
        return deck_data

    # Find all rows (<tr>) that contain the deck data
    rows = main_page_soup.find_all('tr')
    for row in rows:
        try:
            # Extract the deck link
            link_tag = row.find('a', href=True, text="View Decklist")
            if not link_tag:
                continue
            deck_url = urllib.parse.urljoin('https://edhrec.com', link_tag['href'])

            # Extract the price
            price_tag = row.find_all('td')[1]
            price = price_tag.text.strip() if price_tag else "N/A"

            # Extract the tag
            tag_div = row.find('div', class_='d-grid')
            tag = tag_div.text.strip() if tag_div else "N/A"

            # Extract the date
            date_tag = row.find_all('td')[-1]
            date = date_tag.text.strip() if date_tag else "N/A"

            # Append extracted data to the list
            deck_data.append({
                'url': deck_url,
                'price': price,
                'tag': tag,
                'date': date
            })

        except Exception as e:
            logging.error(f"Error extracting data from row: {e}")
            continue

    logging.info(f"Extracted {len(deck_data)} decklist links with metadata.")
    return deck_data

def extract_card_names(deck_page_soup):
    """
    Extracts and returns a list of card names from a deck page soup.
    """
    if not deck_page_soup:
        return None

    # Find the <a> tag with the specific text
    a_tags = deck_page_soup.find_all('a', string=lambda text: text and "Buy this decklist from Card Kingdom" in text)
    if not a_tags:
        logging.warning("No 'Buy this decklist from Card Kingdom' link found.")
        return None

    card_kingdom_link = a_tags[0].get('href', '')
    if not card_kingdom_link:
        logging.warning("Card Kingdom link does not have an href attribute.")
        return None

    # Parse the URL to get the 'c' parameter
    parsed_url = urllib.parse.urlparse(card_kingdom_link)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    c_param = query_params.get('c', [])
    if not c_param:
        logging.warning("No 'c' parameter found in the Card Kingdom link.")
        return None

    # Decode the 'c' parameter
    decoded_c = urllib.parse.unquote(c_param[0])

    # Split by newline characters
    # Handle both '\r\n' and '\n' as separators
    if '\r\n' in decoded_c:
        card_entries = decoded_c.split('\r\n')
    else:
        card_entries = decoded_c.split('\n')

    # Remove leading '1 ' from each entry and strip whitespace
    card_names = []
    for entry in card_entries:
        entry = entry.strip()
        if entry.startswith('1 '):
            card_name = entry[2:].strip()
            card_names.append(card_name)
        else:
            # Handle unexpected formats
            card_names.append(entry)

    return card_names if card_names else None

def scrape_all_pages(base_url, driver):
    """
    Iterates through pages and extracts decklist links from all pages.
    """
    all_deck_data = []

    # Fetch the first page to get total number of decks
    main_page_soup = get_soup_with_requests(base_url)
    # Fallback to Selenium if requests fails
    if not main_page_soup or not main_page_soup.find_all('td'):
        logging.info(f"Falling back to Selenium for page: {base_url}")
        main_page_soup = get_soup_with_selenium(base_url, driver)

    total_decks = get_total_decks(main_page_soup)
    if total_decks is None:
        logging.warning("Could not extract total number of decks. Defaulting to 1000.")
        total_decks = 34000
    else:
        logging.info(f"Total decks found: {total_decks}")

    decks_per_page = 10  # As per the website's pagination
    total_pages = (total_decks + decks_per_page - 1) // decks_per_page

    # Limit to maximum of 1000 decks
    max_pages = min(total_pages, (34000 + decks_per_page - 1) // decks_per_page)

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            current_page_url = base_url
        else:
            current_page_url = base_url + f'?page={page_num}'

        logging.info(f"Fetching page {page_num}: {current_page_url}")

        # Fetch the page using requests first
        main_page_soup = get_soup_with_requests(current_page_url)

        # Fallback to Selenium if requests fails
        if not main_page_soup or not main_page_soup.find_all('td'):
            logging.info(f"Falling back to Selenium for page: {current_page_url}")
            main_page_soup = get_soup_with_selenium(current_page_url, driver)

        # Extract decklist links from the current page
        deck_data = extract_decklist_links(main_page_soup)
        all_deck_data.extend(deck_data)

        # Optional: Delay between page requests to be polite

        # Stop if we have collected 1000 decks
        if len(all_deck_data) >= 34000:
            break

    logging.info(f"Scraped data from {len(all_deck_data)} decks across all pages.")
    return all_deck_data[:34000]  # Return only up to 1000 decks


def extract_decklist_links(main_page_soup):
    deck_data = []
    if not main_page_soup:
        logging.error("Main page soup is None. Cannot extract deck links.")
        return deck_data

    # Find all rows (<tr>) that contain the deck data
    rows = main_page_soup.find_all('tr')
    for row in rows:
        try:
            # Extract the deck link
            link_tag = row.find('a', href=True, text="View Decklist")
            if not link_tag:
                continue
            deck_url = urllib.parse.urljoin('https://edhrec.com', link_tag['href'])

            # Extract the price
            price_tag = row.find_all('td')[1]
            price = price_tag.text.strip() if price_tag else "N/A"

            # Extract the tag
            tag_div = row.find('div', class_='d-grid')
            tag = tag_div.text.strip() if tag_div else "N/A"

            # Extract the date
            date_tag = row.find_all('td')[-1]
            date = date_tag.text.strip() if date_tag else "N/A"

            # Append extracted data to the list
            deck_data.append({
                'url': deck_url,
                'price': price,
                'tag': tag,
                'date': date
            })

        except Exception as e:
            logging.error(f"Error extracting data from row: {e}")
            continue

    logging.info(f"Extracted {len(deck_data)} decklist links with metadata.")
    return deck_data


def scrape_all_pages(commander_name, driver):
    """
    Iterates through pages and extracts decklist links from all pages.
    """
    all_deck_data = []
    
    base_url = 'https://edhrec.com/deckpreview/'
    
    edhrec = EDHRec()
    decks = edhrec.get_commander_decks(commander_name)
    
    decklist_table = decks["table"]

    total_decks = len(decklist_table)
    logging.info(f"Total decks found: {total_decks}")



    i = 0
    url_list = []

    for element in decks['table']:
        deck_data = []
        current_page_url = base_url + element['urlhash']

        url_list.append(current_page_url)
        

        try:
            price = element['price']

            tag_list = element['tags']

            date = element['savedate']

            # Append extracted data to the list
            deck_data.append({
                'url': current_page_url,
                'price': price,
                'tag': tag_list,
                'date': date
            })

        except Exception as e:
            logging.error(f"Error extracting data from row: {e}")
            continue

        # Extract decklist links from the current page
        all_deck_data.extend(deck_data)

        # Optional: Delay between page requests to be polite

        # Stop if we have collected 1000 decks


    logging.info(f"Scraped data from {len(all_deck_data)} decks across all pages.")
    return all_deck_data  # Return only up to 1000 decks

def main():
    main_url = 'https://edhrec.com/decks/atraxa-praetors-voice'
    output_csv = 'decklists_skyler.csv'
    commander_name = 'Atraxa, Praetors Voice'

    # Initialize Selenium WebDriver (Headless)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    try:
        driver = webdriver.Chrome(options=chrome_options)
    except Exception as e:
        logging.error(f"Error initializing Selenium WebDriver: {e}")
        sys.exit("Selenium WebDriver initialization failed. Please ensure ChromeDriver is installed and in PATH.")

    try:
        # Scrape all pages to gather deck data
        deck_data = scrape_all_pages(commander_name, driver)

        if not deck_data:
            logging.error("No decklist links found. Exiting.")
            driver.quit()
            sys.exit("No decklist links found. Please check the website structure or your network connection.")

        # Prepare CSV file
        with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)

            # Write header row
            writer.writerow(
                ['Deck Number', 'Commander Name', 'Price', 'Tag', 'Date'] + [f'Card {i}' for i in range(1, 101)])

            deck_count = 0
            for idx, deck_info in enumerate(deck_data, start=0):
                logging.info(f"Processing deck {idx}: {deck_info['url']}")
                # Try fetching with requests
                deck_page_soup = get_soup_with_requests(deck_info['url'])

                # If requests failed or content not found, fallback to Selenium
                if not deck_page_soup or not deck_page_soup.find('a', string=lambda
                        text: text and "Buy this decklist from Card Kingdom" in text):
                    logging.info(f"Falling back to Selenium for deck {idx}.")
                    deck_page_soup = get_soup_with_selenium(deck_info['url'], driver)

                card_names = extract_card_names(deck_page_soup)

                if card_names:
                    # Exclude the commander
                    card_names = [card for card in card_names if card != commander_name]

                    # Write deck information
                    row = [idx, commander_name, deck_info['price'], deck_info['tag'], deck_info['date']] + card_names
                    writer.writerow(row)
                    logging.info(f"Successfully extracted deck {idx} with {len(card_names)} cards.")
                else:
                    logging.warning(f"Skipping deck {idx} due to extraction failure.")

                # Optional: Delay between requests to be polite to the server
                 # 1-second delay


        logging.info(f"Scraping completed. {deck_count} decks saved to {output_csv}.")
        print(f"Scraping completed. {deck_count} decks saved to {output_csv}.")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
