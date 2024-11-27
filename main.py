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
import random
import concurrent.futures

# Configure logging
logging.basicConfig(
    filename='decklist_scraper.log',
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

def extract_card_names_from_soup(deck_page_soup):
    """
    Helper function to extract card names from a BeautifulSoup object of a deck page.
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

def fetch_page_data(base_url, page_num):
    """
    Fetches and processes a single page to extract decklist links.
    This function is designed to be executed in parallel.
    """
    # Initialize Selenium WebDriver (headless)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    # You can add the path to your ChromeDriver if it's not in PATH
    # chrome_options.binary_location = "/path/to/chrome"

    try:
        driver = webdriver.Chrome(options=chrome_options)
    except Exception as e:
        logging.error(f"Error initializing Selenium WebDriver: {e}")
        return []

    try:
        if page_num == 1:
            current_page_url = base_url
        else:
            current_page_url = f"{base_url}?page={page_num}"

        logging.info(f"Fetching page {page_num}: {current_page_url}")

        # Try fetching with requests
        main_page_soup = get_soup_with_requests(current_page_url)

        # If requests failed or content not found, fallback to Selenium
        if not main_page_soup or not main_page_soup.find_all('td'):
            logging.info(f"Falling back to Selenium for page: {current_page_url}")
            main_page_soup = get_soup_with_selenium(current_page_url, driver)

        # Extract decklist links from the current page
        deck_data = extract_decklist_links(main_page_soup)

        logging.info(f"Extracted {len(deck_data)} decklist links from page {page_num}.")

        return deck_data

    except Exception as e:
        logging.error(f"Error processing page {page_num}: {e}")
        return []

    finally:
        driver.quit()

def scrape_random_pages(base_url, target_deck_count=100, max_workers=20):
    """
    Randomly selects pages and extracts decklist links in parallel until the target number of unique decks is collected.
    """
    all_deck_data = []
    deck_urls_collected = set()

    # Fetch the first page to get total number of decks
    main_page_soup = get_soup_with_requests(base_url)

    if not main_page_soup or not main_page_soup.find_all('td'):
        logging.info(f"Falling back to Selenium for page: {base_url}")
        # Initialize a temporary driver for the first page
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        try:
            driver = webdriver.Chrome(options=chrome_options)
            main_page_soup = get_soup_with_selenium(base_url, driver)
        except Exception as e:
            logging.error(f"Error initializing Selenium WebDriver for the first page: {e}")
            return all_deck_data
        finally:
            driver.quit()

    total_decks = get_total_decks(main_page_soup)
    if total_decks is None:
        logging.warning("Could not extract total number of decks. Defaulting to 100.")
        total_decks = 100
    else:
        logging.info(f"Total decks found: {total_decks}")

    decks_per_page = 10  # As per the website's pagination
    total_pages = (total_decks + decks_per_page - 1) // decks_per_page

    # Determine the number of pages to sample
    pages_needed = (target_deck_count + decks_per_page - 1) // decks_per_page
    pages_to_sample = min(total_pages, pages_needed * 2)  # Double the pages to ensure enough decks

    # Generate a list of random page numbers
    random_page_numbers = random.sample(range(1, total_pages + 1), pages_to_sample)
    logging.info(f"Sampling {pages_to_sample} pages out of {total_pages} total pages.")

    # Use ThreadPoolExecutor to fetch pages in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all page fetching tasks
        future_to_page = {
            executor.submit(fetch_page_data, base_url, page_num): page_num
            for page_num in random_page_numbers
        }

        for future in concurrent.futures.as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                deck_data = future.result()
                for deck in deck_data:
                    if len(all_deck_data) >= target_deck_count:
                        break
                    deck_url = deck['url']
                    if deck_url not in deck_urls_collected:
                        all_deck_data.append(deck)
                        deck_urls_collected.add(deck_url)
            except Exception as e:
                logging.error(f"Error fetching page {page_num}: {e}")

            if len(all_deck_data) >= target_deck_count:
                break

    logging.info(f"Collected {len(all_deck_data)} unique decks from random pages.")
    return all_deck_data[:target_deck_count]  # Return only up to the target number of decks


def extract_card_names(deck_info):
    """
    Extracts card names from a deck URL. Returns a tuple of (deck_number, row_data) or None if failed.
    """
    idx, deck_info, commander_name = deck_info
    deck_url = deck_info['url']
    logging.info(f"Processing deck {idx}: {deck_url}")

    # Initialize Selenium WebDriver (headless)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    try:
        driver = webdriver.Chrome(options=chrome_options)
    except Exception as e:
        logging.error(f"Error initializing Selenium WebDriver for deck {idx}: {e}")
        return None

    try:
        # Try fetching with requests
        deck_page_soup = get_soup_with_requests(deck_url)

        # If requests failed or content not found, fallback to Selenium
        if not deck_page_soup or not deck_page_soup.find('a', string=lambda text: text and "Buy this decklist from Card Kingdom" in text):
            logging.info(f"Falling back to Selenium for deck {idx}.")
            deck_page_soup = get_soup_with_selenium(deck_url, driver)

        card_names = None
        if deck_page_soup:
            card_names = extract_card_names_from_soup(deck_page_soup)

        if card_names:
            # Exclude the commander
            card_names = [card for card in card_names if card != commander_name]

            # Prepare deck information
            row = [idx, commander_name, deck_info['price'], deck_info['tag'], deck_info['date']] + card_names
            logging.info(f"Successfully extracted deck {idx} with {len(card_names)} cards.")
            return row
        else:
            logging.warning(f"Skipping deck {idx} due to extraction failure.")
            return None

    except Exception as e:
        logging.error(f"Error processing deck {idx}: {e}")
        return None

    finally:
        driver.quit()

def main():
    main_url = 'https://edhrec.com/decks/atraxa-praetors-voice'
    output_csv = 'decklists.csv'
    commander_name = 'Atraxa, Praetors Voice'
    target_deck_count = 100  # Set your desired number of decks
    max_workers_pages = 20  # Number of parallel processes for page fetching
    max_workers_decks = 20  # Number of threads for card extraction

    # Scrape random pages to gather deck data
    deck_data = scrape_random_pages(main_url, target_deck_count=target_deck_count, max_workers=max_workers_pages)

    if not deck_data:
        logging.error("No decklist links found. Exiting.")
        sys.exit("No decklist links found. Please check the website structure or your network connection.")

    # Prepare CSV file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)

        # Write header row
        writer.writerow(
            ['Deck Number', 'Commander Name', 'Price', 'Tag', 'Date'] + [f'Card {i}' for i in range(1, 101)]
        )

        deck_count = 0

        # Use ThreadPoolExecutor for multithreading
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_decks) as executor:
            # Prepare the arguments for extract_card_names
            deck_infos = [
                (idx, deck_info, commander_name)
                for idx, deck_info in enumerate(deck_data, start=1)
            ]
            future_to_deck = {executor.submit(extract_card_names, deck_info): deck_info for deck_info in deck_infos}

            for future in concurrent.futures.as_completed(future_to_deck):
                result = future.result()
                if result:
                    writer.writerow(result)
                    deck_count += 1
                # Stop if we've reached the target number of decks
                if deck_count >= target_deck_count:
                    break

    logging.info(f"Scraping completed. {deck_count} decks saved to {output_csv}.")
    print(f"Scraping completed. {deck_count} decks saved to {output_csv}.")

if __name__ == "__main__":
    main()