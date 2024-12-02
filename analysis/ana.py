import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from tqdm import tqdm  # Import tqdm for the progress bar
import random
import community.community_louvain as louvain  # Import the Louvain community detection package


# Load the CSV into a pandas DataFrame
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    return df


# Generate the co-occurrence matrix for a random subset of decks
def generate_cooccurrence_matrix(df, n_random_decks):
    # Sample n_random_decks decks randomly
    sampled_decks = df.sample(n=n_random_decks, random_state=42)  # Adjust random_state for reproducibility

    cooccurrence = {}

    # Use tqdm to add a progress bar while iterating through the sampled decks
    for _, row in tqdm(sampled_decks.iterrows(), total=sampled_decks.shape[0], desc="Processing Sampled Decks",
                       unit="deck"):
        cards = [card for card in row[5:] if pd.notna(card)]  # Cards are from Card 1 to Card 100
        # Get all unique pairs of cards that appear together in this deck
        for card1, card2 in combinations(cards, 2):
            if card1 != card2:  # Ignore self-loops
                pair = tuple(
                    sorted([card1, card2]))  # Sort to avoid (card1, card2) and (card2, card1) being different edges
                if pair in cooccurrence:
                    cooccurrence[pair] += 1
                else:
                    cooccurrence[pair] = 1
    return cooccurrence


# Build the graph from the co-occurrence data
def build_graph(cooccurrence):
    G = nx.Graph()
    for (card1, card2), weight in cooccurrence.items():
        G.add_edge(card1, card2, weight=weight)
    return G


# Subgraph with n random cards (from the co-occurrence graph)
def get_random_subgraph(G, n):
    # Get a list of nodes (cards) from the graph
    nodes = list(G.nodes())

    # Randomly sample n nodes (cards)
    random_nodes = random.sample(nodes, n)

    # Create a subgraph with only the selected random nodes
    subgraph = G.subgraph(random_nodes)

    return subgraph

# Map each card to its Tag from the original dataframe
def map_cards_to_tags(df):
    card_to_tag = {}
    for _, row in df.iterrows():
        tag = row["Tag"]
        cards = [card for card in row[5:] if pd.notna(card)]  # Cards are from Card 1 to Card 100
        for card in cards:
            if card not in card_to_tag:
                card_to_tag[card] = []
            card_to_tag[card].append(tag)
    return card_to_tag



# Community detection with Louvain method
def community_detection_louvain(G, resolution=1.4):
    # Apply Louvain community detection
    partition = louvain.best_partition(G, resolution=resolution)

    # Count the number of communities detected
    num_communities = len(set(partition.values()))
    print(f"Number of communities found: {num_communities}")

    # Group cards by community
    communities = {}
    for card, community in partition.items():
        if community not in communities:
            communities[community] = []
        communities[community].append(card)

    # Show the size of each community and the cards within them
    print("Top 5 largest communities:")
    for community, cards in sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        print(f"Community {community}: {len(cards)} cards")
        print(f"Cards: {', '.join(cards[:10])}...")  # Show only the first 10 cards for brevity

    return partition, communities



# Export data to Gephi-compatible CSV files (nodes and edges)
def export_to_gephi(G, output_dir):
    # Export nodes (cards) to a CSV file
    nodes = pd.DataFrame(G.nodes(), columns=['Id'])
    nodes['Label'] = nodes['Id']

    # Save to nodes.csv
    nodes.to_csv(f'{output_dir}/nodes.csv', index=False)

    # Export edges (pairs of cards) to a CSV file
    edges = pd.DataFrame(G.edges(data=True), columns=['Source', 'Target', 'Weight'])

    # Save to edges.csv
    edges.to_csv(f'{output_dir}/edges.csv', index=False)

    print(f"Gephi files saved to {output_dir}/nodes.csv and {output_dir}/edges.csv")


# Analysis: Basic insights
def analyze_network(G):
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {np.mean([deg for _, deg in G.degree()])}")

    # Degree centrality (most connected nodes)
    degree_centrality = nx.degree_centrality(G)
    print("Top 5 most connected cards (by degree centrality):")
    for card, centrality in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{card}: {centrality}")

    # Clustering coefficient (measuring clustering in the network)
    clustering_coefficient = nx.clustering(G)
    print(f"Average clustering coefficient: {np.mean(list(clustering_coefficient.values()))}")

    # Betweenness centrality (nodes that act as bridges)
    betweenness_centrality = nx.betweenness_centrality(G)
    print("Top 5 cards with highest betweenness centrality:")
    for card, centrality in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{card}: {centrality}")


# Visualize the network
def visualize_network(G, num_nodes=100):
    plt.figure(figsize=(12, 12))

    # Extract the largest connected component for better visualization
    largest_component = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_component)

    pos = nx.spring_layout(subgraph, seed=42)  # Use spring layout for positioning nodes
    plt.title("Card Co-Occurrence Network")
    nx.draw(subgraph, pos, with_labels=True, node_size=30, font_size=8, alpha=0.7, edge_color="gray")
    plt.show()
    # Export gefx file
    nx.write_gexf(G, 'card_cooccurrence_network_100.gexf')


def main(csv_file, output_dir, n_random_decks=100, n_random_cards=100, resolution=1.4):
    # Load the data
    df = load_data(csv_file)

    # Generate the card-to-tag mapping
    card_to_tag = map_cards_to_tags(df)

    # Generate co-occurrence data with a progress bar for a random subset of decks
    cooccurrence = generate_cooccurrence_matrix(df, n_random_decks)

    # Build the graph
    G = build_graph(cooccurrence)

    # Export to Gephi
    export_to_gephi(G, output_dir)

    # Analyze the network
    analyze_network(G)

    # Community detection with Louvain method
    partition, communities = community_detection_louvain(G, resolution=resolution)

    # Create a DataFrame for communities and include the Tag values
    community_data = []
    for community, cards in communities.items():
        for card in cards:
            # Get the tags for the card (join them in case there are multiple tags)
            tags = ", ".join(set(card_to_tag.get(card, [])))
            community_data.append({"Community": community, "Card": card, "Tags": tags})

    community_df = pd.DataFrame(community_data)
    community_df.to_csv(f"{output_dir}/communities_with_tags.csv", index=False)
    print(f"Communities with tags saved to {output_dir}/communities_with_tags.csv")

    # Visualize the network (Optional)
    visualize_network(G, num_nodes=100)



# Run the main function
if __name__ == "__main__":
    csv_file = '../data/decklists_firkraag.csv'  # Replace with the path to your CSV file
    output_dir = 'ana_output'  # Directory to save Gephi files
    n_random_decks = 10  # Number of random decks to sample
    n_random_cards = 100  # Set the number of random cards to consider for community detection
    main(csv_file, output_dir, n_random_decks, n_random_cards)
