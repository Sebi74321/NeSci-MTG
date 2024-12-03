import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from tqdm import tqdm  # Import tqdm for the progress bar
import random
import community.community_louvain as louvain  # Import the Louvain community detection package


# Load the CSV into a pandas DataFrame
def load_data(csv_files):
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    return pd.concat(dfs, ignore_index=True)


# Generate the co-occurrence matrix for a random subset of decks
def generate_cooccurrence_matrix(df, n_random_decks):
    sampled_decks = df.sample(n=n_random_decks, random_state=42)
    cooccurrence = {}
    card_counts = {}

    for _, row in tqdm(sampled_decks.iterrows(), total=sampled_decks.shape[0], desc="Processing Sampled Decks", unit="deck"):
        cards = [card for card in row[5:] if pd.notna(card)]
        for card in cards:
            if card in card_counts:
                card_counts[card] += 1
            else:
                card_counts[card] = 1
        for card1, card2 in combinations(cards, 2):
            if card1 != card2:
                pair = tuple(sorted([card1, card2]))
                if pair in cooccurrence:
                    cooccurrence[pair] += 1
                else:
                    cooccurrence[pair] = 1

    filtered_card_counts = {card: count for card, count in card_counts.items() if count >= 3}
    filtered_cooccurrence = {pair: weight for pair, weight in cooccurrence.items() if pair[0] in filtered_card_counts and pair[1] in filtered_card_counts}

    return filtered_cooccurrence, filtered_card_counts



# Build the graph from the co-occurrence data
def build_graph(cooccurrence):
    G = nx.Graph()
    for (card1, card2), weight in cooccurrence.items():
        if weight > 3:
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
def export_to_gephi(G, card_counts, output_dir):
    # Export nodes (cards) to a CSV file
    nodes = pd.DataFrame(G.nodes(), columns=['Id'])
    nodes['Label'] = nodes['Id']
    nodes['Count'] = nodes['Id'].map(card_counts)  # Add the count of each card

    # Save to nodes.csv
    nodes.to_csv(f'{output_dir}/nodes_1000shared.csv', index=False)

    # Export edges (pairs of cards) to a CSV file
    edges = pd.DataFrame(G.edges(data=True), columns=['Source', 'Target', 'Weight'])

    # Save to edges.csv
    edges.to_csv(f'{output_dir}/edges_1000shared.csv', index=False)

    nx.write_gexf(G, output_dir + '/card_cooccurrence_network_1000shared.gexf')


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

def main(csv_files, output_dir, n_random_decks_per_file=1000, n_random_cards=100, resolution=1.4):
    # Load the data from multiple CSV files
    df = load_data(csv_files)

    # Generate the card-to-tag mapping
    card_to_tag = map_cards_to_tags(df)

    # Generate co-occurrence data with a progress bar for a random subset of decks
    cooccurrence, card_counts = generate_cooccurrence_matrix(df, n_random_decks_per_file * len(csv_files))

    # Build the graph
    G = build_graph(cooccurrence)

    # Export to Gephi
    export_to_gephi(G, card_counts, output_dir)

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
            community_data.append({"Community": community, "Id": card, "Tags": tags})

    community_df = pd.DataFrame(community_data)
    community_df.to_csv(f"{output_dir}/communities_with_tags.csv", index=False)
    print(f"Communities with tags saved to {output_dir}/communities_with_tags.csv")

    # Visualize the network (Optional)
    visualize_network(G, num_nodes=100)


# Run the main function
if __name__ == "__main__":
    csv_files = ['../data/decklists_firkraag.csv', '../data/decklists_ghyrson.csv','../data/decklists_aegar.csv','../data/decklists_bria.csv','../data/decklists_jhoiraWC.csv','../data/decklists_neera.csv','../data/decklists_river_song.csv','../data/decklists_veyran.csv','../data/decklists_yusri.csv']  # Replace with the paths to your CSV files
    output_dir = 'ana_output'  # Directory to save Gephi files
    n_random_decks = 1000 # Number of random decks to sample
    n_random_cards = 100  # Set the number of random cards to consider for community detection
    main(csv_files, output_dir, n_random_decks, n_random_cards)
