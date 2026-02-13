
import pandas as pd
import numpy as np

def analyze_data():
    print("--- Analyzing Dataset Properties ---")
    
    # Load Labels
    train_labels = pd.read_csv('data/public/train_labels.csv')
    test_labels = pd.read_csv('data/public/test_labels.csv')
    
    # Class Imbalance
    train_counts = train_labels['target'].value_counts()
    test_counts = test_labels['target'].value_counts()
    
    print(f"\nTraining Set ({len(train_labels)} samples):")
    print(train_counts)
    print(f"Class Balance (Actives/Total): {train_counts.get(1, 0) / len(train_labels):.3f}")

    print(f"\nTest Set ({len(test_labels)} samples):")
    print(test_counts)
    print(f"Class Balance (Actives/Total): {test_counts.get(1, 0) / len(test_labels):.3f}")
    
    # Graph Sizes
    train_nodes = pd.read_csv('data/public/train_nodes.csv')
    train_edges = pd.read_csv('data/public/train_edges.csv')
    
    nodes_per_graph = train_nodes.groupby('graph_id').size()
    edges_per_graph = train_edges.groupby('graph_id').size()
    
    print("\nGraph Structure Stats (Training):")
    print(f"Avg Nodes: {nodes_per_graph.mean():.2f} +/- {nodes_per_graph.std():.2f}")
    print(f"Avg Edges: {edges_per_graph.mean():.2f} +/- {edges_per_graph.std():.2f}")
    print(f"Max Nodes: {nodes_per_graph.max()}")
    print(f"Min Nodes: {nodes_per_graph.min()}")

if __name__ == "__main__":
    analyze_data()
