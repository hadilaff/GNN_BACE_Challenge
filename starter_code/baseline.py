import pandas as pd
import torch
import torch.nn.functional as F
# FIX 1: Updated the import for DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import os

# --- 1. Load Graphs from CSVs ---
def load_graphs_from_csv(nodes_path, edges_path, labels_path=None):
    """Loads graph data from node and edge CSV files."""
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    data_list = []
    graph_labels = {}
    
    if labels_path:
        labels_df = pd.read_csv(labels_path)
        graph_labels = dict(zip(labels_df['id'], labels_df['target']))

    graph_ids = nodes_df['graph_id'].unique()

    for graph_id in graph_ids:
        graph_nodes = nodes_df[nodes_df['graph_id'] == graph_id]
        graph_edges = edges_df[edges_df['graph_id'] == graph_id]

        node_features = graph_nodes.drop(columns=['graph_id', 'node_id']).values
        x = torch.tensor(node_features, dtype=torch.float)

        if not graph_edges.empty:
            edge_index = torch.tensor(graph_edges[['src', 'dst']].values, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(graph_edges[['ef_0']].values, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        if graph_id in graph_labels:
            data.y = torch.tensor([graph_labels[graph_id]], dtype=torch.long)
        
        data.graph_id = graph_id
        data_list.append(data)
        
    return data_list

# --- 2. GNN Model (same as before) ---
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

# --- 3. Main Training and Evaluation Script ---
if __name__ == '__main__':
    # Load data
    train_graphs = load_graphs_from_csv(
        '../data/public/train_nodes.csv',
        '../data/public/train_edges.csv',
        '../data/public/train_labels.csv'
    )
    
    train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)

    # FIX 2: Define the device variable once
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    in_channels = train_graphs[0].num_node_features
    model = GCN(in_channels=in_channels, hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    def train():
        model.train()
        for data in train_loader:
            # FIX 2: Use the 'device' variable
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch, data.edge_attr)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

    def evaluate(loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in loader:
                # FIX 2: Use the 'device' variable
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch, data.edge_attr)
                pred = out.argmax(dim=1)
                all_preds.append(pred.cpu())
                all_labels.append(data.y.cpu())
        
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        return f1_score(all_labels, all_preds, average='macro')

    # Train
    for epoch in range(1, 51):
        train()
        val_f1 = evaluate(val_loader)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Val F1: {val_f1:.4f}')

    # Predict on test set
    test_graphs = load_graphs_from_csv(
        '../data/public/test_nodes.csv',
        '../data/public/test_edges.csv'
    )
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    
    model.eval()
    all_preds = []
    all_ids = []
    with torch.no_grad():
        for data in test_loader:
            # FIX 2: Use the 'device' variable
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch, data.edge_attr)
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_ids.extend(data.graph_id)

    test_preds = torch.cat(all_preds, dim=0).numpy()
    submission_df = pd.DataFrame({'id': all_ids, 'target': test_preds})
    submission_df.to_csv('../submissions/inbox/submission.csv', index=False)
    print("Submission file created at '../submissions/inbox/submission.csv'")