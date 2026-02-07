import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from rdkit import Chem
import os
import tqdm

# --- Configuration ---
INPUT_CSV_PATH = 'raw_data/bace.csv'
OUTPUT_DIR = 'data/public/'

# Set a random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def graph_to_csvs(mol, graph_id, atom_type_map):
    """Converts an RDKit molecule object to node and edge data lists."""
    if mol is None:
        return None, None

    # Node features using the pre-computed atom_type_map
    num_atom_types = len(atom_type_map)
    nodes_data = []
    for i, atom in enumerate(mol.GetAtoms()):
        features = np.zeros(num_atom_types)
        atomic_num = atom.GetAtomicNum()
        if atomic_num in atom_type_map:
            features[atom_type_map[atomic_num]] = 1
        nodes_data.append([graph_id, i] + features.tolist())

    # Edge data
    edges_data = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        edges_data.append([graph_id, i, j, bond_type])
        edges_data.append([graph_id, j, i, bond_type]) # Undirected graph

    return nodes_data, edges_data


# --- Main Script ---
print("Loading raw data...")
df = pd.read_csv(INPUT_CSV_PATH)
essential_columns = ['mol', 'CID', 'Class']
df = df[essential_columns].copy()
df.rename(columns={'mol': 'smiles', 'CID': 'id', 'Class': 'target'}, inplace=True)

# --- NEW: Find all unique atom types across the entire dataset FIRST ---
print("Finding all unique atom types in the dataset...")
all_atom_types = set()
for smiles in tqdm.tqdm(df['smiles']):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        for atom in mol.GetAtoms():
            all_atom_types.add(atom.GetAtomicNum())

# Create a consistent mapping for all molecules
sorted_atom_types = sorted(list(all_atom_types))
atom_type_map = {atom_type: i for i, atom_type in enumerate(sorted_atom_types)}
print(f"Found {len(sorted_atom_types)} unique atom types: {sorted_atom_types}")

# Define column names based on the global atom types
num_node_features = len(atom_type_map)
node_cols = ['graph_id', 'node_id'] + [f'nf_{i}' for i in range(num_node_features)]
edge_cols = ['graph_id', 'src', 'dst', 'ef_0']
print(f"Node feature columns will be: {node_cols}")


# Split data
train_df, test_df = train_test_split(df, test_size=0.25, stratify=df['target'], random_state=RANDOM_STATE)

# --- Process and Save Data ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_train_nodes = []
all_train_edges = []
all_test_nodes = []
all_test_edges = []

print("Processing training data...")
for idx, row in tqdm.tqdm(train_df.iterrows(), total=len(train_df)):
    mol = Chem.MolFromSmiles(row['smiles'])
    nodes, edges = graph_to_csvs(mol, row['id'], atom_type_map) # Pass the map here
    if nodes:
        all_train_nodes.extend(nodes)
    if edges:
        all_train_edges.extend(edges)

print("Processing test data...")
for idx, row in tqdm.tqdm(test_df.iterrows(), total=len(test_df)):
    mol = Chem.MolFromSmiles(row['smiles'])
    nodes, edges = graph_to_csvs(mol, row['id'], atom_type_map) # And here
    if nodes:
        all_test_nodes.extend(nodes)
    if edges:
        all_test_edges.extend(edges)

# Save to CSVs
print("Saving CSV files...")
pd.DataFrame(all_train_nodes, columns=node_cols).to_csv(os.path.join(OUTPUT_DIR, 'train_nodes.csv'), index=False)
pd.DataFrame(all_train_edges, columns=edge_cols).to_csv(os.path.join(OUTPUT_DIR, 'train_edges.csv'), index=False)
pd.DataFrame(all_test_nodes, columns=node_cols).to_csv(os.path.join(OUTPUT_DIR, 'test_nodes.csv'), index=False)
pd.DataFrame(all_test_edges, columns=edge_cols).to_csv(os.path.join(OUTPUT_DIR, 'test_edges.csv'), index=False)

# Save labels and sample submission
train_df[['id', 'target']].to_csv(os.path.join(OUTPUT_DIR, 'train_labels.csv'), index=False)
test_df[['id']].to_csv(os.path.join(OUTPUT_DIR, 'sample_submission.csv'), index=False)
# This is the hidden ground truth for the organizer
test_df[['id', 'target']].to_csv(os.path.join(OUTPUT_DIR, 'test_labels.csv'), index=False)

print("\n Data preparation complete!")
print(f"Data saved in '{OUTPUT_DIR}'")