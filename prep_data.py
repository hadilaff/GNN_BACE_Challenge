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


# --- NEW: Scaffold Splitting & Class Imbalance ---
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold

def generate_scaffold(smiles, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def scaffold_split(df, smile_col='smiles', balanced=False, seed=42):
    """
    Split the dataset based on scaffolds.
    grouped data such that scaffolds in test set are different from those in train set.
    """
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        scaffold = generate_scaffold(row[smile_col])
        if scaffold:
            scaffolds[scaffold].append(idx)

    # Sort scaffolds by size (largest first) to pack them efficiently
    scaffold_sets = sorted(scaffolds.values(), key=len, reverse=True)

    train_inds, test_inds = [], []
    train_cutoff = 0.75 * len(df)

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            test_inds.extend(scaffold_set)
        else:
            train_inds.extend(scaffold_set)

    return df.loc[train_inds], df.loc[test_inds]

# 1. Scaffold Split
print("Performing Scaffold Split...")
train_df, test_df = scaffold_split(df)
print(f"Train/Test Split: {len(train_df)}/{len(test_df)}")

# 2. Induce Class Imbalance (Training Set Only)
# Goal: Reduce actives (target=1) to ~10% of the training set
print("Inducing Class Imbalance in Training Set...")
train_zeros = train_df[train_df['target'] == 0]
train_ones = train_df[train_df['target'] == 1]

# Calculate how many ones we want
# N_1 = N_0 / 9  (approx, to get 10% prevalence)
target_ones_count = int(len(train_zeros) / 9)
if len(train_ones) > target_ones_count:
    train_ones = train_ones.sample(n=target_ones_count, random_state=RANDOM_STATE)
    train_df = pd.concat([train_zeros, train_ones]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Downsampled actives. New Training Class Balance: \n{train_df['target'].value_counts(normalize=True)}")
else:
    print("Warning: Not enough actives to downsample to 10% (already low or calc issue).")

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