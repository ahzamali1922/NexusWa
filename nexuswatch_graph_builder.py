# nexuswatch_graph_builder.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
import time
from sklearn.preprocessing import StandardScaler

print("Starting Feature-Rich Graph Construction...")
t0 = time.time()

import os

# Paths - Updated to match your ingestion pipeline's folder structure
PREFIX = os.environ.get("DATASET_PREFIX", "HI-Medium") # Default to HI-Medium if not set
OUT_DIR = Path(f"./output/{PREFIX}")

accounts_file = OUT_DIR / "accounts.parquet"
trans_file = OUT_DIR / "transactions_clean.parquet"

# 1. Load Data
print("Loading Parquet files...")
accounts_df = pd.read_parquet(accounts_file)
trans_df = pd.read_parquet(trans_file)

# 2. Map Account Strings to Integer IDs (0 to N-1)
print("Mapping node IDs...")
unique_accounts = accounts_df['Account Number'].unique()
account_mapping = {acc: i for i, acc in enumerate(unique_accounts)}

# 3. Build Edge Index
print("Building edge index...")
src = trans_df['From Account'].map(account_mapping).values
dst = trans_df['To Account'].map(account_mapping).values
edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)

# 4. Define Node Labels (1 = Fraudulent, 0 = Legitimate)
print("Assigning node labels...")
num_nodes = len(unique_accounts)
node_labels = torch.zeros(num_nodes, dtype=torch.float)

fraud_mask = trans_df['Is Laundering'] == 1
fraud_accounts = set(trans_df[fraud_mask]['From Account']).union(set(trans_df[fraud_mask]['To Account']))
fraud_indices = [account_mapping[acc] for acc in fraud_accounts if acc in account_mapping]
node_labels[fraud_indices] = 1.0

# ==========================================
# 5. BUILD NODE FEATURES (x)
# ==========================================
print("Engineering Node Features...")

# A. Base Account Features (Align exactly with account_mapping)
ordered_accounts = pd.DataFrame({'Account Number': unique_accounts})
node_features_df = ordered_accounts.merge(accounts_df, on='Account Number', how='left')

# Clean Bank ID
node_features_df['Bank ID'] = pd.to_numeric(node_features_df['Bank ID'], errors='coerce').fillna(0)

# One-Hot Encode Entity Type
entity_dummies = pd.get_dummies(node_features_df['Entity Type'], prefix='Entity', dummy_na=True)

# B. Aggregated Transaction Features (Degree & Volume)
print("Aggregating transaction volumes per account...")
out_stats = trans_df.groupby('From Account')['Amount Paid'].agg(['count', 'sum']).rename(columns={'count': 'out_degree', 'sum': 'total_sent'})
in_stats = trans_df.groupby('To Account')['Amount Received'].agg(['count', 'sum']).rename(columns={'count': 'in_degree', 'sum': 'total_received'})

# Merge stats back to nodes
node_features_df = node_features_df.merge(out_stats, left_on='Account Number', right_index=True, how='left')
node_features_df = node_features_df.merge(in_stats, left_on='Account Number', right_index=True, how='left')

# Fill NaNs for accounts with no in/out transactions
node_features_df.fillna({'out_degree': 0, 'total_sent': 0, 'in_degree': 0, 'total_received': 0}, inplace=True)

# C. Log Transform & Scale
# Neural networks hate large unscaled numbers (like millions of dollars). We use log1p: log(1 + x)
node_features_df['total_sent'] = np.log1p(node_features_df['total_sent'])
node_features_df['total_received'] = np.log1p(node_features_df['total_received'])

numeric_features = node_features_df[['Bank ID', 'out_degree', 'total_sent', 'in_degree', 'total_received']]
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(numeric_features)

# Combine numeric and one-hot encoded features
x_numpy = np.hstack([scaled_numeric, entity_dummies.values])
x = torch.tensor(x_numpy, dtype=torch.float)
print(f"Node feature matrix shape: {x.shape}")

# ==========================================
# 6. BUILD EDGE FEATURES (edge_attr)
# ==========================================
print("Engineering Edge Features...")

# Log transform the specific transaction amount
trans_df['log_Amount'] = np.log1p(trans_df['Amount Paid'].fillna(0))

# One-Hot Encode Payment Format
format_dummies = pd.get_dummies(trans_df['Payment Format'], prefix='Fmt')

# Combine edge features (Amount, Hour, DayOfWeek + Payment Formats)
edge_features_df = pd.concat([
    trans_df[['log_Amount', 'Hour', 'DayOfWeek']].fillna(0), 
    format_dummies
], axis=1)

edge_scaler = StandardScaler()
scaled_edge_features = edge_scaler.fit_transform(edge_features_df)
edge_attr = torch.tensor(scaled_edge_features, dtype=torch.float)
print(f"Edge feature matrix shape: {edge_attr.shape}")

# ==========================================
# 7. BUILD AND SAVE PYG DATA
# ==========================================
print("Constructing PyG Data object...")
graph_data = Data(
    x=x,
    edge_index=edge_index, 
    edge_attr=edge_attr,
    y=node_labels
)

out_path = OUT_DIR / "nexuswatch_graph.pt"
torch.save(graph_data, out_path)

print(f"\n{'-'*50}")
print(f"SUCCESS!")
print(f"Nodes: {graph_data.num_nodes:,} (Features per node: {graph_data.num_node_features})")
print(f"Edges: {graph_data.num_edges:,} (Features per edge: {graph_data.num_edge_features})")
print(f"Graph saved to: {out_path}")
print(f"Total time: {time.time() - t0:.2f}s")
print(f"{'-'*50}")