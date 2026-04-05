# nexuswatch_graph_builder.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
import time
from sklearn.preprocessing import StandardScaler
import os
import gc

print("Starting Feature-Rich Graph Construction...")
t0 = time.time()

PREFIX = os.environ.get("DATASET_PREFIX", "HI-Medium")
OUT_DIR = Path(f"./output/{PREFIX}")

accounts_file = OUT_DIR / "accounts.parquet"
trans_file = OUT_DIR / "transactions_clean.parquet"

# 1. Load Data
print("Loading Parquet files...")
accounts_df = pd.read_parquet(accounts_file)
trans_df = pd.read_parquet(trans_file)

# 2. Map Account Strings to Integer IDs
print("Mapping node IDs...")
unique_accounts = accounts_df['Account Number'].unique()
account_mapping = {acc: i for i, acc in enumerate(unique_accounts)}

# 3. Build Edge Index
print("Building edge index...")
src = trans_df['From Account'].map(account_mapping).values
dst = trans_df['To Account'].map(account_mapping).values
edge_index = torch.tensor(np.vstack((src, dst)), dtype=torch.long)

# 4. Define Node Labels
print("Assigning node labels...")
num_nodes = len(unique_accounts)
node_labels = torch.zeros(num_nodes, dtype=torch.float)

fraud_mask = trans_df['Is Laundering'] == 1
fraud_accounts = set(trans_df[fraud_mask]['From Account']).union(set(trans_df[fraud_mask]['To Account']))
fraud_indices = [account_mapping[acc] for acc in fraud_accounts if acc in account_mapping]
node_labels[fraud_indices] = 1.0

# ==========================================
# 5. DATA SPLITTING (TRAIN/VAL/TEST MASKS)
# ==========================================
print("Generating Train/Val/Test masks (70/15/15)...")
indices = np.arange(num_nodes)
np.random.shuffle(indices)

train_end = int(0.70 * num_nodes)
val_end = int(0.85 * num_nodes)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[indices[:train_end]] = True
val_mask[indices[train_end:val_end]] = True
test_mask[indices[val_end:]] = True

# ==========================================
# 6. BUILD NODE FEATURES (x)
# ==========================================
print("Engineering Node Features...")
ordered_accounts = pd.DataFrame({'Account Number': unique_accounts})
node_features_df = ordered_accounts.merge(accounts_df, on='Account Number', how='left')
entity_dummies = pd.get_dummies(node_features_df['Entity Type'], prefix='Entity', dummy_na=True)

out_stats = trans_df.groupby('From Account')['Amount Paid'].agg(['count', 'sum']).rename(columns={'count': 'out_degree', 'sum': 'total_sent'})
in_stats = trans_df.groupby('To Account')['Amount Received'].agg(['count', 'sum']).rename(columns={'count': 'in_degree', 'sum': 'total_received'})

node_features_df = node_features_df.merge(out_stats, left_on='Account Number', right_index=True, how='left')
node_features_df = node_features_df.merge(in_stats, left_on='Account Number', right_index=True, how='left')
node_features_df.fillna({'out_degree': 0, 'total_sent': 0, 'in_degree': 0, 'total_received': 0}, inplace=True)

node_features_df['total_sent'] = np.log1p(node_features_df['total_sent'])
node_features_df['total_received'] = np.log1p(node_features_df['total_received'])

numeric_features = node_features_df[['out_degree', 'total_sent', 'in_degree', 'total_received']]
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(numeric_features)

x_numpy = np.hstack([scaled_numeric, entity_dummies.values])
x = torch.tensor(x_numpy, dtype=torch.float)

# Free up memory before heavy edge engineering
del ordered_accounts
del node_features_df
del entity_dummies
del numeric_features
gc.collect()

# ==========================================
# 7. BUILD EDGE FEATURES (edge_attr) - MEMORY OPTIMIZED
# ==========================================
print("Engineering Edge Features...")

# 1. Log transform & downcast to float32 to save 50% memory
trans_df['log_Amount'] = np.log1p(trans_df['Amount Paid'].fillna(0)).astype(np.float32)

# 2. One-Hot Encode Payment Format (cast directly to float32)
format_dummies = pd.get_dummies(trans_df['Payment Format'], prefix='Fmt', dtype=np.float32)

# 3. Combine edge features (downcast Hour/DayOfWeek)
edge_features_df = pd.concat([
    trans_df[['log_Amount', 'Hour', 'DayOfWeek']].fillna(0).astype(np.float32), 
    format_dummies
], axis=1)

# 🔥 CRITICAL: Delete massive dataframes we no longer need and clear RAM
del trans_df
del format_dummies
gc.collect() 
print("  -> Cleared raw transaction data from RAM.")

# 4. Scale features
print("  -> Scaling edge features...")
edge_scaler = StandardScaler()
# StandardScaler returns float64 by default, instantly downcast back to float32
scaled_edge_features = edge_scaler.fit_transform(edge_features_df).astype(np.float32)

# 🔥 CRITICAL: Delete the unscaled dataframe
del edge_features_df
gc.collect()

# 5. Convert to tensor
edge_attr = torch.tensor(scaled_edge_features, dtype=torch.float)

# 🔥 CRITICAL: Delete the numpy array
del scaled_edge_features
gc.collect()

print(f"Edge feature matrix shape: {edge_attr.shape}")

# ==========================================
# 8. BUILD AND SAVE PYG DATA
# ==========================================
print("Constructing PyG Data object...")
graph_data = Data(
    x=x,
    edge_index=edge_index, 
    edge_attr=edge_attr,
    y=node_labels,
    train_mask=train_mask,
    val_mask=val_mask,
    test_mask=test_mask
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