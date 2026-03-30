# nexuswatch_graph_builder.py
import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path
import time

print("Starting Graph Construction...")
t0 = time.time()

# Paths
OUT_DIR = Path("./output")
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

# 3. Build Edge Index (The connections between nodes)
print("Building edge index...")
src = trans_df['From Account'].map(account_mapping).values
dst = trans_df['To Account'].map(account_mapping).values
edge_index = torch.tensor([src, dst], dtype=torch.long)

# 4. Define Node Labels (1 = Fraudulent, 0 = Legitimate)
print("Assigning node labels...")
num_nodes = len(unique_accounts)
node_labels = torch.zeros(num_nodes, dtype=torch.float)

fraud_mask = trans_df['Is Laundering'] == 1
fraud_accounts = set(trans_df[fraud_mask]['From Account']).union(set(trans_df[fraud_mask]['To Account']))
fraud_indices = [account_mapping[acc] for acc in fraud_accounts if acc in account_mapping]

node_labels[fraud_indices] = 1.0

# 5. Build the PyG Data Object
print("Constructing PyG Data object...")
# (We are keeping node/edge features simple for this initial test)
graph_data = Data(
    num_nodes=num_nodes,
    edge_index=edge_index, 
    y=node_labels
)

# 6. Save to Disk
out_path = OUT_DIR / "nexuswatch_graph.pt"
torch.save(graph_data, out_path)

print(f"\nSUCCESS!")
print(f"Nodes: {graph_data.num_nodes:,}")
print(f"Edges: {graph_data.num_edges:,}")
print(f"Graph saved to: {out_path}")
print(f"Total time: {time.time() - t0:.2f}s")