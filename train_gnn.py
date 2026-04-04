import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
import time
from sklearn.metrics import f1_score, precision_score, recall_score

from torch_geometric.loader import NeighborLoader
from models.gnn_model import NexusGraphSAGE

# 🔥 Prevent CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 1. Setup
PREFIX = os.environ.get("DATASET_PREFIX", "HI-Medium")
GRAPH_PATH = Path(f"./output/{PREFIX}/nexuswatch_graph.pt")

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)
SAVE_PATH = MODEL_DIR / f"gnn_model_{PREFIX}.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

if device.type == "cuda":
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")

# 2. Load Graph (KEEP ON CPU)
print(f"Loading graph from {GRAPH_PATH}...")
graph_data = torch.load(GRAPH_PATH, weights_only=False)

print(f"Nodes: {graph_data.num_nodes:,} | Edges: {graph_data.num_edges:,}")

# 3. Class Imbalance
num_fraud = graph_data.y.sum().item()
num_legit = graph_data.num_nodes - num_fraud
imbalance_ratio = num_legit / max(num_fraud, 1)

pos_weight = torch.tensor([imbalance_ratio]).to(device)

print(f"Legit:Fraud Ratio = {imbalance_ratio:.1f}:1")

# 4. Model
model = NexusGraphSAGE(
    in_channels=graph_data.num_node_features,
    hidden_channels=128,
    out_channels=1
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=5e-4)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 🔥 5. Neighbor Sampling Loader (KEY FIX)
train_loader = NeighborLoader(
    graph_data,
    num_neighbors=[15, 10],   # 2-layer GraphSAGE
    batch_size=2048,          # reduce if memory issue
    shuffle=True
)

# 6. Training
epochs = 80
fraud_weight = 12.0

print("Starting mini-batch training...")
model.train()

t0 = time.time()

for epoch in range(epochs):
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index).squeeze()

        base_loss = criterion(out, batch.y)
        weights = torch.where(batch.y == 1, fraud_weight, 1.0)
        loss = (base_loss * weights).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 🔥 Evaluate on last batch (fast approximation)
    with torch.no_grad():
        probs = torch.sigmoid(out)
        preds = (probs > 0.7).float()

        y_true = batch.y.cpu().numpy()
        y_pred = preds.cpu().numpy()

        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        acc = (preds == batch.y).float().mean().item()

    print(
        f"Epoch {epoch:03d}/{epochs} | "
        f"Loss: {total_loss:.4f} | "
        f"Acc: {acc:.4f} | "
        f"F1: {f1:.4f} | "
        f"P: {precision:.4f} | "
        f"R: {recall:.4f}"
    )

print(f"\n✅ Training complete in {time.time() - t0:.2f}s")

# 7. Save Model
torch.save(model.state_dict(), SAVE_PATH)
print(f"💾 Model saved successfully to: {SAVE_PATH}")