# train_gnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
import time

# Import the model we just created
from models.gnn_model import NexusGraphSAGE

# 1. Setup Paths & Device
PREFIX = os.environ.get("DATASET_PREFIX", "HI-Medium")
GRAPH_PATH = Path(f"./output/{PREFIX}/nexuswatch_graph.pt")
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)
SAVE_PATH = MODEL_DIR / f"gnn_model_{PREFIX}.pt"

# Your RTX A6000 will be picked up here
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Training starting on: {device}\n")

# 2. Load the Graph Data
print(f"Loading graph from {GRAPH_PATH}...")
graph_data = torch.load(GRAPH_PATH, weights_only=False).to(device)
print(f"Nodes: {graph_data.num_nodes:,} | Edges: {graph_data.num_edges:,}")

# 3. Handle Class Imbalance
# Fraud is rare. We need to penalize the model heavily for missing a fraudster.
num_fraud = graph_data.y.sum().item()
num_legit = graph_data.num_nodes - num_fraud
imbalance_ratio = num_legit / max(num_fraud, 1)

pos_weight = torch.tensor([imbalance_ratio]).to(device)
print(f"Legit:Fraud Ratio = {imbalance_ratio:.1f}:1")
print(f"Applying pos_weight of {imbalance_ratio:.2f} to loss function.\n")

# 4. Initialize Model, Optimizer, and Loss Function
model = NexusGraphSAGE(
    in_channels=graph_data.num_node_features, 
    hidden_channels=64, 
    out_channels=1
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# BCEWithLogitsLoss is mathematically more stable than combining Sigmoid + BCELoss
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 

# 5. Training Loop
print("Starting training loop...")
model.train()
epochs = 200
t0 = time.time()

for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Forward pass
    out = model(graph_data.x, graph_data.edge_index).squeeze()
    
    # Calculate loss
    loss = criterion(out, graph_data.y)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:03d}/{epochs} | Loss: {loss.item():.4f}")

print(f"\n✅ Training complete in {time.time() - t0:.2f}s")

# 6. Save the trained weights
torch.save(model.state_dict(), SAVE_PATH)
print(f"💾 Model saved successfully to: {SAVE_PATH}")