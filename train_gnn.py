import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
import time
from sklearn.metrics import f1_score, precision_score, recall_score

from torch_geometric.loader import NeighborLoader
from models.gnn_model import NexusGraph

# 🔥 Prevent CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
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

    # 🔥 FIX: Cap the penalty so the model stops spamming false positives
    adjusted_weight = min(imbalance_ratio, 10.0) 
    pos_weight = torch.tensor([adjusted_weight]).to(device)

    print(f"Legit:Fraud Ratio = {imbalance_ratio:.1f}:1")
    print(f"Using adjusted pos_weight: {adjusted_weight:.1f}")

    # 4. Model
    model = NexusGraph(
        in_channels=graph_data.num_node_features,
        edge_dim=graph_data.num_edge_features,
        hidden_channels=128,
        out_channels=1
    ).to(device)

    # 🔥 FIX: Lower learning rate for more stable steps
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 🔥 5. Neighbor Sampling Loader
    train_loader = NeighborLoader(
        graph_data,
        num_neighbors=[15, 10],
        batch_size=8192,          # Dialed back slightly for faster CPU sampling
        shuffle=True,
        num_workers=0,            # Avoid Windows multiprocessing crash
        pin_memory=True           # Fast-lane RAM transfer to GPU
    )

    # 6. Training with Early Stopping
    epochs = 150                # Increased max epochs
    best_f1 = 0.0               
    patience = 15               
    epochs_no_improve = 0       

    print("Starting mini-batch training...")
    model.train()

    t0 = time.time()

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index, batch.edge_attr).squeeze()

            loss = criterion(out, batch.y.float())

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 🔥 Evaluate on last batch
        with torch.no_grad():
            probs = torch.sigmoid(out)
            preds = (probs > 0.7).float()

            y_true = batch.y.cpu().numpy()
            y_pred = preds.cpu().numpy()

            f1 = f1_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            acc = (preds == batch.y).float().mean().item()

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Loss: {total_loss:.4f} | "
            f"Acc: {acc:.4f} | "
            f"F1: {f1:.4f} | "
            f"P: {precision:.4f} | "
            f"R: {recall:.4f}"
        )

        # 🔥 EARLY STOPPING & CHECKPOINTING
        if f1 > best_f1:
            best_f1 = f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH) 
            print(f"   🌟 New best F1 score! Model saved to {SAVE_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n🛑 Early stopping triggered! No improvement for {patience} epochs.")
                break 

    print(f"\n✅ Training complete in {time.time() - t0:.2f}s")
    print(f"🏆 Best F1 Score achieved: {best_f1:.4f}")

if __name__ == '__main__':
    main()