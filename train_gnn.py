import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
import time
import numpy as np
from sklearn.metrics import f1_score
from torch_geometric.loader import NeighborLoader
from models.gnn_model import NexusGraph

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    PREFIX = os.environ.get("DATASET_PREFIX", "HI-Medium")
    GRAPH_PATH = Path(f"./output/{PREFIX}/nexuswatch_graph.pt")
    
    SAVE_PATH = Path("./models") / f"gnn_model_{PREFIX}.pt"
    SAVE_PATH.parent.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device} for {PREFIX}")

    graph_data = torch.load(GRAPH_PATH, weights_only=False)

    train_y = graph_data.y[graph_data.train_mask]
    num_fraud = train_y.sum().item()
    num_legit = len(train_y) - num_fraud
    imbalance_ratio = num_legit / max(num_fraud, 1)
    
    # Cap pos_weight at 12.0 for a slightly stronger fraud signal
    adjusted_weight = min(imbalance_ratio, 12.0) 
    pos_weight = torch.tensor([adjusted_weight]).to(device)

    # UPDATED: Initialize with 256 hidden channels
    model = NexusGraph(
        in_channels=graph_data.num_node_features,
        edge_dim=graph_data.num_edge_features,
        hidden_channels=256,
        out_channels=1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # FIX: Removed the verbose=True argument that was causing the crash
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    batch_size = 2048 if "Large" in PREFIX else 8192

    # UPDATED: 3 Hops of neighbor sampling [15, 10, 5] for our 3 layers
    train_loader = NeighborLoader(
        graph_data, num_neighbors=[15, 10, 5], input_nodes=graph_data.train_mask,
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    val_loader = NeighborLoader(
        graph_data, num_neighbors=[15, 10, 5], input_nodes=graph_data.val_mask,
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    epochs, patience, epochs_no_improve, best_f1 = 150, 15, 0, 0.0
    val_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8] # Dynamic threshold search

    print("Starting mini-batch training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.edge_attr).squeeze()
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size].float())
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation Phase
        model.eval()
        val_probs, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr).squeeze()
                probs = torch.sigmoid(out[:batch.batch_size])
                val_probs.append(probs.cpu())
                val_labels.append(batch.y[:batch.batch_size].cpu())

        val_probs = torch.cat(val_probs).numpy()
        val_labels = torch.cat(val_labels).numpy()
        
        # SMART VALIDATION: Find the best F1 across multiple thresholds
        epoch_best_f1 = 0.0
        for t in val_thresholds:
            preds = (val_probs > t).astype(float)
            f1 = f1_score(val_labels, preds, zero_division=0)
            if f1 > epoch_best_f1:
                epoch_best_f1 = f1

        print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f} | Val F1: {epoch_best_f1:.4f}")
        
        # Step the LR Scheduler
        scheduler.step(epoch_best_f1)

        # Early Stopping Logic
        if epoch_best_f1 > best_f1:
            best_f1 = epoch_best_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH) 
            print(f"   🌟 New best F1 score! Model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping triggered!")
                break 

if __name__ == '__main__':
    main()