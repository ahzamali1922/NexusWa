import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import os
import time
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.loader import NeighborLoader
from models.gnn_model import NexusGraph

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    PREFIX = os.environ.get("DATASET_PREFIX", "HI-Medium")
    GRAPH_PATH = Path(f"./output/{PREFIX}/nexuswatch_graph.pt")
    
    # Create models directory if it doesn't exist
    SAVE_PATH = Path("./models") / f"gnn_model_{PREFIX}.pt"
    SAVE_PATH.parent.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    print(f"Loading graph from {GRAPH_PATH}...")
    graph_data = torch.load(GRAPH_PATH, weights_only=False)

    # Calculate Imbalance (Use train_mask only to prevent leakage)
    train_y = graph_data.y[graph_data.train_mask]
    num_fraud = train_y.sum().item()
    num_legit = len(train_y) - num_fraud
    imbalance_ratio = num_legit / max(num_fraud, 1)
    
    adjusted_weight = min(imbalance_ratio, 10.0) 
    pos_weight = torch.tensor([adjusted_weight]).to(device)

    model = NexusGraph(
        in_channels=graph_data.num_node_features,
        edge_dim=graph_data.num_edge_features,
        hidden_channels=128,
        out_channels=1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Dynamic batch sizing for LI-Large
    batch_size = 2048 if "Large" in PREFIX else 8192

    # Isolate training loader
    train_loader = NeighborLoader(
        graph_data, num_neighbors=[15, 10], input_nodes=graph_data.train_mask,
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    # Dedicated validation loader
    val_loader = NeighborLoader(
        graph_data, num_neighbors=[15, 10], input_nodes=graph_data.val_mask,
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    epochs, patience, epochs_no_improve, best_f1 = 150, 15, 0, 0.0

    print("Starting mini-batch training...")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch.x, batch.edge_index, batch.edge_attr).squeeze()
            
            # FIX: Slice 'out' to match the target batch size (ignore sampled neighbors)
            loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size].float())
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate over the ENTIRE validation loader
        model.eval()
        val_probs, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr).squeeze()
                
                # FIX: Slice 'out' here as well before applying sigmoid
                probs = torch.sigmoid(out[:batch.batch_size])
                
                val_probs.append(probs.cpu())
                val_labels.append(batch.y[:batch.batch_size].cpu())

        val_probs = torch.cat(val_probs).numpy()
        val_labels = torch.cat(val_labels).numpy()
        val_preds = (val_probs > 0.5).astype(float) # Neutral 0.5 for validation

        f1 = f1_score(val_labels, val_preds, zero_division=0)

        print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f} | Val F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
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