import os
import torch
import numpy as np
from pathlib import Path
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Import your model (Make sure this matches your actual import!)
from models.gnn_model import NexusGATv2 # Or whatever your model class is named

def evaluate_all():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Evaluating on device: {device}\n")

    prefixes = ["HI-Small", "HI-Medium", "LI-Small", "LI-Medium", "LI-Large"]
    
    # We will test these thresholds to find the best F1
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for prefix in prefixes:
        print(f"{'='*40}")
        print(f"📊 Evaluating: {prefix}")
        print(f"{'='*40}")

        graph_path = Path(f"./output/{prefix}/nexuswatch_graph.pt")
        model_path = Path(f"./models/gnn_model_{prefix}.pt")

        if not graph_path.exists() or not model_path.exists():
            print(f"⚠️ Missing files for {prefix}. Skipping...\n")
            continue

        # 1. Load Data
        graph_data = torch.load(graph_path, weights_only=False).to(device)
        
        # 2. Initialize Model
        # Update this if your model takes different initialization arguments
        model = NexusGATv2(in_channels=graph_data.num_node_features).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        # 3. Setup Mini-Batch Inference (THE FIX FOR OOM)
        # Using a NeighborLoader allows us to evaluate the graph in chunks
        eval_loader = NeighborLoader(
            graph_data,
            num_neighbors=[-1, -1, -1],  # -1 means take all neighbors during inference
            batch_size=4096,             # Process 4096 nodes at a time
            shuffle=False,
            num_workers=0                # Keep at 0 for Windows to avoid hanging
        )

        all_probs = []
        all_labels = []

        print("🔄 Running mini-batch inference...")
        with torch.no_grad():
            for batch in eval_loader:
                # Forward pass on the subgraph batch
                # If your model takes edge_attr, include it here: model(batch.x, batch.edge_index, batch.edge_attr)
                logits = model(batch.x, batch.edge_index).squeeze()
                
                # Convert logits to probabilities
                probs = torch.sigmoid(logits)
                
                # Only keep predictions for the target nodes in this batch (not the sampled neighbors)
                all_probs.append(probs[:batch.batch_size].cpu())
                all_labels.append(batch.y[:batch.batch_size].cpu())

        # Concatenate all batches back into full arrays
        full_probs = torch.cat(all_probs).numpy()
        full_labels = torch.cat(all_labels).numpy()

        # 4. Find the Best Threshold
        best_f1 = 0.0
        best_thresh = 0.5
        best_metrics = {}

        print("🔍 Threshold Tuning:")
        for t in thresholds_to_test:
            preds = (full_probs >= t).astype(int)
            
            acc = accuracy_score(full_labels, preds)
            f1 = f1_score(full_labels, preds, zero_division=0)
            prec = precision_score(full_labels, preds, zero_division=0)
            rec = recall_score(full_labels, preds, zero_division=0)

            # Track the best F1 Score
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
                best_metrics = {"acc": acc, "prec": prec, "rec": rec}

        # 5. Output the Winner
        print(f"\n✅ BEST THRESHOLD for {prefix}: {best_thresh}")
        print(f"🔥 BEST F1 SCORE : {best_f1:.4f}")
        print(f"🎯 Accuracy      : {best_metrics.get('acc', 0):.4f}")
        print(f"🎯 Precision     : {best_metrics.get('prec', 0):.4f}")
        print(f"🎯 Recall        : {best_metrics.get('rec', 0):.4f}\n")

if __name__ == "__main__":
    evaluate_all()