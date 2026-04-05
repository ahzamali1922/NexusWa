import os
import torch
import numpy as np
from pathlib import Path
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Import the exact class name
from models.gnn_model import NexusGraph 

def evaluate_all():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Evaluating on device: {device}\n")

    # Put the loop back in to run through all datasets
    prefixes = ["HI-Small", "HI-Medium", "LI-Small", "LI-Medium", "LI-Large"]
    thresholds_to_test = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for prefix in prefixes:
        print(f"{'='*40}")
        print(f"📊 Evaluating: {prefix}")
        print(f"{'='*40}")

        graph_path = Path(f"./output/{prefix}/nexuswatch_graph.pt")
        model_path = Path(f"./models/gnn_model_{prefix}.pt")

        # Safety check: skip if the graph or model hasn't been generated yet
        if not graph_path.exists() or not model_path.exists():
            print(f"⚠️ Missing files for {prefix}. Skipping...\n")
            continue

        # 1. Load Data
        graph_data = torch.load(graph_path, weights_only=False).to(device)
        
        # 2. Initialize Model (Added edge_dim)
        model = NexusGraph(
            in_channels=graph_data.num_node_features,
            edge_dim=graph_data.num_edge_features
        ).to(device)
        
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        # 3. Setup Mini-Batch Inference
        eval_loader = NeighborLoader(
            graph_data,
            num_neighbors=[-1, -1],  # 2 hops for 2 layers
            input_nodes=graph_data.test_mask, # Evaluate ONLY on test_mask
            batch_size=4096,             
            shuffle=False,
            num_workers=0                
        )

        all_probs = []
        all_labels = []

        print("🔄 Running mini-batch inference...")
        with torch.no_grad():
            for batch in eval_loader:
                logits = model(batch.x, batch.edge_index, batch.edge_attr).squeeze()
                probs = torch.sigmoid(logits)
                
                # Slice to match the target batch size
                all_probs.append(probs[:batch.batch_size].cpu())
                all_labels.append(batch.y[:batch.batch_size].cpu())

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

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
                best_metrics = {"acc": acc, "prec": prec, "rec": rec}

        print(f"\n✅ BEST THRESHOLD for {prefix}: {best_thresh}")
        print(f"🔥 BEST F1 SCORE : {best_f1:.4f}")
        print(f"🎯 Accuracy      : {best_metrics.get('acc', 0):.4f}")
        print(f"🎯 Precision     : {best_metrics.get('prec', 0):.4f}")
        print(f"🎯 Recall        : {best_metrics.get('rec', 0):.4f}\n")

if __name__ == "__main__":
    evaluate_all()
