import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models.gnn_model import NexusGraphSAGE

# All datasets
prefixes = ["HI-Small", "HI-Medium", "LI-Small", "LI-Medium", "LI-Large"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\n🚀 Evaluating on device: {device}\n")

for prefix in prefixes:
    print("\n==============================")
    print(f"📊 Evaluating: {prefix}")
    print("==============================")

    graph_path = Path(f"./output/{prefix}/nexuswatch_graph.pt")
    model_path = Path(f"./models/gnn_model_{prefix}.pt")

    # Load graph
    graph_data = torch.load(graph_path, weights_only=False).to(device)

    # Load model
    model = NexusGraphSAGE(
        in_channels=graph_data.num_node_features,
        hidden_channels=64,
        out_channels=1
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Inference
    with torch.no_grad():
        logits = model(graph_data.x, graph_data.edge_index).squeeze()
        probs = torch.sigmoid(logits)

    # Ground truth
    y_true = graph_data.y.cpu().numpy()

    # Threshold tuning
    best_f1 = 0
    best_threshold = 0

    print("\n🔍 Threshold Tuning:")

    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        preds = (probs > t).cpu().numpy()

        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        precision = precision_score(y_true, preds)
        recall = recall_score(y_true, preds)

        print(f"\nThreshold {t}")
        print(f"Accuracy  : {acc:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"\n✅ BEST THRESHOLD for {prefix}: {best_threshold}")
    print(f"🔥 BEST F1 SCORE: {best_f1:.4f}")