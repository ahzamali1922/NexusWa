# services/gnn_inference.py
import torch
import os
from pathlib import Path
from models.gnn_model import NexusGraph

threshold_map = {
    "HI-Small": 0.8,
    "HI-Medium": 0.8,
    "LI-Small": 0.8,
    "LI-Medium": 0.8,
    "LI-Large": 0.7
}

class GraphDetector:
    def __init__(self, model_path: Path, graph_path: Path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Graph from: {graph_path}...")
        self.graph = torch.load(graph_path, weights_only=False).to(self.device)
        
        print(f"Loading Model from: {model_path}...")
        self.model = NexusGraph(
            in_channels=self.graph.num_node_features, 
            edge_dim=self.graph.num_edge_features,
            hidden_channels=128, 
            out_channels=1
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def detect_anomalies(self):
        with torch.no_grad():
            logits = self.model(self.graph.x, self.graph.edge_index, self.graph.edge_attr)
            probabilities = torch.sigmoid(logits).squeeze()

            prefix = os.environ.get("DATASET_PREFIX", "HI-Medium")
            threshold = threshold_map.get(prefix, 0.7)
            
            flagged_indices = (probabilities > threshold).nonzero(as_tuple=True)[0]
            
            risk_scores = {
                int(idx): round(float(probabilities[idx]), 4) 
                for idx in flagged_indices
            }
            
            return int(flagged_indices.numel()), risk_scores

# FIX: Robust absolute pathing based on script location
BASE_DIR = Path(__file__).resolve().parent.parent 
PREFIX = os.environ.get("DATASET_PREFIX", "HI-Medium")

detector = GraphDetector(
    model_path=BASE_DIR / f"models/gnn_model_{PREFIX}.pt", 
    graph_path=BASE_DIR / f"output/{PREFIX}/nexuswatch_graph.pt"
)