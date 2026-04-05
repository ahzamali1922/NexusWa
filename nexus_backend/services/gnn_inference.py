# services/gnn_inference.py
import torch # type: ignore
import os
from pathlib import Path

# Import the updated model architecture
from models.gnn_model import NexusGraph

# Dataset-specific thresholds (from evaluation)
threshold_map = {
    "HI-Small": 0.8,
    "HI-Medium": 0.8,
    "LI-Small": 0.8,
    "LI-Medium": 0.8,
    "LI-Large": 0.7
}

class GraphDetector:
    def __init__(self, model_path: str, graph_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Graph from: {graph_path}...")
        # 1. Load the Graph
        self.graph = torch.load(graph_path, weights_only=False).to(self.device)
        
        print(f"Loading Model from: {model_path}...")
        # 2. Initialize the Model dynamically (Fixed hidden channels and added edge_dim)
        self.model = NexusGraph(
            in_channels=self.graph.num_node_features, 
            edge_dim=self.graph.num_edge_features,
            hidden_channels=128, 
            out_channels=1
        ).to(self.device)
        
        # 3. Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        
        # 4. Set model to evaluation mode
        self.model.eval()
        
        print(f"✅ Graph & Model loaded successfully on {self.device}")

    def detect_anomalies(self):
        """Runs a forward pass and returns nodes above the risk threshold."""
        with torch.no_grad():
            # Passed edge_attr here!
            logits = self.model(self.graph.x, self.graph.edge_index, self.graph.edge_attr)
            
            # Convert raw logits to probabilities (0.0 to 1.0) using Sigmoid
            probabilities = torch.sigmoid(logits).squeeze()

            # Get dataset prefix
            prefix = os.environ.get("DATASET_PREFIX", "HI-Medium")

            # Select best threshold
            threshold = threshold_map.get(prefix, 0.7)

            print(f"Using threshold: {threshold} for {prefix}")
            
            # Find nodes that cross the risk threshold
            flagged_indices = (probabilities > threshold).nonzero(as_tuple=True)[0]
            
            # Convert to CPU and format nicely for the JSON API response
            risk_scores = {
                int(idx): round(float(probabilities[idx]), 4) 
                for idx in flagged_indices
            }
            
            return int(flagged_indices.numel()), risk_scores

# Determine paths dynamically based on the dataset prefix
PREFIX = os.environ.get("DATASET_PREFIX", "HI-Medium")

# Instantiate a global instance to be used by the FastAPI router
detector = GraphDetector(
    model_path=f"../models/gnn_model_{PREFIX}.pt", 
    graph_path=f"../output/{PREFIX}/nexuswatch_graph.pt"
)