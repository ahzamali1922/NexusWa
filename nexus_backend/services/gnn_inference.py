# services/gnn_inference.py
import torch
import os
from pathlib import Path

# Import the model architecture we created in Step 2
from models.gnn_model import NexusGraphSAGE

class GraphDetector:
    def __init__(self, model_path: str, graph_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Graph from: {graph_path}...")
        # 1. Load the Graph (remembering weights_only=False for PyG objects)
        self.graph = torch.load(graph_path, weights_only=False).to(self.device)
        
        print(f"Loading Model from: {model_path}...")
        # 2. Initialize the Model architecture dynamically based on node features
        self.model = NexusGraphSAGE(
            in_channels=self.graph.num_node_features, 
            hidden_channels=64, 
            out_channels=1
        ).to(self.device)
        
        # 3. Load the trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        
        # 4. Set model to evaluation mode (turns off dropout, etc.)
        self.model.eval()
        
        print(f"✅ Graph & Model loaded successfully on {self.device}")

    def detect_anomalies(self, threshold: float = 0.85):
        """Runs a forward pass and returns nodes above the risk threshold."""
        with torch.no_grad():
            # The real forward pass!
            logits = self.model(self.graph.x, self.graph.edge_index)
            
            # Convert raw logits to probabilities (0.0 to 1.0) using Sigmoid
            probabilities = torch.sigmoid(logits).squeeze()
            
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