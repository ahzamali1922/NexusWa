# services/gnn_inference.py
import torch
from torch_geometric.data import Data

class GraphDetector:
    def __init__(self, model_path: str, graph_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Load the Graph
        self.graph = torch.load(graph_path).to(self.device)
        
        # 2. Load the Model (assuming you have a PyTorch class defined)
        # self.model = MyGraphSAGEModel().to(self.device)
        # self.model.load_state_dict(torch.load(model_path))
        # self.model.eval()
        
        print(f"Graph & Model loaded on {self.device}")

    def detect_anomalies(self, threshold: float = 0.85):
        """Runs a forward pass and returns nodes above the risk threshold."""
        with torch.no_grad():
            # Example forward pass
            # logits = self.model(self.graph.x, self.graph.edge_index)
            # probabilities = torch.sigmoid(logits).squeeze()
            
            # MOCK DATA for now:
            probabilities = torch.rand(self.graph.num_nodes).to(self.device)
            
            flagged_indices = (probabilities > threshold).nonzero(as_tuple=True)[0]
            
            # Convert to CPU for JSON serialization
            risk_scores = {
                int(idx): float(probabilities[idx]) 
                for idx in flagged_indices
            }
            
            return int(flagged_indices.numel()), risk_scores

# Instantiate a global instance to be used by the router
detector = GraphDetector(
    model_path="../models/gnn_model.pt", 
    graph_path="../output/nexuswatch_graph.pt"
)