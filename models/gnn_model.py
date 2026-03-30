# models/gnn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class NexusGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 1):
        super(NexusGraphSAGE, self).__init__()
        
        # GraphSAGE layers aggregate information from a node's neighbors
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2)
        
        # Final linear layer to output a raw risk score (logit)
        self.classifier = torch.nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index):
        # Pass 1: Look at immediate neighbors
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Pass 2: Look at neighbors of neighbors (2-hop topology)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Output classification score
        logits = self.classifier(x)
        return logits