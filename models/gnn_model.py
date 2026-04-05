# models/gnn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class NexusGraph(torch.nn.Module):
    # INCREASED: hidden_channels from 128 to 256
    def __init__(self, in_channels: int, edge_dim: int, hidden_channels: int = 256, out_channels: int = 1):
        super(NexusGraph, self).__init__()

        # Layer 1: 256 channels
        self.conv1 = TransformerConv(in_channels, hidden_channels, edge_dim=edge_dim)
        # Layer 2: 128 channels
        self.conv2 = TransformerConv(hidden_channels, hidden_channels // 2, edge_dim=edge_dim)
        # Layer 3: 64 channels (NEW - Allows the model to see 3 hops deep!)
        self.conv3 = TransformerConv(hidden_channels // 2, hidden_channels // 4, edge_dim=edge_dim)

        # Classifier
        self.classifier = torch.nn.Linear(hidden_channels // 4, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # 🔹 Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        # 🔹 Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        # 🔹 Layer 3 (NEW)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        # 🔹 Output
        logits = self.classifier(x)
        return logits