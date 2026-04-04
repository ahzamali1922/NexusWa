# models/gnn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class NexusGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 1):
        super(NexusGraphSAGE, self).__init__()

        # Graph layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2)

        # Classifier
        self.classifier = torch.nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index):
        # 🔹 Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)  # 🔥 stronger dropout

        # 🔹 Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)  # 🔥 ADD THIS (VERY IMPORTANT)

        # 🔹 Output
        logits = self.classifier(x)
        return logits