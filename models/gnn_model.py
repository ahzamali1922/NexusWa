# models/gnn_model.py
import torch # type: ignore
import torch.nn.functional as F # type: ignore
from torch_geometric.nn import TransformerConv # type: ignore

class NexusGraph(torch.nn.Module):
    def __init__(self, in_channels: int, edge_dim: int, hidden_channels: int = 128, out_channels: int = 1):
        super(NexusGraph, self).__init__()

        # Graph layers using TransformerConv to support edge features natively
        self.conv1 = TransformerConv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels // 2, edge_dim=edge_dim)

        # Classifier
        self.classifier = torch.nn.Linear(hidden_channels // 2, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # 🔹 Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 🔹 Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 🔹 Output
        logits = self.classifier(x)
        return logits