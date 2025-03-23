import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch_geometric.nn import GCNConv
from torchvision.models import resnet18, ResNet18_Weights

class CCMBase(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, edge_dim=None, dropout=0.1):
        """
        GCN-based Edge Classification Model with Edge Features.

        Args:
            in_channels (int): Number of input node features.
            hidden_channels (int): Number of hidden units in GCN layers.
            num_classes (int): Number of output classes for edge classification.
            edge_dim (int, optional): Dimension of edge attributes (if available).
            dropout (float): Dropout rate for MLP layers.
        """
        super(CCMBase, self).__init__()

        self.feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor.fc = nn.Identity()  # Remove classification layer
        # self.feature_extractor.fc = nn.Identity()  # Remove classification layer
        # GCN layers for node feature extraction
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)

        # Edge attribute processing (if edge_dim is provided)
        self.edge_mlp = None
        if edge_dim is not None:
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_dim, hidden_channels),  
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)  
            )

        # Define edge feature size
        edge_feature_dim = 2 * hidden_channels + (hidden_channels if edge_dim is not None else 0)

        # MLP for edge classification
        self.mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)  
        )

    def forward(self, data):
        """
        Forward pass for edge classification.

        Args:
            data (torch_geometric.data.Data): Graph data containing node features, edge_index, and edge_attr.

        Returns:
            torch.Tensor: Predicted logits for each edge of shape [num_edges, num_classes].
        """
        x, edge_index, edge_attr = data.x, data.edge_index, getattr(data, "edge_attr", None)
        # print(x.shape)
        # Ensure edge_index has the correct shape
        if edge_index.numel() == 0:
            raise ValueError("Empty edge_index detected, skipping batch.")

        if edge_index.shape[0] != 2:
            raise ValueError(f"Invalid edge_index shape: {edge_index.shape}, expected [2, num_edges]")

        x = self.feature_extractor(x)
        # Node feature extraction using GCN
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))

        # Aggregate pairwise node features for edges
        row, col = edge_index  # row = source nodes, col = target nodes
        edge_features = torch.cat([x[row], x[col]], dim=-1)  # Shape: [num_edges, 2 * hidden_channels]

        # If edge attributes exist, process and concatenate them
        if edge_attr is not None and self.edge_mlp is not None:
            edge_emb = self.edge_mlp(edge_attr)  
            edge_features = torch.cat([edge_features, edge_emb], dim=-1)  

        # Predict edge labels
        edge_logits = self.mlp(edge_features)  

        return edge_logits