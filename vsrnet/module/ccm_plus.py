import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torchvision.models import resnet34, ResNet34_Weights

class CMMPlus(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, num_classes, edge_dim=None, dropout=0.1):
        """
        GAT-based Edge Classification Model with Edge Features.

        Args:
            in_channels (int): Number of input node features.
            hidden_channels (int): Number of hidden units in GAT layers.
            num_heads (int): Number of attention heads for GAT.
            num_classes (int): Number of output classes for edge classification.
            edge_dim (int, optional): Dimension of edge attributes (if available).
            dropout (float): Dropout rate for attention and MLP layers.
        """
        super(CMMPlus, self).__init__()

        self.feature_extractor = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.feature_extractor.fc = nn.Identity()  # Remove classification layer

        # Multi-head GAT layers for node feature extraction
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=True, dropout=dropout)

        # Edge attribute processing (only if edge_dim is provided)
        self.edge_mlp = None
        if edge_dim is not None:
            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_dim, hidden_channels),  # Map edge attributes to hidden space
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)  # Match hidden size
            )

        # Self-Attention Layer for edge features (Transformer-style)
        edge_feature_dim = 2 * hidden_channels + (hidden_channels if edge_dim is not None else 0)
        self.edge_attention = nn.TransformerEncoderLayer(
            d_model=edge_feature_dim,  # Edge feature dimension (concatenated node & edge_attr)
            nhead=4,  # Multi-head attention
            dim_feedforward=hidden_channels * 4,  # Expansion in feedforward layer
            dropout=dropout,
            batch_first=True  # Ensures proper input handling
        )

        # MLP to classify edges
        self.mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)  # Output shape: [num_edges, num_classes]
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

        x = self.feature_extractor(x)

        # Ensure edge_index has the correct shape
        if edge_index.numel() == 0:
            raise ValueError("Empty edge_index detected, skipping batch.")

        if edge_index.shape[0] != 2:
            raise ValueError(f"Invalid edge_index shape: {edge_index.shape}, expected [2, num_edges]")

        # Node feature extraction using multi-head GAT
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))

        # Aggregate pairwise node features for edges
        row, col = edge_index  # row = source nodes, col = target nodes
        edge_features = torch.cat([x[row], x[col]], dim=-1)  # Shape: [num_edges, 2 * hidden_channels]

        # If edge attributes exist, process and concatenate them
        if edge_attr is not None and self.edge_mlp is not None:
            edge_emb = self.edge_mlp(edge_attr)  # Transform edge attributes
            edge_features = torch.cat([edge_features, edge_emb], dim=-1)  # Concatenate edge attributes

        # Self-attention layer for edge features
        edge_features = self.edge_attention(edge_features.unsqueeze(0)).squeeze(0)  # [num_edges, feature_dim]

        # Predict edge labels
        edge_logits = self.mlp(edge_features)  # Shape: [num_edges, num_classes]

        return torch.sigmoid(edge_logits).squeeze()
