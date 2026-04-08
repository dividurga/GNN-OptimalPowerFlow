"""
GNN model: two graph convolution layers followed by two fully-connected layers.
Supports GCN, GraphConv, SAGEConv, GATConv, ChebConv — matching the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GATConv, ChebConv

SUPPORTED_TYPES = ('GCN', 'GraphConv', 'SAGEConv', 'GATConv', 'ChebConv')


class GNNPowerFlow(nn.Module):
    def __init__(
        self,
        n_bus: int,
        feat_in: int = 7,
        feat_size1: int = 12,
        feat_size2: int = 12,
        hidden_size: int = 128,
        gnn_type: str = 'GraphConv',
        dropout: float = 0.0,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        if gnn_type not in SUPPORTED_TYPES:
            raise ValueError(f"gnn_type must be one of {SUPPORTED_TYPES}, got '{gnn_type}'")

        self.n_bus      = n_bus
        self.feat_size2 = feat_size2
        self.use_bn     = use_batch_norm

        def make_conv(in_ch, out_ch):
            if gnn_type == 'GCN':       return GCNConv(in_ch, out_ch)
            if gnn_type == 'GraphConv': return GraphConv(in_ch, out_ch)
            if gnn_type == 'SAGEConv':  return SAGEConv(in_ch, out_ch)
            if gnn_type == 'GATConv':   return GATConv(in_ch, out_ch)
            if gnn_type == 'ChebConv':  return ChebConv(in_ch, out_ch, K=2)

        self.conv1 = make_conv(feat_in,   feat_size1)
        self.conv2 = make_conv(feat_size1, feat_size2)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(feat_size1)
            self.bn2 = nn.BatchNorm1d(feat_size2)

        self.dropout = nn.Dropout(dropout)
        self.lin1    = nn.Linear(n_bus * feat_size2, hidden_size)
        self.lin2    = nn.Linear(hidden_size, n_bus * 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch_size = data.num_graphs

        x = self.conv1(x, edge_index)
        if self.use_bn: x = self.bn1(x)
        x = self.dropout(F.relu(x))

        x = self.conv2(x, edge_index)
        if self.use_bn: x = self.bn2(x)
        x = self.dropout(F.relu(x))

        x = x.view(batch_size, self.n_bus * self.feat_size2)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x   # shape: [batch_size, n_bus * 2]
