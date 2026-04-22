"""
Message Passing Network (MPN) — ported from PoweFlowNet/networks/MPN.py
and adapted to the /gnn/ pipeline (7-feature input, [batch, n_bus*2] output).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, TAGConv
from torch_geometric.utils import degree


class EdgeAggregation(MessagePassing):
    """Aggregate node + edge features via a small MLP (PNA-style)."""

    def __init__(self, nfeature_dim, efeature_dim, hidden_dim, output_dim):
        super().__init__(aggr='add')
        self.edge_aggr = nn.Sequential(
            nn.Linear(nfeature_dim * 2 + efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def message(self, x_i, x_j, edge_attr):
        return self.edge_aggr(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr, norm=norm)


class MPN(nn.Module):
    """
    Message Passing Network for power flow.
    - One EdgeAggregation layer to mix node + edge features
    - n_gnn_layers TAGConv layers
    - Final projection to [n_bus * 2] (V, δ)
    """

    def __init__(
        self,
        n_bus: int,
        nfeature_dim: int = 7,
        efeature_dim: int = 4,
        hidden_dim: int = 129,
        n_gnn_layers: int = 4,
        K: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.n_bus = n_bus
        self.dropout_rate = dropout_rate

        self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)

        self.convs = nn.ModuleList()
        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
        for _ in range(n_gnn_layers - 2):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
        self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        self.out_proj = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch_size = data.num_graphs

        x = self.edge_aggr(x, edge_index, edge_attr)
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
            x = nn.functional.relu(x)
        x = self.convs[-1](x=x, edge_index=edge_index)

        x = self.out_proj(x)  # [total_nodes, 2]
        return x.view(batch_size, self.n_bus * 2)
