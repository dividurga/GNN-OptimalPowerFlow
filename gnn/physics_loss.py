"""
Physics-informed losses for AC power flow.

Ported from PoweFlowNet/utils/custom_loss_functions.py (PowerImbalance class)
and adapted to the /gnn/ pipeline's normalisation scheme:
  - inputs x are normalised with x_mean/x_std (except bus-type flags at cols 4:7)
  - targets y=(V, δ) are normalised with y_mean/y_std
  - edge_attr=(R, X) in per-unit, normalised with edge_mean/edge_std
  - P, Q in the raw dataset are in MW/MVAr; divide by sn_mva for per-unit
  - δ is in degrees (verified on the IEEE 14-bus dataset)

Reference AC power flow equations:
    g_ij =  R / (R² + X²)
    b_ij = -X / (R² + X²)
    e_i  = Vm_i · cos(Va_i),   f_i = Vm_i · sin(Va_i)
    P_ji = g·(e_i·e_j - e_i² + f_i·f_j - f_i²) + b·(f_i·e_j - e_i·f_j)
    Q_ji = g·(f_i·e_j - e_i·f_j)              + b·(-e_i·e_j + e_i² - f_i·f_j + f_i²)
    ΔP_i = -Σ P_ji + P_demand_i,   ΔQ_i = -Σ Q_ji + Q_demand_i
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class PowerImbalance(MessagePassing):
    """
    Squared AC power imbalance (ΔP² + ΔQ²) averaged over nodes.

    Args at construction:
        x_mean, x_std     : [n_bus, 7]  node feature stats (P, Q at cols 0, 1)
        y_mean, y_std     : [n_bus, 2]  target stats (V, δ)
        edge_mean, edge_std : [2]       edge attribute stats (R, X)
        sn_mva            : float       base MVA (converts MW/MVAr → per-unit)

    forward(y_pred, x_input, edge_index, edge_attr) → scalar loss
        y_pred     : [batch, n_bus*2]  normalised predictions (V, δ)
        x_input    : [total_nodes, 7]  normalised input features
        edge_index : [2, E]
        edge_attr  : [E, 2]  normalised R, X

    The raw dataset records LOAD demand per bus but not generator output,
    so the KCL residual is only meaningful at pure-load buses. A per-bus
    `load_bus_mask` (True at load/passive buses, False at gen/slack) is
    applied before averaging; pass it via the constructor.
    """

    def __init__(self, x_mean, x_std, y_mean, y_std, edge_mean, edge_std, sn_mva,
                 load_bus_mask=None):
        super().__init__(aggr='add', flow='target_to_source')
        self.register_buffer('x_mean', x_mean.clone())
        self.register_buffer('x_std',  x_std.clone())
        self.register_buffer('y_mean', y_mean.clone())
        self.register_buffer('y_std',  y_std.clone())
        self.register_buffer('edge_mean', edge_mean.clone())
        self.register_buffer('edge_std',  edge_std.clone())
        self.sn_mva = sn_mva

        n_bus = y_mean.shape[0]
        if load_bus_mask is None:
            load_bus_mask = torch.ones(n_bus, dtype=torch.bool)
        self.register_buffer('load_bus_mask', load_bus_mask.clone())

    def _denormalize_inputs(self, y_pred_norm, x_input_norm, edge_attr_norm, n_bus, batch_size):
        """Return [total_nodes, 4]=(Vm_pu, Va_deg, P_pu, Q_pu) and edge_attr_pu."""
        # y_pred_norm: [batch, n_bus*2] → [total_nodes, 2] denormalised V, δ
        y_pred = y_pred_norm.view(batch_size, n_bus, 2)
        y_pred = y_pred * self.y_std + self.y_mean         # broadcasts over batch
        y_pred = y_pred.view(batch_size * n_bus, 2)        # [N, 2] = (Vm, Va_deg)

        # x_input_norm: [total_nodes, 7] → denormalise only P, Q (cols 0, 1)
        x_pq = x_input_norm.view(batch_size, n_bus, 7)[:, :, 0:2]
        pq_mean = self.x_mean[:, 0:2]
        pq_std  = self.x_std[:, 0:2]
        x_pq = x_pq * pq_std + pq_mean                     # broadcasts over batch
        x_pq = x_pq.view(batch_size * n_bus, 2) / self.sn_mva  # → per-unit

        x_phys = torch.cat([y_pred, x_pq], dim=-1)         # [N, 4] = (Vm, Va, P, Q)

        # edge_attr: global 2-vector stats
        edge_pu = edge_attr_norm * self.edge_std + self.edge_mean  # [E, 2]
        return x_phys, edge_pu

    def message(self, x_i, x_j, edge_attr):
        """Compute injected power P_ji, Q_ji along each edge (AC PF equations)."""
        r = edge_attr[:, 0:1]
        x = edge_attr[:, 1:2]
        g_ij =  r / (r**2 + x**2)
        b_ij = -x / (r**2 + x**2)

        vm_i = x_i[:, 0:1]
        va_i = x_i[:, 1:2] * (torch.pi / 180.0)   # degrees → radians
        vm_j = x_j[:, 0:1]
        va_j = x_j[:, 1:2] * (torch.pi / 180.0)

        e_i = vm_i * torch.cos(va_i); f_i = vm_i * torch.sin(va_i)
        e_j = vm_j * torch.cos(va_j); f_j = vm_j * torch.sin(va_j)

        Pji = g_ij * (e_i * e_j - e_i**2 + f_i * f_j - f_i**2) + b_ij * (f_i * e_j - e_i * f_j)
        Qji = g_ij * (f_i * e_j - e_i * f_j) + b_ij * (-e_i * e_j + e_i**2 - f_i * f_j + f_i**2)
        return torch.cat([Pji, Qji], dim=-1)

    def update(self, aggregated, x):
        """Power balance residual at each node: ΔP_i = -Σ P_ji + P_demand."""
        dP = -aggregated[:, 0:1] + x[:, 2:3]
        dQ = -aggregated[:, 1:2] + x[:, 3:4]
        return torch.cat([dP, dQ], dim=-1)

    def forward(self, y_pred, x_input, edge_index, edge_attr):
        n_bus  = self.y_mean.shape[0]
        batch_size = y_pred.view(-1, n_bus * 2).shape[0]

        x_phys, edge_pu = self._denormalize_inputs(
            y_pred, x_input, edge_attr, n_bus, batch_size
        )
        dPQ = self.propagate(edge_index, x=x_phys, edge_attr=edge_pu)  # [N, 2]
        sq = dPQ.square().sum(dim=-1)                                    # [N]

        # Mask out gen/slack buses where P_gen isn't in the dataset
        mask = self.load_bus_mask.repeat(batch_size)                     # [N]
        return sq[mask].mean()


class MixedMSEPowerImbalance(nn.Module):
    """
    Mixed loss: alpha * MSE(y_pred, y_true) + (1 - alpha) * physics_scale * PowerImbalance.

    MSE is computed in the DENORMALISED target space (matches train.py's MSE
    so scales are comparable with other baselines).
    """

    def __init__(self, x_mean, x_std, y_mean, y_std, edge_mean, edge_std, sn_mva,
                 load_bus_mask=None, alpha: float = 0.9, physics_scale: float = 0.02):
        super().__init__()
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha
        self.physics_scale = physics_scale
        self.power_imbalance = PowerImbalance(
            x_mean, x_std, y_mean, y_std, edge_mean, edge_std, sn_mva, load_bus_mask
        )
        self.register_buffer('y_mean', y_mean.clone())
        self.register_buffer('y_std',  y_std.clone())

    def forward(self, y_pred, y_true, x_input, edge_index, edge_attr):
        n_bus = self.y_mean.shape[0]
        batch_size = y_pred.view(-1, n_bus * 2).shape[0]

        # Denormalised MSE on V, δ (matches main-pipeline MSE convention)
        y_pred_r = y_pred.view(batch_size, n_bus, 2) * self.y_std + self.y_mean
        y_true_r = y_true.view(batch_size, n_bus, 2) * self.y_std + self.y_mean
        mse = (y_pred_r - y_true_r).pow(2).mean()

        phys = self.power_imbalance(y_pred, x_input, edge_index, edge_attr)
        return self.alpha * mse + (1.0 - self.alpha) * self.physics_scale * phys
