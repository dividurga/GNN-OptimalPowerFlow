"""
Data loading, preprocessing, and graph construction utilities.
Mirrors the paper's make_dataset / normalize_dataset methodology exactly.
"""

import os

import numpy as np
import pandas as pd
import torch
import pandapower as pp
import pandapower.networks as nw
from torch_geometric.data import Data, DataLoader


NETWORK_MAP = {
    14:  nw.case14,
    30:  nw.case30,
    57:  nw.case57,
    118: nw.case118,
}


def get_branch_info(n_bus: int):
    """
    Return per-branch admittance-matrix entries + topology for ALL branches.

    Edge features are MATPOWER-style branch admittance matrix elements so the
    physics loss's `S = V_self · conj(Y_self·V_self + Y_mutual·V_target)` is
    exact for any branch (lines, off-tap transformers, phase shifters).
    Formulas follow MATPOWER / pypower `makeYbus`:

        z   = R + jX
        y   = 1/z
        τ   = TAP if TAP != 0 else 1.0     (SHIFT=0 on all IEEE cases)
        Y_ff = (y + j·B_c/2) / τ²
        Y_ft = -y / τ
        Y_tf = -y / τ
        Y_tt = y + j·B_c/2

    Each branch has TWO directed-edge feature rows:
      - forward (from→to):  self=Y_ff, mutual=Y_ft
      - reverse (to→from):  self=Y_tt, mutual=Y_tf

    Returns:
        from_buses : list[int], length n_branches (original branch direction)
        to_buses   : list[int], length n_branches
        edge_y_fwd : np.ndarray [n_branches, 4]  (G_self, B_self, G_mut, B_mut)
        edge_y_rev : np.ndarray [n_branches, 4]
        n_lines    : int — first n_lines branches are lines (outage-able);
                           the rest are transformers (always in service)
        sn_mva     : float — base MVA
    """
    net = NETWORK_MAP[n_bus]()
    pp.runpp(net)
    branch = net._ppc['branch']
    n_lines = len(net.line)

    R   = branch[:, 2].real.astype(np.float64)
    X   = branch[:, 3].real.astype(np.float64)
    B_c = branch[:, 4].real.astype(np.float64)
    tap = branch[:, 8].real.astype(np.float64)
    tap = np.where(tap == 0.0, 1.0, tap)

    z = R + 1j * X
    y = 1.0 / z
    Y_ff = (y + 1j * B_c / 2.0) / (tap ** 2)
    Y_tt = y + 1j * B_c / 2.0
    Y_ft = -y / tap
    Y_tf = -y / tap

    from_buses = branch[:, 0].real.astype(int).tolist()
    to_buses   = branch[:, 1].real.astype(int).tolist()

    edge_y_fwd = np.stack(
        [Y_ff.real, Y_ff.imag, Y_ft.real, Y_ft.imag], axis=1
    ).astype(np.float32)
    edge_y_rev = np.stack(
        [Y_tt.real, Y_tt.imag, Y_tf.real, Y_tf.imag], axis=1
    ).astype(np.float32)

    return from_buses, to_buses, edge_y_fwd, edge_y_rev, n_lines, float(net.sn_mva)


def get_load_bus_mask(n_bus: int) -> torch.Tensor:
    """
    Return a boolean mask of length n_bus that is True for buses WITHOUT
    generation (pure load / passive buses). The physics power-balance
    equation is reliable only at these buses, because the raw dataset
    stores per-bus LOAD demand but not generator output.
    """
    net = NETWORK_MAP[n_bus]()
    gen_buses = set(net.gen['bus'].tolist()) | set(net.ext_grid['bus'].tolist())
    mask = torch.tensor([i not in gen_buses for i in range(n_bus)], dtype=torch.bool)
    return mask


def get_bus_shunt_pu(n_bus: int) -> torch.Tensor:
    """
    Return per-bus shunt admittance `[n_bus, 2] = (G_sh, B_sh)` in per-unit
    on the system base. MATPOWER convention: ppc bus matrix columns 4 (GS)
    and 5 (BS) store shunt conductance/susceptance in MW / MVAr at V=1, so
    pu admittance is `(GS + jBS) / sn_mva`.

    Bus shunts (capacitors, reactors) are ground-connected admittances that
    inject/absorb reactive power; they don't appear as branches but do affect
    KCL at their host bus. Including them in `PowerImbalance.update()` closes
    the residual on buses like case14/bus-8.
    """
    net = NETWORK_MAP[n_bus]()
    pp.runpp(net)
    bus = net._ppc['bus']
    sn_mva = float(net.sn_mva)
    G_sh = bus[:n_bus, 4].real.astype(np.float32) / sn_mva
    B_sh = bus[:n_bus, 5].real.astype(np.float32) / sn_mva
    return torch.tensor(np.stack([G_sh, B_sh], axis=1), dtype=torch.float)


def load_excel_datasets(dataset_dir: str, file_indices: list[int]):
    """
    Load and concatenate rows from multiple PF_Dataset_N.xlsx files.

    Returns:
        bus_data   : np.ndarray, shape [n_samples, n_bus * 4]  (P, Q, V, δ per bus)
        line_status: np.ndarray or None, shape [n_samples, n_lines]  (1=in_service, 0=out)
                     None when the dataset has no line-outage columns (no-outage datasets).
    """
    dfs = []
    for idx in file_indices:
        path = os.path.join(dataset_dir, f"PF_Dataset_{idx}.xlsx")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        dfs.append(pd.read_excel(path))
    df = pd.concat(dfs, ignore_index=True)
    label_cols = [
        c for c in df.columns
        if c in ('Dataset', 'Data') or df[c].dtype == object or c.startswith('PF Dataset')
    ]
    df = df.drop(columns=label_cols)
    df = df.dropna()  # drop samples with any NaN (e.g. from disconnected buses)

    line_cols = [c for c in df.columns if c.startswith('line_') and c.endswith('_in_service')]
    bus_cols  = [c for c in df.columns if c not in line_cols]

    bus_data    = df[bus_cols].values.astype(np.float32)
    line_status = df[line_cols].values.astype(np.float32) if line_cols else None
    return bus_data, line_status


def make_dataset(
    dataset: np.ndarray,
    n_bus: int,
    line_status: np.ndarray | None,
    from_buses: list[int],
    to_buses: list[int],
    edge_y_fwd: np.ndarray,
    edge_y_rev: np.ndarray,
    n_lines: int,
):
    """
    Convert raw flat dataset rows into (x, y) tensors + per-sample edge
    indices and admittance edge attributes.

    `from_buses`/`to_buses`/`edge_y_fwd`/`edge_y_rev` cover ALL branches
    (lines + transformers). First `n_lines` branches are lines (subject to
    outage filtering via `line_status`); the rest are transformers (always
    in service).

    Each branch produces TWO directed edges in the final graph:
      - forward  (from_bus → to_bus): edge_attr row = edge_y_fwd[branch]
      - reverse  (to_bus → from_bus): edge_attr row = edge_y_rev[branch]

    Edge-attr columns: `[G_self, B_self, G_mutual, B_mutual]` per directed
    edge — MATPOWER admittance-matrix entries that the physics loss consumes
    directly.

    x shape: [n_samples, n_bus, 7]  — P, Q, V, δ, is_pv, is_pq, is_slack
    y shape: [n_samples, n_bus, 2]  — V, δ  (Newton-Raphson ground truth)
    edge_indices: list of [2, 2·n_active] tensors
    edge_attrs:   list of [2·n_active, 4] tensors
    """
    x_raw, y_raw = [], []
    edge_indices, edge_attrs = [], []

    fb_all = np.asarray(from_buses, dtype=int)
    tb_all = np.asarray(to_buses,   dtype=int)

    def _build_sample_edges(active_branch_mask: np.ndarray):
        """Return (ei, ea) for one sample. `active_branch_mask` is length n_branches."""
        fb = fb_all[active_branch_mask]
        tb = tb_all[active_branch_mask]
        yf = edge_y_fwd[active_branch_mask]   # [n_active, 4]
        yr = edge_y_rev[active_branch_mask]

        src = np.concatenate([fb, tb], axis=0)
        dst = np.concatenate([tb, fb], axis=0)
        ea = np.concatenate([yf, yr], axis=0)

        ei = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        ea = torch.tensor(ea, dtype=torch.float)
        return ei, ea

    base_active = np.ones(len(fb_all), dtype=bool)
    base_ei, base_ea = _build_sample_edges(base_active)

    for i in range(len(dataset)):
        x_sample, y_sample = [], []
        for n in range(n_bus):
            is_slack = int(n == 0)
            is_pv    = int(n != 0 and dataset[i, 4 * n + 1] == 0)
            is_pq    = int(n != 0 and dataset[i, 4 * n + 1] != 0)

            x_sample.append([
                dataset[i, 4 * n],      # P
                dataset[i, 4 * n + 1],  # Q
                dataset[i, 4 * n + 2],  # V
                dataset[i, 4 * n + 3],  # δ
                float(is_pv),
                float(is_pq),
                float(is_slack),
            ])
            y_sample.append([
                dataset[i, 4 * n + 2],  # V  (target)
                dataset[i, 4 * n + 3],  # δ  (target)
            ])

        if line_status is not None:
            active = base_active.copy()
            active[:n_lines] = line_status[i].astype(bool)
            ei, ea = _build_sample_edges(active)
        else:
            ei, ea = base_ei, base_ea

        x_raw.append(x_sample)
        y_raw.append(y_sample)
        edge_indices.append(ei)
        edge_attrs.append(ea)

    x = torch.tensor(x_raw, dtype=torch.float)
    y = torch.tensor(y_raw, dtype=torch.float)
    return x, y, edge_indices, edge_attrs


def normalize_dataset(x: torch.Tensor, y: torch.Tensor):
    """
    Z-score normalise x and y.  Bus-type flag columns (4, 5, 6) in x are left
    unchanged since they are categorical {0, 1}.

    Returns: x_norm, y_norm, x_mean, y_mean, x_std, y_std
    """
    x_mean, x_std = x.mean(0), x.std(0)
    y_mean, y_std = y.mean(0), y.std(0)

    x_std[x_std == 0] = 1
    y_std[y_std == 0] = 1

    x_norm = (x - x_mean) / x_std
    x_norm[:, :, 4:] = x[:, :, 4:]   # restore bus-type flags un-normalised

    y_norm = (y - y_mean) / y_std
    return x_norm, y_norm, x_mean, y_mean, x_std, y_std


def denormalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return tensor * std + mean


def build_edge_index(n_bus: int) -> torch.Tensor:
    """Build bidirectional edge index from the base network (lines + transformers, all in service)."""
    from_buses, to_buses, *_ = get_branch_info(n_bus)
    return torch.tensor(
        [from_buses + to_buses, to_buses + from_buses], dtype=torch.long
    )


def build_dataloaders(
    dataset_dir: str,
    n_bus: int,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    batch_size: int = 16,
):
    """
    Load datasets, build (x, y) tensors, normalise, and return DataLoaders
    plus the training normalisation statistics needed for evaluation.

    Each `Data` object carries its own `edge_index` and `edge_attr`
    (per-directed-edge admittance `[G_self, B_self, G_mutual, B_mutual]`),
    reflecting per-sample line outage status when `line_<n>_in_service`
    columns are present; falls back to the full topology otherwise.

    stats dict keys:
        x_mean, x_std      — node feature stats [n_bus, 7]
        y_mean, y_std      — target stats [n_bus, 2]
        edge_mean, edge_std — edge attribute stats [4] (per-column z-score)
        sn_mva             — base MVA of the network (for physics loss)
        load_bus_mask      — boolean mask of buses without generators
    """
    from_buses, to_buses, edge_y_fwd, edge_y_rev, n_lines, sn_mva = get_branch_info(n_bus)

    raw_train, ls_train = load_excel_datasets(dataset_dir, train_indices)
    raw_val,   ls_val   = load_excel_datasets(dataset_dir, val_indices)
    raw_test,  ls_test  = load_excel_datasets(dataset_dir, test_indices)

    x_train, y_train, ei_train, ea_train = make_dataset(
        raw_train, n_bus, ls_train, from_buses, to_buses, edge_y_fwd, edge_y_rev, n_lines)
    x_val,   y_val,   ei_val,   ea_val   = make_dataset(
        raw_val,   n_bus, ls_val,   from_buses, to_buses, edge_y_fwd, edge_y_rev, n_lines)
    x_test,  y_test,  ei_test,  ea_test  = make_dataset(
        raw_test,  n_bus, ls_test,  from_buses, to_buses, edge_y_fwd, edge_y_rev, n_lines)

    x_train, y_train, x_mean, y_mean, x_std, y_std = normalize_dataset(x_train, y_train)

    # Edge attribute stats: per-column z-score across all training directed edges.
    all_train_edges = torch.cat(ea_train, dim=0)     # [total_edges, 4]
    edge_mean = all_train_edges.mean(0)
    edge_std  = all_train_edges.std(0)
    edge_std[edge_std == 0] = 1

    def _normalize_edges(ea_list):
        return [(ea - edge_mean) / edge_std for ea in ea_list]

    ea_train = _normalize_edges(ea_train)
    ea_val   = _normalize_edges(ea_val)
    ea_test  = _normalize_edges(ea_test)

    def _apply_train_stats(x_raw):
        x_norm = (x_raw - x_mean) / x_std
        x_norm[:, :, 4:] = x_raw[:, :, 4:]   # restore bus-type flags un-normalised
        return x_norm

    x_val  = _apply_train_stats(x_val)
    x_test = _apply_train_stats(x_test)
    y_val  = (y_val  - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    def _to_loader(x, y, eis, eas, shuffle):
        data_list = [
            Data(x=xi, y=yi, edge_index=ei, edge_attr=ea)
            for xi, yi, ei, ea in zip(x, y, eis, eas)
        ]
        return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

    train_loader = _to_loader(x_train, y_train, ei_train, ea_train, shuffle=True)
    val_loader   = _to_loader(x_val,   y_val,   ei_val,   ea_val,   shuffle=False)
    test_loader  = _to_loader(x_test,  y_test,  ei_test,  ea_test,  shuffle=False)

    stats = dict(
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        edge_mean=edge_mean, edge_std=edge_std, sn_mva=sn_mva,
        load_bus_mask=get_load_bus_mask(n_bus),
        bus_shunt_pu=get_bus_shunt_pu(n_bus),
    )
    return train_loader, val_loader, test_loader, stats
