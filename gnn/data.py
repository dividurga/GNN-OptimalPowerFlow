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
    Return per-unit [R, X] and from/to bus indices for ALL branches in the
    base network — both lines AND transformers. Uses net._ppc['branch'] after
    pp.runpp(net); ppc branch ordering is: first `n_lines` rows are lines
    (matching net.line index), remaining rows are transformers.

    Returns:
        from_buses : list[int], length n_branches
        to_buses   : list[int], length n_branches
        rx         : np.ndarray, shape [n_branches, 2]  (R_pu, X_pu)
        n_lines    : int       — first n_lines entries are lines (outage-able);
                                  the rest are transformers (always in service)
        sn_mva     : float     — base MVA
    """
    net = NETWORK_MAP[n_bus]()
    pp.runpp(net)
    branch = net._ppc['branch']
    n_lines = len(net.line)
    from_buses = branch[:, 0].real.astype(int).tolist()
    to_buses   = branch[:, 1].real.astype(int).tolist()
    rx = np.stack([branch[:, 2].real.astype(np.float32),
                   branch[:, 3].real.astype(np.float32)], axis=1)
    return from_buses, to_buses, rx, n_lines, float(net.sn_mva)


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
    branch_rx: np.ndarray,
    n_lines: int,
):
    """
    Convert raw flat dataset rows into (x, y) tensors, per-sample edge indices,
    and per-sample edge attributes.

    `from_buses`/`to_buses`/`branch_rx` cover ALL branches (lines + transformers);
    the first `n_lines` entries are lines (subject to outage filtering via
    `line_status`), the rest are transformers (always in service).

    x shape: [n_samples, n_bus, 7]  — P, Q, V, δ, is_pv, is_pq, is_slack
    y shape: [n_samples, n_bus, 2]  — V, δ  (Newton-Raphson ground truth)
    edge_indices: list of [2, 2*n_active] tensors, one per sample.
    edge_attrs:   list of [2*n_active, 2] tensors, mirrored to match edge_index.
    """
    x_raw, y_raw, edge_indices, edge_attrs = [], [], [], []

    trafo_from = from_buses[n_lines:]
    trafo_to   = to_buses[n_lines:]
    trafo_rx   = branch_rx[n_lines:]
    line_from  = from_buses[:n_lines]
    line_to    = to_buses[:n_lines]
    line_rx    = branch_rx[:n_lines]

    base_fb = line_from + trafo_from
    base_tb = line_to   + trafo_to
    base_ei = torch.tensor([base_fb + base_tb, base_tb + base_fb], dtype=torch.long)
    base_rx = np.concatenate([line_rx, trafo_rx], axis=0)
    base_ea = torch.tensor(np.concatenate([base_rx, base_rx], axis=0), dtype=torch.float)

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
            active = line_status[i].astype(bool)
            fb = [line_from[j] for j, a in enumerate(active) if a] + trafo_from
            tb = [line_to[j]   for j, a in enumerate(active) if a] + trafo_to
            ei = torch.tensor([fb + tb, tb + fb], dtype=torch.long)
            rx_active = np.concatenate([line_rx[active], trafo_rx], axis=0)
            ea = torch.tensor(np.concatenate([rx_active, rx_active], axis=0), dtype=torch.float)
        else:
            ei = base_ei
            ea = base_ea

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
    from_buses, to_buses, _, _, _ = get_branch_info(n_bus)
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

    Each Data object carries its own edge_index and edge_attr (per-unit R, X),
    reflecting per-sample line outage status when line-status columns are
    present in the dataset files. Falls back to the full base topology for
    datasets without outage columns.

    stats dict keys:
        x_mean, x_std      — node feature stats [n_bus, 7]
        y_mean, y_std      — target stats [n_bus, 2]
        edge_mean, edge_std — edge attribute stats [2] (applied globally)
        sn_mva             — base MVA of the network (for physics loss)
    """
    from_buses, to_buses, branch_rx, n_lines, sn_mva = get_branch_info(n_bus)

    raw_train, ls_train = load_excel_datasets(dataset_dir, train_indices)
    raw_val,   ls_val   = load_excel_datasets(dataset_dir, val_indices)
    raw_test,  ls_test  = load_excel_datasets(dataset_dir, test_indices)

    x_train, y_train, ei_train, ea_train = make_dataset(raw_train, n_bus, ls_train, from_buses, to_buses, branch_rx, n_lines)
    x_val,   y_val,   ei_val,   ea_val   = make_dataset(raw_val,   n_bus, ls_val,   from_buses, to_buses, branch_rx, n_lines)
    x_test,  y_test,  ei_test,  ea_test  = make_dataset(raw_test,  n_bus, ls_test,  from_buses, to_buses, branch_rx, n_lines)

    x_train, y_train, x_mean, y_mean, x_std, y_std = normalize_dataset(x_train, y_train)

    # Edge attribute stats: computed globally across all training edges
    all_train_edges = torch.cat(ea_train, dim=0)
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

    def _to_loader(x, y, edge_indices, edge_attrs, shuffle):
        data_list = [
            Data(x=xi, y=yi, edge_index=ei, edge_attr=ea)
            for xi, yi, ei, ea in zip(x, y, edge_indices, edge_attrs)
        ]
        return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

    train_loader = _to_loader(x_train, y_train, ei_train, ea_train, shuffle=True)
    val_loader   = _to_loader(x_val,   y_val,   ei_val,   ea_val,   shuffle=False)
    test_loader  = _to_loader(x_test,  y_test,  ei_test,  ea_test,  shuffle=False)

    stats = dict(
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
        edge_mean=edge_mean, edge_std=edge_std, sn_mva=sn_mva,
        load_bus_mask=get_load_bus_mask(n_bus),
    )
    return train_loader, val_loader, test_loader, stats
