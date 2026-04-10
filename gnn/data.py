"""
Data loading, preprocessing, and graph construction utilities.
Mirrors the paper's make_dataset / normalize_dataset methodology exactly.
"""

import os

import numpy as np
import pandas as pd
import torch
import pandapower.networks as nw
from torch_geometric.data import Data, DataLoader


NETWORK_MAP = {
    14:  nw.case14,
    30:  nw.case30,
    57:  nw.case57,
    118: nw.case118,
}


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
    label_cols = [c for c in df.columns if c in ('Dataset', 'Data')]
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
):
    """
    Convert raw flat dataset rows into (x, y) tensors and per-sample edge indices.

    x shape: [n_samples, n_bus, 7]  — P, Q, V, δ, is_pv, is_pq, is_slack
    y shape: [n_samples, n_bus, 2]  — V, δ  (Newton-Raphson ground truth)
    edge_indices: list of [2, n_active_edges] tensors, one per sample.
                  Uses the full base topology when no line_status is provided.
    """
    x_raw, y_raw, edge_indices = [], [], []
    base_ei = torch.tensor(
        [from_buses + to_buses, to_buses + from_buses], dtype=torch.long
    )

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
            fb = [from_buses[j] for j, a in enumerate(active) if a]
            tb = [to_buses[j]   for j, a in enumerate(active) if a]
            ei = torch.tensor([fb + tb, tb + fb], dtype=torch.long)
        else:
            ei = base_ei

        x_raw.append(x_sample)
        y_raw.append(y_sample)
        edge_indices.append(ei)

    x = torch.tensor(x_raw, dtype=torch.float)
    y = torch.tensor(y_raw, dtype=torch.float)
    return x, y, edge_indices


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
    """Build bidirectional edge index from pandapower network (all lines in service)."""
    net = NETWORK_MAP[n_bus]()
    from_buses = net.line['from_bus'].values.tolist()
    to_buses   = net.line['to_bus'].values.tolist()
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

    Each Data object carries its own edge_index, reflecting per-sample line
    outage status when line-status columns are present in the dataset files.
    Falls back to the full base topology for datasets without outage columns.
    """
    net = NETWORK_MAP[n_bus]()
    from_buses = net.line['from_bus'].values.tolist()
    to_buses   = net.line['to_bus'].values.tolist()

    raw_train, ls_train = load_excel_datasets(dataset_dir, train_indices)
    raw_val,   ls_val   = load_excel_datasets(dataset_dir, val_indices)
    raw_test,  ls_test  = load_excel_datasets(dataset_dir, test_indices)

    x_train, y_train, ei_train = make_dataset(raw_train, n_bus, ls_train, from_buses, to_buses)
    x_val,   y_val,   ei_val   = make_dataset(raw_val,   n_bus, ls_val,   from_buses, to_buses)
    x_test,  y_test,  ei_test  = make_dataset(raw_test,  n_bus, ls_test,  from_buses, to_buses)

    x_train, y_train, x_mean, y_mean, x_std, y_std = normalize_dataset(x_train, y_train)

    def _apply_train_stats(x_raw):
        x_norm = (x_raw - x_mean) / x_std
        x_norm[:, :, 4:] = x_raw[:, :, 4:]   # restore bus-type flags un-normalised
        return x_norm

    x_val  = _apply_train_stats(x_val)
    x_test = _apply_train_stats(x_test)
    y_val  = (y_val  - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    def _to_loader(x, y, edge_indices, shuffle):
        data_list = [Data(x=xi, y=yi, edge_index=ei) for xi, yi, ei in zip(x, y, edge_indices)]
        return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

    train_loader = _to_loader(x_train, y_train, ei_train, shuffle=True)
    val_loader   = _to_loader(x_val,   y_val,   ei_val,   shuffle=False)
    test_loader  = _to_loader(x_test,  y_test,  ei_test,  shuffle=False)

    stats = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    return train_loader, val_loader, test_loader, stats
