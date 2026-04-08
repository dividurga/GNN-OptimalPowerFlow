"""
Data loading, preprocessing, and graph construction utilities.
Mirrors the paper's make_dataset / normalize_dataset methodology exactly.
"""

import os
import glob

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


def load_excel_datasets(dataset_dir: str, file_indices: list[int]) -> np.ndarray:
    """Load and concatenate rows from multiple PF_Dataset_N.xlsx files."""
    dfs = []
    for idx in file_indices:
        path = os.path.join(dataset_dir, f"PF_Dataset_{idx}.xlsx")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        dfs.append(pd.read_excel(path))
    df = pd.concat(dfs, ignore_index=True)
    # Drop the 'Dataset' and 'Data' label columns (first two cols from generation script)
    label_cols = [c for c in df.columns if c in ('Dataset', 'Data')]
    df = df.drop(columns=label_cols)
    return df.values.astype(np.float32)


def make_dataset(dataset: np.ndarray, n_bus: int):
    """
    Convert raw flat dataset rows into (x, y) tensors.

    x shape: [n_samples, n_bus, 7]  — P, Q, V, δ, is_pv, is_pq, is_slack
    y shape: [n_samples, n_bus, 2]  — V, δ  (Newton-Raphson ground truth)
    """
    x_raw, y_raw = [], []

    for i in range(len(dataset)):
        x_sample, y_sample = [], []
        for n in range(n_bus):
            is_slack = int(n == 0)
            is_pv    = int(n != 0 and dataset[i, 4 * n + 2] == 0)
            is_pq    = int(n != 0 and dataset[i, 4 * n + 2] != 0)

            x_sample.append([
                dataset[i, 4 * n + 1],  # P
                dataset[i, 4 * n + 2],  # Q
                dataset[i, 4 * n + 3],  # V
                dataset[i, 4 * n + 4],  # δ
                float(is_pv),
                float(is_pq),
                float(is_slack),
            ])
            y_sample.append([
                dataset[i, 4 * n + 3],  # V  (target)
                dataset[i, 4 * n + 4],  # δ  (target)
            ])

        x_raw.append(x_sample)
        y_raw.append(y_sample)

    x = torch.tensor(x_raw, dtype=torch.float)
    y = torch.tensor(y_raw, dtype=torch.float)
    return x, y


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
    """Build bidirectional edge index from pandapower network topology."""
    net = NETWORK_MAP[n_bus]()
    from_buses = net.line['from_bus'].values.tolist()
    to_buses   = net.line['to_bus'].values.tolist()
    edge_index = torch.tensor(
        [from_buses + to_buses, to_buses + from_buses], dtype=torch.long
    )
    return edge_index


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
    """
    edge_index = build_edge_index(n_bus)

    raw_train = load_excel_datasets(dataset_dir, train_indices)
    raw_val   = load_excel_datasets(dataset_dir, val_indices)
    raw_test  = load_excel_datasets(dataset_dir, test_indices)

    x_train, y_train = make_dataset(raw_train, n_bus)
    x_val,   y_val   = make_dataset(raw_val,   n_bus)
    x_test,  y_test  = make_dataset(raw_test,  n_bus)

    x_train, y_train, x_mean, y_mean, x_std, y_std = normalize_dataset(x_train, y_train)

    # Normalise val/test with training statistics
    x_val  = (x_val  - x_mean) / x_std;  x_val[:, :, 4:]  = (x_val  * x_std + x_mean)[:, :, 4:]
    x_test = (x_test - x_mean) / x_std;  x_test[:, :, 4:] = (x_test * x_std + x_mean)[:, :, 4:]

    def _fix_flags(x_norm, x_raw):
        """Restore un-normalised bus-type flags after applying train stats."""
        x_norm[:, :, 4:] = x_raw[:, :, 4:]
        return x_norm

    x_raw_val,  _ = make_dataset(raw_val,  n_bus)
    x_raw_test, _ = make_dataset(raw_test, n_bus)
    x_val  = _fix_flags((x_raw_val  - x_mean) / x_std, x_raw_val)
    x_test = _fix_flags((x_raw_test - x_mean) / x_std, x_raw_test)

    y_val  = (y_val  - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    def _to_loader(x, y, shuffle):
        data_list = [Data(x=xi, y=yi, edge_index=edge_index) for xi, yi in zip(x, y)]
        return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

    train_loader = _to_loader(x_train, y_train, shuffle=True)
    val_loader   = _to_loader(x_val,   y_val,   shuffle=False)
    test_loader  = _to_loader(x_test,  y_test,  shuffle=False)

    stats = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    return train_loader, val_loader, test_loader, stats
