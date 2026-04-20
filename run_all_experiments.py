#!/usr/bin/env python3
"""
run_all_experiments.py — Master experiment runner for GNN-OptimalPowerFlow.

Trains all models on all specified bus systems, evaluates across every
generalisation axis defined in FOR_NOAH_ALL_TESTS.md, aggregates results
into a single CSV, and generates comparison metric plots.

Models  : GCN, GraphConv, SAGEConv, GATConv, ChebConv  (MSE loss)
          MPN                                            (mixed MSE+physics)

Training sets
  train_normal  →  test_indist, test_topo, test_load_tail,
                   test_load_extreme, test_corr
  train_n       →  test_nk1, test_nk2, test_nk3

Usage
-----
  # Full run (all buses, all models):
  python run_all_experiments.py

  # Single bus (e.g. from SLURM):
  python run_all_experiments.py --buses 14

  # Quick smoke-test:
  python run_all_experiments.py --buses 14 --models GraphConv MPN --epochs 50

  # Skip training, re-plot from existing results:
  python run_all_experiments.py --skip_training

  # Force CUDA:
  python run_all_experiments.py --device cuda
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader

from gnn.data import (
    get_branch_info,
    load_excel_datasets,
    make_dataset,
    build_dataloaders,
)
from gnn.metrics import compute_all
from gnn.model import GNNPowerFlow, SUPPORTED_TYPES
from gnn.mpn import MPN
from gnn.physics_loss import MixedMSEPowerImbalance


# ─────────────────────────── constants ──────────────────────────────────────

ALL_GNN_TYPES = list(SUPPORTED_TYPES) + ['MPN']

TRAIN_NORMAL_AXES = {
    'test_indist':       'In-dist\n(baseline)',
    'test_topo':         'Topology\n(unseen lines)',
    'test_load_tail':    'Load Tail\n[40–80%]',
    'test_load_extreme': 'Load Extreme\n[80–120%]',
    'test_corr':         'Correlated\nLoads',
}

TRAIN_N_AXES = {
    'test_nk1': 'N-1',
    'test_nk2': 'N-2',
    'test_nk3': 'N-3',
}

COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
MODEL_COLORS = {m: COLORS[i % len(COLORS)] for i, m in enumerate(ALL_GNN_TYPES)}


# ─────────────────────────── data helpers ───────────────────────────────────

def make_test_loader(test_dir, test_files, n_bus, branch_info, stats, batch_size):
    """
    Load a test dataset and normalise with TRAINING stats — not re-fit from the
    test set itself. This is critical for OOD evaluation to be meaningful.
    """
    from_buses, to_buses, branch_rx, n_lines, _ = branch_info
    x_mean, x_std = stats['x_mean'], stats['x_std']
    y_mean, y_std = stats['y_mean'], stats['y_std']
    edge_mean, edge_std = stats['edge_mean'], stats['edge_std']

    raw, ls = load_excel_datasets(test_dir, test_files)
    x, y, edge_indices, edge_attrs = make_dataset(
        raw, n_bus, ls, from_buses, to_buses, branch_rx, n_lines
    )

    x_norm = (x - x_mean) / x_std
    x_norm[:, :, 4:] = x[:, :, 4:]   # restore bus-type flags (categorical, not normalised)
    y_norm = (y - y_mean) / y_std
    edge_attrs = [(ea - edge_mean) / edge_std for ea in edge_attrs]

    data_list = [
        Data(x=xi, y=yi, edge_index=ei, edge_attr=ea)
        for xi, yi, ei, ea in zip(x_norm, y_norm, edge_indices, edge_attrs)
    ]
    return DataLoader(data_list, batch_size=batch_size, shuffle=False)


# ─────────────────────────── model helpers ──────────────────────────────────

def build_model(gnn_type, n_bus, device):
    """Construct model; moves GATConv to CPU on MPS (known instability)."""
    if gnn_type == 'MPN':
        model = MPN(
            n_bus=n_bus, nfeature_dim=7, efeature_dim=2,
            hidden_dim=129, n_gnn_layers=4, K=3, dropout_rate=0.0,
        )
    else:
        model = GNNPowerFlow(
            n_bus=n_bus, feat_in=7, feat_size1=12, feat_size2=12,
            hidden_size=128, gnn_type=gnn_type, dropout=0.0, use_batch_norm=True,
        )
    if gnn_type == 'GATConv' and device.type == 'mps':
        return model.to('cpu'), torch.device('cpu')
    return model.to(device), device


def denorm_mse(y_pred, y_true, y_mean, y_std, batch_size):
    ym = y_mean.view(-1).repeat(batch_size)
    ys = y_std.view(-1).repeat(batch_size)
    p = y_pred.view(-1) * ys + ym
    t = y_true.view(-1) * ys + ym
    return torch.mean((p - t) ** 2)


# ─────────────────────────── training ───────────────────────────────────────

def train_model(gnn_type, n_bus, dataset_dir, output_dir,
                device, epochs, patience, batch_size, lr, weight_decay):
    """
    Train one model on dataset_dir.  Skips if checkpoint already exists.

    Saves:
      best_{gnn_type}_model.pt   — best checkpoint by val MSE
      train_stats.pt             — normalisation stats needed for evaluation
      loss_curves.png
    Returns (checkpoint_path, stats_dict).
    """
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path  = os.path.join(output_dir, f'best_{gnn_type}_model.pt')
    stats_path = os.path.join(output_dir, 'train_stats.pt')

    if os.path.exists(ckpt_path) and os.path.exists(stats_path):
        print(f"    [cached] {ckpt_path}")
        return ckpt_path, torch.load(stats_path, map_location='cpu')

    # train_normal/train_n have 8 files (1–7 train, 8 val).
    # We pass val indices as the dummy test split so build_dataloaders doesn't
    # look for a separate test file in these directories.
    train_loader, val_loader, _, stats = build_dataloaders(
        dataset_dir=dataset_dir,
        n_bus=n_bus,
        train_indices=list(range(1, 8)),
        val_indices=[8],
        test_indices=[8],
        batch_size=batch_size,
    )
    torch.save(stats, stats_path)

    model, mdev = build_model(gnn_type, n_bus, device)
    y_mean = stats['y_mean'].to(mdev)
    y_std  = stats['y_std'].to(mdev)

    loss_fn = None
    if gnn_type == 'MPN':
        loss_fn = MixedMSEPowerImbalance(
            x_mean=stats['x_mean'], x_std=stats['x_std'],
            y_mean=stats['y_mean'], y_std=stats['y_std'],
            edge_mean=stats['edge_mean'], edge_std=stats['edge_std'],
            sn_mva=stats['sn_mva'],
            load_bus_mask=stats['load_bus_mask'],
            alpha=0.9, physics_scale=0.02,
        ).to(mdev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30
    )

    best_val, best_epoch, no_improve = float('inf'), 0, 0
    train_hist, val_hist = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(mdev)
            optimizer.zero_grad()
            y_pred = model(batch)
            if loss_fn is not None:
                loss = loss_fn(y_pred, batch.y.view(batch.num_graphs, -1),
                               batch.x, batch.edge_index, batch.edge_attr)
            else:
                loss = denorm_mse(y_pred, batch.y, y_mean, y_std, batch.num_graphs)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.num_graphs
        train_loss = total / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            vt = 0.0
            for batch in val_loader:
                batch = batch.to(mdev)
                vt += denorm_mse(model(batch), batch.y, y_mean, y_std,
                                 batch.num_graphs).item() * batch.num_graphs
        val_loss = vt / len(val_loader.dataset)

        train_hist.append(train_loss)
        val_hist.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, best_epoch, no_improve = val_loss, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    early stop @ epoch {epoch} (best epoch {best_epoch})")
                break

        if epoch % 50 == 0 or epoch == 1:
            print(f"    ep {epoch:4d} | train {train_loss:.5f} | val {val_loss:.5f}")

    plt.figure(figsize=(9, 4))
    plt.plot(train_hist, label='Train')
    plt.plot(val_hist,   label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (denorm)')
    plt.title(f'IEEE {n_bus}-bus | {gnn_type}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=150)
    plt.close()

    print(f"    best val {best_val:.5f} @ epoch {best_epoch}")
    return ckpt_path, stats


# ─────────────────────────── evaluation ─────────────────────────────────────

@torch.no_grad()
def evaluate_loader(model, loader, y_mean, y_std, n_bus, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        yp = model(batch).view(-1, n_bus, 2)
        yt = batch.y.view(-1, n_bus, 2)
        bs = yp.size(0)
        ym = y_mean.unsqueeze(0).expand(bs, -1, -1)
        ys = y_std.unsqueeze(0).expand(bs, -1, -1)
        preds.append((yp * ys + ym).cpu())
        trues.append((yt * ys + ym).cpu())
    return compute_all(torch.cat(preds), torch.cat(trues))


# ─────────────────────────── plotting ───────────────────────────────────────

def _bar_chart(df_sub, bus, metric, axes_map, train_set, out_path):
    test_dirs = [d for d in axes_map if not df_sub[df_sub['test_dir'] == d].empty]
    if not test_dirs:
        return
    models = df_sub['gnn_type'].unique()
    x = np.arange(len(test_dirs))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(test_dirs)), 5))
    for i, m in enumerate(models):
        vals = [
            df_sub[(df_sub['gnn_type'] == m) & (df_sub['test_dir'] == td)][metric].values
            for td in test_dirs
        ]
        vals = [v[0] if len(v) else np.nan for v in vals]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=m,
               color=MODEL_COLORS.get(m, '#888888'))

    ax.set_xticks(x)
    ax.set_xticklabels([axes_map[d] for d in test_dirs], fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(f'IEEE {bus}-bus  |  {metric}  (trained on {train_set})')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _nk_line(df_sub, bus, metric, out_path):
    models = df_sub['gnn_type'].unique()
    k_dirs = ['test_nk1', 'test_nk2', 'test_nk3']
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in models:
        vals = [
            df_sub[(df_sub['gnn_type'] == m) & (df_sub['test_dir'] == d)][metric].values
            for d in k_dirs
        ]
        vals = [v[0] if len(v) else np.nan for v in vals]
        ax.plot([1, 2, 3], vals, marker='o', label=m,
                color=MODEL_COLORS.get(m, '#888888'))
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel('N-k depth (k)')
    ax.set_ylabel(metric)
    ax.set_title(f'IEEE {bus}-bus  |  N-k Degradation  —  {metric}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _heatmap(df_sub, bus, metric, axes_map, train_set, out_path):
    test_dirs = list(axes_map.keys())
    models = sorted(df_sub['gnn_type'].unique())
    data = np.full((len(models), len(test_dirs)), np.nan)
    for i, m in enumerate(models):
        for j, td in enumerate(test_dirs):
            v = df_sub[(df_sub['gnn_type'] == m) & (df_sub['test_dir'] == td)][metric].values
            if len(v):
                data[i, j] = v[0]

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(test_dirs)),
                                    max(3, 0.65 * len(models))))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, ax=ax, label=metric)
    labels = [axes_map[d].replace('\n', ' ') for d in test_dirs]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models, fontsize=9)
    vmax = np.nanmax(data)
    for i in range(len(models)):
        for j in range(len(test_dirs)):
            if not np.isnan(data[i, j]):
                color = 'white' if vmax > 0 and data[i, j] > vmax * 0.7 else 'black'
                ax.text(j, i, f'{data[i, j]:.4f}', ha='center', va='center',
                        fontsize=7, color=color)
    ax.set_title(f'IEEE {bus}-bus  |  {metric} heatmap  (trained on {train_set})')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _cross_bus_bar(df, buses, models, metric, test_dir, train_set, out_path):
    """One chart per test axis showing all bus sizes side-by-side."""
    x = np.arange(len(buses))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(buses)), 5))
    for i, m in enumerate(models):
        vals = [
            df[(df['bus'] == b) & (df['gnn_type'] == m) &
               (df['train_set'] == train_set) & (df['test_dir'] == test_dir)][metric].values
            for b in buses
        ]
        vals = [v[0] if len(v) else np.nan for v in vals]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, label=m,
               color=MODEL_COLORS.get(m, '#888888'))
    ax.set_xticks(x)
    ax.set_xticklabels([f'{b}-bus' for b in buses])
    ax.set_ylabel(metric)
    ax.set_title(f'{test_dir}  |  {metric}  (trained on {train_set})')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_plots(df, buses, models, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    metrics = ['NRMSE', 'R2', 'MAE']

    for bus in buses:
        df_n  = df[(df['bus'] == bus) & (df['train_set'] == 'train_normal')]
        df_nk = df[(df['bus'] == bus) & (df['train_set'] == 'train_n')]

        for metric in metrics:
            if not df_n.empty:
                _bar_chart(df_n, bus, metric, TRAIN_NORMAL_AXES, 'train_normal',
                           os.path.join(plots_dir, f'{bus}bus_{metric}_bar.png'))
                _heatmap(df_n, bus, metric, TRAIN_NORMAL_AXES, 'train_normal',
                         os.path.join(plots_dir, f'{bus}bus_{metric}_heatmap.png'))
            if not df_nk.empty:
                _nk_line(df_nk, bus, metric,
                         os.path.join(plots_dir, f'{bus}bus_{metric}_nk.png'))

    # Cross-bus summary: all bus sizes on one chart per test axis
    if len(buses) > 1:
        for metric in metrics:
            for test_dir in TRAIN_NORMAL_AXES:
                _cross_bus_bar(df, buses, models, metric, test_dir, 'train_normal',
                               os.path.join(plots_dir,
                                            f'allbuses_{metric}_{test_dir}.png'))
            for test_dir in TRAIN_N_AXES:
                _cross_bus_bar(df, buses, models, metric, test_dir, 'train_n',
                               os.path.join(plots_dir,
                                            f'allbuses_{metric}_{test_dir}.png'))

    print(f"Plots saved to: {plots_dir}/")


# ─────────────────────────── main ───────────────────────────────────────────

def get_device(override):
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(
        description='Train + evaluate all GNN models across all generalisation axes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--buses',        type=int, nargs='+', default=[14, 30, 57, 118],
                        help='Which bus systems to run')
    parser.add_argument('--models',       type=str, nargs='+', default=ALL_GNN_TYPES,
                        choices=ALL_GNN_TYPES, metavar='MODEL',
                        help='Which GNN types to train')
    parser.add_argument('--datasets_dir', type=str, default='Datasets',
                        help='Root directory containing NNBus/ subdirectories')
    parser.add_argument('--results_dir',  type=str, default='Results',
                        help='Root directory for checkpoints, stats, and plots')
    parser.add_argument('--epochs',       type=int,   default=500)
    parser.add_argument('--patience',     type=int,   default=100)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--skip_training', action='store_true',
                        help='Do not train; only evaluate existing checkpoints and re-plot')
    parser.add_argument('--skip_plotting', action='store_true',
                        help='Do not generate plots (useful for headless quick runs)')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help='Force a device (default: auto-detect cuda > mps > cpu)')
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device       : {device}")
    print(f"Buses        : {args.buses}")
    print(f"Models       : {args.models}")
    print(f"Datasets dir : {args.datasets_dir}")
    print(f"Results dir  : {args.results_dir}\n")

    records = []

    for bus in args.buses:
        bus_dir     = os.path.join(args.datasets_dir, f'{bus}Bus')
        branch_info = get_branch_info(bus)   # topology is the same for all experiments on this bus

        print(f"\n{'='*64}")
        print(f"  IEEE {bus}-bus")
        print(f"{'='*64}")

        # ── Experiment 1: train on train_normal, eval on generalisation axes ──
        train_normal_dir = os.path.join(bus_dir, 'train_normal')
        if not os.path.isdir(train_normal_dir):
            print(f"  [WARN] {train_normal_dir} not found — skipping train_normal experiments")
        else:
            for gnn_type in args.models:
                out_dir = os.path.join(args.results_dir, f'{bus}Bus_{gnn_type}_trainNormal')
                print(f"\n  ▶ {gnn_type} / train_normal")

                if args.skip_training:
                    ckpt       = os.path.join(out_dir, f'best_{gnn_type}_model.pt')
                    stats_path = os.path.join(out_dir, 'train_stats.pt')
                    if not (os.path.exists(ckpt) and os.path.exists(stats_path)):
                        print(f"    [WARN] checkpoint or stats not found, skipping")
                        continue
                    stats = torch.load(stats_path, map_location='cpu')
                else:
                    ckpt, stats = train_model(
                        gnn_type, bus, train_normal_dir, out_dir,
                        device, args.epochs, args.patience,
                        args.batch_size, args.lr, args.weight_decay,
                    )

                model, mdev = build_model(gnn_type, bus, device)
                model.load_state_dict(torch.load(ckpt, map_location=mdev))
                y_mean = stats['y_mean'].to(mdev)
                y_std  = stats['y_std'].to(mdev)

                for test_subdir, label in TRAIN_NORMAL_AXES.items():
                    test_path = os.path.join(bus_dir, test_subdir)
                    if not os.path.isdir(test_path):
                        print(f"    [skip] {test_subdir} not found")
                        continue
                    try:
                        loader = make_test_loader(
                            test_path, [1, 2, 3, 4], bus,
                            branch_info, stats, args.batch_size,
                        )
                        m = evaluate_loader(model, loader, y_mean, y_std, bus, mdev)
                        tag = label.replace('\n', ' ')
                        print(f"    {test_subdir:22s}  NRMSE={m['NRMSE']:.4f}  R²={m['R2']:.4f}")
                        records.append(dict(
                            bus=bus, gnn_type=gnn_type,
                            train_set='train_normal', test_dir=test_subdir,
                            label=tag, **m,
                        ))
                    except Exception as e:
                        print(f"    [ERROR] {test_subdir}: {e}")

        # ── Experiment 2: train on train_n, eval on N-k axes ──────────────
        train_n_dir = os.path.join(bus_dir, 'train_n')
        if not os.path.isdir(train_n_dir):
            print(f"  [WARN] {train_n_dir} not found — skipping N-k experiments")
        else:
            for gnn_type in args.models:
                out_dir = os.path.join(args.results_dir, f'{bus}Bus_{gnn_type}_trainN')
                print(f"\n  ▶ {gnn_type} / train_n")

                if args.skip_training:
                    ckpt       = os.path.join(out_dir, f'best_{gnn_type}_model.pt')
                    stats_path = os.path.join(out_dir, 'train_stats.pt')
                    if not (os.path.exists(ckpt) and os.path.exists(stats_path)):
                        print(f"    [WARN] checkpoint or stats not found, skipping")
                        continue
                    stats = torch.load(stats_path, map_location='cpu')
                else:
                    ckpt, stats = train_model(
                        gnn_type, bus, train_n_dir, out_dir,
                        device, args.epochs, args.patience,
                        args.batch_size, args.lr, args.weight_decay,
                    )

                model, mdev = build_model(gnn_type, bus, device)
                model.load_state_dict(torch.load(ckpt, map_location=mdev))
                y_mean = stats['y_mean'].to(mdev)
                y_std  = stats['y_std'].to(mdev)

                for test_subdir, label in TRAIN_N_AXES.items():
                    test_path = os.path.join(bus_dir, test_subdir)
                    if not os.path.isdir(test_path):
                        print(f"    [skip] {test_subdir} not found")
                        continue
                    try:
                        loader = make_test_loader(
                            test_path, [1, 2, 3, 4], bus,
                            branch_info, stats, args.batch_size,
                        )
                        m = evaluate_loader(model, loader, y_mean, y_std, bus, mdev)
                        print(f"    {test_subdir:22s}  NRMSE={m['NRMSE']:.4f}  R²={m['R2']:.4f}")
                        records.append(dict(
                            bus=bus, gnn_type=gnn_type,
                            train_set='train_n', test_dir=test_subdir,
                            label=label, **m,
                        ))
                    except Exception as e:
                        print(f"    [ERROR] {test_subdir}: {e}")

    # ── Save aggregate CSV ────────────────────────────────────────────────
    df = pd.DataFrame(records)
    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, 'all_experiments.csv')

    # Merge with any previously saved results from other bus runs
    if os.path.exists(csv_path) and not df.empty:
        prev = pd.read_csv(csv_path)
        # Drop stale rows for buses we just re-ran
        prev = prev[~prev['bus'].isin(args.buses)]
        df = pd.concat([prev, df], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"\nAll results saved to: {csv_path}")
    if not df.empty:
        print(df[['bus', 'gnn_type', 'train_set', 'test_dir',
                   'NRMSE', 'R2', 'MAE']].to_string(index=False))

    if args.skip_plotting or df.empty:
        return

    # Use all buses present in the full CSV for cross-bus plots
    all_buses = sorted(df['bus'].unique().tolist())
    generate_plots(df, all_buses, args.models,
                   os.path.join(args.results_dir, 'plots'))


if __name__ == '__main__':
    main()
