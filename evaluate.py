"""
Evaluation script for GNN-OptimalPowerFlow.

Loads a trained checkpoint and computes MSE, RMSE, NRMSE, MAE, R² on
the test set (or any specified dataset files), then saves results to CSV.

Usage:
    python evaluate.py --bus 14 --gnn_type SAGEConv
    python evaluate.py --bus 30 --gnn_type GAT --test_files 8 9 10
    python evaluate.py --bus 57 --gnn_type GCN --dataset_dir Datasets/57Bus_outages
"""

import argparse
import os

import torch
import pandas as pd

from gnn.data    import build_dataloaders
from gnn.model   import GNNPowerFlow, SUPPORTED_TYPES
from gnn.metrics import compute_all


def denormalize(t, mean, std):
    return t * std + mean


@torch.no_grad()
def collect_predictions(model, loader, y_mean, y_std, n_bus, device):
    model.eval()
    all_pred, all_true = [], []

    for batch in loader:
        batch = batch.to(device)
        y_pred = model(batch).view(-1, n_bus, 2)
        y_true = batch.y.view(-1, n_bus, 2)

        bs = y_pred.size(0)
        ym = y_mean.unsqueeze(0).expand(bs, -1, -1).to(device)
        ys = y_std.unsqueeze(0).expand(bs, -1, -1).to(device)

        all_pred.append(denormalize(y_pred, ym, ys).cpu())
        all_true.append(denormalize(y_true, ym, ys).cpu())

    return torch.cat(all_pred), torch.cat(all_true)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bus',         type=int,   default=14,          choices=[14, 30, 57, 118])
    parser.add_argument('--gnn_type',    type=str,   default='GraphConv', choices=list(SUPPORTED_TYPES))
    parser.add_argument('--dataset_dir', type=str,   default=None)
    parser.add_argument('--train_files', type=int,   nargs='+', default=list(range(1, 17)))
    parser.add_argument('--val_files',   type=int,   nargs='+', default=[17, 18])
    parser.add_argument('--test_files',  type=int,   nargs='+', default=[19, 20])
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--feat_size1',  type=int,   default=12)
    parser.add_argument('--feat_size2',  type=int,   default=12)
    parser.add_argument('--hidden_size', type=int,   default=128)
    parser.add_argument('--dropout',     type=float, default=0.0)
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--checkpoint',  type=str,   default=None,
                        help='Path to .pt checkpoint. Defaults to Results/<N>Bus_<gnn_type>/...')
    parser.add_argument('--output_dir',  type=str,   default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or os.path.join('Datasets', f'{args.bus}Bus')
    output_dir  = args.output_dir  or os.path.join('Results', f'{args.bus}Bus_{args.gnn_type}')
    checkpoint  = args.checkpoint  or os.path.join(output_dir, f'[{args.bus} bus] Best_{args.gnn_type}_model.pt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device      : {device}")
    print(f"Bus system  : IEEE {args.bus}-bus")
    print(f"GNN type    : {args.gnn_type}")
    print(f"Checkpoint  : {checkpoint}\n")

    # ------------------------------------------------------------------ data
    _, val_loader, test_loader, stats = build_dataloaders(
        dataset_dir   = dataset_dir,
        n_bus         = args.bus,
        train_indices = args.train_files,
        val_indices   = args.val_files,
        test_indices  = args.test_files,
        batch_size    = args.batch_size,
    )
    y_mean = stats['y_mean']
    y_std  = stats['y_std']

    # ----------------------------------------------------------------- model
    model = GNNPowerFlow(
        n_bus          = args.bus,
        feat_in        = 7,
        feat_size1     = args.feat_size1,
        feat_size2     = args.feat_size2,
        hidden_size    = args.hidden_size,
        gnn_type       = args.gnn_type,
        dropout        = args.dropout,
        use_batch_norm = not args.no_batch_norm,
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    print("Checkpoint loaded.\n")

    # --------------------------------------------------------------- evaluate
    results = {}
    for split, loader in [('val', val_loader), ('test', test_loader)]:
        y_pred, y_true = collect_predictions(model, loader, y_mean, y_std, args.bus, device)
        metrics = compute_all(y_pred, y_true)
        results[split] = metrics
        print(f"--- {split.upper()} SET ---")
        for k, v in metrics.items():
            print(f"  {k:6s}: {v:.6f}")
        print()

    # ------------------------------------------------------------------ save
    os.makedirs(output_dir, exist_ok=True)
    rows = [{'split': split, **m} for split, m in results.items()]
    df = pd.DataFrame(rows)
    out_csv = os.path.join(output_dir, f'[{args.bus} bus] Evaluation Metrics.csv')
    df.to_csv(out_csv, index=False)
    print(f"Metrics saved to: {out_csv}")


if __name__ == '__main__':
    main()
