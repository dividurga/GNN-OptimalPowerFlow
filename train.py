"""
Training script for GNN-OptimalPowerFlow.

Trains one GNN model (selectable architecture) on a chosen IEEE bus system,
saves the best checkpoint, and logs train/val loss curves.

Usage:
    python train.py --bus 14 --gnn_type SAGEConv
    python train.py --bus 30 --gnn_type GAT --epochs 800 --batch_size 32
    python train.py --bus 57 --gnn_type GCN --dataset_dir Datasets/57Bus_outages
"""

import argparse
import json
import os

import torch
import matplotlib.pyplot as plt

from gnn.data    import build_dataloaders
from gnn.model   import GNNPowerFlow, SUPPORTED_TYPES


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def denormalize(t, mean, std):
    return t * std + mean


def train_one_epoch(model, loader, optimizer, y_mean, y_std, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        y_pred = model(batch).view(-1)
        y_true = batch.y.view(-1)

        bs = batch.num_graphs
        n_bus = model.n_bus

        y_mean_exp = y_mean.view(-1).repeat(bs).to(device)
        y_std_exp  = y_std.view(-1).repeat(bs).to(device)

        loss = torch.mean(
            (denormalize(y_pred, y_mean_exp, y_std_exp) -
             denormalize(y_true, y_mean_exp, y_std_exp)) ** 2
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * bs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, y_mean, y_std, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        y_pred = model(batch).view(-1)
        y_true = batch.y.view(-1)
        bs = batch.num_graphs

        y_mean_exp = y_mean.view(-1).repeat(bs).to(device)
        y_std_exp  = y_std.view(-1).repeat(bs).to(device)

        loss = torch.mean(
            (denormalize(y_pred, y_mean_exp, y_std_exp) -
             denormalize(y_true, y_mean_exp, y_std_exp)) ** 2
        )
        total_loss += loss.item() * bs

    return total_loss / len(loader.dataset)


def save_loss_plot(train_losses, val_losses, out_path, title):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (denormalised)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bus',         type=int,   default=14,           choices=[14, 30, 57, 118])
    parser.add_argument('--gnn_type',    type=str,   default='GraphConv',  choices=list(SUPPORTED_TYPES))
    parser.add_argument('--dataset_dir', type=str,   default=None,
                        help='Path to dataset directory. Defaults to PertubedDatasets/<N>Bus/')
    parser.add_argument('--train_files', type=int,   nargs='+', default=list(range(1, 17)),
                        help='Dataset file indices to use for training (default: 1-16)')
    parser.add_argument('--val_files',   type=int,   nargs='+', default=[17, 18],
                        help='Dataset file indices to use for validation (default: 17-18)')
    parser.add_argument('--test_files',  type=int,   nargs='+', default=[19, 20],
                        help='Dataset file indices to use for testing (default: 19-20)')
    parser.add_argument('--epochs',      type=int,   default=800)
    parser.add_argument('--patience',    type=int,   default=100)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--lr',          type=float, default=5e-5)
    parser.add_argument('--weight_decay',type=float, default=1e-6)
    parser.add_argument('--feat_size1',  type=int,   default=12)
    parser.add_argument('--feat_size2',  type=int,   default=12)
    parser.add_argument('--hidden_size', type=int,   default=128)
    parser.add_argument('--dropout',     type=float, default=0.0)
    parser.add_argument('--no_batch_norm', action='store_true')
    parser.add_argument('--output_dir',  type=str,   default=None,
                        help='Where to save results. Defaults to Results/<N>Bus_<gnn_type>/')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or os.path.join('PerturbedDatasets', f'{args.bus}Bus')
    if args.output_dir:
        output_dir = args.output_dir
    else:
        suffix = '_outages' if 'outage' in dataset_dir.lower() else ''
        output_dir = os.path.join('Results', f'{args.bus}Bus_{args.gnn_type}{suffix}')
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device      : {device}")
    print(f"Bus system  : IEEE {args.bus}-bus")
    print(f"GNN type    : {args.gnn_type}")
    print(f"Dataset dir : {dataset_dir}")
    print(f"Output dir  : {output_dir}\n")

    # ------------------------------------------------------------------ data
    train_loader, val_loader, test_loader, stats = build_dataloaders(
        dataset_dir  = dataset_dir,
        n_bus        = args.bus,
        train_indices = args.train_files,
        val_indices   = args.val_files,
        test_indices  = args.test_files,
        batch_size    = args.batch_size,
    )
    y_mean = stats['y_mean'].to(device)
    y_std  = stats['y_std'].to(device)

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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, verbose=True
    )

    # --------------------------------------------------------------- training
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_count = 0
    best_epoch = 0
    ckpt_path = os.path.join(output_dir, f'[{args.bus} bus] Best_{args.gnn_type}_model.pt')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, y_mean, y_std, device)
        val_loss   = eval_one_epoch(model, val_loader, y_mean, y_std, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch} | Best epoch: {best_epoch}")
                break

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Best: {best_val_loss:.6f}")

    print(f"\nBest val loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"Checkpoint saved to: {ckpt_path}")

    # ------------------------------------------------------------ loss curves
    plot_path = os.path.join(output_dir, f'[{args.bus} bus] Loss Curves.png')
    save_loss_plot(
        train_losses, val_losses, plot_path,
        title=f'IEEE {args.bus}-bus | {args.gnn_type} — Train/Val Loss'
    )
    print(f"Loss curve saved to: {plot_path}")

    # ---------------------------------------------------------- save hyperparams
    hp = vars(args)
    hp['dataset_dir'] = dataset_dir
    hp['best_epoch'] = best_epoch
    hp['best_val_loss'] = best_val_loss
    with open(os.path.join(output_dir, f'[{args.bus} bus] Hyperparameters.json'), 'w') as f:
        json.dump(hp, f, indent=2)


if __name__ == '__main__':
    main()
