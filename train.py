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

from gnn.data         import build_dataloaders
from gnn.model        import GNNPowerFlow, SUPPORTED_TYPES
from gnn.mpn          import MPN
from gnn.physics_loss import PowerImbalance, MixedMSEPowerImbalance


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def denormalize(t, mean, std):
    return t * std + mean


def denorm_mse(y_pred, y_true, y_mean, y_std, batch_size):
    """MSE in denormalised target space (stable comparison scale across losses)."""
    y_mean_exp = y_mean.view(-1).repeat(batch_size)
    y_std_exp  = y_std.view(-1).repeat(batch_size)
    return torch.mean(
        (denormalize(y_pred.view(-1), y_mean_exp, y_std_exp) -
         denormalize(y_true.view(-1), y_mean_exp, y_std_exp)) ** 2
    )


def compute_train_loss(loss_type, loss_fn, y_pred, batch, y_mean, y_std):
    """Compute the configured training loss; y_pred shape [batch, n_bus*2]."""
    if loss_type == 'mse':
        return denorm_mse(y_pred, batch.y, y_mean, y_std, batch.num_graphs)
    if loss_type == 'physics':
        return loss_fn(y_pred, batch.x, batch.edge_index, batch.edge_attr)
    if loss_type == 'mixed':
        y_true = batch.y.view(batch.num_graphs, -1)
        return loss_fn(y_pred, y_true, batch.x, batch.edge_index, batch.edge_attr)
    raise ValueError(f"unknown loss_type: {loss_type}")


def train_one_epoch(model, loader, optimizer, y_mean, y_std, device,
                    loss_type='mse', loss_fn=None):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        y_pred = model(batch)                       # [batch, n_bus*2]
        loss = compute_train_loss(loss_type, loss_fn, y_pred, batch, y_mean, y_std)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, y_mean, y_std, device):
    """Always reports denormalised MSE for cross-model comparability."""
    model.eval()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        y_pred = model(batch)
        loss = denorm_mse(y_pred, batch.y, y_mean, y_std, batch.num_graphs)
        total_loss += loss.item() * batch.num_graphs

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
    parser.add_argument('--gnn_type',    type=str,   default='GraphConv',  choices=list(SUPPORTED_TYPES) + ['MPN'])
    parser.add_argument('--train_loss',  type=str,   default='mse',        choices=['mse', 'physics', 'mixed'],
                        help='Training loss. MSE (default) matches other baselines. physics=PowerImbalance only, mixed=MSE+physics.')
    parser.add_argument('--mixed_alpha',   type=float, default=0.9,
                        help='Weight on MSE in mixed loss (1-alpha on physics). Default 0.9.')
    parser.add_argument('--physics_scale', type=float, default=0.02,
                        help='Scale applied to the physics term in mixed loss. Default 0.02.')
    parser.add_argument('--mpn_hidden',    type=int,   default=129,   help='MPN hidden dim (default 129).')
    parser.add_argument('--mpn_layers',    type=int,   default=4,     help='MPN #TAGConv layers (default 4).')
    parser.add_argument('--mpn_K',         type=int,   default=3,     help='TAGConv filter order K (default 3).')
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
    parser.add_argument('--device',      type=str,   default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help='Force a specific device (default: auto-detect)')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or os.path.join('PerturbedDatasets', f'{args.bus}Bus')
    if args.output_dir:
        output_dir = args.output_dir
    else:
        suffix = '_outages' if 'outage' in dataset_dir.lower() else ''
        loss_suffix = '' if args.train_loss == 'mse' else f'_{args.train_loss}'
        output_dir = os.path.join('Results', f'{args.bus}Bus_{args.gnn_type}{loss_suffix}{suffix}')
    os.makedirs(output_dir, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
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
    if args.gnn_type == 'MPN':
        model = MPN(
            n_bus        = args.bus,
            nfeature_dim = 7,
            efeature_dim = 2,
            hidden_dim   = args.mpn_hidden,
            n_gnn_layers = args.mpn_layers,
            K            = args.mpn_K,
            dropout_rate = args.dropout,
        ).to(device)
    else:
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
    print(f"Trainable parameters: {total_params:,}")
    print(f"Train loss          : {args.train_loss}\n")

    # ------------------------------------------------------------- loss_fn
    loss_fn = None
    if args.train_loss == 'physics':
        loss_fn = PowerImbalance(
            x_mean=stats['x_mean'], x_std=stats['x_std'],
            y_mean=stats['y_mean'], y_std=stats['y_std'],
            edge_mean=stats['edge_mean'], edge_std=stats['edge_std'],
            sn_mva=stats['sn_mva'],
            load_bus_mask=stats['load_bus_mask'],
        ).to(device)
    elif args.train_loss == 'mixed':
        loss_fn = MixedMSEPowerImbalance(
            x_mean=stats['x_mean'], x_std=stats['x_std'],
            y_mean=stats['y_mean'], y_std=stats['y_std'],
            edge_mean=stats['edge_mean'], edge_std=stats['edge_std'],
            sn_mva=stats['sn_mva'],
            load_bus_mask=stats['load_bus_mask'],
            alpha=args.mixed_alpha,
            physics_scale=args.physics_scale,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30
    )

    # --------------------------------------------------------------- training
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_count = 0
    best_epoch = 0
    ckpt_path = os.path.join(output_dir, f'[{args.bus} bus] Best_{args.gnn_type}_model.pt')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, y_mean, y_std, device,
            loss_type=args.train_loss, loss_fn=loss_fn,
        )
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
