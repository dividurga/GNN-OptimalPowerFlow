"""
Evaluation metrics: MSE, RMSE, NRMSE, MAE, R².
All functions operate on denormalised tensors.
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)


def nrmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
    y_true_np = y_true.detach().cpu().numpy().reshape(-1)
    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    return float(rmse / (np.std(y_true_np) + 1e-8))


def compute_all(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
    """
    Compute MSE, RMSE, NRMSE, MAE, R² on denormalised tensors.
    Both inputs shape: [n_samples, n_bus, 2]  or  [n, 2].
    """
    p = y_pred.detach().cpu().numpy().reshape(-1, 2)
    t = y_true.detach().cpu().numpy().reshape(-1, 2)

    mse_val  = mean_squared_error(t, p)
    rmse_val = np.sqrt(mse_val)
    nrmse_val = rmse_val / (np.std(t) + 1e-8)
    mae_val  = mean_absolute_error(t, p)
    r2_val   = r2_score(t, p)

    return dict(MSE=mse_val, RMSE=rmse_val, NRMSE=nrmse_val, MAE=mae_val, R2=r2_val)
