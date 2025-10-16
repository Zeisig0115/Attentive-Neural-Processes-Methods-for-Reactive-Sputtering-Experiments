# -*- coding: utf-8 -*-
import math
import torch
import numpy as np

# ============ 点预测指标 ============
def point_metrics(mu: torch.Tensor, y: torch.Tensor):
    """
    mu, y: shape [n, 1] 或 [n]
    返回 (rmse, r2)
    """
    mu = mu.view(-1)
    y  = y.view(-1)
    se = (mu - y) ** 2
    rmse = float(torch.sqrt(torch.mean(se)).item())

    denom = torch.sum((y - torch.mean(y)) ** 2)
    if denom <= 1e-20:
        r2 = float("nan")  # 或者返回 0.0，取决于你想要的行为
    else:
        r2 = float(1.0 - torch.sum(se) / denom)
    return rmse, r2


# ============ 概率指标（NLL / MSLL / CRPS） ============
def mixture_nll_from_moments(y: torch.Tensor, mean_s: torch.Tensor, var_s: torch.Tensor) -> float:
    S = mean_s.shape[0]
    y = y.view(1, -1, 1)
    var_s = var_s.clamp_min(1e-12)
    log_probs = -0.5 * ( math.log(2 * math.pi) + torch.log(var_s) + (y - mean_s) ** 2 / var_s )
    log_mix = torch.logsumexp(log_probs, dim=0) - math.log(S)
    nll = -log_mix.mean().item()
    return nll


def msll_against_gaussian_baseline(y: torch.Tensor, nll: float, train_mean: float, train_var: float) -> float:
    """MSLL = NLL - baseline_NLL"""
    train_var = max(train_var, 1e-12)
    log_p = -0.5 * ( math.log(2 * math.pi * train_var) + ((y - train_mean) ** 2) / train_var ) # [n, 1]
    baseline_nll = (-log_p).mean().item()
    return nll - baseline_nll


def crps_mc(posterior, y: torch.Tensor, T: int = 512) -> float:
    with torch.no_grad():
        z1 = posterior.rsample(sample_shape=torch.Size([T]))
        z2 = posterior.rsample(sample_shape=torch.Size([T]))

        while z1.ndim >= 3 and z1.shape[-1] == 1:
            z1 = z1.squeeze(-1)
            z2 = z2.squeeze(-1)

        if z1.ndim == 2:      # [T, n]
            pass
        elif z1.ndim >= 3:    # [T, ..., n]
            T_, n_ = z1.shape[0], z1.shape[-1]
            mix = int(np.prod(z1.shape[1:-1]))
            z1 = z1.reshape(T_ * mix, n_)
            z2 = z2.reshape(T_ * mix, n_)
        else:
            raise RuntimeError(f"未预期的采样维度: {z1.shape}")

        y_row = y.view(1, -1)
        term1 = (z1 - y_row).abs().mean(dim=0)
        term2 = (z1 - z2).abs().mean(dim=0)
        return (term1 - 0.5 * term2).mean().item()



