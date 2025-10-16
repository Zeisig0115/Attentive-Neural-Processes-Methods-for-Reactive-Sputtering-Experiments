#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FBGP（SAAS）不传入每点噪声版本：让模型自行学习观测噪声（同方差）
- 不再使用 auc_std / replicates 计算 yvar
- 数据加载仅提取特征与目标列
- 评估仍使用 observation_noise=True（包含“学习到的噪声”）
"""

import argparse
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd
import torch
import pyro
import time

from fit_model import (
    fit_saasbo,
    fit_gp,
    fit_fullyb_gp,
    fit_saas_gp,
    fit_prf,
    fit_mlp_ensemble,
)

from metrics import (
    point_metrics,
    mixture_nll_from_moments,
    msll_against_gaussian_baseline,
    crps_mc,
)


def set_seeds(seed: int):
    torch.set_default_dtype(torch.double)
    torch.manual_seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_numeric_table_no_noise(
    csv_path: str,
    target_col: str,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    FEATURE_WHITELIST = ["power", "time", "ar", "n2", "pressure", "voltage", "current", "ri"]

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"未找到目标列 '{target_col}'；现有列：{list(df.columns)}")

    col_map = {c.lower(): c for c in df.columns if c != target_col}
    feat_cols = [col_map[c] for c in FEATURE_WHITELIST if c in col_map]
    if not feat_cols:
        raise ValueError("白名单特征在数据中一个都没找到；请检查列名或大小写。")

    X_df = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")

    all_df = pd.concat([X_df, y.rename(target_col)], axis=1)
    all_df = all_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")

    X = all_df[feat_cols].to_numpy()
    y = all_df[target_col].to_numpy().reshape(-1, 1).astype(float)

    if X.shape[1] == 0:
        raise ValueError("白名单中的特征在数据里均为空或无法转为数值。")
    if X.shape[0] == 0:
        raise ValueError("清洗后没有有效样本；请检查数据是否含有 NaN/±inf 或目标/特征是否可转换为数值。")

    return feat_cols, X, y


def make_bounds_from_train(X_train: torch.Tensor) -> torch.Tensor:
    lb = X_train.min(dim=0).values.clone()
    ub = X_train.max(dim=0).values.clone()
    same = ub <= lb
    if same.any():
        ub[same] = lb[same] + 1e-6
    return torch.stack([lb, ub])  # [2, d]


@dataclass
class FoldResult:
    rmse: float
    r2: float
    nll: float
    msll: float
    crps: float
    n_test: int


# ========= 单折训练与评估（学习噪声） =========
def fold_eval_learn_noise(
    X_all: np.ndarray,
    y_all: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    warmup: int,
    num_samples: int,
    thinning: int,
    seed: int,
    crps_T: int,
    model_type: str = "gp",
    device: str = "auto",
) -> FoldResult:
    # 划分
    Xtr_np, ytr_np = X_all[idx_train], y_all[idx_train]
    Xte_np, yte_np = X_all[idx_test], y_all[idx_test]

    dev = get_device(device)
    Xtr = torch.as_tensor(Xtr_np, dtype=torch.double, device=dev)
    ytr = torch.as_tensor(ytr_np, dtype=torch.double, device=dev)
    Xte = torch.as_tensor(Xte_np, dtype=torch.double, device=dev)
    yte = torch.as_tensor(yte_np, dtype=torch.double, device=dev)

    finite_mask = torch.isfinite(Xtr).all(dim=0)
    Xtr = Xtr[:, finite_mask]
    Xte = Xte[:, finite_mask]

    std = Xtr.std(dim=0)
    keep = std > 1e-12
    Xtr = Xtr[:, keep]
    Xte = Xte[:, keep]

    assert torch.isfinite(ytr).all() and torch.isfinite(yte).all()
    if ytr.std() <= 0:
        raise ValueError("训练目标方差为 0，无法拟合。")

    # 选择模型
    if model_type == "saasbo":
        model = fit_saasbo(Xtr, ytr, warmup, num_samples, thinning, seed)
    elif model_type == "fullyb_gp":
        model = fit_fullyb_gp(Xtr, ytr, warmup, num_samples, thinning, seed)
    elif model_type == "saas_gp":
        model = fit_saas_gp(Xtr, ytr, seed=seed)
    elif model_type == "prf":
        model = fit_prf(Xtr, ytr, seed=seed)
    elif model_type == "mlp":
        model = fit_mlp_ensemble(Xtr, ytr, seed=seed)
    else:
        model = fit_gp(Xtr, ytr, seed=seed)

    with torch.no_grad():
        post = model.posterior(Xte, observation_noise=True)
        dist = post.distribution

        if hasattr(dist, "component_distribution"):
            comp = dist.component_distribution
            mu_s = comp.mean   # [..., n, m]
            var_s = comp.variance
        else:
            mu_s = post.mean   # [n, m] or [n]
            var_s = post.variance

        if mu_s.ndim == 1:
            mu_s = mu_s.unsqueeze(-1)  # [n,1]
            var_s = var_s.unsqueeze(-1)
        if mu_s.ndim == 2:
            mu_s = mu_s.unsqueeze(0)  # [1, n, m]
            var_s = var_s.unsqueeze(0)

        reduce_dims = tuple(range(0, mu_s.ndim - 2))
        mu_mix = mu_s.mean(dim=reduce_dims)  # [n, m]
        if mu_mix.ndim == 1:
            mu_mix = mu_mix.view(-1, 1)

        S = int(np.prod(mu_s.shape[:-2]))
        n, m = mu_s.shape[-2], mu_s.shape[-1]
        mu_s_flat = mu_s.reshape(S, n, m)   # [S, n, m]
        var_s_flat = var_s.reshape(S, n, m)

        rmse, r2 = point_metrics(mu_mix, yte)
        nll = mixture_nll_from_moments(yte, mu_s_flat, var_s_flat)

        tr_mean = float(ytr.mean().item())
        tr_var = float(ytr.var(unbiased=False).item())
        msll = msll_against_gaussian_baseline(yte, nll, tr_mean, tr_var)

        crps = crps_mc(post, yte.view(-1), T=crps_T)

    return FoldResult(
        rmse=rmse,
        r2=r2,
        nll=nll,
        msll=msll,
        crps=crps,
        n_test=Xte.shape[0],
    )


# ========= 5-fold CV 主流程 =========
def kfold_indices(n: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)

    results = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i], axis=0)
        results.append((train_idx, test_idx))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="./data/Nitride (Dataset 1) NTi.csv")
    parser.add_argument("--target", type=str, default="N/Ti")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model", type=str, choices=["gp", "saasbo", "fullyb_gp", "saas_gp", "prf", "mlp"], default="gp")
    parser.add_argument("--warmup_steps", type=int, default=128)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--thinning", type=int, default=16)
    parser.add_argument("--crps_T", type=int, default=512)
    args = parser.parse_args()

    set_seeds(args.seed)
    feat_names, X_np, y_np = load_numeric_table_no_noise(
        csv_path=args.csv,
        target_col=args.target,
    )

    n, d = X_np.shape
    print(f"[INFO] Loaded n={n}, d={d} from {args.csv}. target='{args.target}'")
    print(f"[INFO] Features: {feat_names}")

    splits = kfold_indices(n=n, k=args.folds, seed=args.seed)
    all_res: List[FoldResult] = []

    for i, (tr_idx, te_idx) in enumerate(splits, 1):
        print(f"\n===== Fold {i}/{args.folds} train={len(tr_idx)} test={len(te_idx)} =====")
        res = fold_eval_learn_noise(
            X_all=X_np,
            y_all=y_np,
            idx_train=tr_idx,
            idx_test=te_idx,
            warmup=args.warmup_steps,
            num_samples=args.num_samples,
            thinning=args.thinning,
            seed=args.seed + i,
            crps_T=args.crps_T,
            model_type=args.model,
            device=args.device,
        )
        print(
            f"[Fold {i}] RMSE={res.rmse:.6f} R2={res.r2:.4f} "
            f"NLL={res.nll:.6f} MSLL={res.msll:.6f} CRPS={res.crps:.6f}"
        )
        all_res.append(res)

    # 汇总均值±标准差（按 fold ）
    def mean_std(vals):
        v = np.asarray(vals, dtype=float)
        if v.size <= 1:
            return float(v.mean()), 0.0
        return float(v.mean()), float(v.std(ddof=1))

    rmse_m, rmse_s = mean_std([r.rmse for r in all_res])
    r2_m, r2_s = mean_std([r.r2 for r in all_res])
    nll_m, nll_s = mean_std([r.nll for r in all_res])
    msll_m, msll_s = mean_std([r.msll for r in all_res])
    crps_m, crps_s = mean_std([r.crps for r in all_res])

    print("\n===== 5-Fold Summary (mean ± std) =====")
    print(f"RMSE : {rmse_m:.6f} ± {rmse_s:.6f}")
    print(f"R2   : {r2_m:.4f} ± {r2_s:.4f}")
    print(f"NLL  : {nll_m:.6f} ± {nll_s:.6f}")
    print(f"MSLL : {msll_m:.6f} ± {msll_s:.6f}")
    print(f"CRPS : {crps_m:.6f} ± {crps_s:.6f}")


if __name__ == "__main__":
    main()
