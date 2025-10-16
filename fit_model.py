# -*- coding: utf-8 -*-
import torch
import pyro
import numpy as np

from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from botorch.fit import fit_fully_bayesian_model_nuts

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

from botorch.models.fully_bayesian import FullyBayesianSingleTaskGP

from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.models.map_saas import add_saas_prior

from sklearn.ensemble import RandomForestRegressor
from model import PRFModel

from torch.utils.data import DataLoader, TensorDataset
from model import MLPRegressor, MLPEnsembleModel


def _set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        import pyro
        pyro.set_rng_seed(seed)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # <—— 新增



def _make_bounds_from_train(X_train: torch.Tensor) -> torch.Tensor:
    lb = X_train.min(dim=0).values.clone()
    ub = X_train.max(dim=0).values.clone()
    same = ub <= lb
    if same.any():
        ub[same] = lb[same] + 1e-6
    return torch.stack([lb, ub])


def fit_saasbo(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    warmup: int, num_samples: int, thinning: int, seed: int
):
    _set_seeds(seed)
    dev = X_tr.device
    bounds_raw = _make_bounds_from_train(X_tr)
    model = SaasFullyBayesianSingleTaskGP(
        train_X=X_tr,
        train_Y=y_tr,
        input_transform=Normalize(d=X_tr.shape[1], bounds=bounds_raw),
        outcome_transform=Standardize(m=1),
    ).to(dev)
    fit_fully_bayesian_model_nuts(
        model,
        warmup_steps=warmup,
        num_samples=num_samples,
        thinning=thinning,
        disable_progbar=False,
    )
    _ = model.posterior(X_tr[: min(5, X_tr.shape[0])], observation_noise=True)
    return model


def fit_gp(X_tr: torch.Tensor, y_tr: torch.Tensor, seed: int):
    _set_seeds(seed)
    dev = X_tr.device
    d = X_tr.shape[1]
    model = SingleTaskGP(
        train_X=X_tr,
        train_Y=y_tr,
        input_transform=Normalize(d=d, bounds=_make_bounds_from_train(X_tr)),
        outcome_transform=Standardize(m=1),
    ).to(dev)  # <—— 新增

    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(dev)  # <—— 新增
    fit_gpytorch_mll(mll)
    _ = model.posterior(X_tr[: min(5, X_tr.shape[0])], observation_noise=True)
    return model


def fit_fullyb_gp(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    warmup: int, num_samples: int, thinning: int, seed: int
):
    _set_seeds(seed)
    dev = X_tr.device
    d = X_tr.shape[1]
    model = FullyBayesianSingleTaskGP(
        train_X=X_tr,
        train_Y=y_tr,
        input_transform=Normalize(d=d, bounds=_make_bounds_from_train(X_tr)),
        outcome_transform=Standardize(m=1),
    ).to(dev)
    fit_fully_bayesian_model_nuts(
        model,
        warmup_steps=warmup,
        num_samples=num_samples,
        thinning=thinning,
        disable_progbar=False,
    )
    _ = model.posterior(X_tr[: min(5, X_tr.shape[0])], observation_noise=True)
    return model


def fit_saas_gp(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    seed: int,
    nu: float = 2.5,
):
    _set_seeds(seed)
    dev = X_tr.device
    d = X_tr.shape[1]
    input_tf  = Normalize(d=d, bounds=_make_bounds_from_train(X_tr))
    outcome_tf = Standardize(m=1)
    base = MaternKernel(nu=nu, ard_num_dims=d)
    add_saas_prior(base_kernel=base, tau=None, log_scale=True)
    covar = ScaleKernel(base)
    model = SingleTaskGP(
        train_X=X_tr,
        train_Y=y_tr,
        covar_module=covar,
        input_transform=input_tf,
        outcome_transform=outcome_tf,
    ).to(dev)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    _ = model.posterior(X_tr[: min(5, X_tr.shape[0])], observation_noise=True)
    return model


def fit_prf(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    seed: int = 0,
    n_estimators: int = 300,
    max_depth=None,
    min_samples_leaf: int = 1,
):
    X_np = X_tr.detach().cpu().numpy()
    y_np = y_tr.view(-1).detach().cpu().numpy()

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        bootstrap=True,
        oob_score=True,       # 用 OOB 估计观测噪声
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_np, y_np)

    obs_var = 0.0
    if getattr(rf, "oob_prediction_", None) is not None:
        resid = y_np - rf.oob_prediction_
        resid = resid[np.isfinite(resid)]
        if resid.size > 1:
            obs_var = float(np.var(resid))
    if not np.isfinite(obs_var) or obs_var <= 0.0:
        resid = y_np - rf.predict(X_np)
        resid = resid[np.isfinite(resid)]
        obs_var = float(np.var(resid)) if resid.size > 1 else 0.0

    return PRFModel(rf=rf, obs_noise_var=obs_var, device=X_tr.device)


def fit_mlp_ensemble(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    seed: int = 0,
    n_ens: int = 10,
    hidden: list[int] = [32, 32],
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
):
    device = X_tr.device
    dtype = X_tr.dtype  # 期望是 torch.double

    # --- 标准化参数（按整份训练集） ---
    x_mean = X_tr.mean(dim=0, keepdim=True)
    x_std = X_tr.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
    y_mean = y_tr.mean(dim=0, keepdim=True)
    y_std = y_tr.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)

    Xn = (X_tr - x_mean) / x_std
    yn = (y_tr - y_mean) / y_std

    ds = TensorDataset(Xn.to(device), yn.to(device))
    dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)

    models = []
    for i in range(n_ens):
        torch.manual_seed(seed + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + i)

        net = MLPRegressor(d_in=X_tr.shape[1], hidden=hidden).to(device=device, dtype=dtype)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        net.train()
        for _ in range(epochs):
            for xb, yb in dl:
                opt.zero_grad()
                mu, log_var = net(xb)
                # 高斯 NLL（省略 0.5*log(2π) 常数项）
                var = torch.exp(log_var).clamp_min(1e-12)
                nll = 0.5 * ((yb - mu).pow(2) / var + log_var)
                loss = nll.mean()
                loss.backward()
                opt.step()

        models.append(net.eval())

    return MLPEnsembleModel(
        models=models,
        x_mean=x_mean.to(device=device, dtype=dtype),
        x_std=x_std.to(device=device, dtype=dtype),
        y_mean=y_mean.to(device=device, dtype=dtype),
        y_std=y_std.to(device=device, dtype=dtype),
        device=device,
    )




