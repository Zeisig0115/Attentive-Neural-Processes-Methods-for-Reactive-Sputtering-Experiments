# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiagNormalPosterior:
    """
    轻量 Posterior，满足你评估管线所需的接口：
    - .mean / .variance -> [n,1] torch.double
    - .rsample(sample_shape) -> [T, n, 1] 或 [n,1]
    - 无 mixture（distribution=None），你的 NLL/MSLL/CRPS 仍可用
    """
    def __init__(self, mean: torch.Tensor, var: torch.Tensor):
        self._mean = mean
        self._var = var
        self.distribution = None  # 保持非-mixture 分支

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def variance(self) -> torch.Tensor:
        return self._var

    @torch.no_grad()
    def rsample(self, sample_shape: torch.Size = torch.Size()):
        eps = torch.randn(
            *sample_shape, *self._mean.shape,
            dtype=self._mean.dtype, device=self._mean.device
        )
        return self._mean + eps * torch.sqrt(self._var.clamp_min(1e-12))


class PRFModel:
    """
    Probabilistic Random Forest baseline：
    - 训练用 sklearn 的 RandomForestRegressor
    - posterior.mean = 树预测均值；posterior.var = 树间方差 (+ 可选观测噪声)
    """
    def __init__(self, rf, obs_noise_var: float, device: torch.device):
        self.rf = rf
        self.obs_noise_var = float(max(obs_noise_var, 0.0))
        self.device = device

    @torch.no_grad()
    def posterior(self, X: torch.Tensor, observation_noise: bool = True):
        # 输入 X: torch.double，[n,d]，任意 device；内部转 numpy 走 sklearn 推理
        X_np = X.detach().cpu().numpy()
        preds_tree = np.stack([est.predict(X_np) for est in self.rf.estimators_], axis=0)  # [S,n]
        mean = preds_tree.mean(axis=0, keepdims=True).T      # [n,1]
        var_model = preds_tree.var(axis=0, ddof=0, keepdims=True).T  # [n,1]
        var = var_model + (self.obs_noise_var if observation_noise else 0.0)

        mean_t = torch.as_tensor(mean, dtype=torch.double, device=self.device)
        var_t  = torch.as_tensor(var,  dtype=torch.double, device=self.device)
        return DiagNormalPosterior(mean_t, var_t)


class MLPRegressor(nn.Module):
    """
    MLP 回归器，输出 (mu, log_var)，用高斯 NLL 训练，支持 torch.double。
    """
    def __init__(self, d_in: int, hidden: list[int] = [64, 64]):
        super().__init__()
        layers = []
        last = d_in
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head_mu = nn.Linear(last, 1)
        self.head_logvar = nn.Linear(last, 1)  # 预测 log(sigma^2)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        mu = self.head_mu(h)                   # [B,1]
        log_var = self.head_logvar(h)          # [B,1]
        return mu, log_var


class MLPEnsembleModel:
    """
    MLP 集成：S 个网络 + 训练集标准化参数
    - 预测时：mean = 平均(mu_i)
             epistemic_var = Var(mu_i)
             aleatoric_var = 平均(exp(log_var_i))
             total_var = epistemic_var + (observation_noise ? aleatoric_var : 0)
    """
    def __init__(self, models: list[MLPRegressor],
                 x_mean: torch.Tensor, x_std: torch.Tensor,
                 y_mean: torch.Tensor, y_std: torch.Tensor,
                 device: torch.device):
        self.models = models
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.device = device
        for m in self.models:
            m.eval()

    @torch.no_grad()
    def _x_normalize(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.x_mean) / self.x_std

    @torch.no_grad()
    def posterior(self, X: torch.Tensor, observation_noise: bool = True):
        Xn = self._x_normalize(X.to(self.device))
        mus = []
        ale_vars = []
        for m in self.models:
            mu_n, logv_n = m(Xn)                         # 归一化空间
            # 反归一化到原 y 空间
            mu = mu_n * self.y_std + self.y_mean
            var = torch.exp(logv_n).clamp_min(1e-12) * (self.y_std ** 2)
            mus.append(mu)                               # [n,1]
            ale_vars.append(var)                         # [n,1]
        mus = torch.stack(mus, dim=0)                    # [S,n,1]
        ale_vars = torch.stack(ale_vars, dim=0)          # [S,n,1]
        mean = mus.mean(dim=0)                           # [n,1]
        var_epi = mus.var(dim=0, unbiased=False)         # [n,1]
        var_ale = ale_vars.mean(dim=0)                   # [n,1]
        var = var_epi + (var_ale if observation_noise else 0.0)
        return DiagNormalPosterior(mean.to(torch.double), var.to(torch.double))
