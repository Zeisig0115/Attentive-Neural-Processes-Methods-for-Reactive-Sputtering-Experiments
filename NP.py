import numpy as np
import pandas as pd
import torch
import random
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    explained_variance_score,
    r2_score
)
import torch.nn as nn
import matplotlib.pyplot as plt


def load_and_preprocess(
    csv_path: str,
    target_col: str,
    train_size: int,
):
    df = pd.read_csv(Path(csv_path))
    total_n = len(df)

    all_indices = np.arange(total_n)
    np.random.shuffle(all_indices)

    train_idx = all_indices[:train_size]
    test_idx = all_indices[train_size:]

    chosen_train_df = df.iloc[train_idx].reset_index(drop=True)
    chosen_test_df = df.iloc[test_idx].reset_index(drop=True)

    X_train_np = chosen_train_df.drop(columns=[target_col]).values.astype("float32")
    y_train_np = chosen_train_df[target_col].values.astype("float32").reshape(-1, 1)

    X_test_np = chosen_test_df.drop(columns=[target_col]).values.astype("float32")
    y_test_np = chosen_test_df[target_col].values.astype("float32").reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train_np).astype("float32")  # ★
    y_train_scaled = y_scaler.fit_transform(y_train_np).astype("float32")  # ★
    X_test_scaled = x_scaler.transform(X_test_np).astype("float32")  # ★
    y_test_scaled = y_scaler.transform(y_test_np).astype("float32")  # ★

    to_tensor = lambda arr: torch.from_numpy(arr)

    return (
        to_tensor(X_train_scaled),
        to_tensor(y_train_scaled),
        to_tensor(X_test_scaled),
        to_tensor(y_test_scaled),
        x_scaler,
        y_scaler,
    )


def make_np_batch(X, Y, num_ctx, min_ctx=3, max_ctx=None):
    N = X.size(0)
    if num_ctx > N:
        raise ValueError("num_ctx bigger than dataset")

    idx = torch.randperm(N)[:num_ctx]
    x_b, y_b = X[idx], Y[idx]  # batch = context + target

    max_ctx = max_ctx or (num_ctx - 1)
    n_ctx = torch.randint(min_ctx, max_ctx + 1, (1,)).item()
    c_idx = torch.randperm(num_ctx)[:n_ctx]

    mask = torch.ones(num_ctx, dtype=torch.bool)
    mask[c_idx] = False
    t_idx = torch.where(mask)[0]

    x_ctx, y_ctx = x_b[c_idx],   y_b[c_idx]       # (N_ctx,  x_dim), (N_ctx,  y_dim)
    x_all, y_all = x_b,          y_b              # (num_ctx,x_dim), (num_ctx,y_dim)
    x_tgt, y_tgt = x_b[t_idx],   y_b[t_idx]       # (N_tgt,  x_dim), (N_tgt,  y_dim)
    return x_ctx, y_ctx, x_all, y_all, n_ctx




class Encoder(nn.Module):
    def __init__(self, x_dim: int, y_dim: int,
                 r_dim: int = 128, hid_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, r_dim)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = torch.cat([x, y], dim=-1)
        return self.net(xy)



def aggregate_mean(r_i: torch.Tensor) -> torch.Tensor:
    if r_i.ndim != 2:
        raise ValueError("r_i 应是 [N_ctx, r_dim]")
    return r_i.mean(dim=0)          # → [r_dim]


class LatentSampler(nn.Module):
    """
    输入  r      : [r_dim]          (或 [B, r_dim] 若批量调用)
    输出 (z, μ, σ)
    ----------
    z_dim : 隐变量维度，可与 r_dim 相同；越大越能表示复杂分布
    """
    def __init__(self, r_dim: int, z_dim: int = 128):
        super().__init__()
        self.to_mu_log  = nn.Sequential(nn.Linear(r_dim, z_dim))
        self.to_log = nn.Linear(r_dim, z_dim)

    def forward(self, r: torch.Tensor):
        """
        参数
        ----
        r : [r_dim] 或 [B, r_dim]

        返回
        ----
        z        : 与 r 同批次大小的 [*, z_dim]
        mu, sigma: 同 shape，用于计算 KL
        """
        mu        = self.to_mu(r)            # [*, z_dim]
        log_sigma = self.to_log(r)           # 同上
        # softplus 保正数，加 1e-4 下限防数值崩
        sigma     = 1e-4 + torch.nn.functional.softplus(log_sigma)

        # 重参数化采样
        eps = torch.randn_like(sigma)
        z   = mu + sigma * eps
        return z, mu, sigma


class Decoder(nn.Module):
    """
    g_dec : (z , x★)  ↦  μ_y , σ_y
    ----------
    x_dim : 特征维度
    z_dim : 隐变量维度
    y_dim : 目标维度 (标量回归 ⇒ 1)
    """
    def __init__(self, x_dim: int, z_dim: int, y_dim: int = 1,
                 hid_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + z_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 2 * y_dim)          # → [μ , log σ]
        )

    def forward(self, z: torch.Tensor, x_tgt: torch.Tensor):
        """
        z      : [z_dim]           (单个 meta‑task), or [B, z_dim]
        x_tgt  : [N_tgt, x_dim]

        返回
        ----
        mu_y, sigma_y : [N_tgt, y_dim]
        """
        if z.ndim == 1:                 # 与目标点对齐
            z = z.expand(x_tgt.size(0), -1)        # [N_tgt, z_dim]
        elif z.ndim == 2:
            # 批量任务: z=[B,z_dim], x_tgt=[B,N_tgt,x_dim] (本示例先不使用)
            raise NotImplementedError

        stats = self.net(torch.cat([x_tgt, z], dim=-1))
        mu, log_sigma = stats.chunk(2, dim=-1)
        sigma = 1e-4 + torch.nn.functional.softplus(log_sigma)
        return mu, sigma


def gaussian_nll(mu_y, sigma_y, y_true):
    """逐点对数似然：-log p(y | μ,σ²)   返回 [N_tgt, 1]"""
    return 0.5 * torch.log(2 * torch.pi * sigma_y**2) + (y_true - mu_y)**2 / (2 * sigma_y**2)

def kl_standard_normal(mu_z, sigma_z):
    """KL(𝓝(μ,σ²) ‖ 𝓝(0,1))  —— 解析公式"""
    return 0.5 * torch.sum(mu_z**2 + sigma_z**2 - 1.0 - torch.log(sigma_z**2))

def np_loss(mu_y, sigma_y, y_tgt, mu_z, sigma_z):
    nll  = gaussian_nll(mu_y, sigma_y, y_tgt).mean()      # 负对数似然
    kl   = kl_standard_normal(mu_z, sigma_z) / y_tgt.size(0)   # 每样本分摊
    return nll + kl, nll.detach(), kl.detach()


class NeuralProcess(nn.Module):
    def __init__(self, x_dim, y_dim,
                 r_dim=64, z_dim=64,
                 aggregator=aggregate_mean):
        super().__init__()
        self.encoder   = Encoder(x_dim, y_dim, r_dim=r_dim)
        if isinstance(aggregator, nn.Module):
            self.aggregator = aggregator  # 注册名直接叫 aggregator
        else:
            self.aggregator = aggregator  # 函数式
        self.latent    = LatentSampler(r_dim, z_dim)
        self.decoder   = Decoder(x_dim, z_dim, y_dim)

    def forward(self, x_ctx, y_ctx, x_tgt):
        # 1) encode each context point
        r_i = self.encoder(x_ctx, y_ctx)       # [N_ctx, r_dim]
        # 2) permutation‑invariant aggregation
        r   = self.aggregator(r_i)             # [r_dim]
        # 3) sample latent z
        z, mu_z, sigma_z = self.latent(r)      # [z_dim], same shape for mu/sigma
        # 4) decode every target x★
        mu_y, sigma_y = self.decoder(z, x_tgt)
        return mu_y, sigma_y, mu_z, sigma_z


def train_step(model, optimizer, X, Y, batch_size=32, min_ctx=3):
    x_c, y_c, x_t, y_t, *_ = make_np_batch(X, Y, batch_size, min_ctx=min_ctx)
    mu_y, sigma_y, mu_z, sigma_z = model(x_c, y_c, x_t)
    loss, nll, kl = np_loss(mu_y, sigma_y, y_t, mu_z, sigma_z)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), nll.item(), kl.item()

@torch.no_grad()
def evaluate_epoch(model, X, Y, reps=100, batch_size=64):
    rmses, nlls = [], []
    for _ in range(reps):
        x_c, y_c, x_t, y_t, *_ = make_np_batch(X, Y, batch_size)
        mu_y, sigma_y, *_ = model(x_c, y_c, x_t)
        rmses.append(torch.sqrt(((mu_y - y_t)**2).mean()).item())
        nlls.append(gaussian_nll(mu_y, sigma_y, y_t).mean().item())
    return np.mean(rmses), np.mean(nlls)


def main(config):
    # 固定随机种子
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 加载数据
    X_tr, y_tr, X_te, y_te, x_scaler, y_scaler = load_and_preprocess(
        config["dataset_path"],
        target_col=config["target_col"],
        train_size=config["train_size"]
    )

    # 模型与优化器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralProcess(
        x_dim=X_tr.shape[1],
        y_dim=1,
        r_dim=config["r_dim"],
        z_dim=config["z_dim"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["T_max"], eta_min=config["lr_min"]
    )

    # 训练循环
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        loss, nll, kl = train_step(
            model, optimizer,
            X_tr.to(device), y_tr.to(device),
            batch_size=config["batch_size"],
            min_ctx=config["min_ctx"]
        )
        scheduler.step()

    # 最终在测试集上评估
    model.eval()
    with torch.no_grad():
        mu_y_scaled, sigma_y_scaled, _, _ = model(
            X_tr.to(device), y_tr.to(device), X_te.to(device)
        )
    mu_y = y_scaler.inverse_transform(mu_y_scaled.cpu().numpy())
    y_true = y_scaler.inverse_transform(y_te.cpu().numpy())

    mse = mean_squared_error(y_true, mu_y)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, mu_y)
    ev = explained_variance_score(y_true, mu_y)
    r2 = r2_score(y_true, mu_y)

    # 返回一个字典，包含你想记录的指标
    return {
        "seed": config["seed"],
        "MSE": mse,
        "RMSE": rmse,
        "MAPE(%)": mape * 100,
        "ExplainedVar": ev,
        "R2": r2
    }


if __name__ == "__main__":
    base_config = {
        "train_size": 40,
        "target_col": "N/Ti",
        "lr": 0.001,
        "lr_min": 1e-4,
        "T_max": 500,
        "epochs": 500,
        "batch_size": 32,
        "min_ctx": 2,
        "eval_reps": 100,
        "eval_batch_size": 8,
        "r_dim": 64,
        "z_dim": 64,
        "hid_dim": 64,
        "dataset_path": "./Nitride (Dataset 1) NTi.csv"
    }

    results = []
    for seed in range(2,3):  # 0,1,...,19 共 20 个种子
        config = base_config.copy()
        config["seed"] = seed
        stats = main(config)
        print(f"Seed {seed:2d} → RMSE: {stats['RMSE']:.4f}, R²: {stats['R2']:.4f}")
        results.append(stats)

    # 存成 DataFrame，导出 CSV
    df = pd.DataFrame(results)
    df.to_csv("baseline.csv", index=False)
    print("所有实验结果已保存到 baseline.csv")


