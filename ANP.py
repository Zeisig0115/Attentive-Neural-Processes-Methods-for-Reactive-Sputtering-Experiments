import math
import random
import pandas as pd
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    explained_variance_score,
    r2_score,
)
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------
# Data loading & preprocessing
# -------------------------------------------------
def load_and_preprocess(
    csv_path: str,
    target_col: str,
    train_size: int,
):
    """Load CSV, split, scale and convert to tensors.

    Args:
        csv_path: Path to the CSV file that holds the raw dataset.
        target_col: Column name of the regression target.
        train_size: Number of rows assigned to the training split. The rest
            become the test split.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              StandardScaler, StandardScaler]:
            * X_train (scaled)
            * y_train (scaled)
            * X_test  (scaled)
            * y_test  (scaled)
            * fitted X scaler
            * fitted y scaler
    """
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

    X_train_scaled = x_scaler.fit_transform(X_train_np).astype("float32")
    y_train_scaled = y_scaler.fit_transform(y_train_np).astype("float32")
    X_test_scaled  = x_scaler.transform(X_test_np).astype("float32")
    y_test_scaled  = y_scaler.transform(y_test_np).astype("float32")

    to_tensor = lambda arr: torch.from_numpy(arr)

    return (
        to_tensor(X_train_scaled),
        to_tensor(y_train_scaled),
        to_tensor(X_test_scaled),
        to_tensor(y_test_scaled),
        x_scaler,
        y_scaler,
    )


# -------------------------------------------------
# Batch construction helper
# -------------------------------------------------
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
    return x_ctx, y_ctx, x_all, y_all, x_tgt, y_tgt, n_ctx




# -------------------------------------------------
# Building blocks
# -------------------------------------------------
class Linear(nn.Module):
    """Xavier-initialised linear layer with configurable gain."""
    def __init__(self, in_dim, out_dim, bias=True, w_init="linear"):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.lin.weight,
                                gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.lin(x)


class MultiheadAttention(nn.Module):
    """
    Classic scaled dot-product multi-head attention implemented from scratch.
    Returns shape (B, N_q, d_model).
    """
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_k = d_model // n_head

        # Projection layers for Q, K, V
        self.wq = Linear(d_model, d_model, w_init="linear")
        self.wk = Linear(d_model, d_model, w_init="linear")
        self.wv = Linear(d_model, d_model, w_init="linear")
        self.out_proj = Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        key, value : (B, N_k, d_model)
        query       : (B, N_q, d_model)
        """
        B, N_k, _ = key.size()
        _, N_q, _ = query.size()

        # Helper: linear projection + reshape heads
        def proj(x, lin):
            x = lin(x)                                   # (B, N, d_model)
            x = x.view(B, -1, self.n_head, self.d_k)     # (B, N, h, d_k)
            return x.permute(0, 2, 1, 3)                 # (B, h, N, d_k)

        Q = proj(query, self.wq)
        K = proj(key,   self.wk)
        V = proj(value, self.wv)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * (self.d_k ** -0.5)
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        out = torch.matmul(attn, V)                      # (B, h, N_q, d_k)
        out = out.permute(0, 2, 1, 3).contiguous()       # (B, N_q, h, d_k)
        out = out.view(B, N_q, -1)                       # (B, N_q, d_model)

        return self.out_proj(out)


class Attention(nn.Module):
    """Attention block with residual connection, norm, and dropout."""
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.mha  = MultiheadAttention(d_model, n_head, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key, value):
        out = self.mha(query, key, value)
        out = self.drop(out)
        return self.norm(out + query)  # residual + layer norm


# -------------------------------------------------
# Latent encoder q(z|C) and p(z|C)
# -------------------------------------------------
class LatentEncoder(nn.Module):
    """Self-attentive encoder producing q(z|·) / p(z|·).

    Attributes:
        input_mlp: Maps concatenated (x, y) to d_model dimensional space.
        self_attn1/self_attn2: Two self-attention blocks over context points.
        stat_mlp: Aggregates to global statistics (μ, log σ).
    """
    def __init__(self, x_dim, y_dim, d_model, z_dim, n_head=8):
        super().__init__()
        # 1) Embed each (x, y) pair
        self.input_mlp = nn.Sequential(
            nn.Linear(x_dim + y_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # 2) Two layers of self-attention
        self.self_attn1 = Attention(d_model, n_head)
        self.self_attn2 = Attention(d_model, n_head)

        # 3) Aggregate across points and map to (μ, log σ)
        self.stat_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * z_dim),
        )

    def forward(self, x, y):
        """Encode context to latent z distribution.

        Args:
            x: Context features, shape (B, N_ctx, x_dim).
            y: Context targets,  shape (B, N_ctx, y_dim).

        Returns:
            z: Sample from q(z|C) or p(z|C).
            mu: Mean of the latent Gaussian.
            sigma: Standard deviation of the latent Gaussian.
        """
        # Concatenate x & y then project
        h = self.input_mlp(torch.cat([x, y], dim=-1))  # (B, N, d_model)
        # Self-attention stack
        s = self.self_attn1(h, h, h)
        s = self.self_attn2(s, s, s)
        # Pool across context points
        s_C = s.mean(dim=1)                            # (B, d_model)
        stats = self.stat_mlp(s_C)                     # (B, 2*z_dim)
        mu, log_sigma = stats.chunk(2, dim=-1)

        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)   # 1

        # Re-parameterisation trick
        eps = torch.randn_like(sigma)
        z   = mu + sigma * eps
        return z, mu, sigma


# -------------------------------------------------
# Deterministic encoder r: (x_ctx, y_ctx, x_tgt) → representation
# -------------------------------------------------
class DeterministicEncoder(nn.Module):
    """
    Produces a deterministic representation r for each target x_tgt
    via cross-attention from context.
    """
    def __init__(self, x_dim, y_dim, d_model, n_head, dropout=0.0):
        super().__init__()
        # Embed each (x_ctx, y_ctx)
        self.input_mlp = nn.Sequential(
            nn.Linear(x_dim + y_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.self_attn1 = Attention(d_model, n_head, dropout)
        self.self_attn2 = Attention(d_model, n_head, dropout)

        # Separate linears for keys & queries
        self.key_mlp   = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(x_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.cross_attn1 = Attention(d_model, n_head, dropout)
        self.cross_attn2 = Attention(d_model, n_head, dropout)

    def forward(self, x_ctx, y_ctx, x_tgt):
        """
        Return r : (B, N_tgt, d_model)
        """
        # Context embedding
        h = self.input_mlp(torch.cat([x_ctx, y_ctx], dim=-1))
        h = self.self_attn1(h, h, h)
        h = self.self_attn2(h, h, h)
        v = h  # values for cross attention

        # K & Q come from context and target respectively
        k = self.key_mlp(x_ctx)
        q = self.key_mlp(x_tgt)

        # Two-layer cross-attention
        r = self.cross_attn1(q, k, v)
        r = self.cross_attn2(r, k, v)
        return r


# -------------------------------------------------
# Decoder: (r, z, x_tgt) → predictive distribution p(y|x_tgt, r, z)
# -------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, x_dim: int, d_model: int, y_dim: int = 1):
        super().__init__()
        hidden = d_model
        # Four-layer MLP ending in (μ, log σ_y)
        self.net = nn.Sequential(
            nn.Linear(x_dim + d_model + d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * y_dim)
        )

    def forward(self,
                r: torch.Tensor,     # deterministic representation
                z: torch.Tensor,     # latent sample
                x_tgt: torch.Tensor  # target inputs
               ):
        B, N, _ = r.size()
        # Broadcast z over N_tgt if needed
        if z.dim() == 2:
            z = z.unsqueeze(1).expand(-1, N, -1)
        inp = torch.cat([x_tgt, r, z], dim=-1)
        stats = self.net(inp)                         # (B, N, 2*y_dim)
        mu_y, log_sigma_y = stats.chunk(2, dim=-1)
        sigma_y = 0.1 + 0.9 * F.softplus(log_sigma_y)
        return mu_y, sigma_y


from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


# -------------------------------------------------
# Full Attentive Neural Process
# -------------------------------------------------
class AttentiveNeuralProcess(nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim=1,
        d_model=128,
        z_dim=128,
        n_head=8,
        dropout=0.0,
    ):
        super().__init__()
        self.latent_encoder = LatentEncoder(x_dim, y_dim, d_model, z_dim, n_head)
        self.det_encoder    = DeterministicEncoder(x_dim, y_dim, d_model, n_head, dropout)
        self.decoder        = Decoder(x_dim, d_model, y_dim)

    def forward(self, x_ctx, y_ctx, x_all, y_all, x_tgt, y_tgt=None):
        """
        Training mode returns (μ_y, σ_y, KL, total_loss, NLL).
        Evaluation mode returns (μ_y, σ_y).
        """

        # Prior from context only
        z_p, mu_p, sig_p = self.latent_encoder(x_ctx, y_ctx)

        # During training, sample posterior using targets as well
        if self.training:
            z_q, mu_q, sig_q = self.latent_encoder(x_all, y_all)
            z = z_q
        else:  # evaluation: use prior
            mu_q, sig_q = mu_p, sig_p
            z = z_p

        # Deterministic path
        r = self.det_encoder(x_ctx, y_ctx, x_tgt)

        # Decode predictions
        mu_y, sig_y = self.decoder(r, z, x_tgt)

        if y_tgt is not None:
            nll = gaussian_nll(mu_y, sig_y, y_tgt).mean()
            kl  = kl_div(mu_q, sig_q, mu_p, sig_p).mean()
            loss = nll + kl / y_tgt.size(1)
            return mu_y, sig_y, kl, loss, nll
        else:
            return mu_y, sig_y


# -------------------------------------------------
# Loss helpers
# -------------------------------------------------
def gaussian_nll(mu, sigma, y):
    """Negative log-likelihood of y under N(μ, σ²)."""
    return 0.5 * (torch.log(2 * math.pi * sigma**2) + (y - mu) ** 2 / sigma**2)


def kl_div(mu_q, sig_q, mu_p, sig_p):
    """KL divergence KL[q(z)||p(z)] for diagonal Gaussians."""
    var_q = sig_q.pow(2)
    var_p = sig_p.pow(2)
    kl_elem = (var_q + (mu_q - mu_p).pow(2)) / var_p - 1 \
              + 2 * (torch.log(sig_p) - torch.log(sig_q))
    kl = 0.5 * kl_elem.sum(dim=-1)  # sum over latent dimension
    return kl


# -------------------------------------------------
# Single optimisation step
# -------------------------------------------------
def train_step(model, opt, X, Y, num_ctx, min_ctx, device, kl_weight):
    """Run one optimisation step for ANP.

    Args:
        model: The AttentiveNeuralProcess instance.
        opt: Optimiser handling model parameters.
        X: Full feature tensor (N, x_dim).
        Y: Full target tensor  (N, 1) or (N, y_dim).
        num_ctx: Batch size (context + target) sent to the model.
        min_ctx: Minimum number of context points per batch.
        device: 'cpu' or 'cuda'.

    Returns:
        Tuple[float, float, float]: loss, negative-log-likelihood, KL divergence.
    """
    x_c, y_c, x_all, y_all, x_t, y_t, n_ctx = make_np_batch(X, Y, num_ctx, min_ctx)
    x_c = x_c.unsqueeze(0).to(device)
    y_c = y_c.unsqueeze(0).to(device)
    x_all = x_all.unsqueeze(0).to(device)
    y_all = y_all.unsqueeze(0).to(device)
    x_t = x_t.unsqueeze(0).to(device)
    y_t = y_t.unsqueeze(0).to(device)

    mu_y, sig_y, kl, loss, nll = model(x_c, y_c, x_all, y_all, x_t, y_t)

    loss = nll + kl_weight * kl

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item(), nll.item(), kl.item()


# -------------------------------------------------
# Training / evaluation loop
# -------------------------------------------------
def main(config):
    """
    Main training loop: sets seeds, prepares data, trains ANP,
    then evaluates on held-out test set and returns metrics.
    """
    # ---- Reproducibility ----
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- Data ----
    X_tr, y_tr, X_te, y_te, x_scaler, y_scaler = load_and_preprocess(
        config["dataset_path"],
        config["target_col"],
        config["train_size"],
    )

    # ---- Model & optimiser ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttentiveNeuralProcess(
        x_dim=X_tr.shape[1],
        y_dim=1,
        d_model=config["d_model"],
        z_dim=config["z_dim"],
        n_head=config["n_head"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["T_max"], eta_min=config["lr_min"]
    )

    kl_warmup_epochs = 2000

    # ---- Training loop ----
    losses = []  # 记录每个 epoch 的总 loss
    nlls = []  # 如果你也想画 nll
    kls = []  # 或者 KL

    # ---- Training loop ----
    for epoch in range(1, config["epochs"] + 1):
        model.train()

        kl_weight = min(1.0, epoch / kl_warmup_epochs)

        loss, nll, kl = train_step(
            model, optimizer, X_tr, y_tr,
            num_ctx=config["num_ctx"],
            min_ctx=config["min_ctx"],
            device=device,
            kl_weight=kl_weight
        )
        scheduler.step()

        losses.append(loss)
        nlls.append(nll)
        kls.append(kl)

        if epoch % config["log_freq"] == 0:
            print(f"Epoch {epoch:4d} → loss={loss:.4f}, nll={nll:.4f}, "
                  f"kl={kl:.4f}, weight={kl_weight:.2f}")


    # epochs = range(1, config["epochs"] + 1)
    #
    # # 1. 计算滑动平均
    # window_size = 50  # 你可以改成 20、100 看看效果
    # loss_series = pd.Series(losses)
    # loss_smooth = loss_series.rolling(window=window_size, min_periods=1).mean()
    #
    # # 2. 画图
    # plt.figure(figsize=(6, 4))
    # plt.plot(epochs, losses, alpha=0.3, label="Raw Loss")
    # plt.plot(epochs, loss_smooth, linewidth=2, label=f"Smooth Loss (w={window_size})")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title(f"Seed {config['seed']} Loss Curve")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # ---- Evaluation ----
    model.eval()
    with torch.no_grad():
        # 1) context = 训练集
        x_c = X_tr.unsqueeze(0).to(device)
        y_c = y_tr.unsqueeze(0).to(device)
        # 2) targets = 测试集
        x_t = X_te.unsqueeze(0).to(device)
        y_t = y_te.unsqueeze(0).to(device)
        # 3) 后验集合 = context ∪ target
        x_all = torch.cat([x_c, x_t], dim=1)
        y_all = torch.cat([y_c, y_t], dim=1)

        # 4) 正确调用 forward（y_tgt 用默认的 None）
        mu_y, _ = model(x_c, y_c, x_all, y_all, x_t)

        # 5) 反标准化、计算指标
        mu_y = mu_y.squeeze(0).cpu().numpy()
        y_pred = y_scaler.inverse_transform(mu_y)
        y_true = y_scaler.inverse_transform(y_t.squeeze(0).cpu().numpy())

    # ---- Metrics ----
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    ev   = explained_variance_score(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"→ Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, "
          f"ExpVar: {ev:.4f}, R²: {r2:.4f}")
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAPE(%)": mape,
        "ExplainedVar": ev,
        "R2": r2
    }


# -------------------------------------------------
# Entry point for multiple random seeds
# -------------------------------------------------
if __name__ == "__main__":
    base_config = {
        "train_size": 40,
        "dataset_path": "./Nitride (Dataset 1) NTi.csv",
        "target_col": "Thickness",
        "lr": 1e-3,
        "lr_min": 1e-4,
        "T_max": 500,
        "epochs": 2000,
        "num_ctx": 32,
        "min_ctx": 3,
        "d_model": 128,
        "z_dim": 128,
        "n_head": 8,
        "dropout": 0.0,
        "log_freq": 50
    }

    all_stats = []
    for seed in range(47,48):
        config = base_config.copy()
        config["seed"] = seed
        print(f"\n=== Running seed {seed} ===")
        stats = main(config)
        stats["seed"] = seed
        all_stats.append(stats)

    # Save aggregate results
    df = pd.DataFrame(all_stats)
    df.to_csv("ANP.csv", index=False)
    print("✅ All results saved to ANP.csv")
