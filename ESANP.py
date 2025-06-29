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
from torch.optim.lr_scheduler import ExponentialLR


def load_and_preprocess(
    csv_path: str,
    target_col: str,
    train_size: int,
    val_size: int,
    seed: int = None,
):
    df = pd.read_csv(Path(csv_path))
    total_n = len(df)

    rng = np.random.RandomState(seed)
    idx = np.arange(total_n)
    rng.shuffle(idx)

    tr_idx  = idx[:train_size]
    val_idx = idx[train_size:train_size+val_size]
    te_idx  = idx[train_size+val_size:]

    df_tr  = df.iloc[tr_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_te  = df.iloc[te_idx].reset_index(drop=True)

    def to_arrays(dframe):
        X = dframe.drop(columns=[target_col]).values.astype("float32")
        y = dframe[target_col].values.astype("float32").reshape(-1, 1)
        return X, y

    X_tr_np, y_tr_np   = to_arrays(df_tr)
    X_val_np, y_val_np = to_arrays(df_val)
    X_te_np, y_te_np   = to_arrays(df_te)

    x_scaler = StandardScaler().fit(X_tr_np)
    y_scaler = StandardScaler().fit(y_tr_np)

    X_tr  = x_scaler.transform(X_tr_np).astype("float32")
    X_val = x_scaler.transform(X_val_np).astype("float32")
    X_te  = x_scaler.transform(X_te_np).astype("float32")

    y_tr  = y_scaler.transform(y_tr_np).astype("float32")
    y_val = y_scaler.transform(y_val_np).astype("float32")
    y_te  = y_scaler.transform(y_te_np).astype("float32")

    to_tensor = lambda arr: torch.from_numpy(arr)
    return (
        to_tensor(X_tr), to_tensor(y_tr),
        to_tensor(X_val), to_tensor(y_val),
        to_tensor(X_te), to_tensor(y_te),
        x_scaler, y_scaler,
    )

# -------------------------------------------------
# Batch construction helper (unchanged)
# -------------------------------------------------
def make_np_batch(X, Y, num_ctx, min_ctx=3):
    N = X.size(0)
    if num_ctx > N:
        raise ValueError("num_ctx bigger than dataset")
    if min_ctx >= num_ctx:
        raise ValueError("min_ctx must be < num_ctx")
    x_all, y_all = X, Y
    n_ctx = torch.randint(min_ctx, num_ctx, (1,)).item()
    c_idx = torch.randperm(N)[:n_ctx]
    mask = torch.ones(N, dtype=torch.bool)
    mask[c_idx] = False
    t_idx = torch.where(mask)[0]
    x_ctx, y_ctx = x_all[c_idx], y_all[c_idx]
    x_tgt, y_tgt = x_all[t_idx], y_all[t_idx]
    return x_ctx, y_ctx, x_all, y_all, x_tgt, y_tgt, n_ctx


# -------------------------------------------------
# Model components (unchanged)
# -------------------------------------------------
class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init="linear"):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.lin.weight,
                                gain=nn.init.calculate_gain(w_init))
    def forward(self, x):
        return self.lin(x)

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.wq = Linear(d_model, d_model, w_init="linear")
        self.wk = Linear(d_model, d_model, w_init="linear")
        self.wv = Linear(d_model, d_model, w_init="linear")
        self.out_proj = Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        B, N_k, _ = key.size()
        _, N_q, _ = query.size()

        def proj(x, lin):
            x = lin(x)
            x = x.view(B, -1, self.n_head, self.d_k)
            return x.permute(0, 2, 1, 3)

        Q = proj(query, self.wq)
        K = proj(key,   self.wk)
        V = proj(value, self.wv)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * (self.d_k ** -0.5)
        attn   = F.softmax(scores, dim=-1)
        attn   = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, N_q, -1)
        return self.out_proj(out)

class Attention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        self.mha  = MultiheadAttention(d_model, n_head, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key, value):
        out = self.mha(query, key, value)
        out = self.drop(out)
        return self.norm(out + query)

class LatentEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, d_model, z_dim, n_head=8):
        super().__init__()
        # self.input_mlp = nn.Sequential(
        #     nn.Linear(x_dim + y_dim, d_model),nn.ReLU(),nn.LayerNorm(d_model),
        #     nn.Linear(d_model, d_model),nn.ReLU(),nn.LayerNorm(d_model),
        #     nn.Linear(d_model, d_model),nn.ReLU(),nn.LayerNorm(d_model),
        #     nn.Linear(d_model, d_model),
        # )
        self.input_mlp = nn.Sequential(
            nn.Linear(x_dim + y_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.self_attn1 = Attention(d_model, n_head)
        self.self_attn2 = Attention(d_model, n_head)
        self.stat_mlp   = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 2 * z_dim),
        )

    def forward(self, x, y):
        h = self.input_mlp(torch.cat([x, y], dim=-1))
        s = self.self_attn1(h, h, h)
        s = self.self_attn2(s, s, s)
        s_C = s.mean(dim=1)
        stats = self.stat_mlp(s_C)
        mu, log_sigma = stats.chunk(2, dim=-1)
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)
        eps   = torch.randn_like(sigma)
        z     = mu + sigma * eps
        return z, mu, sigma

class DeterministicEncoder(nn.Module):
    def __init__(self, x_dim, y_dim, d_model, n_head, dropout=0.0):
        super().__init__()
        self.input_mlp  = nn.Sequential(
            nn.Linear(x_dim + y_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model),       nn.ReLU(),
            nn.Linear(d_model, d_model),       nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.self_attn1 = Attention(d_model, n_head, dropout)
        self.self_attn2 = Attention(d_model, n_head, dropout)
        self.key_mlp    = nn.Sequential(
            nn.Linear(x_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.query_mlp  = nn.Sequential(
            nn.Linear(x_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.cross_attn1 = Attention(d_model, n_head, dropout)
        self.cross_attn2 = Attention(d_model, n_head, dropout)

    def forward(self, x_ctx, y_ctx, x_tgt):
        h = self.input_mlp(torch.cat([x_ctx, y_ctx], dim=-1))
        h = self.self_attn1(h, h, h)
        h = self.self_attn2(h, h, h)
        v = h
        k = self.key_mlp(x_ctx)
        q = self.query_mlp(x_tgt)
        r = self.cross_attn1(q, k, v)
        r = self.cross_attn2(r, k, v)
        return r

class Decoder(nn.Module):
    def __init__(self, x_dim, d_model, y_dim=1):
        super().__init__()
        hidden = d_model
        self.input_norm = nn.LayerNorm(x_dim + d_model + d_model)
        self.net = nn.Sequential(
            nn.Linear(x_dim + d_model + d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                    nn.ReLU(),
            nn.Linear(hidden, hidden),                    nn.ReLU(),
            nn.Linear(hidden, hidden),                    nn.ReLU(),
            nn.Linear(hidden, 2 * y_dim)
        )

    def forward(self, r, z, x_tgt):
        B, N, _ = r.size()
        if z.dim() == 2:
            z = z.unsqueeze(1).expand(-1, N, -1)
        inp         = torch.cat([x_tgt, r, z], dim=-1)

        # inp_normed = self.input_norm(inp)

        stats       = self.net(inp)
        mu_y, lsig  = stats.chunk(2, dim=-1)
        sigma_y     = 0.1 + 0.9 * F.softplus(lsig)
        return mu_y, sigma_y

class AttentiveNeuralProcess(nn.Module):
    def __init__(self, x_dim, y_dim=1, d_model=128, z_dim=128, n_head=8, dropout=0.0):
        super().__init__()
        self.latent_encoder = LatentEncoder(x_dim, y_dim, d_model, z_dim, n_head)
        self.det_encoder    = DeterministicEncoder(x_dim, y_dim, d_model, n_head, dropout)
        self.decoder        = Decoder(x_dim, d_model, y_dim)

    def forward(self, x_ctx, y_ctx, x_all, y_all, x_tgt, y_tgt=None):
        z_p, mu_p, sig_p = self.latent_encoder(x_ctx, y_ctx)
        if self.training:
            z_q, mu_q, sig_q = self.latent_encoder(x_all, y_all)
            z = z_q
        else:
            mu_q, sig_q = mu_p, sig_p
            z = z_p

        r      = self.det_encoder(x_ctx, y_ctx, x_tgt)
        mu_y, sig_y = self.decoder(r, z, x_tgt)

        if y_tgt is not None:
            nll  = gaussian_nll(mu_y, sig_y, y_tgt).mean()
            kl   = kl_div(mu_q, sig_q, mu_p, sig_p).mean()
            loss = nll + kl / y_tgt.size(1)
            return mu_y, sig_y, kl, loss, nll
        else:
            return mu_y, sig_y

def gaussian_nll(mu, sigma, y):
    return 0.5 * (torch.log(2 * math.pi * sigma**2) + (y - mu)**2 / sigma**2)

def kl_div(mu_q, sig_q, mu_p, sig_p):
    var_q   = sig_q.pow(2)
    var_p   = sig_p.pow(2)
    kl_elem = (var_q + (mu_q - mu_p).pow(2)) / var_p - 1 \
             + 2 * (torch.log(sig_p) - torch.log(sig_q))
    return 0.5 * kl_elem.sum(dim=-1)

def train_step(model, opt, X, Y, num_ctx, min_ctx, device, kl_weight):
    x_c, y_c, x_all, y_all, x_t, y_t, _ = make_np_batch(X, Y, num_ctx, min_ctx)
    x_c, y_c = x_c.unsqueeze(0).to(device), y_c.unsqueeze(0).to(device)
    x_all, y_all = x_all.unsqueeze(0).to(device), y_all.unsqueeze(0).to(device)
    x_t, y_t     = x_t.unsqueeze(0).to(device),   y_t.unsqueeze(0).to(device)

    mu_y, sig_y, kl, _, nll = model(x_c, y_c, x_all, y_all, x_t, y_t)
    loss = nll + kl_weight * kl
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item(), nll.item(), kl.item()

# -------------------------------------------------
# Main with Early Stopping
# -------------------------------------------------
def main(config):
    # reproducibility
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # load data
    X_tr, y_tr, X_val, y_val, X_te, y_te, x_scaler, y_scaler = \
        load_and_preprocess(
            config["dataset_path"],
            config["target_col"],
            config["train_size"],
            config["val_size"],
            config["seed"]
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttentiveNeuralProcess(
        x_dim=X_tr.shape[1],
        y_dim=1,
        d_model=config["bottleneck"],
        z_dim=config["bottleneck"],
        n_head=config["n_head"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = ExponentialLR(
        optimizer,
        gamma=0.995  # 每个 epoch 学习率乘以 0.99
    )

    kl_warmup_epochs = 200
    best_val_nll    = float('inf')
    patience        = config.get("patience", 10)
    no_improve      = 0

    losses, nlls, kls = [], [], []

    for epoch in range(1, config["epochs"]+1):
        model.train()
        kl_weight = min(1.0, epoch / kl_warmup_epochs)
        loss, nll, kl = train_step(
            model, optimizer, X_tr, y_tr,
            num_ctx=config["num_ctx"],
            min_ctx=config["min_ctx"],
            device=device,
            kl_weight=kl_weight,
        )
        scheduler.step()

        losses.append(loss)
        nlls.append(nll)
        kls.append(kl)

        # --- 验证集评估 ---
        model.eval()
        with torch.no_grad():
            x_c = X_tr.unsqueeze(0).to(device)
            y_c = y_tr.unsqueeze(0).to(device)
            x_t = X_val.unsqueeze(0).to(device)
            y_t = y_val.unsqueeze(0).to(device)
            mu_y, _ = model(
                x_c, y_c,
                torch.cat([x_c, x_t], dim=1),
                torch.cat([y_c, y_t], dim=1),
                x_t
            )
            val_nll = gaussian_nll(mu_y, 0.1 + 0.9*torch.ones_like(mu_y), y_t)\
                      .mean().item()

        # early stopping check
        if val_nll + 1e-4 < best_val_nll:
            best_val_nll = val_nll
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"→ Early stopping at epoch {epoch}. Best val NLL={best_val_nll:.4f}")
                break

        if epoch % config["log_freq"] == 0:
            print(f"Epoch {epoch:4d} | train_loss={loss:.4f}, train_nll={nll:.4f}, "
                  f"train_kl={kl:.4f}, val_nll={val_nll:.4f}, kl_wt={kl_weight:.2f}")

    # # --- 画训练损失曲线 ---
    # epochs = range(1, len(losses)+1)
    # loss_smooth = pd.Series(losses).rolling(window=50, min_periods=1).mean()
    # plt.figure(figsize=(6,4))
    # plt.plot(epochs, loss_smooth, linewidth=2, label="Smooth Loss")
    # plt.xlabel("Epoch"); plt.ylabel("Loss")
    # plt.title(f"Seed {config['seed']} Loss Curve")
    # plt.grid(True); plt.legend(); plt.show()

    # --- 最佳模型载入 & 测试集评估 ---
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    with torch.no_grad():
        # x_c = torch.cat([X_tr, X_val], dim=0).unsqueeze(0).to(device)
        # y_c = torch.cat([y_tr, y_val], dim=0).unsqueeze(0).to(device)
        x_c = X_tr.unsqueeze(0).to(device)
        y_c = y_tr.unsqueeze(0).to(device)
        x_t = X_te.unsqueeze(0).to(device)
        y_t = y_te.unsqueeze(0).to(device)
        mu_y, _ = model(
            x_c, y_c,
            torch.cat([x_c, x_t], dim=1),
            torch.cat([y_c, y_t], dim=1),
            x_t
        )
        mu_np   = mu_y.squeeze(0).cpu().numpy()
        y_pred  = y_scaler.inverse_transform(mu_np)
        y_true  = y_scaler.inverse_transform(y_t.squeeze(0).cpu().numpy())

    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)*100
    ev   = explained_variance_score(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"→ Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%, "
          f"ExplainedVar: {ev:.4f}, R2: {r2:.4f}")

    return {"MSE": mse, "RMSE": rmse, "MAPE(%)": mape,
            "ExplainedVar": ev, "R2": r2}

# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    base_config = {
        "train_size": 32,
        "val_size": 8,
        "dataset_path": "./Nitride (Dataset 1) NTi.csv",
        "target_col": "Thickness",
        "lr": 1e-3,
        "epochs": 500,
        "num_ctx": 32,
        "min_ctx": 8,
        "bottleneck": 128,
        "n_head": 8,
        "dropout": 0.0,
        "log_freq": 10,
        "patience": 200,
    }

    all_stats = []
    for seed in range(47,48):
        config = base_config.copy()
        config["seed"] = int(seed)
        print(f"\n=== Seed {seed} ===")
        stats = main(config)
        stats["seed"] = int(seed)
        all_stats.append(stats)

    df = pd.DataFrame(all_stats)
    mean_stats = df.mean(numeric_only=True)
    std_stats  = df.std(numeric_only=True)

    print("\n--- Individual Runs ---")
    print(df.to_string(index=False))

    print("\n--- Summary (mean ± std) ---")
    for metric in ["MSE","RMSE","MAPE(%)","ExplainedVar","R2"]:
        m = mean_stats[metric]
        s = std_stats[metric]
        print(f"{metric:12s}: {m:.4f} ± {s:.4f}")
