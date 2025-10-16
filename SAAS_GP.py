import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll, fit_fully_bayesian_model_nuts
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.manual_seed(42)

# df = pd.read_csv("./data/Nitride (Dataset 1) NTi.csv")
df = pd.read_csv("./data/Thickness.csv")
feature_cols = ["Process", "Target", "Power", "Voltage", "Current", "N2/Ar", "Deposition Time"]
# feature_cols = ["Power", "Current", "Voltage", "Time", "Ar", "N2", "Pressure", "RI"]
# feature_cols = ["Pressure", "RI"] # database 2 N/Ti
# feature_cols = ["Power", "Current", "Voltage", "Time", "Ar", "N2", "Pressure", "RI"] # database 2 Thickness
X = torch.tensor(df[feature_cols].values, dtype=torch.double)
y = torch.tensor(df["Thickness"].values, dtype=torch.double).unsqueeze(-1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

metrics_exact = {"rmse": [], "mape": [], "r2": []}
metrics_saas  = {"rmse": [], "mape": [], "r2": []}



import torch
import torch.nn.functional as F

def get_ard_lengthscales(gp_model):
    cov = gp_model.covar_module
    kern = cov.base_kernel if hasattr(cov, "base_kernel") else cov
    if hasattr(kern, "lengthscale"):
        ls = kern.lengthscale.detach().cpu().numpy().flatten()
        return ls

    elif hasattr(kern, "raw_lengthscale"):
        raw = kern.raw_lengthscale.detach().cpu()  # [num_samples, d]
        ls_samples = F.softplus(raw)               # [num_samples, d]
        return ls_samples.mean(dim=0).numpy()      # posterior mean

    else:
        raise RuntimeError("Error")


for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # —— 1) Exact GP ——
    gp_exact = SingleTaskGP(
        X_train, y_train,
        input_transform=Normalize(d=X_train.size(-1)),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp_exact.likelihood, gp_exact)
    fit_gpytorch_mll(mll, options={"max_iter": 100})

    gp_exact.eval(); gp_exact.likelihood.eval()
    with torch.no_grad():
        post_exact = gp_exact.posterior(X_test)
        mean_e = post_exact.mean.squeeze(-1).cpu().numpy()
    y_true = y_test.squeeze(-1).cpu().numpy()

    rmse_e = np.sqrt(mean_squared_error(y_true, mean_e))
    mape_e = mean_absolute_percentage_error(y_true, mean_e)
    r2_e   = r2_score(y_true, mean_e)
    metrics_exact["rmse"].append(rmse_e)
    metrics_exact["mape"].append(mape_e)
    metrics_exact["r2"].append(r2_e)
    print(f"[Exact GP] Fold {fold}: RMSE={rmse_e:.4f}, MAPE={mape_e:.4f}, R2={r2_e:.4f}")

    # —— 2) SAASBO (Fully Bayesian GP with NUTS) ——
    gp_saas = SaasFullyBayesianSingleTaskGP(
        train_X=X_train,
        train_Y=y_train,
        input_transform=Normalize(d=X_train.size(-1)),
        outcome_transform=Standardize(m=1),
    )
    fit_fully_bayesian_model_nuts(
        gp_saas,
        max_tree_depth=6,
        warmup_steps=512,
        num_samples=256,
        thinning=16,
        disable_progbar=True,
        jit_compile=False,
    )
    gp_saas.eval()

    with torch.no_grad():
        post_s = gp_saas.posterior(X_test)
        mean_s = post_s.mean.mean(dim=0).squeeze(-1).cpu().numpy()

    rmse_s = np.sqrt(mean_squared_error(y_true, mean_s))
    mape_s = mean_absolute_percentage_error(y_true, mean_s)
    r2_s   = r2_score(y_true, mean_s)
    metrics_saas["rmse"].append(rmse_s)
    metrics_saas["mape"].append(mape_s)
    metrics_saas["r2"].append(r2_s)
    print(f"[SAASBO   ] Fold {fold}: RMSE={rmse_s:.4f}, MAPE={mape_s:.4f}, R2={r2_s:.4f}")

    std_s = np.sqrt(post_s.variance.mean(dim=0).squeeze(-1).cpu().numpy())
    lower = mean_s - 1.96 * std_s
    upper = mean_s + 1.96 * std_s

    # plt.figure()
    # plt.errorbar(y_true, mean_s, yerr=1.96*std_s, fmt='o', ecolor='gray', capsize=3)
    # plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    # plt.xlabel('True N/Ti')
    # plt.ylabel('Predicted N/Ti')
    # plt.title(f'Fold {fold} SAASBO Pred vs True (95% CI)')
    # plt.show()

    ls_exact = get_ard_lengthscales(gp_exact)
    ls_saas = get_ard_lengthscales(gp_saas)

    print("Exact GP lengthscales:")
    for feat, val in zip(feature_cols, ls_exact):
        print(f"  {feat:>8s} : {val:.4f}")

    print("SAASBO posterior mean lengthscales:")
    for feat, val in zip(feature_cols, ls_saas):
        print(f"  {feat:>8s} : {val:.4f}")

for name, m in [("Exact GP", metrics_exact), ("SAASBO", metrics_saas)]:
    mean_rmse = np.mean(m["rmse"]); std_rmse = np.std(m["rmse"], ddof=1)
    mean_mape = np.mean(m["mape"]); std_mape = np.std(m["mape"], ddof=1)
    mean_r2   = np.mean(m["r2"]);   std_r2   = np.std(m["r2"],   ddof=1)
    print(f"\n{name} Overall (5-fold):")
    print(f"  RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"  MAPE = {mean_mape:.4f} ± {std_mape:.4f}")
    print(f"  R2   = {mean_r2:.4f} ± {std_r2:.4f}")
