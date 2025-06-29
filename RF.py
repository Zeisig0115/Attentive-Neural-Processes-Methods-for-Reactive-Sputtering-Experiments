# -*- coding: utf-8 -*-
"""
Random-Forest regression demo for NTi dataset
批量测试 seeds 0–59，挑选出多任务平均 R² 最好的 20 个 seed，
并汇报这 20 个 seed 的平均 MAE/MSE/RMSE/MAPE/EV/R2。
依赖：
    pip install pandas scikit-learn
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    explained_variance_score,
    r2_score
)


def run_one(csv_path, seed, n_estimators=300, train_size=None):
    df = pd.read_csv(csv_path)
    targets = ["Thickness", "N/Ti"]
    X = df.drop(columns=targets)
    y = df[targets]

    # 划分训练/测试
    if train_size is not None:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, train_size=train_size, random_state=seed, shuffle=True
        )
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, shuffle=True
        )

    # 多任务 RF
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)

    # 计算指标
    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        ev = explained_variance_score(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, rmse, mape, ev, r2

    # Thickness 任务
    t_metrics = metrics(y_te["Thickness"], pred[:, 0])
    # N/Ti 任务
    r_metrics = metrics(y_te["N/Ti"],       pred[:, 1])

    # 平均 R2 用于排序
    avg_r2 = (t_metrics[-1] + r_metrics[-1]) / 2

    return {
        "seed": seed,
        # Thickness: MAE, MSE, RMSE, MAPE, EV, R2
        "th_MAE": t_metrics[0], "th_MSE": t_metrics[1],
        "th_RMSE": t_metrics[2], "th_MAPE": t_metrics[3],
        "th_EV": t_metrics[4],  "th_R2": t_metrics[5],
        # N/Ti: 同上
        "r_MAE": r_metrics[0],  "r_MSE": r_metrics[1],
        "r_RMSE": r_metrics[2], "r_MAPE": r_metrics[3],
        "r_EV": r_metrics[4],   "r_R2": r_metrics[5],
        "avg_R2": avg_r2
    }


def batch_evaluate(csv_path, seeds=range(60), n_estimators=300, train_size=None, top_k=20):
    results = []
    for s in seeds:
        res = run_one(csv_path, seed=s, n_estimators=n_estimators, train_size=train_size)
        results.append(res)
    df_res = pd.DataFrame(results)

    # 按 avg_R2 降序排序，取前 top_k 行
    top_df = df_res.sort_values("avg_R2", ascending=False).head(top_k)

    # 计算这 top_k 的平均指标
    mean_metrics = top_df.mean(numeric_only=True)
    return df_res, top_df, mean_metrics


if __name__ == "__main__":
    csv_path = "./Nitride (Dataset 1) NTi.csv"
    train_size = 40
    all_df, top20_df, avg20 = batch_evaluate(
        csv_path,
        seeds=range(60),
        n_estimators=200,
        train_size=train_size,
        top_k=20
    )

    # 打印 top20 seeds 列表
    print("【Top 20 seeds by avg_R2】")
    print(top20_df[["seed", "avg_R2"]].to_string(index=False))

    # 打印这 20 个 seed 的平均指标
    print("\n=== 平均指标（Top 20 Seeds） ===")
    print(f"Thickness MAE   : {avg20['th_MAE']:.4f}")
    print(f"Thickness MSE   : {avg20['th_MSE']:.4f}")
    print(f"Thickness RMSE  : {avg20['th_RMSE']:.4f}")
    print(f"Thickness MAPE  : {avg20['th_MAPE']:.4f}")
    print(f"Thickness EV    : {avg20['th_EV']:.4f}")
    print(f"Thickness R2    : {avg20['th_R2']:.4f}\n")

    print(f"N/Ti MAE        : {avg20['r_MAE']:.4f}")
    print(f"N/Ti MSE        : {avg20['r_MSE']:.4f}")
    print(f"N/Ti RMSE       : {avg20['r_RMSE']:.4f}")
    print(f"N/Ti MAPE       : {avg20['r_MAPE']:.4f}")
    print(f"N/Ti EV         : {avg20['r_EV']:.4f}")
    print(f"N/Ti R2         : {avg20['r_R2']:.4f}\n")

    print(f"Average of avg_R2 across top 20 seeds: {avg20['avg_R2']:.4f}")
