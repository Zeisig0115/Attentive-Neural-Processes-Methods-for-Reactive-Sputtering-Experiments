import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset (update the path if needed)
df = pd.read_csv("./data/Nitride (Dataset 1) NTi.csv")

# Split with fixed seed
seed = 34
np.random.seed(seed)
indices = np.random.permutation(len(df))
train = df.iloc[indices[:40]].reset_index(drop=True)
test = df.iloc[indices[40:]].reset_index(drop=True)

# Plot all features in one large figure with subplots
features = df.columns.tolist()
n_features = len(features)
n_cols = 3
n_rows = int(np.ceil(n_features / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes = axes.flatten()

for idx, col in enumerate(features):
    ax = axes[idx]
    ax.hist(train[col].dropna(), bins=20, alpha=0.6)
    ax.hist(test[col].dropna(), bins=20, alpha=0.6)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    if idx == 0:
        ax.legend(['Train', 'Test'])

# Turn off any unused subplots
for j in range(n_features, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
