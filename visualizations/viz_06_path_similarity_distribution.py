"""
viz_06_path_similarity_distribution.py
KDE + rug plot of Path Similarity scores (0-1) per model.
Higher = better directional agreement between true and generated paths.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

MODELS = {
    'BiLSTM-Attn':  '../fracture_results/case1/path_metrics.csv',
    'Trans-GAT':    '../fracture_results/case2/path_metrics.csv',
    'LSTM-Stop':    '../fracture_results/case3/path_metrics.csv',
    'CNN-GRU-MDN':  '../fracture_results/case4/path_metrics.csv',
}
COLORS = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00']
OUT = 'plots/06_path_similarity_distribution.png'

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for ax, ((model, path), color) in zip(axes, zip(MODELS.items(), COLORS)):
    if not os.path.exists(path):
        ax.set_title(f'{model}\n(no data)', fontsize=12)
        continue

    df = pd.read_csv(path)
    if 'path_similarity' not in df.columns:
        ax.set_title(f'{model}\n(no path_similarity column)', fontsize=12)
        continue

    vals = df['path_similarity'].dropna().values
    vals = vals[np.isfinite(vals) & (vals >= 0) & (vals <= 1)]
    if len(vals) < 2:
        ax.set_title(f'{model}\n(insufficient data)', fontsize=12)
        continue

    # KDE
    kde = gaussian_kde(vals, bw_method='scott')
    xs = np.linspace(0, 1, 400)
    ax.fill_between(xs, kde(xs), alpha=0.25, color=color)
    ax.plot(xs, kde(xs), color=color, linewidth=2.5)

    # Rug plot
    ax.plot(vals, np.zeros_like(vals) - 0.05 * kde(xs).max(),
            '|', color=color, alpha=0.3, markersize=8)

    # Stats
    med = np.median(vals)
    mu = np.mean(vals)
    ax.axvline(med, color='black', linestyle='--', linewidth=1.5, label=f'Median={med:.3f}')
    ax.axvline(mu,  color='gray',  linestyle=':',  linewidth=1.5, label=f'Mean={mu:.3f}')

    ax.set_xlim(0, 1)
    ax.set_xlabel('Path Similarity', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(model, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Path Similarity Score Distributions per Model\n(1 = perfect directional match)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
