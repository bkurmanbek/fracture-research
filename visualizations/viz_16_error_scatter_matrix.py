"""
viz_16_error_scatter_matrix.py
Scatter matrix (pair plot) of Hausdorff, Fréchet, Endpoint Error, and Length Error
for each model. Shows correlations between different error metrics.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

MODELS = {
    'BiLSTM-Attn':  '../fracture_results/case1/path_metrics.csv',
    'Trans-GAT':    '../fracture_results/case2/path_metrics.csv',
    'LSTM-Stop':    '../fracture_results/case3/path_metrics.csv',
    'CNN-GRU-MDN':  '../fracture_results/case4/path_metrics.csv',
}
COLORS = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00']
METRICS = ['hausdorff', 'frechet', 'endpoint_error', 'length_error']
MLABELS = ['Hausdorff', 'Fréchet', 'Endpoint Error', 'Length Error']
OUT = 'plots/16_error_scatter_matrix.png'

n = len(METRICS)
fig, axes = plt.subplots(n, n, figsize=(16, 14))

# Collect data
dfs = {}
for model, path in MODELS.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[[m for m in METRICS if m in df.columns]].dropna()
        dfs[model] = df

for row_i, m_row in enumerate(METRICS):
    for col_j, m_col in enumerate(METRICS):
        ax = axes[row_i, col_j]
        if row_i == col_j:
            # Diagonal: KDE of single metric
            for (model, df), color in zip(dfs.items(), COLORS):
                if m_row not in df.columns:
                    continue
                vals = df[m_row].values
                vals = vals[np.isfinite(vals)]
                if len(vals) < 2:
                    continue
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(vals, bw_method='scott')
                xs = np.linspace(vals.min(), vals.max(), 200)
                ax.plot(xs, kde(xs), color=color, linewidth=2, label=model)
            ax.set_ylabel('Density', fontsize=7)
        else:
            # Off-diagonal: scatter
            for (model, df), color in zip(dfs.items(), COLORS):
                if m_col not in df.columns or m_row not in df.columns:
                    continue
                x_vals = df[m_col].values
                y_vals = df[m_row].values
                mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                ax.scatter(x_vals[mask], y_vals[mask],
                           color=color, alpha=0.35, s=10, edgecolors='none')

        if row_i == n - 1:
            ax.set_xlabel(MLABELS[col_j], fontsize=8)
        else:
            ax.set_xticklabels([])
        if col_j == 0:
            ax.set_ylabel(MLABELS[row_i], fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

# Legend on first diagonal
handles = [plt.Line2D([0], [0], color=c, linewidth=2, label=m)
           for m, c in zip(dfs.keys(), COLORS)]
axes[0, n-1].legend(handles=handles, fontsize=8, loc='upper right')

plt.suptitle('Error Metric Correlation Matrix (All Models)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
