"""
viz_18_length_error_vs_path_length.py
Scatter of Length Error vs. True Path Length per model.
Reveals whether longer fractures are harder to predict (error grows with length?).
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
OUT = 'plots/18_length_error_vs_path_length.png'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, ((model, path), color) in zip(axes, zip(MODELS.items(), COLORS)):
    if not os.path.exists(path):
        ax.set_title(f'{model}\n(no data)', fontsize=12)
        continue
    df = pd.read_csv(path)
    if 'length_error' not in df.columns or 'true_n_pts' not in df.columns:
        ax.set_title(f'{model}\n(columns missing)', fontsize=12)
        continue

    n_pts = df['true_n_pts'].values
    err   = df['length_error'].values
    mask  = np.isfinite(n_pts) & np.isfinite(err)
    n_pts, err = n_pts[mask], err[mask]

    ax.scatter(n_pts, err, alpha=0.45, s=20, color=color, edgecolors='none')

    # Rolling mean trend line
    sort_idx = np.argsort(n_pts)
    n_sort, e_sort = n_pts[sort_idx], err[sort_idx]
    window = max(3, len(n_sort) // 10)
    if len(n_sort) >= window:
        roll_mean = pd.Series(e_sort).rolling(window, center=True, min_periods=1).mean().values
        ax.plot(n_sort, roll_mean, color='black', linewidth=2, label='Rolling mean')

    ax.set_xlabel('True # Points (path length)', fontsize=11)
    ax.set_ylabel('Length Error', fontsize=11)
    ax.set_title(model, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

plt.suptitle('Length Error vs. True Path Length\n(does error grow for longer fractures?)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
