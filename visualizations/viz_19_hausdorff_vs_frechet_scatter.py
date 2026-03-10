"""
viz_19_hausdorff_vs_frechet_scatter.py
Scatter plot of Hausdorff vs Fréchet distance colored by model.
Shows the relationship between the two shape-distance metrics.
Points near y=x indicate both metrics agree; spread shows divergence.
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
OUT = 'plots/19_hausdorff_vs_frechet.png'

fig, ax = plt.subplots(figsize=(10, 8))
all_vals = []

for (model, path), color in zip(MODELS.items(), COLORS):
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    if 'hausdorff' not in df.columns or 'frechet' not in df.columns:
        continue
    h = df['hausdorff'].dropna().values
    f = df['frechet'].dropna().values
    # Align by index
    df_clean = df[['hausdorff', 'frechet']].dropna()
    h = df_clean['hausdorff'].values
    f = df_clean['frechet'].values
    mask = np.isfinite(h) & np.isfinite(f) & (h >= 0) & (f >= 0)
    h, f = h[mask], f[mask]
    if len(h) == 0:
        continue
    all_vals.extend(h.tolist() + f.tolist())
    ax.scatter(h, f, alpha=0.4, s=18, color=color, edgecolors='none', label=model)

if all_vals:
    lim = np.percentile(all_vals, 98) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1.5, label='y = x', alpha=0.6)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

ax.set_xlabel('Hausdorff Distance', fontsize=12)
ax.set_ylabel('Fréchet Distance', fontsize=12)
ax.set_title('Hausdorff vs. Fréchet Distance per Fracture Path',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
