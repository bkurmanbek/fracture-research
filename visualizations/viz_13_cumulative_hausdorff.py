"""
viz_13_cumulative_hausdorff.py
Empirical CDF (ECDF) curves of Hausdorff distances per model — all on one plot.
Useful for showing the proportion of fractures within a given error threshold.
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
THRESHOLDS = [0.05, 0.10, 0.20]    # mark vertical reference lines
OUT = 'plots/13_cumulative_hausdorff.png'

fig, ax = plt.subplots(figsize=(11, 7))

for (model, path), color in zip(MODELS.items(), COLORS):
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    if 'hausdorff' not in df.columns:
        continue
    vals = df['hausdorff'].dropna().values
    vals = vals[np.isfinite(vals) & (vals >= 0)]
    if len(vals) == 0:
        continue

    sorted_vals = np.sort(vals)
    ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    # Add (0,0) for clean plot start
    sorted_vals = np.concatenate([[0], sorted_vals])
    ecdf = np.concatenate([[0], ecdf])

    med = np.median(vals[vals > 0])
    ax.step(sorted_vals, ecdf, color=color, linewidth=2.5,
            label=f'{model}  (med={med:.3f})')

# Reference thresholds
for th in THRESHOLDS:
    ax.axvline(th, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(th, 0.02, f'{th}', ha='center', va='bottom', fontsize=9, color='gray')

ax.set_xlabel('Hausdorff Distance (normalized)', fontsize=12)
ax.set_ylabel('Cumulative Proportion of Fractures', fontsize=12)
ax.set_title('Empirical CDF of Hausdorff Distance per Model', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(0, 1.02)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
