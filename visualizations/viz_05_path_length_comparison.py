"""
viz_05_path_length_comparison.py
Scatter + histogram: True path length vs. Generated path length per model.
Shows how well each model reproduces the correct number of steps.
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
OUT = 'plots/05_path_length_comparison.png'

fig, axes = plt.subplots(2, 4, figsize=(22, 10))

for col, ((model, path), color) in enumerate(zip(MODELS.items(), COLORS)):
    ax_scatter = axes[0, col]
    ax_hist    = axes[1, col]

    if not os.path.exists(path):
        ax_scatter.set_title(f'{model}\n(no data)')
        continue

    df = pd.read_csv(path)
    true_col = 'true_n_pts' if 'true_n_pts' in df.columns else 'true_length'
    gen_col  = 'gen_n_pts'  if 'gen_n_pts'  in df.columns else 'gen_length'
    true_n = df[true_col].values
    gen_n  = df[gen_col].values
    mask = np.isfinite(true_n) & np.isfinite(gen_n)
    true_n, gen_n = true_n[mask], gen_n[mask]

    # Scatter: true vs generated lengths
    ax_scatter.scatter(true_n, gen_n, alpha=0.5, s=25, color=color, edgecolors='none')
    lim = max(true_n.max(), gen_n.max()) * 1.05
    ax_scatter.plot([0, lim], [0, lim], 'k--', linewidth=1.5, label='Perfect')
    ax_scatter.set_xlabel('True # Points', fontsize=10)
    ax_scatter.set_ylabel('Generated # Points', fontsize=10)
    ax_scatter.set_title(model, fontsize=11, fontweight='bold')
    ax_scatter.legend(fontsize=9)
    ax_scatter.grid(True, alpha=0.3)

    # Histogram overlay
    bins = np.arange(1, max(true_n.max(), gen_n.max()) + 2) - 0.5
    ax_hist.hist(true_n, bins=bins, density=True, alpha=0.5, color='#2c7bb6', label='True')
    ax_hist.hist(gen_n,  bins=bins, density=True, alpha=0.5, color=color,     label='Generated')
    ax_hist.set_xlabel('# Points', fontsize=10)
    ax_hist.set_ylabel('Density', fontsize=10)
    ax_hist.set_title(f'{model} — Length Dist.', fontsize=10, fontweight='bold')
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3)

plt.suptitle('True vs. Generated Fracture Path Lengths', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
