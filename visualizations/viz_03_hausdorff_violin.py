"""
viz_03_hausdorff_violin.py
Violin + box plot of Hausdorff distances across all 4 models.
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
PALETTE = ['#4dac26', '#b8e186', '#d01c8b', '#f1b6da']
OUT = 'plots/03_hausdorff_violin.png'

data, labels = [], []
for model, path in MODELS.items():
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    vals = df['hausdorff'].dropna().values
    vals = vals[np.isfinite(vals)]
    if len(vals) > 0:
        data.append(vals)
        labels.append(model)

fig, ax = plt.subplots(figsize=(10, 7))
vp = ax.violinplot(data, positions=range(len(data)), showmedians=True, showextrema=True)

for i, (body, color) in enumerate(zip(vp['bodies'], PALETTE[:len(data)])):
    body.set_facecolor(color)
    body.set_alpha(0.6)

vp['cmedians'].set_color('black')
vp['cmedians'].set_linewidth(2.5)
vp['cmaxes'].set_linewidth(1.5)
vp['cmins'].set_linewidth(1.5)
vp['cbars'].set_linewidth(1.5)

# Overlay mean dots
for i, vals in enumerate(data):
    ax.scatter(i, np.mean(vals), color='black', zorder=5, s=60, marker='D', label='Mean' if i == 0 else '')

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.set_ylabel('Hausdorff Distance (normalized units)', fontsize=12)
ax.set_title('Hausdorff Distance Distribution per Model', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
