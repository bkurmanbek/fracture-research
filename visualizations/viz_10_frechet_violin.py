"""
viz_10_frechet_violin.py
Violin + strip plot of Fréchet distances across all 4 models.
Fréchet captures the "leash" distance — the minimum leash length to
walk a dog from one path to another simultaneously.
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
PALETTE = ['#377eb8', '#4daf4a', '#e41a1c', '#ff7f00']
OUT = 'plots/10_frechet_violin.png'

data, labels, colors_used = [], [], []
for (model, path), color in zip(MODELS.items(), PALETTE):
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    if 'frechet' not in df.columns:
        continue
    vals = df['frechet'].dropna().values
    vals = vals[np.isfinite(vals) & (vals >= 0)]
    if len(vals) > 0:
        data.append(vals)
        labels.append(model)
        colors_used.append(color)

fig, ax = plt.subplots(figsize=(11, 7))
positions = range(len(data))
vp = ax.violinplot(data, positions=positions, showmedians=True,
                   showextrema=True, widths=0.65)

for body, color in zip(vp['bodies'], colors_used):
    body.set_facecolor(color)
    body.set_alpha(0.55)

vp['cmedians'].set_color('black')
vp['cmedians'].set_linewidth(2.5)
for part in ['cmaxes', 'cmins', 'cbars']:
    vp[part].set_linewidth(1.5)

# Strip plot (jittered raw points — sample up to 300 for readability)
rng = np.random.default_rng(42)
for i, (vals, color) in enumerate(zip(data, colors_used)):
    sample = vals if len(vals) <= 300 else rng.choice(vals, 300, replace=False)
    jitter = rng.uniform(-0.12, 0.12, size=len(sample))
    ax.scatter(i + jitter, sample, color=color, alpha=0.25, s=12, zorder=3)

# Mean diamonds
for i, vals in enumerate(data):
    ax.scatter(i, np.mean(vals), color='black', zorder=6, s=80, marker='D',
               label='Mean' if i == 0 else '')

# Quartile text annotations
for i, vals in enumerate(data):
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    ax.text(i, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else vals.max() * 1.1,
            f'Q1={q1:.3f}\nM={med:.3f}\nQ3={q3:.3f}',
            ha='center', va='bottom', fontsize=7.5, color='gray')

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.set_ylabel('Fréchet Distance (normalized units)', fontsize=12)
ax.set_title('Fréchet Distance Distribution per Model', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
