"""
viz_15_segment_length_boxplot.py
Side-by-side box plots of segment lengths: True vs Generated for each model.
Four panels (one per model) — shows how well each model reproduces individual
step sizes of fracture segments.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

MODELS = {
    'BiLSTM-Attn':  '../fracture_results/case1/segment_lengths.csv',
    'Trans-GAT':    '../fracture_results/case2/segment_lengths.csv',
    'LSTM-Stop':    '../fracture_results/case3/segment_lengths.csv',
    'CNN-GRU-MDN':  '../fracture_results/case4/segment_lengths.csv',
}
COLORS = {'true': '#2c7bb6', 'generated': '#d7191c'}
OUT = 'plots/15_segment_length_boxplot.png'

fig, axes = plt.subplots(1, 4, figsize=(18, 7), sharey=False)

for ax, (model, path) in zip(axes, MODELS.items()):
    if not os.path.exists(path):
        ax.set_title(f'{model}\n(no data)', fontsize=10)
        continue
    df = pd.read_csv(path)

    groups, group_labels, box_colors = [], [], []
    for typ in ['true', 'generated']:
        vals = df[df['type'] == typ]['value'].dropna().values
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) > 0:
            groups.append(vals)
            group_labels.append(typ.capitalize())
            box_colors.append(COLORS[typ])

    if not groups:
        ax.set_title(f'{model}\n(no data)', fontsize=10)
        continue

    bp = ax.boxplot(groups, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markersize=3, alpha=0.3))
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(group_labels, fontsize=11)
    ax.set_ylabel('Segment Length (normalized)', fontsize=10)
    ax.set_title(model, fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    # Add mean ± std annotation
    for i, (vals, color) in enumerate(zip(groups, box_colors), start=1):
        ax.text(i, ax.get_ylim()[1] * 0.95,
                f'μ={np.mean(vals):.4f}\nσ={np.std(vals):.4f}',
                ha='center', va='top', fontsize=8, color=color)

plt.suptitle('Segment Length Distributions: True vs. Generated', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
