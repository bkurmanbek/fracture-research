"""
viz_04_all_metrics_boxplot.py
Side-by-side box plots for Hausdorff, Fréchet, Endpoint Error, Length Error,
and Path Similarity — one panel per metric, all 4 models.
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
METRICS = ['hausdorff', 'frechet', 'endpoint_error', 'length_error', 'path_similarity']
METRIC_LABELS = ['Hausdorff', 'Fréchet', 'Endpoint Error', 'Length Error', 'Path Similarity']
COLORS = ['#1a9641', '#a6d96a', '#fdae61', '#d7191c']
OUT = 'plots/04_all_metrics_boxplot.png'

fig, axes = plt.subplots(1, 5, figsize=(22, 7))

for ax, metric, label in zip(axes, METRICS, METRIC_LABELS):
    data, names = [], []
    for (model, path), color in zip(MODELS.items(), COLORS):
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if metric not in df.columns:
            continue
        vals = df[metric].dropna().values
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            data.append(vals)
            names.append(model)

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker='o', markersize=3, alpha=0.4))
    for patch, color in zip(bp['boxes'], COLORS[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

plt.suptitle('Path-Level Metric Distributions Across Models', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
