"""
viz_14_radar_chart.py
Radar (spider) chart comparing all 4 models across 5 normalized metrics.
Each axis 0=worst, 1=best. Gives a single-glance comparison polygon per model.
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
METRICS = ['hausdorff', 'frechet', 'endpoint_error', 'length_error', 'path_similarity']
LABELS  = ['Hausdorff', 'Fréchet', 'Endpoint\nError', 'Length\nError', 'Path\nSimilarity']
# For error metrics: lower=better → invert; for similarity: higher=better → keep
INVERT  = [True, True, True, True, False]
OUT = 'plots/14_radar_chart.png'

# Gather mean values
rows, model_names = [], []
for model, path in MODELS.items():
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    row = []
    for metric in METRICS:
        if metric in df.columns:
            v = df[metric].dropna().values
            v = v[np.isfinite(v)]
            row.append(np.mean(v) if len(v) > 0 else np.nan)
        else:
            row.append(np.nan)
    rows.append(row)
    model_names.append(model)

matrix = np.array(rows, dtype=float)

# Normalize: 0=worst, 1=best per metric
norm = np.zeros_like(matrix)
for j in range(matrix.shape[1]):
    col = matrix[:, j]
    finite = col[np.isfinite(col)]
    if len(finite) == 0:
        continue
    cmin, cmax = finite.min(), finite.max()
    if cmax == cmin:
        norm[:, j] = 1.0
    else:
        norm[:, j] = (col - cmin) / (cmax - cmin)
        if INVERT[j]:
            norm[:, j] = 1 - norm[:, j]

N = len(METRICS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # close polygon

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

for i, (model, color) in enumerate(zip(model_names, COLORS)):
    values = norm[i].tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2.5, label=model, marker='o', markersize=6)
    ax.fill(angles, values, color=color, alpha=0.12)

# Axis labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(LABELS, fontsize=11)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='gray')
ax.set_ylim(0, 1)
ax.set_title('Normalized Performance Radar Chart\n(1 = best per metric)',
             fontsize=13, fontweight='bold', y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=11)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
