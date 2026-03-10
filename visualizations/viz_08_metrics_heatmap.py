"""
viz_08_metrics_heatmap.py
Heatmap of mean metric values: rows = models, columns = metrics.
Values normalized 0-1 per column (lower=better for errors, higher=better for similarity).
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
METRIC_LABELS = ['Hausdorff', 'Fréchet', 'Endpoint\nError', 'Length\nError', 'Path\nSimilarity']
# True = lower is better; False = higher is better
LOWER_BETTER = [True, True, True, True, False]
OUT = 'plots/08_metrics_heatmap.png'

rows, model_names = [], []
for model, path in MODELS.items():
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    row = []
    for metric in METRICS:
        if metric in df.columns:
            vals = df[metric].dropna().values
            vals = vals[np.isfinite(vals)]
            row.append(np.mean(vals) if len(vals) > 0 else np.nan)
        else:
            row.append(np.nan)
    rows.append(row)
    model_names.append(model)

matrix = np.array(rows, dtype=float)  # shape: (n_models, n_metrics)

# Normalize each column: 0=best, 1=worst (for colormap: dark=best)
norm_matrix = np.zeros_like(matrix)
for j in range(matrix.shape[1]):
    col = matrix[:, j]
    finite = col[np.isfinite(col)]
    if len(finite) == 0:
        continue
    cmin, cmax = finite.min(), finite.max()
    if cmax == cmin:
        norm_matrix[:, j] = 0.5
    else:
        normalized = (col - cmin) / (cmax - cmin)
        if not LOWER_BETTER[j]:
            normalized = 1 - normalized   # flip: high path_similarity → low heat
        norm_matrix[:, j] = normalized

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(norm_matrix, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')

# Annotate cells with raw values
for i in range(len(model_names)):
    for j in range(len(METRICS)):
        val = matrix[i, j]
        text = f'{val:.4f}' if np.isfinite(val) else 'N/A'
        ax.text(j, i, text, ha='center', va='center', fontsize=10,
                color='black', fontweight='bold')

ax.set_xticks(range(len(METRICS)))
ax.set_xticklabels(METRIC_LABELS, fontsize=11)
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels(model_names, fontsize=12, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label('Relative Performance\n(green=best, red=worst)', fontsize=10)

ax.set_title('Mean Metric Heatmap Across Models\n(raw values shown; color = normalized rank)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
