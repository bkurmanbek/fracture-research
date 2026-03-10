"""
viz_07_endpoint_error_distribution.py
Overlaid KDE curves of endpoint error for all 4 models on a single axes.
Shows how each model terminates paths relative to true endpoints.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

MODELS = {
    'BiLSTM-Attn':  '../fracture_results/case1/path_metrics.csv',
    'Trans-GAT':    '../fracture_results/case2/path_metrics.csv',
    'LSTM-Stop':    '../fracture_results/case3/path_metrics.csv',
    'CNN-GRU-MDN':  '../fracture_results/case4/path_metrics.csv',
}
COLORS = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00']
OUT = 'plots/07_endpoint_error_distribution.png'

fig, (ax_main, ax_log) = plt.subplots(1, 2, figsize=(15, 6))

upper_p = []
for (model, path), color in zip(MODELS.items(), COLORS):
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    if 'endpoint_error' not in df.columns:
        continue
    vals = df['endpoint_error'].dropna().values
    vals = vals[np.isfinite(vals) & (vals >= 0)]
    if len(vals) < 2:
        continue
    upper_p.append(np.percentile(vals, 95))

x_max = max(upper_p) * 1.1 if upper_p else 1.0

for (model, path), color in zip(MODELS.items(), COLORS):
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    if 'endpoint_error' not in df.columns:
        continue
    vals = df['endpoint_error'].dropna().values
    vals = vals[np.isfinite(vals) & (vals >= 0)]
    if len(vals) < 2:
        continue

    med = np.median(vals)
    kde = gaussian_kde(vals, bw_method='scott')

    # Linear scale
    xs = np.linspace(0, x_max, 400)
    ax_main.fill_between(xs, kde(xs), alpha=0.15, color=color)
    ax_main.plot(xs, kde(xs), color=color, linewidth=2.5,
                 label=f'{model} (med={med:.3f})')

    # Log scale — histogram
    ax_log.hist(vals, bins=40, density=True, alpha=0.4, color=color,
                label=f'{model}', histtype='stepfilled', edgecolor=color, linewidth=0.5)

ax_main.set_xlabel('Endpoint Error (normalized)', fontsize=12)
ax_main.set_ylabel('Density', fontsize=12)
ax_main.set_title('Endpoint Error KDE (Linear Scale)', fontsize=13, fontweight='bold')
ax_main.legend(fontsize=10)
ax_main.grid(True, alpha=0.3)
ax_main.set_xlim(0, x_max)

ax_log.set_xlabel('Endpoint Error (normalized)', fontsize=12)
ax_log.set_ylabel('Density (log)', fontsize=12)
ax_log.set_title('Endpoint Error Histogram (Log Scale)', fontsize=13, fontweight='bold')
ax_log.set_yscale('log')
ax_log.legend(fontsize=10)
ax_log.grid(True, alpha=0.3)

plt.suptitle('Endpoint Error Distribution: All Models', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
