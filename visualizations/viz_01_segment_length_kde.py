"""
viz_01_segment_length_kde.py
KDE + histogram of segment lengths: True vs. Generated for all 4 models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

RESULTS = {
    'BiLSTM-Attn':  '../fracture_results/case1/segment_lengths.csv',
    'Trans-GAT':    '../fracture_results/case2/segment_lengths.csv',
    'LSTM-Stop':    '../fracture_results/case3/segment_lengths.csv',
    'CNN-GRU-MDN':  '../fracture_results/case4/segment_lengths.csv',
}
COLORS = {'true': '#2c7bb6', 'generated': '#d7191c'}
OUT = 'plots/01_segment_length_kde.png'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, (model, path) in zip(axes, RESULTS.items()):
    if not os.path.exists(path):
        ax.set_title(f'{model}\n(no data)', fontsize=12)
        continue
    df = pd.read_csv(path)
    for typ, color, label in [('true', COLORS['true'], 'True'), ('generated', COLORS['generated'], 'Generated')]:
        vals = df[df['type'] == typ]['value'].dropna().values
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) < 2:
            continue
        ax.hist(vals, bins=40, density=True, alpha=0.30, color=color)
        kde = gaussian_kde(vals, bw_method='scott')
        xs = np.linspace(0, np.percentile(vals, 99), 300)
        ax.plot(xs, kde(xs), color=color, linewidth=2.5, label=label)
    ax.set_title(model, fontsize=13, fontweight='bold')
    ax.set_xlabel('Segment Length (normalized)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Segment Length Distributions: True vs. Generated', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
