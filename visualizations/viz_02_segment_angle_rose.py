"""
viz_02_segment_angle_rose.py
Rose (polar histogram) of segment angles: True vs. Generated for all 4 models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS = {
    'BiLSTM-Attn':  '../fracture_results/case1/segment_angles.csv',
    'Trans-GAT':    '../fracture_results/case2/segment_angles.csv',
    'LSTM-Stop':    '../fracture_results/case3/segment_angles.csv',
    'CNN-GRU-MDN':  '../fracture_results/case4/segment_angles.csv',
}
OUT = 'plots/02_segment_angle_rose.png'
N_BINS = 36

fig, axes = plt.subplots(2, 4, figsize=(20, 9), subplot_kw=dict(projection='polar'))

for col, (model, path) in enumerate(RESULTS.items()):
    for row, (typ, color, label) in enumerate([('true', '#2c7bb6', 'True'), ('generated', '#d7191c', 'Generated')]):
        ax = axes[row, col]
        if not os.path.exists(path):
            ax.set_title(f'{model}\n{label}\n(no data)', fontsize=9)
            continue
        df = pd.read_csv(path)
        vals = df[df['type'] == typ]['value'].dropna().values
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        counts, bin_edges = np.histogram(vals, bins=N_BINS, range=(-np.pi, np.pi))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        width = 2 * np.pi / N_BINS
        bars = ax.bar(bin_centers, counts / counts.max(), width=width * 0.9,
                      color=color, alpha=0.75, edgecolor='white', linewidth=0.3)
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.set_title(f'{model}\n{label}', fontsize=9, fontweight='bold', pad=8)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)

plt.suptitle('Segment Angle Rose Diagrams: True vs. Generated', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
