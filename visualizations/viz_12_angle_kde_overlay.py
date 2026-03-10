"""
viz_12_angle_kde_overlay.py
Wrapped KDE (circular density) of segment angles: True vs Generated.
One panel per model. Uses segment_angles.csv.
Shows directional bias: do generated paths follow the same orientation as true ones?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

MODELS = {
    'BiLSTM-Attn':  '../fracture_results/case1/segment_angles.csv',
    'Trans-GAT':    '../fracture_results/case2/segment_angles.csv',
    'LSTM-Stop':    '../fracture_results/case3/segment_angles.csv',
    'CNN-GRU-MDN':  '../fracture_results/case4/segment_angles.csv',
}
COLORS = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00']
OUT = 'plots/12_angle_kde_overlay.png'

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, ((model, path), model_color) in zip(axes, zip(MODELS.items(), COLORS)):
    if not os.path.exists(path):
        ax.set_title(f'{model}\n(no data)', fontsize=12)
        continue
    df = pd.read_csv(path)

    for typ, color, label in [('true', '#2c7bb6', 'True'), ('generated', model_color, 'Generated')]:
        vals = df[df['type'] == typ]['value'].dropna().values
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        # Circular KDE: mirror the data to handle wrap-around at ±π
        vals_aug = np.concatenate([vals - 2*np.pi, vals, vals + 2*np.pi])
        kde = gaussian_kde(vals_aug, bw_method=0.3)
        xs = np.linspace(-np.pi, np.pi, 500)
        density = kde(xs) * 3   # correct for triplication
        ax.fill_between(xs, density, alpha=0.20, color=color)
        ax.plot(xs, density, color=color, linewidth=2.5, label=label)

    ax.set_xlabel('Segment Angle (radians)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(model, fontsize=13, fontweight='bold')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Circular KDE of Segment Angles: True vs. Generated',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
