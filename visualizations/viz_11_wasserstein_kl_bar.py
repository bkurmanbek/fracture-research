"""
viz_11_wasserstein_kl_bar.py
Grouped bar chart of Wasserstein distances (length & angle) and KL divergence
for models that have distributional_metrics.csv (Cases 1 and 4).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

DIST_FILES = {
    'BiLSTM-Attn':  '../fracture_results/case1/distributional_metrics.csv',
    'CNN-GRU-MDN':  '../fracture_results/case4/distributional_metrics.csv',
}
# For models without a separate distributional file, derive from segment_lengths/angles
SEGMENT_FILES = {
    'Trans-GAT':  {
        'lengths': '../fracture_results/case2/segment_lengths.csv',
        'angles':  '../fracture_results/case2/segment_angles.csv',
    },
    'LSTM-Stop': {
        'lengths': '../fracture_results/case3/segment_lengths.csv',
        'angles':  '../fracture_results/case3/segment_angles.csv',
    },
}
OUT = 'plots/11_wasserstein_kl_bar.png'

from scipy.stats import wasserstein_distance

def compute_wasserstein(seg_path, key='length'):
    if not os.path.exists(seg_path):
        return np.nan
    df = pd.read_csv(seg_path)
    t = df[df['type'] == 'true']['value'].dropna().values
    g = df[df['type'] == 'generated']['value'].dropna().values
    t = t[np.isfinite(t)]
    g = g[np.isfinite(g)]
    if len(t) < 2 or len(g) < 2:
        return np.nan
    return float(wasserstein_distance(t, g))

results = {}

# Load from distributional_metrics.csv where available
for model, path in DIST_FILES.items():
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    row = {}
    for col in ['wasserstein_length', 'wasserstein_angle', 'kl_divergence']:
        if col in df.columns:
            row[col] = float(df[col].iloc[0]) if not df[col].isna().all() else np.nan
    results[model] = row

# Compute Wasserstein from segment files for other models
for model, files in SEGMENT_FILES.items():
    wl = compute_wasserstein(files['lengths'], 'length')
    wa = compute_wasserstein(files['angles'], 'angle')
    results[model] = {
        'wasserstein_length': wl,
        'wasserstein_angle':  wa,
        'kl_divergence':      np.nan,   # not computed for these cases
    }

# Build plot
model_names = list(results.keys())
metrics = ['wasserstein_length', 'wasserstein_angle', 'kl_divergence']
metric_labels = ['Wasserstein\n(Length)', 'Wasserstein\n(Angle)', 'KL Divergence']
colors = ['#4393c3', '#2166ac', '#92c5de']

x = np.arange(len(model_names))
width = 0.22
offsets = np.linspace(-(len(metrics)-1)/2 * width, (len(metrics)-1)/2 * width, len(metrics))

fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=False)

for ax_idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
    ax = axes[ax_idx]
    vals = [results.get(m, {}).get(metric, np.nan) for m in model_names]
    bar_colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00'][:len(model_names)]
    bars = ax.bar(x, vals, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_height() * 0.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha='right', fontsize=10)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

plt.suptitle('Distributional Metrics: Wasserstein Distances and KL Divergence',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
