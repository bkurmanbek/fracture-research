"""
viz_17_true_data_properties.py
Distributions of ground-truth fracture properties from the test set CSV.
Useful as a background characterization figure for the paper.
Shows: fracture length (#pts), segment lengths, segment angles, curvature.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os

TEST_CSV = '../test_fractures_processed.csv'
OUT = 'plots/17_true_data_properties.png'

if not os.path.exists(TEST_CSV):
    print(f'ERROR: {TEST_CSV} not found. Adjust path if needed.')
    exit(1)

df = pd.read_csv(TEST_CSV)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution of fracture lengths (# points per fracture)
ax = axes[0, 0]
if 'fracture_id' in df.columns:
    lengths = df.groupby('fracture_id').size().values
else:
    lengths = np.array([])

if len(lengths) > 0:
    bins = np.arange(lengths.min(), lengths.max() + 2) - 0.5
    ax.hist(lengths, bins=bins, color='#4393c3', edgecolor='white', linewidth=0.5, density=False)
    ax.axvline(np.median(lengths), color='red', linestyle='--', linewidth=2,
               label=f'Median={np.median(lengths):.0f}')
    ax.set_xlabel('Number of Points per Fracture', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Fracture Length Distribution (Test Set)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

# 2. Segment lengths (step sizes)
ax = axes[0, 1]
coord_cols = [c for c in df.columns if c.lower() in ('x', 'y', 'x_norm', 'y_norm')]
x_col = next((c for c in coord_cols if 'x' in c.lower()), None)
y_col = next((c for c in coord_cols if 'y' in c.lower()), None)

seg_lengths = []
if x_col and y_col and 'fracture_id' in df.columns:
    for fid, grp in df.groupby('fracture_id'):
        xs = grp[x_col].values
        ys = grp[y_col].values
        if len(xs) >= 2:
            diffs = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
            seg_lengths.extend(diffs.tolist())

if len(seg_lengths) > 1:
    sl = np.array(seg_lengths)
    sl = sl[sl > 0]
    ax.hist(sl, bins=50, density=True, color='#4393c3', alpha=0.4, edgecolor='white')
    kde = gaussian_kde(sl, bw_method='scott')
    xs_kde = np.linspace(0, np.percentile(sl, 99), 300)
    ax.plot(xs_kde, kde(xs_kde), color='#2166ac', linewidth=2.5)
    ax.set_xlabel('Segment Length (normalized)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('True Segment Length Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

# 3. Segment angles
ax = axes[1, 0]
seg_angles = []
if x_col and y_col and 'fracture_id' in df.columns:
    for fid, grp in df.groupby('fracture_id'):
        xs = grp[x_col].values
        ys = grp[y_col].values
        if len(xs) >= 2:
            angs = np.arctan2(np.diff(ys), np.diff(xs))
            seg_angles.extend(angs.tolist())

if len(seg_angles) > 0:
    sa = np.array(seg_angles)
    sa = sa[np.isfinite(sa)]
    N_BINS = 36
    counts, bin_edges = np.histogram(sa, bins=N_BINS, range=(-np.pi, np.pi))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.bar(bin_centers, counts, width=2*np.pi/N_BINS*0.9,
           color='#4393c3', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Segment Angle (radians)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('True Segment Angle Distribution', fontsize=12, fontweight='bold')
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.grid(True, alpha=0.3)

# 4. Mean curvature per fracture
ax = axes[1, 1]
curv_col = next((c for c in df.columns if 'curvature' in c.lower()), None)
if curv_col and 'fracture_id' in df.columns:
    mean_curv = df.groupby('fracture_id')[curv_col].mean().values
    mean_curv = mean_curv[np.isfinite(mean_curv)]
    if len(mean_curv) > 1:
        ax.hist(mean_curv, bins=30, color='#4393c3', edgecolor='white',
                linewidth=0.5, density=True, alpha=0.7)
        kde = gaussian_kde(mean_curv, bw_method='scott')
        xs_kde = np.linspace(0, mean_curv.max(), 300)
        ax.plot(xs_kde, kde(xs_kde), color='#2166ac', linewidth=2.5)
        ax.set_xlabel('Mean Curvature per Fracture', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('True Mean Curvature Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No curvature column found', ha='center', va='center',
            fontsize=12, transform=ax.transAxes)
    ax.set_title('Curvature', fontsize=12, fontweight='bold')

plt.suptitle('Ground-Truth Fracture Property Distributions (Test Set)',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
