"""
viz_20_summary_panel.py
Combined summary figure for paper — 3×4 panel layout:
Row 1: Hausdorff violin (4 models) | Segment length KDE comparison (4 models)
Row 2: Path similarity KDE (4 models) | Endpoint error CDF (all models)
Row 3: Segment angle rose (1 model each — best 2 + true comparison)
Designed to fit comfortably in a single-column or two-column journal figure.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import os

MODELS = {
    'BiLSTM-Attn':  {
        'paths': '../fracture_results/case1/path_metrics.csv',
        'segl':  '../fracture_results/case1/segment_lengths.csv',
        'sega':  '../fracture_results/case1/segment_angles.csv',
    },
    'Trans-GAT':    {
        'paths': '../fracture_results/case2/path_metrics.csv',
        'segl':  '../fracture_results/case2/segment_lengths.csv',
        'sega':  '../fracture_results/case2/segment_angles.csv',
    },
    'LSTM-Stop':    {
        'paths': '../fracture_results/case3/path_metrics.csv',
        'segl':  '../fracture_results/case3/segment_lengths.csv',
        'sega':  '../fracture_results/case3/segment_angles.csv',
    },
    'CNN-GRU-MDN':  {
        'paths': '../fracture_results/case4/path_metrics.csv',
        'segl':  '../fracture_results/case4/segment_lengths.csv',
        'sega':  '../fracture_results/case4/segment_angles.csv',
    },
}
MODEL_COLORS = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00']
TRUE_COLOR = '#2c7bb6'
OUT = 'plots/20_summary_panel.png'

fig = plt.figure(figsize=(22, 16))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

model_names = list(MODELS.keys())

# ─── Row 0: Hausdorff violin per model ───────────────────────────────────────
hausdorff_data, hd_colors = [], []
for (model, files), color in zip(MODELS.items(), MODEL_COLORS):
    p = files['paths']
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    if 'hausdorff' not in df.columns:
        continue
    v = df['hausdorff'].dropna().values
    v = v[np.isfinite(v) & (v >= 0)]
    if len(v):
        hausdorff_data.append(v)
        hd_colors.append(color)

ax_viol = fig.add_subplot(gs[0, :2])
if hausdorff_data:
    vp = ax_viol.violinplot(hausdorff_data, positions=range(len(hausdorff_data)),
                            showmedians=True, showextrema=True)
    for body, color in zip(vp['bodies'], hd_colors):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    vp['cmedians'].set_color('black')
    vp['cmedians'].set_linewidth(2.5)
    for i, v in enumerate(hausdorff_data):
        ax_viol.scatter(i, np.mean(v), color='black', zorder=5, s=50, marker='D',
                        label='Mean' if i == 0 else '')
ax_viol.set_xticks(range(len(model_names[:len(hausdorff_data)])))
ax_viol.set_xticklabels(model_names[:len(hausdorff_data)], fontsize=10, fontweight='bold')
ax_viol.set_ylabel('Hausdorff Distance', fontsize=10)
ax_viol.set_title('(a) Hausdorff Distance per Model', fontsize=11, fontweight='bold')
ax_viol.grid(True, axis='y', alpha=0.3)
ax_viol.set_ylim(bottom=0)
ax_viol.legend(fontsize=9)

# ─── Row 0 col 2-3: Segment length KDE overlay ───────────────────────────────
ax_len = fig.add_subplot(gs[0, 2:])
for (model, files), color in zip(MODELS.items(), MODEL_COLORS):
    p = files['segl']
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    for typ, ls in [('true', '--'), ('generated', '-')]:
        vals = df[df['type'] == typ]['value'].dropna().values
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals, bw_method='scott')
        xs = np.linspace(0, np.percentile(vals, 98), 300)
        label = f'{model} ({typ})' if typ == 'generated' else None
        ax_len.plot(xs, kde(xs), color=color, linewidth=2 if typ == 'generated' else 1,
                    linestyle=ls, alpha=0.85, label=label)
ax_len.set_xlabel('Segment Length (normalized)', fontsize=10)
ax_len.set_ylabel('Density', fontsize=10)
ax_len.set_title('(b) Segment Length KDE\n(solid=generated, dashed=true)', fontsize=11, fontweight='bold')
ax_len.legend(fontsize=8, ncol=2)
ax_len.grid(True, alpha=0.3)

# ─── Row 1: Path similarity KDE ──────────────────────────────────────────────
ax_sim = fig.add_subplot(gs[1, :2])
for (model, files), color in zip(MODELS.items(), MODEL_COLORS):
    p = files['paths']
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    if 'path_similarity' not in df.columns:
        continue
    vals = df['path_similarity'].dropna().values
    vals = vals[np.isfinite(vals) & (vals >= 0) & (vals <= 1)]
    if len(vals) < 2:
        continue
    kde = gaussian_kde(vals, bw_method='scott')
    xs = np.linspace(0, 1, 300)
    med = np.median(vals)
    ax_sim.fill_between(xs, kde(xs), alpha=0.15, color=color)
    ax_sim.plot(xs, kde(xs), color=color, linewidth=2.5, label=f'{model} (med={med:.2f})')
ax_sim.set_xlabel('Path Similarity', fontsize=10)
ax_sim.set_ylabel('Density', fontsize=10)
ax_sim.set_title('(c) Path Similarity Distribution', fontsize=11, fontweight='bold')
ax_sim.legend(fontsize=9)
ax_sim.grid(True, alpha=0.3)
ax_sim.set_xlim(0, 1)

# ─── Row 1 col 2-3: ECDF of Hausdorff ────────────────────────────────────────
ax_cdf = fig.add_subplot(gs[1, 2:])
for (model, files), color in zip(MODELS.items(), MODEL_COLORS):
    p = files['paths']
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p)
    if 'hausdorff' not in df.columns:
        continue
    vals = df['hausdorff'].dropna().values
    vals = vals[np.isfinite(vals) & (vals >= 0)]
    if len(vals) == 0:
        continue
    sv = np.sort(vals)
    ecdf = np.arange(1, len(sv)+1) / len(sv)
    sv = np.concatenate([[0], sv])
    ecdf = np.concatenate([[0], ecdf])
    med = np.median(vals)
    ax_cdf.step(sv, ecdf, color=color, linewidth=2.5, label=f'{model} (med={med:.3f})')
ax_cdf.set_xlabel('Hausdorff Distance', fontsize=10)
ax_cdf.set_ylabel('Cumulative Proportion', fontsize=10)
ax_cdf.set_title('(d) ECDF of Hausdorff Distance', fontsize=11, fontweight='bold')
ax_cdf.legend(fontsize=9)
ax_cdf.grid(True, alpha=0.3)
ax_cdf.set_ylim(0, 1.02)
ax_cdf.set_xlim(left=0)

# ─── Row 2: Segment angle rose (polar) — one per model ───────────────────────
N_BINS = 36
for col_idx, ((model, files), color) in enumerate(zip(MODELS.items(), MODEL_COLORS)):
    ax_rose = fig.add_subplot(gs[2, col_idx], projection='polar')
    p = files['sega']
    if not os.path.exists(p):
        ax_rose.set_title(f'{model}\n(no data)', fontsize=9)
        continue
    df = pd.read_csv(p)
    for typ, clr, ls in [('true', TRUE_COLOR, 0.4), ('generated', color, 0.75)]:
        vals = df[df['type'] == typ]['value'].dropna().values
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        counts, bin_edges = np.histogram(vals, bins=N_BINS, range=(-np.pi, np.pi))
        bc = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        w = 2 * np.pi / N_BINS
        lbl = 'True' if typ == 'true' else 'Generated'
        ax_rose.bar(bc, counts / counts.max(), width=w*0.85,
                    color=clr, alpha=ls, label=lbl, edgecolor='white', linewidth=0.2)
    ax_rose.set_theta_zero_location('E')
    ax_rose.set_theta_direction(1)
    ax_rose.set_yticks([])
    ax_rose.set_title(f'({"efgh"[col_idx]}) {model}', fontsize=10, fontweight='bold', pad=10)
    ax_rose.tick_params(labelsize=7)
    if col_idx == 0:
        ax_rose.legend(fontsize=8, loc='lower left', bbox_to_anchor=(-0.15, -0.15))

plt.suptitle('Fracture Path Generation Quality: Distribution Summary',
             fontsize=16, fontweight='bold', y=1.01)
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
