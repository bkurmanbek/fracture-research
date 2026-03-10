"""
viz_09_length_stratified_hausdorff.py
Grouped bar chart: mean Hausdorff distance per fracture length category.
Categories: Short (3-5 pts), Medium (6-15), Long (16-30), Very Long (>30).
Uses length_stratified_metrics.csv from Case 4; path_metrics.csv for other cases
(bins computed on the fly from true_n_pts).
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
CATEGORIES = ['Short\n(3-5)', 'Medium\n(6-15)', 'Long\n(16-30)', 'Very Long\n(>30)']
OUT = 'plots/09_length_stratified_hausdorff.png'


def length_category(n):
    if n <= 5:
        return 'Short\n(3-5)'
    elif n <= 15:
        return 'Medium\n(6-15)'
    elif n <= 30:
        return 'Long\n(16-30)'
    else:
        return 'Very Long\n(>30)'


# Try loading from case4 stratified file first; otherwise compute from path_metrics
case4_strat = '../fracture_results/case4/length_stratified_metrics.csv'

fig, ax = plt.subplots(figsize=(13, 7))
n_models = len(MODELS)
n_cats   = len(CATEGORIES)
x = np.arange(n_cats)
width = 0.2
offsets = np.linspace(-(n_models-1)/2 * width, (n_models-1)/2 * width, n_models)

for idx, ((model, path), color) in enumerate(zip(MODELS.items(), COLORS)):
    means = []
    for cat_label in CATEGORIES:
        cat_short = cat_label.replace('\n', ' ')

        # For case4, try loading pre-computed stratified file
        if 'case4' in path and os.path.exists(case4_strat):
            strat_df = pd.read_csv(case4_strat)
            # Find matching row
            cat_col = 'length_category' if 'length_category' in strat_df.columns else 'category'
            match = strat_df[strat_df[cat_col].str.contains(
                cat_short.split('(')[0].strip(), case=False, na=False)]
            if not match.empty and 'mean_hausdorff' in strat_df.columns:
                means.append(float(match['mean_hausdorff'].iloc[0]))
                continue

        # Compute on the fly from path_metrics.csv
        if not os.path.exists(path):
            means.append(np.nan)
            continue
        df = pd.read_csv(path)
        if 'true_n_pts' not in df.columns or 'hausdorff' not in df.columns:
            means.append(np.nan)
            continue
        df['cat'] = df['true_n_pts'].apply(length_category)
        sub = df[df['cat'] == cat_label]['hausdorff'].dropna().values
        sub = sub[np.isfinite(sub)]
        means.append(np.mean(sub) if len(sub) > 0 else np.nan)

    bars = ax.bar(x + offsets[idx], means, width=width * 0.9,
                  color=color, alpha=0.8, label=model, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, means):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_xticks(x)
ax.set_xticklabels(CATEGORIES, fontsize=11)
ax.set_ylabel('Mean Hausdorff Distance', fontsize=12)
ax.set_title('Hausdorff Distance by Fracture Length Category', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
print(f'Saved: {OUT}')
plt.close()
