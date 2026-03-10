# Deep Learning Approaches for Autoregressive Fracture Network Path Prediction

Reproducibility repository for the paper:
> *Deep Learning Approaches for Autoregressive Fracture Network Path Prediction: A Comprehensive Study on the Teapot Dome Dataset*
> Bakytzhank Kurmanbek, Nazarbayev University

---

## Repository Structure

```
fracture-network-prediction/
├── README.md                        ← this file
├── requirements.txt                 ← Python dependencies
│
├── data/
│   ├── Teapod_fractures_XY_coordinates.csv   ← raw fracture coordinate data
│   ├── train_fractures_processed.csv          ← preprocessed training set (1,197 fractures, 5,541 pts)
│   └── test_fractures_processed.csv           ← preprocessed test set (139 fractures, 625 pts)
│
├── models/
│   ├── case1_bilstm_multihead_attention.py    ← BiLSTM + Multi-head Attention
│   ├── case2_transformer_gat_hybrid.py        ← Transformer-GAT Hybrid
│   ├── case3_advanced_lstm_stopping.py        ← LSTM with Stopping Prediction
│   └── case4_cnn_gru_mdn.py                  ← CNN-GRU with Mixture Density Network
│
├── visualizations/
│   ├── run_all_viz.sh                         ← runs all 20 plots in sequence
│   ├── viz_01_segment_length_kde.py           ← KDE of segment lengths
│   ├── viz_02_segment_angle_rose.py           ← rose diagram of angles
│   ├── ... (viz_03 through viz_20)
│   └── plots/                                 ← generated PNG outputs
│
├── results/
│   ├── case1/   ← evaluation_metrics.csv, path_metrics.csv, distributional_metrics.csv, plots/
│   ├── case2/   ← evaluation_metrics.csv, path_metrics.csv, plots/
│   ├── case3/   ← evaluation_metrics.csv, path_metrics.csv, stopping_metrics.csv, plots/
│   └── case4/   ← evaluation_metrics.csv, path_metrics.csv, distributional_metrics.csv,
│                   length_stratified_metrics.csv, curvature_stratified_metrics.csv, plots/
│
├── factcheck/
│   ├── FULL_FACTCHECK.md                      ← complete fact-check of all paper claims
│   ├── metrics_factcheck.md                   ← single-step metrics detail
│   ├── path_metrics_factcheck.md              ← path-level metrics detail
│   └── distributional_stopping_factcheck.md   ← distributional + stopping detail
│
└── paper/
    └── fracture_network_prediction.tex        ← LaTeX source
```

---

## Reproducing Results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the four models

Run from the `models/` directory. Results are saved to `results/case{1-4}/`.

```bash
cd models/

# Case 1 — BiLSTM + Multi-head Attention (TensorFlow/Keras)
python case1_bilstm_multihead_attention.py

# Case 2 — Transformer-GAT Hybrid (PyTorch)
python case2_transformer_gat_hybrid.py

# Case 3 — LSTM with Stopping Prediction (TensorFlow/Keras)
python case3_advanced_lstm_stopping.py

# Case 4 — CNN-GRU-MDN (PyTorch)
python case4_cnn_gru_mdn.py
```

> **Note:** Each script expects `train_fractures_processed.csv` and `test_fractures_processed.csv`
> in the working directory (or set the path in the CONFIG dict at the top of each file).
> Copy the data files from `data/` to `models/` before running, or adjust the paths.

### 3. Generate visualizations

```bash
cd visualizations/
bash run_all_viz.sh
# Plots saved to visualizations/plots/
```

---

## Data Description

| File | Description |
|------|-------------|
| `Teapod_fractures_XY_coordinates.csv` | Raw XY coordinates of mapped fracture traces, Teapot Dome field (Naval Petroleum Reserve No. 3, Wyoming) |
| `train_fractures_processed.csv` | 1,197 fractures / 5,541 points. Geographic split: excludes southeastern region. Contains 25 columns including coordinates, geometric features, and neighborhood features. |
| `test_fractures_processed.csv` | 139 fractures / 625 points. Test region: x ∈ [795000–805000] ft, y ∈ [950000–970000] ft (~18.6 km²). |

**Key columns:** `fracture_id`, `point_idx`, `coord_x`, `coord_y`, `prev_angle`, `next_angle`, `prev_length`, `next_length`, `curvature`, `closest_seg_{0-3}_p{1,2}_{x,y}`

---

## Results Summary (Reproduced — Full Test Set)

### Single-Step Prediction (normalized coordinate units)

| Model | MSE | RMSE | MAE |
|-------|-----|------|-----|
| BiLSTM-Attn | 0.1196 | 0.3458 | 0.2816 |
| Trans-GAT | 0.2771 | 0.5264 | 0.2789 |
| LSTM-Stop | 2.9854 | 1.7278 | 1.3383 |
| CNN-GRU-MDN | 1.1935 | 1.0925 | 0.8677 |

### Path-Level Metrics (normalized units, full test set)

| Model | N | Hausdorff | Fréchet | Path Sim |
|-------|---|-----------|---------|----------|
| BiLSTM-Attn | 139 | 9421.88 ± 2530 | 9773.70 ± 2512 | 0.726 ± 0.276 |
| Trans-GAT | 111 | 0.87 ± 1.07 | 2.36 ± 1.49 | 0.996 ± 0.029 |
| LSTM-Stop | 111 | 6.89 ± 2.86 | 7.40 ± 2.71 | 0.865 ± 0.120 |
| CNN-GRU-MDN | 111 | 3.17 ± 1.50 | 3.52 ± 1.57 | 0.999 ± 0.010 |

> **Note on BiLSTM-Attn:** Case 1 generates paths in raw UTM coordinates (metres), not in
> per-fracture normalized units as used by Cases 2–4. This explains the large Hausdorff values.

> **Note on N=111 vs N=139:** Cases 2–4 exclude 28 fractures that have only 2 points,
> as their feature preprocessors require ≥ 3 points to compute segment angles and curvatures.

---

## Known Discrepancies Between Paper and Code

See `factcheck/FULL_FACTCHECK.md` for the complete audit. Key issues:

- **Max epochs:** Paper states 100 for all models; code trains for **50 epochs**
- **BiLSTM attention heads:** Paper says 4; code uses **8**
- **CNN-GRU-MDN K (MDN mixtures):** Paper says 3; code uses **5**
- **CNN-GRU-MDN GRU hidden dim:** Paper says 128; code uses **256**
- **Stopping label definition:** Paper says "penultimate point"; code labels **last point**
- **Dilation rates (Case 4):** Paper says (1,2,4); code uses **(1,2,4,8)**
- **All numeric results:** No paper table values are reproduced exactly; normalization schemes differ between models and from what the paper implies
- **Baselines (Linear Extrapolation, Random Walk):** Not implemented in this repository

---

## Environment

Tested on:
- Python 3.11
- TensorFlow 2.21 (Cases 1, 3)
- PyTorch 2.4.1 + CUDA 12.4 (Cases 2, 4)
- GPU: NVIDIA GeForce RTX 4090 (optional; all models run on CPU too)

---

## Citation

```bibtex
@article{kurmanbek2025fracture,
  title={Deep Learning Approaches for Autoregressive Fracture Network Path Prediction},
  author={Kurmanbek, Bakytzhank},
  journal={...},
  year={2025}
}
```
