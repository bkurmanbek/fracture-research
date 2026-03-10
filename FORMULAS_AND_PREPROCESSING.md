# Fracture Network Prediction — Formulas, Preprocessing & Technical Reference

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Preprocessing Steps](#2-preprocessing-steps)
   - 2.1 [Case 1 – BiLSTM + Multi-Head Attention](#21-case-1--bilstm--multi-head-attention)
   - 2.2 [Cases 2 & 4 – Transformer-GAT and CNN-GRU-MDN](#22-cases-2--4--transformer-gat-and-cnn-gru-mdn)
   - 2.3 [Case 3 – LSTM with Stopping Prediction](#23-case-3--lstm-with-stopping-prediction)
3. [Derived Features & Geometric Calculations](#3-derived-features--geometric-calculations)
4. [Loss Functions](#4-loss-functions)
5. [Evaluation Metrics](#5-evaluation-metrics)
   - 5.1 [Point-Level Metrics](#51-point-level-metrics)
   - 5.2 [Path-Level Metrics](#52-path-level-metrics)
   - 5.3 [Distributional Metrics](#53-distributional-metrics)
   - 5.4 [Stopping Prediction Metrics](#54-stopping-prediction-metrics)
6. [Compliance & Quality Scores](#6-compliance--quality-scores)
7. [Model Architecture Parameters](#7-model-architecture-parameters)
8. [Key Dataset Statistics](#8-key-dataset-statistics)
9. [Known Paper vs. Code Discrepancies](#9-known-paper-vs-code-discrepancies)

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Source | Teapot Dome field fracture data (UTM coordinates, feet) |
| Raw fractures | 1,453 |
| Raw points | 6,377 |
| Training fractures | 1,197 |
| Training points | 5,541 |
| Test fractures | 139 (111 with ≥ 3 pts usable for Cases 2/4) |
| Test points | 625 |
| Test spatial region | x ∈ [795,009, 804,963] ft, y ∈ [950,035, 969,986] ft (~18.6 km²) |
| Fracture length range | 30 ft – 13,073 ft |
| Median fracture length | 550 ft |
| Median point density | 4 pts/fracture |
| Median segment spacing | 182 ft |

**Input columns per point**: `fracture_id`, `point_idx`, `coord_x`, `coord_y`, `prev_angle`, `next_angle`, `prev_length`, `next_length`, `curvature`

**Test fractures by length category**:

| Category | Points | Count | % |
|---|---|---|---|
| Short | 3–5 | 73 | 66% |
| Medium | 6–15 | 37 | 33% |
| Long | 16–30 | 1 | 1% |
| Very Long | > 30 | 0 | 0% |

**Test fractures by curvature category**:

| Category | κ range | Count | % |
|---|---|---|---|
| Low | κ < 0.01 | 1 | 1% |
| Medium | 0.01 ≤ κ < 0.05 | 64 | 58% |
| High | κ ≥ 0.05 | 46 | 41% |

---

## 2. Preprocessing Steps

### 2.1 Case 1 – BiLSTM + Multi-Head Attention

**Step 1 — Raw feature extraction**

From the processed CSV, each point contributes a 5-dimensional feature vector:

```
f = [prev_angle, next_angle, prev_length, next_length, curvature]
```

and a 2-dimensional coordinate vector:

```
c = [coord_x, coord_y]
```

**Step 2 — StandardScaler normalization**

Fit on training set only; applied to both training and test sets.

For coordinates (fit on all training coordinate pairs flattened):
```
X_coords_reshaped = X_coords.reshape(-1, 2)
coord_scaler = StandardScaler().fit(X_coords_reshaped)
X_coords_norm = coord_scaler.transform(X_coords_reshaped).reshape(X_coords.shape)
y_norm = coord_scaler.transform(y)                      # targets use same scaler
```

For features (fit on all training feature vectors flattened):
```
X_features_reshaped = X_features.reshape(-1, 5)
feature_scaler = StandardScaler().fit(X_features_reshaped)
X_features_norm = feature_scaler.transform(X_features_reshaped).reshape(X_features.shape)
```

StandardScaler formula for each dimension d:

$$z_d = \frac{x_d - \mu_d}{\sigma_d}$$

where μ_d and σ_d are the mean and standard deviation computed on the training split for that dimension.

**Step 3 — Concatenated input tensor**

```
X = concat([X_coords_norm, X_features_norm], axis=-1)
# Shape: (N_sequences, sequence_length=10, 7)
```

**Step 4 — Sliding-window sequence creation**

For a fracture with n points (requires n ≥ 11):
- Window size: `k = 10`
- Produces `n − k` training examples
- Input window i: `X[i] = points[i : i+k]`
- Target: `y[i] = points[i+k]` (next coordinate in normalized space)

---

### 2.2 Cases 2 & 4 – Transformer-GAT and CNN-GRU-MDN

These models use a **per-fracture centroid-and-scale normalization** followed by derived feature computation.

**Step 1 — Per-fracture centroid normalization**

For each fracture independently:
```
cx = mean(x_coords)           # centroid x
cy = mean(y_coords)           # centroid y
x_norm = x - cx
y_norm = y - cy
```

**Step 2 — Per-fracture scale normalization**

Compute the median segment length in the centroid-normalized space:

$$\ell_i = \sqrt{(\Delta x_i)^2 + (\Delta y_i)^2}, \quad i = 1, \ldots, n-1$$

$$\text{scale} = \text{median}(\ell_1, \ldots, \ell_{n-1})$$

Apply:
```
x_norm /= scale       (if scale > 0, else leave as-is)
y_norm /= scale
```

The `(centroid_x, centroid_y, scale)` values are stored for inverse-transformation.

**Step 3 — Derived features per segment**

After normalization, for each segment i from point i to point i+1:

| Feature | Formula | Notes |
|---|---|---|
| `delta_r` | $\sqrt{(\Delta x_i)^2 + (\Delta y_i)^2}$ | Segment Euclidean length |
| `delta_theta` | $\arctan2(\Delta y_i,\ \Delta x_i)$ | Segment angle ∈ (−π, π] |
| `log_delta_r` | $\ln(\delta r_i + \varepsilon)$ | Log-length; ε = 1e-8 |
| `sin_theta` | $\sin(\delta\theta_i)$ | Trigonometric projection |
| `cos_theta` | $\cos(\delta\theta_i)$ | Trigonometric projection |
| `delta_angle` | $\delta\theta_i - \delta\theta_{i-1}$ (wrapped to (−π, π]) | Turning angle |
| `curvature_trajectory` | $\Delta\text{angle}_i - \Delta\text{angle}_{i-1}$ | 2nd derivative of angle |

Whole-fracture scalar features (broadcast to every sequence step):

| Feature | Formula |
|---|---|
| `mean_curvature` | $\overline{\vert\Delta\text{angle}\vert}$ across the fracture |
| `length_variance` | $\text{Var}(\ell_1, \ldots, \ell_{n-1})$ |
| `tortuosity` | $\frac{\sum_i \ell_i}{\lVert p_\text{end} - p_\text{start} \rVert}$ |

Angle-difference wrapping:

$$\text{wrap}(\phi) = \phi - 2\pi \cdot \text{round}\!\left(\frac{\phi}{2\pi}\right), \quad \phi \in (-\pi, \pi]$$

**Step 4 — Global feature standardization**

Computed on the training set:
```
log_delta_r_mean, log_delta_r_std  = mean/std over all training log_delta_r values
delta_theta_mean, delta_theta_std  = mean/std over all training delta_theta values
```

Applied to both splits:

$$\hat{f} = \frac{f - \mu_\text{train}}{\sigma_\text{train} + \varepsilon}$$

**Step 5 — Exclusion criteria**

Fractures with fewer than 3 points are excluded (cannot compute angles or curvatures).
→ 28 test fractures excluded, leaving **111 usable test fractures** for Cases 2 and 4.

**Step 6 — Sliding-window sequence creation** (same as Case 1)

Window size `k = 10`; fractures with n < k+1 = 11 points produce no training examples.

---

### 2.3 Case 3 – LSTM with Stopping Prediction

Preprocessing is identical to Case 1 (StandardScaler on coords and features), but the target is **dual**:

```
y_coord = point[i+k]          # coordinate target (same as Case 1)
y_stop  = 1 if (i+k) == (n-1) else 0   # stopping target
```

`y_stop = 1` marks the **last point** of a fracture; all earlier points have `y_stop = 0`.

> Note: The paper describes labeling the penultimate point, but the code labels the final point.

---

## 3. Derived Features & Geometric Calculations

### Curvature at an interior point

For point i (with i ∈ [1, n−2]):

$$\mathbf{v}_1 = p_i - p_{i-1}, \quad \mathbf{v}_2 = p_{i+1} - p_i$$

$$\kappa_i = \arccos\!\left(\text{clip}\!\left(\frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\lVert\mathbf{v}_1\rVert\,\lVert\mathbf{v}_2\rVert},\ -1,\ 1\right)\right)$$

Returns turning angle in radians, ∈ [0, π].

### Tortuosity

$$\tau = \frac{\sum_{i=1}^{n-1} \lVert p_i - p_{i-1} \rVert}{\lVert p_{n-1} - p_0 \rVert}$$

τ = 1 for a perfectly straight fracture; τ > 1 for curved fractures.

### Segment angle (direction)

$$\theta_i = \arctan2(y_{i+1} - y_i,\ x_{i+1} - x_i), \quad \theta_i \in (-\pi, \pi]$$

### Log segment length

$$\ell_i^{\log} = \ln(\ell_i + \varepsilon), \quad \varepsilon = 10^{-8}$$

Used to reduce skewness in the segment-length distribution.

---

## 4. Loss Functions

### Case 1 — BiLSTM + Multi-Head Attention

**Mean Squared Error (coordinate regression)**:

$$\mathcal{L}_\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \lVert \hat{y}_i - y_i \rVert^2 = \frac{1}{N} \sum_{i=1}^{N} \left[(\hat{x}_i - x_i)^2 + (\hat{y}_i - y_i)^2\right]$$

- Optimizer: **AdamW** — lr = 1e-3, weight decay = 0.01
- Learning rate schedule: ReduceLROnPlateau — factor = 0.5, patience = 5
- Early stopping: patience = 15

---

### Case 2 — Transformer-GAT Hybrid

Same MSE loss as Case 1.

$$\mathcal{L} = \mathcal{L}_\text{MSE}$$

- Optimizer: **AdamW** — lr = 3e-4, weight decay = 0.01
- Early stopping: patience = 20

---

### Case 3 — LSTM with Stopping Prediction

**Combined loss (coordinate + stopping)**:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{coord} + \lambda_\text{stop}\,\mathcal{L}_\text{stop}$$

**Coordinate loss** (MSE):

$$\mathcal{L}_\text{coord} = \frac{1}{N} \sum_{i=1}^{N} \lVert \hat{y}_i^\text{coord} - y_i^\text{coord} \rVert^2$$

**Stopping loss** (Binary Cross-Entropy):

$$\mathcal{L}_\text{stop} = -\frac{1}{N} \sum_{i=1}^{N} \left[s_i \log \hat{s}_i + (1 - s_i) \log(1 - \hat{s}_i)\right]$$

where $s_i \in \{0, 1\}$ is the ground-truth stopping label and $\hat{s}_i \in (0, 1)$ is the predicted stopping probability.

**Hyperparameters**:
- $\lambda_\text{stop} = 1.0$
- Optimizer: **Adam** — lr = 1e-3
- Early stopping: patience = 20

---

### Case 4 — CNN-GRU with Mixture Density Network

**MDN negative log-likelihood loss**:

The model outputs a K-component Gaussian mixture over polar coordinates $(\Delta r, \Delta\theta)$:
- $\pi_k$ — mixing weights (softmax, $\sum_k \pi_k = 1$)
- $\mu_r^{(k)},\ \sigma_r^{(k)}$ — Gaussian parameters for step length
- $\mu_\theta^{(k)},\ \sigma_\theta^{(k)}$ — Gaussian parameters for step angle

Gaussian probability density for component k:

$$\mathcal{N}(\Delta r \mid \mu_r^{(k)}, \sigma_r^{(k)}) = \frac{1}{\sigma_r^{(k)}\sqrt{2\pi}}\,\exp\!\left(-\frac{(\Delta r - \mu_r^{(k)})^2}{2\,(\sigma_r^{(k)})^2}\right)$$

$$\mathcal{N}(\Delta\theta \mid \mu_\theta^{(k)}, \sigma_\theta^{(k)}) = \frac{1}{\sigma_\theta^{(k)}\sqrt{2\pi}}\,\exp\!\left(-\frac{(\Delta\theta - \mu_\theta^{(k)})^2}{2\,(\sigma_\theta^{(k)})^2}\right)$$

Joint probability under independence assumption:

$$p_k(\Delta r, \Delta\theta) = \mathcal{N}(\Delta r \mid \mu_r^{(k)}, \sigma_r^{(k)}) \cdot \mathcal{N}(\Delta\theta \mid \mu_\theta^{(k)}, \sigma_\theta^{(k)})$$

Mixture probability:

$$p(\Delta r, \Delta\theta) = \sum_{k=1}^{K} \pi_k\, p_k(\Delta r, \Delta\theta)$$

MDN loss (negative log-likelihood, averaged over batch):

$$\mathcal{L}_\text{MDN} = -\frac{1}{N} \sum_{i=1}^{N} \log\!\left(\sum_{k=1}^{K} \pi_k^{(i)}\, p_k(\Delta r_i, \Delta\theta_i)\right)$$

**Combined loss** (MDN + stopping):

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{MDN} + \lambda_\text{stop}\,\mathcal{L}_\text{BCE-stop}$$

**Hyperparameters**:
- K = 5 mixture components
- $\lambda_\text{stop} = 1.0$ (code value; paper claims 0.1)
- Optimizer: **Adam** — lr = 1e-3
- Early stopping: patience = 15

---

## 5. Evaluation Metrics

### 5.1 Point-Level Metrics

**Mean Squared Error**:

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \lVert \hat{y}_i - y_i \rVert^2$$

**Root Mean Squared Error**:

$$\text{RMSE} = \sqrt{\text{MSE}}$$

**Mean Absolute Error**:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \lVert \hat{y}_i - y_i \rVert$$

**Coordinate-wise decomposition**:

$$\text{MSE}_x = \frac{1}{N}\sum(\hat{x}_i - x_i)^2, \quad \text{MSE}_y = \frac{1}{N}\sum(\hat{y}_i - y_i)^2$$

$$\text{MAE}_x = \frac{1}{N}\sum|\hat{x}_i - x_i|, \quad \text{MAE}_y = \frac{1}{N}\sum|\hat{y}_i - y_i|$$

All point-level metrics are computed in **normalized space** (post-StandardScaler or post-centroid-scale normalization depending on the model).

---

### 5.2 Path-Level Metrics

These are computed on **multi-step generated paths** vs. **ground-truth paths** (both in original coordinate space after inverse-transforming predictions).

**Hausdorff Distance**:

$$H(A, B) = \max\!\left(\sup_{a \in A} \inf_{b \in B} d(a, b),\;\sup_{b \in B} \inf_{a \in A} d(a, b)\right)$$

Implemented via `scipy.spatial.distance.directed_hausdorff`:

$$H(A,B) = \max\!\left(h(A \to B),\; h(B \to A)\right)$$

**Discrete Fréchet Distance** (dynamic programming):

Initialize:
$$\text{ca}[0, 0] = \lVert P_0 - Q_0 \rVert$$

$$\text{ca}[i, 0] = \max\!\left(\text{ca}[i-1, 0],\; \lVert P_i - Q_0 \rVert\right)$$

$$\text{ca}[0, j] = \max\!\left(\text{ca}[0, j-1],\; \lVert P_0 - Q_j \rVert\right)$$

Recurrence:
$$\text{ca}[i, j] = \max\!\left(\min\!\left(\text{ca}[i-1,j],\; \text{ca}[i-1,j-1],\; \text{ca}[i,j-1]\right),\; \lVert P_i - Q_j \rVert\right)$$

Result: $d_F(P, Q) = \text{ca}[n-1, m-1]$

**Path Similarity (direction cosine similarity)**:

Direction vectors:
$$\mathbf{d}_i^{(1)} = p_{i+1}^{(1)} - p_i^{(1)}, \quad \mathbf{d}_i^{(2)} = p_{i+1}^{(2)} - p_i^{(2)}$$

Normalized:
$$\hat{\mathbf{d}}_i^{(1)} = \frac{\mathbf{d}_i^{(1)}}{\lVert \mathbf{d}_i^{(1)} \rVert + \varepsilon}, \quad \hat{\mathbf{d}}_i^{(2)} = \frac{\mathbf{d}_i^{(2)}}{\lVert \mathbf{d}_i^{(2)} \rVert + \varepsilon}$$

Cosine similarity mapped to [0, 1]:
$$\text{PathSim} = \frac{1}{L} \sum_{i=1}^{L} \frac{\hat{\mathbf{d}}_i^{(1)} \cdot \hat{\mathbf{d}}_i^{(2)} + 1}{2}, \quad L = \min(n-1, m-1)$$

Range: 0 (opposite directions) → 1 (perfectly aligned).

**Endpoint Error**:

$$E_\text{end} = \lVert \hat{p}_{-1} - p_{-1} \rVert_2$$

Euclidean distance between the final predicted point and the final true point.

**Relative Length Error**:

$$E_\text{len} = \frac{\left|\hat{L}_\text{path} - L_\text{path}\right|}{L_\text{path} + \varepsilon}$$

where:
$$L_\text{path} = \sum_{i=1}^{n-1} \lVert p_{i+1} - p_i \rVert_2, \quad \varepsilon = 10^{-6}$$

---

### 5.3 Distributional Metrics

**1D Wasserstein Distance (Earth Mover's Distance)**:

$$W_1(P, Q) = \inf_{\gamma \in \Gamma(P,Q)} \mathbb{E}_{(x,y)\sim\gamma}\left[|x - y|\right]$$

Computed via `scipy.stats.wasserstein_distance` separately for:
- Segment lengths: $W_1(\{\ell_i^\text{true}\}, \{\ell_i^\text{gen}\})$
- Segment angles: $W_1(\{\theta_i^\text{true}\}, \{\theta_i^\text{gen}\})$

**KL Divergence** (histogram approximation with 50 bins):

$$D_\text{KL}(P \| Q) = \sum_{b} P(b)\,\log\frac{P(b)}{Q(b)}$$

Smoothed to avoid log(0):
$$P(b) = \frac{h_P(b) + \varepsilon}{\sum_b (h_P(b) + \varepsilon)}, \quad \varepsilon = 10^{-10}$$

where $h_P(b)$ is the histogram density count for bin b.

Computed for length and angle distributions of generated vs. true paths.

---

### 5.4 Stopping Prediction Metrics (Cases 3 & 4)

**Accuracy**:

$$\text{Acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[\hat{s}_i = s_i]$$

**Area Under ROC Curve (AUC)**:

$$\text{AUC} = \int_0^1 \text{TPR}\,d(\text{FPR})$$

computed by `tf.keras.metrics.AUC` (interpolation method).

**Precision, Recall, F1**:

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Average Path Length Error** (stopping-based):

$$\bar{E}_L = \frac{1}{M} \sum_{j=1}^{M} \left|\hat{n}_j - n_j\right|$$

where $n_j$ is the true number of steps for fracture j, and $\hat{n}_j$ is the predicted number of steps (determined by when the model first outputs stop = 1).

---

## 6. Compliance & Quality Scores

These scores evaluate whether generated paths are statistically consistent with the training distribution.

### Gaussian compliance score

$$\text{score}(v, \mu, \sigma) = \exp\!\left(-\frac{1}{2}\left(\frac{v - \mu}{\sigma}\right)^2\right)$$

Returns 1.0 at perfect match (v = μ), ≈ 0.607 at 1σ, ≈ 0.135 at 2σ, ≈ 0.011 at 3σ.

### Segment-length compliance

$$\text{score}_\text{seg} = \text{score}\!\left(\bar{\ell}_\text{gen},\; \mu_\text{train}^\ell,\; \sigma_\text{train}^\ell\right)$$

where $\bar{\ell}_\text{gen}$ is the mean segment length of the generated path.

### Path-length compliance

$$\text{score}_\text{path} = \text{score}\!\left(L_\text{gen},\; \mu_\text{train}^L,\; \sigma_\text{train}^L\right)$$

### Coordinate-bounds compliance

$$\text{score}_\text{bounds} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}\!\left[x_\min \le \hat{x}_i \le x_\max \text{ and } y_\min \le \hat{y}_i \le y_\max\right]$$

Fraction of generated points within the bounding box of training data.

### Oscillation detection

Consecutive direction vectors $\mathbf{d}_i$ (normalized):

$$\text{dot}_i = \mathbf{d}_i \cdot \mathbf{d}_{i+1}$$

$$\text{is\_oscillating} = \frac{\sum_i \mathbf{1}[\text{dot}_i < -\tau_\text{osc}]}{|\{\text{dot}_i\}|} \ge 0.5$$

where $\tau_\text{osc}$ is the oscillation threshold (default 0.5).

### Stagnation detection

$$\bar{m} = \frac{1}{W} \sum_{i=1}^{W} \lVert p_i - p_{i-1} \rVert, \quad \text{is\_stagnating} = \bar{m} < \tau_\text{stag}$$

where W is the stagnation window size and $\tau_\text{stag}$ is the movement threshold.

---

## 7. Model Architecture Parameters

| Parameter | Case 1 (BiLSTM-Attn) | Case 2 (Transformer-GAT) | Case 3 (LSTM-Stop) | Case 4 (CNN-GRU-MDN) |
|---|---|---|---|---|
| LSTM/GRU units | 512 (4 BiLSTM layers) | — | 256, 128 (stacked) | 256 (BiGRU) |
| Attention heads | 8 | 16 | — | — |
| Transformer layers | — | 6 | — | — |
| GAT heads | — | 8 | — | — |
| CNN filters | — | — | — | 64, 128, 128 |
| CNN dilation rates | — | — | — | 1, 2, 4, 8 |
| MDN mixtures K | — | — | — | 5 |
| Dropout rate | 0.3 | 0.2 | 0.3 | 0.3 |
| Sequence length k | 10 | 10 | 10 | 10 |
| Batch size | 32 | 32 | 32 | 32 |
| Learning rate | 1e-3 | 3e-4 | 1e-3 | 1e-3 |
| Weight decay | 0.01 | 0.01 | — | — |
| Optimizer | AdamW | AdamW | Adam | Adam |
| Max epochs | 50 | 50 | 50 | 50 |
| Early stop patience | 15 | 20 | 20 | 15 |
| LR reduce patience | 5 | 5 | 5 | 5 |
| LR reduce factor | 0.5 | 0.5 | 0.5 | 0.5 |
| λ_stop | — | — | 1.0 | 1.0 |
| Framework | TensorFlow/Keras | PyTorch | TensorFlow/Keras | PyTorch |

---

## 8. Key Dataset Statistics

### Segment-length distribution (training set)

Computed as:
$$\ell_i = \lVert p_{i+1} - p_i \rVert_2$$

Stored statistics:
- `mean`, `std`, `median`
- `min`, `max`
- `q25` (25th percentile), `q75` (75th percentile)

### Segment-angle distribution (training set)

$$\theta_i = \arctan2(y_{i+1} - y_i,\ x_{i+1} - x_i), \quad \theta_i \in (-\pi, \pi]$$

Stored: `mean`, `std`, `min`, `max`

### Curvature distribution (training set)

$$\kappa_i = \arccos\!\left(\text{clip}\!\left(\frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\lVert\mathbf{v}_1\rVert\lVert\mathbf{v}_2\rVert}, -1, 1\right)\right) \in [0, \pi]$$

Stored: `mean`, `std`, `max`

### Path-length distribution (training set)

$$L_j = \sum_{i=1}^{n_j - 1} \ell_i$$

Stored: `mean`, `std`, `min`, `max`

All of the above statistics are used exclusively from the **training set** to parameterize the compliance scorers. They are never recomputed on the test set.

---

## 9. Known Paper vs. Code Discrepancies

The following differences were identified between the published paper descriptions and the actual code implementation:

| Parameter / Claim | Paper | Code (actual) |
|---|---|---|
| Max training epochs | 100 | **50** |
| BiLSTM attention heads | 4 | **8** |
| CNN-GRU-MDN: K (mixtures) | 3 | **5** |
| CNN-GRU-MDN: GRU hidden units | 128 | **256** |
| CNN-GRU-MDN: λ_stop | 0.1 | **1.0** |
| CNN-GRU-MDN: dilation rates | (1, 2, 4) | **(1, 2, 4, 8)** |
| Stopping label definition | Penultimate point | **Last point** |
| Linear Extrapolation baseline | Described | **Not implemented** |
| Random Walk baseline | Described | **Not implemented** |

All formulas, parameters, and statistics in this document are based on the **code as implemented**, not the paper text.
