# SVM Kernel Comparison — Complete Code Walkthrough

**Target Script:** [`01_svm_kernel_comparison.py`](file:///c:/Users/Jashwanth/Documents/Projects/ML-Seminar/01_svm_kernel_comparison.py)  
**Topic:** Support Vector Machines with Different Kernels (Linear, Polynomial, RBF)  
**Seminar:** Team 15 ML Technical Seminar (Assigned Topic 1)  
**Environment:** Python 3.13 / scikit-learn 1.6 / NumPy / Matplotlib  

---

## Table of Contents
1. [Architectural Overview & Imports](#1-architectural-overview--imports)
2. [Global Constants & Experimental Hyperparameters](#2-global-constants--experimental-hyperparameters)
3. [Stage 1: Synthetic Dataset Generation (`make_moons`)](#3-stage-1-synthetic-dataset-generation-make_moons)
4. [Stage 2: Train/Test Partitioning & Split Hygiene](#4-stage-2-traintest-partitioning--split-hygiene)
5. [Stage 3: Feature Standardization (`StandardScaler`) & Leakage Prevention](#5-stage-3-feature-standardization-standardscaler--leakage-prevention)
6. [Stage 4: Model Instantiation & Mathematical Configurations](#6-stage-4-model-instantiation--mathematical-configurations)
7. [Stage 5: Model Fitting & Multi-Metric Extraction](#7-stage-5-model-fitting--multi-metric-extraction)
8. [Stage 6: Decision Surface Reconstruction & Mesh Grid Mechanics](#8-stage-6-decision-surface-reconstruction--mesh-grid-mechanics)
9. [Stage 7: Comparative Visualization & Figure Generation](#9-stage-7-comparative-visualization--figure-generation)
10. [Stage 8: Persistence & Output Serialization](#10-stage-8-persistence--output-serialization)
11. [Theoretical Alignment: Mapping Code Constructs to SVM Mathematics](#11-theoretical-alignment-mapping-code-constructs-to-svm-mathematics)

---

## 1. Architectural Overview & Imports

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

### Technical Explanation
- **`os`**: Manages filesystem operations, directory tree creation (`os.makedirs`), and path normalization across Windows/POSIX environments.
- **`numpy` (`np`)**: Powers high-performance vector/matrix operations: mesh grid arrays (`np.meshgrid`), multidimensional array flattening (`ravel()`), and column stacking (`np.c_`).
- **`matplotlib.pyplot` & `ListedColormap`**: Generates publication-quality 2D contour maps and metric bar graphs using discrete RGB color mapping.
- **`sklearn.datasets.make_moons`**: Generates two interlocking non-linear crescent point distributions.
- **`sklearn.model_selection.train_test_split`**: Partitions data arrays into training and testing subsets using stratified pseudo-random shuffling.
- **`sklearn.preprocessing.StandardScaler`**: Centers features to zero mean ($\mu = 0$) and scales to unit variance ($\sigma = 1$).
- **`sklearn.svm.SVC`**: Scikit-learn's C-Support Vector Classification implementation, backed under the hood by the highly optimized C library `libsvm`.
- **`sklearn.metrics`**: Implements standard evaluation functions for binary classification.

---

## 2. Global Constants & Experimental Hyperparameters

```python
RANDOM_STATE = 42
N_SAMPLES = 500
NOISE = 0.25
TEST_SIZE = 0.30
C_VALUE = 1.0
GAMMA = 'scale'
POLY_DEGREE = 3

OUTPUT_DIR = os.path.join('outputs', 'svm')
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

### Technical Rationale
- **`RANDOM_STATE = 42`**: Seeds both NumPy's internal Mersenne Twister PRNG and scikit-learn's random generators. This ensures identical dataset synthesis, identical sample splitting, and identical solver initialization on every run.
- **`N_SAMPLES = 500`**: Balances statistical stability against computational throughput. With 500 samples, metric estimates have low standard errors, while fitting takes milliseconds.
- **`NOISE = 0.25`**: Adds isotropic Gaussian noise $\mathcal{N}(0, \sigma^2)$ with standard deviation $\sigma = 0.25$ to point coordinates. This creates realistic overlapping at crescent boundaries, forcing soft-margin slack variables ($\xi_i > 0$) to become active.
- **`TEST_SIZE = 0.30`**: Allocates 70% of data (350 samples) for model fitting and 30% (150 samples) for out-of-sample generalization testing.
- **`C_VALUE = 1.0`**: Sets the soft-margin penalty parameter $C$. Holding $C=1.0$ constant across all three models isolates the kernel function as the single independent variable.
- **`GAMMA = 'scale'`**: Configures the kernel coefficient $\gamma$. In scikit-learn, `'scale'` sets:
  $$\gamma = \frac{1}{n_{\text{features}} \cdot \text{Var}(X)}$$
  This automatically standardizes the Gaussian exponential scale relative to input variance.
- **`POLY_DEGREE = 3`**: Sets the polynomial exponent $d=3$, testing cubic boundary curvature.
- **`os.makedirs(..., exist_ok=True)`**: Safely creates the destination directory `outputs/svm/` if it does not already exist without throwing an `OSError`.

---

## 3. Stage 1: Synthetic Dataset Generation (`make_moons`)

```python
X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE,
                  random_state=RANDOM_STATE)
```

### Technical Explanation
- Generates two 2D interleaved crescent moons.
- **Feature Matrix `X`**: NumPy array of shape `(500, 2)` representing Cartesian coordinates `(x1, x2)`.
- **Target Vector `y`**: Integer array of shape `(500,)` with binary labels $y \in \{0, 1\}$.
- **Class Balance**: Exactly 250 samples in Class 0 and 250 samples in Class 1. This balanced 50/50 prior eliminates class imbalance skew in Accuracy and F1 metrics.
- **Geometric Manifold**: The upper moon curves downward in an arc from $x_1 \approx -1.0$ to $1.0$; the lower moon curves upward from $x_1 \approx 0.0$ to $2.0$. Because the crescents wrap around each other, no single straight line can separate them without severe error.

---

## 4. Stage 2: Train/Test Partitioning & Split Hygiene

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
```

### Technical Explanation
- Partitions the 500 instances:
  - `X_train`: Shape `(350, 2)` — used for fitting models and support vector selection.
  - `X_test`: Shape `(150, 2)` — held out strictly for final metric evaluation.
  - `y_train`: Shape `(350,)`
  - `y_test`: Shape `(150,)`
- **Reproducibility Guarantee**: `random_state=42` guarantees that every execution produces the identical 350 training indices and 150 testing indices.

---

## 5. Stage 3: Feature Standardization (`StandardScaler`) & Leakage Prevention

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Technical Explanation
- **Standardization Formula**: For each feature $j \in \{1, 2\}$:
  $$x_{\text{scaled}} = \frac{x - \mu_j}{\sigma_j}$$
- **Why Scaling is Mandatory for SVM**:
  1. The margin width is $2 / \|w\|_2$. If feature 1 has variance 1,000 and feature 2 has variance 1, the optimizer will collapse the weight on feature 1 to minimize $\|w\|^2$, completely ignoring feature 2.
  2. The RBF kernel computes Euclidean distance $\|x_i - x_j\|^2 = (x_{i1} - x_{j1})^2 + (x_{i2} - x_{j2})^2$. Unscaled features distort Euclidean spheres into stretched ellipsoids.
- **Preventing Data Leakage**:
  - `scaler.fit_transform(X_train)`: Computes empirical mean $\mu_{\text{train}}$ and variance $\sigma_{\text{train}}^2$ strictly from the 350 training instances.
  - `scaler.transform(X_test)`: Applies $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ to the 150 test instances.
  - **CRITICAL**: Never call `scaler.fit(X)` on the full dataset prior to splitting. Fitting on the full dataset leaks test set distribution parameters into training, artificially inflating performance and violating clean empirical validation standards.

---

## 6. Stage 4: Model Instantiation & Mathematical Configurations

```python
kernels = {
    'Linear': SVC(kernel='linear', C=C_VALUE, gamma=GAMMA,
                  random_state=RANDOM_STATE),
    'Polynomial (d=3)': SVC(kernel='poly', C=C_VALUE, degree=POLY_DEGREE,
                            gamma=GAMMA, random_state=RANDOM_STATE),
    'RBF': SVC(kernel='rbf', C=C_VALUE, gamma=GAMMA,
               random_state=RANDOM_STATE),
}
```

### Technical Explanation
- **Model 1: `SVC(kernel='linear')`**:
  - Kernel equation: $K(x_i, x_j) = x_i^T x_j$
  - Constructs a flat 1D line in 2D space: $w_1 x_1 + w_2 x_2 + b = 0$.
- **Model 2: `SVC(kernel='poly', degree=3)`**:
  - Kernel equation: $K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$ with $d=3, r=1.0$.
  - Maps 2D input into a 10-dimensional polynomial monomial space, allowing cubic curvature.
- **Model 3: `SVC(kernel='rbf')`**:
  - Kernel equation: $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$.
  - Implicitly maps points into an infinite-dimensional reproducing kernel Hilbert space via localized Gaussian radial basis functions.

---

## 7. Stage 5: Model Fitting & Multi-Metric Extraction

```python
results = {}
for name, model in kernels.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    n_sv = model.n_support_.sum()

    results[name] = {
        'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1': f1,
        'n_support_vectors': n_sv, 'model': model
    }
```

### Technical Explanation
- **`model.fit(X_train_scaled, y_train)`**: Solves the convex dual quadratic program using the Sequential Minimal Optimization (SMO) decomposition algorithm implemented in `libsvm`.
- **`model.predict(X_test_scaled)`**: Evaluates the decision function on test points:
  $$f(x_{\text{test}}) = \text{sign}\left(\sum_{i \in \text{SV}} \alpha_i y_i K(x_i, x_{\text{test}}) + b\right)$$
- **Metrics Computed**:
  - **Accuracy**: $(TP + TN) / (TP + TN + FP + FN)$ — overall correctness.
  - **Precision**: $TP / (TP + FP)$ — proportion of predicted positives that are true positives.
  - **Recall**: $TP / (TP + FN)$ — proportion of actual positives correctly identified.
  - **F1 Score**: $2 \cdot (\text{Precision} \cdot \text{Recall}) / (\text{Precision} + \text{Recall})$ — harmonic mean balancing precision and recall.
  - **`model.n_support_.sum()`**: Sums the count of support vectors across both classes ($N_{\text{SV}} = N_{\text{SV}, 0} + N_{\text{SV}, 1}$).
- **Exact Extracted Results**:
  - **Linear**: Acc = 0.8533, Prec = 0.8630, Rec = 0.8400, F1 = 0.8514, SVs = 117
  - **Polynomial (d=3)**: Acc = 0.8733, Prec = 0.9242, Rec = 0.8133, F1 = 0.8652, SVs = 118
  - **RBF**: Acc = 0.9467, Prec = 0.9855, Rec = 0.9067, F1 = 0.9444, SVs = 95

---

## 8. Stage 6: Decision Surface Reconstruction & Mesh Grid Mechanics

```python
def plot_decision_boundary(ax, model, X, y, scaler, title, metrics):
    h = 0.02  # mesh step size

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
```

### Technical Explanation
- **`h = 0.02`**: Dense step size across the feature plane. A smaller step size produces smooth, high-resolution boundary lines without rasterization artifacts.
- **Bounding Box**: Adds a $0.5$ padding margin beyond the minimum and maximum data coordinates to ensure the entire decision region is visible.
- **`np.meshgrid`**: Generates coordinate matrices representing a dense 2D spatial grid.
- **`np.c_[xx.ravel(), yy.ravel()]`**: Flattens the 2D coordinate matrices into an array of query coordinates of shape `(M, 2)` where $M \approx 70,000$ points.
- **`model.predict(mesh_points)`**: Evaluates the trained SVM model on all 70,000 grid points.
- **`Z.reshape(xx.shape)`**: Reconstructs the predictions into a 2D matrix matching the grid topology.

---

## 9. Stage 7: Comparative Visualization & Figure Generation

```python
    cmap_bg = ListedColormap(['#FFE0B2', '#B3E5FC'])
    cmap_pts = ListedColormap(['#E65100', '#01579B'])

    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg)
    ax.contour(xx, yy, Z, colors='#455A64', linewidths=1.5, linestyles='--')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pts,
                         edgecolors='white', s=30, linewidths=0.8, alpha=0.85)
```

### Technical Explanation
- **`ax.contourf`**: Fills the continuous decision regions with soft pastel colors (`#FFE0B2` for Class 0, `#B3E5FC` for Class 1) at 40% transparency (`alpha=0.4`).
- **`ax.contour`**: Renders the decision boundary where $f(x) = 0$ as a distinct dashed charcoal line (`#455A64`, `linewidths=1.5`, `linestyles='--'`).
- **`ax.scatter`**: Overlays the actual training data points with white borders for clean visual pop and high contrast.
- **`ax.text(..., bbox=props)`**: Places an anchored monospace text box in the bottom-right corner displaying test Accuracy and F1 score.

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, (name, r) in zip(axes, results.items()):
    plot_decision_boundary(ax, r['model'], X_train_scaled, y_train, scaler, f'{name} Kernel', r)
```
- Creates a panoramic $1 \times 3$ subplot canvas (18 inches wide, 5.5 inches tall), rendering Linear, Polynomial, and RBF decision boundaries side-by-side for direct visual comparison.

---

## 10. Stage 8: Persistence & Output Serialization

```python
# Save decision boundaries plot
fig.savefig(os.path.join(OUTPUT_DIR, 'svm_decision_boundaries.png'),
            dpi=200, bbox_inches='tight', facecolor='white')

# Save metrics bar chart
fig2.savefig(os.path.join(OUTPUT_DIR, 'svm_metrics_comparison.png'),
             dpi=200, bbox_inches='tight', facecolor='white')

# Save metrics summary table
with open(os.path.join(OUTPUT_DIR, 'svm_results.txt'), 'w') as f:
    ...
```

### Technical Explanation
- **`dpi=200` & `bbox_inches='tight'`**: Generates razor-sharp figures at 3600 x 1100 resolution, trimmed of excess whitespace padding.
- **`svm_results.txt`**: Writes an formatted ASCII scorecard recording parameters, train/test ratios, metrics, and support vector counts.

---

## 11. Theoretical Alignment: Mapping Code Constructs to SVM Mathematics

| Code Construct in `01_svm_kernel_comparison.py` | Mathematical Concept | Formal Theoretical Definition |
|---|---|---|
| `C_VALUE = 1.0` | Soft-Margin Regularization Constant | $\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^n \xi_i$ |
| `GAMMA = 'scale'` | Kernel Width Parameter $\gamma$ | $K(x, z) = \exp(-\gamma \|x - z\|^2), \quad \gamma = \frac{1}{d \cdot \text{Var}(X)}$ |
| `POLY_DEGREE = 3` | Monomial Degree $d$ | $K(x, z) = (\gamma x^T z + r)^d$ |
| `model.fit()` | Sequential Minimal Optimization (SMO) | Dual QP: $\max_\alpha \sum \alpha_i - \frac{1}{2}\sum \alpha_i \alpha_j y_i y_j K(x_i, x_j)$ |
| `model.n_support_.sum()` | Support Vector Count ($N_{\text{SV}}$) | Cardinality: $|\{i : \alpha_i > 0\}|$ |
| `model.predict()` | Kernelized Decision Rule | $f(x) = \text{sign}\left(\sum_{i \in \text{SV}} \alpha_i y_i K(x_i, x) + b\right)$ |
| `StandardScaler()` | Variance Normalization | Distance invariance: $\|x_i - x_j\|^2 = \sum \frac{(x_{ik} - x_{jk})^2}{\sigma_k^2}$ |
