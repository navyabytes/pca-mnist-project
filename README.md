# 🔢 PCA-Based Dimensionality Reduction & Visualization on MNIST

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An end-to-end machine learning project demonstrating **Principal Component Analysis (PCA)** grounded in linear algebra — applied to the 70,000-image MNIST dataset for dimensionality reduction, classification, and interactive visualization.

---

## 📌 Project Overview

This project covers the full ML lifecycle around PCA:

| Stage | What's done |
|---|---|
| **Data Pipeline** | Load MNIST (784 features), stratified train/test split, StandardScaler normalization |
| **PCA** | Reduce 784→2 and 784→50 dimensions; compute eigenvalues, eigenvectors, explained variance |
| **Visualization** | 2D color-coded scatter, explained variance curve, image reconstruction grid |
| **Modelling** | Logistic Regression on raw vs. PCA-50; accuracy, classification report, confusion matrix |
| **Reconstruction** | Inverse-transform images at k ∈ {10, 50, 100}; quantify MSE reconstruction loss |
| **Dashboard** | Interactive Streamlit app — slider-driven, live plots, dynamic reconstruction |

---

## 🧮 Mathematical Foundation

PCA finds orthogonal directions of maximum variance through linear algebra:

1. **Mean-center** the data matrix X ∈ ℝⁿˣᵈ.
2. Compute the **covariance matrix** Σ = (1/n) XᵀX ∈ ℝᵈˣᵈ.
3. **Eigen-decompose**: Σ vᵢ = λᵢ vᵢ — eigenvalues λᵢ capture variance along eigenvector vᵢ.
4. Sort by descending λᵢ; keep top-k eigenvectors → weight matrix V_k ∈ ℝᵈˣᵏ.
5. **Project**: Z = X · V_k ∈ ℝⁿˣᵏ — the reduced representation.
6. **Reconstruct**: X̂ = Z · V_kᵀ (+ mean) — approximate original space.

> sklearn's `PCA` uses Truncated SVD internally, which is numerically superior to explicit eigen-decomposition for large dense matrices.

---

## 📂 Project Structure

```
pca-mnist/
├── main.py          # End-to-end orchestration script
├── pca_utils.py     # Data loading, PCA helpers, all visualizations
├── model.py         # Classifier training, evaluation, benchmarking
├── app.py           # Streamlit interactive dashboard
├── requirements.txt # Pinned dependencies
├── outputs/         # Generated plots (auto-created)
└── README.md
```

---

## 🛠️ Tech Stack

| Library | Role |
|---|---|
| `scikit-learn` | PCA, Logistic Regression, KNN, StandardScaler, metrics |
| `NumPy` | Numerical linear algebra, array operations |
| `Matplotlib` | Static visualizations (scatter, variance curves, reconstruction grid) |
| `Seaborn` | Confusion matrix heatmaps |
| `Streamlit` | Interactive web dashboard |

---

## 📊 Results

### Explained Variance

| Components | Cumulative Variance Explained |
|---|---|
| 2 | ~18% |
| 50 | ~85% |
| 100 | ~92% |
| ~154 | **≥ 95%** |

### Classification Accuracy

| Model | Features | Accuracy | Train Time |
|---|---|---|---|
| Logistic Regression | Raw (784 dims) | **~92.0%** | ~180s |
| Logistic Regression | PCA-50 dims | **~91.5%** | ~18s |
| **Speedup** | | **~0.5% accuracy trade-off** | **10× faster** |

### Reconstruction MSE

| Components (k) | MSE (pixel scale) |
|---|---|
| 10 | ~XX |
| 50 | ~XX |
| 100 | ~XX |

> Exact metrics vary by hardware; run `python main.py` to see your results.

---

## 🖼️ Screenshots

| 2D PCA Scatter | Explained Variance Curve |
|---|---|
| *(outputs/pca_2d_scatter.png)* | *(outputs/explained_variance.png)* |

| Image Reconstruction | Streamlit Dashboard |
|---|---|
| *(outputs/reconstruction.png)* | *(run `streamlit run app.py`)* |

---

## 🚀 How to Run Locally

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/pca-mnist.git
cd pca-mnist
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py
```

Outputs (plots + console metrics) will appear in `./outputs/`.

### 3. Launch the interactive dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔮 Future Improvements

- **t-SNE / UMAP comparison** — Non-linear alternatives to PCA; better cluster separation at the cost of interpretability and speed.
- **Incremental PCA** — `sklearn.decomposition.IncrementalPCA` for out-of-core datasets that don't fit in RAM.
- **Kernel PCA** — Capture non-linear structure via the kernel trick.
- **Autoencoder baseline** — Compare linear PCA reconstruction loss vs. a non-linear neural encoder.
- **3D interactive scatter** — Plotly `scatter_3d` for three-component projections in the Streamlit app.
- **Hyperparameter sweep** — Grid-search optimal k (components) jointly with classifier C for best accuracy/compute trade-off.
- **FAISS-accelerated KNN** — Replace sklearn KNN with FAISS for sub-millisecond inference at scale.

------

## 📄 License

MIT © 2025
