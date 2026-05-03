"""
pca_utils.py
------------
Utility functions for PCA-based dimensionality reduction on MNIST.
Covers data loading, preprocessing, PCA fitting, variance analysis,
and image reconstruction.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

def load_mnist(test_size: float = 0.2, random_state: int = 42):
    """
    Load MNIST from sklearn, split into train/test, and return raw arrays.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    print("Loading MNIST dataset …")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X, y = mnist.data.astype(np.float32), mnist.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def standardize(X_train: np.ndarray, X_test: np.ndarray):
    """
    Fit StandardScaler on training data and transform both splits.

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
    scaler : fitted StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# PCA Core
# ---------------------------------------------------------------------------

def fit_pca(X_train_scaled: np.ndarray, n_components: int):
    """
    Fit a PCA model with the specified number of components.

    Mathematical background
    -----------------------
    1. Compute the covariance matrix  Σ = (1/n) Xᵀ X  (after mean-centering).
    2. Eigen-decompose: Σ vᵢ = λᵢ vᵢ.
       - λᵢ  : eigenvalues  → variance captured by each PC.
       - vᵢ  : eigenvectors → principal directions (components).
    3. Sort by descending eigenvalue; keep top-k eigenvectors.
    4. Project: Z = X · V_k  (k × 784 weight matrix).

    sklearn's PCA uses truncated SVD internally (numerically superior).

    Returns
    -------
    pca : fitted PCA object
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train_scaled)
    return pca


def transform(pca: PCA, X_scaled: np.ndarray) -> np.ndarray:
    """Project scaled data into the PCA subspace."""
    return pca.transform(X_scaled)


def variance_summary(pca: PCA) -> dict:
    """
    Return a dict with explained variance ratio and cumulative variance.
    """
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    return {
        "explained_variance_ratio": evr,
        "cumulative_variance": cumvar,
        "total_variance_explained": cumvar[-1],
        "n_components": pca.n_components_,
    }


def components_for_variance(X_train_scaled: np.ndarray, threshold: float = 0.95) -> int:
    """
    Return the minimum number of PCA components needed to explain
    `threshold` fraction of total variance.
    """
    pca_full = PCA(random_state=42).fit(X_train_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n = int(np.searchsorted(cumvar, threshold) + 1)
    print(f"  {n} components explain ≥ {threshold*100:.0f}% variance.")
    return n


# ---------------------------------------------------------------------------
# Image Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_images(
    X_scaled: np.ndarray,
    scaler: StandardScaler,
    n_components_list: list,
) -> dict:
    """
    Reconstruct images using varying numbers of PCA components.

    Returns
    -------
    dict mapping n_components → reconstructed pixel array (original scale)
    """
    reconstructions = {}
    for n in n_components_list:
        pca = PCA(n_components=n, random_state=42).fit(X_scaled)
        X_proj = pca.transform(X_scaled)
        X_rec_scaled = pca.inverse_transform(X_proj)
        X_rec = scaler.inverse_transform(X_rec_scaled)
        reconstructions[n] = np.clip(X_rec, 0, 255)
    return reconstructions


def reconstruction_loss(X_original: np.ndarray, X_reconstructed: np.ndarray) -> float:
    """Mean squared error between original and reconstructed pixel values."""
    return float(np.mean((X_original - X_reconstructed) ** 2))


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_2d_scatter(
    X_2d: np.ndarray,
    y: np.ndarray,
    title: str = "PCA – 2D Projection of MNIST",
    save_path: str = None,
):
    """Color-coded 2D scatter plot of PCA-reduced data."""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab10", 10)
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=y, cmap=cmap, alpha=0.35, s=6, linewidths=0,
    )
    cbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Digit Class", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_explained_variance(pca_full: PCA, highlight: list = None, save_path: str = None):
    """
    Plot individual and cumulative explained variance ratio vs. number of components.
    """
    evr = pca_full.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    n = len(evr)
    xs = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    ax1.bar(xs, evr, color="#4C72B0", alpha=0.6, label="Individual variance")
    ax2.plot(xs, cumvar, color="#DD8452", linewidth=2, label="Cumulative variance")
    ax2.axhline(0.95, color="gray", linestyle="--", linewidth=1, label="95% threshold")

    if highlight:
        for h in highlight:
            ax1.axvline(h, color="red", linestyle=":", linewidth=1.5, label=f"k={h}")

    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("Explained Variance Ratio", color="#4C72B0")
    ax2.set_ylabel("Cumulative Variance", color="#DD8452")
    ax1.set_title("PCA Explained Variance – MNIST", fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_reconstructions(
    original: np.ndarray,
    reconstructions: dict,
    n_samples: int = 5,
    save_path: str = None,
):
    """
    Grid showing original images and reconstructions at various component counts.
    """
    n_rows = 1 + len(reconstructions)
    fig = plt.figure(figsize=(n_samples * 2, n_rows * 2.2))
    gs = gridspec.GridSpec(n_rows, n_samples, hspace=0.4)

    for col in range(n_samples):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(original[col].reshape(28, 28), cmap="gray")
        ax.axis("off")
        if col == 0:
            ax.set_ylabel("Original", fontsize=9)

    for row_idx, (n_comp, rec) in enumerate(reconstructions.items(), start=1):
        loss = reconstruction_loss(original[:n_samples], rec[:n_samples])
        for col in range(n_samples):
            ax = fig.add_subplot(gs[row_idx, col])
            ax.imshow(rec[col].reshape(28, 28), cmap="gray")
            ax.axis("off")
        fig.text(
            0.01, 1 - row_idx / n_rows + 0.03,
            f"k={n_comp}  MSE={loss:.1f}",
            va="center", fontsize=8, color="#333"
        )

    fig.suptitle("Image Reconstruction at Varying PCA Components", fontsize=13, fontweight="bold")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
