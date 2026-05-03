"""
main.py
-------
End-to-end orchestration script for the PCA-MNIST project.

Pipeline
--------
1. Load & standardize MNIST
2. Fit PCA (2D and 50D); report variance
3. Visualize 2D projection and explained variance curve
4. Benchmark Logistic Regression: raw vs. PCA-50
5. Reconstruct images at k ∈ {10, 50, 100} components
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt

from pca_utils import (
    load_mnist,
    standardize,
    fit_pca,
    transform,
    variance_summary,
    components_for_variance,
    reconstruct_images,
    reconstruction_loss,
    plot_2d_scatter,
    plot_explained_variance,
    plot_reconstructions,
)
from model import benchmark, plot_confusion_matrix
from sklearn.decomposition import PCA

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # ------------------------------------------------------------------ #
    # 1. Data Pipeline                                                     #
    # ------------------------------------------------------------------ #
    X_train, X_test, y_train, y_test = load_mnist(test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, scaler = standardize(X_train, X_test)

    # ------------------------------------------------------------------ #
    # 2. PCA – 2 components                                               #
    # ------------------------------------------------------------------ #
    print("\n--- PCA: 2 Components ---")
    pca_2 = fit_pca(X_train_scaled, n_components=2)
    X_train_2d = transform(pca_2, X_train_scaled)
    X_test_2d  = transform(pca_2, X_test_scaled)
    v2 = variance_summary(pca_2)
    print(f"  Variance explained (2 PCs): {v2['total_variance_explained']*100:.2f}%")

    # ------------------------------------------------------------------ #
    # 3. PCA – 50 components                                              #
    # ------------------------------------------------------------------ #
    print("\n--- PCA: 50 Components ---")
    pca_50 = fit_pca(X_train_scaled, n_components=50)
    X_train_50 = transform(pca_50, X_train_scaled)
    X_test_50  = transform(pca_50, X_test_scaled)
    v50 = variance_summary(pca_50)
    print(f"  Variance explained (50 PCs): {v50['total_variance_explained']*100:.2f}%")

    # How many components for 95% variance?
    components_for_variance(X_train_scaled, threshold=0.95)

    # ------------------------------------------------------------------ #
    # 4. Visualizations                                                    #
    # ------------------------------------------------------------------ #
    print("\n--- Generating Visualizations ---")

    # 2D scatter
    fig_scatter = plot_2d_scatter(
        X_train_2d[:8000], y_train[:8000],
        title="PCA – 2D Projection of MNIST (train, 8k samples)",
        save_path=os.path.join(OUTPUT_DIR, "pca_2d_scatter.png"),
    )
    plt.close(fig_scatter)

    # Explained variance curve (full PCA)
    pca_full = PCA(random_state=42).fit(X_train_scaled)
    fig_var = plot_explained_variance(
        pca_full,
        highlight=[2, 50, 100],
        save_path=os.path.join(OUTPUT_DIR, "explained_variance.png"),
    )
    plt.close(fig_var)

    # ------------------------------------------------------------------ #
    # 5. Model Benchmarking                                               #
    # ------------------------------------------------------------------ #
    print("\n--- Model Benchmarking ---")
    results = benchmark(
        X_train_raw=X_train_scaled,
        X_test_raw=X_test_scaled,
        X_train_pca=X_train_50,
        X_test_pca=X_test_50,
        y_train=y_train,
        y_test=y_test,
        pca_n=50,
    )

    # Confusion matrices
    for key in ("raw", "pca"):
        tag = "RAW_784" if key == "raw" else "PCA_50"
        fig_cm = plot_confusion_matrix(
            results[key]["confusion_matrix"],
            title=f"Confusion Matrix – {tag}",
            save_path=os.path.join(OUTPUT_DIR, f"confusion_{tag.lower()}.png"),
        )
        plt.close(fig_cm)

    # ------------------------------------------------------------------ #
    # 6. Image Reconstruction                                             #
    # ------------------------------------------------------------------ #
    print("\n--- Image Reconstruction ---")
    k_list = [10, 50, 100]
    reconstructions = reconstruct_images(X_train_scaled[:20], scaler, k_list)

    losses = {
        k: reconstruction_loss(X_train[:5], reconstructions[k][:5])
        for k in k_list
    }
    for k, mse in losses.items():
        print(f"  k={k:>3} components → MSE = {mse:.2f}")

    fig_rec = plot_reconstructions(
        original=X_train[:5],
        reconstructions={k: reconstructions[k] for k in k_list},
        n_samples=5,
        save_path=os.path.join(OUTPUT_DIR, "reconstruction.png"),
    )
    plt.close(fig_rec)

    # ------------------------------------------------------------------ #
    # 7. Final Summary                                                    #
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(f"  Raw  accuracy  : {results['raw']['accuracy']*100:.2f}%")
    print(f"  PCA-50 accuracy: {results['pca']['accuracy']*100:.2f}%")
    print(f"  Train speedup  : {results['raw']['train_time']/results['pca']['train_time']:.1f}×")
    print(f"\n  Outputs saved to: ./{OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
