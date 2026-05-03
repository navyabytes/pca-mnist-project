"""
model.py
--------
Classification pipeline: train Logistic Regression and KNN on raw vs.
PCA-reduced MNIST features, then evaluate and compare.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 300,
    C: float = 1.0,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Fit a multinomial Logistic Regression classifier.

    Parameters
    ----------
    X_train      : Feature matrix (n_samples, n_features)
    y_train      : Integer labels
    max_iter     : Solver iteration budget
    C            : Inverse regularisation strength
    random_state : Reproducibility seed

    Returns
    -------
    Fitted LogisticRegression model
    """
    clf = LogisticRegression(
        max_iter=max_iter,
        C=C,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 5,
) -> KNeighborsClassifier:
    """
    Fit a K-Nearest Neighbours classifier.

    Parameters
    ----------
    X_train    : Feature matrix
    y_train    : Integer labels
    n_neighbors: k for KNN

    Returns
    -------
    Fitted KNeighborsClassifier model
    """
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label: str = "Model",
) -> dict:
    """
    Run inference and return accuracy, report, and confusion matrix.

    Parameters
    ----------
    clf    : Fitted sklearn classifier
    X_test : Test feature matrix
    y_test : True labels
    label  : Human-readable name for console output

    Returns
    -------
    dict with keys: accuracy, report, confusion_matrix
    """
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    elapsed = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Inference : {elapsed*1000:.1f} ms (full test set)")
    print(f"\n{report}")

    return {"accuracy": acc, "report": report, "confusion_matrix": cm, "label": label}


# ---------------------------------------------------------------------------
# Benchmarking: raw vs PCA
# ---------------------------------------------------------------------------

def benchmark(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    X_train_pca: np.ndarray,
    X_test_pca: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    pca_n: int = 50,
) -> dict:
    """
    Train Logistic Regression on raw and PCA-50 features, time both,
    and return comparative results.

    Returns
    -------
    dict with results for 'raw' and 'pca' conditions
    """
    results = {}

    print("\n[1/2] Training on RAW features (784 dims) …")
    t0 = time.perf_counter()
    clf_raw = train_logistic_regression(X_train_raw, y_train, max_iter=300)
    results["raw"] = {
        "train_time": time.perf_counter() - t0,
        **evaluate_model(clf_raw, X_test_raw, y_test, label=f"Logistic Regression – RAW (784 dims)"),
    }

    print(f"\n[2/2] Training on PCA features ({pca_n} dims) …")
    t0 = time.perf_counter()
    clf_pca = train_logistic_regression(X_train_pca, y_train, max_iter=300)
    results["pca"] = {
        "train_time": time.perf_counter() - t0,
        **evaluate_model(clf_pca, X_test_pca, y_test, label=f"Logistic Regression – PCA ({pca_n} dims)"),
    }

    # Summary table
    print("\n" + "="*55)
    print("  COMPARISON SUMMARY")
    print("="*55)
    for key, res in results.items():
        print(
            f"  {res['label'][:40]:<40} "
            f"Acc={res['accuracy']*100:.2f}%  "
            f"Train={res['train_time']:.1f}s"
        )

    return results


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str = None,
):
    """Heatmap of a confusion matrix."""
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
        ax=ax,
        linewidths=0.4,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
