"""
app.py
------
Streamlit interactive dashboard for PCA-MNIST exploration.

Run
---
    streamlit run app.py
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------------------------------------------------ #
# Page config                                                          #
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="PCA Explorer – MNIST",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# Custom CSS                                                           #
# ------------------------------------------------------------------ #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    h1 { font-weight: 800; letter-spacing: -1px; }
    h2, h3 { font-weight: 600; }
    code, .stCode { font-family: 'JetBrains Mono', monospace; }
    .metric-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: #e0e0e0;
    }
    .stSlider > div > div { background: #0f3460; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Data loading (cached)                                               #
# ------------------------------------------------------------------ #
@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    from sklearn.datasets import load_digits

    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y, scaler


@st.cache_data(show_spinner="Fitting full PCA …")
def fit_full_pca(_X_scaled):
    pca = PCA(random_state=42)
    pca.fit(_X_scaled)
    return pca


# ------------------------------------------------------------------ #
# Header                                                               #
# ------------------------------------------------------------------ #
st.title("🔢 PCA Explorer — MNIST Handwritten Digits")
st.markdown(
    "Interactively explore **Principal Component Analysis** on 1,797 digit images "
    "(64 features each). Adjust the slider to see how dimensionality affects "
    "reconstruction quality and explained variance in real time."
)

# ------------------------------------------------------------------ #
# Load data                                                            #
# ------------------------------------------------------------------ #
with st.spinner("Preparing dataset …"):
    X_raw, X_scaled, y, scaler = load_data()
    pca_full = fit_full_pca(X_scaled)

cumvar = np.cumsum(pca_full.explained_variance_ratio_)

# ------------------------------------------------------------------ #
# Sidebar controls                                                     #
# ------------------------------------------------------------------ #
st.sidebar.header("⚙️ Controls")

n_components = st.sidebar.slider(
    "Number of PCA Components (k)",
    min_value=1, max_value=200, value=50, step=1,
)

digit_filter = st.sidebar.multiselect(
    "Filter digits for scatter plot",
    options=list(range(10)),
    default=list(range(10)),
)

n_recon_samples = st.sidebar.slider(
    "Reconstruction samples to display",
    min_value=3, max_value=10, value=5,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Math behind PCA**\n\n"
    "1. Center data → compute covariance Σ\n"
    "2. Eigen-decompose: Σ v = λ v\n"
    "3. Sort by λ (descending)\n"
    "4. Project: Z = X · V_k"
)

# ------------------------------------------------------------------ #
# Key metrics row                                                      #
# ------------------------------------------------------------------ #
var_explained = float(cumvar[n_components - 1]) * 100
var_remaining = 100 - var_explained
col1, col2, col3, col4 = st.columns(4)
col1.metric("Components selected", f"{n_components}", f"of {X_scaled.shape[1]}")
col2.metric("Variance explained", f"{var_explained:.2f}%")
col3.metric("Variance discarded", f"{var_remaining:.2f}%")
col4.metric("Compression ratio", f"{784 / n_components:.1f}×")

st.markdown("---")

# ------------------------------------------------------------------ #
# Row 1 – Variance curve  +  2D scatter                               #
# ------------------------------------------------------------------ #
left, right = st.columns([1.1, 1])

with left:
    st.subheader("Explained Variance vs. Components")
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    xs = np.arange(1, len(cumvar) + 1)
    ax1.plot(xs, cumvar * 100, color="#4fc3f7", linewidth=2)
    ax1.axvline(n_components, color="#ef5350", linestyle="--", linewidth=1.5,
                label=f"k={n_components} → {var_explained:.1f}%")
    ax1.axhline(95, color="#aaa", linestyle=":", linewidth=1, label="95% threshold")
    ax1.fill_between(xs, cumvar * 100, alpha=0.15, color="#4fc3f7")
    ax1.set_xlabel("Number of Components", fontsize=10)
    ax1.set_ylabel("Cumulative Variance (%)", fontsize=10)
    ax1.set_xlim(0, 300)
    ax1.legend(fontsize=8)
    ax1.set_facecolor("#0d1117")
    fig1.patch.set_facecolor("#0d1117")
    ax1.tick_params(colors="#ccc")
    ax1.spines[:].set_color("#333")
    ax1.yaxis.label.set_color("#ccc")
    ax1.xaxis.label.set_color("#ccc")
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

with right:
    st.subheader("2D PCA Scatter (first 5,000 samples)")

    # Fit 2D PCA on filtered digits
    mask = np.isin(y, digit_filter) if digit_filter else np.ones(len(y), dtype=bool)
    X_filt = X_scaled[mask][:5000]
    y_filt = y[mask][:5000]

    pca2 = PCA(n_components=2, random_state=42).fit(X_scaled)
    X_2d = pca2.transform(X_filt)

    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    cmap = plt.cm.get_cmap("tab10", 10)
    sc = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=y_filt, cmap=cmap,
                     alpha=0.4, s=5, linewidths=0)
    cbar = fig2.colorbar(sc, ax=ax2, ticks=sorted(digit_filter))
    cbar.ax.tick_params(colors="#ccc", labelsize=8)
    ax2.set_facecolor("#0d1117")
    fig2.patch.set_facecolor("#0d1117")
    ax2.tick_params(colors="#ccc")
    ax2.spines[:].set_color("#333")
    ax2.set_xlabel("PC 1", color="#ccc", fontsize=10)
    ax2.set_ylabel("PC 2", color="#ccc", fontsize=10)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ------------------------------------------------------------------ #
# Row 2 – Image Reconstruction                                        #
# ------------------------------------------------------------------ #
st.markdown("---")
st.subheader(f"Image Reconstruction with k = {n_components} Components")

# Use a fixed set of samples for speed
@st.cache_data(show_spinner=False)
def get_sample_indices(n=10):
    rng = np.random.default_rng(0)
    return rng.choice(len(X_raw), size=n, replace=False)

sample_idx = get_sample_indices(n_recon_samples)
X_sample = X_scaled[sample_idx]
X_sample_raw = X_raw[sample_idx]

pca_k = PCA(n_components=n_components, random_state=42).fit(X_scaled)
X_proj = pca_k.transform(X_sample)
X_rec_scaled = pca_k.inverse_transform(X_proj)
X_rec = np.clip(scaler.inverse_transform(X_rec_scaled), 0, 255)
mse = float(np.mean((X_sample_raw - X_rec) ** 2))

cols_orig = st.columns(n_recon_samples)
cols_rec  = st.columns(n_recon_samples)

for i in range(n_recon_samples):
    with cols_orig[i]:
        fig_o, ax_o = plt.subplots(figsize=(1.5, 1.5))
        ax_o.imshow(X_sample_raw[i].reshape(8, 8), cmap="gray")
        ax_o.axis("off")
        if i == 0:
            ax_o.set_title("Original", fontsize=7, color="#ccc")
        fig_o.patch.set_alpha(0)
        st.pyplot(fig_o, use_container_width=True)
        plt.close(fig_o)

    with cols_rec[i]:
        fig_r, ax_r = plt.subplots(figsize=(1.5, 1.5))
        ax_r.imshow(X_rec[i].reshape(8, 8), cmap="gray")
        ax_r.axis("off")
        if i == 0:
            ax_r.set_title("Reconstructed", fontsize=7, color="#ccc")
        fig_r.patch.set_alpha(0)
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

st.caption(
    f"**Reconstruction MSE: {mse:.2f}** — "
    f"Lower is better. k={n_components} captures "
    f"{var_explained:.1f}% of the pixel-space variance."
)

# ------------------------------------------------------------------ #
# Row 3 – Per-component variance bar                                  #
# ------------------------------------------------------------------ #
st.markdown("---")
st.subheader("Individual Explained Variance (first 100 components)")
evr = pca_full.explained_variance_ratio_ * 100
n = len(evr)

fig3, ax3 = plt.subplots(figsize=(10, 2.8))
colors = ["#ef5350" if i < n_components else "#37474f" for i in range(n)]
ax3.bar(range(1, n + 1), evr, color=colors, width=0.8)
ax3.set_xlabel("Component index", color="#ccc", fontsize=9)
ax3.set_ylabel("% Variance", color="#ccc", fontsize=9)
ax3.set_facecolor("#0d1117")
fig3.patch.set_facecolor("#0d1117")
ax3.tick_params(colors="#ccc", labelsize=7)
ax3.spines[:].set_color("#333")
st.pyplot(fig3, use_container_width=True)
plt.close(fig3)

st.caption("Red bars = selected components; grey = excluded.")

# ------------------------------------------------------------------ #
# Footer                                                               #
# ------------------------------------------------------------------ #
st.markdown("---")
st.markdown(
    "<small>Built with **Streamlit · scikit-learn · NumPy · Matplotlib** | "
    "PCA-MNIST Project</small>",
    unsafe_allow_html=True,
)
