"""Figures (matplotlib, colour-blind-safe palette, grey = noise, legends always present)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import CLUSTER_COLORS, FEATURES, NOISE_COLOR, SEASON_ORDER
from .interpret import label_names

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def _color_for(name: str) -> str:
    if name == "Noise":
        return NOISE_COLOR
    return CLUSTER_COLORS[int(name[1:]) % len(CLUSTER_COLORS)]


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_k_distance(kd: np.ndarray, k: int, knee_idx: int, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(kd, color=CLUSTER_COLORS[0], lw=1.6, label=f"sorted {k}-distance")
    ax.scatter([knee_idx], [kd[knee_idx]], color=CLUSTER_COLORS[5], zorder=3, s=45,
               label=f"knee, eps ≈ {kd[knee_idx]:.2f}")
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel(f"Distance to {k}-th nearest neighbour (scaled units)")
    ax.set_title("k-distance graph for eps selection")
    ax.legend(frameon=False)
    _save(fig, out)


def plot_grid(grid: pd.DataFrame, chosen: pd.Series, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title, cmap in [
        (axes[0], "silhouette", "Silhouette (non-noise points)", "Blues"),
        (axes[1], "noise_ratio", "Noise ratio", "Oranges"),
    ]:
        pv = grid.pivot(index="min_samples", columns="eps", values=col)
        im = ax.imshow(pv.values, aspect="auto", cmap=cmap, origin="lower")
        ax.set_xticks(range(len(pv.columns)), [f"{e:g}" for e in pv.columns], rotation=45)
        ax.set_yticks(range(len(pv.index)), pv.index)
        ax.set_xlabel("eps")
        ax.set_ylabel("min_samples")
        ax.set_title(title)
        ax.grid(False)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        xi = list(pv.columns).index(chosen["eps"])
        yi = list(pv.index).index(int(chosen["min_samples"]))
        ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, ec="black", lw=2))
        fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, out)


def plot_pca(X: np.ndarray, labels: np.ndarray, out: Path, title: str) -> None:
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    names = label_names(labels).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for n in sorted(set(names), key=lambda s: (s == "Noise", s)):
        m = names == n
        ax.scatter(Z[m, 0], Z[m, 1], s=10 if n != "Noise" else 18, alpha=0.55 if n != "Noise" else 0.9,
                   color=_color_for(n), marker="o" if n != "Noise" else "x",
                   label=f"{n} (n={m.sum()})")
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.0f}% var.)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.0f}% var.)")
    ax.set_title(title)
    ax.legend(frameon=False, markerscale=1.5)
    _save(fig, out)


def plot_profiles(z: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 0.8 + 0.6 * len(z)))
    lim = max(1.0, float(np.abs(z.values).max()))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(FEATURES)), FEATURES)
    ax.set_yticks(range(len(z)), z.index)
    ax.grid(False)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z.values[i, j]:+.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Cluster profile (mean z-score per feature)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="z-score")
    _save(fig, out)


def plot_seasons(season_pct: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(season_pct))
    cols = [c for c in SEASON_ORDER if c in season_pct.columns]
    shades = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]
    for c, col in zip(cols, shades):
        ax.barh(season_pct.index, season_pct[c], left=bottom, color=col, edgecolor="white", lw=1.5, label=c)
        bottom += season_pct[c].to_numpy()
    ax.set_xlabel("% of cluster's observations")
    ax.set_title("Seasonal composition of each cluster")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", visible=False)
    _save(fig, out)


def plot_noise_timeline(df: pd.DataFrame, labels: np.ndarray, out: Path) -> None:
    d = pd.DataFrame({"datetime": df["datetime"], "noise": labels == -1})
    monthly = d.set_index("datetime")["noise"].resample("MS").mean() * 100
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(monthly.index, monthly.values, color=CLUSTER_COLORS[5], lw=2, label="% of observations flagged as noise")
    ax.set_ylabel("Noise share (%)")
    ax.set_title("When do anomalous observations occur?")
    ax.legend(frameon=False)
    _save(fig, out)
