"""Turn cluster labels into findings: profiles, noise diagnostics, seasonality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURES, SEASON_ORDER, SEASONS


def label_names(labels: np.ndarray) -> pd.Series:
    """-1 -> 'Noise', otherwise 'C0', 'C1', ... (ordered by cluster id)."""
    return pd.Series(np.where(labels == -1, "Noise", [f"C{l}" for l in labels]))


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Size, share and per-feature median / mean in ORIGINAL units."""
    d = df[FEATURES].copy()
    d["cluster"] = label_names(labels).to_numpy()
    g = d.groupby("cluster")
    out = g.size().rename("size").to_frame()
    out["share_%"] = (100 * out["size"] / len(d)).round(2)
    for f in FEATURES:
        out[f"{f}_median"] = g[f].median().round(2)
        out[f"{f}_mean"] = g[f].mean().round(2)
    return out.sort_values("size", ascending=False)


def noise_drivers(X_scaled: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """For every noise point: which feature deviates most (in z-units), and in which direction?"""
    mask = labels == -1
    if not mask.any():
        return pd.DataFrame(columns=["driver", "count", "share_%"])
    Z = X_scaled[mask]
    j = np.abs(Z).argmax(axis=1)
    sign = np.where(Z[np.arange(len(Z)), j] >= 0, "high", "low")
    drivers = pd.Series([f"{FEATURES[i]} {s}" for i, s in zip(j, sign)])
    out = drivers.value_counts().rename("count").to_frame()
    out["share_%"] = (100 * out["count"] / out["count"].sum()).round(1)
    out.index.name = "driver"
    return out.reset_index()


def seasonal_table(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Row-normalised (%) distribution of each cluster over the four seasons."""
    season = df["datetime"].dt.month.map(SEASONS)
    ct = pd.crosstab(label_names(labels).to_numpy(), season.to_numpy(), normalize="index") * 100
    ct = ct.reindex(columns=[s for s in SEASON_ORDER if s in ct.columns]).round(1)
    ct.index.name = "cluster"
    return ct


def station_table(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Share of each station's observations that are noise / in each cluster."""
    ct = pd.crosstab(df["station"].to_numpy(), label_names(labels).to_numpy(), normalize="index") * 100
    ct.index.name = "station"
    return ct.round(1)


def cluster_zscores(X_scaled: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Mean z-score per cluster and feature (input to the profile heatmap)."""
    d = pd.DataFrame(X_scaled, columns=FEATURES)
    d["cluster"] = label_names(labels).to_numpy()
    return d.groupby("cluster")[FEATURES].mean().sort_index()
