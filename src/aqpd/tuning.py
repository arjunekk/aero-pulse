"""Parameter selection for DBSCAN: k-distance graph + (eps, min_samples) grid search."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from .config import RANDOM_STATE


def k_distances(X: np.ndarray, k: int) -> np.ndarray:
    """Sorted distance of every point to its k-th nearest neighbour *including itself*.

    DBSCAN's ``min_samples`` counts the point itself, so min_samples=k matches this definition.
    """
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    d, _ = nn.kneighbors(X)
    return np.sort(d[:, -1])


def knee_point(y: np.ndarray) -> int:
    """Index of the 'elbow' of a convex increasing curve.

    Normalise the curve to the unit square and take the point farthest below the chord
    joining its endpoints (a light-weight version of the Kneedle idea).
    """
    n = len(y)
    x = np.linspace(0, 1, n)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-12)
    return int(np.argmax(x - yn))


def _quality(X: np.ndarray, labels: np.ndarray, sample_size: int) -> tuple[float, float, float]:
    mask = labels != -1
    if len(np.unique(labels[mask])) < 2:
        return np.nan, np.nan, np.nan
    Xm, lm = X[mask], labels[mask]
    try:
        sil = silhouette_score(Xm, lm, sample_size=min(sample_size, len(Xm)), random_state=RANDOM_STATE)
    except ValueError:  # a subsample can contain a single cluster
        sil = np.nan
    return float(sil), float(davies_bouldin_score(Xm, lm)), float(calinski_harabasz_score(Xm, lm))


def grid_search(
    X: np.ndarray,
    eps_grid: list[float],
    minpts_grid: list[int],
    sample_size: int = 10000,
) -> pd.DataFrame:
    """Fit DBSCAN on every (eps, min_samples) pair and record structure + quality."""
    rows = []
    for m in minpts_grid:
        for eps in eps_grid:
            labels = DBSCAN(eps=eps, min_samples=m, n_jobs=-1).fit_predict(X)
            core = labels[labels != -1]
            n_clusters = len(np.unique(core))
            sil, db, ch = _quality(X, labels, sample_size)
            sizes = np.bincount(core) if n_clusters else np.array([0])
            rows.append(
                dict(
                    eps=eps,
                    min_samples=m,
                    n_clusters=n_clusters,
                    noise_ratio=float((labels == -1).mean()),
                    largest_cluster_share=float(sizes.max() / len(labels)),
                    silhouette=sil,
                    davies_bouldin=db,
                    calinski_harabasz=ch,
                )
            )
    return pd.DataFrame(rows)


def select_params(
    grid: pd.DataFrame,
    min_clusters: int = 2,
    noise_range: tuple[float, float] = (0.005, 0.15),
) -> tuple[pd.Series, str]:
    """Pick the best silhouette among 'sensible' configurations.

    Sensible = at least ``min_clusters`` clusters and a noise share inside ``noise_range``
    (we want *some* anomalies, but "noise" should stay the exception, not the majority).
    Returns the chosen row and a human-readable note on which rule applied.
    """
    ok = grid.dropna(subset=["silhouette"])
    strict = ok[
        (ok.n_clusters >= min_clusters)
        & (ok.noise_ratio >= noise_range[0])
        & (ok.noise_ratio <= noise_range[1])
    ]
    if len(strict):
        return strict.sort_values("silhouette", ascending=False).iloc[0], "constrained"
    if len(ok):
        return ok.sort_values("silhouette", ascending=False).iloc[0], "fallback: no config met the constraints"
    raise ValueError("No (eps, min_samples) pair produced >= 2 clusters; widen the grid.")
