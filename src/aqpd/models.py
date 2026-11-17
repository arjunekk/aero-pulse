"""Final DBSCAN fit plus baselines used to put its results in context."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .config import RANDOM_STATE
from .tuning import _quality


def fit_dbscan(X: np.ndarray, eps: float, min_samples: int) -> tuple[np.ndarray, DBSCAN]:
    model = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(X)
    return model.labels_, model


def fit_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)


def fit_hdbscan(X: np.ndarray, min_samples: int, min_cluster_size: int | None = None) -> np.ndarray:
    mcs = min_cluster_size or max(10, int(0.01 * len(X)))
    return HDBSCAN(min_cluster_size=mcs, min_samples=min_samples, copy=True).fit_predict(X)


def outlier_flags_iforest(X: np.ndarray, contamination: float) -> np.ndarray:
    c = float(np.clip(contamination, 0.001, 0.5))
    return IsolationForest(contamination=c, random_state=RANDOM_STATE).fit_predict(X) == -1


def outlier_flags_lof(X: np.ndarray, contamination: float, n_neighbors: int = 20) -> np.ndarray:
    c = float(np.clip(contamination, 0.001, 0.5))
    return LocalOutlierFactor(n_neighbors=n_neighbors, contamination=c).fit_predict(X) == -1


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else float("nan")


def summarize(name: str, X: np.ndarray, labels: np.ndarray, sample_size: int = 10000) -> dict:
    """Comparable summary row: quality is computed on non-noise points only."""
    sil, db, ch = _quality(X, labels, sample_size)
    core = labels[labels != -1]
    return dict(
        model=name,
        n_clusters=int(len(np.unique(core))),
        noise_ratio=float((labels == -1).mean()),
        coverage=float((labels != -1).mean()),
        silhouette=sil,
        davies_bouldin=db,
        calinski_harabasz=ch,
    )
