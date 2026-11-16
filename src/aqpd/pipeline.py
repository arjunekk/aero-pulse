"""End-to-end pipeline: load -> preprocess -> tune -> fit -> compare -> interpret -> save."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn

from . import viz
from .config import FEATURES
from .data import load_beijing
from .interpret import (
    cluster_zscores,
    label_names,
    noise_drivers,
    profile_clusters,
    seasonal_table,
    station_table,
)
from .models import (
    fit_dbscan,
    fit_hdbscan,
    fit_kmeans,
    jaccard,
    outlier_flags_iforest,
    outlier_flags_lof,
    summarize,
)
from .preprocess import preprocess
from .tuning import grid_search, k_distances, knee_point, select_params

DEFAULT_EPS = [round(x, 2) for x in np.arange(0.2, 1.01, 0.1)]
DEFAULT_MINPTS = [5, 8, 10, 15, 20, 30]


def run(
    data_dir: str | Path,
    out_dir: str | Path = "results",
    stations: list[str] | None = None,
    resample: str | None = "D",
    min_obs: int = 18,
    scaler: str = "standard",
    eps_grid: list[float] | None = None,
    minpts_grid: list[int] | None = None,
    k_for_knee: int = 8,
    df: pd.DataFrame | None = None,
) -> dict:
    """Run everything and write tables/figures/metrics under ``out_dir``. Returns the metrics dict."""
    t0 = time.time()
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    raw = df if df is not None else load_beijing(data_dir, stations)
    clean, X, _, rep = preprocess(raw, resample=resample, min_obs=min_obs, scaler=scaler)
    print(f"[data] {rep.rows_raw} raw rows -> {rep.rows_final} rows used ({resample or 'hourly'})")

    # 1) eps heuristic
    kd = k_distances(X, k_for_knee)
    knee = knee_point(kd)
    viz.plot_k_distance(kd, k_for_knee, knee, out / "figures" / "01_k_distance.png")

    # 2) grid search + selection
    grid = grid_search(X, eps_grid or DEFAULT_EPS, minpts_grid or DEFAULT_MINPTS)
    grid.to_csv(out / "tables" / "grid_search.csv", index=False)
    best, rule = select_params(grid)
    eps, m = float(best["eps"]), int(best["min_samples"])
    viz.plot_grid(grid, best, out / "figures" / "02_grid_search.png")
    print(f"[tune] selected eps={eps}, min_samples={m} ({rule})")

    # 3) final DBSCAN
    labels, _ = fit_dbscan(X, eps, m)
    n_clusters = int(len(np.unique(labels[labels != -1])))
    noise_ratio = float((labels == -1).mean())
    print(f"[dbscan] {n_clusters} clusters, noise {100 * noise_ratio:.2f}%")

    # 4) baselines / anomaly cross-checks
    comp = [summarize("DBSCAN", X, labels)]
    k = max(2, n_clusters)
    comp.append(summarize(f"KMeans (k={k})", X, fit_kmeans(X, k)))
    comp.append(summarize("HDBSCAN", X, fit_hdbscan(X, min_samples=m)))
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(out / "tables" / "model_comparison.csv", index=False)

    noise = labels == -1
    iso = outlier_flags_iforest(X, noise_ratio)
    lof = outlier_flags_lof(X, noise_ratio)
    agreement = {
        "jaccard_dbscan_vs_isolation_forest": jaccard(noise, iso),
        "jaccard_dbscan_vs_lof": jaccard(noise, lof),
        "jaccard_isolation_forest_vs_lof": jaccard(iso, lof),
    }

    # 5) interpretation
    prof = profile_clusters(clean, labels)
    prof.to_csv(out / "tables" / "cluster_profiles.csv")
    drivers = noise_drivers(X, labels)
    drivers.to_csv(out / "tables" / "noise_drivers.csv", index=False)
    seasons = seasonal_table(clean, labels)
    seasons.to_csv(out / "tables" / "seasonal_composition.csv")
    station_table(clean, labels).to_csv(out / "tables" / "station_composition.csv")
    z = cluster_zscores(X, labels)

    labelled = clean.copy()
    labelled["cluster"] = label_names(labels).to_numpy()
    labelled.to_csv(out / "tables" / "labelled_observations.csv", index=False)

    viz.plot_pca(X, labels, out / "figures" / "03_clusters_pca.png",
                 f"DBSCAN clusters (eps={eps:g}, min_samples={m}) in PCA space")
    viz.plot_profiles(z, out / "figures" / "04_cluster_profiles.png")
    viz.plot_seasons(seasons, out / "figures" / "05_seasonal_composition.png")
    viz.plot_noise_timeline(clean, labels, out / "figures" / "06_noise_timeline.png")

    metrics = {
        "preprocessing": rep.to_dict(),
        "k_distance": {"k": k_for_knee, "knee_eps": float(kd[knee])},
        "selected": {"eps": eps, "min_samples": m, "rule": rule,
                     **{c: (None if pd.isna(best[c]) else float(best[c]))
                        for c in ["silhouette", "davies_bouldin", "calinski_harabasz"]}},
        "dbscan": {"n_clusters": n_clusters, "noise_ratio": noise_ratio,
                   "n_noise": int(noise.sum()), "n_total": int(len(labels))},
        "model_comparison": comp_df.to_dict(orient="records"),
        "anomaly_agreement": agreement,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                        "matplotlib": matplotlib.__version__},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"[done] outputs in {out}/ ({metrics['runtime_seconds']}s)")
    return metrics
