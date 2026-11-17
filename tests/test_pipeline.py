import json

import numpy as np
import pandas as pd
import pytest

from aqpd.config import FEATURES
from aqpd.data import load_beijing
from aqpd.pipeline import run
from aqpd.preprocess import preprocess
from aqpd.tuning import k_distances, knee_point, select_params


def test_loader_reads_all_stations(data_dir):
    df = load_beijing(data_dir)
    assert set(df["station"]) == {"Alpha", "Beta"}
    assert {"datetime", "station", *FEATURES} <= set(df.columns)


def test_loader_station_filter_and_missing_dir(data_dir, tmp_path):
    assert set(load_beijing(data_dir, ["alpha"])["station"]) == {"Alpha"}
    with pytest.raises(FileNotFoundError):
        load_beijing(tmp_path / "nope")


def test_preprocess_daily_and_scaling(data_dir):
    df = load_beijing(data_dir)
    clean, X, _, rep = preprocess(df, resample="D", min_obs=18)
    assert not clean[FEATURES].isna().any().any()
    assert X.shape == (len(clean), 4)
    assert np.allclose(X.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(X.std(axis=0), 1, atol=1e-8)
    assert rep.rows_final <= rep.rows_after_dedup


def test_preprocess_sets_negative_pollutants_nan():
    t = pd.date_range("2020-01-01", periods=48, freq="h")
    df = pd.DataFrame({"datetime": t, "station": "s", "PM2.5": 10.0, "PM10": 20.0, "NO2": 30.0, "TEMP": 5.0})
    df.loc[3, "PM2.5"] = -5.0
    _, _, _, rep = preprocess(df, resample=None)
    assert rep.negative_values_set_nan == 1
    assert rep.rows_final == 47


def test_knee_point_finds_bend():
    y = np.concatenate([np.linspace(0, 1, 90), np.linspace(1, 20, 10)])
    assert knee_point(y) >= 80


def test_k_distance_sorted():
    X = np.random.default_rng(0).normal(size=(200, 4))
    kd = k_distances(X, 5)
    assert (np.diff(kd) >= 0).all()


def test_select_params_respects_noise_constraint():
    g = pd.DataFrame({
        "eps": [0.3, 0.5], "min_samples": [5, 5], "n_clusters": [2, 2], "n_major_clusters": [2, 2],
        "noise_ratio": [0.5, 0.05], "silhouette": [0.9, 0.4],
    })
    best, rule = select_params(g)
    assert best["eps"] == 0.5 and rule.startswith("constrained")


def test_end_to_end(data_dir, tmp_path):
    out = tmp_path / "out"
    m = run(data_dir, out, eps_grid=[0.3, 0.5, 0.8, 1.2], minpts_grid=[5, 10, 20])
    assert (out / "metrics.json").exists()
    assert json.loads((out / "metrics.json").read_text())["dbscan"]["n_total"] > 0
    for f in ["01_k_distance", "02_grid_search", "03_clusters_pca", "04_cluster_profiles",
              "05_seasonal_composition", "06_noise_timeline"]:
        assert (out / "figures" / f"{f}.png").exists()
    assert m["dbscan"]["n_clusters"] >= 1


def test_select_params_prefers_two_major_clusters():
    g = pd.DataFrame({
        "eps": [0.5, 0.7], "min_samples": [8, 8], "n_clusters": [2, 2], "n_major_clusters": [1, 2],
        "noise_ratio": [0.02, 0.03], "silhouette": [0.9, 0.3],
    })
    best, rule = select_params(g)
    assert best["eps"] == 0.7 and rule.startswith("constrained")


def test_select_params_relaxes_when_no_two_major():
    g = pd.DataFrame({
        "eps": [0.5], "min_samples": [8], "n_clusters": [2], "n_major_clusters": [1],
        "noise_ratio": [0.02], "silhouette": [0.2],
    })
    _, rule = select_params(g)
    assert rule.startswith("relaxed")
