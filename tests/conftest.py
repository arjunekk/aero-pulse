"""Synthetic PRSA-format data used ONLY to test the code. Never use it for report results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic_station(station: str, days: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = pd.date_range("2015-01-01", periods=days * 24, freq="h")
    doy = t.dayofyear.to_numpy()
    season = np.cos(2 * np.pi * (doy - 15) / 365)  # +1 in mid-winter
    temp = 14 - 16 * season + rng.normal(0, 3, len(t))
    base = np.exp(4.0 + 0.5 * season + rng.normal(0, 0.35, len(t)))
    pm25 = base * rng.lognormal(0, 0.1, len(t))
    pm10 = pm25 * 1.4 + rng.normal(0, 5, len(t))
    no2 = 40 + 0.15 * pm25 + rng.normal(0, 6, len(t))
    df = pd.DataFrame({
        "year": t.year, "month": t.month, "day": t.day, "hour": t.hour,
        "PM2.5": pm25, "PM10": pm10.clip(1), "NO2": no2.clip(1), "TEMP": temp, "station": station,
    })
    # inject a few extreme episodes and missing values
    spike = rng.choice(len(df), size=60, replace=False)
    df.loc[spike, ["PM2.5", "PM10"]] *= 6
    miss = rng.choice(len(df), size=300, replace=False)
    df.loc[miss, "PM2.5"] = np.nan
    return df


@pytest.fixture()
def data_dir(tmp_path):
    for i, s in enumerate(["Alpha", "Beta"]):
        make_synthetic_station(s, seed=i).to_csv(
            tmp_path / f"PRSA_Data_{s}_20150101-20160204.csv", index=False
        )
    return tmp_path
