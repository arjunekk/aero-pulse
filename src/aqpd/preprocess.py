"""Cleaning, aggregation, transformation and scaling.

Design decisions (each one is a viva question):

* We do NOT remove statistical outliers. Extreme pollution episodes are exactly what the
  project wants DBSCAN to flag, so removing them beforehand would destroy the finding.
  We only remove physically impossible values (negative concentrations).
* Rows with missing values are dropped (not imputed): imputing would invent observations
  and bias density estimates, which DBSCAN is directly sensitive to.
* Optional temporal aggregation (daily mean, requiring a minimum number of valid hourly
  readings) reduces autocorrelation between consecutive hours and keeps DBSCAN tractable.
* log1p on pollutants: concentrations are right-skewed (lognormal-like); in raw units a few
  huge values would dominate Euclidean distance and the eps-neighbourhood would be useless.
* Standardisation: DBSCAN is distance based, so all features must be on a comparable scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from .config import FEATURES, POLLUTANTS


@dataclass
class PreprocessReport:
    rows_raw: int
    rows_after_dedup: int
    negative_values_set_nan: int
    missing_cells_before: dict
    resample: str | None
    rows_after_aggregation: int
    rows_final: int
    stations: list
    date_min: str
    date_max: str

    def to_dict(self) -> dict:
        return asdict(self)


def preprocess(
    df: pd.DataFrame,
    resample: str | None = "D",
    min_obs: int = 18,
    log_pollutants: bool = True,
    scaler: str = "standard",
) -> tuple[pd.DataFrame, np.ndarray, object, PreprocessReport]:
    """Return ``(clean_df, X_scaled, fitted_scaler, report)``.

    ``clean_df`` keeps original units (plus datetime, station) so clusters can be interpreted
    in real units; ``X_scaled`` is what the algorithms see.
    """
    rows_raw = len(df)
    df = df.drop_duplicates(subset=["station", "datetime"]).sort_values(["station", "datetime"])
    rows_dedup = len(df)

    df = df.copy()
    neg = int(df[POLLUTANTS].lt(0).sum().sum())
    for c in POLLUTANTS:
        df.loc[df[c] < 0, c] = np.nan
    missing_before = {c: int(df[c].isna().sum()) for c in FEATURES}

    if resample:
        grp = df.set_index("datetime").groupby("station")[FEATURES]
        means = grp.resample(resample).mean()
        counts = grp.resample(resample).count()
        keep = (counts >= min_obs).all(axis=1)
        out = means[keep].dropna().reset_index()
    else:
        out = df.dropna(subset=FEATURES)
    rows_agg = len(out)
    out = out.reset_index(drop=True)

    X = out[FEATURES].to_numpy(dtype=float).copy()
    if log_pollutants:
        idx = [FEATURES.index(c) for c in POLLUTANTS]
        X[:, idx] = np.log1p(X[:, idx])
    sc = RobustScaler() if scaler == "robust" else StandardScaler()
    X_scaled = sc.fit_transform(X)

    report = PreprocessReport(
        rows_raw=rows_raw,
        rows_after_dedup=rows_dedup,
        negative_values_set_nan=neg,
        missing_cells_before=missing_before,
        resample=resample,
        rows_after_aggregation=rows_agg,
        rows_final=len(out),
        stations=sorted(out["station"].unique().tolist()),
        date_min=str(out["datetime"].min()),
        date_max=str(out["datetime"].max()),
    )
    return out, X_scaled, sc, report
