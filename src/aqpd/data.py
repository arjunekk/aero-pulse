"""Dataset loading.

Primary dataset: UCI "Beijing Multi-Site Air-Quality Data" (12 monitoring stations, hourly,
2013-03-01 to 2017-02-28). Files are named ``PRSA_Data_<Station>_<dates>.csv`` and contain
columns ``year, month, day, hour, PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd,
WSPM, station``.

``load_generic`` lets you point the same pipeline at another faculty-approved dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import FEATURES

_TIME_COLS = ["year", "month", "day", "hour"]


def load_beijing(data_dir: str | Path, stations: Iterable[str] | None = None) -> pd.DataFrame:
    """Load and concatenate all ``PRSA_Data_*.csv`` files in ``data_dir``.

    Returns a frame with columns ``datetime, station`` + FEATURES (hourly rows, NaNs kept).
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("PRSA_Data_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No PRSA_Data_*.csv files found in '{data_dir}'. Download the dataset from the UCI "
            "repository (see README) and extract the CSVs there."
        )
    wanted = {s.lower() for s in stations} if stations else None
    frames: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(f)
        missing = [c for c in _TIME_COLS + FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"{f.name} is missing expected columns: {missing}")
        if "station" not in df.columns:
            df["station"] = f.stem.split("_")[2]
        if wanted and str(df["station"].iloc[0]).lower() not in wanted:
            continue
        df["datetime"] = pd.to_datetime(df[_TIME_COLS])
        frames.append(df[["datetime", "station", *FEATURES]])
    if not frames:
        raise ValueError(f"No station files matched {sorted(wanted)} in '{data_dir}'.")
    return pd.concat(frames, ignore_index=True)


def load_generic(
    path: str | Path,
    datetime_col: str,
    rename: dict[str, str] | None = None,
    station_col: str | None = None,
) -> pd.DataFrame:
    """Load any CSV that has a timestamp and the four assigned attributes.

    ``rename`` maps the file's column names to PM2.5 / PM10 / NO2 / TEMP.
    """
    df = pd.read_csv(path)
    if rename:
        df = df.rename(columns=rename)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing attributes after renaming: {missing}")
    out = pd.DataFrame({"datetime": pd.to_datetime(df[datetime_col])})
    out["station"] = df[station_col].astype(str) if station_col else "site"
    for c in FEATURES:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    return out
