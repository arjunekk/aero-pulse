"""CLI:  python -m aqpd --data-dir data/raw --out results"""

from __future__ import annotations

import argparse

from .pipeline import run


def main() -> None:
    p = argparse.ArgumentParser(description="Air Quality Pattern Discovery (DBSCAN)")
    p.add_argument("--data-dir", default="data/raw", help="folder with PRSA_Data_*.csv files")
    p.add_argument("--out", default="results", help="output folder for tables/figures/metrics")
    p.add_argument("--stations", nargs="*", default=None, help="subset of stations (default: all)")
    p.add_argument("--resample", default="D", help="pandas offset alias (D=daily); 'none' for hourly")
    p.add_argument("--min-obs", type=int, default=18, help="min valid readings per bin (daily: 18/24)")
    p.add_argument("--scaler", choices=["standard", "robust"], default="standard")
    p.add_argument("--eps", nargs="*", type=float, default=None, help="eps grid override")
    p.add_argument("--min-samples", nargs="*", type=int, default=None, help="min_samples grid override")
    a = p.parse_args()
    run(
        a.data_dir, a.out, a.stations,
        None if a.resample.lower() == "none" else a.resample,
        a.min_obs, a.scaler, a.eps, a.min_samples,
    )


if __name__ == "__main__":
    main()
