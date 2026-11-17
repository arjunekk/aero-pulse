# Aero-Pulse: Air Quality Pattern Discovery (DBSCAN)


Density-based clustering of air-quality observations (PM2.5, PM10, NO2, temperature) to
find dense pollution regimes and unusual observations (DBSCAN "noise").

## Pipeline

1. **Load** the multi-station hourly data (`aqpd/data.py`).
2. **Preprocess** (`aqpd/preprocess.py`): drop duplicates, negative concentrations to NaN, drop
   incomplete rows, daily means (>= 18 valid hours), `log1p` on pollutants, standardise.
   Statistical outliers are deliberately *kept*: they are what we want to detect.
3. **Tune DBSCAN** (`aqpd/tuning.py`): k-distance knee for eps, then an (eps, min_samples) grid
   scored by silhouette / Davies-Bouldin / Calinski-Harabasz on non-noise points, with a
   constraint that noise stays between 0.5% and 15%.
4. **Compare** against K-Means and HDBSCAN; cross-check the noise points against Isolation
   Forest and Local Outlier Factor (Jaccard overlap).
5. **Interpret** (`aqpd/interpret.py`): cluster profiles in original units, which feature makes
   each noise point anomalous, seasonal and per-station composition.

## Dataset

UCI Machine Learning Repository, *Beijing Multi-Site Air-Quality Data* (12 stations, hourly,
2013-03-01 to 2017-02-28). Download from
<https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data>, extract the
`PRSA_Data_*.csv` files into `data/raw/`. The raw data is not committed.
Check the dataset page for the exact citation and licence before submitting the report.

## Run

```bash
python -m venv .venv
source .venv/Scripts/activate        # Git Bash on Windows
pip install -r requirements.txt
pip install -e .
pytest                                # tests use synthetic data only
python -m aqpd --data-dir data/raw --out results
```

Useful options: `--stations Aotizhongxin Dongsi`, `--resample none` (hourly),
`--scaler robust`, `--eps 0.3 0.5 0.7`, `--min-samples 5 10`.

Outputs: `results/metrics.json`, `results/tables/*.csv`, `results/figures/*.png`.
Every number in the report must come from these files.

## Layout

`src/` also holds earlier exploration scripts (OpenAQ probing/auditing); the DBSCAN pipeline lives in the `src/aqpd/` package.

```
src/aqpd/      config, data, preprocess, tuning, models, interpret, viz, pipeline, CLI
tests/         unit + end-to-end tests on synthetic data
data/raw/      dataset goes here (git-ignored)
results/       generated outputs from your run
```
