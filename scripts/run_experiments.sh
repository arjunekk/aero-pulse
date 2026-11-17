#!/usr/bin/env bash
# Runs the three experiments used in the report. Run from the project root with the venv active.
set -euo pipefail
python -m aqpd --data-dir data/raw --out results/pooled
python -m aqpd --data-dir data/raw --stations Dongsi --out results/dongsi
python -m aqpd --data-dir data/raw --stations Dingling --out results/dingling
echo "All experiments finished. Outputs are in results/pooled, results/dongsi, results/dingling."
