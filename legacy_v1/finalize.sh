#!/usr/bin/env bash
# Post-sweep finalization: image metrics, color-correction ablation, plots,
# viewer packaging. Safe to re-run.
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

echo "== computing image quality metrics =="
python -m src.image_metrics sweep > results/v1/image_metrics.json
echo "  -> results/v1/image_metrics.json"

echo "== color-correction ablation =="
python run_correction_ablation.py

echo "== fine-grained sweep near break point =="
python run_fine_sweep.py

echo "== plots =="
python make_plots.py

echo "== viewer package =="
python package_viewer.py

echo "== markdown results table =="
python legacy_v1/results_summary.py | tee results/v1/summary.md

echo
echo "DONE. Ready to commit + push."
