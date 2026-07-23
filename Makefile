.PHONY: help venv depth sweep colmap ablation plots viewer all clean

PY := .venv/bin/python
PIP := .venv/bin/pip

help:
	@echo "Targets:"
	@echo "  venv     - create virtualenv and install requirements"
	@echo "  depth    - estimate per-frame depth (~40s)"
	@echo "  sweep    - generate degraded images across turbidity levels (~1min)"
	@echo "  colmap   - run COLMAP SfM on every sweep level (~15-20min CPU)"
	@echo "  ablation - color-correction ablation (~15min CPU)"
	@echo "  plots    - render figures from metrics JSON"
	@echo "  viewer   - package interactive viewer for GitHub Pages"
	@echo "  all      - everything end-to-end"
	@echo "  clean    - remove intermediate outputs (keeps figures + metrics JSON)"

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

depth:
	$(PY) -c "from pathlib import Path; from src.depth import batch_estimate_depth; batch_estimate_depth(sorted(Path('data').glob('*.JPG')), Path('depth'))"

sweep:
	$(PY) -c "from pathlib import Path; from src.sweep import make_sweep, DEFAULT_LEVELS; make_sweep(Path('data'), Path('depth'), Path('sweep'), DEFAULT_LEVELS)"

colmap:
	$(PY) run_colmap_sweep.py

ablation:
	$(PY) run_correction_ablation.py

plots:
	$(PY) make_plots.py

viewer:
	$(PY) package_viewer.py

all: depth sweep colmap ablation plots viewer

clean:
	rm -rf sweep/ sweep_corrected/ depth/ results/level_*/ results/sweep_log.txt
