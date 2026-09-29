.PHONY: help venv data depth prep reference main standoff aliked ablation study plots viewer test all clean legacy-v1

PY := .venv/bin/python
PIP := .venv/bin/pip

help:
	@echo "Study v2 (see README):"
	@echo "  venv       create virtualenv with pinned requirements"
	@echo "  data       download south-building, extract + verify the 30 frames"
	@echo "  depth      DPT depth maps (cached in depth/, ~40 s)"
	@echo "  prep       1600 px working copies of the frames"
	@echo "  reference  clear-air reference + replicates (~8 min)"
	@echo "  main       6 water types x 3 seeds, SIFT (~45 min)"
	@echo "  standoff   1C and 3C at 1.5 m and 6 m (~20 min)"
	@echo "  transition optical depth 2.4-3.2 fill-in (~20 min)"
	@echo "  aliked     ALIKED + LightGlue across water types (~25 min)"
	@echo "  ablation   colour-correction ablation (~30 min)"
	@echo "  sensitivity veil x10/x30 and photon budget x4/x0.25 at the transition (~20 min)"
	@echo "  study      all stages above, in order (resumable)"
	@echo "  plots      results/v2/summary.md + figures/v2/"
	@echo "  viewer     package viewer/ for GitHub Pages"
	@echo "  test       unit tests"
	@echo "  all        data depth prep study plots viewer"
	@echo "  legacy-v1  rerun the original v1 study (kept for comparison)"

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PY) scripts/get_data.py

depth:
	$(PY) -m src.depth data depth

prep:
	$(PY) -c "import cv2, pathlib; o = pathlib.Path('work/data_1600'); o.mkdir(parents=True, exist_ok=True); [cv2.imwrite(str(o / p.name), cv2.resize(cv2.imread(str(p)), (1600, 1200), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95]) for p in sorted(pathlib.Path('data').glob('*.JPG'))]"

reference main standoff transition aliked sensitivity:
	$(PY) run_study.py $@

ablation:
	$(PY) run_study.py ablation --levels 3C 5C

study: reference main standoff transition aliked ablation sensitivity

plots:
	$(PY) make_plots_v2.py

viewer:
	$(PY) package_viewer.py

test:
	$(PY) -m pytest -q

all: data depth prep study plots viewer

clean:
	rm -rf work/sets work/runs

legacy-v1:
	$(PY) -c "from pathlib import Path; from src.sweep import make_sweep, DEFAULT_LEVELS; make_sweep(Path('data'), Path('depth'), Path('sweep'), DEFAULT_LEVELS)"
	$(PY) legacy_v1/run_colmap_sweep.py
	$(PY) legacy_v1/run_correction_ablation.py
	$(PY) legacy_v1/make_plots.py
