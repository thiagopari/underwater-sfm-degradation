"""Fine-grained intensity sweep near the discovered break point.

The coarse sweep uses one Kd value per Jerlov type. This driver sweeps the
intensity multiplier *within* a chosen type to find the precise turbidity
level at which reconstruction starts to fail. Useful once the coarse sweep
has identified which Jerlov type is on the boundary.
"""

from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # run from repo root: python legacy_v1/<script>.py


import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

from src.jerlov import degrade, DegradeConfig
from src.colmap_runner import run_sfm


DATA = Path("data")
DEPTH = Path("depth")
SWEEP = Path("sweep_fine")
RESULTS = Path("results/v1")

# Coarse sweep showed SIFT is robust even at Jerlov 1C. Zoom in near the
# actual break at 3C -> 5C by pushing intensity beyond the standard Kd values.
JERLOV_TYPE = "3C"
INTENSITIES = [1.25, 1.5, 1.75]


def _degrade_dir(intensity: float) -> Path:
    tag = f"fine_{JERLOV_TYPE}_x{intensity:.2f}"
    out_dir = SWEEP / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if list(out_dir.glob("*.JPG")):
        return out_dir
    cfg = DegradeConfig(jerlov_type=JERLOV_TYPE, intensity=intensity)
    for p in sorted(DATA.glob("*.JPG")):
        depth = np.load(DEPTH / (p.stem + ".npy"))
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if depth.shape[:2] != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        out = degrade(rgb, depth, cfg)
        out_bgr = cv2.cvtColor((out * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / p.name), out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_dir


def main():
    SWEEP.mkdir(exist_ok=True)
    all_metrics = {}
    t0 = time.time()

    for i, intensity in enumerate(INTENSITIES):
        tag = f"fine_{JERLOV_TYPE}_x{intensity:.2f}"
        result_dir = RESULTS / tag
        cache = result_dir / "metrics.json"

        if cache.exists():
            all_metrics[tag] = json.loads(cache.read_text())
            print(f"[{i+1}/{len(INTENSITIES)}] {tag}  (cached)")
            continue

        print(f"\n[{i+1}/{len(INTENSITIES)}] {tag}")
        img_dir = _degrade_dir(intensity)
        t1 = time.time()
        try:
            metrics = run_sfm(image_dir=img_dir, work_dir=result_dir)
        except Exception as e:
            metrics = {"error": str(e)}
        dt = time.time() - t1
        metrics["runtime_s"] = dt
        n_reg = metrics.get("num_registered", 0)
        n_pts = metrics.get("num_3d_points", 0)
        reproj = metrics.get("mean_reproj_error", 0)
        print(f"  registered={n_reg}/{metrics.get('num_input_images', '?')}  points={n_pts}  reproj={reproj:.3f}  ({dt:.1f}s)")
        all_metrics[tag] = metrics

    out_path = RESULTS / "fine_metrics.json"
    out_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nTotal: {time.time()-t0:.1f}s  ->  {out_path}")


if __name__ == "__main__":
    main()
