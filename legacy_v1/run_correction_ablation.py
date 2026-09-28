"""Color-correction ablation:

For each degraded sweep level, apply two corrections and re-run COLMAP:
  1. Shades-of-Gray white balance (naive, per-channel gain, no depth).
  2. Depth-aware Sea-thru-style inverse (uses the *known* Kd + depth, so
     it's an oracle test of "can perfect physics-based correction recover
     reconstruction quality?").

Compares metrics against the uncorrected sweep to isolate how much of the
SfM degradation is driven by color cast (recoverable by naive WB) vs by
depth-dependent contrast loss + backscatter (needs Sea-thru-style depth
information).
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

from src.color_correction import shades_of_gray, sea_thru_inverse
from src.colmap_runner import run_sfm


SWEEP = Path("sweep")
CORRECTED = Path("sweep_corrected")
RESULTS = Path("results/v1")
DEPTH = Path("depth")

# Only run on levels where the coarse sweep showed degradation.
# Format: (level_dir_name, jerlov_type, intensity)
LEVELS_TO_CORRECT = [
    ("level_05_3C_x1.00", "3C", 1.0),
    ("level_06_5C_x1.00", "5C", 1.0),
]

METHODS = ["shades_of_gray", "sea_thru"]


def _correct_dir(src_dir: Path, dst_dir: Path, method: str, jerlov_type: str, intensity: float) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_dir.glob("*.JPG")) + sorted(src_dir.glob("*.jpg")):
        out_p = dst_dir / p.name
        if out_p.exists():
            continue
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if method == "shades_of_gray":
            corr = shades_of_gray(rgb)
        elif method == "sea_thru":
            depth = np.load(DEPTH / (p.stem + ".npy"))
            if depth.shape[:2] != rgb.shape[:2]:
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            corr = sea_thru_inverse(rgb, depth, jerlov_type=jerlov_type, intensity=intensity)
        else:
            corr = rgb

        out_bgr = cv2.cvtColor((corr * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_p), out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    CORRECTED.mkdir(exist_ok=True)
    all_metrics: dict = {}
    t0 = time.time()

    total = len(LEVELS_TO_CORRECT) * len(METHODS)
    i = 0
    for level_name, jt, intensity in LEVELS_TO_CORRECT:
        src_dir = SWEEP / level_name
        if not src_dir.exists():
            print(f"skip {level_name}: source dir missing")
            continue
        for method in METHODS:
            i += 1
            corr_tag = f"{level_name}__{method}"
            corr_dir = CORRECTED / corr_tag
            result_dir = RESULTS / (corr_tag + "_run")
            cache = result_dir / "metrics.json"

            if cache.exists():
                all_metrics[corr_tag] = json.loads(cache.read_text())
                print(f"[{i}/{total}] {corr_tag} (cached)")
                continue

            print(f"\n[{i}/{total}] {corr_tag}")
            print(f"  applying {method} correction...")
            _correct_dir(src_dir, corr_dir, method, jt, intensity)

            t1 = time.time()
            print(f"  running COLMAP...")
            try:
                metrics = run_sfm(image_dir=corr_dir, work_dir=result_dir)
            except Exception as e:
                metrics = {"error": str(e)}
            dt = time.time() - t1
            metrics["runtime_s"] = dt
            metrics["correction_method"] = method
            metrics["source_level"] = level_name
            n_reg = metrics.get("num_registered", 0)
            n_pts = metrics.get("num_3d_points", 0)
            reproj = metrics.get("mean_reproj_error", 0)
            print(f"  registered={n_reg}/{metrics.get('num_input_images', '?')}  points={n_pts}  reproj={reproj:.3f}  ({dt:.1f}s)")
            all_metrics[corr_tag] = metrics

    out_path = RESULTS / "correction_metrics.json"
    out_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    print(f"Aggregated metrics: {out_path}")


if __name__ == "__main__":
    main()
