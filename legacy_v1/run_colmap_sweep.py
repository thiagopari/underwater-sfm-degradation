"""Driver: run COLMAP SfM on every sweep level and aggregate metrics."""

from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # run from repo root: python legacy_v1/<script>.py


import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.colmap_runner import run_sfm


def main():
    sweep_root = Path("sweep")
    results_root = Path("results/v1")
    results_root.mkdir(exist_ok=True)

    levels = sorted(p for p in sweep_root.iterdir() if p.is_dir() and p.name.startswith("level_"))
    print(f"Running COLMAP on {len(levels)} levels")

    all_metrics = {}
    # Reuse existing metrics.json files where present (skip already-run levels)
    for level in levels:
        m_path = results_root / level.name / "metrics.json"
        if m_path.exists():
            try:
                all_metrics[level.name] = json.loads(m_path.read_text())
            except Exception:
                pass

    t0 = time.time()
    for i, level in enumerate(levels):
        tag = level.name
        if tag in all_metrics and all_metrics[tag].get("num_input_images", 0) > 0:
            m = all_metrics[tag]
            print(f"\n[{i+1}/{len(levels)}] {tag}  (cached)")
            print(f"  registered={m.get('num_registered',0)}/{m.get('num_input_images','?')}  points={m.get('num_3d_points',0)}  reproj={m.get('mean_reproj_error',0):.3f}")
            continue
        t1 = time.time()
        print(f"\n[{i+1}/{len(levels)}] {tag}", flush=True)
        try:
            metrics = run_sfm(
                image_dir=level,
                work_dir=results_root / tag,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            metrics = {"error": str(e)}
        dt = time.time() - t1
        metrics["runtime_s"] = dt
        n_reg = metrics.get("num_registered", 0)
        n_pts = metrics.get("num_3d_points", 0)
        reproj = metrics.get("mean_reproj_error", 0)
        print(f"  registered={n_reg}/{metrics.get('num_input_images', '?')}  points={n_pts}  reproj={reproj:.3f}  ({dt:.1f}s)")
        all_metrics[tag] = metrics

    (results_root / "sweep_metrics.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    print(f"Aggregated metrics: {results_root / 'sweep_metrics.json'}")


if __name__ == "__main__":
    main()
