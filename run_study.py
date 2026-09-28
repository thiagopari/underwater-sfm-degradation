"""Study v2 driver: degrade, reconstruct, score. Resumable; one COLMAP job at a time.

    python run_study.py reference        # clear-air replicates (COLMAP run-to-run floor)
    python run_study.py main             # 6 water types x 3 seeds, SIFT, 3 m standoff
    python run_study.py standoff         # 1C and 3C at 1.5 m and 6 m, 2 seeds
    python run_study.py transition       # fill optical depth 2.4-3.2 (3C at 3.5/4 m, 1C at 4.5/5 m)
    python run_study.py aliked           # ALIKED + LightGlue, 6 water types + clear air
    python run_study.py ablation         # colour correction at the failure region

Each finished run appends one JSON record to results/v2/runs.jsonl; runs already
recorded are skipped, so an interrupted stage can simply be restarted.
Needs work/runs/reference_sift_s0 (made by `python run_study.py reference`).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from src.color_correction import sea_thru_estimated, sea_thru_oracle, shades_of_gray
from src.colmap_runner import run_sfm
from src.depth_align import align_all, load_rel_maps
from src.geometry import pose_errors, read_model
from src.optics import WATER_TYPES, Sensor, Water, channel_curves, degrade_image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "work" / "data_1600"
WORK = ROOT / "work"
RESULTS = ROOT / "results" / "v2"
REF_DIR = WORK / "runs" / "reference_sift_s0"
THREADS = 6
MIN_FREE_MEM_PCT = 25


def free_memory_pct() -> int:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
        return int(out.strip().splitlines()[-1].split(":")[-1].strip().rstrip("%"))
    except Exception:
        return 100  # not macOS: don't block


def wait_for_memory():
    while (pct := free_memory_pct()) < MIN_FREE_MEM_PCT:
        print(f"  waiting: free memory {pct}% < {MIN_FREE_MEM_PCT}%", flush=True)
        time.sleep(30)


def done_tags() -> set[str]:
    f = RESULTS / "runs.jsonl"
    if not f.exists():
        return set()
    return {json.loads(line)["tag"] for line in f.read_text().splitlines() if line.strip()}


def record(rec: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


_RANGE_CACHE: dict[float, tuple[dict, dict]] = {}


def ranges_for(standoff: float):
    if standoff not in _RANGE_CACHE:
        ref = read_model(REF_DIR / "sparse" / "0")
        cam = next(iter(ref.cameras.values()))
        rel = load_rel_maps(ROOT / "depth", list(ref.images), (cam.width, cam.height))
        ranges, diag = align_all(ref, rel, standoff_m=standoff)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"depth_alignment_{standoff:g}m.json").write_text(json.dumps(diag, indent=2))
        _RANGE_CACHE[standoff] = (ranges, diag)
    return _RANGE_CACHE[standoff]


def make_set(tag: str, water_type: str | None, standoff: float, seed: int,
             correction: str | None = None) -> tuple[Path, float]:
    """Write the degraded (and optionally corrected) image set; return (dir, mean gain)."""
    out = WORK / "sets" / tag
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if water_type is None:
        for p in sorted(DATA.glob("*.JPG")):
            shutil.copyfile(p, out / p.name)
        return out, 1.0
    ranges, _ = ranges_for(standoff)
    ref = read_model(REF_DIR / "sparse" / "0")
    focal = next(iter(ref.cameras.values())).K[0, 0]
    water, sensor = Water(water_type), Sensor()
    curves = channel_curves(water)
    gains = []
    for k, p in enumerate(sorted(DATA.glob("*.JPG"))):
        rng = np.random.default_rng(10_000 * seed + k)
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        img, g = degrade_image(rgb, ranges[p.name], water, sensor, focal, rng, curves)
        gains.append(g)
        if correction == "shades_of_gray":
            img = (shades_of_gray(img.astype(np.float32) / 255.0) * 255.0 + 0.5).astype(np.uint8)
        elif correction == "sea_thru_estimated":
            img, _ = sea_thru_estimated(img, ranges[p.name])
        elif correction == "sea_thru_oracle":
            img = sea_thru_oracle(img, ranges[p.name], water, g, curves)
        cv2.imwrite(str(out / p.name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, sensor.jpeg_quality])
    return out, float(np.mean(gains))


def run_one(stage: str, water_type: str | None, standoff: float, seed: int,
            features: str = "sift", correction: str | None = None):
    tag = "_".join(str(x) for x in (stage, water_type or "clear", f"{standoff:g}m", features,
                                    correction or "none", f"s{seed}"))
    if tag in done_tags():
        print(f"skip {tag}")
        return
    wait_for_memory()
    t0 = time.time()
    image_dir, gain = make_set(tag, water_type, standoff, seed, correction)
    m = run_sfm(image_dir, WORK / "runs" / tag, features=features, seed=seed, num_threads=THREADS)
    rec = {
        "tag": tag, "stage": stage, "water_type": water_type or "clear", "standoff_m": standoff,
        "seed": seed, "features": features, "correction": correction or "none", "mean_gain": gain,
        "num_registered": m["num_registered"], "num_3d_points": m["num_3d_points"],
        "mean_reproj_error": m["mean_reproj_error"], "num_models": m.get("num_models", 0),
        "mean_features": float(np.mean(m["features_per_image"])) if m["features_per_image"] else 0.0,
    }
    if m.get("model_dir"):
        _, diag = ranges_for(standoff)
        ref = read_model(REF_DIR / "sparse" / "0")
        pe = pose_errors(ref, read_model(Path(m["model_dir"])), metres_per_unit=diag["metres_per_unit"])
        rec.update({k: v for k, v in pe.items() if k != "per_image"})
    rec["runtime_s"] = round(time.time() - t0, 1)
    (WORK / "runs" / tag / "database.db").unlink(missing_ok=True)
    record(rec)
    print(f"done {tag}: reg {rec['num_registered']}/30 pts {rec['num_3d_points']} "
          f"ATE {rec.get('ate_rmse_m')} m ({rec['runtime_s']} s)", flush=True)


STAGES = {
    "reference": lambda: [run_one("reference", None, 3.0, s) for s in (1, 2)],
    "main": lambda: [run_one("main", w, 3.0, s) for s in (0, 1, 2) for w in WATER_TYPES],
    "standoff": lambda: [run_one("standoff", w, d, s) for s in (0, 1) for w in ("1C", "3C") for d in (1.5, 6.0)],
    "transition": lambda: [run_one("transition", w, d, s) for s in (0, 1) for w, d in (("3C", 3.5), ("3C", 4.0), ("1C", 4.5), ("1C", 5.0))],
    "aliked": lambda: [run_one("aliked", w, 3.0, 0, features="aliked_lightglue") for w in (None, *WATER_TYPES)],
}


def ablation(levels: list[str]):
    for s in (0, 1):
        for w in levels:
            for corr in ("shades_of_gray", "sea_thru_estimated", "sea_thru_oracle"):
                run_one("ablation", w, 3.0, s, correction=corr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[*STAGES, "ablation"])
    ap.add_argument("--levels", nargs="+", default=["3C", "5C"], help="water types for the ablation stage")
    args = ap.parse_args()
    if not (REF_DIR / "sparse" / "0").exists():
        print("building reference reconstruction (clear air, seed 0)")
        run_sfm(DATA, REF_DIR, seed=0, num_threads=THREADS)
    if args.stage == "ablation":
        ablation(args.levels)
    else:
        STAGES[args.stage]()
