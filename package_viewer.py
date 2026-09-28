"""Package the interactive viewer (static site, deployed to GitHub Pages).

Reads the seed-0 v2 image sets in work/sets/ and results/v2/runs.jsonl, writes:
    viewer/assets/manifest.json
    viewer/assets/frames/<level>/<frame>.JPG   (downsized to 800 px wide)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

VIEWER_ROOT = Path("viewer")
ASSETS = VIEWER_ROOT / "assets"
FRAMES = ASSETS / "frames"
SETS = Path("work/sets")
RUNS = Path("results/v2/runs.jsonl")
LEVELS = ["clear", "IB", "II", "III", "1C", "3C", "5C"]

MAX_W = 800   # target viewer image width in pixels
JPEG_Q = 82


def _downsize_and_write(src: Path, dst: Path) -> None:
    bgr = cv2.imread(str(src))
    if bgr is None:
        return
    h, w = bgr.shape[:2]
    if w > MAX_W:
        scale = MAX_W / w
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])


def _set_dir(level: str) -> Path:
    return SETS / ("reference_clear_3m_sift_none_s1" if level == "clear" else f"main_{level}_3m_sift_none_s0")


def _metrics() -> dict:
    runs = [json.loads(l) for l in RUNS.read_text().splitlines() if l.strip()]
    out = {}
    for level in LEVELS:
        rs = [r for r in runs if r["water_type"] == level and r["features"] == "sift"
              and r["correction"] == "none" and r["standoff_m"] == 3.0 and r["stage"] in ("main", "reference")]
        if rs:
            mean = lambda k: float(np.mean([r[k] for r in rs if r.get(k) is not None])) if any(r.get(k) is not None for r in rs) else None
            out[level] = {"n": len(rs), **{k: mean(k) for k in ("num_registered", "num_accurate", "ate_rmse_m", "num_3d_points", "mean_gain")}}
    return out


def main():
    frames = sorted(p.name for p in _set_dir("clear").glob("*.JPG"))
    assert frames, "missing work/sets: run run_study.py reference/main first"
    if FRAMES.exists():
        shutil.rmtree(FRAMES)
    for level in LEVELS:
        for frame in frames:
            src = _set_dir(level) / frame
            if src.exists():
                _downsize_and_write(src, FRAMES / level / frame)
    manifest = {"levels": [{"tag": l, "short": l} for l in LEVELS], "frames": frames, "metrics": _metrics()}
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {ASSETS / 'manifest.json'} with {len(LEVELS)} levels x {len(frames)} frames.")


if __name__ == "__main__":
    main()
