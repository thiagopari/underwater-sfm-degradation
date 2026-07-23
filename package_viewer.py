"""Package the viewer for GitHub Pages deployment.

Reads sweep/ + results/sweep_metrics.json, produces:
    viewer/assets/manifest.json
    viewer/assets/frames/<level_tag>/<frame>.jpg   (downsized to ~800px wide)

The viewer is a static site — after running this, `viewer/` can be served
by any static host (GitHub Pages, netlify, ...).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2


VIEWER_ROOT = Path("viewer")
ASSETS = VIEWER_ROOT / "assets"
FRAMES = ASSETS / "frames"
SWEEP = Path("sweep")
RESULTS = Path("results")

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


def _short_tag(level_dir: str) -> str:
    """level_02_II_x1.00 -> II   ;   level_00_baseline -> baseline"""
    parts = level_dir.split("_")
    if "baseline" in level_dir:
        return "baseline"
    # level_02_II_x1.00 -> ["level","02","II","x1.00"] -> "II"
    return parts[2] if len(parts) > 2 else level_dir


def main():
    assert SWEEP.exists(), f"missing {SWEEP} — run degrade sweep first"
    metrics_path = RESULTS / "sweep_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    # discover levels
    level_dirs = sorted(p for p in SWEEP.iterdir() if p.is_dir() and p.name.startswith("level_"))
    frames = sorted(p.name for p in level_dirs[0].glob("*.JPG"))
    if not frames:
        frames = sorted(p.name for p in level_dirs[0].glob("*.jpg"))

    # copy downsized frames
    if FRAMES.exists():
        shutil.rmtree(FRAMES)
    for level in level_dirs:
        out_dir = FRAMES / level.name
        for frame in frames:
            src = level / frame
            if src.exists():
                _downsize_and_write(src, out_dir / frame)

    # manifest
    manifest = {
        "levels": [
            {"tag": level.name, "short": _short_tag(level.name)}
            for level in level_dirs
        ],
        "frames": frames,
        "metrics": metrics,
    }
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {ASSETS/'manifest.json'} with {len(level_dirs)} levels x {len(frames)} frames.")


if __name__ == "__main__":
    main()
