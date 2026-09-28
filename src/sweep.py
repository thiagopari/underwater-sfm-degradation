"""[v1] Batch-degrade the dataset across a turbidity sweep."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from .jerlov import degrade, DegradeConfig, JERLOV_TYPES, describe


def make_sweep(
    image_dir: Path,
    depth_dir: Path,
    out_root: Path,
    levels: list[tuple[str, float]],
    also_copy_baseline: bool = True,
) -> dict[str, Path]:
    """Generate degraded copies of the input images across a sweep.

    Args:
        image_dir: source images.
        depth_dir: per-image .npy depth caches (from src.depth.batch_estimate_depth).
        out_root:  root of output sweep dirs.
        levels:    list of (jerlov_type, intensity) tuples defining each sweep step.
        also_copy_baseline: if True, also create a level_0_baseline dir with
                            unmodified originals for the COLMAP baseline run.

    Returns dict mapping level tag -> output dir.
    """
    image_dir = Path(image_dir)
    depth_dir = Path(depth_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sources = sorted(image_dir.glob("*.JPG")) + sorted(image_dir.glob("*.jpg"))
    if not sources:
        raise RuntimeError(f"No images in {image_dir}")

    dirs: dict[str, Path] = {}

    if also_copy_baseline:
        d = out_root / "level_00_baseline"
        d.mkdir(exist_ok=True)
        dirs["baseline"] = d
        for src in sources:
            dst = d / src.name
            if not dst.exists():
                shutil.copyfile(src, dst)

    for i, (jt, intensity) in enumerate(levels, start=1):
        tag = f"level_{i:02d}_{jt}_x{intensity:.2f}"
        d = out_root / tag
        d.mkdir(exist_ok=True)
        dirs[tag] = d
        cfg = DegradeConfig(jerlov_type=jt, intensity=intensity)
        for src in sources:
            depth_path = depth_dir / (src.stem + ".npy")
            if not depth_path.exists():
                raise FileNotFoundError(depth_path)
            depth = np.load(depth_path)

            bgr = cv2.imread(str(src))
            if bgr is None:
                raise FileNotFoundError(src)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if depth.shape[:2] != rgb.shape[:2]:
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            out = degrade(rgb, depth, cfg)
            out_bgr = cv2.cvtColor((out * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(d / src.name), out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return dirs


DEFAULT_LEVELS: list[tuple[str, float]] = [
    ("I",   1.0),   # clearest ocean
    ("II",  1.0),   # clear coastal
    ("III", 1.0),   # turbid coastal
    ("1C",  1.0),   # harbor moderate
    ("3C",  1.0),   # harbor very turbid
    ("5C",  1.0),   # harbor extremely turbid
]
