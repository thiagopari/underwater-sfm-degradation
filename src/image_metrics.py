"""Image-quality metrics that help explain SfM behavior downstream.

SIFT is intensity-based and invariant to smooth color shifts; what really
kills feature matching is loss of local contrast and edge sharpness. These
per-image metrics let us correlate 'how bad does the image look' with
'how bad does the SfM get'.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def rms_contrast(gray: np.ndarray) -> float:
    """Root-mean-square contrast (std of intensity)."""
    g = gray.astype(np.float32)
    return float(g.std())


def gradient_magnitude(gray: np.ndarray) -> float:
    """Mean magnitude of the Sobel gradient — proxy for edge density."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return float(mag.mean())


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of Laplacian — classic sharpness metric."""
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(lap.var())


def compute_folder_metrics(image_dir: Path) -> dict:
    image_dir = Path(image_dir)
    files = sorted(image_dir.glob("*.JPG")) + sorted(image_dir.glob("*.jpg"))
    rows = []
    for p in files:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rows.append({
            "rms_contrast": rms_contrast(gray),
            "gradient_mag": gradient_magnitude(gray),
            "laplacian_var": laplacian_variance(gray),
        })
    if not rows:
        return {"n": 0}
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg["n"] = len(rows)
    return agg


def sweep_folder_metrics(sweep_root: Path) -> dict[str, dict]:
    """Compute image metrics for every level_ subdirectory."""
    sweep_root = Path(sweep_root)
    out = {}
    for level in sorted(sweep_root.iterdir()):
        if level.is_dir() and level.name.startswith("level_"):
            out[level.name] = compute_folder_metrics(level)
    return out


if __name__ == "__main__":
    import json
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "sweep")
    print(json.dumps(sweep_folder_metrics(root), indent=2))
