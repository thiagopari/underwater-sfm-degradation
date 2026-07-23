"""Monocular depth estimation via HuggingFace transformers (DPT).

Returns a per-pixel depth map rescaled to a plausible underwater standoff
range (default 0.5m to 5m) so downstream Jerlov degradation produces visible
effects across the image.

The absolute scale is not calibrated; this study is about *relative* depth
structure driving the degradation, not metric reconstruction. The COLMAP
baseline reconstruction (level_00) serves as our geometric ground truth
downstream.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


_MODEL_CACHE: dict = {}

# small (~85MB), fast, works fine on Apple Silicon MPS
DEFAULT_MODEL = "Intel/dpt-swinv2-tiny-256"


def _load(model_name: str = DEFAULT_MODEL):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
    pipe = pipeline("depth-estimation", model=model_name, device=device)
    _MODEL_CACHE[model_name] = pipe
    return pipe


def estimate_depth(
    bgr: np.ndarray,
    depth_min_m: float = 0.5,
    depth_max_m: float = 5.0,
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """Estimate a depth map from a BGR image.

    Args:
        bgr:          HxWx3 uint8, BGR (cv2 convention).
        depth_min_m:  Minimum output depth after rescaling.
        depth_max_m:  Maximum output depth after rescaling.

    Returns:
        HxW float32 depth map in meters (linearly rescaled to the given range).
    """
    pipe = _load(model_name)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    result = pipe(pil)
    # result["predicted_depth"] is a torch tensor (relative depth, larger = closer for DPT-swinv2)
    depth_t = result["predicted_depth"]
    if depth_t.ndim == 3:
        depth_t = depth_t.squeeze(0)
    depth_rel = depth_t.detach().cpu().numpy().astype(np.float32)
    # Resize to full image size
    depth_rel = cv2.resize(depth_rel, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)

    dmin, dmax = float(depth_rel.min()), float(depth_rel.max())
    if dmax - dmin < 1e-6:
        return np.full(rgb.shape[:2], 0.5 * (depth_min_m + depth_max_m), dtype=np.float32)

    # DPT-swinv2 tends to output larger = closer; invert to depth semantics
    depth_norm = 1.0 - (depth_rel - dmin) / (dmax - dmin)   # 0=nearest, 1=farthest
    depth_m = depth_min_m + (depth_max_m - depth_min_m) * depth_norm
    return depth_m.astype(np.float32)


def batch_estimate_depth(
    image_paths: list[Path],
    cache_dir: Path,
    force: bool = False,
    **kwargs,
) -> dict[str, Path]:
    """Estimate + cache depth for a batch of images. Returns {stem: cache_path}."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for p in image_paths:
        p = Path(p)
        cache_path = cache_dir / (p.stem + ".npy")
        if not force and cache_path.exists():
            out[p.stem] = cache_path
            continue
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise FileNotFoundError(p)
        depth = estimate_depth(bgr, **kwargs)
        np.save(cache_path, depth)
        out[p.stem] = cache_path
    return out


if __name__ == "__main__":
    import sys
    import time
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    cache_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "depth")
    images = sorted(data_dir.glob("*.JPG")) + sorted(data_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images found in {data_dir}")
    print(f"Estimating depth for {len(images)} images -> {cache_dir}")
    t0 = time.time()
    mapping = batch_estimate_depth(images, cache_dir)
    print(f"Done in {time.time() - t0:.1f}s ({len(mapping)} maps)")
