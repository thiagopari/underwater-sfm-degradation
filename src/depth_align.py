"""Metric, multi-view-consistent range maps from DPT + the clear-air COLMAP model.

v1 min-max normalised every DPT map to 0.5-5 m independently, so the same wall
had a different range in every view, and it treated DPT's inverse depth as if
it were linear depth. Here each DPT map is fitted to the sparse depths of the
reference reconstruction (scale + shift in inverse depth, the usual MiDaS/DPT
alignment), and one global scale sets the median scene standoff. The result is
a line-of-sight range map that agrees across views up to DPT's own error.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .geometry import Model

# v1 cached maps store depth = 0.5 + 4.5 * (1 - rel), rel = min-max-normalised DPT output
_V1_MIN, _V1_SPAN = 0.5, 4.5


def relative_disparity_from_v1_cache(depth_v1: np.ndarray) -> np.ndarray:
    """Recover DPT's normalised output (larger = closer) from a v1 cached map."""
    return 1.0 - (depth_v1 - _V1_MIN) / _V1_SPAN


def fit_inverse_depth(rel: np.ndarray, inv_depth: np.ndarray, iters: int = 500,
                      tol: float = 0.1, seed: int = 0) -> tuple[float, float, np.ndarray]:
    """Robustly fit inv_depth ~= s * rel + o (RANSAC on 2-point samples, LS refit).

    tol is the inlier threshold as a fraction of the median inverse depth.
    Returns (s, o, inlier_mask).
    """
    rng = np.random.default_rng(seed)
    thr = tol * float(np.median(inv_depth))
    best = np.zeros(len(rel), bool)
    for _ in range(iters):
        i, j = rng.choice(len(rel), 2, replace=False)
        if abs(rel[i] - rel[j]) < 1e-6:
            continue
        s = (inv_depth[i] - inv_depth[j]) / (rel[i] - rel[j])
        o = inv_depth[i] - s * rel[i]
        inl = np.abs(s * rel + o - inv_depth) < thr
        if inl.sum() > best.sum():
            best = inl
    A = np.stack([rel[best], np.ones(best.sum())], 1)
    s, o = np.linalg.lstsq(A, inv_depth[best], rcond=None)[0]
    return float(s), float(o), best


def align_all(ref: Model, rel_maps: dict[str, np.ndarray], standoff_m: float,
              far_factor: float = 20.0) -> tuple[dict[str, np.ndarray], dict]:
    """Return per-image metric range maps (metres) and alignment diagnostics.

    rel_maps: image name -> HxW DPT relative disparity at the reference image size.
    standoff_m: the median scene depth is scaled to this value.
    far_factor: pixels DPT puts beyond far_factor x the median depth (e.g. sky)
                are clamped there; they end up fully veiled either way.
    """
    fits, med_depths = {}, []
    for name, img in ref.images.items():
        if name not in rel_maps or len(img.point_ids) < 20:
            continue
        X = np.stack([ref.points[p] for p in img.point_ids if p in ref.points])
        xy = np.stack([xy for xy, p in zip(img.xy, img.point_ids) if p in ref.points])
        z = (img.R @ X.T).T[:, 2] + img.t[2]
        ok = z > 0
        rel = rel_maps[name]
        u = np.clip(np.round(xy[ok, 0]).astype(int), 0, rel.shape[1] - 1)
        v = np.clip(np.round(xy[ok, 1]).astype(int), 0, rel.shape[0] - 1)
        s, o, inl = fit_inverse_depth(rel[v, u], 1.0 / z[ok])
        pred = 1.0 / np.clip(s * rel[v, u] + o, 1e-9, None)
        fits[name] = {"s": s, "o": o, "n_points": int(ok.sum()), "inlier_frac": float(inl.mean()),
                      "median_abs_rel_err": float(np.median(np.abs(pred[inl] - z[ok][inl]) / z[ok][inl]))}
        med_depths.append(float(np.median(z[ok])))
    scale = standoff_m / float(np.median(med_depths))   # metres per reference unit
    ranges = {}
    for name, f in fits.items():
        img = ref.images[name]
        cam = ref.cameras[img.camera_id]
        K = cam.K
        rel = rel_maps[name]
        z_max = far_factor * float(np.median(med_depths))
        inv = np.maximum(f["s"] * rel + f["o"], 1.0 / z_max)
        z = 1.0 / inv * scale
        h, w = rel.shape
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        ray = np.sqrt(((uu - K[0, 2]) / K[0, 0]) ** 2 + ((vv - K[1, 2]) / K[1, 1]) ** 2 + 1.0)
        ranges[name] = (z * ray).astype(np.float32)
    diag = {"metres_per_unit": scale, "standoff_m": standoff_m, "per_image": fits,
            "median_inlier_frac": float(np.median([f["inlier_frac"] for f in fits.values()])),
            "median_abs_rel_err": float(np.median([f["median_abs_rel_err"] for f in fits.values()]))}
    return ranges, diag


def load_rel_maps(depth_dir: Path, names: list[str], size: tuple[int, int]) -> dict[str, np.ndarray]:
    """Load v1 cached DPT maps and convert to relative disparity at size (w, h)."""
    out = {}
    for n in names:
        d = np.load(Path(depth_dir) / (Path(n).stem + ".npy"))
        out[n] = cv2.resize(relative_disparity_from_v1_cache(d), size, interpolation=cv2.INTER_LINEAR)
    return out
