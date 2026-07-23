"""Simple color-correction baselines for the ablation study.

These are NOT full Sea-thru reimplementations (Sea-thru requires per-pixel
depth + solving for backscatter coefficients from dark-channel priors).
They are inexpensive baselines a marine-robotics practitioner might try
before reaching for Sea-thru, useful for asking: "does *any* channel-wise
white-balance recover enough color to help SfM?"

The point of the ablation is to establish that the SfM breakdown we observe
is driven by structural information loss (turbidity, contrast, backscatter),
not merely by the color cast — because a naive white balance ONLY fixes the
color cast.
"""

from __future__ import annotations

import numpy as np


def gray_world(rgb: np.ndarray) -> np.ndarray:
    """Classic Gray World white balance.

    Assumes the average color of the scene is neutral gray, and rescales each
    channel so its mean matches the overall mean.
    """
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    means = rgb.reshape(-1, 3).mean(axis=0) + 1e-6
    global_mean = means.mean()
    gains = global_mean / means
    out = rgb * gains[None, None, :]
    return np.clip(out, 0.0, 1.0)


def shades_of_gray(rgb: np.ndarray, p: float = 6.0) -> np.ndarray:
    """Shades-of-Gray (Minkowski-norm) white balance.

    Generalization of Gray World; typically produces better results.
    p=1 -> gray world, p=infinity -> max-RGB. p=6 is a common default.
    """
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    flat = rgb.reshape(-1, 3)
    norms = (np.mean(flat ** p, axis=0)) ** (1.0 / p) + 1e-6
    global_norm = norms.mean()
    gains = global_norm / norms
    out = rgb * gains[None, None, :]
    return np.clip(out, 0.0, 1.0)


def uwcnn_like_stretch(rgb: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Per-channel contrast stretch (percentile clip) — a common baseline
    used by underwater imaging pipelines before deep-learning methods.
    """
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    out = np.empty_like(rgb)
    for c in range(3):
        ch = rgb[..., c]
        p_lo = np.percentile(ch, lo)
        p_hi = np.percentile(ch, hi)
        if p_hi - p_lo < 1e-6:
            out[..., c] = ch
        else:
            out[..., c] = np.clip((ch - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    return out


def sea_thru_inverse(
    rgb_degraded: np.ndarray,
    depth: np.ndarray,
    jerlov_type: str = "1C",
    intensity: float = 1.0,
) -> np.ndarray:
    """Depth-aware inverse of the Jerlov degradation model.

    Given the observed (degraded) RGB image and a per-pixel depth map,
    analytically invert the forward model:

        I = J * exp(-Kd*z) + B_inf * (1 - exp(-Kd*z))
        =>  J = (I - B_inf * (1 - trans)) / trans      where trans = exp(-Kd*z)

    Where trans is very small (deep pixels), the inversion is numerically
    unstable — we clip transmission to a minimum floor. This is a
    simplified Sea-thru: it uses the *known* Kd + veiling color of the
    same water model that was used to synthesize the degradation, so it
    tests whether a *perfect* physics-based correction can recover
    reconstruction quality.
    """
    from .jerlov import JERLOV_TYPES

    props = JERLOV_TYPES[jerlov_type]
    kd = np.asarray(props["kd_rgb"], dtype=np.float32) * float(intensity)
    b_inf = np.asarray(props["b_inf_rgb"], dtype=np.float32)

    z = depth[..., None].astype(np.float32)
    trans = np.exp(-kd[None, None, :] * z)                 # HxWx3
    trans = np.maximum(trans, 0.05)                        # numerical floor
    veiling = b_inf[None, None, :] * (1.0 - trans)         # HxWx3
    recovered = (rgb_degraded - veiling) / trans
    return np.clip(recovered, 0.0, 1.0)


def correct(rgb: np.ndarray, method: str = "gray_world", **kw) -> np.ndarray:
    """Dispatch table. sea_thru requires kw={'depth': ndarray, 'jerlov_type': str, 'intensity': float}."""
    if method == "gray_world":
        return gray_world(rgb)
    if method == "shades_of_gray":
        return shades_of_gray(rgb)
    if method == "stretch":
        return uwcnn_like_stretch(rgb)
    if method == "sea_thru":
        return sea_thru_inverse(rgb, kw["depth"], kw.get("jerlov_type", "1C"), kw.get("intensity", 1.0))
    if method == "none":
        return rgb
    raise ValueError(f"Unknown method: {method}")
