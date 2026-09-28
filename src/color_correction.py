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


# --------------------------------------------------------------------------
# Study v2: corrections for images produced by src.optics
# --------------------------------------------------------------------------

def _fit_backscatter(r: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y ~= B_inf * (1 - exp(-beta r)) by grid search over beta."""
    best = (np.inf, 0.0, 1.0)
    for beta in np.geomspace(0.1, 5.0, 200):  # physical range for coastal/oceanic water
        f = 1.0 - np.exp(-beta * r)
        b_inf = float((y * f).sum() / max((f * f).sum(), 1e-12))
        err = float(((y - b_inf * f) ** 2).sum())
        if err < best[0]:
            best = (err, b_inf, beta)
    return best[1], best[2]


def sea_thru_estimated(rgb_u8: np.ndarray, range_m: np.ndarray, n_bins: int = 10,
                       dark_pct: float = 1.0) -> tuple[np.ndarray, dict]:
    """Sea-thru-style correction with parameters estimated from the image.

    Follows the structure of Akkaynak & Treibitz (CVPR 2019): backscatter is fit
    to the darkest pixels in each range bin, then per-channel attenuation is fit
    to the range-binned mean of the backscatter-free signal (a grey-world-per-
    range assumption standing in for Sea-thru's local illuminant map). Range is
    assumed known, as it would be from stereo or SfM; water parameters are not.
    """
    from .optics import linear_to_srgb, srgb_to_linear

    lin = srgb_to_linear(rgb_u8.astype(np.float32) / 255.0)
    r = range_m.astype(np.float32)
    edges = np.quantile(r, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, n_bins - 1)
    near = r < 0.99 * r.max()  # exclude clamped far pixels (sky) from the attenuation fit
    out = np.empty_like(lin)
    params = {}
    for c in range(3):
        ch = lin[..., c]
        rb, dark, mean_r, mean_d, w = [], [], [], [], []
        for k in range(n_bins):
            m = idx == k
            if m.sum() < 50:
                continue
            rb.append(float(r[m].mean()))
            dark.append(float(np.percentile(ch[m], dark_pct)))
        b_inf, beta_b = _fit_backscatter(np.array(rb), np.array(dark))
        D = np.clip(ch - b_inf * (1.0 - np.exp(-beta_b * r)), 0.0, None)
        for k in range(n_bins):
            m = (idx == k) & near
            if m.sum() < 50:
                continue
            mean_r.append(float(r[m].mean()))
            mean_d.append(max(float(D[m].mean()), 1e-6))
            w.append(float(m.sum()))
        A = np.stack([np.ones(len(mean_r)), -np.array(mean_r)], 1) * np.sqrt(w)[:, None]
        logA, beta_d = np.linalg.lstsq(A, np.log(mean_d) * np.sqrt(w), rcond=None)[0]
        beta_d = max(float(beta_d), 0.0)
        out[..., c] = D / (np.exp(logA) * np.exp(-beta_d * r)) * 0.18   # mean reflectance -> 18% grey
        params[c] = {"B_inf": b_inf, "beta_B": float(beta_b), "beta_D": beta_d}
    p99 = float(np.percentile(out, 99))
    out = out * (0.9 / max(p99, 1e-6))
    return (linear_to_srgb(out) * 255.0 + 0.5).astype(np.uint8), params


def sea_thru_oracle(rgb_u8: np.ndarray, range_m: np.ndarray, water, gain: float,
                    curves=None) -> np.ndarray:
    """Invert src.optics with the true water parameters and exposure gain.

    An upper bound, not a method: it knows everything the simulator used.
    Noise, quantisation and the forward-scatter blur are not inverted (the
    forward term is folded into the transmission).
    """
    from .optics import channel_curves, linear_to_srgb, srgb_to_linear

    curves = curves or channel_curves(water)
    T, F, B = curves.at(range_m)
    lin = srgb_to_linear(rgb_u8.astype(np.float32) / 255.0) / gain
    rho = np.clip((lin - B) / np.maximum(T + F, 1e-4), 0.0, 1.0)
    return (linear_to_srgb(rho) * 255.0 + 0.5).astype(np.uint8)
