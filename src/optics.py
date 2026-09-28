"""Spectral underwater image formation + camera sensor model (study v2).

Replaces the v1 model in ``src/jerlov.py``, which used the diffuse attenuation
coefficient Kd along the line of sight, hand-picked veiling colours and no
sensor noise. Here every per-channel quantity is integrated over wavelength
from measured inherent optical properties (IOPs), so the direct and
backscatter coefficients differ and depend on range, as in the revised
underwater image formation model of Akkaynak & Treibitz (CVPR 2018).

Per wavelength lambda and line-of-sight range r (all radiances relative to a
white Lambertian target at the surface, i.e. E_surface/pi):

    E(l)      = D65(l) * exp(-Kd(l) * d_amb)          ambient light at depth d_amb
    direct    = rho * E(l) * exp(-c(l) r)             unscattered target light
    forward   = rho * E(l) * exp(-c r) (exp(eta b r) - 1)
                                                      small-angle forward scatter,
                                                      rendered blurred
    veiling   = bb(l) E(l) / (2 c(l)) * (1 - exp(-c r))
                                                      single-scattering path
                                                      radiance (B_inf estimate)

with c = a + b, bb = backscatter_ratio * b and Kd ~= 1.0395 (a + bb) / mu_d
(Gordon 1989). Each camera channel integrates these against a Gaussian
spectral sensitivity, white-balanced for daylight at the surface.

The camera then auto-exposes (digital gain so the 99th-percentile luminance
lands at ``target_p99``), which is where turbidity really hurts SfM: fewer
photons, more gain, more shot noise.

Known simplifications (documented in the README): ambient illumination only
(no vehicle lights), a single blur width for forward scatter, Gaussian channel
sensitivities, and an order-of-magnitude veiling estimate.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
WATER_TYPES = ("IB", "II", "III", "1C", "3C", "5C")
DESCRIPTIONS = {
    "IB": "average open ocean",
    "II": "clear coastal / open ocean",
    "III": "turbid open ocean",
    "1C": "clearest coastal",
    "3C": "coastal",
    "5C": "turbid coastal",
}

# Camera channel sensitivities: Gaussians (centre, sigma) in nm.
_CHANNELS = ((600.0, 35.0), (540.0, 35.0), (460.0, 30.0))  # R, G, B


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


@lru_cache(maxsize=None)
def _tables():
    iop_rows = _read_csv(DATA_DIR / "jerlov_iop_williamson2022.csv")
    wl = np.array(sorted({int(r["wavelength_nm"]) for r in iop_rows}), dtype=np.float64)
    iop = {}
    for t in WATER_TYPES:
        rows = sorted((r for r in iop_rows if r["water_type"] == t), key=lambda r: int(r["wavelength_nm"]))
        iop[t] = (np.array([float(r["a_per_m"]) for r in rows]), np.array([float(r["b_per_m"]) for r in rows]))
    d65_rows = {int(r["wavelength_nm"]): float(r["relative_power"]) for r in _read_csv(DATA_DIR / "cie_d65.csv")}
    d65 = np.array([d65_rows[int(w)] for w in wl])
    sens = np.stack([np.exp(-0.5 * ((wl - mu) / s) ** 2) for mu, s in _CHANNELS])  # 3 x L
    return wl, iop, d65 / d65.max(), sens


def iop(water_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (wavelengths_nm, a, b) for a Jerlov type."""
    wl, table, _, _ = _tables()
    a, b = table[water_type]
    return wl, a, b


@dataclass
class Water:
    water_type: str = "1C"
    backscatter_ratio: float | None = None   # bb/b; default 0.01 open ocean, 0.02 coastal
    ambient_depth_m: float = 5.0             # depth of the scene below the surface
    forward_scatter_frac: float = 0.5        # eta: share of scattering kept near-forward
    forward_blur_deg: float = 0.5            # angular width (sigma) of the forward-scatter halo
    mu_d: float = 0.85                       # mean cosine of downwelling light (for Kd)

    def __post_init__(self):
        if self.water_type not in WATER_TYPES:
            raise ValueError(f"unknown water type {self.water_type!r}; choose from {WATER_TYPES}")
        if self.backscatter_ratio is None:
            self.backscatter_ratio = 0.01 if self.water_type in ("IB", "II", "III") else 0.02


@dataclass
class Sensor:
    full_well_e: float = 5000.0   # electrons for a surface-lit white target at base exposure
    read_noise_e: float = 3.0
    target_p99: float = 0.9       # auto-exposure target for the 99th-percentile luminance
    max_gain: float = 256.0
    jpeg_quality: int = 92


@dataclass
class ChannelCurves:
    """Per-channel T(r), F(r), B(r) tabulated on a range grid."""
    r: np.ndarray
    T: np.ndarray  # 3 x N direct transmission (includes ambient dimming)
    F: np.ndarray  # 3 x N forward-scatter (blurred) contribution
    B: np.ndarray  # 3 x N veiling radiance

    def at(self, range_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = []
        for curve in (self.T, self.F, self.B):
            out.append(np.stack([np.interp(range_m, self.r, curve[c]) for c in range(3)], axis=-1).astype(np.float32))
        return tuple(out)


def channel_curves(water: Water, r_max: float = 200.0, n: int = 1024) -> ChannelCurves:
    wl, table, d65, sens = _tables()
    a, b = table[water.water_type]
    c = a + b
    bb = water.backscatter_ratio * b
    kd = 1.0395 * (a + bb) / water.mu_d
    E = d65 * np.exp(-kd * water.ambient_depth_m)                  # L
    norm = sens @ d65                                             # 3: daylight white balance at the surface
    r = np.concatenate([[0.0], np.geomspace(1e-3, r_max, n - 1)])
    att = np.exp(-np.outer(r, c))                                 # N x L
    T = (sens * E) @ att.T / norm[:, None]
    F = (sens * E) @ (att * (np.exp(np.outer(r, water.forward_scatter_frac * b)) - 1.0)).T / norm[:, None]
    B = (sens * (E * bb / (2.0 * c))) @ (1.0 - att).T / norm[:, None]
    return ChannelCurves(r=r, T=T, F=F, B=B)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def render_linear(rho: np.ndarray, range_m: np.ndarray, water: Water, focal_px: float,
                  curves: ChannelCurves | None = None) -> np.ndarray:
    """Noise-free linear radiance of an underwater view, relative to surface white.

    rho:      HxWx3 linear RGB of the clear-air image (treated as reflectance).
    range_m:  HxW line-of-sight range in metres.
    """
    curves = curves or channel_curves(water)
    T, F, B = curves.at(range_m)
    sigma = max(focal_px * np.tan(np.deg2rad(water.forward_blur_deg)), 0.5)
    blurred = cv2.GaussianBlur(rho, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    return (rho * T + blurred * F + B).astype(np.float32)


def luminance(lin: np.ndarray) -> np.ndarray:
    return lin @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def auto_exposure_gain(lin: np.ndarray, sensor: Sensor) -> float:
    p99 = float(np.percentile(luminance(lin), 99))
    return float(np.clip(sensor.target_p99 / max(p99, 1e-6), 1.0 / 8.0, sensor.max_gain))


def expose(lin: np.ndarray, sensor: Sensor, rng: np.random.Generator | None,
           gain: float | None = None) -> tuple[np.ndarray, float]:
    """Photon/read noise, auto-exposure gain, sRGB encoding. Returns (uint8 RGB, gain).

    rng=None gives the noise-free image (used by tests and the oracle).
    """
    g = auto_exposure_gain(lin, sensor) if gain is None else gain
    electrons = np.clip(lin, 0.0, None) * sensor.full_well_e
    if rng is not None:
        electrons = rng.poisson(electrons).astype(np.float32)
        electrons += rng.normal(0.0, sensor.read_noise_e, size=electrons.shape).astype(np.float32)
    value = electrons / sensor.full_well_e * g
    return (linear_to_srgb(value) * 255.0 + 0.5).astype(np.uint8), g


def degrade_image(rgb_u8: np.ndarray, range_m: np.ndarray, water: Water, sensor: Sensor,
                  focal_px: float, rng: np.random.Generator | None,
                  curves: ChannelCurves | None = None) -> tuple[np.ndarray, float]:
    """Clear-air sRGB uint8 image -> simulated underwater sRGB uint8 image."""
    rho = srgb_to_linear(rgb_u8.astype(np.float32) / 255.0)
    lin = render_linear(rho, range_m, water, focal_px, curves)
    return expose(lin, sensor, rng)


def effective_coefficients(water: Water, r: float = 3.0) -> dict[str, list[float]]:
    """Wideband per-channel direct (beta_D) and backscatter (beta_B) coefficients at range r.

    Reported in the README so readers can compare with v1's Kd values.
    """
    cur = channel_curves(water)
    T0, _, _ = cur.at(np.array([1e-6]))
    Tr, _, Br = cur.at(np.array([r]))
    _, _, Binf = cur.at(np.array([cur.r[-1]]))
    beta_d = -np.log(Tr[0] / T0[0]) / r
    beta_b = -np.log(np.clip(1.0 - Br[0] / Binf[0], 1e-9, None)) / r
    return {"beta_D": beta_d.round(3).tolist(), "beta_B": beta_b.round(3).tolist(),
            "B_inf": Binf[0].round(4).tolist(), "ambient_T": T0[0].round(3).tolist()}
