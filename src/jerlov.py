"""Wavelength-dependent Jerlov underwater optics degradation.

Applies a Sea-thru-like image formation model to convert clear-air RGB + depth
into a synthetically-underwater RGB image, using published Jerlov water-type
attenuation coefficients.

Model (per RGB channel c):
    I_c = J_c * exp(-Kd_c * z) + B_inf_c * (1 - exp(-Kd_c * z))

where J is the clear-air reflectance, z is the per-pixel depth from the
camera, Kd is the diffuse attenuation coefficient at the channel's
representative wavelength, and B_inf is the ambient veiling-light color.

This is a simplification of Akkaynak & Treibitz (CVPR 2019), which shows the
direct-signal and backscatter coefficients are formally different; we tie them
to the same Kd for a compact one-parameter-per-channel model that captures the
dominant color-cast and contrast-loss behavior.

Kd values (per Jerlov 1976 / Solonenko & Mobley 2015) are approximate at
representative R/G/B wavelengths of 650/550/450 nm.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# (Kd_R, Kd_G, Kd_B) in m^-1, and (B_inf_R, B_inf_G, B_inf_B) in normalized [0,1]
JERLOV_TYPES: dict[str, dict[str, tuple[float, float, float]]] = {
    "I":  {"kd_rgb": (0.34, 0.075, 0.017), "b_inf_rgb": (0.02, 0.15, 0.35)},
    "IA": {"kd_rgb": (0.36, 0.088, 0.026), "b_inf_rgb": (0.03, 0.17, 0.36)},
    "IB": {"kd_rgb": (0.38, 0.110, 0.043), "b_inf_rgb": (0.04, 0.19, 0.38)},
    "II": {"kd_rgb": (0.42, 0.130, 0.075), "b_inf_rgb": (0.05, 0.22, 0.42)},
    "III":{"kd_rgb": (0.50, 0.200, 0.130), "b_inf_rgb": (0.07, 0.28, 0.45)},
    "1C": {"kd_rgb": (0.80, 0.450, 0.300), "b_inf_rgb": (0.10, 0.35, 0.50)},
    "3C": {"kd_rgb": (1.10, 0.700, 0.550), "b_inf_rgb": (0.15, 0.40, 0.55)},
    "5C": {"kd_rgb": (1.50, 1.100, 0.900), "b_inf_rgb": (0.20, 0.45, 0.60)},
}

_TYPE_DESCRIPTIONS = {
    "I":  "clearest open ocean",
    "IA": "clear open ocean",
    "IB": "average open ocean",
    "II": "clear coastal water",
    "III":"turbid coastal water",
    "1C": "harbor (moderately turbid)",
    "3C": "harbor (very turbid)",
    "5C": "harbor (extremely turbid)",
}


def describe(jerlov_type: str) -> str:
    return _TYPE_DESCRIPTIONS.get(jerlov_type, jerlov_type)


@dataclass
class DegradeConfig:
    jerlov_type: str = "II"
    intensity: float = 1.0        # global multiplier on Kd, lets you slide continuously
    depth_offset_m: float = 0.0   # add to depth (e.g., to model camera standoff)


def degrade(
    rgb: np.ndarray,
    depth: np.ndarray,
    cfg: DegradeConfig | None = None,
) -> np.ndarray:
    """Apply Jerlov degradation to a single RGB image.

    Args:
        rgb:   HxWx3 float in [0,1] (RGB order).
        depth: HxW float, per-pixel range from camera in meters.
        cfg:   DegradeConfig.

    Returns:
        HxWx3 float in [0,1] (RGB), the degraded image.
    """
    if cfg is None:
        cfg = DegradeConfig()
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)

    props = JERLOV_TYPES[cfg.jerlov_type]
    kd = np.asarray(props["kd_rgb"], dtype=np.float32) * float(cfg.intensity)
    b_inf = np.asarray(props["b_inf_rgb"], dtype=np.float32)

    z = (depth + cfg.depth_offset_m)[..., None]                    # HxWx1
    trans = np.exp(-kd[None, None, :] * z)                         # HxWx3
    veiling = b_inf[None, None, :] * (1.0 - trans)                 # HxWx3
    out = rgb * trans + veiling
    return np.clip(out, 0.0, 1.0)


def degrade_uint8(rgb_u8: np.ndarray, depth: np.ndarray, cfg: DegradeConfig | None = None) -> np.ndarray:
    """Convenience wrapper for uint8 input/output (RGB order)."""
    out = degrade(rgb_u8.astype(np.float32) / 255.0, depth, cfg)
    return (out * 255.0 + 0.5).astype(np.uint8)


if __name__ == "__main__":
    # smoke test
    rgb = np.full((32, 32, 3), 0.8, dtype=np.float32)
    depth = np.linspace(0.5, 5.0, 32 * 32).reshape(32, 32).astype(np.float32)
    for jt in ["I", "II", "III", "1C", "5C"]:
        out = degrade(rgb, depth, DegradeConfig(jerlov_type=jt))
        near = out[0, 0]
        far = out[-1, -1]
        print(f"{jt}: near RGB={near.round(2).tolist()}  far RGB={far.round(2).tolist()}")
