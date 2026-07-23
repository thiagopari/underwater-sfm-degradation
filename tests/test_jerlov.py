"""Sanity tests for the Jerlov degradation model.

These verify the physical intuitions that the model MUST satisfy:
  1. Red attenuates faster than green, which attenuates faster than blue.
  2. Turbid water types attenuate faster than clear ones at the same depth.
  3. Deeper pixels shift toward the veiling-light color.
  4. Zero depth is (approximately) a no-op.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.jerlov import degrade, DegradeConfig, JERLOV_TYPES


def _flat(color, depth_m, shape=(16, 16)):
    rgb = np.tile(np.array(color, dtype=np.float32), shape + (1,))
    depth = np.full(shape, float(depth_m), dtype=np.float32)
    return rgb, depth


def test_red_attenuates_faster_than_blue():
    """At any nontrivial depth, R channel loss > B channel loss."""
    rgb, depth = _flat([0.8, 0.8, 0.8], depth_m=2.0)
    for jt in ["I", "II", "III", "1C"]:
        out = degrade(rgb, depth, DegradeConfig(jerlov_type=jt))
        r_loss = 0.8 - out[0, 0, 0]
        b_loss = 0.8 - out[0, 0, 2]
        assert r_loss > b_loss, f"{jt}: r_loss={r_loss:.3f} b_loss={b_loss:.3f}"


def test_turbid_attenuates_more_than_clear():
    """At the same depth, Jerlov 5C attenuates more than Jerlov I on all channels."""
    rgb, depth = _flat([0.7, 0.7, 0.7], depth_m=3.0)
    clear = degrade(rgb, depth, DegradeConfig(jerlov_type="I"))
    turbid = degrade(rgb, depth, DegradeConfig(jerlov_type="5C"))
    for c in range(3):
        clear_loss = 0.7 - clear[0, 0, c]
        turbid_loss = 0.7 - turbid[0, 0, c]
        assert turbid_loss > clear_loss, f"channel {c}: clear_loss={clear_loss:.3f} turbid_loss={turbid_loss:.3f}"


def test_zero_depth_is_near_identity():
    """At zero depth, transmission=1 and veiling=0, so output == input."""
    rgb, depth = _flat([0.4, 0.5, 0.6], depth_m=0.0)
    for jt in JERLOV_TYPES:
        out = degrade(rgb, depth, DegradeConfig(jerlov_type=jt))
        assert np.allclose(out, rgb, atol=1e-4), f"{jt} at z=0 changed the pixel"


def test_deep_pixels_approach_veiling_color():
    """As depth grows, the pixel converges to B_inf regardless of initial color.

    For Jerlov I, Kd_blue ~ 0.017 m^-1 -> need z >> 58m to saturate blue channel.
    """
    rgb_red, depth_deep = _flat([1.0, 0.0, 0.0], depth_m=1000.0)
    for jt in JERLOV_TYPES:
        out = degrade(rgb_red, depth_deep, DegradeConfig(jerlov_type=jt))
        b_inf = np.array(JERLOV_TYPES[jt]["b_inf_rgb"], dtype=np.float32)
        assert np.allclose(out[0, 0], b_inf, atol=1e-3), (
            f"{jt} did not converge to B_inf: got {out[0,0]} expected {b_inf}"
        )


def test_intensity_scales_attenuation():
    """intensity=2 attenuates more than intensity=1 for same type + depth."""
    rgb, depth = _flat([0.8, 0.8, 0.8], depth_m=1.0)
    o1 = degrade(rgb, depth, DegradeConfig(jerlov_type="II", intensity=1.0))
    o2 = degrade(rgb, depth, DegradeConfig(jerlov_type="II", intensity=2.0))
    for c in range(3):
        # more attenuation => further from original
        assert abs(0.8 - o2[0, 0, c]) > abs(0.8 - o1[0, 0, c])


def test_output_in_range():
    """Output always in [0,1]."""
    rgb, depth = _flat([0.9, 0.5, 0.1], depth_m=2.0)
    for jt in JERLOV_TYPES:
        out = degrade(rgb, depth, DegradeConfig(jerlov_type=jt))
        assert out.min() >= 0.0
        assert out.max() <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
