"""Tests for color-correction baselines."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.color_correction import gray_world, shades_of_gray, uwcnn_like_stretch, sea_thru_inverse, correct
from src.jerlov import degrade, DegradeConfig


def test_gray_world_neutralizes_cast():
    """Applying gray world to an image with a red cast should reduce the cast."""
    # image with strong red bias
    rgb = np.zeros((32, 32, 3), dtype=np.float32)
    rgb[..., 0] = 0.8  # bright red
    rgb[..., 1] = 0.4
    rgb[..., 2] = 0.2
    out = gray_world(rgb)
    means = out.reshape(-1, 3).mean(axis=0)
    # channel means should be closer together after correction
    assert (means.max() - means.min()) < (0.8 - 0.2)


def test_shades_of_gray_stays_in_range():
    rgb = np.random.rand(24, 24, 3).astype(np.float32)
    out = shades_of_gray(rgb, p=6.0)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_stretch_uses_full_range():
    """Percentile stretch should push most pixels into [0,1] range."""
    rgb = 0.3 + 0.2 * np.random.rand(16, 16, 3).astype(np.float32)
    out = uwcnn_like_stretch(rgb, lo=1.0, hi=99.0)
    # after stretch, min close to 0 and max close to 1 per channel
    for c in range(3):
        assert out[..., c].min() < 0.05
        assert out[..., c].max() > 0.95


def test_sea_thru_inverse_recovers_input():
    """Applying degrade then sea_thru with same params should ~recover the input.
    (Only valid for depths where transmission stays above the numerical floor.)"""
    rgb = np.random.rand(16, 16, 3).astype(np.float32) * 0.6 + 0.2
    depth = np.full((16, 16), 1.0, dtype=np.float32)  # 1m depth, well within trans floor
    jt, ix = "II", 1.0
    degraded = degrade(rgb, depth, DegradeConfig(jerlov_type=jt, intensity=ix))
    recovered = sea_thru_inverse(degraded, depth, jerlov_type=jt, intensity=ix)
    # allow some tolerance since we clip to [0,1] in degrade
    assert np.allclose(recovered, rgb, atol=0.02), f"max diff: {np.abs(recovered-rgb).max()}"


def test_dispatch():
    rgb = np.random.rand(8, 8, 3).astype(np.float32)
    for method in ["gray_world", "shades_of_gray", "stretch", "none"]:
        out = correct(rgb, method=method)
        assert out.shape == rgb.shape


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
