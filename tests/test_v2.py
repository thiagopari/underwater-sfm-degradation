"""Tests for the v2 optics, depth alignment, geometry and correction code."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.color_correction import _fit_backscatter, sea_thru_oracle
from src.depth_align import fit_inverse_depth, relative_disparity_from_v1_cache
from src.geometry import qvec_to_R, rotation_angle_deg, umeyama
from src.optics import (WATER_TYPES, Sensor, Water, channel_curves, effective_coefficients, expose,
                        iop, linear_to_srgb, render_linear, srgb_to_linear)


# ---------------------------------------------------------------- IOP table
def test_attenuation_increases_with_turbidity():
    """Beam attenuation c = a + b grows from open ocean to turbid coastal water at every wavelength."""
    cs = [sum(iop(t)[1:]) for t in WATER_TYPES]
    for lo, hi in zip(cs, cs[1:]):
        assert np.all(hi >= lo - 1e-9)


@pytest.mark.parametrize("t", ["3C", "5C"])
def test_coastal_water_absorbs_blue_more_than_green(t):
    """Dissolved organic matter: in coastal types a(450) > a(550). v1 had this backwards."""
    wl, a, _ = iop(t)
    assert a[wl == 450][0] > a[wl == 550][0]


def test_red_is_most_absorbed_in_open_ocean():
    wl, a, _ = iop("IB")
    assert a[wl == 650][0] > a[wl == 550][0] > a[wl == 450][0]


# ---------------------------------------------------------------- renderer
def test_zero_range_is_only_ambient_dimming():
    """At r = 0 there is no path: output = reflectance x ambient transmission, no veil."""
    w = Water("1C")
    rho = np.full((8, 8, 3), 0.5, np.float32)
    out = render_linear(rho, np.zeros((8, 8), np.float32), w, focal_px=1000)
    amb = np.array(effective_coefficients(w)["ambient_T"], np.float32)
    assert np.allclose(out[0, 0], 0.5 * amb, atol=2e-3)


def test_far_pixels_converge_to_veiling_light():
    w = Water("3C")
    cur = channel_curves(w)
    rho = np.full((8, 8, 3), 0.9, np.float32)
    out = render_linear(rho, np.full((8, 8), 150.0, np.float32), w, focal_px=1000, curves=cur)
    _, _, b_inf = cur.at(np.array([cur.r[-1]]))
    assert np.allclose(out[0, 0], b_inf[0], rtol=0.05)


def test_signal_decreases_with_range_and_turbidity():
    rho = np.full((4, 4, 3), 0.6, np.float32)
    near = render_linear(rho, np.full((4, 4), 1.0, np.float32), Water("II"), 1000)
    far = render_linear(rho, np.full((4, 4), 4.0, np.float32), Water("II"), 1000)
    turbid = render_linear(rho, np.full((4, 4), 4.0, np.float32), Water("5C"), 1000)
    assert np.all(far < near) and np.all(turbid < far)


def test_direct_and_backscatter_coefficients_differ_slightly():
    """Wideband integration makes beta_D != beta_B (Akkaynak & Treibitz 2018)."""
    e = effective_coefficients(Water("II"))
    assert not np.allclose(e["beta_D"], e["beta_B"], atol=1e-4)


def test_srgb_round_trip():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


def test_noise_free_exposure_is_deterministic_and_noise_adds_variance():
    lin = np.full((64, 64, 3), 0.02, np.float32)
    a, g = expose(lin, Sensor(), None)
    b, _ = expose(lin, Sensor(), None)
    c, _ = expose(lin, Sensor(), np.random.default_rng(0))
    assert np.array_equal(a, b) and a.std() == 0
    assert c.std() > 0 and g > 1  # dark scene -> auto-exposure gain -> visible noise


# ---------------------------------------------------------------- depth alignment
def test_v1_cache_inversion():
    rel = np.linspace(0, 1, 11)
    v1 = 0.5 + 4.5 * (1 - rel)
    assert np.allclose(relative_disparity_from_v1_cache(v1), rel)


def test_inverse_depth_fit_recovers_scale_shift_with_outliers():
    rng = np.random.default_rng(1)
    rel = rng.uniform(0, 1, 400)
    inv = 0.8 * rel + 0.1
    inv[:60] = rng.uniform(0.05, 1.5, 60)  # 15% outliers
    s, o, inl = fit_inverse_depth(rel, inv)
    assert s == pytest.approx(0.8, rel=0.01) and o == pytest.approx(0.1, abs=0.01)
    assert inl[60:].mean() > 0.95


# ---------------------------------------------------------------- geometry
def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(20, 3))
    R = qvec_to_R(np.array([0.9, 0.1, -0.3, 0.2]))
    dst = 2.5 * (R @ src.T).T + np.array([1.0, -2.0, 0.5])
    s, R_est, t = umeyama(src, dst)
    assert s == pytest.approx(2.5) and np.allclose(R_est, R) and np.allclose(t, [1.0, -2.0, 0.5])


def test_rotation_angle():
    th = np.deg2rad(30)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    assert rotation_angle_deg(Rz) == pytest.approx(30.0)


# ---------------------------------------------------------------- corrections
def test_backscatter_fit_recovers_parameters():
    r = np.linspace(0.5, 8, 30)
    y = 0.3 * (1 - np.exp(-0.7 * r))
    b_inf, beta = _fit_backscatter(r, y)
    assert b_inf == pytest.approx(0.3, rel=0.05) and beta == pytest.approx(0.7, rel=0.05)


def test_oracle_inverts_noise_free_render():
    w = Water("1C")
    rng = np.random.default_rng(3)
    rho = rng.uniform(0.1, 0.9, (32, 32, 3)).astype(np.float32)
    rng_m = np.full((32, 32), 2.0, np.float32)
    cur = channel_curves(w)
    lin = render_linear(rho, rng_m, w, focal_px=0.0, curves=cur)  # focal 0 -> no blur
    img, g = expose(lin, Sensor(), None)
    back = srgb_to_linear(sea_thru_oracle(img, rng_m, w, g, cur) / 255.0)
    assert np.abs(back - rho).mean() < 0.03


def test_rotation_alignment_recovers_world_rotation():
    rng = np.random.default_rng(4)
    A = qvec_to_R(np.array([0.8, 0.3, 0.4, -0.2]))
    R_ref = [qvec_to_R(rng.normal(size=4)) for _ in range(10)]
    R_est = [R @ A for R in R_ref]  # R_est_i A^T = R_ref_i
    from src.geometry import align_rotations, relative_rotation_errors
    assert np.allclose(align_rotations(R_ref, R_est), A, atol=1e-8)
    assert relative_rotation_errors(R_ref, R_est).max() < 1e-5
