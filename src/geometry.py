"""COLMAP text-model I/O and pose-accuracy evaluation against a reference model.

The clear-air COLMAP reconstruction is the reference. A degraded run is scored
by aligning its camera centres to the reference with a similarity transform
(Umeyama 1991) and reporting absolute trajectory error (ATE) and rotation
error. Registration counts alone say nothing about whether the poses are
right; these metrics do.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def K(self) -> np.ndarray:
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            f, cx, cy = p[0], p[1], p[2]
            fx = fy = f
        else:  # PINHOLE, OPENCV, ... start with fx, fy, cx, cy
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


@dataclass
class Image:
    name: str
    camera_id: int
    R: np.ndarray          # world -> camera rotation
    t: np.ndarray          # world -> camera translation
    xy: np.ndarray         # N x 2 keypoints with a 3D point
    point_ids: np.ndarray  # N

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass
class Model:
    cameras: dict[int, Camera]
    images: dict[str, Image]
    points: dict[int, np.ndarray]  # point3D_id -> xyz


def qvec_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _data_lines(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                yield line.rstrip("\n")


def read_model(model_dir: Path) -> Model:
    model_dir = Path(model_dir)
    cameras = {}
    for line in _data_lines(model_dir / "cameras.txt"):
        p = line.split()
        cameras[int(p[0])] = Camera(p[1], int(p[2]), int(p[3]), np.array([float(v) for v in p[4:]]))
    images = {}
    lines = list(_data_lines(model_dir / "images.txt"))
    for header, obs in zip(lines[0::2], lines[1::2]):
        h = header.split()
        vals = np.array([float(v) for v in obs.split()]) if obs.strip() else np.zeros(0)
        vals = vals.reshape(-1, 3)
        keep = vals[:, 2] >= 0 if len(vals) else np.zeros(0, bool)
        images[h[9]] = Image(
            name=h[9], camera_id=int(h[8]),
            R=qvec_to_R(np.array([float(v) for v in h[1:5]])),
            t=np.array([float(v) for v in h[5:8]]),
            xy=vals[keep, :2], point_ids=vals[keep, 2].astype(np.int64),
        )
    points = {}
    for line in _data_lines(model_dir / "points3D.txt"):
        p = line.split()
        points[int(p[0])] = np.array([float(p[1]), float(p[2]), float(p[3])])
    return Model(cameras, images, points)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (s, R, t) with dst ~= s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def align_rotations(R_ref: list[np.ndarray], R_est: list[np.ndarray]) -> np.ndarray:
    """Best world rotation A with R_est_i A^T ~= R_ref_i (chordal L2 mean, projected to SO(3))."""
    M = sum(Rr.T @ Re for Rr, Re in zip(R_ref, R_est))
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def relative_rotation_errors(R_ref: list[np.ndarray], R_est: list[np.ndarray]) -> np.ndarray:
    """Alignment-free: angle between reference and estimated relative rotations, all camera pairs."""
    errs = []
    for i in range(len(R_ref)):
        for j in range(i + 1, len(R_ref)):
            rel_ref = R_ref[i] @ R_ref[j].T
            rel_est = R_est[i] @ R_est[j].T
            errs.append(rotation_angle_deg(rel_ref @ rel_est.T))
    return np.array(errs)


def pose_errors(ref: Model, est: Model, metres_per_unit: float,
                pos_tol_m: float = 0.10, rot_tol_deg: float = 2.0) -> dict:
    """Align est to ref on shared images and report pose accuracy.

    metres_per_unit converts reference-model units to metres (the study fixes
    the reference scale so the median scene standoff equals the chosen value).
    """
    names = sorted(set(ref.images) & set(est.images))
    out = {"num_compared": len(names), "ate_rmse_m": None, "ate_median_m": None,
           "rot_err_median_deg": None, "num_accurate": 0}
    if len(names) < 3:
        return out
    C_ref = np.stack([ref.images[n].center for n in names])
    C_est = np.stack([est.images[n].center for n in names])
    s, R, t = umeyama(C_est, C_ref)
    C_al = (s * (R @ C_est.T)).T + t
    pos_err = np.linalg.norm(C_al - C_ref, axis=1) * metres_per_unit
    # Rotation error after the best rotation-only alignment, so a slightly mis-estimated
    # centre-based alignment (e.g. roll about a near-collinear camera path) is not
    # counted against every camera.
    R_ref = [ref.images[n].R for n in names]
    R_est = [est.images[n].R for n in names]
    A = align_rotations(R_ref, R_est)
    rot_err = np.array([rotation_angle_deg(Rr @ (Re @ A.T).T) for Rr, Re in zip(R_ref, R_est)])
    rel_err = relative_rotation_errors(R_ref, R_est)
    out.update({
        "ate_rmse_m": float(np.sqrt((pos_err ** 2).mean())),
        "ate_median_m": float(np.median(pos_err)),
        "rot_err_median_deg": float(np.median(rot_err)),
        "rel_rot_err_median_deg": float(np.median(rel_err)),
        "pos_err_max_m": float(pos_err.max()),
        "num_accurate": int(((pos_err < pos_tol_m) & (rot_err < rot_tol_deg)).sum()),
        "per_image": {n: [float(p), float(r)] for n, p, r in zip(names, pos_err, rot_err)},
    })
    return out
