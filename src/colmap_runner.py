"""Wrapper around COLMAP CLI to run incremental SfM and collect metrics."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


COLMAP_BIN = "colmap"


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"COLMAP command failed ({' '.join(cmd)}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def run_sfm(
    image_dir: Path,
    work_dir: Path,
    max_num_features: int = 8192,
    peak_threshold: float = 0.001,
    init_min_num_inliers: int = 15,
    init_min_tri_angle: float = 4.0,
    max_image_size: int = 1600,
    clean: bool = True,
) -> dict[str, Any]:
    """Run COLMAP feature extractor + matcher + mapper on image_dir.

    Uses relaxed SIFT thresholds so heavily-degraded scenes still get *some*
    features, which lets us measure exactly when the pipeline breaks. Caps
    image resolution for tractable CPU-only runtime.
    """
    image_dir = Path(image_dir)
    work_dir = Path(work_dir)
    if clean and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    _run([
        COLMAP_BIN, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.peak_threshold", str(peak_threshold),
        "--SiftExtraction.max_num_features", str(max_num_features),
        "--FeatureExtraction.max_image_size", str(max_image_size),
        "--FeatureExtraction.use_gpu", "0",
    ])
    _run([
        COLMAP_BIN, "exhaustive_matcher",
        "--database_path", str(db_path),
        "--FeatureMatching.use_gpu", "0",
    ])
    _run([
        COLMAP_BIN, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.init_min_num_inliers", str(init_min_num_inliers),
        "--Mapper.init_min_tri_angle", str(init_min_tri_angle),
    ])

    # Read features/keypoints per image from the DB
    metrics = _summarize(db_path, sparse_dir, image_dir)
    (work_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _summarize(db_path: Path, sparse_dir: Path, image_dir: Path) -> dict[str, Any]:
    import sqlite3

    metrics: dict[str, Any] = {
        "image_dir": str(image_dir),
        "num_input_images": len(list(image_dir.glob("*.JPG")) + list(image_dir.glob("*.jpg"))),
        "features_per_image": [],
        "num_matches_pairs": 0,
        "num_registered": 0,
        "num_3d_points": 0,
        "mean_track_length": 0.0,
        "mean_reproj_error": 0.0,
    }

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rows = cur.execute(
        "SELECT image_id, rows FROM keypoints"
    ).fetchall()
    metrics["features_per_image"] = [int(r[1]) for r in rows]

    matches_pairs = cur.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    metrics["num_matches_pairs"] = int(matches_pairs)
    con.close()

    # Read reconstructed model with COLMAP text export
    models = sorted(sparse_dir.iterdir()) if sparse_dir.exists() else []
    if models:
        # Use the largest (most images) reconstruction
        best_model = None
        best_size = -1
        for m in models:
            if not m.is_dir():
                continue
            images_txt = m / "images.txt"
            if not images_txt.exists():
                # try converting
                try:
                    _run([
                        COLMAP_BIN, "model_converter",
                        "--input_path", str(m),
                        "--output_path", str(m),
                        "--output_type", "TXT",
                    ])
                except Exception:
                    continue
            if not images_txt.exists():
                continue
            n_imgs = _count_registered(images_txt)
            if n_imgs > best_size:
                best_size = n_imgs
                best_model = m

        if best_model is not None:
            metrics["num_registered"] = _count_registered(best_model / "images.txt")
            pts, mean_track, mean_reproj = _stats_points3d(best_model / "points3D.txt")
            metrics["num_3d_points"] = pts
            metrics["mean_track_length"] = mean_track
            metrics["mean_reproj_error"] = mean_reproj

    return metrics


def _count_registered(images_txt: Path) -> int:
    """Each registered image occupies 2 lines in images.txt; count the header lines."""
    n = 0
    with open(images_txt) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            n += 1
    # each image = 2 non-comment lines
    return n // 2


def _stats_points3d(points_txt: Path) -> tuple[int, float, float]:
    n = 0
    total_track_len = 0
    total_err = 0.0
    if not points_txt.exists():
        return 0, 0.0, 0.0
    with open(points_txt) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            # POINT3D_ID X Y Z R G B ERROR TRACK[]
            err = float(parts[7])
            track_len = (len(parts) - 8) // 2
            total_err += err
            total_track_len += track_len
            n += 1
    if n == 0:
        return 0, 0.0, 0.0
    return n, total_track_len / n, total_err / n
