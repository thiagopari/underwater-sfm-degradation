"""Visualization utilities for the study — comparison strips, per-level thumbnails, plots."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


PLT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#333",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "text.color": "#222",
    "xtick.color": "#333",
    "ytick.color": "#333",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
}
plt.rcParams.update(PLT_STYLE)


def comparison_strip(
    sweep_root: Path,
    image_name: str,
    out_path: Path,
    max_width_px: int = 480,
) -> Path:
    """Create a horizontal strip of one image across all sweep levels."""
    sweep_root = Path(sweep_root)
    levels = sorted(p for p in sweep_root.iterdir() if p.is_dir() and p.name.startswith("level_"))

    imgs = []
    labels = []
    for level in levels:
        p = level / image_name
        if not p.exists():
            # try lowercase
            candidates = list(level.glob(image_name.rsplit(".", 1)[0] + ".*"))
            if not candidates:
                continue
            p = candidates[0]
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w > max_width_px:
            scale = max_width_px / w
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        imgs.append(rgb)
        # short pretty label
        name = level.name.replace("level_", "").replace("_x1.00", "")
        labels.append(name)

    if not imgs:
        raise RuntimeError("No images found for comparison")

    fig, axes = plt.subplots(1, len(imgs), figsize=(2.4 * len(imgs), 3.2))
    if len(imgs) == 1:
        axes = [axes]
    for ax, im, lb in zip(axes, imgs, labels):
        ax.imshow(im)
        ax.set_title(lb, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"Turbidity sweep — {image_name}", y=1.02, fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return out_path


def depth_visualization(depth: np.ndarray, out_path: Path) -> Path:
    """Save a colormapped depth visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(depth, cmap="viridis")
    ax.set_title("Estimated depth (m)")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.7, label="meters")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return out_path


def fine_sweep_plot(fine_metrics: dict, out_path: Path) -> Path:
    """Plot fine-grained sweep of intensity within a single Jerlov type."""
    plt.rcParams.update(PLT_STYLE)
    tags = sorted(fine_metrics.keys())
    # extract intensities from tags like "fine_3C_x1.25"
    intensities = []
    for t in tags:
        try:
            intensities.append(float(t.rsplit("x", 1)[1]))
        except Exception:
            intensities.append(0.0)

    def _get(k):
        return [fine_metrics[t].get(k, 0) for t in tags]

    reg = _get("num_registered")
    pts = _get("num_3d_points")
    reproj = _get("mean_reproj_error")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    color = "#2563eb"
    for ax, y, ylabel, title in [
        (axes[0], reg,    "Images registered",        "Registration vs intensity"),
        (axes[1], pts,    "3D points",                "Reconstruction density vs intensity"),
        (axes[2], reproj, "Mean reproj err (px)",     "Reprojection error vs intensity"),
    ]:
        ax.plot(intensities, y, "-o", color=color, linewidth=2, markersize=7)
        ax.set_xlabel("Kd intensity multiplier")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)

    fig.suptitle("Fine-grained sweep — Jerlov 3C, varying Kd intensity", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path


def ablation_visual(
    sweep_dir: Path,
    corrected_root: Path,
    level_name: str,
    image_name: str,
    out_path: Path,
) -> Path:
    """4-panel figure: baseline, degraded, +Shades-of-Gray, +Sea-thru."""
    plt.rcParams.update(PLT_STYLE)
    paths = [
        (sweep_dir / "level_00_baseline" / image_name, "baseline (clear air)"),
        (sweep_dir / level_name / image_name, f"degraded ({_short_from(level_name)})"),
        (corrected_root / f"{level_name}__shades_of_gray" / image_name, "+ Shades-of-Gray"),
        (corrected_root / f"{level_name}__sea_thru" / image_name, "+ Sea-thru (oracle)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, (p, title) in zip(axes, paths):
        bgr = cv2.imread(str(p)) if p.exists() else None
        if bgr is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.axis("off")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w > 500:
            scale = 500 / w
            rgb = cv2.resize(rgb, (int(w*scale), int(h*scale)))
        ax.imshow(rgb)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle(f"Color-correction ablation — {level_name}", y=1.02, fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return out_path


def _short_from(level_name: str) -> str:
    if "baseline" in level_name:
        return "baseline"
    parts = level_name.split("_")
    return parts[2] if len(parts) > 2 else level_name


def depth_grid(depth_dir: Path, out_path: Path, cols: int = 6) -> Path:
    """Grid montage of all cached depth maps."""
    depth_dir = Path(depth_dir)
    files = sorted(depth_dir.glob("*.npy"))
    if not files:
        raise RuntimeError(f"no .npy files in {depth_dir}")
    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 1.6))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for i, p in enumerate(files):
        d = np.load(p)
        ax = axes[i]
        ax.imshow(d, cmap="viridis", aspect="auto")
        ax.set_title(p.stem, fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    for j in range(len(files), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Per-image estimated depth (0.5–5m, viridis)", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return out_path


def image_quality_plot(image_metrics: dict, sfm_metrics: dict, out_path: Path) -> Path:
    """Two-axis plot: image RMS contrast + Laplacian sharpness on one axis,
    SfM registration rate on the other. Shows how the pipeline collapses
    only after image quality passes a threshold.
    """
    tags = sorted(image_metrics.keys())
    labels = [t.replace("level_", "").replace("_x1.00", "").split("_")[-1] if "baseline" not in t else "baseline" for t in tags]

    contrast = [image_metrics[t].get("rms_contrast", 0) for t in tags]
    sharp    = [image_metrics[t].get("laplacian_var", 0) for t in tags]

    def _reg(m: dict, t: str) -> float:
        d = m.get(t, {})
        return 100.0 * d.get("num_registered", 0) / max(d.get("num_input_images", 1), 1)

    reg = [_reg(sfm_metrics, t) for t in tags]

    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    color1 = "#f59e0b"
    color2 = "#2563eb"

    ax1.plot(labels, contrast, "-o", color=color1, label="RMS contrast", linewidth=2)
    ax1.set_ylabel("Image quality proxy (RMS contrast)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, max(contrast) * 1.15 + 1)

    ax2 = ax1.twinx()
    ax2.bar(labels, reg, color=color2, alpha=0.35, edgecolor=color2, label="SfM registration %")
    ax2.set_ylabel("Images registered (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 105)
    ax2.grid(False)

    ax1.set_title("Image contrast drops smoothly; SfM registration drops off a cliff")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path


def metrics_plots(metrics: dict, out_dir: Path) -> list[Path]:
    """Produce four small plots: registration rate, points, features, reproj error."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort by level name
    tags = sorted(metrics.keys())
    labels = [t.replace("level_", "").replace("_x1.00", "") for t in tags]

    def _get(key, default=0):
        return [metrics[t].get(key, default) if isinstance(metrics[t], dict) else default for t in tags]

    reg = _get("num_registered")
    n_input = _get("num_input_images", 30)
    reg_rate = [(r / max(n, 1)) * 100 for r, n in zip(reg, n_input)]
    pts = _get("num_3d_points")
    reproj = _get("mean_reproj_error", 0.0)
    features_lists = [metrics[t].get("features_per_image", []) if isinstance(metrics[t], dict) else [] for t in tags]
    mean_feats = [np.mean(fl) if fl else 0 for fl in features_lists]

    accent = "#2563eb"

    out_paths = []

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(labels, reg_rate, color=accent, edgecolor="white")
    ax.set_ylabel("Images registered (%)")
    ax.set_ylim(0, 105)
    ax.set_title("SfM registration rate vs. water condition")
    for b, v in zip(bars, reg_rate):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    p = out_dir / "registration_rate.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); out_paths.append(p)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(labels, pts, color=accent, edgecolor="white")
    ax.set_ylabel("Reconstructed 3D points")
    ax.set_title("Reconstruction density vs. water condition")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    p = out_dir / "points_3d.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); out_paths.append(p)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(labels, mean_feats, color=accent, edgecolor="white")
    ax.set_ylabel("Mean SIFT features / image")
    ax.set_title("Feature extraction vs. water condition")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    p = out_dir / "features_per_image.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); out_paths.append(p)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(labels, reproj, color=accent, edgecolor="white")
    ax.set_ylabel("Mean reprojection error (px)")
    ax.set_title("Reprojection error vs. water condition")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    p = out_dir / "reproj_error.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); out_paths.append(p)

    return out_paths
