"""Make a portfolio-quality hero figure: depth + degraded strip in one image."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .viz import PLT_STYLE


def hero_figure(
    sweep_root: Path,
    depth_path: Path,
    image_name: str,
    out_path: Path,
) -> Path:
    """Two-row hero: top = depth colormap, bottom = images across sweep."""
    plt.rcParams.update(PLT_STYLE)
    sweep_root = Path(sweep_root)
    levels = sorted(p for p in sweep_root.iterdir() if p.is_dir() and p.name.startswith("level_"))

    depth = np.load(depth_path)

    # collect images across sweep
    imgs = []
    labels = []
    for level in levels:
        p = level / image_name
        if not p.exists():
            continue
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w > 420:
            scale = 420 / w
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        imgs.append(rgb)
        name = level.name.replace("level_", "").replace("_x1.00", "")
        # readable label
        parts = name.split("_", 1)
        labels.append(parts[1] if len(parts) > 1 else name)

    n = len(imgs)
    fig = plt.figure(figsize=(2.2 * n, 5.2))
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1.6], hspace=0.08, wspace=0.05)

    # top row: single wide depth image spanning first 2 cells + text explanation
    ax_depth = fig.add_subplot(gs[0, :2])
    im = ax_depth.imshow(depth, cmap="viridis")
    ax_depth.set_xticks([]); ax_depth.set_yticks([])
    ax_depth.set_title("Estimated depth (0.5–5 m)", fontsize=10, loc="left")
    cbar = plt.colorbar(im, ax=ax_depth, shrink=0.7)
    cbar.ax.tick_params(labelsize=8)

    ax_text = fig.add_subplot(gs[0, 2:])
    ax_text.axis("off")
    ax_text.text(
        0.01, 0.72,
        "Underwater SfM Degradation Study",
        fontsize=15, fontweight="bold", va="top",
    )
    ax_text.text(
        0.01, 0.48,
        "Applies a wavelength-dependent Jerlov model to a standard clear-air "
        "photogrammetry dataset,\nthen measures how COLMAP structure-from-motion "
        "degrades across seven water types —\nfrom clearest open ocean to "
        "extremely turbid harbor.",
        fontsize=9.5, va="top", color="#333",
    )

    # bottom row: image strip
    for i, (im, lb) in enumerate(zip(imgs, labels)):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(lb, fontsize=9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path
