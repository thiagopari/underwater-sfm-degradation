"""Render all result figures from the aggregated metrics JSON."""

from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # run from repo root: python legacy_v1/<script>.py


import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt

from src.viz import metrics_plots, image_quality_plot, fine_sweep_plot, PLT_STYLE


def _short(tag: str) -> str:
    if "baseline" in tag:
        return "baseline"
    parts = tag.split("_")
    return parts[2] if len(parts) > 2 else tag


def combined_ablation_plot(sweep_metrics: dict, corr_metrics: dict, out_path: Path) -> Path:
    """Grouped bars per level: uncorrected vs Shades-of-Gray vs Sea-thru."""
    plt.rcParams.update(PLT_STYLE)

    # Group correction metrics by source level
    grouped: dict[str, dict[str, dict]] = {}
    for k, v in corr_metrics.items():
        # keys like "level_05_3C_x1.00__shades_of_gray"
        if "__" not in k:
            continue
        level, method = k.rsplit("__", 1)
        grouped.setdefault(level, {})[method] = v

    # Only compare on levels we actually corrected
    tags = sorted(grouped.keys())
    if not tags:
        # nothing to plot yet
        return out_path

    labels = [_short(t) for t in tags]

    def _reg_rate(d: dict) -> float:
        n = max(d.get("num_input_images", 1), 1)
        return 100.0 * d.get("num_registered", 0) / n

    def _pts(d: dict) -> int:
        return d.get("num_3d_points", 0)

    baseline_reg = [_reg_rate(sweep_metrics.get(t, {})) for t in tags]
    baseline_pts = [_pts(sweep_metrics.get(t, {})) for t in tags]
    sog_reg = [_reg_rate(grouped[t].get("shades_of_gray", {})) for t in tags]
    sog_pts = [_pts(grouped[t].get("shades_of_gray", {})) for t in tags]
    sea_reg = [_reg_rate(grouped[t].get("sea_thru", {})) for t in tags]
    sea_pts = [_pts(grouped[t].get("sea_thru", {})) for t in tags]

    x = np.arange(len(tags))
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.bar(x - w, baseline_reg, width=w, color="#334155", label="uncorrected", edgecolor="white")
    ax1.bar(x,     sog_reg,      width=w, color="#f59e0b", label="Shades-of-Gray", edgecolor="white")
    ax1.bar(x + w, sea_reg,      width=w, color="#22c55e", label="Sea-thru (oracle depth)", edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Images registered (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Does color correction recover registration?")
    ax1.legend(loc="lower left", fontsize=9)

    ax2.bar(x - w, baseline_pts, width=w, color="#334155", label="uncorrected", edgecolor="white")
    ax2.bar(x,     sog_pts,      width=w, color="#f59e0b", label="Shades-of-Gray", edgecolor="white")
    ax2.bar(x + w, sea_pts,      width=w, color="#22c55e", label="Sea-thru (oracle depth)", edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylabel("Reconstructed 3D points")
    ax2.set_title("Does color correction recover reconstruction density?")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path


def main():
    results = Path("results/v1")
    figures = Path("figures/v1")
    figures.mkdir(exist_ok=True)

    sweep_metrics = json.loads((results / "sweep_metrics.json").read_text())
    paths = metrics_plots(sweep_metrics, figures)
    for p in paths:
        print(f"wrote {p}")

    img_path = results / "image_metrics.json"
    if img_path.exists():
        img_metrics = json.loads(img_path.read_text())
        p = image_quality_plot(img_metrics, sweep_metrics, figures / "image_vs_sfm.png")
        print(f"wrote {p}")

    corr_path = results / "correction_metrics.json"
    if corr_path.exists():
        corr = json.loads(corr_path.read_text())
        p = combined_ablation_plot(sweep_metrics, corr, figures / "ablation_correction.png")
        print(f"wrote {p}")

    fine_path = results / "fine_metrics.json"
    if fine_path.exists():
        fine = json.loads(fine_path.read_text())
        p = fine_sweep_plot(fine, figures / "fine_sweep.png")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
