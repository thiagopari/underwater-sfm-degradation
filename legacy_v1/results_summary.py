"""Print a markdown table of sweep + correction metrics; also generate a
short interpretation paragraph that can be pasted into the README.
"""

from __future__ import annotations
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))  # run from repo root: python legacy_v1/<script>.py


import json
import sys
from pathlib import Path


TYPE_DESCS = {
    "baseline":  "clear air (reference)",
    "I":  "clearest open ocean",
    "II": "clear coastal water",
    "III":"turbid coastal water",
    "1C": "harbor (moderately turbid)",
    "3C": "harbor (very turbid)",
    "5C": "harbor (extremely turbid)",
}


def _short(tag: str) -> str:
    if "baseline" in tag:
        return "baseline"
    parts = tag.split("_")
    return parts[2] if len(parts) > 2 else tag


def _row(tag: str, m: dict) -> str:
    short = _short(tag)
    n_reg = m.get("num_registered", 0)
    n_in = m.get("num_input_images", 30)
    feats = m.get("features_per_image", [])
    mean_feats = int(sum(feats) / len(feats)) if feats else 0
    n_pts = m.get("num_3d_points", 0)
    reproj = m.get("mean_reproj_error", 0.0)
    return (
        f"| **{short}** | {TYPE_DESCS.get(short, short)} | "
        f"{n_reg} / {n_in} ({100*n_reg/max(n_in,1):.0f}%) | "
        f"{mean_feats:,} | {n_pts:,} | {reproj:.3f} |"
    )


def main():
    sweep = json.loads(Path("results/v1/sweep_metrics.json").read_text())
    tags = sorted(sweep.keys())

    print("| Level | Water condition | Registered | Mean features/img | 3D points | Mean reproj err (px) |")
    print("|-------|-----------------|------------|-------------------|-----------|----------------------|")
    for t in tags:
        print(_row(t, sweep[t]))

    corr_path = Path("results/v1/correction_metrics.json")
    if corr_path.exists():
        corr = json.loads(corr_path.read_text())
        # Group by source level
        grouped: dict[str, dict[str, dict]] = {}
        for k, v in corr.items():
            if "__" not in k:
                continue
            level, method = k.rsplit("__", 1)
            grouped.setdefault(level, {})[method] = v

        print()
        print("### Color-correction ablation")
        print()
        print("| Level | Method | Registered | 3D points | Δ points vs uncorrected |")
        print("|-------|--------|------------|-----------|-------------------------|")
        for level in sorted(grouped.keys()):
            base = sweep.get(level, {})
            n_pts_b = base.get("num_3d_points", 0)
            for method in sorted(grouped[level].keys()):
                c = grouped[level][method]
                n_reg = c.get("num_registered", 0)
                n_pts = c.get("num_3d_points", 0)
                delta = n_pts - n_pts_b
                sign = "+" if delta >= 0 else ""
                print(f"| **{_short(level)}** | {method} | {n_reg}/{c.get('num_input_images', 30)} | {n_pts:,} | {sign}{delta:,} |")


if __name__ == "__main__":
    main()
