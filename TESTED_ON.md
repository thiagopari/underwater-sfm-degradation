# Tested-on environment

The full pipeline (30 south-building images, 7 turbidity levels + 4 ablation
runs, all metrics + plots + viewer packaging) was validated end-to-end on
the following environment. Full runtime approximately 25 minutes wall-clock.

- **Hardware:** MacBook Air, Apple Silicon (M-series), 16 GB unified memory.
- **OS:** macOS (Darwin 25.5.0 arm64).
- **Python:** 3.10.6 (via pyenv).
- **COLMAP:** 4.0.4, CPU-only build (`brew install colmap`, no CUDA).
- **PyTorch:** 2.13.0 with MPS enabled for DPT depth inference.
- **HuggingFace transformers:** 5.14.1.
- **OpenCV:** 5.0.0 (headless).
- **matplotlib:** 3.10.9.
- **numpy:** 2.2.6.

### Per-phase runtime observed on the tested hardware

| Phase | Runtime |
|-------|---------|
| Environment install | ~2 min |
| Depth estimation (30 images, DPT-SwinV2 on MPS) | ~40 s |
| Turbidity sweep image generation (7 levels × 30 images) | ~1 min |
| COLMAP sweep (7 runs, CPU-only, `max_image_size=1600`) | ~14 min |
| Image-quality metrics | <5 s |
| Color-correction ablation (4 COLMAP runs) | ~10 min |
| Plot generation | ~10 s |
| Viewer packaging | ~10 s |
| **Total end-to-end** | **~28 min** |

### Notes for other systems

- **Linux with a CUDA GPU:** substantially faster. COLMAP with GPU-SIFT
  matching typically 5–10× faster on the same dataset. Remove the
  `--FeatureExtraction.use_gpu 0` and `--FeatureMatching.use_gpu 0`
  overrides in `src/colmap_runner.py` (or pass `use_gpu=True` if I add
  the option).
- **Intel Mac / no MPS:** DPT depth inference will fall back to CPU
  (~5× slower for depth, but depth is <1 min total either way — not
  a bottleneck).
- **Older Python:** the code uses `from __future__ import annotations`
  plus type hints; tested on 3.10. Should run on 3.9+ but not verified.
