# Tested-on environment (study v2)

- **Hardware:** MacBook Air, Apple Silicon (M4), 16 GB unified memory
- **OS:** macOS (Darwin 25.x, arm64)
- **COLMAP:** 4.0.4, CPU-only build (`brew install colmap`, no CUDA)
- **Python packages:** exactly as pinned in `requirements.txt`

## Runtime (one COLMAP process at a time, 6 threads)

| Stage | Runs | Wall clock |
|---|---|---|
| Reference + 2 clear-air replicates | 3 | ~8 min |
| Main sweep (6 water types x 3 seeds, SIFT) | 18 | ~42 min |
| Standoff (1C, 3C at 1.5 m and 6 m, 2 seeds) | 8 | ~18 min |
| Transition (3C at 3.5/4 m, 1C at 4.5/5 m, 2 seeds) | 8 | ~17 min |
| ALIKED + LightGlue (7 conditions, 1 seed) | 7 | ~90 min |
| Colour-correction ablation (3C, 5C x 3 methods x 2 seeds) | 12 | ~25 min |
| **Total** | **56** | **~3.5 h** |

A SIFT run takes 2-2.5 min and peaks at about 2.4 GB of memory; an ALIKED +
LightGlue run takes about 13 min on CPU. `run_study.py` waits whenever free
memory drops below 25% and skips runs that are already recorded, so an
interrupted stage can be restarted. On macOS, run long stages under
`caffeinate -i` so the machine does not sleep mid-run.

## Other systems

- **Linux + CUDA:** set `use_gpu` to 1 in `src/colmap_runner.py` for a large
  speed-up (SIFT GPU matching, ALIKED/LightGlue on GPU).
- **No MPS / CUDA:** only the one-off DPT depth step uses PyTorch; the cached
  maps in `depth/` are enough for the rest of the study.
