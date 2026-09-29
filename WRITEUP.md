# From v1 to v2: what was wrong, and what changed

The first version of this study (July 2026, kept in `legacy_v1/`, `results/v1/`,
`figures/v1/`) concluded that off-the-shelf RGB SfM "survives through Jerlov 1C"
and that the usable band was "wider than sonar-vendor marketing implies". A
multi-reviewer audit in September 2026 found that those conclusions came from
the simulator rather than the water. This file records what the audit found and
how v2 addresses each point, because the corrections are part of the result.

## What the audit found

| # | Problem in v1 | Why it matters | v2 fix |
|---|---|---|---|
| 1 | Attenuated along the line of sight with the **diffuse** coefficient Kd (`src/jerlov.py`) | Kd describes how sunlight dims with depth; the camera path needs **beam** attenuation c = a + b, typically several times larger in coastal water. Degradation was understated. A quick check with 2.5 x Kd already collapsed 1C to 25/30 cameras. | Per-wavelength c from measured IOPs (Williamson & Hollins 2022) |
| 2 | Coastal types had blue attenuating **less** than green | Backwards: dissolved organic matter absorbs blue in coastal water, which turns it green-yellow, not blue | Spectral rendering from measured a(lambda), b(lambda); coastal images now look coastal |
| 3 | Veiling colours hand-picked; direct and backscatter tied to one coefficient | Not physical; contradicts Akkaynak & Treibitz (2018) | Single-scattering veiling estimate b_b E / (2c); wideband integration makes beta_D != beta_B |
| 4 | **Each image's depth min-max normalised to 0.5-5 m**, and DPT inverse depth used as linear depth | The same wall had a different range in every view; the "5 m" choice decided where the cliff appeared | DPT fitted to the clear-air COLMAP model (scale + shift in inverse depth, 82% inliers, 2.7% median error); one global scale sets the standoff; standoff is swept |
| 5 | No sensor noise | The main way turbidity hurts matching is fewer photons, more gain, more noise. Without it, SIFT's affine invariance made the degradation nearly free. | Poisson shot noise + read noise, auto-exposure gain, 8-bit sRGB, JPEG |
| 6 | Only registration and point counts | "Registered" does not mean "right" | Camera ATE and rotation error against the clear-air reference after similarity alignment; "accurate pose" = < 10 cm and < 2 deg |
| 7 | One COLMAP run per condition, no seeds | Differences could have been noise | 3 seeds (main sweep), 2 (standoff, ablation); clear-air replicates show the run-to-run floor is 0.3 mm |
| 8 | "Oracle Sea-thru rescues 5C" presented as a headline | The oracle inverted the exact model that created the damage, so it was true by construction | Kept only as a labelled upper bound, next to a Sea-thru-style correction whose parameters are **estimated** from the image |
| 9 | Unsupported claims about sonar vendors and industry practice; "a cliff" from 7 coarse points | Not evidence | Removed |
| 10 | Factual slips in the old writeup (128 vs 30 images, "registration starts to hurt at 3C" when it was 30/30, "nothing hand-tuned") | Credibility | This file replaces it |
| 11 | Data step not reproducible (no URL, no checksums), unpinned deps, `make all` skipped stages | Nobody could rerun it | `scripts/get_data.py` (URL + SHA-256 per frame), pinned `requirements.txt`, `make study` runs every stage, resumable |

## Found in the independent review of v2 (and fixed)

- **Rotation error used a centres-only alignment.** A near-collinear camera path
  left a roll ambiguity, which made one 5C run look like 10 wrong cameras. Now
  rotations are compared after the best rotation-only alignment, plus an
  alignment-free relative-rotation error (`src/geometry.py`), and every run was
  rescored from its saved model (`run_study.py rescore`).
- **Only the largest model was counted.** The "drop" to 16–17 cameras at τ ≈ 2.7
  turned out to be the map splitting into two sub-models with 29–30 cameras
  registered in total. Both counts are now reported, and the findings describe
  three regimes (complete map, split map, lost cameras).
- **Veiling light probably too weak, and the τ collapse partly by construction.**
  A sensitivity stage (veil x10/x30, photon budget x4/x0.25) shows how far the
  transition moves, and the README now says that collapsing onto optical depth is
  expected in this model rather than discovered.
- **Number slips** ("within 10 mm", run count, runtime) corrected.

## What stayed

The question, the dataset (COLMAP's south-building, 30 frames), COLMAP
incremental SfM with relaxed SIFT thresholds, the idea of a controlled sweep, and
the colour-correction ablation. The v1 numbers are preserved in `results/v1/` so
the difference can be checked.

## What v2 still does not do

See *Limitations* in the README. The big ones: the scene is a terrestrial
building rather than a hull or seabed, lighting is ambient only (no vehicle
lights), forward scatter uses one blur width, and there is no validation on real
underwater footage yet.
