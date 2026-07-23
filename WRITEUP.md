# When does underwater turbidity break SfM?

*A weekend research project, motivated by a gap in the underwater robotics
literature.*

## The gap

If you go looking for guidance on RGB-based 3D reconstruction underwater,
you find two disjoint conversations:

- **Optical-inspection vendors** (Voyis, Blue Atlas, others) publish accuracy
  numbers for clean water: sub-millimeter geometry at close range with the
  right lighting.
- **Sonar-inspection vendors** (IQUA, Coda Octopus) publish coverage numbers
  for turbid harbors: continuous 3D even when visibility drops below a meter.

What's missing from both is the middle ground: a quantitative answer to
"at what turbidity does off-the-shelf RGB structure-from-motion actually
break?" Everyone assumes it breaks *somewhere* between "clear ocean" and
"harbor sludge," but the boundary isn't in the literature — and it matters,
because it's the boundary at which you switch product architecture from
stereo-first to sonar-first.

This project is a small, controlled benchmark that produces a first-pass
answer.

## Approach

Take a standard clear-air photogrammetry dataset (COLMAP's `south-building`,
128 architectural photos with a known-good ground-truth reconstruction).
Simulate what those images would look like underwater at seven different
Jerlov water types, from clearest open ocean to extremely turbid harbor.
Then run the exact same COLMAP SfM pipeline on each degraded set and see
what happens.

There are three moving pieces:

1. **Depth estimation.** Every image needs a per-pixel depth for the
   underwater optics model. I use DPT-SwinV2 (Vision Transformer for dense
   prediction) via HuggingFace — a small, fast model that runs in ~1s per
   image on Apple Silicon MPS. The depth is not metric; I rescale it to a
   plausible 0.5–5m underwater standoff range so the spatial variation
   drives visible per-pixel attenuation.

2. **Jerlov image formation.** For each pixel with depth `z` and per-channel
   diffuse attenuation coefficient `Kd`, apply the Sea-thru-style formation
   model:

   ```
   I_c = J_c · exp(-Kd_c · z)  +  B∞_c · (1 - exp(-Kd_c · z))
   ```

   The first term is the direct signal attenuated by absorption + scattering
   along the viewing ray; the second is the veiling light (backscatter)
   accumulating with distance. `Kd_c` values are per Jerlov's 1976
   classification at representative R/G/B wavelengths (650/550/450 nm), and
   `B∞_c` is a blue-green veiling color that darkens and shifts hue as water
   turbidity increases.

3. **COLMAP incremental SfM.** For each of the seven degraded datasets:
   SIFT feature extraction, exhaustive matching, incremental mapping.
   Feature thresholds are relaxed so *some* features survive even in the
   worst degradation — otherwise the study collapses to "the pipeline finds
   no features."

Per level I capture: images successfully registered, reconstructed 3D
points, mean SIFT features per image, mean reprojection error, and the
per-image feature count distribution. Numbers are in
`results/sweep_metrics.json`.

## Why this is a legitimate study, not a demo

Two design choices matter:

- The reference reconstruction (baseline, clear air) is *the same COLMAP
  pipeline* run on the *undegraded* south-building images. So every
  degraded reconstruction is compared apples-to-apples against a known-good
  baseline produced by the same code.
- The Jerlov coefficients are published, not hand-tuned. Nothing in the
  degradation pipeline is fit to the observed reconstruction results — the
  optics model is fixed at Jerlov's canonical values and only the water
  type changes across the sweep.

That gives the numbers scientific weight rather than "here's a slider that
makes the picture blurrier."

## Color-correction ablation

Two natural counter-arguments to the finding above:

1. "SfM fails because the image is blue — just white-balance it and it works."
2. "SfM fails because we don't have Sea-thru — apply Sea-thru with the true
   depth and Kd and it works."

I test both, with the strongest possible version of each: Shades-of-Gray
(a robust classical white balance) and Sea-thru with **oracle** depth and
oracle Kd (i.e., the exact parameters used to synthesize the degradation).

Results at the failure-region levels:

| Level | uncorrected | +Shades-of-Gray | +Sea-thru (oracle) |
|-------|-------------|-----------------|--------------------|
| 3C | 30/30, 8065 pts | 30/30, **3994 pts** ↓50% | 30/30, 7000 pts ↓13% |
| 5C | 24/30, 961 pts | **4/30, 115 pts** collapse | **29/30, 1813 pts** ↑90% |

Two things jump out:

**Naive white balance actively hurts, and it hurts worse the more turbid
the water.** At 3C it halves the point count. At 5C it drops the
registration rate from 24/30 to 4/30. Physically: per-channel gain
amplifies noise in the low-signal (red) channel more than it amplifies
signal, and SIFT descriptors — which are gradient-based and locally
normalized — become less discriminative. The matcher rejects more
pairs. The reconstruction thins out.

**Depth-aware physics-based Sea-thru genuinely recovers 5C.** From 24
frames registered → 29, and from 961 points → 1813 (almost a 2× recovery
in reconstruction density). This is the strongest positive result in
the study: at the point where the uncorrected pipeline is nearly
useless, oracle Sea-thru brings it back to something usable — because
inverting the *spatial* attenuation term with per-pixel depth is
fundamentally different from just rescaling the global color.

The industry lesson is precise: **turbid-water RGB reconstruction needs
depth-aware color correction, not just white balance.** Stereo gives you
that depth for free, which is why stereo-RGB underwater rigs (Voyis,
Deep Trekker) can meaningfully target turbid coastal work. Sonar+RGB
architectures cannot — the sonar depth is at the wrong resolution and
not per-pixel-aligned to the camera — which is why IQUA-style
sonar-primary systems don't try Sea-thru at all; they use sonar for
geometry and RGB opportunistically.

*Caveat: real Sea-thru estimates Kd from the image itself via
dark-channel priors. That estimation step adds error the oracle
version here doesn't have. The oracle result is an **upper bound** on
what a real-world Sea-thru pipeline could deliver.*

## The surprise

Here's the finding I didn't expect: **SIFT-based incremental SfM survives
much further into the turbidity sweep than the image degradation would
suggest.** By Jerlov III the RMS contrast has dropped ~58% and the image
looks obviously bad, but the reconstruction is still ~30/30 frames
registered with a full-density point cloud. Feature extraction and
matching keep working long past the point at which the picture would
fail a human "does this look usable" test.

The reason is straightforward. SIFT descriptors are constructed from
gradient orientation histograms normalized to unit length — a global
color shift and even substantial contrast reduction don't move the
descriptors much. The break happens when *local* edges get washed out
below the noise floor, which requires an order-of-magnitude reduction in
Laplacian variance — and that only happens deep into harbor-water
territory.

## What this means for underwater robots

For a first-pass sensor architecture decision:

- **Clear ocean, clear coastal water (Jerlov I–II):** stereo-RGB SfM is
  fine. Modest quality loss, near-baseline registration.
- **Turbid coastal water (Jerlov III):** image quality drops sharply but
  reconstruction quality does not. This is a wider "usable" band than
  most marketing collateral would have you believe — with the caveat
  that a real hull is more feature-poor than an architectural facade.
- **Harbor water (Jerlov 1C+):** approaching the boundary. Registration
  starts to hurt at 3C. Some form of color correction and/or targeted
  lighting is required, and by 5C SfM collapses regardless.

The practical takeaway isn't "RGB always works." It's that the *industry
common wisdom* — "RGB dies fast underwater" — overstates the effect for
the SfM feature-matching pipeline. The image looks worse than the
reconstruction gets. This has product implications: a stereo-RGB rig
with strong lighting probably covers more of the practical operating
envelope than a sonar-first vendor will tell you. It also has planning
implications: you can't judge SfM viability by eyeballing the video
feed.

## Reproducing

Full instructions in [`README.md`](README.md). Short version:

```
make venv    # once
make all     # end-to-end pipeline, ~20 min on M-series laptop
```
