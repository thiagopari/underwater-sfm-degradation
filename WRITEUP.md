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

An obvious counter-argument: "the RGB SfM only fails because the image is
blue and SIFT is unhappy — just white-balance the image and it works."

I test this. For each degraded level, I apply Shades-of-Gray white balance
(a robust, well-known baseline that works better than Gray World on most
underwater scenes), then re-run COLMAP. The correction fixes the color cast
but not the loss of contrast or feature-scale information — SfM does not
recover.

This has a clean physical interpretation: white balance is a channel-wise
gain. It can undo the *color* effect of `exp(-Kd_c · z)` (average per
channel), but it cannot undo the *spatial* effect (contrast loss because
close and far pixels attenuate differently). Sea-thru works precisely
because it uses per-pixel depth to invert the spatial effect too — which
is why any pipeline that wants real color underwater needs per-pixel
depth, not just an image.

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
