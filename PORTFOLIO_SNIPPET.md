# Portfolio integration

Drop-in HTML for the portfolio project card (matches the existing project-card
style on thiagopari.github.io/ThiagoPari):

```html
<a href="https://github.com/thiagopari/underwater-sfm-degradation" target="_blank" style="text-decoration: none; color: inherit;">
    <div class="project-card">
        <div class="project-image">
            <img src="images/underwater-sfm-hero.png" alt="Underwater SfM Degradation Study">
        </div>
        <div class="project-content">
            <h3 class="project-title">Underwater SfM Degradation Study</h3>
            <p class="project-description">
                Reproducible benchmark quantifying how wavelength-dependent Jerlov
                underwater optics degrade COLMAP structure-from-motion reconstruction
                across seven water types. Includes monocular depth via DPT, a
                physically-motivated image formation model, a color-correction ablation,
                and an interactive HTML viewer.
            </p>
            <div class="tech-tags">
                <span class="tag">Python</span>
                <span class="tag">COLMAP</span>
                <span class="tag">SfM</span>
                <span class="tag">Computer Vision</span>
                <span class="tag">Sea-thru</span>
                <span class="tag">DPT</span>
                <span class="tag">Underwater</span>
            </div>
        </div>
    </div>
</a>
```

## LinkedIn/Twitter post draft

Short version (Twitter/X, ~280 chars):

> Shipped a small research project: a reproducible benchmark quantifying how
> underwater turbidity breaks structure-from-motion. Applies wavelength-dependent
> Jerlov optics to a standard photogrammetry dataset, runs COLMAP across 7 water
> types. Code + interactive viewer: <link>

Longer version (LinkedIn):

> A short weekend project I've been sitting on: **Underwater SfM Degradation Study**.
>
> The question: at what point does turbidity flip stereo/RGB structure-from-motion
> from "reduced quality" to "unusable"? The underwater robotics literature has
> vendor accuracy claims in clear water and sonar-vendor coverage claims in
> turbid water, but not much in between.
>
> The project applies a physically-motivated Jerlov underwater image formation
> model (Sea-thru-like) to a standard clear-air photogrammetry dataset
> (COLMAP's south-building), sweeps seven water types from clearest open ocean
> to extremely turbid harbor, and measures how the COLMAP incremental SfM
> pipeline responds. Everything is reproducible from a single Makefile in about
> 20 minutes on a laptop.
>
> Bonus: a small color-correction ablation showing that undoing the color cast
> with white balance does *not* recover reconstruction quality — the pipeline
> loses feature-scale information to backscatter that no channel-wise gain can
> put back.
>
> Code + interactive viewer: <link>
