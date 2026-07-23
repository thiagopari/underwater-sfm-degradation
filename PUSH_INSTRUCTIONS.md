# Pushing this to GitHub

The repo is currently local-only at `~/Projects/underwater-sfm-degradation/`.
To publish it:

## 1. Create the remote repo

Two options:

**Option A — via GitHub web UI** (no gh CLI required):
1. Go to https://github.com/new
2. Repository name: `underwater-sfm-degradation`
3. Owner: `thiagopari` (or your preferred account)
4. Public
5. Do **not** check "Initialize with README" — we already have one
6. Click "Create repository"
7. Copy the repo URL (e.g., `https://github.com/thiagopari/underwater-sfm-degradation.git`)

**Option B — with gh CLI**:
```bash
brew install gh
gh auth login
gh repo create thiagopari/underwater-sfm-degradation --public --source=. --remote=origin --push
```
Option B does the whole thing in one shot.

## 2. Push (if you used option A)

```bash
cd ~/Projects/underwater-sfm-degradation
git remote add origin https://github.com/thiagopari/underwater-sfm-degradation.git
git branch -M main
git push -u origin main
```

## 3. (Optional) Enable GitHub Pages for the interactive viewer

The `viewer/` directory is a static HTML site. To serve it:

1. In the repo's Settings → Pages
2. Source: Deploy from a branch
3. Branch: `main`, folder: `/viewer` (or `/` and add a redirect)
4. Save

After ~1 minute, the viewer will be live at
`https://thiagopari.github.io/underwater-sfm-degradation/`

## 4. Add to your portfolio

See `PORTFOLIO_SNIPPET.md` for a ready-to-paste project card and LinkedIn/X
post drafts.

## What's in the repo

- `README.md` — main writeup with figures and results tables.
- `WRITEUP.md` — narrative long-form version.
- `PORTFOLIO_SNIPPET.md` — portfolio card + social post drafts.
- `PUSH_INSTRUCTIONS.md` — (this file — you can delete after pushing).
- `Makefile` — one-command reproduction of the full pipeline.
- `src/` — modules: `jerlov`, `depth`, `sweep`, `colmap_runner`, `color_correction`, `image_metrics`, `viz`, `hero_figure`.
- `tests/` — pytest unit tests for the Jerlov model + color correction.
- `figures/` — output figures for the README.
- `results/*.json` — aggregated metrics (small, kept in git).
- `viewer/` — static HTML interactive viewer (deployable to GitHub Pages).
- `.github/workflows/ci.yml` — CI that runs the unit tests on push.

**Excluded** from git (see `.gitignore`):
- `data/` — 30 south-building images (regenerable from COLMAP public release).
- `depth/` — cached depth maps (regenerable from `data/` via `make depth`).
- `sweep/`, `sweep_corrected/`, `sweep_fine/` — degraded image outputs.
- `results/level_*/` — per-level COLMAP databases (large).
- `.venv/`, `__pycache__/`.
