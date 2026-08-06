# User Guide

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements-dev.txt
pip install -e .
```

## CLI

Once installed, the `mitomorph` console script (or `python -m mitomorph.cli`) exposes four subcommands:

```bash
# Analyze a single image
mitomorph analyze path/to/image.tif --animal-id M123 --condition SCI --time-point "6 weeks"

# Batch-process a directory
mitomorph batch data/raw/ --output-dir data/processed/

# Fine-tune the segmentation model (once training data exists)
mitomorph train --train-dir data/raw/train --val-dir data/raw/val --checkpoint-dir data/models

# Build a PDF report from stored results
mitomorph report data/processed/ --output data/processed/reports/summary.pdf
```

Every subcommand accepts `--config path/to/config.yaml` to override `config/default_config.yaml`.

## Dashboard

```bash
streamlit run dashboard/app.py
```

Opens a three-tab interface, all real:
- **Analyze** — upload an image, run real preprocessing + segmentation +
  morphometrics (if a checkpoint is present), and automatically save the
  run.
- **Results** — lists every saved run (`mitomorph.data.database`'s
  `segmentation_runs` table) with its overlay image and stats; select
  one to view it in detail.
- **Mask Correction** (FR-13) — pick a run, uncheck any false-positive
  regions, preview the corrected overlay live, and save — this
  recomputes morphometrics on the corrected mask and updates the saved
  run (marked `corrected`).

Note: Results/Mask Correction track segmentation + morphometrics output
only (`segmentation_runs` table) — not the full §5.2
`AnalysisResult`/`analyses` table, which also needs cell-type and health
classification fields that are still stubs.

## Current state

Segmentation is real if `data/models/segmentation_unet.pt` is present
(gitignored — not committed). With it, `analyze`/`batch` will get through
image loading, validation, preprocessing, segmentation, and morphometric
feature extraction, then raise `NotImplementedError` at cell-type
classification (the next stub stage). Without a checkpoint, it stops one
stage earlier, at segmentation. See [api_reference.md](api_reference.md)
for exactly which modules are real vs. stub.

The current checkpoint was trained externally on Kaggle (GPU), using the
`UNetResNet34` architecture in `segmentation/models/unet.py` against the
public EPFL/Lucchi EM mitochondria dataset — Val Dice 0.80, Val IoU 0.73
on that dataset's held-out test volume. It has **not** been validated on
fluorescence microscopy images (Tom20/COX IV/MitoTracker), which is what
this pipeline is actually built for — expect it to detect nothing on
fluorescence input, since EM and fluorescence have very different
intensity/texture statistics. Fine-tuning on real annotated lab data
(`mitomorph train`, once that data exists) is what would make it usable
for real analysis.

## Notebooks

`notebooks/` has five tutorial shells (preprocessing, segmentation,
morphometrics, full pipeline, cell-type-specific analysis) with markdown
walkthroughs and empty code cells wired to the current API — fill them in as
each stage is implemented.
