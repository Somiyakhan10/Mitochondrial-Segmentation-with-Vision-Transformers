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

Opens a three-tab interface: **Analyze** (upload + run), **Results**
(overview, once analyses exist), **Mask Correction** (manual segmentation
editing, FR-13 — placeholder until implemented).

## Current state

Most of the deep-learning stages (segmentation inference, health
classification, XAI) are interface stubs — see [api_reference.md](api_reference.md)
for exactly which modules are real vs. stub. Running `analyze`/`batch` today
will get through image loading, validation, and preprocessing, then raise
`NotImplementedError` at the segmentation stage. This is expected until a
segmentation checkpoint is trained (see SRS §6, Phase 2–4).

## Notebooks

`notebooks/` has five tutorial shells (preprocessing, segmentation,
morphometrics, full pipeline, cell-type-specific analysis) with markdown
walkthroughs and empty code cells wired to the current API — fill them in as
each stage is implemented.
