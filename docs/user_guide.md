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

Opens a four-tab interface, all real:
- **Analyze** — upload an image, run real preprocessing + segmentation +
  morphometrics (if a checkpoint is present), and automatically save the
  run. Shows the mask overlay, a per-pixel confidence heatmap, a region
  area histogram, and an area-vs-circularity scatter plot (FR-37).
- **Results** — lists every saved run (`mitomorph.data.database`'s
  `segmentation_runs` table) with its overlay image and stats; filter by
  condition and see real box-plot comparisons across conditions (FR-36,
  once 2+ conditions have saved runs); select any run to view it in
  detail, including its own scatter plot.
- **Mask Correction** (FR-13) — pick a run, uncheck any false-positive
  regions, preview the corrected overlay live, and save — this
  recomputes morphometrics on the corrected mask and updates the saved
  run (marked `corrected`).
- **Validate** — upload an image plus a known ground-truth mask to
  measure real segmentation accuracy: Dice, IoU, precision, recall, F1,
  a pixel-level confusion matrix heatmap, and a predicted-vs-ground-truth
  overlay (NFR-05, NFR-06). Needs real labeled data — see
  "Getting a validation pair" below.

Note: Results/Mask Correction track segmentation + morphometrics output
only (`segmentation_runs` table) — not the full §5.2
`AnalysisResult`/`analyses` table, which also needs cell-type and health
classification fields that are still stubs.

### Getting a validation pair

The Validate tab needs a microscopy image and its matching ground-truth
mask (a binary TIFF marking true mitochondria) — not just any image. If
you used the Kaggle notebook that trained the current checkpoint
(EPFL/Lucchi dataset), export a fresh slice + its mask:

```python
import tifffile

slice_idx = 100  # any index into the held-out test volume
tifffile.imwrite("/kaggle/working/validate_image.tif", test_images[slice_idx])
tifffile.imwrite("/kaggle/working/validate_mask.tif", test_masks[slice_idx])
```

Download both, then note: the microscopy image must have at least 2
channels (FR-02) even though this dataset is single-channel — duplicate
it before uploading:

```python
import numpy as np
import tifffile

image = tifffile.imread("validate_image.tif")
tifffile.imwrite(
    "validate_image_packaged.ome.tif",
    np.stack([image, image]),
    metadata={"axes": "CYX", "Channel": {"Name": ["Tom20", "Tom20_duplicate"]}},
)
```

Upload `validate_image_packaged.ome.tif` and `validate_mask.tif` (unmodified)
into the Validate tab's two file slots.

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
averaged across the full held-out test volume (165 slices) during
training. Spot-checked afterward via the dashboard's Validate tab on an
individual held-out slice: Dice 0.95, IoU 0.90, Precision 0.92, Recall
0.98 — consistent with (and on this slice, better than) the training-time
average. It has **not** been validated on fluorescence microscopy images
(Tom20/COX IV/MitoTracker), which is what this pipeline is actually
built for — expect it to detect nothing on fluorescence input, since EM
and fluorescence have very different intensity/texture statistics.
Fine-tuning on real annotated lab data (`mitomorph train`, once that
data exists) is what would make it usable for real analysis.

## Notebooks

`notebooks/` has five tutorial shells (preprocessing, segmentation,
morphometrics, full pipeline, cell-type-specific analysis) with markdown
walkthroughs and empty code cells wired to the current API — fill them in as
each stage is implemented.
