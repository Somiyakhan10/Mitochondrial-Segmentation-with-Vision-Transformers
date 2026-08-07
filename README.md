# MitoMorph — Mitochondrial Morphology Analysis Pipeline

Deep-learning pipeline for automated segmentation, classification, and
morphometric quantification of mitochondria in spinal cord tissue
microscopy images, built to accelerate spinal cord injury (SCI)
mitochondrial dysfunction research. See [docs/srs.md](docs/srs.md) for
the full requirements document this project implements.

## Status

Segmentation now runs on a real trained model: a U-Net/ResNet34
(`data/models/segmentation_unet.pt`) trained on the EPFL/Lucchi
hippocampus electron microscopy mitochondria dataset. Validated against
real held-out ground truth via the dashboard's Validate tab: Dice 0.95,
IoU 0.90, Precision 0.92, Recall 0.98 on a held-out EM test slice. No
lab-specific fluorescence data/annotations exist yet, so this
checkpoint has not been validated against this pipeline's primary
input modality — it's proof that the train → predict → validate →
display path works correctly end-to-end, not a validated detector for
Tom20/COX IV/MitoTracker images.

The dashboard (`streamlit run dashboard/app.py`) has four working tabs:
Analyze (upload, segment, view morphometrics + confidence/shape
charts), Results (browse saved runs, filter and compare by condition
with real box plots), Mask Correction (reject false-positive regions),
and Validate (confusion matrix + Dice/IoU/precision/recall/F1 against
a ground-truth mask).

Cell-type classification, health classification, XAI, and PDF
reporting are still fully-typed stubs that raise `NotImplementedError`.
See [docs/api_reference.md](docs/api_reference.md) for the real/stub
status of every module.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements-dev.txt
pip install -e .
```

## Quickstart

```bash
mitomorph analyze path/to/image.tif --animal-id M123 --condition SCI --time-point "6 weeks"
mitomorph batch data/raw/ --output-dir data/processed/
streamlit run dashboard/app.py
pytest
```

`data/models/` is gitignored (trained weights aren't committed) — if you
have `segmentation_unet.pt`, place it there and the CLI/dashboard will
pick it up automatically. See [docs/user_guide.md](docs/user_guide.md)
for full CLI/dashboard usage.

## Project structure

```
config/                 Default analysis parameters (config/default_config.yaml)
src/mitomorph/           
  preprocessing/         Image I/O, validation, channel ID, normalization, denoising, Z-stacks
  segmentation/           U-Net/Attention U-Net models, training, inference, checkpointing, metrics
  celltype/               Neuronal vs. non-neuronal classification
  morphometrics/          Single-mitochondrion + network features, dysfunction indices, QC
  classification/         Health classifier, treatment-response predictor
  xai/                     Grad-CAM, SHAP, uncertainty estimation
  validation/              Cross-validation, MLflow tracking, benchmarking
  reporting/               Figures, temporal plots, PDF reports, CSV/Excel/JSON export
  data/                    Result schema, time-series model, SQLite metadata store
  integration/             Lab database client interface
  pipeline.py              Orchestrates every stage above
  cli.py                   analyze / batch / train / report subcommands
dashboard/app.py         Streamlit UI: Analyze, Results, Mask Correction, Validate tabs
scripts/                 Thin CLI wrapper scripts
notebooks/                Tutorial notebook shells
tests/                    Unit + integration tests
docs/                     SRS, API reference, user guide
```

## SRS traceability

| Area | FR/NFR IDs | Key modules |
|---|---|---|
| Image I/O & preprocessing | FR-01–FR-07 | `preprocessing/` |
| Segmentation | FR-08–FR-14 | `segmentation/` |
| Cell-type specificity | FR-15–FR-18 | `celltype/`, `preprocessing/channel_utils.py` |
| Morphometrics | FR-19–FR-25 | `morphometrics/` |
| Classification & prediction | FR-26–FR-30 | `classification/` |
| Explainable AI | FR-31–FR-34 | `xai/` |
| Reporting & visualization | FR-35–FR-40 | `reporting/`, `dashboard/` |
| Data management | FR-41–FR-44 | `data/`, `integration/` |
| Performance | NFR-01–NFR-04 | `pipeline.py`, `segmentation/infer.py` |
| Accuracy | NFR-05–NFR-07 | `validation/`, `morphometrics/quality_control.py` |
| Usability | NFR-08–NFR-11 | `notebooks/`, `cli.py`, `docs/` |
| Maintainability | NFR-12–NFR-15 | package-wide; `.github/workflows/`, `tests/` |

Full per-module status is in [docs/api_reference.md](docs/api_reference.md).

## Testing

```bash
pytest
black --check src/ tests/
flake8 src/ tests/
```
