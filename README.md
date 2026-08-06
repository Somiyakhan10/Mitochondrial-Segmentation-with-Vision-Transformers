# MitoMorph — Mitochondrial Morphology Analysis Pipeline

Deep-learning pipeline for automated segmentation, classification, and
morphometric quantification of mitochondria in spinal cord tissue
microscopy images, built to accelerate Prof. Patel's SCI mitochondrial
dysfunction research. See [docs/srs.md](docs/srs.md) for the full
requirements document this project implements.

## Status

This is a **project scaffold**, not a trained pipeline. No lab
data/annotations exist yet, so modules that don't depend on trained
weights (image I/O, normalization, morphometric feature extraction,
config, database, export, checkpointing) are implemented and tested;
everything requiring a trained model or annotated data (segmentation
inference, health classification, XAI, PDF/figure rendering) is a
fully-typed stub that raises `NotImplementedError`. See
[docs/api_reference.md](docs/api_reference.md) for the real/stub status of
every module.

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

See [docs/user_guide.md](docs/user_guide.md) for full CLI/dashboard usage.

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
dashboard/app.py         Streamlit UI skeleton
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
