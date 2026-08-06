# API Reference

Per-module summary of the `mitomorph` package. **Real** modules have working,
tested logic. **Stub** modules define the intended interface (types,
docstrings, FR/NFR traceability) but raise `NotImplementedError` in the body
until trained models or annotated lab data exist to build them against.

## Cross-cutting

| Module | Status | Purpose |
|---|---|---|
| `mitomorph.exceptions` | Real | `MitoMorphError` hierarchy used across the package |
| `mitomorph.logger` | Real | `get_logger(name)` / `configure_logging()` |
| `mitomorph.config.loader` | Real | `load_config`, `validate_config`, `merge_overrides` (FR-42) |
| `mitomorph.pipeline` | Real (wiring) | `MitoPipeline.run()` orchestrates every stage below |
| `mitomorph.cli` | Real | `analyze` / `batch` / `train` / `report` subcommands (FR-07, NFR-09) |

## Preprocessing (`mitomorph.preprocessing`)

| Module | Status | FR |
|---|---|---|
| `io` | Real | FR-01, FR-02 — `load_image()` for TIFF/OME-TIFF/CZI |
| `validators` | Real | `validate_image()`, `validate_metadata()` |
| `channel_utils` | Real | FR-02, FR-15 — marker-name-based channel identification |
| `normalization` | Real | FR-04 — `zscore_normalize()`, `percentile_normalize()` |
| `zstack` | Real | FR-06 — `max_intensity_projection()`, `focus_stack()` |
| `illumination` | Stub | FR-03 — needs a validated flat-field reference or tuned rolling-ball radius |
| `denoising` | Stub | FR-05 — needs tuning against real noise characteristics |

## Segmentation (`mitomorph.segmentation`)

| Module | Status | FR |
|---|---|---|
| `metrics` | Real | FR-14, NFR-05 — `iou_score()`, `dice_score()` |
| `checkpoint` | Real | NFR-14 — `save_checkpoint()`, `load_checkpoint()`, `resume_training()` |
| `models.unet` | Stub | FR-08 — `UNetResNet34` |
| `models.attention_unet` | Stub | FR-09 — `AttentionUNet` |
| `augmentations` | Stub | FR-10, FR-11 — MONAI training/validation transforms |
| `model_zoo` | Stub (registry is real) | FR-10 — `list_available_models()` real; `download_pretrained_weights()` stub |
| `train` | Stub | FR-11 — fine-tuning entrypoint |
| `infer` | Stub | FR-12 — `segment()` |

## Cell type (`mitomorph.celltype`)

| Module | Status | FR |
|---|---|---|
| `classifier` | Stub | FR-15–FR-17 — `classify_neuronal()` |

## Morphometrics (`mitomorph.morphometrics`)

| Module | Status | FR |
|---|---|---|
| `single_features` | Real | FR-19, FR-20 — `extract_single_features()` |
| `dysfunction_indices` | Real | FR-22–FR-24 — fragmentation/swelling/density |
| `quality_control` | Real | NFR-07 — `validate_features()` |
| `network_features` | Stub | FR-21 — skeletonization/branch-point detection |

## Classification (`mitomorph.classification`)

| Module | Status | FR |
|---|---|---|
| `health_classifier` | Stub (pipeline construction is real) | FR-26–FR-28 |
| `outcome_predictor` | Stub | FR-29, FR-30 |

## Explainable AI (`mitomorph.xai`)

| Module | Status | FR |
|---|---|---|
| `gradcam` | Stub | FR-31 |
| `shap_analysis` | Stub | FR-32, FR-33 |
| `uncertainty` | Stub | FR-34 |

## Validation (`mitomorph.validation`)

| Module | Status | NFR |
|---|---|---|
| `cross_validation` | Real | NFR-05, NFR-06 — `make_folds()` |
| `performance_tracker` | Stub (`meets_targets()` real) | NFR-05, NFR-06 |
| `benchmarking` | Stub | — |

## Reporting (`mitomorph.reporting`)

| Module | Status | FR |
|---|---|---|
| `export` | Real | FR-43 — `to_csv()`, `to_excel()`, `to_json()` |
| `figures` | Stub | FR-35–FR-37 |
| `temporal_plots` | Stub | FR-38 |
| `pdf_report` | Stub | FR-39 |

## Data (`mitomorph.data`)

| Module | Status | FR |
|---|---|---|
| `schema` | Real | §5.2 — `AnalysisResult` and related dataclasses |
| `temporal` | Real | §1.3 — `TimeSeriesData` |
| `database` | Real | FR-41 — SQLite-backed `AnalysisDatabase` |

## Integration (`mitomorph.integration`)

| Module | Status | FR |
|---|---|---|
| `lab_api` | Stub | FR-44 (Low priority) |
