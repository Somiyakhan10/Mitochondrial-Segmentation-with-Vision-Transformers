"""Streamlit dashboard (FR-13, FR-40, NFR-10).

Page structure, styling, upload widget, section layout, the
segmentation + morphometrics preview, run persistence (Results tab),
region-level mask correction (Mask Correction tab), and ground-truth
validation (Validate tab: confusion matrix, Dice/IoU/precision/recall/F1)
are all real (a trained checkpoint is loaded from
data/models/segmentation_unet.pt if present). Cell-type classification
and health scoring are still stubs pending further training data.
"""

from __future__ import annotations

import io
import tempfile
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tifffile
from skimage.measure import label, regionprops

from mitomorph.data.database import AnalysisDatabase
from mitomorph.data.segmentation_runs import load_run_artifacts, save_run_artifacts
from mitomorph.morphometrics.dysfunction_indices import fragmentation_index, mitochondrial_density
from mitomorph.morphometrics.single_features import extract_single_features
from mitomorph.preprocessing.channel_utils import extract_mitochondrial_channel
from mitomorph.preprocessing.io import load_image
from mitomorph.preprocessing.normalization import zscore_normalize
from mitomorph.preprocessing.validators import validate_image
from mitomorph.reporting.figures import (
    plot_condition_comparison,
    plot_confusion_matrix,
    plot_feature_correlation,
    plot_feature_histogram,
)
from mitomorph.segmentation.infer import load_model, segment
from mitomorph.segmentation.metrics import (
    confusion_matrix,
    dice_score,
    f1_score,
    iou_score,
    precision_score,
    recall_score,
)

st.set_page_config(page_title="MitoMorph", layout="wide")

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = REPO_ROOT / "data" / "models" / "segmentation_unet.pt"
DB_PATH = REPO_ROOT / "data" / "processed" / "mitomorph.db"
RUNS_DIR = REPO_ROOT / "data" / "processed" / "runs"
OVERLAYS_DIR = REPO_ROOT / "data" / "processed" / "overlays"
OVERLAY_FACECOLOR = "#121018"


@st.cache_resource
def _get_model():
    if not CHECKPOINT_PATH.exists():
        return None
    return load_model(CHECKPOINT_PATH)


@st.cache_resource
def _get_db() -> AnalysisDatabase:
    return AnalysisDatabase(DB_PATH)


def _run_segmentation_preview(uploaded_file):
    """Run preprocessing + real segmentation + morphometrics on an uploaded image."""
    model = _get_model()
    if model is None:
        return None

    suffix = Path(uploaded_file.name).suffix or ".tif"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        image = load_image(tmp_path)
        validate_image(image)
        mito_channel = extract_mitochondrial_channel(image)
        normalized = zscore_normalize(mito_channel)

        result = segment(model, normalized, confidence_threshold=0.5)
        labeled_mask = label(result.mask)
        features = extract_single_features(result.mask)
        total_area = float(normalized.shape[-2] * normalized.shape[-1])
        return {
            "image": mito_channel,
            "mask": result.mask,
            "labeled_mask": labeled_mask,
            "confidence": result.confidence,
            "features": features,
            "fragmentation_index": fragmentation_index(features, total_area),
            "mitochondrial_density": mitochondrial_density(features, total_area),
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _render_overlay(image_arr: np.ndarray, mask: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_arr, cmap="gray")
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = [1.0, 0.25, 0.35, 0.45]
    ax.imshow(overlay)
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    fig.tight_layout(pad=0)
    return fig


def _render_confidence_heatmap(confidence: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(confidence, cmap="inferno", vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _style_for_dark_theme(fig: plt.Figure) -> plt.Figure:
    """Recolor a matplotlib figure to match the dashboard's dark card background.

    Uses an explicit opaque dark facecolor rather than a transparent one:
    st.pyplot()'s rendering doesn't reliably honor a transparent figure
    patch, which was leaving light-on-white (near-invisible) text/points
    behind — this makes the background match the theme unconditionally.
    """
    fig.patch.set_facecolor("#1A1826")
    fig.patch.set_alpha(1.0)
    for ax in fig.axes:
        ax.set_facecolor("#1A1826")
        ax.tick_params(colors="#A8A4C0")
        ax.xaxis.label.set_color("#EDEBF7")
        ax.yaxis.label.set_color("#EDEBF7")
        ax.title.set_color("#EDEBF7")
        for spine in ax.spines.values():
            spine.set_color("#332F4A")
    return fig


def _features_to_df(features) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "area": f.area,
                "perimeter": f.perimeter,
                "circularity": f.circularity,
                "aspect_ratio": f.aspect_ratio,
                "solidity": f.solidity,
            }
            for f in features
        ]
    )


def _compute_stats(image: np.ndarray, labeled_mask: np.ndarray) -> dict:
    mask = labeled_mask > 0
    features = extract_single_features(mask)
    total_area = float(image.shape[-2] * image.shape[-1])
    n_regions = len(features)
    mean_area = (sum(f.area for f in features) / n_regions) if n_regions else 0.0
    return {
        "mask": mask,
        "region_count": n_regions,
        "mean_area": mean_area,
        "fragmentation_index": fragmentation_index(features, total_area),
        "mitochondrial_density": mitochondrial_density(features, total_area),
    }


def _save_new_run(
    image: np.ndarray,
    labeled_mask: np.ndarray,
    filename: str,
    animal_id: str,
    condition: str,
    time_point: str,
) -> int:
    """Persist a new segmentation run's artifacts + summary row (FR-40, FR-41)."""
    stats = _compute_stats(image, labeled_mask)

    run_uuid = uuid.uuid4().hex[:12]
    data_path = save_run_artifacts(RUNS_DIR, run_uuid, image, labeled_mask)

    OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
    overlay_path = OVERLAYS_DIR / f"{run_uuid}.png"
    fig = _render_overlay(image, stats["mask"])
    fig.savefig(overlay_path, dpi=120, facecolor=OVERLAY_FACECOLOR)
    plt.close(fig)

    return _get_db().insert_segmentation_run(
        filename=filename,
        animal_id=animal_id or None,
        condition=condition or None,
        time_point=time_point or None,
        region_count=stats["region_count"],
        mean_area=stats["mean_area"],
        fragmentation_index=stats["fragmentation_index"],
        mitochondrial_density=stats["mitochondrial_density"],
        overlay_path=str(overlay_path),
        data_path=str(data_path),
    )


def _save_corrected_run(
    run_id: int, image: np.ndarray, labeled_mask: np.ndarray, data_path: str, overlay_path: str
) -> None:
    """Overwrite an existing run's artifacts + summary row after manual correction (FR-13)."""
    stats = _compute_stats(image, labeled_mask)

    save_run_artifacts(Path(data_path).parent, Path(data_path).stem, image, labeled_mask)
    fig = _render_overlay(image, stats["mask"])
    fig.savefig(overlay_path, dpi=120, facecolor=OVERLAY_FACECOLOR)
    plt.close(fig)

    _get_db().update_segmentation_run(
        run_id,
        region_count=stats["region_count"],
        mean_area=stats["mean_area"],
        fragmentation_index=stats["fragmentation_index"],
        mitochondrial_density=stats["mitochondrial_density"],
        corrected=1,
    )


CUSTOM_CSS = """
<style>
    #MainMenu, footer {visibility: hidden;}

    .block-container {padding-top: 2.5rem; max-width: 1200px;}

    .hero {
        margin-bottom: 0.25rem;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        color: #EDEBF7;
    }
    .hero-subtitle {
        color: #A8A4C0;
        font-size: 1.02rem;
        margin: 0.15rem 0 1.75rem 0;
    }

    div[data-testid="stMetric"] {
        background: #1E1B2B;
        border: 1px solid #332F4A;
        border-radius: 14px;
        padding: 1rem 1.1rem;
    }

    .card {
        background: #1A1826;
        border: 1px solid #2A2740;
        border-radius: 16px;
        padding: 1.5rem 1.6rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.25rem;
    }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .badge-live { background: rgba(74, 222, 128, 0.16); color: #4ADE80; }
    .badge-stub { background: rgba(251, 191, 36, 0.16); color: #FBBF24; }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1 class="hero-title">MitoMorph</h1>
    </div>
    <p class="hero-subtitle">
        Automated segmentation, classification, and morphometric quantification of
        mitochondria in spinal cord tissue microscopy images.
    </p>
    """,
    unsafe_allow_html=True,
)

_segmentation_status = "Live" if CHECKPOINT_PATH.exists() else "Stub"

status_col1, status_col2, status_col3, status_col4 = st.columns(4)
status_col1.metric("Preprocessing", "Live", help="Image I/O, normalization, channel ID, Z-stack projection")
status_col2.metric("Morphometrics", "Live", help="Feature extraction, dysfunction indices, quality control")
status_col3.metric(
    "Segmentation",
    _segmentation_status,
    help="U-Net/ResNet34, trained on EM data (Val Dice 0.80). Untested on fluorescence images.",
)
status_col4.metric("Classification", "Stub", help="Cell-type & health classifiers need a trained checkpoint")

tab_analyze, tab_results, tab_correct, tab_validate = st.tabs(
    ["Analyze", "Results", "Mask Correction", "Validate"]
)

with tab_analyze:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload & Analyze")
    uploaded_file = st.file_uploader("Microscopy image (TIFF/OME-TIFF/CZI)", type=["tif", "tiff", "czi"])

    col1, col2, col3 = st.columns(3)
    animal_id = col1.text_input("Animal ID")
    condition = col2.selectbox("Condition", ["Naive", "SCI", "SCI + PTEN-KO", "SCI + PGC1a"])
    time_point = col3.text_input("Time point (e.g. '6 weeks post-SCI')")
    st.caption(
        "These label the saved run for tracking and comparison in the Results tab — "
        "they don't change the detected regions, which depend only on the image."
    )

    if st.button("Run analysis", type="primary", disabled=uploaded_file is None):
        if _get_model() is None:
            st.warning("No trained segmentation checkpoint found. Segmentation can't run without one.")
        else:
            try:
                with st.spinner("Running preprocessing + segmentation..."):
                    result = _run_segmentation_preview(uploaded_file)
                    run_id = _save_new_run(
                        result["image"],
                        result["labeled_mask"],
                        uploaded_file.name,
                        animal_id,
                        condition,
                        time_point,
                    )
            except Exception as exc:  # noqa: BLE001 — show a clean message, not a raw traceback
                st.error(f"Analysis failed: {exc}")
            else:
                n_regions = len(result["features"])
                saved_label = ", ".join(filter(None, [animal_id or None, condition, time_point or None]))
                st.success(f"{n_regions} region(s) detected. Saved as run #{run_id} ({saved_label}).")
                fig_col, stats_col = st.columns([2, 1])
                with fig_col:
                    st.pyplot(_render_overlay(result["image"], result["mask"]))
                with stats_col:
                    st.metric("Detected regions", n_regions)
                    st.metric("Fragmentation index", f"{result['fragmentation_index']:.3e}")
                    st.metric("Mitochondrial density", f"{result['mitochondrial_density']:.4f}")
                    if n_regions:
                        mean_area = sum(f.area for f in result["features"]) / n_regions
                        st.metric("Mean region area (px²)", f"{mean_area:.1f}")
                if n_regions == 0:
                    st.info(
                        "Zero regions is expected on fluorescence images — this checkpoint "
                        "was trained on electron microscopy data."
                    )

                st.markdown("##### Model confidence map")
                st.caption(
                    "Raw per-pixel detection confidence (before the 0.5 threshold is applied). "
                    "Brighter areas are where the model is more confident it's looking at a "
                    "mitochondrion."
                )
                st.pyplot(_style_for_dark_theme(_render_confidence_heatmap(result["confidence"])))

                if n_regions >= 3:
                    chart_col1, chart_col2 = st.columns(2)
                    feature_df = _features_to_df(result["features"])
                    with chart_col1:
                        st.markdown("##### Region area distribution")
                        st.pyplot(
                            _style_for_dark_theme(
                                plot_feature_histogram(feature_df["area"].tolist(), xlabel="area (px²)")
                            )
                        )
                    with chart_col2:
                        st.markdown("##### Region shape correlation (FR-37)")
                        st.pyplot(
                            _style_for_dark_theme(plot_feature_correlation(feature_df, "area", "circularity"))
                        )

                st.caption(
                    "Note: re-analyzing the same image always produces the same detected "
                    "regions and charts — the model is deterministic and only looks at pixel "
                    "data. Animal ID/Condition/Time point change only the label saved with the run."
                )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_results:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Results")
    runs = _get_db().list_segmentation_runs()

    if not runs:
        st.info("No analyses recorded yet. Run an analysis in the Analyze tab first.")
    else:
        all_conditions = sorted({r["condition"] for r in runs if r["condition"]})
        selected_conditions = st.multiselect(
            "Filter by condition", options=all_conditions, default=all_conditions
        )
        filtered_runs = [r for r in runs if not r["condition"] or r["condition"] in selected_conditions]

        if not filtered_runs:
            st.info("No runs match the selected condition(s).")
        else:
            table = pd.DataFrame(
                [
                    {
                        "Run": r["run_id"],
                        "File": r["filename"],
                        "Animal": r["animal_id"] or "—",
                        "Condition": r["condition"] or "—",
                        "Time point": r["time_point"] or "—",
                        "Regions": r["region_count"],
                        "Frag. index": round(r["fragmentation_index"], 6),
                        "Density": round(r["mitochondrial_density"], 4),
                        "Corrected": "Yes" if r["corrected"] else "No",
                        "When": r["created_at"],
                    }
                    for r in filtered_runs
                ]
            )
            st.dataframe(table, use_container_width=True, hide_index=True)

            conditions_present = sorted({r["condition"] for r in filtered_runs if r["condition"]})
            if len(conditions_present) >= 2:
                st.markdown("##### Comparison across conditions (FR-36)")
                st.caption(
                    "Only shows conditions with at least one saved run. To add "
                    "'SCI + PTEN-KO' or 'SCI + PGC1a' here, run an analysis in the "
                    "Analyze tab with that condition selected."
                )
                comparison_df = pd.DataFrame(
                    [
                        {
                            "Condition": r["condition"],
                            "Regions": r["region_count"],
                            "Fragmentation index": r["fragmentation_index"],
                            "Mitochondrial density": r["mitochondrial_density"],
                        }
                        for r in filtered_runs
                        if r["condition"]
                    ]
                )
                chart_cols = st.columns(3)
                for chart_col, metric in zip(
                    chart_cols, ["Regions", "Fragmentation index", "Mitochondrial density"]
                ):
                    with chart_col:
                        st.pyplot(
                            _style_for_dark_theme(
                                plot_condition_comparison(
                                    comparison_df, metric, groupby="Condition", kind="box"
                                )
                            )
                        )

            run_by_id = {r["run_id"]: r for r in filtered_runs}
            selected_id = st.selectbox(
                "View run",
                options=list(run_by_id.keys()),
                format_func=lambda rid: f"#{rid} — {run_by_id[rid]['filename']}",
            )
            selected = run_by_id[selected_id]

            img_col, stats_col = st.columns([2, 1])
            with img_col:
                st.image(selected["overlay_path"])
            with stats_col:
                st.metric("Detected regions", selected["region_count"])
                st.metric("Fragmentation index", f"{selected['fragmentation_index']:.3e}")
                st.metric("Mitochondrial density", f"{selected['mitochondrial_density']:.4f}")
                st.metric("Mean region area (px²)", f"{selected['mean_area']:.1f}")

            selected_image, selected_labeled_mask = load_run_artifacts(selected["data_path"])
            selected_features = extract_single_features(selected_labeled_mask > 0)
            if len(selected_features) >= 3:
                st.markdown("##### Region shape correlation (FR-37)")
                st.pyplot(
                    _style_for_dark_theme(
                        plot_feature_correlation(_features_to_df(selected_features), "area", "circularity")
                    )
                )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_correct:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Manual Segmentation Correction")
    runs = _get_db().list_segmentation_runs()

    if not runs:
        st.info("No segmentation runs to correct yet — run an analysis in the Analyze tab first.")
    else:
        run_by_id = {r["run_id"]: r for r in runs}
        selected_id = st.selectbox(
            "Run to correct",
            options=list(run_by_id.keys()),
            format_func=lambda rid: f"#{rid} — {run_by_id[rid]['filename']}",
            key="correct_run_select",
        )
        selected = run_by_id[selected_id]
        image, labeled_mask = load_run_artifacts(selected["data_path"])
        region_areas = {int(p.label): int(p.area) for p in regionprops(labeled_mask)}
        region_ids = list(region_areas.keys())

        if not region_ids:
            st.info("This run has no detected regions to correct.")
        else:
            kept_ids = st.multiselect(
                "Regions to keep (remove any false positives)",
                options=region_ids,
                default=region_ids,
                format_func=lambda rid: f"Region {rid} ({region_areas[rid]} px²)",
            )
            corrected_mask = np.isin(labeled_mask, kept_ids)
            st.pyplot(_render_overlay(image, corrected_mask))
            st.caption(f"{len(kept_ids)} of {len(region_ids)} region(s) kept.")

            if st.button("Save correction", type="primary"):
                corrected_labeled_mask = np.where(corrected_mask, labeled_mask, 0)
                _save_corrected_run(
                    selected_id,
                    image,
                    corrected_labeled_mask,
                    selected["data_path"],
                    selected["overlay_path"],
                )
                st.success(f"Correction saved to run #{selected_id}.")
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with tab_validate:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Validate Against Ground Truth")
    st.caption(
        "Upload a microscopy image together with a known ground-truth mask (a binary TIFF "
        "marking true mitochondria) to measure segmentation accuracy against it (NFR-05, "
        "NFR-06). This needs real labeled data — it can't validate against an image alone."
    )

    val_col1, val_col2 = st.columns(2)
    val_image_file = val_col1.file_uploader(
        "Microscopy image", type=["tif", "tiff", "czi"], key="validate_image"
    )
    val_mask_file = val_col2.file_uploader(
        "Ground-truth mask (TIFF)", type=["tif", "tiff"], key="validate_mask"
    )

    run_validation = st.button(
        "Run validation", type="primary", disabled=val_image_file is None or val_mask_file is None
    )
    if run_validation:
        if _get_model() is None:
            st.warning("No trained segmentation checkpoint found. Segmentation can't run without one.")
        else:
            try:
                with st.spinner("Running segmentation and comparing to ground truth..."):
                    suffix = Path(val_image_file.name).suffix or ".tif"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(val_image_file.getvalue())
                        tmp_path = tmp.name
                    try:
                        image_obj = load_image(tmp_path)
                        validate_image(image_obj)
                        mito_channel = extract_mitochondrial_channel(image_obj)
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

                    normalized = zscore_normalize(mito_channel)
                    pred_result = segment(_get_model(), normalized, confidence_threshold=0.5)

                    gt_mask = tifffile.imread(io.BytesIO(val_mask_file.getvalue())) > 127
                    if gt_mask.shape != pred_result.mask.shape:
                        raise ValueError(
                            f"Ground-truth mask shape {gt_mask.shape} doesn't match image "
                            f"shape {pred_result.mask.shape}"
                        )
            except Exception as exc:  # noqa: BLE001 — show a clean message, not a raw traceback
                st.error(f"Validation failed: {exc}")
            else:
                metric_cols = st.columns(5)
                metric_cols[0].metric("Dice", f"{dice_score(pred_result.mask, gt_mask):.3f}")
                metric_cols[1].metric("IoU", f"{iou_score(pred_result.mask, gt_mask):.3f}")
                metric_cols[2].metric("Precision", f"{precision_score(pred_result.mask, gt_mask):.3f}")
                metric_cols[3].metric("Recall", f"{recall_score(pred_result.mask, gt_mask):.3f}")
                metric_cols[4].metric("F1", f"{f1_score(pred_result.mask, gt_mask):.3f}")

                cm_col, overlay_col = st.columns(2)
                with cm_col:
                    st.markdown("##### Confusion matrix (pixel-level)")
                    cm = confusion_matrix(pred_result.mask, gt_mask)
                    st.pyplot(_style_for_dark_theme(plot_confusion_matrix(cm)))
                with overlay_col:
                    st.markdown("##### Predicted (red) vs. ground truth (green)")
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(mito_channel, cmap="gray")
                    gt_overlay = np.zeros((*gt_mask.shape, 4))
                    gt_overlay[gt_mask] = [0.2, 1.0, 0.3, 0.4]
                    ax.imshow(gt_overlay)
                    pred_overlay = np.zeros((*pred_result.mask.shape, 4))
                    pred_overlay[pred_result.mask] = [1.0, 0.25, 0.35, 0.4]
                    ax.imshow(pred_overlay)
                    ax.axis("off")
                    fig.patch.set_alpha(0.0)
                    fig.tight_layout(pad=0)
                    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About")
    st.write(
        "MitoMorph automates segmentation, classification, and morphometric quantification "
        "of mitochondria in spinal cord tissue microscopy images, for spinal cord injury (SCI) "
        "research."
    )
    st.markdown("### Pipeline status")
    st.markdown(
        '<span class="badge badge-live">Live</span>&nbsp; Preprocessing, segmentation, '
        "morphometrics, run persistence, mask correction, comparison figures, "
        "ground-truth validation",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="badge badge-stub">Stub</span>&nbsp; Cell-type & health '
        "classification, XAI, PDF reports",
        unsafe_allow_html=True,
    )
    if CHECKPOINT_PATH.exists():
        st.caption(
            "Segmentation checkpoint: U-Net/ResNet34 trained on EPFL/Lucchi EM data "
            "(Val Dice 0.80, Val IoU 0.73). Not validated on fluorescence images."
        )
