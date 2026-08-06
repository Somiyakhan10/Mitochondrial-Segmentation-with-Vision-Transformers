"""Streamlit dashboard (FR-13, FR-40, NFR-10).

Page structure, styling, upload widget, section layout, and the
segmentation + morphometrics preview are real (a trained checkpoint is
loaded from data/models/segmentation_unet.pt if present). Cell-type
classification, health scoring, and manual mask correction are still
stubs pending further training data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from mitomorph.exceptions import MitoMorphError
from mitomorph.morphometrics.dysfunction_indices import fragmentation_index, mitochondrial_density
from mitomorph.morphometrics.single_features import extract_single_features
from mitomorph.preprocessing.channel_utils import extract_mitochondrial_channel
from mitomorph.preprocessing.io import load_image
from mitomorph.preprocessing.normalization import zscore_normalize
from mitomorph.preprocessing.validators import validate_image
from mitomorph.segmentation.infer import load_model, segment

st.set_page_config(page_title="MitoMorph", page_icon="\U0001f9ec", layout="wide")

CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "data" / "models" / "segmentation_unet.pt"


@st.cache_resource
def _get_model():
    if not CHECKPOINT_PATH.exists():
        return None
    return load_model(CHECKPOINT_PATH)


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
        features = extract_single_features(result.mask)
        total_area = float(normalized.shape[-2] * normalized.shape[-1])
        return {
            "image": mito_channel,
            "mask": result.mask,
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


CUSTOM_CSS = """
<style>
    #MainMenu, footer {visibility: hidden;}

    .block-container {padding-top: 2.5rem; max-width: 1200px;}

    .hero {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.25rem;
    }
    .hero-icon {
        font-size: 2.4rem;
        line-height: 1;
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
        <div class="hero-icon">\U0001f9ec</div>
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

tab_analyze, tab_results, tab_correct = st.tabs(
    ["\U0001f4c2 Analyze", "\U0001f4ca Results", "✏️ Mask Correction"]
)

with tab_analyze:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload & Analyze")
    uploaded_file = st.file_uploader("Microscopy image (TIFF/OME-TIFF/CZI)", type=["tif", "tiff", "czi"])

    col1, col2, col3 = st.columns(3)
    animal_id = col1.text_input("Animal ID")
    condition = col2.selectbox("Condition", ["Naive", "SCI", "SCI + PTEN-KO", "SCI + PGC1a"])
    time_point = col3.text_input("Time point (e.g. '6 weeks post-SCI')")

    if st.button("Run analysis", type="primary", disabled=uploaded_file is None):
        if _get_model() is None:
            st.warning(
                f"No trained segmentation checkpoint found at `{CHECKPOINT_PATH}`. "
                "Preprocessing and morphometrics are real, but segmentation needs a "
                "checkpoint to run.",
                icon="⚠️",
            )
        else:
            try:
                with st.spinner("Running preprocessing + segmentation..."):
                    result = _run_segmentation_preview(uploaded_file)
            except MitoMorphError as exc:
                st.error(f"Analysis failed: {exc}", icon="🚫")
            else:
                n_regions = len(result["features"])
                st.success(
                    f"Segmentation complete — {n_regions} region(s) detected "
                    "(confidence threshold 0.5). Cell-type classification and health "
                    "scoring are still stubs pending further training data.",
                    icon="✅",
                )
                fig_col, stats_col = st.columns([2, 1])
                with fig_col:
                    st.pyplot(_render_overlay(result["image"], result["mask"]))
                with stats_col:
                    st.metric("Detected regions", n_regions)
                    st.metric("Fragmentation index", f"{result['fragmentation_index']:.4f}")
                    st.metric("Mitochondrial density", f"{result['mitochondrial_density']:.4f}")
                    if n_regions:
                        mean_area = sum(f.area for f in result["features"]) / n_regions
                        st.metric("Mean region area (px²)", f"{mean_area:.1f}")
                if n_regions == 0:
                    st.info(
                        "Zero regions detected is expected on fluorescence images: this "
                        "checkpoint was trained on electron microscopy data (see the "
                        "Segmentation status badge above), so its confidence stays below "
                        "the 0.5 threshold on other imaging modalities.",
                        icon="ℹ️",
                    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_results:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Results")
    st.info(
        "No analyses recorded yet. Once mitomorph.data.database.AnalysisDatabase has entries, "
        "this tab will show overlay images, morphometric statistics tables (FR-40), and "
        "comparative charts across conditions.",
        icon="ℹ️",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab_correct:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Manual Segmentation Correction")
    st.info(
        "Interactive mask correction (FR-13) is not yet implemented. Planned: display the "
        "predicted mask overlaid on the image with editable region boundaries, and persist "
        "corrections back through mitomorph.data.database.",
        icon="ℹ️",
    )
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
        "morphometrics, data storage, export",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="badge badge-stub">Stub</span>&nbsp; Cell-type & health '
        "classification, XAI, reporting",
        unsafe_allow_html=True,
    )
    if CHECKPOINT_PATH.exists():
        st.caption(
            "Segmentation checkpoint: U-Net/ResNet34 trained on EPFL/Lucchi EM data "
            "(Val Dice 0.80, Val IoU 0.73). Not validated on fluorescence images."
        )
