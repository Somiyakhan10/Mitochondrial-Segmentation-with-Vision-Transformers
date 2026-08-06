"""Streamlit dashboard (FR-13, FR-40, NFR-10).

Page structure, styling, upload widget, and section layout are real;
wiring to pipeline results, mask-overlay rendering, and manual mask
correction are stubs pending a trained segmentation model.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="MitoMorph", page_icon="\U0001f9ec", layout="wide")

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

status_col1, status_col2, status_col3, status_col4 = st.columns(4)
status_col1.metric("Preprocessing", "Live", help="Image I/O, normalization, channel ID, Z-stack projection")
status_col2.metric("Morphometrics", "Live", help="Feature extraction, dysfunction indices, quality control")
status_col3.metric("Segmentation", "Stub", help="Needs a trained checkpoint")
status_col4.metric("Classification", "Stub", help="Needs a trained checkpoint")

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
        st.info(
            "Analysis requires a trained segmentation model (mitomorph.segmentation.infer) "
            "and health classifier (mitomorph.classification.health_classifier), neither of "
            "which exist yet. This button will call MitoPipeline.run() once they do.",
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
        '<span class="badge badge-live">Live</span>&nbsp; Preprocessing, morphometrics, '
        "data storage, export",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="badge badge-stub">Stub</span>&nbsp; Segmentation, cell-type & health '
        "classification, XAI, reporting",
        unsafe_allow_html=True,
    )
