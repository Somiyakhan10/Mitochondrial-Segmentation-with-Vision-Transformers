"""Streamlit dashboard skeleton (FR-13, FR-40, NFR-10).

Runnable placeholder: page structure, upload widget, and section layout
are real; wiring to pipeline results, mask-overlay rendering, and manual
mask correction are stubs pending a trained segmentation model.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="MitoMorph", layout="wide")

st.title("Mitochondrial Morphology Analysis Pipeline")
st.caption("Prof. Patel's Lab — Spinal Cord Injury Research")

tab_analyze, tab_results, tab_correct = st.tabs(["Analyze", "Results", "Mask Correction"])

with tab_analyze:
    st.header("Upload & Analyze")
    uploaded_file = st.file_uploader("Microscopy image (TIFF/OME-TIFF/CZI)", type=["tif", "tiff", "czi"])
    col1, col2, col3 = st.columns(3)
    animal_id = col1.text_input("Animal ID")
    condition = col2.selectbox("Condition", ["Naive", "SCI", "SCI + PTEN-KO", "SCI + PGC1a"])
    time_point = col3.text_input("Time point (e.g. '6 weeks post-SCI')")

    if st.button("Run analysis", disabled=uploaded_file is None):
        st.info(
            "Analysis requires a trained segmentation model (mitomorph.segmentation.infer) "
            "and health classifier (mitomorph.classification.health_classifier), neither of "
            "which exist yet. This button will call MitoPipeline.run() once they do."
        )

with tab_results:
    st.header("Results")
    st.info(
        "No analyses recorded yet. Once mitomorph.data.database.AnalysisDatabase has entries, "
        "this tab will show overlay images, morphometric statistics tables (FR-40), and "
        "comparative charts across conditions."
    )

with tab_correct:
    st.header("Manual Segmentation Correction")
    st.info(
        "Interactive mask correction (FR-13) is not yet implemented. Planned: display the "
        "predicted mask overlaid on the image with editable region boundaries, and persist "
        "corrections back through mitomorph.data.database."
    )

st.sidebar.header("About")
st.sidebar.write(
    "MitoMorph automates segmentation, classification, and morphometric quantification of "
    "mitochondria in spinal cord tissue microscopy images."
)
