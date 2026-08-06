# Software Requirements Specification (SRS)
## Mitochondrial Morphology Analysis Pipeline for Spinal Cord Injury Research

### A Computer Vision Project Aligned with Mitochondrial Dysfunction Research in Chronic SCI

---

### 1. Introduction

#### 1.1 Purpose
This document defines the software requirements for a deep learning-based image analysis pipeline designed to automatically segment, classify, and quantify mitochondrial morphology from microscopy images of spinal cord tissue. The system will provide researchers with high-throughput, reproducible analysis of mitochondrial health, enabling rapid assessment of therapeutic interventions for spinal cord injury (SCI).

#### 1.2 Alignment with Ongoing Research
This project directly complements published research on mitochondrial dysfunction in chronic SCI, specifically findings that:

- **Neuronal mitochondria exhibit ~50% loss of respiratory capacity** at 21 weeks post-SCI (Experimental Neurology 2026)
- **PTEN knockout restores mitochondrial bioenergetic abilities** in chronic SCI conditions
- **PGC1α upregulation can restore mitochondrial function** when applied during chronic SCI

Rather than replicating existing methods, this project provides a **computational tool** to accelerate this research by automating the analysis of mitochondrial morphology, which is currently a manual, time-consuming process in most labs.

#### 1.3 Scope
The system will:
- Automatically segment mitochondria from fluorescence microscopy images
- Classify mitochondrial morphological types (healthy, fragmented, swollen, dysfunctional)
- Quantify morphometric parameters (area, perimeter, circularity, aspect ratio, network complexity)
- Track mitochondrial changes over time in response to interventions
- Generate comparative reports between experimental conditions
- Integrate explainable AI to identify morphological features most indicative of dysfunction

#### 1.4 Intended Audience
- Lab researchers studying mitochondrial transplantation
- SCI researchers evaluating mitochondrial-targeted therapeutics
- Neuroscientists investigating mitochondrial dynamics in neurodegeneration

---

### 2. System Overview

#### 2.1 System Architecture

Input Layer (fluorescence microscopy images: TIFF/OME-TIFF, mitochondrial
markers Tom20/COX IV/MitoTracker, neuronal markers NeuN/NFH, multi-channel
Z-stacks/time-series)
→ Preprocessing Layer (illumination correction, intensity normalization,
denoising, Z-stack maximum intensity projection)
→ Segmentation Layer (U-Net/ResNet34, Attention U-Net, cell-type
classification neuron vs. non-neuron)
→ Morphometric Analysis Layer (single-mitochondria features, network
features, dysfunction indicators)
→ Classification & Prediction Layer (health classifier, functional
outcome predictor)
→ Output Layer (interactive dashboard, PDF report, CSV/Excel/JSON export)

#### 2.2 Technology Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| Deep Learning | PyTorch, MONAI | Medical imaging optimized, GPU support |
| Image Processing | OpenCV, scikit-image | Robust microscopy image handling |
| Segmentation Models | U-Net, Attention U-Net | State-of-the-art for biomedical segmentation |
| Pre-trained Weights | EM mitochondria datasets | Transfer learning reduces training data needs |
| Morphometric Analysis | scikit-image, CellProfiler | Established morphometric tools |
| Visualization | Matplotlib, Seaborn, Plotly | Publication-quality figures |
| Dashboard | Streamlit | Rapid prototyping, interactive |
| Model Management | MLflow | Experiment tracking |
| Explainable AI | Captum, Grad-CAM | Visualize model decision-making |

---

### 3. Functional Requirements

#### 3.1 Image Input and Preprocessing

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | System shall accept fluorescence microscopy images in TIFF, OME-TIFF, and CZI formats | High |
| FR-02 | System shall support multi-channel images with mitochondrial markers (Tom20, COX IV, MitoTracker) and nuclear markers | High |
| FR-03 | System shall perform illumination correction using flat-field or rolling ball background subtraction | High |
| FR-04 | System shall normalize intensity using Z-score or percentile-based methods | High |
| FR-05 | System shall denoise images using non-local means or BM3D algorithms | High |
| FR-06 | System shall process Z-stacks with maximum intensity projection (MIP) or focus stacking | Medium |
| FR-07 | System shall support batch processing of entire experimental datasets | High |

#### 3.2 Mitochondrial Segmentation

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-08 | System shall implement U-Net architecture with ResNet34 encoder for mitochondrial segmentation | High |
| FR-09 | System shall implement Attention U-Net as alternative architecture for overlapping mitochondria | High |
| FR-10 | System shall be pre-trained on EM mitochondrial datasets (e.g., MitoEM, EMPIAR) | High |
| FR-11 | System shall support fine-tuning on lab-specific microscopy data | High |
| FR-12 | System shall provide confidence scores for each segmented mitochondrial region | Medium |
| FR-13 | System shall support manual correction of segmentation masks via interactive interface | Medium |
| FR-14 | System shall calculate Intersection over Union (IoU) and Dice scores for validation | High |

#### 3.3 Cell-Type Specific Analysis

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-15 | System shall classify neuronal vs. non-neuronal mitochondria using NeuN or NFH markers | High |
| FR-16 | System shall provide separate morphometric analysis for neuronal mitochondria | High |
| FR-17 | System shall detect and exclude non-neuronal mitochondria from analysis | High |
| FR-18 | System shall generate comparison reports between neuronal and non-neuronal mitochondria | Medium |

#### 3.4 Morphometric Feature Extraction

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-19 | System shall extract single mitochondria features: area, perimeter, major/minor axis length, aspect ratio, circularity, solidity, eccentricity | High |
| FR-20 | System shall extract Feret diameter, extent, and equivalent diameter | High |
| FR-21 | System shall extract network features: total mitochondrial footprint, network size, branch points, end points | High |
| FR-22 | System shall calculate fragmentation index (number of mitochondria / total area) | High |
| FR-23 | System shall calculate swelling score (area/perimeter ratio deviation from healthy baseline) | High |
| FR-24 | System shall calculate mitochondrial density (mitochondrial area / cell area) | High |
| FR-25 | System shall support cristae density measurement from electron microscopy images | Low |

#### 3.5 Classification and Prediction

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-26 | System shall classify mitochondria into four categories: Healthy, Fragmented, Swollen, Dysfunctional | High |
| FR-27 | System shall implement Random Forest classifier trained on expert-annotated mitochondria | High |
| FR-28 | System shall predict mitochondrial health score (0-100) based on morphometric features | High |
| FR-29 | System shall correlate morphological features with predicted respiratory capacity (from Seahorse data if available) | Medium |
| FR-30 | System shall identify mitochondria most likely to respond to PTEN-KO or PGC1α treatment | Medium |

#### 3.6 Explainable AI

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-31 | System shall implement Grad-CAM to generate heatmaps highlighting regions most important for classification | High |
| FR-32 | System shall calculate SHAP values for morphometric features to identify most discriminative features | High |
| FR-33 | System shall generate feature importance plots for each classification decision | High |
| FR-34 | System shall provide uncertainty estimates for each prediction | Medium |

#### 3.7 Reporting and Visualization

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-35 | System shall generate publication-quality figures for segmentation masks overlaid on original images | High |
| FR-36 | System shall generate box plots and bar charts comparing morphometric features across conditions (Naïve vs SCI vs PTEN-KO vs PGC1α) | High |
| FR-37 | System shall generate scatter plots showing correlation between morphological features | High |
| FR-38 | System shall generate heatmaps for mitochondrial network connectivity | Medium |
| FR-39 | System shall create comprehensive PDF reports with experimental details, analysis parameters, results, and statistics | High |
| FR-40 | System shall support interactive dashboard for exploring individual mitochondria classifications | Medium |

#### 3.8 Data Management

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-41 | System shall maintain database of analyzed images with associated metadata (experimental conditions, treatment, time point) | High |
| FR-42 | System shall support version control for analysis parameters | Medium |
| FR-43 | System shall export all data in CSV, Excel, and JSON formats | High |
| FR-44 | System shall support integration with existing lab databases | Low |

---

### 4. Non-Functional Requirements

#### 4.1 Performance

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-01 | Mitochondrial segmentation shall complete within 30 seconds per image on GPU (NVIDIA Tesla T4 or equivalent) | High |
| NFR-02 | Morphometric analysis shall complete within 10 seconds per image | High |
| NFR-03 | System shall support batch processing of at least 100 images per run | High |
| NFR-04 | System shall operate with GPU memory ≤ 8GB | Medium |

#### 4.2 Accuracy

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-05 | Segmentation model shall achieve Dice score ≥ 0.85 on validation set | High |
| NFR-06 | Classification model shall achieve accuracy ≥ 0.80 on validation set | High |
| NFR-07 | Morphometric measurements shall be reproducible with coefficient of variation < 5% | High |

#### 4.3 Usability

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-08 | System shall provide Jupyter Notebook tutorials for all major workflows | High |
| NFR-09 | System shall include command-line interface for batch processing | High |
| NFR-10 | System shall provide interactive GUI for visualization and manual correction | Medium |
| NFR-11 | System shall include comprehensive documentation with examples from real SCI research workflows | High |

#### 4.4 Maintainability

| ID | Requirement | Priority |
|----|-------------|----------|
| NFR-12 | Code shall follow PEP 8 Python style guidelines | High |
| NFR-13 | System shall be modular with separate components for preprocessing, segmentation, and analysis | High |
| NFR-14 | All model training pipelines shall be reproducible with version control | High |
| NFR-15 | System shall include comprehensive unit tests and integration tests | Medium |

---

### 5. Data Specifications

#### 5.1 Input Data Requirements

**Image Format:** TIFF, OME-TIFF, CZI (Zeiss), LIF (Leica)
**Resolution:** 1024×1024 pixels minimum, 4096×4096 recommended
**Channels:** Minimum 2 (mitochondrial marker + nuclear/neuronal marker)
**Bit Depth:** 8-bit, 16-bit, or 32-bit

**Required Metadata:** Experimental condition (Naïve, SCI, SCI+PTEN-KO,
SCI+PGC1α), time point post-SCI (weeks), animal ID, treatment details,
imaging parameters (magnification, objective, exposure time)

**Recommended Mitochondrial Markers:** Tom20, COX IV, MitoTracker, VDAC/Porin
**Neuronal Markers for Cell-Type Specificity:** NeuN, NFH, MAP2

#### 5.2 Output Data Schema

```json
{
    "analysis_id": "SCI_001",
    "image_info": {
        "filename": "mouse_123_tom20.tif",
        "experiment": "SCI 6 weeks",
        "animal_id": "M123",
        "time_point": "6 weeks post-SCI"
    },
    "segmentation": {
        "total_mitochondria": 342,
        "neuronal_mitochondria": 187,
        "non_neuronal_mitochondria": 155,
        "segmentation_confidence": 0.92
    },
    "morphometric_summary": {
        "mean_area": 0.85,
        "mean_aspect_ratio": 2.3,
        "mean_circularity": 0.67,
        "fragmentation_index": 4.2,
        "mitochondrial_density": 0.34,
        "network_size": 12.5
    },
    "classification": {
        "healthy": 25,
        "fragmented": 45,
        "swollen": 18,
        "dysfunctional": 12,
        "health_score": 52.3
    },
    "feature_importance": {
        "aspect_ratio": 0.35,
        "area": 0.28,
        "circularity": 0.22,
        "fragmentation_index": 0.15
    },
    "compared_to_baseline": {
        "condition": "SCI vs Naïve",
        "significance": "p < 0.01",
        "change_percent": -32.5
    }
}
```

---

### 6. Project Timeline

| Phase | Tasks | Estimated Duration |
|-------|-------|-------------------|
| Phase 1 | Literature review and requirements finalization | 1 week |
| Phase 2 | Data collection and annotation (collaboration with a research lab) | 2-3 weeks |
| Phase 3 | Preprocessing pipeline development | 1 week |
| Phase 4 | U-Net implementation and training | 2 weeks |
| Phase 5 | Morphometric feature extraction | 1-2 weeks |
| Phase 6 | Classification model development | 1 week |
| Phase 7 | Explainable AI integration | 1 week |
| Phase 8 | Reporting and visualization | 1-2 weeks |
| Phase 9 | Validation and testing | 1 week |
| Phase 10 | Documentation and deployment | 1 week |

---

### 7. Expected Contributions to SCI Research

1. High-throughput analysis — automated processing of hundreds of images
2. Reproducibility — standardized, objective morphometric measurements
3. Novel insights — ML might identify subtle morphological patterns
4. Predictive capability — classification could predict dysfunction severity before Seahorse experiments
5. Visualization — publication-ready figures for grants and papers
6. Cell-type specificity — automated separation of neuronal vs. non-neuronal mitochondria

---

### 8. References

1. Author, S.P., et al. (2026). Proteomics reveal PTEN as a critical mediator of sustained mitochondrial dysfunction during chronic spinal cord injury. *Experimental Neurology*, 404, 115880.
2. Author, S.P., et al. (2022). Erodible thermogelling hydrogels for localized mitochondrial transplantation to the spinal cord. *Mitochondrion*, 65, 91-101.
3. Fecher, C., et al. (2019). Cell-type-specific profiling of brain mitochondria reveals functional and molecular diversity. *Nature Neuroscience*, 22, 1731-1742.
4. Gollihue, J.L., et al. (2018). Effects of mitochondrial transplantation on bioenergetics, cellular incorporation, and functional recovery after spinal cord injury. *Journal of Neurotrauma*, 35, 1800-1818.
5. Stewart, A.N., et al. (2023). PTEN knockout using retrogradely transported AAVs transiently restores locomotor abilities in both acute and chronic spinal cord injury. *Experimental Neurology*, 368, 114502.
6. Ronneberger, O., et al. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*, 234-241.
7. Oktay, O., et al. (2018). Attention U-Net: Learning where to look for the pancreas. *Medical Imaging with Deep Learning*.

---

### 9. Glossary

| Term | Definition |
|------|------------|
| SCI | Spinal Cord Injury |
| PTEN | Phosphatase and Tensin Homologue protein |
| PGC1α | Peroxisome Proliferator-Activated Receptor-γ Coactivator 1-α |
| Mitochondrial Respiration | Oxygen consumption rate, measure of mitochondrial function |
| Seahorse Assay | Method for measuring mitochondrial respiratory capacity |
| U-Net | Convolutional neural network architecture for image segmentation |
| Grad-CAM | Gradient-weighted Class Activation Mapping (XAI technique) |
| Dice Score | Measure of segmentation overlap quality |
| Morphometry | Quantitative measurement of shape and structure |
| Fragmentation | Process of mitochondria dividing into smaller units |
| Fusion | Process of mitochondria merging into networks |

---

### 10. Success Criteria

1. Segmentation accuracy exceeds Dice score of 0.85 on validation data
2. Analysis speed processes 100 images in under 1 hour
3. Results replicate published findings showing significantly different morphometric features between naïve and chronic SCI conditions
4. Cell-type specificity correctly separates neuronal from non-neuronal mitochondria
5. Classification accuracy exceeds 80% for mitochondrial health categories
6. Publication-quality figures can be generated automatically
7. Integration with a research lab's workflow is smooth and requires minimal training
