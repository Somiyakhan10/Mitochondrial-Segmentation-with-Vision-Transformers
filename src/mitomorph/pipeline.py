"""End-to-end pipeline orchestration (SRS §2.1).

Wires preprocessing -> segmentation -> cell-type classification ->
morphometrics -> health classification -> XAI -> reporting. Several
stages are stubs (see their module docstrings) that raise
``NotImplementedError`` until real logic is filled in; this module fixes
the call order and data flow so that filling in one stage at a time
doesn't require touching this file again. Once every stage stub is
implemented, :meth:`MitoPipeline.run` works end-to-end unmodified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mitomorph.celltype.classifier import classify_neuronal
from mitomorph.config.loader import load_config
from mitomorph.logger import get_logger
from mitomorph.morphometrics.dysfunction_indices import fragmentation_index, mitochondrial_density
from mitomorph.morphometrics.quality_control import validate_features
from mitomorph.morphometrics.single_features import extract_single_features
from mitomorph.preprocessing.channel_utils import extract_mitochondrial_channel
from mitomorph.preprocessing.io import load_image
from mitomorph.preprocessing.normalization import zscore_normalize
from mitomorph.preprocessing.validators import validate_image, validate_metadata
from mitomorph.preprocessing.zstack import max_intensity_projection
from mitomorph.segmentation.infer import load_model, segment

logger = get_logger(__name__)


class MitoPipeline:
    """Orchestrates the full mitochondrial morphology analysis pipeline for a single image."""

    def __init__(self, config_path: str | Path, segmentation_model: Any | None = None):
        self.config = load_config(config_path)
        self.segmentation_model = segmentation_model or self._load_default_segmentation_model()

    def _load_default_segmentation_model(self) -> Any | None:
        checkpoint_path = Path(self.config["segmentation"]["checkpoint_path"])
        if not checkpoint_path.exists():
            logger.info(
                "No segmentation checkpoint found at %s; segmentation stage is unavailable", checkpoint_path
            )
            return None
        logger.info("Loading segmentation checkpoint from %s", checkpoint_path)
        return load_model(checkpoint_path)

    def run(self, image_path: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
        """Run preprocessing -> segmentation -> cell-type -> morphometrics ->
        classification -> reporting on a single image.

        Args:
            image_path: path to the raw microscopy image.
            metadata: experimental metadata (``experimental_condition``,
                ``time_point``, ``animal_id``, ...).

        Returns:
            An :class:`~mitomorph.data.schema.AnalysisResult`-shaped dict
            once every stage is implemented.
        """
        logger.info("Starting pipeline run for %s", image_path)

        image = load_image(image_path)
        validate_image(image)
        validate_metadata(metadata)

        mito_channel = extract_mitochondrial_channel(image)
        remaining_axes = image.axes.replace("C", "")
        if "Z" in remaining_axes:
            projected = max_intensity_projection(mito_channel, z_axis=remaining_axes.index("Z"))
        else:
            projected = mito_channel
        normalized = zscore_normalize(projected)

        segmentation_result = segment(self.segmentation_model, normalized)
        mask = segmentation_result.mask

        classify_neuronal(mask, image)

        features = extract_single_features(mask)
        validate_features(features)
        total_area = float(normalized.shape[-2] * normalized.shape[-1])
        fragmentation_index(features, total_area)
        mitochondrial_density(features, total_area)

        # Health classification (mitomorph.classification.health_classifier),
        # XAI (mitomorph.xai), and report assembly (mitomorph.reporting) are
        # the next stages once a trained segmentation model and classifier exist.
        raise NotImplementedError(
            "Reached the health-classification stage, which requires a trained classifier; "
            "preprocessing, segmentation, cell-type, and morphometric stages above are wired."
        )
