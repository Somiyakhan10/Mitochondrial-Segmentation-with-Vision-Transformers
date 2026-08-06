"""Dataclasses mirroring the SRS §5.2 output JSON schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ImageInfo:
    filename: str
    experiment: str
    animal_id: str
    time_point: str


@dataclass
class SegmentationSummary:
    total_mitochondria: int
    neuronal_mitochondria: int
    non_neuronal_mitochondria: int
    segmentation_confidence: float


@dataclass
class MorphometricSummary:
    mean_area: float
    mean_aspect_ratio: float
    mean_circularity: float
    fragmentation_index: float
    mitochondrial_density: float
    network_size: float


@dataclass
class ClassificationSummary:
    healthy: int
    fragmented: int
    swollen: int
    dysfunctional: int
    health_score: float


@dataclass
class BaselineComparison:
    condition: str
    significance: str
    change_percent: float


@dataclass
class AnalysisResult:
    """Mirrors the JSON output schema defined in SRS §5.2."""

    analysis_id: str
    image_info: ImageInfo
    segmentation: SegmentationSummary
    morphometric_summary: MorphometricSummary
    classification: ClassificationSummary
    feature_importance: dict[str, float] = field(default_factory=dict)
    compared_to_baseline: BaselineComparison | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisResult:
        baseline = data.get("compared_to_baseline")
        return cls(
            analysis_id=data["analysis_id"],
            image_info=ImageInfo(**data["image_info"]),
            segmentation=SegmentationSummary(**data["segmentation"]),
            morphometric_summary=MorphometricSummary(**data["morphometric_summary"]),
            classification=ClassificationSummary(**data["classification"]),
            feature_importance=dict(data.get("feature_importance", {})),
            compared_to_baseline=BaselineComparison(**baseline) if baseline else None,
        )
