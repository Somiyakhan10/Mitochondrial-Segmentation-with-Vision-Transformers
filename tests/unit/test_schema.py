from __future__ import annotations

from mitomorph.data.schema import (
    AnalysisResult,
    BaselineComparison,
    ClassificationSummary,
    ImageInfo,
    MorphometricSummary,
    SegmentationSummary,
)
from mitomorph.data.temporal import TimeSeriesData


def _make_result(analysis_id="A1", time_point="6 weeks", health_score=52.3) -> AnalysisResult:
    return AnalysisResult(
        analysis_id=analysis_id,
        image_info=ImageInfo(filename="m.tif", experiment="SCI 6 weeks", animal_id="M123", time_point=time_point),
        segmentation=SegmentationSummary(
            total_mitochondria=342, neuronal_mitochondria=187, non_neuronal_mitochondria=155,
            segmentation_confidence=0.92,
        ),
        morphometric_summary=MorphometricSummary(
            mean_area=0.85, mean_aspect_ratio=2.3, mean_circularity=0.67,
            fragmentation_index=4.2, mitochondrial_density=0.34, network_size=12.5,
        ),
        classification=ClassificationSummary(
            healthy=25, fragmented=45, swollen=18, dysfunctional=12, health_score=health_score
        ),
        feature_importance={"aspect_ratio": 0.35},
        compared_to_baseline=BaselineComparison(condition="SCI vs Naive", significance="p < 0.01", change_percent=-32.5),
    )


def test_analysis_result_roundtrip():
    result = _make_result()
    restored = AnalysisResult.from_dict(result.to_dict())
    assert restored == result


def test_analysis_result_roundtrip_no_baseline():
    result = _make_result()
    result.compared_to_baseline = None
    restored = AnalysisResult.from_dict(result.to_dict())
    assert restored.compared_to_baseline is None


def test_timeseries_sorts_time_points_numerically():
    series = TimeSeriesData(animal_id="M123", condition="SCI")
    series.add_measurement("21 weeks", _make_result(analysis_id="A2", time_point="21 weeks", health_score=30.0))
    series.add_measurement("6 weeks", _make_result(analysis_id="A1", time_point="6 weeks", health_score=52.3))
    assert series.time_points() == ["6 weeks", "21 weeks"]


def test_health_score_trend_order():
    series = TimeSeriesData(animal_id="M123", condition="SCI")
    series.add_measurement("21 weeks", _make_result(analysis_id="A2", time_point="21 weeks", health_score=30.0))
    series.add_measurement("6 weeks", _make_result(analysis_id="A1", time_point="6 weeks", health_score=52.3))
    trend = series.health_score_trend()
    assert list(trend.keys()) == ["6 weeks", "21 weeks"]
    assert trend["6 weeks"] == 52.3
