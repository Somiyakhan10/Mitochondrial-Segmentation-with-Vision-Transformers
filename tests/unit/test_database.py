from __future__ import annotations

from mitomorph.data.database import AnalysisDatabase
from mitomorph.data.schema import (
    AnalysisResult,
    ClassificationSummary,
    ImageInfo,
    MorphometricSummary,
    SegmentationSummary,
)


def _make_result(analysis_id: str, animal_id: str = "M123", time_point: str = "6 weeks") -> AnalysisResult:
    return AnalysisResult(
        analysis_id=analysis_id,
        image_info=ImageInfo(filename="m.tif", experiment="SCI", animal_id=animal_id, time_point=time_point),
        segmentation=SegmentationSummary(
            total_mitochondria=10,
            neuronal_mitochondria=5,
            non_neuronal_mitochondria=5,
            segmentation_confidence=0.9,
        ),
        morphometric_summary=MorphometricSummary(
            mean_area=1.0,
            mean_aspect_ratio=1.0,
            mean_circularity=1.0,
            fragmentation_index=1.0,
            mitochondrial_density=1.0,
            network_size=1.0,
        ),
        classification=ClassificationSummary(
            healthy=1, fragmented=1, swollen=1, dysfunctional=1, health_score=50.0
        ),
    )


def test_insert_and_get(tmp_path):
    db = AnalysisDatabase(tmp_path / "test.db")
    result = _make_result("A1")
    db.insert(result)
    assert db.get("A1") == result
    db.close()


def test_get_missing_returns_none(tmp_path):
    db = AnalysisDatabase(tmp_path / "test.db")
    assert db.get("does-not-exist") is None
    db.close()


def test_list_by_animal(tmp_path):
    db = AnalysisDatabase(tmp_path / "test.db")
    db.insert(_make_result("A1", animal_id="M1", time_point="6 weeks"))
    db.insert(_make_result("A2", animal_id="M1", time_point="21 weeks"))
    db.insert(_make_result("A3", animal_id="M2"))
    assert {r.analysis_id for r in db.list_by_animal("M1")} == {"A1", "A2"}
    db.close()


def test_delete(tmp_path):
    db = AnalysisDatabase(tmp_path / "test.db")
    db.insert(_make_result("A1"))
    db.delete("A1")
    assert db.get("A1") is None
    db.close()


def test_context_manager(tmp_path):
    with AnalysisDatabase(tmp_path / "test.db") as db:
        db.insert(_make_result("A1"))
        assert db.get("A1") is not None


def test_insert_or_replace(tmp_path):
    db = AnalysisDatabase(tmp_path / "test.db")
    db.insert(_make_result("A1", time_point="6 weeks"))
    db.insert(_make_result("A1", time_point="21 weeks"))
    assert db.get("A1").image_info.time_point == "21 weeks"
    db.close()
