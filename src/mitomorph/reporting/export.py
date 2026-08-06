"""Export analysis results to CSV/Excel/JSON (FR-43)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mitomorph.data.schema import AnalysisResult


def _flatten(result: AnalysisResult) -> dict[str, Any]:
    d = result.to_dict()
    flat: dict[str, Any] = {"analysis_id": d["analysis_id"]}
    for section in ("image_info", "segmentation", "morphometric_summary", "classification"):
        for k, v in d[section].items():
            flat[f"{section}.{k}"] = v
    for k, v in d.get("feature_importance", {}).items():
        flat[f"feature_importance.{k}"] = v
    if d.get("compared_to_baseline"):
        for k, v in d["compared_to_baseline"].items():
            flat[f"compared_to_baseline.{k}"] = v
    return flat


def to_json(results: list[AnalysisResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8")


def to_csv(results: list[AnalysisResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_flatten(r) for r in results]).to_csv(path, index=False)


def to_excel(results: list[AnalysisResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_flatten(r) for r in results]).to_excel(path, index=False)
