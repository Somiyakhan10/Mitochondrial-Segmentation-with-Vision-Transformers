"""Client interface for integrating with an existing lab database (FR-44, Low priority)."""

from __future__ import annotations

from typing import Any

from mitomorph.data.schema import AnalysisResult


class LabDatabaseClient:
    """Client interface for pushing/pulling analysis results to/from a lab's existing database.

    No lab database endpoint has been specified yet; this defines the
    intended interface so a real client can be swapped in later.
    """

    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url
        self.api_key = api_key

    def push_result(self, result: AnalysisResult) -> None:
        raise NotImplementedError("Lab database endpoint not yet specified (FR-44, Low priority)")

    def fetch_metadata(self, animal_id: str) -> dict[str, Any]:
        raise NotImplementedError("Lab database endpoint not yet specified (FR-44, Low priority)")
