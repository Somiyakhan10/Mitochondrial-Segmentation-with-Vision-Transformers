"""SQLite-backed metadata store for analyzed images (FR-41)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mitomorph.data.schema import AnalysisResult
from mitomorph.exceptions import DatabaseError

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    experiment TEXT,
    animal_id TEXT,
    time_point TEXT,
    result_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class AnalysisDatabase:
    """Thin wrapper around a SQLite database of analysis results and metadata."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(SCHEMA)
        self._conn.commit()

    def insert(self, result: AnalysisResult) -> None:
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO analyses "
                "(analysis_id, filename, experiment, animal_id, time_point, result_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    result.analysis_id,
                    result.image_info.filename,
                    result.image_info.experiment,
                    result.image_info.animal_id,
                    result.image_info.time_point,
                    json.dumps(result.to_dict()),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to insert analysis {result.analysis_id}: {exc}") from exc

    def get(self, analysis_id: str) -> AnalysisResult | None:
        row = self._conn.execute(
            "SELECT result_json FROM analyses WHERE analysis_id = ?", (analysis_id,)
        ).fetchone()
        return AnalysisResult.from_dict(json.loads(row[0])) if row else None

    def list_by_animal(self, animal_id: str) -> list[AnalysisResult]:
        rows = self._conn.execute(
            "SELECT result_json FROM analyses WHERE animal_id = ? ORDER BY created_at", (animal_id,)
        ).fetchall()
        return [AnalysisResult.from_dict(json.loads(r[0])) for r in rows]

    def delete(self, analysis_id: str) -> None:
        self._conn.execute("DELETE FROM analyses WHERE analysis_id = ?", (analysis_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> AnalysisDatabase:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
