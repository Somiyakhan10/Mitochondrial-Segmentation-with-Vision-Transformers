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

CREATE TABLE IF NOT EXISTS segmentation_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    filename TEXT NOT NULL,
    animal_id TEXT,
    condition TEXT,
    time_point TEXT,
    region_count INTEGER NOT NULL,
    mean_area REAL,
    fragmentation_index REAL NOT NULL,
    mitochondrial_density REAL NOT NULL,
    overlay_path TEXT NOT NULL,
    data_path TEXT NOT NULL,
    corrected INTEGER NOT NULL DEFAULT 0
);
"""


class AnalysisDatabase:
    """Thin wrapper around a SQLite database of analysis results and metadata.

    ``segmentation_runs`` stores what the dashboard's Analyze tab can
    actually produce today (segmentation + morphometrics only) — distinct
    from ``analyses``/:class:`~mitomorph.data.schema.AnalysisResult`,
    which mirrors the full SRS §5.2 schema (cell-type + health
    classification included) and is populated once those stages are real.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
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

    def insert_segmentation_run(self, **fields: object) -> int:
        """Insert a segmentation-run record. Returns the new ``run_id``."""
        columns = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        try:
            cursor = self._conn.execute(
                f"INSERT INTO segmentation_runs ({columns}) VALUES ({placeholders})",
                tuple(fields.values()),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise DatabaseError(f"Failed to insert segmentation run: {exc}") from exc
        return int(cursor.lastrowid)

    def list_segmentation_runs(self) -> list[dict]:
        # created_at has only second resolution, so run_id (monotonic) breaks ties deterministically.
        rows = self._conn.execute(
            "SELECT * FROM segmentation_runs ORDER BY created_at DESC, run_id DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def get_segmentation_run(self, run_id: int) -> dict | None:
        row = self._conn.execute("SELECT * FROM segmentation_runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def update_segmentation_run(self, run_id: int, **fields: object) -> None:
        set_clause = ", ".join(f"{key} = ?" for key in fields)
        self._conn.execute(
            f"UPDATE segmentation_runs SET {set_clause} WHERE run_id = ?",
            (*fields.values(), run_id),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> AnalysisDatabase:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
