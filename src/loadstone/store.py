"""Store: a local SQLite index over studies.

Indexing is a convenience only. Every number stays readable from the study
directory itself (JSON manifests, JSONL rows, CSV summaries) without the DB.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .util import read_json, utc_now

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
    study_id TEXT PRIMARY KEY,
    protocol_id TEXT,
    protocol_sha256 TEXT,
    suite TEXT,
    comparison_regime TEXT,
    device TEXT,
    precision TEXT,
    path TEXT,
    status TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT,
    study_id TEXT,
    model_id TEXT,
    stage TEXT,
    seed INTEGER,
    learning_rate REAL,
    status TEXT,
    updates_completed INTEGER,
    cell_macro_query_accuracy REAL,
    elapsed_s REAL,
    summary_path TEXT,
    PRIMARY KEY (job_id, study_id)
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    return con


def index_study(study_dir: str | Path, db_path: str | Path) -> int:
    study_dir = Path(study_dir)
    resolved = read_json(study_dir / "protocol.resolved.json")
    con = connect(db_path)
    study_id = resolved.get("study_id") or study_dir.name
    con.execute(
        "INSERT OR REPLACE INTO studies VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            study_id,
            resolved.get("protocol_id"),
            resolved.get("protocol_sha256"),
            resolved.get("suite"),
            resolved.get("comparison_regime"),
            resolved.get("environment", {}).get("device"),
            resolved.get("precision"),
            str(study_dir),
            resolved.get("status"),
            utc_now(),
        ),
    )
    n = 0
    for summary_path in sorted((study_dir / "jobs").glob("*/summary.json")):
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        con.execute(
            "INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                s["job_id"],
                study_id,
                s.get("model_id"),
                s.get("stage"),
                s.get("seed"),
                s.get("learning_rate"),
                s.get("status"),
                s.get("updates_completed"),
                (s.get("final_validation") or {}).get("cell_macro_query_accuracy"),
                s.get("elapsed_s"),
                str(summary_path),
            ),
        )
        n += 1
    con.commit()
    con.close()
    return n


def list_studies(db_path: str | Path) -> list[dict]:
    if not Path(db_path).exists():
        return []
    con = connect(db_path)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM studies ORDER BY created_at DESC")]
    con.close()
    return rows


def study_jobs(db_path: str | Path, study_id: str) -> list[dict]:
    con = connect(db_path)
    con.row_factory = sqlite3.Row
    rows = [
        dict(r)
        for r in con.execute(
            "SELECT * FROM jobs WHERE study_id = ? ORDER BY stage, model_id, seed", (study_id,)
        )
    ]
    con.close()
    return rows
