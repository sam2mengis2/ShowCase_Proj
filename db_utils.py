"""
db_utils.py — standalone helpers for inspecting / managing the SQLite DB.
Run directly:  python db_utils.py
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "tsionvision.db"


def init_db(path: Path = DB_PATH):
    """Create tables. Safe to call multiple times (CREATE IF NOT EXISTS)."""
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS scans (
                id              TEXT PRIMARY KEY,
                created_at      TEXT NOT NULL,

                patient_id      TEXT,
                patient_name    TEXT,
                patient_dob     TEXT,
                patient_sex     TEXT,
                study_date      TEXT,
                study_time      TEXT,
                modality        TEXT,
                institution     TEXT,
                study_desc      TEXT,
                series_desc     TEXT,
                manufacturer    TEXT,
                rows            INTEGER,
                cols            INTEGER,

                dicom_meta_json TEXT,

                clinical_notes  TEXT,
                file_path       TEXT,
                file_name       TEXT,

                prediction      TEXT,
                confidence      REAL,
                all_probs_json  TEXT,
                inference_at    TEXT,
                inference_error TEXT
            );
        """)
    print(f"DB ready at {path}")


def list_scans(path: Path = DB_PATH, limit: int = 20):
    """Print a summary table of recent scans."""
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, created_at, patient_id, modality, prediction, confidence "
            "FROM scans ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    if not rows:
        print("No scans in database yet.")
        return
    print(f"{'ID':36}  {'Created':20}  {'Patient':12}  {'Mod':5}  {'Prediction':12}  Conf")
    print("-" * 95)
    for r in rows:
        conf = f"{r['confidence']*100:.1f}%" if r["confidence"] is not None else "—"
        print(
            f"{r['id']}  {r['created_at'][:19]}  "
            f"{(r['patient_id'] or '—'):12}  {(r['modality'] or '—'):5}  "
            f"{(r['prediction'] or 'pending'):12}  {conf}"
        )


def get_scan(scan_id: str, path: Path = DB_PATH):
    """Pretty-print a single scan record."""
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    if row is None:
        print(f"Scan {scan_id!r} not found.")
        return
    data = dict(row)
    # Pretty-print nested JSON fields
    for key in ("dicom_meta_json", "all_probs_json"):
        if data.get(key):
            try:
                data[key] = json.loads(data[key])
            except Exception:
                pass
    print(json.dumps(data, indent=2, default=str))


def delete_scan(scan_id: str, path: Path = DB_PATH):
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
    print(f"Deleted scan {scan_id}")


if __name__ == "__main__":
    init_db()
    print()
    list_scans()
