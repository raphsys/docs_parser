#!/usr/bin/env python3
"""Ingest a PAGEPRINT->PAGETRANSLATE audit run into a SQLite database.

Builds two tables (pages, elements) consumed by the Streamlit dashboard, and
copies the per-page assets (rendered page, bbox overlay, background if any).

Manual edits made in the dashboard are preserved across re-ingest: rows flagged
``edited=1`` keep their ``translatable``/``translation``/``role`` values
(matched by page_tag + source_text).
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.pipeline_dashboard.elements import build_elements, page_meta

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    page_tag TEXT PRIMARY KEY, pdf TEXT, page_num INTEGER, page_role TEXT,
    source_image TEXT, bboxes_image TEXT, background TEXT, background_image TEXT,
    runtime_status TEXT, quality_status TEXT, publication_status TEXT,
    selection_health TEXT, input_data_path TEXT
);
CREATE TABLE IF NOT EXISTS elements (
    id INTEGER PRIMARY KEY AUTOINCREMENT, page_tag TEXT, ord INTEGER,
    category TEXT, granularity TEXT, role TEXT, object_type TEXT,
    translatable INTEGER, source_text TEXT, translation TEXT, status TEXT,
    needs_review INTEGER, qa_reasons TEXT, bbox TEXT, reason TEXT,
    edited INTEGER DEFAULT 0, updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_elements_page ON elements(page_tag);
"""


def _load_edits(con: sqlite3.Connection) -> dict:
    edits = {}
    try:
        for row in con.execute("SELECT page_tag, source_text, translatable, translation, role FROM elements WHERE edited=1"):
            edits[(row[0], row[1])] = {"translatable": row[2], "translation": row[3], "role": row[4]}
    except sqlite3.OperationalError:
        pass
    return edits


def ingest_dir(audit_dir, db_path=None, *, source_lang: str = "en", target_lang: str = "fr") -> dict:
    """Ingest every input_data_*.json in audit_dir into a SQLite DB. Callable."""
    audit_dir = Path(audit_dir)
    db_path = Path(db_path) if db_path else audit_dir / "pipeline_dashboard.db"
    assets_dir = audit_dir / "dashboard_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)
    edits = _load_edits(con)
    # Rebuild from scratch (data is derived) so schema changes always apply.
    con.execute("DROP TABLE IF EXISTS pages")
    con.execute("DROP TABLE IF EXISTS elements")
    con.executescript(SCHEMA)

    files = sorted(glob.glob(str(audit_dir / "input_data_*.json")))
    now = datetime.now().isoformat(timespec="seconds")
    for path in files:
        tag = Path(path).stem.replace("input_data_", "")
        input_data = json.loads(Path(path).read_text(encoding="utf-8"))
        pt_path = audit_dir / f"pagetranslate_result_{tag}.json"
        pt_result = json.loads(pt_path.read_text(encoding="utf-8")) if pt_path.is_file() else {}
        meta = page_meta(input_data, pt_result)

        # copy assets
        page_assets = assets_dir / tag
        page_assets.mkdir(parents=True, exist_ok=True)
        src_img = meta.get("source_image")
        src_copy = ""
        if src_img and Path(src_img).is_file():
            dest = page_assets / f"page_{Path(src_img).name}"
            shutil.copy2(src_img, dest)
            src_copy = str(dest)
        bbox_src = audit_dir / f"pageprint_bboxes_{tag}.png"
        bbox_copy = ""
        if bbox_src.is_file():
            dest = page_assets / f"bboxes_{tag}.png"
            shutil.copy2(bbox_src, dest)
            bbox_copy = str(dest)
        # Background trame: real exported layer if any, else the synthesised
        # 'text masked' preview produced by run_page (background_<tag>.png).
        bg_image = ""
        if meta.get("background_path") and Path(meta["background_path"]).is_file():
            dest = page_assets / f"background_{Path(meta['background_path']).name}"
            shutil.copy2(meta["background_path"], dest)
            bg_image = str(dest)
        else:
            synth = audit_dir / f"background_{tag}.png"
            if synth.is_file():
                dest = page_assets / f"background_{tag}.png"
                shutil.copy2(synth, dest)
                bg_image = str(dest)

        con.execute(
            "INSERT OR REPLACE INTO pages VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (tag, (pt_result or {}).get("pdf"), (pt_result or {}).get("page"), meta["page_role"],
             src_copy, bbox_copy, ("oui" if meta["background"] else "non"), bg_image,
             meta["runtime_status"], meta["quality_status"], meta["publication_status"], meta["selection_health"],
             str(Path(path).resolve())),
        )

        for el in build_elements(input_data, pt_result, source_lang=source_lang, target_lang=target_lang):
            key = (tag, el["source_text"])
            edited = 0
            translatable = int(el["translatable"])
            translation = el["translation"]
            role = el["role"]
            if key in edits:
                e = edits[key]
                translatable = int(e["translatable"]) if e["translatable"] is not None else translatable
                translation = e["translation"] if e["translation"] is not None else translation
                role = e["role"] if e["role"] is not None else role
                edited = 1
            con.execute(
                "INSERT INTO elements (page_tag, ord, category, granularity, role, object_type, translatable, "
                "source_text, translation, status, needs_review, qa_reasons, bbox, reason, edited, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tag, el["ord"], el["category"], el["granularity"], role, el["object_type"], translatable,
                 el["source_text"], translation, el["status"], int(el["needs_review"]), el["qa_reasons"],
                 el["bbox"], el["reason"], edited, now),
            )

    con.commit()
    n_pages = con.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
    n_el = con.execute("SELECT COUNT(*) FROM elements").fetchone()[0]
    con.close()
    return {"db_path": str(db_path), "assets_dir": str(assets_dir),
            "pages": n_pages, "elements": n_el, "edits_preserved": len(edits)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--db", default=None)
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="fr")
    args = parser.parse_args()
    res = ingest_dir(args.audit_dir, args.db, source_lang=args.source_lang, target_lang=args.target_lang)
    print(f"DB : {res['db_path']}")
    print(f"Assets : {res['assets_dir']}")
    print(f"Pages : {res['pages']} | Éléments : {res['elements']} | Éditions préservées : {res['edits_preserved']}")
    print(f"\nLancer le dashboard :\n  streamlit run tools/pipeline_dashboard/app.py -- --db {res['db_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
