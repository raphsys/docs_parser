#!/usr/bin/env python3
"""Audit PAGEPRINT -> PAGETRANSLATE on random real PDF pages.

For each randomly selected page:
  - run the pipeline (PDF -> PAGEPRINT input_data),
  - run PAGETRANSLATE with the chosen engine,
  - render the page with PAGEPRINT bboxes overlaid,
  - write a side-by-side extraction/translation report,
  - collect functional audit metrics.

Outputs (in --out):
  input_data_pNNN.json, pagetranslate_result_pNNN.json,
  pageprint_bboxes_pNNN.png, extraction_vs_translation_pNNN.md,
  src_<doc>_pNNN.png, audit_compact.json, README.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
from translation_engines import create_translation_engine


# --- heuristic detectors (clearly approximate, used only for the audit) -------
COMMAND_RE = re.compile(r"^\s*(dir|copy|del|findstr|cd|ls|cp|mv|rm|cat|grep|mkdir|pwd|chmod|git|pip|python3?|sudo)\b", re.IGNORECASE)
PATH_RE = re.compile(r"([A-Za-z]:\\[^\s]+|/[\w.][\w./-]+|\*\.\w+)")
SQL_RE = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|COMMIT|ROLLBACK|START TRANSACTION|CREATE TABLE|DROP TABLE)\b")
PUBLISHER_RE = re.compile(r"(Estad[ií]sticos|e-?Books?\s*&\s*Papers|All rights reserved|©|\bISBN\b)", re.IGNORECASE)
FUNCTION_WORDS = re.compile(r"\b(the|and|or|to|of|in|for|with|that|this|you|will|can|is|are|a|an)\b", re.IGNORECASE)
TABLE_HINT_RE = re.compile(r"\b(Table\s+\d|Command\s+Function|Total number|True positives|Precision|Recall)\b")
INDEX_HINT_RE = re.compile(r"[A-Za-z].*,\s*\d{2,3}(?:[–-]\d{2,3})?(?:,\s*\d{2,3})*\s*$")

ROLE_COLORS = {
    "title": (220, 30, 30), "section_heading": (220, 90, 30), "subsection_heading": (220, 140, 30),
    "body_paragraph": (30, 120, 220), "body": (30, 120, 220), "list_item": (30, 170, 120),
    "figure_caption": (160, 60, 200), "table_caption": (160, 60, 200), "table_body_cell": (200, 160, 30),
    "code_block": (90, 90, 90), "code_line": (90, 90, 90), "index_entry": (30, 160, 160),
    "page_footer": (150, 150, 150), "page_header": (150, 150, 150), "publisher_mark": (120, 0, 0),
}
LEVEL_COLORS = {"block": (0, 90, 200), "line": (0, 160, 90), "phrase": (200, 120, 0), "region": (200, 0, 120)}


def _looks_natural(text: str) -> bool:
    text = (text or "").strip()
    words = re.findall(r"[A-Za-z]{3,}", text)
    return len(words) >= 6 and len(FUNCTION_WORDS.findall(text)) >= 2


def _color_for(role, level):
    if role and role in ROLE_COLORS:
        return ROLE_COLORS[role]
    return LEVEL_COLORS.get(level, (255, 0, 0))


def _scale(geom):
    sx = geom.get("scale_x_px_per_pt")
    sy = geom.get("scale_y_px_per_pt")
    if sx and sy:
        return float(sx), float(sy)
    w, h = geom.get("width"), geom.get("height")
    rw, rh = geom.get("render_width_px"), geom.get("render_height_px")
    if w and h and rw and rh:
        return rw / w, rh / h
    return 1.0, 1.0


def render_bboxes(input_data: dict, image_path: Path, out_path: Path) -> bool:
    if not image_path or not Path(image_path).is_file():
        return False
    geom = (input_data.get("page") or {}).get("geometry") or {}
    sx, sy = _scale(geom)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # Draw block/line/phrase/region units that carry geometry.
    drawn = 0
    for unit in input_data.get("units") or []:
        level = unit.get("level")
        if level not in {"block", "line", "phrase", "region"}:
            continue
        bbox = (unit.get("geometry") or {}).get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        role = (unit.get("understanding") or {}).get("role")
        color = _color_for(role, level)
        x0, y0, x1, y1 = bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy
        width = 3 if level == "block" else (2 if level == "line" else 1)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        drawn += 1
    # Protected/special regions in red dashes-like thick border.
    for region in input_data.get("regions") or []:
        bbox = region.get("bbox") or (region.get("geometry") or {}).get("bbox")
        rtype = region.get("region_type") or region.get("type") or ""
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        if "protected" in str(rtype):
            x0, y0, x1, y1 = bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 255), width=2)
    img.save(out_path)
    return True


def page_audit(input_data: dict, trial: dict) -> dict:
    units = trial.get("translation_units") or []
    role_none = sum(1 for u in units if not (u.get("role") or u.get("understanding", {}).get("role")))
    ss = input_data.get("semantic_system") or {}
    phrases = len(ss.get("semantic_phrases") or [])
    groups = len(ss.get("semantic_groups") or [])
    debug = trial.get("debug") or {}
    quality = trial.get("quality") or {}
    needs_review = int(quality.get("needs_review_count") or 0)
    generic_coalesced = sum(1 for u in units if len(u.get("source_unit_ids") or []) > 1) if debug.get("generic_coalescer_used") else 0

    publisher_sent = code_sent = natural_protected = 0
    table_text_present = index_text_present = False
    for u in units:
        src = u.get("source_text") or u.get("text") or ""
        if PUBLISHER_RE.search(src):
            publisher_sent += 1
        if COMMAND_RE.search(src) or PATH_RE.search(src) or SQL_RE.search(src):
            code_sent += 1
        if TABLE_HINT_RE.search(src):
            table_text_present = True
        if INDEX_HINT_RE.search(src):
            index_text_present = True
    for u in input_data.get("views", {}).get("protected_visual_units") or []:
        if _looks_natural(u.get("text") or u.get("source_text") or ""):
            natural_protected += 1

    # table/index region detection from pageprint
    region_types = {str(r.get("region_type") or r.get("type") or "") for r in input_data.get("regions") or []}
    has_table_region = any("table" in t for t in region_types) or bool((input_data.get("indexes") or {}).get("tables"))
    page_role = (input_data.get("page") or {}).get("page_role") or (input_data.get("page_intelligence") or {}).get("page_role")
    has_index_role = "index" in str(page_role or "").lower()

    return {
        "translation_unit_count": len(units),
        "role_none": role_none,
        "semantic_phrases": phrases,
        "semantic_groups": groups,
        "semantic_system_empty": phrases == 0 and groups == 0,
        "needs_review": needs_review,
        "generic_coalesced_units": generic_coalesced,
        "fallback_selector_used": bool(debug.get("fallback_selector_used")),
        "selection_mode": debug.get("selection_mode"),
        "page_role": page_role,
        "table_text_present": table_text_present,
        "table_region_detected": has_table_region,
        "table_false_negative": bool(table_text_present and not has_table_region),
        "index_text_present": index_text_present,
        "index_role_detected": has_index_role,
        "index_false_negative": bool(index_text_present and not has_index_role),
        "publisher_mark_sent": publisher_sent,
        "code_or_command_sent": code_sent,
        "natural_text_marked_protected": natural_protected,
        "pipeline_status": trial.get("pipeline_status"),
        "translation_runtime_status": trial.get("translation_runtime_status"),
        "linguistic_quality_status": trial.get("linguistic_quality_status"),
        "publication_readiness_status": trial.get("publication_readiness_status"),
    }


def write_side_by_side(units: list[dict], out_path: Path, page_label: str) -> None:
    lines = [f"# Extraction PAGEPRINT vs Traduction PAGETRANSLATE — {page_label}", ""]
    lines.append("| # | role | strategy | status | review | source (PAGEPRINT) | traduction (PAGETRANSLATE) |")
    lines.append("|---|------|----------|--------|--------|--------------------|----------------------------|")
    for i, u in enumerate(units, 1):
        role = u.get("role") or (u.get("understanding") or {}).get("role") or "—"
        strat = u.get("strategy") or u.get("translation_strategy") or "—"
        status = u.get("status") or "—"
        review = "⚠️" if (u.get("quality") or {}).get("needs_review") else ""
        src = (u.get("source_text") or u.get("text") or "").replace("|", "\\|").replace("\n", " ")[:160]
        tgt = (u.get("translated_text") or "").replace("|", "\\|").replace("\n", " ")[:160]
        lines.append(f"| {i} | {role} | {strat} | {status} | {review} | {src} | {tgt} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_pages(pdf_dir: Path, count: int, seed: int, min_pages: int) -> list[tuple[Path, int]]:
    pdfs = []
    for p in sorted(pdf_dir.glob("*.pdf")):
        try:
            doc = fitz.open(p)
            n = doc.page_count
            doc.close()
        except Exception:
            continue
        if n >= min_pages:
            pdfs.append((p, n))
    if not pdfs:
        # fall back to any pdf
        for p in sorted(pdf_dir.glob("*.pdf")):
            try:
                doc = fitz.open(p); n = doc.page_count; doc.close()
            except Exception:
                continue
            pdfs.append((p, n))
    rng = random.Random(seed)
    picks = []
    attempts = 0
    seen = set()
    while len(picks) < count and attempts < count * 50 and pdfs:
        pdf, n = rng.choice(pdfs)
        page = rng.randint(1, n)
        key = (str(pdf), page)
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        picks.append((pdf, page))
        attempts += 1
    return picks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="tests/doc_pdf")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--min-pages", type=int, default=20, help="Only sample PDFs with at least this many pages.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--engine", default="ct2")
    parser.add_argument("--inventory", default="ai_models/translation/model_inventory.json")
    parser.add_argument("--model", default="opus_mt_tc_big_en_fr")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="fr")
    parser.add_argument("--enable-ocr", action="store_true")
    args = parser.parse_args()

    out = Path(args.out or f"ocr_results/pageprint_pagetranslate_audit_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_pages").mkdir(exist_ok=True)

    picks = collect_pages(Path(args.pdf_dir), args.count, args.seed, args.min_pages)
    orchestrator = PipelineOrchestrator(
        enable_ocr=args.enable_ocr, enable_understanding=True,
        enable_postprocessors=True, enable_special_regions=True,
        save_render_dir=str(out / "source_pages"),
    )
    engine = create_translation_engine(
        args.engine, inventory_path=args.inventory, model_name=args.model,
        source_lang=args.source_lang, target_lang=args.target_lang,
    )

    per_page = []
    for idx, (pdf, page) in enumerate(picks, 1):
        tag = f"{pdf.stem[:24]}_p{page:04d}"
        started = time.perf_counter()
        try:
            doc_result = orchestrator.run(str(pdf), pages=str(page),
                                          language={"source_lang": args.source_lang, "target_lang": args.target_lang})
            page_entries = [p for p in (doc_result.get("pages") or []) if p.get("status") == "ok"]
            if not page_entries:
                per_page.append({"tag": tag, "pdf": pdf.name, "page": page, "error": "extraction_failed"})
                print(f"[{idx}/{len(picks)}] {tag}: extraction KO", flush=True)
                continue
            input_data = page_entries[0]["input_data"]
            trial = build_page_translation(
                input_data, translator=engine,
                target_lang=args.target_lang, source_lang=args.source_lang,
                allow_fallback=True,
            )
        except Exception as exc:
            per_page.append({"tag": tag, "pdf": pdf.name, "page": page, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{idx}/{len(picks)}] {tag}: ERROR {exc}", flush=True)
            continue

        (out / f"input_data_{tag}.json").write_text(json.dumps(input_data, ensure_ascii=False, indent=2), encoding="utf-8")
        units = trial.get("translation_units") or []
        result_compact = {
            "pdf": pdf.name, "page": page,
            "statuses": {k: trial.get(k) for k in ("pipeline_status", "translation_runtime_status", "linguistic_quality_status", "publication_readiness_status")},
            "debug": trial.get("debug"),
            "units": [{
                "unit_id": u.get("unit_id"), "role": u.get("role") or (u.get("understanding") or {}).get("role"),
                "strategy": u.get("strategy") or u.get("translation_strategy"),
                "status": u.get("status"), "needs_review": (u.get("quality") or {}).get("needs_review"),
                "source_text": u.get("source_text") or u.get("text"),
                "translated_text": u.get("translated_text"),
                "protected": u.get("protected") or [],
            } for u in units],
        }
        (out / f"pagetranslate_result_{tag}.json").write_text(json.dumps(result_compact, ensure_ascii=False, indent=2), encoding="utf-8")
        write_side_by_side(units, out / f"extraction_vs_translation_{tag}.md", f"{pdf.name} p{page}")
        img_path = (input_data.get("assets") or {}).get("source_image_path")
        render_bboxes(input_data, Path(img_path) if img_path else None, out / f"pageprint_bboxes_{tag}.png")

        metrics = page_audit(input_data, trial)
        metrics.update({"tag": tag, "pdf": pdf.name, "page": page, "duration_s": round(time.perf_counter() - started, 1)})
        per_page.append(metrics)
        print(f"[{idx}/{len(picks)}] {tag}: units={metrics['translation_unit_count']} role_none={metrics['role_none']} "
              f"sem_empty={metrics['semantic_system_empty']} review={metrics['needs_review']} ({metrics['duration_s']}s)", flush=True)

    ok_pages = [p for p in per_page if "error" not in p]
    def s(key):
        return sum(int(p.get(key) or 0) for p in ok_pages)
    critical = {
        "pages_audited": len(per_page),
        "pages_ok": len(ok_pages),
        "translation_units_total": s("translation_unit_count"),
        "role_none_translation_units": s("role_none"),
        "semantic_system_empty_pages": sum(1 for p in ok_pages if p.get("semantic_system_empty")),
        "needs_review_total": s("needs_review"),
        "generic_coalesced_units": s("generic_coalesced_units"),
        "natural_text_marked_protected": s("natural_text_marked_protected"),
        "table_false_negative_pages": sum(1 for p in ok_pages if p.get("table_false_negative")),
        "index_false_negative_pages": sum(1 for p in ok_pages if p.get("index_false_negative")),
        "publisher_mark_sent_to_translation": s("publisher_mark_sent"),
        "code_or_command_sent_to_translation": s("code_or_command_sent"),
    }
    functional_ok = (
        critical["role_none_translation_units"] == 0
        and critical["semantic_system_empty_pages"] == 0
        and critical["table_false_negative_pages"] == 0
        and critical["index_false_negative_pages"] == 0
        and critical["publisher_mark_sent_to_translation"] == 0
        and critical["code_or_command_sent_to_translation"] == 0
        and critical["natural_text_marked_protected"] == 0
    )
    audit = {
        "schema_status": "ok" if ok_pages else "ko",
        "functional_status": "ok" if functional_ok else "ko",
        "engine": args.engine, "model": args.model,
        "critical_counts": critical,
        "pages": per_page,
    }
    (out / "audit_compact.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_readme(out, audit)
    print("\n=== AUDIT ===")
    print(json.dumps({"functional_status": audit["functional_status"], **critical}, ensure_ascii=False, indent=2))
    print("Output:", out)
    return 0


def _write_readme(out: Path, audit: dict) -> None:
    c = audit["critical_counts"]
    lines = [
        f"# Audit PAGEPRINT → PAGETRANSLATE", "",
        f"- moteur: `{audit['engine']}` / modèle `{audit['model']}`",
        f"- schema_status: **{audit['schema_status']}**",
        f"- functional_status: **{audit['functional_status']}**", "",
        "## Compteurs critiques", "",
        "| métrique | valeur |", "|---|---|",
    ]
    for k, v in c.items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Pages", "", "| page | units | role_none | sem_empty | review | table_FN | index_FN | pub | code | dur(s) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for p in audit["pages"]:
        if "error" in p:
            lines.append(f"| {p['tag']} | ERROR: {p['error']} |  |  |  |  |  |  |  |  |")
            continue
        lines.append(f"| {p['tag']} | {p['translation_unit_count']} | {p['role_none']} | {p['semantic_system_empty']} | "
                     f"{p['needs_review']} | {p['table_false_negative']} | {p['index_false_negative']} | "
                     f"{p['publisher_mark_sent']} | {p['code_or_command_sent']} | {p['duration_s']} |")
    lines += ["", "## Fichiers par page",
              "- `input_data_<tag>.json` — sortie PAGEPRINT complète",
              "- `pagetranslate_result_<tag>.json` — unités source↔traduction",
              "- `extraction_vs_translation_<tag>.md` — tableau côte à côte",
              "- `pageprint_bboxes_<tag>.png` — page annotée (bboxes par rôle/niveau)",
              "- `source_pages/` — pages source rendues"]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
