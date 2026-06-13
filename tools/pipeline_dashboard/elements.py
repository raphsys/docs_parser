"""Build flat element records for the pipeline dashboard.

One record per extracted element of a page, joining PAGEPRINT output
(granularity, role, type, translatable) with PAGETRANSLATE output (translation,
status, QA reasons). Textual and non-textual (formula/table/figure/region)
elements are unified.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_translation_selection import _structural, audit_page
from pagetranslate.quality import unit_quality


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip().lower()


def _translation_map(pt_result: dict) -> dict:
    return {_norm(u.get("source_text")): u for u in (pt_result or {}).get("units") or []}


def build_elements(input_data: dict, pt_result: dict, *, source_lang: str = "en", target_lang: str = "fr") -> list[dict]:
    audit = audit_page(input_data)
    st = audit["structural"]
    tmap = _translation_map(pt_result)
    profile = {"source_lang": source_lang, "target_lang": target_lang}
    rows: list[dict] = []
    n = 0

    def add(**kw):
        nonlocal n
        n += 1
        rows.append({
            "ord": n,
            "category": kw.get("category"),
            "granularity": kw.get("granularity") or "—",
            "role": kw.get("role"),
            "object_type": kw.get("object_type"),
            "translatable": bool(kw.get("translatable")),
            "source_text": kw.get("source_text") or "",
            "translation": kw.get("translation") or "",
            "status": kw.get("status") or "",
            "needs_review": bool(kw.get("needs_review")),
            "qa_reasons": kw.get("qa_reasons") or "",
            "bbox": str(kw.get("bbox") or ""),
            "reason": kw.get("reason") or "",
        })

    for it in audit["translatable"]:
        src = it["text"]
        u = tmap.get(_norm(src)) or {}
        translation = u.get("translated_text") or ""
        q = unit_quality(src, translation, {}, profile) if translation else {}
        add(category="texte", granularity=it["granularity"], role=it.get("role"),
            object_type=it.get("object_type"), translatable=True, source_text=src,
            translation=translation, status=u.get("status"), needs_review=u.get("needs_review"),
            qa_reasons=", ".join(q.get("qa_reasons") or []), reason=it.get("reason"), bbox=it.get("bbox"))

    for it in audit["non_translatable"]:
        add(category="texte_exclu", granularity=it["granularity"], role=it.get("role"),
            object_type=it.get("object_type"), translatable=False, source_text=it["text"],
            reason=it.get("reason"), bbox=it.get("bbox"))

    for f in st["formulas"]:
        add(category="formule", role="formula", object_type="formula_expression",
            translatable=False, source_text=f.get("text"), reason=f.get("preservation_mode") or "formula_zone",
            bbox=f.get("bbox"))
    for c in st["code_blocks"]:
        add(category="code", role="code_block", object_type="code", translatable=False,
            source_text=c.get("text"), reason="code_zone", bbox=c.get("bbox"))
    for t in st["tables"]:
        add(category="table", role="table", object_type=f"{t.get('cells')} cellules", translatable=False,
            source_text=f"table {t.get('table_id')} — colonnes: {t.get('columns')}", reason="table_zone", bbox=t.get("bbox"))
    for fig in st["figures"]:
        add(category="figure", role="figure", object_type=f"{fig.get('diagram_labels')} labels", translatable=False,
            source_text="(zone figure / diagramme)", reason="figure_zone", bbox=fig.get("bbox"))
    for rtype, count in sorted(st["regions"].items(), key=lambda x: -x[1]):
        add(category="region", role=rtype, object_type=f"x{count}", translatable=False,
            source_text=f"(zone détectée : {rtype})", reason="special_zone")

    return rows


def page_meta(input_data: dict, pt_result: dict) -> dict:
    audit = audit_page(input_data)
    st = _structural(input_data)
    s = audit["summary"]
    statuses = (pt_result or {}).get("statuses") or {}
    return {
        "page_role": audit["page_role"],
        "selection_health": s["selection_health"],
        "translatable_count": s["translatable_count"],
        "background": bool(st["background"]["has_background_layer"]),
        "background_path": st["background"].get("background_path"),
        "source_image": st["background"].get("source_image_path"),
        "runtime_status": statuses.get("translation_runtime_status"),
        "quality_status": statuses.get("linguistic_quality_status"),
        "publication_status": statuses.get("publication_readiness_status"),
    }
