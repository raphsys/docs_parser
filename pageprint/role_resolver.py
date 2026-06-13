"""Resolve documentary roles for PAGEPRINT units."""

from __future__ import annotations

import re


CAPTION_RE = re.compile(r"^\s*(figure|fig\.|table|tab\.)\s+\d+(?:[.-]\d+)?", re.IGNORECASE)
TOC_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*\s+)?\S.{3,}?(?:\.{2,}|\s{2,})\s*[ivxlcdm\d]+\s*$", re.IGNORECASE)
INDEX_RE = re.compile(r"^\s*[A-Za-z][^,]{1,80},\s*(?:\d+[,-–\s]*)+\s*$")
PAGE_REF_RE = re.compile(r"^\s*(?:[ivxlcdm]+|\d+)(?:[-–]\d+)?\s*$", re.IGNORECASE)
SECTION_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s*$")
PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/[\w.-]+/|\.{1,2}/)[^\s]+")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
COMMAND_RE = re.compile(r"^(?:copy|dir|del|findstr|mkdir|rmdir|cd|ls|cat|grep|sudo|docker|npm|pip|python|git)\b", re.IGNORECASE)
# Shared publisher/watermark patterns (single source of truth).
from .structure_builders.publisher_mark_builder import PUBLISHER_RE, WATERMARK_RE


ROLE_TO_OBJECT = {
    "body_paragraph": ("natural_text", "prose"),
    "title": ("natural_text", "title"),
    "subtitle": ("natural_text", "subtitle"),
    "section_heading": ("natural_text", "heading"),
    "subsection_heading": ("natural_text", "heading"),
    "list_item": ("natural_text", "list_item_text"),
    "list_marker": ("marker", "list_marker"),
    "figure_caption": ("natural_text", "caption_text"),
    "figure_label": ("reference", "figure_label"),
    "table_caption": ("natural_text", "caption_text"),
    "table_header_cell": ("table_cell", "table_header"),
    "table_body_cell": ("table_cell", "table_body"),
    "table_numeric_cell": ("table_cell", "numeric"),
    "formula_expression": ("formula_expression", "formula"),
    "formula_explanation": ("natural_text", "formula_explanation"),
    "code_block": ("code", "code_block"),
    "code_line": ("code", "code_line"),
    "code_token": ("code_token", "code_token"),
    "command_name": ("command_name", "code_token"),
    "path": ("path", "protected_token"),
    "file_name": ("file_name", "protected_token"),
    "url": ("url", "protected_token"),
    "email": ("email", "protected_token"),
    "page_header": ("running_text", "header"),
    "page_footer": ("running_text", "footer"),
    "publisher_mark": ("publisher_mark", "artifact"),
    "watermark": ("watermark", "artifact"),
    "author_name": ("proper_name", "person_name"),
    "author_bio": ("natural_text", "biography"),
    "index_entry": ("index_entry", "index_entry"),
    "index_head_term": ("technical_term", "index_term"),
    "index_subentry": ("natural_text", "index_subentry"),
    "index_page_reference": ("page_reference", "reference"),
    "page_reference": ("page_reference", "reference"),
    "toc_title": ("natural_text", "toc_title"),
    "toc_entry": ("toc_entry", "toc_entry"),
    "toc_entry_title": ("natural_text", "toc_entry_title"),
    "toc_section_number": ("section_number", "reference"),
    "toc_page_reference": ("page_reference", "reference"),
    "toc_bullet_marker": ("marker", "toc_marker"),
    "diagram_label": ("diagram_label", "label"),
    "diagram_text_label": ("natural_text", "diagram_label_text"),
}


def resolve_roles(
    units: list[dict],
    *,
    page_intelligence: dict | None = None,
    document_context: dict | None = None,
) -> dict:
    """Assign roles in-place and return role metrics."""
    page_intelligence = page_intelligence or {}
    role_counts: dict[str, int] = {}
    unresolved = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        role, reason, confidence = resolve_unit_role(
            unit,
            page_intelligence=page_intelligence,
            document_context=document_context or {},
        )
        understanding = unit.setdefault("understanding", {})
        understanding["role"] = role
        object_type, semantic_kind = ROLE_TO_OBJECT.get(role, ("unknown", "unknown"))
        if object_type != "unknown":
            understanding["object_type"] = object_type
        elif not understanding.get("object_type"):
            understanding["object_type"] = "unknown"
        if semantic_kind != "unknown":
            understanding["semantic_kind"] = semantic_kind
        elif not understanding.get("semantic_kind"):
            understanding["semantic_kind"] = "unknown"
        unit.setdefault("confidence", {})["role"] = confidence
        unit.setdefault("provenance", {}).setdefault("decision_trace", []).append({
            "stage": "role_resolution",
            "target_id": unit.get("unit_id"),
            "decision": f"role={role}",
            "reason": reason,
            "confidence": confidence,
        })
        role_counts[role] = role_counts.get(role, 0) + 1
        if role == "unknown":
            unresolved.append(unit.get("unit_id"))
    return {
        "schema_version": "pageprint.role_resolution.v1",
        "role_counts": role_counts,
        "unresolved_unit_ids": unresolved,
    }


def infer_page_role(role_counts: dict | None, logical_structures: dict | None = None, *, current: str | None = None) -> str | None:
    """Promote a generic page_role to index/toc/table_page from dominant content.

    Uses logical structures first (reliable), then resolved role counts.
    Only overrides a generic page_role (None/unknown/body).
    """
    current_l = str(current or "").lower()
    if current_l not in {"", "none", "unknown", "body", "body_text", "body_text_two_column"}:
        return current
    rc = role_counts or {}
    ls = logical_structures or {}
    index_n = max(len(ls.get("index_entries") or []), rc.get("index_entry", 0) + rc.get("index_head_term", 0))
    toc_n = max(len(ls.get("toc_entries") or []), rc.get("toc_entry", 0) + rc.get("toc_entry_title", 0))
    table_cells = sum(len(t.get("cells") or []) for t in ls.get("tables") or [])
    table_cells = max(table_cells, rc.get("table_body_cell", 0) + rc.get("table_header_cell", 0))
    named = index_n + toc_n + table_cells + rc.get("body_paragraph", 0) + rc.get("list_item", 0)
    named = named or 1
    if index_n >= 10 and index_n / named > 0.4:
        return "index"
    if toc_n >= 10 and toc_n / named > 0.4:
        return "toc"
    if table_cells >= 10 and table_cells / named > 0.5:
        return "table_page"
    return current


def resolve_unit_role(unit: dict, *, page_intelligence: dict, document_context: dict) -> tuple[str, str, float]:
    understanding = unit.get("understanding") or {}
    current = str(understanding.get("role") or "").strip()
    text = str((unit.get("content") or {}).get("text") or "").strip()
    level = unit.get("level")
    page_role = str(page_intelligence.get("page_role") or "").lower()
    layout_type = str(page_intelligence.get("layout_type") or "").lower()
    resolved = (unit.get("evidence") or {}).get("resolved_understanding") or {}
    resolved_object = str(resolved.get("object_type") or "").lower()

    if text:
        if PUBLISHER_RE.search(text):
            return "publisher_mark", "publisher_mark_pattern", 0.88
        if WATERMARK_RE.search(text):
            return "watermark", "watermark_pattern", 0.88
        if PAGE_REF_RE.fullmatch(text) and _in_margin(unit, page_intelligence):
            return "page_reference", "page_number_margin", 0.82
    if current and current not in {"body", "paragraph", "text", "unknown", "None"}:
        return _normalize_legacy_role(current, text, page_role), "legacy_role_normalized", 0.82
    if level in {"page", "region", "image", "drawing", "overlay", "table"}:
        return _container_role(level, understanding), "structural_container", 0.85
    if not text:
        return "unknown", "empty_text", 0.2
    if page_role == "toc":
        if PAGE_REF_RE.fullmatch(text):
            return "toc_page_reference", "toc_page_reference_pattern", 0.86
        if SECTION_RE.fullmatch(text):
            return "toc_section_number", "toc_section_number_pattern", 0.86
        return "toc_entry_title" if level in {"phrase", "span"} else "toc_entry", "toc_page_context", 0.82
    if page_role == "index":
        if PAGE_REF_RE.fullmatch(text):
            return "index_page_reference", "index_page_reference_pattern", 0.86
        if INDEX_RE.fullmatch(text):
            return "index_entry", "index_pattern", 0.84
        return "index_head_term", "index_page_context", 0.76
    if CAPTION_RE.match(text):
        return ("table_caption" if text.lower().startswith(("table", "tab.")) else "figure_caption"), "caption_pattern", 0.88
    if TOC_RE.match(text):
        return "toc_entry", "toc_pattern", 0.78
    if INDEX_RE.match(text):
        return "index_entry", "index_pattern", 0.78
    if URL_RE.fullmatch(text):
        return "url", "url_pattern", 0.96
    if EMAIL_RE.fullmatch(text):
        return "email", "email_pattern", 0.96
    if PATH_RE.search(text):
        return "path", "path_pattern", 0.92
    if COMMAND_RE.match(text):
        return "command_name" if len(text.split()) == 1 else "code_line", "command_pattern", 0.86
    if resolved_object == "formula_expression":
        return "formula_expression", "resolved_formula_evidence", 0.78
    if resolved_object in {"code_line", "code"}:
        return "code_line", "resolved_code_evidence", 0.78
    if level == "cell" or layout_type == "table_dominant":
        return "table_body_cell", "table_context", 0.72
    if level in {"block", "line", "phrase", "span"}:
        return "body_paragraph" if len(text.split()) >= 4 else "title", "textual_default", 0.68
    return "unknown", "no_role_rule_matched", 0.35


def _normalize_legacy_role(role: str, text: str, page_role: str) -> str:
    role_l = role.lower()
    if role_l in {"body", "paragraph", "text"}:
        return "body_paragraph"
    if role_l in ROLE_TO_OBJECT:
        return role_l
    if "caption" in role_l:
        return "table_caption" if text.lower().startswith(("table", "tab.")) else "figure_caption"
    if "heading" in role_l or "title" in role_l:
        return "title" if "title" in role_l else "section_heading"
    if "footer" in role_l:
        return "page_footer"
    if "header" in role_l:
        return "page_header"
    if "toc" in role_l:
        return "toc_entry_title" if "title" in role_l else "toc_entry"
    if "index" in role_l:
        return "index_page_reference" if "reference" in role_l else "index_entry"
    if "formula" in role_l or "equation" in role_l:
        return "formula_expression"
    if "code" in role_l:
        return "code_line"
    if page_role == "toc":
        return "toc_entry"
    if page_role == "index":
        return "index_entry"
    return "unknown"


def _container_role(level: str, understanding: dict) -> str:
    object_type = str(understanding.get("object_type") or "").lower()
    if "table" in object_type:
        return "table_body_cell" if level == "cell" else "unknown"
    if "publisher" in object_type or "logo" in object_type:
        return "publisher_mark"
    return "unknown"


def _in_margin(unit: dict, page_intelligence: dict) -> bool:
    bbox = (unit.get("geometry") or {}).get("bbox")
    page_geometry = page_intelligence.get("page_geometry") or {}
    height = float(page_geometry.get("height") or 0.0)
    dpi = float(page_geometry.get("render_dpi") or 72.0)
    if height and dpi:
        height = height * 72.0 / dpi
    if not height or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    y0 = float(bbox[1])
    y1 = float(bbox[3])
    return y1 < height * 0.12 or y0 > height * 0.88
