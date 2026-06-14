"""Build heading logical units, separate from body paragraphs (directive 18.3)."""

from __future__ import annotations

from .common import bbox_of, eligible_text_units, reading_order, role_of, text_of

_HEADING_ROLES = {"title", "subtitle", "section_heading", "subsection_heading", "chapter_heading"}


def build_headings(units: list[dict], *, page_intelligence: dict | None = None) -> list[dict]:
    out = []
    idx = 0
    for unit in eligible_text_units(units):
        if unit.get("level") not in {"line", "phrase", "block"}:
            continue
        role = role_of(unit)
        if role not in _HEADING_ROLES:
            continue
        text = text_of(unit)
        if not text:
            continue
        idx += 1
        out.append({
            "logical_unit_id": f"heading_{idx:04d}",
            "type": "heading",
            "role": role,
            "text": text,
            "source_unit_ids": [unit["unit_id"]],
            "bbox": bbox_of(unit),
            "reading_order_index": reading_order(unit),
        })
    return out
