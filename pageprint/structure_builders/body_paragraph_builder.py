from __future__ import annotations

import re

from .common import bbox_of, bbox_union, eligible_text_units, reading_order, role_of, text_of

_NATURAL_TEXT_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’\-]*")
_STRONG_MATH_RE = re.compile(r"^[=×÷±∑∫√≈≠≤≥]|[=≈≠≤≥±×÷∑∫√].*[=≈≠≤≥±×÷∑∫√]")
_CODE_SIGNAL_RE = re.compile(r"\b(?:SELECT|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|INSERT|UPDATE|DELETE|JOIN|WITH|VALUES|CREATE|ALTER|DROP)\b|[;{}]", re.IGNORECASE)


def _looks_like_rescuable_prose(unit: dict) -> bool:
    text = text_of(unit)
    if not text:
        return False
    words = _NATURAL_TEXT_RE.findall(text)
    if len(words) < 5:
        return False
    if _STRONG_MATH_RE.search(text) and len(words) <= 2:
        return False
    if _CODE_SIGNAL_RE.search(text):
        return False
    role = role_of(unit)
    # A long sentence can be mislabeled as path/code/formula by upstream.  It is
    # still a translation candidate; inline URLs/formulas are protected later.
    return role in {"unknown", "path", "code_line", "code_block", "formula_expression", "section_heading", "table_body_cell", "table_header_cell"}



def build_body_paragraphs(units: list[dict], *, page_intelligence: dict | None = None) -> list[dict]:
    page_role = str((page_intelligence or {}).get("page_role") or "").lower()
    if page_role in {"toc", "index", "table", "cover"}:
        return []
    text_units = eligible_text_units(units)
    if any(unit.get("level") == "line" for unit in text_units):
        text_units = [unit for unit in text_units if unit.get("level") == "line"]
    by_id = {u.get("unit_id"): u for u in units if isinstance(u, dict) and u.get("unit_id")}
    by_parent: dict[str, list[dict]] = {}
    for unit in text_units:
        if unit.get("level") not in {"line", "phrase"}:
            continue
        if _has_heading_parent(unit, by_id) or _has_table_parent(unit, by_id):
            # A section title often has child lines/phrases whose local style is
            # too weak to be recognised as headings.  Do not also turn those
            # children into a body paragraph.
            continue
        role = role_of(unit)
        # Headings/titles must not be absorbed into body paragraphs, except when
        # upstream clearly mislabeled a normal prose sentence (common near code,
        # formulas and inline URLs).
        if role not in {"body_paragraph", "paragraph", "body"} and not _looks_like_rescuable_prose(unit):
            continue
        parent_id = _block_parent(unit)
        by_parent.setdefault(parent_id or "__page__", []).append(unit)

    paragraphs = []
    counter = 1
    for _parent, group in by_parent.items():
        rows = _prefer_line_rows(group)
        for para_rows in _split_paragraph_rows(rows):
            text = " ".join(text_of(row) for row in para_rows if text_of(row)).strip()
            if not text:
                continue
            paragraphs.append({
                "logical_unit_id": f"body_para_{counter:04d}",
                "type": "body_paragraph",
                "text": text,
                "source_unit_ids": [row["unit_id"] for row in para_rows],
                "line_unit_ids": [row["unit_id"] for row in para_rows if row.get("level") == "line"],
                "bbox": bbox_union([bbox_of(row) for row in para_rows]),
                "role": "body_paragraph",
                "parse_strategy": "line_continuity",
            })
            counter += 1
    return paragraphs


def _has_heading_parent(unit: dict, by_id: dict[str, dict]) -> bool:
    heading_roles = {"title", "subtitle", "section_heading", "subsection_heading", "chapter_heading"}
    parent_id = unit.get("parent_id")
    while parent_id:
        parent = by_id.get(parent_id)
        if parent is None:
            return False
        if role_of(parent) in heading_roles:
            return True
        parent_id = parent.get("parent_id")
    return False


def _has_table_parent(unit: dict, by_id: dict[str, dict]) -> bool:
    parent_id = unit.get("parent_id")
    while parent_id:
        parent = by_id.get(parent_id)
        if parent is None:
            return False
        if role_of(parent) in {"table", "table_body_cell", "table_header_cell"}:
            return True
        if parent.get("level") in {"table", "cell"}:
            return True
        parent_id = parent.get("parent_id")
    return False


def _block_parent(unit: dict) -> str | None:
    parent_id = unit.get("parent_id")
    if unit.get("level") == "line":
        return parent_id
    if unit.get("level") == "phrase":
        # Phrase parent is normally a line; grouping by that line's parent is
        # not available here without a full unit map, so the line id is still a
        # safe local paragraph bucket.
        return parent_id
    return parent_id


def _prefer_line_rows(group: list[dict]) -> list[dict]:
    lines = sorted([unit for unit in group if unit.get("level") == "line"], key=reading_order)
    if lines:
        return lines
    return sorted([unit for unit in group if unit.get("level") == "phrase"], key=reading_order)


def _split_paragraph_rows(rows: list[dict]) -> list[list[dict]]:
    if not rows:
        return []
    output: list[list[dict]] = []
    current: list[dict] = []
    previous_bottom = None
    previous_height = None
    for row in rows:
        bbox = bbox_of(row) or [0, 0, 0, 0]
        height = max(1.0, float(bbox[3]) - float(bbox[1])) if len(bbox) == 4 else 1.0
        gap = float(bbox[1]) - float(previous_bottom) if previous_bottom is not None and len(bbox) == 4 else 0.0
        starts_new = bool(current and previous_height and gap > previous_height * 1.4)
        if starts_new:
            output.append(current)
            current = []
        current.append(row)
        previous_bottom = bbox[3] if len(bbox) == 4 else previous_bottom
        previous_height = height
    if current:
        output.append(current)
    return output
