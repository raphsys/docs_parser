from __future__ import annotations

from .common import bbox_of, bbox_union, eligible_text_units, reading_order, role_of, text_of


def build_body_paragraphs(units: list[dict], *, page_intelligence: dict | None = None) -> list[dict]:
    page_role = str((page_intelligence or {}).get("page_role") or "").lower()
    if page_role in {"toc", "index", "table", "cover"}:
        return []
    text_units = eligible_text_units(units)
    if any(unit.get("level") == "line" for unit in text_units):
        text_units = [unit for unit in text_units if unit.get("level") == "line"]
    by_parent: dict[str, list[dict]] = {}
    for unit in text_units:
        if unit.get("level") not in {"line", "phrase"}:
            continue
        role = role_of(unit)
        if role not in {"body_paragraph", "paragraph", "body", "title", "section_heading"}:
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
