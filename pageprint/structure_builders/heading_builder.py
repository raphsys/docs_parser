"""Build heading logical units without block/line/phrase duplication."""

from __future__ import annotations

from .common import bbox_of, bbox_union, eligible_text_units, reading_order, role_of, text_of

_HEADING_ROLES = {"title", "subtitle", "section_heading", "subsection_heading", "chapter_heading"}


def build_headings(units: list[dict], *, page_intelligence: dict | None = None) -> list[dict]:
    """Return one logical heading per visual heading row/block.

    Older code emitted the block, its line and its phrase children as separate
    headings.  That produced duplicate translation units such as:
    ``CHAPTER 2 ...`` + ``CHAPTER 2`` + ``Deep learning ...``.  This builder
    groups heading children by their source block and merges them, while dropping
    page-number children.
    """
    text_units = eligible_text_units(units)
    by_id = {u.get("unit_id"): u for u in text_units if u.get("unit_id")}
    groups: dict[str, list[dict]] = {}

    for unit in text_units:
        if unit.get("level") not in {"block", "line", "phrase"}:
            continue
        if role_of(unit) not in _HEADING_ROLES:
            continue
        text = text_of(unit)
        if not text:
            continue
        block_id = _ancestor_block_id(unit, by_id) or unit.get("unit_id")
        groups.setdefault(block_id, []).append(unit)

    output = []
    for idx, (_block_id, group) in enumerate(sorted(groups.items(), key=lambda item: min(reading_order(u) for u in item[1])), start=1):
        rows = _preferred_heading_rows(group, by_id)
        if not rows:
            continue
        text = " ".join(text_of(row) for row in rows if text_of(row)).strip()
        if not text:
            continue
        role = _dominant_role(rows)
        output.append({
            "logical_unit_id": f"heading_{idx:04d}",
            "type": "heading",
            "role": role,
            "text": text,
            "source_unit_ids": [row["unit_id"] for row in rows],
            "bbox": bbox_union([bbox_of(row) for row in rows]),
            "reading_order_index": min(reading_order(row) for row in rows),
            "parse_strategy": "merged_heading_rows",
        })
    return output


def _ancestor_block_id(unit: dict, by_id: dict[str, dict]) -> str | None:
    if unit.get("level") == "block":
        return unit.get("unit_id")
    parent_id = unit.get("parent_id")
    while parent_id:
        parent = by_id.get(parent_id)
        if parent is None:
            return parent_id if "_block_" in str(parent_id) else None
        if parent.get("level") == "block":
            return parent.get("unit_id")
        parent_id = parent.get("parent_id")
    return None


def _preferred_heading_rows(group: list[dict], by_id: dict[str, dict]) -> list[dict]:
    # Use child rows when present; skip the parent block to avoid page-number
    # contamination and duplicate translation. Prefer lines over phrases because
    # adjacent phrases in the same running header are separate glyph runs.
    lines = [u for u in group if u.get("level") == "line" and role_of(u) in _HEADING_ROLES]
    if lines:
        return sorted(lines, key=reading_order)
    phrases = [u for u in group if u.get("level") == "phrase" and role_of(u) in _HEADING_ROLES]
    if phrases:
        return sorted(phrases, key=reading_order)
    blocks = [u for u in group if u.get("level") == "block"]
    return sorted(blocks, key=reading_order)


def _dominant_role(rows: list[dict]) -> str:
    for role in ("title", "chapter_heading", "section_heading", "subsection_heading", "subtitle"):
        if any(role_of(row) == role for row in rows):
            return role
    return role_of(rows[0]) if rows else "section_heading"
