from __future__ import annotations

import re

from .common import bbox_of, eligible_text_units, role_of, text_of

ROW_LEVEL_ORDER = ("line", "phrase", "block")
BULLET_RE = re.compile(r"^\s*(?P<marker>[•▪■●*-])\s*(?P<rest>.+?)\s*$")
SECTION_RE = re.compile(r"^\s*(?P<section>\d+(?:\.\d+)*\.?)\s+(?P<rest>.+?)\s*$")
PAGE_REF_RE = re.compile(r"^(?:\d{1,4}|[ivxlcdm]{1,12})$", re.IGNORECASE)
LEADERS_RE = re.compile(r"\.{2,}|\s{2,}")


def build_toc_entries(units: list[dict], *, page_intelligence: dict | None = None) -> list[dict]:
    """Build one logical TOC unit per visual row.

    The builder deliberately avoids block+line+phrase+span duplication. It
    prefers line rows when available, then phrase rows, then block rows.
    """
    page_role = str((page_intelligence or {}).get("page_role") or "").lower()
    candidates = _toc_row_candidates(units, page_role=page_role)
    output = []
    for idx, unit in enumerate(candidates, start=1):
        parsed = parse_toc_row(text_of(unit))
        if not parsed["title_text"]:
            continue
        output.append({
            "logical_unit_id": f"toc_entry_{idx:04d}",
            "type": "toc_entry",
            "marker": parsed.get("marker"),
            "section_number": parsed.get("section_number"),
            "title_text": parsed["title_text"],
            "page_reference": parsed.get("page_reference"),
            "source_unit_ids": [unit["unit_id"]],
            "title_unit_ids": [unit["unit_id"]],
            "preserve_unit_ids": [],
            "protected_values": [
                value for value in (
                    parsed.get("marker"),
                    parsed.get("section_number"),
                    parsed.get("page_reference"),
                )
                if value
            ],
            "bbox": bbox_of(unit),
            "parse_strategy": parsed.get("parse_strategy"),
        })
    return output


def parse_toc_row(text: str) -> dict:
    raw = str(text or "").strip()
    marker = None
    section = None
    page = None
    work = raw

    bullet = BULLET_RE.match(work)
    if bullet:
        marker = bullet.group("marker")
        work = bullet.group("rest").strip()

    section_match = SECTION_RE.match(work)
    if section_match:
        section = section_match.group("section").rstrip(".")
        work = section_match.group("rest").strip()

    strategy = "title_only"
    if LEADERS_RE.search(work):
        parts = [part.strip(" .") for part in LEADERS_RE.split(work) if part.strip(" .")]
        if len(parts) >= 2 and PAGE_REF_RE.fullmatch(parts[-1]):
            page = parts[-1]
            work = " ".join(parts[:-1]).strip()
            strategy = "leaders_or_column_gap"

    if page is None:
        tokens = work.rsplit(None, 1)
        if len(tokens) == 2 and PAGE_REF_RE.fullmatch(tokens[1]):
            work, page = tokens[0].strip(), tokens[1]
            strategy = "trailing_page_reference"

    return {
        "marker": marker,
        "section_number": section,
        "title_text": work.strip(" ."),
        "page_reference": page,
        "parse_strategy": strategy,
    }


def _toc_row_candidates(units: list[dict], *, page_role: str) -> list[dict]:
    text_units = eligible_text_units(units)
    toc_like = [
        unit for unit in text_units
        if page_role == "toc" or role_of(unit).startswith("toc") or _looks_like_toc_row(text_of(unit))
    ]
    if not toc_like:
        return []

    by_id = {unit.get("unit_id"): unit for unit in text_units}
    has_textual_child = _has_textual_child(text_units)
    for level in ROW_LEVEL_ORDER:
        if level == "line":
            rows = [unit for unit in toc_like if unit.get("level") == "line"]
            if rows:
                return rows
        rows = [
            unit for unit in toc_like
            if unit.get("level") == level and not has_textual_child.get(unit.get("unit_id"), False)
        ]
        if rows:
            return rows
    return [
        unit for unit in toc_like
        if unit.get("level") != "span" and unit.get("parent_id") not in by_id
    ]


def _has_textual_child(text_units: list[dict]) -> dict[str, bool]:
    text_ids = {unit.get("unit_id") for unit in text_units}
    output = {unit_id: False for unit_id in text_ids}
    for unit in text_units:
        parent_id = unit.get("parent_id")
        while parent_id:
            if parent_id in output:
                output[parent_id] = True
            parent = next((candidate for candidate in text_units if candidate.get("unit_id") == parent_id), None)
            parent_id = parent.get("parent_id") if parent else None
    return output


def _looks_like_toc_row(text: str) -> bool:
    parsed = parse_toc_row(text)
    if not (parsed.get("page_reference") and parsed.get("title_text")):
        return False
    # Require a TOC-specific signal so index entries ("term, 27,") are not
    # mistaken for TOC rows (directive PR-Lot 2).
    if re.search(r"\.{2,}", text) or re.search(r"\s{2,}\S*\d", text):
        return True
    if re.match(r"^\s*\d+(?:\.\d+)*\s+\S", text):  # leading section number
        return True
    return False
