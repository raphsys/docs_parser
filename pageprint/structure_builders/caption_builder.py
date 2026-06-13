from __future__ import annotations

import re

from .common import bbox_of, eligible_text_units, reading_order, role_of, text_of

CAPTION_RE = re.compile(r"^\s*(?P<label>Figure|Fig\.|Table|Tab\.)\s+(?P<number>\d+(?:[.-]\d+)?)\s*[:.-]?\s*(?P<text>.*)$", re.IGNORECASE)


def build_captions(units: list[dict], *, page_intelligence: dict | None = None) -> list[dict]:
    output = []
    for idx, unit in enumerate(_caption_rows(units), start=1):
        text = text_of(unit)
        role = role_of(unit)
        match = CAPTION_RE.match(text)
        if "caption" not in role and not match:
            continue
        label = None
        number = None
        caption_text = text
        if match:
            label = match.group("label")
            number = match.group("number")
            caption_text = match.group("text").strip()
        output.append({
            "logical_unit_id": f"caption_{idx:04d}",
            "caption_id": f"caption_{idx:04d}",
            "type": "caption",
            "label": label,
            "number": number,
            "caption_text": caption_text,
            "preserve": [number] if number else [],
            "translatable_text": caption_text,
            "source_unit_ids": [unit["unit_id"]],
            "bbox": bbox_of(unit),
            "parse_strategy": "caption_split" if match else "role_caption",
        })
    return output


def _caption_rows(units: list[dict]) -> list[dict]:
    text_units = eligible_text_units(units)
    matches = [
        unit for unit in text_units
        if "caption" in role_of(unit) or CAPTION_RE.match(text_of(unit))
    ]
    lines = [unit for unit in matches if unit.get("level") == "line"]
    if lines:
        return sorted(lines, key=reading_order)
    phrases = [unit for unit in matches if unit.get("level") == "phrase"]
    if phrases:
        return sorted(phrases, key=reading_order)
    return sorted([unit for unit in matches if unit.get("level") not in {"span"}], key=reading_order)
