"""Small text helpers shared by PAGETRANSLATE modules."""

from __future__ import annotations

import re
from typing import Any


def normalize_spaces(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def unit_text(unit: dict) -> str:
    content = unit.get("content") or {}
    return normalize_spaces(
        content.get("text")
        or content.get("normalized_text")
        or content.get("raw_text")
        or unit.get("text")
    )


def bbox_union(boxes: list[list[float] | tuple[float, ...] | None]) -> list[float] | None:
    valid = [box for box in boxes if isinstance(box, (list, tuple)) and len(box) == 4]
    if not valid:
        return None
    return [
        min(float(box[0]) for box in valid),
        min(float(box[1]) for box in valid),
        max(float(box[2]) for box in valid),
        max(float(box[3]) for box in valid),
    ]


def ancestor_id(unit_id: str, unit_map: dict[str, dict], *, level: str) -> str | None:
    current = unit_map.get(unit_id)
    seen = set()
    while current and current.get("unit_id") not in seen:
        seen.add(current.get("unit_id"))
        if current.get("level") == level:
            return current.get("unit_id")
        current = unit_map.get(current.get("parent_id"))
    return None


def reading_order(unit: dict) -> int:
    value = (unit.get("geometry") or {}).get("reading_order_index")
    return int(value) if isinstance(value, int) else 0
