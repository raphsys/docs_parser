from __future__ import annotations

import re

from .common import bbox_of, eligible_text_units, role_of, text_of

LIST_RE = re.compile(r"^\s*(?P<marker>(?:[-*•▪]|\d+[.)]|[a-z][.)]))\s+(?P<text>.+)$", re.IGNORECASE)


def build_list_items(units: list[dict], *, page_intelligence: dict | None = None) -> list[dict]:
    output = []
    for idx, unit in enumerate(eligible_text_units(units), start=1):
        text = text_of(unit)
        role = role_of(unit)
        match = LIST_RE.match(text)
        if role != "list_item" and not match:
            continue
        marker = match.group("marker") if match else None
        item_text = match.group("text").strip() if match else text
        output.append({
            "logical_unit_id": f"list_item_{idx:04d}",
            "type": "list_item",
            "marker": marker,
            "text": item_text,
            "continuation_unit_ids": [],
            "marker_policy": "preserve_text_exactly",
            "text_policy": "translate",
            "source_unit_ids": [unit["unit_id"]],
            "bbox": bbox_of(unit),
        })
    return output
