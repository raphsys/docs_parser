from __future__ import annotations

from .common import bbox_of, eligible_text_units, role_of, text_of


def build_author_bios(units: list[dict], *, page_intelligence: dict | None = None) -> list[dict]:
    output = []
    for idx, unit in enumerate(eligible_text_units(units), start=1):
        if role_of(unit) not in {"author_name", "author_bio"}:
            continue
        output.append({
            "logical_unit_id": f"author_{idx:04d}",
            "type": role_of(unit),
            "text": text_of(unit),
            "source_unit_ids": [unit["unit_id"]],
            "bbox": bbox_of(unit),
        })
    return output
