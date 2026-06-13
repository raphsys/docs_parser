from __future__ import annotations

from .common import bbox_of, eligible_text_units, role_of, text_of


def build_formula_units(units: list[dict], *, page_intelligence: dict | None = None) -> list[dict]:
    formula_units = [u for u in eligible_text_units(units) if role_of(u) == "formula_expression"]
    return [
        {
            "logical_unit_id": f"formula_{idx:04d}",
            "type": "formula_expression",
            "text": text_of(unit),
            "source_unit_ids": [unit["unit_id"]],
            "preservation_mode": (unit.get("policy") or {}).get("preservation_mode"),
            "bbox": bbox_of(unit),
        }
        for idx, unit in enumerate(formula_units, start=1)
    ]
