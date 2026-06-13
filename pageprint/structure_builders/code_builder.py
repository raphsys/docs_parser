from __future__ import annotations

from .common import bbox_of, eligible_text_units, role_of, text_of


def build_code_blocks(units: list[dict], *, page_intelligence: dict | None = None) -> list[dict]:
    code_units = [u for u in eligible_text_units(units) if role_of(u) in {"code_block", "code_line", "command_name", "path"}]
    return [
        {
            "logical_unit_id": f"code_{idx:04d}",
            "type": "code_line" if role_of(unit) != "code_block" else "code_block",
            "text": text_of(unit),
            "source_unit_ids": [unit["unit_id"]],
            "translation_mode": "preserve_text_exactly",
            "bbox": bbox_of(unit),
        }
        for idx, unit in enumerate(code_units, start=1)
    ]
