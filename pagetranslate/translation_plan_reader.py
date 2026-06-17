"""Read PAGEPRINT views.translation_plan for PAGETRANSLATE."""

from __future__ import annotations

from .text_utils import normalize_spaces
from .selector import strip_running_header_page_number


def read_translation_plan(input_data: dict) -> list[dict]:
    plan = ((input_data.get("views") or {}).get("translation_plan") or [])
    output = []
    for idx, item in enumerate(plan, start=1):
        if not isinstance(item, dict):
            continue
        raw_source_text = normalize_spaces(item.get("source_text"))
        if not raw_source_text:
            continue
        role = item.get("role")
        bbox = item.get("bbox") or (item.get("render_target") or {}).get("bbox")
        source_text, preprocess = strip_running_header_page_number(raw_source_text, role=role, bbox=bbox)
        if not source_text:
            continue
        translation_unit_id = item.get("translation_unit_id") or f"tp_{idx:04d}"
        output.append({
            "translation_unit_id": translation_unit_id,
            "unit_id": item.get("unit_id") or translation_unit_id,
            "level": item.get("level") or "semantic_phrase",
            "parent_id": item.get("parent_id"),
            "source_unit_ids": list(item.get("source_unit_ids") or []),
            "logical_unit_id": item.get("logical_unit_id"),
            "source_text": source_text,
            "bbox": item.get("bbox") or (item.get("render_target") or {}).get("bbox"),
            "reading_order_index": item.get("reading_order_index") or idx,
            "role": role,
            "preprocess": preprocess,
            "object_type": item.get("object_type"),
            "object_class": item.get("object_class"),
            "semantic_kind": item.get("semantic_kind"),
            "strategy": item.get("translation_strategy") or "layout_constrained",
            "translation_mode": item.get("translation_mode") or "translate",
            "render_policy": (item.get("render_target") or {}).get("render_policy"),
            "coverage_required": item.get("coverage_required") or "strict",
            "protected": list(item.get("protected_tokens") or []),
            "translatable": item.get("translation_mode", "translate") == "translate",
            "context": dict(item.get("context") or {}),
            "render_target": dict(item.get("render_target") or {}),
            "qa_requirements": dict(item.get("qa_requirements") or {}),
            "original_translation_unit_id": translation_unit_id,
            "original_unit_id": item.get("unit_id") or translation_unit_id,
            "plan_item": item,
        })
    return output
