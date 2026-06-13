"""Functional checks for PAGETRANSLATE output."""

from __future__ import annotations


PRESERVE_ROLES = {"command_name", "path", "file_name", "url", "email", "toc_page_reference", "index_page_reference"}


def validate_functional_translation(result: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    debug = result.get("debug") or {}
    selection_mode = debug.get("selection_mode")
    metrics = {
        "translation_item_role_none": 0,
        "translation_without_render_target": 0,
        "protected_token_missing_after_restore": 0,
        "preserved_role_translated": 0,
        "reconstruction_unit_role_missing": 0,
        "reconstruction_unit_render_target_missing": 0,
        "fallback_selector_usage": 1 if selection_mode == "fallback_selector" else 0,
        "generic_coalescer_used": 1 if debug.get("generic_coalescer_used") else 0,
        "translation_plan_input_count": int(debug.get("translation_plan_input_count") or 0),
    }
    if selection_mode == "translation_plan_empty":
        errors.append("translation_plan_empty_no_fallback")
    if selection_mode == "fallback_disabled":
        errors.append("translation_plan_missing_fallback_disabled")
    for item in result.get("translation_units") or []:
        if not item.get("role"):
            metrics["translation_item_role_none"] += 1
            errors.append(f"translation_item_role_missing:{item.get('translation_unit_id')}")
        if not item.get("render_target"):
            metrics["translation_without_render_target"] += 1
            errors.append(f"translation_render_target_missing:{item.get('translation_unit_id')}")
        target = item.get("translated_text") or ""
        for token in item.get("protected") or []:
            if token and token not in target:
                metrics["protected_token_missing_after_restore"] += 1
                errors.append(f"protected_token_missing:{item.get('translation_unit_id')}:{token}")
        if item.get("role") in PRESERVE_ROLES and item.get("status") == "translated":
            metrics["preserved_role_translated"] += 1
            errors.append(f"preserved_role_translated:{item.get('translation_unit_id')}")
    translated_input = result.get("translated_input_data") or {}
    for item in ((translated_input.get("views") or {}).get("reconstruction_units") or []):
        if not item.get("role"):
            metrics["reconstruction_unit_role_missing"] += 1
            errors.append(f"reconstruction_unit_role_missing:{item.get('unit_id')}")
        if not item.get("render_target"):
            metrics["reconstruction_unit_render_target_missing"] += 1
            errors.append(f"reconstruction_unit_render_target_missing:{item.get('unit_id')}")
    return {
        "functional_valid": not errors,
        "functional_status": "ok" if not errors else "ko",
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }
