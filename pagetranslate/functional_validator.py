"""Functional checks for PAGETRANSLATE output."""

from __future__ import annotations


PRESERVE_ROLES = {"command_name", "path", "file_name", "url", "email", "toc_page_reference", "index_page_reference"}
TEXT_LEVELS = {"block", "line", "phrase", "span", "word"}
VALID_EXCLUSION_REASONS = {
    "artifact", "publisher_mark", "watermark", "page_number", "formula", "code",
    "protected_visual_region", "background_only", "exclude_as_artifact",
}


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
        "original_text_unit_count": 0,
        "original_text_missing_disposition": 0,
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
    coverage = audit_original_text_coverage(translated_input)
    metrics["original_text_unit_count"] = coverage["original_text_unit_count"]
    metrics["original_text_missing_disposition"] = coverage["missing_count"]
    for item in coverage["missing"]:
        errors.append(f"original_text_missing_disposition:{item['unit_id']}")
    return {
        "functional_valid": not errors,
        "functional_status": "ok" if not errors else "ko",
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
        "original_text_coverage": coverage,
    }


def audit_original_text_coverage(translated_input: dict) -> dict:
    """Every original PagePrint text unit must survive with a disposition.

    A text unit is covered when it is rendered translated, preserved as original,
    covered by a rendered parent, covered by rendered children, or explicitly
    excluded with a valid reason.
    """
    units = [
        u for u in translated_input.get("units") or []
        if isinstance(u, dict)
        and u.get("unit_id")
        and u.get("level") in TEXT_LEVELS
        and _text(u)
    ]
    by_id = {u["unit_id"]: u for u in units}
    children: dict[str, list[str]] = {}
    for u in translated_input.get("units") or []:
        pid = u.get("parent_id")
        if pid:
            children.setdefault(pid, []).append(u.get("unit_id"))

    views = translated_input.get("views") or {}
    rendered = {
        sid
        for ru in views.get("reconstruction_units") or []
        for sid in ru.get("source_unit_ids") or []
    }
    excluded = {}
    for ex in views.get("exclusion_plan") or []:
        reason = str(ex.get("reason") or "")
        for sid in ex.get("source_unit_ids") or []:
            excluded[sid] = reason

    def ancestors(uid: str):
        parent = (translated_input.get("_unit_parent_index") or {}).get(uid)
        if parent is None:
            parent = next((u.get("parent_id") for u in translated_input.get("units") or [] if u.get("unit_id") == uid), None)
        while parent:
            yield parent
            parent = next((u.get("parent_id") for u in translated_input.get("units") or [] if u.get("unit_id") == parent), None)

    def covered_by_children(uid: str) -> bool:
        kids = [k for k in children.get(uid, []) if k in by_id]
        return bool(kids) and all(is_covered(k) for k in kids)

    def is_covered(uid: str) -> bool:
        if uid in rendered:
            return True
        if any(a in rendered for a in ancestors(uid)):
            return True
        if uid in excluded and excluded[uid] in VALID_EXCLUSION_REASONS:
            return True
        return covered_by_children(uid)

    missing = []
    for u in units:
        if not is_covered(u["unit_id"]):
            missing.append({
                "unit_id": u["unit_id"],
                "level": u.get("level"),
                "text": _text(u)[:160],
                "bbox": (u.get("geometry") or {}).get("bbox"),
            })

    return {
        "status": "ok" if not missing else "ko",
        "original_text_unit_count": len(units),
        "missing_count": len(missing),
        "missing": missing,
    }


def _text(unit: dict) -> str:
    return str((unit.get("content") or {}).get("text") or "").strip()
