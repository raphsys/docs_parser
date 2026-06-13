"""Functional PAGEPRINT validation beyond schema checks."""

from __future__ import annotations

import re


CAPTION_RAW_RE = re.compile(r"^\s*(?:figure|fig\.|table|tab\.)\s+\d", re.IGNORECASE)

def validate_functional(input_data: dict) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    views = input_data.get("views") or {}
    translation_plan = views.get("translation_plan") or []
    reconstruction_plan = views.get("reconstruction_plan") or []
    semantic_system = input_data.get("semantic_system") or {}
    page_role = str((input_data.get("page_intelligence") or {}).get("page_role") or (input_data.get("page") or {}).get("page_role") or "").lower()
    units_by_id = {
        unit.get("unit_id"): unit
        for unit in input_data.get("units") or []
        if isinstance(unit, dict) and unit.get("unit_id")
    }

    metrics = {
        "role_none_translation_units": 0,
        "object_type_none_translation_units": 0,
        "word_char_translation_units": 0,
        "mixed_block_translation_units": 0,
        "partial_protected_parent_background": 0,
        "natural_text_marked_preserve_visual": 0,
        "reconstruction_units_missing_roles": 0,
        "toc_without_entries": 0,
        "index_without_entries": 0,
        "semantic_system_empty_for_body": 0,
        "translation_plan_empty_but_translatable_text_exists": 0,
        "toc_entries_exist_but_no_translation_segments": 0,
        "toc_entries_exist_but_no_translation_plan": 0,
        "index_entries_exist_but_no_translation_plan": 0,
        "tables_exist_but_no_cell_translation_plan": 0,
        "table_pages_without_tables": 0,
        "index_pages_without_index_entries": 0,
        "logical_units_exist_but_no_translation_segments": 0,
        "fallback_required_after_pageprint": 0,
        "mixed_block_in_translation_plan": 0,
        "publisher_mark_sent_to_translation": 0,
        "watermark_sent_to_translation": 0,
        "caption_raw_block_translation": 0,
        "raw_table_row_translation": 0,
        "translation_plan_compile_errors": len(views.get("translation_plan_compile_errors") or []),
    }
    errors.extend(views.get("translation_plan_compile_errors") or [])

    for item in translation_plan:
        if not item.get("role") or item.get("role") == "unknown":
            metrics["role_none_translation_units"] += 1
            errors.append(f"translation_plan_role_missing:{item.get('translation_unit_id')}")
        if not item.get("object_type") or item.get("object_type") == "unknown":
            metrics["object_type_none_translation_units"] += 1
            errors.append(f"translation_plan_object_type_missing:{item.get('translation_unit_id')}")
        for source_id in item.get("source_unit_ids") or []:
            source = units_by_id.get(source_id)
            if source and source.get("level") in {"word", "char"}:
                metrics["word_char_translation_units"] += 1
                errors.append(f"translation_plan_word_char_source:{item.get('translation_unit_id')}:{source_id}")
        if not item.get("render_target"):
            errors.append(f"translation_plan_render_target_missing:{item.get('translation_unit_id')}")
        source_text = str(item.get("source_text") or "")
        if CAPTION_RAW_RE.match(source_text):
            metrics["caption_raw_block_translation"] += 1
            errors.append(f"caption_raw_block_translation:{item.get('translation_unit_id')}")
        if item.get("role") in {"publisher_mark", "watermark"}:
            key = "publisher_mark_sent_to_translation" if item.get("role") == "publisher_mark" else "watermark_sent_to_translation"
            metrics[key] += 1
            errors.append(f"{key}:{item.get('translation_unit_id')}")
        if item.get("role", "").startswith("table_") and ("  " in source_text or "\t" in source_text):
            metrics["raw_table_row_translation"] += 1
            errors.append(f"raw_table_row_translation:{item.get('translation_unit_id')}")
        for source_id in item.get("source_unit_ids") or []:
            source = units_by_id.get(source_id) or {}
            source_role = (source.get("understanding") or {}).get("role")
            if source_role in {"publisher_mark", "watermark"}:
                key = "publisher_mark_sent_to_translation" if source_role == "publisher_mark" else "watermark_sent_to_translation"
                metrics[key] += 1
                errors.append(f"{key}:{item.get('translation_unit_id')}:{source_id}")
            if source.get("level") == "block" and item.get("role") not in {"body_paragraph", "title", "section_heading"}:
                metrics["mixed_block_in_translation_plan"] += 1
                errors.append(f"mixed_block_in_translation_plan:{item.get('translation_unit_id')}:{source_id}")

    for unit in units_by_id.values():
        policy = unit.get("policy") or {}
        if policy.get("render_policy") == "background_only":
            for membership in (unit.get("understanding") or {}).get("region_memberships") or []:
                if membership.get("coverage_mode") == "partial_inline":
                    metrics["partial_protected_parent_background"] += 1
                    errors.append(f"partial_region_parent_background:{unit.get('unit_id')}")
        if policy.get("preservation_mode") == "preserve_as_visual_overlay":
            role = (unit.get("understanding") or {}).get("role")
            text = str((unit.get("content") or {}).get("text") or "")
            if role == "body_paragraph" and len(text.split()) >= 5:
                metrics["natural_text_marked_preserve_visual"] += 1
                errors.append(f"natural_text_preserved_as_visual:{unit.get('unit_id')}")

    for item in reconstruction_plan:
        if not item.get("role"):
            metrics["reconstruction_units_missing_roles"] += 1
            errors.append(f"reconstruction_unit_role_missing:{item.get('reconstruction_unit_id')}")

    logical = (views.get("logical_structures") or semantic_system.get("logical_structures") or {})
    translation_segments = semantic_system.get("translation_segments") or []
    logical_units = logical.get("logical_units") or []
    translatable_text_units = _translatable_text_units(units_by_id.values())
    if page_role == "toc" and not logical.get("toc_entries"):
        metrics["toc_without_entries"] = 1
        errors.append("toc_page_without_toc_entries")
    if logical.get("toc_entries") and not translation_segments:
        metrics["toc_entries_exist_but_no_translation_segments"] = 1
        errors.append("toc_entries_exist_but_no_translation_segments")
    if logical.get("toc_entries") and not translation_plan:
        metrics["toc_entries_exist_but_no_translation_plan"] = 1
        errors.append("toc_entries_exist_but_no_translation_plan")
    if page_role == "index" and not logical.get("index_entries"):
        metrics["index_without_entries"] = 1
        metrics["index_pages_without_index_entries"] = 1
        errors.append("index_page_without_index_entries")
    if page_role == "table" and not logical.get("tables"):
        metrics["table_pages_without_tables"] = 1
        errors.append("table_page_without_tables")
    if logical.get("index_entries") and not translation_plan:
        metrics["index_entries_exist_but_no_translation_plan"] = 1
        errors.append("index_entries_exist_but_no_translation_plan")
    if logical.get("tables") and not any(item.get("role", "").startswith("table_") for item in translation_plan):
        metrics["tables_exist_but_no_cell_translation_plan"] = 1
        errors.append("tables_exist_but_no_cell_translation_plan")
    if logical_units and not translation_segments:
        metrics["logical_units_exist_but_no_translation_segments"] = 1
        errors.append("logical_units_exist_but_no_translation_segments")
    if translatable_text_units and not translation_plan:
        metrics["translation_plan_empty_but_translatable_text_exists"] = 1
        metrics["fallback_required_after_pageprint"] = 1
        errors.append("translation_plan_empty_but_translatable_text_exists")
        errors.append("fallback_required_after_pageprint")
    if page_role in {"body", "body_text"} and not (semantic_system.get("semantic_phrases") or semantic_system.get("translation_segments")):
        metrics["semantic_system_empty_for_body"] = 1
        errors.append("semantic_system_empty_but_page_has_translatable_text")

    return {
        "functional_valid": not errors,
        "functional_status": "ok" if not errors else "ko",
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }


def _translatable_text_units(units) -> list[dict]:
    forbidden_roles = {
        "page_reference",
        "section_number",
        "toc_page_reference",
        "toc_section_number",
        "toc_bullet_marker",
        "index_page_reference",
        "command_name",
        "path",
        "file_name",
        "code",
        "watermark",
        "publisher_mark",
    }
    output = []
    for unit in units:
        if unit.get("level") not in {"block", "line", "phrase", "span", "cell"}:
            continue
        text = str((unit.get("content") or {}).get("text") or "").strip()
        if not text:
            continue
        policy = unit.get("policy") or {}
        role = (unit.get("understanding") or {}).get("role")
        if policy.get("translatable") is True and role not in forbidden_roles:
            output.append(unit)
    return output
