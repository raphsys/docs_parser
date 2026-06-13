"""Projection of translations back into PagePrint INPUT_DATA."""

from __future__ import annotations

from .text_utils import normalize_spaces


def project_translations(translated_input: dict, translated_units: list[dict]) -> list[dict]:
    unit_map = {
        unit.get("unit_id"): unit
        for unit in translated_input.get("units") or []
        if isinstance(unit, dict) and unit.get("unit_id")
    }
    projections = []
    for item in translated_units:
        target_ids = item.get("source_unit_ids") or [item.get("unit_id")]
        direct_unit = unit_map.get(item.get("unit_id"))
        if direct_unit is not None:
            _set_unit_translation(direct_unit, item)
            _backfill_single_span_child(direct_unit, unit_map, item["translated_text"])
            projected_ids = [direct_unit["unit_id"]]
            reconstruction_unit = None
        else:
            projected_ids = _project_to_source_units(unit_map, target_ids, item)
            _project_to_semantic_system(translated_input, item)
            reconstruction_unit = _semantic_reconstruction_unit(item, unit_map)

        projections.append({
            "unit_id": item.get("unit_id"),
            "level": item.get("level"),
            "translation_unit_id": item.get("translation_unit_id"),
            "status": item.get("status"),
            "target_text": item.get("translated_text"),
            "bbox": item.get("bbox"),
            "source_unit_ids": target_ids,
            "projected_unit_ids": projected_ids,
            "reconstruction_compatible": True,
            "projection_strategy": "direct_unit" if direct_unit else "semantic_source_units",
            "reconstruction_unit": reconstruction_unit,
        })

    _aggregate_parent_translations(unit_map)
    translated_input.setdefault("views", {})["reconstruction_units"] = _reconstruction_units(
        translated_input,
        translated_units,
    )
    return projections


def _set_unit_translation(unit: dict, item: dict) -> None:
    unit.setdefault("content", {})["translated_text"] = item["translated_text"]
    unit.setdefault("translation", {}).update({
        "translation_id": item["translation_id"],
        "translation_unit_id": item["translation_unit_id"],
        "source_text": item["source_text"],
        "target_text": item["translated_text"],
        "status": item["status"],
        "strategy": item["strategy"],
        "sentence": item.get("sentence") or {},
        "quality": item.get("quality") or {},
        "protections": item.get("protections") or [],
        "context": item.get("context") or {},
    })


def _project_to_source_units(unit_map: dict[str, dict], target_ids: list[str], item: dict) -> list[str]:
    projected = []
    for idx, unit_id in enumerate(target_ids):
        unit = unit_map.get(unit_id)
        if not unit:
            continue
        unit.setdefault("translation", {}).update({
            "inherited_from_translation_unit_id": item["translation_unit_id"],
            "semantic_source_unit_id": item["unit_id"],
            "semantic_projection_index": idx,
            "status": item["status"],
            "consumed_by_translation_unit_id": item["translation_unit_id"],
            "skip_individual_render": True,
            "source_units_consumed": True,
        })
        projected.append(unit_id)
    return projected


def _project_to_semantic_system(translated_input: dict, item: dict) -> None:
    semantic_system = translated_input.get("semantic_system") or {}
    for key in ("semantic_phrases", "semantic_groups"):
        for entry in semantic_system.get(key) or []:
            if isinstance(entry, dict) and entry.get("unit_id") == item.get("unit_id"):
                entry["translated_text"] = item["translated_text"]
                entry["translation"] = {
                    "translation_id": item["translation_id"],
                    "translation_unit_id": item["translation_unit_id"],
                    "status": item["status"],
                    "quality": item.get("quality") or {},
                    "protections": item.get("protections") or [],
                }


def _backfill_single_span_child(unit: dict, unit_map: dict[str, dict], translated_text: str) -> None:
    children = [
        unit_map[cid]
        for cid in unit.get("children_ids") or []
        if cid in unit_map
    ]
    span_children = [child for child in children if child.get("level") == "span"]
    if len(span_children) == 1:
        span_children[0].setdefault("content", {})["translated_text"] = translated_text
        span_children[0].setdefault("translation", {}).update({
            "inherited_from_unit_id": unit["unit_id"],
            "skip_individual_render": True,
        })


def _aggregate_parent_translations(unit_map: dict[str, dict]) -> None:
    ordered = sorted(
        unit_map.values(),
        key=lambda unit: (unit.get("geometry") or {}).get("reading_order_index") or 0,
        reverse=True,
    )
    for unit in ordered:
        children = [
            unit_map[cid]
            for cid in unit.get("children_ids") or []
            if cid in unit_map
        ]
        translated_children = [
            normalize_spaces((child.get("content") or {}).get("translated_text"))
            for child in children
            if child.get("level") in {"phrase", "line", "span", "block"}
        ]
        translated_children = [text for text in translated_children if text]
        if translated_children and not normalize_spaces((unit.get("content") or {}).get("translated_text")):
            unit.setdefault("content", {})["translated_text"] = normalize_spaces(" ".join(translated_children))


def _style_from_source_ids(source_unit_ids, unit_map: dict[str, dict]) -> tuple[dict, str | None]:
    """Dominant typographic style resolved from the consumed source units."""
    for sid in source_unit_ids or []:
        unit = (unit_map or {}).get(sid)
        if not unit:
            continue
        style = _dominant_style(unit, unit_map)
        if _style_has_real_values(style):
            return style, sid
    return {}, None


def _semantic_reconstruction_unit(item: dict, unit_map: dict[str, dict] | None = None) -> dict:
    render_target = item.get("render_target") or {}
    context = item.get("context") or {}
    # Separate bboxes: never let a (possibly first-line) render_target.bbox shrink the layout.
    logical_bbox = item.get("bbox")
    layout_bbox = render_target.get("layout_bbox") or logical_bbox or render_target.get("bbox")
    patch_bbox = render_target.get("patch_bbox") or render_target.get("coverage_bbox") or logical_bbox or render_target.get("bbox")
    coverage_bbox = render_target.get("coverage_bbox") or logical_bbox or render_target.get("bbox")
    anchor_bbox = render_target.get("anchor_bbox") or layout_bbox
    bbox = layout_bbox
    style, style_source_unit_id = _style_from_source_ids(item.get("source_unit_ids") or [], unit_map or {})
    return {
        "style": style,
        "style_source_unit_id": render_target.get("style_source_unit_id") or style_source_unit_id,
        "reconstruction_unit_id": render_target.get("reconstruction_unit_id"),
        "unit_id": item.get("unit_id"),
        "translation_unit_id": item.get("translation_unit_id"),
        "logical_unit_id": item.get("logical_unit_id"),
        "level": item.get("level"),
        "render_level": item.get("level"),
        "role": item.get("role"),
        "object_type": item.get("object_type"),
        "semantic_kind": item.get("semantic_kind"),
        "page_role": context.get("page_role"),
        "preservation_mode": item.get("preservation_mode"),
        "text": item.get("source_text"),
        "translated_text": item.get("translated_text"),
        "bbox": bbox,
        "layout_bbox": layout_bbox,
        "patch_bbox": patch_bbox,
        "coverage_bbox": coverage_bbox,
        "anchor_bbox": anchor_bbox,
        "source_unit_ids": item.get("source_unit_ids") or [],
        "consume_source_units": True,
        "source_units_consumed": True,
        "preferred_over_children": True,
        "skip_original_units": True,
        "render_as": item.get("level"),
        "overflow_policy": "shrink_or_reflow",
        "line_break_policy": item.get("strategy") or "semantic_reflow",
        "layout_budget": _layout_budget(bbox),
        "style_source": "dominant_source_span",
        "render_target": render_target,
        "render_contract": {
            "strategy": item.get("strategy"),
            "render_policy": item.get("render_policy"),
            "coverage_required": item.get("coverage_required"),
            "render_target": render_target,
        },
        "translation": {
            "translation_id": item.get("translation_id"),
            "translation_unit_id": item.get("translation_unit_id"),
            "status": item.get("status"),
            "quality": item.get("quality") or {},
        },
    }


def _reconstruction_units(translated_input: dict, translated_units: list[dict]) -> list[dict]:
    output = []
    unit_map = {
        unit.get("unit_id"): unit
        for unit in translated_input.get("units") or []
        if isinstance(unit, dict) and unit.get("unit_id")
    }
    for item in translated_units:
        if item.get("status") == "error":
            continue
        if not normalize_spaces(item.get("translated_text")):
            continue
        if item.get("level") in {"semantic_phrase", "semantic_group"}:
            output.append(_semantic_reconstruction_unit(item, unit_map))
            continue
        unit = unit_map.get(item.get("unit_id"))
        if not unit:
            continue
        output.append(_direct_reconstruction_unit(unit, item, unit_map))
    return output


def _direct_reconstruction_unit(unit: dict, item: dict, unit_map: dict[str, dict]) -> dict:
    render_target = item.get("render_target") or {}
    bbox = render_target.get("bbox") or item.get("bbox") or (unit.get("geometry") or {}).get("bbox")
    understanding = unit.get("understanding") or {}
    context = item.get("context") or {}
    return {
        "unit_id": unit.get("unit_id"),
        "translation_unit_id": item.get("translation_unit_id"),
        "logical_unit_id": item.get("logical_unit_id"),
        "level": unit.get("level"),
        "render_level": unit.get("level"),
        "role": item.get("role") or understanding.get("role"),
        "object_type": item.get("object_type") or understanding.get("object_type"),
        "semantic_kind": item.get("semantic_kind") or understanding.get("semantic_kind"),
        "page_role": context.get("page_role") or understanding.get("page_role"),
        "preservation_mode": item.get("preservation_mode") or (unit.get("policy") or {}).get("preservation_mode"),
        "text": item.get("source_text") or (unit.get("content") or {}).get("text"),
        "translated_text": item.get("translated_text"),
        "bbox": bbox,
        "source_unit_ids": item.get("source_unit_ids") or [unit.get("unit_id")],
        "consume_source_units": False,
        "preferred_over_children": True,
        "overflow_policy": "shrink_or_reflow",
        "line_break_policy": item.get("strategy") or "layout_constrained",
        "layout_budget": _layout_budget(bbox),
        "style": _dominant_style(unit, unit_map),
        "render_target": render_target,
        "render_contract": unit.get("render_contract") or {},
        "translation": item,
    }


def _dominant_style(unit: dict, unit_map: dict[str, dict]) -> dict:
    style = (unit.get("visual") or {}).get("style") or {}
    if _style_has_real_values(style):
        return style
    for child_id in unit.get("children_ids") or []:
        child = unit_map.get(child_id)
        if not child:
            continue
        child_style = _dominant_style(child, unit_map)
        if _style_has_real_values(child_style):
            return child_style
    return style


def _style_has_real_values(style: dict) -> bool:
    ignored = {"flags", "font_size_unit"}
    return any(value is not None for key, value in (style or {}).items() if key not in ignored)


def _layout_budget(bbox: object) -> dict:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return {"bbox_reliable": False}
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    return {
        "bbox_reliable": bool(width and height),
        "width": width,
        "height": height,
        "area": width * height,
    }
