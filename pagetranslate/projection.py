"""Projection of translations back into PagePrint INPUT_DATA."""

from __future__ import annotations

from .text_utils import normalize_spaces
from .text_survival import append_uncovered_source_line_fallbacks
from source_ownership import filter_translation_units_by_ownership, build_source_ownership, is_non_translatable_owner


def project_translations(translated_input: dict, translated_units: list[dict]) -> list[dict]:
    unit_map = {
        unit.get("unit_id"): unit
        for unit in translated_input.get("units") or []
        if isinstance(unit, dict) and unit.get("unit_id")
    }
    translated_units = filter_translation_units_by_ownership(translated_input, translated_units)
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
    # Do not aggregate protected/preserved child text back into a parent.  That
    # would reintroduce formula/code glyphs into translated natural-text blocks.
    pseudo_input = {"units": list(unit_map.values())}
    ownership = build_source_ownership(pseudo_input)
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
            and not is_non_translatable_owner(ownership, child.get("unit_id"))
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
    translated_units = filter_translation_units_by_ownership(translated_input, translated_units)
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
            # Atomic translation units may have synthetic ids such as
            # seg_XXXX_line_YYY while their geometry/style source is the
            # PAGEPRINT line id stored in source_unit_ids. In that case the
            # translated text is valid and must be consumed, not replaced by an
            # identity fallback.
            for sid in item.get("source_unit_ids") or []:
                unit = unit_map.get(sid)
                if unit:
                    break
        if not unit:
            continue
        output.append(_direct_reconstruction_unit(unit, item, unit_map))
    output.extend(_preserve_uncovered_original_units(translated_input, output, unit_map))
    output = _dedupe_parent_child_units(output, translated_input)
    return output


def _is_ancestor_id(anc: str, desc: str) -> bool:
    """Convention de nommage pageprint : pXXX_block_NNN[_line_MMM[_phrase_KKK...]].
    `anc` est ancêtre de `desc` si desc démarre par `anc` + séparateur."""
    return bool(anc) and bool(desc) and desc != anc and desc.startswith(anc + "_")


def _dedupe_parent_child_units(output: list[dict], translated_input: dict) -> list[dict]:
    """UNE seule unité de sortie par texte source : interdit qu'un même texte
    source alimente pagereconstruct par DEUX granularités (bloc parent + ses
    phrases). Si une unité couvre un texte source dont un DESCENDANT est aussi
    couvert par une autre unité, on supprime le PARENT (granularité fine
    prioritaire). Détection par ascendance d'id (children_ids souvent vide)."""
    def sources(u: dict) -> list[str]:
        return [s for s in (u.get("source_unit_ids") or []) if s]

    all_sources: list[str] = []
    for v in output:
        all_sources.extend(sources(v))

    kept: list[dict] = []
    for i, u in enumerate(output):
        u_sources = sources(u)
        others = [s for j, v in enumerate(output) if j != i for s in sources(v)]
        # u est parent-doublon si un autre couvre un descendant d'une de ses sources
        redundant = any(_is_ancestor_id(s, o) for s in u_sources for o in others)
        if not redundant:
            kept.append(u)
    return kept


def _preserve_uncovered_original_units(translated_input: dict, reconstruction_units: list[dict], unit_map: dict[str, dict]) -> list[dict]:
    """Hard text survival fallback.

    After atomic line splitting, this should normally add very few units.  If
    PAGEPRINT has visible lines that never reached PAGETRANSLATE, they are
    rendered as source text with an explicit identity_fallback translation id.
    Silent disappearance is forbidden.
    """
    return append_uncovered_source_line_fallbacks(translated_input, reconstruction_units, unit_map)


def _preserved_reconstruction_unit(unit: dict, unit_map: dict[str, dict]) -> dict:
    understanding = unit.get("understanding") or {}
    policy = unit.get("policy") or {}
    bbox = (unit.get("geometry") or {}).get("bbox")
    uid = unit.get("unit_id")
    text = (unit.get("content") or {}).get("text")
    reason = (
        policy.get("non_translatable_reason")
        or policy.get("preservation_reason")
        or policy.get("translation_strategy")
        or understanding.get("role")
        or "not_selected_for_translation"
    )
    return {
        "unit_id": uid,
        "translation_unit_id": None,
        "logical_unit_id": uid,
        "level": unit.get("level"),
        "render_level": unit.get("level"),
        "role": understanding.get("role"),
        "object_type": understanding.get("object_type") or policy.get("unit_type"),
        "semantic_kind": understanding.get("semantic_kind"),
        "page_role": understanding.get("page_role"),
        "preservation_mode": "preserve_original",
        "text": text,
        "translated_text": text,
        "bbox": bbox,
        "layout_bbox": bbox,
        "patch_bbox": None,
        "coverage_bbox": bbox,
        "anchor_bbox": bbox,
        "source_unit_ids": [uid],
        "consume_source_units": False,
        "source_units_consumed": False,
        "preferred_over_children": False,
        "skip_original_units": False,
        "render_as": unit.get("level"),
        "overflow_policy": "preserve_original",
        "line_break_policy": "source_layout",
        "layout_budget": _layout_budget(bbox),
        "style": _dominant_style(unit, unit_map),
        "render_target": {"bbox": bbox, "layout_bbox": bbox, "coverage_bbox": bbox, "style_source_unit_id": uid},
        "render_contract": {
            "mode": "preserve_original",
            "preservation_mode": "preserve_original",
            "reason": reason,
        },
        "translation": {"status": "preserved", "reason": reason},
    }


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
        "render_contract": {
            **(unit.get("render_contract") or {}),
            "mode": "translated_text",
            "strategy": item.get("strategy"),
        },
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
