"""Compile PAGEPRINT downstream execution plans."""

from __future__ import annotations

from .structure_builders.common import bbox_of, eligible_text_units, role_of, text_of


TRANSLATE_MODES = {"layout_constrained", "paragraph_flow", "semantic_reflow", "toc_row_layout"}
FORBIDDEN_TRANSLATION_ROLES = {
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


def compile_views(
    units: list[dict],
    *,
    semantic_system: dict | None = None,
    logical_structures: dict | None = None,
    page_intelligence: dict | None = None,
) -> dict:
    unit_map = {u.get("unit_id"): u for u in units if isinstance(u, dict) and u.get("unit_id")}
    translation_plan, plan_errors, plan_warnings = _translation_plan(unit_map, semantic_system or {}, page_intelligence or {})
    preservation_plan = _preservation_plan(units, logical_structures or {})
    reconstruction_plan = _reconstruction_plan(units, translation_plan, preservation_plan)
    exclusion_plan = _exclusion_plan(units, logical_structures or {})
    metrics = {
        "translation_plan_count": len(translation_plan),
        "preservation_plan_count": len(preservation_plan),
        "reconstruction_plan_count": len(reconstruction_plan),
        "logical_unit_count": len((logical_structures or {}).get("logical_units") or []),
        "semantic_segment_count": len((semantic_system or {}).get("translation_segments") or []),
        "translation_plan_compile_errors": len(plan_errors),
        "translation_plan_compile_warnings": len(plan_warnings),
    }
    return {
        "translation_plan": translation_plan,
        "preservation_plan": preservation_plan,
        "reconstruction_plan": reconstruction_plan,
        "exclusion_plan": exclusion_plan,
        "logical_structures": logical_structures or {},
        "metrics": metrics,
        "translation_plan_compile_errors": plan_errors,
        "translation_plan_compile_warnings": plan_warnings,
    }


def _translation_plan(unit_map: dict[str, dict], semantic_system: dict, page_intelligence: dict) -> tuple[list[dict], list[str], list[str]]:
    entries = semantic_system.get("translation_segments") or semantic_system.get("semantic_phrases") or []
    output = []
    errors = []
    warnings = []
    for idx, entry in enumerate(entries, start=1):
        source_ids = [sid for sid in entry.get("source_unit_ids") or [] if sid in unit_map]
        if not source_ids:
            errors.append(f"translation_segment_without_source_units:{entry.get('translation_segment_id') or idx}")
            continue
        sample = unit_map[source_ids[0]]
        policy = sample.get("policy") or {}
        understanding = sample.get("understanding") or {}
        role = entry.get("role") or understanding.get("role")
        object_type = entry.get("object_type") or understanding.get("object_type")
        semantic_kind = entry.get("semantic_kind") or understanding.get("semantic_kind")
        if role in FORBIDDEN_TRANSLATION_ROLES:
            warnings.append(f"forbidden_role_excluded:{entry.get('translation_segment_id') or idx}:{role}")
            continue
        if not role or role == "unknown" or not object_type or object_type == "unknown":
            errors.append(f"translation_segment_missing_role_or_object:{entry.get('translation_segment_id') or idx}")
            continue
        entry_mode = entry.get("translation_mode")
        mode = "translate" if entry_mode == "translate" or (entry_mode is None and policy.get("translatable") is True) else "needs_review"
        if mode != "translate":
            warnings.append(f"translation_segment_not_translatable:{entry.get('translation_segment_id') or idx}")
            continue
        level = sample.get("level")
        if level in {"word", "char"}:
            errors.append(f"translation_segment_word_char_source:{entry.get('translation_segment_id') or idx}:{source_ids[0]}")
            continue
        source_text = entry.get("source_text") or entry.get("text") or text_of(sample)
        if not source_text:
            errors.append(f"translation_segment_empty_text:{entry.get('translation_segment_id') or idx}")
            continue
        render_target = entry.get("render_target") or _render_target_for_segment(entry, unit_map, source_ids, idx, role)
        protected_tokens = [
            token for token in list(dict.fromkeys((entry.get("protected_tokens") or entry.get("protected") or []) + (policy.get("protected_tokens") or [])))
            if token and token in source_text
        ]
        output.append({
            "translation_unit_id": f"tp_{idx:04d}",
            "unit_id": entry.get("translation_segment_id") or entry.get("unit_id") or f"tp_{idx:04d}",
            "level": entry.get("semantic_level") or "semantic_phrase",
            "source_unit_ids": source_ids,
            "logical_unit_id": entry.get("logical_unit_id"),
            "source_text": source_text,
            "role": role,
            "object_type": object_type,
            "semantic_kind": semantic_kind,
            "translation_mode": mode,
            "translation_strategy": policy.get("translation_strategy") or entry.get("translation_strategy") or "layout_constrained",
            "protected_tokens": protected_tokens,
            "context": {
                "page_role": page_intelligence.get("page_role"),
                "page_family": page_intelligence.get("page_family"),
                "layout_type": page_intelligence.get("layout_type"),
            },
            "render_target": render_target,
            "qa_requirements": {
                "preserve_numbers": True,
                "preserve_protected_tokens": True,
                "check_overflow": True,
            },
            "reason_included": "semantic_segment_translatable",
            "bbox": entry.get("bbox") or bbox_of(sample),
        })
    return output, errors, warnings


def _preservation_plan(units: list[dict], logical_structures: dict | None = None) -> list[dict]:
    output = []
    counter = 1
    logical_structures = logical_structures or {}
    for entry in logical_structures.get("toc_entries") or []:
        for kind, value in (
            ("toc_bullet_marker", entry.get("marker")),
            ("toc_section_number", entry.get("section_number")),
            ("toc_page_reference", entry.get("page_reference")),
        ):
            if not value:
                continue
            output.append({
                "preservation_id": f"pres_{counter:04d}",
                "source_unit_ids": entry.get("source_unit_ids") or [],
                "logical_unit_id": entry.get("logical_unit_id"),
                "text": value,
                "preservation_mode": "preserve_text_exactly",
                "render_mode": "fixed_preserve",
                "reason": kind,
                "bbox": entry.get("bbox"),
            })
            counter += 1
    for entry in logical_structures.get("index_entries") or []:
        for ref in entry.get("page_refs") or []:
            output.append({
                "preservation_id": f"pres_{counter:04d}",
                "source_unit_ids": entry.get("source_unit_ids") or [],
                "logical_unit_id": entry.get("logical_unit_id"),
                "text": ref,
                "preservation_mode": "preserve_text_exactly",
                "render_mode": "fixed_preserve",
                "reason": "index_page_reference",
                "bbox": entry.get("bbox"),
            })
            counter += 1
        for subentry in entry.get("subentries") or []:
            for ref in subentry.get("page_refs") or []:
                output.append({
                    "preservation_id": f"pres_{counter:04d}",
                    "source_unit_ids": subentry.get("source_unit_ids") or entry.get("source_unit_ids") or [],
                    "logical_unit_id": entry.get("logical_unit_id"),
                    "text": ref,
                    "preservation_mode": "preserve_text_exactly",
                    "render_mode": "fixed_preserve",
                    "reason": "index_page_reference",
                    "bbox": subentry.get("bbox") or entry.get("bbox"),
                })
                counter += 1
    for caption in logical_structures.get("captions") or []:
        for kind, value in (
            ("caption_label", caption.get("label")),
            ("caption_number", caption.get("number")),
        ):
            if not value:
                continue
            output.append({
                "preservation_id": f"pres_{counter:04d}",
                "source_unit_ids": caption.get("source_unit_ids") or [],
                "logical_unit_id": caption.get("logical_unit_id"),
                "text": value,
                "preservation_mode": "preserve_text_exactly",
                "render_mode": "fixed_preserve",
                "reason": kind,
                "bbox": caption.get("bbox"),
            })
            counter += 1
    for table in logical_structures.get("tables") or []:
        for cell in table.get("cells") or []:
            if cell.get("translation_mode") != "preserve_text_exactly":
                continue
            output.append({
                "preservation_id": f"pres_{counter:04d}",
                "source_unit_ids": cell.get("source_unit_ids") or [],
                "logical_unit_id": cell.get("cell_id"),
                "text": cell.get("text"),
                "preservation_mode": "preserve_text_exactly",
                "render_mode": "fixed_preserve",
                "reason": cell.get("cell_kind") or cell.get("role"),
                "bbox": cell.get("bbox"),
            })
            counter += 1
    for artifact_group in ("publisher_marks", "watermarks", "page_numbers"):
        for artifact in logical_structures.get(artifact_group) or []:
            mode = artifact.get("preservation_mode") or "exclude_as_artifact"
            if mode == "exclude_as_artifact":
                continue
            output.append({
                "preservation_id": f"pres_{counter:04d}",
                "source_unit_ids": artifact.get("source_unit_ids") or [],
                "logical_unit_id": artifact.get("logical_unit_id"),
                "text": artifact.get("text"),
                "preservation_mode": mode,
                "render_mode": "fixed_preserve",
                "reason": artifact.get("type") or artifact_group,
                "bbox": artifact.get("bbox"),
            })
            counter += 1
    for unit in _leaf_text_units(eligible_text_units(units)):
        policy = unit.get("policy") or {}
        mode = policy.get("preservation_mode") or (unit.get("preservation_policy") or {}).get("mode")
        if not mode or mode == "none":
            continue
        if mode == "protect_token_inside_translation":
            continue
        output.append({
            "preservation_id": f"pres_{counter:04d}",
            "source_unit_ids": [unit["unit_id"]],
            "text": text_of(unit),
            "preservation_mode": mode,
            "render_mode": policy.get("render_policy"),
            "reason": policy.get("preservation_reason") or (unit.get("preservation_policy") or {}).get("reason"),
            "bbox": bbox_of(unit),
        })
        counter += 1
    return output


def _leaf_text_units(text_units: list[dict]) -> list[dict]:
    ids = {u.get("unit_id") for u in text_units}
    ancestors = set()
    by_id = {u.get("unit_id"): u for u in text_units}
    for unit in text_units:
        parent_id = unit.get("parent_id")
        while parent_id:
            if parent_id in ids:
                ancestors.add(parent_id)
            parent = by_id.get(parent_id)
            parent_id = parent.get("parent_id") if parent else None
    return [u for u in text_units if u.get("unit_id") not in ancestors and not (u.get("level") == "span" and u.get("parent_id") in ids)]


def _reconstruction_plan(units: list[dict], translation_plan: list[dict], preservation_plan: list[dict]) -> list[dict]:
    output = []
    consumed: set[str] = set()
    for idx, item in enumerate(translation_plan, start=1):
        consumed.update(item.get("source_unit_ids") or [])
        rt = item.get("render_target") or {}
        output.append({
            "reconstruction_unit_id": rt.get("reconstruction_unit_id"),
            "translation_unit_id": item.get("translation_unit_id"),
            "unit_id": item.get("unit_id"),
            "source_unit_ids": item.get("source_unit_ids") or [],
            "role": item.get("role"),
            "object_type": item.get("object_type"),
            "semantic_kind": item.get("semantic_kind"),
            "bbox": rt.get("bbox"),
            "layout_bbox": rt.get("layout_bbox") or rt.get("bbox"),
            "patch_bbox": rt.get("patch_bbox"),
            "coverage_bbox": rt.get("coverage_bbox"),
            "anchor_bbox": rt.get("anchor_bbox"),
            "style_source_unit_id": rt.get("style_source_unit_id"),
            "render_contract": {"mode": "translated_text", "strategy": item.get("translation_strategy")},
            "text_source": "translation_plan",
            "consume_source_unit_ids": item.get("source_unit_ids") or [],
        })
    for idx, item in enumerate(preservation_plan, start=1):
        source_ids = item.get("source_unit_ids") or []
        consumed.update(source_ids)
        output.append({
            "reconstruction_unit_id": f"ru_pres_{idx:04d}",
            "source_unit_ids": source_ids,
            "role": item.get("reason") or _role_from_source(units, source_ids),
            "object_type": item.get("preservation_mode"),
            "semantic_kind": "preserved",
            "bbox": item.get("bbox"),
            "style_source_unit_id": source_ids[0] if source_ids else None,
            "render_contract": {"mode": item.get("render_mode"), "preservation_mode": item.get("preservation_mode")},
            "text_source": "preservation_plan",
            "consume_source_unit_ids": source_ids,
        })
    return output


def _exclusion_plan(units: list[dict], logical_structures: dict | None = None) -> list[dict]:
    output = []
    counter = 1
    for artifact_group in ("publisher_marks", "watermarks"):
        for artifact in (logical_structures or {}).get(artifact_group) or []:
            output.append({
                "exclusion_id": f"excl_{counter:04d}",
                "source_unit_ids": artifact.get("source_unit_ids") or [],
                "logical_unit_id": artifact.get("logical_unit_id"),
                "reason": artifact.get("type") or artifact_group,
                "bbox": artifact.get("bbox"),
            })
            counter += 1
    for idx, unit in enumerate(units, start=1):
        policy = unit.get("policy") or {}
        if policy.get("preservation_mode") != "exclude_as_artifact":
            continue
        output.append({
            "exclusion_id": f"excl_{counter:04d}",
            "source_unit_ids": [unit.get("unit_id")],
            "reason": policy.get("non_translatable_reason") or "artifact",
            "bbox": bbox_of(unit),
        })
        counter += 1
    return output


_FLOW_ROLES = {"body_paragraph", "paragraph", "body", "list_item", "author_bio", "index_subentry"}


def _bbox_union(boxes) -> list | None:
    boxes = [b for b in boxes if isinstance(b, (list, tuple)) and len(b) == 4]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _best_style_source_id(unit_map: dict, source_ids: list[str]) -> str | None:
    for sid in source_ids:
        u = unit_map.get(sid)
        style = ((u or {}).get("visual") or {}).get("style") or {}
        if any(v is not None for k, v in style.items() if k not in {"flags", "font_size_unit", "font_size_px"}):
            return sid
    return source_ids[0] if source_ids else None


def _render_target_for_segment(entry: dict, unit_map: dict, source_ids: list[str], idx: int, role: str | None) -> dict:
    """Segment-aware render target: layout = logical block (never first line)."""
    source_boxes = [bbox_of(unit_map[sid]) for sid in source_ids if sid in unit_map]
    coverage = _bbox_union(source_boxes)
    logical = entry.get("bbox") or coverage
    anchor = bbox_of(unit_map[source_ids[0]]) if source_ids and source_ids[0] in unit_map else logical
    layout = logical  # flow text lays out in the full logical block
    patch = coverage or logical
    return {
        "reconstruction_unit_id": f"ru_{idx:04d}",
        "bbox": layout,
        "layout_bbox": layout,
        "patch_bbox": patch,
        "coverage_bbox": coverage,
        "anchor_bbox": anchor,
        "style_source_unit_id": _best_style_source_id(unit_map, source_ids),
        "consume_source_unit_ids": source_ids,
    }


def _render_target(unit: dict, source_ids: list[str], idx: int) -> dict:  # legacy/back-compat
    return {
        "reconstruction_unit_id": f"ru_{idx:04d}",
        "bbox": bbox_of(unit),
        "style_source_unit_id": source_ids[0] if source_ids else unit.get("unit_id"),
        "consume_source_unit_ids": source_ids,
    }


def _role_from_source(units: list[dict], source_ids: list[str]) -> str | None:
    source_set = set(source_ids)
    for unit in units:
        if unit.get("unit_id") in source_set:
            return role_of(unit)
    return None
