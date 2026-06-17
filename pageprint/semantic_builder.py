"""Build PAGEPRINT semantic units from roles and logical structures."""

from __future__ import annotations

import re

from .role_resolver import _is_strong_math
from .structure_builders.common import bbox_of, eligible_text_units, role_of, text_of
from .structure_builders.publisher_mark_builder import PUBLISHER_RE, WATERMARK_RE
from .text_postprocessors import merge_hyphenated_segments, repair_hyphenation


TRANSLATABLE_ROLES = {
    "body_paragraph",
    "title",
    "subtitle",
    "section_heading",
    "subsection_heading",
    "list_item",
    "figure_caption",
    "table_caption",
    "table_header_cell",
    "table_body_cell",
    "formula_explanation",
    "toc_entry_title",
    "index_head_term",
    "index_subentry",
    "author_bio",
}


def build_semantic_system(
    page_structure: dict,
    units: list[dict],
    *,
    logical_structures: dict | None = None,
    legacy_semantic_system: dict | None = None,
) -> dict:
    semantic_phrases = list((legacy_semantic_system or {}).get("semantic_phrases") or [])
    semantic_groups = list((legacy_semantic_system or {}).get("semantic_groups") or [])
    logical_structures = logical_structures or {}
    logical_text_units = []
    translation_segments = _build_translation_segments_from_logical_structures(logical_structures)
    if translation_segments:
        semantic_phrases = _semantic_phrases_from_segments(translation_segments)
    elif not semantic_phrases:
        semantic_phrases = _build_semantic_phrases(units)
    for idx, phrase in enumerate(semantic_phrases, start=1):
        logical_text_units.append({
            "logical_unit_id": phrase.get("logical_unit_id") or f"logical_text_{idx:04d}",
            "type": phrase.get("role") or "text_unit",
            "source_unit_ids": phrase.get("source_unit_ids") or [],
            "text": phrase.get("text"),
            "bbox": phrase.get("bbox"),
        })
        if not translation_segments and phrase.get("translation_mode", "translate") == "translate":
            translation_segments.append({
                "translation_segment_id": f"seg_{idx:04d}",
                "logical_unit_id": phrase.get("logical_unit_id"),
                "source_unit_ids": phrase.get("source_unit_ids") or [],
                "source_text": phrase.get("text"),
                "role": phrase.get("role"),
                "object_type": phrase.get("object_type"),
                "semantic_kind": phrase.get("semantic_kind"),
                "translation_mode": "translate",
                "protected_tokens": phrase.get("protected") or [],
                "bbox": phrase.get("bbox"),
            })
    translation_segments = _augment_missing_translation_segments(
        translation_segments,
        units,
        start_index=len(translation_segments) + 1,
    )
    translation_segments = _dedupe_translation_segments(merge_hyphenated_segments(translation_segments))
    return {
        "schema_version": "pageprint.semantic_system.v2",
        "semantic_phrases": semantic_phrases,
        "semantic_groups": semantic_groups,
        "logical_text_units": logical_text_units,
        "translation_segments": translation_segments,
        "logical_structures": logical_structures,
        "segment_source": _segment_source(logical_structures, translation_segments),
    }


def build_semantic_system_from_logical_structures(input_data: dict) -> dict:
    segments = _build_translation_segments_from_logical_structures(input_data.get("logical_structures") or {})
    return {
        "schema_version": "pageprint.semantic_system.v2",
        "semantic_phrases": _semantic_phrases_from_segments(segments),
        "semantic_groups": [],
        "logical_text_units": [],
        "translation_segments": segments,
        "logical_structures": input_data.get("logical_structures") or {},
        "segment_source": _segment_source(input_data.get("logical_structures") or {}, segments),
    }


def _build_translation_segments_from_logical_structures(logical_structures: dict) -> list[dict]:
    segments: list[dict] = []
    counter = 1

    def add_segment(**kwargs) -> None:
        nonlocal counter
        raw_text = str(kwargs.get("source_text") or "").strip()
        source_text = repair_hyphenation(raw_text)
        source_unit_ids = [sid for sid in kwargs.get("source_unit_ids") or [] if sid]
        if not source_text or not source_unit_ids:
            return
        # Never send publisher/scan artefacts to the engine.
        if PUBLISHER_RE.search(source_text) or WATERMARK_RE.search(source_text):
            return
        # Never translate a strong-math line ("= …", "× 6 ×") even if a logical
        # structure (mis-detected toc_entry / list_item) swept it in: it is
        # preserved as a formula, and translating it self-collides with its own
        # protected region. Prose with an inline "1 × 1" is not strong math.
        if _is_strong_math(source_text):
            return
        # A pure-symbol segment ("*", "=", "•") is a formula/artifact fragment,
        # not translatable text; sending it to the engine only collides with the
        # formula it belongs to.
        if not re.search(r"[A-Za-zÀ-ÿ]", source_text):
            return
        protected = list(dict.fromkeys([
            token for token in [
                *(kwargs.get("protected_tokens") or []),
                *_technical_tokens(source_text),
            ]
            if token and token in source_text
        ]))
        segments.append({
            "translation_segment_id": f"seg_{counter:04d}",
            "logical_unit_id": kwargs.get("logical_unit_id"),
            "source_unit_ids": source_unit_ids,
            "source_text": source_text,
            "source_text_raw": raw_text,
            "text": source_text,
            "role": kwargs.get("role"),
            "object_type": kwargs.get("object_type") or "natural_text",
            "semantic_kind": kwargs.get("semantic_kind"),
            "translation_mode": "translate",
            "translation_strategy": kwargs.get("translation_strategy") or "layout_constrained",
            "protected_tokens": protected,
            "protected": protected,
            "render_target": kwargs.get("render_target") or {},
            "bbox": kwargs.get("bbox"),
            "semantic_level": "semantic_phrase",
        })
        counter += 1

    for entry in logical_structures.get("toc_entries") or []:
        add_segment(
            logical_unit_id=entry.get("logical_unit_id"),
            source_unit_ids=entry.get("title_unit_ids") or entry.get("source_unit_ids"),
            source_text=entry.get("title_text"),
            role="toc_entry_title" if entry.get("type") == "toc_entry" else "toc_title",
            object_type="natural_text",
            semantic_kind="toc_entry_title",
            protected_tokens=entry.get("protected_values") or [],
            bbox=entry.get("bbox"),
        )

    for heading in logical_structures.get("headings") or []:
        add_segment(
            logical_unit_id=heading.get("logical_unit_id"),
            source_unit_ids=heading.get("source_unit_ids"),
            source_text=heading.get("text"),
            role=heading.get("role") or "section_heading",
            object_type="natural_text",
            semantic_kind="heading",
            bbox=heading.get("bbox"),
            translation_strategy="layout_constrained",
        )

    for paragraph in logical_structures.get("body_paragraphs") or []:
        add_segment(
            logical_unit_id=paragraph.get("logical_unit_id"),
            source_unit_ids=paragraph.get("source_unit_ids") or paragraph.get("line_unit_ids"),
            source_text=paragraph.get("text"),
            role="body_paragraph",
            object_type="natural_text",
            semantic_kind="prose",
            bbox=paragraph.get("bbox"),
            translation_strategy="paragraph_flow",
        )

    for entry in logical_structures.get("index_entries") or []:
        add_segment(
            logical_unit_id=entry.get("logical_unit_id"),
            source_unit_ids=entry.get("source_unit_ids"),
            source_text=entry.get("head_term"),
            role="index_head_term",
            object_type="technical_term",
            semantic_kind="index_term",
            protected_tokens=entry.get("page_refs") or [],
            bbox=entry.get("bbox"),
        )
        for sub_idx, subentry in enumerate(entry.get("subentries") or [], start=1):
            add_segment(
                logical_unit_id=f"{entry.get('logical_unit_id')}_sub{sub_idx}",
                source_unit_ids=entry.get("source_unit_ids"),
                source_text=subentry.get("text"),
                role="index_subentry",
                object_type="natural_text",
                semantic_kind="index_subentry",
                protected_tokens=subentry.get("page_refs") or [],
                bbox=entry.get("bbox"),
            )

    for table in logical_structures.get("tables") or []:
        for cell in table.get("cells") or []:
            role = cell.get("role") or "table_body_cell"
            if role in {"command_name", "path", "file_name", "code", "table_numeric_cell"}:
                continue
            if cell.get("translation_mode") == "preserve_text_exactly":
                continue
            add_segment(
                logical_unit_id=cell.get("cell_id") or table.get("logical_unit_id"),
                source_unit_ids=cell.get("source_unit_ids"),
                source_text=cell.get("text"),
                role=role if str(role).startswith("table_") else "table_body_cell",
                object_type="table_cell",
                semantic_kind="table_cell_text",
                bbox=cell.get("bbox"),
                translation_strategy="layout_constrained",
            )

    for caption in logical_structures.get("captions") or []:
        add_segment(
            logical_unit_id=caption.get("logical_unit_id"),
            source_unit_ids=caption.get("source_unit_ids"),
            source_text=caption.get("translatable_text") or caption.get("caption_text"),
            role="table_caption" if str(caption.get("label") or "").lower().startswith("tab") else "figure_caption",
            object_type="natural_text",
            semantic_kind="caption_text",
            protected_tokens=caption.get("preserve") or [],
            bbox=caption.get("bbox"),
        )

    # Figure-internal labels are currently preserved with the figure pixels.
    # Translating isolated labels without rebuilding the whole figure creates
    # duplicate labels and destroys diagrams. Captions are still handled above.

    for item in logical_structures.get("list_items") or []:
        add_segment(
            logical_unit_id=item.get("logical_unit_id"),
            source_unit_ids=item.get("source_unit_ids"),
            source_text=item.get("text"),
            role="list_item",
            object_type="natural_text",
            semantic_kind="list_item_text",
            protected_tokens=[item.get("marker")] if item.get("marker") else [],
            bbox=item.get("bbox"),
        )

    for entry in logical_structures.get("author_entries") or logical_structures.get("author_bios") or []:
        add_segment(
            logical_unit_id=entry.get("logical_unit_id"),
            source_unit_ids=entry.get("source_unit_ids"),
            source_text=entry.get("text"),
            role=entry.get("type") or "author_bio",
            object_type="natural_text",
            semantic_kind="biography",
            bbox=entry.get("bbox"),
        )

    return _dedupe_translation_segments(merge_hyphenated_segments(segments))


def _augment_missing_translation_segments(segments: list[dict], units: list[dict], *, start_index: int = 1) -> list[dict]:
    """Rescue natural text lines that logical builders missed.

    Logical structures are preferable, but when a page is misclassified upstream
    (table/formula/code overreach), relying only on them leaves real text in the
    clean background and out of PAGETRANSLATE.  This pass adds conservative line
    candidates not already covered by an existing segment.  It never rescues
    figure labels, strong math, code, page references or pure numbers.
    """
    output = list(segments or [])
    by_id = {u.get("unit_id"): u for u in units if isinstance(u, dict) and u.get("unit_id")}
    covered = set()
    for seg in output:
        for sid in seg.get("source_unit_ids") or []:
            covered.add(sid)
            covered.update(_descendant_ids(sid, by_id))
    counter = int(start_index or 1)
    for unit in eligible_text_units(units):
        uid = unit.get("unit_id")
        if not uid or uid in covered:
            continue
        if unit.get("level") not in {"line", "phrase"}:
            continue
        if _has_covered_ancestor(uid, covered, by_id):
            continue
        role = role_of(unit)
        text = repair_hyphenation(text_of(unit))
        if not _is_rescuable_translation_line(text, role, unit):
            continue
        protected = list(dict.fromkeys(_technical_tokens(text)))
        output.append({
            "translation_segment_id": f"seg_rescue_{counter:04d}",
            "logical_unit_id": f"rescued_text_{counter:04d}",
            "source_unit_ids": [uid],
            "source_text": text,
            "source_text_raw": text_of(unit),
            "text": text,
            "role": "body_paragraph" if role in {"unknown", "path", "code_line", "code_block", "formula_expression"} else role,
            "object_type": "natural_text",
            "semantic_kind": "prose_rescued",
            "translation_mode": "translate",
            "translation_strategy": "paragraph_flow" if len(text.split()) >= 6 else "layout_constrained",
            "protected_tokens": protected,
            "protected": protected,
            "bbox": bbox_of(unit),
            "semantic_level": "semantic_phrase",
            "rescue_reason": "logical_structure_gap",
        })
        covered.add(uid)
        counter += 1
    return output


def _descendant_ids(unit_id: str, by_id: dict[str, dict]) -> set[str]:
    prefix = str(unit_id or "") + "_"
    return {uid for uid in by_id if str(uid).startswith(prefix)}


def _has_covered_ancestor(unit_id: str, covered: set[str], by_id: dict[str, dict]) -> bool:
    parent = by_id.get(unit_id, {}).get("parent_id")
    while parent:
        if parent in covered:
            return True
        parent = by_id.get(parent, {}).get("parent_id")
    return False


def _is_rescuable_translation_line(text: str, role: str, unit: dict) -> bool:
    s = re.sub(r"\s+", " ", str(text or "").strip())
    if not s or len(s) < 3:
        return False
    if role in {"page_reference", "toc_page_reference", "toc_section_number", "index_page_reference", "publisher_mark", "watermark", "diagram_label", "axis_label", "legend_label"}:
        return False
    if role in {"formula_expression"} and _is_strong_math(s):
        return False
    if role in {"code_line", "code_block", "command_name"} and _looks_like_code_line(s):
        return False
    if not re.search(r"[A-Za-zÀ-ÿ]", s):
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’\-]*", s)
    if len(words) >= 5:
        return True
    if role in TRANSLATABLE_ROLES and len(words) >= 1:
        return True
    return False


def _looks_like_code_line(text: str) -> bool:
    s = str(text or "").strip()
    if not s:
        return False
    if len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’\-]*", s)) >= 7 and not re.search(r"\b(?:SELECT|CREATE|INSERT|UPDATE|DELETE|FROM|WHERE|JOIN)\b", s, re.IGNORECASE):
        return False
    return bool(re.search(r"\b(?:SELECT|FROM|WHERE|GROUP\s+BY|ORDER\s+BY|INSERT|UPDATE|DELETE|JOIN|WITH|VALUES|CREATE|ALTER|DROP)\b|[;{}]", s, re.IGNORECASE))


def _dedupe_translation_segments(segments: list[dict]) -> list[dict]:
    """Remove parent/child and text/bbox duplicate translation segments."""
    output: list[dict] = []
    used_units: set[str] = set()
    for segment in segments:
        text = re.sub(r"\s+", " ", str(segment.get("source_text") or segment.get("text") or "").strip()).lower()
        source_ids = {sid for sid in segment.get("source_unit_ids") or [] if sid}
        if source_ids and source_ids <= used_units:
            continue
        bbox = segment.get("bbox")
        duplicate = False
        for kept in output:
            kept_text = re.sub(r"\s+", " ", str(kept.get("source_text") or kept.get("text") or "").strip()).lower()
            if text and text == kept_text and _bbox_overlap_ratio(bbox, kept.get("bbox")) >= 0.65:
                duplicate = True
                break
            kept_ids = set(kept.get("source_unit_ids") or [])
            if source_ids and kept_ids and source_ids & kept_ids and _bbox_overlap_ratio(bbox, kept.get("bbox")) >= 0.55:
                duplicate = True
                break
        if duplicate:
            continue
        output.append(segment)
        used_units.update(source_ids)
    for idx, segment in enumerate(output, start=1):
        segment["translation_segment_id"] = f"seg_{idx:04d}"
    return output


def _bbox_overlap_ratio(a, b) -> float:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != 4 or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1.0, min(max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0), max(0.0, bx1 - bx0) * max(0.0, by1 - by0)))
    return inter / area


def _segment_source(logical_structures: dict, segments: list[dict]) -> str:
    if not segments:
        return "visual_fallback"
    if logical_structures.get("body_paragraphs"):
        return "logical_body_paragraphs"
    if logical_structures.get("tables"):
        return "logical_tables"
    if logical_structures.get("index_entries"):
        return "logical_index_entries"
    if logical_structures.get("toc_entries"):
        return "logical_toc_entries"
    if logical_structures.get("captions") or logical_structures.get("figures"):
        return "logical_figures_captions"
    return "logical_structures"


def _semantic_phrases_from_segments(segments: list[dict]) -> list[dict]:
    return [
        {
            "unit_id": segment.get("translation_segment_id"),
            "logical_unit_id": segment.get("logical_unit_id"),
            "semantic_level": "semantic_phrase",
            "text": segment.get("source_text"),
            "source_unit_ids": segment.get("source_unit_ids") or [],
            "role": segment.get("role"),
            "object_type": segment.get("object_type"),
            "semantic_kind": segment.get("semantic_kind"),
            "translation_mode": segment.get("translation_mode") or "translate",
            "translation_strategy": segment.get("translation_strategy") or "layout_constrained",
            "protected": segment.get("protected_tokens") or [],
            "bbox": segment.get("bbox"),
        }
        for segment in segments
    ]


def _technical_tokens(text: str) -> list[str]:
    common_titles = {"CONTENTS", "INTRODUCTION", "BACKGROUND", "SUMMARY", "CONCLUSION", "CHAPTER", "APPENDIX"}
    tokens = []
    for token in re.findall(r"\b[A-Z][A-Z0-9&./+-]{1,12}\b", text or ""):
        if token in common_titles:
            continue
        if len(token) <= 8 or any(ch.isdigit() for ch in token):
            tokens.append(token)
    return tokens


def _build_semantic_phrases(units: list[dict]) -> list[dict]:
    output = []
    selected_units = _select_translation_source_units(units)
    for idx, unit in enumerate(selected_units, start=1):
        role = role_of(unit)
        policy = unit.get("policy") or {}
        if role not in TRANSLATABLE_ROLES:
            continue
        if policy.get("translatable") is False or policy.get("translation_strategy") == "needs_role_resolution":
            continue
        text = text_of(unit)
        if not text:
            continue
        text = repair_hyphenation(text)
        output.append({
            "unit_id": f"semantic_phrase_{idx:04d}",
            "semantic_level": "semantic_phrase",
            "text": text,
            "source_unit_ids": [unit["unit_id"]],
            "role": role,
            "object_type": (unit.get("understanding") or {}).get("object_type"),
            "semantic_kind": (unit.get("understanding") or {}).get("semantic_kind"),
            "translation_mode": "translate",
            "translation_strategy": policy.get("translation_strategy") or "layout_constrained",
            "protected": policy.get("protected_tokens") or [],
            "bbox": bbox_of(unit),
            "structural_context": {
                "block_unit_id": _ancestor_block_id(unit),
                "source_role": role,
            },
        })
    return output


def _ancestor_block_id(unit: dict) -> str | None:
    if unit.get("level") == "block":
        return unit.get("unit_id")
    return unit.get("parent_id")


def _select_translation_source_units(units: list[dict]) -> list[dict]:
    """Pick one textual granularity per branch: phrase > line > block."""
    by_id = {u.get("unit_id"): u for u in units if isinstance(u, dict) and u.get("unit_id")}
    text_units = eligible_text_units(units)
    textual_descendants: dict[str, set[str]] = {u.get("unit_id"): set() for u in text_units}
    textual_ids = set(textual_descendants)
    for unit in text_units:
        cursor = unit
        while cursor.get("parent_id") in by_id:
            parent_id = cursor.get("parent_id")
            if parent_id in textual_ids:
                textual_descendants.setdefault(parent_id, set()).add(unit["unit_id"])
            cursor = by_id[parent_id]
    selected = []
    for unit in text_units:
        level = unit.get("level")
        descendants = textual_descendants.get(unit.get("unit_id")) or set()
        if level == "span" and unit.get("parent_id") in textual_ids:
            continue
        if descendants and level in {"block", "line"}:
            continue
        selected.append(unit)
    return selected
