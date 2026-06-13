"""Coalesce visual translation units into semantic phrases when needed."""

from __future__ import annotations

from .text_utils import bbox_union, normalize_spaces

try:
    from pageprint.graph_query import can_merge_for_translation
except Exception:  # pragma: no cover - fallback for standalone packaging
    can_merge_for_translation = None


PROTECTED_STRATEGIES = {"background_only", "exact_preserve", "keep_original"}
PROTECTED_TYPES = {
    "protected_visual",
    "formula",
    "equation",
    "code",
    "code_visible",
    "symbolic_expression",
    "chemical_formula",
}


def coalesce_translation_units(units: list[dict]) -> list[dict]:
    """Merge consecutive open visual units into synthetic semantic phrases.

    This is a safety net for PagePrint outputs that do not yet provide
    semantic_phrases. Existing semantic units are left untouched.
    """
    output: list[dict] = []
    idx = 0
    while idx < len(units):
        item = units[idx]
        if not _can_start_coalesce(item):
            output.append(item)
            idx += 1
            continue

        group = [item]
        cursor = idx
        while group[-1].get("sentence", {}).get("continues_to_next"):
            nxt = units[cursor + 1] if cursor + 1 < len(units) else None
            if not nxt or not _can_join(group[-1], nxt):
                break
            group.append(nxt)
            cursor += 1

        if len(group) == 1:
            output.append(item)
            idx += 1
            continue

        output.append(_coalesced_unit(group))
        idx = cursor + 1

    for pos, item in enumerate(output, start=1):
        item["translation_unit_id"] = f"tu_{pos:04d}"
    return output


def _can_start_coalesce(item: dict) -> bool:
    if item.get("level") in {"semantic_phrase", "semantic_group"}:
        return False
    if item.get("level") not in {"phrase", "line"}:
        return False
    if not item.get("sentence", {}).get("continues_to_next"):
        return False
    return not _is_protected(item)


def _can_join(previous: dict, current: dict) -> bool:
    if current.get("level") in {"semantic_phrase", "semantic_group"}:
        return False
    if current.get("level") not in {"phrase", "line"}:
        return False
    if previous.get("block_id") != current.get("block_id"):
        return False
    if _is_protected(current):
        return False
    if can_merge_for_translation is not None and not can_merge_for_translation(previous, current):
        return False
    previous_strategy = previous.get("strategy") or "layout_constrained"
    current_strategy = current.get("strategy") or "layout_constrained"
    return previous_strategy == current_strategy


def _is_protected(item: dict) -> bool:
    strategy = str(item.get("strategy") or "").lower()
    object_type = str(item.get("object_type") or "").lower()
    semantic_kind = str(item.get("semantic_kind") or "").lower()
    return bool(strategy in PROTECTED_STRATEGIES or object_type in PROTECTED_TYPES or semantic_kind in PROTECTED_TYPES)


def _coalesced_unit(group: list[dict]) -> dict:
    first = group[0]
    last = group[-1]
    source_ids = [
        source_id
        for item in group
        for source_id in (item.get("source_unit_ids") or [item.get("unit_id")])
        if source_id
    ]
    text = normalize_spaces(" ".join(item.get("source_text") or "" for item in group))
    return {
        **first,
        "translation_unit_id": first.get("translation_unit_id"),
        "unit_id": "synthetic_semantic_phrase:" + "+".join(source_ids),
        "level": "semantic_phrase",
        "source_unit_ids": source_ids,
        "source_text": text,
        "bbox": bbox_union([item.get("bbox") for item in group]),
        "reading_order_index": first.get("reading_order_index"),
        "strategy": first.get("strategy") or "semantic_reflow",
        "semantic_kind": first.get("semantic_kind") or "prose",
        "sentence": {
            **last.get("sentence", {}),
            "is_sentence_start": first.get("sentence", {}).get("is_sentence_start", True),
            "continues_from_previous": first.get("sentence", {}).get("continues_from_previous", False),
            "continues_to_next": last.get("sentence", {}).get("continues_to_next", False),
            "is_sentence_end": last.get("sentence", {}).get("is_sentence_end", True),
            "is_multiline_phrase": True,
            "coalesced_from_visual_units": True,
            "coalesced_source_unit_ids": source_ids,
        },
        "coalesced": {
            "from_unit_ids": [item.get("unit_id") for item in group],
            "from_levels": [item.get("level") for item in group],
            "reason": "open_sentence_continuation",
        },
    }
