"""Atomic text-survival utilities for PAGETRANSLATE/PAGERECONSTRUCT.

Hard invariant:

    Every visible source line must have an output path.

This is deliberately stricter than a normal translation pipeline. It favours
text presence over layout beauty:

    1. PAGEPRINT line units become canonical atomic translation units.
    2. Oversized semantic units are split down to line units.
    3. Any visible source line missing from translation_plan is appended.
    4. Any still-uncovered source line is reinserted as audited identity fallback.
"""

from __future__ import annotations

import copy
import re
from typing import Iterable

from source_ownership import build_source_ownership, is_non_translatable_owner, source_ids_have_non_translatable_owner

try:
    from .text_utils import normalize_spaces
except Exception:  # pragma: no cover
    def normalize_spaces(value):
        return " ".join(str(value or "").split())


MAX_TRANSLATION_CHARS = 220
MAX_TRANSLATION_LINES = 1
MIN_TRUNCATION_CHAR_RATIO = 0.35
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z0-9(\"'“])")
_TEXT_ROLES_TO_SKIP_AS_TRANSLATION_CONTEXT = {"word", "char"}


def _text_of(unit: dict | None) -> str:
    if not isinstance(unit, dict):
        return ""
    c = unit.get("content") or {}
    return normalize_spaces(c.get("text") or unit.get("text") or unit.get("source_text") or "")


def _bbox_of(unit_or_item: dict | None):
    if not isinstance(unit_or_item, dict):
        return None
    b = unit_or_item.get("bbox")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return [float(x) for x in b]
    g = unit_or_item.get("geometry") or {}
    b = g.get("bbox")
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return [float(x) for x in b]
    rt = unit_or_item.get("render_target") or {}
    for key in ("layout_bbox", "coverage_bbox", "patch_bbox", "bbox"):
        b = rt.get(key)
        if isinstance(b, (list, tuple)) and len(b) == 4:
            return [float(x) for x in b]
    return None


def _union(boxes: Iterable) -> list | None:
    bs = []
    for b in boxes or []:
        if isinstance(b, (list, tuple)) and len(b) == 4:
            bs.append([float(x) for x in b])
    if not bs:
        return None
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def _reading_key(unit: dict) -> tuple:
    g = unit.get("geometry") or {}
    b = _bbox_of(unit) or [0, 0, 0, 0]
    ro = g.get("reading_order_index")
    return (ro if ro is not None else 10**9, float(b[1]), float(b[0]))


def _unit_map(input_data: dict) -> dict[str, dict]:
    return {u.get("unit_id"): u for u in input_data.get("units") or [] if isinstance(u, dict) and u.get("unit_id")}


def _children_map(input_data: dict) -> dict[str, list[str]]:
    return {u.get("unit_id"): list(u.get("children_ids") or []) for u in input_data.get("units") or [] if isinstance(u, dict) and u.get("unit_id")}


def _descendants(uid: str, cmap: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    seen = set()
    stack = list(cmap.get(uid) or [])
    while stack:
        cid = stack.pop(0)
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
        stack.extend(cmap.get(cid) or [])
    return out


def _role_of(unit: dict | None) -> str:
    if not isinstance(unit, dict):
        return ""
    return str((unit.get("understanding") or {}).get("role") or "")


def _object_type_of(unit: dict | None) -> str:
    if not isinstance(unit, dict):
        return ""
    return str((unit.get("understanding") or {}).get("object_type") or "")


def _in_hard_special_region(unit: dict | None) -> bool:
    if not isinstance(unit, dict):
        return False
    policy = unit.get("policy") or {}
    constraints = unit.get("constraints") or {}
    understanding = unit.get("understanding") or {}
    tags = {
        str(policy.get("unit_type") or "").lower(),
        str(understanding.get("role") or "").lower(),
        str(understanding.get("object_type") or "").lower(),
        str(understanding.get("semantic_kind") or "").lower(),
    }
    if tags & {"formula_region", "formula", "equation", "math_expression", "code_region", "code", "protected_visual_region", "protected_visual"}:
        return True
    if policy.get("skip_translation") or constraints.get("skip_translation") or policy.get("protected_visual"):
        return True
    for m in understanding.get("region_memberships") or []:
        rt = str(m.get("region_type") or "").lower()
        ratio = float(m.get("overlap_ratio") or 0.0)
        mode = str(m.get("coverage_mode") or "").lower()
        if rt in {"formula_region", "code_region", "protected_visual_region"} and (ratio >= 0.35 or mode in {"full_coverage", "dominant_overlap"}):
            return True
    return False


def _is_visible_text_line(unit: dict) -> bool:
    if not isinstance(unit, dict) or unit.get("level") != "line":
        return False
    text = _text_of(unit)
    if not text:
        return False
    if _role_of(unit) in _TEXT_ROLES_TO_SKIP_AS_TRANSLATION_CONTEXT:
        return False
    if _in_hard_special_region(unit):
        return False
    policy = unit.get("policy") or {}
    if policy.get("visible") is False:
        return False
    bbox = _bbox_of(unit)
    if not bbox:
        return False
    w = max(0.0, bbox[2] - bbox[0])
    h = max(0.0, bbox[3] - bbox[1])
    return w > 0.5 and h > 0.5


def _all_visible_lines(input_data: dict) -> list[dict]:
    return sorted([u for u in input_data.get("units") or [] if _is_visible_text_line(u)], key=_reading_key)


def _line_units_for_item(item: dict, unit_map: dict[str, dict], cmap: dict[str, list[str]]) -> list[dict]:
    out = []
    seen = set()

    def add_line(u: dict | None):
        if not isinstance(u, dict):
            return
        uid = u.get("unit_id")
        if uid and uid not in seen and _is_visible_text_line(u):
            out.append(u)
            seen.add(uid)

    for sid in item.get("source_unit_ids") or []:
        u = unit_map.get(sid)
        if not isinstance(u, dict):
            continue
        if u.get("level") == "line":
            add_line(u)
        else:
            for did in _descendants(sid, cmap):
                du = unit_map.get(did)
                if isinstance(du, dict) and du.get("level") == "line":
                    add_line(du)

    if not out:
        src = normalize_spaces(item.get("source_text") or "")
        if src:
            for u in unit_map.values():
                if not _is_visible_text_line(u):
                    continue
                txt = _text_of(u)
                if txt and txt in src and u.get("unit_id") not in seen:
                    add_line(u)

    out.sort(key=_reading_key)
    return out


def _refresh_render_target(item: dict, line: dict, bbox: list | None) -> dict:
    rt = copy.deepcopy(item.get("render_target") or {})
    if not bbox:
        return rt
    uid = line.get("unit_id")
    rt.update({
        "bbox": bbox,
        "layout_bbox": bbox,
        "coverage_bbox": bbox,
        "patch_bbox": bbox,
        "anchor_bbox": bbox,
        "style_source_unit_id": rt.get("style_source_unit_id") or uid,
        "consume_source_unit_ids": [uid] if uid else [],
    })
    return rt


def _line_item_from_plan(item: dict, line: dict, *, index: int, total: int) -> dict:
    base_tuid = item.get("translation_unit_id") or item.get("unit_id") or "tp"
    base_uid = item.get("unit_id") or base_tuid
    uid = line.get("unit_id")
    bbox = _bbox_of(line) or _bbox_of(item)
    role = item.get("role") or _role_of(line)
    object_type = item.get("object_type") or _object_type_of(line)
    text = _text_of(line)

    ni = copy.deepcopy(item)
    ni.update({
        "translation_unit_id": f"{base_tuid}_line_{index:03d}",
        "unit_id": f"{base_uid}_line_{index:03d}",
        "parent_translation_unit_id": base_tuid,
        "parent_unit_id": base_uid,
        "split_index": index,
        "split_count": total,
        "level": "line",
        "source_unit_ids": [uid] if uid else [],
        "source_text": text,
        "bbox": bbox,
        "reading_order_index": (line.get("geometry") or {}).get("reading_order_index") or item.get("reading_order_index") or index,
        "role": role,
        "object_type": object_type,
        "semantic_kind": item.get("semantic_kind") or (line.get("understanding") or {}).get("semantic_kind"),
        "render_target": _refresh_render_target(item, line, bbox),
        "reason_included": f"{item.get('reason_included') or 'translation_plan'}+atomic_line_text_survival",
    })
    return ni


def _standalone_line_item(line: dict, *, index: int) -> dict:
    uid = line.get("unit_id")
    bbox = _bbox_of(line)
    role = _role_of(line) or "body_paragraph"
    object_type = _object_type_of(line) or "natural_text"
    text = _text_of(line)
    tuid = f"tp_line_survival_{index:04d}"
    return {
        "translation_unit_id": tuid,
        "unit_id": f"seg_line_survival_{index:04d}",
        "level": "line",
        "parent_id": None,
        "source_unit_ids": [uid] if uid else [],
        "logical_unit_id": uid,
        "source_text": text,
        "bbox": bbox,
        "reading_order_index": (line.get("geometry") or {}).get("reading_order_index") or index,
        "role": role,
        "object_type": object_type,
        "object_class": (line.get("understanding") or {}).get("object_class"),
        "semantic_kind": (line.get("understanding") or {}).get("semantic_kind"),
        "strategy": "layout_constrained",
        "translation_mode": "translate",
        "render_policy": "anchored_line",
        "coverage_required": "strict",
        "protected": [],
        "translatable": True,
        "context": {},
        "render_target": {
            "reconstruction_unit_id": f"ru_line_survival_{index:04d}",
            "bbox": bbox,
            "layout_bbox": bbox,
            "patch_bbox": bbox,
            "coverage_bbox": bbox,
            "anchor_bbox": bbox,
            "style_source_unit_id": uid,
            "consume_source_unit_ids": [uid] if uid else [],
        },
        "qa_requirements": {"text_presence_required": True},
        "reason_included": "visible_line_missing_from_translation_plan+atomic_text_survival",
    }


def _text_chunks(text: str, *, max_chars: int = MAX_TRANSLATION_CHARS) -> list[str]:
    text = normalize_spaces(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        words = text.split()
        cur, n = [], 0
        for w in words:
            if cur and n + len(w) + 1 > max_chars:
                chunks.append(" ".join(cur))
                cur, n = [], 0
            cur.append(w)
            n += len(w) + 1
        if cur:
            chunks.append(" ".join(cur))
        return chunks
    cur, n = [], 0
    for sentence in sentences:
        if cur and n + len(sentence) + 1 > max_chars:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(sentence)
        n += len(sentence) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _split_item_without_lines(item: dict) -> list[dict]:
    text = normalize_spaces(item.get("source_text") or "")
    chunks = _text_chunks(text)
    if len(chunks) <= 1:
        return [item]
    base_tuid = item.get("translation_unit_id") or item.get("unit_id") or "tp"
    base_uid = item.get("unit_id") or base_tuid
    out = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        ni = copy.deepcopy(item)
        ni.update({
            "translation_unit_id": f"{base_tuid}_chunk_{idx:02d}",
            "unit_id": f"{base_uid}_chunk_{idx:02d}",
            "parent_translation_unit_id": base_tuid,
            "parent_unit_id": base_uid,
            "split_index": idx,
            "split_count": total,
            "source_text": chunk,
            "reason_included": f"{item.get('reason_included') or 'translation_plan'}+atomic_text_chunk_survival",
        })
        out.append(ni)
    return out


def _source_related(a: str, b: str) -> bool:
    return bool(a and b) and (a == b or a.startswith(b + "_") or b.startswith(a + "_"))


def _covered_by_any_line(uid: str, covered: set[str]) -> bool:
    return any(_source_related(uid, cid) for cid in covered)


def split_translation_units_for_text_survival(input_data: dict, units: list[dict]) -> list[dict]:
    """Expand translation units to atomic PAGEPRINT line units.

    Ownership/Lifecycle v1: text survival guarantees visible natural text only.
    A formula/code/protected visual zone is covered by preservation, not by a
    TextOp fallback.  Therefore preserved/excluded source ids are never split
    back into translation units here.
    """
    unit_map = _unit_map(input_data)
    cmap = _children_map(input_data)
    ownership = build_source_ownership(input_data)
    out: list[dict] = []
    covered_line_ids: set[str] = set()

    for item in units or []:
        text = normalize_spaces(item.get("source_text") or "")
        if not text:
            continue
        sids = [sid for sid in item.get("source_unit_ids") or [] if sid]
        if source_ids_have_non_translatable_owner(ownership, sids):
            # Mixed parent blocks should normally be split into their descendant
            # visible lines. If the item itself is already an owned special line,
            # suppress it completely.
            if len(sids) == 1 and is_non_translatable_owner(ownership, sids[0]):
                continue
        if any(_in_hard_special_region(unit_map.get(sid)) for sid in sids):
            continue

        line_units = [ln for ln in _line_units_for_item(item, unit_map, cmap)
                      if not is_non_translatable_owner(ownership, ln.get("unit_id"))]
        if line_units:
            total = len(line_units)
            for idx, line in enumerate(line_units, start=1):
                uid = line.get("unit_id")
                if uid:
                    covered_line_ids.add(uid)
                out.append(_line_item_from_plan(item, line, index=idx, total=total))
            continue

        if not source_ids_have_non_translatable_owner(ownership, sids):
            for ni in _split_item_without_lines(item):
                out.append(ni)

    next_index = len(out) + 1
    for line in _all_visible_lines(input_data):
        uid = line.get("unit_id")
        if not uid or is_non_translatable_owner(ownership, uid) or _covered_by_any_line(uid, covered_line_ids):
            continue
        out.append(_standalone_line_item(line, index=next_index))
        covered_line_ids.add(uid)
        next_index += 1

    deduped: list[dict] = []
    seen_sources: set[str] = set()
    for item in sorted(out, key=lambda it: (it.get("reading_order_index") if it.get("reading_order_index") is not None else 10**9,
                                            (_bbox_of(it) or [0, 0, 0, 0])[1],
                                            (_bbox_of(it) or [0, 0, 0, 0])[0],
                                            str(it.get("translation_unit_id") or ""))):
        sids = [s for s in item.get("source_unit_ids") or [] if s]
        if len(sids) == 1 and sids[0] in seen_sources:
            continue
        deduped.append(item)
        for sid in sids:
            seen_sources.add(sid)
    return deduped


def _is_suspiciously_truncated(item: dict) -> bool:
    src = normalize_spaces(item.get("source_text") or "")
    tgt = normalize_spaces(item.get("translated_text") or "")
    if not src:
        return False
    if not tgt:
        return True
    if len(src) < 70:
        return False
    ratio = len(tgt) / max(1, len(src))
    if ratio < MIN_TRUNCATION_CHAR_RATIO:
        return True
    quality = item.get("quality") or {}
    if quality.get("empty_translation"):
        return True
    trace = item.get("engine_trace") or {}
    if trace.get("truncated") is True:
        return True
    return False


def repair_truncated_translation_units(input_data: dict, translated_units: list[dict]) -> list[dict]:
    """If the engine still returns an obvious empty/truncated line, keep source."""
    repaired = []
    for item in translated_units or []:
        ni = copy.deepcopy(item)
        if _is_suspiciously_truncated(ni):
            src = normalize_spaces(ni.get("source_text") or "")
            tgt = normalize_spaces(ni.get("translated_text") or "")
            if src and src not in tgt:
                ni["translated_text"] = (tgt + "\n" + src).strip() if tgt else src
                ni["status"] = "translated_with_source_survival_fallback"
                ni.setdefault("quality", {})["text_survival_fallback_applied"] = True
                ni.setdefault("quality", {})["needs_review"] = True
                ni.setdefault("quality", {}).setdefault("qa_reasons", []).append("text_survival_fallback_applied")
                ni.setdefault("engine_trace", {})["text_survival_fallback_applied"] = True
        repaired.append(ni)
    return repaired


def _related(a: str, b: str) -> bool:
    return bool(a and b) and (a == b or a.startswith(b + "_") or b.startswith(a + "_"))


def append_uncovered_source_line_fallbacks(translated_input: dict, reconstruction_units: list[dict], unit_map: dict[str, dict]) -> list[dict]:
    """Append identity fallback reconstruction units for still-uncovered lines.

    Ownership/Lifecycle v1: preserved visual/code/formula units must not be
    resurrected as raw text fallbacks.
    """
    ownership = build_source_ownership(translated_input)
    covered: set[str] = set()
    for ru in reconstruction_units or []:
        for sid in ru.get("source_unit_ids") or []:
            if sid:
                covered.add(sid)

    additions = []
    for unit in sorted(unit_map.values(), key=_reading_key):
        uid = unit.get("unit_id")
        if not uid or not _is_visible_text_line(unit):
            continue
        if is_non_translatable_owner(ownership, uid):
            continue
        if any(_related(uid, c) for c in covered):
            continue
        bbox = _bbox_of(unit)
        text = _text_of(unit)
        if not bbox or not text:
            continue
        understanding = unit.get("understanding") or {}
        style = (unit.get("visual") or {}).get("style") or {}
        additions.append({
            "unit_id": f"text_survival_{uid}",
            "translation_unit_id": f"identity_fallback::{uid}",
            "logical_unit_id": uid,
            "level": "line",
            "render_level": "line",
            "role": understanding.get("role") or "body_paragraph",
            "object_type": understanding.get("object_type") or "natural_text",
            "semantic_kind": understanding.get("semantic_kind"),
            "page_role": understanding.get("page_role"),
            "preservation_mode": "text_survival_identity_fallback",
            "text": text,
            "source_text": text,
            "translated_text": text,
            "bbox": bbox,
            "layout_bbox": bbox,
            "patch_bbox": bbox,
            "coverage_bbox": bbox,
            "anchor_bbox": bbox,
            "source_unit_ids": [uid],
            "consume_source_units": True,
            "source_units_consumed": True,
            "preferred_over_children": True,
            "skip_original_units": True,
            "render_as": "line",
            "overflow_policy": "must_render_identity_fallback",
            "line_break_policy": "source_line",
            "layout_budget": {
                "bbox_reliable": True,
                "width": max(0.0, float(bbox[2]) - float(bbox[0])),
                "height": max(0.0, float(bbox[3]) - float(bbox[1])),
                "area": max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1])),
            },
            "style": style,
            "render_target": {
                "bbox": bbox,
                "layout_bbox": bbox,
                "coverage_bbox": bbox,
                "patch_bbox": bbox,
                "anchor_bbox": bbox,
                "style_source_unit_id": uid,
                "consume_source_unit_ids": [uid],
            },
            "render_contract": {
                "mode": "text_survival_identity_fallback",
                "reason": "visible_source_line_without_output",
                "must_render": True,
            },
            "translation": {"status": "identity_fallback", "reason": "visible_source_line_without_output"},
        })
        covered.add(uid)
    if additions:
        translated_input.setdefault("text_survival", {})["identity_fallback_count"] = len(additions)
    return additions
