"""Global source-unit ownership/lifecycle contract.

This module is intentionally dependency-light and shared by PAGEPRINT,
PAGETRANSLATE, PAGERECONSTRUCT and PUBREADY.  It answers one question before any
layout/rendering decision:

    for each PAGEPRINT source_unit_id, who owns the final output?

The states are mutually exclusive.  The important invariant is that a unit owned
by a preserved visual/special zone must not also enter the translation or TextOp
pipeline.  Its final coverage is a PreservationOp, not translated text.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

TEXT_LEVELS = {"block", "line", "phrase", "span", "word", "char"}
SPECIAL_REGION_TYPES = {
    "formula_region",
    "code_region",
    "protected_visual_region",
}
SPECIAL_CANDIDATE_REGION_TYPES = {
    "formula_candidate_region",
    "code_candidate_region",
    "visual_candidate_region",
}
PRESERVED_VISUAL_REASONS = {
    "formula",
    "formula_region",
    "formula_expression",
    "equation",
    "math_expression",
    "code",
    "code_region",
    "code_line",
    "code_block",
    "protected_visual",
    "protected_visual_region",
    "image",
    "figure",
    "diagram",
    "chart",
}
PRESERVED_TEXT_REASONS = {
    "page_number",
    "page_reference",
    "toc_page_reference",
    "toc_section_number",
    "index_page_reference",
    "caption_label",
    "caption_number",
    "list_marker",
    "section_number",
}
EXCLUDED_REASONS = {
    "artifact",
    "publisher_mark",
    "watermark",
    "background_only",
    "exclude_as_artifact",
}
FINAL_TRANSLATION_STATES = {"translated_text", "translation_candidate"}
FINAL_PRESERVATION_STATES = {"preserved_visual", "preserved_text_exact"}
FINAL_EXCLUSION_STATES = {"excluded", "background_only", "background_visual"}
NON_TRANSLATABLE_STATES = FINAL_PRESERVATION_STATES | FINAL_EXCLUSION_STATES


@dataclass
class SourceOwnershipEntry:
    source_unit_id: str
    state: str = "translation_candidate"
    owner: str | None = None
    reason: str | None = None
    bbox: list | None = None
    level: str | None = None
    text: str | None = None
    region_id: str | None = None
    preservation_id: str | None = None
    findings: list[str] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("findings") is None:
            d["findings"] = []
        return d


def text_of(unit: dict | None) -> str:
    if not isinstance(unit, dict):
        return ""
    return str((unit.get("content") or {}).get("text") or unit.get("text") or unit.get("source_text") or "").strip()


def bbox_of(item: dict | None) -> list | None:
    if not isinstance(item, dict):
        return None
    for key in ("bbox", "layout_bbox", "coverage_bbox", "patch_bbox", "anchor_bbox"):
        b = item.get(key)
        if _valid_bbox(b):
            return [float(x) for x in b]
    g = item.get("geometry") or {}
    b = g.get("bbox")
    if _valid_bbox(b):
        return [float(x) for x in b]
    rt = item.get("render_target") or {}
    for key in ("layout_bbox", "coverage_bbox", "patch_bbox", "bbox", "anchor_bbox"):
        b = rt.get(key)
        if _valid_bbox(b):
            return [float(x) for x in b]
    return None


def _valid_bbox(b) -> bool:
    return isinstance(b, (list, tuple)) and len(b) == 4


def _area(b) -> float:
    if not _valid_bbox(b):
        return 0.0
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def _inter_area(a, b) -> float:
    if not (_valid_bbox(a) and _valid_bbox(b)):
        return 0.0
    x0, y0 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x1, y1 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def overlap_ratio(inner, outer) -> float:
    denom = max(1e-6, _area(inner))
    return _inter_area(inner, outer) / denom


def region_coverage_ratio(region_bbox, unit_bbox) -> float:
    denom = max(1e-6, _area(region_bbox))
    return _inter_area(region_bbox, unit_bbox) / denom


def hierarchy(units: list[dict]) -> tuple[dict[str, str], dict[str, list[str]]]:
    parent_by_id: dict[str, str] = {}
    children_by_parent: dict[str, list[str]] = {}
    for u in units or []:
        if not isinstance(u, dict):
            continue
        uid = u.get("unit_id")
        if not uid:
            continue
        parent = u.get("parent_id") or u.get("parent_unit_id")
        if parent:
            parent_by_id[uid] = parent
            children_by_parent.setdefault(parent, []).append(uid)
        for cid in u.get("children_ids") or []:
            parent_by_id.setdefault(cid, uid)
            children_by_parent.setdefault(uid, []).append(cid)
    return parent_by_id, {k: list(dict.fromkeys(v)) for k, v in children_by_parent.items()}


def descendants(uid: str, children_by_parent: dict[str, list[str]]) -> set[str]:
    out: set[str] = set()
    stack = list(children_by_parent.get(uid) or [])
    while stack:
        current = stack.pop(0)
        if current in out:
            continue
        out.add(current)
        stack.extend(children_by_parent.get(current) or [])
    return out


def ancestors(uid: str, parent_by_id: dict[str, str]) -> set[str]:
    out: set[str] = set()
    parent = parent_by_id.get(uid)
    while parent:
        out.add(parent)
        parent = parent_by_id.get(parent)
    return out


def covered_ids(source_ids: Iterable[str], unit_ids: set[str], parent_by_id: dict[str, str],
                children_by_parent: dict[str, list[str]], *, include_ancestors: bool = False,
                include_descendants: bool = True) -> set[str]:
    out: set[str] = set()
    for sid in source_ids or []:
        if sid in unit_ids:
            out.add(sid)
        if include_descendants:
            out |= (descendants(sid, children_by_parent) & unit_ids)
        if include_ancestors:
            out |= (ancestors(sid, parent_by_id) & unit_ids)
    return out


def _unit_level(unit: dict) -> str:
    return str(unit.get("level") or unit.get("unit_level") or unit.get("type") or unit.get("unit_type") or "")


def _ownership_threshold(level: str | None, region_type: str | None) -> float:
    # Blocks are often larger containers; do not preserve them unless the special
    # region dominates the block.  Line/phrase/span and fine tokens can be owned
    # when the special region clearly covers them.
    if level in {"block"}:
        return 0.60
    if level in {"line", "phrase"}:
        return 0.50
    if level in {"span", "word", "char"}:
        return 0.35
    return 0.50


def _special_kind(region: dict) -> str | None:
    rt = str(region.get("region_type") or region.get("object_type") or region.get("role") or "").lower()
    if rt in SPECIAL_REGION_TYPES or rt in SPECIAL_CANDIDATE_REGION_TYPES:
        if "formula" in rt or "equation" in rt or "math" in rt:
            return "formula"
        if "code" in rt:
            return "code"
        return "protected_visual"
    if "formula" in rt or "equation" in rt or "math" in rt:
        return "formula"
    if "code" in rt:
        return "code"
    if "protected_visual" in rt:
        return "protected_visual"
    return None


def hard_special_regions(data: dict, *, include_candidates: bool = True) -> list[dict]:
    out = []
    seen = set()
    for r in data.get("regions") or []:
        if not isinstance(r, dict):
            continue
        rt = str(r.get("region_type") or "").lower()
        if rt not in SPECIAL_REGION_TYPES and not (include_candidates and rt in SPECIAL_CANDIDATE_REGION_TYPES):
            continue
        b = bbox_of(r)
        if not b:
            continue
        kind = _special_kind(r) or "protected_visual"
        key = (tuple(round(float(x), 1) for x in b), kind, rt)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "region_id": r.get("region_id") or f"special_{len(out)+1:04d}",
            "region_type": rt,
            "kind": kind,
            "bbox": b,
            "confidence": float(r.get("confidence") or 0.0),
            "source": r.get("source") or r.get("detection_source") or "pageprint_region",
            "members": r.get("members") or {},
        })
    return out


def _set_state(entry: SourceOwnershipEntry, *, state: str, owner: str, reason: str,
               region_id: str | None = None, preservation_id: str | None = None) -> None:
    priority = {
        "translation_candidate": 0,
        "excluded": 1,
        "background_only": 1,
        "preserved_text_exact": 2,
        "preserved_visual": 3,
        "translated_text": 4,
    }
    # Translation is assigned later by PAGERECONSTRUCT.  Do not let it overwrite
    # preservation/exclusion; those are upstream ownership decisions.
    if state == "translated_text" and entry.state in NON_TRANSLATABLE_STATES:
        entry.findings = list(entry.findings or []) + ["translation_conflicts_with_non_translatable_owner"]
        return
    if priority.get(state, 0) >= priority.get(entry.state, 0):
        entry.state = state
        entry.owner = owner
        entry.reason = reason
        entry.region_id = region_id or entry.region_id
        entry.preservation_id = preservation_id or entry.preservation_id


def build_source_ownership(data: dict) -> dict[str, dict]:
    units = [u for u in data.get("units") or [] if isinstance(u, dict) and u.get("unit_id")]
    unit_map = {u["unit_id"]: u for u in units}
    unit_ids = set(unit_map)
    parent_by_id, children_by_parent = hierarchy(units)
    entries: dict[str, SourceOwnershipEntry] = {}
    for uid, u in unit_map.items():
        level = _unit_level(u)
        txt = text_of(u)
        textual_container = level in TEXT_LEVELS
        default_state = "translation_candidate" if textual_container else "background_visual"
        entries[uid] = SourceOwnershipEntry(
            source_unit_id=uid,
            state=default_state,
            owner="pageprint_text" if textual_container else "non_text_or_container",
            reason="visible_text_default" if txt else "textual_container_default" if textual_container else "empty_or_non_text_unit",
            bbox=bbox_of(u),
            level=level,
            text=txt or None,
            findings=[],
        )

    # Preservation plan has first-class ownership authority.
    preservation_plan = data.get("preservation_plan") or ((data.get("views") or {}).get("preservation_plan") or [])
    for p in preservation_plan or []:
        if not isinstance(p, dict):
            continue
        reason = str(p.get("reason") or p.get("role") or p.get("preservation_mode") or "preserve").lower()
        mode = str(p.get("preservation_mode") or "").lower()
        state = "preserved_text_exact" if mode == "preserve_text_exactly" or reason in PRESERVED_TEXT_REASONS else "preserved_visual"
        # Exact-text preservation entries (page refs, labels, markers) are often
        # attached to a broader logical row.  They must not steal ownership of
        # every descendant/title in that row.  Visual preservation (formula/code)
        # does own descendants.
        sids = covered_ids(p.get("source_unit_ids") or [], unit_ids, parent_by_id, children_by_parent,
                           include_descendants=(state == "preserved_visual"), include_ancestors=False)
        preserve_text = str(p.get("text") or "").strip()
        for sid in sids:
            # Page refs/caption labels are frequently extracted from a wider TOC
            # or caption line.  If the preserved text is only a substring of the
            # unit text, it does not own the whole unit; the natural text portion
            # must still be translated.  Exact text ownership applies only when
            # the preserved item matches the source unit text, or the unit is
            # already typed as page furniture.
            if state == "preserved_text_exact":
                utext = text_of(unit_map.get(sid))
                urole = str(((unit_map.get(sid) or {}).get("understanding") or {}).get("role") or "").lower()
                if preserve_text and utext and preserve_text != utext and urole not in PRESERVED_TEXT_REASONS:
                    continue
            _set_state(entries[sid], state=state, owner="preservation_plan", reason=reason,
                       preservation_id=p.get("preservation_id") or p.get("id"))

    # Exclusion plan owns artifacts/page furniture that should not become text.
    exclusion_plan = data.get("exclusion_plan") or ((data.get("views") or {}).get("exclusion_plan") or [])
    for e in exclusion_plan or []:
        if not isinstance(e, dict):
            continue
        reason = str(e.get("reason") or "excluded").lower()
        state = "background_only" if reason == "background_only" else "excluded"
        sids = covered_ids(e.get("source_unit_ids") or [], unit_ids, parent_by_id, children_by_parent,
                           include_descendants=True, include_ancestors=False)
        for sid in sids:
            _set_state(entries[sid], state=state, owner="exclusion_plan", reason=reason)

    # Hard special regions own all units clearly inside them.  This catches
    # formula/code regions even when PAGEPRINT still leaves line/phrase units in
    # units[].  The region is the source of truth for display formulas.
    for r in hard_special_regions(data, include_candidates=False):
        rb = r["bbox"]
        member_ids = []
        for key in ("block_ids", "line_ids", "phrase_ids", "span_ids", "word_ids", "char_ids"):
            member_ids.extend([sid for sid in (r.get("members") or {}).get(key) or [] if sid in unit_ids])
        for uid, u in unit_map.items():
            ub = bbox_of(u)
            if not ub:
                continue
            level = _unit_level(u)
            unit_cover = overlap_ratio(ub, rb)
            if uid not in member_ids and unit_cover < _ownership_threshold(level, r.get("region_type")):
                continue
            _set_state(entries[uid], state="preserved_visual", owner="special_region", reason=r.get("kind") or "protected_visual", region_id=r.get("region_id"))

    return {uid: e.to_dict() for uid, e in entries.items()}


def ownership_state(ownership: dict[str, dict], source_unit_id: str | None) -> str | None:
    if not source_unit_id:
        return None
    entry = ownership.get(source_unit_id) or {}
    return entry.get("state")


def is_non_translatable_owner(ownership: dict[str, dict], source_unit_id: str | None) -> bool:
    return ownership_state(ownership, source_unit_id) in NON_TRANSLATABLE_STATES


def source_ids_have_non_translatable_owner(ownership: dict[str, dict], source_ids: Iterable[str]) -> bool:
    return any(is_non_translatable_owner(ownership, sid) for sid in source_ids or [])


def all_source_ids_non_translatable(ownership: dict[str, dict], source_ids: Iterable[str]) -> bool:
    sids = [sid for sid in source_ids or [] if sid]
    return bool(sids) and all(is_non_translatable_owner(ownership, sid) for sid in sids)


def filter_translation_units_by_ownership(input_data: dict, items: list[dict]) -> list[dict]:
    ownership = build_source_ownership(input_data)
    out = []
    for item in items or []:
        sids = [sid for sid in item.get("source_unit_ids") or [] if sid]
        if all_source_ids_non_translatable(ownership, sids):
            continue
        # If the item is directly sourced from a preserved formula/code line,
        # suppress it. Mixed parent blocks survive; text_survival will split them
        # down to the surviving visible lines.
        if len(sids) == 1 and is_non_translatable_owner(ownership, sids[0]):
            continue
        out.append(item)
    return out


def annotate_input_data_ownership(data: dict) -> dict:
    ownership = build_source_ownership(data)
    data.setdefault("views", {})["source_ownership"] = ownership
    for u in data.get("units") or []:
        uid = u.get("unit_id") if isinstance(u, dict) else None
        if uid and uid in ownership:
            u["source_ownership"] = ownership[uid]
    return data


def audit_source_ownership(plan: dict | None, data: dict) -> dict:
    ownership = build_source_ownership(data)
    translated = set()
    preserved = set()
    render_ops = (plan or {}).get("render_ops") or []
    layers = (plan or {}).get("layers") or {}
    for item in layers.get("translated_text") or []:
        translated.update(item.get("source_unit_ids") or [])
    for op in render_ops:
        if op.get("op_type") == "text":
            translated.update(op.get("source_unit_ids") or [])
        elif op.get("op_type") == "preservation":
            preserved.update(op.get("source_unit_ids") or [])
    conflicts = []
    for sid, entry in ownership.items():
        state = entry.get("state")
        if state in FINAL_PRESERVATION_STATES and sid in translated:
            conflicts.append({"source_unit_id": sid, "type": "preserved_source_translated", "state": state})
        if state == "translation_candidate" and sid not in translated and sid not in preserved:
            # Containers may be covered by children; leave detailed text coverage to
            # source_text_lifecycle_ledger.  This audit is about exclusivity.
            pass
    return {
        "status": "ko" if conflicts else "ok",
        "hard_blockers": sorted({c["type"] for c in conflicts}),
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "ownership": ownership,
    }
