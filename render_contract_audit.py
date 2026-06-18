"""Render contract propagation audit for Ownership/Lifecycle v2.

This module verifies the second half of the ownership contract:

    source ownership says preserved_visual
    -> the reconstruction/render contract actually treats it as preserved visual

It is deliberately geometry-aware.  Real pages often preserve a formula through a
single region-level PreservationOp while several line/span/word units inside the
region carry preserved_visual ownership.  Therefore an item can satisfy the
contract either by direct source_unit_id membership or by bbox coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable

from source_ownership import (
    build_source_ownership,
    bbox_of,
    overlap_ratio,
    region_coverage_ratio,
)

PRESERVED_VISUAL_STATE = "preserved_visual"
DEFAULT_COVERAGE_THRESHOLD = 0.72
DEFAULT_PATCH_OVERLAP_THRESHOLD = 0.12


@dataclass
class RenderContractRow:
    source_unit_id: str
    state: str
    level: str | None = None
    reason: str | None = None
    bbox: list | None = None
    in_translation_units: bool = False
    in_translated_text_layer: bool = False
    in_text_ops: bool = False
    in_protected_regions: bool = False
    in_preserved_layers: bool = False
    in_preservation_ops: bool = False
    covered_by_patch: bool = False
    status: str = "ok"
    blockers: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _valid_bbox(b) -> bool:
    return isinstance(b, (list, tuple)) and len(b) == 4


def _bbox_key(b) -> tuple | None:
    if not _valid_bbox(b):
        return None
    return tuple(round(float(x), 2) for x in b)


def _collect_source_ids(items: Iterable[dict], *, op_type: str | None = None) -> set[str]:
    out: set[str] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if op_type is not None and item.get("op_type") != op_type:
            continue
        out.update(str(s) for s in (item.get("source_unit_ids") or []) if s)
    return out


def _collect_items(items: Iterable[dict], *, op_type: str | None = None) -> list[dict]:
    out = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if op_type is not None and item.get("op_type") != op_type:
            continue
        out.append(item)
    return out


def _collect_boxes(items: Iterable[dict]) -> list[list]:
    boxes = []
    for item in items or []:
        b = bbox_of(item)
        if b:
            boxes.append(b)
    return boxes


def _covered_by_any_bbox(unit_bbox, boxes: Iterable[list], *, threshold: float = DEFAULT_COVERAGE_THRESHOLD) -> bool:
    if not _valid_bbox(unit_bbox):
        return False
    for b in boxes or []:
        if not _valid_bbox(b):
            continue
        # Unit coverage handles normal case; region coverage handles equal/tiny
        # bboxes and protects from degenerate formulas that have oversized unit
        # boxes around a smaller preservation bbox.
        if overlap_ratio(unit_bbox, b) >= threshold:
            return True
        if region_coverage_ratio(b, unit_bbox) >= 0.92:
            return True
    return False


def _overlap_any(unit_bbox, boxes: Iterable[list], *, threshold: float = DEFAULT_PATCH_OVERLAP_THRESHOLD) -> bool:
    if not _valid_bbox(unit_bbox):
        return False
    for b in boxes or []:
        if _valid_bbox(b) and overlap_ratio(unit_bbox, b) >= threshold:
            return True
    return False


def _translation_units_from_data(data: dict) -> list[dict]:
    out: list[dict] = []
    for key in ("translated_units", "translation_units"):
        vals = data.get(key) or []
        if isinstance(vals, list):
            out.extend(v for v in vals if isinstance(v, dict))
    tr = data.get("translation_result") or {}
    for key in ("translated_units", "translation_units"):
        vals = tr.get(key) or []
        if isinstance(vals, list):
            out.extend(v for v in vals if isinstance(v, dict))
    return out


def _layers(plan: dict) -> dict:
    return plan.get("layers") or {}


def _render_ops(plan: dict) -> list[dict]:
    return [op for op in (plan.get("render_ops") or []) if isinstance(op, dict)]


def _protected_region_items(plan: dict) -> list[dict]:
    return _collect_items(plan.get("protected_regions") or [])


def _preserved_layer_items(plan: dict) -> list[dict]:
    layers = _layers(plan)
    return _collect_items(layers.get("preserved_underlays") or []) + _collect_items(layers.get("preserved_overlays") or [])


def _patch_items(plan: dict) -> list[dict]:
    return _collect_items((_layers(plan)).get("patches") or [])


def _patch_is_destructive_for_unit(unit_bbox, patch: dict) -> bool:
    b = bbox_of(patch)
    if not _valid_bbox(unit_bbox) or not _valid_bbox(b):
        return False
    # plan_patches may already compute protected_overlap_ratio.  A high ratio
    # means the patch was recognised as protected and normally skipped by
    # render_ops.  Still, if it remains in the plan and overlaps a preserved unit,
    # report it unless the patch is explicitly non-destructive.
    reason = str(patch.get("reason") or patch.get("method") or "").lower()
    if "debug" in reason or "non_destructive" in reason:
        return False
    return overlap_ratio(unit_bbox, b) >= DEFAULT_PATCH_OVERLAP_THRESHOLD


def audit_render_contract(plan: dict, data: dict, *, coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD) -> dict:
    """Return a hard audit of preserved_visual propagation to render contract.

    Parameters
    ----------
    plan:
        PAGERECONSTRUCT plan dict, normally containing layers, protected_regions
        and render_ops.
    data:
        PAGEPRINT/translated input_data dict containing units and regions.
    """
    plan = plan or {}
    data = data or {}
    ownership = build_source_ownership(data)
    preserved_visual = {sid: e for sid, e in ownership.items() if e.get("state") == PRESERVED_VISUAL_STATE}

    translation_units = _translation_units_from_data(data)
    translated_layer = _collect_items((_layers(plan)).get("translated_text") or [])
    render_ops = _render_ops(plan)
    text_ops = _collect_items(render_ops, op_type="text")
    preservation_ops = _collect_items(render_ops, op_type="preservation")
    protected_regions = _protected_region_items(plan)
    preserved_layers = _preserved_layer_items(plan)
    patches = _patch_items(plan)

    ids_in_translation_units = _collect_source_ids(translation_units)
    ids_in_translated_layer = _collect_source_ids(translated_layer)
    ids_in_text_ops = _collect_source_ids(text_ops, op_type=None)
    ids_in_preservation_ops = _collect_source_ids(preservation_ops)
    ids_in_protected_regions = _collect_source_ids(protected_regions)
    ids_in_preserved_layers = _collect_source_ids(preserved_layers)

    protected_boxes = _collect_boxes(protected_regions)
    preserved_layer_boxes = _collect_boxes(preserved_layers)
    preservation_op_boxes = _collect_boxes(preservation_ops)

    rows: list[RenderContractRow] = []
    hard_blockers: list[str] = []

    render_ops_missing = bool(preserved_visual) and not render_ops
    if render_ops_missing:
        hard_blockers.append("render_ops_missing")

    for sid, entry in sorted(preserved_visual.items()):
        ub = entry.get("bbox")
        in_translation_units = sid in ids_in_translation_units
        in_translated_layer = sid in ids_in_translated_layer
        in_text_ops = sid in ids_in_text_ops
        in_protected_regions = sid in ids_in_protected_regions or _covered_by_any_bbox(ub, protected_boxes, threshold=coverage_threshold)
        in_preserved_layers = sid in ids_in_preserved_layers or _covered_by_any_bbox(ub, preserved_layer_boxes, threshold=coverage_threshold)
        in_preservation_ops = sid in ids_in_preservation_ops or _covered_by_any_bbox(ub, preservation_op_boxes, threshold=coverage_threshold)
        covered_by_patch = any(_patch_is_destructive_for_unit(ub, p) for p in patches)

        blockers: list[str] = []
        if in_translation_units:
            blockers.append("preserved_visual_in_translation_units")
        if in_translated_layer:
            blockers.append("preserved_visual_in_translated_text_layer")
        if in_text_ops:
            blockers.append("preserved_visual_as_textop")
        if not in_protected_regions:
            blockers.append("preserved_visual_missing_protected_region")
        if not in_preserved_layers:
            blockers.append("preserved_visual_missing_preserved_layer")
        if not in_preservation_ops:
            blockers.append("preserved_visual_missing_preservationop")
        if covered_by_patch:
            blockers.append("preserved_visual_covered_by_patch")
        if render_ops_missing:
            blockers.append("render_ops_missing")

        hard_blockers.extend(blockers)
        rows.append(RenderContractRow(
            source_unit_id=sid,
            state=PRESERVED_VISUAL_STATE,
            level=entry.get("level"),
            reason=entry.get("reason"),
            bbox=ub,
            in_translation_units=in_translation_units,
            in_translated_text_layer=in_translated_layer,
            in_text_ops=in_text_ops,
            in_protected_regions=in_protected_regions,
            in_preserved_layers=in_preserved_layers,
            in_preservation_ops=in_preservation_ops,
            covered_by_patch=covered_by_patch,
            status="ko" if blockers else "ok",
            blockers=blockers,
            evidence={
                "bbox_key": _bbox_key(ub),
                "region_id": entry.get("region_id"),
                "owner": entry.get("owner"),
            },
        ))

    hard_blockers_sorted = sorted(set(hard_blockers))
    ko_rows = [r.to_dict() for r in rows if r.status != "ok"]
    ok_rows = [r.to_dict() for r in rows if r.status == "ok"]
    return {
        "schema_version": "render_contract_audit.v2",
        "status": "ko" if hard_blockers_sorted else "ok",
        "preserved_visual_count": len(preserved_visual),
        "checked_count": len(rows),
        "ok_count": len(ok_rows),
        "ko_count": len(ko_rows),
        "hard_blockers": hard_blockers_sorted,
        "rows": [r.to_dict() for r in rows],
        "ko_rows": ko_rows,
        "summary": {
            "translation_units_source_id_count": len(ids_in_translation_units),
            "translated_layer_source_id_count": len(ids_in_translated_layer),
            "textop_source_id_count": len(ids_in_text_ops),
            "protected_region_count": len(protected_regions),
            "preserved_layer_count": len(preserved_layers),
            "preservationop_count": len(preservation_ops),
            "patch_count": len(patches),
        },
    }


def compact_render_contract_audit(audit: dict, *, max_rows: int = 25) -> dict:
    """Return a compact object suitable for render_policy and CLI summaries."""
    return {
        "schema_version": audit.get("schema_version"),
        "status": audit.get("status"),
        "preserved_visual_count": audit.get("preserved_visual_count", 0),
        "checked_count": audit.get("checked_count", 0),
        "ko_count": audit.get("ko_count", 0),
        "hard_blockers": audit.get("hard_blockers") or [],
        "ko_rows_sample": (audit.get("ko_rows") or [])[:max_rows],
        "summary": audit.get("summary") or {},
    }
