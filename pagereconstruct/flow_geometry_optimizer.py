"""Geometry flow optimizer for translated page reconstruction.

This is the actual correction for the visual geometry problem:
    - recompute translated block height;
    - expand available width when safe;
    - group atomic line blocks into paragraph flows;
    - cascade adjacent blocks only when needed;
    - avoid protected figures/tables/formulas/code;
    - never delete text.

The module is CPU-only and deterministic, but model-aware through
ai_layout_advisor.py. It uses the local ai_models inventory as policy hints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict

from .multiblock_layout_solver import (
    BBox, LayoutPatch, MultiBlockSolveResult, FlowRegion,
    _bbox, _block_bbox, _block_id, _role, _is_locked, _height,
    _overlap_ratio, collect_protected_obstacles,
)
from .text_measure import measure_block
from .ai_layout_advisor import build_layout_policy_hint

_LOCKED_ROLES = {
    "formula", "formula_expression", "equation",
    "code", "code_block", "code_line",
    "table_body_cell", "table_header_cell", "table_numeric_cell",
    "diagram_label", "diagram_text_label", "axis_label", "legend_label",
    "page_number", "page_reference", "toc_page_reference",
    "caption_label", "caption_number", "publisher_mark", "watermark",
}
_FLOW_ROLES = {
    "body_paragraph", "body", "paragraph", "list_item",
    "caption", "figure_caption", "figure_caption_text",
    "table_caption", "table_caption_text",
    "footnote", "section_heading", "subsection_heading", "chapter_heading",
    "title", "subtitle", "author_bio", "bibliography_entry", "index_entry",
}


def _page_size(contract: Any) -> tuple[float, float]:
    page = getattr(contract, "page", None) or getattr(contract, "page_info", None)
    ps = getattr(page, "page_size", None)
    if isinstance(ps, (list, tuple)) and len(ps) == 2 and ps[0] and ps[1]:
        return float(ps[0]), float(ps[1])
    w = float(getattr(page, "width_pt", 0.0) or getattr(page, "width", 0.0) or 0.0)
    h = float(getattr(page, "height_pt", 0.0) or getattr(page, "height", 0.0) or 0.0)
    if w and h:
        return w, h
    boxes = [_block_bbox(b) for b in getattr(contract, "blocks", []) or []]
    boxes = [b for b in boxes if b]
    if boxes:
        return max(b[2] for b in boxes) + 24.0, max(b[3] for b in boxes) + 24.0
    return 595.0, 842.0


def _style_for_measure(block: Any) -> dict:
    st = getattr(block, "style", None)
    if st is None:
        return {"font_size_pt": 10.0, "flags": {}, "alignment": "left"}
    flags = {
        "bold": bool(getattr(st, "bold", False)),
        "italic": bool(getattr(st, "italic", False)),
        "monospace": getattr(st, "font_class", "") == "mono",
        "serif": getattr(st, "font_class", "serif") == "serif",
    }
    try:
        from .composition.intrablock_composer import _clamp_render_size
        size = _clamp_render_size(getattr(st, "font_size_pt", None) or 10.0, getattr(block, "role", ""))
    except Exception:
        size = float(getattr(st, "font_size_pt", None) or 10.0)
    return {
        "font_size_pt": size,
        "flags": flags,
        "color": getattr(st, "color", "#000000"),
        "alignment": getattr(st, "alignment", "left"),
    }


def _block_text(block: Any) -> str:
    return " ".join(str(getattr(block, "translated_text", "") or getattr(block, "source_text", "") or "").split())


def _measure_required_height(block: Any, bbox: BBox, *, min_ratio: float = 0.92) -> tuple[float, int, dict]:
    text = _block_text(block)
    if not text:
        return _height(bbox), 0, {}
    style = _style_for_measure(block)
    align = style.get("alignment") if style.get("alignment") in {"left", "center", "right"} else "left"
    tall = [bbox[0], bbox[1], bbox[2], bbox[1] + 100000.0]
    try:
        lay = measure_block(text, tall, style, align=align, min_ratio=min_ratio)
    except Exception:
        # crude fallback
        avg = max(2.5, float(style.get("font_size_pt", 10.0)) * 0.50)
        chars = max(8, int((bbox[2] - bbox[0]) / avg))
        n = max(1, (len(text) + chars - 1) // chars)
        line_h = float(style.get("font_size_pt", 10.0)) * 1.25
        return max(_height(bbox), n * line_h), n, {}
    lines = lay.get("lines") or []
    line_h = float(lay.get("line_h") or (style.get("font_size_pt", 10.0) * 1.25))
    n = max(1, len(lines))
    return max(_height(bbox), n * line_h), n, lay


def _needed_height(block: Any, bbox: BBox) -> float:
    # Compatibility with existing tests/imports.
    return _measure_required_height(block, bbox)[0]


def _is_flow_block(block: Any) -> bool:
    role = _role(block)
    if role in _LOCKED_ROLES or _is_locked(block):
        return False
    b = _block_bbox(block)
    if not b or not _block_text(block):
        return False
    if role in _FLOW_ROLES:
        return True
    # Atomic fallback lines often arrive as unknown but have translated text.
    return (b[2] - b[0]) >= 45.0


def _source_parent_key(block: Any) -> str:
    bid = _block_id(block)
    tuid = str(getattr(block, "translation_unit_id", "") or "")
    for marker in ("_line_", "_chunk_"):
        if marker in tuid:
            return "tuid:" + tuid.split(marker)[0]
    sids = list(getattr(block, "source_unit_ids", []) or [])
    if sids:
        sid = str(sids[0])
        for marker in ("_line_", "_phrase_", "_span_", "_word_", "_char_"):
            if marker in sid:
                return "src:" + sid.split(marker)[0]
        return "src:" + sid
    return "blk:" + bid


def _same_column(a: BBox, b: BBox) -> bool:
    ac = (a[0] + a[2]) / 2.0
    bc = (b[0] + b[2]) / 2.0
    return abs(ac - bc) <= max(36.0, min(a[2] - a[0], b[2] - b[0]) * 0.35)


def _groups(blocks: list[Any]) -> list[list[Any]]:
    blocks = [b for b in blocks if _block_bbox(b)]
    blocks.sort(key=lambda b: (_block_bbox(b)[1], _block_bbox(b)[0]))
    raw: list[list[Any]] = []
    current: list[Any] = []
    last_key = None
    last_box = None
    for b in blocks:
        key = _source_parent_key(b)
        bb = _block_bbox(b)
        close = last_box is not None and bb[1] - last_box[3] <= 18.0 and _same_column(bb, last_box)
        if current and (key == last_key or close):
            current.append(b)
        else:
            if current:
                raw.append(current)
            current = [b]
        last_key, last_box = key, bb
    if current:
        raw.append(current)
    return raw


def _union_box(boxes: list[BBox]) -> BBox:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _h_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] <= b[0] + 1.0 or a[0] >= b[2] - 1.0)


def _v_overlap(a: BBox, b: BBox) -> bool:
    return not (a[3] <= b[1] + 1.0 or a[1] >= b[3] - 1.0)


def _obstacle_boxes(contract: Any) -> list[BBox]:
    obs = list(collect_protected_obstacles(contract) or [])
    for b in getattr(contract, "blocks", []) or []:
        if _role(b) in _LOCKED_ROLES or _is_locked(b):
            bb = _block_bbox(b)
            if bb:
                obs.append(bb)
    # Deduplicate coarse.
    out = []
    for b in obs:
        if not any(_overlap_ratio(b, o) > 0.80 for o in out):
            out.append(b)
    return out


def _available_width(group_box: BBox, page_w: float, obstacles: list[BBox], hint) -> tuple[float, float]:
    margin = 24.0
    x0, _, x1, _ = group_box
    max_x1 = min(page_w - margin, x1 + float(hint.max_width_growth_pt or 0.0))
    # Do not grow into obstacles that vertically intersect the group and sit to the right.
    for ob in obstacles:
        if _v_overlap(group_box, ob) and ob[0] >= x1 - 2.0:
            max_x1 = min(max_x1, ob[0] - 4.0)
    if max_x1 < x1:
        max_x1 = x1
    return x0, max_x1


def _jump_below_obstacles(box: BBox, obstacles: list[BBox], *, gap: float, page_h: float) -> BBox:
    out = box
    changed = True
    while changed:
        changed = False
        for ob in sorted(obstacles, key=lambda x: x[1]):
            if _h_overlap(out, ob) and _v_overlap(out, ob):
                h = out[3] - out[1]
                ny0 = ob[3] + gap
                if ny0 + h <= page_h - 6.0:
                    out = (out[0], ny0, out[2], ny0 + h)
                    changed = True
                    break
    return out


def _patch(block: Any, old: BBox, new: BBox, strategy: str, findings: list[dict]) -> LayoutPatch:
    return LayoutPatch(_block_id(block), old, new, strategy, round(new[1] - old[1], 3), findings)


def solve_flow_geometry(contract: Any, *, normalized: dict | None = None, enabled: bool = True) -> MultiBlockSolveResult:
    if not enabled:
        return MultiBlockSolveResult("skipped", {}, [], [{"type": "flow_geometry_disabled"}])

    hint = build_layout_policy_hint(contract, normalized)
    page_w, page_h = _page_size(contract)
    flow_blocks = [b for b in getattr(contract, "blocks", []) or [] if _is_flow_block(b)]
    if not flow_blocks:
        return MultiBlockSolveResult("skipped", {}, [], [{"type": "flow_geometry_no_flow_blocks"}])

    obstacles = _obstacle_boxes(contract)
    patches: Dict[str, LayoutPatch] = {}
    findings: list[dict] = [{
        "type": "ai_layout_policy_hint",
        "severity": "info",
        "policy": hint.to_dict(),
    }]

    min_gap = float(hint.min_gap_pt or 2.0)
    para_gap = float(hint.para_gap_pt or 3.0)
    heading_gap = float(hint.heading_gap_pt or 4.5)

    # Split by columns first; prevents left/right independent flows from pushing
    # each other.
    columns: list[list[Any]] = []
    for b in sorted(flow_blocks, key=lambda x: (_block_bbox(x)[0], _block_bbox(x)[1])):
        bb = _block_bbox(b)
        placed = False
        for col in columns:
            cb = _block_bbox(col[0])
            if cb and _same_column(bb, cb):
                col.append(b)
                placed = True
                break
        if not placed:
            columns.append([b])

    for col in columns:
        col.sort(key=lambda b: (_block_bbox(b)[1], _block_bbox(b)[0]))
        cursor_y: float | None = None

        for group in _groups(col):
            group_boxes = [_block_bbox(b) for b in group if _block_bbox(b)]
            if not group_boxes:
                continue
            old_group = _union_box(group_boxes)
            gx0, gx1 = _available_width(old_group, page_w, obstacles, hint)
            # Keep individual indent but allow more right-side space.
            req_items = []
            total_h = 0.0
            for b in group:
                old = _block_bbox(b)
                if not old:
                    continue
                nb_for_measure = (old[0], old[1], max(old[2], gx1), old[3])
                req_h, n_lines, lay = _measure_required_height(b, nb_for_measure)
                req_h = max(req_h, old[3] - old[1])
                req_items.append((b, old, req_h, n_lines))
                total_h += req_h
            if not req_items:
                continue
            total_h += min_gap * max(0, len(req_items) - 1)

            gap_before = para_gap
            if any(_role(b) in {"section_heading", "subsection_heading", "chapter_heading", "title"} for b in group):
                gap_before = heading_gap

            if cursor_y is None:
                start_y = old_group[1]
            else:
                original_gap = old_group[1] - cursor_y
                start_y = max(old_group[1], cursor_y + min(max(original_gap, gap_before), max(gap_before, 9.0)))

            candidate_group = (gx0, start_y, gx1, start_y + total_h)
            candidate_group = _jump_below_obstacles(candidate_group, obstacles, gap=para_gap, page_h=page_h)
            start_y = candidate_group[1]

            # Page bottom protection: if we cannot fit everything, compress gaps
            # first, then leave a review finding. Text is still not dropped.
            bottom_limit = page_h - 8.0
            if start_y + total_h > bottom_limit:
                available = max(1.0, bottom_limit - start_y)
                if available >= sum(x[2] for x in req_items):
                    min_gap_eff = max(0.2, (available - sum(x[2] for x in req_items)) / max(1, len(req_items) - 1))
                else:
                    min_gap_eff = 0.2
                    findings.append({
                        "type": "flow_geometry_bottom_pressure",
                        "severity": "review",
                        "group_top": old_group[1],
                        "available_height": round(available, 2),
                        "required_height": round(total_h, 2),
                    })
            else:
                min_gap_eff = min_gap

            y = start_y
            for b, old, req_h, n_lines in req_items:
                # Preserve left indent but use widened right edge when possible.
                nx0 = old[0]
                nx1 = max(old[2], gx1)
                new = (nx0, y, nx1, y + req_h)
                # If the new block itself hits a protected visual object, jump it.
                new = _jump_below_obstacles(new, obstacles, gap=para_gap, page_h=page_h)
                changed = abs(new[0] - old[0]) > 0.5 or abs(new[1] - old[1]) > 0.5 or abs(new[2] - old[2]) > 0.5 or abs(new[3] - old[3]) > 0.5
                if changed:
                    strategy = "ai_model_aware_width_expand_vertical_reflow"
                    if new[2] <= old[2] + 0.5:
                        strategy = "ai_model_aware_vertical_reflow"
                    findings_patch = [{
                        "type": "flow_geometry_patch",
                        "block_id": _block_id(b),
                        "old_bbox": list(old),
                        "new_bbox": list(new),
                        "wrapped_lines": n_lines,
                        "strategy": strategy,
                    }]
                    patches[_block_id(b)] = _patch(b, old, new, strategy, findings_patch)
                y = new[3] + min_gap_eff
            cursor_y = max(cursor_y or 0.0, y)

    regions = [FlowRegion("ai_model_aware_page_flow", (0.0, 0.0, page_w, page_h),
                          [_block_id(b) for b in flow_blocks], obstacles,
                          "ai_model_aware_vertical_flow")]
    status = "review" if patches else "skipped"
    return MultiBlockSolveResult(status, patches, regions, findings)


def apply_flow_geometry_patches_in_place(contract: Any, result: MultiBlockSolveResult) -> Any:
    for block in getattr(contract, "blocks", []) or []:
        bid = _block_id(block)
        patch = result.patches_by_block_id.get(bid)
        if not patch:
            continue
        layout = getattr(block, "layout", None)
        if layout is None:
            continue
        nb = [float(x) for x in patch.new_bbox]
        if hasattr(layout, "layout_bbox"):
            setattr(layout, "layout_bbox", nb)
        if hasattr(layout, "safe_bbox"):
            setattr(layout, "safe_bbox", nb)
        if hasattr(layout, "overflow_bbox"):
            setattr(layout, "overflow_bbox", nb)
    if hasattr(contract, "findings"):
        contract.findings.extend(result.findings)
        for p in result.patches_by_block_id.values():
            contract.findings.extend(p.findings)
    return contract
