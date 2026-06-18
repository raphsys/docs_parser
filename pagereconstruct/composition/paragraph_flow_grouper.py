"""Safe render-time paragraph flow grouping, v2.

Atomic lines remain in FinalReconstructionContract for traceability.  At render
execution time only, contiguous prose lines are converted to virtual paragraph
blocks with enough horizontal/vertical room and with preserved visual regions as
hard obstacles.  This is the layout counterpart of Ownership/Lifecycle: formula
or figure zones are not touched, and prose must not be painted in their
interlines.
"""
from __future__ import annotations

from copy import deepcopy
from statistics import median
from typing import Any, Iterable

_FLOW_ROLES = {
    "body_paragraph", "body", "paragraph", "list_item",
    "figure_caption", "figure_caption_text", "table_caption", "table_caption_text",
    "caption", "footnote", "author_bio", "bibliography_entry",
}
_STOP_ROLES = {
    "page_number", "page_reference", "toc_page_reference", "toc_section_number",
    "toc_entry", "toc_entry_title", "toc_entry_page", "toc_leader",
    "section_heading", "subsection_heading", "chapter_heading", "title", "subtitle",
    "formula", "formula_expression", "equation", "code", "code_line", "code_block",
    "table_body_cell", "table_header_cell", "table_numeric_cell",
    "diagram_label", "axis_label", "legend_label", "publisher_mark", "watermark",
    "index_head_term", "index_page_reference",
}


def _bbox(block: Any):
    layout = getattr(block, "layout", None)
    if layout is None:
        return None
    for attr in ("layout_bbox", "coverage_bbox", "source_bbox"):
        b = getattr(layout, attr, None)
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                x0, y0, x1, y1 = [float(x) for x in b]
                if x1 > x0 and y1 > y0:
                    return [x0, y0, x1, y1]
            except Exception:
                pass
    return None


def _union(boxes: Iterable[list[float]]) -> list[float] | None:
    bs = [b for b in boxes if isinstance(b, list) and len(b) == 4]
    if not bs:
        return None
    return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]


def _text(block: Any) -> str:
    return " ".join(str(getattr(block, "translated_text", "") or getattr(block, "source_text", "") or "").split())


def _role(block: Any) -> str:
    return str(getattr(block, "role", "") or "")


def _looks_like_toc_or_index(text: str) -> bool:
    s = str(text or "")
    if "....." in s or " . . ." in s or "…" in s:
        return True
    # A label/title followed by a terminal page number is usually TOC/index; it
    # must stay anchored, not be merged into normal paragraph flow.
    parts = s.rsplit(" ", 1)
    return len(parts) == 2 and parts[1].isdigit() and len(parts[0]) > 8 and len(parts[1]) <= 4


def _eligible(block: Any) -> bool:
    role = _role(block)
    txt = _text(block)
    if role in _STOP_ROLES or role not in _FLOW_ROLES:
        return False
    if not txt or _looks_like_toc_or_index(txt):
        return False
    layout = getattr(block, "layout", None)
    if bool(getattr(layout, "bbox_locked", False)):
        return False
    b = _bbox(block)
    if not b:
        return False
    return (b[2] - b[0]) >= 90.0


def _same_column(a: list[float], b: list[float]) -> bool:
    ax = (a[0] + a[2]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    left_close = abs(a[0] - b[0]) <= 18.0
    return left_close or abs(ax - bx) <= max(42.0, min(a[2] - a[0], b[2] - b[0]) * 0.28)


def _close_vertical(prev: list[float], cur: list[float]) -> bool:
    gap = cur[1] - prev[3]
    prev_h = max(1.0, prev[3] - prev[1])
    cur_h = max(1.0, cur[3] - cur[1])
    return -1.5 <= gap <= max(10.0, min(prev_h, cur_h) * 1.05)


def _page_width(contract: Any) -> float | None:
    pi = getattr(contract, "page_info", None)
    ps = getattr(pi, "page_size", None)
    if isinstance(ps, (list, tuple)) and len(ps) == 2 and ps[0]:
        return float(ps[0])
    return None


def _page_height(contract: Any) -> float | None:
    pi = getattr(contract, "page_info", None)
    ps = getattr(pi, "page_size", None)
    if isinstance(ps, (list, tuple)) and len(ps) == 2 and ps[1]:
        return float(ps[1])
    return None


def _protected_boxes(contract: Any) -> list[list[float]]:
    boxes = []
    pres = getattr(contract, "preservation", None)
    for obj in getattr(pres, "objects", []) or []:
        b = getattr(obj, "bbox", None)
        if isinstance(b, (list, tuple)) and len(b) == 4:
            boxes.append([float(x) for x in b])
    for block in getattr(contract, "blocks", []) or []:
        for r in getattr(block, "protected_regions", []) or []:
            b = r.get("bbox") if isinstance(r, dict) else getattr(r, "bbox", None)
            if isinstance(b, (list, tuple)) and len(b) == 4:
                boxes.append([float(x) for x in b])
    return boxes


def _intersects(a: list[float], b: list[float], pad: float = 1.0) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def _crosses_protected(prev: list[float], cur: list[float], obstacles: list[list[float]]) -> bool:
    band = [min(prev[0], cur[0]), prev[3], max(prev[2], cur[2]), cur[1]]
    if band[3] <= band[1]:
        return False
    return any(_intersects(band, o, pad=0.5) for o in obstacles)


def _expand_right_if_safe(box: list[float], page_w: float | None, group: list[Any], obstacles: list[list[float]]) -> list[float]:
    if not page_w:
        return box
    margin = max(18.0, page_w * 0.035)
    max_x1 = page_w - margin
    if max_x1 <= box[2]:
        return box
    growth_cap = 180.0 if len(group) >= 2 else 90.0
    candidate = [box[0], box[1], min(max_x1, box[2] + growth_cap), box[3]]
    # Horizontal expansion is forbidden if it would hit a preserved object in the
    # same vertical band.
    if any(_intersects(candidate, o, pad=1.0) and not _intersects(box, o, pad=1.0) for o in obstacles):
        return box
    return candidate


def _estimated_group_height(group: list[Any], box: list[float]) -> float:
    boxes = [_bbox(b) for b in group]
    hs = [max(1.0, b[3] - b[1]) for b in boxes if b]
    base = median(hs) if hs else 10.0
    line_h = max(7.0, base * 1.22)
    text = " ".join(_text(b) for b in group)
    width = max(20.0, box[2] - box[0])
    # Approximate wrap lines.  The composer will measure exactly later; this is
    # only to reserve enough vertical space and prevent immediate interline hits.
    avg_char = max(3.0, line_h * 0.44)
    est = max(len(group), int((len(text) * avg_char) // width) + 1)
    return est * line_h + max(2.0, line_h * 0.2)


def _limit_bottom_by_obstacle(box: list[float], obstacles: list[list[float]], page_h: float | None) -> list[float]:
    bottom = box[3]
    for o in obstacles:
        same_col = not (box[2] <= o[0] or o[2] <= box[0])
        if same_col and o[1] > box[1]:
            bottom = min(bottom, o[1] - 2.0)
    if page_h:
        bottom = min(bottom, page_h - 6.0)
    if bottom < box[1] + 6.0:
        bottom = box[3]
    return [box[0], box[1], box[2], bottom]


def _make_group(first: Any, group: list[Any], contract: Any, obstacles: list[list[float]]) -> Any:
    if len(group) <= 1:
        return first
    boxes = [_bbox(b) for b in group]
    ub = _union([b for b in boxes if b])
    if not ub:
        return first
    ub = _expand_right_if_safe(ub, _page_width(contract), group, obstacles)
    needed_h = _estimated_group_height(group, ub)
    if needed_h > (ub[3] - ub[1]):
        ub[3] = ub[1] + needed_h
    ub = _limit_bottom_by_obstacle(ub, obstacles, _page_height(contract))
    vb = deepcopy(first)
    ids = [str(getattr(b, "block_id", "")) for b in group if getattr(b, "block_id", "")]
    sid: list[str] = []
    tuid: list[str] = []
    src_texts: list[str] = []
    tr_texts: list[str] = []
    for b in group:
        sid.extend(list(getattr(b, "source_unit_ids", []) or []))
        if getattr(b, "translation_unit_id", None):
            tuid.append(str(getattr(b, "translation_unit_id")))
        src_texts.append(str(getattr(b, "source_text", "") or ""))
        tr_texts.append(_text(b))
    vb.block_id = "flowgrp_" + (ids[0] if ids else "paragraph")
    vb.source_unit_ids = list(dict.fromkeys(sid))
    vb.translation_unit_id = "group::" + "+".join(tuid[:8]) if tuid else getattr(first, "translation_unit_id", None)
    vb.source_text = " ".join(" ".join(src_texts).split())
    vb.translated_text = " ".join(" ".join(tr_texts).split())
    layout = getattr(vb, "layout", None)
    if layout is not None:
        for attr in ("layout_bbox", "safe_bbox", "overflow_bbox"):
            if hasattr(layout, attr):
                setattr(layout, attr, list(ub))
        if hasattr(layout, "coverage_bbox"):
            setattr(layout, "coverage_bbox", list(ub))
    if hasattr(vb, "findings"):
        vb.findings.append({"type": "render_time_paragraph_flow_group_v2", "members": ids, "bbox": list(ub)})
    return vb


def blocks_for_render(contract: Any) -> list[Any]:
    import os
    blocks = list(getattr(contract, "blocks", []) or [])
    if os.getenv("RECON_DISABLE_PARAGRAPH_FLOW_GROUPING", "").strip().lower() in {"1", "true", "yes"}:
        return blocks
    if not blocks:
        return []
    obstacles = _protected_boxes(contract)
    ordered = sorted(blocks, key=lambda b: ((_bbox(b) or [0, 0, 0, 0])[1], (_bbox(b) or [0, 0, 0, 0])[0]))
    out: list[Any] = []
    i = 0
    while i < len(ordered):
        b = ordered[i]
        if not _eligible(b):
            out.append(b)
            i += 1
            continue
        group = [b]
        last_box = _bbox(b)
        j = i + 1
        while j < len(ordered):
            nb = ordered[j]
            nbx = _bbox(nb)
            if (not _eligible(nb) or not last_box or not nbx or not _same_column(last_box, nbx)
                    or not _close_vertical(last_box, nbx) or _crosses_protected(last_box, nbx, obstacles)):
                break
            group.append(nb)
            last_box = nbx
            j += 1
        out.append(_make_group(group[0], group, contract, obstacles) if len(group) >= 2 else b)
        i = j
    return out
