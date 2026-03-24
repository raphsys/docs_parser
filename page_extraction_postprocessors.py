import copy
import re

import fitz


def _rect_from_bbox(b):
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return fitz.Rect(0, 0, 0, 0)
    return fitz.Rect(float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def _bbox_from_rect(r):
    return [int(round(r.x0)), int(round(r.y0)), int(round(r.x1)), int(round(r.y1))]


def _line_sort_key(line):
    b = line.get("bbox", [0, 0, 0, 0])
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return (0.0, 0.0)
    return (float(b[1]), float(b[0]))


def _block_text(block):
    parts = []
    for line in block.get("lines", []) or []:
        ltxt = (line.get("line_text") or "").strip()
        if not ltxt:
            for phrase in line.get("phrases", []) or []:
                ptxt = (phrase.get("texte") or phrase.get("translated_text") or "").strip()
                if ptxt:
                    parts.append(ptxt)
            continue
        parts.append(ltxt)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _word_count(text):
    return len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", text or ""))


def _block_style_signature(block):
    for line in block.get("lines", []) or []:
        for phrase in line.get("phrases", []) or []:
            for span in phrase.get("spans", []) or []:
                st = span.get("style", {}) or {}
                flags = st.get("flags", {}) if isinstance(st.get("flags"), dict) else {}
                return (
                    st.get("font", ""),
                    round(float(st.get("size", 0.0) or 0.0), 1),
                    st.get("color", "#000000"),
                    bool(flags.get("bold")),
                    bool(flags.get("italic")),
                )
    return ("", 0.0, "#000000", False, False)


def _style_is_compatible(sig_a, sig_b):
    if not sig_a or not sig_b:
        return False
    same_font = sig_a[0] == sig_b[0]
    same_color = sig_a[2] == sig_b[2]
    size_close = abs(float(sig_a[1]) - float(sig_b[1])) <= 1.2
    weight_close = sig_a[3] == sig_b[3]
    italic_close = sig_a[4] == sig_b[4]
    return same_font and same_color and size_close and weight_close and italic_close


def _horizontal_overlap_ratio(r1, r2):
    inter = max(0.0, min(r1.x1, r2.x1) - max(r1.x0, r2.x0))
    den = max(1.0, min(r1.width, r2.width))
    return inter / den


def _vertical_overlap_ratio(r1, r2):
    inter = max(0.0, min(r1.y1, r2.y1) - max(r1.y0, r2.y0))
    den = max(1.0, min(r1.height, r2.height))
    return inter / den


def _same_left_edge(r1, r2, tolerance=24.0):
    return abs(float(r1.x0) - float(r2.x0)) <= float(tolerance)


def _same_right_edge(r1, r2, tolerance=24.0):
    return abs(float(r1.x1) - float(r2.x1)) <= float(tolerance)


def _recompute_block_fields(block):
    lines = list(block.get("lines", []) or [])
    lines.sort(key=_line_sort_key)
    block["lines"] = lines
    if lines:
        rect = None
        for ln in lines:
            lb = _rect_from_bbox(ln.get("bbox", [0, 0, 0, 0]))
            rect = lb if rect is None else (rect | lb)
        if rect is not None:
            block["bbox"] = _bbox_from_rect(rect)
    line_texts = []
    for line in lines:
        ltxt = (line.get("line_text") or "").strip()
        if not ltxt:
            parts = []
            for phrase in line.get("phrases", []) or []:
                ptxt = (phrase.get("texte") or phrase.get("translated_text") or "").strip()
                if ptxt:
                    parts.append(ptxt)
            ltxt = re.sub(r"\s+", " ", " ".join(parts)).strip()
            if ltxt:
                line["line_text"] = ltxt
        if ltxt:
            line_texts.append(ltxt)
    block["line_texts"] = line_texts
    block_text = _block_text(block)
    if block_text:
        block["text"] = block_text
        block["raw_text"] = block_text
    return block


def compute_fragmentation_metrics(page_data):
    blocks = page_data.get("blocks") or []
    if not blocks:
        return {
            "score": 0.0,
            "block_count": 0,
            "small_block_ratio": 0.0,
            "single_line_ratio": 0.0,
            "micro_block_count": 0,
        }

    small_blocks = 0
    single_line_blocks = 0
    micro_blocks = 0
    for block in blocks:
        text = _block_text(block)
        words = _word_count(text)
        line_count = len(block.get("lines", []) or [])
        bb = _rect_from_bbox(block.get("bbox", [0, 0, 0, 0]))
        if line_count <= 1:
            single_line_blocks += 1
        if words <= 6 and line_count <= 2:
            small_blocks += 1
        if words <= 3 and bb.width <= 180 and bb.height <= 42:
            micro_blocks += 1

    total = max(1, len(blocks))
    small_ratio = small_blocks / total
    single_line_ratio = single_line_blocks / total
    micro_ratio = micro_blocks / total
    score = min(1.0, 0.45 * small_ratio + 0.35 * single_line_ratio + 0.20 * micro_ratio)
    return {
        "score": round(score, 4),
        "block_count": len(blocks),
        "small_block_ratio": round(small_ratio, 4),
        "single_line_ratio": round(single_line_ratio, 4),
        "micro_block_count": micro_blocks,
    }


def _should_merge_table_fragments(a, b, page_w):
    ra = _rect_from_bbox(a.get("bbox", [0, 0, 0, 0]))
    rb = _rect_from_bbox(b.get("bbox", [0, 0, 0, 0]))
    if ra.get_area() <= 0 or rb.get_area() <= 0:
        return False
    if a.get("source") != b.get("source"):
        return False
    if a.get("role", "body") != b.get("role", "body"):
        return False
    if a.get("role", "body") not in {"body", "title", "section_heading"}:
        return False
    if not _style_is_compatible(_block_style_signature(a), _block_style_signature(b)):
        return False
    ta = _block_text(a)
    tb = _block_text(b)
    if not ta or not tb:
        return False
    wa = _word_count(ta)
    wb = _word_count(tb)
    if max(wa, wb) > 18:
        return False
    if max(ra.width, rb.width) > page_w * 0.55:
        return False

    h_ov = _horizontal_overlap_ratio(ra, rb)
    v_gap = max(0.0, rb.y0 - ra.y1)
    if h_ov >= 0.72 and v_gap <= max(10.0, 0.35 * max(ra.height, rb.height)):
        return True
    return False


def _should_merge_table_row_fragments(a, b, page_w):
    ra = _rect_from_bbox(a.get("bbox", [0, 0, 0, 0]))
    rb = _rect_from_bbox(b.get("bbox", [0, 0, 0, 0]))
    if ra.get_area() <= 0 or rb.get_area() <= 0:
        return False
    if a.get("source") != b.get("source"):
        return False
    if a.get("role", "body") != b.get("role", "body"):
        return False
    if a.get("role", "body") not in {"body", "title", "section_heading"}:
        return False
    if not _style_is_compatible(_block_style_signature(a), _block_style_signature(b)):
        return False
    ta = _block_text(a)
    tb = _block_text(b)
    if not ta or not tb:
        return False
    wa = _word_count(ta)
    wb = _word_count(tb)
    if max(wa, wb) > 10:
        return False
    if max(ra.width, rb.width) > page_w * 0.42:
        return False
    if min(ra.width, rb.width) < 24:
        return False
    v_ov = _vertical_overlap_ratio(ra, rb)
    h_gap = max(0.0, rb.x0 - ra.x1)
    row_height = max(ra.height, rb.height)
    if v_ov < 0.55:
        return False
    if h_gap > max(28.0, row_height * 1.4):
        return False
    if abs(ra.y0 - rb.y0) > max(10.0, row_height * 0.35):
        return False
    return True


def _illustration_rect(page_data):
    candidates = []
    for img in page_data.get("images", []) or []:
        candidates.append(_rect_from_bbox(img.get("bbox", [0, 0, 0, 0])))
    for zone in page_data.get("non_text_zones", []) or []:
        candidates.append(_rect_from_bbox(zone))
    usable = [r for r in candidates if r.get_area() > 0]
    if not usable:
        return fitz.Rect(0, 0, 0, 0)
    rect = usable[0]
    for other in usable[1:]:
        rect |= other
    return rect


def _annotation_side(rect, illustration):
    if illustration.get_area() <= 0:
        return "unknown"
    cx = (rect.x0 + rect.x1) * 0.5
    cy = (rect.y0 + rect.y1) * 0.5
    if rect.x1 <= illustration.x0:
        return "left"
    if rect.x0 >= illustration.x1:
        return "right"
    if rect.y1 <= illustration.y0:
        return "above"
    if rect.y0 >= illustration.y1:
        return "below"
    icx = (illustration.x0 + illustration.x1) * 0.5
    icy = (illustration.y0 + illustration.y1) * 0.5
    if abs(cx - icx) > abs(cy - icy):
        return "left" if cx < icx else "right"
    return "above" if cy < icy else "below"


def _is_annotation_candidate(block, illustration, page_w, page_h):
    rect = _rect_from_bbox(block.get("bbox", [0, 0, 0, 0]))
    if rect.get_area() <= 0:
        return False
    role = str(block.get("role", "body") or "body").strip().lower()
    source = str(block.get("source", "") or "").strip().lower()
    if source not in {"native", "native_phrase", "native_line"}:
        return False
    if role not in {"title", "body", "section_heading"}:
        return False
    text = _block_text(block)
    words = _word_count(text)
    if not text or words > 18:
        return False
    if rect.width > page_w * 0.45 or rect.height > page_h * 0.18:
        return False
    if illustration.get_area() > 0:
        overlap = rect & illustration
        if overlap.get_area() > 0 and overlap.get_area() / max(1.0, rect.get_area()) > 0.12:
            return False
        side = _annotation_side(rect, illustration)
        if side == "unknown":
            return False
    return True


def _should_merge_annotation_fragments(a, b, illustration, page_w, page_h):
    if not (_is_annotation_candidate(a, illustration, page_w, page_h) and _is_annotation_candidate(b, illustration, page_w, page_h)):
        return False
    ra = _rect_from_bbox(a.get("bbox", [0, 0, 0, 0]))
    rb = _rect_from_bbox(b.get("bbox", [0, 0, 0, 0]))
    if not _style_is_compatible(_block_style_signature(a), _block_style_signature(b)):
        return False
    if a.get("source") != b.get("source"):
        return False
    side_a = _annotation_side(ra, illustration)
    side_b = _annotation_side(rb, illustration)
    if side_a != side_b:
        return False
    ta = _block_text(a).strip()
    tb = _block_text(b).strip()
    vertical_stack = (
        (_same_left_edge(ra, rb, tolerance=36.0) or _same_right_edge(ra, rb, tolerance=36.0))
        and _horizontal_overlap_ratio(ra, rb) >= 0.45
        and max(0.0, rb.y0 - ra.y1) <= max(22.0, 0.65 * max(ra.height, rb.height))
    )
    same_row = (
        _vertical_overlap_ratio(ra, rb) >= 0.6
        and max(0.0, rb.x0 - ra.x1) <= max(20.0, 0.25 * page_w)
        and abs(ra.y0 - rb.y0) <= max(10.0, 0.25 * max(ra.height, rb.height))
    )
    if vertical_stack and _looks_like_annotation_continuation(ta, tb):
        return True
    if same_row and _looks_like_annotation_continuation(ta, tb, same_row=True):
        return True
    return False


def _looks_like_annotation_continuation(text_a, text_b, same_row=False):
    a = (text_a or "").strip()
    b = (text_b or "").strip()
    if not a or not b:
        return False
    if b[:1] in {"(", "[", ",", ";", ":"}:
        return True
    if re.match(r"^[a-zà-ÿ0-9]", b):
        return True
    if a.endswith(("-", "(", "/", ":", ",")):
        return True
    if "(" in a and ")" not in a:
        return True
    if same_row and _word_count(a) <= 3 and _word_count(b) <= 4 and b[:1].islower():
        return True
    return False


def _is_chart_tick_like(text):
    s = (text or "").strip()
    if not s:
        return False
    return bool(re.fullmatch(r"[\d.,%]+", s))


def _is_short_all_caps_label(text):
    s = (text or "").strip()
    return bool(re.fullmatch(r"[A-Z]{2,8}", s))


def _is_chart_label_candidate(block, page_w, page_h):
    rect = _rect_from_bbox(block.get("bbox", [0, 0, 0, 0]))
    if rect.get_area() <= 0:
        return False
    role = str(block.get("role", "body") or "body").strip().lower()
    if role not in {"title", "section_heading", "body"}:
        return False
    txt = _block_text(block).strip()
    if not txt:
        return False
    if _word_count(txt) > 10:
        return False
    if rect.width > page_w * 0.55 or rect.height > page_h * 0.14:
        return False
    return True


def _should_merge_chart_fragments(a, b, page_w, page_h):
    if not (_is_chart_label_candidate(a, page_w, page_h) and _is_chart_label_candidate(b, page_w, page_h)):
        return False
    ta = _block_text(a).strip()
    tb = _block_text(b).strip()
    if not ta or not tb:
        return False
    if _is_chart_tick_like(ta) or _is_chart_tick_like(tb):
        return False
    if _is_short_all_caps_label(ta) or _is_short_all_caps_label(tb):
        return False
    ra = _rect_from_bbox(a.get("bbox", [0, 0, 0, 0]))
    rb = _rect_from_bbox(b.get("bbox", [0, 0, 0, 0]))
    if not _style_is_compatible(_block_style_signature(a), _block_style_signature(b)):
        return False
    combined_width = max(ra.x1, rb.x1) - min(ra.x0, rb.x0)
    combined_words = _word_count(ta) + _word_count(tb)
    same_row = (
        _vertical_overlap_ratio(ra, rb) >= 0.65
        and max(0.0, rb.x0 - ra.x1) <= max(26.0, 0.08 * page_w)
        and abs(ra.y0 - rb.y0) <= max(8.0, 0.25 * max(ra.height, rb.height))
    )
    stacked = (
        (_same_left_edge(ra, rb, tolerance=26.0) or _same_right_edge(ra, rb, tolerance=26.0))
        and _horizontal_overlap_ratio(ra, rb) >= 0.45
        and max(0.0, rb.y0 - ra.y1) <= max(12.0, 0.45 * max(ra.height, rb.height))
    )
    if same_row and combined_width <= page_w * 0.42 and combined_words <= 6:
        return True
    if stacked and _looks_like_annotation_continuation(ta, tb):
        return True
    return False


def _merge_two_blocks(base, extra):
    rb = _rect_from_bbox(base.get("bbox", [0, 0, 0, 0]))
    re = _rect_from_bbox(extra.get("bbox", [0, 0, 0, 0]))
    base["bbox"] = _bbox_from_rect(rb | re)
    base["lines"] = list(base.get("lines", []) or []) + list(extra.get("lines", []) or [])
    return _recompute_block_fields(base)


def _set_structure_hints(block, **kwargs):
    hints = dict(block.get("structure_hints") or {})
    for key, value in kwargs.items():
        if value is None:
            continue
        if key == "group_ids":
            group_ids = dict(hints.get("group_ids") or {})
            group_ids.update(value or {})
            hints["group_ids"] = group_ids
        else:
            hints[key] = value
    block["structure_hints"] = hints
    return block


def _cluster_positions(values, tolerance):
    clusters = []
    for value in sorted(float(v) for v in values):
        placed = False
        for cluster in clusters:
            center = cluster["center"]
            if abs(value - center) <= tolerance:
                cluster["values"].append(value)
                cluster["center"] = sum(cluster["values"]) / len(cluster["values"])
                placed = True
                break
        if not placed:
            clusters.append({"center": value, "values": [value]})
    return clusters


def _annotate_table_native_structure(page_data):
    blocks = list(page_data.get("blocks") or [])
    if len(blocks) < 2:
        return None

    row_clusters = []
    for block in sorted(blocks, key=lambda b: (_rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).y0, _rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).x0)):
        rect = _rect_from_bbox(block.get("bbox", [0, 0, 0, 0]))
        if rect.get_area() <= 0:
            continue
        mid_y = (rect.y0 + rect.y1) * 0.5
        placed = False
        for row in row_clusters:
            tol = max(18.0, min(rect.height, row["height"]) * 0.9)
            if abs(mid_y - row["center_y"]) <= tol:
                row["blocks"].append(block)
                row["center_y"] = sum(((_rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).y0 + _rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).y1) * 0.5) for b in row["blocks"]) / len(row["blocks"])
                row["height"] = max(row["height"], rect.height)
                placed = True
                break
        if not placed:
            row_clusters.append({"blocks": [block], "center_y": mid_y, "height": rect.height})

    if len(row_clusters) < 1:
        return None

    column_centers = []
    for row in row_clusters:
        for block in row["blocks"]:
            rect = _rect_from_bbox(block.get("bbox", [0, 0, 0, 0]))
            column_centers.append((rect.x0 + rect.x1) * 0.5)
    columns = _cluster_positions(column_centers, tolerance=36.0)

    table_rect = None
    row_infos = []
    header_row_ids = []
    stub_column_group_id = f"native_table_col_0" if columns else None
    for row_idx, row in enumerate(row_clusters):
        sorted_blocks = sorted(row["blocks"], key=lambda b: _rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).x0)
        row_rect = None
        row_id = f"native_table_row_{row_idx}"
        title_like = 0
        for block in sorted_blocks:
            role = str(block.get("role") or "body").strip().lower()
            if role in {"title", "section_heading"}:
                title_like += 1
        first_text = _block_text(sorted_blocks[0]).strip() if sorted_blocks else ""
        first_has_alpha = bool(re.search(r"[A-Za-zÀ-ÿ]", first_text))
        first_word_count = _word_count(first_text)
        short_numeric_row = all(
            bool(re.fullmatch(r"[\d.,xX%+\-– ]+", _block_text(block).strip()))
            or (
                _word_count(_block_text(block)) <= 2
                and not re.search(r"[A-Za-zÀ-ÿ]", _block_text(block).strip())
            )
            for block in sorted_blocks
        )
        row_role = "body"
        if row_idx == 0:
            row_role = "header"
        elif title_like >= max(1, len(sorted_blocks) // 2) and not (row_idx >= 2 and first_has_alpha):
            row_role = "header"
        elif short_numeric_row and row_idx <= 2:
            row_role = "header"
        elif row_idx >= 2 and first_has_alpha and first_word_count <= 4:
            row_role = "body"
        if row_role == "header":
            header_row_ids.append(row_id)
        cell_infos = []
        for cell_idx, block in enumerate(sorted_blocks):
            rect = _rect_from_bbox(block.get("bbox", [0, 0, 0, 0]))
            row_rect = rect if row_rect is None else (row_rect | rect)
            col_center = (rect.x0 + rect.x1) * 0.5
            best_col_idx = min(range(len(columns)), key=lambda idx: abs(col_center - columns[idx]["center"])) if columns else 0
            cell_id = f"native_table_cell_{block.get('id')}"
            if row_role == "header":
                structural_role_hint = "table_header_cell"
                cell_role = "header"
            elif best_col_idx == 0 and len(columns) >= 2:
                structural_role_hint = "table_stub_cell"
                cell_role = "stub"
            else:
                structural_role_hint = "table_value_cell"
                cell_role = "value"
            _set_structure_hints(
                block,
                band_role_hint="table_band",
                structural_role_hint=structural_role_hint,
                layout_behavior_hint="locked_in_cell",
                group_ids={
                    "table_id": "native_table_main",
                    "table_row_group_id": row_id,
                    "table_column_group_id": f"native_table_col_{best_col_idx}",
                    "cell_id": cell_id,
                },
            )
            cell_infos.append(
                {
                    "id": cell_id,
                    "bbox": _bbox_from_rect(rect),
                    "block_id": str(block.get("id") or ""),
                    "column_group_id": f"native_table_col_{best_col_idx}",
                    "structural_role": structural_role_hint,
                    "cell_role": cell_role,
                }
            )
        if row_rect is None:
            continue
        table_rect = row_rect if table_rect is None else (table_rect | row_rect)
        row_infos.append(
            {
                "id": row_id,
                "bbox": _bbox_from_rect(row_rect),
                "block_ids": [str(block.get("id") or "") for block in sorted_blocks],
                "cells": cell_infos,
                "row_role": row_role,
            }
        )

    if table_rect is None:
        return None

    return {
        "table_id": "native_table_main",
        "bbox": _bbox_from_rect(table_rect),
        "row_groups": row_infos,
        "column_groups": [
            {"id": f"native_table_col_{idx}", "center_x": round(cluster["center"], 2)}
            for idx, cluster in enumerate(columns)
        ],
        "header_row_group_ids": header_row_ids,
        "stub_column_group_id": stub_column_group_id,
    }


def _annotate_annotation_native_structure(page_data, excluded_block_ids=None):
    blocks = list(page_data.get("blocks") or [])
    dims = page_data.get("dimensions") or {}
    page_w = float(dims.get("width", 0.0) or 0.0)
    page_h = float(dims.get("height", 0.0) or 0.0)
    illustration = _illustration_rect(page_data)
    excluded_ids = {str(bid) for bid in (excluded_block_ids or []) if str(bid)}
    if len(blocks) < 1 or illustration.get_area() <= 0:
        return None

    candidates = []
    for block in blocks:
        if str(block.get("id") or "") in excluded_ids:
            continue
        if not _is_annotation_candidate(block, illustration, page_w, page_h):
            continue
        rect = _rect_from_bbox(block.get("bbox", [0, 0, 0, 0]))
        side = _annotation_side(rect, illustration)
        if side == "unknown":
            continue
        candidates.append((block, rect, side))
    if not candidates:
        return None

    groups = []
    for block, rect, side in sorted(candidates, key=lambda item: (item[2], item[1].y0, item[1].x0)):
        mid_primary = (rect.y0 + rect.y1) * 0.5 if side in {"left", "right"} else (rect.x0 + rect.x1) * 0.5
        placed = False
        for group in groups:
            if group["side"] != side:
                continue
            if abs(mid_primary - group["primary_center"]) > max(48.0, group["primary_span"] * 0.7):
                continue
            group["blocks"].append(block)
            group["rect"] |= rect
            primary_values = [
                ((_rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).y0 + _rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).y1) * 0.5)
                if side in {"left", "right"}
                else ((_rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).x0 + _rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).x1) * 0.5)
                for b in group["blocks"]
            ]
            group["primary_center"] = sum(primary_values) / len(primary_values)
            group["primary_span"] = max(primary_values) - min(primary_values) if len(primary_values) > 1 else rect.height
            placed = True
            break
        if not placed:
            groups.append(
                {
                    "side": side,
                    "blocks": [block],
                    "rect": fitz.Rect(rect),
                    "primary_center": mid_primary,
                    "primary_span": rect.height if side in {"left", "right"} else rect.width,
                }
            )

    group_infos = []
    for idx, group in enumerate(groups):
        group_id = f"native_annotation_group_{idx}"
        bbox = _bbox_from_rect(group["rect"])
        for block in group["blocks"]:
            _set_structure_hints(
                block,
                band_role_hint="annotation_band",
                structural_role_hint="diagram_label",
                layout_behavior_hint="anchored",
                attachment_target_hint="illustration_main",
                group_ids={"annotation_group_id": group_id},
                side_hint=group["side"],
            )
        group_infos.append(
            {
                "id": group_id,
                "side": group["side"],
                "bbox": bbox,
                "block_ids": [str(block.get("id") or "") for block in group["blocks"]],
                "attachment_target_id": "illustration_main",
            }
        )

    return {
        "illustration_bbox": _bbox_from_rect(illustration),
        "groups": group_infos,
    }


def _annotate_chart_native_structure(page_data):
    chart = page_data.get("chart_structure") or {}
    chart_bbox = chart.get("chart_area_bbox")
    if not chart_bbox:
        return None

    blocks_by_id = {str(block.get("id") or ""): block for block in page_data.get("blocks") or []}
    def group_bbox(ids):
        rect = None
        for bid in ids or []:
            block = blocks_by_id.get(str(bid))
            if not block:
                continue
            br = _rect_from_bbox(block.get("bbox", [0, 0, 0, 0]))
            if br.get_area() <= 0:
                continue
            rect = br if rect is None else (rect | br)
        return _bbox_from_rect(rect) if rect is not None else None

    for bid in chart.get("y_tick_block_ids") or []:
        block = blocks_by_id.get(str(bid))
        if block:
            _set_structure_hints(
                block,
                band_role_hint="axis_band",
                structural_role_hint="chart_tick_label",
                layout_behavior_hint="anchored",
                group_ids={"tick_group_id": "native_chart_ticks_y", "axis_group_id": "native_chart_axis_y"},
            )
    for bid in chart.get("y_axis_label_ids") or []:
        block = blocks_by_id.get(str(bid))
        if block:
            _set_structure_hints(
                block,
                band_role_hint="axis_band",
                structural_role_hint="chart_axis_label",
                layout_behavior_hint="anchored",
                group_ids={"axis_group_id": "native_chart_axis_y"},
            )
    for bid in chart.get("x_axis_label_ids") or []:
        block = blocks_by_id.get(str(bid))
        if block:
            _set_structure_hints(
                block,
                band_role_hint="axis_band",
                structural_role_hint="chart_axis_label",
                layout_behavior_hint="anchored",
                group_ids={"axis_group_id": "native_chart_axis_x"},
            )
    for bid in chart.get("x_tick_block_ids") or []:
        if str(bid) in {str(v) for v in (chart.get("x_axis_label_ids") or [])}:
            continue
        block = blocks_by_id.get(str(bid))
        if block:
            _set_structure_hints(
                block,
                band_role_hint="axis_band",
                structural_role_hint="chart_tick_label",
                layout_behavior_hint="anchored",
                group_ids={"tick_group_id": "native_chart_ticks_x", "axis_group_id": "native_chart_axis_x"},
            )
    for bid in chart.get("legend_label_ids") or []:
        block = blocks_by_id.get(str(bid))
        if block:
            _set_structure_hints(
                block,
                band_role_hint="legend_band",
                structural_role_hint="chart_legend_label",
                layout_behavior_hint="anchored",
                attachment_target_hint="chart_main",
                group_ids={"legend_group_id": "native_chart_legend_0", "series_group_id": "native_chart_series_0"},
            )

    return {
        "chart_id": "native_chart_main",
        "chart_area_bbox": list(chart_bbox),
        "plot_area_bbox": list(chart.get("plot_area_bbox") or chart_bbox),
        "y_tick_group": {
            "id": "native_chart_ticks_y",
            "block_ids": list(chart.get("y_tick_block_ids") or []),
            "bbox": group_bbox(chart.get("y_tick_block_ids") or []),
        },
        "x_tick_group": {
            "id": "native_chart_ticks_x",
            "block_ids": list(chart.get("x_tick_block_ids") or []),
            "bbox": group_bbox(chart.get("x_tick_block_ids") or []),
        },
        "axis_groups": [
            {
                "id": "native_chart_axis_y",
                "block_ids": list(chart.get("y_axis_label_ids") or []),
                "bbox": group_bbox(chart.get("y_axis_label_ids") or []),
            },
            {
                "id": "native_chart_axis_x",
                "block_ids": list(chart.get("x_axis_label_ids") or []),
                "bbox": group_bbox(chart.get("x_axis_label_ids") or []),
            },
        ],
        "legend_group": {
            "id": "native_chart_legend_0",
            "block_ids": list(chart.get("legend_label_ids") or []),
            "bbox": group_bbox(chart.get("legend_label_ids") or []),
        },
        "series_groups": [
            {
                "id": "native_chart_series_0",
                "block_ids": list(chart.get("series_label_ids") or chart.get("legend_label_ids") or []),
                "bbox": group_bbox(chart.get("series_label_ids") or chart.get("legend_label_ids") or []),
            }
        ],
    }


def _postprocess_table_page(page_data):
    blocks = list(page_data.get("blocks") or [])
    if len(blocks) < 2:
        return page_data, False

    page_w = float((page_data.get("dimensions") or {}).get("width", 0.0) or 0.0)
    ordered = sorted(
        [copy.deepcopy(b) for b in blocks],
        key=lambda b: (_rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).y0, _rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).x0),
    )

    merged = []
    changed = False
    for blk in ordered:
        target_idx = None
        for idx in range(len(merged) - 1, -1, -1):
            cur = merged[idx]
            if _should_merge_table_fragments(cur, blk, page_w=page_w):
                target_idx = idx
                break
        if target_idx is None:
            merged.append(blk)
        else:
            merged[target_idx] = _merge_two_blocks(merged[target_idx], blk)
            changed = True

    second_pass = []
    for blk in merged:
        target_idx = None
        for idx in range(len(second_pass) - 1, -1, -1):
            cur = second_pass[idx]
            if _should_merge_table_row_fragments(cur, blk, page_w=page_w):
                target_idx = idx
                break
        if target_idx is None:
            second_pass.append(blk)
        else:
            second_pass[target_idx] = _merge_two_blocks(second_pass[target_idx], blk)
            changed = True

    if changed:
        page_data["blocks"] = second_pass
    return page_data, changed


def _postprocess_annotated_page(page_data):
    blocks = list(page_data.get("blocks") or [])
    if len(blocks) < 2:
        return page_data, False

    dims = page_data.get("dimensions") or {}
    page_w = float(dims.get("width", 0.0) or 0.0)
    page_h = float(dims.get("height", 0.0) or 0.0)
    illustration = _illustration_rect(page_data)

    ordered = sorted(
        [copy.deepcopy(b) for b in blocks],
        key=lambda b: (_rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).y0, _rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).x0),
    )

    merged = []
    changed = False
    for blk in ordered:
        target_idx = None
        for idx in range(len(merged) - 1, -1, -1):
            cur = merged[idx]
            if _should_merge_annotation_fragments(cur, blk, illustration=illustration, page_w=page_w, page_h=page_h):
                target_idx = idx
                break
        if target_idx is None:
            merged.append(blk)
        else:
            merged[target_idx] = _merge_two_blocks(merged[target_idx], blk)
            changed = True

    if changed:
        page_data["blocks"] = merged
    return page_data, changed


def _postprocess_chart_page(page_data):
    blocks = list(page_data.get("blocks") or [])
    if len(blocks) < 2:
        return page_data, False

    dims = page_data.get("dimensions") or {}
    page_w = float(dims.get("width", 0.0) or 0.0)
    page_h = float(dims.get("height", 0.0) or 0.0)
    ordered = sorted(
        [copy.deepcopy(b) for b in blocks],
        key=lambda b: (_rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).y0, _rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).x0),
    )

    merged = []
    changed = False
    for blk in ordered:
        target_idx = None
        for idx in range(len(merged) - 1, -1, -1):
            cur = merged[idx]
            if _should_merge_chart_fragments(cur, blk, page_w=page_w, page_h=page_h):
                target_idx = idx
                break
        if target_idx is None:
            merged.append(blk)
        else:
            merged[target_idx] = _merge_two_blocks(merged[target_idx], blk)
            changed = True

    if changed:
        page_data["blocks"] = merged
    return page_data, changed


def _extract_chart_structure(page_data):
    dims = page_data.get("dimensions") or {}
    page_w = float(dims.get("width", 0.0) or 0.0)
    page_h = float(dims.get("height", 0.0) or 0.0)
    blocks = list(page_data.get("blocks") or [])
    if not blocks or page_w <= 0 or page_h <= 0:
        return None

    candidates = []
    for block in blocks:
        text = _block_text(block).strip()
        rect = _rect_from_bbox(block.get("bbox", [0, 0, 0, 0]))
        if not text or rect.get_area() <= 0:
            continue
        candidates.append((block, text, rect))

    y_ticks = []
    for block, text, rect in candidates:
        if not _is_chart_tick_like(text):
            continue
        if rect.width > page_w * 0.08 or rect.height > page_h * 0.06:
            continue
        if rect.x0 > page_w * 0.45:
            continue
        if rect.y0 < page_h * 0.45:
            continue
        y_ticks.append((block, text, rect))

    if len(y_ticks) < 4:
        return None

    y_ticks.sort(key=lambda item: item[2].y0)
    y_tick_rect = y_ticks[0][2]
    for _, _, rect in y_ticks[1:]:
        y_tick_rect |= rect
    chart_left = min(page_w, y_tick_rect.x1 + 12.0)
    chart_top = max(0.0, y_tick_rect.y0 - 24.0)
    chart_bottom = min(page_h, y_tick_rect.y1 + 12.0)

    overlapping = []
    for block, text, rect in candidates:
        if rect.x1 <= chart_left:
            continue
        if rect.y1 < chart_top - 24.0 or rect.y0 > chart_bottom + 40.0:
            continue
        if str(block.get("role") or "").strip().lower() == "figure_caption":
            continue
        overlapping.append((block, text, rect))
    if not overlapping:
        return None

    chart_right = max(rect.x1 for _, _, rect in overlapping)
    chart_area = [chart_left, chart_top, min(page_w, chart_right + 12.0), chart_bottom]

    legend_ids = []
    y_axis_label_ids = []
    x_axis_label_ids = []
    x_tick_ids = []
    for block, text, rect in candidates:
        bid = str(block.get("id") or "")
        if not bid:
            continue
        role = str(block.get("role") or "").strip().lower()
        if rect.x1 <= chart_left and rect.y0 <= chart_bottom and rect.y1 >= chart_top and not _is_chart_tick_like(text):
            if rect.height > rect.width * 2.2 or rect.height >= page_h * 0.05:
                y_axis_label_ids.append(bid)
                continue
        if (
            rect.y0 >= chart_bottom - 24.0
            and rect.y1 <= min(page_h, chart_bottom + page_h * 0.18)
            and rect.x1 >= chart_left - 24.0
            and rect.x0 <= chart_area[2] + 32.0
        ):
            if role == "figure_caption":
                continue
            numeric_tokens = re.findall(r"\b\d+(?:[.,]\d+)?\b", text or "")
            has_alpha = bool(re.search(r"[A-Za-zÀ-ÿ]", text or ""))
            looks_mixed_axis = len(numeric_tokens) >= 3 and has_alpha
            if _is_chart_tick_like(text):
                x_tick_ids.append(bid)
                continue
            if looks_mixed_axis and role in {"title", "section_heading", "header"}:
                x_tick_ids.append(bid)
            if role in {"title", "section_heading", "header"}:
                x_axis_label_ids.append(bid)
                continue
        if rect.x0 >= chart_left and rect.x1 <= chart_area[2] and rect.y0 >= chart_top and rect.y1 <= chart_bottom:
            if role in {"title", "section_heading"} and not _is_chart_tick_like(text):
                if _word_count(text) <= 4:
                    legend_ids.append(bid)

    def bbox_union(ids):
        rect = None
        for bid in ids or []:
            block = next((b for b, _, _ in candidates if str(b.get("id") or "") == str(bid)), None)
            if not block:
                continue
            br = _rect_from_bbox(block.get("bbox", [0, 0, 0, 0]))
            if br.get_area() <= 0:
                continue
            rect = br if rect is None else (rect | br)
        if rect is None:
            return None
        return [round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)]

    x_ticks_bbox = bbox_union(x_tick_ids)
    plot_left = min(page_w, chart_left + 6.0)
    plot_top = min(page_h, chart_top + 6.0)
    plot_bottom = chart_bottom - 6.0
    if x_ticks_bbox:
        plot_bottom = min(plot_bottom, max(plot_top + 24.0, float(x_ticks_bbox[1]) - 8.0))
    plot_area_bbox = [
        round(plot_left, 2),
        round(plot_top, 2),
        round(max(plot_left + 24.0, chart_area[2] - 6.0), 2),
        round(max(plot_top + 24.0, plot_bottom), 2),
    ]

    return {
        "chart_area_bbox": [round(v, 2) for v in chart_area],
        "plot_area_bbox": plot_area_bbox,
        "y_tick_block_ids": [str(block.get("id") or "") for block, _, _ in y_ticks],
        "x_tick_block_ids": sorted(set(x_tick_ids)),
        "y_axis_label_ids": sorted(set(y_axis_label_ids)),
        "x_axis_label_ids": sorted(set(x_axis_label_ids)),
        "legend_label_ids": sorted(set(legend_ids)),
        "series_label_ids": sorted(set(legend_ids)),
    }


def apply_page_extraction_postprocessors(page_data):
    if not isinstance(page_data, dict):
        return page_data, {"changed": False, "applied": [], "fragmentation": compute_fragmentation_metrics({})}

    work = copy.deepcopy(page_data)
    applied = []
    changed = False

    fragmentation = compute_fragmentation_metrics(work)
    work["fragmentation_metrics"] = fragmentation

    layout_type = str(work.get("layout_type") or "").strip().lower()
    page_family = str(work.get("page_family") or "").strip().lower()
    document_type = str(work.get("document_type") or "").strip().lower()

    if layout_type == "table_dominant" or page_family in {"table_page", "table_diagram_example"} or document_type in {"form", "invoice", "receipt"}:
        work, local_changed = _postprocess_table_page(work)
        if local_changed:
            changed = True
            applied.append("table_dominant_merge")

    if layout_type == "annotated_page" or page_family in {"illustrated_label_page", "chart_label_page", "body_with_diagram"}:
        work, local_changed = _postprocess_annotated_page(work)
        if local_changed:
            changed = True
            applied.append("annotated_page_grouping")

    if page_family == "chart_label_page":
        work, local_changed = _postprocess_chart_page(work)
        if local_changed:
            changed = True
            applied.append("chart_label_grouping")
        chart_structure = _extract_chart_structure(work)
        if chart_structure:
            work["chart_structure"] = chart_structure
            applied.append("chart_structure")

    chart_structure = work.get("chart_structure") or {}
    chart_block_ids = set()
    for key in ("y_tick_block_ids", "y_axis_label_ids", "x_axis_label_ids", "legend_label_ids"):
        chart_block_ids.update(str(bid) for bid in (chart_structure.get(key) or []) if str(bid))

    native_structure = {
        "table": None,
        "annotations": None,
        "chart": None,
    }
    if layout_type == "table_dominant" or page_family in {"table_page", "table_diagram_example"} or document_type in {"form", "invoice", "receipt"}:
        native_structure["table"] = _annotate_table_native_structure(work)
    if page_family == "chart_label_page":
        native_structure["chart"] = _annotate_chart_native_structure(work)
    if layout_type == "annotated_page" or page_family in {"illustrated_label_page", "chart_label_page", "body_with_diagram"}:
        native_structure["annotations"] = _annotate_annotation_native_structure(work, excluded_block_ids=chart_block_ids)
    work["native_structure"] = native_structure

    if changed:
        work["fragmentation_metrics"] = compute_fragmentation_metrics(work)

    work["extraction_postprocess"] = {
        "changed": bool(changed),
        "applied": applied,
        "fragmentation_before": fragmentation,
        "fragmentation_after": work.get("fragmentation_metrics"),
    }
    return work, work["extraction_postprocess"]
