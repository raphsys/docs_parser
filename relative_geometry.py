import re


SCHEMA_VERSION = "relative_geometry.v1"


_RTL_RANGES = (
    "\u0590-\u05FF"  # Hebrew
    "\u0600-\u06FF"  # Arabic
    "\u0750-\u077F"
    "\u08A0-\u08FF"
    "\uFB50-\uFDFF"
    "\uFE70-\uFEFF"
)
_RTL_RE = re.compile(f"[{_RTL_RANGES}]")
_LTR_RE = re.compile(r"[A-Za-zÀ-ÿ]")

_PAPER_SIZES_MM = {
    "A5": (148.0, 210.0),
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A2": (420.0, 594.0),
    "Letter": (215.9, 279.4),
    "Legal": (215.9, 355.6),
}


def enrich_page_relative_geometry(page_data):
    if not isinstance(page_data, dict):
        return page_data

    dims = page_data.get("dimensions") or {}
    page_w = float(dims.get("width", 0.0) or 0.0)
    page_h = float(dims.get("height", 0.0) or 0.0)
    dpi = float(dims.get("dpi", 0.0) or 0.0)
    if page_w <= 0.0 or page_h <= 0.0:
        return page_data

    page_bbox = [0.0, 0.0, page_w, page_h]
    blocks = [block for block in (page_data.get("blocks") or []) if isinstance(block, dict)]
    layout_direction = _resolve_layout_direction(page_data, blocks)
    columns = _normalize_columns(((page_data.get("layout") or {}).get("columns") or []), page_w)

    page_features = _page_features(page_bbox, blocks, dpi=dpi, layout_direction=layout_direction, columns=columns)

    block_order = _reading_order(blocks, layout_direction=layout_direction, columns=columns)
    block_tree = []
    reading_order_ids = []
    for block_index, block in enumerate(block_order, start=1):
        block_bbox = _bbox(block)
        block_tree.append(
            _annotate_node(
                node=block,
                node_type="block",
                parent_bbox=page_bbox,
                parent_id=f"page_{int(page_data.get('page') or 0)}",
                layout_direction=layout_direction,
                sibling_order=block_index,
                child_extractor=_block_children,
                child_type_fn=_block_child_type,
                container_block_bbox=block_bbox,
                container_block_id=str(block.get("id") or f"block_{block_index}"),
                reading_order_path=[block_index],
                columns=columns,
            )
        )
        reading_order_ids.append(str(block.get("id") or f"block_{block_index}"))

    flat_nodes = []
    _flatten_nodes(block_tree, flat_nodes)

    relative_layout = {
        "schema_version": SCHEMA_VERSION,
        "page_id": f"page_{int(page_data.get('page') or 0)}",
        "layout_direction": layout_direction,
        "page_bbox": page_bbox,
        "page_features": page_features,
        "columns": columns,
        "reading_order": reading_order_ids,
        "children": block_tree,
        "flat_nodes": flat_nodes,
    }

    page_data["layout_direction"] = layout_direction
    page_data["relative_layout"] = relative_layout
    page_data["relative_layout_flat"] = flat_nodes
    page_data.setdefault("layout", {})
    page_data["layout"]["relative_layout"] = relative_layout
    page_data["layout"]["relative_layout_flat"] = flat_nodes
    page_data["layout"]["relative_layout_version"] = SCHEMA_VERSION
    return page_data


def _page_features(page_bbox, blocks, dpi=0.0, layout_direction="ltr", columns=None):
    page_w = float(page_bbox[2]) - float(page_bbox[0])
    page_h = float(page_bbox[3]) - float(page_bbox[1])
    surface = max(0.0, page_w * page_h)
    occupied_surface = sum(_bbox_area(_bbox(block)) for block in blocks)
    block_count = len(blocks)
    line_count = sum(len(block.get("lines") or []) for block in blocks)
    phrase_count = sum(len(line.get("phrases") or []) for block in blocks for line in (block.get("lines") or []))
    span_count = sum(len(phrase.get("spans") or []) for block in blocks for line in (block.get("lines") or []) for phrase in (line.get("phrases") or []))

    width_mm = 0.0
    height_mm = 0.0
    paper_guess = "unknown"
    paper_error_mm = 0.0
    if dpi > 0.0:
        width_mm = page_w * 25.4 / dpi
        height_mm = page_h * 25.4 / dpi
        paper_guess, paper_error_mm = _guess_paper_size(width_mm, height_mm)

    return {
        "width": round(page_w, 4),
        "height": round(page_h, 4),
        "unit": "px",
        "dpi": round(dpi, 4) if dpi > 0.0 else 0.0,
        "orientation": "landscape" if page_w > page_h else "portrait",
        "aspect_ratio": round(page_w / max(1.0, page_h), 6),
        "surface": round(surface, 4),
        "occupied_surface": round(occupied_surface, 4),
        "occupancy_ratio": round(occupied_surface / max(1.0, surface), 6),
        "block_count": int(block_count),
        "column_count": int(len(columns or [])),
        "line_count": int(line_count),
        "phrase_count": int(phrase_count),
        "span_count": int(span_count),
        "density_per_mpx": round((block_count * 1_000_000.0) / max(1.0, surface), 6),
        "width_mm": round(width_mm, 4) if width_mm > 0.0 else 0.0,
        "height_mm": round(height_mm, 4) if height_mm > 0.0 else 0.0,
        "paper_guess": paper_guess,
        "paper_error_mm": round(paper_error_mm, 4) if paper_guess != "unknown" else 0.0,
        "volume": {
            "surface_px2": round(surface, 4),
            "occupied_surface_px2": round(occupied_surface, 4),
            "free_surface_px2": round(max(0.0, surface - occupied_surface), 4),
        },
        "layout_direction": layout_direction,
    }


def _guess_paper_size(width_mm, height_mm):
    child = sorted([float(width_mm), float(height_mm)])
    best_name = "unknown"
    best_error = None
    for name, dims in _PAPER_SIZES_MM.items():
        ref = sorted([float(dims[0]), float(dims[1])])
        error = abs(child[0] - ref[0]) + abs(child[1] - ref[1])
        if best_error is None or error < best_error:
            best_name = name
            best_error = error
    if best_error is None or best_error > 20.0:
        return "unknown", 0.0
    return best_name, best_error


def _resolve_layout_direction(page_data, blocks):
    explicit = str(page_data.get("layout_direction") or page_data.get("reading_direction") or "").strip().lower()
    if explicit in {"ltr", "rtl"}:
        return explicit
    texts = []
    for block in blocks:
        txt = str(block.get("text") or block.get("raw_text") or "").strip()
        if txt:
            texts.append(txt)
    sample = " ".join(texts[:24])
    rtl_count = len(_RTL_RE.findall(sample))
    ltr_count = len(_LTR_RE.findall(sample))
    return "rtl" if rtl_count > ltr_count else "ltr"


def _annotate_node(
    node,
    node_type,
    parent_bbox,
    parent_id,
    layout_direction,
    sibling_order,
    child_extractor,
    child_type_fn,
    container_block_bbox=None,
    container_block_id="",
    reading_order_path=None,
    columns=None,
):
    bbox = _bbox(node)
    if not bbox:
        return {
            "id": str(node.get("id") or f"{node_type}_{sibling_order}"),
            "type": node_type,
            "parent_id": parent_id,
            "reading_order_index": sibling_order,
            "reading_order_path": list(reading_order_path or [sibling_order]),
            "bbox": None,
            "children": [],
        }

    relative_bbox = _relative_bbox(bbox, parent_bbox)
    borders_abs = _borders_from_bbox(bbox)
    borders_rel = _borders_from_bbox(relative_bbox)
    border_relations = {
        name: {
            "segment_absolute": dict(segment_abs),
            "segment_relative_to_parent": dict(borders_rel[name]),
            "relative_to_parent_borders": _border_to_parent_relations(segment_abs, parent_bbox),
        }
        for name, segment_abs in borders_abs.items()
    }

    inline_payload = {
        "schema_version": SCHEMA_VERSION,
        "type": node_type,
        "parent_id": str(parent_id or ""),
        "reading_order_index": int(sibling_order),
        "reading_order_path": list(reading_order_path or [sibling_order]),
        "layout_direction": layout_direction,
        "bbox_absolute": list(bbox),
        "bbox_relative_to_parent": list(relative_bbox),
        "size": {
            "width": round(float(bbox[2]) - float(bbox[0]), 4),
            "height": round(float(bbox[3]) - float(bbox[1]), 4),
            "surface": round(_bbox_area(bbox), 4),
        },
        "borders": border_relations,
    }
    if node_type == "block":
        inline_payload["column_id"] = _column_id_for_bbox(bbox, columns)
    if container_block_bbox and node_type != "block":
        container_relative_bbox = _relative_bbox(bbox, container_block_bbox)
        container_borders_rel = _borders_from_bbox(container_relative_bbox)
        inline_payload["container_block_id"] = str(container_block_id or "")
        inline_payload["bbox_relative_to_container_block"] = list(container_relative_bbox)
        inline_payload["borders_relative_to_container_block"] = {
            name: {
                "segment_relative_to_container_block": dict(container_borders_rel[name]),
                "relative_to_container_block_borders": _border_to_parent_relations(segment_abs, container_block_bbox),
            }
            for name, segment_abs in borders_abs.items()
        }
    node["relative_geometry"] = inline_payload
    node["reading_order_index"] = int(sibling_order)
    node["reading_order_path"] = list(reading_order_path or [sibling_order])
    node["layout_direction"] = layout_direction
    if node_type == "block":
        node["column_id"] = inline_payload.get("column_id")

    children = [child for child in child_extractor(node) if isinstance(child, dict)]
    ordered_children = _reading_order(children, layout_direction=layout_direction)
    child_tree = []
    for child_index, child in enumerate(ordered_children, start=1):
        child_tree.append(
            _annotate_node(
                node=child,
                node_type=child_type_fn(child),
                parent_bbox=bbox,
                parent_id=str(node.get("id") or f"{node_type}_{sibling_order}"),
                layout_direction=layout_direction,
                sibling_order=child_index,
                child_extractor=_children_for_type,
                child_type_fn=_child_type_for_nested,
                container_block_bbox=bbox if node_type == "block" else container_block_bbox,
                container_block_id=str(node.get("id") or f"{node_type}_{sibling_order}") if node_type == "block" else container_block_id,
                reading_order_path=list(reading_order_path or [sibling_order]) + [child_index],
                columns=None,
            )
        )

    payload = {
        "id": str(node.get("id") or f"{node_type}_{sibling_order}"),
        "type": node_type,
        "parent_id": str(parent_id or ""),
        "reading_order_index": int(sibling_order),
        "reading_order_path": list(reading_order_path or [sibling_order]),
        "layout_direction": layout_direction,
        "bbox": list(bbox),
        "bbox_relative_to_parent": list(relative_bbox),
        "size": dict(inline_payload["size"]),
        "borders": border_relations,
        "attributes": _node_attributes(node, node_type),
        "children": child_tree,
    }
    if node_type == "block":
        payload["column_id"] = inline_payload.get("column_id")
    if container_block_bbox and node_type != "block":
        payload["container_block_id"] = str(container_block_id or "")
        payload["bbox_relative_to_container_block"] = list(inline_payload.get("bbox_relative_to_container_block") or [])
        payload["borders_relative_to_container_block"] = dict(inline_payload.get("borders_relative_to_container_block") or {})
    return payload


def _flatten_nodes(nodes, collector):
    for node in nodes:
        flat = {
            "id": node.get("id"),
            "type": node.get("type"),
            "parent_id": node.get("parent_id"),
            "reading_order_index": node.get("reading_order_index"),
            "reading_order_path": list(node.get("reading_order_path") or []),
            "layout_direction": node.get("layout_direction"),
            "bbox": list(node.get("bbox") or []),
            "bbox_relative_to_parent": list(node.get("bbox_relative_to_parent") or []),
            "size": dict(node.get("size") or {}),
            "borders": dict(node.get("borders") or {}),
            "attributes": dict(node.get("attributes") or {}),
        }
        if node.get("column_id") is not None:
            flat["column_id"] = node.get("column_id")
        if node.get("container_block_id") is not None:
            flat["container_block_id"] = node.get("container_block_id")
        if node.get("bbox_relative_to_container_block") is not None:
            flat["bbox_relative_to_container_block"] = list(node.get("bbox_relative_to_container_block") or [])
        if node.get("borders_relative_to_container_block") is not None:
            flat["borders_relative_to_container_block"] = dict(node.get("borders_relative_to_container_block") or {})
        collector.append(flat)
        _flatten_nodes(node.get("children") or [], collector)


def _block_children(block):
    return list(block.get("lines") or [])


def _block_child_type(_child):
    return "line"


def _children_for_type(node):
    if node.get("phrases") is not None:
        return list(node.get("phrases") or [])
    if node.get("spans") is not None:
        return list(node.get("spans") or [])
    return []


def _child_type_for_nested(node):
    if node.get("spans") is not None:
        return "phrase"
    if node.get("texte") is not None or node.get("style") is not None:
        return "span"
    return "node"


def _node_attributes(node, node_type):
    payload = {
        "role": str(node.get("role") or ""),
        "source": str(node.get("source") or ""),
        "source_kind": str(node.get("source_kind") or ""),
        "alignment": str(node.get("alignment") or ""),
    }
    if node_type == "block":
        payload["unit_type"] = str(node.get("unit_type") or "")
        payload["translation_strategy"] = str(node.get("translation_strategy") or "")
        payload["style_class"] = str(node.get("style_class") or "")
        payload["text_preview"] = str((node.get("text") or node.get("raw_text") or "").strip())[:240]
    elif node_type in {"line", "phrase", "span"}:
        payload["text_preview"] = str((node.get("text") or node.get("line_text") or node.get("texte") or "").strip())[:240]
    return payload


def _normalize_columns(columns, page_width):
    normalized = []
    for idx, column in enumerate(columns or []):
        if not isinstance(column, dict):
            continue
        try:
            x0 = float(column.get("x0", 0.0) or 0.0)
            x1 = float(column.get("x1", 0.0) or 0.0)
        except Exception:
            continue
        if x1 < x0:
            x0, x1 = x1, x0
        if x1 <= x0:
            continue
        normalized.append(
            {
                "id": str(column.get("id") if column.get("id") is not None else idx),
                "index": int(idx),
                "x0": round(max(0.0, x0), 4),
                "x1": round(min(float(page_width), x1), 4),
            }
        )
    return normalized


def _column_id_for_bbox(bbox, columns):
    column = _column_for_bbox(bbox, columns)
    return None if not column else column.get("id")


def _column_for_bbox(bbox, columns):
    if not bbox or not columns:
        return None
    center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
    best = None
    best_score = None
    for column in columns:
        x0 = float(column.get("x0", 0.0) or 0.0)
        x1 = float(column.get("x1", 0.0) or 0.0)
        overlap = max(0.0, min(float(bbox[2]), x1) - max(float(bbox[0]), x0))
        center_penalty = 0.0 if x0 <= center_x <= x1 else min(abs(center_x - x0), abs(center_x - x1))
        score = (overlap, -center_penalty)
        if best_score is None or score > best_score:
            best_score = score
            best = column
    return best


def _bbox(node):
    bbox = (node or {}).get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def _bbox_area(bbox):
    if not bbox:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _relative_bbox(bbox, parent_bbox):
    return [
        round(float(bbox[0]) - float(parent_bbox[0]), 4),
        round(float(bbox[1]) - float(parent_bbox[1]), 4),
        round(float(bbox[2]) - float(parent_bbox[0]), 4),
        round(float(bbox[3]) - float(parent_bbox[1]), 4),
    ]


def _borders_from_bbox(bbox):
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return {
        "left": {"x_haut": round(x0, 4), "y_haut": round(y0, 4), "x_bas": round(x0, 4), "y_bas": round(y1, 4)},
        "top": {"x_haut": round(x0, 4), "y_haut": round(y0, 4), "x_bas": round(x1, 4), "y_bas": round(y0, 4)},
        "right": {"x_haut": round(x1, 4), "y_haut": round(y0, 4), "x_bas": round(x1, 4), "y_bas": round(y1, 4)},
        "bottom": {"x_haut": round(x0, 4), "y_haut": round(y1, 4), "x_bas": round(x1, 4), "y_bas": round(y1, 4)},
    }


def _border_to_parent_relations(segment, parent_bbox):
    px0, py0, px1, py1 = [float(v) for v in parent_bbox]
    xh = float(segment["x_haut"])
    yh = float(segment["y_haut"])
    xb = float(segment["x_bas"])
    yb = float(segment["y_bas"])
    return {
        "to_left": {"haut": round(xh - px0, 4), "bas": round(xb - px0, 4)},
        "to_top": {"haut": round(yh - py0, 4), "bas": round(yb - py0, 4)},
        "to_right": {"haut": round(px1 - xh, 4), "bas": round(px1 - xb, 4)},
        "to_bottom": {"haut": round(py1 - yh, 4), "bas": round(py1 - yb, 4)},
    }


def _reading_order(nodes, layout_direction="ltr", columns=None):
    enriched = []
    for idx, node in enumerate(nodes):
        bbox = _bbox(node)
        if not bbox:
            continue
        enriched.append({"node": node, "bbox": bbox, "index": idx, "column": _column_for_bbox(bbox, columns)})
    if len(enriched) <= 1:
        return [entry["node"] for entry in enriched]

    if columns and len(columns) >= 2:
        spanning = []
        grouped = {str(column.get("id")): [] for column in columns}
        for entry in enriched:
            bbox = entry["bbox"]
            overlap_columns = []
            for column in columns:
                overlap = max(0.0, min(float(bbox[2]), float(column["x1"])) - max(float(bbox[0]), float(column["x0"])))
                if overlap >= max(12.0, 0.15 * (float(bbox[2]) - float(bbox[0]))):
                    overlap_columns.append(column)
            if len(overlap_columns) >= 2 and (float(bbox[2]) - float(bbox[0])) >= 0.6 * sum(float(c["x1"]) - float(c["x0"]) for c in overlap_columns[:2]):
                spanning.append(entry)
            elif entry["column"] is not None:
                grouped[str(entry["column"]["id"])].append(entry)
            else:
                spanning.append(entry)

        spanning_sorted = _reading_order_linear(spanning, layout_direction=layout_direction)
        top_threshold = min((entry["bbox"][1] for group in grouped.values() for entry in group), default=None)
        bottom_threshold = max((entry["bbox"][3] for group in grouped.values() for entry in group), default=None)
        pre = []
        middle = []
        post = []
        for node in spanning_sorted:
            bbox = _bbox(node)
            if top_threshold is not None and bbox and bbox[3] <= top_threshold:
                pre.append(node)
            elif bottom_threshold is not None and bbox and bbox[1] >= bottom_threshold:
                post.append(node)
            else:
                middle.append(node)

        if layout_direction == "rtl":
            ordered_columns = sorted(columns, key=lambda col: (-float(col["x1"]), col["index"]))
        else:
            ordered_columns = sorted(columns, key=lambda col: (float(col["x0"]), col["index"]))

        ordered = list(pre)
        for column in ordered_columns:
            ordered.extend(_reading_order_linear(grouped.get(str(column["id"])) or [], layout_direction=layout_direction))
        ordered.extend(middle)
        ordered.extend(post)
        seen = set()
        unique = []
        for node in ordered:
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            unique.append(node)
        return unique

    return _reading_order_linear(enriched, layout_direction=layout_direction)


def _reading_order_linear(nodes_or_entries, layout_direction="ltr"):
    if not nodes_or_entries:
        return []
    if isinstance(nodes_or_entries[0], dict) and "node" in nodes_or_entries[0] and "bbox" in nodes_or_entries[0]:
        enriched = list(nodes_or_entries)
    else:
        enriched = []
        for idx, node in enumerate(nodes_or_entries):
            bbox = _bbox(node)
            if not bbox:
                continue
            enriched.append({"node": node, "bbox": bbox, "index": idx})
    if len(enriched) <= 1:
        return [entry["node"] for entry in enriched]

    heights = [max(1.0, entry["bbox"][3] - entry["bbox"][1]) for entry in enriched]
    heights_sorted = sorted(heights)
    median_height = heights_sorted[len(heights_sorted) // 2]
    tolerance = max(3.0, median_height * 0.6)
    enriched.sort(key=lambda entry: (entry["bbox"][1], entry["bbox"][0], entry["index"]))

    rows = []
    for entry in enriched:
        bbox = entry["bbox"]
        if not rows:
            rows.append({"top": bbox[1], "bottom": bbox[3], "items": [entry]})
            continue
        current = rows[-1]
        vertical_overlap = min(current["bottom"], bbox[3]) - max(current["top"], bbox[1])
        if bbox[1] <= current["bottom"] + tolerance or vertical_overlap > 0.0:
            current["items"].append(entry)
            current["top"] = min(current["top"], bbox[1])
            current["bottom"] = max(current["bottom"], bbox[3])
        else:
            rows.append({"top": bbox[1], "bottom": bbox[3], "items": [entry]})

    ordered = []
    for row in rows:
        if layout_direction == "rtl":
            row_items = sorted(row["items"], key=lambda entry: (-entry["bbox"][2], entry["bbox"][1], entry["index"]))
        else:
            row_items = sorted(row["items"], key=lambda entry: (entry["bbox"][0], entry["bbox"][1], entry["index"]))
        ordered.extend(entry["node"] for entry in row_items)
    return ordered
