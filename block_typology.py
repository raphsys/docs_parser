def _norm(value, default=""):
    raw = default if value is None else value
    return str(raw).strip().lower()


def classify_block_typology(unit, context=None):
    context = context or {}
    unit = unit or {}
    hints = unit.get("structure_hints") or {}
    semantic = unit.get("semantic") or {}
    role = _norm(unit.get("role") or context.get("block_role") or context.get("role"))
    semantic_type = _norm(semantic.get("type"))
    band_role = _norm(hints.get("band_role_hint") or context.get("band_role"))
    structural_role = _norm(hints.get("structural_role_hint") or context.get("structural_role"))
    layout_behavior = _norm(hints.get("layout_behavior_hint") or context.get("layout_behavior"))
    layout_type = _norm(context.get("layout_type"))
    page_family_group = _norm(context.get("page_family_group"))
    line_count = len(unit.get("lines") or []) if isinstance(unit.get("lines"), list) else 0

    is_heading_like = role in {"title", "section_heading", "figure_caption"} or semantic_type == "heading"
    is_locked_cell = layout_behavior in {"locked_in_cell", "locked_in_table"}
    is_annotated_page = layout_type in {"annotated_page", "mixed_blocks", "table_dominant"} or page_family_group in {
        "body_with_figure",
        "body_with_diagram",
        "mixed_page",
        "table_page",
    }
    is_visual_label = role in {"diagram_text_label", "diagram_label"} or structural_role in {
        "diagram_label",
        "chart_axis_label",
        "chart_tick_label",
        "chart_legend_label",
        "chart_series_label",
    }

    subtype = "generic_block"
    if is_visual_label:
        subtype = "visual_label"
    elif is_locked_cell and structural_role in {"table_stub_cell", "table_value_cell"} and line_count >= 5:
        subtype = "locked_code_table"
    elif is_locked_cell and structural_role in {"table_stub_cell", "table_value_cell"} and line_count <= 4:
        subtype = "editorial_locked_callout"
    elif is_heading_like and band_role in {"annotation_band", "caption_band", "table_band", "title_band"} and line_count <= 4:
        subtype = "editorial_short_callout"
    elif is_annotated_page and line_count <= 4 and band_role in {"annotation_band", "caption_band", "table_band", ""}:
        subtype = "editorial_short_callout"
    elif band_role == "table_band" and line_count >= 5:
        subtype = "dense_table_body"

    return {
        "role": role,
        "semantic_type": semantic_type,
        "band_role": band_role,
        "structural_role": structural_role,
        "layout_behavior": layout_behavior,
        "layout_type": layout_type,
        "page_family_group": page_family_group,
        "line_count": line_count,
        "is_heading_like": is_heading_like,
        "is_locked_cell": is_locked_cell,
        "is_annotated_page": is_annotated_page,
        "is_visual_label": is_visual_label,
        "subtype": subtype,
    }
