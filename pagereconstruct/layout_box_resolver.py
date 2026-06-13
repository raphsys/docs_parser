"""Resolve layout/patch/anchor bboxes per role (directive Lot 3, ABSOLUTE PRIORITY).

The core failure: a multi-line paragraph is erased over its full area but the
translation is re-drawn inside the FIRST-LINE bbox. Fix: for flow text, the
layout box must be the full logical block; the patch box covers the source
lines; the anchor stays the first line.
"""

from __future__ import annotations

_FLOW_ROLES = {"body_paragraph", "paragraph", "body", "list_item", "author_bio",
               "index_subentry", "formula_explanation"}
_HEADING_ROLES = {"title", "subtitle", "section_heading", "subsection_heading", "chapter_heading"}
_LOCKED_ROLES = {"table_body_cell", "table_header_cell", "table_numeric_cell",
                 "diagram_label", "axis_label", "legend_label", "code_line", "formula_expression"}


def _height(b):
    return (float(b[3]) - float(b[1])) if isinstance(b, (list, tuple)) and len(b) == 4 else 0.0


def resolve_layout(role: str | None, layout_bbox, coverage_bbox, anchor_bbox=None) -> tuple:
    """Return (layout_bbox, patch_bbox, anchor_bbox, findings)."""
    role = str(role or "")
    findings = []
    lb = layout_bbox
    cov = coverage_bbox or layout_bbox
    anchor = anchor_bbox or layout_bbox

    if role in _FLOW_ROLES:
        # Flow text must be laid out in the full logical block, not one line.
        if cov and _height(lb) < 0.5 * _height(cov):
            findings.append({"type": "layout_bbox_repaired_from_coverage", "role": role,
                             "from_h": round(_height(lb), 1), "to_h": round(_height(cov), 1)})
            lb = cov
        patch = cov or lb
    elif role in _HEADING_ROLES:
        # Heading may stay compact but must contain the full source text area.
        if cov and _height(lb) < 0.5 * _height(cov):
            lb = cov
            findings.append({"type": "heading_layout_bbox_expanded", "role": role})
        patch = cov or lb
    elif role in _LOCKED_ROLES:
        patch = lb  # locked: do not expand
    else:
        patch = cov or lb

    return lb, patch, anchor, findings
