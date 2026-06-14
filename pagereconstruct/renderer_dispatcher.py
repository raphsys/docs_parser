"""Map a unit's renderer/role to a concrete renderer. Unknown role NEVER becomes
the paragraph renderer (directive Lot 8)."""

from __future__ import annotations

from .renderers import (
    AnchoredLabelRenderer,
    AnchoredLabelReviewRenderer,
    CodeRenderer,
    FormulaRenderer,
    HeadingRenderer,
    ListItemRenderer,
    ParagraphRenderer,
    PreservationRenderer,
    TableCellRenderer,
)

_PARAGRAPH = ParagraphRenderer()
_LIST = ListItemRenderer()
_HEADING = HeadingRenderer()
_TABLE = TableCellRenderer()
_CODE = CodeRenderer()
_FORMULA = FormulaRenderer()
_ANCHORED = AnchoredLabelRenderer()
_REVIEW = AnchoredLabelReviewRenderer()
_PRESERVE = PreservationRenderer()

_BY_RENDERER = {
    "paragraph": _PARAGRAPH, "list_item": _LIST, "caption": _PARAGRAPH,
    "heading": _HEADING, "table": _TABLE, "code": _CODE, "formula": _FORMULA,
    "anchored_label": _ANCHORED, "anchored_label_review": _REVIEW,
    "preservation": _PRESERVE,
}

_BY_ROLE = {
    "body_paragraph": _PARAGRAPH, "paragraph": _PARAGRAPH, "author_bio": _PARAGRAPH,
    "index_subentry": _PARAGRAPH, "formula_explanation": _PARAGRAPH,
    "list_item": _LIST,
    "title": _HEADING, "subtitle": _HEADING, "section_heading": _HEADING,
    "subsection_heading": _HEADING, "chapter_heading": _HEADING,
    "figure_caption": _PARAGRAPH, "table_caption": _PARAGRAPH,
    "table_body_cell": _TABLE, "table_header_cell": _TABLE, "table_numeric_cell": _TABLE,
    "code_line": _CODE, "code_block": _CODE, "command_name": _CODE,
    "formula_expression": _FORMULA,
    "toc_entry_title": _ANCHORED, "index_head_term": _ANCHORED,
    "diagram_label": _ANCHORED, "diagram_text_label": _ANCHORED,
    "axis_label": _ANCHORED, "legend_label": _ANCHORED,
    "publisher_mark": _PRESERVE, "page_number": _PRESERVE, "watermark": _PRESERVE,
    "page_header": _PRESERVE, "page_footer": _PRESERVE,
}


def dispatch(renderer_name: str | None, role: str | None):
    if renderer_name and renderer_name in _BY_RENDERER:
        return _BY_RENDERER[renderer_name]
    if role and role in _BY_ROLE:
        return _BY_ROLE[role]
    return _REVIEW  # unknown -> review, never paragraph
