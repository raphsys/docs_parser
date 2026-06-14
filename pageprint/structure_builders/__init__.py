"""Logical structure builders for PAGEPRINT."""

from __future__ import annotations

from .body_paragraph_builder import build_body_paragraphs
from .caption_builder import build_captions
from .code_builder import build_code_blocks
from .figure_builder import build_figures
from .formula_builder import build_formula_units
from .heading_builder import build_headings
from .index_builder import build_index_entries
from .list_builder import build_list_items
from .publisher_mark_builder import build_artifacts
from .table_builder import build_tables
from .toc_builder import build_toc_entries


def build_logical_structures(units: list[dict], *, page_intelligence: dict | None = None) -> dict:
    page_intelligence = page_intelligence or {}
    toc_entries = build_toc_entries(units, page_intelligence=page_intelligence)
    index_entries = build_index_entries(units, page_intelligence=page_intelligence)
    body_paragraphs = build_body_paragraphs(units, page_intelligence=page_intelligence)
    headings = build_headings(units, page_intelligence=page_intelligence)
    tables = build_tables(units, page_intelligence=page_intelligence)
    captions = build_captions(units, page_intelligence=page_intelligence)
    figures = build_figures(units, captions=captions, page_intelligence=page_intelligence)
    list_items = build_list_items(units, page_intelligence=page_intelligence)
    code_blocks = build_code_blocks(units, page_intelligence=page_intelligence)
    formulas = build_formula_units(units, page_intelligence=page_intelligence)
    artifacts = build_artifacts(units, page_intelligence=page_intelligence)
    logical_units = [
        *toc_entries,
        *index_entries,
        *headings,
        *body_paragraphs,
        *tables,
        *captions,
        *figures,
        *list_items,
        *code_blocks,
        *formulas,
        *artifacts.get("publisher_marks", []),
        *artifacts.get("watermarks", []),
        *artifacts.get("page_numbers", []),
    ]
    return {
        "schema_version": "pageprint.logical_structures.v1",
        "logical_units": logical_units,
        "toc_entries": toc_entries,
        "index_entries": index_entries,
        "headings": headings,
        "body_paragraphs": body_paragraphs,
        "tables": tables,
        "captions": captions,
        "figures": figures,
        "list_items": list_items,
        "code_blocks": code_blocks,
        "formula_units": formulas,
        "publisher_marks": artifacts.get("publisher_marks", []),
        "watermarks": artifacts.get("watermarks", []),
        "page_numbers": artifacts.get("page_numbers", []),
        "artifacts": artifacts,
    }


__all__ = ["build_logical_structures"]
