from __future__ import annotations

import re

from .common import bbox_of, bbox_union, eligible_text_units, reading_order, role_of, text_of


COMMAND_RE = re.compile(r"^(?:copy|dir|del|findstr|mkdir|rmdir|cd|ls|cat|grep|sudo|docker|npm|pip|python|git)\b", re.IGNORECASE)
PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/[\w.-]+/|\.{1,2}/)[^\s]+")
NUMERIC_RE = re.compile(r"^\s*[-+]?\d+(?:[.,]\d+)?(?:\s*[%$€£]|[A-Za-z]{1,4})?\s*$")
FORMULA_RE = re.compile(r"[=∑√±×÷]|(?:\b[a-zA-Z]\s*[+\-*/]\s*[a-zA-Z0-9])")


def build_tables(units: list[dict], *, page_intelligence: dict | None = None) -> list[dict]:
    page_intelligence = page_intelligence or {}
    text_units = eligible_text_units(units)
    rows = _detect_rows_from_lines(text_units, page_intelligence=page_intelligence)
    if rows:
        cell_units = []
    else:
        cell_units = [
            unit for unit in eligible_text_units(units)
            if unit.get("level") == "cell" or (unit.get("level") == "phrase" and role_of(unit).startswith("table_"))
        ]
    if cell_units:
        rows = [[unit] for unit in sorted(cell_units, key=reading_order)]
    if not rows:
        return []

    cells = []
    source_units = []
    columns = []
    for row_idx, row in enumerate(rows, start=1):
        row_cells = row if all(isinstance(cell, dict) and cell.get("_synthetic_cell") for cell in row) else [
            _cell_from_unit(unit, row_idx, col_idx)
            for col_idx, unit in enumerate(row, start=1)
        ]
        for col_idx, cell in enumerate(row_cells, start=1):
            cell_kind = _cell_kind(cell.get("text") or "")
            role = _cell_role(row_idx, cell_kind)
            source_units.extend(cell.get("source_unit_ids") or [])
            cells.append({
                **cell,
                "cell_id": f"tbl_001_r{row_idx}_c{col_idx}",
                "row_index": row_idx,
                "column_index": col_idx,
                "role": role,
                "cell_kind": cell_kind,
                "translation_mode": "translate" if cell_kind in {"natural_text", "header_text"} else "preserve_text_exactly",
            })
            if row_idx == 1:
                columns.append({"column_id": f"c{col_idx}", "header_text": cell.get("text")})
    return [{
        "logical_unit_id": "tbl_001",
        "table_id": "tbl_001",
        "type": "table",
        "columns": columns,
        "rows": [{"row_id": f"tbl_001_r{idx}", "row_index": idx} for idx in range(1, len(rows) + 1)],
        "cells": cells,
        "source_unit_ids": list(dict.fromkeys(source_units)),
        "bbox": bbox_union([cell.get("bbox") for cell in cells]),
        "detection_strategy": "native_cells" if cell_units else "aligned_rows",
    }]


def _detect_rows_from_lines(text_units: list[dict], *, page_intelligence: dict) -> list[list[dict]]:
    page_role = str(page_intelligence.get("page_role") or "").lower()
    layout_type = str(page_intelligence.get("layout_type") or "").lower()
    line_units = [unit for unit in text_units if unit.get("level") == "line"]
    candidate_rows = []
    for line in line_units:
        text = text_of(line)
        parts = [part.strip() for part in re.split(r"\s{2,}|\t+", text) if part.strip()]
        if len(parts) >= 2:
            candidate_rows.append(_split_line_into_cells(line, parts))
    if len(candidate_rows) >= 2:
        return candidate_rows
    if page_role == "table" or layout_type == "table_dominant":
        phrase_rows = _group_phrases_by_y([unit for unit in text_units if unit.get("level") in {"phrase", "cell"}])
        if len(phrase_rows) >= 2 and max(len(row) for row in phrase_rows) >= 2:
            return phrase_rows
    return []


def _split_line_into_cells(line: dict, parts: list[str]) -> list[dict]:
    bbox = bbox_of(line)
    cells = []
    width = (float(bbox[2]) - float(bbox[0])) / len(parts) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else 0
    for idx, part in enumerate(parts, start=1):
        cb = None
        if width:
            cb = [float(bbox[0]) + width * (idx - 1), float(bbox[1]), float(bbox[0]) + width * idx, float(bbox[3])]
        cells.append({
            "_synthetic_cell": True,
            "source_unit_ids": [line["unit_id"]],
            "text": part,
            "bbox": cb or bbox,
        })
    return cells


def _group_phrases_by_y(phrases: list[dict]) -> list[list[dict]]:
    rows: list[list[dict]] = []
    for phrase in sorted(phrases, key=lambda u: (float((bbox_of(u) or [0, 0, 0, 0])[1]), float((bbox_of(u) or [0, 0, 0, 0])[0]))):
        bbox = bbox_of(phrase) or [0, 0, 0, 0]
        placed = False
        for row in rows:
            rb = bbox_of(row[0]) or [0, 0, 0, 0]
            if abs(float(bbox[1]) - float(rb[1])) <= max(3.0, (float(rb[3]) - float(rb[1])) * 0.6):
                row.append(phrase)
                placed = True
                break
        if not placed:
            rows.append([phrase])
    return [sorted(row, key=lambda u: float((bbox_of(u) or [0, 0, 0, 0])[0])) for row in rows]


def _cell_from_unit(unit: dict, row_idx: int, col_idx: int) -> dict:
    return {
        "source_unit_ids": [unit["unit_id"]],
        "text": text_of(unit),
        "bbox": bbox_of(unit),
    }


def _cell_kind(text: str) -> str:
    text = text.strip()
    if not text:
        return "empty"
    if PATH_RE.search(text):
        return "path"
    if COMMAND_RE.match(text):
        return "command"
    if NUMERIC_RE.fullmatch(text):
        return "numeric"
    if FORMULA_RE.search(text):
        return "formula"
    if len(text.split()) <= 2 and any(ch in text for ch in "{}[]();"):
        return "code"
    return "natural_text"


def _cell_role(row_idx: int, cell_kind: str) -> str:
    if cell_kind in {"command", "path", "code"}:
        return "command_name" if cell_kind == "command" else cell_kind
    if cell_kind in {"numeric", "formula"}:
        return "table_numeric_cell"
    return "table_header_cell" if row_idx == 1 else "table_body_cell"
