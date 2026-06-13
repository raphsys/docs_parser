from __future__ import annotations

import re

from .common import bbox_of, bbox_union, eligible_text_units, reading_order, role_of, text_of

INDEX_RE = re.compile(r"^\s*(?P<head>[^,]{1,100}),\s*(?P<refs>(?:\d+(?:[-–]\d+)?[,;\s]*)+)\s*$")
SUBENTRY_RE = re.compile(r"^\s{2,}(?P<text>[^,]{1,100}),\s*(?P<refs>(?:\d+(?:[-–]\d+)?[,;\s]*)+)\s*$")


def build_index_entries(units: list[dict], *, page_intelligence: dict | None = None) -> list[dict]:
    page_role = str((page_intelligence or {}).get("page_role") or "").lower()
    rows = _index_rows(units)
    output = []
    current_entry: dict | None = None
    counter = 1
    for unit in rows:
        text = text_of(unit)
        role = role_of(unit)
        submatch = SUBENTRY_RE.match(_raw_text(unit))
        match = INDEX_RE.match(text)
        if page_role != "index" and not role.startswith("index") and not match and not submatch:
            continue
        if submatch and current_entry:
            subentry = {
                "text": submatch.group("text").strip(),
                "page_refs": _refs(submatch.group("refs")),
                "source_unit_ids": [unit["unit_id"]],
                "bbox": bbox_of(unit),
            }
            current_entry.setdefault("subentries", []).append(subentry)
            current_entry["source_unit_ids"] = list(dict.fromkeys(current_entry.get("source_unit_ids", []) + [unit["unit_id"]]))
            current_entry["bbox"] = bbox_union([current_entry.get("bbox"), bbox_of(unit)])
            continue

        head = text.strip()
        refs: list[str] = []
        if match:
            head = match.group("head").strip()
            refs = _refs(match.group("refs"))
        current_entry = {
            "logical_unit_id": f"index_entry_{counter:04d}",
            "type": "index_entry",
            "head_term": head,
            "subentries": [],
            "page_refs": refs,
            "source_unit_ids": [unit["unit_id"]],
            "bbox": bbox_of(unit),
            "parse_strategy": "index_line",
        }
        output.append(current_entry)
        counter += 1
    return output


def _index_rows(units: list[dict]) -> list[dict]:
    text_units = eligible_text_units(units)
    lines = [unit for unit in text_units if unit.get("level") == "line"]
    if lines:
        return sorted(lines, key=reading_order)
    phrases = [unit for unit in text_units if unit.get("level") == "phrase"]
    return sorted(phrases, key=reading_order)


def _raw_text(unit: dict) -> str:
    return str((unit.get("content") or {}).get("text") or "")


def _refs(text: str) -> list[str]:
    return [ref for ref in re.split(r"[,;]\s*|\s+", text.strip()) if ref]
