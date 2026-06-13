"""Sentence boundary analysis for translation units."""

from __future__ import annotations

import re
from collections import defaultdict

from .text_utils import normalize_spaces, word_count


ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "sr.",
    "jr.",
    "st.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
    "fig.",
    "figs.",
    "eq.",
    "eqs.",
    "sec.",
    "secs.",
    "ch.",
    "vol.",
    "no.",
    "pp.",
    "p.",
    "al.",
}


def annotate_sentence_boundaries(units: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in units:
        grouped[item.get("block_id") or item.get("parent_id") or "__page__"].append(item)

    for group_units in grouped.values():
        group_units.sort(key=lambda item: item.get("reading_order_index") or 0)
        previous_open = False
        for idx, item in enumerate(group_units):
            text = normalize_spaces(item.get("source_text"))
            next_item = group_units[idx + 1] if idx + 1 < len(group_units) else None
            terminal = terminal_punctuation(text)
            ends_sentence = sentence_end(text)
            atomic_label = is_atomic_text_unit(item, text)
            break_type = _break_type(item, next_item, ends_sentence, atomic_label)
            starts_sentence = idx == 0 or not previous_open or starts_sentence_like(text)
            continues_after = bool(
                next_item
                and not ends_sentence
                and not atomic_label
                and break_type in {"same_line_continuation", "soft_wrap"}
                and _same_paragraph_like(item, next_item)
            )
            item["sentence"] = {
                "is_sentence_start": bool(starts_sentence),
                "is_sentence_end": bool(ends_sentence or atomic_label or next_item is None),
                "continues_from_previous": bool(idx > 0 and previous_open),
                "continues_to_next": continues_after,
                "terminal_punctuation": terminal,
                "word_count": word_count(text),
                "char_count": len(text),
                "boundary_type": "atomic_label" if atomic_label else ("terminal" if ends_sentence else "open"),
                "break_type": break_type,
                "is_multiline_phrase": _is_multiline(item),
                "start_reason": "first_or_after_boundary" if starts_sentence else "continuation",
                "end_reason": (
                    "atomic_label" if atomic_label
                    else "terminal_punctuation" if ends_sentence
                    else "end_of_group" if next_item is None
                    else "soft_wrap" if break_type == "soft_wrap"
                    else "continues_next"
                ),
            }
            previous_open = bool(continues_after)
    return units


def terminal_punctuation(text: str) -> str | None:
    s = normalize_spaces(text)
    if not s:
        return None
    m = re.search(r"([.!?…]+)[\"')\]\}»]*$", s)
    return m.group(1) if m else None


def looks_like_abbreviation(text: str) -> bool:
    s = normalize_spaces(text)
    if not s:
        return False
    tail = s.split()[-1].strip("\"')]}»").lower()
    if tail in ABBREVIATIONS:
        return True
    if re.fullmatch(r"(?:[A-Z]\.){1,4}", tail, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"[A-Z][a-z]{0,3}\.", tail) and tail[:-1] in {"jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}:
        return True
    return False


def starts_sentence_like(text: str) -> bool:
    s = normalize_spaces(text)
    if not s:
        return False
    return bool(re.match(r"^[A-ZÀ-ÖØ-Þ0-9«“\"'(\[]", s))


def sentence_end(text: str) -> bool:
    punct = terminal_punctuation(text)
    return bool(punct and not looks_like_abbreviation(text))


def is_atomic_text_unit(item: dict, text: str) -> bool:
    role = str(item.get("role") or "").lower()
    object_type = str(item.get("object_type") or "").lower()
    strategy = str(item.get("strategy") or "").lower()
    wc = word_count(text)
    if strategy in {"exact_preserve", "keep_original", "background_only"}:
        return True
    label_roles = {"title", "section_heading", "caption", "figure_caption", "diagram_label", "header", "footer"}
    label_types = {"short_label", "diagram_label", "chart_label", "formula_label", "table_cell_text"}
    if role in label_roles:
        return True
    if object_type in {"citation", "reference_link", "formula", "code_visible", "chart_tick_label"}:
        return True
    return bool((role in label_roles or object_type in label_types) and wc <= 6 and not sentence_end(text))


def _break_type(item: dict, next_item: dict | None, ends_sentence: bool, atomic_label: bool) -> str:
    if atomic_label:
        return "atomic_label"
    if ends_sentence:
        return "terminal"
    if not next_item:
        return "end_of_block"
    y2 = _bbox_y2(item.get("bbox"))
    next_y1 = _bbox_y1(next_item.get("bbox"))
    if y2 is not None and next_y1 is not None and next_y1 > y2 + 6:
        return "soft_wrap"
    return "same_line_continuation"


def _same_paragraph_like(item: dict, next_item: dict) -> bool:
    if _is_list_or_structural_boundary(item) or _is_list_or_structural_boundary(next_item):
        return False
    text = normalize_spaces(item.get("source_text"))
    next_text = normalize_spaces(next_item.get("source_text"))
    if not text or not next_text:
        return False
    if _looks_like_short_list_item(text) and _looks_like_short_list_item(next_text):
        return False
    x0 = _bbox_x0(item.get("bbox"))
    next_x0 = _bbox_x0(next_item.get("bbox"))
    y1 = _bbox_y1(item.get("bbox"))
    y2 = _bbox_y2(item.get("bbox"))
    next_y1 = _bbox_y1(next_item.get("bbox"))
    if x0 is not None and next_x0 is not None and next_x0 > x0 + 24:
        return False
    if y1 is not None and y2 is not None and next_y1 is not None:
        line_height = max(1.0, y2 - y1)
        vertical_gap = next_y1 - y2
        if vertical_gap > max(18.0, line_height * 1.2):
            return False
    if starts_sentence_like(next_text) and word_count(text) <= 5 and not text.endswith(("-", "–", "—", ",")):
        return False
    return True


def _is_list_or_structural_boundary(item: dict) -> bool:
    role = str(item.get("role") or "").lower()
    object_type = str(item.get("object_type") or "").lower()
    text = normalize_spaces(item.get("source_text"))
    if role in {"list_item", "bullet", "table_cell", "section_heading", "title", "caption"}:
        return True
    if object_type in {"table_cell_text", "chart_label", "diagram_label", "short_label"}:
        return True
    return bool(re.match(r"^(?:[-*•▪■]|\d+[.)]|[A-Za-z][.)])\s+", text))


def _looks_like_short_list_item(text: str) -> bool:
    s = normalize_spaces(text)
    if re.match(r"^(?:[-*•▪■]|\d+[.)]|[A-Za-z][.)])\s+", s):
        return True
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", s)
    if len(words) > 5:
        return False
    return bool(words and s[:1].isupper() and not terminal_punctuation(s))


def _is_multiline(item: dict) -> bool:
    source_ids = item.get("source_unit_ids") or []
    return bool(len(source_ids) > 1 or item.get("level") in {"semantic_phrase", "semantic_group"})


def _bbox_y1(bbox: object) -> float | None:
    return float(bbox[1]) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None


def _bbox_y2(bbox: object) -> float | None:
    return float(bbox[3]) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None


def _bbox_x0(bbox: object) -> float | None:
    return float(bbox[0]) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None
