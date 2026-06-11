from __future__ import annotations

import re
from statistics import median
from typing import Any

import fitz


def _rect_from_bbox(bbox: Any) -> fitz.Rect:
    if isinstance(bbox, fitz.Rect):
        return fitz.Rect(bbox)
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return fitz.Rect([float(v) for v in bbox])
        except Exception:
            return fitz.Rect(0, 0, 0, 0)
    return fitz.Rect(0, 0, 0, 0)


def _bbox_from_rect(rect: fitz.Rect) -> list[int]:
    return [int(round(rect.x0)), int(round(rect.y0)), int(round(rect.x1)), int(round(rect.y1))]


def _normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _formula_rects_px(formula_regions: list[dict]) -> list[fitz.Rect]:
    rects = []
    for region in formula_regions or []:
        subregions = list((region or {}).get("formula_subregions") or [])
        if not subregions:
            subregions = [{"bbox": (region or {}).get("visual_bbox") or (region or {}).get("bbox")}]
        for sub in subregions:
            rect = _rect_from_bbox((sub or {}).get("bbox"))
            if rect.get_area() <= 0:
                continue
            rect.x0 -= 2
            rect.y0 -= 2
            rect.x1 += 2
            rect.y1 += 2
            rects.append(rect)
    return rects


def _word_rect_px(word, sx: float, sy: float) -> fitz.Rect:
    return fitz.Rect(float(word[0]) * sx, float(word[1]) * sy, float(word[2]) * sx, float(word[3]) * sy)


def _intersects_any(rect: fitz.Rect, rects: list[fitz.Rect]) -> bool:
    cx = (rect.x0 + rect.x1) * 0.5
    cy = (rect.y0 + rect.y1) * 0.5
    for other in rects:
        if other.x0 <= cx <= other.x1 and other.y0 <= cy <= other.y1:
            return True
        if (rect & other).get_area() / max(1.0, rect.get_area()) >= 0.45:
            return True
    return False


def _formula_like_word(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return True
    if re.search(r"[∂∑∏√∞≈≠≤≥±×÷−*/=^_{}]|[α-ωΑ-Ω]", text):
        return True
    if re.fullmatch(r"[(){}\[\],.;:]+", text):
        return True
    return False


def _line_text(words: list[dict]) -> str:
    return _normalize_text(" ".join(word["text"] for word in words if word.get("text")))


def _natural_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-zÀ-ÿ]{3,}", text or ""))


def _line_block(block_id: str, bbox: list[int], text: str, role: str, source_line_indices: list[int]) -> dict:
    style = {"font": "Times-Roman", "size": 10.0, "color": "#000000", "flags": {"serif": True}}
    span = {"bbox": list(bbox), "texte": text, "text": text, "style": dict(style)}
    phrase = {"bbox": list(bbox), "texte": text, "text": text, "spans": [span]}
    line = {"bbox": list(bbox), "line_text": text, "text": text, "phrases": [phrase]}
    return {
        "id": block_id,
        "role": role,
        "source": "native_pdf_logical_rebuilder",
        "source_kind": "logical_pdf_text",
        "bbox": list(bbox),
        "text": text,
        "raw_text": text,
        "line_texts": [text],
        "lines": [line],
        "render_policy": "paragraph_flow" if role == "body" else "anchored_text",
        "style": dict(style),
        "style_attributes": {
            "font_family_primary": "Times-Roman",
            "font_size_pt_median": 10.0,
            "font_size_pt_max": 10.0,
            "color_primary": "#000000",
            "flags_any": {"serif": True},
        },
        "logical_rebuilder": {"source_line_indices": list(source_line_indices)},
    }


def _merge_lines_to_paragraphs(lines: list[dict], page_width: float) -> list[dict]:
    paragraphs = []
    current = []
    current_rect = None
    for line in lines:
        rect = line["rect"]
        text = line["text"]
        if not current:
            current = [line]
            current_rect = fitz.Rect(rect)
            continue
        assert current_rect is not None
        gap = rect.y0 - current_rect.y1
        same_column = abs(rect.x0 - current_rect.x0) <= 26 or (
            max(0.0, min(rect.x1, current_rect.x1) - max(rect.x0, current_rect.x0)) / max(1.0, min(rect.width, current_rect.width)) >= 0.45
        )
        previous_text = current[-1]["text"]
        hard_break = previous_text.endswith((".", ":", ";", "?", "!")) and gap > max(9.0, rect.height * 0.85)
        title_like = rect.width < page_width * 0.45 and _natural_word_count(text) <= 5
        if same_column and gap <= max(16.0, rect.height * 1.25) and not hard_break and not title_like:
            current.append(line)
            current_rect |= rect
        else:
            paragraphs.append({"lines": current, "rect": current_rect})
            current = [line]
            current_rect = fitz.Rect(rect)
    if current and current_rect is not None:
        paragraphs.append({"lines": current, "rect": current_rect})
    return paragraphs


def rebuild_logical_blocks_from_pdf(page_data: dict, pdf_page, *, sx: float = 1.0, sy: float = 1.0) -> tuple[dict, dict]:
    if pdf_page is None or not isinstance(page_data, dict):
        return page_data, {"changed": False, "reason": "no_pdf_page"}
    formula_regions = list((page_data or {}).get("formula_regions") or [])
    if len(formula_regions) < 2:
        return page_data, {"changed": False, "reason": "not_formula_dense"}
    formula_rects = _formula_rects_px(formula_regions)
    if not formula_rects:
        return page_data, {"changed": False, "reason": "no_formula_rects"}
    try:
        words = pdf_page.get_text("words") or []
    except Exception as exc:
        return page_data, {"changed": False, "reason": f"pdf_words_failed:{exc.__class__.__name__}"}
    grouped: dict[tuple[int, int], list[dict]] = {}
    for word in words:
        if len(word) < 7:
            continue
        text = _normalize_text(str(word[4] or ""))
        if not text:
            continue
        rect = _word_rect_px(word, sx, sy)
        if rect.get_area() <= 0 or _intersects_any(rect, formula_rects):
            continue
        if _formula_like_word(text):
            continue
        key = (int(word[5]), int(word[6]))
        grouped.setdefault(key, []).append({"text": text, "rect": rect, "word_no": int(word[7]) if len(word) > 7 else 0})
    lines = []
    for line_index, (_key, items) in enumerate(sorted(grouped.items(), key=lambda kv: (min(w["rect"].y0 for w in kv[1]), min(w["rect"].x0 for w in kv[1])))):
        items = sorted(items, key=lambda item: item["rect"].x0)
        text = _line_text(items)
        if _natural_word_count(text) < 2:
            continue
        rect = fitz.Rect(items[0]["rect"])
        for item in items[1:]:
            rect |= item["rect"]
        lines.append({"text": text, "rect": rect, "source_line_index": line_index})
    if len(lines) < 3:
        return page_data, {"changed": False, "reason": "too_few_logical_lines"}
    page_width = float((page_data.get("dimensions") or {}).get("width") or getattr(pdf_page.rect, "width", 0) * sx)
    paragraphs = _merge_lines_to_paragraphs(lines, page_width)
    rebuilt = []
    for idx, para in enumerate(paragraphs):
        text = _normalize_text(" ".join(line["text"] for line in para["lines"]))
        if _natural_word_count(text) < 2:
            continue
        role = "title" if _natural_word_count(text) <= 5 and para["rect"].width < page_width * 0.55 else "body"
        rebuilt.append(_line_block(f"logical_pdf_text_{idx}", _bbox_from_rect(para["rect"]), text, role, [line["source_line_index"] for line in para["lines"]]))
    if len(rebuilt) < 3:
        return page_data, {"changed": False, "reason": "too_few_rebuilt_blocks"}

    kept = []
    for block in page_data.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        role = str(block.get("role") or "").lower()
        object_class = str(block.get("object_class") or "").lower()
        object_type = str(block.get("object_type") or "").lower()
        render_policy = str(block.get("render_policy") or block.get("render_mode") or "").lower()
        source_kind = str(block.get("source_kind") or "").lower()
        if (
            bool(block.get("formula_region_id"))
            or object_class == "formula"
            or object_type == "formula_region"
            or role in {"equation_inline", "equation_block", "formula"}
            or source_kind == "formula_region_carrier"
            or render_policy in {"skip", "source_overlay", "background_only"}
        ):
            kept.append(block)
    out = dict(page_data)
    out["blocks"] = kept + rebuilt
    out["logical_block_rebuilder"] = {
        "changed": True,
        "source": "pdf_words_minus_formula_regions",
        "kept_non_text_blocks": len(kept),
        "rebuilt_text_blocks": len(rebuilt),
        "input_word_count": len(words),
    }
    return out, {"changed": True, "rebuilt_text_blocks": len(rebuilt), "kept_non_text_blocks": len(kept)}
