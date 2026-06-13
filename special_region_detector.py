import copy
import json
import os
import re
import unicodedata

import fitz


FORMULA_ROLES = {"equation_inline", "equation_block", "formula"}
FORMULA_OBJECT_TYPES = {
    "formula_block",
    "formula_line",
    "formula_equation",
    "formula_region",
    "inline_formula",
    "inline_formula_cluster",
}
CODE_ROLES = {"code", "code_block", "code_line"}
MATH_FONT_HINTS = (
    "math",
    "symbol",
    "mtmi",
    "mtsyn",
    "mtex",
    "mtextra",
    "cmex",
    "cmsy",
    "cmmi",
    "msam",
    "msbm",
    "stix",
)
FORMULA_WORDS = {
    "argmax",
    "argmin",
    "cos",
    "cosh",
    "det",
    "dim",
    "exp",
    "log",
    "max",
    "min",
    "relu",
    "sigmoid",
    "sin",
    "sinh",
    "softmax",
    "sqrt",
    "sum",
    "tan",
    "tanh",
    "target",
    "targetk",
}
NATURAL_SHORT_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "be",
    "by",
    "for",
    "in",
    "is",
    "it",
    "of",
    "or",
    "the",
    "to",
    "we",
}
MATH_SYMBOL_PATTERN = re.compile(r"[∂∑∏∫√∞≈≠≤≥±×÷−∆Ω∗·δµμ=<>*/^_{}()[\]\\|¬→←↔⇒⇔∈∉⊂⊃⊆⊇∧∨∩∪]|[α-ωΑ-Ω]")
CONTROL_FORMULA_CHARS = {"\x02", "\x03", "\x04", "\x05", "\x06", "\x07"}
LIST_MARKER_TEXTS = {
    "■",
    "•",
    "▪",
    "◦",
    "‣",
    "⁃",
    "·",
    "◆",
    "▶",
    "▷",
}
PROSE_MARKERS = {
    "above",
    "calculated",
    "derivative",
    "function",
    "given",
    "gradient",
    "here",
    "input",
    "layer",
    "neuron",
    "number",
    "output",
    "previous",
    "process",
    "represented",
    "shown",
    "softmax",
    "therefore",
    "updated",
    "weight",
}


def _rect_from_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return fitz.Rect(0, 0, 0, 0)
    try:
        return fitz.Rect([float(v) for v in bbox])
    except Exception:
        return fitz.Rect(0, 0, 0, 0)


def _bbox_from_rect(rect):
    return [int(round(rect.x0)), int(round(rect.y0)), int(round(rect.x1)), int(round(rect.y1))]


def _expanded_rect(rect, pad_x=1.0, pad_y=1.0):
    if rect.get_area() <= 0:
        return rect
    return fitz.Rect(rect.x0 - pad_x, rect.y0 - pad_y, rect.x1 + pad_x, rect.y1 + pad_y)


def _intersection_area(left, right):
    return max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0)) * max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))


def _intersection_ratio(left, right):
    inter = _intersection_area(left, right)
    den = max(1.0, min(float(left.get_area()), float(right.get_area())))
    return inter / den


def _horizontal_overlap_ratio(left, right):
    inter = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    return inter / max(1.0, min(left.width, right.width))


def _vertical_overlap_ratio(left, right):
    inter = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    return inter / max(1.0, min(left.height, right.height))


def _rect_to_px(bbox_pt, sx, sy):
    if not isinstance(bbox_pt, (list, tuple)) or len(bbox_pt) != 4:
        return fitz.Rect(0, 0, 0, 0)
    return fitz.Rect(
        float(bbox_pt[0]) * float(sx or 1.0),
        float(bbox_pt[1]) * float(sy or 1.0),
        float(bbox_pt[2]) * float(sx or 1.0),
        float(bbox_pt[3]) * float(sy or 1.0),
    )


def _is_math_font(font_name):
    low = str(font_name or "").lower()
    return any(hint in low for hint in MATH_FONT_HINTS)


def _is_symbol_category(char):
    if not char:
        return False
    try:
        return unicodedata.category(char).startswith("S")
    except Exception:
        return False


def _is_formula_symbol(char):
    if not char:
        return False
    if char in CONTROL_FORMULA_CHARS:
        return True
    if MATH_SYMBOL_PATTERN.search(char):
        return True
    return _is_symbol_category(char) and char not in {"©", "®", "™"}


def _is_list_marker_text(text):
    s = re.sub(r"\s+", "", str(text or ""))
    return len(s) == 1 and s in LIST_MARKER_TEXTS


def _is_natural_word(text):
    return bool(re.fullmatch(r"[A-Za-zÀ-ÿ]{3,}(?:['-][A-Za-zÀ-ÿ]{2,})?", text or ""))


def _is_alpha_slash_word(text):
    parts = str(text or "").split("/")
    if len(parts) != 2:
        return False
    return all(len(part) >= 4 and all(ch.isalpha() for ch in part) for part in parts)


def _token_formula_compatible(token):
    text = re.sub(r"\s+", "", str((token or {}).get("text") or ""))
    if not text:
        return False
    if _is_list_marker_text(text):
        return False
    low = text.lower()
    if low in NATURAL_SHORT_WORDS:
        return False
    if _is_alpha_slash_word(text):
        return False
    if _token_formula_score(token) >= 1.0:
        return True
    if (token or {}).get("has_math_font") or (token or {}).get("has_symbol"):
        return True
    if low in FORMULA_WORDS:
        return True
    if re.fullmatch(r"[A-Za-z]{1,4}\d+[A-Za-z]{0,4}", text):
        return True
    if re.fullmatch(r"[A-Za-z]{1,3}[0-9]{0,4}", text):
        return True
    if re.fullmatch(r"[A-Za-z]{1,3}[ijxyzknm]?", text):
        return True
    if re.fullmatch(r"\d+(?:[.,]\d+)?", text):
        return True
    if re.fullmatch(r"[()\[\]{}<>+\-−=*/∗·^_,.;:|]+", text):
        return True
    if MATH_SYMBOL_PATTERN.search(text):
        return True
    return False


def _token_is_anchor(token):
    text = str((token or {}).get("text") or "")
    if _is_list_marker_text(text):
        return False
    if _is_alpha_slash_word(text):
        return False
    if (token or {}).get("has_symbol") or (token or {}).get("has_control"):
        return True
    if (token or {}).get("has_math_font") and not _is_natural_word(text):
        return True
    return bool(re.search(r"[A-Za-z]\d+[A-Za-z0-9]*|[A-Za-z]\([A-Za-z0-9]\)|\d+\s*[+\-−*/=]", text))


def _tokenize_pdf_line(chars):
    tokens = []
    current = []
    prev = None
    heights = [max(0.0, float((ch.get("rect") or fitz.Rect()).height)) for ch in chars]
    widths = [max(0.0, float((ch.get("rect") or fitz.Rect()).width)) for ch in chars if not (ch.get("c") or "").isspace()]
    avg_height = sum(heights) / max(1, len(heights))
    avg_width = sum(widths) / max(1, len(widths))
    word_gap = max(1.5, min(avg_height * 0.22, avg_width * 0.72))
    for ch in chars:
        c = ch.get("c") or ""
        rect = ch.get("rect") or fitz.Rect(0, 0, 0, 0)
        gap = 0.0 if prev is None else float(rect.x0 - (prev.get("rect") or fitz.Rect()).x1)
        separator = c.isspace() or (prev is not None and gap > word_gap)
        if separator and current:
            tokens.append(_token_from_chars(current))
            current = []
        if not c.isspace():
            current.append(ch)
        prev = ch
    if current:
        tokens.append(_token_from_chars(current))
    return tokens


def _token_from_chars(chars):
    rect = chars[0]["rect"]
    for ch in chars[1:]:
        rect |= ch["rect"]
    text = "".join(ch.get("c") or "" for ch in chars)
    return {
        "text": text,
        "rect": rect,
        "has_math_font": any(ch.get("math_font") for ch in chars),
        "has_symbol": any(ch.get("formula_symbol") for ch in chars),
        "has_control": any((ch.get("c") or "") in CONTROL_FORMULA_CHARS for ch in chars),
        "chars": chars,
    }


def _token_classes(token):
    text = re.sub(r"\s+", "", str((token or {}).get("text") or ""))
    classes = set()
    if not text:
        return classes
    low = text.lower()
    if (token or {}).get("has_control") or any(ch in CONTROL_FORMULA_CHARS for ch in text):
        classes.add("control")
    if (token or {}).get("has_math_font"):
        classes.add("math_font")
    if (token or {}).get("has_symbol") or MATH_SYMBOL_PATTERN.search(text):
        classes.add("symbol")
    if re.fullmatch(r"[+\-−=*/∗·^_,.;:|()[\]{}<>]+", text):
        classes.add("operator")
    if re.fullmatch(r"\d+(?:[.,]\d+)?", text):
        classes.add("number")
    if re.fullmatch(r"[A-Za-z]{1,4}\d+[A-Za-z0-9]*", text):
        classes.add("variable")
    elif re.fullmatch(r"[A-Za-z]{1,3}[ijxyzknm]?", text) and low not in NATURAL_SHORT_WORDS:
        classes.add("variable")
    if low in FORMULA_WORDS:
        classes.add("function")
    if _is_natural_word(text) and low not in FORMULA_WORDS and "variable" not in classes:
        classes.add("natural")
    return classes


def _token_formula_score(token):
    classes = _token_classes(token)
    if not classes:
        return 0.0
    score = 0.0
    if "control" in classes:
        score += 2.0
    if "math_font" in classes:
        score += 1.5
    if "symbol" in classes:
        score += 1.5
    if "operator" in classes:
        score += 1.0
    if "variable" in classes:
        score += 1.0
    if "function" in classes:
        score += 0.8
    if "number" in classes:
        score += 0.35
    if "natural" in classes:
        score -= 1.0
    return score


def _line_formula_signature(tokens, line_text):
    natural_words = [
        word
        for word in re.findall(r"[A-Za-zÀ-ÿ]{3,}(?:['-][A-Za-zÀ-ÿ]{2,})?", line_text or "")
        if word.lower() not in FORMULA_WORDS
    ]
    natural_lows = {word.lower() for word in natural_words}
    scores = [_token_formula_score(token) for token in tokens]
    positive = sum(1 for score in scores if score > 0)
    strong = sum(1 for score in scores if score >= 1.5)
    anchors = sum(1 for token in tokens if _token_is_anchor(token))
    return {
        "natural_words": natural_words,
        "natural_lows": natural_lows,
        "positive": positive,
        "strong": strong,
        "anchors": anchors,
        "score": sum(scores),
        "token_count": len(tokens),
        "has_prose_marker": bool(natural_lows.intersection(PROSE_MARKERS)),
    }


def _line_text_from_chars(chars):
    return "".join(ch.get("c") or "" for ch in chars)


def _line_is_display_formula(tokens, line_text, line_rect, page_width):
    if not tokens:
        return False
    signature = _line_formula_signature(tokens, line_text)
    anchors = int(signature["anchors"])
    math_tokens = int(signature["positive"])
    natural_words = list(signature["natural_words"])
    compact = re.sub(r"\s+", "", line_text or "")
    if not compact:
        return False
    if signature["has_prose_marker"]:
        return False
    has_fraction_or_derivative = bool(re.search(r"∂|δ|∑|∏|∫|√|=", line_text or ""))
    if anchors >= 2 and math_tokens >= max(2, len(tokens) * 0.55) and len(natural_words) <= 3:
        return True
    if has_fraction_or_derivative and len(natural_words) == 0 and math_tokens >= 1:
        return True
    if line_rect.width <= max(80.0, page_width * 0.22) and anchors >= 1 and len(natural_words) == 0:
        return True
    return False


def _candidate_from_pdf_rect(rect, source, confidence, index, formula_text=""):
    rect = _expanded_rect(rect, pad_x=2.0, pad_y=2.0)
    return {
        "rect": rect,
        "special_class": "formula",
        "source": source,
        "block_ids": [],
        "confidence": confidence,
        "preserve_subregions": [
            {
                "id": f"pdf_formula_subregion_{index}",
                "block_id": "",
                "bbox": _bbox_from_rect(rect),
                "policy": "preserve_visual",
                "source": source,
                "text_hint": re.sub(r"[\x00-\x1f]+", " ", formula_text or "").strip(),
            }
        ],
    }


def _chars_in_px_rect(pdf_page, rect_px, sx=1.0, sy=1.0):
    if pdf_page is None or not isinstance(rect_px, fitz.Rect) or rect_px.get_area() <= 0:
        return []
    try:
        raw = pdf_page.get_text("rawdict")
    except Exception:
        return []
    chars = []
    for block in raw.get("blocks", []) or []:
        if block.get("type") not in (None, 0):
            continue
        for line in block.get("lines", []) or []:
            for span in line.get("spans", []) or []:
                font = str(span.get("font") or "")
                size = float(span.get("size") or 0.0)
                math_font = _is_math_font(font)
                span_bbox = span.get("bbox")
                for raw_char in span.get("chars", []) or []:
                    char = raw_char.get("c") or ""
                    bbox = raw_char.get("bbox") or span_bbox
                    char_rect = _rect_to_px(bbox, sx, sy)
                    if not char or char_rect.get_area() <= 0:
                        continue
                    if _intersection_area(char_rect, rect_px) <= 0:
                        continue
                    chars.append(
                        {
                            "c": char,
                            "rect": char_rect,
                            "font": font,
                            "size": size,
                            "math_font": math_font,
                            "formula_symbol": _is_formula_symbol(char),
                        }
                    )
    return sorted(chars, key=lambda ch: (round(ch["rect"].y0, 1), ch["rect"].x0))


def _native_line_chars_in_px_rect(pdf_page, rect_px, sx=1.0, sy=1.0):
    if pdf_page is None or not isinstance(rect_px, fitz.Rect) or rect_px.get_area() <= 0:
        return []
    try:
        raw = pdf_page.get_text("rawdict")
    except Exception:
        return []
    lines = []
    for block in raw.get("blocks", []) or []:
        if block.get("type") not in (None, 0):
            continue
        for line in block.get("lines", []) or []:
            chars = []
            line_rect = None
            for span in line.get("spans", []) or []:
                font = str(span.get("font") or "")
                size = float(span.get("size") or 0.0)
                math_font = _is_math_font(font)
                span_bbox = span.get("bbox")
                for raw_char in span.get("chars", []) or []:
                    char = raw_char.get("c") or ""
                    bbox = raw_char.get("bbox") or span_bbox
                    char_rect = _rect_to_px(bbox, sx, sy)
                    if not char or char_rect.get_area() <= 0:
                        continue
                    if _intersection_area(char_rect, rect_px) <= 0:
                        continue
                    item = {
                        "c": char,
                        "rect": char_rect,
                        "font": font,
                        "size": size,
                        "math_font": math_font,
                        "formula_symbol": _is_formula_symbol(char),
                    }
                    chars.append(item)
                    line_rect = fitz.Rect(char_rect) if line_rect is None else (line_rect | char_rect)
            if chars and line_rect is not None:
                chars.sort(key=lambda ch: ch["rect"].x0)
                lines.append({"chars": chars, "rect": line_rect})
    return sorted(lines, key=lambda item: (item["rect"].y0, item["rect"].x0))


def _cluster_chars_by_baseline(chars):
    if not chars:
        return []
    lines = []
    for ch in chars:
        rect = ch["rect"]
        placed = False
        for line in lines:
            line_rect = line["rect"]
            if _vertical_overlap_ratio(line_rect, rect) >= 0.22 or abs(rect.y0 - line_rect.y0) <= max(3.0, rect.height * 0.45):
                line["chars"].append(ch)
                line["rect"] |= rect
                placed = True
                break
        if not placed:
            lines.append({"chars": [ch], "rect": fitz.Rect(rect)})
    for line in lines:
        line["chars"].sort(key=lambda item: item["rect"].x0)
    return sorted(lines, key=lambda item: (item["rect"].y0, item["rect"].x0))


def _formula_subregion_should_merge(left, right):
    if left.get_area() <= 0 or right.get_area() <= 0:
        return False
    if (left & right).get_area() > 0:
        return True
    horizontal_gap = max(0.0, max(left.x0, right.x0) - min(left.x1, right.x1))
    vertical_gap = max(0.0, max(left.y0, right.y0) - min(left.y1, right.y1))
    same_row = _vertical_overlap_ratio(left, right) >= 0.10 and horizontal_gap <= max(18.0, 1.8 * min(left.height, right.height))
    same_stack = _horizontal_overlap_ratio(left, right) >= 0.10 and vertical_gap <= max(14.0, 1.15 * min(left.height, right.height))
    aligned_stack = (
        vertical_gap <= max(10.0, 0.32 * max(left.height, right.height))
        and abs((left.x0 + left.x1) * 0.5 - (right.x0 + right.x1) * 0.5) <= max(34.0, 0.42 * max(left.width, right.width))
    )
    return bool(same_row or same_stack or aligned_stack)


def _merge_formula_subregions(rects, container_rect=None):
    clean = []
    for rect in rects or []:
        if not isinstance(rect, fitz.Rect) or rect.get_area() <= 4.0:
            continue
        item = _expanded_rect(rect, pad_x=0.8, pad_y=0.8)
        if isinstance(container_rect, fitz.Rect) and container_rect.get_area() > 0:
            item = fitz.Rect(
                max(container_rect.x0, item.x0),
                max(container_rect.y0, item.y0),
                min(container_rect.x1, item.x1),
                min(container_rect.y1, item.y1),
            )
        if item.get_area() > 4.0:
            clean.append(item)
    if not clean:
        return []

    components = [[rect] for rect in sorted(clean, key=lambda item: (item.y0, item.x0))]
    changed = True
    while changed:
        changed = False
        merged = []
        while components:
            current = components.pop(0)
            current_rect = fitz.Rect(current[0])
            for rect in current[1:]:
                current_rect |= rect
            absorbed_indexes = []
            for idx, other in enumerate(components):
                other_rect = fitz.Rect(other[0])
                for rect in other[1:]:
                    other_rect |= rect
                if _formula_subregion_should_merge(current_rect, other_rect):
                    current.extend(other)
                    current_rect |= other_rect
                    absorbed_indexes.append(idx)
                    changed = True
            components = [component for idx, component in enumerate(components) if idx not in absorbed_indexes]
            merged.append(current)
        components = merged

    out = []
    for component in components:
        rect = fitz.Rect(component[0])
        for item in component[1:]:
            rect |= item
        rect = _expanded_rect(rect, pad_x=0.8, pad_y=0.8)
        if isinstance(container_rect, fitz.Rect) and container_rect.get_area() > 0:
            rect = fitz.Rect(
                max(container_rect.x0, rect.x0),
                max(container_rect.y0, rect.y0),
                min(container_rect.x1, rect.x1),
                min(container_rect.y1, rect.y1),
            )
        if rect.get_area() > 4.0:
            out.append(rect)
    return sorted(out, key=lambda item: (item.y0, item.x0))


def _tight_formula_rect_from_pdf(rect_px, pdf_page=None, sx=1.0, sy=1.0):
    refined_rect, _subregions = _tight_formula_geometry_from_pdf(rect_px, pdf_page=pdf_page, sx=sx, sy=sy)
    return refined_rect


def _tight_formula_geometry_from_pdf(rect_px, pdf_page=None, sx=1.0, sy=1.0):
    """Return a glyph-tight formula bbox inside a candidate region.

    PDF text extraction often gives overbroad native blocks around formulas.
    This pass re-reads the source glyphs and keeps only formula-compatible
    tokens. Natural prose words inside the same native block are deliberately
    excluded, so formula crops do not carry stray text.
    """
    if pdf_page is None or not isinstance(rect_px, fitz.Rect) or rect_px.get_area() <= 0:
        return rect_px, []
    native_lines = _native_line_chars_in_px_rect(pdf_page, rect_px, sx=sx, sy=sy)
    if not native_lines:
        chars = _chars_in_px_rect(pdf_page, rect_px, sx=sx, sy=sy)
        native_lines = _cluster_chars_by_baseline(chars)
    if not native_lines:
        return rect_px, []
    kept_rect = None
    subregions = []
    kept_token_count = 0
    for line in native_lines:
        tokens = _tokenize_pdf_line(line["chars"])
        if not tokens:
            continue
        line_text = _line_text_from_chars(line["chars"])
        anchors = sum(1 for token in tokens if _token_is_anchor(token))
        formula_tokens = [token for token in tokens if _token_formula_compatible(token)]
        signature = _line_formula_signature(tokens, line_text)
        natural_words = list(signature["natural_words"])
        if not formula_tokens:
            continue
        # Preserve compact display rows and formula-heavy inline runs. Reject
        # prose rows that only touch the candidate because of bbox overreach.
        if signature["has_prose_marker"]:
            continue
        if signature["score"] < 1.5 and len(natural_words) >= 2:
            continue
        if len(natural_words) >= 4 and anchors < 2:
            continue
        if anchors == 0 and len(formula_tokens) < 2 and len(natural_words) >= 2:
            continue
        preserve_whole_formula_band = _line_is_display_formula(tokens, line_text, line["rect"], rect_px.width)
        preserve_whole_formula_band = preserve_whole_formula_band or (
            anchors >= 2
            and signature["positive"] >= max(2, len(tokens) * 0.50)
            and len(natural_words) <= 2
            and signature["score"] >= 3.0
        )
        if preserve_whole_formula_band:
            run_rect = fitz.Rect(line["rect"])
            kept_token_count += len(formula_tokens)
        else:
            run_rect = None
            for token in formula_tokens:
                token_rect = token["rect"]
                run_rect = fitz.Rect(token_rect) if run_rect is None else (run_rect | token_rect)
                kept_token_count += 1
        if run_rect is not None:
            run_rect = _expanded_rect(run_rect, pad_x=2.0, pad_y=1.8)
            run_rect = fitz.Rect(
                max(rect_px.x0, run_rect.x0),
                max(rect_px.y0, run_rect.y0),
                min(rect_px.x1, run_rect.x1),
                min(rect_px.y1, run_rect.y1),
            )
            if run_rect.get_area() > 4.0:
                subregions.append(run_rect)
            kept_rect = fitz.Rect(run_rect) if kept_rect is None else (kept_rect | run_rect)
    if kept_rect is None or kept_token_count == 0:
        return rect_px, []
    kept_rect = _expanded_rect(kept_rect, pad_x=2.0, pad_y=1.8)
    # Do not let refinement grow the region beyond the original candidate.
    kept_rect = fitz.Rect(
        max(rect_px.x0, kept_rect.x0),
        max(rect_px.y0, kept_rect.y0),
        min(rect_px.x1, kept_rect.x1),
        min(rect_px.y1, kept_rect.y1),
    )
    if kept_rect.get_area() < 4.0:
        return rect_px, []
    if kept_rect.height < 5.0 or (rect_px.height <= 40.0 and kept_rect.height < rect_px.height * 0.45):
        return rect_px, []
    subregions = _merge_formula_subregions(subregions, container_rect=kept_rect)
    if subregions:
        kept_rect = fitz.Rect(subregions[0])
        for subregion in subregions[1:]:
            kept_rect |= subregion
    return kept_rect, subregions


def _pdf_text_for_px_rect(pdf_page, rect_px, sx=1.0, sy=1.0):
    if pdf_page is None or not isinstance(rect_px, fitz.Rect) or rect_px.get_area() <= 0:
        return ""
    try:
        sx = float(sx or 1.0)
        sy = float(sy or 1.0)
        clip = fitz.Rect(rect_px.x0 / sx, rect_px.y0 / sy, rect_px.x1 / sx, rect_px.y1 / sy)
        words = pdf_page.get_text("words", clip=clip) or []
    except Exception:
        return ""
    return re.sub(r"\s+", " ", " ".join(str(word[4]) for word in words if len(word) >= 5)).strip()


def _rect_is_overbroad_natural_text(rect_px, page_width, pdf_page=None, sx=1.0, sy=1.0):
    if not isinstance(rect_px, fitz.Rect) or rect_px.get_area() <= 0:
        return True
    if page_width <= 0 and pdf_page is not None:
        try:
            page_width = float(pdf_page.rect.width) * float(sx or 1.0)
        except Exception:
            page_width = 0.0
    if page_width <= 0 or rect_px.width < max(160.0, page_width * 0.34):
        return False
    text = _pdf_text_for_px_rect(pdf_page, rect_px, sx=sx, sy=sy)
    if not text:
        return False
    natural_words = re.findall(r"[A-Za-zÀ-ÿ]{3,}(?:['-][A-Za-zÀ-ÿ]{2,})?", text)
    natural_lows = {word.lower() for word in natural_words}
    prose_markers = {"and", "are", "been", "find", "first", "given", "here", "need", "shown", "solve", "therefore", "which", "where"}
    math_marks = len(re.findall(r"[∂∑∏√∞≈≠≤≥±×÷−*/=^_{}()[\]\\<>]|[A-Za-z]\s*\^\s*[A-Za-z0-9]", text))
    if len(natural_words) >= 5 and natural_lows.intersection(prose_markers):
        return True
    if len(natural_words) >= 5 and math_marks < max(3, len(natural_words) * 0.45):
        return True
    return False


def _pdf_formula_candidates(pdf_page, sx=1.0, sy=1.0):
    if pdf_page is None:
        return []
    try:
        raw = pdf_page.get_text("rawdict")
    except Exception:
        return []
    if not isinstance(raw, dict):
        return []
    page_width = float(getattr(getattr(pdf_page, "rect", None), "width", 0.0) or 0.0) * float(sx or 1.0)
    candidates = []
    candidate_index = 0
    for block in raw.get("blocks", []) or []:
        if block.get("type") not in (None, 0):
            continue
        for line in block.get("lines", []) or []:
            chars = []
            for span in line.get("spans", []) or []:
                font = str(span.get("font") or "")
                size = float(span.get("size") or 0.0)
                math_font = _is_math_font(font)
                span_bbox = span.get("bbox")
                for raw_char in span.get("chars", []) or []:
                    char = raw_char.get("c") or ""
                    bbox = raw_char.get("bbox") or span_bbox
                    rect = _rect_to_px(bbox, sx, sy)
                    if not char or rect.get_area() <= 0:
                        continue
                    formula_symbol = _is_formula_symbol(char)
                    chars.append(
                        {
                            "c": char,
                            "rect": rect,
                            "font": font,
                            "size": size,
                            "math_font": math_font,
                            "formula_symbol": formula_symbol,
                        }
                    )
            if not chars:
                continue
            chars.sort(key=lambda ch: ch["rect"].x0)
            tokens = _tokenize_pdf_line(chars)
            if not tokens:
                continue
            line_text = _line_text_from_chars(chars)
            line_rect = tokens[0]["rect"]
            for token in tokens[1:]:
                line_rect |= token["rect"]
            if _line_is_display_formula(tokens, line_text, line_rect, page_width):
                candidates.append(_candidate_from_pdf_rect(line_rect, "pdf_glyph_formula_line", 0.94, candidate_index, line_text))
                candidate_index += 1
                continue
            anchor_indexes = [idx for idx, token in enumerate(tokens) if _token_is_anchor(token)]
            consumed = set()
            for anchor_index in anchor_indexes:
                if anchor_index in consumed:
                    continue
                left = anchor_index
                right = anchor_index
                while left > 0 and _token_formula_compatible(tokens[left - 1]):
                    gap = tokens[left]["rect"].x0 - tokens[left - 1]["rect"].x1
                    if gap > max(16.0, tokens[left]["rect"].height * 1.2):
                        break
                    left -= 1
                while right + 1 < len(tokens) and _token_formula_compatible(tokens[right + 1]):
                    gap = tokens[right + 1]["rect"].x0 - tokens[right]["rect"].x1
                    if gap > max(16.0, tokens[right]["rect"].height * 1.2):
                        break
                    right += 1
                run_tokens = tokens[left : right + 1]
                run_text = "".join(token.get("text") or "" for token in run_tokens)
                natural_words = [word for word in re.findall(r"[A-Za-zÀ-ÿ]{3,}", run_text or "") if word.lower() not in FORMULA_WORDS]
                if any(_is_list_marker_text(token.get("text")) for token in run_tokens):
                    if not re.search(r"[∂δ∑∏∫√=]", run_text or ""):
                        continue
                if len(natural_words) > 2 and not re.search(r"∂|δ|∑|∏|∫|√|=", run_text or ""):
                    continue
                rect = run_tokens[0]["rect"]
                for token in run_tokens[1:]:
                    rect |= token["rect"]
                if rect.width < 3 or rect.height < 3:
                    continue
                for idx in range(left, right + 1):
                    consumed.add(idx)
                candidates.append(_candidate_from_pdf_rect(rect, "pdf_glyph_formula_inline", 0.90, candidate_index, run_text))
                candidate_index += 1
    return candidates


def _ai_formula_candidates(page_image):
    model_path = os.environ.get("DOCS_PARSER_SPECIAL_REGION_MODEL", "").strip()
    if not model_path or not os.path.exists(model_path):
        return [], {"available": False, "reason": "no_model_configured"}
    try:
        import onnxruntime as ort  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        return [], {"available": False, "reason": f"onnxruntime_unavailable:{exc.__class__.__name__}"}
    if page_image is None:
        return [], {"available": True, "runtime": "onnxruntime", "model": model_path, "reason": "no_page_image"}
    class_path = os.environ.get("DOCS_PARSER_SPECIAL_REGION_CLASSES", "").strip()
    class_names = ["formula", "equation", "inline_formula", "code", "technical_expression"]
    if class_path and os.path.exists(class_path):
        try:
            payload = json.loads(open(class_path, "r", encoding="utf-8").read())
            if isinstance(payload, list):
                class_names = [str(item) for item in payload]
            elif isinstance(payload, dict) and isinstance(payload.get("names"), list):
                class_names = [str(item) for item in payload["names"]]
        except Exception:
            pass
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        shape = list(input_meta.shape or [])
        target_h = int(shape[2]) if len(shape) == 4 and isinstance(shape[2], int) else 640
        target_w = int(shape[3]) if len(shape) == 4 and isinstance(shape[3], int) else 640
        image = page_image.convert("RGB")
        src_w, src_h = image.size
        resized = image.resize((target_w, target_h))
        arr = np.asarray(resized).astype("float32") / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        outputs = session.run(None, {input_name: arr})
    except Exception as exc:
        return [], {"available": True, "runtime": "onnxruntime", "model": model_path, "reason": f"inference_failed:{exc.__class__.__name__}"}

    detections = []
    for output in outputs:
        data = np.asarray(output)
        if data.ndim == 3:
            data = data[0]
        if data.ndim == 2 and data.shape[0] in {5, 6, 7} and data.shape[1] > data.shape[0]:
            data = data.T
        if data.ndim != 2 or data.shape[1] < 5:
            continue
        for row in data:
            values = row.tolist()
            x0, y0, x1, y1 = [float(v) for v in values[:4]]
            score = float(values[4])
            class_id = 0
            if len(values) > 6:
                class_scores = values[5:]
                class_id = int(max(range(len(class_scores)), key=lambda idx: class_scores[idx]))
                score *= float(class_scores[class_id])
            elif len(values) == 6:
                class_id = int(values[5])
            if score < float(os.environ.get("DOCS_PARSER_SPECIAL_REGION_SCORE", "0.35")):
                continue
            class_name = class_names[class_id] if 0 <= class_id < len(class_names) else "formula"
            class_low = class_name.lower()
            if not any(marker in class_low for marker in ("formula", "equation", "math", "code", "technical")):
                continue
            # Accept both xyxy and cxcywh style YOLO outputs.
            if x1 <= x0 or y1 <= y0:
                cx, cy, w, h = x0, y0, max(0.0, x1), max(0.0, y1)
                x0, y0, x1, y1 = cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
            scale_x = src_w / max(1.0, float(target_w))
            scale_y = src_h / max(1.0, float(target_h))
            rect = fitz.Rect(x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
            if rect.get_area() <= 4:
                continue
            special_class = "code" if "code" in class_low else "formula"
            detections.append(
                {
                    "rect": rect,
                    "special_class": special_class,
                    "source": "onnx_region_detector",
                    "block_ids": [],
                    "confidence": min(0.99, max(0.0, score)),
                    "preserve_subregions": [
                        {
                            "id": f"onnx_special_region_{len(detections)}",
                            "block_id": "",
                            "bbox": _bbox_from_rect(rect),
                            "policy": "preserve_visual",
                            "source": "onnx_region_detector",
                            "class_name": class_name,
                        }
                    ],
                }
            )
    return detections, {"available": True, "runtime": "onnxruntime", "model": model_path, "detections": len(detections)}


def _block_text(block):
    parts = []
    for line in (block or {}).get("lines", []) or []:
        line_text = (line.get("line_text") or "").strip()
        if line_text:
            parts.append(line_text)
            continue
        for phrase in line.get("phrases", []) or []:
            text = (phrase.get("texte") or phrase.get("translated_text") or "").strip()
            if text:
                parts.append(text)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _block_id(block, index):
    return str((block or {}).get("id") or f"block_{index}")


def _block_hints(block):
    return dict((block or {}).get("structure_hints") or {})


def _block_is_formula_candidate(block):
    role = str((block or {}).get("role") or "").strip().lower()
    object_type = str((block or {}).get("object_type") or "").strip().lower()
    object_class = str((block or {}).get("object_class") or "").strip().lower()
    hints = _block_hints(block)
    structural_hint = str(hints.get("structural_role_hint") or "").strip().lower()
    text = _block_text(block)
    math_marks = len(re.findall(r"[∂∑∏√∞≈≠≤≥±×÷−*/=^_{}]|[A-Za-z]\s*\^\s*[A-Za-z0-9]", text or ""))
    words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", text or "")
    if role in FORMULA_ROLES or object_class == "formula" or object_type in FORMULA_OBJECT_TYPES or structural_hint == "formula_block":
        if len(words) >= 5 and math_marks < max(3, len(words)):
            return False
        return True
    if len(words) >= 5:
        return False
    return math_marks >= 3 and math_marks >= max(1, len(words) * 2) and bool(text.strip())


def _span_font_names(block):
    fonts = []
    for line in (block or {}).get("lines", []) or []:
        for phrase in line.get("phrases", []) or []:
            for span in phrase.get("spans", []) or []:
                font = str(((span or {}).get("style") or {}).get("font") or "").strip()
                if font:
                    fonts.append(font)
    return fonts


def _block_is_code_candidate(block):
    role = str((block or {}).get("role") or "").strip().lower()
    object_type = str((block or {}).get("object_type") or "").strip().lower()
    object_class = str((block or {}).get("object_class") or "").strip().lower()
    if role in CODE_ROLES or object_type in CODE_ROLES or object_class == "code":
        return True
    text = _block_text(block)
    if re.search(r"\b(?:sudo|mkdir|python|npm|curl|git)\b", text or "", flags=re.IGNORECASE):
        return True
    sql_hits = re.findall(r"\b(?:SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)\b", text or "")
    if len(sql_hits) >= 2 or any(hit in {"SELECT", "INSERT", "UPDATE", "DELETE"} for hit in sql_hits):
        return True
    natural_words = re.findall(r"[A-Za-zÀ-ÿ]{3,}(?:['-][A-Za-zÀ-ÿ]{2,})?", text or "")
    if len(natural_words) >= 5 and not re.search(r"(?:/[A-Za-z0-9_.-]+){2,}|[A-Za-z0-9_.-]+\.(?:py|js|ts|json|sql|yaml|yml)", text or ""):
        return False
    return bool(re.search(r"[A-Za-z_][A-Za-z0-9_]*\([^)]{0,80}\)|(?:/[A-Za-z0-9_.-]+){2,}|[A-Za-z0-9_.-]+\.(?:py|js|ts|json|sql|yaml|yml)", text or ""))


def _is_translatable_text_block(block):
    if not isinstance(block, dict) or _block_is_formula_candidate(block) or _block_is_code_candidate(block):
        return False
    role = str(block.get("role") or "").strip().lower()
    if role in {"page_header", "page_footer", "header", "footer", "diagram_label"}:
        return False
    text = _block_text(block)
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text or ""):
        return False
    natural_words = re.findall(r"[A-Za-zÀ-ÿ]{3,}(?:['-][A-Za-zÀ-ÿ]{2,})?", text or "")
    if len(natural_words) < 3:
        return False
    math_marks = len(re.findall(r"[∂∑∏√∞≈≠≤≥±×÷−*/=^_{}()[\]<>]", text or ""))
    return math_marks < max(3, len(natural_words))


def _should_merge_regions(left, right):
    if left.get_area() <= 0 or right.get_area() <= 0:
        return False
    if (left & right).get_area() > 0:
        return True
    horizontal_gap = max(0.0, max(left.x0, right.x0) - min(left.x1, right.x1))
    vertical_gap = max(0.0, max(left.y0, right.y0) - min(left.y1, right.y1))
    if _vertical_overlap_ratio(left, right) >= 0.22 and horizontal_gap <= max(28.0, 0.18 * max(left.width, right.width)):
        return True
    if _horizontal_overlap_ratio(left, right) >= 0.18 and vertical_gap <= max(6.0, 0.10 * max(left.height, right.height)):
        return True
    return False


def _text_should_attach(region_rect, block_rect):
    if region_rect.get_area() <= 0 or block_rect.get_area() <= 0:
        return False
    if _intersection_ratio(region_rect, block_rect) >= 0.08 or _intersection_ratio(block_rect, region_rect) >= 0.08:
        return True
    horizontal_gap = max(0.0, max(region_rect.x0, block_rect.x0) - min(region_rect.x1, block_rect.x1))
    vertical_gap = max(0.0, max(region_rect.y0, block_rect.y0) - min(region_rect.y1, block_rect.y1))
    if _horizontal_overlap_ratio(region_rect, block_rect) >= 0.25 and vertical_gap <= max(16.0, 0.35 * max(region_rect.height, block_rect.height)):
        return True
    if _vertical_overlap_ratio(region_rect, block_rect) >= 0.25 and horizontal_gap <= max(44.0, 0.20 * max(region_rect.width, block_rect.width)):
        return True
    return False


def _merge_candidates(candidates):
    components = []
    for candidate in candidates:
        rect = candidate["rect"]
        for component in components:
            if any(_should_merge_regions(rect, other["rect"]) for other in component):
                component.append(candidate)
                break
        else:
            components.append([candidate])

    changed = True
    while changed:
        changed = False
        merged = []
        while components:
            current = components.pop(0)
            current_rects = [item["rect"] for item in current]
            absorbed = []
            for idx, other in enumerate(components):
                if any(_should_merge_regions(left, right) for left in current_rects for right in [item["rect"] for item in other]):
                    current.extend(other)
                    current_rects = [item["rect"] for item in current]
                    absorbed.append(idx)
                    changed = True
            components = [component for idx, component in enumerate(components) if idx not in absorbed]
            merged.append(current)
        components = merged
    return components


def detect_special_regions(page_data, page_image=None, pdf_page=None, sx=1.0, sy=1.0):
    work = copy.deepcopy(page_data or {})
    blocks = list(work.get("blocks") or [])
    dims = dict(work.get("dimensions") or {})
    try:
        page_width = float(dims.get("width") or 0.0)
    except Exception:
        page_width = 0.0
    candidates = []
    pdf_candidates = _pdf_formula_candidates(pdf_page, sx=sx, sy=sy)
    candidates.extend(pdf_candidates)
    ai_candidates, ai_info = _ai_formula_candidates(page_image)
    candidates.extend(ai_candidates)

    layout_ai = work.get("layout_ai_structure") or {}
    for idx, region in enumerate(layout_ai.get("formula_regions") or []):
        rect = _rect_from_bbox((region or {}).get("bbox"))
        if rect.get_area() > 0:
            candidates.append({"rect": rect, "special_class": "formula", "source": "layout_ai_formula_region", "block_ids": [], "confidence": 0.78})

    for idx, block in enumerate(blocks):
        rect = _rect_from_bbox((block or {}).get("bbox"))
        if rect.get_area() <= 0:
            continue
        block_id = _block_id(block, idx)
        if _block_is_formula_candidate(block):
            candidates.append({"rect": rect, "special_class": "formula", "source": "block_formula_signal", "block_ids": [block_id], "confidence": 0.86})
        elif _block_is_code_candidate(block):
            candidates.append({"rect": rect, "special_class": "code", "source": "block_code_signal", "block_ids": [block_id], "confidence": 0.80})

    special_regions = []
    for region_index, component in enumerate(_merge_candidates(candidates)):
        visual_rect = None
        formula_block_ids = []
        code_block_ids = []
        sources = set()
        confidence = 0.0
        special_class_votes = []
        preserve_subregions = []
        for item in component:
            item_rect = item["rect"]
            sources.add(str(item.get("source") or ""))
            confidence = max(confidence, float(item.get("confidence") or 0.0))
            special_class_votes.append(str(item.get("special_class") or "special"))
            is_overbroad_formula_text = item.get("special_class") != "code" and _rect_is_overbroad_natural_text(
                item_rect,
                page_width,
                pdf_page=pdf_page,
                sx=sx,
                sy=sy,
            )
            for block_id in item.get("block_ids") or []:
                if item.get("special_class") == "code":
                    code_block_ids.append(block_id)
                else:
                    formula_block_ids.append(block_id)
                if is_overbroad_formula_text:
                    continue
                preserve_subregions.append({"block_id": block_id, "bbox": _bbox_from_rect(item_rect), "policy": "preserve_visual"})
                visual_rect = item_rect if visual_rect is None else (visual_rect | item_rect)
            for subregion in item.get("preserve_subregions") or []:
                if isinstance(subregion, dict):
                    sub_rect = _rect_from_bbox(subregion.get("bbox"))
                    if item.get("special_class") != "code" and _rect_is_overbroad_natural_text(
                        sub_rect,
                        page_width,
                        pdf_page=pdf_page,
                        sx=sx,
                        sy=sy,
                    ):
                        continue
                    preserve_subregions.append(dict(subregion))
                    if sub_rect.get_area() > 0:
                        visual_rect = sub_rect if visual_rect is None else (visual_rect | sub_rect)
        if not preserve_subregions:
            visual_rect = component[0]["rect"]
            for item in component[1:]:
                visual_rect |= item["rect"]
            preserve_subregions.append({"block_id": "", "bbox": _bbox_from_rect(visual_rect), "policy": "preserve_visual"})
        elif visual_rect is None:
            visual_rect = component[0]["rect"]
        special_class = "code" if special_class_votes.count("code") > special_class_votes.count("formula") else "formula"
        if special_class == "formula":
            refined_rect, refined_subregions = _tight_formula_geometry_from_pdf(visual_rect, pdf_page=pdf_page, sx=sx, sy=sy)
            if refined_rect.get_area() > 0:
                visual_rect = refined_rect
                refined_subregions = refined_subregions or [refined_rect]
                preserve_subregions = [
                    {
                        "block_id": "",
                        "bbox": _bbox_from_rect(sub_rect),
                        "policy": "preserve_visual",
                        "source": "formula_glyph_tightener",
                    }
                    for sub_rect in refined_subregions
                ]
        special_regions.append(
            {
                "id": f"special_region_{region_index}",
                "region_type": special_class,
                "special_class": special_class,
                "object_type": special_class,
                "object_class": special_class,
                "visual_bbox": _bbox_from_rect(visual_rect),
                "bbox": _bbox_from_rect(visual_rect),
                "preserve_subregions": preserve_subregions,
                "formula_block_ids": sorted(set(formula_block_ids)),
                "code_block_ids": sorted(set(code_block_ids)),
                "translatable_block_ids": [],
                "text_subregions": [],
                "render_policy": "preserve_source_region",
                "translation_policy": "preserve_visual_region",
                "protected_visual": True,
                "preserve_original_pixels": True,
                "skip_translation": True,
                "skip_text_reconstruction": True,
                "detection_source": "+".join(sorted(s for s in sources if s)) or "cpu_heuristic",
                "confidence": round(confidence, 3),
            }
        )

    work["special_regions"] = special_regions
    return work, {
        "changed": bool(special_regions),
        "special_region_count": len(special_regions),
        "detector": "cpu_pdf_glyph_heuristic_v3",
        "pdf_glyph_candidate_count": len(pdf_candidates),
        "ai": ai_info,
    }
