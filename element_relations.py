import re


SCHEMA_VERSION = "element_relations.v1"

_TERMINAL_PUNCT_RE = re.compile(r"[.!?…:;]\s*$")
_LOWER_START_RE = re.compile(r"^[a-zà-ÿ]")
_UPPER_START_RE = re.compile(r"^[A-ZÀ-ß]")
_LIST_MARKER_RE = re.compile(r"^(?:[•▪◦·\-\*]|\d+[.)]|[A-Za-z][.)])\s*")


def enrich_element_relations(page_data):
    if not isinstance(page_data, dict):
        return page_data

    layout_direction = str(page_data.get("layout_direction") or "ltr").strip().lower() or "ltr"
    blocks = [block for block in (page_data.get("blocks") or []) if isinstance(block, dict)]
    block_relations = []
    flat_relations = []

    for block in blocks:
        enriched = _enrich_block_relations(block, layout_direction=layout_direction)
        if not enriched:
            continue
        block["element_relations"] = enriched
        block_relations.append(enriched)
        flat_relations.extend(list(enriched.get("pair_relations") or []))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "element_type": "phrase",
        "layout_direction": layout_direction,
        "block_relations": block_relations,
        "flat_relations": flat_relations,
    }
    page_data["element_relations"] = payload
    page_data.setdefault("layout", {})
    page_data["layout"]["element_relations"] = payload
    page_data["layout"]["element_relations_version"] = SCHEMA_VERSION
    return page_data


def _enrich_block_relations(block, layout_direction="ltr"):
    phrases = _ordered_phrases_in_block(block, layout_direction=layout_direction)
    if not phrases:
        return {
            "schema_version": SCHEMA_VERSION,
            "block_id": str(block.get("id") or ""),
            "element_type": "phrase",
            "reading_order": [],
            "pair_relations": [],
        }

    for idx, item in enumerate(phrases, start=1):
        phrase = item["phrase"]
        phrase_id = item["phrase_id"]
        phrase["element_relation_node"] = {
            "phrase_id": phrase_id,
            "reading_order_index": idx,
            "line_index": int(item["line_index"]),
        }

    relations = []
    for idx in range(len(phrases) - 1):
        prev_item = phrases[idx]
        next_item = phrases[idx + 1]
        relation = _infer_pair_relation(
            block=block,
            previous=prev_item,
            current=next_item,
            sequence_index=idx + 1,
            layout_direction=layout_direction,
        )
        relations.append(relation)
        prev_item["phrase"]["flow_to_next_phrase"] = relation
        next_item["phrase"]["flow_from_previous_phrase"] = relation

    return {
        "schema_version": SCHEMA_VERSION,
        "block_id": str(block.get("id") or ""),
        "element_type": "phrase",
        "layout_direction": layout_direction,
        "reading_order": [item["phrase_id"] for item in phrases],
        "pair_relations": relations,
    }


def _ordered_phrases_in_block(block, layout_direction="ltr"):
    items = []
    block_id = str(block.get("id") or "block")
    lines = [line for line in (block.get("lines") or []) if isinstance(line, dict)]
    lines = sorted(
        lines,
        key=lambda line: (
            float(((line.get("bbox") or [0, 0, 0, 0])[1])),
            float(((line.get("bbox") or [0, 0, 0, 0])[0])),
        ),
    )
    for line_idx, line in enumerate(lines):
        phrases = [phrase for phrase in (line.get("phrases") or []) if isinstance(phrase, dict)]
        phrases = sorted(phrases, key=lambda phrase: _phrase_sort_key(phrase, layout_direction=layout_direction))
        for phrase_idx, phrase in enumerate(phrases):
            phrase_id = str(
                phrase.get("unit_id")
                or phrase.get("id")
                or f"{block_id}:line:{line_idx}:phrase:{phrase_idx}"
            )
            items.append(
                {
                    "phrase": phrase,
                    "phrase_id": phrase_id,
                    "phrase_index": phrase_idx,
                    "line": line,
                    "line_index": int(line.get("line_index", line_idx) or line_idx),
                    "line_bbox": _bbox(line.get("bbox")),
                    "bbox": _bbox(phrase.get("bbox")),
                    "text": _clean_text(phrase.get("text") or phrase.get("texte") or ""),
                }
            )
    return items


def _phrase_sort_key(phrase, layout_direction="ltr"):
    bbox = _bbox(phrase.get("bbox"))
    if not bbox:
        return (0.0, 0.0)
    if layout_direction == "rtl":
        return (bbox[1], -bbox[2])
    return (bbox[1], bbox[0])


def _infer_pair_relation(block, previous, current, sequence_index, layout_direction="ltr"):
    prev_bbox = previous["bbox"] or [0.0, 0.0, 0.0, 0.0]
    curr_bbox = current["bbox"] or [0.0, 0.0, 0.0, 0.0]
    block_bbox = _bbox(block.get("bbox")) or [0.0, 0.0, 0.0, 0.0]

    same_line = previous["line_index"] == current["line_index"]
    line_delta = int(current["line_index"]) - int(previous["line_index"])
    inline_gap = float(curr_bbox[0]) - float(prev_bbox[2]) if same_line else 0.0
    vertical_gap = max(0.0, float(curr_bbox[1]) - float(prev_bbox[3]))
    indent_delta = float(curr_bbox[0]) - float(prev_bbox[0])

    prev_text = previous["text"]
    curr_text = current["text"]
    prev_terminal = bool(_TERMINAL_PUNCT_RE.search(prev_text))
    prev_hyphen = bool(prev_text.endswith("-"))
    curr_lower = bool(_LOWER_START_RE.match(curr_text))
    curr_upper = bool(_UPPER_START_RE.match(curr_text))
    curr_marker = bool(_LIST_MARKER_RE.match(curr_text) or current["phrase"].get("leading_marker"))
    curr_hard_break = bool(current["phrase"].get("hard_break_before", False) or current["line"].get("hard_break_before", False))
    prev_break_after = bool(previous["phrase"].get("line_break_after", True))
    same_style = _style_signature(previous["phrase"]) == _style_signature(current["phrase"])
    line_height_ref = max(1.0, float((prev_bbox[3] - prev_bbox[1]) or 0.0), float((curr_bbox[3] - curr_bbox[1]) or 0.0))
    wrapped_continuation = (
        not same_line
        and line_delta == 1
        and not prev_terminal
        and not curr_marker
        and (curr_lower or prev_hyphen or not curr_upper)
        and abs(indent_delta) <= max(18.0, line_height_ref * 1.2)
    )
    inline_continuation = same_line and not curr_marker and inline_gap <= max(24.0, line_height_ref * 1.2)

    if inline_continuation:
        visual_relation = "continues_inline"
    elif wrapped_continuation:
        visual_relation = "continues_wrapped_line"
    else:
        visual_relation = "new_structural_unit"

    if prev_hyphen:
        logical_relation = "same_token_continuation"
    elif inline_continuation and not prev_terminal:
        logical_relation = "same_sentence_continuation"
    elif wrapped_continuation:
        logical_relation = "same_paragraph_continuation"
    elif curr_marker:
        logical_relation = "new_list_item"
    elif prev_terminal:
        logical_relation = "new_sentence_or_unit"
    elif curr_hard_break:
        logical_relation = "new_structural_unit"
    else:
        logical_relation = "uncertain"

    continuation = visual_relation in {"continues_inline", "continues_wrapped_line"}
    confidence = _relation_confidence(
        continuation=continuation,
        same_line=same_line,
        wrapped_continuation=wrapped_continuation,
        same_style=same_style,
        prev_terminal=prev_terminal,
        curr_marker=curr_marker,
        curr_hard_break=curr_hard_break,
        inline_gap=inline_gap,
        vertical_gap=vertical_gap,
        line_height_ref=line_height_ref,
    )
    ai_review_required = confidence < 0.72 or logical_relation == "uncertain"

    return {
        "relation_id": f"{str(block.get('id') or 'block')}::phrase_flow::{sequence_index}",
        "block_id": str(block.get("id") or ""),
        "sequence_index": int(sequence_index),
        "source_phrase_id": previous["phrase_id"],
        "target_phrase_id": current["phrase_id"],
        "source_line_index": int(previous["line_index"]),
        "target_line_index": int(current["line_index"]),
        "visual_relation": visual_relation,
        "logical_relation": logical_relation,
        "continuation": bool(continuation),
        "confidence": round(confidence, 4),
        "ai_review_required": bool(ai_review_required),
        "understanding": {
            "mode": "semantic_heuristics",
            "external_model_used": False,
            "ai_ready": True,
        },
        "signals": {
            "same_line": bool(same_line),
            "line_delta": int(line_delta),
            "inline_gap_px": round(inline_gap, 4),
            "vertical_gap_px": round(vertical_gap, 4),
            "indent_delta_px": round(indent_delta, 4),
            "previous_terminal_punctuation": bool(prev_terminal),
            "previous_ends_hyphen": bool(prev_hyphen),
            "current_starts_lowercase": bool(curr_lower),
            "current_starts_uppercase": bool(curr_upper),
            "current_has_list_marker": bool(curr_marker),
            "current_hard_break_before": bool(curr_hard_break),
            "previous_line_break_after": bool(prev_break_after),
            "same_style_signature": bool(same_style),
            "layout_direction": layout_direction,
            "block_left_px": round(block_bbox[0], 4),
            "block_right_px": round(block_bbox[2], 4),
        },
        "text": {
            "source": prev_text[:240],
            "target": curr_text[:240],
        },
    }


def _relation_confidence(
    continuation,
    same_line,
    wrapped_continuation,
    same_style,
    prev_terminal,
    curr_marker,
    curr_hard_break,
    inline_gap,
    vertical_gap,
    line_height_ref,
):
    score = 0.55
    if continuation:
        score += 0.18
    if same_line:
        score += 0.16
    if wrapped_continuation:
        score += 0.14
    if same_style:
        score += 0.08
    if prev_terminal:
        score -= 0.16
    if curr_marker:
        score -= 0.18
    if curr_hard_break and not wrapped_continuation:
        score -= 0.12
    if same_line and inline_gap > max(24.0, line_height_ref * 1.2):
        score -= 0.12
    if not same_line and vertical_gap > max(20.0, line_height_ref * 1.8):
        score -= 0.1
    return max(0.0, min(0.99, score))


def _style_signature(node):
    style = (node or {}).get("style") or (node or {}).get("resolved_style") or {}
    if not style and isinstance((node or {}).get("spans"), list):
        for span in (node.get("spans") or []):
            candidate = (span or {}).get("style") or (span or {}).get("resolved_style") or {}
            if candidate:
                style = candidate
                break
    flags = style.get("flags") or {}
    return (
        str(style.get("font") or "").strip().lower(),
        round(float(style.get("size", style.get("font_size_px", 0.0)) or 0.0), 3),
        str(style.get("color") or "").strip().lower(),
        bool(flags.get("bold") or style.get("bold")),
        bool(flags.get("italic") or style.get("italic")),
    )


def _bbox(bbox):
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


def _clean_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()
