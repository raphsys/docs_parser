import math
import re

from element_relations_ai import get_element_relations_ai_enricher


SCHEMA_VERSION = "positioning_policy.v1"

_HORIZONTAL_HYPOTHESES = {
    "start": "this fragment should stay attached to the start side of its container when translated",
    "end": "this fragment should stay attached to the end side of its container when translated",
    "center": "this fragment should remain centered in its container when translated",
}

_VERTICAL_HYPOTHESES = {
    "top": "this fragment should stay attached to the top side of its container when translated",
    "bottom": "this fragment should stay attached to the bottom side of its container when translated",
    "middle": "this fragment should remain vertically centered in its container when translated",
}

_ROLE_HYPOTHESES = {
    "flow_text": "this fragment behaves like running text in a reading flow",
    "centered_title": "this fragment behaves like a centered title or heading",
    "end_value": "this fragment behaves like an end aligned value or counter",
    "attached_label": "this fragment behaves like a short attached label",
}


def enrich_positioning_policy(page_data):
    if not isinstance(page_data, dict):
        return page_data

    layout_direction = str(page_data.get("layout_direction") or "ltr").strip().lower() or "ltr"
    blocks = [block for block in (page_data.get("blocks") or []) if isinstance(block, dict)]
    block_payloads = []
    flat_policies = []

    for block in blocks:
        payload = _enrich_block_policy(block, layout_direction=layout_direction)
        if not payload:
            continue
        block["positioning_policy"] = payload
        block_payloads.append(payload)
        flat_policies.extend(list(payload.get("phrase_policies") or []))

    page_payload = {
        "schema_version": SCHEMA_VERSION,
        "layout_direction": layout_direction,
        "block_policies": block_payloads,
        "flat_policies": flat_policies,
    }
    page_data["positioning_policy"] = page_payload
    page_data["positioning_policy_flat"] = flat_policies
    page_data.setdefault("layout", {})
    page_data["layout"]["positioning_policy"] = page_payload
    page_data["layout"]["positioning_policy_flat"] = flat_policies
    page_data["layout"]["positioning_policy_version"] = SCHEMA_VERSION
    return page_data


def _enrich_block_policy(block, layout_direction="ltr"):
    block_bbox = _bbox(block.get("bbox"))
    if not block_bbox:
        return {
            "schema_version": SCHEMA_VERSION,
            "block_id": str(block.get("id") or ""),
            "phrase_policies": [],
        }

    block_width = max(1.0, block_bbox[2] - block_bbox[0])
    block_height = max(1.0, block_bbox[3] - block_bbox[1])
    phrases = _ordered_phrases(block, layout_direction=layout_direction)
    if not phrases:
        return {
            "schema_version": SCHEMA_VERSION,
            "block_id": str(block.get("id") or ""),
            "phrase_policies": [],
        }

    ai_helper = get_element_relations_ai_enricher()
    phrase_policies = []
    for index, item in enumerate(phrases, start=1):
        phrase = item["phrase"]
        phrase_bbox = item["bbox"]
        left_space = max(0.0, phrase_bbox[0] - block_bbox[0])
        right_space = max(0.0, block_bbox[2] - phrase_bbox[2])
        top_space = max(0.0, phrase_bbox[1] - block_bbox[1])
        bottom_space = max(0.0, block_bbox[3] - phrase_bbox[3])

        start_space = left_space if layout_direction != "rtl" else right_space
        end_space = right_space if layout_direction != "rtl" else left_space
        start_ratio = start_space / block_width
        end_ratio = end_space / block_width
        top_ratio = top_space / block_height
        bottom_ratio = bottom_space / block_height

        phrase_width = max(1.0, phrase_bbox[2] - phrase_bbox[0])
        phrase_height = max(1.0, phrase_bbox[3] - phrase_bbox[1])
        width_ratio = phrase_width / block_width
        height_ratio = phrase_height / block_height
        center_x = (phrase_bbox[0] + phrase_bbox[2]) / 2.0
        center_y = (phrase_bbox[1] + phrase_bbox[3]) / 2.0
        block_center_x = (block_bbox[0] + block_bbox[2]) / 2.0
        block_center_y = (block_bbox[1] + block_bbox[3]) / 2.0
        center_x_offset_ratio = abs(center_x - block_center_x) / max(1.0, block_width / 2.0)
        center_y_offset_ratio = abs(center_y - block_center_y) / max(1.0, block_height / 2.0)

        alignment = _resolve_alignment(item["phrase"], item["line"], block)
        flow_prev = item["phrase"].get("flow_from_previous_phrase") or {}
        flow_next = item["phrase"].get("flow_to_next_phrase") or {}
        in_flow = bool(flow_prev.get("continuation")) or bool(flow_next.get("continuation"))
        phrase_text = item["text"]
        short_text = len(phrase_text) <= 24
        numeric_like = _is_numeric_like(phrase_text)
        centered_alignment = alignment == "center"
        end_alignment = alignment == "right" if layout_direction != "rtl" else alignment == "left"
        start_alignment = alignment in {"left", "justify"} if layout_direction != "rtl" else alignment in {"right", "justify"}

        semantic = _semantic_scores(
            ai_helper=ai_helper,
            layout_direction=layout_direction,
            block=block,
            phrase=item["phrase"],
            text=phrase_text,
            block_text=item["block_text"],
            alignment=alignment,
            start_space=start_space,
            end_space=end_space,
            top_space=top_space,
            bottom_space=bottom_space,
            width_ratio=width_ratio,
            height_ratio=height_ratio,
            in_flow=in_flow,
            numeric_like=numeric_like,
            short_text=short_text,
        )

        horizontal_scores = _normalize_scores(
            {
                "start": (
                    0.34 * (1.0 - _clamp01(start_ratio))
                    + 0.18 * _clamp01((end_ratio - start_ratio) + 0.5)
                    + 0.18 * float(semantic["horizontal"].get("start", 1.0 / 3.0))
                    + 0.14 * (1.0 if in_flow else 0.0)
                    + 0.10 * (1.0 if start_alignment else 0.0)
                    + 0.06 * (1.0 if width_ratio >= 0.45 else 0.0)
                ),
                "end": (
                    0.34 * (1.0 - _clamp01(end_ratio))
                    + 0.18 * _clamp01((start_ratio - end_ratio) + 0.5)
                    + 0.18 * float(semantic["horizontal"].get("end", 1.0 / 3.0))
                    + 0.12 * (1.0 if numeric_like else 0.0)
                    + 0.10 * (1.0 if end_alignment else 0.0)
                    + 0.08 * float(semantic["roles"].get("end_value", 0.0))
                ),
                "center": (
                    0.28 * (1.0 - _clamp01(center_x_offset_ratio))
                    + 0.22 * (1.0 - _clamp01(abs(start_ratio - end_ratio) * 2.0))
                    + 0.18 * float(semantic["horizontal"].get("center", 1.0 / 3.0))
                    + 0.14 * (1.0 if centered_alignment else 0.0)
                    + 0.10 * float(semantic["roles"].get("centered_title", 0.0))
                    + 0.08 * (1.0 if short_text else 0.0)
                ),
            }
        )

        vertical_scores = _normalize_scores(
            {
                "top": (
                    0.34 * (1.0 - _clamp01(top_ratio))
                    + 0.18 * _clamp01((bottom_ratio - top_ratio) + 0.5)
                    + 0.18 * float(semantic["vertical"].get("top", 1.0 / 3.0))
                    + 0.16 * (1.0 if in_flow else 0.0)
                    + 0.14 * (1.0 if index == 1 else 0.0)
                ),
                "bottom": (
                    0.34 * (1.0 - _clamp01(bottom_ratio))
                    + 0.18 * _clamp01((top_ratio - bottom_ratio) + 0.5)
                    + 0.18 * float(semantic["vertical"].get("bottom", 1.0 / 3.0))
                    + 0.15 * (1.0 if index == len(phrases) else 0.0)
                    + 0.15 * (1.0 if numeric_like and short_text else 0.0)
                ),
                "middle": (
                    0.30 * (1.0 - _clamp01(center_y_offset_ratio))
                    + 0.22 * (1.0 - _clamp01(abs(top_ratio - bottom_ratio) * 2.0))
                    + 0.18 * float(semantic["vertical"].get("middle", 1.0 / 3.0))
                    + 0.18 * (1.0 if centered_alignment else 0.0)
                    + 0.12 * (1.0 if short_text else 0.0)
                ),
            }
        )

        x_primary, x_secondary = _top_two(horizontal_scores)
        y_primary, y_secondary = _top_two(vertical_scores)
        x_conf = float(horizontal_scores.get(x_primary, 0.0))
        y_conf = float(vertical_scores.get(y_primary, 0.0))
        combined_conf = round(math.sqrt(max(0.0, x_conf) * max(0.0, y_conf)), 4)

        phrase_policy = {
            "schema_version": SCHEMA_VERSION,
            "phrase_id": item["phrase_id"],
            "block_id": str(block.get("id") or ""),
            "reading_order_index": int(index),
            "layout_direction": layout_direction,
            "anchors": {
                "horizontal": {
                    "primary": x_primary,
                    "secondary": x_secondary,
                    "scores": {key: round(float(value), 4) for key, value in horizontal_scores.items()},
                },
                "vertical": {
                    "primary": y_primary,
                    "secondary": y_secondary,
                    "scores": {key: round(float(value), 4) for key, value in vertical_scores.items()},
                },
            },
            "primary_position_reference": {
                "mode": f"{y_primary}_{x_primary}",
                "horizontal": x_primary,
                "vertical": y_primary,
                "confidence": combined_conf,
            },
            "expansion_policy": {
                "horizontal": _horizontal_expansion(x_primary),
                "vertical": _vertical_expansion(y_primary, in_flow=in_flow),
                "translation_positioning_mode": f"{y_primary}_{x_primary}_{_horizontal_expansion(x_primary)}",
            },
            "space_metrics": {
                "left_px": round(left_space, 4),
                "right_px": round(right_space, 4),
                "top_px": round(top_space, 4),
                "bottom_px": round(bottom_space, 4),
                "start_px": round(start_space, 4),
                "end_px": round(end_space, 4),
                "left_ratio": round(left_space / block_width, 6),
                "right_ratio": round(right_space / block_width, 6),
                "top_ratio": round(top_space / block_height, 6),
                "bottom_ratio": round(bottom_space / block_height, 6),
                "center_x_offset_ratio": round(center_x_offset_ratio, 6),
                "center_y_offset_ratio": round(center_y_offset_ratio, 6),
            },
            "semantic_context": {
                "model_used": bool(semantic["model_used"]),
                "review_ready": bool(semantic["review_ready"]),
                "horizontal_scores": {key: round(float(value), 4) for key, value in semantic["horizontal"].items()},
                "vertical_scores": {key: round(float(value), 4) for key, value in semantic["vertical"].items()},
                "role_scores": {key: round(float(value), 4) for key, value in semantic["roles"].items()},
            },
            "signals": {
                "alignment": alignment,
                "in_flow": bool(in_flow),
                "short_text": bool(short_text),
                "numeric_like": bool(numeric_like),
                "width_ratio": round(width_ratio, 6),
                "height_ratio": round(height_ratio, 6),
                "flow_from_previous": str(flow_prev.get("logical_relation") or ""),
                "flow_to_next": str(flow_next.get("logical_relation") or ""),
                "block_role": str(block.get("role") or ""),
                "block_unit_type": str(block.get("unit_type") or ""),
            },
            "formula": {
                "horizontal": {
                    "start": "0.34*edge_closeness_start + 0.18*free_space_after + 0.18*semantic_start + 0.14*flow + 0.10*alignment_start + 0.06*wide_fragment",
                    "end": "0.34*edge_closeness_end + 0.18*free_space_before + 0.18*semantic_end + 0.12*numeric_like + 0.10*alignment_end + 0.08*semantic_end_value",
                    "center": "0.28*center_closeness + 0.22*margin_symmetry + 0.18*semantic_center + 0.14*alignment_center + 0.10*semantic_centered_title + 0.08*short_text",
                },
                "vertical": {
                    "top": "0.34*edge_closeness_top + 0.18*free_space_below + 0.18*semantic_top + 0.16*flow + 0.14*first_in_block",
                    "bottom": "0.34*edge_closeness_bottom + 0.18*free_space_above + 0.18*semantic_bottom + 0.15*last_in_block + 0.15*compact_value",
                    "middle": "0.30*middle_closeness + 0.22*vertical_symmetry + 0.18*semantic_middle + 0.18*alignment_center + 0.12*short_text",
                },
            },
        }
        phrase["positioning_policy"] = phrase_policy
        phrase_policies.append(phrase_policy)

    return {
        "schema_version": SCHEMA_VERSION,
        "block_id": str(block.get("id") or ""),
        "phrase_policies": phrase_policies,
    }


def _semantic_scores(
    ai_helper,
    layout_direction,
    block,
    phrase,
    text,
    block_text,
    alignment,
    start_space,
    end_space,
    top_space,
    bottom_space,
    width_ratio,
    height_ratio,
    in_flow,
    numeric_like,
    short_text,
):
    neutral3 = {"start": 1.0 / 3.0, "end": 1.0 / 3.0, "center": 1.0 / 3.0}
    neutralv = {"top": 1.0 / 3.0, "bottom": 1.0 / 3.0, "middle": 1.0 / 3.0}
    neutral_roles = {"flow_text": 0.25, "centered_title": 0.25, "end_value": 0.25, "attached_label": 0.25}

    if ai_helper is None:
        return {"model_used": False, "review_ready": False, "horizontal": neutral3, "vertical": neutralv, "roles": neutral_roles}
    if _should_skip_semantic_ai(block, phrase, text):
        return {"model_used": False, "review_ready": False, "horizontal": neutral3, "vertical": neutralv, "roles": neutral_roles}

    runtime = ai_helper._get_runtime()
    if runtime is None:
        return {"model_used": False, "review_ready": False, "horizontal": neutral3, "vertical": neutralv, "roles": neutral_roles}

    premise = (
        f"Container role: {str(block.get('role') or 'unknown')}. "
        f"Container unit_type: {str(block.get('unit_type') or 'unknown')}. "
        f"Container alignment: {alignment}. "
        f"Layout direction: {layout_direction}. "
        f"Container text preview: {str(block_text or '')[:260]}. "
        f"Fragment text: {str(text or '')[:220]}. "
        f"start_space={start_space:.2f}; end_space={end_space:.2f}; top_space={top_space:.2f}; bottom_space={bottom_space:.2f}; "
        f"width_ratio={width_ratio:.3f}; height_ratio={height_ratio:.3f}; "
        f"in_flow={int(bool(in_flow))}; numeric_like={int(bool(numeric_like))}; short_text={int(bool(short_text))}."
    )

    horizontal = ai_helper.score_hypotheses(premise, _HORIZONTAL_HYPOTHESES) or {}
    vertical = ai_helper.score_hypotheses(premise, _VERTICAL_HYPOTHESES) or {}
    roles = ai_helper.score_hypotheses(premise, _ROLE_HYPOTHESES) or {}

    return {
        "model_used": True,
        "review_ready": True,
        "horizontal": _fill_scores(horizontal, ("start", "end", "center"), default=1.0 / 3.0),
        "vertical": _fill_scores(vertical, ("top", "bottom", "middle"), default=1.0 / 3.0),
        "roles": _fill_scores(roles, ("flow_text", "centered_title", "end_value", "attached_label"), default=0.25),
    }


def _fill_scores(scores, keys, default):
    if not scores:
        return {key: float(default) for key in keys}
    out = {}
    total = 0.0
    for key in keys:
        value = float(scores.get(key, 0.0) or 0.0)
        out[key] = value
        total += value
    if total <= 0.0:
        return {key: float(default) for key in keys}
    return {key: value / total for key, value in out.items()}


def _ordered_phrases(block, layout_direction="ltr"):
    items = []
    block_id = str(block.get("id") or "block")
    block_text = str(block.get("text") or "").strip()
    lines = [line for line in (block.get("lines") or []) if isinstance(line, dict)]
    lines = sorted(lines, key=lambda line: _sort_bbox(_bbox(line.get("bbox"))))
    for line_idx, line in enumerate(lines):
        line_bbox = _bbox(line.get("bbox"))
        phrases = [phrase for phrase in (line.get("phrases") or []) if isinstance(phrase, dict)]
        phrases = sorted(phrases, key=lambda phrase: _phrase_sort_key(_bbox(phrase.get("bbox")), layout_direction=layout_direction))
        for phrase_idx, phrase in enumerate(phrases):
            bbox = _bbox(phrase.get("bbox"))
            if not bbox:
                continue
            items.append(
                {
                    "phrase": phrase,
                    "phrase_id": str(phrase.get("unit_id") or phrase.get("id") or f"{block_id}:line:{line_idx}:phrase:{phrase_idx}"),
                    "line": line,
                    "bbox": bbox,
                    "text": _clean_text(phrase.get("text") or phrase.get("texte") or ""),
                    "block_text": block_text,
                    "line_bbox": line_bbox,
                }
            )
    return items


def _resolve_alignment(phrase, line, block):
    for node in (phrase, line, block):
        if not isinstance(node, dict):
            continue
        value = str(node.get("alignment") or "").strip().lower()
        if value in {"left", "right", "center", "justify"}:
            return value
    return "left"


def _sort_bbox(bbox):
    if not bbox:
        return (0.0, 0.0)
    return (bbox[1], bbox[0])


def _phrase_sort_key(bbox, layout_direction="ltr"):
    if not bbox:
        return (0.0, 0.0)
    if layout_direction == "rtl":
        return (bbox[1], -bbox[2])
    return (bbox[1], bbox[0])


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
    return " ".join(str(text or "").split()).strip()


def _normalize_scores(scores):
    total = sum(max(0.0, float(value)) for value in (scores or {}).values())
    if total <= 0.0:
        count = max(1, len(scores or {}))
        return {key: 1.0 / count for key in (scores or {})}
    return {key: max(0.0, float(value)) / total for key, value in (scores or {}).items()}


def _top_two(scores):
    ordered = sorted((scores or {}).items(), key=lambda item: (-float(item[1]), item[0]))
    primary = ordered[0][0] if ordered else "start"
    secondary = ordered[1][0] if len(ordered) > 1 else primary
    return primary, secondary


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _horizontal_expansion(anchor):
    if anchor == "end":
        return "grow_to_start"
    if anchor == "center":
        return "grow_symmetrically"
    return "grow_to_end"


def _vertical_expansion(anchor, in_flow=False):
    if anchor == "bottom":
        return "grow_up"
    if anchor == "middle":
        return "preserve_middle" if not in_flow else "grow_symmetrically_vertical"
    return "grow_down"


def _is_numeric_like(text):
    s = str(text or "").strip()
    if not s:
        return False
    allowed = set("0123456789.,:%+-/()[] ")
    return all(ch in allowed for ch in s)


def _should_skip_semantic_ai(block, phrase, text):
    if _looks_like_code_fragment(block, phrase, text):
        return True
    return False


def _looks_like_code_fragment(block, phrase, text):
    text_value = _clean_text(text)
    if not text_value:
        return False
    unit_candidates = [
        str((phrase or {}).get("unit_type") or "").strip().lower(),
        str((block or {}).get("unit_type") or "").strip().lower(),
    ]
    if "code_visible" in unit_candidates:
        return True
    if bool((block or {}).get("immutable_code_block")):
        return True
    spans = list((phrase or {}).get("spans") or [])
    for span in spans:
        if not isinstance(span, dict):
            continue
        style = span.get("style") or {}
        flags = style.get("flags") or {}
        font_name = str(style.get("font") or "").strip().lower()
        if bool(flags.get("monospace")) or "courier" in font_name:
            return True
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", text_value):
        return True
    if "=" in text_value and re.search(r"[A-Za-z_][A-Za-z0-9_]*", text_value):
        return True
    if text_value.count(",") >= 2 and ("=" in text_value or "_" in text_value):
        return True
    return False
