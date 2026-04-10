import math


SCHEMA_VERSION = "element_ruleset.v1"


def enrich_element_rulesets(page_data):
    if not isinstance(page_data, dict):
        return page_data

    layout_direction = str(page_data.get("layout_direction") or "ltr").strip().lower() or "ltr"
    blocks = [block for block in (page_data.get("blocks") or []) if isinstance(block, dict)]
    block_rulesets = []
    flat_rulesets = []

    for block in blocks:
        payload = _enrich_block_rulesets(page_data, block, layout_direction=layout_direction)
        if not payload:
            continue
        block["element_rulesets"] = payload
        block["translation_rulesets"] = payload
        block["translation_ruleset_summary"] = dict(payload.get("summary") or {})
        block_rulesets.append(payload)
        flat_rulesets.extend(list(payload.get("element_rulesets") or []))

    _annotate_toc_rows_with_rulesets(page_data)

    page_payload = {
        "schema_version": SCHEMA_VERSION,
        "element_type": "phrase",
        "layout_direction": layout_direction,
        "block_rulesets": block_rulesets,
        "flat_rulesets": flat_rulesets,
    }
    page_data["element_rulesets"] = page_payload
    page_data["translation_rulesets"] = page_payload
    page_data.setdefault("layout", {})
    page_data["layout"]["element_rulesets"] = page_payload
    page_data["layout"]["translation_rulesets"] = page_payload
    page_data["layout"]["element_rulesets_version"] = SCHEMA_VERSION
    return page_data


def _enrich_block_rulesets(page_data, block, layout_direction="ltr"):
    phrases = _ordered_phrases(block, layout_direction=layout_direction)
    block_id = str(block.get("id") or "")
    rulesets = []

    for item in phrases:
        phrase = item["phrase"]
        ruleset = _build_phrase_ruleset(
            page_data=page_data,
            block=block,
            phrase=phrase,
            phrase_id=item["phrase_id"],
            line=item["line"],
            layout_direction=layout_direction,
        )
        phrase["element_ruleset"] = ruleset
        phrase["translation_ruleset"] = ruleset
        rulesets.append(ruleset)

    return {
        "schema_version": SCHEMA_VERSION,
        "block_id": block_id,
        "element_type": "phrase",
        "element_rulesets": rulesets,
        "summary": _summarize_block_rulesets(rulesets),
    }


def _build_phrase_ruleset(page_data, block, phrase, phrase_id, line, layout_direction="ltr"):
    policy = phrase.get("positioning_policy") or {}
    anchors = policy.get("anchors") or {}
    horizontal = anchors.get("horizontal") or {}
    vertical = anchors.get("vertical") or {}
    horizontal_scores = _score_map(horizontal.get("scores") or {}, ("start", "end", "center"))
    vertical_scores = _score_map(vertical.get("scores") or {}, ("top", "bottom", "middle"))
    combined_modes = _combined_mode_scores(horizontal_scores, vertical_scores)

    primary_horizontal = str(horizontal.get("primary") or _top_score_key(horizontal_scores, default="start"))
    secondary_horizontal = str(horizontal.get("secondary") or _second_score_key(horizontal_scores, default=primary_horizontal))
    primary_vertical = str(vertical.get("primary") or _top_score_key(vertical_scores, default="top"))
    secondary_vertical = str(vertical.get("secondary") or _second_score_key(vertical_scores, default=primary_vertical))

    flow_prev = phrase.get("flow_from_previous_phrase") or {}
    flow_next = phrase.get("flow_to_next_phrase") or {}
    semantic_context = policy.get("semantic_context") or {}
    role_scores = _score_map(
        semantic_context.get("role_scores") or {},
        ("flow_text", "centered_title", "end_value", "attached_label"),
    )
    resolved_role_scores = _resolved_role_scores(
        role_scores=role_scores,
        primary_horizontal=primary_horizontal,
        primary_vertical=primary_vertical,
        signals=policy.get("signals") or {},
    )
    generic_semantic_role = _top_score_key(resolved_role_scores, default="flow_text")
    signal_numeric_like = bool((policy.get("signals") or {}).get("numeric_like"))
    signal_short_text = bool((policy.get("signals") or {}).get("short_text"))
    if primary_horizontal == "end" and signal_numeric_like and signal_short_text:
        generic_semantic_role = "end_value"
    anchor_confidence = round(float((policy.get("primary_position_reference") or {}).get("confidence") or 0.0), 4)
    continuity_confidence = round(
        max(float(flow_prev.get("confidence") or 0.0), float(flow_next.get("confidence") or 0.0)),
        4,
    )

    rel = phrase.get("relative_geometry") or {}
    bbox_relative = _bbox(rel.get("bbox_relative_to_container_block")) or _bbox(phrase.get("bbox")) or [0.0, 0.0, 0.0, 0.0]
    width = max(0.0, bbox_relative[2] - bbox_relative[0])
    height = max(0.0, bbox_relative[3] - bbox_relative[1])
    container_block_id = str(rel.get("container_block_id") or block.get("id") or "")
    space_metrics = policy.get("space_metrics") or {}

    line_text = _clean_text(line.get("line_text") or "")
    phrase_text = _clean_text(phrase.get("text") or phrase.get("texte") or "")
    signals = policy.get("signals") or {}
    phrase_bbox = _bbox(phrase.get("bbox")) or bbox_relative
    hard_break_before = bool(phrase.get("hard_break_before") or line.get("hard_break_before"))
    hard_break_after = bool(phrase.get("line_break_after", True))
    continuity_class = _continuity_class(flow_prev, flow_next)
    specialized_semantic_role, role_source, role_details = _resolve_specialized_role(
        page_data=page_data,
        block=block,
        line=line,
        phrase=phrase,
        phrase_text=phrase_text,
        phrase_bbox=phrase_bbox,
        primary_horizontal=primary_horizontal,
        primary_vertical=primary_vertical,
        generic_role=generic_semantic_role,
        signals=signals,
        layout_direction=layout_direction,
    )
    semantic_role = specialized_semantic_role
    semantic_confidence = round(float(resolved_role_scores.get(generic_semantic_role, 0.0)), 4)
    effective_horizontal, effective_secondary_horizontal, effective_vertical, effective_secondary_vertical = _apply_semantic_anchor_overrides(
        semantic_role=semantic_role,
        primary_horizontal=primary_horizontal,
        secondary_horizontal=secondary_horizontal,
        primary_vertical=primary_vertical,
        secondary_vertical=secondary_vertical,
    )
    overall_confidence = round(
        (0.55 * anchor_confidence) + (0.25 * semantic_confidence) + (0.20 * continuity_confidence),
        4,
    )

    rules = {
        "preserve_horizontal_anchor": effective_horizontal,
        "preserve_vertical_anchor": effective_vertical,
        "secondary_horizontal_anchor": effective_secondary_horizontal,
        "secondary_vertical_anchor": effective_secondary_vertical,
        "translation_positioning_mode": f"{effective_vertical}_{effective_horizontal}_{_horizontal_growth_for_rule(effective_horizontal)}",
        "horizontal_growth": _horizontal_growth_for_rule(effective_horizontal),
        "vertical_growth": _vertical_growth_for_rule(
            effective_vertical,
            in_flow=continuity_class in {"mid_flow", "head_continuation", "tail_continuation"},
        ),
        "keep_with_previous": bool(flow_prev.get("continuation")),
        "keep_with_next": bool(flow_next.get("continuation")),
        "hard_break_before": hard_break_before,
        "hard_break_after": hard_break_after,
        "continuity_class": continuity_class,
        "semantic_role": semantic_role,
    }

    constraints = {
        "available_space": {
            "left_px": round(float(space_metrics.get("left_px") or 0.0), 4),
            "right_px": round(float(space_metrics.get("right_px") or 0.0), 4),
            "top_px": round(float(space_metrics.get("top_px") or 0.0), 4),
            "bottom_px": round(float(space_metrics.get("bottom_px") or 0.0), 4),
            "start_px": round(float(space_metrics.get("start_px") or 0.0), 4),
            "end_px": round(float(space_metrics.get("end_px") or 0.0), 4),
        },
        "preserve_center_if_possible": effective_horizontal == "center" or effective_vertical == "middle",
        "allow_horizontal_reflow": not bool(flow_prev.get("continuation")) and not bool(flow_next.get("continuation")),
        "allow_vertical_reflow": continuity_class in {"standalone", "tail_continuation"},
        "container_block_id": container_block_id,
    }

    override_conditions = _override_conditions(
        primary_horizontal=effective_horizontal,
        primary_vertical=effective_vertical,
        space_metrics=space_metrics,
        continuity_class=continuity_class,
        semantic_role=semantic_role,
    )

    ruleset = {
        "schema_version": SCHEMA_VERSION,
        "ruleset_id": f"{container_block_id}::{phrase_id}::ruleset",
        "element_type": "phrase",
        "phrase_id": phrase_id,
        "block_id": str(block.get("id") or ""),
        "container_block_id": container_block_id,
        "reading_order_index": int(
            ((phrase.get("element_relation_node") or {}).get("reading_order_index"))
            or policy.get("reading_order_index")
            or 0
        ),
        "layout_direction": layout_direction,
        "text_preview": phrase_text[:240],
        "geometry": {
            "bbox_relative_to_block": bbox_relative,
            "width_px": round(width, 4),
            "height_px": round(height, 4),
            "space_metrics": constraints["available_space"],
            "space_ratios": {
                "left_ratio": round(float(space_metrics.get("left_ratio") or 0.0), 6),
                "right_ratio": round(float(space_metrics.get("right_ratio") or 0.0), 6),
                "top_ratio": round(float(space_metrics.get("top_ratio") or 0.0), 6),
                "bottom_ratio": round(float(space_metrics.get("bottom_ratio") or 0.0), 6),
                "center_x_offset_ratio": round(float(space_metrics.get("center_x_offset_ratio") or 0.0), 6),
                "center_y_offset_ratio": round(float(space_metrics.get("center_y_offset_ratio") or 0.0), 6),
            },
        },
        "position_reference_priority": {
            "horizontal": _sorted_scores(horizontal_scores),
            "vertical": _sorted_scores(vertical_scores),
            "combined_modes": combined_modes,
        },
        "continuity": {
            "with_previous": _relation_snapshot(flow_prev),
            "with_next": _relation_snapshot(flow_next),
        },
        "semantics": {
            "role": semantic_role,
            "generic_role": generic_semantic_role,
            "role_confidence": semantic_confidence,
            "role_scores": {key: round(float(value), 4) for key, value in resolved_role_scores.items()},
            "raw_role_scores": {key: round(float(value), 4) for key, value in role_scores.items()},
            "model_used": bool(semantic_context.get("model_used")),
            "review_ready": bool(semantic_context.get("review_ready")),
            "specialized_role_source": role_source,
            "specialized_role_details": role_details,
        },
        "rules": rules,
        "constraints": constraints,
        "override_conditions": override_conditions,
        "signals": {
            "alignment": str(signals.get("alignment") or ""),
            "numeric_like": bool(signals.get("numeric_like")),
            "short_text": bool(signals.get("short_text")),
            "in_flow": bool(signals.get("in_flow")),
            "block_role": str(signals.get("block_role") or block.get("role") or ""),
            "block_unit_type": str(signals.get("block_unit_type") or block.get("unit_type") or ""),
            "line_text_preview": line_text[:240],
        },
        "confidence": {
            "overall": overall_confidence,
            "anchor": anchor_confidence,
            "continuity": continuity_confidence,
            "semantic_role": semantic_confidence,
        },
    }
    return ruleset


def _ordered_phrases(block, layout_direction="ltr"):
    items = []
    block_id = str(block.get("id") or "block")
    lines = [line for line in (block.get("lines") or []) if isinstance(line, dict)]
    lines = sorted(lines, key=lambda line: _sort_bbox(_bbox(line.get("bbox"))))
    for line_idx, line in enumerate(lines):
        phrases = [phrase for phrase in (line.get("phrases") or []) if isinstance(phrase, dict)]
        phrases = sorted(
            phrases,
            key=lambda phrase: _phrase_sort_key(_bbox(phrase.get("bbox")), layout_direction=layout_direction),
        )
        for phrase_idx, phrase in enumerate(phrases):
            items.append(
                {
                    "phrase": phrase,
                    "phrase_id": str(
                        phrase.get("unit_id")
                        or phrase.get("id")
                        or f"{block_id}:line:{line_idx}:phrase:{phrase_idx}"
                    ),
                    "line": line,
                }
            )
    return items


def _summarize_block_rulesets(rulesets):
    rulesets = [ruleset for ruleset in (rulesets or []) if isinstance(ruleset, dict)]
    if not rulesets:
        return {}
    role_counts = {}
    h_counts = {}
    v_counts = {}
    best = None
    best_conf = -1.0
    for ruleset in rulesets:
        rules = ruleset.get("rules") or {}
        conf = float(((ruleset.get("confidence") or {}).get("overall")) or 0.0)
        role = str(rules.get("semantic_role") or "")
        h_anchor = str(rules.get("preserve_horizontal_anchor") or "")
        v_anchor = str(rules.get("preserve_vertical_anchor") or "")
        if role:
            role_counts[role] = int(role_counts.get(role, 0)) + 1
        if h_anchor:
            h_counts[h_anchor] = int(h_counts.get(h_anchor, 0)) + 1
        if v_anchor:
            v_counts[v_anchor] = int(v_counts.get(v_anchor, 0)) + 1
        if conf > best_conf:
            best_conf = conf
            best = ruleset
    best_rules = (best or {}).get("rules") or {}
    best_semantics = (best or {}).get("semantics") or {}
    return {
        "dominant_semantic_role": _max_count_key(role_counts),
        "preferred_horizontal_anchor": _max_count_key(h_counts),
        "preferred_vertical_anchor": _max_count_key(v_counts),
        "top_confidence_ruleset_id": str((best or {}).get("ruleset_id") or ""),
        "top_confidence_overall": round(best_conf if best_conf >= 0.0 else 0.0, 4),
        "top_confidence_semantic_role": str(best_rules.get("semantic_role") or ""),
        "top_confidence_translation_mode": str(best_rules.get("translation_positioning_mode") or ""),
        "role_counts": dict(role_counts),
        "horizontal_anchor_counts": dict(h_counts),
        "vertical_anchor_counts": dict(v_counts),
        "specialized_role_source": str(best_semantics.get("specialized_role_source") or ""),
    }


def _annotate_toc_rows_with_rulesets(page_data):
    if not _is_toc_page(page_data):
        return
    toc = (page_data or {}).get("toc") or {}
    rows = toc.get("toc_rows") or []
    if not isinstance(rows, list) or not rows:
        return

    phrase_rulesets = []
    for block in (page_data.get("blocks") or []):
        for line in (block.get("lines") or []):
            for phrase in (line.get("phrases") or []):
                ruleset = phrase.get("element_ruleset") or {}
                if not ruleset:
                    continue
                phrase_bbox = _bbox(phrase.get("bbox"))
                if not phrase_bbox:
                    continue
                phrase_rulesets.append(
                    {
                        "bbox": phrase_bbox,
                        "text": _clean_text(phrase.get("text") or phrase.get("texte") or ""),
                        "ruleset": ruleset,
                    }
                )

    for row in rows:
        if not isinstance(row, dict):
            continue
        label_matches = _match_rulesets_for_bbox(phrase_rulesets, row.get("label_bbox"))
        page_matches = _match_rulesets_for_bbox(phrase_rulesets, row.get("page_bbox"))
        row["label_rulesets"] = [match["ruleset"] for match in label_matches]
        row["page_rulesets"] = [match["ruleset"] for match in page_matches]
        row["label_ruleset"] = label_matches[0]["ruleset"] if label_matches else None
        row["page_ruleset"] = page_matches[0]["ruleset"] if page_matches else None
        row["row_ruleset_summary"] = {
            "label_semantic_roles": [str(((match["ruleset"].get("rules") or {}).get("semantic_role")) or "") for match in label_matches],
            "page_semantic_roles": [str(((match["ruleset"].get("rules") or {}).get("semantic_role")) or "") for match in page_matches],
            "label_primary_horizontal": str((((label_matches[0]["ruleset"].get("rules") or {}).get("preserve_horizontal_anchor")) if label_matches else "") or ""),
            "page_primary_horizontal": str((((page_matches[0]["ruleset"].get("rules") or {}).get("preserve_horizontal_anchor")) if page_matches else "") or ""),
        }


def _match_rulesets_for_bbox(phrase_rulesets, bbox):
    bbox = _bbox(bbox)
    if not bbox:
        return []
    matches = []
    for item in phrase_rulesets or []:
        score = _bbox_overlap_ratio(item.get("bbox"), bbox)
        if score < 0.25:
            continue
        matches.append((score, item))
    matches.sort(key=lambda pair: (-float(pair[0]), str((((pair[1].get("ruleset") or {}).get("rules") or {}).get("semantic_role")) or "")))
    return [item for _, item in matches]


def _max_count_key(counts):
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))
    return str(ordered[0][0] or "")


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


def _bbox_overlap_ratio(a, b):
    a = _bbox(a)
    b = _bbox(b)
    if not a or not b:
        return 0.0
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    a_area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / a_area


def _score_map(scores, keys):
    out = {}
    for key in keys:
        out[key] = max(0.0, float((scores or {}).get(key, 0.0) or 0.0))
    total = sum(out.values())
    if total <= 0.0:
        return {key: 1.0 / max(1, len(keys)) for key in keys}
    return {key: value / total for key, value in out.items()}


def _sorted_scores(scores):
    ordered = sorted((scores or {}).items(), key=lambda item: (-float(item[1]), item[0]))
    return [{"reference": key, "score": round(float(value), 4)} for key, value in ordered]


def _combined_mode_scores(horizontal_scores, vertical_scores):
    combined = []
    for vertical_key, vertical_value in (vertical_scores or {}).items():
        for horizontal_key, horizontal_value in (horizontal_scores or {}).items():
            score = math.sqrt(max(0.0, float(vertical_value)) * max(0.0, float(horizontal_value)))
            combined.append(
                {
                    "mode": f"{vertical_key}_{horizontal_key}",
                    "score": score,
                }
            )
    combined = sorted(combined, key=lambda item: (-float(item["score"]), item["mode"]))
    return [
        {
            "mode": item["mode"],
            "score": round(float(item["score"]), 4),
        }
        for item in combined
    ]


def _top_score_key(scores, default):
    ordered = _sorted_scores(scores)
    return ordered[0]["reference"] if ordered else default


def _second_score_key(scores, default):
    ordered = _sorted_scores(scores)
    return ordered[1]["reference"] if len(ordered) > 1 else default


def _relation_snapshot(relation):
    if not isinstance(relation, dict):
        return {
            "exists": False,
            "continuation": False,
            "visual_relation": "",
            "logical_relation": "",
            "confidence": 0.0,
            "resolved_by": "",
        }
    return {
        "exists": True,
        "relation_id": str(relation.get("relation_id") or ""),
        "continuation": bool(relation.get("continuation")),
        "visual_relation": str(relation.get("visual_relation") or ""),
        "logical_relation": str(relation.get("logical_relation") or ""),
        "confidence": round(float(relation.get("confidence") or 0.0), 4),
        "resolved_by": str(relation.get("resolved_by") or ""),
    }


def _continuity_class(flow_prev, flow_next):
    prev_cont = bool((flow_prev or {}).get("continuation"))
    next_cont = bool((flow_next or {}).get("continuation"))
    if prev_cont and next_cont:
        return "mid_flow"
    if prev_cont:
        return "tail_continuation"
    if next_cont:
        return "head_continuation"
    return "standalone"


def _resolved_role_scores(role_scores, primary_horizontal, primary_vertical, signals):
    boosted = dict(role_scores or {})
    for key in ("flow_text", "centered_title", "end_value", "attached_label"):
        boosted.setdefault(key, 0.0)

    alignment = str((signals or {}).get("alignment") or "").strip().lower()
    numeric_like = bool((signals or {}).get("numeric_like"))
    short_text = bool((signals or {}).get("short_text"))
    in_flow = bool((signals or {}).get("in_flow"))

    if numeric_like and primary_horizontal == "end":
        boosted["end_value"] += 0.60
    if numeric_like and short_text and primary_horizontal == "end":
        boosted["end_value"] += 0.10
    if primary_horizontal == "center" and alignment == "center":
        boosted["centered_title"] += 0.20
    if primary_vertical == "middle" and alignment == "center" and short_text:
        boosted["centered_title"] += 0.10
    if in_flow:
        boosted["flow_text"] += 0.15
    if short_text and primary_horizontal == "start" and not in_flow:
        boosted["attached_label"] += 0.10
    return _score_map(boosted, ("flow_text", "centered_title", "end_value", "attached_label"))


def _apply_semantic_anchor_overrides(
    semantic_role,
    primary_horizontal,
    secondary_horizontal,
    primary_vertical,
    secondary_vertical,
):
    effective_horizontal = primary_horizontal
    effective_secondary_horizontal = secondary_horizontal
    effective_vertical = primary_vertical
    effective_secondary_vertical = secondary_vertical

    if semantic_role == "toc_page_number":
        effective_horizontal = "end"
        effective_secondary_horizontal = "start" if primary_horizontal != "start" else "center"
    elif semantic_role == "toc_section_number":
        effective_horizontal = "start"
        effective_secondary_horizontal = "end" if primary_horizontal != "end" else "center"
    elif semantic_role == "toc_heading" and primary_horizontal != "center":
        effective_secondary_horizontal = primary_horizontal
        effective_horizontal = "center"

    return (
        effective_horizontal,
        effective_secondary_horizontal,
        effective_vertical,
        effective_secondary_vertical,
    )


def _horizontal_growth_for_rule(anchor):
    if anchor == "end":
        return "grow_to_start"
    if anchor == "center":
        return "grow_symmetrically"
    return "grow_to_end"


def _vertical_growth_for_rule(anchor, in_flow=False):
    if anchor == "bottom":
        return "grow_up"
    if anchor == "middle":
        return "grow_symmetrically_vertical" if in_flow else "preserve_middle"
    return "grow_down"


def _resolve_specialized_role(
    page_data,
    block,
    line,
    phrase,
    phrase_text,
    phrase_bbox,
    primary_horizontal,
    primary_vertical,
    generic_role,
    signals,
    layout_direction,
):
    if _is_toc_page(page_data):
        toc_role, toc_details = _resolve_toc_role(
            page_data=page_data,
            block=block,
            line=line,
            phrase=phrase,
            phrase_text=phrase_text,
            phrase_bbox=phrase_bbox,
            primary_horizontal=primary_horizontal,
            primary_vertical=primary_vertical,
            generic_role=generic_role,
            signals=signals,
            layout_direction=layout_direction,
        )
        if toc_role:
            return toc_role, "toc_specialization", toc_details
    return generic_role, "generic_semantic_role", {}


def _resolve_toc_role(
    page_data,
    block,
    line,
    phrase,
    phrase_text,
    phrase_bbox,
    primary_horizontal,
    primary_vertical,
    generic_role,
    signals,
    layout_direction,
):
    text = _clean_text(phrase_text)
    if not text:
        return None, {}

    match_kind, match_row = _match_toc_row(page_data, phrase_bbox)
    line_info = _line_role_context(line, phrase, layout_direction=layout_direction)
    numeric_like = bool((signals or {}).get("numeric_like"))
    short_text = bool((signals or {}).get("short_text"))
    alignment = str((signals or {}).get("alignment") or "").strip().lower()
    roman_like = _is_roman_numeral(text)
    section_like = _is_toc_section_number(text)
    heading_like = _is_toc_heading_text(text)
    word_like = _has_word(text)
    page_number_like = (numeric_like or roman_like) and short_text

    details = {
        "row_match_kind": match_kind or "",
        "row_role": str((match_row or {}).get("role") or ""),
        "line_has_page_number_candidate": bool(line_info.get("has_page_number_candidate")),
        "line_has_section_marker_candidate": bool(line_info.get("has_section_marker_candidate")),
    }

    if heading_like:
        return "toc_heading", details
    if match_row and str((match_row or {}).get("role") or "") in {"part_title", "chapter_title"}:
        if page_number_like and match_kind == "page":
            return "toc_page_number", details
        return "toc_heading", details
    if match_kind == "page" and page_number_like:
        return "toc_page_number", details
    if primary_horizontal == "end" and page_number_like:
        return "toc_page_number", details
    if section_like and primary_horizontal == "start":
        return "toc_section_number", details
    if line_info.get("next_is_page_number_candidate") and (section_like or _looks_like_toc_marker(text)):
        return "toc_section_number", details
    if match_kind == "label" and word_like:
        return "toc_entry_title", details
    if line_info.get("has_page_number_candidate") and word_like and primary_horizontal != "end":
        return "toc_entry_title", details
    if line_info.get("has_section_marker_candidate") and word_like:
        return "toc_entry_title", details
    if generic_role == "centered_title" and alignment == "center":
        return "toc_heading", details
    return None, details


def _is_toc_page(page_data):
    if not isinstance(page_data, dict):
        return False
    if str(page_data.get("page_role") or "").strip().lower() == "toc":
        return True
    if str(page_data.get("layout_type") or "").strip().lower() == "toc_page":
        return True
    toc = page_data.get("toc") or {}
    return bool(toc.get("toc_rows"))


def _match_toc_row(page_data, phrase_bbox):
    rows = ((page_data or {}).get("toc") or {}).get("toc_rows") or []
    if not rows or not phrase_bbox:
        return None, None

    best_kind = None
    best_row = None
    best_score = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        for kind, key in (("label", "label_bbox"), ("page", "page_bbox")):
            row_bbox = _bbox(row.get(key))
            if not row_bbox:
                continue
            score = _bbox_overlap_ratio(phrase_bbox, row_bbox)
            if score > best_score:
                best_score = score
                best_kind = kind
                best_row = row
    if best_score < 0.25:
        return None, None
    return best_kind, best_row


def _line_role_context(line, phrase, layout_direction="ltr"):
    phrases = [item for item in (line.get("phrases") or []) if isinstance(item, dict)]
    ordered = sorted(phrases, key=lambda item: _phrase_sort_key(_bbox(item.get("bbox")), layout_direction=layout_direction))
    current_idx = 0
    for idx, item in enumerate(ordered):
        if item is phrase or str(item.get("id") or item.get("unit_id") or "") == str(phrase.get("id") or phrase.get("unit_id") or ""):
            current_idx = idx
            break

    def phrase_text(item):
        return _clean_text(item.get("text") or item.get("texte") or "")

    def is_page_num(item):
        text = phrase_text(item)
        return bool(text) and len(text) <= 12 and (_is_numeric_like_text(text) or _is_roman_numeral(text))

    def is_section_num(item):
        text = phrase_text(item)
        return _is_toc_section_number(text)

    has_page_number_candidate = any(is_page_num(item) for item in ordered)
    has_section_marker_candidate = any(is_section_num(item) for item in ordered)
    next_item = ordered[current_idx + 1] if current_idx + 1 < len(ordered) else None
    prev_item = ordered[current_idx - 1] if current_idx - 1 >= 0 else None
    return {
        "has_page_number_candidate": has_page_number_candidate,
        "has_section_marker_candidate": has_section_marker_candidate,
        "next_is_page_number_candidate": bool(next_item and is_page_num(next_item)),
        "prev_is_section_marker_candidate": bool(prev_item and is_section_num(prev_item)),
    }


def _bbox_overlap_ratio(a, b):
    a = _bbox(a)
    b = _bbox(b)
    if not a or not b:
        return 0.0
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def _is_toc_heading_text(text):
    s = _clean_text(text).lower()
    return s in {"contents", "table of contents", "sommaire"} or s.startswith("contents ")


def _has_word(text):
    s = str(text or "")
    return any(ch.isalpha() for ch in s)


def _is_roman_numeral(text):
    s = _clean_text(text).upper()
    if not s or len(s) > 12:
        return False
    return all(ch in {"I", "V", "X", "L", "C", "D", "M"} for ch in s)


def _is_toc_section_number(text):
    s = _clean_text(text)
    if not s or len(s) > 16:
        return False
    parts = s.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return False
    return len(parts) >= 2 or (len(parts) == 1 and len(s) <= 3)


def _looks_like_toc_marker(text):
    s = _clean_text(text)
    return _is_toc_section_number(s) or s.startswith(("Part ", "PART "))


def _is_numeric_like_text(text):
    s = str(text or "").strip()
    if not s:
        return False
    allowed = set("0123456789.,:%+-/()[] ")
    return all(ch in allowed for ch in s)


def _override_conditions(primary_horizontal, primary_vertical, space_metrics, continuity_class, semantic_role):
    conditions = []
    start_px = float(space_metrics.get("start_px") or 0.0)
    end_px = float(space_metrics.get("end_px") or 0.0)
    top_px = float(space_metrics.get("top_px") or 0.0)
    bottom_px = float(space_metrics.get("bottom_px") or 0.0)

    if primary_horizontal == "start" and end_px > start_px:
        conditions.append("prefer_growth_toward_end_space")
    if primary_horizontal == "end" and start_px > end_px:
        conditions.append("prefer_growth_toward_start_space")
    if primary_horizontal == "center":
        conditions.append("preserve_horizontal_center_until_overflow")
    if primary_vertical == "middle":
        conditions.append("preserve_vertical_middle_until_overflow")
    if primary_vertical == "top" and bottom_px > top_px:
        conditions.append("prefer_growth_downward")
    if primary_vertical == "bottom" and top_px > bottom_px:
        conditions.append("prefer_growth_upward")
    if continuity_class in {"mid_flow", "head_continuation", "tail_continuation"}:
        conditions.append("preserve_text_flow_relationship")
    if semantic_role == "end_value":
        conditions.append("protect_value_alignment")
    if semantic_role == "centered_title":
        conditions.append("protect_title_centering")
    if semantic_role == "toc_page_number":
        conditions.append("protect_value_alignment")
        conditions.append("preserve_toc_row_pairing")
    if semantic_role == "toc_entry_title":
        conditions.append("preserve_toc_row_pairing")
    if semantic_role == "toc_heading":
        conditions.append("protect_title_centering")
    if semantic_role == "toc_section_number":
        conditions.append("protect_section_marker_alignment")
        conditions.append("preserve_toc_row_pairing")
    return conditions


def _clean_text(text):
    return " ".join(str(text or "").split()).strip()
