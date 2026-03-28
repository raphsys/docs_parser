import re

import fitz


def _normalize_spaces(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def _normalize_text(text):
    s = _normalize_spaces(text).lower()
    s = re.sub(r"[’`]", "'", s)
    s = s.replace("·", "'")
    s = re.sub(r"[^a-z0-9à-ÿ'\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(text):
    s = _normalize_text(text)
    if not s:
        return []
    return re.findall(r"[a-z0-9à-ÿ][a-z0-9à-ÿ'\-]*", s)


def _text_signature(text):
    toks = _tokenize(text)
    if not toks:
        return set()
    stop = {
        "the", "and", "for", "with", "from", "this", "that", "into", "using",
        "les", "des", "pour", "avec", "dans", "cette", "cette", "sur", "une",
        "de", "du", "la", "le", "et", "en", "un", "une", "a", "to", "of",
    }
    return {t for t in toks if len(t) >= 3 and t not in stop}


def _is_reference_like_render_unit(unit, expected_text):
    unit_type = _normalize_spaces((unit or {}).get("unit_type") or "").lower()
    strategy = _normalize_spaces((unit or {}).get("translation_strategy") or "").lower()
    expected = _normalize_spaces(expected_text)
    if unit_type in {"citation", "reference_link"}:
        return True
    if re.search(r"(https?://|www\.)", expected, flags=re.IGNORECASE):
        return True
    if strategy == "exact_preserve" and len(_tokenize(expected)) >= 6:
        return True
    return False


def _english_marker_count(text):
    s = _normalize_text(text)
    if not s:
        return 0
    markers = {
        "the", "and", "with", "for", "from", "this", "that", "what", "where",
        "why", "building", "model", "architecture", "using", "image",
        "classification", "dropout", "layer", "layers", "overfitting",
    }
    return sum(1 for tok in _tokenize(s) if tok in markers)


def _retained_source_token_count(source_text, translated_text):
    src_sig = _text_signature(source_text)
    if not src_sig:
        return 0
    tgt_tokens = set(_tokenize(translated_text))
    count = 0
    for tok in src_sig:
        if tok in tgt_tokens and not re.fullmatch(r"[A-Z]{2,6}", tok, flags=re.IGNORECASE):
            count += 1
    return count


def _looks_page_number(text):
    s = _normalize_spaces(text)
    return bool(re.fullmatch(r"[\divxlcdm]+[.)]?", s, flags=re.IGNORECASE))


def _looks_symbolic(text):
    s = _normalize_spaces(text)
    if not s:
        return True
    if re.fullmatch(r"[•▪◦·\-\*■]+", s):
        return True
    if re.fullmatch(r"[A-Z]{1,5}", s):
        return True
    if re.fullmatch(r"\([A-Za-z0-9_\-\s]{1,24}\)", s):
        return True
    return False


def _is_shared_technical_term_fr(text):
    tokens = _tokenize(text)
    if not tokens:
        return False
    allowed = {
        "image",
        "images",
        "convolution",
        "convolutions",
        "pixel",
        "pixels",
    }
    return all(tok in allowed for tok in tokens)


def _is_preservable_symbolic_clause(text, document_type="", layout_type=""):
    src = _normalize_spaces(text)
    if not src:
        return False
    if not re.search(r"[<>=%]", src):
        return False
    if not re.search(r"\b(value is|which means|means that|less than|greater than)\b", src, flags=re.IGNORECASE):
        return False

    doc_kind = _normalize_spaces(document_type).lower()
    layout_kind = _normalize_spaces(layout_type).lower()
    if layout_kind == "table_dominant" and doc_kind in {"form", "invoice", "receipt", "mixed_unknown"}:
        return True

    # Fallback for short operator clauses that are structurally tied to a diagram,
    # table cell, or formula explanation even when page-level metadata is incomplete.
    tokens = _tokenize(src)
    if len(tokens) > 8:
        return False
    if len(src) > 48:
        return False
    return bool(re.search(r"\b(value|means|than)\b", src, flags=re.IGNORECASE))


def _is_preservable_short_label_fr(text, unit_type=""):
    unit_type = _normalize_spaces(unit_type).lower()
    if unit_type not in {"short_label", "chart_label"}:
        return False
    raw = _normalize_spaces(text)
    tokens = _tokenize(text)
    if len(tokens) != 1:
        return False
    if len(raw) > 32:
        return False
    if re.search(r"\d", raw):
        return False
    return bool(re.fullmatch(r"[A-ZÀ-Ý][A-Za-zÀ-ÿ'\-]{2,31}", raw))


def _is_figure_reference_label(text):
    raw = _normalize_spaces(text)
    if not raw:
        return False
    return bool(
        re.fullmatch(
            r"[\(\[]?(?:fig(?:ure)?|figure)\s+\d+(?:\.\d+){0,3}(?:[\)\]])?(?:[.:;])?",
            raw,
            flags=re.IGNORECASE,
        )
    )


def _is_numeric_heading_marker(text):
    raw = _normalize_spaces(text)
    if not raw:
        return False
    return bool(re.fullmatch(r"\d+(?:\.\d+){1,5}\.?", raw))


def _guess_source_text(unit):
    return _normalize_spaces(
        unit.get("texte_original")
        or unit.get("raw_text")
        or unit.get("text")
        or unit.get("texte")
        or ""
    )


def _guess_translated_text(unit):
    return _normalize_spaces(
        unit.get("translated_text")
        or unit.get("texte")
        or unit.get("text")
        or ""
    )


def _iter_phrase_units(page):
    if not isinstance(page, dict):
        return
    descriptor = page.get("layout_descriptor") or ((page.get("layout") or {}).get("layout_descriptor")) or {}
    descriptor_elements = {
        str(el.get("id")): el
        for el in (descriptor.get("elements") or [])
        if isinstance(el, dict) and el.get("id")
    }
    descriptor_groups = {
        str(gr.get("id")): gr
        for gr in (descriptor.get("groups") or [])
        if isinstance(gr, dict) and gr.get("id")
    }
    descriptor_constraints = {}
    for constraint in (descriptor.get("constraints") or []):
        if not isinstance(constraint, dict):
            continue
        element_id = str(constraint.get("element_id") or "")
        if not element_id:
            continue
        descriptor_constraints.setdefault(element_id, []).append(constraint)
    page_family = page.get("page_family") or ((page.get("layout") or {}).get("page_family")) or ""
    page_family_group = page.get("page_family_group") or ((page.get("layout") or {}).get("page_family_group")) or page_family
    document_type = page.get("document_type") or ((page.get("layout") or {}).get("document_type")) or ""
    layout_type = page.get("layout_type") or ((page.get("layout") or {}).get("layout_type")) or ""
    style_profile = page.get("style_profile") or ((page.get("layout") or {}).get("style_profile")) or ""
    blocks = page.get("blocks", []) or []
    for b_idx, block in enumerate(blocks):
        block_id = str(block.get("id") or "")
        descriptor_block = descriptor_elements.get(block_id) if block_id else None
        paragraph_group = None
        if isinstance(descriptor_block, dict):
            paragraph_id = str(descriptor_block.get("paragraph_id") or "").strip()
            if paragraph_id:
                paragraph_group = descriptor_groups.get(paragraph_id)
        block_constraints = descriptor_constraints.get(block_id, []) if block_id else []
        preserve_sentence_integrity = bool(
            any(str(c.get("type") or "") == "no_internal_sentence_break" for c in block_constraints)
            or ((paragraph_group or {}).get("constraints") or {}).get("can_break_inside_sentence") is False
        )
        lines = block.get("lines", []) or []
        for l_idx, line in enumerate(lines):
            phrases = line.get("phrases", []) or []
            for p_idx, phrase in enumerate(phrases):
                spans = phrase.get("spans", []) or []
                skip_flags = [bool(sp.get("skip_render", False)) for sp in spans if isinstance(sp, dict)]
                unit = {
                    "page_id": page.get("page", 0),
                    "block_index": b_idx,
                    "line_index": l_idx,
                    "phrase_index": p_idx,
                    "unit_id": phrase.get("unit_id") or f"p{page.get('page', 0)}:b{b_idx}:l{l_idx}:p{p_idx}",
                    "role": phrase.get("role") or block.get("role") or "body",
                    "source_kind": phrase.get("source_kind") or block.get("source_kind") or "",
                    "translation_strategy": phrase.get("translation_strategy") or block.get("translation_strategy") or "semantic_reflow",
                    "coverage_required": phrase.get("coverage_required") or block.get("coverage_required") or "strict",
                    "translatable": bool(phrase.get("translatable", block.get("translatable", True))),
                    "render_policy": phrase.get("render_policy") or line.get("render_policy") or block.get("render_policy") or "",
                    "block_render_policy": block.get("render_policy") or "",
                    "render_mode": phrase.get("render_mode") or line.get("render_mode") or block.get("render_mode") or "",
                    "unit_type": phrase.get("unit_type") or line.get("unit_type") or block.get("unit_type") or "",
                    "page_family": page_family,
                    "page_family_group": page_family_group,
                    "document_type": document_type,
                    "layout_type": layout_type,
                    "style_profile": style_profile,
                    "preserve_sentence_integrity": preserve_sentence_integrity,
                    "descriptor_paragraph_id": str((paragraph_group or {}).get("id") or ""),
                    "block_unit_id": block.get("unit_id") or block_id or f"blk_{b_idx}",
                    "block_source_text": _normalize_spaces(block.get("text") or ""),
                    "block_translated_text": _normalize_spaces(block.get("translated_text") or ""),
                    "block_is_primary_render_unit": l_idx == 0 and p_idx == 0,
                    "visible_text": _normalize_spaces(phrase.get("text") or ""),
                    "has_skipped_spans": any(skip_flags),
                    "all_spans_skip_render": bool(skip_flags) and all(skip_flags),
                    "source_text": _guess_source_text(phrase),
                    "translated_text": _guess_translated_text(phrase),
                }
                yield unit


def _iter_toc_units(page):
    if not isinstance(page, dict):
        return
    if page.get("schema_version") != "layout.v2":
        return
    if str(page.get("page_role", "")).strip().lower() != "toc":
        return
    rows = ((page.get("toc") or {}).get("toc_rows") or [])
    page_id = page.get("page", 0)
    for idx, row in enumerate(rows):
        label = _normalize_spaces(row.get("label") or "")
        translated_label = _normalize_spaces(row.get("translated_label") or row.get("label") or "")
        page_num = _normalize_spaces(row.get("page") or "")
        if label:
            yield {
                "page_id": page_id,
                "block_index": idx,
                "line_index": 0,
                "phrase_index": 0,
                "unit_id": f"toc:p{page_id}:row{idx}:label",
                "role": "toc_label",
                "source_kind": "toc_row_label",
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "translatable": True,
                "source_text": label,
                "translated_text": translated_label,
            }
        if page_num:
            yield {
                "page_id": page_id,
                "block_index": idx,
                "line_index": 0,
                "phrase_index": 1,
                "unit_id": f"toc:p{page_id}:row{idx}:page",
                "role": "toc_page_number",
                "source_kind": "toc_row_page",
                "translation_strategy": "exact_preserve",
                "coverage_required": "strict",
                "translatable": False,
                "source_text": page_num,
                "translated_text": page_num,
            }


def _iter_units(page):
    toc_units = list(_iter_toc_units(page) or [])
    if toc_units:
        for unit in toc_units:
            yield unit
        return
    for unit in _iter_phrase_units(page):
        yield unit


def _classify_unit_status(unit, target_lang="fr"):
    source_text = unit["source_text"]
    translated_text = unit["translated_text"]
    strategy = unit["translation_strategy"]
    required = unit["coverage_required"]
    translatable = unit["translatable"]
    unit_type = _normalize_spaces(unit.get("unit_type") or "").lower()

    if not source_text:
        return "ignored", "empty_source"
    if required != "strict":
        return "optional", "coverage_optional"
    if not translatable or strategy == "exact_preserve":
        visible_text = _normalize_spaces(unit.get("visible_text") or "")
        if translated_text == source_text:
            return "covered", "preserved"
        if visible_text and _normalize_text(visible_text) and _normalize_text(visible_text) in _normalize_text(translated_text):
            return "covered", "preserved_visible_snippet"
        return "warning", "preserve_mismatch"
    if _looks_page_number(source_text):
        return "covered", "page_marker"
    if _looks_symbolic(source_text):
        return "covered", "symbolic"
    if not translated_text:
        return "missing", "missing_translation"

    src_norm = _normalize_text(source_text)
    tgt_norm = _normalize_text(translated_text)
    src_sig = _text_signature(source_text)
    tgt_sig = _text_signature(translated_text)

    if strategy == "layout_constrained":
        if src_norm == tgt_norm:
            if unit_type in {"reference_link", "citation"}:
                return "covered", "reference_or_citation_preserved"
            if target_lang == "fr" and _is_figure_reference_label(source_text):
                return "covered", "figure_reference_preserved"
            if target_lang == "fr" and _is_shared_technical_term_fr(source_text):
                return "covered", "shared_technical_term"
            if target_lang == "fr" and _is_preservable_symbolic_clause(
                source_text,
                document_type=unit.get("document_type") or "",
                layout_type=unit.get("layout_type") or "",
            ):
                return "covered", "preserved_symbolic_clause"
            if target_lang == "fr" and _is_preservable_short_label_fr(source_text, unit_type=unit_type):
                return "covered", "preserved_short_label"
            if target_lang == "fr" and re.search(r"[A-Za-z]", source_text):
                return "warning", "layout_unchanged"
            return "covered", "layout_preserved"
        retained = _retained_source_token_count(source_text, translated_text)
        if retained >= 2 and _english_marker_count(translated_text) >= 1:
            return "warning", "source_language_leak"
        if len(_tokenize(translated_text)) <= 1 and len(_tokenize(source_text)) >= 3:
            return "warning", "overcompressed"
        return "covered", "layout_translated"

    if strategy == "semantic_reflow":
        if src_norm == tgt_norm:
            return "warning", "unchanged_reflow"
        if len(_tokenize(translated_text)) < max(1, int(len(_tokenize(source_text)) * 0.3)):
            return "warning", "too_short"
        retained = _retained_source_token_count(source_text, translated_text)
        if retained >= 3 and _english_marker_count(translated_text) >= 2:
            return "warning", "source_language_leak"
        return "covered", "semantic_translated"

    return "covered", "default"


def analyze_document_coverage(source_pages, translated_pages, target_lang="fr"):
    src_units = []
    tr_units = []
    for page in source_pages or []:
        src_units.extend(list(_iter_units(page)))
    for page in translated_pages or []:
        tr_units.extend(list(_iter_units(page)))

    translated_by_id = {u["unit_id"]: u for u in tr_units}
    findings = []
    summary = {
        "source_units": 0,
        "strict_units": 0,
        "covered_units": 0,
        "warning_units": 0,
        "missing_units": 0,
        "preserved_units": 0,
        "layout_constrained_units": 0,
        "semantic_reflow_units": 0,
        "coverage_score": 0.0,
        "publication_blocked": False,
    }

    for src in src_units:
        translated = translated_by_id.get(src["unit_id"], {})
        unit = dict(src)
        unit["translated_text"] = translated.get("translated_text", src.get("translated_text", ""))
        status, reason = _classify_unit_status(unit, target_lang=target_lang)
        strategy = unit["translation_strategy"]
        if strategy == "exact_preserve":
            summary["preserved_units"] += 1
        elif strategy == "layout_constrained":
            summary["layout_constrained_units"] += 1
        elif strategy == "semantic_reflow":
            summary["semantic_reflow_units"] += 1

        if unit["source_text"]:
            summary["source_units"] += 1
        if unit["coverage_required"] == "strict" and unit["source_text"]:
            summary["strict_units"] += 1
        if status == "covered":
            summary["covered_units"] += 1
        elif status == "warning":
            summary["warning_units"] += 1
        elif status == "missing":
            summary["missing_units"] += 1

        if status in {"warning", "missing"}:
            findings.append(
                {
                    "unit_id": unit["unit_id"],
                    "page_id": unit["page_id"],
                    "role": unit["role"],
                    "source_kind": unit["source_kind"],
                    "translation_strategy": unit["translation_strategy"],
                    "coverage_required": unit["coverage_required"],
                    "status": status,
                    "reason": reason,
                    "source_text": unit["source_text"],
                    "translated_text": unit["translated_text"],
                }
            )

    strict_units = max(1, summary["strict_units"])
    penalty = summary["warning_units"] * 0.5 + summary["missing_units"] * 1.0
    score = max(0.0, 1.0 - (penalty / strict_units))
    summary["coverage_score"] = round(score, 4)
    summary["publication_blocked"] = summary["missing_units"] > 0
    findings.sort(key=lambda x: (0 if x["status"] == "missing" else 1, x["page_id"], x["unit_id"]))
    return {
        "summary": summary,
        "findings": findings[:200],
    }


def _extract_rendered_page_texts(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    try:
        for page in doc:
            words = page.get_text("words") or []
            if words:
                words = sorted(words, key=lambda w: (w[5], w[6], w[7], round(w[1], 3), round(w[0], 3)))
                text = " ".join(str(w[4]) for w in words if str(w[4]).strip())
            else:
                text = page.get_text("text") or ""
            pages.append(_normalize_spaces(text))
    finally:
        doc.close()
    return pages


def _expected_rendered_text(unit):
    strategy = unit.get("translation_strategy") or "semantic_reflow"
    translatable = bool(unit.get("translatable", True))
    render_policy = _normalize_spaces(unit.get("render_policy") or "").lower()
    block_render_policy = _normalize_spaces(unit.get("block_render_policy") or "").lower()
    render_mode = _normalize_spaces(unit.get("render_mode") or "").lower()
    visible_text = _normalize_spaces(unit.get("visible_text") or "")
    if render_policy == "background_only" or render_mode == "background_only":
        return ""
    if block_render_policy == "paragraph_flow" and str(unit.get("role") or "").strip().lower() == "body":
        if not bool(unit.get("block_is_primary_render_unit")):
            return ""
        block_text = _normalize_spaces(unit.get("block_translated_text") or unit.get("block_source_text") or "")
        if block_text:
            return block_text
    if bool(unit.get("all_spans_skip_render")):
        return ""
    if bool(unit.get("has_skipped_spans")) and visible_text and (not translatable or strategy == "exact_preserve"):
        return visible_text
    if not translatable or strategy == "exact_preserve":
        return _normalize_spaces(unit.get("source_text") or unit.get("translated_text") or "")
    return _normalize_spaces(unit.get("translated_text") or unit.get("source_text") or "")


def _page_index_for_unit(unit, page_count):
    raw = unit.get("page_id", 0)
    try:
        idx = int(raw)
    except Exception:
        idx = 0
    if idx < 0:
        idx = 0
    if idx >= page_count and idx > 0:
        idx -= 1
    if idx < 0:
        idx = 0
    if idx >= page_count:
        idx = max(0, page_count - 1)
    return idx


def _ordered_subsequence_ratio(expected_tokens, page_tokens):
    if not expected_tokens:
        return 0.0
    pos = 0
    matched = 0
    for tok in expected_tokens:
        while pos < len(page_tokens) and page_tokens[pos] != tok:
            pos += 1
        if pos >= len(page_tokens):
            break
        matched += 1
        pos += 1
    return matched / max(1, len(expected_tokens))


def _classify_rendered_presence(unit, page_text, full_text=""):
    expected = _expected_rendered_text(unit)
    required = (unit.get("coverage_required") or "strict").lower()
    if required != "strict":
        return "optional", "coverage_optional", expected
    if not expected:
        return "ignored", "empty_expected", expected

    page_norm = _normalize_text(page_text)
    expected_norm = _normalize_text(expected)
    if not expected_norm:
        return "ignored", "empty_expected", expected
    if expected_norm in page_norm:
        return "covered", "substring_match", expected
    full_norm = _normalize_text(full_text)
    if full_norm and expected_norm in full_norm:
        return "covered", "document_substring_match", expected

    expected_tokens = _tokenize(expected)
    page_tokens = _tokenize(page_text)
    full_tokens = _tokenize(full_text)
    if not expected_tokens:
        return "ignored", "empty_expected", expected
    if len(expected_tokens) == 1 and expected_tokens[0] in set(page_tokens):
        return "covered", "single_token_match", expected

    ratio = _ordered_subsequence_ratio(expected_tokens, page_tokens)
    if ratio >= 1.0:
        return "covered", "ordered_token_match", expected
    full_ratio = _ordered_subsequence_ratio(expected_tokens, full_tokens)
    if full_ratio >= 1.0:
        return "covered", "document_ordered_token_match", expected
    if _is_reference_like_render_unit(unit, expected):
        expected_sig = _text_signature(expected)
        page_sig = _text_signature(page_text)
        full_sig = _text_signature(full_text)
        page_sig_ratio = len(expected_sig & page_sig) / max(1, len(expected_sig))
        full_sig_ratio = len(expected_sig & full_sig) / max(1, len(expected_sig))
        if ratio >= 0.6 or page_sig_ratio >= 0.6:
            return "covered", "reference_like_token_match", expected
        if full_ratio >= 0.6 or full_sig_ratio >= 0.6:
            return "covered", "document_reference_like_token_match", expected
    role = _normalize_spaces(unit.get("role") or "").lower()
    if role == "header":
        page_compact = re.sub(r"\s+", "", page_norm)
        expected_compact = re.sub(r"\s+", "", expected_norm)
        full_compact = re.sub(r"\s+", "", full_norm)
        if expected_compact and expected_compact in page_compact:
            return "covered", "header_compact_match", expected
        if expected_compact and full_compact and expected_compact in full_compact:
            return "covered", "document_header_compact_match", expected
    if _is_numeric_heading_marker(expected):
        if ratio >= 0.6:
            return "covered", "heading_marker_partial_match", expected
        if full_ratio >= 0.6:
            return "covered", "document_heading_marker_partial_match", expected
    sentence_strict = bool(unit.get("preserve_sentence_integrity"))
    if ratio >= 0.6:
        return ("warning", "sentence_truncated", expected) if sentence_strict else ("warning", "partial_token_match", expected)
    if full_ratio >= 0.6:
        return ("warning", "document_sentence_truncated", expected) if sentence_strict else ("warning", "document_partial_token_match", expected)
    if sentence_strict and (ratio >= 0.3 or full_ratio >= 0.3):
        return "missing", "sentence_not_fully_rendered", expected
    return "missing", "not_rendered", expected


def analyze_rendered_text_coverage(source_pages, translated_pages, pdf_path, target_lang="fr"):
    del source_pages, target_lang
    rendered_pages = _extract_rendered_page_texts(pdf_path)
    full_text = _normalize_spaces(" ".join(rendered_pages))
    translated_units = []
    for page in translated_pages or []:
        translated_units.extend(list(_iter_units(page)))

    findings = []
    summary = {
        "expected_units": 0,
        "strict_units": 0,
        "rendered_covered_units": 0,
        "rendered_warning_units": 0,
        "rendered_missing_units": 0,
        "rendered_coverage_score": 0.0,
        "publication_blocked": False,
    }

    for unit in translated_units:
        expected = _expected_rendered_text(unit)
        if expected:
            summary["expected_units"] += 1
        if (unit.get("coverage_required") or "strict").lower() == "strict" and expected:
            summary["strict_units"] += 1

        page_idx = _page_index_for_unit(unit, len(rendered_pages))
        page_text = rendered_pages[page_idx] if rendered_pages else ""
        status, reason, expected_text = _classify_rendered_presence(unit, page_text, full_text=full_text)

        if status == "covered":
            summary["rendered_covered_units"] += 1
        elif status == "warning":
            summary["rendered_warning_units"] += 1
        elif status == "missing":
            summary["rendered_missing_units"] += 1

        if status in {"warning", "missing"}:
            findings.append(
                {
                    "unit_id": unit.get("unit_id"),
                    "page_id": unit.get("page_id"),
                    "role": unit.get("role"),
                    "source_kind": unit.get("source_kind"),
                    "translation_strategy": unit.get("translation_strategy"),
                    "coverage_required": unit.get("coverage_required"),
                    "status": status,
                    "reason": reason,
                    "source_text": unit.get("source_text", ""),
                    "translated_text": unit.get("translated_text", ""),
                    "expected_rendered_text": expected_text,
                }
            )

    strict_units = max(1, summary["strict_units"])
    penalty = summary["rendered_warning_units"] * 0.5 + summary["rendered_missing_units"] * 1.0
    summary["rendered_coverage_score"] = round(max(0.0, 1.0 - (penalty / strict_units)), 4)
    summary["publication_blocked"] = (
        summary["rendered_missing_units"] > 0 or summary["rendered_warning_units"] > 0
    )
    findings.sort(key=lambda x: (0 if x["status"] == "missing" else 1, x["page_id"], x["unit_id"]))
    return {
        "summary": summary,
        "findings": findings[:200],
    }
