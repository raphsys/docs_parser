import re
from collections import Counter
from page_family_registry import PAGE_FAMILY_REGISTRY, get_family_group
from page_profile_registry import DOCUMENT_TYPES, LAYOUT_TYPES, STYLE_PROFILES


class PageCaseClassifier:
    KNOWN_PAGE_FAMILIES = {name for name in PAGE_FAMILY_REGISTRY.keys() if name != "unknown"}

    def _block_text(self, block):
        text = (block.get("translated_text") or block.get("text") or "").strip()
        if text:
            return re.sub(r"\s+", " ", text)
        parts = []
        for line in block.get("lines") or []:
            line_text = (line.get("translated_text") or line.get("line_text") or "").strip()
            if line_text:
                parts.append(line_text)
                continue
            for phrase in line.get("phrases") or []:
                phrase_text = (phrase.get("translated_text") or phrase.get("text") or phrase.get("texte") or "").strip()
                if phrase_text:
                    parts.append(phrase_text)
        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    def _bbox_area(self, bbox):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return 0.0
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except Exception:
            return 0.0
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def _safe_ratio(self, a, b):
        if not b:
            return 0.0
        return round(float(a) / float(b), 4)

    def _mean(self, values):
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return 0.0
        return round(sum(vals) / float(len(vals)), 4)

    def _is_code_like_text(self, text):
        s = self._block_text({"text": text})
        if not s or len(s) > 200:
            return False
        if re.search(r"\b(from|import|def|class|return|lambda)\b", s, flags=re.IGNORECASE):
            return True
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", s):
            return True
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\(", s):
            return True
        if re.search(r"[{}\[\]_]|==|!=|<=|>=|=>|:=|=\s*['\"]", s):
            return True
        return False

    def _is_reference_like_text(self, text):
        s = self._block_text({"text": text})
        if not s:
            return False
        if re.search(r"(https?://\S+|www\.\S+|doi:\s*\S+|arxiv:\s*\S+)", s, flags=re.IGNORECASE):
            return True
        if re.search(r"\b(et al\.|Google AI Blog|blog|website)\b", s, flags=re.IGNORECASE):
            return True
        return False

    def _is_citation_like_text(self, text):
        s = self._block_text({"text": text})
        if not s:
            return False
        if re.search(r"\b(et al\.|vol\.|no\.|pp\.|ISBN|ISSN|Google AI Blog)\b", s, flags=re.IGNORECASE):
            return True
        if (
            re.search(r"[“\"].+[”\"]", s)
            and re.search(r"\b(19|20)\d{2}\b", s)
        ):
            return True
        return False

    def extract_features(self, page_data, lines, page_role="body"):
        blocks = page_data.get("blocks") or []
        images = page_data.get("images") or []
        drawings = page_data.get("drawings") or []
        non_text_zones = page_data.get("non_text_zones") or []
        layout = page_data.get("layout") or {}
        columns = layout.get("columns") or []
        dims = page_data.get("dimensions") or {}
        page_w = float(dims.get("width", 0.0) or 0.0)
        page_h = float(dims.get("height", 0.0) or 0.0)
        page_area = max(1.0, page_w * page_h)

        role_counter = Counter()
        figure_captions = 0
        body_blocks = 0
        short_native_labels = 0
        diagram_roles = 0
        tableish_lines = 0
        equation_blocks = 0
        section_heading_blocks = 0
        header_footer_blocks = 0
        short_text_blocks = 0
        short_label_lines = 0
        code_like_blocks = 0
        reference_like_blocks = 0
        citation_like_blocks = 0
        num_words = 0
        total_chars = 0
        digit_chars = 0
        uppercase_chars = 0
        punctuation_chars = 0
        currency_symbol_count = 0
        date_pattern_count = 0
        email_count = 0
        url_count = 0
        scientific_pattern_hits = 0
        form_pattern_hits = 0
        invoice_pattern_hits = 0
        text_block_count = 0
        title_block_count = 0
        table_block_count = 0
        picture_block_count = 0
        caption_block_count = 0
        header_block_count = 0
        footer_block_count = 0
        formula_block_count = 0
        list_block_count = 0
        font_sizes = []
        text_area = 0.0
        figure_area = 0.0
        table_area = 0.0
        block_areas = []
        left_edges = []
        top_edges = []
        line_word_counts = []
        line_char_counts = []

        scientific_terms = {"abstract", "keywords", "references", "bibliography", "appendix"}
        invoice_terms = {"invoice", "total", "subtotal", "tax", "amount", "vat", "balance"}
        toc_like_lines = 0
        page_marker_lines = 0

        for block in blocks:
            block_role = str(block.get("role") or "body").strip().lower()
            role_counter.update([block_role])
            text = self._block_text(block)
            source = str(block.get("source") or "ocr").strip().lower()
            line_count = len(block.get("lines") or [])
            block_bbox = block.get("bbox") or []
            area = self._bbox_area(block_bbox)
            if area > 0:
                block_areas.append(area)
                left_edges.append(float(block_bbox[0]))
                top_edges.append(float(block_bbox[1]))

            if block_role == "figure_caption":
                figure_captions += 1
                caption_block_count += 1
            if block_role == "body":
                body_blocks += 1
                text_block_count += 1
            if block_role in {"diagram_label", "diagram_text_label", "axis_label", "legend_label"}:
                diagram_roles += 1
            if block_role == "equation_inline":
                equation_blocks += 1
                formula_block_count += 1
            if block_role == "section_heading":
                section_heading_blocks += 1
                title_block_count += 1
            if block_role in {"header", "footer"}:
                header_footer_blocks += 1
            if block_role == "header":
                header_block_count += 1
            if block_role == "footer":
                footer_block_count += 1
            if block_role in {"title", "subtitle"}:
                title_block_count += 1
            if text and len(text) <= 32:
                short_text_blocks += 1
            if self._is_code_like_text(text):
                code_like_blocks += 1
            if self._is_reference_like_text(text):
                reference_like_blocks += 1
            if self._is_citation_like_text(text):
                citation_like_blocks += 1
            if (
                block_role == "title"
                and source == "native"
                and text
                and len(text) <= 120
                and line_count <= 3
            ):
                short_native_labels += 1
            if block_role in {"list_item", "list_marker"}:
                list_block_count += 1

            if block_role in {"body", "section_heading", "title", "subtitle", "figure_caption"}:
                text_area += area
            if block_role in {"figure_caption"}:
                figure_area += area

            low_text = text.lower()
            if any(term in low_text for term in scientific_terms):
                scientific_pattern_hits += 1
            if any(term in low_text for term in invoice_terms):
                invoice_pattern_hits += 1
            if re.search(r"\b(name|date|signature|address|phone|email)\s*:", low_text):
                form_pattern_hits += 1
            currency_symbol_count += len(re.findall(r"[$€£¥]", text))
            date_pattern_count += len(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text))
            email_count += len(re.findall(r"\b\S+@\S+\.\S+\b", text))
            url_count += len(re.findall(r"(https?://\S+|www\.\S+)", text, flags=re.IGNORECASE))

            block_has_tableish = False
            for line in block.get("lines") or []:
                line_text = (line.get("translated_text") or line.get("line_text") or "").strip()
                if not line_text:
                    continue
                words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", line_text)
                line_word_counts.append(len(words))
                line_char_counts.append(len(line_text))
                num_words += len(words)
                total_chars += len(line_text)
                digit_chars += sum(1 for ch in line_text if ch.isdigit())
                uppercase_chars += sum(1 for ch in line_text if ch.isupper())
                punctuation_chars += sum(1 for ch in line_text if not ch.isalnum() and not ch.isspace())
                word_count = len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", line_text))
                if (
                    word_count >= 1
                    and word_count <= 7
                    and len(line_text) <= 64
                    and not re.search(r"[.;:!?]$", line_text)
                    and not self._is_reference_like_text(line_text)
                    and not self._is_code_like_text(line_text)
                ):
                    short_label_lines += 1
                if (
                    "|" in line_text
                    or "\t" in line_text
                    or len(re.findall(r"\b\d+(?:[.,]\d+)?\b", line_text)) >= 3
                ):
                    tableish_lines += 1
                    block_has_tableish = True
                if re.search(r"\.{2,}\s*\d{1,4}$", line_text) or re.search(r"\b\d{1,4}$", line_text):
                    page_marker_lines += 1
                if re.search(r"\.{2,}\s*\d{1,4}$", line_text):
                    toc_like_lines += 1
                for phrase in line.get("phrases") or []:
                    for span in phrase.get("spans") or []:
                        style = span.get("style") or {}
                        size = style.get("size")
                        if size:
                            try:
                                font_sizes.append(float(size))
                            except Exception:
                                pass
            if block_has_tableish:
                table_area += area
                table_block_count += 1

        image_count = len(images)
        drawing_count = len(drawings)
        major_non_text = image_count + len(non_text_zones)
        visual_non_text = major_non_text + drawing_count
        total_lines = max(1, len(lines or []))
        short_line_count = 0
        for entry in lines or []:
            text = self._block_text(entry.get("block") or {})
            if text and len(text) <= 80:
                short_line_count += 1

        whitespace_ratio = max(0.0, 1.0 - min(1.0, (text_area + sum(self._bbox_area(item.get("bbox")) for item in images or [])) / page_area))
        scientific_pattern_score = min(1.0, round((scientific_pattern_hits * 1.5 + citation_like_blocks + equation_blocks) / 8.0, 4))
        form_pattern_score = min(1.0, round((form_pattern_hits + max(0, tableish_lines - 1)) / 6.0, 4))
        invoice_pattern_score = min(1.0, round((invoice_pattern_hits + currency_symbol_count + date_pattern_count) / 8.0, 4))
        toc_pattern_score = min(1.0, round((toc_like_lines * 2 + page_marker_lines) / 10.0, 4))

        return {
            "page_role": str(page_role or "body").strip().lower(),
            "page_width": round(page_w, 2),
            "page_height": round(page_h, 2),
            "aspect_ratio": round(page_w / page_h, 4) if page_h else 0.0,
            "column_count": len(columns) or 1,
            "block_count": len(blocks),
            "line_count": total_lines,
            "role_counts": dict(role_counter),
            "figure_captions": figure_captions,
            "body_blocks": body_blocks,
            "short_native_labels": short_native_labels,
            "diagram_roles": diagram_roles,
            "tableish_lines": tableish_lines,
            "equation_blocks": equation_blocks,
            "section_heading_blocks": section_heading_blocks,
            "header_footer_blocks": header_footer_blocks,
            "short_text_blocks": short_text_blocks,
            "short_label_lines": short_label_lines,
            "code_like_blocks": code_like_blocks,
            "reference_like_blocks": reference_like_blocks,
            "citation_like_blocks": citation_like_blocks,
            "short_line_ratio": round(short_line_count / float(total_lines), 4),
            "image_count": image_count,
            "drawing_count": drawing_count,
            "major_non_text": major_non_text,
            "visual_non_text": visual_non_text,
            "num_blocks_total": len(blocks),
            "num_text_blocks": text_block_count,
            "num_title_blocks": title_block_count,
            "num_table_blocks": table_block_count,
            "num_figure_blocks": image_count,
            "num_caption_blocks": caption_block_count,
            "num_header_blocks": header_block_count,
            "num_footer_blocks": footer_block_count,
            "num_formula_blocks": formula_block_count,
            "num_list_blocks": list_block_count,
            "num_words": num_words,
            "avg_words_per_line": self._mean(line_word_counts),
            "avg_chars_per_line": self._mean(line_char_counts),
            "digit_ratio": self._safe_ratio(digit_chars, total_chars),
            "uppercase_ratio": self._safe_ratio(uppercase_chars, total_chars),
            "punctuation_ratio": self._safe_ratio(punctuation_chars, total_chars),
            "currency_symbol_count": currency_symbol_count,
            "date_pattern_count": date_pattern_count,
            "email_count": email_count,
            "url_count": url_count,
            "toc_pattern_score": toc_pattern_score,
            "toc_like_lines": toc_like_lines,
            "page_marker_lines": page_marker_lines,
            "form_pattern_score": form_pattern_score,
            "scientific_pattern_score": scientific_pattern_score,
            "invoice_pattern_score": invoice_pattern_score,
            "text_coverage_ratio": self._safe_ratio(text_area, page_area),
            "table_coverage_ratio": self._safe_ratio(table_area, page_area),
            "figure_coverage_ratio": self._safe_ratio(figure_area + sum(self._bbox_area(item.get("bbox")) for item in images or []), page_area),
            "whitespace_ratio": round(whitespace_ratio, 4),
            "block_area_mean": self._mean(block_areas),
            "font_size_levels": len({round(v, 1) for v in font_sizes}),
            "font_size_mean": self._mean(font_sizes),
        }

    def _build_regions(self, page_data):
        regions = []
        for block in page_data.get("blocks") or []:
            bbox = block.get("bbox")
            if not bbox:
                continue
            block_role = str(block.get("role") or "text").strip().lower()
            region_type = {
                "section_heading": "section_header",
                "figure_caption": "caption",
                "equation_inline": "formula",
                "equation_block": "formula",
                "body": "text",
            }.get(block_role, block_role)
            regions.append({
                "type": region_type,
                "bbox": bbox,
                "score": 0.9,
            })
        return regions

    def _pick_best_label(self, scores, default):
        if not isinstance(scores, dict) or not scores:
            return default, 0.0
        label = max(scores, key=scores.get)
        return label, round(float(scores.get(label, 0.0)), 4)

    def score_document_types(self, features, page_family="unknown"):
        scores = {name: 0.0 for name in DOCUMENT_TYPES}
        col_count = int(features.get("column_count", 1))
        scientific = float(features.get("scientific_pattern_score", 0.0))
        form_score = float(features.get("form_pattern_score", 0.0))
        invoice_score = float(features.get("invoice_pattern_score", 0.0))
        toc_score = float(features.get("toc_pattern_score", 0.0))
        figure_ratio = float(features.get("figure_coverage_ratio", 0.0))
        url_count = int(features.get("url_count", 0))
        citation_like = int(features.get("citation_like_blocks", 0))

        if page_family == "toc" or toc_score >= 0.65:
            scores["book_page"] += 0.72
            scores["manual_guide"] += 0.48
        if col_count >= 2 and scientific >= 0.35:
            scores["scientific_paper"] += 0.85
        if page_family.startswith("body_text") and scientific < 0.35:
            scores["book_page"] += 0.72
        if page_family in {"body_with_figure", "body_with_diagram", "illustrated_label_page", "chart_label_page"}:
            scores["manual_guide"] += 0.72
            scores["book_page"] += 0.42
        if features.get("figure_captions", 0) >= 1 and scientific < 0.35:
            scores["manual_guide"] += 0.2
        if page_family in {"mixed_page", "mixed_dense_illustrated", "mixed_formula_annotation_page"}:
            scores["report"] += 0.55
            scores["manual_guide"] += 0.45
        if form_score >= 0.5:
            scores["form"] += 0.88
        if invoice_score >= 0.45:
            scores["invoice"] += 0.9
            scores["receipt"] += 0.45
        if url_count >= 2 and scientific < 0.2:
            scores["web_print"] += 0.7
        if citation_like >= 2 and scientific >= 0.35:
            scores["scientific_paper"] += 0.08
        if figure_ratio >= 0.45 and int(features.get("num_words", 0)) <= 80:
            scores["slide"] += 0.75
            scores["advertisement_poster"] += 0.45
        if max(scores.values()) <= 0.01:
            scores["mixed_unknown"] = 0.5
        else:
            scores["mixed_unknown"] = 0.15
        return {k: round(min(1.0, max(0.0, v)), 4) for k, v in scores.items()}

    def score_layout_types(self, features, page_family="unknown", page_role="body"):
        scores = {name: 0.0 for name in LAYOUT_TYPES}
        col_count = int(features.get("column_count", 1))
        figure_ratio = float(features.get("figure_coverage_ratio", 0.0))
        table_ratio = float(features.get("table_coverage_ratio", 0.0))
        text_ratio = float(features.get("text_coverage_ratio", 0.0))
        scientific = float(features.get("scientific_pattern_score", 0.0))
        table_blocks = int(features.get("num_table_blocks", 0))
        visual_non_text = int(features.get("visual_non_text", 0))

        if page_role == "toc":
            scores["toc_page"] = 0.98
        if col_count == 1:
            scores["single_column"] += 0.72
        elif col_count == 2:
            scores["double_column"] += 0.82
        elif col_count >= 3:
            scores["multi_column"] += 0.9
        if page_family in {"table_page", "table_diagram_example"} or table_blocks >= 2 or table_ratio >= 0.28:
            scores["table_dominant"] += 0.9
        if figure_ratio >= 0.35 and text_ratio <= 0.35:
            scores["image_dominant"] += 0.82
        if page_family in {"body_with_diagram", "illustrated_label_page", "chart_label_page", "mixed_formula_annotation_page"}:
            scores["annotated_page"] += 0.88
        if visual_non_text >= 4 and int(features.get("short_label_lines", 0)) >= 4:
            scores["annotated_page"] += 0.12
        if page_family in {"mixed_page", "mixed_dense_illustrated"}:
            scores["mixed_blocks"] += 0.82
        if text_ratio >= 0.45 and figure_ratio < 0.2:
            scores["dense_text"] += 0.74
        if page_family in {"narrative_reference_page", "citation_heavy_body_page"} or scientific >= 0.45:
            scores["reference_page"] += 0.52
        if col_count == 2 and text_ratio >= 0.35 and table_blocks == 0:
            scores["double_column"] += 0.08
            scores["table_dominant"] = max(0.0, scores["table_dominant"] - 0.15)
        return {k: round(min(1.0, max(0.0, v)), 4) for k, v in scores.items()}

    def score_style_profiles(self, features, page_family="unknown", page_role="body"):
        scores = {name: 0.0 for name in STYLE_PROFILES}
        text_ratio = float(features.get("text_coverage_ratio", 0.0))
        figure_ratio = float(features.get("figure_coverage_ratio", 0.0))
        whitespace = float(features.get("whitespace_ratio", 0.0))
        table_ratio = float(features.get("table_coverage_ratio", 0.0))
        scientific = float(features.get("scientific_pattern_score", 0.0))

        if scientific >= 0.45 or page_family in {"body_text_two_column_equations", "citation_heavy_body_page"}:
            scores["academic_dense"] += 0.84
        if page_family in {"table_page", "table_diagram_example"} or table_ratio >= 0.28:
            scores["tabular_structured"] += 0.88
        if page_family in {"illustrated_label_page", "chart_label_page", "body_with_figure"}:
            scores["editorial_visual"] += 0.8
        if int(features.get("visual_non_text", 0)) >= 4 and int(features.get("short_label_lines", 0)) >= 4:
            scores["editorial_visual"] += 0.08
        if figure_ratio >= 0.45 and whitespace >= 0.35:
            scores["marketing_visual"] += 0.72
        if page_role == "toc" or (whitespace >= 0.5 and text_ratio < 0.35):
            scores["minimalist"] += 0.52
        if page_family in {"mixed_page", "mixed_dense_illustrated", "mixed_formula_annotation_page"}:
            scores["mixed_irregular"] += 0.82
        if max(scores.values()) <= 0.01 and text_ratio >= 0.4:
            scores["administrative_clean"] += 0.5
        return {k: round(min(1.0, max(0.0, v)), 4) for k, v in scores.items()}

    def score_known_types(self, features):
        role = str(features.get("page_role") or "body").lower()
        if role == "toc":
            return {"toc": 1.0}

        scores = {
            "body_text": 0.15,
            "body_text_two_column": 0.0,
            "body_text_two_column_sectioned": 0.0,
            "body_text_two_column_equations": 0.0,
            "body_text_single_column_sparse": 0.0,
            "body_with_figure": 0.0,
            "body_with_diagram": 0.0,
            "illustrated_label_page": 0.0,
            "chart_label_page": 0.0,
            "table_page": 0.0,
            "table_diagram_example": 0.0,
            "mixed_page": 0.0,
            "mixed_dense_illustrated": 0.0,
            "mixed_formula_annotation_page": 0.0,
            "narrative_reference_page": 0.0,
            "citation_heavy_body_page": 0.0,
        }

        col_count = int(features.get("column_count", 1))
        visual_non_text = int(features.get("visual_non_text", 0))
        major_non_text = int(features.get("major_non_text", 0))
        figure_captions = int(features.get("figure_captions", 0))
        equation_blocks = int(features.get("equation_blocks", 0))
        section_heading_blocks = int(features.get("section_heading_blocks", 0))
        header_footer_blocks = int(features.get("header_footer_blocks", 0))
        body_blocks = int(features.get("body_blocks", 0))
        short_text_blocks = int(features.get("short_text_blocks", 0))
        short_label_lines = int(features.get("short_label_lines", 0))
        block_count = int(features.get("block_count", 0))
        short_native_labels = int(features.get("short_native_labels", 0))
        code_like_blocks = int(features.get("code_like_blocks", 0))
        reference_like_blocks = int(features.get("reference_like_blocks", 0))
        citation_like_blocks = int(features.get("citation_like_blocks", 0))

        if int(features.get("tableish_lines", 0)) >= 6:
            scores["table_page"] += 0.75
        if int(features.get("tableish_lines", 0)) >= 2 and short_native_labels >= 3 and equation_blocks >= 1:
            scores["table_diagram_example"] += 0.82
        if visual_non_text >= 1 and (short_native_labels >= 4 or short_label_lines >= 5):
            scores["illustrated_label_page"] += 0.8
        elif visual_non_text >= 1 and short_label_lines >= 4 and body_blocks >= 3:
            scores["illustrated_label_page"] += 0.72
        if (
            visual_non_text >= 1
            and (short_native_labels >= 3 or short_label_lines >= 3)
            and (
                int(features.get("tableish_lines", 0)) >= 1
                or figure_captions >= 1
                or int(features.get("drawing_count", 0)) >= 5
            )
        ):
            scores["chart_label_page"] += 0.78
            if figure_captions >= 1 and visual_non_text >= 5 and short_native_labels >= 3:
                scores["chart_label_page"] += 0.08
        if figure_captions >= 1:
            scores["body_with_figure"] += 0.45
        if major_non_text >= 1:
            scores["body_with_figure"] += 0.15
            scores["body_with_diagram"] += 0.2
            scores["mixed_page"] += 0.2
        if int(features.get("short_native_labels", 0)) >= 2:
            scores["body_with_figure"] += 0.2
        if int(features.get("diagram_roles", 0)) >= 1:
            scores["body_with_diagram"] += 0.45
        if equation_blocks >= 1:
            scores["body_with_diagram"] += 0.1
            scores["body_with_figure"] += 0.1
        if body_blocks >= 1 and major_non_text >= 1:
            scores["mixed_page"] += 0.25
        if body_blocks >= 2 and major_non_text >= 1 and short_native_labels >= 3:
            scores["mixed_dense_illustrated"] += 0.8
            scores["illustrated_label_page"] = max(0.0, scores["illustrated_label_page"] - 0.08)
        if body_blocks >= 1 and major_non_text >= 1 and equation_blocks >= 2 and short_native_labels >= 2:
            scores["mixed_formula_annotation_page"] += 0.84
        if visual_non_text == 0 and reference_like_blocks >= 1 and body_blocks >= 1:
            scores["narrative_reference_page"] += 0.78
        if visual_non_text == 0 and citation_like_blocks >= 1 and body_blocks >= 1:
            scores["citation_heavy_body_page"] += 0.88
        if body_blocks >= 3 and major_non_text == 0:
            scores["body_text"] += 0.35
        if code_like_blocks >= 1 and int(features.get("tableish_lines", 0)) >= 1:
            scores["table_diagram_example"] += 0.08
        if header_footer_blocks >= 2 and col_count >= 2:
            scores["body_text"] += 0.05
            scores["mixed_page"] += 0.05
        if float(features.get("short_line_ratio", 0.0)) >= 0.55:
            scores["body_with_diagram"] += 0.05
            scores["mixed_page"] += 0.05
        if visual_non_text >= 1 and short_label_lines >= 3:
            scores["body_with_diagram"] += 0.18
            scores["mixed_page"] += 0.1

        if major_non_text == 0 and figure_captions == 0:
            if col_count >= 2 and body_blocks >= 1:
                scores["body_text_two_column"] += 0.6
                if header_footer_blocks >= 1:
                    scores["body_text_two_column"] += 0.1
                if section_heading_blocks >= 1:
                    scores["body_text_two_column_sectioned"] += 0.72
                if equation_blocks >= 1:
                    scores["body_text_two_column_equations"] += 0.72
                if section_heading_blocks >= 1 and equation_blocks >= 1:
                    scores["body_text_two_column_equations"] += 0.08
            elif col_count == 1 and visual_non_text <= 1 and block_count <= 4 and short_text_blocks >= max(1, block_count - 1):
                scores["body_text_single_column_sparse"] += 0.62

        if scores["body_text_two_column_sectioned"] > 0:
            scores["body_text_two_column"] = max(
                0.0, scores["body_text_two_column"] - 0.1
            )
        if scores["body_text_two_column_equations"] > 0:
            scores["body_text_two_column"] = max(
                0.0, scores["body_text_two_column"] - 0.1
            )

        if scores["table_page"] >= 0.75:
            scores["body_text"] = min(scores["body_text"], 0.2)
            scores["mixed_page"] = min(scores["mixed_page"], 0.2)
        if scores["table_diagram_example"] >= 0.75:
            scores["table_page"] = max(scores["table_page"], 0.7)
            scores["body_with_diagram"] = max(scores["body_with_diagram"], 0.45)
            scores["mixed_page"] = min(scores["mixed_page"], 0.25)
        if scores["illustrated_label_page"] >= 0.75:
            scores["body_with_diagram"] = max(scores["body_with_diagram"], 0.6)
            scores["body_with_figure"] = max(scores["body_with_figure"], 0.35)
        if scores["chart_label_page"] >= 0.75:
            scores["body_with_figure"] = min(max(scores["body_with_figure"], 0.6), 0.72)
            scores["mixed_dense_illustrated"] = min(scores["mixed_dense_illustrated"], 0.7)
            scores["mixed_page"] = min(scores["mixed_page"], 0.45)
        if scores["mixed_formula_annotation_page"] >= 0.75:
            scores["mixed_page"] = max(scores["mixed_page"], 0.75)
            scores["body_with_diagram"] = max(scores["body_with_diagram"], 0.45)
        if scores["mixed_dense_illustrated"] >= 0.75:
            scores["mixed_page"] = max(scores["mixed_page"], 0.7)
            scores["body_with_diagram"] = max(scores["body_with_diagram"], 0.4)
        if scores["narrative_reference_page"] >= 0.75:
            scores["body_text"] = max(scores["body_text"], 0.55)
        if scores["citation_heavy_body_page"] >= 0.75:
            scores["body_text"] = max(scores["body_text"], 0.55)

        return {k: round(min(1.0, max(0.0, v)), 4) for k, v in scores.items()}

    def classify(self, page_data, lines, page_role="body"):
        features = self.extract_features(page_data, lines, page_role=page_role)
        scores = self.score_known_types(features)
        if not scores:
            return {
                "document_type": "mixed_unknown",
                "layout_type": "mixed_blocks",
                "style_profile": "mixed_irregular",
                "confidence": {
                    "document_type": 0.0,
                    "layout_type": 0.0,
                    "page_role": 0.98 if page_role == "toc" else 0.0,
                    "style_profile": 0.0,
                    "page_family": 0.0,
                },
                "document_scores": {},
                "layout_scores": {},
                "style_scores": {},
                "regions": self._build_regions(page_data),
                "page_family": "unknown",
                "page_family_group": "unknown",
                "family_confidence": 0.0,
                "best_known_family": "body_text",
                "best_known_family_group": "body_text",
                "known_scores": {},
                "is_known_family": False,
                "fallback_policy": "safe_mixed",
                "unknown_signature": "empty_or_unclassified",
                "features": features,
            }

        best_family = max(scores, key=scores.get)
        best_score = float(scores.get(best_family, 0.0))
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = best_score - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
        is_known = best_score >= 0.55 and margin >= 0.08

        if is_known:
            page_family = best_family
            fallback_policy = f"known::{best_family}"
            unknown_signature = ""
        else:
            page_family = "unknown"
            fallback_policy = "safe_mixed"
            unknown_signature = (
                f"role={features['page_role']}"
                f"|cols={features['column_count']}"
                f"|non_text={features['visual_non_text']}"
                f"|captions={features['figure_captions']}"
                f"|diagram={features['diagram_roles']}"
                f"|tableish={features['tableish_lines']}"
                f"|equations={features['equation_blocks']}"
            )

        document_scores = self.score_document_types(features, page_family=page_family)
        layout_scores = self.score_layout_types(features, page_family=page_family, page_role=page_role)
        style_scores = self.score_style_profiles(features, page_family=page_family, page_role=page_role)
        document_type, document_conf = self._pick_best_label(document_scores, "mixed_unknown")
        layout_type, layout_conf = self._pick_best_label(layout_scores, "mixed_blocks")
        style_profile, style_conf = self._pick_best_label(style_scores, "mixed_irregular")

        return {
            "document_type": document_type,
            "layout_type": layout_type,
            "style_profile": style_profile,
            "confidence": {
                "document_type": document_conf,
                "layout_type": layout_conf,
                "page_role": 0.98 if page_role == "toc" else 0.75,
                "style_profile": style_conf,
                "page_family": round(best_score, 4),
            },
            "document_scores": document_scores,
            "layout_scores": layout_scores,
            "style_scores": style_scores,
            "regions": self._build_regions(page_data),
            "page_family": page_family,
            "page_family_group": get_family_group(page_family),
            "family_confidence": round(best_score, 4),
            "best_known_family": best_family,
            "best_known_family_group": get_family_group(best_family),
            "known_scores": scores,
            "is_known_family": is_known,
            "fallback_policy": fallback_policy,
            "unknown_signature": unknown_signature,
            "features": features,
        }
