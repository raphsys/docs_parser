import re


class PagePolicyMatrix:
    def _norm(self, value, default=""):
        raw = default if value is None else value
        return str(raw).strip().lower()

    def _is_equation_like(self, text):
        s = self._norm(text)
        if not s:
            return False
        if re.search(r"[=<>±×÷∑∫∞≈≠≤≥√∆∂µλΩα-ωΑ-Ω]", s):
            return True
        if re.search(r"\b[a-z]\s*/\s*[a-z]\b", s):
            return True
        if re.search(r"\b[dD][A-Za-z]\s*/\s*d[A-Za-z]\b", s):
            return True
        return False

    def _is_code_like(self, text):
        s = re.sub(r"\s+", " ", str(text or "")).strip()
        if not s:
            return False
        if len(s) > 160:
            return False
        if re.search(r"^\s*from\s+[A-Za-z_][A-Za-z0-9_\.]*\s+import\s+[A-Za-z_\*\.,\s]+$", s, flags=re.IGNORECASE):
            return True
        if re.search(r"^\s*import\s+[A-Za-z_][A-Za-z0-9_\.]*(\s+as\s+[A-Za-z_][A-Za-z0-9_]*)?$", s, flags=re.IGNORECASE):
            return True
        if re.search(r"^\s*(def|class)\s+[A-Za-z_][A-Za-z0-9_]*", s, flags=re.IGNORECASE):
            return True
        if re.search(r"\b(return|lambda)\b", s, flags=re.IGNORECASE) and re.search(r"[()\[\]{}:=_]|->", s):
            return True
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", s):
            return True
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\(", s):
            return True
        if re.search(r"[{}\[\]_]|==|!=|<=|>=|=>|:=|=\s*['\"]", s):
            return True
        return False

    def _word_count(self, text):
        return len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", str(text or "")))

    def _looks_explanatory_annotated_label(self, text):
        s = re.sub(r"\s+", " ", str(text or "")).strip()
        if not s:
            return False
        words = self._word_count(s)
        if words < 2 or words > 18:
            return False
        if len(s) > 120:
            return False
        if s.endswith(".") or s.endswith("..."):
            return False
        if "(" in s and ")" in s:
            return True
        return bool(re.search(r"\b(system|device|detection|interpretation|vision|input|output|activation)\b", s, flags=re.IGNORECASE))

    def _is_reference_like(self, text):
        s = re.sub(r"\s+", " ", str(text or "")).strip()
        if not s:
            return False
        return bool(re.search(r"(https?://\S+|www\.\S+|doi:\s*\S+|arxiv:\s*\S+)", s, flags=re.IGNORECASE))

    def _is_citation_like(self, text):
        s = re.sub(r"\s+", " ", str(text or "")).strip()
        if not s:
            return False
        return bool(
            re.search(r"[“\"].+[”\"]", s)
            or re.search(r"\b(et al\.|vol\.|no\.|pp\.|Google AI Blog|blog)\b", s, flags=re.IGNORECASE)
        )

    def classify_unit_type(
        self,
        text,
        role,
        source_kind="",
        page_family="body_text",
        page_family_group="body_text",
        document_type="mixed_unknown",
        layout_type="mixed_blocks",
        style_profile="mixed_irregular",
    ):
        txt = re.sub(r"\s+", " ", str(text or "")).strip()
        role = self._norm(role, "body")
        source_kind = self._norm(source_kind)
        page_family = self._norm(page_family, "body_text")
        page_family_group = self._norm(page_family_group, page_family or "body_text")
        document_type = self._norm(document_type, "mixed_unknown")
        layout_type = self._norm(layout_type, "mixed_blocks")
        style_profile = self._norm(style_profile, "mixed_irregular")
        if not txt:
            return "empty"
        citation_like = self._is_citation_like(txt)
        reference_like = self._is_reference_like(txt)
        if citation_like:
            return "citation"
        if reference_like:
            # A prose sentence may embed a link or DOI but should still be translated as body text.
            if self._word_count(txt) <= 8 or re.fullmatch(r"(https?://\S+|www\.\S+|doi:\s*\S+|arxiv:\s*\S+)", txt, flags=re.IGNORECASE):
                return "reference_link"
        if self._is_code_like(txt):
            return "code_visible"
        if role in {"equation_inline", "equation_block"}:
            return "formula" if self._is_equation_like(txt) else "formula_label"
        if role in {"diagram_label", "diagram_text_label", "axis_label", "legend_label"}:
            return "diagram_label"
        if (
            layout_type == "annotated_page"
            and role in {"title", "body", "section_heading"}
            and self._looks_explanatory_annotated_label(txt)
            and not self._is_reference_like(txt)
            and not self._is_code_like(txt)
        ):
            return "diagram_label"
        short_words = self._word_count(txt)
        short_title_like = (
            role in {"title", "section_heading", "figure_caption", "body"}
            and short_words >= 1
            and short_words <= 7
            and len(txt) <= 64
            and not re.search(r"[.;:!?]$", txt)
        )
        if short_words <= 4 and len(txt) <= 48:
            if page_family in {"chart_label_page"}:
                return "chart_label"
            if page_family_group in {"body_with_figure", "body_with_diagram", "mixed_page", "table_page"}:
                return "short_label"
        if short_title_like and (
            page_family in {"illustrated_label_page", "chart_label_page", "mixed_formula_annotation_page", "table_diagram_example", "mixed_dense_illustrated"}
            or layout_type in {"annotated_page", "table_dominant", "image_dominant"}
            or style_profile in {"editorial_visual", "tabular_structured", "mixed_irregular"}
        ):
            return "short_label"
        if short_title_like and source_kind.startswith("native") and role in {"title", "figure_caption", "section_heading"}:
            return "short_label"
        if (
            role == "body"
            and source_kind.startswith("native")
            and short_words >= 1
            and short_words <= 4
            and len(txt) <= 40
            and not re.search(r"[.;:!?]$", txt)
        ):
            return "short_label"
        return "narrative_body"

    def classify_unit_policy(
        self,
        text,
        role,
        source_kind="",
        page_role="body",
        page_family="body_text",
        page_family_group="body_text",
        document_type="mixed_unknown",
        layout_type="mixed_blocks",
        style_profile="mixed_irregular",
        fallback_policy="",
    ):
        txt = re.sub(r"\s+", " ", str(text or "")).strip()
        role = self._norm(role, "body")
        source_kind = self._norm(source_kind)
        page_role = self._norm(page_role, "body")
        page_family = self._norm(page_family, "body_text")
        page_family_group = self._norm(page_family_group, page_family or "body_text")
        document_type = self._norm(document_type, "mixed_unknown")
        layout_type = self._norm(layout_type, "mixed_blocks")
        style_profile = self._norm(style_profile, "mixed_irregular")
        fallback_policy = self._norm(fallback_policy)

        if not txt:
            return {
                "unit_type": "empty",
                "translatable": False,
                "translation_strategy": "ignore",
                "coverage_required": "optional",
                "render_policy": "skip",
            }
        unit_type = self.classify_unit_type(
            text=txt,
            role=role,
            source_kind=source_kind,
            page_family=page_family,
            page_family_group=page_family_group,
            document_type=document_type,
            layout_type=layout_type,
            style_profile=style_profile,
        )

        if page_role == "toc":
            if re.fullmatch(r"\d{1,4}|[ivxlcdm]+", txt, flags=re.IGNORECASE):
                return {
                    "unit_type": "toc_page_marker",
                    "translatable": False,
                    "translation_strategy": "exact_preserve",
                    "coverage_required": "strict",
                    "render_policy": "fixed_preserve",
                }
            return {
                "unit_type": "toc_label",
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if role == "diagram_label":
            return {
                "unit_type": unit_type,
                "translatable": False,
                "translation_strategy": "exact_preserve",
                "coverage_required": "strict",
                "render_policy": "background_only",
            }

        if self._is_code_like(txt):
            return {
                "unit_type": unit_type,
                "translatable": False,
                "translation_strategy": "exact_preserve",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        short_label = self._word_count(txt) <= 4 and len(txt) <= 48
        if (
            page_family in {"table_diagram_example", "mixed_dense_illustrated"}
            and role in {"title", "section_heading", "body"}
            and short_label
            and not self._is_equation_like(txt)
        ):
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if page_family in {"table_diagram_example", "mixed_dense_illustrated"} and role == "body":
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if page_family in {"illustrated_label_page", "chart_label_page"} and unit_type in {"short_label", "chart_label"}:
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if unit_type == "short_label" and role in {"title", "section_heading", "figure_caption"}:
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if page_family == "mixed_formula_annotation_page" and unit_type in {"formula_label", "short_label"}:
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if layout_type == "annotated_page" and unit_type in {"short_label", "chart_label", "formula_label", "diagram_label"}:
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if layout_type == "annotated_page" and role == "body":
            long_narrative = unit_type == "narrative_body" and self._word_count(txt) >= 12
            if long_narrative:
                return {
                    "unit_type": unit_type,
                    "translatable": True,
                    "translation_strategy": "layout_constrained",
                    "coverage_required": "strict",
                    "render_policy": "paragraph_flow",
                }
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if layout_type == "table_dominant" and role == "body":
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if layout_type in {"image_dominant", "mixed_blocks"} and role == "body" and unit_type in {"short_label", "chart_label", "diagram_label"}:
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if layout_type == "reference_page" or document_type in {"scientific_paper", "web_print"}:
            if unit_type in {"reference_link", "citation"}:
                return {
                    "unit_type": unit_type,
                    "translatable": unit_type == "citation",
                    "translation_strategy": "layout_constrained" if unit_type == "citation" else "exact_preserve",
                    "coverage_required": "strict",
                    "render_policy": "anchored_text",
                }

        if unit_type == "reference_link":
            return {
                "unit_type": unit_type,
                "translatable": False,
                "translation_strategy": "exact_preserve",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if unit_type == "citation":
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if role in {"header", "footer"} and re.fullmatch(r"\d{1,4}|[ivxlcdm]+", txt, flags=re.IGNORECASE):
            return {
                "unit_type": unit_type,
                "translatable": False,
                "translation_strategy": "exact_preserve",
                "coverage_required": "strict",
                "render_policy": "fixed_preserve",
            }

        if re.fullmatch(r"[•▪◦·\-\*]", txt) or re.fullmatch(r"\d{1,4}([.)]|)", txt):
            return {
                "unit_type": "list_or_marker",
                "translatable": False,
                "translation_strategy": "exact_preserve",
                "coverage_required": "strict",
                "render_policy": "fixed_preserve",
            }

        if role == "equation_inline":
            if self._is_equation_like(txt):
                return {
                    "unit_type": unit_type,
                    "translatable": False,
                    "translation_strategy": "exact_preserve",
                    "coverage_required": "strict",
                    "render_policy": "fixed_preserve",
                }
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if role in {"title", "section_heading", "figure_caption", "diagram_text_label"}:
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if page_family in {"body_text_two_column_equations"} and role == "body":
            if unit_type == "narrative_body" and self._word_count(txt) >= 12:
                return {
                    "unit_type": unit_type,
                    "translatable": True,
                    "translation_strategy": "layout_constrained",
                    "coverage_required": "strict",
                    "render_policy": "paragraph_flow",
                }
            if unit_type in {"reference_link", "citation", "formula", "formula_label"}:
                return {
                    "unit_type": unit_type,
                    "translatable": unit_type not in {"reference_link", "formula"},
                    "translation_strategy": "layout_constrained" if unit_type in {"citation", "formula_label"} else "exact_preserve",
                    "coverage_required": "strict",
                    "render_policy": "anchored_text" if unit_type in {"citation", "formula_label"} else "fixed_preserve",
                }
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if layout_type == "double_column" and role == "body":
            if document_type in {"scientific_paper", "book_page", "manual_guide"}:
                if unit_type == "narrative_body" and self._word_count(txt) >= 12:
                    return {
                        "unit_type": unit_type,
                        "translatable": True,
                        "translation_strategy": "layout_constrained",
                        "coverage_required": "strict",
                        "render_policy": "paragraph_flow",
                    }
                if unit_type in {"reference_link", "citation", "formula", "formula_label"}:
                    return {
                        "unit_type": unit_type,
                        "translatable": unit_type not in {"reference_link", "formula"},
                        "translation_strategy": "layout_constrained" if unit_type in {"citation", "formula_label"} else "exact_preserve",
                        "coverage_required": "strict",
                        "render_policy": "anchored_text" if unit_type in {"citation", "formula_label"} else "fixed_preserve",
                    }
                return {
                    "unit_type": unit_type,
                    "translatable": True,
                    "translation_strategy": "layout_constrained",
                    "coverage_required": "strict",
                    "render_policy": "anchored_text",
                }

        if page_family_group == "body_with_figure" and role == "body":
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if page_family_group == "body_with_diagram" and role == "body":
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if page_family_group == "table_page" and role == "body":
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if style_profile == "tabular_structured" and role == "body":
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if fallback_policy == "safe_mixed" and role == "body":
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        if source_kind in {"native_span", "native_phrase"} and len(txt.split()) <= 10:
            return {
                "unit_type": unit_type,
                "translatable": True,
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "render_policy": "anchored_text",
            }

        return {
            "unit_type": unit_type,
            "translatable": True,
            "translation_strategy": "semantic_reflow",
            "coverage_required": "strict",
            "render_policy": "paragraph_flow" if role == "body" else "anchored_text",
        }
