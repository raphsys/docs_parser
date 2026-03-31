import re

from page_case_classifier import PageCaseClassifier


class PageCaseClassifierV2:
    VERSION = "page_case.v2"

    def __init__(self):
        self.legacy = PageCaseClassifier()

    def classify(self, page_data, lines, page_role="body"):
        legacy = self.legacy.classify(page_data, lines, page_role=page_role)
        features = legacy.get("features") or {}
        role = str(page_role or features.get("page_role") or "body").strip().lower()

        layout_tendencies = self._layout_tendencies(features, role)
        reading_modes = self._reading_modes(features, legacy, role)
        archetype_signals = self._archetype_signals(features, legacy, role)
        translation_sensitivity = self._translation_sensitivity(features, legacy, role)
        risk_flags = self._risk_flags(features, legacy, reading_modes, translation_sensitivity, role)

        return {
            "version": self.VERSION,
            "page_role": role,
            "layout_tendencies": layout_tendencies,
            "reading_modes": reading_modes,
            "page_archetype_signals": archetype_signals,
            "translation_sensitivity_signals": translation_sensitivity,
            "risk_flags": risk_flags,
            "legacy_bridge": {
                "page_family": legacy.get("page_family"),
                "page_family_group": legacy.get("page_family_group"),
                "layout_type": legacy.get("layout_type"),
                "document_type": legacy.get("document_type"),
                "style_profile": legacy.get("style_profile"),
                "fallback_policy": legacy.get("fallback_policy"),
                "family_confidence": float(legacy.get("family_confidence") or 0.0),
                "is_known_family": bool(legacy.get("is_known_family")),
            },
            "feature_snapshot": {
                "column_count": int(features.get("column_count", 1) or 1),
                "text_coverage_ratio": float(features.get("text_coverage_ratio", 0.0) or 0.0),
                "table_coverage_ratio": float(features.get("table_coverage_ratio", 0.0) or 0.0),
                "figure_coverage_ratio": float(features.get("figure_coverage_ratio", 0.0) or 0.0),
                "toc_pattern_score": float(features.get("toc_pattern_score", 0.0) or 0.0),
                "scientific_pattern_score": float(features.get("scientific_pattern_score", 0.0) or 0.0),
                "form_pattern_score": float(features.get("form_pattern_score", 0.0) or 0.0),
                "whitespace_ratio": float(features.get("whitespace_ratio", 0.0) or 0.0),
                "short_line_ratio": float(features.get("short_line_ratio", 0.0) or 0.0),
            },
            "legacy_snapshot": legacy,
        }

    def _layout_tendencies(self, features, page_role):
        col_count = int(features.get("column_count", 1) or 1)
        text_ratio = float(features.get("text_coverage_ratio", 0.0) or 0.0)
        table_ratio = float(features.get("table_coverage_ratio", 0.0) or 0.0)
        figure_ratio = float(features.get("figure_coverage_ratio", 0.0) or 0.0)
        whitespace = float(features.get("whitespace_ratio", 0.0) or 0.0)
        short_line_ratio = float(features.get("short_line_ratio", 0.0) or 0.0)
        visual_non_text = int(features.get("visual_non_text", 0) or 0)
        ai_region_coverage = float(features.get("ai_region_coverage_ratio", 0.0) or 0.0)

        return {
            "columnarity": "multi" if col_count >= 3 else ("double" if col_count == 2 else "single"),
            "text_density": self._density_label(text_ratio, whitespace),
            "visual_density": self._density_label(figure_ratio + min(0.25, visual_non_text * 0.03), max(0.0, 1.0 - figure_ratio)),
            "table_density": self._density_label(table_ratio, max(0.0, 1.0 - table_ratio)),
            "lineation": "fragmented" if short_line_ratio >= 0.45 else "continuous",
            "toc_likeliness": round(float(features.get("toc_pattern_score", 0.0) or 0.0), 4),
            "page_role_bias": "toc" if page_role == "toc" else "body",
            "ai_support_density": round(ai_region_coverage, 4),
        }

    def _reading_modes(self, features, legacy, page_role):
        col_count = int(features.get("column_count", 1) or 1)
        toc_score = float(features.get("toc_pattern_score", 0.0) or 0.0)
        table_ratio = float(features.get("table_coverage_ratio", 0.0) or 0.0)
        figure_ratio = float(features.get("figure_coverage_ratio", 0.0) or 0.0)
        visual_non_text = int(features.get("visual_non_text", 0) or 0)
        short_line_ratio = float(features.get("short_line_ratio", 0.0) or 0.0)
        scientific = float(features.get("scientific_pattern_score", 0.0) or 0.0)
        table_blocks = int(features.get("num_table_blocks", 0) or 0)

        two_col = 0.15 + (0.55 if col_count == 2 else 0.0) + (0.15 if scientific >= 0.35 else 0.0)
        linear = 0.25 + (0.45 if col_count == 1 else 0.0) + (0.1 if short_line_ratio < 0.3 else 0.0)
        anchored = 0.05 + min(0.55, figure_ratio + visual_non_text * 0.04)
        toc = 0.0
        if page_role == "toc":
            toc = 0.98
        else:
            toc = min(0.92, toc_score + (0.08 if short_line_ratio >= 0.4 else 0.0))
        table = min(0.95, table_ratio + table_blocks * 0.12)
        glossary = 0.0
        if self._looks_like_glossary_page(features, legacy):
            glossary = 0.88

        return {
            "linear_flow": round(min(1.0, linear), 4),
            "columnar_flow": round(min(1.0, two_col), 4),
            "anchored_overlay_flow": round(min(1.0, anchored), 4),
            "toc_row_flow": round(min(1.0, toc), 4),
            "tabular_grid_flow": round(min(1.0, table), 4),
            "glossary_pair_flow": round(min(1.0, glossary), 4),
        }

    def _archetype_signals(self, features, legacy, page_role):
        page_family = str(legacy.get("page_family") or "").strip().lower()
        signals = {
            "toc": 1.0 if page_role == "toc" else float(features.get("toc_pattern_score", 0.0) or 0.0),
            "editorial_body": 0.0,
            "annotated_visual": 0.0,
            "table_containing": 0.0,
            "glossary_like": 0.0,
            "chapter_opening": 0.0,
        }
        body_blocks = int(features.get("body_blocks", 0) or 0)
        section_heads = int(features.get("section_heading_blocks", 0) or 0)
        figure_ratio = float(features.get("figure_coverage_ratio", 0.0) or 0.0)
        table_ratio = float(features.get("table_coverage_ratio", 0.0) or 0.0)
        whitespace = float(features.get("whitespace_ratio", 0.0) or 0.0)
        title_blocks = int(features.get("num_title_blocks", 0) or 0)

        if body_blocks >= 2:
            signals["editorial_body"] += 0.55
        if int(features.get("column_count", 1) or 1) == 2:
            signals["editorial_body"] += 0.18
        if page_family.startswith("body_text"):
            signals["editorial_body"] += 0.22

        if page_family in {"body_with_figure", "body_with_diagram", "illustrated_label_page", "chart_label_page", "mixed_dense_illustrated"}:
            signals["annotated_visual"] += 0.72
        signals["annotated_visual"] += min(0.22, figure_ratio + int(features.get("visual_non_text", 0) or 0) * 0.03)

        if page_family in {"table_page", "table_diagram_example"}:
            signals["table_containing"] += 0.8
        signals["table_containing"] += min(0.2, table_ratio + int(features.get("num_table_blocks", 0) or 0) * 0.08)

        if self._looks_like_glossary_page(features, legacy):
            signals["glossary_like"] = 0.88

        if self._looks_like_chapter_opening(features, legacy, page_role):
            signals["chapter_opening"] += 0.68
        if "chapter" in str(page_family):
            signals["chapter_opening"] += 0.12

        return {k: round(min(1.0, max(0.0, v)), 4) for k, v in signals.items()}

    def _translation_sensitivity(self, features, legacy, page_role):
        page_family = str(legacy.get("page_family") or "").strip().lower()
        layout_type = str(legacy.get("layout_type") or "").strip().lower()
        short_line_ratio = float(features.get("short_line_ratio", 0.0) or 0.0)
        text_ratio = float(features.get("text_coverage_ratio", 0.0) or 0.0)
        figure_ratio = float(features.get("figure_coverage_ratio", 0.0) or 0.0)
        table_ratio = float(features.get("table_coverage_ratio", 0.0) or 0.0)

        return {
            "line_break_sensitivity": round(min(1.0, 0.2 + short_line_ratio + (0.2 if page_role == "toc" else 0.0)), 4),
            "overflow_risk": round(min(1.0, 0.15 + text_ratio * 0.7 + (0.15 if layout_type == "double_column" else 0.0)), 4),
            "anchoring_sensitivity": round(min(1.0, 0.1 + figure_ratio + (0.15 if page_family in {"body_with_figure", "body_with_diagram", "table_diagram_example"} else 0.0)), 4),
            "grid_alignment_sensitivity": round(min(1.0, 0.1 + table_ratio + (0.2 if page_family in {"table_page", "table_diagram_example"} else 0.0)), 4),
            "lexical_preservation_bias": round(min(1.0, 0.15 + int(features.get("code_like_blocks", 0) or 0) * 0.12 + int(features.get("reference_like_blocks", 0) or 0) * 0.08), 4),
        }

    def _risk_flags(self, features, legacy, reading_modes, translation_sensitivity, page_role):
        flags = []
        if page_role == "toc" or reading_modes.get("toc_row_flow", 0.0) >= 0.8:
            flags.append(self._flag("toc_row_fragmentation", max(
                translation_sensitivity.get("line_break_sensitivity", 0.0),
                reading_modes.get("toc_row_flow", 0.0),
            )))
        if reading_modes.get("tabular_grid_flow", 0.0) >= 0.55:
            flags.append(self._flag("grid_alignment_loss", max(
                reading_modes.get("tabular_grid_flow", 0.0),
                translation_sensitivity.get("grid_alignment_sensitivity", 0.0),
            )))
        if reading_modes.get("anchored_overlay_flow", 0.0) >= 0.45:
            flags.append(self._flag("anchor_attachment_drift", max(
                reading_modes.get("anchored_overlay_flow", 0.0),
                translation_sensitivity.get("anchoring_sensitivity", 0.0),
            )))
        if translation_sensitivity.get("overflow_risk", 0.0) >= 0.75:
            flags.append(self._flag("translation_overflow", translation_sensitivity.get("overflow_risk", 0.0)))
        if self._looks_like_glossary_page(features, legacy):
            flags.append(self._flag("key_value_pair_breakage", 0.88))
        return flags

    def _flag(self, code, severity):
        return {"code": code, "severity": round(float(severity or 0.0), 4)}

    def _looks_like_glossary_page(self, features, legacy):
        page_role = str(features.get("page_role") or "").strip().lower()
        page_family = str(legacy.get("page_family") or "").strip().lower()
        toc_score = float(features.get("toc_pattern_score", 0.0) or 0.0)
        table_ratio = float(features.get("table_coverage_ratio", 0.0) or 0.0)
        if page_role == "toc" or toc_score >= 0.75:
            return False
        if page_family in {"table_page", "table_diagram_example", "body_text_two_column_equations"}:
            return False
        if table_ratio >= 0.18:
            return False
        if page_role in {"glossary", "abbreviations"}:
            return True
        block_count = int(features.get("block_count", 0) or 0)
        short_blocks = int(features.get("short_text_blocks", 0) or 0)
        col_count = int(features.get("column_count", 1) or 1)
        return block_count >= 8 and short_blocks >= max(4, block_count // 3) and col_count <= 2

    def _looks_like_chapter_opening(self, features, legacy, page_role):
        if page_role == "toc":
            return False
        page_family = str(legacy.get("page_family") or "").strip().lower()
        layout_type = str(legacy.get("layout_type") or "").strip().lower()
        title_blocks = int(features.get("num_title_blocks", 0) or 0)
        section_heads = int(features.get("section_heading_blocks", 0) or 0)
        body_blocks = int(features.get("body_blocks", 0) or 0)
        whitespace = float(features.get("whitespace_ratio", 0.0) or 0.0)
        short_line_ratio = float(features.get("short_line_ratio", 0.0) or 0.0)
        figure_ratio = float(features.get("figure_coverage_ratio", 0.0) or 0.0)
        table_ratio = float(features.get("table_coverage_ratio", 0.0) or 0.0)
        tableish_lines = int(features.get("tableish_lines", 0) or 0)
        code_like_blocks = int(features.get("code_like_blocks", 0) or 0)
        col_count = int(features.get("column_count", 1) or 1)

        if "chapter" in page_family:
            return True
        if page_family in {"table_page", "table_diagram_example", "body_text_two_column_equations"}:
            return False
        if layout_type == "table_dominant" or table_ratio >= 0.16 or tableish_lines >= 3 or code_like_blocks >= 2:
            return False
        if title_blocks >= 2 and body_blocks <= 3 and whitespace >= 0.22 and figure_ratio <= 0.35:
            return True
        if (
            title_blocks >= 1
            and section_heads <= 1
            and body_blocks <= 2
            and whitespace >= 0.35
            and short_line_ratio <= 0.45
        ):
            return True
        if layout_type == "double_column" and col_count == 2 and title_blocks >= 2 and body_blocks <= 2:
            return True
        return False

    def _density_label(self, signal, whitespace):
        score = float(signal or 0.0)
        blank = float(whitespace or 0.0)
        if score >= 0.5:
            return "high"
        if score >= 0.24:
            return "medium"
        if blank >= 0.55:
            return "sparse"
        return "low"
