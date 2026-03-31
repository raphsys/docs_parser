import fitz
import os
import re
import uuid
import unicodedata
import json
import xml.etree.ElementTree as ET
from collections import Counter
from statistics import median
from PIL import Image, ImageDraw, ImageStat
from block_typology import classify_block_typology
from background_inpainter import get_background_inpainter
from font_resolver import FontResolver
from text_composer import TextComposer, ComposeOptions

class DocumentReconstructor:
    def __init__(self):
        self.pixel_to_point = 72.0 / 150.0
        self.font_resolver = FontResolver()
        self._page_font_aliases = {}
        self._font_objects = {}
        self.strict_fidelity = os.getenv("LAYOUT_STRICT_FIDELITY", "0") == "1"
        self.sequential_layout_mode = os.getenv("LAYOUT_SEQUENTIAL_FLOW", "0") == "1"
        self.layout_correction = os.getenv("LAYOUT_CORRECTION", "0") == "1"
        self.max_shift_steps = int(os.getenv("LAYOUT_MAX_SHIFT_STEPS", "24"))
        self.overlap_threshold = float(os.getenv("LAYOUT_OVERLAP_THRESHOLD", "0.25"))
        self.flow_layout_mode = os.getenv("LAYOUT_FLOW_MODE", "0") == "1"
        self.flow_zone_pad = float(os.getenv("LAYOUT_FLOW_ZONE_PAD", "8.0"))
        self.flow_min_font_scale = float(os.getenv("LAYOUT_FLOW_MIN_FONT_SCALE", "0.72"))
        self.flow_min_font_pt = float(os.getenv("LAYOUT_FLOW_MIN_FONT_PT", "5.5"))
        self.layout_debug_overlay = os.getenv("LAYOUT_DEBUG_OVERLAY", "1") == "1"
        self.layout_debug_dpi = int(os.getenv("LAYOUT_DEBUG_DPI", "150"))
        self.native_first_mode = os.getenv("LAYOUT_NATIVE_FIRST", "1") == "1"
        self.translation_reflow_mode = os.getenv("LAYOUT_TRANSLATION_REFLOW", "1") == "1"
        self.fixed_font_size = os.getenv("LAYOUT_FIXED_FONT_SIZE", "1") == "1"
        self.fixed_spacing = os.getenv("LAYOUT_FIXED_SPACING", "1") == "1"
        self.overflow_policy = os.getenv(
            "LAYOUT_OVERFLOW_POLICY",
            "single_page_if_possible_else_paginate",
        ).strip().lower()
        self.page_overflow_to_next_page = os.getenv("LAYOUT_PAGE_OVERFLOW_TO_NEXT_PAGE", "0") == "1"
        self.style_audit_enabled = os.getenv("LAYOUT_STYLE_AUDIT", "1") == "1"
        self._restored_background_rects = {}
        self._prepared_visual_groups = {}
        self._typography_group_cache = {}
        self._local_background_profile_cache = {}
        self.dynamic_equation_overlays = os.getenv("LAYOUT_DYNAMIC_EQUATION_OVERLAYS", "1") == "1"
        self.dynamic_symbol_overlays = os.getenv("LAYOUT_DYNAMIC_SYMBOL_OVERLAYS", "1") == "1"
        self.dynamic_risk_overlays = os.getenv("LAYOUT_DYNAMIC_RISK_OVERLAYS", "1") == "1"
        self.pro_strict_mode = os.getenv("LAYOUT_PRO_STRICT", "1") == "1"
        self.dynamic_overlay_pad_px = int(os.getenv("LAYOUT_DYNAMIC_OVERLAY_PAD_PX", "1"))
        self.equation_diff_threshold = float(os.getenv("LAYOUT_EQUATION_DIFF_THRESHOLD", "22.0"))
        self.native_block_diff_threshold = float(os.getenv("LAYOUT_NATIVE_BLOCK_DIFF_THRESHOLD", "26.0"))
        self.background_inpainter = get_background_inpainter()
        if self.pro_strict_mode:
            self.overlap_threshold = min(self.overlap_threshold, 0.08)
        self.text_composer = TextComposer()
        if self.strict_fidelity:
            self.sequential_layout_mode = False
            self.layout_correction = False
            self.flow_layout_mode = False
        trusted_dirs_env = os.getenv("LAYOUT_XML_TRUSTED_DIRS", "").strip()
        if trusted_dirs_env:
            trusted_dirs = [p.strip() for p in trusted_dirs_env.split(os.pathsep) if p.strip()]
        else:
            trusted_dirs = [os.path.join(os.getcwd(), "ocr_results")]
        self._layout_xml_trusted_dirs = [os.path.realpath(p) for p in trusted_dirs]
        self._layout_xml_max_bytes = max(1024, int(os.getenv("LAYOUT_XML_MAX_BYTES", "1048576")))

    def _should_paginate_on_overflow(self):
        if self.overflow_policy == "strict_single_page":
            return False
        if self.overflow_policy in {"single_page_if_possible_else_paginate", "paginate"}:
            return True
        return bool(self.page_overflow_to_next_page)

    def _default_style(self):
        return {"font": "helv", "size": 12, "color": "#000000", "flags": {}}

    def _merge_styles(self, preferred, fallback):
        pref = preferred if isinstance(preferred, dict) else {}
        fb = fallback if isinstance(fallback, dict) else {}
        out = dict(self._default_style())
        out.update(fb)
        out.update(pref)
        flags = {}
        if isinstance(fb.get("flags"), dict):
            flags.update(fb.get("flags", {}))
        if isinstance(pref.get("flags"), dict):
            flags.update(pref.get("flags", {}))
        out["flags"] = flags
        return out

    def _item_typography_key(self, item):
        if not isinstance(item, dict):
            return ("content", "")
        tclass = str(item.get("descriptor_typographic_class") or "").strip().lower() or "content"
        gids = item.get("descriptor_group_ids") or {}
        group_id = (
            gids.get("paragraph_chain_group_id")
            or gids.get("same_band_group_id")
            or gids.get("same_row_group_id")
            or gids.get("toc_entry_group_id")
            or gids.get("section_sibling_group_id")
            or gids.get("annotation_group_id")
            or gids.get("legend_group_id")
            or gids.get("axis_group_id")
            or gids.get("tick_group_id")
            or gids.get("series_group_id")
            or gids.get("cell_id")
            or gids.get("table_row_group_id")
            or item.get("descriptor_paragraph_id")
            or item.get("descriptor_section_id")
            or item.get("descriptor_region_id")
            or item.get("source_block_id")
            or item.get("text")
        )
        return (tclass, str(group_id or ""))

    def _descriptor_v3_primary_family(self, item):
        if not isinstance(item, dict):
            return ""
        contract = item.get("descriptor_v3_contract") or {}
        family = str(contract.get("primary_structure_family") or "").strip().lower()
        if family:
            return family
        descriptor_v3 = item.get("descriptor_v3") or {}
        return str(descriptor_v3.get("primary_structure_family") or "").strip().lower()

    def _descriptor_v3_structure_priority(self, item):
        if not isinstance(item, dict):
            return ""
        render_unit = item.get("descriptor_v3_render_unit") or {}
        return str(render_unit.get("structure_priority") or "").strip().lower()

    def _item_preserve_extracted_typography(self, item):
        if not isinstance(item, dict):
            return False
        if item.get("preserve_line_style_variation"):
            return True
        family = self._descriptor_v3_primary_family(item)
        priority = self._descriptor_v3_structure_priority(item)
        role = str(item.get("role") or "").strip().lower()
        tclass = str(item.get("descriptor_typographic_class") or "").strip().lower()
        structural_role = str(item.get("descriptor_structural_role") or "").strip().lower()
        source = str(item.get("source") or "").strip().lower()
        if family in {"toc", "glossary_pairs", "chapter_opening"} and priority == "primary":
            return True
        if structural_role in {"abbreviation_key", "abbreviation_value"}:
            return True
        if role in {"title", "section_heading", "header"} and source == "native":
            return True
        if tclass in {"abbreviation_key", "abbreviation_value", "running_header", "running_footer", "section_title"}:
            return True
        return False

    def _item_native_style_fidelity_mode(self, item):
        if not isinstance(item, dict):
            return False
        source = str(item.get("source") or "").strip().lower()
        if source != "native":
            return False
        if self._item_preserve_extracted_typography(item):
            return True
        family = self._descriptor_v3_primary_family(item)
        role = str(item.get("role") or "").strip().lower()
        if family in {"section_flow", "dense_paragraph_flow", "chapter_opening"} and role in {"body", "title", "section_heading", "header", "footer"}:
            return True
        return False

    def _min_fontsize_for_item(self, item, base_fs, strict=False):
        fs = float(base_fs or 0.0)
        if fs <= 0.0:
            return 5.5
        if self._item_native_style_fidelity_mode(item):
            keep_ratio = 0.96 if strict else 0.94
            return max(5.8, fs * keep_ratio)
        if self._item_preserve_extracted_typography(item):
            keep_ratio = 0.95 if strict else 0.92
            return max(5.8, fs * keep_ratio)
        if strict:
            return max(5.5, min(fs, 7.0))
        return max(4.8, min(fs, 6.6))

    def _overflow_limit_for_item(self, item, default_limit):
        limit = float(default_limit or 1.0)
        if self._item_native_style_fidelity_mode(item):
            return max(limit, 1.08)
        if self._item_preserve_extracted_typography(item):
            return max(limit, 1.06)
        return limit

    def _needs_conservative_right_padding(self, item):
        if not isinstance(item, dict):
            return False
        if str(item.get("role") or "").strip().lower() != "body":
            return False
        if not item.get("translated_block"):
            return False
        page_data = item.get("page_data") or {}
        layout_type = str(page_data.get("layout_type") or "").strip().lower()
        document_type = str(page_data.get("document_type") or "").strip().lower()
        if layout_type == "table_dominant" and document_type == "form":
            return True
        if item.get("paragraph_flow_mode") and layout_type == "double_column" and document_type in {"scientific_paper", "book_page", "manual_guide"}:
            return True
        return False

    def _line_right_padding_for_item(self, item, fontsize, strict=False):
        fs = max(1.0, float(fontsize or 0.0))
        if self._needs_conservative_right_padding(item):
            base = max(7.0, min(14.0, fs * 0.8))
            return max(base, 9.0 if strict else 8.0)
        if self._allow_exact_line_left_relief(item):
            base = max(8.0, min(16.0, fs * 0.95))
            return max(base, 12.0 if strict else 10.0)
        if self._item_native_style_fidelity_mode(item):
            base = max(5.0, min(12.0, fs * 0.72))
            return max(base, 7.0 if strict else 6.0)
        if self._item_preserve_extracted_typography(item):
            base = max(4.0, min(10.0, fs * 0.58))
            return max(base, 5.0 if strict else 4.0)
        if strict:
            return max(2.5, min(6.0, fs * 0.42))
        return max(2.0, min(4.5, fs * 0.32))

    def _relation_flow_hint(self, item):
        if not isinstance(item, dict):
            return ""
        gids = item.get("descriptor_group_ids") or {}
        if str(item.get("role") or "").strip().lower() != "body":
            return ""
        if str(item.get("descriptor_typographic_class") or "").strip().lower() != "editorial_body":
            return ""
        if gids.get("paragraph_chain_group_id"):
            return "paragraph_chain"
        if gids.get("toc_entry_group_id"):
            return ""
        if gids.get("same_row_group_id"):
            return ""
        if gids.get("same_band_group_id"):
            region_type = str(item.get("descriptor_region_type") or "").strip().lower()
            band_role = str(item.get("descriptor_band_role") or "").strip().lower()
            if region_type in {"text", "text_band", "column"} and band_role == "content_band":
                return "content_band"
        return ""

    def _allow_exact_line_left_relief(self, item):
        if not isinstance(item, dict):
            return False
        if not item.get("keep_exact_line") or not item.get("translated_block"):
            return False
        if str(item.get("role") or "").strip().lower() != "body":
            return False
        if self._is_abbreviation_entry_role(item.get("descriptor_structural_role")):
            return False
        return str(item.get("descriptor_typographic_class") or "").strip().lower() == "editorial_body"

    def _allow_strict_line_items_for_anchored_text_body(self, block_role, descriptor_typographic_class, descriptor_structural_role):
        role = str(block_role or "").strip().lower()
        if role != "body":
            return True
        tclass = str(descriptor_typographic_class or "").strip().lower()
        srole = str(descriptor_structural_role or "").strip().lower()
        if tclass == "editorial_body" or srole in {"body_paragraph", "opening_paragraph"}:
            return False
        return True

    def _translated_body_line_fit_metrics(self, line_entries, source, overflow_ratio=1.04):
        entries = line_entries if isinstance(line_entries, list) else []
        source = str(source or "").strip().lower() or "ocr"
        max_ratio = 0.0
        overflow_lines = 0
        line_count = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            text = self._clean_text_for_render(entry.get("text", "")).strip()
            bbox = entry.get("bbox")
            if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                rect = fitz.Rect([float(v) * self.pixel_to_point for v in bbox])
            except Exception:
                continue
            if rect.get_area() <= 0:
                continue
            line_count += 1
            style = self._merge_styles(entry.get("style", {}), {})
            fs = self._get_original_fontsize(style, max(1.0, rect.height), source)
            _, fontfile, builtin, fontname = self._resolve_style_font(None, style, text=text)
            fontname = builtin or fontname
            measured = self._measure_text_width(text, fs, fontname, fontfile)
            ratio = measured / max(8.0, rect.width)
            max_ratio = max(max_ratio, float(ratio))
            if measured > max(8.0, rect.width) * float(overflow_ratio or 1.0):
                overflow_lines += 1
        return {
            "line_count": line_count,
            "overflow_lines": overflow_lines,
            "max_ratio": max_ratio,
            "fits": overflow_lines == 0,
        }

    def _translated_body_lines_fit_source_slots(self, line_entries, source, overflow_ratio=1.04):
        return bool(
            self._translated_body_line_fit_metrics(
                line_entries,
                source,
                overflow_ratio=overflow_ratio,
            ).get("fits")
        )

    def _should_keep_strict_line_items_for_anchored_body(self, line_entries, source):
        metrics = self._translated_body_line_fit_metrics(line_entries, source, overflow_ratio=1.04)
        if metrics.get("fits"):
            return True
        overflow_lines = int(metrics.get("overflow_lines") or 0)
        line_count = int(metrics.get("line_count") or 0)
        max_ratio = float(metrics.get("max_ratio") or 0.0)
        if overflow_lines <= 1 and max_ratio <= 1.08:
            return True
        if line_count <= 2 and overflow_lines <= 1 and max_ratio <= 1.12:
            return True
        return False

    def _should_keep_multiline_locked_editorial_block(
        self,
        page_data,
        block,
        descriptor_layout_behavior,
        descriptor_structural_role,
        descriptor_typographic_class,
        line_entries,
        source,
        translated_block,
    ):
        if not translated_block:
            return False
        if not isinstance(page_data, dict) or not isinstance(block, dict):
            return False
        if str(block.get("role") or "").strip().lower() != "body":
            return False
        if str(block.get("render_policy") or "").strip().lower() != "anchored_text":
            return False
        if str(page_data.get("layout_type") or "").strip().lower() != "table_dominant":
            return False
        if str(page_data.get("document_type") or "").strip().lower() != "form":
            return False
        if str(descriptor_layout_behavior or "").strip().lower() not in {"locked_in_cell", "locked_in_table"}:
            return False
        if str(descriptor_structural_role or "").strip().lower() != "table_value_cell":
            return False
        if str(descriptor_typographic_class or "").strip().lower() != "editorial_body":
            return False
        metrics = self._translated_body_line_fit_metrics(line_entries, source, overflow_ratio=1.04)
        line_count = int(metrics.get("line_count") or 0)
        overflow_lines = int(metrics.get("overflow_lines") or 0)
        max_ratio = float(metrics.get("max_ratio") or 0.0)
        if line_count < 2:
            return False
        if overflow_lines <= 0:
            return False
        return max_ratio >= 1.08

    def _should_allow_relation_flow_override(self, item, page_data, fallback_policy, anchored_figure_page, table_locked_block):
        if not isinstance(item, dict):
            return False
        if str(fallback_policy or "").strip().lower() != "safe_mixed":
            return False
        if anchored_figure_page or table_locked_block:
            return False
        if not item.get("translated_block"):
            return False
        relation_hint = self._relation_flow_hint(item)
        if relation_hint not in {"paragraph_chain", "content_band"}:
            return False
        page_data = page_data if isinstance(page_data, dict) else {}
        if str(page_data.get("layout_type") or "").strip().lower() not in {"single_column", "double_column", "text_heavy"}:
            return False
        if str(page_data.get("document_type") or "").strip().lower() not in {"scientific_paper", "book_page", "manual_guide", "mixed_unknown"}:
            return False
        return True

    def _is_abbreviation_entry_role(self, structural_role):
        role = str(structural_role or "").strip().lower()
        return role in {"abbreviation_key", "abbreviation_value"}

    def _is_abbreviation_key_role(self, structural_role):
        return str(structural_role or "").strip().lower() == "abbreviation_key"

    def _is_abbreviation_value_role(self, structural_role):
        return str(structural_role or "").strip().lower() == "abbreviation_value"

    def _relation_group_bbox(self, item, relation_type):
        if not isinstance(item, dict):
            return None
        relation_type = str(relation_type or "").strip().lower()
        gids = item.get("descriptor_group_ids") or {}
        type_to_group_key = {
            "same_band": "same_band_group_id",
            "same_row": "same_row_group_id",
            "continues_paragraph": "paragraph_chain_group_id",
            "inside_toc_entry": "toc_entry_group_id",
            "section_sibling": "section_sibling_group_id",
        }
        group_key = type_to_group_key.get(relation_type)
        group_id = str(gids.get(group_key) or "").strip() if group_key else ""
        if not group_id:
            return None
        reconstruction_plan = item.get("descriptor_reconstruction_plan") or {}
        group_entries = reconstruction_plan.get("relation_groups") or {}
        for group in group_entries.get(relation_type) or []:
            if not isinstance(group, dict):
                continue
            if str(group.get("id") or "").strip() != group_id:
                continue
            bbox = group.get("bbox")
            if isinstance(bbox, fitz.Rect):
                return fitz.Rect(bbox)
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    return fitz.Rect(
                        float(bbox[0]) * self.pixel_to_point,
                        float(bbox[1]) * self.pixel_to_point,
                        float(bbox[2]) * self.pixel_to_point,
                        float(bbox[3]) * self.pixel_to_point,
                    )
                except Exception:
                    return None
        return None

    def _normalized_style_for_item(self, item, style_override=None):
        style = dict(style_override if isinstance(style_override, dict) else (item.get("style") or {}))
        if self._item_preserve_extracted_typography(item):
            return style
        cache_key = self._item_typography_key(item)
        cached = self._typography_group_cache.get(cache_key) or {}
        font_name = str(style.get("font") or "").strip()
        if cached.get("font") and not font_name:
            style["font"] = cached["font"]
        elif font_name and not cached.get("font"):
            cached["font"] = font_name
            self._typography_group_cache[cache_key] = cached
        flags = style.get("flags") or {}
        if cached.get("flags") and not flags:
            style["flags"] = dict(cached["flags"])
        elif flags and not cached.get("flags"):
            cached["flags"] = dict(flags)
            self._typography_group_cache[cache_key] = cached
        return style

    def _normalized_fontsize_for_item(self, item, style, bbox_h_pt, source):
        raw_fs = self._preferred_fontsize_for_item(item, style, bbox_h_pt, source)
        if self._item_preserve_extracted_typography(item):
            return float(raw_fs)
        cache_key = self._item_typography_key(item)
        cached = self._typography_group_cache.get(cache_key) or {}
        if "font_size" not in cached:
            cached["font_size"] = float(raw_fs)
            self._typography_group_cache[cache_key] = cached
            return float(raw_fs)
        current = float(cached["font_size"])
        tclass = str(item.get("descriptor_typographic_class") or "").strip().lower()
        if tclass in {"editorial_body", "diagram_label", "chart_axis_label", "chart_tick_label", "chart_legend_label", "figure_caption", "table_header_cell", "table_stub_cell", "table_value_cell", "abbreviation_key", "abbreviation_value"}:
            return current
        return current

    def _layout_ai_font_hint_pt(self, item, bbox_h_pt):
        if not isinstance(item, dict):
            return 0.0
        role = str(item.get("role") or "").strip().lower()
        tclass = str(item.get("descriptor_typographic_class") or "").strip().lower()
        line_hint = float(item.get("layout_ai_text_line_height_pt") or 0.0)
        block_hint = float(item.get("layout_ai_block_height_pt") or 0.0)
        source_line_count = int(item.get("source_line_count") or len(item.get("source_lines") or []) or 1)
        candidates = []
        if line_hint > 0.0:
            candidates.append(line_hint * 0.90)
        if block_hint > 0.0 and source_line_count > 0:
            candidates.append((block_hint / max(1, source_line_count)) * 0.88)
        if not candidates:
            return 0.0
        hint = max(candidates)
        sensitive = {
            "editorial_body",
            "diagram_label",
            "chart_axis_label",
            "chart_tick_label",
            "chart_legend_label",
            "figure_caption",
            "running_header",
            "running_footer",
            "section_title",
        }
        if role in {"title", "section_heading", "header", "footer", "figure_caption", "diagram_label", "diagram_text_label"}:
            sensitive.add("role_sensitive")
        max_from_bbox = max(5.0, float(bbox_h_pt or 0.0) * 1.02)
        if (tclass in sensitive) or ("role_sensitive" in sensitive and role in {"title", "section_heading", "header", "footer", "figure_caption", "diagram_label", "diagram_text_label"}):
            return min(max_from_bbox, hint)
        return min(max_from_bbox, hint * 0.96)

    def _preferred_fontsize_for_item(self, item, style, bbox_h_pt, source):
        raw_fs = self._get_original_fontsize(style, bbox_h_pt, source)
        if self._item_preserve_extracted_typography(item):
            return float(raw_fs)
        if source == "native" and isinstance(style.get("size"), (int, float)) and float(style.get("size") or 0.0) > 0.0:
            return float(raw_fs)
        ai_hint = self._layout_ai_font_hint_pt(item, bbox_h_pt)
        if ai_hint <= 0.0:
            return float(raw_fs)
        role = str(item.get("role") or "").strip().lower()
        tclass = str(item.get("descriptor_typographic_class") or "").strip().lower()
        if tclass in {"editorial_body", "diagram_label", "chart_axis_label", "chart_tick_label", "chart_legend_label", "figure_caption", "running_header", "running_footer", "section_title"}:
            return max(float(raw_fs), float(ai_hint))
        if role in {"title", "section_heading", "header", "footer", "figure_caption", "diagram_label", "diagram_text_label"}:
            return max(float(raw_fs), float(ai_hint))
        return float(raw_fs)

    def _should_preserve_double_column_lineation(self, page_data, block, descriptor_region_type, translated_text, source_lines):
        if not isinstance(page_data, dict) or not isinstance(block, dict):
            return False
        if str(block.get("role") or "").strip().lower() != "body":
            return False
        if str(page_data.get("layout_type") or "").strip().lower() != "double_column":
            return False
        if str(page_data.get("document_type") or "").strip().lower() not in {"scientific_paper", "book_page", "manual_guide"}:
            return False
        if str(descriptor_region_type or "").strip().lower() not in {"text", "text_band", "column"}:
            return False
        if str(block.get("render_policy") or "").strip().lower() == "anchored_text":
            return False
        unit_type = str(block.get("unit_type") or "").strip().lower()
        if unit_type in {"short_label", "chart_label", "formula_label", "diagram_label", "reference_link", "citation", "code_visible"}:
            return False
        clean_lines = [self._clean_text_for_render(line).strip() for line in (source_lines or []) if str(line).strip()]
        if len(clean_lines) < 4:
            return False
        src_text = self._clean_text_for_render(self._get_block_source_text(block))
        tr_text = self._clean_text_for_render(translated_text)
        src_words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", src_text)
        tr_words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", tr_text)
        if len(src_words) < 16 or len(tr_words) < 16:
            return False
        word_ratio = len(tr_words) / max(1, len(src_words))
        if word_ratio < 0.55 or word_ratio > 1.45:
            return False
        avg_chars_per_line = len(tr_text) / max(1, len(clean_lines))
        return avg_chars_per_line <= 72.0

    def _should_keep_local_source_slot_geometry_for_anchored_body(
        self,
        page_data,
        block,
        descriptor_region_type,
        descriptor_typographic_class,
        descriptor_structural_role,
        line_entries,
        source,
    ):
        if not isinstance(page_data, dict) or not isinstance(block, dict):
            return False
        if str(block.get("role") or "").strip().lower() != "body":
            return False
        if str(block.get("render_policy") or "").strip().lower() != "anchored_text":
            return False
        if str(page_data.get("layout_type") or "").strip().lower() != "double_column":
            return False
        if str(page_data.get("document_type") or "").strip().lower() not in {"scientific_paper", "book_page", "manual_guide"}:
            return False
        if str(descriptor_region_type or "").strip().lower() not in {"text", "text_band", "column"}:
            return False
        tclass = str(descriptor_typographic_class or "").strip().lower()
        srole = str(descriptor_structural_role or "").strip().lower()
        if tclass != "editorial_body" and srole not in {"body_paragraph", "opening_paragraph"}:
            return False
        metrics = self._translated_body_line_fit_metrics(line_entries, source, overflow_ratio=1.04)
        line_count = int(metrics.get("line_count") or 0)
        overflow_lines = int(metrics.get("overflow_lines") or 0)
        max_ratio = float(metrics.get("max_ratio") or 0.0)
        if line_count < 2 or overflow_lines <= 0:
            return False
        if max_ratio > 2.4:
            return False
        return True

    def _has_meaningful_line_style_variation(self, line_entries):
        entries = line_entries if isinstance(line_entries, list) else []
        signatures = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            text = self._clean_text_for_render(entry.get("text", "")).strip()
            if not text:
                continue
            style = self._merge_styles(entry.get("style", {}), {})
            font_key = str(style.get("font_key_normalized") or style.get("font") or "").strip().lower()
            font_key = re.sub(r"[^a-z0-9]+", "", font_key)
            if not font_key:
                font_key = "unknown"
            flags = style.get("flags") or {}
            signatures.add(
                (
                    font_key,
                    bool(flags.get("bold")),
                    bool(flags.get("italic")),
                    bool(flags.get("monospace")),
                )
            )
            if len(signatures) >= 2:
                return True
        return False

    def _structured_source_lines_with_styles(self, item):
        lines = []
        styles = []
        if not isinstance(item, dict):
            return lines, styles
        raw_lines = item.get("source_lines", []) or []
        raw_styles = item.get("source_line_styles", []) or []
        for idx_line, raw_line in enumerate(raw_lines):
            clean_line = self._clean_text_for_render(raw_line).strip()
            if not clean_line:
                continue
            lines.append(clean_line)
            style_override = raw_styles[idx_line] if idx_line < len(raw_styles) and isinstance(raw_styles[idx_line], dict) else None
            styles.append(dict(style_override) if isinstance(style_override, dict) else None)
        return lines, styles

    def _inline_style_signature(self, style):
        st = style if isinstance(style, dict) else {}
        flags = st.get("flags") or {}
        font_key = str(st.get("font_key_normalized") or st.get("font") or "").strip().lower()
        font_key = re.sub(r"[^a-z0-9]+", "", font_key)
        return (
            font_key,
            bool(flags.get("bold")),
            bool(flags.get("italic")),
            bool(flags.get("monospace")),
            str(st.get("color") or "").strip().lower(),
        )

    def _should_render_inline_style_segments(self, item, segments):
        if not isinstance(item, dict) or not item.get("translated_block"):
            return False
        role = str(item.get("role") or "").strip().lower()
        srole = str(item.get("descriptor_structural_role") or "").strip().lower()
        if role not in {"title", "header", "section_heading"} and srole not in {"table_header_cell", "table_stub_cell"}:
            return False
        segs = segments if isinstance(segments, list) else []
        if len(segs) < 2:
            return False
        signatures = {
            self._inline_style_signature(seg.get("style", {}))
            for seg in segs
            if isinstance(seg, dict)
        }
        return len(signatures) >= 2

    def _partition_translated_line_to_segments(self, translated_text, source_segments):
        text = self._clean_text_for_render(translated_text).strip()
        segments = source_segments if isinstance(source_segments, list) else []
        if not text or len(segments) < 2:
            return []
        tokens = text.split()
        if len(tokens) < len(segments):
            return []
        weights = []
        for seg in segments:
            if not isinstance(seg, dict):
                return []
            source_text = self._clean_text_for_render(seg.get("text", "")).strip()
            word_count = max(1, len(source_text.split()))
            weights.append(float(word_count))
        total_weight = sum(weights)
        if total_weight <= 0:
            return []
        total_tokens = len(tokens)
        boundaries = []
        cumulative = 0.0
        prev = 0
        for idx, weight in enumerate(weights[:-1], start=1):
            cumulative += weight
            raw_boundary = int(round(total_tokens * cumulative / total_weight))
            remaining_segments = len(weights) - idx
            lower = prev + 1
            upper = total_tokens - remaining_segments
            boundary = max(lower, min(upper, raw_boundary))
            boundaries.append(boundary)
            prev = boundary
        parts = []
        start = 0
        for boundary in boundaries + [total_tokens]:
            part = " ".join(tokens[start:boundary]).strip()
            if not part:
                return []
            parts.append(part)
            start = boundary
        return parts if len(parts) == len(segments) else []

    def _should_use_uniform_preserved_line_fontsize(self, item):
        if not isinstance(item, dict):
            return False
        if item.get("preserve_line_style_variation"):
            return False
        if not item.get("preserve_linebreaks"):
            return False
        if not item.get("translated_block"):
            return False
        if not item.get("keep_source_slot_geometry"):
            return False
        if item.get("keep_exact_line"):
            return False
        if str(item.get("role") or "").strip().lower() != "body":
            return False
        page_data = item.get("page_data") or {}
        if str(page_data.get("layout_type") or "").strip().lower() != "double_column":
            return False
        if str(page_data.get("document_type") or "").strip().lower() not in {"scientific_paper", "book_page", "manual_guide"}:
            return False
        region_type = str(item.get("descriptor_region_type") or "").strip().lower()
        if region_type not in {"text", "text_band", "column"}:
            return False
        source_line_count = int(item.get("source_line_count") or len(item.get("source_lines") or []) or 0)
        if source_line_count < 4:
            return False
        return True

    def _fit_uniform_preserved_line_fontsize(
        self,
        lines,
        slot_widths,
        base_fs,
        fontname,
        fontfile,
        overflow_limit=1.03,
        min_font_pt=None,
    ):
        clean_lines = [self._clean_text_for_render(line).strip() for line in (lines or []) if str(line).strip()]
        clean_widths = [float(width) for width in (slot_widths or []) if float(width or 0.0) > 0.0]
        if not clean_lines or not clean_widths:
            return float(base_fs)
        fs = max(5.0, float(base_fs or 0.0))
        min_fs = max(5.8, float(min_font_pt if min_font_pt is not None else max(5.8, fs * 0.86)))
        while fs > min_fs + 1e-6:
            overflow = False
            for idx, line in enumerate(clean_lines):
                width = clean_widths[min(idx, len(clean_widths) - 1)]
                line_w = self._measure_text_width(line, fs, fontname, fontfile)
                if line_w > width * overflow_limit:
                    overflow = True
                    break
            if not overflow:
                return fs
            fs = max(min_fs, fs - 0.2)
        return fs

    def _style_from_block(self, block):
        if not isinstance(block, dict):
            return {}
        if isinstance(block.get("resolved_style"), dict):
            return block.get("resolved_style", {})
        if isinstance(block.get("style"), dict):
            return block.get("style", {})
        return {}

    def _resolve_style_font(self, page, style, text=""):
        style_dict = style if isinstance(style, dict) else {}
        probe_text = self._clean_text_for_render(text or "")
        resolved = self.font_resolver.resolve(style_dict, text=probe_text)
        fontfile = resolved.get("fontfile")
        builtin = resolved.get("builtin")
        if page is None:
            fontname = builtin or str(style_dict.get("font") or "helv")
        else:
            fontname = self._resolve_page_fontname(page, fontfile, builtin)
        return resolved, fontfile, builtin, fontname

    def _normalize_alignment(self, alignment):
        a = (alignment or "left").strip().lower()
        if a not in {"left", "center", "right", "justify"}:
            return "left"
        # Justification must be explicitly detected.
        return a

    def _alignment_payload(self, raw_alignment, source="block", default="left"):
        raw = "" if raw_alignment is None else str(raw_alignment).strip()
        normalized = self._normalize_alignment(raw if raw else default)
        payload = {
            "alignment": normalized,
            "alignment_raw": raw,
            "alignment_source": source,
            "alignment_defaulted": not bool(raw),
            "justify_explicit": raw.lower() == "justify",
        }
        if payload["alignment_defaulted"]:
            payload["alignment_fallback_reason"] = "missing_alignment"
        return payload

    def _resolve_applied_alignment(self, expected_alignment, line_w, left, right, is_last_line=False):
        expected = self._normalize_alignment(expected_alignment)
        avail_w = max(10.0, float(right) - float(left))
        if line_w >= avail_w:
            return "left", "line_wider_than_slot"
        if expected == "justify" and is_last_line:
            return "left", "justify_last_line_left_aligned"
        return expected, ""

    def _item_requires_anchored_render(self, item, anchored_figure_page=False):
        if not isinstance(item, dict):
            return False
        role = str(item.get("role") or "").strip().lower()
        if role in {"header", "footer", "figure_caption", "section_heading", "list_marker"}:
            return True
        if self._is_abbreviation_entry_role(item.get("descriptor_structural_role")):
            return True
        if self._should_render_equation_as_anchored_text(item):
            return True
        if anchored_figure_page and (item.get("is_diagram_label") or role in {"title", "diagram_text_label"}):
            return True
        descriptor_layout_behavior = str(item.get("descriptor_layout_behavior") or "").strip().lower()
        descriptor_band_role = str(item.get("descriptor_band_role") or "").strip().lower()
        descriptor_region_type = str(item.get("descriptor_region_type") or "").strip().lower()
        anchor_target_bbox = item.get("anchor_target_bbox")
        if descriptor_layout_behavior in {"anchored", "locked_in_cell", "locked_in_table"}:
            return True
        if descriptor_band_role in {"annotation_band", "caption_band", "header_band", "legend_band", "axis_band", "table_band"}:
            return True
        if descriptor_region_type in {"annotation_band", "caption_band", "header_band", "table_cell", "table_row", "caption", "header"}:
            return True
        if (
            descriptor_band_role == "title_band"
            and isinstance(anchor_target_bbox, fitz.Rect)
            and anchor_target_bbox.get_area() > 0
        ):
            return True
        return False

    def _item_requires_exact_slot_render(self, item):
        if not isinstance(item, dict):
            return False
        if not bool(item.get("translated_block")):
            return False
        role = str(item.get("role") or "").strip().lower()
        descriptor_band_role = str(item.get("descriptor_band_role") or "").strip().lower()
        descriptor_group_render_mode = str(item.get("descriptor_group_render_mode") or "").strip().lower()
        descriptor_structural_role = str(item.get("descriptor_structural_role") or "").strip().lower()
        descriptor_layout_behavior = str(item.get("descriptor_layout_behavior") or "").strip().lower()
        source_text = self._clean_text_for_render(item.get("source_text", "")).strip()
        rendered_text = self._clean_text_for_render(item.get("text", "")).strip()
        typology = classify_block_typology(
            {
                "role": item.get("role"),
                "semantic": {"type": "heading" if str(item.get("role") or "").strip().lower() in {"title", "section_heading"} else "body"},
                "structure_hints": {
                    "band_role_hint": descriptor_band_role,
                    "structural_role_hint": descriptor_structural_role,
                    "layout_behavior_hint": descriptor_layout_behavior,
                },
                "lines": list(item.get("source_lines") or []),
            }
        )
        length_ratio_ok = bool(
            source_text
            and rendered_text
            and len(rendered_text) <= max(len(source_text) + 2, int(len(source_text) * 1.08))
        )
        if typology["subtype"] == "visual_label":
            if descriptor_band_role in {"annotation_band", "legend_band", "axis_band"}:
                return True
            if descriptor_group_render_mode in {"annotation_group", "chart_legend_group", "chart_axis_group", "chart_series_group"}:
                return True
            return role in {"diagram_text_label", "diagram_label"}
        if self._is_abbreviation_key_role(descriptor_structural_role):
            return True
        editorial_short_callout = bool(
            rendered_text
            and source_text
            and rendered_text.lower() != source_text.lower()
            and length_ratio_ok
            and role in {"title", "section_heading", "body", "figure_caption"}
            and len(item.get("source_lines") or []) <= 1
            and typology["subtype"] in {"editorial_locked_callout", "editorial_short_callout"}
            and not re.search(r"[_=]{6,}", source_text)
            and not re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*", source_text)
            and not re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(", source_text)
        )
        return editorial_short_callout

    def _expand_anchor_target_span(self, item, left, right, x0, block_right):
        if not isinstance(item, dict):
            return x0, block_right
        anchor_target_bbox = item.get("anchor_target_bbox")
        if not isinstance(anchor_target_bbox, fitz.Rect) or anchor_target_bbox.get_area() <= 0:
            return x0, block_right
        descriptor_layout_behavior = str(item.get("descriptor_layout_behavior") or "").strip().lower()
        descriptor_band_role = str(item.get("descriptor_band_role") or "").strip().lower()
        role = str(item.get("role") or "").strip().lower()
        if not (
            item.get("is_diagram_label")
            or role in {"title", "header", "figure_caption", "diagram_text_label"}
            or descriptor_layout_behavior == "anchored"
            or descriptor_band_role in {"annotation_band", "caption_band", "header_band", "legend_band", "axis_band", "title_band"}
        ):
            return x0, block_right
        span_left = max(left, float(x0))
        span_right = min(right, max(float(block_right), span_left + 8.0))
        anchor_preferred_side = str(item.get("anchor_preferred_side") or "").strip().lower()
        if anchor_preferred_side == "left_of":
            span_right = min(right, max(span_right, anchor_target_bbox.x0 - 2.0))
        elif anchor_preferred_side == "right_of":
            span_left = max(left, max(span_left, min(right - 8.0, anchor_target_bbox.x1 + 2.0)))
            span_right = min(right, max(span_right, span_left + max(36.0, anchor_target_bbox.width * 0.55)))
        else:
            span_left = max(left, min(span_left, anchor_target_bbox.x0))
            span_right = min(right, max(span_right, anchor_target_bbox.x1))
        if span_right <= span_left + 8.0:
            span_right = min(right, span_left + 8.0)
        return span_left, span_right

    def _compose_exact_slot_text(self, text, slot_w, slot_h, base_fs, fontname, fontfile, source="ocr", alignment="left", max_font_shrink=1.6, min_font_pt=5.0, line_height_factor=1.18):
        comp = self.text_composer.compose_text_in_box(
            text=self._clean_text_for_render(text),
            box_w=max(8.0, slot_w),
            box_h=max(8.0, slot_h),
            base_font_pt=float(base_fs),
            line_height_factor=float(line_height_factor),
            measure_fn=lambda t, fsz: self._measure_text_width(t, fsz, fontname, fontfile),
            alignment=self._normalize_alignment(alignment),
            lang="en",
            options=ComposeOptions(
                enable_hyphenation=(source != "native"),
                max_font_shrink=float(max_font_shrink),
                min_font_pt=float(min_font_pt),
                step_pt=0.2,
            ),
        )
        lines = [self._clean_text_for_render(line).strip() for line in (comp.get("lines") or []) if str(line).strip()]
        return {
            "lines": lines,
            "font_size": float(comp.get("font_size", base_fs) or base_fs),
            "overflow": self._clean_text_for_render(comp.get("overflow") or ""),
        }

    def _baseline_ratio(self, style, fontsize):
        flags = style.get("flags", {}) if isinstance(style, dict) else {}
        font_name = str(style.get("font", "")).lower() if isinstance(style, dict) else ""
        ratio = 0.80
        if flags.get("serif"):
            ratio = 0.78
        if flags.get("italic"):
            ratio = min(ratio, 0.79)
        if any(k in font_name for k in ("times", "baskerville", "garamond")):
            ratio = min(ratio, 0.775)
        if any(k in font_name for k in ("arial", "helvetica", "franklin", "gothic")):
            ratio = max(ratio, 0.81)
        if "mono" in font_name:
            ratio = 0.79
        if fontsize >= 14:
            ratio -= 0.01
        elif fontsize <= 8:
            ratio += 0.005
        return max(0.74, min(0.84, ratio))

    def _should_reorder_top_header_number(self, text):
        clean = self._clean_text_for_render(text)
        if not clean:
            return False
        if not re.match(r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9\s\-\']+\s+\d{1,3}$", clean):
            return False
        if re.match(r"^(?:chapter|chapitre|part|partie)\b", clean, flags=re.IGNORECASE):
            return False
        if re.match(r"^(?:appendix|annexe)\s+[A-Z0-9]+\b", clean, flags=re.IGNORECASE):
            return False
        return True

    def _resolve_header_item_collisions(self, items, page_w_pt):
        if not isinstance(items, list) or not items:
            return
        groups = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            if str(item.get("role") or "").strip().lower() != "header":
                continue
            bbox = item.get("bbox")
            slots = item.get("slots") or []
            if not isinstance(bbox, fitz.Rect) or bbox.get_area() <= 0 or not slots:
                continue
            key = round(float(bbox.y0), 1)
            groups.setdefault(key, []).append(item)
        for _, group in groups.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda it: float((it.get("slots") or [it.get("bbox")])[0].x0))
            prev_end = None
            prev_text = ""
            prev_start_x = None
            for item in group:
                text = self._clean_text_for_render(item.get("text", ""))
                slots = item.get("slots") or []
                bbox = item.get("bbox")
                if not text or not slots or not isinstance(bbox, fitz.Rect):
                    continue
                slot0 = slots[0]
                if not isinstance(slot0, fitz.Rect):
                    continue
                style = item.get("style") or {}
                resolved = self.font_resolver.resolve(style, text=text) if hasattr(self, "font_resolver") else {}
                fontfile = resolved.get("fontfile")
                builtin = resolved.get("builtin")
                try:
                    fontname = self._resolve_page_fontname(None, fontfile, builtin)
                except Exception:
                    fontname = builtin or "helv"
                try:
                    fontsize = float(style.get("font_size_pt") or style.get("size") or 9.0)
                except Exception:
                    fontsize = 9.0
                text_w = max(0.0, self._measure_text_width(text, fontsize, fontname, fontfile))
                text_w = max(text_w * 1.24, float(slot0.width) + max(4.0, fontsize * 0.55))
                min_gap = max(10.0, fontsize * 0.72)
                chapter_like_prev = bool(re.match(r"^(?:chapter|chapitre|part|partie)\b", prev_text, flags=re.IGNORECASE))
                if re.match(r"^(?:chapter|chapitre|part|partie)\b", prev_text, flags=re.IGNORECASE):
                    text_w = max(text_w, float(slot0.width) + max(10.0, fontsize * 1.2))
                    min_gap = max(min_gap, 18.0)
                cur_x = float(slot0.x0)
                min_start_x = prev_end + min_gap if prev_end is not None else cur_x
                if chapter_like_prev and prev_start_x is not None:
                    min_start_x = max(min_start_x, prev_start_x + max(92.0, fontsize * 10.0))
                if cur_x < min_start_x:
                    shift = min(page_w_pt - cur_x - 6.0, min_start_x - cur_x)
                    if shift > 0.5:
                        old_bbox = fitz.Rect(bbox)
                        for idx, slot in enumerate(slots):
                            if isinstance(slot, fitz.Rect):
                                slots[idx] = fitz.Rect(slot.x0 + shift, slot.y0, slot.x1 + shift, slot.y1)
                        item["slots"] = slots
                        item["row_start_x_pt"] = float(item.get("row_start_x_pt") or bbox.x0) + shift
                        new_bbox = fitz.Rect(bbox.x0 + shift, bbox.y0, bbox.x1 + shift, bbox.y1)
                        item["bbox"] = new_bbox
                        item["whiteout_bbox"] = fitz.Rect(
                            min(old_bbox.x0, new_bbox.x0) - 1.5,
                            min(old_bbox.y0, new_bbox.y0) - 0.6,
                            max(old_bbox.x1, new_bbox.x1) + 1.5,
                            max(old_bbox.y1, new_bbox.y1) + 0.6,
                        )
                        slot0 = slots[0]
                        cur_x = float(slot0.x0)
                prev_end = max(prev_end or 0.0, cur_x + text_w)
                prev_text = text
                prev_start_x = cur_x

    def reconstruct(self, structure, output_path):
        doc = fitz.open()
        debug_store = {}
        self._style_audit_records = []
        pages_list = structure.get("pages", [])
        if not pages_list and "blocks" in structure: pages_list = [structure]

        for i, page_data in enumerate(pages_list):
            self._augment_page_data_from_layout_xml(page_data)
            dims = page_data.get("dimensions", {"width": 885, "height": 1110})
            w_pt, h_pt = dims["width"] * self.pixel_to_point, dims["height"] * self.pixel_to_point
            page = doc.new_page(width=w_pt, height=h_pt)
            page_index = int(page.number)
            
            # Fond Maître
            bg_path = page_data.get("background_path")
            if bg_path and os.path.exists(bg_path):
                page.insert_image(page.rect, filename=bg_path)
            self._inject_dynamic_immutable_overlays(page_data)

            has_translated = self.translation_reflow_mode and self._has_translated_content(page_data)
            forbidden_rects = (
                self._collect_translation_forbidden_rects(page_data)
                if has_translated
                else self._collect_forbidden_rects(page_data)
            )
            self._rendered_signatures = set()
            self._restored_background_rects = {}
            self._prepared_visual_groups = {}
            self._typography_group_cache = {}
            if has_translated:
                self._reconstruct_translated_anchored(doc, page, page_data, debug_store, forbidden_rects=forbidden_rects)
            elif self.native_first_mode and self._has_native_blocks(page_data):
                self._reconstruct_strict(page, page_data, forbidden_rects)
            elif self.sequential_layout_mode:
                self._reconstruct_sequential_flow(doc, page, page_data, debug_store, forbidden_rects=forbidden_rects)
            elif self.flow_layout_mode:
                self._reconstruct_with_flow(page, page_data, forbidden_rects)
            else:
                self._reconstruct_strict(page, page_data, forbidden_rects)
            # A page object can become invalid after page insertions; reload it.
            page = doc[page_index]
            self._postcheck_equation_fidelity(page, page_data)
            self._postcheck_native_block_fidelity(page, page_data)
            # Immutable overlays must be placed last so translated text cannot hide them.
            self._insert_immutable_overlays(page, page_data)

        if self.layout_debug_overlay:
            self._save_layout_debug_overlays(doc, debug_store, output_path)
        doc.save(output_path)
        doc.close()
        self._save_style_audit_report(output_path)
        return output_path

    def _parse_bbox_csv(self, s):
        try:
            vals = [float(x.strip()) for x in str(s or "").split(",")]
            if len(vals) != 4:
                return None
            return vals
        except Exception:
            return None

    def _resolve_trusted_layout_xml_path(self, xml_path):
        if not xml_path:
            return None
        try:
            candidate = os.path.realpath(str(xml_path))
        except Exception:
            return None
        if not candidate.lower().endswith(".xml"):
            return None
        if not os.path.isfile(candidate):
            return None
        try:
            size = os.path.getsize(candidate)
        except Exception:
            return None
        if size > self._layout_xml_max_bytes:
            return None
        for root in self._layout_xml_trusted_dirs:
            try:
                if os.path.commonpath([candidate, root]) == root:
                    return candidate
            except Exception:
                continue
        return None

    def _augment_page_data_from_layout_xml(self, page_data):
        xml_path = self._resolve_trusted_layout_xml_path(page_data.get("layout_xml_path"))
        if not xml_path:
            return
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception:
            return
        xml_blocks = {}
        for b_el in root.findall("./blocks/block"):
            bid = (b_el.get("id") or "").strip()
            if not bid:
                continue
            lines = []
            line_texts = []
            for l_el in b_el.findall("./line"):
                ltxt = (l_el.findtext("text") or "").strip()
                if ltxt:
                    line_texts.append(ltxt)
                lines.append(
                    {
                        "index": int(l_el.get("index") or 0),
                        "bbox": self._parse_bbox_csv(l_el.get("bbox")),
                        "marker": (l_el.get("marker") or ""),
                        "indent_px": float(l_el.get("indent_px") or 0.0),
                        "hard_break_before": (l_el.get("hard_break_before") == "1"),
                        "line_break_after": (l_el.get("line_break_after") != "0"),
                        "text": ltxt,
                    }
                )
            xml_blocks[bid] = {"line_texts": line_texts, "lines": lines}

        for block in page_data.get("blocks", []):
            bid = str(block.get("id", "")).strip()
            xb = xml_blocks.get(bid)
            if not xb:
                continue
            if xb.get("line_texts"):
                block["line_texts"] = list(xb["line_texts"])
                block["render_text_with_breaks"] = "\n".join(xb["line_texts"]).strip()
            lines = block.get("lines", [])
            for idx, ln in enumerate(lines):
                xl = xb["lines"][idx] if idx < len(xb["lines"]) else None
                if not xl:
                    continue
                ln["line_index"] = int(xl.get("index", idx))
                ln["leading_marker"] = xl.get("marker", "")
                ln["indent_px"] = float(xl.get("indent_px", 0.0) or 0.0)
                ln["hard_break_before"] = bool(xl.get("hard_break_before", False))
                ln["line_break_after"] = bool(xl.get("line_break_after", True))
                if xl.get("text"):
                    ln["line_text"] = xl["text"]
                for ph in ln.get("phrases", []):
                    ph["line_index"] = ln["line_index"]
                    ph["leading_marker"] = ln["leading_marker"]
                    ph["indent_px"] = ln["indent_px"]
                    ph["hard_break_before"] = ln["hard_break_before"]
                    ph["line_break_after"] = ln["line_break_after"]

    # -------------------------
    # TOC / Sommaire heuristics
    # -------------------------
    def _looks_like_toc_page(self, page_data):
        """Heuristic detection: many lines ending with page numbers + TOC title near top."""
        if not isinstance(page_data, dict):
            return False
        if str(page_data.get("schema_version") or "").strip().lower() == "layout.v2":
            explicit_role = str(page_data.get("page_role") or "").strip().lower()
            if explicit_role:
                return explicit_role == "toc"
        blocks = page_data.get("blocks", []) or []
        dims = page_data.get("dimensions", {}) or {}
        page_h_px = float(dims.get("height", 1.0) or 1.0)
        top_hits = 0
        toc_line_hits = 0
        total_lines = 0
        block_number_hits = 0

        for b in blocks:
            if b.get("render_mode") == "background_only":
                continue
            bb = b.get("bbox") or [0, 0, 0, 0]
            try:
                by0 = float(bb[1])
            except Exception:
                by0 = 1e9
            txt = (b.get("translated_text") or b.get("text") or "").strip()
            if by0 <= page_h_px * 0.22 and re.search(r"\b(CONTENTS|SOMMAIRE|TABLE\s+OF\s+CONTENTS)\b", txt, flags=re.I):
                top_hits += 1
            if txt:
                if re.search(r"\b\d+(?:\.\d+)*\b", txt) and len(re.findall(r"\b\d{1,3}\b", txt)) >= 1:
                    block_number_hits += 1

            for ln in b.get("lines", []) or []:
                total_lines += 1
                lt = (ln.get("translated_text") or ln.get("line_text") or "").strip()
                if not lt:
                    parts = []
                    for ph in ln.get("phrases", []) or []:
                        t = (ph.get("translated_text") or ph.get("texte") or "").strip()
                        if t:
                            parts.append(t)
                    lt = " ".join(parts).strip()
                if not lt:
                    continue

                if re.search(r"^\s*\d+(?:\.\d+)*\s+.+\s+\d{1,3}\s*$", lt):
                    toc_line_hits += 1
                elif re.search(r"^\s*[A-Za-z].+\s+\d{1,3}\s*$", lt) and len(lt) <= 110:
                    toc_line_hits += 1

        if top_hits >= 1 and toc_line_hits >= 6:
            return True
        if top_hits >= 1 and block_number_hits >= 8 and len(blocks) >= 10:
            return True
        if toc_line_hits >= 10 and total_lines >= 12:
            return True
        return False

    def _split_toc_line(self, s):
        s = self._clean_text_for_render(s or "")
        m = re.search(r"(.*?)\s+(\d{1,3})\s*$", s)
        if not m:
            return s, ""
        left = self._clean_text_for_render(m.group(1)).strip()
        right = m.group(2).strip()
        return left, right

    def _render_toc_item(self, page, item, anchor_y, left, right, zone_top, zone_bottom, forbidden_rects):
        """Render a TOC-like block using tab stop + dot leaders + right-aligned page numbers."""
        style = item.get("style", {}) or {}
        source = item.get("source", "ocr")
        _, fontfile, builtin, fontname = self._resolve_style_font(page, style, text=item.get("text", ""))
        base_fs = self._get_original_fontsize(style, max(1.0, float(item.get("slot_h_pt", 10.0))), source)
        rgb = self._resolve_text_color(style, item)

        bbox = item.get("bbox")
        if isinstance(bbox, fitz.Rect):
            x_left = max(left, min(bbox.x0, right - 90.0))
            x_right = min(right, max(bbox.x1, x_left + 140.0))
        else:
            x_left = left
            x_right = right
        x_right = max(x_right, right - max(60.0, (right - left) * 0.22))

        line_h = max(1.0, base_fs * 1.30)
        gap_y = max(1.5, base_fs * 0.35)

        if item.get("preserve_linebreaks") and item.get("use_structured_source_lines"):
            raw_lines = item.get("source_lines", []) or []
        else:
            raw_lines = (item.get("text", "") or "").split("\n")
        raw_lines = [self._clean_text_for_render(x).strip() for x in raw_lines if str(x).strip()]

        y = max(zone_top, anchor_y)
        used_bottom = y
        blue_rect = None

        for raw in raw_lines:
            if y + line_h > zone_bottom:
                break
            probe = fitz.Rect(x_left, y, x_right, y + line_h)
            for _ in range(6):
                collisions = [fr for fr in forbidden_rects if (probe & fr).get_area() > 0]
                if not collisions:
                    break
                y = max(fr.y1 for fr in collisions) + max(1.0, gap_y * 0.5)
                if y + line_h > zone_bottom:
                    break
                probe = fitz.Rect(x_left, y, x_right, y + line_h)
            if y + line_h > zone_bottom:
                break

            left_txt, right_txt = self._split_toc_line(raw)

            if left_txt:
                self._safe_insert_text_dedup(page, (x_left, y + line_h * 0.82), left_txt, base_fs, fontname, rgb)

            num_x = None
            if right_txt:
                w_num = self._measure_text_width(right_txt, base_fs, fontname, fontfile)
                num_x = max(x_left + 40.0, x_right - w_num)
                self._safe_insert_text_dedup(page, (num_x, y + line_h * 0.82), right_txt, base_fs, fontname, rgb)

            if left_txt and right_txt and num_x is not None:
                w_left = self._measure_text_width(left_txt, base_fs, fontname, fontfile)
                lead_start = x_left + w_left + max(6.0, base_fs * 0.6)
                lead_end = num_x - max(6.0, base_fs * 0.6)
                if lead_end - lead_start >= max(10.0, base_fs * 2.0):
                    dot = "."
                    w_dot = max(1.0, self._measure_text_width(dot, base_fs, fontname, fontfile))
                    n = int(max(0.0, (lead_end - lead_start) / w_dot))
                    leader = dot * max(2, min(220, n))
                    self._safe_insert_text_dedup(page, (lead_start, y + line_h * 0.82), leader, base_fs, fontname, rgb)

            used_bottom = max(used_bottom, y + line_h)
            y += line_h + gap_y

        if used_bottom > anchor_y:
            blue_rect = fitz.Rect(x_left, anchor_y, x_right, used_bottom)
            forbidden_rects.append(fitz.Rect(blue_rect))

        return used_bottom, blue_rect

    def _collect_translation_forbidden_rects(self, page_data):
        # In translation mode, keep only real blocking zones.
        # Small immutable inline overlays (symbols/references inside paragraphs)
        # should not push whole blocks down.
        rects = []
        page_role = str((page_data or {}).get("page_role", "")).strip().lower()
        document_type = str((page_data or {}).get("document_type") or ((page_data or {}).get("layout") or {}).get("document_type") or "").strip().lower()
        page_family = str((page_data or {}).get("page_family") or ((page_data or {}).get("layout") or {}).get("page_family") or "").strip().lower()
        page_family_group = str((page_data or {}).get("page_family_group") or ((page_data or {}).get("layout") or {}).get("page_family_group") or page_family).strip().lower()
        layout_type = str((page_data or {}).get("layout_type") or ((page_data or {}).get("layout") or {}).get("layout_type") or "").strip().lower()
        page_case = (page_data or {}).get("page_case") or {}
        fallback_policy = str(page_case.get("fallback_policy") or "").strip().lower()
        mixed_like_page = (
            layout_type in {"mixed_blocks", "annotated_page"}
            and document_type not in {"scientific_paper", "book_page", "manual_guide"}
        ) or page_family in {"mixed_page"} or page_family_group in {"mixed_page"} or fallback_policy == "safe_mixed"
        # NOTE: non_text_zones can be noisy and may overlap paragraph regions.
        # We intentionally ignore them in translated anchored mode.
        for im in page_data.get("images", []):
            bbox = im.get("bbox") if isinstance(im, dict) else im
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            area_px = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
            if area_px < 4500:
                continue
            x0, y0, x1, y1 = [float(v) * self.pixel_to_point for v in bbox]
            rects.append(fitz.Rect(x0, y0, x1, y1))
        for ov in page_data.get("immutable_overlays", []):
            bbox = ov.get("bbox") if isinstance(ov, dict) else None
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            kind = (ov.get("kind") or ov.get("reason") or "").lower() if isinstance(ov, dict) else ""
            ov_text = str(ov.get("text", "")).strip() if isinstance(ov, dict) else ""
            area_px = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
            # Keep only large/structural overlays as blockers.
            is_page_marker = bool(re.fullmatch(r"\d{1,4}|[ivxlcdm]+", ov_text, flags=re.IGNORECASE))
            if area_px < 4500 and kind not in {"diagram_block"} and not (page_role == "toc" or is_page_marker):
                continue
            x0, y0, x1, y1 = [float(v) * self.pixel_to_point for v in bbox]
            rects.append(fitz.Rect(x0, y0, x1, y1))
        if mixed_like_page:
            for zone in page_data.get("non_text_zones", []) or []:
                bbox = zone.get("bbox") if isinstance(zone, dict) else zone
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                area_px = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
                if area_px < 5000:
                    continue
                x0, y0, x1, y1 = [float(v) * self.pixel_to_point for v in bbox]
                rects.append(fitz.Rect(x0, y0, x1, y1))
        return rects

    def _save_style_audit_report(self, output_path):
        if not self.style_audit_enabled:
            return
        records = getattr(self, "_style_audit_records", None) or []
        if not records:
            return
        out_dir = os.path.dirname(output_path) or "."
        base = os.path.splitext(os.path.basename(output_path))[0]
        out_path = os.path.join(out_dir, f"{base}_style_audit.json")
        summary = {
            "records": len(records),
            "font_fallback_count": sum(1 for r in records if r.get("font_fallback")),
            "size_delta_nonzero": sum(1 for r in records if abs(float(r.get("size_delta_pt", 0.0))) > 1e-6),
            "color_mismatch": sum(1 for r in records if r.get("expected_color") != r.get("applied_color")),
            "alignment_mismatch": sum(1 for r in records if r.get("expected_alignment") != r.get("applied_alignment")),
            "alignment_fallback_reasons": dict(
                Counter(
                    str(r.get("alignment_fallback_reason", "")).strip()
                    for r in records
                    if str(r.get("alignment_fallback_reason", "")).strip()
                )
            ),
        }
        payload = {"summary": summary, "records": records}
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _reconstruct_strict(self, page, page_data, forbidden_rects):
        placed_rects = []
        seen_spans = []
        for block in page_data.get("blocks", []):
            if block.get("render_mode") == "background_only":
                continue
            for line in block.get("lines", []):
                for phrase in line.get("phrases", []):
                    if phrase.get("render_mode") == "background_only":
                        continue
                    for span in phrase.get("spans", []):
                        if span.get("skip_render"):
                            continue
                        if self.layout_correction and self._is_duplicate_span(span, seen_spans):
                            continue
                        self._insert_hierarchical_span(
                            page,
                            span,
                            source=block.get("source", "ocr"),
                            placed_rects=placed_rects,
                            forbidden_rects=forbidden_rects,
                            allow_shift=(block.get("source", "ocr") != "native"),
                        )

    def _reconstruct_with_flow(self, page, page_data, forbidden_rects):
        items = self._extract_flow_items(page_data, forbidden_rects)
        flow_items = [it for it in items if it["kind"] != "diagram_label"]
        strict_items = [it for it in items if it["kind"] == "diagram_label"]
        placed_rects = []

        # 1) Keep diagram labels at original location.
        for item in strict_items:
            for span in item.get("spans", []):
                self._insert_hierarchical_span(
                    page,
                    span,
                    source=item.get("source", "ocr"),
                    placed_rects=[],
                    forbidden_rects=[],
                    allow_shift=False,
                )

        # 2) Reflow body/captions through allowed frames (excluding figures).
        frames = self._build_flow_frames(page.rect, flow_items, forbidden_rects)
        if not frames:
            frames = [fitz.Rect(page.rect)]
        frame_idx = 0
        cursor_y = frames[0].y0
        for item in flow_items:
            frame_idx, cursor_y = self._place_item_in_frames(
                page=page,
                item=item,
                frames=frames,
                frame_idx=frame_idx,
                cursor_y=cursor_y,
                placed_rects=placed_rects,
                forbidden_rects=forbidden_rects,
            )

    def _has_translated_content(self, page_data):
        if isinstance(page_data, dict) and page_data.get("schema_version") == "layout.v2":
            if str(page_data.get("page_role", "")).strip().lower() == "toc":
                rows = ((page_data.get("toc") or {}).get("toc_rows") or [])
                for row in rows:
                    label = str(row.get("label") or "").strip()
                    translated = str(row.get("translated_label") or "").strip()
                    if translated and translated != label:
                        return True
        for block in page_data.get("blocks", []):
            if self._is_translated_block(block):
                return True
        return False

    def _reconstruct_sequential_flow(self, doc, first_page, page_data, debug_store=None, forbidden_rects=None):
        items = self._extract_block_slot_items(page_data)
        if not items:
            return
        forbidden_rects = forbidden_rects or []

        layout = page_data.get("layout", {}) or {}
        margins_px = layout.get("margins", {}) or {}
        margin_l = max(8.0, float(margins_px.get("left", 0.0)) * self.pixel_to_point)
        margin_r = max(8.0, float(margins_px.get("right", 0.0)) * self.pixel_to_point)
        margin_t = max(8.0, float(margins_px.get("top", 0.0)) * self.pixel_to_point)
        margin_b = max(8.0, float(margins_px.get("bottom", 0.0)) * self.pixel_to_point)
        header_band = layout.get("header_band", [0, 0])
        footer_band = layout.get("footer_band", [0, 0])

        page = first_page
        right_safety = 20.0
        left = page.rect.x0 + margin_l
        right = page.rect.x1 - margin_r
        top = page.rect.y0 + margin_t
        bottom = page.rect.y1 - margin_b
        header_bottom = page.rect.y0 + (max(0.0, float(header_band[1])) * self.pixel_to_point if len(header_band) == 2 else top)
        footer_top = page.rect.y0 + (max(0.0, float(footer_band[0])) * self.pixel_to_point if len(footer_band) == 2 else bottom)
        if header_bottom > bottom:
            header_bottom = min(bottom, top + page.rect.height * 0.12)
        if footer_top < top:
            footer_top = max(top, bottom - page.rect.height * 0.12)
        body_top = min(bottom, max(top, header_bottom + 2.0))
        body_bottom = max(body_top + 10.0, min(bottom, footer_top - 2.0))

        header_items = [it for it in items if it.get("role") == "header"]
        footer_items = [it for it in items if it.get("role") == "footer"]
        caption_items = [it for it in items if it.get("role") == "figure_caption"]
        diagram_items = [it for it in items if it.get("is_diagram_label")]
        equation_items = [it for it in items if it.get("role") == "equation_inline"]
        diagram_text_items = [it for it in items if it.get("role") == "diagram_text_label"]
        body_items = [
            it
            for it in items
            if it.get("role") not in {"header", "footer", "equation_inline", "diagram_text_label", "figure_caption"} and not it.get("is_diagram_label")
        ]

        for item in diagram_items:
            _, _, blue_rect, used_slots = self._render_block_slots(
                page=page,
                item=item,
                anchor_y=max(top, min(item["bbox"].y0, bottom - 8.0)),
                left=left,
                right=right,
                zone_top=max(top, item["bbox"].y0 - 2.0),
                zone_bottom=min(bottom, item["bbox"].y1 + max(6.0, item.get("slot_h_pt", 8.0) * 1.5)),
            )
            self._append_debug_rects(debug_store, page, blue_rect, used_slots)

        for item in caption_items:
            _, _, blue_rect, used_slots = self._render_block_slots(
                page=page,
                item=item,
                anchor_y=max(top, min(item["bbox"].y0, bottom - 8.0)),
                left=left,
                right=right,
                zone_top=max(top, item["bbox"].y0 - max(4.0, item.get("slot_h_pt", 8.0) * 0.4)),
                zone_bottom=min(bottom, item["bbox"].y1 + max(8.0, item.get("slot_h_pt", 8.0) * 2.0)),
            )
            self._append_debug_rects(debug_store, page, blue_rect, used_slots)

        # Keep equations/formulas at original location to avoid reflow loss.
        for item in equation_items:
            if self._should_render_equation_as_anchored_text(item):
                zone_top, zone_bottom = self._anchored_zone_bounds(item, top, bottom)
                _, _, blue_rect, used_slots = self._render_block_slots(
                    page=page,
                    item=item,
                    anchor_y=max(top, min(item["bbox"].y0, bottom - 8.0)),
                    left=left,
                    right=right,
                    zone_top=zone_top,
                    zone_bottom=zone_bottom,
                )
                self._append_debug_rects(debug_store, page, blue_rect, used_slots)
            else:
                self._render_fixed_item(page, item)
        for item in diagram_text_items:
            self._render_fixed_item(page, item)

        cursor_y = body_top
        page_forbidden = list(forbidden_rects)
        for item in caption_items:
            bb = item.get("bbox")
            if isinstance(bb, fitz.Rect) and bb.get_area() > 0:
                page_forbidden.append(fitz.Rect(bb))
        for item in body_items:
            remaining = item.get("text", "").strip()
            if not remaining:
                continue
            original_remaining = remaining
            anchor_y = max(body_top, cursor_y, item["bbox"].y0)
            anchor_y = self._shift_anchor_below_forbidden(
                anchor_y=anchor_y,
                item=item,
                left=left,
                right=right,
                zone_top=body_top,
                zone_bottom=body_bottom,
                forbidden_rects=page_forbidden,
            )
            last_bottom = anchor_y
            while remaining:
                override_text = remaining
                if item.get("preserve_linebreaks") and remaining == original_remaining:
                    override_text = None
                remaining, used_bottom, blue_rect, used_slots = self._render_block_slots(
                    page=page,
                    item=item,
                    anchor_y=anchor_y,
                    left=left,
                    right=right,
                    zone_top=body_top,
                    zone_bottom=body_bottom,
                    override_text=override_text,
                )
                self._append_debug_rects(debug_store, page, blue_rect, used_slots)
                last_bottom = max(last_bottom, used_bottom)
                if blue_rect is not None:
                    page_forbidden.append(fitz.Rect(blue_rect))
                if not remaining:
                    break
                if not self._should_paginate_on_overflow():
                    break
                page = doc.new_page(width=page.rect.width, height=page.rect.height)
                left = page.rect.x0 + margin_l
                right = page.rect.x1 - margin_r
                top = page.rect.y0 + margin_t
                bottom = page.rect.y1 - margin_b
                body_top = min(bottom, max(top, header_bottom + 2.0))
                body_bottom = max(body_top + 10.0, min(bottom, footer_top - 2.0))
                anchor_y = body_top
                last_bottom = body_top
                page_forbidden = []
            cursor_y = min(body_bottom, last_bottom + max(4.0, item.get("slot_h_pt", 10.0) * 0.45))

        for item in header_items:
            descriptor_region_bbox = item.get("descriptor_region_bbox")
            slot_h = max(8.0, float(item.get("slot_h_pt", 8.0) or 8.0))
            header_zone_bottom = max(top + 8.0, min(header_bottom, bottom))
            header_zone_bottom = max(header_zone_bottom, min(bottom, item["bbox"].y1 + max(10.0, slot_h * 2.0)))
            if isinstance(descriptor_region_bbox, fitz.Rect) and descriptor_region_bbox.get_area() > 0:
                header_zone_bottom = max(header_zone_bottom, min(bottom, descriptor_region_bbox.y1))
            _, _, blue_rect, used_slots = self._render_block_slots(
                page=page,
                item=item,
                anchor_y=max(top, min(item["bbox"].y0, header_bottom - 8.0)),
                left=left,
                right=right,
                zone_top=top,
                zone_bottom=header_zone_bottom,
            )
            self._append_debug_rects(debug_store, page, blue_rect, used_slots)

        for item in footer_items:
            _, _, blue_rect, used_slots = self._render_block_slots(
                page=page,
                item=item,
                anchor_y=max(top, min(item["bbox"].y0, bottom - 8.0)),
                left=left,
                right=right,
                zone_top=max(top, min(footer_top, bottom - 8.0)),
                zone_bottom=bottom,
            )
            self._append_debug_rects(debug_store, page, blue_rect, used_slots)

    def _reconstruct_translated_anchored(self, doc, first_page, page_data, debug_store=None, forbidden_rects=None):
        items = self._extract_block_slot_items(page_data)
        if not items:
            return
        forbidden_rects = forbidden_rects or []
        page = first_page
        root_page_index = int(first_page.number)
        base_page_rect = fitz.Rect(page.rect)
        base_page_width = float(base_page_rect.width)
        base_page_height = float(base_page_rect.height)
        page_forbidden = list(forbidden_rects)
        top = page.rect.y0 + 2.0
        bottom = page.rect.y1 - 2.0
        left = page.rect.x0 + 2.0
        right = page.rect.x1 - 2.0
        page_family = str(page_data.get("page_family") or ((page_data.get("layout") or {}).get("page_family")) or "").strip().lower()
        layout_type = str(page_data.get("layout_type") or ((page_data.get("layout") or {}).get("layout_type")) or "").strip().lower()
        document_type = str(page_data.get("document_type") or ((page_data.get("layout") or {}).get("document_type")) or "").strip().lower()
        anchored_figure_page = (
            layout_type in {"annotated_page", "table_dominant", "image_dominant", "mixed_blocks"}
            and document_type not in {"scientific_paper", "book_page", "manual_guide"}
        ) or page_family in {"body_with_figure", "body_with_diagram", "mixed_page", "table_page"}

        # --- layout.v2 TOC fast path ---
        if isinstance(page_data, dict) and page_data.get("schema_version") == "layout.v2":
            if page_data.get("page_role") == "toc" and isinstance(page_data.get("toc"), dict):
                toc = page_data.get("toc") or {}
                rows = toc.get("toc_rows") or []
                tab = toc.get("tab_stops") or {}
                if rows:
                    self._render_toc_rows_v2(
                        page=page,
                        rows=rows,
                        tab_stops=tab,
                        zone_top=top,
                        zone_bottom=bottom,
                        left=left,
                        right=right,
                    )
                    return

        # Keep natural reading order and preserve relative Y as much as possible.
        body_items = [it for it in items if not self._item_requires_anchored_render(it, anchored_figure_page=anchored_figure_page)]
        # Strict fidelity: keep source block segmentation to preserve natural
        # paragraph/list line breaks and avoid artificial merges.
        if not self.pro_strict_mode:
            body_items = self._merge_translated_body_items(body_items)
        fixed_items = [it for it in items if self._item_requires_anchored_render(it, anchored_figure_page=anchored_figure_page)]
        late_fixed_roles = {"header", "footer"}
        early_fixed_items = [it for it in fixed_items if it.get("role") not in late_fixed_roles]
        late_fixed_items = [it for it in fixed_items if it.get("role") in late_fixed_roles]
        toc_mode = self._looks_like_toc_page(page_data)

        visual_groups, non_group_items = self._group_visual_items(items)

        # Render fixed/sensitive items first at source location.
        for group in visual_groups:
            page = doc[root_page_index]
            group_items = self._dedupe_visual_group_items(group.get("items") or [])
            if not group_items:
                continue
            group_bbox = group.get("bbox")
            group_left = max(left, group_bbox.x0) if isinstance(group_bbox, fitz.Rect) else left
            group_right = min(right, group_bbox.x1) if isinstance(group_bbox, fitz.Rect) else right
            group_top = max(top, group_bbox.y0) if isinstance(group_bbox, fitz.Rect) else top
            group_bottom = min(bottom, group_bbox.y1) if isinstance(group_bbox, fitz.Rect) else bottom
            if isinstance(group_bbox, fitz.Rect) and group_bbox.get_area() > 0:
                self._prepare_visual_group_background(page, group_items[0], group_bbox, group_items=group_items)
            local_forbidden = []
            for item in group_items:
                blue_rect = None
                used_slots = []
                zone_top, zone_bottom = self._anchored_zone_bounds(item, group_top, group_bottom)
                if isinstance(group_bbox, fitz.Rect) and group_bbox.get_area() > 0:
                    relative_y = max(0.0, item["bbox"].y0 - group_bbox.y0)
                    group_anchor_y = max(zone_top, min(zone_bottom - 8.0, group_top + relative_y))
                else:
                    group_anchor_y = max(top, min(item["bbox"].y0, bottom - 8.0))
                _, _, blue_rect, used_slots = self._render_block_slots(
                    page=page,
                    item=item,
                    anchor_y=group_anchor_y,
                    left=group_left,
                    right=group_right,
                    zone_top=zone_top,
                    zone_bottom=zone_bottom,
                    forbidden_rects=local_forbidden,
                )
                self._append_debug_rects(debug_store, page, blue_rect, used_slots)
                for used in used_slots or []:
                    if isinstance(used, fitz.Rect) and used.get_area() > 0:
                        local_forbidden.append(fitz.Rect(used))
                if isinstance(blue_rect, fitz.Rect) and blue_rect.get_area() > 0:
                    local_forbidden.append(fitz.Rect(blue_rect))

        grouped_ids = {id(it) for group in visual_groups for it in (group.get("items") or [])}
        early_fixed_items = [it for it in early_fixed_items if id(it) not in grouped_ids]
        late_fixed_items = [it for it in late_fixed_items if id(it) not in grouped_ids]
        body_items = [it for it in body_items if id(it) not in grouped_ids]

        for item in early_fixed_items:
            blue_rect = None
            used_slots = []
            toc_like_fixed = bool(
                toc_mode
                and item.get("role") in {"header", "section_heading"}
                and re.search(r"\b\d{1,3}\b", self._clean_text_for_render(item.get("text", "")))
            )
            if toc_like_fixed:
                used_bottom, blue_rect = self._render_toc_item(
                    page=page,
                    item=item,
                    anchor_y=max(top, min(item["bbox"].y0, bottom - 8.0)),
                    left=left,
                    right=right,
                    zone_top=max(top, item["bbox"].y0 - max(4.0, item.get("slot_h_pt", 8.0) * 0.6)),
                    zone_bottom=min(bottom, item["bbox"].y1 + max(12.0, item.get("slot_h_pt", 8.0) * 1.4)),
                    forbidden_rects=page_forbidden,
                )
            elif (
                self._item_requires_anchored_render(item, anchored_figure_page=anchored_figure_page)
            ):
                zone_top, zone_bottom = self._anchored_zone_bounds(item, top, bottom)
                _, _, blue_rect, used_slots = self._render_block_slots(
                    page=page,
                    item=item,
                    anchor_y=max(top, min(item["bbox"].y0, bottom - 8.0)),
                    left=left,
                    right=right,
                    zone_top=zone_top,
                    zone_bottom=zone_bottom,
                    forbidden_rects=page_forbidden,
                )
                self._append_debug_rects(debug_store, page, blue_rect, used_slots)
            else:
                self._render_fixed_item(page, item)
            bb = item.get("bbox")
            add_forbidden = bool(self._item_requires_anchored_render(item, anchored_figure_page=anchored_figure_page))
            if self._is_visual_group_item(item):
                add_forbidden = False
            if add_forbidden and isinstance(bb, fitz.Rect) and bb.get_area() > 0:
                page_forbidden.append(fitz.Rect(bb))
            if add_forbidden:
                for used in used_slots or []:
                    if isinstance(used, fitz.Rect) and used.get_area() > 0:
                        page_forbidden.append(fitz.Rect(used))
            if add_forbidden and isinstance(blue_rect, fitz.Rect) and blue_rect.get_area() > 0:
                page_forbidden.append(fitz.Rect(blue_rect))
        if toc_mode:
            toc_marker_items = []
            kept_body_items = []
            for item in body_items:
                txt = self._clean_text_for_render(item.get("text", ""))
                if re.fullmatch(r"\d{1,4}|[ivxlcdm]+", txt, flags=re.IGNORECASE):
                    toc_marker_items.append(item)
                else:
                    kept_body_items.append(item)
            for item in toc_marker_items:
                _, _, blue_rect, used_slots = self._render_block_slots(
                    page=page,
                    item=item,
                    anchor_y=max(top, min(item["bbox"].y0, bottom - 8.0)),
                    left=left,
                    right=right,
                    zone_top=max(top, item["bbox"].y0 - 2.0),
                    zone_bottom=min(bottom, item["bbox"].y1 + 4.0),
                    render=True,
                    forbidden_rects=page_forbidden,
                )
                self._append_debug_rects(debug_store, page, blue_rect, used_slots)
                for used in used_slots or []:
                    if isinstance(used, fitz.Rect) and used.get_area() > 0:
                        page_forbidden.append(fitz.Rect(used))
                if isinstance(blue_rect, fitz.Rect) and blue_rect.get_area() > 0:
                    page_forbidden.append(fitz.Rect(blue_rect))
            body_items = kept_body_items


        for idx, item in enumerate(body_items):
            # Adding pages can invalidate previously held page handles in PyMuPDF.
            # Always refresh the root/source page before rendering the next block.
            page = doc[root_page_index]
            descriptor_region_type = str(item.get("descriptor_region_type") or "").strip().lower()
            role = str(item.get("role") or "").strip().lower()
            editorial_text_locked = bool(
                role == "body"
                and descriptor_region_type == "text"
                and item.get("translated_block")
                and item.get("strict_bbox_mode")
                and layout_type == "double_column"
                and document_type in {"manual_guide", "book_page", "scientific_paper"}
            )
            if (not item.get("paragraph_flow_mode")) and item.get("preserve_linebreaks") and item.get("use_structured_source_lines"):
                src_lines = [
                    self._clean_text_for_render(x).strip()
                    for x in item.get("source_lines", [])
                    if str(x).strip()
                ]
                remaining = "\n".join(src_lines).strip() if src_lines else item.get("text", "").strip()
            else:
                remaining = item.get("text", "").strip()
            if not remaining:
                continue
            # In translated mode, do not hard-cap a paragraph to the next block's source Y.
            # Let the paragraph expand vertically (same X/line model), then push following
            # blocks down via forbidden zones when collisions happen.
            if item.get("paragraph_flow_mode"):
                # Strict block conformance requested by user:
                # translated paragraph must stay inside its source block bbox.
                next_y = min(bottom, float(item["bbox"].y1) + max(1.0, item.get("slot_h_pt", 8.0) * 0.1))
            elif editorial_text_locked:
                next_y = min(bottom, float(item["bbox"].y1) + 1.0)
            elif item.get("strict_bbox_mode") and not item.get("translated_block"):
                next_y = min(bottom, float(item["bbox"].y1) + 1.0)
            else:
                next_y = bottom
            if item.get("paragraph_flow_mode"):
                # Strict anchor: keep original vertical start for translated paragraph blocks.
                anchor_y = max(top, float(item["bbox"].y0))
            elif editorial_text_locked:
                anchor_y = max(top, float(item["bbox"].y0))
            elif item.get("strict_bbox_mode") and not item.get("translated_block"):
                anchor_y = max(top, float(item["bbox"].y0))
            else:
                anchor_y = self._shift_anchor_below_forbidden(
                    anchor_y=max(top, float(item["bbox"].y0)),
                    item=item,
                    left=left,
                    right=right,
                    zone_top=top,
                    zone_bottom=next_y,
                    forbidden_rects=page_forbidden,
                )
            # Dry-run to avoid placing a translated block over already rendered zones.
            # We simulate wrapping with the current anchor and push downward if needed.
            if (not editorial_text_locked) and (not item.get("paragraph_flow_mode")) and not (item.get("strict_bbox_mode") and not item.get("translated_block")):
                for _ in range(8):
                    _, _, probe_blue, _ = self._render_block_slots(
                        page=page,
                        item=item,
                        anchor_y=anchor_y,
                        left=left,
                        right=right,
                    zone_top=top,
                    zone_bottom=next_y,
                    override_text=remaining,
                    render=False,
                    forbidden_rects=page_forbidden,
                )
                    if not isinstance(probe_blue, fitz.Rect):
                        break
                    collisions = [fr for fr in page_forbidden if (probe_blue & fr).get_area() > 0]
                    if not collisions:
                        break
                    push_y = max(fr.y1 for fr in collisions) + max(1.5, item.get("slot_h_pt", 8.0) * 0.2)
                    if push_y <= anchor_y + 0.2:
                        break
                    anchor_y = min(next_y - 2.0, push_y)
                    if anchor_y >= next_y - 2.0:
                        break
            if toc_mode and item.get("role") in {"body", "section_heading", "title"}:
                item["preserve_linebreaks"] = True
                item["use_structured_source_lines"] = True
                used_bottom, blue_rect = self._render_toc_item(
                    page=page,
                    item=item,
                    anchor_y=anchor_y,
                    left=left,
                    right=right,
                    zone_top=top,
                    zone_bottom=next_y,
                    forbidden_rects=page_forbidden,
                )
                remaining = ""
                used_slots = []
            else:
                remaining, used_bottom, blue_rect, used_slots = self._render_block_slots(
                    page=page,
                    item=item,
                    anchor_y=anchor_y,
                    left=left,
                    right=right,
                    zone_top=top,
                    zone_bottom=next_y,
                    override_text=remaining,
                    render=True,
                    forbidden_rects=page_forbidden,
                )
            self._append_debug_rects(debug_store, page, blue_rect, used_slots)
            if not self._is_visual_group_item(item):
                for used in used_slots or []:
                    if isinstance(used, fitz.Rect) and used.get_area() > 0:
                        page_forbidden.append(fitz.Rect(used))
                if blue_rect is not None:
                    page_forbidden.append(fitz.Rect(blue_rect))

            # Overflow continuation pages: keep next source blocks anchored on page 1.
            if remaining and self._should_paginate_on_overflow() and not item.get("strict_bbox_mode"):
                flow_page = doc.new_page(width=base_page_width, height=base_page_height)
                f_top = flow_page.rect.y0 + 2.0
                f_bottom = flow_page.rect.y1 - 2.0
                f_left = flow_page.rect.x0 + 2.0
                f_right = flow_page.rect.x1 - 2.0
                flow_anchor = f_top
                while remaining:
                    prev_remaining = remaining
                    remaining, used_bottom, blue_rect, used_slots = self._render_block_slots(
                        page=flow_page,
                        item=item,
                        anchor_y=flow_anchor,
                        left=f_left,
                        right=f_right,
                        zone_top=f_top,
                        zone_bottom=f_bottom,
                        override_text=remaining,
                    )
                    self._append_debug_rects(debug_store, flow_page, blue_rect, used_slots)
                    flow_anchor = min(f_bottom, used_bottom + max(3.0, item.get("slot_h_pt", 8.0) * 0.35))
                    if not remaining:
                        break
                    if remaining == prev_remaining and not used_slots:
                        break
                    flow_page = doc.new_page(width=base_page_width, height=base_page_height)
                    f_top = flow_page.rect.y0 + 2.0
                    f_bottom = flow_page.rect.y1 - 2.0
                    f_left = flow_page.rect.x0 + 2.0
                    f_right = flow_page.rect.x1 - 2.0
                    flow_anchor = f_top

        for item in late_fixed_items:
            page = doc[root_page_index]
            blue_rect = None
            used_slots = []
            if self._item_requires_anchored_render(item, anchored_figure_page=anchored_figure_page):
                zone_top, zone_bottom = self._anchored_zone_bounds(item, top, bottom)
                _, _, blue_rect, used_slots = self._render_block_slots(
                    page=page,
                    item=item,
                    anchor_y=max(top, min(item["bbox"].y0, bottom - 8.0)),
                    left=left,
                    right=right,
                    zone_top=zone_top,
                    zone_bottom=zone_bottom,
                    forbidden_rects=page_forbidden,
                )
                self._append_debug_rects(debug_store, page, blue_rect, used_slots)
                if (not self._is_visual_group_item(item)) and isinstance(blue_rect, fitz.Rect) and blue_rect.get_area() > 0:
                    page_forbidden.append(fitz.Rect(blue_rect))
            else:
                self._render_fixed_item(page, item)

    def _render_toc_rows_v2(self, page, rows, tab_stops, zone_top, zone_bottom, left, right):
        """
        Render TOC rows from canonical layout.v2 structure:
          - label left-aligned with indent
          - page number right-aligned at tab_stops.page_num_right_x
          - dot leaders between label end and number start
        """
        col_left = float(tab_stops.get("column_left_x", left)) * self.pixel_to_point
        col_right = float(tab_stops.get("column_right_x", right)) * self.pixel_to_point
        page_num_right_x = self._resolve_toc_page_num_right_x(tab_stops, left, right, col_left=col_left, col_right=col_right)
        # The master background still leaves visible TOC page-number ghosts on
        # this document family. Clean only the page-number gutter and keep the
        # rest of the page intact.
        try:
            num_gutter = fitz.Rect(max(col_right - 18.0, page_num_right_x - 28.0), zone_top, right - 2.0, zone_bottom)
            page.draw_rect(num_gutter, color=None, fill=(1, 1, 1), overlay=True)
        except Exception:
            pass
        def get_native_fs(row):
            st = row.get("style") or {}
            source = str(row.get("source") or "native").strip().lower()
            try:
                raw_size = st.get("size")
                if isinstance(raw_size, (int, float)) and raw_size > 0:
                    if source == "native":
                        return float(raw_size)
                    return float(raw_size) * self.pixel_to_point
                raw_size = st.get("font_size_pt")
                if isinstance(raw_size, (int, float)) and raw_size > 0:
                    return float(raw_size)
            except Exception:
                pass
            role = str(row.get("role") or "").strip().lower()
            if role == "part_title":
                return 15.0
            if role == "chapter_title":
                return 12.5
            if role in {"section_heading", "subentry", "subentry_marker"}:
                return 10.0
            if role == "toc_title":
                return 13.0
            return 10.0

        def get_label_fs(row):
            native_fs = max(4.2, get_native_fs(row))
            bbox_fs = self._fontsize_from_bbox(row.get("label_bbox"))
            if isinstance(bbox_fs, (int, float)) and bbox_fs > 0.0:
                # A TOC label bbox can span multiple native source lines. Use
                # it only as a mild sanity cap, never as a way to inflate the
                # actual font size above the native style size.
                return min(float(bbox_fs), native_fs * 1.08)
            return native_fs

        def get_page_fs(row, fallback):
            bbox_fs = self._fontsize_from_bbox(row.get("page_bbox"))
            if isinstance(bbox_fs, (int, float)) and bbox_fs > 0.0:
                return float(bbox_fs)
            return max(4.2, float(fallback or 4.2))

        def wrap_row_label(label, width, fs, fontname, fontfile):
            clean = (label or "").strip()
            if not clean:
                return []
            words = clean.split()
            if not words:
                return []
            lines = []
            current = words[0]
            for word in words[1:]:
                candidate = f"{current} {word}"
                if self._measure_text_width(candidate, fs, fontname, fontfile) <= max(12.0, width):
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)
            return lines

        band_counts = {}
        for row in rows:
            band_id = row.get("source_band_id")
            if band_id is None:
                continue
            band_counts[band_id] = band_counts.get(band_id, 0) + 1
        dense_toc = len(rows) >= 40
        vertical_compactness = 1.0

        def layout_rows(scale):
            plans = []
            total_h = 0.0
            for row in rows:
                indent_level = int(row.get("indent_level", 0) or 0)
                row_role = str(row.get("role") or "").strip().lower()
                raw_indent_pt = float(row.get("indent_px", 0.0) or 0.0) * self.pixel_to_point
                label_bbox = row.get("label_bbox") or []
                page_bbox = row.get("page_bbox") or []
                band_id = row.get("source_band_id")
                band_size = int(band_counts.get(band_id, 1) or 1)
                # Compress source indentation to preserve hierarchy without starving label width.
                indent_px = min(max(0.0, raw_indent_pt * 0.38), 18.0 * max(0, indent_level))
                indent_px = max(indent_px, 10.0 * max(0, indent_level))
                marker = (row.get("marker") or "").strip()
                label = (row.get("translated_label") or row.get("label") or "").strip()
                label = self._format_toc_label_for_render(row_role, label)
                page_num = (row.get("page") or "").strip()
                if marker:
                    label = (marker + " " + label).strip()

                st = row.get("style") or {}
                page_style = row.get("page_style") or st
                resolved = self.font_resolver.resolve(st) if hasattr(self, "font_resolver") else {}
                fontfile = resolved.get("fontfile")
                builtin = resolved.get("builtin")
                fontname = self._resolve_page_fontname(page, fontfile, builtin) if hasattr(self, "_resolve_page_fontname") else "helv"
                rgb = self._resolve_text_color(st, {"style": st}) if hasattr(self, "_resolve_text_color") else (0, 0, 0)
                native_fs = max(4.2, get_label_fs(row))
                row_fs = max(4.2, native_fs * scale)
                page_fs = max(4.2, get_page_fs(row, row_fs) * scale)
                page_resolved = self.font_resolver.resolve(page_style) if hasattr(self, "font_resolver") else {}
                page_fontfile = page_resolved.get("fontfile")
                page_builtin = page_resolved.get("builtin")
                page_fontname = self._resolve_page_fontname(page, page_fontfile, page_builtin) if hasattr(self, "_resolve_page_fontname") else fontname
                page_rgb = self._resolve_text_color(page_style, {"style": page_style}) if hasattr(self, "_resolve_text_color") else rgb

                # TOC labels often wrap to 2 lines after translation. Keep a
                # looser baseline grid so extracted words do not overlap.
                compact = max(0.68 if dense_toc else 0.76, min(1.0, vertical_compactness))
                line_h = max(1.0, row_fs * (1.34 if row_role == "part_title" else 1.28) * compact)
                gap_y = max(0.45, row_fs * 0.16 * compact)
                pre_gap = 0.0
                post_gap = 0.0

                x_left = max(left + 6.0, min(col_left + indent_px, page_num_right_x - 90.0))
                if isinstance(label_bbox, (list, tuple)) and len(label_bbox) == 4:
                    try:
                        x_left = max(left + 6.0, min(float(label_bbox[0]) * self.pixel_to_point, page_num_right_x - 90.0))
                    except Exception:
                        pass
                if row_role == "toc_title":
                    if isinstance(label_bbox, (list, tuple)) and len(label_bbox) == 4:
                        try:
                            x_left = max(left + 6.0, float(label_bbox[0]) * self.pixel_to_point)
                        except Exception:
                            x_left = max(left + 6.0, col_left - 14.0)
                    else:
                        x_left = max(left + 6.0, col_left - 14.0)
                    pre_gap = 1.0
                    post_gap = 2.0
                elif row_role == "part_title":
                    if isinstance(page_bbox, (list, tuple)) and len(page_bbox) == 4:
                        try:
                            x_left = max(left + 74.0, (float(page_bbox[0]) * self.pixel_to_point) + 8.0)
                        except Exception:
                            x_left = max(left + 64.0, col_left + 30.0)
                    elif isinstance(label_bbox, (list, tuple)) and len(label_bbox) == 4:
                        try:
                            x_left = max(left + 74.0, float(label_bbox[0]) * self.pixel_to_point + 44.0)
                        except Exception:
                            x_left = max(left + 64.0, col_left + 30.0)
                    else:
                        x_left = max(left + 74.0, col_left + 40.0)
                    pre_gap = max(2.0, row_fs * 0.12)
                    post_gap = max(1.0, row_fs * 0.08)
                elif row_role == "chapter_title":
                    if isinstance(label_bbox, (list, tuple)) and len(label_bbox) == 4:
                        try:
                            x_left = max(left + 16.0, float(label_bbox[0]) * self.pixel_to_point)
                        except Exception:
                            x_left = max(left + 24.0, col_left + 24.0)
                    else:
                        x_left = max(left + 24.0, col_left + 24.0)
                    pre_gap = max(2.0, row_fs * 0.2)
                    post_gap = max(1.0, row_fs * 0.12)
                elif row_role == "section_heading":
                    if isinstance(label_bbox, (list, tuple)) and len(label_bbox) == 4:
                        try:
                            x_left = max(left + 16.0, float(label_bbox[0]) * self.pixel_to_point)
                        except Exception:
                            x_left = max(left + 36.0, col_left + 34.0)
                    else:
                        x_left = max(left + 36.0, col_left + 34.0)
                    pre_gap = max(1.0, row_fs * 0.08)
                    post_gap = max(0.6, row_fs * 0.05)
                elif row_role == "subentry":
                    if isinstance(label_bbox, (list, tuple)) and len(label_bbox) == 4:
                        try:
                            x_left = max(left + 16.0, float(label_bbox[0]) * self.pixel_to_point)
                        except Exception:
                            x_left = max(left + 56.0, col_left + 54.0 + min(12.0, indent_px * 0.18))
                    else:
                        x_left = max(left + 56.0, col_left + 54.0 + min(12.0, indent_px * 0.18))
                elif row_role == "subentry_marker":
                    if isinstance(label_bbox, (list, tuple)) and len(label_bbox) == 4:
                        try:
                            x_left = max(left + 16.0, float(label_bbox[0]) * self.pixel_to_point)
                        except Exception:
                            x_left = max(left + 64.0, col_left + 62.0 + min(14.0, indent_px * 0.16))
                    else:
                            x_left = max(left + 64.0, col_left + 62.0 + min(14.0, indent_px * 0.16))
                pre_gap *= compact
                post_gap *= compact
                native_label_right = None
                if isinstance(label_bbox, (list, tuple)) and len(label_bbox) == 4:
                    try:
                        native_label_right = float(label_bbox[2]) * self.pixel_to_point
                    except Exception:
                        native_label_right = None
                w_num = self._measure_text_width(page_num, page_fs, page_fontname, page_fontfile) if page_num else 0.0
                num_x = max(x_left + 18.0, page_num_right_x - w_num) if page_num else page_num_right_x
                use_native_num_slot = False
                if row_role != "part_title" and isinstance(page_bbox, (list, tuple)) and len(page_bbox) == 4 and page_num:
                    try:
                        native_num_right = min(right - 8.0, float(page_bbox[2]) * self.pixel_to_point)
                        min_native_slot = x_left + max(42.0, row_fs * 5.2)
                        if isinstance(native_label_right, (int, float)) and native_label_right > x_left:
                            min_native_slot = max(min_native_slot, native_label_right + max(8.0, row_fs * 0.8))
                        if native_num_right >= min_native_slot:
                            num_x = max(x_left + 18.0, native_num_right - w_num)
                            use_native_num_slot = True
                    except Exception:
                        pass
                if row_role == "chapter_title":
                    num_x = max(x_left + 32.0, page_num_right_x - w_num - 6.0)
                elif row_role == "section_heading":
                    num_x = max(x_left + 26.0, page_num_right_x - w_num - 3.0)
                if row_role == "part_title":
                    num_x = max(x_left + 48.0, page_num_right_x - w_num)
                    label_col_right = min(right - 10.0, num_x - max(12.0, row_fs * 1.2))
                elif use_native_num_slot and isinstance(page_bbox, (list, tuple)) and len(page_bbox) == 4:
                    try:
                        label_col_right = min(col_right, (float(page_bbox[0]) * self.pixel_to_point) - max(10.0, row_fs * 1.1))
                    except Exception:
                        label_col_right = min(col_right, num_x - max(10.0, row_fs * 1.1))
                else:
                    label_col_right = min(col_right, num_x - max(10.0, row_fs * 1.1))
                if isinstance(native_label_right, (int, float)) and native_label_right > x_left:
                    label_col_right = min(
                        max(label_col_right, native_label_right + max(2.0, row_fs * 0.2)),
                        num_x - max(8.0, row_fs * 0.85),
                    )
                label_max_w = max(18.0, label_col_right - x_left)
                source_line_capacity = 1
                label_bbox_fs = self._fontsize_from_bbox(label_bbox)
                if isinstance(label_bbox_fs, (int, float)) and label_bbox_fs and native_fs > 0:
                    source_line_capacity = max(1, int(round(float(label_bbox_fs) / max(1.0, native_fs))))
                if band_size > 1:
                    source_line_capacity = min(source_line_capacity, 1)
                min_ratio = 0.56 if dense_toc and row_role in {"section_heading", "subentry", "subentry_marker"} else (0.64 if row_role in {"section_heading", "subentry", "subentry_marker"} else 0.72)
                min_row_fs = max(4.0, native_fs * min_ratio)
                if row_role == "part_title" and label:
                    label_lines = [label]
                    while (
                        self._measure_text_width(label_lines[0], row_fs, fontname, fontfile) > label_max_w
                        and row_fs > min_row_fs
                    ):
                        row_fs -= 0.25
                    if self._measure_text_width(label_lines[0], row_fs, fontname, fontfile) > label_max_w:
                        label_lines = wrap_row_label(label, label_max_w, row_fs, fontname, fontfile)
                else:
                    label_lines = wrap_row_label(label, label_max_w, row_fs, fontname, fontfile) if label else []
                while label and len(label_lines) > source_line_capacity and row_fs > min_row_fs:
                    row_fs = max(min_row_fs, row_fs - 0.25)
                    label_lines = wrap_row_label(label, label_max_w, row_fs, fontname, fontfile) if label else []
                if not label_lines:
                    label_lines = [""]
                if len(label_lines) > max(2, source_line_capacity):
                    merged = " ".join(label_lines)
                    row_fs = max(min_row_fs, row_fs - 0.45)
                    row_line_h = max(1.0, row_fs * (1.34 if row_role == "part_title" else 1.28))
                    row_gap_y = max(0.8, row_fs * 0.16)
                    label_lines = wrap_row_label(merged, max(label_max_w, num_x - x_left - 4.0), row_fs, fontname, fontfile) or [merged]
                else:
                    row_line_h = line_h
                    row_gap_y = gap_y
                row_h = pre_gap + len(label_lines) * row_line_h + row_gap_y + post_gap
                source_y = float(row.get("y", 0.0) or 0.0) * self.pixel_to_point
                if row_role == "part_title" and isinstance(label_bbox, (list, tuple)) and len(label_bbox) == 4:
                    try:
                        source_y = max(zone_top, (float(label_bbox[1]) * self.pixel_to_point) - max(2.0, row_fs * 0.35))
                    except Exception:
                        pass
                plans.append(
                    {
                        "label_lines": label_lines,
                        "page_num": page_num,
                        "x_left": x_left,
                        "num_x": num_x,
                        "source_y": source_y,
                        "label_bbox": label_bbox,
                        "source_band_id": band_id,
                        "source_band_size": band_size,
                        "source_band_lane": int(row.get("source_band_lane", 0) or 0),
                        "fontname": fontname,
                        "fontfile": fontfile,
                        "rgb": rgb,
                        "style": st,
                        "fs": row_fs,
                        "page_fontname": page_fontname,
                        "page_fontfile": page_fontfile,
                        "page_rgb": page_rgb,
                        "page_style": page_style,
                        "page_fs": page_fs,
                        "line_h": row_line_h,
                        "gap_y": row_gap_y,
                        "pre_gap": pre_gap,
                        "post_gap": post_gap,
                        "chapter_marker": str(row.get("chapter_marker") or "").strip(),
                        "chapter_marker_bbox": row.get("chapter_marker_bbox") or row.get("page_bbox"),
                        "chapter_marker_style": row.get("chapter_marker_style") or row.get("page_style") or {},
                    }
                )
                total_h += row_h
            return plans, total_h

        def estimate_used_height(plans):
            if not plans:
                return 0.0, zone_top
            def transition_gap(prev_role, current_role, fs):
                cur = str(current_role or "").strip().lower()
                return 0.0
            est_y = max(zone_top, float(rows[0].get("y", zone_top)) * self.pixel_to_point)
            est_band_id = None
            est_band_source_y = None
            est_band_y = est_y
            est_band_bottom = est_y
            start_y = est_y
            prev_row_role = ""
            for row, plan in zip(rows, plans):
                row_h = len(plan["label_lines"]) * plan["line_h"] + plan["gap_y"]
                source_y = max(zone_top, plan.get("source_y", zone_top))
                row_role = str(row.get("role") or "").strip().lower()
                band_id = plan.get("source_band_id")
                band_size = int(plan.get("source_band_size", 1) or 1)
                if row_role == "part_title":
                    est_y = source_y
                    est_band_id = band_id
                    est_band_source_y = source_y
                    est_band_y = est_y
                    est_band_bottom = est_y + row_h + plan.get("post_gap", 0.0)
                elif est_band_id is not None and band_id == est_band_id:
                    prior_band_multiline = est_band_bottom > (est_band_y + max(1.5, plan["fs"] * 1.12))
                    if band_size > 1 and (len(plan["label_lines"]) > 1 or prior_band_multiline):
                        est_y = max(source_y, est_band_bottom + plan.get("pre_gap", 0.0))
                    else:
                        est_y = est_band_y
                elif est_band_source_y is not None and abs(source_y - est_band_source_y) <= max(1.5, plan["fs"] * 0.18):
                    est_band_id = band_id
                    est_band_y = source_y
                    est_y = est_band_y
                else:
                    est_y = max(source_y, est_band_bottom + plan.get("pre_gap", 0.0) + transition_gap(prev_row_role, row_role, plan["fs"]))
                    est_band_id = band_id
                    est_band_source_y = source_y
                    est_band_y = est_y
                est_band_bottom = max(
                    est_band_bottom,
                    est_y + len(plan["label_lines"]) * plan["line_h"] + plan["gap_y"] + plan.get("post_gap", 0.0),
                )
                prev_row_role = row_role
            return max(0.0, est_band_bottom - start_y), start_y

        available_h = max(20.0, zone_bottom - zone_top)
        scale = 1.0
        plans, _ = layout_rows(scale)
        used_h, y = estimate_used_height(plans)
        min_scale = 0.64 if dense_toc else 0.76
        while used_h > available_h and scale > min_scale:
            scale = max(min_scale, scale - 0.04)
            plans, _ = layout_rows(scale)
            used_h, y = estimate_used_height(plans)
        if used_h > available_h:
            vertical_compactness = max(0.74 if dense_toc else 0.9, min(1.0, available_h / max(used_h, 1.0)))
            plans, _ = layout_rows(scale)
            used_h, y = estimate_used_height(plans)
        while dense_toc and used_h > available_h and scale > 0.58:
            scale = max(0.58, scale - 0.02)
            plans, _ = layout_rows(scale)
            used_h, y = estimate_used_height(plans)
        if used_h < available_h:
            y = max(zone_top, min(y, zone_top + (available_h - used_h) * 0.25))

        rendered_chapter_markers = set()
        current_band_id = None
        band_source_y = None
        band_y = y
        band_bottom = y
        prev_row_role = ""
        for row, plan in zip(rows, plans):
            row_h = len(plan["label_lines"]) * plan["line_h"] + plan["gap_y"]
            source_y = max(zone_top, plan.get("source_y", zone_top))
            row_role = str(row.get("role") or "").strip().lower()
            band_id = plan.get("source_band_id")
            extra_gap = 0.0
            if row_role == "part_title":
                y = source_y
                current_band_id = band_id
                band_source_y = source_y
                band_y = y
                band_bottom = y + row_h + plan.get("post_gap", 0.0)
            elif current_band_id is not None and band_id == current_band_id:
                prior_band_multiline = band_bottom > (band_y + max(1.5, plan["fs"] * 1.12))
                if int(plan.get("source_band_size", 1) or 1) > 1 and (len(plan["label_lines"]) > 1 or prior_band_multiline):
                    y = max(source_y, band_bottom + plan.get("pre_gap", 0.0))
                else:
                    y = band_y
            elif band_source_y is not None and abs(source_y - band_source_y) <= max(1.5, plan["fs"] * 0.18):
                current_band_id = band_id
                band_y = source_y
                y = band_y
            else:
                y = max(source_y, band_bottom + plan.get("pre_gap", 0.0) + extra_gap)
                current_band_id = band_id
                band_source_y = source_y
                band_y = y
            if y + row_h > zone_bottom:
                overflow = (y + row_h) - zone_bottom
                if overflow <= max(3.0, plan["line_h"] * 0.9):
                    y = max(zone_top, y - overflow)
                if y + row_h > zone_bottom:
                    break
            chapter_number = str(row.get("chapter_number") or "").strip()
            try:
                wipe_left = max(left + 22.0, col_left - 8.0)
                if row_role == "toc_title":
                    wipe_left = max(left + 34.0, col_left - 6.0)
                elif row_role == "part_title":
                    wipe_left = max(left + 72.0, plan["x_left"] - 4.0)
                wipe_rect = fitz.Rect(wipe_left, max(zone_top, y - 1.0), min(right - 2.0, page_num_right_x + 20.0), min(zone_bottom, y + row_h + 1.0))
                page.draw_rect(wipe_rect, color=None, fill=(1, 1, 1), overlay=True)
            except Exception:
                pass
            chapter_marker = str(plan.get("chapter_marker") or "").strip()
            pending_chapter_title_marker = None
            if row_role == "part_title" and chapter_marker and chapter_marker not in rendered_chapter_markers:
                try:
                    marker_bbox = plan.get("chapter_marker_bbox") or []
                    marker_style = plan.get("chapter_marker_style") or {}
                    marker_fs = self._fontsize_from_bbox(marker_bbox, fallback=max(18.0, plan["fs"] * 2.2))
                    marker_resolved = self.font_resolver.resolve(marker_style) if hasattr(self, "font_resolver") else {}
                    marker_fontfile = marker_resolved.get("fontfile")
                    marker_builtin = marker_resolved.get("builtin")
                    marker_fontname = self._resolve_page_fontname(page, marker_fontfile, marker_builtin) if hasattr(self, "_resolve_page_fontname") else "helv"
                    marker_rgb = self._resolve_text_color(marker_style, {"style": marker_style}) if hasattr(self, "_resolve_text_color") else (0.77, 0.62, 0.27)
                    if isinstance(marker_bbox, (list, tuple)) and len(marker_bbox) == 4:
                        marker_rect = fitz.Rect([float(v) * self.pixel_to_point for v in marker_bbox])
                    else:
                        marker_rect = fitz.Rect(max(left + 4.0, plan["x_left"] - 42.0), max(zone_top, y - 2.0), max(left + 8.0, plan["x_left"] - 6.0), min(zone_bottom, y + row_h + 2.0))
                    marker_wipe = fitz.Rect(marker_rect)
                    page.draw_rect(marker_wipe, color=None, fill=(1, 1, 1), overlay=True)
                    marker_x = marker_rect.x0
                    marker_y = marker_rect.y0 + marker_fs * self._baseline_ratio(marker_style, marker_fs)
                    self._safe_insert_text_dedup(
                        page,
                        (marker_x, marker_y),
                        chapter_marker,
                        marker_fs,
                        marker_fontname,
                        marker_rgb,
                    )
                    rendered_chapter_markers.add(chapter_marker)
                except Exception:
                    pass
            elif row_role == "chapter_title" and chapter_number and chapter_number not in rendered_chapter_markers:
                try:
                    label_bbox = plan.get("label_bbox") or []
                    if isinstance(label_bbox, (list, tuple)) and len(label_bbox) == 4:
                        try:
                            label_rect = fitz.Rect([float(v) * self.pixel_to_point for v in label_bbox])
                        except Exception:
                            label_rect = fitz.Rect(max(left + 4.0, plan["x_left"] - 42.0), max(zone_top, y - 2.0), max(left + 8.0, plan["x_left"] - 6.0), min(zone_bottom, y + row_h + 2.0))
                    else:
                        label_rect = fitz.Rect(max(left + 4.0, plan["x_left"] - 42.0), max(zone_top, y - 2.0), max(left + 8.0, plan["x_left"] - 6.0), min(zone_bottom, y + row_h + 2.0))
                    marker_wipe = fitz.Rect(max(left + 4.0, plan["x_left"] - 42.0), max(zone_top, label_rect.y0 - 3.0), max(left + 8.0, plan["x_left"] - 6.0), min(zone_bottom, label_rect.y1 + 3.0))
                    page.draw_rect(marker_wipe, color=None, fill=(1, 1, 1), overlay=True)
                    pending_chapter_title_marker = {
                        "x": max(left + 4.0, plan["x_left"] - 34.0),
                        "y": label_rect.y0 + max(18.0, plan["fs"] * 2.2) * self._baseline_ratio(plan.get("style") or {}, max(18.0, plan["fs"] * 2.2)),
                        "fs": max(18.0, plan["fs"] * 2.2),
                    }
                except Exception:
                    pending_chapter_title_marker = {
                        "x": max(left + 4.0, plan["x_left"] - 34.0),
                        "y": y + max(18.0, plan["fs"] * 2.2) * self._baseline_ratio(plan.get("style") or {}, max(18.0, plan["fs"] * 2.2)),
                        "fs": max(18.0, plan["fs"] * 2.2),
                    }
            last_line = ""
            for idx, line in enumerate(plan["label_lines"]):
                baseline_y = y + idx * plan["line_h"] + plan["fs"] * self._baseline_ratio(plan.get("style") or {}, plan["fs"])
                if line:
                    self._safe_insert_text_dedup(
                        page,
                        (plan["x_left"], baseline_y),
                        line,
                        plan["fs"],
                        plan["fontname"],
                        plan["rgb"],
                    )
                    last_line = line

            if plan["page_num"] and last_line:
                try:
                    line_w = self._measure_text_width(last_line, plan["fs"], plan["fontname"], plan["fontfile"])
                    lead_start = plan["x_left"] + line_w + max(4.0, plan["fs"] * 0.45)
                    lead_end = plan["num_x"] - max(5.0, plan["page_fs"] * 0.55)
                    baseline_y = y + max(0, len(plan["label_lines"]) - 1) * plan["line_h"] + plan["page_fs"] * 0.72
                    if lead_end - lead_start >= 18.0:
                        step = max(4.8, plan["page_fs"] * 0.7)
                        x = lead_start
                        while x <= lead_end:
                            page.draw_circle(
                                fitz.Point(x, baseline_y),
                                radius=max(0.35, plan["page_fs"] * 0.05),
                                color=None,
                                fill=(0.78, 0.63, 0.29),
                            )
                            x += step
                except Exception:
                    pass

            if plan["page_num"]:
                baseline_y = y + max(0, len(plan["label_lines"]) - 1) * plan["line_h"] + plan["page_fs"] * self._baseline_ratio(plan.get("page_style") or {}, plan["page_fs"])
                self._safe_insert_text_dedup(
                    page,
                    (plan["num_x"], baseline_y),
                    plan["page_num"],
                    plan["page_fs"],
                    plan["page_fontname"],
                    plan["page_rgb"],
                )
            if pending_chapter_title_marker is not None:
                self._safe_insert_text_dedup(
                    page,
                    (pending_chapter_title_marker["x"], pending_chapter_title_marker["y"]),
                    chapter_number,
                    pending_chapter_title_marker["fs"],
                    plan["fontname"],
                    plan["rgb"],
                )
                rendered_chapter_markers.add(chapter_number)

            band_bottom = max(band_bottom, y + len(plan["label_lines"]) * plan["line_h"] + plan["gap_y"] + plan.get("post_gap", 0.0))
            prev_row_role = row_role

    def _resolve_toc_page_num_right_x(self, tab_stops, left, right, col_left=None, col_right=None):
        page_num_right_x = float(tab_stops.get("page_num_right_x", right)) * self.pixel_to_point
        if col_left is None:
            col_left = float(tab_stops.get("column_left_x", left)) * self.pixel_to_point
        if col_right is None:
            col_right = float(tab_stops.get("column_right_x", right)) * self.pixel_to_point

        # Some TOC pages detect page-number tab stops too far inside the text
        # column, which over-constrains translated labels and causes heavy
        # wrapping compared with the source layout. When the detected stop is
        # implausibly left of the content right edge, fall back to a right-gutter
        # anchor near the actual page edge.
        if page_num_right_x < (col_right - 18.0):
            page_num_right_x = max(col_right + 18.0, right - 18.0)
        return min(right - 12.0, max(col_left + 120.0, page_num_right_x))

    def _format_toc_label_for_render(self, row_role, label):
        text = self._clean_text_for_render(label or "")
        role = str(row_role or "").strip().lower()
        if role != "part_title":
            return text
        # The decorative "PART 2" block usually remains visually intact on the
        # master background. Render only the translated thematic title to the
        # right so the result stays close to the source composition.
        text = re.sub(r"^\s*(?:partie|part)\s+\d+\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*(?:partie|part)\s+", "", text, flags=re.IGNORECASE)
        return text.upper()

    def _merge_translated_body_items(self, body_items):
        if not body_items:
            return []
        merged = []
        for it in sorted(body_items, key=lambda x: (x["bbox"].y0, x["bbox"].x0)):
            if not merged:
                merged.append(dict(it))
                continue
            prev = merged[-1]
            r0 = prev["bbox"]
            r1 = it["bbox"]
            if not isinstance(r0, fitz.Rect) or not isinstance(r1, fitz.Rect):
                merged.append(dict(it))
                continue
            same_role = (prev.get("role") == it.get("role") == "body")
            y_gap = r1.y0 - r0.y1
            x_overlap = max(0.0, min(r0.x1, r1.x1) - max(r0.x0, r1.x0)) / max(1.0, min(r0.width, r1.width))
            if same_role and y_gap <= max(10.0, prev.get("slot_h_pt", 10.0) * 1.2) and x_overlap >= 0.45:
                # Merge paragraph fragments to improve professional continuity.
                prev["text"] = self._clean_text_for_render(f"{prev.get('text','').rstrip()} {it.get('text','').lstrip()}")
                prev["bbox"] = r0 | r1
                prev_slots = [fitz.Rect(s) for s in prev.get("slots", [])]
                cur_slots = [fitz.Rect(s) for s in it.get("slots", [])]
                prev["slots"] = sorted(prev_slots + cur_slots, key=lambda r: (r.y0, r.x0))
                prev["source_lines"] = list(prev.get("source_lines", [])) + list(it.get("source_lines", []))
                prev["preserve_linebreaks"] = True
                continue
            merged.append(dict(it))
        return merged

    def _shift_anchor_below_forbidden(self, anchor_y, item, left, right, zone_top, zone_bottom, forbidden_rects):
        y = max(zone_top, anchor_y)
        if not forbidden_rects:
            return y
        h = max(item.get("slot_h_pt", 8.0), min(item["bbox"].height, zone_bottom - zone_top))
        x0 = max(left, item["bbox"].x0)
        x1 = min(right, item["bbox"].x1)
        if x1 <= x0:
            x0, x1 = left, right
        for _ in range(128):
            probe = fitz.Rect(x0, y, x1, min(zone_bottom, y + h))
            collisions = []
            for fr in forbidden_rects:
                if (probe & fr).get_area() > 0:
                    collisions.append(fr)
            if not collisions:
                return y
            y = max(y, max(fr.y1 for fr in collisions) + 1.0)
            if y >= zone_bottom - 2.0:
                return max(zone_top, min(anchor_y, zone_bottom - 2.0))
        return max(zone_top, min(y, zone_bottom - 2.0))

    def _render_fixed_item(self, page, item):
        text = self._clean_text_for_render(item.get("text", "")).strip()
        if not text:
            return
        style = self._normalized_style_for_item(item)
        source = item.get("source", "ocr")
        bbox = item.get("bbox")
        if not isinstance(bbox, fitz.Rect):
            return
        _, fontfile, builtin, fontname = self._resolve_style_font(page, style, text=text)
        fs = self._normalized_fontsize_for_item(item, style, max(1.0, bbox.height), source)
        region_type = str(item.get("descriptor_region_type") or "").strip().lower()
        role = str(item.get("role") or "").strip().lower()
        typographic_class = str(item.get("descriptor_typographic_class") or "").strip().lower()
        if role == "header":
            fs = max(fs, 8.8)
        elif role in {"title", "section_heading"} or region_type in {"annotation_band", "caption_band", "header_band"}:
            fs = max(fs, 8.2)
        if typographic_class in {"diagram_label", "chart_axis_label", "chart_tick_label", "chart_legend_label"}:
            fs = max(fs, 7.4)
        try:
            c = style.get("color", "#000000").lstrip("#")
            rgb = tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except Exception:
            rgb = (0, 0, 0)
        max_w = max(8.0, bbox.width)
        line_w = self._measure_text_width(text, fs, fontname, fontfile)
        if line_w > max_w and not self.fixed_font_size:
            min_fs = self._min_fontsize_for_item(item, fs, strict=True)
            while line_w > max_w and fs > min_fs + 1e-6:
                fs = max(min_fs, fs - 0.2)
                line_w = self._measure_text_width(text, fs, fontname, fontfile)
        align = self._normalize_alignment(item.get("alignment", "left"))
        draw_x = bbox.x0
        if align == "center" and line_w < bbox.width:
            draw_x = bbox.x0 + (bbox.width - line_w) / 2.0
        elif align == "right" and line_w < bbox.width:
            draw_x = bbox.x1 - line_w
        br = self._baseline_ratio(style, fs)
        baseline = bbox.y0 + min(max(1.0, fs * br), max(1.0, bbox.height - 0.6))
        if self._should_whiteout_before_render(item):
            self._whiteout_rect(page, item.get("whiteout_bbox", bbox))
        elif self._should_restore_background_before_render(item):
            self._restore_background_rect(page, item, bbox, kind="fixed_visual_text_bg_restore")
        target_rect = fitz.Rect(
            draw_x,
            baseline - max(fs * 0.95, bbox.height * 0.9),
            draw_x + line_w,
            baseline + max(1.0, fs * 0.2),
        )
        self._safe_insert_text_dedup(page, (draw_x, baseline), text, fs, fontname, rgb)

    def _should_whiteout_before_render(self, item):
        if not isinstance(item, dict):
            return False
        if str(item.get("source") or "").strip().lower() != "native":
            return False
        role = str(item.get("role") or "").strip().lower()
        if role == "list_marker":
            return True
        if self._should_whiteout_per_line(item):
            return False
        if self._should_restore_background_before_render(item):
            return False
        source_text = self._clean_text_for_render(item.get("source_text", "")).strip()
        rendered_text = self._clean_text_for_render(item.get("text", "")).strip()
        if not source_text or not rendered_text:
            return False
        if source_text == rendered_text:
            return False
        region_type = str(item.get("descriptor_region_type") or "").strip().lower()
        if role in {"title", "section_heading", "header", "figure_caption", "diagram_label", "diagram_text_label"}:
            return True
        if region_type in {"annotation_band", "caption_band", "header_band", "table_cell", "table_row"}:
            return True
        return bool(item.get("translated_block"))

    def _should_whiteout_per_line(self, item):
        if not isinstance(item, dict):
            return False
        if str(item.get("source") or "").strip().lower() != "native":
            return False
        if self._should_restore_background_before_render(item):
            return False
        role = str(item.get("role") or "").strip().lower()
        structural_role = str(item.get("descriptor_structural_role") or "").strip().lower()
        band_role = str(item.get("descriptor_band_role") or "").strip().lower()
        source_text = self._clean_text_for_render(item.get("source_text", "")).strip()
        rendered_text = self._clean_text_for_render(item.get("text", "")).strip()
        if not source_text or not rendered_text:
            return False
        if source_text == rendered_text:
            return bool(
                item.get("translated_block")
                and role == "body"
                and band_role in {"table_band", "header_band"}
            )
        if role == "body" and band_role == "text_band":
            return True
        if structural_role in {"opening_paragraph", "body_paragraph", "continuation_paragraph"}:
            return True
        return False

    def _should_restore_background_before_render(self, item):
        if not isinstance(item, dict):
            return False
        if str(item.get("source") or "").strip().lower() != "native":
            return False
        source_text = self._clean_text_for_render(item.get("source_text", "")).strip()
        rendered_text = self._clean_text_for_render(item.get("text", "")).strip()
        if not source_text or not rendered_text or source_text == rendered_text:
            return False
        visual_text = item.get("descriptor_visual_text") or {}
        text_embedding_mode = str(visual_text.get("text_embedding_mode") or "").strip().lower()
        replacement_strategy = str(visual_text.get("background_replacement_strategy") or "").strip().lower()
        band_role = str(item.get("descriptor_band_role") or "").strip().lower()
        structural_role = str(item.get("descriptor_structural_role") or "").strip().lower()
        attachment_target_id = str(item.get("descriptor_attachment_target_id") or "").strip().lower()
        region_type = str(item.get("descriptor_region_type") or "").strip().lower()
        ai_region_type = str(item.get("descriptor_ai_region_type") or "").strip().lower()
        if replacement_strategy in {"crop_restore", "text_erase_then_overlay"}:
            return True
        if text_embedding_mode == "embedded_in_visual":
            return True
        if band_role in {"annotation_band", "legend_band", "axis_band", "table_band", "sidebar"}:
            return True
        if structural_role in {"diagram_label", "chart_axis_label", "chart_tick_label", "chart_legend_label"}:
            return True
        if attachment_target_id in {"illustration_main", "chart_main"}:
            return True
        if region_type in {
            "annotation_band",
            "chart_area",
            "chart_plot_area",
            "chart_x_axis",
            "chart_y_axis",
            "chart_x_ticks",
            "chart_y_ticks",
            "chart_legend",
            "table",
            "table_row",
            "table_cell",
            "sidebar",
        }:
            return True
        if ai_region_type in {"image", "chart", "figure", "table", "sidebar"}:
            return True
        if self._item_has_nonwhite_background(item):
            return True
        return False

    def _parse_hex_rgb(self, color):
        if not isinstance(color, str):
            return None
        s = color.strip()
        if not s:
            return None
        if s.startswith("#"):
            s = s[1:]
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) != 6 or re.search(r"[^0-9a-fA-F]", s):
            return None
        try:
            return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
        except Exception:
            return None

    def _item_has_nonwhite_background(self, item):
        if not isinstance(item, dict):
            return False
        style = item.get("style") or {}
        bg_rgb = self._parse_hex_rgb(style.get("highlight_color") or style.get("background_color") or "")
        if bg_rgb is not None:
            avg = sum(bg_rgb) / 3.0
            if avg < 247.0 or max(bg_rgb) - min(bg_rgb) >= 5:
                return True
        page_data = item.get("page_data")
        bbox = item.get("bbox")
        if not isinstance(page_data, dict) or not isinstance(bbox, fitz.Rect) or bbox.get_area() <= 0:
            return False
        background_path = page_data.get("background_path") or page_data.get("source_image_path")
        if not background_path or not os.path.exists(background_path):
            return False
        try:
            key = (
                os.path.realpath(background_path),
                int(round(bbox.x0 / self.pixel_to_point)),
                int(round(bbox.y0 / self.pixel_to_point)),
                int(round(bbox.x1 / self.pixel_to_point)),
                int(round(bbox.y1 / self.pixel_to_point)),
            )
            cached = self._local_background_profile_cache.get(key)
            if cached is not None:
                return bool(cached)
            with Image.open(background_path).convert("RGB") as im:
                x0 = max(0, min(im.width, key[1]))
                y0 = max(0, min(im.height, key[2]))
                x1 = max(x0 + 1, min(im.width, key[3]))
                y1 = max(y0 + 1, min(im.height, key[4]))
                crop = im.crop((x0, y0, x1, y1)).resize((18, 18))
                stat = ImageStat.Stat(crop)
                mean = stat.mean[:3]
                stddev = stat.stddev[:3]
                avg = sum(mean) / max(1.0, len(mean))
                contrast = max(mean) - min(mean)
                nonwhite = avg < 247.0 or contrast >= 4.5 or max(stddev) >= 3.0
            self._local_background_profile_cache[key] = bool(nonwhite)
            return bool(nonwhite)
        except Exception:
            return False

    def _prefer_text_erased_overlay(self, item):
        if not isinstance(item, dict):
            return False
        visual_text = item.get("descriptor_visual_text") or {}
        if str(visual_text.get("background_replacement_strategy") or "").strip().lower() == "text_erase_then_overlay":
            return True
        text_embedding_mode = str(visual_text.get("text_embedding_mode") or "").strip().lower()
        band_role = str(item.get("descriptor_band_role") or "").strip().lower()
        region_type = str(item.get("descriptor_region_type") or "").strip().lower()
        ai_region_type = str(item.get("descriptor_ai_region_type") or "").strip().lower()
        if text_embedding_mode == "embedded_in_visual":
            return True
        if band_role in {"annotation_band", "legend_band", "axis_band"}:
            return True
        if region_type in {"annotation_band", "chart_area", "chart_plot_area", "chart_legend"}:
            return True
        if ai_region_type in {"image", "chart", "figure"}:
            return True
        return False

    def _estimate_patch_fill_rgb(self, crop, inner_rect):
        if crop is None:
            return (255, 255, 255)
        x0, y0, x1, y1 = [int(v) for v in inner_rect]
        w, h = crop.size
        samples = []
        pad = 2
        rx0 = max(0, x0 - pad)
        ry0 = max(0, y0 - pad)
        rx1 = min(w, x1 + pad)
        ry1 = min(h, y1 + pad)
        px = crop.load()
        for yy in range(ry0, ry1):
            for xx in range(rx0, rx1):
                if x0 <= xx < x1 and y0 <= yy < y1:
                    continue
                samples.append(px[xx, yy])
        if not samples:
            samples = list(crop.getdata())
        if not samples:
            return (255, 255, 255)
        channels = list(zip(*samples))
        return tuple(int(sum(ch) / max(1, len(ch))) for ch in channels[:3])

    def _estimate_local_background_image(self, crop, inner_rect):
        if crop is None:
            return None
        x0, y0, x1, y1 = [int(v) for v in inner_rect]
        inner_w = max(1, x1 - x0)
        inner_h = max(1, y1 - y0)
        candidates = []
        if y0 > 0:
            candidates.append(crop.crop((x0, 0, x1, y0)))
        if y1 < crop.height:
            candidates.append(crop.crop((x0, y1, x1, crop.height)))
        if x0 > 0:
            candidates.append(crop.crop((0, y0, x0, y1)))
        if x1 < crop.width:
            candidates.append(crop.crop((x1, y0, crop.width, y1)))
        patches = []
        for cand in candidates:
            if cand.width <= 0 or cand.height <= 0:
                continue
            patches.append(cand.resize((inner_w, inner_h)))
        if not patches:
            fill_rgb = self._estimate_patch_fill_rgb(crop, inner_rect)
            patch = Image.new("RGB", (inner_w, inner_h), fill_rgb)
            return patch
        patch = patches[0]
        for extra in patches[1:]:
            patch = Image.blend(patch, extra, 0.5)
        return patch

    def _save_text_erased_overlay(self, page_data, bbox, kind="visual_text_erase"):
        source_img = page_data.get("source_image_path")
        if not source_img or not os.path.exists(source_img):
            return None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        if getattr(self, "background_inpainter", None):
            out_dir = os.path.dirname(page_data.get("background_path", "")) or "ocr_results"
            ov = self.background_inpainter.save_inpaint_overlay(
                source_image_path=source_img,
                crop_bbox=bbox,
                mask_rects=[bbox],
                out_dir=out_dir,
                kind=kind,
            )
            if ov:
                return ov
        try:
            x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
            pad = max(2, int(self.dynamic_overlay_pad_px) + 2)
            with Image.open(source_img).convert("RGB") as im:
                crop_x0 = max(0, x0 - pad)
                crop_y0 = max(0, y0 - pad)
                crop_x1 = min(im.width, x1 + pad)
                crop_y1 = min(im.height, y1 + pad)
                if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
                    return None
                crop = im.crop((crop_x0, crop_y0, crop_x1, crop_y1))
                inner = (
                    max(0, x0 - crop_x0),
                    max(0, y0 - crop_y0),
                    min(crop.width, x1 - crop_x0),
                    min(crop.height, y1 - crop_y0),
                )
                patch = self._estimate_local_background_image(crop, inner)
                if patch is None:
                    fill_rgb = self._estimate_patch_fill_rgb(crop, inner)
                    patch = Image.new("RGB", (max(1, inner[2] - inner[0]), max(1, inner[3] - inner[1])), fill_rgb)
                crop.paste(patch, (inner[0], inner[1]))
                out_dir = os.path.dirname(page_data.get("background_path", "")) or "ocr_results"
                os.makedirs(out_dir, exist_ok=True)
                out_name = f"dynamic_overlay_{kind}_{uuid.uuid4().hex[:12]}.png"
                out_path = os.path.join(out_dir, out_name)
                crop.save(out_path)
                return {"path": out_path, "bbox": [crop_x0, crop_y0, crop_x1, crop_y1], "kind": kind}
        except Exception:
            return None

    def _insert_restored_overlay(self, page, ov):
        if not isinstance(ov, dict):
            return False
        path = ov.get("path")
        bbox = ov.get("bbox")
        if not path or not os.path.exists(path):
            return False
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        try:
            x0, y0, x1, y1 = [float(v) * self.pixel_to_point for v in bbox]
            restore_rect = fitz.Rect(x0, y0, x1, y1)
            restore_rect = self._clamp_rect_to_page(restore_rect, page.rect)
            if restore_rect.get_area() <= 0:
                return False
            page_idx = int(getattr(page, "number", 0) or 0)
            prior_rects = self._restored_background_rects.setdefault(page_idx, [])
            for prev in prior_rects:
                inter = restore_rect & prev
                if inter.get_area() <= 0:
                    continue
                overlap_ratio = inter.get_area() / max(1.0, min(restore_rect.get_area(), prev.get_area()))
                if overlap_ratio >= 0.25:
                    return False
            page.insert_image(restore_rect, filename=path, overlay=True, keep_proportion=False)
            prior_rects.append(fitz.Rect(restore_rect))
            return True
        except Exception:
            return False

    def _expanded_text_mask_rect(self, item, rect):
        if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
            return rect
        role = str((item or {}).get("role") or "").strip().lower()
        structural_role = str((item or {}).get("descriptor_structural_role") or "").strip().lower()
        base_h = max(6.0, rect.height)
        pad_x = max(1.2, min(8.0, base_h * 0.28))
        pad_y = max(0.9, min(5.0, base_h * 0.20))
        if role in {"header", "figure_caption", "title"} or structural_role in {"running_header", "figure_caption", "diagram_label"}:
            pad_x += 0.8
            pad_y += 0.5
        return fitz.Rect(rect.x0 - pad_x, rect.y0 - pad_y, rect.x1 + pad_x, rect.y1 + pad_y)

    def _restore_inpainted_group_background(self, page, item, group_rect, group_items=None, kind="background_restore"):
        inpainter = getattr(self, "background_inpainter", None)
        if inpainter is None or not getattr(inpainter, "enabled", False):
            return False
        if not isinstance(item, dict):
            return False
        page_data = item.get("page_data")
        if not isinstance(page_data, dict):
            return False
        source_img = page_data.get("source_image_path")
        if not source_img or not os.path.exists(source_img):
            return False
        crop_rect = self._clamp_rect_to_page(fitz.Rect(group_rect), page.rect)
        if crop_rect.get_area() <= 0:
            return False
        mask_rects = []
        for member in (group_items or [item]):
            bb = member.get("bbox")
            if not isinstance(bb, fitz.Rect) or bb.get_area() <= 0:
                continue
            mask_rect = self._expanded_text_mask_rect(member, fitz.Rect(bb))
            mask_rect = self._clamp_rect_to_page(mask_rect, crop_rect)
            if mask_rect.get_area() <= 0:
                continue
            mask_rects.append(
                [
                    mask_rect.x0 / self.pixel_to_point,
                    mask_rect.y0 / self.pixel_to_point,
                    mask_rect.x1 / self.pixel_to_point,
                    mask_rect.y1 / self.pixel_to_point,
                ]
            )
        if not mask_rects:
            return False
        crop_bbox = [
            crop_rect.x0 / self.pixel_to_point,
            crop_rect.y0 / self.pixel_to_point,
            crop_rect.x1 / self.pixel_to_point,
            crop_rect.y1 / self.pixel_to_point,
        ]
        out_dir = os.path.dirname(page_data.get("background_path", "")) or "ocr_results"
        ov = inpainter.save_inpaint_overlay(
            source_image_path=source_img,
            crop_bbox=crop_bbox,
            mask_rects=mask_rects,
            out_dir=out_dir,
            kind=kind,
        )
        if not ov:
            return False
        return self._insert_restored_overlay(page, ov)

    def _restore_background_rect(self, page, item, rect, kind="background_restore"):
        if not isinstance(item, dict):
            return False
        page_data = item.get("page_data")
        if not isinstance(page_data, dict):
            return False
        try:
            use_text_erased_overlay = self._prefer_text_erased_overlay(item)
            if use_text_erased_overlay:
                rect = self._expanded_visual_erase_rect(item, fitz.Rect(rect))
                rect = self._clamp_rect_to_page(rect, page.rect)
            bbox_px = [
                rect.x0 / self.pixel_to_point,
                rect.y0 / self.pixel_to_point,
                rect.x1 / self.pixel_to_point,
                rect.y1 / self.pixel_to_point,
            ]
            if use_text_erased_overlay:
                ov = self._save_text_erased_overlay(page_data, bbox_px, kind=kind)
            else:
                ov = self._save_background_crop_overlay(page_data, bbox_px, kind=kind)
            return self._insert_restored_overlay(page, ov)
        except Exception:
            return False

    def _save_background_crop_overlay(self, page_data, bbox, kind="background_restore"):
        background_img = page_data.get("background_path") or page_data.get("source_image_path")
        if not background_img or not os.path.exists(background_img):
            return None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
            pad = max(0, int(self.dynamic_overlay_pad_px))
            with Image.open(background_img).convert("RGB") as im:
                x0 = max(0, x0 - pad)
                y0 = max(0, y0 - pad)
                x1 = min(im.width, x1 + pad)
                y1 = min(im.height, y1 + pad)
                if x1 <= x0 or y1 <= y0:
                    return None
                crop = im.crop((x0, y0, x1, y1))
                out_dir = os.path.dirname(page_data.get("background_path", "")) or "ocr_results"
                os.makedirs(out_dir, exist_ok=True)
                out_name = f"dynamic_overlay_{kind}_{uuid.uuid4().hex[:12]}.png"
                out_path = os.path.join(out_dir, out_name)
                crop.save(out_path)
                return {"path": out_path, "bbox": [x0, y0, x1, y1], "kind": kind}
        except Exception:
            return None

    def _visual_group_key(self, page, item):
        if not isinstance(item, dict):
            return None
        mode = str(item.get("descriptor_group_render_mode") or "").strip().lower()
        if mode not in {"annotation_group", "chart_legend_group", "chart_axis_group", "chart_series_group"}:
            return None
        gids = item.get("descriptor_group_ids") or {}
        group_id = (
            gids.get("annotation_group_id")
            or gids.get("legend_group_id")
            or gids.get("axis_group_id")
            or gids.get("tick_group_id")
            or gids.get("series_group_id")
        )
        if not group_id:
            return None
        return (int(getattr(page, "number", 0) or 0), mode, str(group_id))

    def _fitz_rect_from_pixels(self, bbox_like):
        if isinstance(bbox_like, fitz.Rect):
            return fitz.Rect(bbox_like)
        if not isinstance(bbox_like, (list, tuple)) or len(bbox_like) != 4:
            return None
        try:
            return fitz.Rect(
                float(bbox_like[0]) * self.pixel_to_point,
                float(bbox_like[1]) * self.pixel_to_point,
                float(bbox_like[2]) * self.pixel_to_point,
                float(bbox_like[3]) * self.pixel_to_point,
            )
        except Exception:
            return None

    def _expanded_visual_erase_rect(self, item, rect):
        if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
            return rect
        result = fitz.Rect(rect)
        visual_obj = item.get("descriptor_visual_text_object") or {}
        obj_rect = self._fitz_rect_from_pixels(visual_obj.get("bbox"))
        if isinstance(obj_rect, fitz.Rect) and obj_rect.get_area() > 0:
            result = result | obj_rect
        visual_text = item.get("descriptor_visual_text") or {}
        if str(visual_text.get("background_replacement_strategy") or "").strip().lower() != "text_erase_then_overlay":
            return result
        priority = str(visual_obj.get("visual_priority") or "").strip().lower()
        pad_x = max(1.5, min(7.0, result.height * 0.30))
        pad_y = max(1.2, min(5.5, result.height * 0.22))
        if priority == "primary":
            pad_x += 1.0
            pad_y += 0.8
        return fitz.Rect(result.x0 - pad_x, result.y0 - pad_y, result.x1 + pad_x, result.y1 + pad_y)

    def _visual_text_group_bbox(self, item):
        if not isinstance(item, dict):
            return None
        visual_group = item.get("descriptor_visual_text_group") or {}
        rect = self._fitz_rect_from_pixels(visual_group.get("bbox"))
        if isinstance(rect, fitz.Rect) and rect.get_area() > 0:
            return rect
        visual_obj = item.get("descriptor_visual_text_object") or {}
        rect = self._fitz_rect_from_pixels(visual_obj.get("bbox"))
        if isinstance(rect, fitz.Rect) and rect.get_area() > 0:
            return rect
        return None

    def _prepare_visual_group_background(self, page, item, group_rect, group_items=None):
        key = self._visual_group_key(page, item)
        if key is None:
            return False
        if key in self._prepared_visual_groups:
            return bool(self._prepared_visual_groups.get(key))
        ok = self._restore_inpainted_group_background(
            page,
            item,
            group_rect,
            group_items=group_items,
            kind=f"{key[1]}_group_bg_restore",
        )
        if not ok:
            ok = self._restore_background_rect(page, item, group_rect, kind=f"{key[1]}_group_bg_restore")
        self._prepared_visual_groups[key] = bool(ok)
        return bool(ok)

    def _is_visual_group_item(self, item):
        if not isinstance(item, dict):
            return False
        mode = str(item.get("descriptor_group_render_mode") or "").strip().lower()
        return mode in {"annotation_group", "chart_legend_group", "chart_axis_group", "chart_series_group"}

    def _group_visual_items(self, items):
        grouped = {}
        passthrough = []
        for item in items or []:
            if not self._is_visual_group_item(item):
                passthrough.append(item)
                continue
            gids = item.get("descriptor_group_ids") or {}
            mode = str(item.get("descriptor_group_render_mode") or "").strip().lower()
            group_id = (
                gids.get("annotation_group_id")
                or gids.get("legend_group_id")
                or gids.get("axis_group_id")
                or gids.get("tick_group_id")
                or gids.get("series_group_id")
            )
            if not group_id:
                passthrough.append(item)
                continue
            key = (mode, str(group_id))
            grouped.setdefault(key, []).append(item)
        ordered_groups = []
        for key, members in grouped.items():
            members = sorted(members, key=lambda it: (it["bbox"].y0, it["bbox"].x0))
            preferred_bbox = None
            for member in members:
                preferred_bbox = self._visual_text_group_bbox(member)
                if isinstance(preferred_bbox, fitz.Rect) and preferred_bbox.get_area() > 0:
                    break
            members_bbox = fitz.Rect(
                min(it["bbox"].x0 for it in members),
                min(it["bbox"].y0 for it in members),
                max(it["bbox"].x1 for it in members),
                max(it["bbox"].y1 for it in members),
            )
            if isinstance(preferred_bbox, fitz.Rect) and preferred_bbox.get_area() > 0:
                group_bbox = preferred_bbox | members_bbox
            else:
                group_bbox = members_bbox
            ordered_groups.append(
                {
                    "mode": key[0],
                    "id": key[1],
                    "items": members,
                    "bbox": group_bbox,
                }
            )
        ordered_groups.sort(key=lambda g: (g["bbox"].y0, g["bbox"].x0))
        passthrough.sort(key=lambda it: (it["bbox"].y0, it["bbox"].x0))
        return ordered_groups, passthrough

    def _visual_group_role_priority(self, item):
        structural_role = str(item.get("descriptor_structural_role") or "").strip().lower()
        role = str(item.get("role") or "").strip().lower()
        if structural_role in {"figure_title", "chart_title"} or role == "title":
            return 0
        if role in {"section_heading", "header"}:
            return 1
        if structural_role in {"diagram_label", "chart_axis_label"}:
            return 2
        if structural_role in {"chart_legend_label", "chart_series_label"}:
            return 3
        if structural_role in {"chart_tick_label"}:
            return 4
        return 5

    def _dedupe_visual_group_items(self, items):
        kept = []
        for item in sorted(items or [], key=lambda it: (self._visual_group_role_priority(it), it["bbox"].y0, it["bbox"].x0)):
            text = self._clean_text_for_render(item.get("text", "")).strip().lower()
            if not text:
                kept.append(item)
                continue
            duplicate = False
            for prev in kept:
                prev_text = self._clean_text_for_render(prev.get("text", "")).strip().lower()
                if prev_text != text:
                    continue
                inter = (item["bbox"] & prev["bbox"]).get_area()
                close = abs(item["bbox"].x0 - prev["bbox"].x0) <= 24.0 and abs(item["bbox"].y0 - prev["bbox"].y0) <= 24.0
                overlap_ratio = inter / max(1.0, min(item["bbox"].get_area(), prev["bbox"].get_area()))
                if overlap_ratio >= 0.35 or close:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(item)
        return sorted(kept, key=lambda it: (self._visual_group_role_priority(it), it["bbox"].y0, it["bbox"].x0))

    def _whiteout_rect(self, page, rect, pad_x=1.5, pad_y=0.8):
        try:
            wipe = fitz.Rect(rect.x0 - pad_x, rect.y0 - pad_y, rect.x1 + pad_x, rect.y1 + pad_y)
            wipe = self._clamp_rect_to_page(wipe, page.rect)
            if wipe.get_area() > 0:
                page.draw_rect(wipe, color=None, fill=(1, 1, 1), overlay=True)
        except Exception:
            return

    def _has_native_blocks(self, page_data):
        for b in page_data.get("blocks", []):
            if b.get("source") == "native":
                return True
        return False

    def _get_block_text(self, block):
        text_parts = []
        for line in block.get("lines", []):
            for phrase in line.get("phrases", []):
                t = self._phrase_text_for_render(phrase)
                if t:
                    text_parts.append(t)
        block_preferred_text = re.sub(r"\s+", " ", (block.get("translated_text") or "").strip())
        return block_preferred_text or re.sub(r"\s+", " ", " ".join(text_parts)).strip()

    def _get_block_source_text(self, block):
        text_parts = []
        for line in block.get("lines", []):
            line_src = self._clean_text_for_render(line.get("line_text", "")).strip()
            if line_src:
                text_parts.append(line_src)
                continue
            for phrase in line.get("phrases", []):
                src = self._clean_text_for_render(phrase.get("texte", "")).strip()
                if src:
                    text_parts.append(src)
        return re.sub(r"\s+", " ", " ".join(text_parts)).strip()

    def _is_symbol_heavy_text(self, text):
        s = text or ""
        if not s:
            return False
        if re.search(r"[=<>±×÷∑∫∞≈≠≤≥√∆∂µλΩα-ωΑ-Ω]", s):
            return True
        letters = sum(1 for c in s if c.isalpha())
        symbols = sum(1 for c in s if not c.isalnum() and not c.isspace())
        if symbols >= 3 and symbols >= max(2, int(0.25 * max(1, letters + symbols))):
            return True
        return False

    def _is_reference_like_text(self, text):
        s = self._clean_text_for_render(text)
        if not s:
            return False
        if re.fullmatch(r"\(?\d+(?:\.\d+){1,4}\)?", s):
            return True
        if re.fullmatch(r"\((\d+|[ivxlcdm]+|[a-z])\)", s, flags=re.IGNORECASE):
            return True
        if re.fullmatch(r"\[\d+([,\-\s]*\d+)*\]", s):
            return True
        return False

    def _should_render_equation_as_anchored_text(self, item):
        if not isinstance(item, dict):
            return False
        if (item.get("role") or "").lower() != "equation_inline":
            return False
        text = self._clean_text_for_render(item.get("text", ""))
        if not text:
            return False
        if self._is_reference_like_text(text):
            return True
        if self._is_symbol_heavy_text(text):
            return False
        if re.search(r"\b[a-zA-Z]\s*/\s*[a-zA-Z]\b", text):
            return False
        if re.search(r"\b[dD][A-Za-z]\s*/\s*d[A-Za-z]\b", text):
            return False
        words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", text)
        return len(words) >= 1

    def _should_lock_equation_overlay(self, block, rendered_text=None, source_text=None):
        if not isinstance(block, dict):
            return False
        if str(block.get("role") or "").strip().lower() != "equation_inline":
            return False
        src = self._clean_text_for_render(source_text if source_text is not None else self._get_block_source_text(block))
        txt = self._clean_text_for_render(rendered_text if rendered_text is not None else self._get_block_text(block))
        candidate = src or txt
        if not candidate:
            return False
        if src and txt and src == txt:
            return True
        if self._is_reference_like_text(candidate):
            return True
        if self._is_symbol_heavy_text(candidate):
            return True
        if re.search(r"\b[a-zA-Z]\s*/\s*[a-zA-Z]\b", candidate):
            return True
        if re.search(r"\b[dD][A-Za-z]\s*/\s*d[A-Za-z]\b", candidate):
            return True
        if len(candidate) <= 4:
            return True
        return False

    def _anchored_zone_bounds(self, item, top, bottom):
        bbox = item.get("bbox")
        if not isinstance(bbox, fitz.Rect):
            return max(top, 0.0), min(bottom, bottom)
        descriptor_region_bbox = item.get("descriptor_region_bbox")
        role = str(item.get("role") or "").strip().lower()
        slot_h = max(8.0, float(item.get("slot_h_pt", 8.0) or 8.0))
        source_lines = [x for x in (item.get("source_lines") or []) if str(x).strip()]
        translated_multiline = bool(
            item.get("translated_block")
            and item.get("preserve_linebreaks")
            and len(source_lines) >= 2
        )
        zone_top = max(top, bbox.y0 - max(4.0, slot_h * 0.6))
        extra = max(8.0, slot_h * 1.2)
        if translated_multiline:
            extra = max(extra, slot_h * (len(source_lines) + 1.6))
        if role in {"header", "footer"}:
            extra = max(extra, slot_h * 2.2)
        zone_bottom = min(bottom, bbox.y1 + extra)
        if isinstance(descriptor_region_bbox, fitz.Rect) and descriptor_region_bbox.get_area() > 0:
            if role in {"header", "footer"}:
                zone_top = max(top, min(zone_top, descriptor_region_bbox.y0))
                zone_bottom = min(bottom, max(zone_bottom, descriptor_region_bbox.y1, bbox.y1 + extra))
            else:
                zone_top = max(zone_top, descriptor_region_bbox.y0)
                zone_bottom = min(zone_bottom, max(zone_top + 6.0, descriptor_region_bbox.y1))
        return zone_top, zone_bottom

    def _overlay_exists(self, overlays, bbox):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return True
        r = fitz.Rect([float(v) for v in bbox])
        for ov in overlays:
            bb = ov.get("bbox") if isinstance(ov, dict) else None
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            o = fitz.Rect([float(v) for v in bb])
            inter = (r & o).get_area()
            if inter <= 0:
                continue
            ratio = inter / max(1e-9, min(r.get_area(), o.get_area()))
            if ratio >= 0.95:
                return True
        return False

    def _layout_descriptor_maps(self, page_data):
        descriptor = (page_data or {}).get("layout_descriptor")
        if not isinstance(descriptor, dict):
            descriptor = ((page_data or {}).get("layout") or {}).get("layout_descriptor")
        if not isinstance(descriptor, dict):
            return {}, {}, {}, {}, {}, {}, {}, {}, {}
        elements = descriptor.get("elements") or []
        groups = descriptor.get("groups") or []
        regions = descriptor.get("regions") or []
        constraints = descriptor.get("constraints") or []
        relations = descriptor.get("relations") or []
        page_organization = descriptor.get("page_organization") or {}
        reconstruction_plan = descriptor.get("reconstruction_plan") or {}
        visual_text_model = descriptor.get("visual_text_model") or {}
        element_map = {str(el.get("id")): el for el in elements if isinstance(el, dict) and el.get("id")}
        group_map = {str(gr.get("id")): gr for gr in groups if isinstance(gr, dict) and gr.get("id")}
        region_map = {str(rg.get("id")): rg for rg in regions if isinstance(rg, dict) and rg.get("id")}
        visual_object_map = {}
        for obj in visual_text_model.get("objects") or []:
            if not isinstance(obj, dict):
                continue
            source_element_id = str(obj.get("source_element_id") or "")
            if source_element_id:
                visual_object_map[source_element_id] = obj
        visual_group_map = {
            str(gr.get("id")): gr
            for gr in (visual_text_model.get("groups") or [])
            if isinstance(gr, dict) and gr.get("id")
        }
        constraint_map = {}
        for constraint in constraints:
            if not isinstance(constraint, dict):
                continue
            element_id = str(constraint.get("element_id") or "")
            if not element_id:
                continue
            constraint_map.setdefault(element_id, []).append(constraint)
        relation_map = {}
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            source_id = str(relation.get("source_id") or "")
            if not source_id:
                continue
            relation_map.setdefault(source_id, []).append(relation)
        return (
            element_map,
            group_map,
            region_map,
            constraint_map,
            relation_map,
            page_organization,
            reconstruction_plan,
            visual_object_map,
            visual_group_map,
        )

    def _layout_descriptor_v3_maps(self, page_data):
        descriptor = (page_data or {}).get("layout_descriptor_v3")
        if not isinstance(descriptor, dict):
            descriptor = ((page_data or {}).get("layout") or {}).get("layout_descriptor_v3")
        if not isinstance(descriptor, dict):
            return {}, {}, {}, {}, {}, {}, {}
        reconstruction_contract = descriptor.get("reconstruction_contract") or {}
        render_model = descriptor.get("render_model") or {}
        dependency_graph = descriptor.get("dependency_graph") or {}
        spatial_graph = descriptor.get("spatial_graph") or {}
        render_unit_map = {
            str(unit.get("source_element_id") or ""): unit
            for unit in (render_model.get("render_units") or [])
            if isinstance(unit, dict) and unit.get("source_element_id")
        }
        container_map = {
            str(container.get("id") or ""): container
            for container in (reconstruction_contract.get("containers") or render_model.get("containers") or [])
            if isinstance(container, dict) and container.get("id")
        }
        placement_constraint_map = {}
        for constraint in reconstruction_contract.get("placement_constraints") or []:
            if not isinstance(constraint, dict):
                continue
            source_element_id = str(constraint.get("source_element_id") or "")
            if not source_element_id:
                continue
            placement_constraint_map.setdefault(source_element_id, []).append(constraint)
        dependency_edge_map = {}
        for edge in reconstruction_contract.get("execution_edges") or dependency_graph.get("edges") or []:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source") or "")
            if not source:
                continue
            dependency_edge_map.setdefault(source, []).append(edge)
        spatial_cluster_map = {
            "row_clusters": list(spatial_graph.get("row_clusters") or []),
            "baseline_clusters": list(spatial_graph.get("baseline_clusters") or []),
        }
        return (
            descriptor,
            reconstruction_contract,
            render_unit_map,
            container_map,
            placement_constraint_map,
            dependency_edge_map,
            spatial_cluster_map,
        )

    def _save_crop_overlay(self, page_data, bbox, kind="dynamic"):
        source_img = page_data.get("source_image_path")
        if not source_img or not os.path.exists(source_img):
            return None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
            pad = max(0, int(self.dynamic_overlay_pad_px))
            with Image.open(source_img).convert("RGB") as im:
                x0 = max(0, x0 - pad)
                y0 = max(0, y0 - pad)
                x1 = min(im.width, x1 + pad)
                y1 = min(im.height, y1 + pad)
                if x1 <= x0 or y1 <= y0:
                    return None
                crop = im.crop((x0, y0, x1, y1))
                out_dir = os.path.dirname(page_data.get("background_path", "")) or "ocr_results"
                os.makedirs(out_dir, exist_ok=True)
                out_name = f"dynamic_overlay_{kind}_{uuid.uuid4().hex[:12]}.png"
                out_path = os.path.join(out_dir, out_name)
                crop.save(out_path)
                return {"path": out_path, "bbox": [x0, y0, x1, y1], "kind": kind}
        except Exception:
            return None

    def _collect_diagram_regions_px(self, blocks):
        out = []
        for b in blocks or []:
            role = (b.get("role") or "").lower()
            if role not in {"diagram_label", "diagram_text_label"}:
                continue
            bb = b.get("bbox")
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            r = fitz.Rect([float(v) for v in bb])
            if r.get_area() <= 0:
                continue
            out.append(r)
        return out

    def _collect_non_text_regions_px(self, page_data):
        out = []
        for z in page_data.get("non_text_zones", []) or []:
            if not isinstance(z, (list, tuple)) or len(z) != 4:
                continue
            r = fitz.Rect([float(v) for v in z])
            if r.get_area() <= 0:
                continue
            out.append(r)
        for im in page_data.get("images", []) or []:
            bb = im.get("bbox") if isinstance(im, dict) else im
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            r = fitz.Rect([float(v) for v in bb])
            if r.get_area() > 0:
                out.append(r)
        for dr in page_data.get("drawings", []) or []:
            bb = dr.get("bbox") if isinstance(dr, dict) else dr
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                continue
            r = fitz.Rect([float(v) for v in bb])
            if r.get_area() > 0:
                out.append(r)
        return out

    def _overlap_ratio(self, r1, r2):
        inter = (r1 & r2).get_area()
        if inter <= 0:
            return 0.0
        return inter / max(1e-9, min(r1.get_area(), r2.get_area()))

    def _block_should_be_image_locked(self, block, non_text_regions, diagram_regions):
        bb = block.get("bbox")
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            return False
        role = (block.get("role") or "body").lower()
        if role in {"diagram_label", "diagram_text_label", "figure_caption"}:
            return True
        if role != "body":
            return False
        rb = fitz.Rect([float(v) for v in bb])
        if rb.get_area() <= 0:
            return False
        txt = self._get_block_text(block)
        word_count = len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", txt))
        for dr in diagram_regions or []:
            inter = (rb & dr).get_area()
            if inter > 0 and inter / max(1e-9, rb.get_area()) >= 0.10:
                return True
        for nz in non_text_regions or []:
            inter = (rb & nz).get_area()
            if inter <= 0:
                continue
            block_cov = inter / max(1e-9, rb.get_area())
            zone_cov = inter / max(1e-9, nz.get_area())
            # Generic professional rule: any meaningful overlap with non-text
            # zones should be image-locked in translated mode to avoid layout drift.
            if block_cov >= 0.10 or zone_cov >= 0.28:
                return True
        return False

    def _block_is_risky_for_reflow(self, block, text):
        role = block.get("role", "body")
        if role in {"equation_inline", "diagram_text_label", "diagram_label"}:
            return True
        bbox = block.get("bbox", [0, 0, 0, 0])
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False
        w = max(1.0, float(bbox[2]) - float(bbox[0]))
        h = max(1.0, float(bbox[3]) - float(bbox[1]))
        if self._is_symbol_heavy_text(text) and h <= 40:
            return True
        if role in {"figure_caption", "header", "footer"} and len(text) <= 220:
            return True
        if h <= 22 and len(text) <= 60:
            return True
        return False

    def _has_unresolved_native_font(self, block):
        if block.get("source") != "native":
            return False
        style = self._style_from_block(block)
        # enrich with first span style when available
        for line in block.get("lines", []):
            for phrase in line.get("phrases", []):
                spans = phrase.get("spans", [])
                if spans:
                    style = self._merge_styles(spans[0].get("style", {}), style)
                    break
            else:
                continue
            break
        resolved = self.font_resolver.resolve(style or {})
        fontfile = resolved.get("fontfile")
        builtin = (resolved.get("builtin") or "").lower()
        requested = (style.get("font") or "").strip().lower() if isinstance(style, dict) else ""
        # If a specific native font was requested but no file resolved, treat as unresolved.
        if requested and not fontfile and builtin in {"helv", "times", "courier"}:
            if requested not in {"helv", "times", "courier", "arial", "helvetica"}:
                return True
        return False

    def _inject_dynamic_immutable_overlays(self, page_data):
        overlays = page_data.setdefault("immutable_overlays", [])
        blocks = page_data.get("blocks", [])
        diagram_regions = self._collect_diagram_regions_px(blocks)
        non_text_regions = self._collect_non_text_regions_px(page_data)
        dims = page_data.get("dimensions", {}) or {}
        page_h_px = float(dims.get("height", 0.0) or 0.0)
        page_family = str(page_data.get("page_family") or ((page_data.get("layout") or {}).get("page_family")) or "").strip().lower()
        layout_type = str(page_data.get("layout_type") or ((page_data.get("layout") or {}).get("layout_type")) or "").strip().lower()
        document_type = str(page_data.get("document_type") or ((page_data.get("layout") or {}).get("document_type")) or "").strip().lower()
        figure_like_page = (
            layout_type in {"annotated_page", "image_dominant", "mixed_blocks"}
            and document_type not in {"scientific_paper", "book_page", "manual_guide"}
        ) or page_family in {"body_with_figure", "body_with_diagram", "mixed_page"}
        for block in blocks:
            bbox = block.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            role = block.get("role", "body")
            text = self._get_block_text(block)
            source_text = self._get_block_source_text(block)
            is_translated = self._is_translated_block(block)
            if self._should_lock_equation_overlay(block, rendered_text=text, source_text=source_text):
                if not self._overlay_exists(overlays, bbox):
                    ov = self._save_crop_overlay(page_data, bbox, kind="equation_inline_locked")
                    if ov:
                        overlays.append(ov)
                block["render_mode"] = "background_only"
                for line in block.get("lines", []):
                    for phrase in line.get("phrases", []):
                        phrase["render_mode"] = "background_only"
                continue
            # Non-translated labels in image/diagram zones must stay visually identical:
            # keep exact red bbox by locking them as source-image overlays.
            try:
                by1 = float(bbox[3])
            except Exception:
                by1 = 0.0
            if (
                (not is_translated)
                and role in {"title", "diagram_label", "diagram_text_label", "equation_inline"}
                and page_h_px > 0
                and by1 <= page_h_px * 0.62
                and not (figure_like_page and role in {"title", "diagram_label", "diagram_text_label"})
            ):
                if not self._overlay_exists(overlays, bbox):
                    ov = self._save_crop_overlay(page_data, bbox, kind="label_original_locked")
                    if ov:
                        overlays.append(ov)
                block["render_mode"] = "background_only"
                for line in block.get("lines", []):
                    for phrase in line.get("phrases", []):
                        phrase["render_mode"] = "background_only"
                continue
            # Never hide translated body blocks behind overlays.
            if is_translated and role == "body":
                continue
            # Translated captions must stay renderable as text even when the
            # page classifier does not recognize the page as figure-like.
            if is_translated and role == "figure_caption":
                continue
            # If extractor already marked a diagram label as background_only,
            # ensure we keep it as immutable image overlay.
            if block.get("render_mode") == "background_only" and role == "diagram_label":
                if not self._overlay_exists(overlays, bbox):
                    ov = self._save_crop_overlay(page_data, bbox, kind="diagram_block")
                    if ov:
                        overlays.append(ov)
                continue
            # In translated mode, keep figure/diagram textual artifacts as immutable overlays
            # to preserve professional layout integrity.
            if is_translated and role in {"diagram_label", "diagram_text_label", "figure_caption"} and not figure_like_page:
                if not self._overlay_exists(overlays, bbox):
                    ov = self._save_crop_overlay(page_data, bbox, kind=f"{role}_translated")
                    if ov:
                        overlays.append(ov)
                block["render_mode"] = "background_only"
                for line in block.get("lines", []):
                    for phrase in line.get("phrases", []):
                        phrase["render_mode"] = "background_only"
                continue
            if (
                self.pro_strict_mode
                and is_translated
                and role != "body"
                and not (figure_like_page and role in {"figure_caption", "title", "diagram_label", "diagram_text_label"})
                and self._block_should_be_image_locked(block, non_text_regions, diagram_regions)
            ):
                lock_kind = "body_overlap_non_text" if role == "body" else f"{role}_translated"
                if not self._overlay_exists(overlays, bbox):
                    ov = self._save_crop_overlay(page_data, bbox, kind=lock_kind)
                    if ov:
                        overlays.append(ov)
                block["render_mode"] = "background_only"
                for line in block.get("lines", []):
                    for phrase in line.get("phrases", []):
                        phrase["render_mode"] = "background_only"
                continue
            # Never auto-hide other translated blocks behind dynamic overlays.
            if is_translated:
                continue
            add_kind = None
            if self.dynamic_equation_overlays and role == "equation_inline":
                add_kind = "equation"
            elif self.dynamic_symbol_overlays and role == "diagram_text_label":
                if self._is_symbol_heavy_text(text) or len(text) <= 40:
                    add_kind = "diagram_text"
            elif self.dynamic_risk_overlays and self._block_is_risky_for_reflow(block, text):
                # conservative: only non-body short/special blocks
                if role != "body":
                    add_kind = "risk"
            if self.dynamic_risk_overlays and self._has_unresolved_native_font(block):
                add_kind = "native_font_fallback"
            if not add_kind:
                continue
            if self._overlay_exists(overlays, bbox):
                continue
            ov = self._save_crop_overlay(page_data, bbox, kind=add_kind)
            if ov:
                overlays.append(ov)
                block["render_mode"] = "background_only"
                for line in block.get("lines", []):
                    for phrase in line.get("phrases", []):
                        phrase["render_mode"] = "background_only"

    def _postcheck_equation_fidelity(self, page, page_data):
        source_img = page_data.get("source_image_path")
        if not source_img or not os.path.exists(source_img):
            return
        eq_blocks = [b for b in page_data.get("blocks", []) if b.get("role") == "equation_inline"]
        if not eq_blocks:
            return
        try:
            with Image.open(source_img).convert("RGB") as src:
                mat = fitz.Matrix(self.layout_debug_dpi / 72.0, self.layout_debug_dpi / 72.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                ren = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                scale = float(self.layout_debug_dpi) / 150.0
                for b in eq_blocks:
                    bb = b.get("bbox")
                    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                        continue
                    x0, y0, x1, y1 = [int(round(float(v))) for v in bb]
                    x0 = max(0, min(src.width, x0)); x1 = max(0, min(src.width, x1))
                    y0 = max(0, min(src.height, y0)); y1 = max(0, min(src.height, y1))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    src_crop = src.crop((x0, y0, x1, y1))
                    rx0, ry0, rx1, ry1 = [int(round(v * scale)) for v in (x0, y0, x1, y1)]
                    rx0 = max(0, min(ren.width, rx0)); rx1 = max(0, min(ren.width, rx1))
                    ry0 = max(0, min(ren.height, ry0)); ry1 = max(0, min(ren.height, ry1))
                    if rx1 <= rx0 or ry1 <= ry0:
                        continue
                    ren_crop = ren.crop((rx0, ry0, rx1, ry1)).resize(src_crop.size, Image.BILINEAR)
                    # Compute grayscale absolute diff using PIL only.
                    src_g = src_crop.convert("L")
                    ren_g = ren_crop.convert("L")
                    diff = 0.0
                    src_px = src_g.load()
                    ren_px = ren_g.load()
                    w, h = src_g.size
                    n = max(1, w * h)
                    for yy in range(h):
                        for xx in range(w):
                            diff += abs(int(src_px[xx, yy]) - int(ren_px[xx, yy]))
                    mad = diff / n
                    if mad > self.equation_diff_threshold:
                        ov = self._save_crop_overlay(page_data, [x0, y0, x1, y1], kind="equation_post")
                        if ov and os.path.exists(ov["path"]):
                            px0, py0, px1, py1 = [float(v) * self.pixel_to_point for v in ov["bbox"]]
                            rect = fitz.Rect(px0, py0, px1, py1)
                            page.insert_image(rect, filename=ov["path"], overlay=True, keep_proportion=False)
        except Exception:
            return

    def _is_translated_block(self, block):
        block_tt = (block.get("translated_text") or "").strip()
        block_src = (block.get("text") or "").strip()
        if not block_src:
            parts = []
            for ln in block.get("lines", []) or []:
                lt = (ln.get("line_text") or "").strip()
                if lt:
                    parts.append(lt)
                    continue
                for ph in ln.get("phrases", []) or []:
                    pt = (ph.get("texte") or "").strip()
                    if pt:
                        parts.append(pt)
            block_src = " ".join(parts).strip()
        if block_tt and block_src and block_tt != block_src:
            return True
        for line in block.get("lines", []):
            for phrase in line.get("phrases", []):
                tt = (phrase.get("translated_text") or "").strip()
                src = (phrase.get("texte") or "").strip()
                if tt and tt != src:
                    return True
        return False

    def _postcheck_native_block_fidelity(self, page, page_data):
        source_img = page_data.get("source_image_path")
        if not source_img or not os.path.exists(source_img):
            return
        if self._has_translated_content(page_data):
            return
        native_blocks = [
            b for b in page_data.get("blocks", [])
            if b.get("source") == "native"
            and b.get("render_mode") != "background_only"
            and not self._is_translated_block(b)
        ]
        if not native_blocks:
            return
        try:
            with Image.open(source_img).convert("RGB") as src:
                mat = fitz.Matrix(self.layout_debug_dpi / 72.0, self.layout_debug_dpi / 72.0)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                ren = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                scale = float(self.layout_debug_dpi) / 150.0
                for b in native_blocks:
                    bb = b.get("bbox")
                    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                        continue
                    x0, y0, x1, y1 = [int(round(float(v))) for v in bb]
                    x0 = max(0, min(src.width, x0))
                    x1 = max(0, min(src.width, x1))
                    y0 = max(0, min(src.height, y0))
                    y1 = max(0, min(src.height, y1))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    src_crop = src.crop((x0, y0, x1, y1))
                    rx0, ry0, rx1, ry1 = [int(round(v * scale)) for v in (x0, y0, x1, y1)]
                    rx0 = max(0, min(ren.width, rx0))
                    rx1 = max(0, min(ren.width, rx1))
                    ry0 = max(0, min(ren.height, ry0))
                    ry1 = max(0, min(ren.height, ry1))
                    if rx1 <= rx0 or ry1 <= ry0:
                        continue
                    ren_crop = ren.crop((rx0, ry0, rx1, ry1)).resize(src_crop.size, Image.BILINEAR)
                    src_g = src_crop.convert("L")
                    ren_g = ren_crop.convert("L")
                    diff = 0.0
                    src_px = src_g.load()
                    ren_px = ren_g.load()
                    w, h = src_g.size
                    n = max(1, w * h)
                    for yy in range(h):
                        for xx in range(w):
                            diff += abs(int(src_px[xx, yy]) - int(ren_px[xx, yy]))
                    mad = diff / n
                    if mad > self.native_block_diff_threshold:
                        ov = self._save_crop_overlay(page_data, [x0, y0, x1, y1], kind="native_post")
                        if ov and os.path.exists(ov["path"]):
                            px0, py0, px1, py1 = [float(v) * self.pixel_to_point for v in ov["bbox"]]
                            rect = fitz.Rect(px0, py0, px1, py1)
                            page.insert_image(rect, filename=ov["path"], overlay=True, keep_proportion=False)
        except Exception:
            return

    def _extract_block_slot_items(self, page_data):
        items = []
        (
            descriptor_elements,
            descriptor_groups,
            descriptor_regions,
            descriptor_constraints,
            descriptor_relations,
            descriptor_page_organization,
            descriptor_reconstruction_plan,
            descriptor_visual_objects,
            descriptor_visual_groups,
        ) = self._layout_descriptor_maps(page_data)
        (
            descriptor_v3,
            descriptor_v3_contract,
            descriptor_v3_render_units,
            descriptor_v3_containers,
            descriptor_v3_placement_constraints,
            descriptor_v3_dependency_edges,
            descriptor_v3_spatial_clusters,
        ) = self._layout_descriptor_v3_maps(page_data)
        dims = page_data.get("dimensions", {}) or {}
        page_w_pt = float(dims.get("width", 1000.0)) * self.pixel_to_point
        page_h_pt = float(dims.get("height", 1000.0)) * self.pixel_to_point
        page_lang = self._resolve_page_lang(page_data)
        page_role = str((page_data or {}).get("page_role", "")).strip().lower()
        # Largest horizontal blue-bbox limit on the page (typically body text column).
        blue_right_candidates = []
        for b0 in page_data.get("blocks", []):
            bb0 = b0.get("bbox")
            if not isinstance(bb0, (list, tuple)) or len(bb0) != 4:
                continue
            if b0.get("render_mode") == "background_only":
                continue
            role0 = (b0.get("role") or "body").lower()
            if role0 not in {"body", "figure_caption", "section_heading", "title"}:
                continue
            blue_right_candidates.append(float(bb0[2]) * self.pixel_to_point)
        page_blue_right_pt = max(blue_right_candidates) if blue_right_candidates else page_w_pt
        for block in page_data.get("blocks", []):
            if block.get("render_mode") == "background_only":
                continue
            block_id = str(block.get("id") or "")
            descriptor_block = descriptor_elements.get(block_id) if block_id else None
            paragraph_group = None
            if isinstance(descriptor_block, dict):
                paragraph_id = str(descriptor_block.get("paragraph_id") or "").strip()
                if paragraph_id:
                    paragraph_group = descriptor_groups.get(paragraph_id)
            paragraph_constraints = (paragraph_group or {}).get("constraints") or {}
            region_id = (paragraph_group or {}).get("region_id") or ((descriptor_block or {}).get("page_region_id"))
            descriptor_region = descriptor_regions.get(str(region_id)) if region_id else None
            descriptor_region_type = str((descriptor_region or {}).get("type") or "").strip().lower()
            descriptor_ai_region_id = str((descriptor_block or {}).get("ai_region_id") or "")
            descriptor_ai_region = descriptor_regions.get(descriptor_ai_region_id) if descriptor_ai_region_id else None
            descriptor_ai_region_type = str((descriptor_ai_region or {}).get("type") or "").strip().lower()
            descriptor_band_role = str((descriptor_block or {}).get("band_role") or "").strip().lower()
            descriptor_structural_role = str((descriptor_block or {}).get("structural_role") or "").strip().lower()
            descriptor_layout_behavior = str((descriptor_block or {}).get("layout_behavior") or "").strip().lower()
            descriptor_attachment_target_id = str((descriptor_block or {}).get("attachment_target_id") or "").strip()
            descriptor_group_ids = (descriptor_block or {}).get("group_ids") or {}
            descriptor_section_id = str((descriptor_block or {}).get("section_id") or "").strip()
            descriptor_typographic_class = str((descriptor_block or {}).get("typographic_class") or "").strip().lower()
            block_structure_hints = dict(block.get("structure_hints") or {})
            layout_ai_label_hint = str(block_structure_hints.get("layout_ai_label_hint") or "").strip().lower()
            layout_ai_text_line_height_pt = float(block_structure_hints.get("layout_ai_text_line_height_hint") or 0.0) * self.pixel_to_point
            layout_ai_text_line_width_pt = float(block_structure_hints.get("layout_ai_text_line_width_hint") or 0.0) * self.pixel_to_point
            layout_ai_block_height_pt = float(block_structure_hints.get("layout_ai_block_height_hint") or 0.0) * self.pixel_to_point
            layout_ai_block_width_pt = float(block_structure_hints.get("layout_ai_block_width_hint") or 0.0) * self.pixel_to_point
            descriptor_visual_object = descriptor_visual_objects.get(block_id) if block_id else None
            descriptor_visual_group = None
            if isinstance(descriptor_visual_object, dict):
                visual_group_id = str(descriptor_visual_object.get("group_id") or "").strip()
                if visual_group_id:
                    descriptor_visual_group = descriptor_visual_groups.get(visual_group_id)
            descriptor_v3_render_unit = descriptor_v3_render_units.get(block_id) if block_id else None
            descriptor_v3_container_entries = []
            if isinstance(descriptor_v3_render_unit, dict):
                for container_id in descriptor_v3_render_unit.get("container_ids") or []:
                    container = descriptor_v3_containers.get(str(container_id))
                    if isinstance(container, dict):
                        descriptor_v3_container_entries.append(container)
            descriptor_v3_block_constraints = descriptor_v3_placement_constraints.get(block_id, []) if block_id else []
            descriptor_v3_block_edges = descriptor_v3_dependency_edges.get(block_id, []) if block_id else []
            block_constraints = descriptor_constraints.get(block_id, []) if block_id else []
            descriptor_region_rect = (
                fitz.Rect([float(v) * self.pixel_to_point for v in descriptor_region.get("bbox")])
                if isinstance(descriptor_region, dict) and isinstance(descriptor_region.get("bbox"), (list, tuple)) and len(descriptor_region.get("bbox")) == 4
                else None
            )
            block_relations = descriptor_relations.get(block_id, []) if block_id else []
            anchor_target_rect = None
            anchor_preferred_side = ""
            for relation in block_relations:
                rel_type = str(relation.get("type") or "").strip().lower()
                if rel_type not in {"anchored_to", "left_of", "right_of", "above", "below"}:
                    continue
                target_id = str(relation.get("target_id") or "")
                target_region = descriptor_regions.get(target_id)
                target_bbox = (target_region or {}).get("bbox")
                if isinstance(target_bbox, (list, tuple)) and len(target_bbox) == 4:
                    anchor_target_rect = fitz.Rect([float(v) * self.pixel_to_point for v in target_bbox])
                    if rel_type in {"left_of", "right_of", "above", "below"}:
                        anchor_preferred_side = rel_type
                    elif not anchor_preferred_side and isinstance(descriptor_region_rect, fitz.Rect):
                        dx = descriptor_region_rect.x0 + descriptor_region_rect.width / 2.0 - (anchor_target_rect.x0 + anchor_target_rect.width / 2.0)
                        dy = descriptor_region_rect.y0 + descriptor_region_rect.height / 2.0 - (anchor_target_rect.y0 + anchor_target_rect.height / 2.0)
                        if abs(dx) >= abs(dy):
                            anchor_preferred_side = "right_of" if dx >= 0 else "left_of"
                        else:
                            anchor_preferred_side = "below" if dy >= 0 else "above"
                    break
            table_locked_block = bool(
                descriptor_region_type in {"table_cell", "table_row"}
                or any(str(c.get("type") or "") == "table_cell_locked" for c in block_constraints)
                or descriptor_layout_behavior in {"locked_in_cell", "locked_in_table"}
            )
            preserve_sentence_integrity = bool(
                any(str(constraint.get("type") or "") == "no_internal_sentence_break" for constraint in block_constraints)
                or (paragraph_constraints and paragraph_constraints.get("can_break_inside_sentence") is False)
            )
            source = block.get("source", "ocr")
            block_is_translated = self._is_translated_block(block)
            block_role = block.get("role", "body")
            text_parts = []
            line_texts = []
            line_entries = []
            line_markers_used = []
            style = self._style_from_block(block)
            slots = []
            translated_phrase_items = []
            span_color_sequence = []
            first_run_text_parts = []
            first_run_bbox = None
            first_run_style = None
            first_run_color = None
            first_run_locked = False
            for line_idx, line in enumerate(block.get("lines", [])):
                this_line_parts = []
                line_visible_rects = []
                line_has_hidden_phrase = False
                line_runs = []
                current_run = None
                line_inline_segments = []
                for phrase in line.get("phrases", []):
                    if phrase.get("render_mode") == "background_only":
                        line_has_hidden_phrase = True
                        if current_run and current_run.get("text_parts"):
                            line_runs.append(current_run)
                        current_run = None
                        continue
                    t = self._phrase_text_for_render(phrase)
                    if t:
                        this_line_parts.append(t)
                    phrase_rect = None
                    pb = phrase.get("bbox") or line.get("bbox")
                    if isinstance(pb, (list, tuple)) and len(pb) == 4:
                        phrase_rect = fitz.Rect([float(v) * self.pixel_to_point for v in pb])
                        if phrase_rect.get_area() > 0:
                            line_visible_rects.append(fitz.Rect(phrase_rect))
                    if phrase.get("spans"):
                        phrase_style = {}
                        for sp_sel in phrase.get("spans", []):
                            if not isinstance(sp_sel, dict):
                                continue
                            sp_txt = self._clean_text_for_render(sp_sel.get("texte", ""))
                            if sp_sel.get("skip_render"):
                                continue
                            if re.fullmatch(r"(?:[•▪◦·\-\*]|\d+[.)]?|[A-Za-z][.)])", sp_txt or ""):
                                continue
                            phrase_style = sp_sel.get("style", {}) if isinstance(sp_sel.get("style"), dict) else {}
                            break
                            if not phrase_style:
                                phrase_style = phrase["spans"][0].get("style", {})
                        style = self._merge_styles(phrase_style, style)
                        pcol_any = str(phrase_style.get("color", "")).strip()
                        if pcol_any:
                            span_color_sequence.append(pcol_any)
                        if line_idx == 0 and not first_run_locked and t:
                            pcol = str(phrase_style.get("color", "")).lower()
                            pb0 = phrase.get("bbox") or line.get("bbox")
                            if first_run_color is None:
                                first_run_color = pcol
                                first_run_style = self._merge_styles(phrase_style, style)
                                first_run_text_parts.append(t)
                                if isinstance(pb0, (list, tuple)) and len(pb0) == 4:
                                    first_run_bbox = fitz.Rect([float(v) * self.pixel_to_point for v in pb0])
                            elif pcol == first_run_color:
                                first_run_text_parts.append(t)
                                if isinstance(pb0, (list, tuple)) and len(pb0) == 4:
                                    r0 = fitz.Rect([float(v) * self.pixel_to_point for v in pb0])
                                    if first_run_bbox is None:
                                        first_run_bbox = fitz.Rect(r0)
                                    else:
                                        first_run_bbox = first_run_bbox | r0
                            else:
                                # first style run ended
                                first_run_locked = True
                    phrase_source_text = self._clean_text_for_render(phrase.get("texte", ""))
                    if t and isinstance(phrase_rect, fitz.Rect) and phrase_rect.get_area() > 0:
                        if current_run is None:
                            current_run = {
                                "text_parts": [],
                                "source_parts": [],
                                "bbox": fitz.Rect(phrase_rect),
                                "style": self._merge_styles(phrase_style if isinstance(locals().get("phrase_style"), dict) else {}, style),
                            }
                        else:
                            current_run["bbox"] = current_run["bbox"] | phrase_rect
                        current_run["text_parts"].append(self._clean_text_for_render(t))
                        if phrase_source_text:
                            current_run["source_parts"].append(phrase_source_text)
                    if block_is_translated and block_role in {"diagram_text_label"}:
                        pb = phrase.get("bbox") or line.get("bbox") or block.get("bbox")
                        if isinstance(pb, (list, tuple)) and len(pb) == 4:
                            pr = fitz.Rect([float(v) * self.pixel_to_point for v in pb])
                            if pr.get_area() > 0:
                                pstyle_base = {}
                                for sp_sel in phrase.get("spans", []):
                                    if not isinstance(sp_sel, dict):
                                        continue
                                    if sp_sel.get("skip_render"):
                                        continue
                                    stx = self._clean_text_for_render(sp_sel.get("texte", ""))
                                    if re.fullmatch(r"(?:[•▪◦·\-\*]|\d+[.)]?|[A-Za-z][.)])", stx or ""):
                                        continue
                                    pstyle_base = sp_sel.get("style", {}) if isinstance(sp_sel.get("style"), dict) else {}
                                    break
                                if not pstyle_base and phrase.get("spans"):
                                    pstyle_base = phrase["spans"][0].get("style", {})
                                pstyle = self._merge_styles(pstyle_base, style)
                                translated_phrase_items.append(
                                    {
                                        "text": self._clean_text_for_render(t),
                                        "source_lines": [self._clean_text_for_render(t)],
                                        "preserve_linebreaks": False,
                                        "bbox": pr,
                                        "slots": [fitz.Rect(pr)],
                                        "slot_w_pt": max(10.0, pr.width),
                                        "slot_h_pt": max(6.0, pr.height),
                                        "slot_gap_x_pt": max(1.5, pr.height * 0.2),
                                        "slot_gap_y_pt": max(2.0, pr.height * 0.28),
                                        "row_start_x_pt": pr.x0,
                                        "style": self._merge_styles(pstyle, {}),
                                        "source": source,
                                        "source_text": self._clean_text_for_render(phrase.get("texte", "")),
                                        "role": phrase.get("role", line.get("role", block.get("role", "body"))),
                                        "lang": (block.get("language") or page_lang or self._infer_text_lang(t)),
                                        "is_title": False,
                                        "is_diagram_label": False,
                                        "style_lock_source": "phrase",
                                        "translated_block": True,
                                        "strict_bbox_mode": True,
                                        "exact_slot_render": True,
                                        "source_block_id": block_id,
                                        "source_line_count": 1,
                                        "layout_ai_label_hint": layout_ai_label_hint,
                                        "layout_ai_text_line_height_pt": layout_ai_text_line_height_pt,
                                        "layout_ai_text_line_width_pt": layout_ai_text_line_width_pt,
                                        "layout_ai_block_height_pt": layout_ai_block_height_pt,
                                        "layout_ai_block_width_pt": layout_ai_block_width_pt,
                                        "descriptor_region_bbox": fitz.Rect(descriptor_region_rect) if isinstance(descriptor_region_rect, fitz.Rect) else None,
                                        "descriptor_region_type": descriptor_region_type,
                                        "descriptor_region_id": str(region_id or ""),
                                        "descriptor_ai_region_type": descriptor_ai_region_type,
                                        "descriptor_ai_region_id": descriptor_ai_region_id,
                                        "descriptor_band_role": descriptor_band_role,
                                        "descriptor_structural_role": descriptor_structural_role,
                                        "descriptor_layout_behavior": descriptor_layout_behavior,
                                        "descriptor_attachment_target_id": descriptor_attachment_target_id,
                                        "descriptor_group_ids": descriptor_group_ids,
                                        "descriptor_group_render_mode": str((descriptor_block or {}).get("group_render_mode") or ""),
                                        "descriptor_typographic_class": descriptor_typographic_class,
                                        "descriptor_visual_text": dict((descriptor_block or {}).get("visual_text") or {}),
                                        "descriptor_visual_text_object": dict(descriptor_visual_object or {}),
                                        "descriptor_visual_text_group": dict(descriptor_visual_group or {}),
                                        "descriptor_section_id": descriptor_section_id,
                                        "descriptor_page_organization": descriptor_page_organization,
                                        "descriptor_reconstruction_plan": descriptor_reconstruction_plan,
                                        "descriptor_v3": descriptor_v3,
                                        "descriptor_v3_contract": descriptor_v3_contract,
                                        "descriptor_v3_render_unit": dict(descriptor_v3_render_unit or {}),
                                        "descriptor_v3_containers": list(descriptor_v3_container_entries),
                                        "descriptor_v3_placement_constraints": list(descriptor_v3_block_constraints),
                                        "descriptor_v3_dependency_edges": list(descriptor_v3_block_edges),
                                        "descriptor_v3_spatial_clusters": dict(descriptor_v3_spatial_clusters or {}),
                                        "page_data": page_data,
                                        "anchor_target_bbox": fitz.Rect(anchor_target_rect) if isinstance(anchor_target_rect, fitz.Rect) else None,
                                        "anchor_preferred_side": anchor_preferred_side,
                                        "preserve_sentence_integrity": bool(preserve_sentence_integrity),
                                        **self._alignment_payload(
                                            "left",
                                            source="exact_slot_phrase",
                                        ),
                                    }
                                )
                    if isinstance(phrase_rect, fitz.Rect) and phrase_rect.get_area() > 0:
                        slots.append(phrase_rect)
                if current_run and current_run.get("text_parts"):
                    line_runs.append(current_run)
                if this_line_parts:
                    should_split_line_to_phrase_items = bool(
                        block_is_translated
                        and block_role == "body"
                        and line_has_hidden_phrase
                        and line_runs
                    )
                    if should_split_line_to_phrase_items:
                        for run in line_runs:
                            run_text = self._clean_text_for_render(" ".join(run.get("text_parts") or []))
                            run_source_text = self._clean_text_for_render(" ".join(run.get("source_parts") or []))
                            run_rect = run.get("bbox")
                            if not run_text or not isinstance(run_rect, fitz.Rect) or run_rect.get_area() <= 0:
                                continue
                            translated_phrase_items.append(
                                {
                                    "text": run_text,
                                    "source_lines": [run_text],
                                    "preserve_linebreaks": False,
                                    "bbox": fitz.Rect(run_rect),
                                    "slots": [fitz.Rect(run_rect)],
                                    "slot_w_pt": max(10.0, run_rect.width),
                                    "slot_h_pt": max(6.0, run_rect.height),
                                    "slot_gap_x_pt": max(1.5, run_rect.height * 0.2),
                                    "slot_gap_y_pt": max(2.0, run_rect.height * 0.28),
                                    "row_start_x_pt": run_rect.x0,
                                    "style": self._merge_styles(run.get("style", {}), {}),
                                    "source": source,
                                    "source_text": run_source_text or run_text,
                                    "role": "body",
                                    "lang": (block.get("language") or page_lang or self._infer_text_lang(run_text)),
                                    "is_title": False,
                                    "is_diagram_label": False,
                                    "style_lock_source": "mixed_inline_phrase",
                                    "translated_block": True,
                                    "strict_bbox_mode": True,
                                    "exact_slot_render": True,
                                    "source_block_id": block_id,
                                    "source_line_count": 1,
                                    "layout_ai_label_hint": layout_ai_label_hint,
                                    "layout_ai_text_line_height_pt": layout_ai_text_line_height_pt,
                                    "layout_ai_text_line_width_pt": layout_ai_text_line_width_pt,
                                    "layout_ai_block_height_pt": layout_ai_block_height_pt,
                                    "layout_ai_block_width_pt": layout_ai_block_width_pt,
                                    "descriptor_layout_behavior": "anchored",
                                    "descriptor_region_bbox": fitz.Rect(descriptor_region_rect) if isinstance(descriptor_region_rect, fitz.Rect) else None,
                                    "descriptor_region_type": descriptor_region_type,
                                    "descriptor_region_id": str(region_id or ""),
                                    "descriptor_ai_region_type": descriptor_ai_region_type,
                                    "descriptor_ai_region_id": descriptor_ai_region_id,
                                    "descriptor_band_role": descriptor_band_role,
                                    "descriptor_structural_role": descriptor_structural_role,
                                    "descriptor_layout_behavior": descriptor_layout_behavior or "anchored",
                                    "descriptor_attachment_target_id": descriptor_attachment_target_id,
                                    "descriptor_group_ids": descriptor_group_ids,
                                    "descriptor_group_render_mode": str((descriptor_block or {}).get("group_render_mode") or ""),
                                    "descriptor_typographic_class": descriptor_typographic_class,
                                    "descriptor_visual_text": dict((descriptor_block or {}).get("visual_text") or {}),
                                    "descriptor_visual_text_object": dict(descriptor_visual_object or {}),
                                    "descriptor_visual_text_group": dict(descriptor_visual_group or {}),
                                    "descriptor_section_id": descriptor_section_id,
                                    "descriptor_page_organization": descriptor_page_organization,
                                    "descriptor_reconstruction_plan": descriptor_reconstruction_plan,
                                    "descriptor_v3": descriptor_v3,
                                    "descriptor_v3_contract": descriptor_v3_contract,
                                    "descriptor_v3_render_unit": dict(descriptor_v3_render_unit or {}),
                                    "descriptor_v3_containers": list(descriptor_v3_container_entries),
                                    "descriptor_v3_placement_constraints": list(descriptor_v3_block_constraints),
                                    "descriptor_v3_dependency_edges": list(descriptor_v3_block_edges),
                                    "descriptor_v3_spatial_clusters": dict(descriptor_v3_spatial_clusters or {}),
                                    "page_data": page_data,
                                    "anchor_target_bbox": fitz.Rect(anchor_target_rect) if isinstance(anchor_target_rect, fitz.Rect) else None,
                                    "anchor_preferred_side": anchor_preferred_side,
                                    "preserve_sentence_integrity": False,
                                    **self._alignment_payload(
                                        "left",
                                        source="mixed_inline_phrase",
                                    ),
                                }
                            )
                        continue
                    text_parts.extend(this_line_parts)
                    line_txt_src = ""
                    if block_is_translated:
                        line_txt_src = line.get("translated_text") or line.get("line_text") or ""
                    if not line_txt_src:
                        line_txt_src = " ".join(this_line_parts)
                    line_txt = self._clean_text_for_render(line_txt_src)
                    original_line_text = self._clean_text_for_render(
                        line.get("line_text")
                        or " ".join(
                            self._clean_text_for_render(ph.get("texte", ""))
                            for ph in (line.get("phrases", []) or [])
                            if self._clean_text_for_render(ph.get("texte", ""))
                        )
                    )
                    line_texts.append(line_txt)
                    line_marker = (line.get("leading_marker") or "").strip()
                    line_markers_used.append(line_marker)
                    lb = line.get("bbox") or block.get("bbox")
                    if line_has_hidden_phrase and line_visible_rects:
                        visible_union = fitz.Rect(line_visible_rects[0])
                        for vr in line_visible_rects[1:]:
                            visible_union = visible_union | vr
                        lb = [visible_union.x0 / self.pixel_to_point, visible_union.y0 / self.pixel_to_point, visible_union.x1 / self.pixel_to_point, visible_union.y1 / self.pixel_to_point]
                    line_style = self._merge_styles(style, {})
                    for ph0 in line.get("phrases", []):
                        if ph0.get("spans"):
                            picked = {}
                            for sp0 in ph0.get("spans", []):
                                if not isinstance(sp0, dict):
                                    continue
                                if sp0.get("skip_render"):
                                    continue
                                st0 = self._clean_text_for_render(sp0.get("texte", ""))
                                if re.fullmatch(r"(?:[•▪◦·\-\*]|\d+[.)]?|[A-Za-z][.)])", st0 or ""):
                                    continue
                                picked = sp0.get("style", {}) if isinstance(sp0.get("style"), dict) else {}
                                break
                            if not picked:
                                picked = ph0["spans"][0].get("style", {})
                            line_style = self._merge_styles(picked, line_style)
                            break
                    for ph0 in line.get("phrases", []):
                        for sp0 in ph0.get("spans", []) or []:
                            if not isinstance(sp0, dict):
                                continue
                            if sp0.get("skip_render"):
                                continue
                            st0 = self._clean_text_for_render(
                                (sp0.get("translated_text") or sp0.get("texte") or "")
                            )
                            if not st0 or re.fullmatch(r"(?:[•▪◦·\-\*]|\d+[.)]?|[A-Za-z][.)])", st0 or ""):
                                continue
                            sb0 = sp0.get("bbox") or sp0.get("bbox_pt")
                            if isinstance(sb0, (list, tuple)) and len(sb0) == 4:
                                try:
                                    srect0 = fitz.Rect([float(v) * self.pixel_to_point for v in sb0])
                                except Exception:
                                    srect0 = None
                            else:
                                srect0 = None
                            if isinstance(srect0, fitz.Rect) and srect0.get_area() > 0:
                                line_inline_segments.append(
                                    {
                                        "text": st0,
                                        "source_text": self._clean_text_for_render(sp0.get("texte", "")),
                                        "bbox": fitz.Rect(srect0),
                                        "style": self._merge_styles(
                                            sp0.get("style", {}) if isinstance(sp0.get("style"), dict) else {},
                                            line_style,
                                        ),
                                    }
                                )
                    line_entries.append(
                        {
                            "text": line_txt,
                            "source_text": original_line_text,
                            "marker": line_marker,
                            "bbox": lb,
                            "indent_px": float(line.get("indent_px", 0.0) or 0.0),
                            "style": line_style,
                            "inline_style_segments": list(line_inline_segments),
                            "marker_bbox": None,
                            "marker_style": {},
                        }
                    )
                else:
                    # Keep structural line fidelity: even if phrase aggregation is empty,
                    # preserve translated/source line text from extracted layout.
                    fallback_line = ""
                    if block_is_translated:
                        fallback_line = line.get("translated_text") or line.get("line_text") or ""
                    if not fallback_line:
                        fallback_line = line.get("line_text") or ""
                    fallback_line = self._clean_text_for_render(fallback_line)
                    if fallback_line:
                        original_line_text = self._clean_text_for_render(
                            line.get("line_text")
                            or " ".join(
                                self._clean_text_for_render(ph.get("texte", ""))
                                for ph in (line.get("phrases", []) or [])
                                if self._clean_text_for_render(ph.get("texte", ""))
                            )
                        )
                        line_texts.append(fallback_line)
                        line_marker = (line.get("leading_marker") or "").strip()
                        line_markers_used.append(line_marker)
                        lb = line.get("bbox") or block.get("bbox")
                        line_entries.append(
                            {
                                "text": fallback_line,
                                "source_text": original_line_text,
                                "marker": line_marker,
                                "bbox": lb,
                                "indent_px": float(line.get("indent_px", 0.0) or 0.0),
                                "style": self._merge_styles(style, {}),
                                "inline_style_segments": [],
                                "marker_bbox": None,
                                "marker_style": {},
                            }
                        )
                    if line_marker and line_entries:
                        for ph0 in line.get("phrases", []):
                            for sp0 in ph0.get("spans", []) or []:
                                st0 = self._clean_text_for_render(sp0.get("texte", ""))
                                if st0 == line_marker:
                                    mbb = sp0.get("bbox")
                                    if isinstance(mbb, (list, tuple)) and len(mbb) == 4:
                                        line_entries[-1]["marker_bbox"] = mbb
                                        line_entries[-1]["marker_style"] = sp0.get("style", {}) if isinstance(sp0.get("style"), dict) else {}
                                        break
                            if line_entries[-1].get("marker_bbox") is not None:
                                break
            block_preferred_text = re.sub(r"\s+", " ", (block.get("translated_text") or "").strip())
            page_family = str(page_data.get("page_family") or ((page_data.get("layout") or {}).get("page_family")) or "").strip().lower()
            page_family_group = str(page_data.get("page_family_group") or ((page_data.get("layout") or {}).get("page_family_group")) or page_family).strip().lower()
            layout_type = str(page_data.get("layout_type") or ((page_data.get("layout") or {}).get("layout_type")) or "").strip().lower()
            document_type = str(page_data.get("document_type") or ((page_data.get("layout") or {}).get("document_type")) or "").strip().lower()
            page_case = page_data.get("page_case") or {}
            fallback_policy = str(page_case.get("fallback_policy") or "").strip().lower()
            prefer_paragraph_flow = str(block.get("render_policy") or "").strip().lower() == "paragraph_flow"
            anchored_figure_page = (
                layout_type in {"annotated_page", "table_dominant", "image_dominant", "mixed_blocks"}
                and document_type not in {"scientific_paper", "book_page", "manual_guide"}
            ) or page_family_group in {"body_with_figure", "body_with_diagram", "mixed_page", "table_page"}
            text_layout_page = bool(
                layout_type in {"single_column", "double_column", "text_heavy"}
                and fallback_policy != "safe_mixed"
                and (not anchored_figure_page or prefer_paragraph_flow)
            )
            paragraph_flow_mode = bool(block.get("translation_compose_mode") == "paragraph_flow" and block_role == "body")
            effective_block_alignment = block.get("alignment", "left")
            if descriptor_layout_behavior == "flow_in_band" and block_role == "body" and block_is_translated:
                paragraph_flow_mode = True
            if (
                not paragraph_flow_mode
                and block_role == "body"
                and document_type in {"scientific_paper", "book_page", "manual_guide"}
                and layout_type == "double_column"
            ):
                unit_type = str(block.get("unit_type") or "").strip().lower()
                looks_narrative = len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", block_preferred_text or "")) >= 12
                if unit_type not in {"short_label", "chart_label", "formula_label", "diagram_label", "reference_link", "citation"} and looks_narrative:
                    paragraph_flow_mode = True
            if (
                not paragraph_flow_mode
                and block_role == "body"
                and block_is_translated
                and str(paragraph_constraints.get("render_mode") or "").strip().lower() == "flow_in_region"
            ):
                paragraph_flow_mode = True
            if (
                not paragraph_flow_mode
                and block_role == "body"
                and block_is_translated
                and descriptor_region_type == "text_band"
            ):
                paragraph_flow_mode = True
            if not block_is_translated:
                paragraph_flow_mode = False
            if self._is_abbreviation_entry_role(descriptor_structural_role):
                paragraph_flow_mode = False
            if anchored_figure_page and block_role == "body" and not prefer_paragraph_flow:
                paragraph_flow_mode = False
            relation_flow_item = {
                "role": block_role,
                "translated_block": block_is_translated,
                "descriptor_typographic_class": descriptor_typographic_class,
                "descriptor_group_ids": descriptor_group_ids,
                "descriptor_region_type": descriptor_region_type,
                "descriptor_band_role": descriptor_band_role,
            }
            allow_relation_flow_override = self._should_allow_relation_flow_override(
                relation_flow_item,
                page_data,
                fallback_policy,
                anchored_figure_page,
                table_locked_block,
            )
            if fallback_policy == "safe_mixed" and block_role == "body" and not allow_relation_flow_override:
                paragraph_flow_mode = False
            if table_locked_block:
                paragraph_flow_mode = False
            if descriptor_band_role in {"annotation_band", "caption_band", "header_band", "legend_band", "axis_band"}:
                paragraph_flow_mode = False
            preserve_double_column_lineation = bool(
                block_is_translated
                and self._should_preserve_double_column_lineation(
                    page_data=page_data,
                    block=block,
                    descriptor_region_type=descriptor_region_type,
                    translated_text=block_preferred_text or " ".join(line_texts),
                    source_lines=line_texts,
                )
            )
            if (
                not preserve_double_column_lineation
                and block_is_translated
                and block_role == "body"
                and layout_type == "double_column"
                and document_type in {"scientific_paper", "book_page", "manual_guide"}
                and str(descriptor_region_type or "").strip().lower() in {"text", "text_band", "column"}
                and self._has_meaningful_line_style_variation(line_entries)
                and len(line_texts or []) >= 4
            ):
                preserve_double_column_lineation = True
            if (not preserve_double_column_lineation) and block_is_translated:
                preserve_double_column_lineation = bool(
                    self._should_keep_local_source_slot_geometry_for_anchored_body(
                        page_data=page_data,
                        block=block,
                        descriptor_region_type=descriptor_region_type,
                        descriptor_typographic_class=descriptor_typographic_class,
                        descriptor_structural_role=descriptor_structural_role,
                        line_entries=line_entries,
                        source=source,
                    )
                )
            if preserve_double_column_lineation:
                paragraph_flow_mode = False
            if block_is_translated and paragraph_flow_mode:
                # Paragraph-level translation mode: use block translated text, then compose.
                text = block_preferred_text or re.sub(r"\s+", " ", " ".join(text_parts)).strip()
            elif block_is_translated:
                # For translated layouts, rely on phrase/line aggregation to preserve local structure.
                text = re.sub(r"\s+", " ", " ".join(text_parts)).strip()
            else:
                text = block_preferred_text or re.sub(r"\s+", " ", " ".join(text_parts)).strip()
            text = self._clean_text_for_render(text)
            if not text:
                if translated_phrase_items:
                    items.extend(translated_phrase_items)
                continue
            bb_for_title = block.get("bbox", [0, 0, 10, 10])
            try:
                by0_pt = float(bb_for_title[1]) * self.pixel_to_point
            except Exception:
                by0_pt = page_h_pt
            if (
                block_role in {"body", "title", "section_heading", "header"}
                and by0_pt <= page_h_pt * 0.16
                and self._should_reorder_top_header_number(text)
            ):
                m_end_num = re.match(r"^(.+?)\s+(\d{1,3})$", text)
                if m_end_num:
                    text = self._clean_text_for_render(f"{m_end_num.group(2)} {m_end_num.group(1)}")
            heading_candidate = self._clean_text_for_render(" ".join(first_run_text_parts))
            if (
                heading_candidate
                and by0_pt <= page_h_pt * 0.16
                and self._should_reorder_top_header_number(heading_candidate)
            ):
                m_end_num_h = re.match(r"^(.+?)\s+(\d{1,3})$", heading_candidate)
                if m_end_num_h:
                    heading_candidate = self._clean_text_for_render(f"{m_end_num_h.group(2)} {m_end_num_h.group(1)}")
            heading_regex = re.match(
                r"^\s*((?:LA|LE|LES|THE)\s+[A-ZÀ-ÿ\s\-]{3,}\([^)]{1,32}\))\s+",
                text,
                flags=re.IGNORECASE,
            )
            heading_is_distinct = bool(
                block_role == "body"
                and heading_candidate
                and first_run_style
                and isinstance(first_run_bbox, fitz.Rect)
                and first_run_bbox.get_area() > 0
                and len(heading_candidate) <= 90
                and len(re.findall(r"[A-Za-zÀ-ÿ]", heading_candidate)) >= 3
                and (
                    sum(1 for c in heading_candidate if c.isalpha() and c.isupper())
                    / max(1, sum(1 for c in heading_candidate if c.isalpha()))
                ) >= 0.45
                and str(first_run_style.get("color", "")).lower() != str(style.get("color", "")).lower()
                and descriptor_region_type not in {"annotation_band", "caption_band", "header_band"}
            )
            if translated_phrase_items:
                items.extend(translated_phrase_items)
            if block_is_translated and block_role in {"diagram_text_label"} and translated_phrase_items:
                continue
            bb = block.get("bbox", [0, 0, 10, 10])
            bbox = fitz.Rect([float(v) * self.pixel_to_point for v in bb])
            if bbox.get_area() <= 0:
                continue
            if not slots:
                slots = [fitz.Rect(bbox)]
            slots.sort(key=lambda r: (r.y0, r.x0))
            hs = [max(6.0, r.height) for r in slots]
            slot_h = float(median(hs))
            gaps_x = []
            gaps_y = []
            row_tol = max(2.0, slot_h * 0.5)
            rows = []
            for s in slots:
                if not rows or abs(s.y0 - rows[-1][-1].y0) > row_tol:
                    rows.append([s])
                else:
                    rows[-1].append(s)
            for row in rows:
                row.sort(key=lambda r: r.x0)
                for i in range(1, len(row)):
                    gx = row[i].x0 - row[i - 1].x1
                    if gx >= 0:
                        gaps_x.append(gx)
            for i in range(1, len(rows)):
                gy = rows[i][0].y0 - rows[i - 1][0].y1
                if gy >= 0:
                    gaps_y.append(gy)
            # One red slot per visual row; each rendered line can then expand to the blue frame right edge.
            row_slots = []
            for row in rows:
                x0 = min(r.x0 for r in row)
                y0 = min(r.y0 for r in row)
                y1 = max(r.y1 for r in row)
                row_slots.append(fitz.Rect(x0, y0, x0 + max(10.0, bbox.x1 - x0), y1))
            if not row_slots:
                row_slots = [fitz.Rect(bbox)]

            preserve_block_left_anchor = bool(
                block_is_translated
                and block_role == "body"
                and layout_type == "double_column"
                and document_type in {"manual_guide", "book_page", "scientific_paper"}
                and descriptor_region_type in {"text", "text_band"}
                and block.get("translation_compose_mode") == "preserved"
            )
            if preserve_double_column_lineation:
                preserve_block_left_anchor = True
            if preserve_block_left_anchor:
                normalized_row_slots = []
                for row in row_slots:
                    normalized_row_slots.append(
                        fitz.Rect(
                            bbox.x0,
                            row.y0,
                            max(bbox.x0 + 10.0, bbox.x1),
                            row.y1,
                        )
                    )
                row_slots = normalized_row_slots or [fitz.Rect(bbox)]

            gap_x = float(median(gaps_x)) if gaps_x else max(2.0, slot_h * 0.25)
            gap_y = float(median(gaps_y)) if gaps_y else max(3.0, slot_h * 0.35)
            row_start_x = min(s.x0 for s in row_slots) if row_slots else bbox.x0
            letters = [c for c in text if c.isalpha()]
            upper_ratio = (sum(1 for c in letters if c.isupper()) / max(1, len(letters))) if letters else 0.0
            flags = style.get("flags", {}) if isinstance(style, dict) else {}
            is_title = bool(
                block.get("role") in {"header", "title", "section_heading"}
                or (len(text) <= 140 and (flags.get("bold") or upper_ratio >= 0.55))
            )
            if descriptor_structural_role in {"running_header", "section_title", "figure_title", "chart_title"}:
                is_title = True
            is_diagram_label = bool(
                block.get("role") == "diagram_label"
                or block.get("role") == "figure_label"
                or (
                bbox.y0 <= page_h_pt * 0.58
                and bbox.height <= 16.0
                and bbox.width <= page_w_pt * 0.70
                and len(text) <= 220
                )
            )
            if descriptor_region_type == "annotation_band":
                is_diagram_label = True
            if descriptor_structural_role in {"diagram_label", "chart_axis_label", "chart_tick_label", "chart_legend_label"}:
                is_diagram_label = True
            if block.get("role") == "diagram_text_label":
                is_diagram_label = False
                is_title = False
            if is_diagram_label:
                is_title = False
            line_markers = line_markers_used
            has_list_markers = any(bool(m) for m in line_markers)
            has_number_only_lines = any(
                bool(re.match(r"^\s*\d+[.)]?\s*$", self._clean_text_for_render(lt)))
                for lt in (line_texts or [])
            )
            marker_text_indent_pt = 0.0
            if line_entries:
                try:
                    marker_text_indent_pt = max(
                        0.0,
                        max(float(le.get("indent_px", 0.0) or 0.0) for le in line_entries) * self.pixel_to_point,
                    )
                except Exception:
                    marker_text_indent_pt = 0.0
            is_structural_role = block.get("role") in {
                "title",
                "section_heading",
                "header",
                "footer",
                "figure_caption",
                "diagram_label",
                "diagram_text_label",
            }
            if descriptor_structural_role in {
                "running_header",
                "running_footer",
                "section_title",
                "figure_caption",
                "figure_title",
                "diagram_label",
                "chart_axis_label",
                "chart_tick_label",
                "chart_legend_label",
                "table_header_cell",
            }:
                is_structural_role = True
            has_hard_breaks = any(
                bool((ln.get("hard_break_before") if isinstance(ln, dict) else False))
                for ln in block.get("lines", [])
            )
            item_lang = block.get("language") or page_lang or self._infer_text_lang(text)
            editorial_header_row_texts = []
            if (
                block_role == "header"
                and layout_type == "double_column"
                and document_type in {"manual_guide", "book_page", "scientific_paper"}
                and line_entries
            ):
                row_groups = []
                header_row_tol = max(2.0, slot_h * 0.5)
                for le in line_entries:
                    bb = le.get("bbox")
                    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                        continue
                    y0_px = float(bb[1])
                    if not row_groups or abs(y0_px - row_groups[-1]["y0"]) > header_row_tol / max(self.pixel_to_point, 1e-6):
                        row_groups.append({"y0": y0_px, "entries": [le]})
                    else:
                        row_groups[-1]["entries"].append(le)
                for row in row_groups:
                    entries = []
                    for le in row["entries"]:
                        bb = le.get("bbox")
                        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                            continue
                        txt = self._clean_text_for_render(
                            le.get("translated_text")
                            or le.get("text")
                            or ""
                        ).strip()
                        phrase_candidates = []
                        for ph in le.get("phrases", []) or []:
                            ptxt = self._clean_text_for_render(
                                ph.get("translated_text")
                                or ph.get("texte")
                                or ""
                            ).strip()
                            if ptxt:
                                phrase_candidates.append(ptxt)
                        if phrase_candidates:
                            txt = " ".join(phrase_candidates).strip()
                        if txt:
                            entries.append((float(bb[0]), txt))
                    if entries:
                        entries.sort(key=lambda it: it[0])
                        merged_row_text = self._clean_text_for_render(" ".join(txt for _, txt in entries)).strip()
                        if merged_row_text:
                            editorial_header_row_texts.append(merged_row_text)

            source_lines_for_render = []
            source_line_styles_for_render = []
            for i, raw_line in enumerate(line_texts):
                lt = self._clean_text_for_render(raw_line)
                line_style_for_render = self._merge_styles(
                    (line_entries[i].get("style", {}) if i < len(line_entries) and isinstance(line_entries[i], dict) else {}),
                    style,
                )
                marker = self._normalize_leading_marker(line_markers[i] if i < len(line_markers) else "")
                if marker:
                    has_any_marker = bool(
                        re.match(r"^\s*(?:[•▪◦·\-\*]|\d+[.)]|[A-Za-z][.)])\s+", lt or "")
                    )
                    if not has_any_marker:
                        if re.fullmatch(r"[•▪◦·\-\*]", marker):
                            lt = f"{marker}    {lt}".strip()
                        else:
                            lt = f"{marker}   {lt}".strip()
                    is_numbered_marker = bool(re.fullmatch(r"(?:\d+[.)]?|[A-Za-z][.)])", marker))
                    # Keep bullet markers inline so line start remains anchored to blue bbox.
                    # Detach only numbered/list ordinal markers that need strict standalone positioning.
                    if is_numbered_marker and page_role != "toc" and block_is_translated and block_role == "body" and i < len(line_entries):
                        le0 = line_entries[i]
                        mbb = le0.get("marker_bbox")
                        if isinstance(mbb, (list, tuple)) and len(mbb) == 4:
                            mrect = fitz.Rect([float(v) * self.pixel_to_point for v in mbb])
                            if mrect.get_area() > 0:
                                items.append(
                                    {
                                        "text": marker,
                                        "source_lines": [marker],
                                        "preserve_linebreaks": False,
                                        "bbox": fitz.Rect(mrect),
                                        "slots": [fitz.Rect(mrect)],
                                        "slot_w_pt": max(6.0, mrect.width),
                                        "slot_h_pt": max(6.0, mrect.height),
                                        "slot_gap_x_pt": max(1.5, mrect.height * 0.2),
                                        "slot_gap_y_pt": max(2.0, mrect.height * 0.25),
                                        "row_start_x_pt": mrect.x0,
                                        "style": self._merge_styles(le0.get("marker_style", {}), le0.get("style", {})),
                                        "source": source,
                                        "source_text": marker,
                                        "alignment": "left",
                                        "justify_explicit": False,
                                        "role": "list_marker",
                                        "lang": item_lang,
                                        "is_title": False,
                                        "is_diagram_label": False,
                                    }
                                )
                        lt = re.sub(r"^\s*(?:\d+[.)]|[A-Za-z][.)])\s*", "", lt).strip()
                # If previous line is a dedicated marker-only line (e.g. "1", "2"),
                # strip accidental duplicated leading marker in current text line.
                if i > 0 and i - 1 < len(line_entries):
                    prev_txt = self._clean_text_for_render(line_entries[i - 1].get("text", ""))
                    if re.match(r"^\s*(?:\d+[.)]?|[•▪◦·\-\*])\s*$", prev_txt):
                        lt = re.sub(r"^\s*(?:\d+[.)]?\s+|[•▪◦·\-\*]\s+)", "", lt).strip()
                source_lines_for_render.append(lt)
                source_line_styles_for_render.append(dict(line_style_for_render))
            if editorial_header_row_texts:
                source_lines_for_render = editorial_header_row_texts
                source_line_styles_for_render = [self._merge_styles(style, {}) for _ in editorial_header_row_texts]
                text = "\n".join(editorial_header_row_texts)

            # Keep numeric/list markers (e.g. standalone "1", "2") fixed at original
            # location; remove them from flowing body text to avoid displacement.
            if block_is_translated and block_role == "body" and line_entries:
                kept_lines = []
                kept_styles = []
                for i, lt in enumerate(source_lines_for_render):
                    le = line_entries[i] if i < len(line_entries) else {}
                    ltxt = self._clean_text_for_render(le.get("text", lt))
                    bb0 = le.get("bbox")
                    bw0 = 0.0
                    if isinstance(bb0, (list, tuple)) and len(bb0) == 4:
                        bw0 = max(0.0, float(bb0[2]) - float(bb0[0]))
                    is_marker_only = bool(
                        re.match(r"^\s*(?:\d+[.)]?|[•▪◦·\-\*])\s*$", ltxt)
                        and bw0 <= 42.0
                    )
                    if is_marker_only and page_role != "toc":
                        bb = le.get("bbox")
                        if isinstance(bb, (list, tuple)) and len(bb) == 4:
                            mrect = fitz.Rect([float(v) * self.pixel_to_point for v in bb])
                            if mrect.get_area() > 0:
                                items.append(
                                    {
                                        "text": ltxt,
                                        "source_lines": [ltxt],
                                        "preserve_linebreaks": False,
                                        "bbox": fitz.Rect(mrect),
                                        "slots": [fitz.Rect(mrect)],
                                        "slot_w_pt": max(8.0, mrect.width),
                                        "slot_h_pt": max(6.0, mrect.height),
                                        "slot_gap_x_pt": max(1.5, mrect.height * 0.2),
                                        "slot_gap_y_pt": max(2.0, mrect.height * 0.25),
                                        "row_start_x_pt": mrect.x0,
                                        "style": self._merge_styles(le.get("style", {}), style),
                                        "source": source,
                                        "source_text": ltxt,
                                        "alignment": "left",
                                        "justify_explicit": False,
                                        "role": "list_marker",
                                        "lang": item_lang,
                                        "is_title": False,
                                        "is_diagram_label": False,
                                    }
                                )
                        continue
                    kept_lines.append(lt)
                    if i < len(source_line_styles_for_render):
                        kept_styles.append(dict(source_line_styles_for_render[i]))
                    else:
                        kept_styles.append(self._merge_styles(style, {}))
                source_lines_for_render = kept_lines
                source_line_styles_for_render = kept_styles
            preserve_linebreaks = bool(
                len(line_texts) >= 1
                and (
                    (block_is_translated and block_role == "body")
                    or is_structural_role
                    or has_list_markers
                    or has_number_only_lines
                    or has_hard_breaks
                )
            )
            if not block_is_translated:
                # Non-translated content must keep original line geometry and typography.
                preserve_linebreaks = True
            if paragraph_flow_mode:
                preserve_linebreaks = False
            word_like_tokens = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", text or "")
            allow_vertical_expand = bool(
                block_role == "body"
                and block_is_translated
                and paragraph_flow_mode
                and text_layout_page
                and len(word_like_tokens) >= 16
                and block.get("render_policy") != "anchored_text"
            )
            if descriptor_region_type == "text_band" and paragraph_flow_mode and block_role == "body":
                allow_vertical_expand = True
            if paragraph_flow_mode and block_role == "body":
                allow_vertical_expand = bool(
                    allow_vertical_expand or paragraph_constraints.get("allow_vertical_expand")
                )
            if self._is_abbreviation_value_role(descriptor_structural_role):
                allow_vertical_expand = True
            if table_locked_block:
                allow_vertical_expand = False
            if (
                block_role == "body"
                and block_is_translated
                and layout_type in {"single_column", "double_column", "text_heavy"}
                and document_type in {"scientific_paper", "book_page", "manual_guide"}
                and str(block.get("unit_type") or "").strip().lower() == "narrative_body"
                and str(effective_block_alignment or "left").strip().lower() in {"right", "center"}
                and not bool(block.get("justify_explicit"))
            ):
                effective_block_alignment = "left"
            accent_color = ""
            for c in span_color_sequence[:18]:
                if str(c).strip():
                    accent_color = c
                    break
            if block_role == "body" and heading_regex:
                heading_text_rx = self._clean_text_for_render(heading_regex.group(1))
                body_text_rx = self._clean_text_for_render(text[len(heading_regex.group(0)):])
                heading_color = ""
                for c in span_color_sequence[:18]:
                    if str(c).strip():
                        heading_color = c
                        break
                if heading_text_rx and body_text_rx:
                    hstyle = self._merge_styles(style, {})
                    if heading_color:
                        hstyle["color"] = heading_color
                    heading_bbox = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, min(bbox.y1, bbox.y0 + max(slot_h * 1.35, 16.0)))
                    items.append(
                        {
                            "text": heading_text_rx,
                            "source_lines": [heading_text_rx],
                            "preserve_linebreaks": False,
                            "bbox": fitz.Rect(heading_bbox),
                            "slots": [fitz.Rect(heading_bbox)],
                            "slot_w_pt": max(10.0, heading_bbox.width),
                            "slot_h_pt": max(6.0, heading_bbox.height),
                            "slot_gap_x_pt": gap_x,
                            "slot_gap_y_pt": gap_y,
                            "row_start_x_pt": heading_bbox.x0,
                            "style": hstyle,
                            "source": source,
                            "role": "section_heading",
                            "lang": item_lang,
                            "is_title": True,
                            "is_diagram_label": False,
                            "accent_color": heading_color or accent_color,
                            **self._alignment_payload(effective_block_alignment, source="block"),
                        }
                    )
                    text = body_text_rx
                    bbox = fitz.Rect(bbox.x0, min(bbox.y1, heading_bbox.y1 + 1.0), bbox.x1, bbox.y1)
                    row_slots = [fitz.Rect(r) for r in row_slots if r.y0 >= heading_bbox.y1 - 0.5]
                    if not row_slots:
                        row_slots = [fitz.Rect(bbox)]
            if heading_is_distinct and not heading_regex:
                heading_text = heading_candidate
                body_text = text
                if body_text.lower().startswith(heading_text.lower()):
                    body_text = self._clean_text_for_render(body_text[len(heading_text):].lstrip(" .:-"))
                if heading_text:
                    items.append(
                        {
                            "text": heading_text,
                            "source_lines": [heading_text],
                            "preserve_linebreaks": False,
                            "bbox": fitz.Rect(first_run_bbox),
                            "slots": [fitz.Rect(first_run_bbox)],
                            "slot_w_pt": max(10.0, first_run_bbox.width),
                            "slot_h_pt": max(6.0, first_run_bbox.height),
                            "slot_gap_x_pt": gap_x,
                            "slot_gap_y_pt": gap_y,
                            "row_start_x_pt": first_run_bbox.x0,
                            "style": self._merge_styles(first_run_style, {}),
                            "source": source,
                            "role": "section_heading",
                            "lang": item_lang,
                            "is_title": True,
                            "is_diagram_label": False,
                            "accent_color": first_run_style.get("color", "") if isinstance(first_run_style, dict) else "",
                            **self._alignment_payload(effective_block_alignment, source="block"),
                        }
                    )
                    bbox = fitz.Rect(bbox.x0, min(bbox.y1, first_run_bbox.y1 + 1.0), bbox.x1, bbox.y1)
                    row_slots = [fitz.Rect(r) for r in row_slots if r.y0 >= first_run_bbox.y1 - 0.5]
                    if not row_slots:
                        row_slots = [fitz.Rect(bbox)]
                text = body_text or text
            # For non-translated figure/image labels, preserve exact source line boxes
            # (red bboxes) instead of block-level recomposition.
            use_strict_line_items = bool(
                (
                    (not block_is_translated)
                    and (is_diagram_label or block_role in {"diagram_text_label", "equation_inline", "title"})
                )
                or (
                    block_is_translated
                    and block_role == "header"
                )
                or (
                    block_is_translated
                    and block_role == "title"
                    and layout_type in {"double_column", "table_dominant"}
                )
                or (
                    anchored_figure_page
                    and block_is_translated
                    and (
                        block_role in {"title", "figure_caption", "diagram_text_label", "diagram_label", "header"}
                        or (block_role == "body" and source == "native")
                    )
                )
                or (
                    block_is_translated
                    and block.get("render_policy") == "anchored_text"
                    and (
                        block_role in {"section_heading", "title", "figure_caption", "diagram_text_label"}
                        or (
                            block_role == "body"
                            and (
                                self._allow_strict_line_items_for_anchored_text_body(
                                    block_role,
                                    descriptor_typographic_class,
                                    descriptor_structural_role,
                                )
                                or self._should_keep_strict_line_items_for_anchored_body(
                                    line_entries,
                                    source,
                                )
                            )
                        )
                    )
                )
                or self._is_abbreviation_key_role(descriptor_structural_role)
                or descriptor_region_type in {"annotation_band", "caption_band"}
                or table_locked_block
            )
            keep_multiline_locked_editorial_block = bool(
                self._should_keep_multiline_locked_editorial_block(
                    page_data=page_data,
                    block=block,
                    descriptor_layout_behavior=descriptor_layout_behavior,
                    descriptor_structural_role=descriptor_structural_role,
                    descriptor_typographic_class=descriptor_typographic_class,
                    line_entries=line_entries,
                    source=source,
                    translated_block=block_is_translated,
                )
            )
            if keep_multiline_locked_editorial_block:
                use_strict_line_items = False
            if use_strict_line_items:
                pushed_any = False
                for le in line_entries:
                    lt = self._clean_text_for_render(le.get("text", ""))
                    bb = le.get("bbox")
                    if not lt or not isinstance(bb, (list, tuple)) or len(bb) != 4:
                        continue
                    lr = fitz.Rect([float(v) * self.pixel_to_point for v in bb])
                    if lr.get_area() <= 0:
                        continue
                    lst = self._merge_styles(le.get("style", {}), style)
                    exact_slot_item = {
                            "text": lt,
                            "source_lines": [lt],
                            "preserve_linebreaks": True,
                            "use_structured_source_lines": True,
                            "has_number_markers": False,
                            "allow_line_overflow": True,
                            "keep_exact_line": True,
                            "paragraph_flow_mode": False,
                            "strict_bbox_mode": True,
                            "bbox": fitz.Rect(lr),
                            "slots": [fitz.Rect(lr)],
                            "slot_w_pt": max(8.0, lr.width),
                            "slot_h_pt": max(6.0, lr.height),
                            "slot_gap_x_pt": gap_x,
                            "slot_gap_y_pt": gap_y,
                            "row_start_x_pt": lr.x0,
                            "style": lst,
                            "source": source,
                            "source_text": self._clean_text_for_render(le.get("source_text", "")),
                            "role": block_role,
                            "lang": item_lang,
                            "is_title": bool(block_role in {"title", "section_heading", "header"}),
                            "is_diagram_label": bool(is_diagram_label),
                            "accent_color": accent_color,
                            "translated_block": bool(block_is_translated),
                            "source_block_id": block_id,
                            "source_line_count": 1,
                            "layout_ai_label_hint": layout_ai_label_hint,
                            "layout_ai_text_line_height_pt": layout_ai_text_line_height_pt,
                            "layout_ai_text_line_width_pt": layout_ai_text_line_width_pt,
                            "layout_ai_block_height_pt": layout_ai_block_height_pt,
                            "layout_ai_block_width_pt": layout_ai_block_width_pt,
                            "inline_style_segments": list(le.get("inline_style_segments") or []),
                            "exact_slot_render": bool(
                                block_is_translated
                                and (
                                    descriptor_region_type in {"annotation_band", "caption_band"}
                                    or descriptor_band_role in {"annotation_band", "legend_band", "axis_band"}
                                    or str((descriptor_block or {}).get("group_render_mode") or "").strip().lower()
                                    in {"annotation_group", "chart_legend_group", "chart_axis_group", "chart_series_group"}
                                    or block_role in {"diagram_text_label", "diagram_label"}
                                )
                            ),
                            "descriptor_region_bbox": fitz.Rect(descriptor_region_rect) if isinstance(descriptor_region_rect, fitz.Rect) else None,
                            "descriptor_region_type": descriptor_region_type,
                            "descriptor_region_id": str(region_id or ""),
                            "descriptor_ai_region_type": descriptor_ai_region_type,
                            "descriptor_ai_region_id": descriptor_ai_region_id,
                            "descriptor_band_role": descriptor_band_role,
                            "descriptor_structural_role": descriptor_structural_role,
                            "descriptor_layout_behavior": descriptor_layout_behavior,
                            "descriptor_attachment_target_id": descriptor_attachment_target_id,
                            "descriptor_group_ids": descriptor_group_ids,
                            "descriptor_group_render_mode": str((descriptor_block or {}).get("group_render_mode") or ""),
                            "descriptor_typographic_class": descriptor_typographic_class,
                            "descriptor_visual_text": dict((descriptor_block or {}).get("visual_text") or {}),
                            "descriptor_visual_text_object": dict(descriptor_visual_object or {}),
                            "descriptor_visual_text_group": dict(descriptor_visual_group or {}),
                            "descriptor_section_id": descriptor_section_id,
                            "descriptor_page_organization": descriptor_page_organization,
                            "descriptor_reconstruction_plan": descriptor_reconstruction_plan,
                            "descriptor_v3": descriptor_v3,
                            "descriptor_v3_contract": descriptor_v3_contract,
                            "descriptor_v3_render_unit": dict(descriptor_v3_render_unit or {}),
                            "descriptor_v3_containers": list(descriptor_v3_container_entries),
                            "descriptor_v3_placement_constraints": list(descriptor_v3_block_constraints),
                            "descriptor_v3_dependency_edges": list(descriptor_v3_block_edges),
                            "descriptor_v3_spatial_clusters": dict(descriptor_v3_spatial_clusters or {}),
                            "page_data": page_data,
                            "anchor_target_bbox": fitz.Rect(anchor_target_rect) if isinstance(anchor_target_rect, fitz.Rect) else None,
                            "anchor_preferred_side": anchor_preferred_side,
                            "preserve_sentence_integrity": bool(preserve_sentence_integrity),
                            **self._alignment_payload("left", source="strict_line"),
                        }
                    exact_slot_item["exact_slot_render"] = self._item_requires_exact_slot_render(exact_slot_item)
                    items.append(exact_slot_item)
                    pushed_any = True
                if pushed_any:
                    continue
            structured_source_lines = [ln for ln in (source_lines_for_render or block.get("line_texts") or line_texts) if str(ln).strip()]
            use_structured_source_lines = bool(
                (not paragraph_flow_mode)
                and structured_source_lines
                and (
                    preserve_linebreaks
                    or (not block_is_translated)
                    or has_list_markers
                    or has_hard_breaks
                    or has_number_only_lines
                )
            )
            items.append(
                {
                    "text": text,
                    "source_lines": structured_source_lines,
                    "source_line_styles": [dict(s) for s in source_line_styles_for_render[: len(structured_source_lines)]],
                    "preserve_linebreaks": preserve_linebreaks,
                    "use_structured_source_lines": use_structured_source_lines,
                    "has_number_markers": bool(has_number_only_lines),
                    "marker_text_indent_pt": float(marker_text_indent_pt),
                    "allow_line_overflow": bool(
                        (not paragraph_flow_mode) and block_is_translated and (is_structural_role or has_list_markers or has_number_only_lines)
                    ),
                    "paragraph_flow_mode": bool(paragraph_flow_mode),
                    "strict_bbox_mode": True,
                    "allow_expand_to_page_right": bool(block_is_translated and block_role == "body"),
                    "allow_vertical_expand": bool(allow_vertical_expand),
                    "expand_right_limit_pt": float(page_blue_right_pt),
                    "descriptor_region_bbox": fitz.Rect(descriptor_region_rect) if isinstance(descriptor_region_rect, fitz.Rect) else None,
                    "descriptor_region_type": descriptor_region_type,
                    "descriptor_region_id": str(region_id or ""),
                    "descriptor_ai_region_type": descriptor_ai_region_type,
                    "descriptor_ai_region_id": descriptor_ai_region_id,
                    "descriptor_band_role": descriptor_band_role,
                    "descriptor_structural_role": descriptor_structural_role,
                    "descriptor_layout_behavior": descriptor_layout_behavior,
                    "descriptor_attachment_target_id": descriptor_attachment_target_id,
                    "descriptor_group_ids": descriptor_group_ids,
                    "descriptor_group_render_mode": str((descriptor_block or {}).get("group_render_mode") or ""),
                    "descriptor_typographic_class": descriptor_typographic_class,
                    "descriptor_visual_text": dict((descriptor_block or {}).get("visual_text") or {}),
                    "descriptor_visual_text_object": dict(descriptor_visual_object or {}),
                    "descriptor_visual_text_group": dict(descriptor_visual_group or {}),
                    "descriptor_section_id": descriptor_section_id,
                    "descriptor_page_organization": descriptor_page_organization,
                    "descriptor_reconstruction_plan": descriptor_reconstruction_plan,
                    "descriptor_v3": descriptor_v3,
                    "descriptor_v3_contract": descriptor_v3_contract,
                    "descriptor_v3_render_unit": dict(descriptor_v3_render_unit or {}),
                    "descriptor_v3_containers": list(descriptor_v3_container_entries),
                    "descriptor_v3_placement_constraints": list(descriptor_v3_block_constraints),
                    "descriptor_v3_dependency_edges": list(descriptor_v3_block_edges),
                    "descriptor_v3_spatial_clusters": dict(descriptor_v3_spatial_clusters or {}),
                    "page_data": page_data,
                    "anchor_target_bbox": fitz.Rect(anchor_target_rect) if isinstance(anchor_target_rect, fitz.Rect) else None,
                    "anchor_preferred_side": anchor_preferred_side,
                    "descriptor_paragraph_id": str((paragraph_group or {}).get("id") or ""),
                    "descriptor_no_internal_sentence_break": bool(
                        any(
                            str(constraint.get("type") or "") == "no_internal_sentence_break"
                            for constraint in block_constraints
                        )
                    ),
                    "preserve_sentence_integrity": bool(
                        False if keep_multiline_locked_editorial_block else preserve_sentence_integrity
                    ),
                    "prefer_local_multiline_reflow": bool(keep_multiline_locked_editorial_block),
                    "bbox": bbox,
                    "slots": row_slots,
                    "slot_w_pt": max(10.0, bbox.width),
                    "slot_h_pt": slot_h,
                    "slot_gap_x_pt": gap_x,
                    "slot_gap_y_pt": gap_y,
                    "row_start_x_pt": row_start_x,
                    "preferred_left_x_pt": bbox.x0,
                    "preserve_block_left_anchor": bool(preserve_block_left_anchor),
                    "keep_source_slot_geometry": bool(preserve_double_column_lineation),
                    "preserve_line_style_variation": bool(
                        preserve_double_column_lineation and self._has_meaningful_line_style_variation(line_entries)
                    ),
                    "style": self._merge_styles(style, {}),
                    "source": source,
                    "source_text": self._get_block_source_text(block),
                    "role": block.get("role", "body"),
                    "lang": item_lang,
                    "is_title": is_title,
                    "is_diagram_label": is_diagram_label,
                    "accent_color": accent_color,
                    "translated_block": bool(block_is_translated),
                    "source_block_id": block_id,
                    "source_line_count": len(structured_source_lines),
                    "layout_ai_label_hint": layout_ai_label_hint,
                    "layout_ai_text_line_height_pt": layout_ai_text_line_height_pt,
                    "layout_ai_text_line_width_pt": layout_ai_text_line_width_pt,
                    "layout_ai_block_height_pt": layout_ai_block_height_pt,
                    "layout_ai_block_width_pt": layout_ai_block_width_pt,
                    **self._alignment_payload(effective_block_alignment, source="block"),
                }
            )

            # Header often contains "page-number + title" merged in one OCR line.
            # On editorial double-column pages, keeping the original merged line
            # preserves the top-band hierarchy better than splitting number/title.
            split_header_number = not (
                block.get("role") == "header"
                and layout_type == "double_column"
                and document_type in {"manual_guide", "book_page", "scientific_paper"}
            )
            if block.get("role") == "header" and split_header_number:
                m = re.match(r"^\s*(\d{1,3})\s+(.+)$", text)
                if not m:
                    m_end = re.match(r"^\s*(.+?)\s+(\d{1,3})\s*$", text)
                    if m_end and len(m_end.group(1).split()) >= 2:
                        text = f"{m_end.group(2)} {m_end.group(1)}"
                        items[-1]["text"] = text
                        m = re.match(r"^\s*(\d{1,3})\s+(.+)$", text)
                if m:
                    num_txt = m.group(1).strip()
                    title_txt = self._clean_text_for_render(m.group(2).strip())
                    if title_txt:
                        items[-1]["text"] = title_txt
                        items[-1]["source_lines"] = [title_txt]
                        items[-1]["preserve_linebreaks"] = False
                        items[-1]["use_structured_source_lines"] = False
                        items[-1]["alignment"] = "center"
                        num_w = max(26.0, min(44.0, bbox.width * 0.22))
                        num_bbox = fitz.Rect(bbox.x0, bbox.y0, bbox.x0 + num_w, bbox.y1)
                        items.append(
                            {
                                "text": num_txt,
                                "source_lines": [num_txt],
                                "bbox": num_bbox,
                                "slots": [fitz.Rect(num_bbox)],
                                "slot_w_pt": num_w,
                                "slot_h_pt": slot_h,
                                "slot_gap_x_pt": gap_x,
                                "slot_gap_y_pt": gap_y,
                                "row_start_x_pt": num_bbox.x0,
                                "style": self._merge_styles(style, {}),
                                "source": source,
                                "role": "header",
                                "is_title": True,
                                "is_diagram_label": False,
                                **self._alignment_payload("right", source="header_page_number"),
                            }
                        )
        self._resolve_header_item_collisions(items, page_w_pt=page_w_pt)
        for it in items:
            try:
                bb = it.get("bbox")
                txt = self._clean_text_for_render(it.get("text", ""))
                if not isinstance(bb, fitz.Rect) or not txt:
                    continue
                if bb.y0 <= page_h_pt * 0.16 and self._should_reorder_top_header_number(txt):
                    m = re.match(r"^(.+?)\s+(\d{1,3})$", txt)
                    if m:
                        it["text"] = self._clean_text_for_render(f"{m.group(2)} {m.group(1)}")
            except Exception:
                continue
        items.sort(key=lambda it: (it["bbox"].y0, it["bbox"].x0))
        return items

    def _resolve_page_lang(self, page_data):
        raw = (page_data.get("language") or page_data.get("detected_language") or "").strip().lower()
        if raw in {"fr", "french"}:
            return "fr"
        if raw in {"en", "english"}:
            return "en"
        if raw in {"es", "spanish"}:
            return "es"
        if raw in {"de", "german"}:
            return "de"
        if raw in {"it", "italian"}:
            return "it"
        if raw in {"pt", "portuguese"}:
            return "pt"
        return ""

    def _infer_text_lang(self, text):
        s = (text or "").lower()
        if re.search(r"[àâçéèêëîïôûùüÿœ]", s):
            return "fr"
        if re.search(r"[ñáéíóú¿¡]", s):
            return "es"
        if re.search(r"[äöüß]", s):
            return "de"
        return "en"

    def _clean_text_for_render(self, text):
        s = unicodedata.normalize("NFC", text or "")
        s = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        # Common OCR artifacts on French documents.
        fixes = {
            "c-ur": "coeur",
            "n-ud": "noeud",
            "n-uds": "noeuds",
            "c-urs": "coeurs",
        }
        for k, v in fixes.items():
            s = s.replace(k, v)
        # Remove immediate duplicated sentence/chunk.
        chunks = [c.strip() for c in re.split(r"(?<=[\.\!\?;:])\s+", s) if c.strip()]
        if chunks:
            dedup = []
            for c in chunks:
                key = re.sub(r"\W+", "", c).lower()
                if dedup and re.sub(r"\W+", "", dedup[-1]).lower() == key:
                    continue
                dedup.append(c)
            s = " ".join(dedup)
        return s

    def _normalize_leading_marker(self, marker):
        m = (marker or "").strip()
        return m

    def _remove_skip_span_tokens_from_text(self, text, spans):
        s = self._clean_text_for_render(text)
        if not s:
            return s
        skip_tokens = []
        for sp in spans or []:
            if not isinstance(sp, dict) or not sp.get("skip_render"):
                continue
            tok = self._clean_text_for_render(sp.get("texte", ""))
            if tok:
                skip_tokens.append(tok)
        for tok in skip_tokens:
            pattern = re.escape(tok).replace(r"\ ", r"\s+")
            s = re.sub(pattern, " ", s, count=1, flags=re.IGNORECASE)
            s = self._clean_text_for_render(s)
        return s

    def _phrase_text_for_render(self, phrase):
        # Prefer translated text when available.
        t = re.sub(r"\s+", " ", (phrase.get("translated_text") or phrase.get("texte") or "").strip())
        spans = phrase.get("spans", []) if isinstance(phrase, dict) else []
        if not spans:
            return t
        if not any(bool(sp.get("skip_render")) for sp in spans if isinstance(sp, dict)):
            return t
        # If immutable/symbol spans were marked skip_render, always strip those tokens
        # from phrase text to avoid duplication with immutable overlays.
        if t:
            stripped = self._remove_skip_span_tokens_from_text(t, spans)
            if stripped:
                return stripped
        # Fallback: rebuild from non-skipped spans.
        kept = []
        for sp in spans:
            if not isinstance(sp, dict):
                continue
            if sp.get("skip_render"):
                continue
            st = self._clean_text_for_render(sp.get("translated_text") or sp.get("texte") or "")
            if st:
                kept.append(st)
        if not kept:
            return ""
        return self._clean_text_for_render(" ".join(kept))

    def _consume_words_for_width(self, words, max_w, fontsize, fontname, fontfile):
        if not words:
            return "", []
        first = words[0]
        if self._measure_text_width(first, fontsize, fontname, fontfile) > max_w:
            chunk = ""
            idx = 0
            for ch in first:
                cand = chunk + ch
                if chunk and self._measure_text_width(cand, fontsize, fontname, fontfile) > max_w:
                    break
                chunk = cand
                idx += 1
            if not chunk:
                chunk = first[0]
                idx = 1
            rest = first[idx:]
            tail = words[1:]
            if rest:
                tail = [rest] + tail
            return chunk, tail
        current = first
        i = 1
        while i < len(words):
            cand = f"{current} {words[i]}"
            if self._measure_text_width(cand, fontsize, fontname, fontfile) <= max_w:
                current = cand
                i += 1
            else:
                break
        return current, words[i:]

    def _render_block_slots(self, page, item, anchor_y, left, right, zone_top, zone_bottom, override_text=None, render=True, forbidden_rects=None):
        text = self._clean_text_for_render(override_text if override_text is not None else item.get("text", "")).strip()
        if not text:
            return "", anchor_y, None, []
        forbidden_rects = forbidden_rects or []
        style = self._normalized_style_for_item(item)
        source = item["source"]
        _, fontfile, builtin, fontname = self._resolve_style_font(page, style, text=text)
        base_fs = self._normalized_fontsize_for_item(item, style, max(1.0, item["slot_h_pt"]), source)
        rgb = self._resolve_text_color(style, item)

        block_w = max(10.0, item["bbox"].width)
        x0 = max(left, min(item["bbox"].x0, right - block_w))
        descriptor_region_bbox = item.get("descriptor_region_bbox")
        descriptor_region_type = str(item.get("descriptor_region_type") or "").strip().lower()
        descriptor_ai_region_type = str(item.get("descriptor_ai_region_type") or "").strip().lower()
        descriptor_band_role = str(item.get("descriptor_band_role") or "").strip().lower()
        descriptor_structural_role = str(item.get("descriptor_structural_role") or "").strip().lower()
        descriptor_layout_behavior = str(item.get("descriptor_layout_behavior") or "").strip().lower()
        descriptor_group_ids = item.get("descriptor_group_ids") or {}
        descriptor_group_render_mode = str(item.get("descriptor_group_render_mode") or "").strip().lower()
        descriptor_typographic_class = str(item.get("descriptor_typographic_class") or "").strip().lower()
        descriptor_page_organization = item.get("descriptor_page_organization") or {}
        descriptor_reconstruction_plan = item.get("descriptor_reconstruction_plan") or {}
        descriptor_visual_text = item.get("descriptor_visual_text") or {}
        descriptor_visual_group = item.get("descriptor_visual_text_group") or {}
        anchor_target_bbox = item.get("anchor_target_bbox")
        anchor_preferred_side = str(item.get("anchor_preferred_side") or "").strip().lower()
        paragraph_chain_bbox = self._relation_group_bbox(item, "continues_paragraph")
        same_band_bbox = self._relation_group_bbox(item, "same_band")
        same_row_bbox = self._relation_group_bbox(item, "same_row")
        section_sibling_bbox = self._relation_group_bbox(item, "section_sibling")
        native_structure = descriptor_page_organization.get("native_structure") or {}
        role = item.get("role")
        header_like = bool(
            str(role or "").strip().lower() in {"header", "footer"}
            or descriptor_structural_role in {"running_header", "running_footer"}
            or descriptor_typographic_class in {"running_header", "running_footer"}
        )

        def _fitz_rect_from_any(bbox_like):
            if isinstance(bbox_like, fitz.Rect):
                return fitz.Rect(bbox_like)
            if isinstance(bbox_like, (list, tuple)) and len(bbox_like) == 4:
                try:
                    return fitz.Rect(
                        float(bbox_like[0]) * self.pixel_to_point,
                        float(bbox_like[1]) * self.pixel_to_point,
                        float(bbox_like[2]) * self.pixel_to_point,
                        float(bbox_like[3]) * self.pixel_to_point,
                    )
                except Exception:
                    return None
            return None

        def _native_group_bbox():
            ann_id = str(descriptor_group_ids.get("annotation_group_id") or "").strip()
            if ann_id:
                for group in descriptor_page_organization.get("annotation_groups") or []:
                    if str(group.get("id") or "") == ann_id:
                        return _fitz_rect_from_any(group.get("bbox"))
            cell_id = str(descriptor_group_ids.get("cell_id") or "").strip()
            row_id = str(descriptor_group_ids.get("table_row_group_id") or "").strip()
            if row_id:
                for row in descriptor_page_organization.get("table_row_groups") or []:
                    if str(row.get("id") or "") == row_id:
                        if cell_id:
                            for cell in row.get("cells") or []:
                                if str(cell.get("id") or "") == cell_id or str(cell.get("block_id") or "") == str(item.get("source_block_id") or ""):
                                    return _fitz_rect_from_any(cell.get("bbox"))
                        return _fitz_rect_from_any(row.get("bbox"))
            chart_groups = descriptor_page_organization.get("chart_groups") or {}
            legend_id = str(descriptor_group_ids.get("legend_group_id") or "").strip()
            if legend_id:
                legend = chart_groups.get("legend_group") or {}
                if str(legend.get("id") or "") == legend_id:
                    return _fitz_rect_from_any(legend.get("bbox"))
            series_id = str(descriptor_group_ids.get("series_group_id") or "").strip()
            if series_id:
                for series in chart_groups.get("series_groups") or []:
                    if str(series.get("id") or "") == series_id:
                        return _fitz_rect_from_any(series.get("bbox"))
            tick_id = str(descriptor_group_ids.get("tick_group_id") or "").strip()
            if tick_id:
                ticks = chart_groups.get("tick_group") or {}
                if str(ticks.get("id") or "") == tick_id:
                    return _fitz_rect_from_any(ticks.get("bbox"))
                ticks_x = chart_groups.get("x_tick_group") or {}
                if str(ticks_x.get("id") or "") == tick_id:
                    return _fitz_rect_from_any(ticks_x.get("bbox"))
            axis_id = str(descriptor_group_ids.get("axis_group_id") or "").strip()
            if axis_id:
                for axis in chart_groups.get("axis_groups") or []:
                    if str(axis.get("id") or "") == axis_id:
                        return _fitz_rect_from_any(axis.get("bbox"))
            return None

        native_group_bbox = _native_group_bbox()
        visual_group_bbox = self._visual_text_group_bbox(item)
        effective_group_bbox = visual_group_bbox if isinstance(visual_group_bbox, fitz.Rect) and visual_group_bbox.get_area() > 0 else native_group_bbox
        group_background_prepared = False
        if (
            render
            and descriptor_group_render_mode in {"annotation_group", "chart_legend_group", "chart_axis_group", "chart_series_group"}
            and isinstance(effective_group_bbox, fitz.Rect)
            and effective_group_bbox.get_area() > 0
        ):
            group_background_prepared = self._prepare_visual_group_background(page, item, effective_group_bbox)
        if header_like:
            region_left = left
            region_right = right
        elif isinstance(descriptor_region_bbox, fitz.Rect) and descriptor_region_bbox.get_area() > 0:
            region_left = max(left, descriptor_region_bbox.x0)
            region_right = min(right, descriptor_region_bbox.x1)
        else:
            region_left = left
            region_right = right
        if (not header_like) and isinstance(effective_group_bbox, fitz.Rect) and effective_group_bbox.get_area() > 0:
            region_left = max(region_left, effective_group_bbox.x0)
            region_right = min(region_right, effective_group_bbox.x1)
        relation_band_bbox = None
        if isinstance(paragraph_chain_bbox, fitz.Rect) and paragraph_chain_bbox.get_area() > 0:
            relation_band_bbox = paragraph_chain_bbox
        elif isinstance(same_band_bbox, fitz.Rect) and same_band_bbox.get_area() > 0:
            relation_band_bbox = same_band_bbox
        if (
            not header_like
            and descriptor_typographic_class == "editorial_body"
            and isinstance(relation_band_bbox, fitz.Rect)
            and relation_band_bbox.get_area() > 0
        ):
            region_left = max(region_left, relation_band_bbox.x0)
            region_right = min(region_right, relation_band_bbox.x1)
        if (
            not header_like
            and descriptor_structural_role == "section_title"
            and isinstance(section_sibling_bbox, fitz.Rect)
            and section_sibling_bbox.get_area() > 0
        ):
            region_left = max(region_left, section_sibling_bbox.x0)
            region_right = min(region_right, section_sibling_bbox.x1)
        strict_anchor_zone = descriptor_region_type in {"annotation_band", "caption_band", "table_cell", "table_row"}
        if descriptor_band_role in {"annotation_band", "caption_band", "header_band", "legend_band", "axis_band", "table_band"}:
            strict_anchor_zone = True
        if header_like:
            strict_anchor_zone = False
        if (not header_like) and isinstance(descriptor_region_bbox, fitz.Rect) and descriptor_region_bbox.get_area() > 0:
            x0 = max(region_left, min(x0, max(region_left, region_right - min(block_w, max(8.0, region_right - region_left)))))
        if (not header_like) and isinstance(effective_group_bbox, fitz.Rect) and effective_group_bbox.get_area() > 0:
            x0 = max(region_left, min(x0, max(region_left, region_right - min(block_w, max(8.0, region_right - region_left)))))
        if strict_anchor_zone and isinstance(descriptor_region_bbox, fitz.Rect) and descriptor_region_bbox.get_area() > 0:
            original_offset = max(0.0, item["bbox"].x0 - descriptor_region_bbox.x0)
            x0 = max(region_left, min(region_left + original_offset, max(region_left, region_right - min(block_w, region_right - region_left))))
        if strict_anchor_zone and isinstance(effective_group_bbox, fitz.Rect) and effective_group_bbox.get_area() > 0:
            original_offset = max(0.0, item["bbox"].x0 - effective_group_bbox.x0)
            x0 = max(region_left, min(region_left + original_offset, max(region_left, region_right - min(block_w, region_right - region_left))))
        if item.get("paragraph_flow_mode") and item.get("has_number_markers"):
            # Keep paragraph text aligned with source text column, not marker column.
            x0 = min(right - 8.0, x0 + max(0.0, float(item.get("marker_text_indent_pt", 0.0) or 0.0)))
        if item.get("paragraph_flow_mode") and isinstance(descriptor_region_bbox, fitz.Rect) and descriptor_region_bbox.get_area() > 0:
            x0 = max(region_left, min(x0, max(region_left, region_right - block_w)))
        y0 = max(zone_top, anchor_y)
        if item.get("paragraph_flow_mode") and isinstance(descriptor_region_bbox, fitz.Rect) and descriptor_region_bbox.get_area() > 0:
            y0 = max(y0, descriptor_region_bbox.y0)
        if item.get("paragraph_flow_mode") and isinstance(effective_group_bbox, fitz.Rect) and effective_group_bbox.get_area() > 0:
            y0 = max(y0, effective_group_bbox.y0)
        if item.get("paragraph_flow_mode") and isinstance(relation_band_bbox, fitz.Rect) and relation_band_bbox.get_area() > 0:
            y0 = max(y0, relation_band_bbox.y0)
        if descriptor_layout_behavior in {"anchored", "locked_in_cell", "locked_in_table"} and isinstance(descriptor_region_bbox, fitz.Rect) and descriptor_region_bbox.get_area() > 0:
            y0 = max(y0, descriptor_region_bbox.y0)
        if descriptor_layout_behavior in {"anchored", "locked_in_cell", "locked_in_table"} and isinstance(effective_group_bbox, fitz.Rect) and effective_group_bbox.get_area() > 0:
            y0 = max(y0, effective_group_bbox.y0)
        dx = x0 - item["bbox"].x0
        dy = y0 - item["bbox"].y0
        block_right = min(right, x0 + block_w)
        if item.get("allow_expand_to_page_right"):
            limit_pt = float(item.get("expand_right_limit_pt", right) or right)
            block_right = min(right, max(x0 + 8.0, limit_pt))
        if (not header_like) and isinstance(descriptor_region_bbox, fitz.Rect) and descriptor_region_bbox.get_area() > 0:
            block_right = min(block_right, region_right)
        if (not header_like) and isinstance(native_group_bbox, fitz.Rect) and native_group_bbox.get_area() > 0:
            block_right = min(block_right, region_right)
        if strict_anchor_zone and isinstance(descriptor_region_bbox, fitz.Rect) and descriptor_region_bbox.get_area() > 0:
            block_right = min(region_right, max(x0 + 8.0, descriptor_region_bbox.x1))
        if strict_anchor_zone and isinstance(native_group_bbox, fitz.Rect) and native_group_bbox.get_area() > 0:
            block_right = min(region_right, max(x0 + 8.0, native_group_bbox.x1))
        if strict_anchor_zone and isinstance(anchor_target_bbox, fitz.Rect) and anchor_target_bbox.get_area() > 0:
            if anchor_preferred_side == "left_of":
                block_right = min(block_right, max(x0 + 8.0, anchor_target_bbox.x0 - 2.0))
            elif anchor_preferred_side == "right_of":
                x0 = max(x0, min(block_right - 8.0, anchor_target_bbox.x1 + 2.0))
            elif anchor_preferred_side == "above":
                zone_bottom = min(zone_bottom, max(y0 + 8.0, anchor_target_bbox.y0 - 2.0))
            elif anchor_preferred_side == "below":
                y0 = max(y0, anchor_target_bbox.y1 + 2.0)
            x0, block_right = self._expand_anchor_target_span(item, left, right, x0, block_right)
        if (
            descriptor_typographic_class == "editorial_body"
            and isinstance(relation_band_bbox, fitz.Rect)
            and relation_band_bbox.get_area() > 0
        ):
            x0 = max(region_left, min(x0, max(region_left, relation_band_bbox.x0)))
            block_right = min(block_right, max(x0 + 8.0, relation_band_bbox.x1))
        if (
            descriptor_structural_role == "section_title"
            and isinstance(section_sibling_bbox, fitz.Rect)
            and section_sibling_bbox.get_area() > 0
        ):
            x0 = max(region_left, min(x0, max(region_left, section_sibling_bbox.x0)))
            block_right = min(block_right, max(x0 + 8.0, section_sibling_bbox.x1))
        if (
            not item.get("paragraph_flow_mode")
            and isinstance(same_row_bbox, fitz.Rect)
            and same_row_bbox.get_area() > 0
            and not self._is_abbreviation_value_role(descriptor_structural_role)
        ):
            zone_bottom = min(zone_bottom, max(y0 + 8.0, same_row_bbox.y1))
        # Strict slot anchoring requested:
        # - line start must match phrase slot x0
        # - line wrapping must use blue box right edge
        # - no center/right/justify alignment
        body_stable_left = False
        if descriptor_structural_role in {"opening_paragraph", "body_paragraph"} and descriptor_band_role == "text_band":
            body_stable_left = True
        # Professional paragraph rule: keep stable vertical rhythm (line starts + line heights)
        # for body text to avoid "stair-step" OCR line jitter.
        body_stable_vertical = body_stable_left
        paragraph_left_x = max(left, min(x0, block_right - 6.0))
        stable_slot_h = max(8.0, float(item.get("slot_h_pt", 10.0)))
        stable_gap_y = max(1.8, min(7.5, float(item.get("slot_gap_y_pt", stable_slot_h * 0.28))))
        stable_next_y = y0
        # Smart horizontal expansion for short/top metadata and headings.
        if role in {"header", "footer"} or descriptor_structural_role in {"running_header", "running_footer"}:
            block_right = right
        elif item.get("is_title") and block_w < (right - left) * 0.72:
            block_right = min(right, x0 + max(block_w, (right - left) * 0.72))
        if (not header_like) and descriptor_band_role == "title_band" and isinstance(descriptor_region_bbox, fitz.Rect) and descriptor_region_bbox.get_area() > 0:
            x0 = max(region_left, descriptor_region_bbox.x0)
            block_right = min(right, descriptor_region_bbox.x1)
        if (not header_like) and descriptor_band_role in {"annotation_band", "legend_band", "axis_band", "table_band"} and isinstance(native_group_bbox, fitz.Rect) and native_group_bbox.get_area() > 0:
            x0 = max(region_left, native_group_bbox.x0)
            block_right = min(region_right, native_group_bbox.x1)
            y0 = max(y0, native_group_bbox.y0)
            zone_bottom = min(zone_bottom, max(y0 + 8.0, native_group_bbox.y1))

        slots = [fitz.Rect(s.x0 + dx, s.y0 + dy, s.x1 + dx, s.y1 + dy) for s in item["slots"]]
        slots.sort(key=lambda r: (r.y0, r.x0))
        preserve_linebreaks = bool(item.get("preserve_linebreaks"))
        strict_bbox_mode = bool(item.get("strict_bbox_mode"))
        preset_lines = []
        preset_line_styles = []
        if preserve_linebreaks:
            source_structured_lines, source_structured_styles = self._structured_source_lines_with_styles(item)
            if override_text is not None:
                preset_lines = [self._clean_text_for_render(x).strip() for x in str(override_text).split("\n") if x.strip()]
                if (
                    source_structured_lines
                    and len(preset_lines) == len(source_structured_lines)
                    and preset_lines == source_structured_lines
                ):
                    preset_line_styles = list(source_structured_styles)
                else:
                    preset_line_styles = [None for _ in preset_lines]
            else:
                preset_lines = list(source_structured_lines)
                preset_line_styles = list(source_structured_styles)
        translated_structured_fit = bool(
            preserve_linebreaks
            and item.get("translated_block")
            and (not item.get("keep_exact_line"))
            and preset_lines
            and (not item.get("keep_source_slot_geometry"))
        )
        if translated_structured_fit:
            strict_bbox_mode = False
        words = [] if preserve_linebreaks else text.split()
        used_bottom = y0
        idx = 0
        used_slots = []
        prev_slot_bottom = None
        fitted_structured_fs = None
        fitted_structured_line_h = None
        uniform_preserved_fs = None
        if render and self._should_whiteout_before_render(item):
            self._whiteout_rect(
                page,
                item.get(
                    "whiteout_bbox",
                    item.get("bbox", fitz.Rect(x0, y0, block_right, max(y0 + 6.0, y0 + item.get("slot_h_pt", 8.0)))),
                ),
            )
        elif render and (not group_background_prepared) and self._should_restore_background_before_render(item):
            self._restore_background_rect(
                page,
                item,
                item.get("bbox", fitz.Rect(x0, y0, block_right, max(y0 + 6.0, y0 + item.get("slot_h_pt", 8.0)))),
                kind="translated_anchor_bg_restore",
            )
        if translated_structured_fit:
            bbox_bottom = min(zone_bottom, max(y0 + 8.0, float(item["bbox"].y1) + dy))
            available_h = max(8.0, bbox_bottom - y0)
            line_count = max(1, len(preset_lines))
            desired_gap = max(1.0, min(float(item.get("slot_gap_y_pt", 2.0)), available_h * 0.08))
            total_gap = desired_gap * max(0, line_count - 1)
            if total_gap >= available_h:
                desired_gap = max(0.5, available_h / max(2.0, line_count * 3.0))
                total_gap = desired_gap * max(0, line_count - 1)
            target_line_h = max(6.0, (available_h - total_gap) / line_count)
            fitted_structured_fs = min(base_fs, max(5.5, target_line_h / 1.22))
            slot_w_fit = max(8.0, block_right - x0)
            while (
                fitted_structured_fs > 5.5 + 1e-6
                and any(
                    self._measure_text_width(line, fitted_structured_fs, fontname, fontfile) > slot_w_fit * 1.01
                    for line in preset_lines
                )
            ):
                fitted_structured_fs = max(5.5, fitted_structured_fs - 0.2)
            fitted_structured_line_h = max(6.0, fitted_structured_fs * 1.22)
            desired_gap = max(
                0.5,
                min(desired_gap, max(0.5, (available_h - fitted_structured_line_h * line_count) / max(1, line_count - 1))),
            )
            slots = []
            cur_slot_y = y0
            for _ in range(line_count):
                next_y = min(bbox_bottom, cur_slot_y + fitted_structured_line_h)
                slots.append(fitz.Rect(x0, cur_slot_y, block_right, max(cur_slot_y + 6.0, next_y)))
                cur_slot_y = next_y + desired_gap
        elif self._should_use_uniform_preserved_line_fontsize(item) and preset_lines:
            width_samples = []
            probe_slots = slots or [fitz.Rect(x0, y0, block_right, max(y0 + 6.0, y0 + item.get("slot_h_pt", 8.0)))]
            for line_idx in range(len(preset_lines)):
                probe_slot = probe_slots[min(line_idx, len(probe_slots) - 1)]
                probe_x0 = max(left, min(probe_slot.x0, block_right - 6.0))
                width_samples.append(max(8.0, block_right - probe_x0))
            uniform_preserved_fs = self._fit_uniform_preserved_line_fontsize(
                preset_lines,
                width_samples,
                base_fs,
                fontname,
                fontfile,
                overflow_limit=1.03,
                min_font_pt=max(5.8, base_fs * 0.88),
            )
        active_style_lock_source = item.get("style_lock_source", "block")
        if item.get("paragraph_flow_mode") and not preserve_linebreaks:
            # Strict block mode: compose the whole paragraph inside its source bbox.
            # No spill to extra pages/rows; keep line starts anchored at block x0.
            if descriptor_reconstruction_plan.get("primary_flow_regions") and descriptor_band_role == "text_band":
                x0 = max(region_left, float(item.get("preferred_left_x_pt", x0) or x0))
            box_bottom = min(zone_bottom, item["bbox"].y1 + dy)
            if item.get("allow_vertical_expand"):
                box_bottom = zone_bottom
            if isinstance(descriptor_region_bbox, fitz.Rect) and descriptor_region_bbox.get_area() > 0:
                region_type = str(item.get("descriptor_region_type") or "").strip().lower()
                if region_type == "text_band":
                    box_bottom = max(box_bottom, descriptor_region_bbox.y1)
                else:
                    box_bottom = min(box_bottom, max(y0 + 8.0, descriptor_region_bbox.y1))
            box_h = max(8.0, box_bottom - y0)
            base_try = base_fs if self.fixed_font_size else min(base_fs, max(7.0, item.get("slot_h_pt", 10.0) * 0.92))
            if descriptor_typographic_class in {"editorial_body", "diagram_label", "chart_axis_label", "chart_tick_label", "chart_legend_label", "figure_caption"}:
                base_try = max(base_try, base_fs * 0.94)
            box_w = max(
                8.0,
                block_right - x0 - self._line_right_padding_for_item(item, base_try, strict=False),
            )
            def _wrap_words_no_split(full_text, width, fsz):
                ws = [w for w in self._clean_text_for_render(full_text).split() if w]
                out = []
                cur = ""
                for w in ws:
                    if not cur:
                        # Keep whole word even if it overflows slightly; never split inside a word.
                        cur = w
                        continue
                    cand = f"{cur} {w}"
                    if self._measure_text_width(cand, fsz, fontname, fontfile) <= width:
                        cur = cand
                    else:
                        out.append(cur)
                        cur = w
                if cur:
                    out.append(cur)
                return out

            fs = float(base_try)
            min_fs = self._min_fontsize_for_item(item, base_try, strict=False)
            lines = _wrap_words_no_split(text, box_w, fs)
            line_h = max(1.0, fs * 1.22)
            while (len(lines) * line_h > box_h) and (fs > min_fs + 1e-6):
                fs = max(min_fs, fs - 0.2)
                line_h = max(1.0, fs * 1.22)
                lines = _wrap_words_no_split(text, box_w, fs)
            y = y0
            for line in lines:
                baseline = y + min(line_h * 0.82, line_h - 1.0)
                if baseline > (y0 + box_h):
                    break
                slot_rect = fitz.Rect(x0, y, x0 + box_w, min(y + line_h, y0 + box_h))
                if render:
                    if self._should_whiteout_per_line(item):
                        self._whiteout_rect(page, slot_rect, pad_x=0.8, pad_y=0.35)
                    self._safe_insert_text_dedup(page, (x0, baseline), line, fs, fontname, rgb)
                used_slots.append(slot_rect)
                used_bottom = max(used_bottom, slot_rect.y1)
                y += line_h
            blue_rect = fitz.Rect(x0, y0, x0 + box_w, max(y0 + item["slot_h_pt"], used_bottom))
            return "", used_bottom, blue_rect, used_slots
        while words or preset_lines:
            current_style_lock_source = active_style_lock_source
            if idx >= len(slots):
                can_extend_slots = bool(
                    item.get("allow_vertical_expand")
                    or (
                        item.get("preserve_sentence_integrity")
                        and isinstance(descriptor_region_bbox, fitz.Rect)
                        and descriptor_region_bbox.get_area() > 0
                    )
                )
                if strict_bbox_mode and not can_extend_slots:
                    break
                if item.get("is_diagram_label") and not can_extend_slots:
                    break
                prev = slots[-1] if slots else fitz.Rect(x0, y0, x0 + item["slot_w_pt"], y0 + item["slot_h_pt"])
                # New rows are appended downward; each line fills available width first.
                nx = x0 + max(0.0, item["row_start_x_pt"] - item["bbox"].x0)
                if body_stable_vertical:
                    ny = stable_next_y
                else:
                    ny = prev.y1 + item["slot_gap_y_pt"]
                slots.append(fitz.Rect(nx, ny, nx + item["slot_w_pt"], ny + item["slot_h_pt"]))
            slot = slots[idx]
            idx += 1
            sx0 = max(left, min(slot.x0, block_right - 6.0))
            exact_slot_render = bool(item.get("exact_slot_render"))
            if exact_slot_render:
                sx1 = max(sx0 + 6.0, min(block_right, slot.x1))
            else:
                # Red slot always extends to blue frame right edge.
                sx1 = block_right
            if body_stable_vertical:
                sy0 = max(zone_top, stable_next_y)
                sy1 = sy0 + stable_slot_h
            else:
                sy0 = max(zone_top, slot.y0)
                sy1 = max(sy0 + 6.0, slot.y1)
                if prev_slot_bottom is not None:
                    min_gap = max(1.5, item.get("slot_h_pt", 8.0) * 0.18)
                    sy0 = max(sy0, prev_slot_bottom + min_gap)
                    sy1 = max(sy1, sy0 + 6.0)
            slot = fitz.Rect(sx0, sy0, sx1, sy1)
            if slot.y1 > zone_bottom:
                break
            slot_w = max(8.0, slot.width)
            slot_h = max(8.0, slot.height)
            preview_style_override = None
            if preserve_linebreaks and preset_lines and preset_line_styles:
                preview_style_override = preset_line_styles[0]
            active_style_lock_source = item.get("style_lock_source", "block")
            if isinstance(preview_style_override, dict) and preview_style_override:
                effective_style = self._normalized_style_for_item(item, self._merge_styles(preview_style_override, style))
                active_style_lock_source = "line"
            else:
                effective_style = style
            preview_text = ""
            if preserve_linebreaks and preset_lines:
                preview_text = preset_lines[0]
            elif words:
                preview_text = " ".join(words[: min(len(words), 24)])
            _, fontfile, builtin, fontname = self._resolve_style_font(page, effective_style, text=preview_text)
            base_fs = self._normalized_fontsize_for_item(item, effective_style, max(1.0, item["slot_h_pt"]), source)
            rgb = self._resolve_text_color(effective_style, item)
            if uniform_preserved_fs is not None:
                fs = float(uniform_preserved_fs)
            else:
                fs = base_fs if (self.fixed_font_size or strict_bbox_mode) else min(base_fs, slot_h * 0.92)
            if descriptor_typographic_class in {"editorial_body", "diagram_label", "chart_axis_label", "chart_tick_label", "chart_legend_label", "figure_caption"}:
                fs = max(fs, base_fs * 0.94)
            if fitted_structured_fs is not None:
                fs = min(fs, fitted_structured_fs)
            expected_align = self._normalize_alignment(item.get("alignment", "left"))
            region_type = str(item.get("descriptor_region_type") or "").strip().lower()
            if item.get("role") == "figure_caption":
                if not self.fixed_font_size:
                    fs = max(6.0, base_fs)
            elif not self.fixed_font_size:
                if item.get("is_title"):
                    fs = max(fs, min(max(11.5, base_fs * 1.22), slot_h * 1.05))
                elif item.get("role") == "section_heading":
                    fs = max(fs, min(max(10.5, base_fs * 1.12), slot_h * 1.02))
                elif item.get("is_diagram_label"):
                    fs = max(min(fs, 9.6), min(base_fs, 8.4))
                else:
                    fs = min(fs, max(8.0, slot_h * 0.78))
            if item.get("role") == "header":
                fs = max(fs, 8.8)
            if region_type in {"annotation_band", "caption_band", "header_band"}:
                fs = max(fs, 8.2)
            fit_right_pad = self._line_right_padding_for_item(
                item,
                fs,
                strict=bool(exact_slot_render or preserve_linebreaks or strict_bbox_mode),
            )
            slot_right_limit = max(slot.x0 + 8.0, min(slot.x1, slot.x1 - fit_right_pad))
            fit_slot_w = max(8.0, slot_right_limit - slot.x0)
            required_line_h = max(1.0, fs * 1.22)
            if slot_h < required_line_h:
                slot = fitz.Rect(slot.x0, slot.y0, slot.x1, slot.y0 + required_line_h)
                slot_h = max(8.0, slot.height)
            if forbidden_rects and (not strict_bbox_mode) and (not strict_anchor_zone):
                probe = fitz.Rect(slot)
                for _ in range(6):
                    collisions = [fr for fr in forbidden_rects if (probe & fr).get_area() > 0]
                    if not collisions:
                        break
                    next_y = max(fr.y1 for fr in collisions) + max(1.0, slot_h * 0.12)
                    shifted = False
                    x_step = max(6.0, min(slot_w * 0.18, 28.0))
                    for dx in (x_step, -x_step, x_step * 2.0, -x_step * 2.0):
                        cand_x0 = max(left, min(right - slot_w, probe.x0 + dx))
                        cand = fitz.Rect(cand_x0, probe.y0, cand_x0 + slot_w, probe.y1)
                        if not any((cand & fr).get_area() > 0 for fr in forbidden_rects):
                            probe = cand
                            shifted = True
                            break
                    if shifted:
                        continue
                    if next_y >= zone_bottom - 2.0:
                        break
                    probe = fitz.Rect(probe.x0, next_y, probe.x1, next_y + slot_h)
                if probe.y1 <= zone_bottom:
                    slot = probe
            if exact_slot_render:
                if preserve_linebreaks and preset_lines:
                    exact_text = self._clean_text_for_render(preset_lines.pop(0)).strip()
                else:
                    exact_text = self._clean_text_for_render(" ".join(words)).strip()
                if not exact_text:
                    continue
                inline_segments = list(item.get("inline_style_segments") or [])
                if item.get("keep_exact_line") and self._should_render_inline_style_segments(item, inline_segments):
                    direct_parts = [
                        self._clean_text_for_render(seg.get("text", "")).strip()
                        for seg in inline_segments
                        if isinstance(seg, dict)
                    ]
                    normalized_direct = self._clean_text_for_render(" ".join(part for part in direct_parts if part)).strip()
                    if (
                        len(direct_parts) == len(inline_segments)
                        and normalized_direct
                        and normalized_direct == exact_text
                    ):
                        partitioned = direct_parts
                    else:
                        partitioned = self._partition_translated_line_to_segments(exact_text, inline_segments)
                    inline_fit = []
                    if partitioned:
                        for seg, seg_text in zip(inline_segments, partitioned):
                            seg_style = self._normalized_style_for_item(
                                item,
                                self._merge_styles(seg.get("style", {}), style),
                            )
                            seg_rect = seg.get("bbox")
                            if not isinstance(seg_rect, fitz.Rect) or seg_rect.get_area() <= 0:
                                inline_fit = []
                                break
                            _, seg_fontfile, seg_builtin, seg_fontname = self._resolve_style_font(page, seg_style, text=seg_text)
                            seg_fs = self._normalized_fontsize_for_item(item, seg_style, max(1.0, seg_rect.height), source)
                            seg_w = self._measure_text_width(seg_text, seg_fs, seg_fontname, seg_fontfile)
                            if seg_w > max(8.0, seg_rect.width) * 1.02:
                                inline_fit = []
                                break
                            inline_fit.append(
                                {
                                    "text": seg_text,
                                    "bbox": fitz.Rect(seg_rect),
                                    "fontname": seg_fontname,
                                    "fontsize": seg_fs,
                                    "rgb": self._resolve_text_color(seg_style, item),
                                }
                            )
                    if inline_fit:
                        if render:
                            if self._should_whiteout_per_line(item):
                                self._whiteout_rect(page, slot, pad_x=0.6, pad_y=0.3)
                            for seg_part in inline_fit:
                                seg_rect = fitz.Rect(
                                    seg_part["bbox"].x0 + dx,
                                    seg_part["bbox"].y0 + dy,
                                    seg_part["bbox"].x1 + dx,
                                    seg_part["bbox"].y1 + dy,
                                )
                                seg_fs = float(seg_part["fontsize"])
                                seg_h = max(1.0, seg_rect.height)
                                seg_baseline = seg_rect.y0 + min(seg_h * 0.82, seg_h - 1.0)
                                self._safe_insert_text_dedup(
                                    page,
                                    (seg_rect.x0, seg_baseline),
                                    seg_part["text"],
                                    seg_fs,
                                    seg_part["fontname"],
                                    seg_part["rgb"],
                                )
                        continue
                comp = self._compose_exact_slot_text(
                    text=exact_text,
                    slot_w=fit_slot_w,
                    slot_h=slot_h,
                    base_fs=fs,
                    fontname=fontname,
                    fontfile=fontfile,
                    source=item.get("source", "ocr"),
                    alignment=item.get("alignment", "left"),
                    max_font_shrink=(
                        1.04 if self._item_native_style_fidelity_mode(item) else (
                        1.08 if self._item_preserve_extracted_typography(item) else 1.12
                        )
                        if (
                            descriptor_typographic_class in {"running_header", "running_footer", "section_title", "figure_caption", "diagram_label", "chart_axis_label", "chart_tick_label", "chart_legend_label", "editorial_body"}
                            or str(item.get("role") or "").strip().lower() in {"title", "section_heading", "header", "footer", "figure_caption", "diagram_label", "diagram_text_label"}
                        )
                        else 1.20 if self._item_native_style_fidelity_mode(item) else (
                        1.24 if self._item_preserve_extracted_typography(item) else 1.35
                        )
                    ),
                    min_font_pt=self._min_fontsize_for_item(
                        item,
                        fs,
                        strict=(
                            descriptor_typographic_class in {"running_header", "running_footer", "section_title", "figure_caption", "diagram_label", "chart_axis_label", "chart_tick_label", "chart_legend_label", "editorial_body"}
                            or str(item.get("role") or "").strip().lower() in {"title", "section_heading", "header", "footer", "figure_caption", "diagram_label", "diagram_text_label"}
                        ),
                    ),
                    line_height_factor=(
                        1.14
                        if descriptor_typographic_class in {"running_header", "running_footer", "section_title", "figure_caption", "diagram_label", "chart_axis_label", "chart_tick_label", "chart_legend_label"}
                        else 1.18
                    ),
                )
                exact_lines = comp.get("lines") or []
                if not exact_lines:
                    continue
                fs = float(comp.get("font_size", fs) or fs)
                line_h = max(1.0, fs * 1.18)
                total_h = line_h * len(exact_lines)
                top_y = slot.y0 + max(0.0, (slot_h - total_h) / 2.0)
                for line_idx, exact_line in enumerate(exact_lines):
                    line_w = self._measure_text_width(exact_line, fs, fontname, fontfile)
                    applied_align, align_fallback_reason = self._resolve_applied_alignment(
                        expected_alignment=item.get("alignment", "left"),
                        line_w=line_w,
                        left=slot.x0,
                        right=slot_right_limit,
                        is_last_line=(line_idx == len(exact_lines) - 1),
                    )
                    line_x = self._compute_aligned_x(
                        alignment=item.get("alignment", "left"),
                        line_w=line_w,
                        left=slot.x0,
                        right=slot_right_limit,
                        preferred_x=slot.x0,
                        is_last_line=(line_idx == len(exact_lines) - 1),
                    )
                    baseline = top_y + line_idx * line_h + min(line_h * 0.82, line_h - 1.0)
                    if render:
                        if self._should_whiteout_per_line(item):
                            self._whiteout_rect(page, slot, pad_x=0.6, pad_y=0.3)
                        line_rect = fitz.Rect(
                            line_x,
                            baseline - line_h * 0.82,
                            line_x + line_w,
                            baseline + max(1.0, line_h * 0.18),
                        )
                        self._safe_insert_text_dedup(page, (line_x, baseline), exact_line, fs, fontname, rgb)
                    if render and self.style_audit_enabled:
                        self._style_audit_records.append(
                            {
                                "page": int(page.number) + 1,
                                "role": item.get("role", "body"),
                                "primary_structure_family": self._descriptor_v3_primary_family(item),
                                "structure_priority": self._descriptor_v3_structure_priority(item),
                                "style_lock_source": active_style_lock_source,
                                "expected_alignment": item.get("alignment", "left"),
                                "applied_alignment": applied_align,
                                "alignment_raw": item.get("alignment_raw", ""),
                                "alignment_source": item.get("alignment_source", "block"),
                                "alignment_defaulted": bool(item.get("alignment_defaulted", False)),
                                "alignment_fallback_reason": align_fallback_reason or item.get("alignment_fallback_reason", ""),
                                "expected_font": effective_style.get("font"),
                                "applied_font": fontname,
                                "font_fallback": (fontname in {"helv", "times", "courier"} and str(effective_style.get("font", "")).strip().lower() not in {"", "helv", "times", "courier", "arial", "helvetica"}),
                            }
                        )
                overflow_text = self._clean_text_for_render(comp.get("overflow", "")).strip()
                if preserve_linebreaks:
                    if overflow_text:
                        preset_lines.insert(0, overflow_text)
                else:
                    words = overflow_text.split() if overflow_text else []
                used_bottom = max(used_bottom, slot.y1)
                used_slots.append(slot)
                prev_slot_bottom = slot.y1
                if body_stable_vertical:
                    stable_next_y = slot.y1 + stable_gap_y
                continue
            if preserve_linebreaks and preset_lines:
                current_line_style = None
                line = preset_lines.pop(0)
                if preset_line_styles:
                    current_line_style = preset_line_styles.pop(0)
                current_style_lock_source = active_style_lock_source
                if isinstance(current_line_style, dict) and current_line_style:
                    effective_style = self._normalized_style_for_item(item, self._merge_styles(current_line_style, style))
                    current_style_lock_source = "line"
                    _, fontfile, builtin, fontname = self._resolve_style_font(page, effective_style, text=line)
                    base_fs = self._normalized_fontsize_for_item(item, effective_style, max(1.0, item["slot_h_pt"]), source)
                    rgb = self._resolve_text_color(effective_style, item)
                    if uniform_preserved_fs is None:
                        fs = base_fs if (self.fixed_font_size or strict_bbox_mode) else min(base_fs, slot_h * 0.92)
                if item.get("keep_exact_line"):
                    line = self._clean_text_for_render(line).strip()
                    if not line:
                        continue
                    inline_segments = list(item.get("inline_style_segments") or [])
                    inline_segment_parts = []
                    if self._should_render_inline_style_segments(item, inline_segments):
                        direct_parts = [
                            self._clean_text_for_render(seg.get("text", "")).strip()
                            for seg in inline_segments
                            if isinstance(seg, dict)
                        ]
                        normalized_direct = self._clean_text_for_render(" ".join(part for part in direct_parts if part)).strip()
                        normalized_line = self._clean_text_for_render(line).strip()
                        if (
                            len(direct_parts) == len(inline_segments)
                            and normalized_direct
                            and normalized_direct == normalized_line
                        ):
                            partitioned = direct_parts
                        else:
                            partitioned = self._partition_translated_line_to_segments(line, inline_segments)
                        if partitioned:
                            inline_fit = []
                            for seg, seg_text in zip(inline_segments, partitioned):
                                seg_style = self._normalized_style_for_item(
                                    item,
                                    self._merge_styles(seg.get("style", {}), style),
                                )
                                seg_rect = seg.get("bbox")
                                if not isinstance(seg_rect, fitz.Rect) or seg_rect.get_area() <= 0:
                                    inline_fit = []
                                    break
                                _, seg_fontfile, seg_builtin, seg_fontname = self._resolve_style_font(page, seg_style, text=seg_text)
                                seg_fs = self._normalized_fontsize_for_item(item, seg_style, max(1.0, seg_rect.height), source)
                                seg_w = self._measure_text_width(seg_text, seg_fs, seg_fontname, seg_fontfile)
                                if seg_w > max(8.0, seg_rect.width) * 1.02:
                                    inline_fit = []
                                    break
                                inline_fit.append(
                                    {
                                        "text": seg_text,
                                        "bbox": fitz.Rect(seg_rect),
                                        "style": seg_style,
                                        "fontfile": seg_fontfile,
                                        "fontname": seg_fontname,
                                        "fontsize": seg_fs,
                                        "rgb": self._resolve_text_color(seg_style, item),
                                    }
                                )
                            inline_segment_parts = inline_fit
                    line_w_now = self._measure_text_width(line, fs, fontname, fontfile)
                    if (not inline_segment_parts) and item.get("translated_block") and line_w_now > fit_slot_w:
                        min_fs = self._min_fontsize_for_item(item, fs, strict=True)
                        while line_w_now > fit_slot_w and fs > min_fs + 1e-6:
                            fs = max(min_fs, fs - 0.2)
                            line_w_now = self._measure_text_width(line, fs, fontname, fontfile)
                else:
                    if (
                        item.get("role") == "body"
                        and item.get("use_structured_source_lines")
                        and re.fullmatch(r"\s*\d+[.)]?\s*", line or "")
                    ):
                        continue
                    keep_source_slot_geometry = bool(item.get("keep_source_slot_geometry"))
                    allow_overflow = bool(item.get("allow_line_overflow", False))
                    prefer_local_multiline_reflow = bool(item.get("prefer_local_multiline_reflow"))
                    effective_preserve_sentence_integrity = bool(item.get("preserve_sentence_integrity"))
                    if prefer_local_multiline_reflow:
                        effective_preserve_sentence_integrity = False
                    if (
                        effective_preserve_sentence_integrity
                        and keep_source_slot_geometry
                        and item.get("translated_block")
                        and item.get("preserve_line_style_variation")
                    ):
                        effective_preserve_sentence_integrity = False
                    prefer_tail_split_before_shrink = bool(
                        prefer_local_multiline_reflow
                        or (
                            keep_source_slot_geometry
                            and item.get("translated_block")
                            and item.get("preserve_line_style_variation")
                            and not effective_preserve_sentence_integrity
                        )
                    )
                    line_w_now = self._measure_text_width(line, fs, fontname, fontfile)
                    overflow_limit = self._overflow_limit_for_item(item, 1.04 if keep_source_slot_geometry else 1.12)
                    overflow_ok = bool(allow_overflow and line_w_now <= fit_slot_w * overflow_limit)
                    split_before_shrink_applied = False
                    if (
                        prefer_tail_split_before_shrink
                        and (not overflow_ok)
                        and line_w_now > fit_slot_w
                    ):
                        wds = line.split()
                        if len(wds) > 1:
                            fitted_line, tail = self._consume_words_for_width(wds, fit_slot_w, fs, fontname, fontfile)
                            if fitted_line and tail:
                                line = fitted_line
                                preset_lines.insert(0, " ".join(tail))
                                preset_line_styles.insert(0, current_line_style)
                                split_before_shrink_applied = True
                                line_w_now = self._measure_text_width(line, fs, fontname, fontfile)
                                overflow_ok = bool(allow_overflow and line_w_now <= fit_slot_w * overflow_limit)
                    if (
                        (not overflow_ok)
                        and item.get("translated_block")
                        and ((not self.fixed_font_size) or keep_source_slot_geometry or prefer_local_multiline_reflow)
                        and not split_before_shrink_applied
                    ):
                        if keep_source_slot_geometry:
                            min_fs = self._min_fontsize_for_item(item, fs, strict=True)
                        elif prefer_local_multiline_reflow:
                            min_fs = self._min_fontsize_for_item(item, fs, strict=True)
                        else:
                            min_fs = self._min_fontsize_for_item(
                                item,
                                fs,
                                strict=bool(effective_preserve_sentence_integrity),
                            )
                        while line_w_now > fit_slot_w and fs > min_fs + 1e-6:
                            fs = max(min_fs, fs - 0.2)
                            line_w_now = self._measure_text_width(line, fs, fontname, fontfile)
                        fit_right_pad = self._line_right_padding_for_item(
                            item,
                            fs,
                            strict=bool(exact_slot_render or preserve_linebreaks or strict_bbox_mode),
                        )
                        slot_right_limit = max(slot.x0 + 8.0, min(slot.x1, slot.x1 - fit_right_pad))
                        fit_slot_w = max(8.0, slot_right_limit - slot.x0)
                        overflow_ok = bool(allow_overflow and line_w_now <= fit_slot_w * overflow_limit)
                    # Keep original font size on structured lines (lists/bullets/numbered lines).
                    # In strict bbox mode, allow shrink only for paragraph-flow composition.
                    if (not overflow_ok) and (not preserve_linebreaks) and ((not self.fixed_font_size) or strict_bbox_mode) and line_w_now > fit_slot_w:
                        min_fs = self._min_fontsize_for_item(item, fs, strict=True)
                        while self._measure_text_width(line, fs, fontname, fontfile) > fit_slot_w and fs > min_fs + 1e-6:
                            fs = max(min_fs, fs - 0.2)
                    line_w_now = self._measure_text_width(line, fs, fontname, fontfile)
                    fit_right_pad = self._line_right_padding_for_item(
                        item,
                        fs,
                        strict=bool(exact_slot_render or preserve_linebreaks or strict_bbox_mode),
                    )
                    slot_right_limit = max(slot.x0 + 8.0, min(slot.x1, slot.x1 - fit_right_pad))
                    fit_slot_w = max(8.0, slot_right_limit - slot.x0)
                    overflow_ok = bool(allow_overflow and line_w_now <= fit_slot_w * overflow_limit)
                    if (not overflow_ok) and line_w_now > fit_slot_w and not effective_preserve_sentence_integrity:
                        if keep_source_slot_geometry and line_w_now <= fit_slot_w * 1.03:
                            overflow_ok = True
                        if not overflow_ok:
                            wds = line.split()
                            if len(wds) > 1:
                                line, tail = self._consume_words_for_width(wds, fit_slot_w, fs, fontname, fontfile)
                                if tail:
                                    preset_lines.insert(0, " ".join(tail))
                                    preset_line_styles.insert(0, current_line_style)
                            elif len(line) > 1 and not keep_source_slot_geometry:
                                chunk = ""
                                i = 0
                                for ch in line:
                                    cand = chunk + ch
                                    if chunk and self._measure_text_width(cand, fs, fontname, fontfile) > fit_slot_w:
                                        break
                                    chunk = cand
                                    i += 1
                                if chunk:
                                    rest = line[i:].strip()
                                    line = chunk
                                    if rest:
                                        preset_lines.insert(0, rest)
                                        preset_line_styles.insert(0, current_line_style)
                    line = line.strip()
                    if not line:
                        continue
            else:
                if self.fixed_font_size:
                    line, words = self._consume_words_for_width(words, fit_slot_w, fs, fontname, fontfile)
                    if not line:
                        continue
                else:
                    remaining_text = " ".join(words).strip()
                    comp = self.text_composer.compose_text_in_box(
                        text=remaining_text,
                        box_w=fit_slot_w,
                        box_h=slot_h,
                        base_font_pt=fs,
                        line_height_factor=1.22,
                        measure_fn=lambda t, fsz: self._measure_text_width(t, fsz, fontname, fontfile),
                        alignment=expected_align,
                        lang=item.get("lang", "en"),
                        options=ComposeOptions(
                            enable_hyphenation=(item.get("source") != "native"),
                            max_font_shrink=1.0,
                            min_font_pt=self._min_fontsize_for_item(item, fs, strict=False),
                            step_pt=0.25,
                        ),
                    )
                    if not comp.get("lines"):
                        continue
                    line = comp["lines"][0]
                    words = (comp.get("overflow") or "").split()
                    fs = comp.get("font_size", fs)
                    required_line_h = max(1.0, fs * 1.22)
                    if slot_h < required_line_h:
                        slot = fitz.Rect(slot.x0, slot.y0, slot.x1, slot.y0 + required_line_h)
                        slot_h = max(8.0, slot.height)
            line_w = self._measure_text_width(line, fs, fontname, fontfile)
            if item.get("role") == "body" and item.get("has_number_markers"):
                # Numeric markers (1,2,...) are rendered as dedicated fixed items.
                # Strip accidental duplicated inline numbering from flowed text lines.
                line = re.sub(r"^\s*\d+[.)]?\s+", "", line or "").strip()
                if re.fullmatch(r"\d+[.)]?", line or ""):
                    line = ""
                if not line:
                    continue
                line_w = self._measure_text_width(line, fs, fontname, fontfile)
            applied_align, align_fallback_reason = self._resolve_applied_alignment(
                expected_alignment=expected_align,
                line_w=line_w,
                left=slot.x0,
                right=slot_right_limit,
                is_last_line=(not words and not preset_lines),
            )
            preferred_left_x = slot.x0
            if (
                preserve_linebreaks
                and item.get("preserve_block_left_anchor")
                and item.get("role") == "body"
            ):
                try:
                    preferred_left_x = max(left, min(right - 6.0, float(item.get("preferred_left_x_pt", slot.x0))))
                except Exception:
                    preferred_left_x = slot.x0
            elif (
                item.get("keep_exact_line")
                and self._allow_exact_line_left_relief(item)
                and expected_align == "left"
                and line_w > fit_slot_w
            ):
                preferred_left_x = max(left, region_left, slot_right_limit - line_w)
            line_x = self._compute_aligned_x(
                alignment=expected_align,
                line_w=line_w,
                left=preferred_left_x,
                right=slot_right_limit,
                preferred_x=preferred_left_x,
                is_last_line=(not words and not preset_lines),
            )
            baseline = slot.y0 + min(slot_h * 0.82, slot_h - 1.0)
            line_rgb = rgb
            if item.get("role") == "body":
                if re.match(r"^\s*(?:LA|LE|LES|THE)\s+[A-ZÀ-ÿ].*\([^)]{1,32}\)", line, flags=re.IGNORECASE):
                    accent = self._hex_to_rgb(item.get("accent_color", ""))
                    if accent is not None:
                        line_rgb = accent
            if render:
                if self._should_whiteout_per_line(item):
                    self._whiteout_rect(page, slot, pad_x=0.8, pad_y=0.35)
                if item.get("keep_exact_line") and inline_segment_parts:
                    for seg_part in inline_segment_parts:
                        seg_rect = fitz.Rect(seg_part["bbox"].x0 + dx, seg_part["bbox"].y0 + dy, seg_part["bbox"].x1 + dx, seg_part["bbox"].y1 + dy)
                        seg_fs = float(seg_part["fontsize"])
                        seg_h = max(1.0, seg_rect.height)
                        seg_baseline = seg_rect.y0 + min(seg_h * 0.82, seg_h - 1.0)
                        self._safe_insert_text_dedup(
                            page,
                            (seg_rect.x0, seg_baseline),
                            seg_part["text"],
                            seg_fs,
                            seg_part["fontname"],
                            seg_part["rgb"],
                        )
                else:
                    line_h = max(1.0, fs * 1.22)
                    line_rect = fitz.Rect(
                        line_x,
                        baseline - line_h * 0.82,
                        line_x + line_w,
                        baseline + max(1.0, line_h * 0.18),
                    )
                    self._safe_insert_text_dedup(page, (line_x, baseline), line, fs, fontname, line_rgb)
            if render and self.style_audit_enabled:
                exp_color = effective_style.get("color", "#000000")
                app_color = "#%02x%02x%02x" % (
                    int(max(0.0, min(1.0, line_rgb[0])) * 255),
                    int(max(0.0, min(1.0, line_rgb[1])) * 255),
                    int(max(0.0, min(1.0, line_rgb[2])) * 255),
                )
                self._style_audit_records.append(
                    {
                        "page": int(page.number) + 1,
                        "role": item.get("role", "body"),
                        "primary_structure_family": self._descriptor_v3_primary_family(item),
                        "structure_priority": self._descriptor_v3_structure_priority(item),
                        "style_lock_source": current_style_lock_source,
                        "expected_alignment": expected_align,
                        "applied_alignment": applied_align,
                        "alignment_raw": item.get("alignment_raw", ""),
                        "alignment_source": item.get("alignment_source", "block"),
                        "alignment_defaulted": bool(item.get("alignment_defaulted", False)),
                        "alignment_fallback_reason": align_fallback_reason or item.get("alignment_fallback_reason", ""),
                        "expected_font": effective_style.get("font"),
                        "applied_font": fontname,
                        "font_fallback": (fontname in {"helv", "times", "courier"} and str(effective_style.get("font", "")).strip().lower() not in {"", "helv", "times", "courier", "arial", "helvetica"}),
                        "expected_size_pt": float(base_fs),
                        "applied_size_pt": float(fs),
                        "size_delta_pt": float(fs - base_fs),
                        "slot_width_pt": float(slot_w),
                        "line_width_pt": float(line_w),
                        "line_x_pt": float(line_x),
                        "expected_color": exp_color if str(exp_color).startswith("#") else f"#{str(exp_color).lstrip('#')}",
                        "applied_color": app_color,
                    }
                )
            used_bottom = max(used_bottom, slot.y1)
            used_slots.append(fitz.Rect(slot))
            prev_slot_bottom = slot.y1
            if body_stable_vertical:
                stable_next_y = slot.y1 + stable_gap_y

        remaining = "\n".join(preset_lines).strip() if preserve_linebreaks else " ".join(words).strip()
        blue_rect = fitz.Rect(x0, y0, block_right, max(y0 + item["slot_h_pt"], used_bottom))
        return remaining, used_bottom, blue_rect, used_slots

    def _resolve_text_color(self, style, item):
        # WYSIWYG-first: keep extracted color exactly.
        try:
            c = style.get("color", "#000000").lstrip("#")
            if len(c) != 6:
                return (0, 0, 0)
            rgb = tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
            return rgb
        except Exception:
            return (0, 0, 0)

    def _hex_to_rgb(self, hex_color):
        try:
            c = str(hex_color or "").strip().lstrip("#")
            if len(c) != 6:
                return None
            return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except Exception:
            return None

    def _safe_insert_text_dedup(self, page, point, text, fontsize, fontname, color):
        sig = (round(point[0], 2), round(point[1], 2), round(float(fontsize), 2), (text or "").strip())
        if sig in self._rendered_signatures:
            return
        self._rendered_signatures.add(sig)
        self._safe_insert_text(page, point, text, fontsize, fontname, color)

    def _append_debug_rects(self, debug_store, page, blue_rect, red_rects):
        if debug_store is None:
            return
        page_number = getattr(page, "number", None)
        if page_number is None:
            return
        key = int(page_number)
        slot = debug_store.setdefault(key, {"blue": [], "red": []})
        if blue_rect is not None:
            slot["blue"].append(fitz.Rect(blue_rect))
        for r in red_rects or []:
            slot["red"].append(fitz.Rect(r))

    def _save_layout_debug_overlays(self, doc, debug_store, output_path):
        out_dir = os.path.dirname(output_path) or "."
        base = os.path.splitext(os.path.basename(output_path))[0]
        def _norm_debug_rect(rect):
            try:
                r = fitz.Rect(rect).normalize()
            except Exception:
                return None
            if r.x1 < r.x0 or r.y1 < r.y0:
                return None
            return r
        for page_idx, rects in debug_store.items():
            if page_idx < 0 or page_idx >= len(doc):
                continue
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=self.layout_debug_dpi, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            draw = ImageDraw.Draw(img)
            s = float(self.layout_debug_dpi) / 72.0
            for r in rects.get("blue", []):
                r = _norm_debug_rect(r)
                if r is None:
                    continue
                draw.rectangle([r.x0 * s, r.y0 * s, r.x1 * s, r.y1 * s], outline=(0, 90, 255), width=3)
            for r in rects.get("red", []):
                r = _norm_debug_rect(r)
                if r is None:
                    continue
                draw.rectangle([r.x0 * s, r.y0 * s, r.x1 * s, r.y1 * s], outline=(255, 0, 0), width=1)
            out_path = os.path.join(out_dir, f"{base}_layout_debug_p{page_idx + 1}.jpg")
            img.save(out_path, quality=92)

    def _extract_sequential_items(self, page_data):
        items = []
        seen = []
        for block in page_data.get("blocks", []):
            if block.get("render_mode") == "background_only":
                continue
            source = block.get("source", "ocr")
            block_align = block.get("alignment", "left")
            block_role = block.get("role", "body")
            block_indent_px = float(block.get("indent_px", 0.0) or 0.0)
            block_style = self._style_from_block(block)
            for line in block.get("lines", []):
                for phrase in line.get("phrases", []):
                    if phrase.get("render_mode") == "background_only":
                        continue
                    text = re.sub(r"\s+", " ", (phrase.get("texte") or "").strip())
                    if not text:
                        continue
                    bbox = phrase.get("bbox") or line.get("bbox") or block.get("bbox") or [0, 0, 0, 0]
                    if len(bbox) != 4:
                        continue
                    rect_pt = fitz.Rect([float(v) * self.pixel_to_point for v in bbox])
                    key = (self._text_key(text), round(rect_pt.x0, 1), round(rect_pt.y0, 1))
                    if key in seen:
                        continue
                    seen.append(key)
                    style = dict(block_style)
                    if phrase.get("spans"):
                        style = self._merge_styles(style, phrase["spans"][0].get("style", {}))
                    items.append(
                        {
                            "text": text,
                            "bbox": rect_pt,
                            "style": self._merge_styles(style, {}),
                            "source": source,
                            "alignment": phrase.get("alignment", line.get("alignment", block_align)),
                            "role": phrase.get("role", line.get("role", block_role)),
                            "indent_pt": float(phrase.get("indent_px", line.get("indent_px", block_indent_px))) * self.pixel_to_point,
                        }
                    )
        items.sort(key=lambda it: (it["bbox"].y0, it["bbox"].x0))
        return items

    def _compute_aligned_x(self, alignment, line_w, left, right, preferred_x, is_last_line=False):
        avail_w = max(10.0, right - left)
        if line_w >= avail_w:
            return left
        align = (alignment or "left").lower()
        if align == "center":
            return max(left, min((left + right - line_w) / 2.0, right - line_w))
        if align == "right":
            return max(left, right - line_w)
        if align == "justify" and not is_last_line:
            return left
        return max(left, min(preferred_x, right - line_w))

    def _render_anchored_item(self, page, item, left, right, zone_top, zone_bottom, right_safety):
        style = item["style"]
        source = item["source"]
        text = item["text"]
        if not text or zone_bottom <= zone_top:
            return

        _, fontfile, builtin, fontname = self._resolve_style_font(page, style, text=text)
        fs = self._get_original_fontsize(style, max(1.0, item["bbox"].height), source)
        line_h = max(1.0, fs * 1.22)
        rgb = self._resolve_text_color(style, item)

        start_x = max(left, min(item["bbox"].x0, right - 40.0))
        max_w = max(30.0, right - right_safety - start_x)
        lines = self._wrap_text_lines(text, max_w, fs, fontname, fontfile)
        if not lines:
            return

        max_lines = max(1, int((zone_bottom - zone_top) / line_h))
        lines = lines[:max_lines]
        y = max(zone_top, min(item["bbox"].y0, zone_bottom - line_h * len(lines)))
        for i, line in enumerate(lines):
            line_w = self._measure_text_width(line, fs, fontname, fontfile)
            line_x = self._compute_aligned_x(
                alignment=item.get("alignment", "left"),
                line_w=line_w,
                left=left,
                right=right - right_safety,
                preferred_x=start_x,
                is_last_line=(i == len(lines) - 1),
            )
            baseline = y + line_h * 0.82
            self._safe_insert_text_dedup(page, (line_x, baseline), line, fs, fontname, rgb)
            y += line_h

    def _extract_flow_items(self, page_data, forbidden_rects):
        items = []
        for block in page_data.get("blocks", []):
            if block.get("render_mode") == "background_only":
                continue
            source = block.get("source", "ocr")
            block_align = block.get("alignment", "left")
            block_role = block.get("role", "body")
            b = block.get("bbox", [0, 0, 10, 10])
            bbox = fitz.Rect([float(v) * self.pixel_to_point for v in b])
            text_parts = []
            spans = []
            style = self._style_from_block(block)
            for line in block.get("lines", []):
                for phrase in line.get("phrases", []):
                    if phrase.get("render_mode") == "background_only":
                        continue
                    ptxt = self._phrase_text_for_render(phrase)
                    if ptxt:
                        text_parts.append(ptxt)
                    for sp in phrase.get("spans", []):
                        spans.append(sp)
                        style = self._merge_styles(style, sp.get("style", {}))
            text = re.sub(r"\s+", " ", " ".join(text_parts)).strip()
            if not text and spans:
                text = re.sub(r"\s+", " ", " ".join((s.get("texte") or "") for s in spans)).strip()
            if not text:
                continue
            kind = self._classify_block_kind(bbox, text, forbidden_rects)
            items.append(
                {
                    "kind": kind,
                    "text": text,
                    "bbox": bbox,
                    "style": self._merge_styles(style, {}),
                    "source": source,
                    "spans": spans,
                    "alignment": block_align,
                    "role": block_role,
                }
            )
        items.sort(key=lambda it: (it["bbox"].y0, it["bbox"].x0))
        return items

    def _classify_block_kind(self, bbox, text, forbidden_rects):
        upper_ratio = 0.0
        letters = [c for c in text if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
        # Text intersecting figure zones is treated as diagram label.
        if self._has_overlap(bbox, forbidden_rects):
            return "diagram_label"
        if len(text) <= 80 and upper_ratio > 0.6:
            return "caption"
        return "body"

    def _build_flow_frames(self, page_rect, flow_items, forbidden_rects):
        if flow_items:
            min_x = min(it["bbox"].x0 for it in flow_items)
            max_x = max(it["bbox"].x1 for it in flow_items)
            x0 = max(page_rect.x0 + 8.0, min_x)
            x1 = min(page_rect.x1 - 8.0, max_x)
            if x1 - x0 < 60:
                x0, x1 = page_rect.x0 + 24.0, page_rect.x1 - 24.0
        else:
            x0, x1 = page_rect.x0 + 24.0, page_rect.x1 - 24.0

        top = page_rect.y0 + 14.0
        bottom = page_rect.y1 - 14.0
        obs = []
        for z in forbidden_rects:
            rz = fitz.Rect(z)
            if rz.x1 <= x0 or rz.x0 >= x1:
                continue
            obs.append(rz)
        obs.sort(key=lambda r: r.y0)

        frames = []
        cur_y = top
        pad = self.flow_zone_pad
        for z in obs:
            y0 = max(top, z.y0 - pad)
            y1 = min(bottom, z.y1 + pad)
            if y0 - cur_y >= 14:
                frames.append(fitz.Rect(x0, cur_y, x1, y0))
            cur_y = max(cur_y, y1)
        if bottom - cur_y >= 14:
            frames.append(fitz.Rect(x0, cur_y, x1, bottom))
        return frames

    def _wrap_text_lines(self, text, max_w, fontsize, fontname, fontfile):
        words = text.split()
        if not words:
            return []
        normalized_words = []
        for w in words:
            if self._measure_text_width(w, fontsize, fontname, fontfile) <= max_w:
                normalized_words.append(w)
                continue
            # Split very long token so it can still be rendered.
            chunk = ""
            for ch in w:
                candidate = chunk + ch
                if chunk and self._measure_text_width(candidate, fontsize, fontname, fontfile) > max_w:
                    normalized_words.append(chunk)
                    chunk = ch
                else:
                    chunk = candidate
            if chunk:
                normalized_words.append(chunk)

        lines = []
        cur = normalized_words[0]
        for w in normalized_words[1:]:
            cand = f"{cur} {w}"
            if self._measure_text_width(cand, fontsize, fontname, fontfile) <= max_w:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    def _place_item_in_frames(self, page, item, frames, frame_idx, cursor_y, placed_rects, forbidden_rects):
        style = item["style"]
        source = item["source"]
        _, fontfile, builtin, fontname = self._resolve_style_font(page, style, text=item["text"])
        base_fs = self._get_original_fontsize(style, max(1.0, item["bbox"].height), source)
        min_fs = max(self.flow_min_font_pt, base_fs * self.flow_min_font_scale)
        para_gap = max(2.0, base_fs * 0.55 if item["kind"] == "body" else base_fs * 0.35)
        rgb = self._resolve_text_color(style, item)

        remaining_text = item["text"]
        while frame_idx < len(frames):
            frame = frames[frame_idx]
            cur_y = max(cursor_y, frame.y0)
            desired_w = max(40.0, min(item["bbox"].width, frame.x1 - frame.x0))
            start_x = max(frame.x0, min(item["bbox"].x0, frame.x1 - desired_w))
            right_pad = self._line_right_padding_for_item(item, base_fs, strict=False)
            max_w = max(20.0, frame.x1 - start_x - right_pad)

            fs = base_fs
            line_h = max(1.0, fs * 1.22)
            avail_lines = int((frame.y1 - cur_y) / line_h)
            while avail_lines <= 0 and fs > min_fs + 1e-6:
                fs = max(min_fs, fs * 0.92)
                line_h = max(1.0, fs * 1.22)
                avail_lines = int((frame.y1 - cur_y) / line_h)
            if avail_lines <= 0:
                frame_idx += 1
                if frame_idx < len(frames):
                    cursor_y = frames[frame_idx].y0
                continue

            lines = self._wrap_text_lines(remaining_text, max_w, fs, fontname, fontfile)
            if not lines:
                return frame_idx, cur_y
            take = min(len(lines), max(1, avail_lines))
            chunk = lines[:take]

            y = cur_y
            for line in chunk:
                baseline = y + line_h * 0.82
                line_w = self._measure_text_width(line, fs, fontname, fontfile)
                line_rect = fitz.Rect(
                    start_x,
                    baseline - line_h * 0.82,
                    start_x + line_w,
                    baseline + max(1.0, line_h * 0.18),
                )
                self._safe_insert_text_dedup(page, (start_x, baseline), line, fs, fontname, rgb)
                y = line_rect.y1

            rendered_rect = fitz.Rect(start_x, cur_y, start_x + max_w, y)
            rendered_rect = self._clamp_rect_to_page(rendered_rect, page.rect)
            if not self._has_overlap(rendered_rect, forbidden_rects):
                placed_rects.append(rendered_rect)

            if take >= len(lines):
                return frame_idx, y + para_gap

            remaining_text = " ".join(lines[take:])
            frame_idx += 1
            if frame_idx < len(frames):
                cursor_y = frames[frame_idx].y0

        # Last resort: clamp inside final frame (no off-page rendering).
        frame = frames[-1]
        fs = max(min_fs, base_fs * 0.85)
        line_h = max(1.0, fs * 1.22)
        y = max(frame.y0, min(cursor_y, frame.y1 - line_h))
        desired_w = max(40.0, min(item["bbox"].width, frame.x1 - frame.x0))
        start_x = max(frame.x0, min(item["bbox"].x0, frame.x1 - desired_w))
        right_pad = self._line_right_padding_for_item(item, fs, strict=True)
        max_w = max(20.0, frame.x1 - start_x - right_pad)
        lines = self._wrap_text_lines(remaining_text, max_w, fs, fontname, fontfile)
        max_lines = max(1, int((frame.y1 - y) / line_h))
        lines = lines[:max_lines]
        if lines:
            lines[-1] = lines[-1] + " ..."
        for line in lines:
            baseline = y + line_h * 0.82
            line_w = self._measure_text_width(line, fs, fontname, fontfile)
            line_rect = fitz.Rect(
                start_x,
                baseline - line_h * 0.82,
                start_x + line_w,
                baseline + max(1.0, line_h * 0.18),
            )
            self._safe_insert_text_dedup(page, (start_x, baseline), line, fs, fontname, rgb)
            y = line_rect.y1
        return len(frames) - 1, min(frame.y1, y + para_gap)

    def _insert_hierarchical_span(self, page, span, source="ocr", placed_rects=None, forbidden_rects=None, allow_shift=True):
        text = span.get("texte", "")
        if not text: return
        style = span.get("style", {})
        bbox = span.get("bbox", [0,0,10,10])
        x0, y0, x1, y1 = [c * self.pixel_to_point for c in bbox]

        _, fontfile, builtin, fontname = self._resolve_style_font(page, style, text=text)

        fs = self._get_original_fontsize(style, y1 - y0, source)
        natural_w = self._measure_text_width(text, fs, fontname, fontfile)
        natural_h = max(fs * 1.15, (y1 - y0) if (y1 - y0) > 0 else fs)

        # Couleur
        rgb = self._resolve_text_color(style, {"style": style, "source": source})

        # Baseline calibration: avoid hardcoded offset and stabilize native placement.
        baseline_ratio = self._baseline_ratio(style, fs)
        baseline_y = y0 + (fs * baseline_ratio)
        target_rect = fitz.Rect(x0, baseline_y - natural_h * 0.8, x0 + natural_w, baseline_y + natural_h * 0.2)
        if source == "native":
            allow_shift = False
        if self.layout_correction and allow_shift:
            target_rect = self._find_non_overlapping_rect(
                target_rect=target_rect,
                page_rect=page.rect,
                placed_rects=placed_rects or [],
                forbidden_rects=forbidden_rects or [],
                step=max(fs * 0.9, 1.0),
            )
        # Always keep rendered text inside page bounds.
        target_rect = self._clamp_rect_to_page(target_rect, page.rect)
        baseline_y = target_rect.y1 - natural_h * 0.2

        try:
            self._safe_insert_text(page, (target_rect.x0, baseline_y), text, fs, fontname, rgb)
            span_item = {
                "source_text": span.get("source_text", text),
                "text": text,
                "translated_block": span.get("translated_block", False),
            }
            if placed_rects is not None:
                placed_rects.append(target_rect)
        except Exception as e:
            print(f"Erreur rendu span: {e}")

    def _safe_insert_text(self, page, point, text, fontsize, fontname, color):
        try:
            page.insert_text(point, text, fontsize=fontsize, fontname=fontname, color=color)
        except Exception:
            # Last-resort stable fallback to built-in Helvetica.
            page.insert_text(point, text, fontsize=fontsize, fontname="helv", color=color)

    def _get_original_fontsize(self, style, bbox_h_pt, source):
        raw_size = style.get("size")
        if isinstance(raw_size, (int, float)) and raw_size > 0:
            # Native spans already carry point sizes from PDF extraction.
            if source == "native":
                return float(raw_size)
            # OCR spans carry pixel-like sizes; convert to points on target page.
            return float(raw_size) * self.pixel_to_point
        raw_pt = style.get("font_size_pt")
        if isinstance(raw_pt, (int, float)) and raw_pt > 0:
            return float(raw_pt)
        return max(1.0, (bbox_h_pt * 0.9))

    def _fontsize_from_bbox(self, bbox, fallback=None):
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                height_pt = max(0.0, (float(bbox[3]) - float(bbox[1])) * self.pixel_to_point)
                if height_pt > 0.0:
                    return height_pt
            except Exception:
                pass
        return fallback

    def _collect_forbidden_rects(self, page_data):
        rects = []
        for z in page_data.get("non_text_zones", []):
            if not isinstance(z, (list, tuple)) or len(z) != 4:
                continue
            x0, y0, x1, y1 = [float(v) * self.pixel_to_point for v in z]
            rects.append(fitz.Rect(x0, y0, x1, y1))
        for im in page_data.get("images", []):
            bbox = im.get("bbox") if isinstance(im, dict) else im
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(v) * self.pixel_to_point for v in bbox]
            rects.append(fitz.Rect(x0, y0, x1, y1))
        for ov in page_data.get("immutable_overlays", []):
            bbox = ov.get("bbox") if isinstance(ov, dict) else None
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(v) * self.pixel_to_point for v in bbox]
            rects.append(fitz.Rect(x0, y0, x1, y1))
        return rects

    def _insert_immutable_overlays(self, page, page_data):
        page_role = str((page_data or {}).get("page_role", "")).strip().lower()
        if page_role == "toc":
            return
        translated_rects = []
        if self._has_translated_content(page_data):
            for block in (page_data.get("blocks") or []):
                if block.get("render_mode") == "background_only":
                    continue
                if not self._is_translated_block(block):
                    continue
                bb = block.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    try:
                        br = fitz.Rect([float(v) * self.pixel_to_point for v in bb])
                        if br.get_area() > 0:
                            translated_rects.append(br)
                    except Exception:
                        pass
        for ov in page_data.get("immutable_overlays", []):
            path = ov.get("path") if isinstance(ov, dict) else None
            bbox = ov.get("bbox") if isinstance(ov, dict) else None
            kind = str(ov.get("kind") or ov.get("reason") or "").strip().lower() if isinstance(ov, dict) else ""
            ov_text = str(ov.get("text", "")).strip() if isinstance(ov, dict) else ""
            if not path or not os.path.exists(path):
                continue
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            area_px = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
            if kind == "immutable_inline" and area_px < 4500:
                continue
            if page_role == "toc" and re.fullmatch(r"\d{1,4}|[ivxlcdm]+", ov_text, flags=re.IGNORECASE):
                continue
            x0, y0, x1, y1 = [float(v) * self.pixel_to_point for v in bbox]
            rect = fitz.Rect(x0, y0, x1, y1)
            if rect.get_area() <= 0:
                continue
            if translated_rects:
                overlap = False
                for tr in translated_rects:
                    inter = (rect & tr).get_area()
                    if inter <= 0:
                        continue
                    ratio = inter / max(1e-9, min(rect.get_area(), tr.get_area()))
                    if ratio >= 0.2:
                        overlap = True
                        break
                if overlap:
                    continue
            try:
                page.insert_image(rect, filename=path, overlay=True, keep_proportion=False)
            except Exception:
                continue

    def _text_key(self, text):
        return re.sub(r"\s+", " ", (text or "").strip().lower())

    def _is_duplicate_span(self, span, seen_spans):
        text_key = self._text_key(span.get("texte", ""))
        if not text_key:
            return False
        b = span.get("bbox", [0, 0, 0, 0])
        if len(b) != 4:
            return False
        r = fitz.Rect([v * self.pixel_to_point for v in b])
        for prev_r, prev_key in seen_spans:
            if prev_key != text_key:
                continue
            inter = (r & prev_r).get_area()
            if inter <= 0:
                continue
            ratio = inter / max(1e-9, min(r.get_area(), prev_r.get_area()))
            if ratio >= 0.6:
                return True
        seen_spans.append((r, text_key))
        return False

    def _has_overlap(self, rect, others):
        for o in others:
            inter = (rect & o).get_area()
            if inter <= 0:
                continue
            ratio = inter / max(1e-9, min(rect.get_area(), o.get_area()))
            if ratio >= self.overlap_threshold:
                return True
        return False

    def _find_non_overlapping_rect(self, target_rect, page_rect, placed_rects, forbidden_rects, step):
        rect = fitz.Rect(target_rect)
        # Keep initial X if possible and search downward first.
        for _ in range(max(0, self.max_shift_steps) + 1):
            if rect.y0 >= page_rect.y0 and rect.y1 <= page_rect.y1:
                if not self._has_overlap(rect, placed_rects) and not self._has_overlap(rect, forbidden_rects):
                    return rect
            x_step = max(4.0, step * 0.9)
            for dx in (x_step, -x_step, x_step * 2.0, -x_step * 2.0):
                cand = fitz.Rect(rect.x0 + dx, rect.y0, rect.x1 + dx, rect.y1)
                cand = self._clamp_rect_to_page(cand, page_rect)
                if cand.y0 >= page_rect.y0 and cand.y1 <= page_rect.y1:
                    if not self._has_overlap(cand, placed_rects) and not self._has_overlap(cand, forbidden_rects):
                        return cand
            rect = fitz.Rect(rect.x0, rect.y0 + step, rect.x1, rect.y1 + step)
        return target_rect

    def _clamp_rect_to_page(self, rect, page_rect):
        r = fitz.Rect(rect)
        page_w = page_rect.x1 - page_rect.x0
        page_h = page_rect.y1 - page_rect.y0
        rect_w = r.x1 - r.x0
        rect_h = r.y1 - r.y0

        # If rectangle is larger than page on an axis, force full-axis coverage.
        if rect_w >= page_w:
            r.x0, r.x1 = page_rect.x0, page_rect.x1
        else:
            if r.x0 < page_rect.x0:
                dx = page_rect.x0 - r.x0
                r = fitz.Rect(r.x0 + dx, r.y0, r.x1 + dx, r.y1)
            if r.x1 > page_rect.x1:
                dx = r.x1 - page_rect.x1
                r = fitz.Rect(r.x0 - dx, r.y0, r.x1 - dx, r.y1)

        if rect_h >= page_h:
            r.y0, r.y1 = page_rect.y0, page_rect.y1
        else:
            if r.y0 < page_rect.y0:
                dy = page_rect.y0 - r.y0
                r = fitz.Rect(r.x0, r.y0 + dy, r.x1, r.y1 + dy)
            if r.y1 > page_rect.y1:
                dy = r.y1 - page_rect.y1
                r = fitz.Rect(r.x0, r.y0 - dy, r.x1, r.y1 - dy)
        return r

    def _resolve_page_fontname(self, page, fontfile, builtin):
        if not fontfile:
            return builtin or "helv"

        key = (id(page), fontfile)
        alias = self._page_font_aliases.get(key)
        if alias:
            return alias

        alias = f"F{len(self._page_font_aliases) + 1}"
        try:
            page.insert_font(fontname=alias, fontfile=fontfile)
            self._page_font_aliases[key] = alias
            return alias
        except Exception:
            return builtin or "helv"

    def _measure_text_width(self, text, fontsize, fontname, fontfile):
        try:
            if fontfile:
                fobj = self._font_objects.get(fontfile)
                if fobj is None:
                    fobj = fitz.Font(fontfile=fontfile)
                    self._font_objects[fontfile] = fobj
                return fobj.text_length(text, fontsize=fontsize)
            return fitz.get_text_length(text, fontname=fontname, fontsize=fontsize)
        except Exception:
            return fitz.get_text_length(text, fontname="helv", fontsize=fontsize)
