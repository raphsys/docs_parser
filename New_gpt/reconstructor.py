import fitz
import os
import re
import uuid
import unicodedata
import json
import xml.etree.ElementTree as ET
from statistics import median
from PIL import Image, ImageDraw
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
        self.dynamic_equation_overlays = os.getenv("LAYOUT_DYNAMIC_EQUATION_OVERLAYS", "1") == "1"
        self.dynamic_symbol_overlays = os.getenv("LAYOUT_DYNAMIC_SYMBOL_OVERLAYS", "1") == "1"
        self.dynamic_risk_overlays = os.getenv("LAYOUT_DYNAMIC_RISK_OVERLAYS", "1") == "1"
        self.pro_strict_mode = os.getenv("LAYOUT_PRO_STRICT", "1") == "1"
        self.dynamic_overlay_pad_px = int(os.getenv("LAYOUT_DYNAMIC_OVERLAY_PAD_PX", "1"))
        self.equation_diff_threshold = float(os.getenv("LAYOUT_EQUATION_DIFF_THRESHOLD", "22.0"))
        self.native_block_diff_threshold = float(os.getenv("LAYOUT_NATIVE_BLOCK_DIFF_THRESHOLD", "26.0"))
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

    def _style_from_block(self, block):
        if not isinstance(block, dict):
            return {}
        if isinstance(block.get("resolved_style"), dict):
            return block.get("resolved_style", {})
        if isinstance(block.get("style"), dict):
            return block.get("style", {})
        return {}

    def _normalize_alignment(self, alignment):
        a = (alignment or "left").strip().lower()
        if a not in {"left", "center", "right", "justify"}:
            return "left"
        # Justification must be explicitly detected.
        return a

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

    def _collect_translation_forbidden_rects(self, page_data):
        # In translation mode, keep only real blocking zones.
        # Small immutable inline overlays (symbols/references inside paragraphs)
        # should not push whole blocks down.
        rects = []
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
            area_px = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
            # Keep only large/structural overlays as blockers.
            if area_px < 4500 and kind not in {"diagram_block"}:
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

        for item in header_items:
            _, _, blue_rect, used_slots = self._render_block_slots(
                page=page,
                item=item,
                anchor_y=max(top, min(item["bbox"].y0, header_bottom - 8.0)),
                left=left,
                right=right,
                zone_top=top,
                zone_bottom=max(top + 8.0, min(header_bottom, bottom)),
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

        # Keep natural reading order and preserve relative Y as much as possible.
        body_items = [
            it
            for it in items
            if it.get("role") not in {"equation_inline", "diagram_text_label", "header", "footer", "figure_caption", "section_heading"}
            and not it.get("is_diagram_label")
        ]
        # Strict fidelity: keep source block segmentation to preserve natural
        # paragraph/list line breaks and avoid artificial merges.
        if not self.pro_strict_mode:
            body_items = self._merge_translated_body_items(body_items)
        fixed_items = [
            it
            for it in items
            if it.get("role") in {"equation_inline", "diagram_text_label", "header", "footer", "figure_caption", "section_heading", "list_marker"}
            or it.get("is_diagram_label")
        ]

        # Render fixed/sensitive items first at source location.
        for item in fixed_items:
            blue_rect = None
            if item.get("role") in {"header", "footer", "figure_caption", "section_heading"}:
                _, _, blue_rect, used_slots = self._render_block_slots(
                    page=page,
                    item=item,
                    anchor_y=max(top, min(item["bbox"].y0, bottom - 8.0)),
                    left=left,
                    right=right,
                    zone_top=max(top, item["bbox"].y0 - max(4.0, item.get("slot_h_pt", 8.0) * 0.6)),
                    zone_bottom=min(bottom, item["bbox"].y1 + max(8.0, item.get("slot_h_pt", 8.0) * 1.2)),
                )
                self._append_debug_rects(debug_store, page, blue_rect, used_slots)
            else:
                self._render_fixed_item(page, item)
            bb = item.get("bbox")
            add_forbidden = bool(
                item.get("is_diagram_label")
                or item.get("role") in {"figure_caption", "header", "footer", "section_heading", "list_marker"}
            )
            if add_forbidden and isinstance(bb, fitz.Rect) and bb.get_area() > 0:
                page_forbidden.append(fitz.Rect(bb))
            if add_forbidden and isinstance(blue_rect, fitz.Rect) and blue_rect.get_area() > 0:
                page_forbidden.append(fitz.Rect(blue_rect))

        for idx, item in enumerate(body_items):
            # Adding pages can invalidate previously held page handles in PyMuPDF.
            # Always refresh the root/source page before rendering the next block.
            page = doc[root_page_index]
            if item.get("preserve_linebreaks") and item.get("use_structured_source_lines"):
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
            next_y = bottom
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
            )
            self._append_debug_rects(debug_store, page, blue_rect, used_slots)
            if blue_rect is not None:
                page_forbidden.append(fitz.Rect(blue_rect))

            # Overflow continuation pages: keep next source blocks anchored on page 1.
            if remaining and self._should_paginate_on_overflow():
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
        style = item.get("style", {}) or {}
        source = item.get("source", "ocr")
        bbox = item.get("bbox")
        if not isinstance(bbox, fitz.Rect):
            return
        resolved = self.font_resolver.resolve(style)
        fontfile = resolved.get("fontfile")
        builtin = resolved.get("builtin")
        fontname = self._resolve_page_fontname(page, fontfile, builtin)
        fs = self._get_original_fontsize(style, max(1.0, bbox.height), source)
        try:
            c = style.get("color", "#000000").lstrip("#")
            rgb = tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except Exception:
            rgb = (0, 0, 0)
        br = self._baseline_ratio(style, fs)
        baseline = bbox.y0 + min(max(1.0, fs * br), max(1.0, bbox.height - 0.6))
        self._safe_insert_text_dedup(page, (bbox.x0, baseline), text, fs, fontname, rgb)

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
        for block in blocks:
            bbox = block.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            role = block.get("role", "body")
            text = self._get_block_text(block)
            is_translated = self._is_translated_block(block)
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
            if is_translated and role in {"diagram_label", "diagram_text_label"}:
                if not self._overlay_exists(overlays, bbox):
                    ov = self._save_crop_overlay(page_data, bbox, kind=f"{role}_translated")
                    if ov:
                        overlays.append(ov)
                block["render_mode"] = "background_only"
                for line in block.get("lines", []):
                    for phrase in line.get("phrases", []):
                        phrase["render_mode"] = "background_only"
                continue
            if self.pro_strict_mode and is_translated and self._block_should_be_image_locked(block, non_text_regions, diagram_regions):
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
        if block_tt and block_tt != block_src:
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
        dims = page_data.get("dimensions", {}) or {}
        page_w_pt = float(dims.get("width", 1000.0)) * self.pixel_to_point
        page_h_pt = float(dims.get("height", 1000.0)) * self.pixel_to_point
        page_lang = self._resolve_page_lang(page_data)
        for block in page_data.get("blocks", []):
            if block.get("render_mode") == "background_only":
                continue
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
                for phrase in line.get("phrases", []):
                    if phrase.get("render_mode") == "background_only":
                        continue
                    t = self._phrase_text_for_render(phrase)
                    if t:
                        text_parts.append(t)
                        this_line_parts.append(t)
                    if phrase.get("spans"):
                        phrase_style = phrase["spans"][0].get("style", {})
                        style = self._merge_styles(style, phrase_style)
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
                    if block_is_translated and block_role in {"diagram_text_label"}:
                        pb = phrase.get("bbox") or line.get("bbox") or block.get("bbox")
                        if isinstance(pb, (list, tuple)) and len(pb) == 4:
                            pr = fitz.Rect([float(v) * self.pixel_to_point for v in pb])
                            if pr.get_area() > 0:
                                pstyle = self._merge_styles(
                                    phrase["spans"][0].get("style", {}) if phrase.get("spans") else {},
                                    style,
                                )
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
                                        "alignment": self._normalize_alignment(
                                            phrase.get("alignment", line.get("alignment", block.get("alignment", "left")))
                                        ),
                                        "justify_explicit": (
                                            str(phrase.get("alignment", line.get("alignment", block.get("alignment", ""))))
                                            .strip()
                                            .lower()
                                            == "justify"
                                        ),
                                        "role": phrase.get("role", line.get("role", block.get("role", "body"))),
                                        "lang": (block.get("language") or page_lang or self._infer_text_lang(t)),
                                        "is_title": False,
                                        "is_diagram_label": False,
                                        "style_lock_source": "phrase",
                                    }
                                )
                    pb = phrase.get("bbox") or line.get("bbox")
                    if isinstance(pb, (list, tuple)) and len(pb) == 4:
                        r = fitz.Rect([float(v) * self.pixel_to_point for v in pb])
                        if r.get_area() > 0:
                            slots.append(r)
                if this_line_parts:
                    line_txt_src = ""
                    if block_is_translated:
                        line_txt_src = line.get("translated_text") or line.get("line_text") or ""
                    if not line_txt_src:
                        line_txt_src = " ".join(this_line_parts)
                    line_txt = self._clean_text_for_render(line_txt_src)
                    line_texts.append(line_txt)
                    line_marker = (line.get("leading_marker") or "").strip()
                    line_markers_used.append(line_marker)
                    lb = line.get("bbox") or block.get("bbox")
                    line_style = self._merge_styles(style, {})
                    for ph0 in line.get("phrases", []):
                        if ph0.get("spans"):
                            line_style = self._merge_styles(ph0["spans"][0].get("style", {}), line_style)
                            break
                    line_entries.append(
                        {
                            "text": line_txt,
                            "marker": line_marker,
                            "bbox": lb,
                            "indent_px": float(line.get("indent_px", 0.0) or 0.0),
                            "style": line_style,
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
            paragraph_flow_mode = bool(block.get("translation_compose_mode") == "paragraph_flow" and block.get("role") in {"body", "figure_caption"})
            if block_is_translated and paragraph_flow_mode:
                text = block_preferred_text or re.sub(r"\s+", " ", " ".join(text_parts)).strip()
            elif block_is_translated:
                # For translated layouts, rely on phrase/line aggregation to preserve local structure.
                text = re.sub(r"\s+", " ", " ".join(text_parts)).strip()
            else:
                text = block_preferred_text or re.sub(r"\s+", " ", " ".join(text_parts)).strip()
            text = self._clean_text_for_render(text)
            bb_for_title = block.get("bbox", [0, 0, 10, 10])
            try:
                by0_pt = float(bb_for_title[1]) * self.pixel_to_point
            except Exception:
                by0_pt = page_h_pt
            if (
                block_role in {"body", "title", "section_heading", "header"}
                and by0_pt <= page_h_pt * 0.16
                and re.match(r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9\s\-\']+\s+\d{1,3}$", text or "")
            ):
                m_end_num = re.match(r"^(.+?)\s+(\d{1,3})$", text)
                if m_end_num:
                    text = self._clean_text_for_render(f"{m_end_num.group(2)} {m_end_num.group(1)}")
            if not text:
                continue
            heading_candidate = self._clean_text_for_render(" ".join(first_run_text_parts))
            if (
                heading_candidate
                and by0_pt <= page_h_pt * 0.16
                and re.match(r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9\s\-\']+\s+\d{1,3}$", heading_candidate)
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
            )
            if block_is_translated and block_role in {"diagram_text_label"} and translated_phrase_items:
                items.extend(translated_phrase_items)
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

            gap_x = float(median(gaps_x)) if gaps_x else max(2.0, slot_h * 0.25)
            gap_y = float(median(gaps_y)) if gaps_y else max(3.0, slot_h * 0.35)
            row_start_x = min(s.x0 for s in row_slots)
            letters = [c for c in text if c.isalpha()]
            upper_ratio = (sum(1 for c in letters if c.isupper()) / max(1, len(letters))) if letters else 0.0
            flags = style.get("flags", {}) if isinstance(style, dict) else {}
            is_title = bool(
                block.get("role") in {"header", "title", "section_heading"}
                or (len(text) <= 140 and (flags.get("bold") or upper_ratio >= 0.55))
            )
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
            is_structural_role = block.get("role") in {
                "title",
                "section_heading",
                "header",
                "footer",
                "figure_caption",
                "diagram_label",
                "diagram_text_label",
            }
            has_hard_breaks = any(
                bool((ln.get("hard_break_before") if isinstance(ln, dict) else False))
                for ln in block.get("lines", [])
            )
            item_lang = block.get("language") or page_lang or self._infer_text_lang(text)
            source_lines_for_render = []
            for i, raw_line in enumerate(line_texts):
                lt = self._clean_text_for_render(raw_line)
                marker = self._normalize_leading_marker(line_markers[i] if i < len(line_markers) else "")
                if marker:
                    has_any_marker = bool(
                        re.match(r"^\s*(?:[•▪◦·\-\*]|\d+[.)]|[A-Za-z][.)])\s+", lt or "")
                    )
                    if not has_any_marker:
                        lt = f"{marker} {lt}".strip()
                    if block_is_translated and block_role == "body" and i < len(line_entries):
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
                                        "alignment": "left",
                                        "justify_explicit": False,
                                        "role": "list_marker",
                                        "lang": item_lang,
                                        "is_title": False,
                                        "is_diagram_label": False,
                                    }
                                )
                        lt = re.sub(r"^\s*(?:[•▪◦·\-\*]|\d+[.)]|[A-Za-z][.)])\s*", "", lt).strip()
                # If previous line is a dedicated marker-only line (e.g. "1", "2"),
                # strip accidental duplicated leading marker in current text line.
                if i > 0 and i - 1 < len(line_entries):
                    prev_txt = self._clean_text_for_render(line_entries[i - 1].get("text", ""))
                    if re.match(r"^\s*(?:\d+[.)]?|[•▪◦·\-\*])\s*$", prev_txt):
                        lt = re.sub(r"^\s*(?:\d+[.)]?\s+|[•▪◦·\-\*]\s+)", "", lt).strip()
                source_lines_for_render.append(lt)
            # Keep numeric/list markers (e.g. standalone "1", "2") fixed at original
            # location; remove them from flowing body text to avoid displacement.
            if block_is_translated and block_role == "body" and line_entries:
                kept_lines = []
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
                    if is_marker_only:
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
                source_lines_for_render = kept_lines
            preserve_linebreaks = bool(
                len(line_texts) >= 1
                and (
                    is_structural_role
                    or has_list_markers
                    or has_number_only_lines
                    or has_hard_breaks
                )
            )
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
                            "alignment": self._normalize_alignment(block.get("alignment", "left")),
                            "justify_explicit": False,
                            "role": "section_heading",
                            "lang": item_lang,
                            "is_title": True,
                            "is_diagram_label": False,
                            "accent_color": heading_color or accent_color,
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
                            "alignment": self._normalize_alignment(block.get("alignment", "left")),
                            "justify_explicit": False,
                            "role": "section_heading",
                            "lang": item_lang,
                            "is_title": True,
                            "is_diagram_label": False,
                            "accent_color": first_run_style.get("color", "") if isinstance(first_run_style, dict) else "",
                        }
                    )
                    bbox = fitz.Rect(bbox.x0, min(bbox.y1, first_run_bbox.y1 + 1.0), bbox.x1, bbox.y1)
                    row_slots = [fitz.Rect(r) for r in row_slots if r.y0 >= first_run_bbox.y1 - 0.5]
                    if not row_slots:
                        row_slots = [fitz.Rect(bbox)]
                text = body_text or text
            items.append(
                {
                    "text": text,
                    "source_lines": (source_lines_for_render or block.get("line_texts") or line_texts),
                    "preserve_linebreaks": preserve_linebreaks,
                    "use_structured_source_lines": bool(has_list_markers or has_hard_breaks or has_number_only_lines),
                    "allow_line_overflow": bool(
                        block_is_translated and (is_structural_role or has_list_markers or has_number_only_lines)
                    ),
                    "bbox": bbox,
                    "slots": row_slots,
                    "slot_w_pt": max(10.0, bbox.width),
                    "slot_h_pt": slot_h,
                    "slot_gap_x_pt": gap_x,
                    "slot_gap_y_pt": gap_y,
                    "row_start_x_pt": row_start_x,
                    "style": self._merge_styles(style, {}),
                    "source": source,
                    "alignment": self._normalize_alignment(block.get("alignment", "left")),
                    "justify_explicit": str(block.get("alignment", "")).strip().lower() == "justify",
                    "role": block.get("role", "body"),
                    "lang": item_lang,
                    "is_title": is_title,
                    "is_diagram_label": is_diagram_label,
                    "accent_color": accent_color,
                }
            )

            # Header often contains "page-number + title" merged in one OCR line.
            if block.get("role") == "header":
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
                        items[-1]["alignment"] = "center"
                        num_w = max(26.0, min(44.0, bbox.width * 0.22))
                        num_bbox = fitz.Rect(bbox.x0, bbox.y0, bbox.x0 + num_w, bbox.y1)
                        items.append(
                            {
                                "text": num_txt,
                                "bbox": num_bbox,
                                "slots": [fitz.Rect(num_bbox)],
                                "slot_w_pt": num_w,
                                "slot_h_pt": slot_h,
                                "slot_gap_x_pt": gap_x,
                                "slot_gap_y_pt": gap_y,
                                "row_start_x_pt": num_bbox.x0,
                                "style": self._merge_styles(style, {}),
                                "source": source,
                                "alignment": "right",
                                "justify_explicit": False,
                                "role": "header",
                                "is_title": True,
                                "is_diagram_label": False,
                            }
                        )
        for it in items:
            try:
                bb = it.get("bbox")
                txt = self._clean_text_for_render(it.get("text", ""))
                if not isinstance(bb, fitz.Rect) or not txt:
                    continue
                if bb.y0 <= page_h_pt * 0.16 and re.match(r"^[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9\s\-\']+\s+\d{1,3}$", txt):
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

    def _render_block_slots(self, page, item, anchor_y, left, right, zone_top, zone_bottom, override_text=None, render=True):
        text = self._clean_text_for_render(override_text if override_text is not None else item.get("text", "")).strip()
        if not text:
            return "", anchor_y, None, []
        style = item["style"]
        source = item["source"]
        resolved = self.font_resolver.resolve(style)
        fontfile = resolved.get("fontfile")
        builtin = resolved.get("builtin")
        fontname = self._resolve_page_fontname(page, fontfile, builtin)
        base_fs = self._get_original_fontsize(style, max(1.0, item["slot_h_pt"]), source)
        rgb = self._resolve_text_color(style, item)

        block_w = max(10.0, item["bbox"].width)
        x0 = max(left, min(item["bbox"].x0, right - block_w))
        y0 = max(zone_top, anchor_y)
        dx = x0 - item["bbox"].x0
        dy = y0 - item["bbox"].y0
        block_right = min(right, x0 + block_w)
        role = item.get("role")
        # Strict slot anchoring requested:
        # - line start must match phrase slot x0
        # - line wrapping must use blue box right edge
        # - no center/right/justify alignment
        body_stable_left = False
        # Professional paragraph rule: keep stable vertical rhythm (line starts + line heights)
        # for body text to avoid "stair-step" OCR line jitter.
        body_stable_vertical = body_stable_left
        paragraph_left_x = max(left, min(x0, block_right - 6.0))
        stable_slot_h = max(8.0, float(item.get("slot_h_pt", 10.0)))
        stable_gap_y = max(1.8, min(7.5, float(item.get("slot_gap_y_pt", stable_slot_h * 0.28))))
        stable_next_y = y0
        # Smart horizontal expansion for short/top metadata and headings.
        if role in {"header", "footer"}:
            block_right = right
        elif item.get("is_title") and block_w < (right - left) * 0.72:
            block_right = min(right, x0 + max(block_w, (right - left) * 0.72))

        slots = [fitz.Rect(s.x0 + dx, s.y0 + dy, s.x1 + dx, s.y1 + dy) for s in item["slots"]]
        slots.sort(key=lambda r: (r.y0, r.x0))
        preserve_linebreaks = bool(item.get("preserve_linebreaks"))
        preset_lines = []
        if preserve_linebreaks:
            if override_text is not None:
                preset_lines = [self._clean_text_for_render(x).strip() for x in str(override_text).split("\n") if x.strip()]
            else:
                preset_lines = [self._clean_text_for_render(x).strip() for x in item.get("source_lines", []) if x and x.strip()]
        words = [] if preserve_linebreaks else text.split()
        used_bottom = y0
        idx = 0
        used_slots = []
        prev_slot_bottom = None
        while words or preset_lines:
            if idx >= len(slots):
                if item.get("is_diagram_label"):
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
            fs = base_fs if self.fixed_font_size else min(base_fs, slot_h * 0.92)
            align = "left"
            if item.get("role") == "figure_caption":
                if not self.fixed_font_size:
                    fs = max(6.0, base_fs)
            elif not self.fixed_font_size:
                if item.get("is_title"):
                    fs = max(fs, min(max(11.5, base_fs * 1.22), slot_h * 1.05))
                elif item.get("role") == "section_heading":
                    fs = max(fs, min(max(10.5, base_fs * 1.12), slot_h * 1.02))
                elif item.get("is_diagram_label"):
                    fs = min(fs, 8.6)
                else:
                    fs = min(fs, max(8.0, slot_h * 0.78))
            if preserve_linebreaks and preset_lines:
                line = preset_lines.pop(0)
                if (
                    item.get("role") == "body"
                    and item.get("use_structured_source_lines")
                    and re.fullmatch(r"\s*\d+[.)]?\s*", line or "")
                ):
                    continue
                allow_overflow = bool(item.get("allow_line_overflow", False))
                line_w_now = self._measure_text_width(line, fs, fontname, fontfile)
                overflow_ok = bool(allow_overflow and line_w_now <= slot_w * 1.12)
                if (not overflow_ok) and (not self.fixed_font_size) and line_w_now > slot_w:
                    min_fs = max(5.0, min(7.0, fs))
                    while self._measure_text_width(line, fs, fontname, fontfile) > slot_w and fs > min_fs + 1e-6:
                        fs = max(min_fs, fs - 0.2)
                line_w_now = self._measure_text_width(line, fs, fontname, fontfile)
                overflow_ok = bool(allow_overflow and line_w_now <= slot_w * 1.12)
                if (not overflow_ok) and line_w_now > slot_w:
                    wds = line.split()
                    if len(wds) > 1:
                        line, tail = self._consume_words_for_width(wds, slot_w, fs, fontname, fontfile)
                        if tail:
                            preset_lines.insert(0, " ".join(tail))
                    elif len(line) > 1:
                        chunk = ""
                        i = 0
                        for ch in line:
                            cand = chunk + ch
                            if chunk and self._measure_text_width(cand, fs, fontname, fontfile) > slot_w:
                                break
                            chunk = cand
                            i += 1
                        if chunk:
                            rest = line[i:].strip()
                            line = chunk
                            if rest:
                                preset_lines.insert(0, rest)
                line = line.strip()
                if not line:
                    continue
            else:
                if self.fixed_font_size:
                    line, words = self._consume_words_for_width(words, slot_w, fs, fontname, fontfile)
                    if not line:
                        continue
                else:
                    remaining_text = " ".join(words).strip()
                    comp = self.text_composer.compose_text_in_box(
                        text=remaining_text,
                        box_w=slot_w,
                        box_h=slot_h,
                        base_font_pt=fs,
                        line_height_factor=1.22,
                        measure_fn=lambda t, fsz: self._measure_text_width(t, fsz, fontname, fontfile),
                        alignment=align,
                        lang=item.get("lang", "en"),
                        options=ComposeOptions(
                            enable_hyphenation=(item.get("source") != "native"),
                            max_font_shrink=1.0,
                            min_font_pt=7.0,
                            step_pt=0.25,
                        ),
                    )
                    if not comp.get("lines"):
                        continue
                    line = comp["lines"][0]
                    words = (comp.get("overflow") or "").split()
                    fs = comp.get("font_size", fs)
            line_w = self._measure_text_width(line, fs, fontname, fontfile)
            line_x = slot.x0
            baseline = slot.y0 + min(slot_h * 0.82, slot_h - 1.0)
            line_rgb = rgb
            if item.get("role") == "body":
                if re.match(r"^\s*(?:LA|LE|LES|THE)\s+[A-ZÀ-ÿ].*\([^)]{1,32}\)", line, flags=re.IGNORECASE):
                    accent = self._hex_to_rgb(item.get("accent_color", ""))
                    if accent is not None:
                        line_rgb = accent
            if render:
                self._safe_insert_text_dedup(page, (line_x, baseline), line, fs, fontname, line_rgb)
            if render and self.style_audit_enabled:
                exp_color = style.get("color", "#000000")
                app_color = "#%02x%02x%02x" % (
                    int(max(0.0, min(1.0, line_rgb[0])) * 255),
                    int(max(0.0, min(1.0, line_rgb[1])) * 255),
                    int(max(0.0, min(1.0, line_rgb[2])) * 255),
                )
                self._style_audit_records.append(
                    {
                        "page": int(page.number) + 1,
                        "role": item.get("role", "body"),
                        "style_lock_source": item.get("style_lock_source", "block"),
                        "expected_alignment": self._normalize_alignment(item.get("alignment", "left")),
                        "applied_alignment": align,
                        "expected_font": style.get("font"),
                        "applied_font": fontname,
                        "font_fallback": (fontname in {"helv", "times", "courier"} and str(style.get("font", "")).strip().lower() not in {"", "helv", "times", "courier", "arial", "helvetica"}),
                        "expected_size_pt": float(base_fs),
                        "applied_size_pt": float(fs),
                        "size_delta_pt": float(fs - base_fs),
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
        for page_idx, rects in debug_store.items():
            if page_idx < 0 or page_idx >= len(doc):
                continue
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=self.layout_debug_dpi, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            draw = ImageDraw.Draw(img)
            s = float(self.layout_debug_dpi) / 72.0
            for r in rects.get("blue", []):
                draw.rectangle([r.x0 * s, r.y0 * s, r.x1 * s, r.y1 * s], outline=(0, 90, 255), width=3)
            for r in rects.get("red", []):
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

        resolved = self.font_resolver.resolve(style)
        fontfile = resolved.get("fontfile")
        builtin = resolved.get("builtin")
        fontname = self._resolve_page_fontname(page, fontfile, builtin)
        fs = self._get_original_fontsize(style, max(1.0, item["bbox"].height), source)
        line_h = max(1.0, fs * 1.22)
        try:
            c = style.get("color", "#000000").lstrip("#")
            rgb = tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except Exception:
            rgb = (0, 0, 0)

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
        resolved = self.font_resolver.resolve(style)
        fontfile = resolved.get("fontfile")
        builtin = resolved.get("builtin")
        fontname = self._resolve_page_fontname(page, fontfile, builtin)
        base_fs = self._get_original_fontsize(style, max(1.0, item["bbox"].height), source)
        min_fs = max(self.flow_min_font_pt, base_fs * self.flow_min_font_scale)
        para_gap = max(2.0, base_fs * 0.55 if item["kind"] == "body" else base_fs * 0.35)
        try:
            c = style.get("color", "#000000").lstrip("#")
            rgb = tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except Exception:
            rgb = (0, 0, 0)

        remaining_text = item["text"]
        while frame_idx < len(frames):
            frame = frames[frame_idx]
            cur_y = max(cursor_y, frame.y0)
            desired_w = max(40.0, min(item["bbox"].width, frame.x1 - frame.x0))
            start_x = max(frame.x0, min(item["bbox"].x0, frame.x1 - desired_w))
            max_w = max(20.0, frame.x1 - start_x - 6.0)

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
                self._safe_insert_text_dedup(page, (start_x, baseline), line, fs, fontname, rgb)
                y += line_h

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
        max_w = max(20.0, frame.x1 - start_x - 6.0)
        lines = self._wrap_text_lines(remaining_text, max_w, fs, fontname, fontfile)
        max_lines = max(1, int((frame.y1 - y) / line_h))
        lines = lines[:max_lines]
        if lines:
            lines[-1] = lines[-1] + " ..."
        for line in lines:
            baseline = y + line_h * 0.82
            self._safe_insert_text_dedup(page, (start_x, baseline), line, fs, fontname, rgb)
            y += line_h
        return len(frames) - 1, min(frame.y1, y + para_gap)

    def _insert_hierarchical_span(self, page, span, source="ocr", placed_rects=None, forbidden_rects=None, allow_shift=True):
        text = span.get("texte", "")
        if not text: return
        style = span.get("style", {})
        bbox = span.get("bbox", [0,0,10,10])
        x0, y0, x1, y1 = [c * self.pixel_to_point for c in bbox]

        resolved = self.font_resolver.resolve(style)
        fontfile = resolved.get("fontfile")
        builtin = resolved.get("builtin")
        fontname = self._resolve_page_fontname(page, fontfile, builtin)

        fs = self._get_original_fontsize(style, y1 - y0, source)
        natural_w = self._measure_text_width(text, fs, fontname, fontfile)
        natural_h = max(fs * 1.15, (y1 - y0) if (y1 - y0) > 0 else fs)

        # Couleur
        try:
            c = style.get("color", "#000000").lstrip('#')
            rgb = tuple(int(c[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        except: rgb = (0, 0, 0)

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
        return max(1.0, (bbox_h_pt * 0.9))

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
        for ov in page_data.get("immutable_overlays", []):
            path = ov.get("path") if isinstance(ov, dict) else None
            bbox = ov.get("bbox") if isinstance(ov, dict) else None
            if not path or not os.path.exists(path):
                continue
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(v) * self.pixel_to_point for v in bbox]
            rect = fitz.Rect(x0, y0, x1, y1)
            if rect.get_area() <= 0:
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
