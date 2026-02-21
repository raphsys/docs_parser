import fitz
import os
import re
import unicodedata
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
        self.sequential_layout_mode = os.getenv("LAYOUT_SEQUENTIAL_FLOW", "1") == "1"
        self.layout_correction = os.getenv("LAYOUT_CORRECTION", "1") == "1"
        self.max_shift_steps = int(os.getenv("LAYOUT_MAX_SHIFT_STEPS", "24"))
        self.overlap_threshold = float(os.getenv("LAYOUT_OVERLAP_THRESHOLD", "0.25"))
        self.flow_layout_mode = os.getenv("LAYOUT_FLOW_MODE", "1") == "1"
        self.flow_zone_pad = float(os.getenv("LAYOUT_FLOW_ZONE_PAD", "8.0"))
        self.flow_min_font_scale = float(os.getenv("LAYOUT_FLOW_MIN_FONT_SCALE", "0.72"))
        self.flow_min_font_pt = float(os.getenv("LAYOUT_FLOW_MIN_FONT_PT", "5.5"))
        self.layout_debug_overlay = os.getenv("LAYOUT_DEBUG_OVERLAY", "1") == "1"
        self.layout_debug_dpi = int(os.getenv("LAYOUT_DEBUG_DPI", "150"))
        self.text_composer = TextComposer()
        if self.strict_fidelity:
            self.sequential_layout_mode = False
            self.layout_correction = False
            self.flow_layout_mode = False

    def reconstruct(self, structure, output_path):
        doc = fitz.open()
        debug_store = {}
        pages_list = structure.get("pages", [])
        if not pages_list and "blocks" in structure: pages_list = [structure]

        for i, page_data in enumerate(pages_list):
            dims = page_data.get("dimensions", {"width": 885, "height": 1110})
            w_pt, h_pt = dims["width"] * self.pixel_to_point, dims["height"] * self.pixel_to_point
            page = doc.new_page(width=w_pt, height=h_pt)
            
            # Fond Maître
            bg_path = page_data.get("background_path")
            if bg_path and os.path.exists(bg_path):
                page.insert_image(page.rect, filename=bg_path)

            forbidden_rects = self._collect_forbidden_rects(page_data)
            self._rendered_signatures = set()
            if self.sequential_layout_mode:
                self._reconstruct_sequential_flow(doc, page, page_data, debug_store)
            elif self.flow_layout_mode:
                self._reconstruct_with_flow(page, page_data, forbidden_rects)
            else:
                self._reconstruct_strict(page, page_data, forbidden_rects)

        if self.layout_debug_overlay:
            self._save_layout_debug_overlays(doc, debug_store, output_path)
        doc.save(output_path)
        doc.close()
        return output_path

    def _reconstruct_strict(self, page, page_data, forbidden_rects):
        placed_rects = []
        seen_spans = []
        for block in page_data.get("blocks", []):
            for line in block.get("lines", []):
                for phrase in line.get("phrases", []):
                    for span in phrase.get("spans", []):
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

    def _reconstruct_sequential_flow(self, doc, first_page, page_data, debug_store=None):
        items = self._extract_block_slot_items(page_data)
        if not items:
            return

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
        diagram_items = [it for it in items if it.get("is_diagram_label")]
        body_items = [it for it in items if it.get("role") not in {"header", "footer"} and not it.get("is_diagram_label")]

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

        cursor_y = body_top
        for item in body_items:
            remaining = item.get("text", "").strip()
            if not remaining:
                continue
            anchor_y = max(body_top, cursor_y, item["bbox"].y0)
            last_bottom = anchor_y
            while remaining:
                remaining, used_bottom, blue_rect, used_slots = self._render_block_slots(
                    page=page,
                    item=item,
                    anchor_y=anchor_y,
                    left=left,
                    right=right,
                    zone_top=body_top,
                    zone_bottom=body_bottom,
                    override_text=remaining,
                )
                self._append_debug_rects(debug_store, page, blue_rect, used_slots)
                last_bottom = max(last_bottom, used_bottom)
                if not remaining:
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
            cursor_y = min(body_bottom, last_bottom + max(4.0, item.get("slot_h_pt", 10.0) * 0.45))

    def _extract_block_slot_items(self, page_data):
        items = []
        dims = page_data.get("dimensions", {}) or {}
        page_w_pt = float(dims.get("width", 1000.0)) * self.pixel_to_point
        page_h_pt = float(dims.get("height", 1000.0)) * self.pixel_to_point
        for block in page_data.get("blocks", []):
            source = block.get("source", "ocr")
            text_parts = []
            style = {}
            slots = []
            for line in block.get("lines", []):
                for phrase in line.get("phrases", []):
                    t = re.sub(r"\s+", " ", (phrase.get("translated_text") or phrase.get("texte") or "").strip())
                    if t:
                        text_parts.append(t)
                    if not style and phrase.get("spans"):
                        style = phrase["spans"][0].get("style", {})
                    pb = phrase.get("bbox") or line.get("bbox")
                    if isinstance(pb, (list, tuple)) and len(pb) == 4:
                        r = fitz.Rect([float(v) * self.pixel_to_point for v in pb])
                        if r.get_area() > 0:
                            slots.append(r)
            block_preferred_text = re.sub(r"\s+", " ", (block.get("translated_text") or "").strip())
            text = block_preferred_text or re.sub(r"\s+", " ", " ".join(text_parts)).strip()
            text = self._clean_text_for_render(text)
            if not text:
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
            if is_diagram_label:
                is_title = False
            items.append(
                {
                    "text": text,
                    "bbox": bbox,
                    "slots": row_slots,
                    "slot_w_pt": max(10.0, bbox.width),
                    "slot_h_pt": slot_h,
                    "slot_gap_x_pt": gap_x,
                    "slot_gap_y_pt": gap_y,
                    "row_start_x_pt": row_start_x,
                    "style": style or {"font": "helv", "size": 12, "color": "#000000", "flags": {}},
                    "source": source,
                    "alignment": block.get("alignment", "left"),
                    "role": block.get("role", "body"),
                    "is_title": is_title,
                    "is_diagram_label": is_diagram_label,
                }
            )

            # Header often contains "page-number + title" merged in one OCR line.
            if block.get("role") == "header":
                m = re.match(r"^\s*(\d{1,3})\s+(.+)$", text)
                if m:
                    num_txt = m.group(1).strip()
                    title_txt = self._clean_text_for_render(m.group(2).strip())
                    if title_txt:
                        items[-1]["text"] = title_txt
                        items[-1]["alignment"] = "center"
                        items.append(
                            {
                                "text": num_txt,
                                "bbox": fitz.Rect(bbox.x0, bbox.y0, bbox.x0 + min(24.0, bbox.width * 0.15), bbox.y1),
                                "slots": [fitz.Rect(s) for s in row_slots[:1]],
                                "slot_w_pt": min(24.0, bbox.width * 0.15),
                                "slot_h_pt": slot_h,
                                "slot_gap_x_pt": gap_x,
                                "slot_gap_y_pt": gap_y,
                                "row_start_x_pt": row_start_x,
                                "style": style or {"font": "helv", "size": 12, "color": "#000000", "flags": {}},
                                "source": source,
                                "alignment": "right",
                                "role": "header",
                                "is_title": True,
                                "is_diagram_label": False,
                            }
                        )
        items.sort(key=lambda it: (it["bbox"].y0, it["bbox"].x0))
        return items

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

    def _render_block_slots(self, page, item, anchor_y, left, right, zone_top, zone_bottom, override_text=None):
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

        slots = [fitz.Rect(s.x0 + dx, s.y0 + dy, s.x1 + dx, s.y1 + dy) for s in item["slots"]]
        slots.sort(key=lambda r: (r.y0, r.x0))
        words = text.split()
        used_bottom = y0
        idx = 0
        used_slots = []
        prev_slot_bottom = None
        while words:
            if idx >= len(slots):
                if item.get("is_diagram_label"):
                    break
                prev = slots[-1] if slots else fitz.Rect(x0, y0, x0 + item["slot_w_pt"], y0 + item["slot_h_pt"])
                # New rows are appended downward; each line fills available width first.
                nx = x0 + max(0.0, item["row_start_x_pt"] - item["bbox"].x0)
                ny = prev.y1 + item["slot_gap_y_pt"]
                slots.append(fitz.Rect(nx, ny, nx + item["slot_w_pt"], ny + item["slot_h_pt"]))
            slot = slots[idx]
            idx += 1
            sx0 = max(left, min(slot.x0, block_right - 6.0))
            # Red slot extends to the right edge of the blue frame.
            sx1 = block_right
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
            fs = min(base_fs, slot_h * 0.92)
            if item.get("is_title"):
                fs = max(fs, min(max(11.5, base_fs * 1.22), slot_h * 1.05))
            elif item.get("role") == "section_heading":
                fs = max(fs, min(max(10.5, base_fs * 1.12), slot_h * 1.02))
            elif item.get("role") == "figure_caption":
                fs = min(fs, max(8.6, base_fs * 0.92))
            elif item.get("is_diagram_label"):
                fs = min(fs, 8.6)
            else:
                fs = min(fs, max(8.0, slot_h * 0.78))
            remaining_text = " ".join(words).strip()
            comp = self.text_composer.compose_text_in_box(
                text=remaining_text,
                box_w=slot_w,
                box_h=slot_h,
                base_font_pt=fs,
                line_height_factor=1.22,
                measure_fn=lambda t, fsz: self._measure_text_width(t, fsz, fontname, fontfile),
                alignment=item.get("alignment", "left"),
                lang="fr",
                options=ComposeOptions(enable_hyphenation=True, max_font_shrink=1.0, min_font_pt=7.0, step_pt=0.25),
            )
            if not comp.get("lines"):
                continue
            line = comp["lines"][0]
            words = (comp.get("overflow") or "").split()
            fs = comp.get("font_size", fs)
            line_w = self._measure_text_width(line, fs, fontname, fontfile)
            align = item.get("alignment", "left")
            if item.get("role") == "figure_caption" and align not in {"left", "center", "right", "justify"}:
                align = "left"
            line_x = self._compute_aligned_x(
                alignment=align,
                line_w=line_w,
                left=slot.x0,
                right=slot.x1,
                preferred_x=slot.x0,
                is_last_line=(len(words) == 0),
            )
            baseline = slot.y0 + min(slot_h * 0.82, slot_h - 1.0)
            self._safe_insert_text_dedup(page, (line_x, baseline), line, fs, fontname, rgb)
            used_bottom = max(used_bottom, slot.y1)
            used_slots.append(fitz.Rect(slot))
            prev_slot_bottom = slot.y1

        remaining = " ".join(words).strip()
        blue_rect = fitz.Rect(x0, y0, block_right, max(y0 + item["slot_h_pt"], used_bottom))
        return remaining, used_bottom, blue_rect, used_slots

    def _resolve_text_color(self, style, item):
        # WYSIWYG-first: keep extracted color, only enforce contrast floor for body text.
        try:
            c = style.get("color", "#000000").lstrip("#")
            if len(c) != 6:
                return (0, 0, 0)
            rgb = tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            if item.get("source") != "native" and item.get("role") == "body" and lum > 0.62:
                return (0.06, 0.06, 0.06)
            return rgb
        except Exception:
            return (0, 0, 0)

    def _safe_insert_text_dedup(self, page, point, text, fontsize, fontname, color):
        sig = (round(point[0], 2), round(point[1], 2), round(float(fontsize), 2), (text or "").strip())
        if sig in self._rendered_signatures:
            return
        self._rendered_signatures.add(sig)
        self._safe_insert_text(page, point, text, fontsize, fontname, color)

    def _append_debug_rects(self, debug_store, page, blue_rect, red_rects):
        if debug_store is None:
            return
        key = int(page.number)
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
            source = block.get("source", "ocr")
            block_align = block.get("alignment", "left")
            block_role = block.get("role", "body")
            block_indent_px = float(block.get("indent_px", 0.0) or 0.0)
            for line in block.get("lines", []):
                for phrase in line.get("phrases", []):
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
                    style = {}
                    if phrase.get("spans"):
                        style = phrase["spans"][0].get("style", {})
                    items.append(
                        {
                            "text": text,
                            "bbox": rect_pt,
                            "style": style or {"font": "helv", "size": 12, "color": "#000000", "flags": {}},
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
            source = block.get("source", "ocr")
            block_align = block.get("alignment", "left")
            block_role = block.get("role", "body")
            b = block.get("bbox", [0, 0, 10, 10])
            bbox = fitz.Rect([float(v) * self.pixel_to_point for v in b])
            text_parts = []
            spans = []
            style = {}
            for line in block.get("lines", []):
                for phrase in line.get("phrases", []):
                    ptxt = (phrase.get("translated_text") or phrase.get("texte") or "").strip()
                    if ptxt:
                        text_parts.append(ptxt)
                    for sp in phrase.get("spans", []):
                        spans.append(sp)
                        if not style:
                            style = sp.get("style", {})
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
                    "style": style or {"font": "helv", "size": 12, "color": "#000000", "flags": {}},
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
        baseline_ratio = 0.80
        flags = style.get("flags", {}) if isinstance(style, dict) else {}
        if flags.get("serif"):
            baseline_ratio = 0.78
        elif flags.get("italic"):
            baseline_ratio = 0.79
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
        return rects

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
