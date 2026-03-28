import hashlib
import os
import re
import tempfile


def _normalize_font_key(font_name: str) -> str:
    raw = (font_name or "").split("+", 1)[-1].strip()
    return re.sub(r"[^a-z0-9]+", "", raw.lower())


def _color_int_to_hex(value) -> str:
    try:
        if isinstance(value, (tuple, list)) and len(value) >= 3:
            r = max(0, min(255, int(round(float(value[0]) * 255 if value[0] <= 1.0 else value[0]))))
            g = max(0, min(255, int(round(float(value[1]) * 255 if value[1] <= 1.0 else value[1]))))
            b = max(0, min(255, int(round(float(value[2]) * 255 if value[2] <= 1.0 else value[2]))))
            return f"#{r:02x}{g:02x}{b:02x}"
        return "#{:06x}".format(int(value) & 0xFFFFFF)
    except Exception:
        return "#000000"


def _rect_to_px(rect, sx: float, sy: float):
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return [0, 0, 0, 0]
    return [
        int(float(rect[0]) * sx),
        int(float(rect[1]) * sy),
        int(float(rect[2]) * sx),
        int(float(rect[3]) * sy),
    ]


def _merge_bbox_list(boxes):
    valid = [b for b in boxes if isinstance(b, (list, tuple)) and len(b) == 4]
    if not valid:
        return [0, 0, 0, 0]
    return [
        min(int(float(b[0])) for b in valid),
        min(int(float(b[1])) for b in valid),
        max(int(float(b[2])) for b in valid),
        max(int(float(b[3])) for b in valid),
    ]


def _line_text_from_spans(spans):
    return "".join((sp.get("texte") or "") for sp in spans).strip()


class NativePDFExtractor:
    """
    Extract native (vector) text structure from a PDF page.
    Coordinates are returned in raster pixel space using sx/sy.

    The API remains compatible with the previous implementation while enriching
    blocks/lines/spans with additional native fidelity metadata:
    - raw text and char-level data when available
    - point-space metrics
    - source_kind / translatability hooks for later translation planning
    """

    def __init__(self, embedded_font_cache_dir=None):
        cache_dir = embedded_font_cache_dir or os.getenv("LAYOUT_EMBEDDED_FONT_CACHE_DIR", "").strip()
        if not cache_dir:
            cache_dir = os.path.join(tempfile.gettempdir(), "docs_parser_embedded_fonts")
        self.embedded_font_cache_dir = cache_dir
        self._embedded_font_file_cache = {}
        self._embedded_font_map_cache = {}

    def _doc_cache_key(self, pdf_page):
        doc = getattr(pdf_page, "parent", None)
        doc_name = ""
        try:
            doc_name = os.path.realpath(getattr(doc, "name", "") or "")
        except Exception:
            doc_name = ""
        return doc_name or f"mem:{id(doc)}"

    def _ensure_embedded_font_cache_dir(self):
        try:
            os.makedirs(self.embedded_font_cache_dir, exist_ok=True)
        except Exception:
            return None
        return self.embedded_font_cache_dir

    def _extract_embedded_font_file(self, pdf_page, xref, font_name):
        doc = getattr(pdf_page, "parent", None)
        if doc is None:
            return None
        cache_key = (self._doc_cache_key(pdf_page), int(xref))
        cached = self._embedded_font_file_cache.get(cache_key)
        if cached and os.path.isfile(cached):
            return cached

        cache_dir = self._ensure_embedded_font_cache_dir()
        if not cache_dir:
            return None

        try:
            extracted = doc.extract_font(int(xref))
        except Exception:
            return None
        if not extracted or len(extracted) < 4:
            return None

        ext = str(extracted[1] or "bin").strip().lower()
        font_bytes = extracted[3]
        if not font_bytes:
            return None
        ext = re.sub(r"[^a-z0-9]+", "", ext) or "bin"
        font_key = _normalize_font_key(font_name or extracted[0] or f"font{xref}") or f"font{xref}"
        digest = hashlib.sha1(font_bytes).hexdigest()[:12]
        out_path = os.path.join(cache_dir, f"{font_key}-{int(xref)}-{digest}.{ext}")
        if not os.path.isfile(out_path):
            with open(out_path, "wb") as fh:
                fh.write(font_bytes)
        self._embedded_font_file_cache[cache_key] = out_path
        return out_path

    def _get_page_embedded_font_map(self, pdf_page):
        cache_key = (self._doc_cache_key(pdf_page), int(getattr(pdf_page, "number", -1)))
        cached = self._embedded_font_map_cache.get(cache_key)
        if isinstance(cached, dict):
            return cached

        font_map = {}
        try:
            page_fonts = pdf_page.get_fonts(full=True) or []
        except Exception:
            page_fonts = []

        for font_entry in page_fonts:
            if not isinstance(font_entry, (list, tuple)) or len(font_entry) < 4:
                continue
            try:
                xref = int(font_entry[0])
            except Exception:
                continue
            font_name = str(font_entry[3] or "").strip()
            if not font_name:
                continue
            font_path = self._extract_embedded_font_file(pdf_page, xref, font_name)
            if not font_path:
                continue
            key_variants = {
                _normalize_font_key(font_name),
                _normalize_font_key(font_name.split("+", 1)[-1]),
            }
            for key in key_variants:
                if key and key not in font_map:
                    font_map[key] = font_path

        self._embedded_font_map_cache[cache_key] = font_map
        return font_map

    def _build_style(self, font_name, span_text, size, color, embedded_font_path=None):
        low_font = (font_name or "").lower()
        return {
            "font": font_name,
            "font_name_raw": font_name,
            "font_key_normalized": _normalize_font_key(font_name),
            "embedded_font_path": embedded_font_path,
            "size": float(size or 12.0),
            "color": _color_int_to_hex(color),
            "flags": {
                "bold": "bold" in low_font,
                "italic": any(x in low_font for x in ("italic", "itali", "oblique")),
                "serif": any(x in low_font for x in ("times", "serif", "tiro", "roman", "baskerville", "garamond")),
                "uppercase": bool(span_text) and span_text.isupper(),
            },
        }

    def _extract_raw_chars(self, raw_line, sx: float, sy: float):
        chars = []
        for raw_span in raw_line.get("spans", []) or []:
            span_bbox = raw_span.get("bbox")
            for ch in raw_span.get("chars", []) or []:
                c = ch.get("c", "")
                cb = ch.get("bbox", span_bbox)
                if not c or not isinstance(cb, (list, tuple)) or len(cb) != 4:
                    continue
                chars.append(
                    {
                        "c": c,
                        "bbox": _rect_to_px(cb, sx, sy),
                        "bbox_pt": [float(cb[0]), float(cb[1]), float(cb[2]), float(cb[3])],
                        "origin_pt": [float(v) for v in (ch.get("origin") or [])[:2]],
                    }
                )
        return chars

    def _chars_summary(self, chars):
        if not chars:
            return {"count": 0, "avg_char_width_pt": 0.0, "avg_char_height_pt": 0.0}
        widths = [max(0.0, float(ch["bbox_pt"][2]) - float(ch["bbox_pt"][0])) for ch in chars]
        heights = [max(0.0, float(ch["bbox_pt"][3]) - float(ch["bbox_pt"][1])) for ch in chars]
        return {
            "count": len(chars),
            "avg_char_width_pt": round(sum(widths) / max(1, len(widths)), 4),
            "avg_char_height_pt": round(sum(heights) / max(1, len(heights)), 4),
        }

    def _translatability_contract(self, text):
        txt = (text or "").strip()
        lower = txt.lower()
        if not txt:
            return {"translatable": False, "translation_strategy": "ignore", "coverage_required": "optional"}
        if re.fullmatch(r"[•▪◦·\-\*]", txt):
            return {"translatable": False, "translation_strategy": "exact_preserve", "coverage_required": "strict"}
        if re.fullmatch(r"\d{1,4}([.)]|)", txt):
            return {"translatable": False, "translation_strategy": "exact_preserve", "coverage_required": "strict"}
        if re.fullmatch(r"[ivxlcdm]+", lower):
            return {"translatable": False, "translation_strategy": "exact_preserve", "coverage_required": "strict"}
        if re.fullmatch(r"[A-Z]{2,8}", txt):
            return {"translatable": False, "translation_strategy": "exact_preserve", "coverage_required": "strict"}
        if len(txt.split()) <= 10:
            return {"translatable": True, "translation_strategy": "layout_constrained", "coverage_required": "strict"}
        return {"translatable": True, "translation_strategy": "semantic_reflow", "coverage_required": "strict"}

    def extract_page(self, pdf_page, sx: float = 1.0, sy: float = 1.0):
        dict_text = pdf_page.get_text("dict")
        raw_text = pdf_page.get_text("rawdict")
        embedded_font_map = self._get_page_embedded_font_map(pdf_page)
        native_blocks = []
        non_text_zones = []
        images = []
        drawings = []

        raw_blocks = raw_text.get("blocks", []) if isinstance(raw_text, dict) else []
        raw_line_index = {}
        for raw_block in raw_blocks:
            rb = raw_block.get("bbox")
            for raw_line in raw_block.get("lines", []) or []:
                lb = raw_line.get("bbox")
                if isinstance(rb, (list, tuple)) and len(rb) == 4 and isinstance(lb, (list, tuple)) and len(lb) == 4:
                    key = (
                        round(float(lb[0]), 2),
                        round(float(lb[1]), 2),
                        round(float(lb[2]), 2),
                        round(float(lb[3]), 2),
                    )
                    raw_line_index[key] = raw_line

        for b in dict_text.get("blocks", []):
            block_bbox = b.get("bbox")
            if not block_bbox:
                continue
            bb_px = _rect_to_px(block_bbox, sx, sy)

            if b.get("type") == 1:
                non_text_zones.append(bb_px)
                images.append({"bbox": bb_px, "bbox_pt": [float(v) for v in block_bbox], "source": "native_pdf_image"})
                continue

            lines = []
            for line_idx, l in enumerate(b.get("lines", []) or []):
                line_bbox_pt = l.get("bbox", block_bbox)
                line_bbox_px = _rect_to_px(line_bbox_pt, sx, sy)
                raw_key = (
                    round(float(line_bbox_pt[0]), 2),
                    round(float(line_bbox_pt[1]), 2),
                    round(float(line_bbox_pt[2]), 2),
                    round(float(line_bbox_pt[3]), 2),
                ) if isinstance(line_bbox_pt, (list, tuple)) and len(line_bbox_pt) == 4 else None
                raw_line = raw_line_index.get(raw_key) if raw_key is not None else None
                line_chars = self._extract_raw_chars(raw_line or {}, sx=sx, sy=sy)

                current_spans = []
                phrases = []
                for span_idx, s in enumerate(l.get("spans", []) or []):
                    txt = s.get("text", "")
                    if not txt:
                        continue
                    sb = s.get("bbox", block_bbox)
                    bbox_px = _rect_to_px(sb, sx, sy)
                    font_name = s.get("font", "")
                    span_payload = {
                        "texte": txt,
                        "bbox": bbox_px,
                        "bbox_pt": [float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3])],
                        "source": "native",
                        "source_kind": "native_span",
                        "span_index": span_idx,
                        "style": self._build_style(
                            font_name,
                            txt,
                            s.get("size", 12.0),
                            s.get("color", 0),
                            embedded_font_path=embedded_font_map.get(_normalize_font_key(font_name)),
                        ),
                    }
                    span_payload.update(self._translatability_contract(txt))
                    current_spans.append(span_payload)

                if not current_spans:
                    continue

                phrase_text = _line_text_from_spans(current_spans)
                phrase_payload = {
                    "texte": phrase_text,
                    "bbox": _merge_bbox_list([sp["bbox"] for sp in current_spans]),
                    "bbox_pt": _merge_bbox_list([sp["bbox_pt"] for sp in current_spans]),
                    "spans": current_spans,
                    "source": "native",
                    "source_kind": "native_phrase",
                }
                phrase_payload.update(self._translatability_contract(phrase_text))
                phrases.append(phrase_payload)

                line_payload = {
                    "bbox": line_bbox_px,
                    "bbox_pt": [float(line_bbox_pt[0]), float(line_bbox_pt[1]), float(line_bbox_pt[2]), float(line_bbox_pt[3])],
                    "phrases": phrases,
                    "source": "native",
                    "source_kind": "native_line",
                    "line_index_native": line_idx,
                    "line_text": phrase_text,
                    "raw_line_text": phrase_text,
                    "chars": line_chars,
                    "char_metrics": self._chars_summary(line_chars),
                }
                line_payload.update(self._translatability_contract(phrase_text))
                lines.append(line_payload)

            if lines:
                block_text = " ".join((ln.get("line_text") or "").strip() for ln in lines if (ln.get("line_text") or "").strip()).strip()
                block_payload = {
                    "id": f"n_{len(native_blocks)}",
                    "bbox": _merge_bbox_list([ln["bbox"] for ln in lines]),
                    "bbox_pt": _merge_bbox_list([ln["bbox_pt"] for ln in lines]),
                    "lines": lines,
                    "source": "native",
                    "source_kind": "native_block",
                    "raw_text": block_text,
                    "text": block_text,
                    "native_metrics": {
                        "line_count": len(lines),
                        "char_count": sum(int((ln.get("char_metrics") or {}).get("count", 0)) for ln in lines),
                    },
                }
                block_payload.update(self._translatability_contract(block_text))
                native_blocks.append(block_payload)

        try:
            for d in pdf_page.get_drawings():
                r = d.get("rect")
                if not r:
                    continue
                rb = [int(r.x0 * sx), int(r.y0 * sy), int(r.x1 * sx), int(r.y1 * sy)]
                if rb[2] > rb[0] and rb[3] > rb[1]:
                    drawings.append({"bbox": rb, "bbox_pt": [float(r.x0), float(r.y0), float(r.x1), float(r.y1)], "source": "native_pdf_drawing"})
        except Exception:
            drawings = []

        return {
            "blocks": native_blocks,
            "non_text_zones": non_text_zones,
            "images": images,
            "drawings": drawings,
        }
