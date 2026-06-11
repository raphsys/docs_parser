"""Autonomous high-fidelity document extraction.

This module is intentionally separate from the current extraction pipeline. It
reads a PDF source directly and produces a richer, explicit document model that
can later be bridged into the existing pipeline after validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
import re
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz


SCHEMA_VERSION = "perfect_extraction.v1"
POINTS_PER_INCH = 72.0
NULL = None


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _project_embedded_font_dir() -> Path:
    return _project_root() / "ai_models" / "fonts" / "embedded"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


def _round_float(value: Any, digits: int = 3) -> Optional[float]:
    value = _safe_float(value)
    if value is None:
        return None
    return round(value, digits)


def _bbox(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    try:
        rect = fitz.Rect(value)
        if rect.is_empty or rect.is_infinite:
            return None
        return [
            round(float(rect.x0), 3),
            round(float(rect.y0), 3),
            round(float(rect.x1), 3),
            round(float(rect.y1), 3),
        ]
    except Exception:
        return None


def _rect(value: Any) -> fitz.Rect:
    try:
        return fitz.Rect(value)
    except Exception:
        return fitz.Rect(0, 0, 0, 0)


def _rect_area(value: Any) -> float:
    rect = _rect(value)
    if rect.is_empty or rect.is_infinite:
        return 0.0
    return max(0.0, float(rect.width)) * max(0.0, float(rect.height))


def _union_bbox(items: Iterable[Any]) -> Optional[List[float]]:
    rect: Optional[fitz.Rect] = None
    for item in items:
        bbox = _bbox(item)
        if not bbox:
            continue
        item_rect = fitz.Rect(bbox)
        rect = item_rect if rect is None else rect | item_rect
    return _bbox(rect) if rect is not None else None


def _rgb_int_to_hex(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        intval = int(value)
        return f"#{(intval >> 16) & 255:02x}{(intval >> 8) & 255:02x}{intval & 255:02x}"
    except Exception:
        return None


def _color_tuple_to_hex(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _rgb_int_to_hex(value)
    try:
        vals = list(value)
        if not vals:
            return None
        if max(vals) <= 1.0:
            vals = [int(round(max(0.0, min(1.0, float(v))) * 255)) for v in vals[:3]]
        else:
            vals = [int(round(max(0.0, min(255.0, float(v))))) for v in vals[:3]]
        while len(vals) < 3:
            vals.append(0)
        return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"
    except Exception:
        return None


def _normalize_font_key(font_name: str) -> str:
    text = str(font_name or "").strip()
    text = re.sub(r"^[A-Z]{6}\+", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", "", text).lower()
    return text


def _font_flags_from_name(font_name: str, flags: Any = 0) -> Dict[str, Any]:
    low = str(font_name or "").lower()
    try:
        int_flags = int(flags or 0)
    except Exception:
        int_flags = 0
    return {
        "raw_flags": int_flags,
        "is_serif": bool(int_flags & 4) or any(v in low for v in ("serif", "times", "baskerville", "garamond")),
        "is_monospace": bool(int_flags & 8) or any(v in low for v in ("mono", "courier", "consolas", "code")),
        "is_bold": bool(int_flags & 16) or any(v in low for v in ("bold", "demi", "black", "heavy")),
        "is_italic": bool(int_flags & 2) or any(v in low for v in ("italic", "oblique", "slant")),
        "is_symbolic": bool(int_flags & 2_000_000) or any(v in low for v in ("symbol", "math", "cmmi", "cmsy", "stix")),
        "is_superscript": bool(int_flags & 1),
        "is_underline": None,
        "is_strikeout": None,
        "is_small_caps": None,
    }


def _case_profile(text: str) -> Dict[str, Any]:
    text = str(text or "")
    letters = [c for c in text if c.isalpha()]
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'-]*", text)
    acronyms = [w for w in words if len(w) >= 2 and w.upper() == w and any(c.isalpha() for c in w)]
    if not letters:
        case_type = "numeric" if re.search(r"\d", text) else "symbolic"
    elif all(c.upper() == c for c in letters):
        case_type = "all_caps"
    elif all(c.lower() == c for c in letters):
        case_type = "lower"
    elif words and all(w[:1].upper() == w[:1] for w in words):
        case_type = "title_case"
    elif text[:1].upper() == text[:1]:
        case_type = "sentence_case"
    else:
        case_type = "mixed"
    return {
        "raw": text,
        "case_type": case_type,
        "must_preserve_case": bool(case_type in {"all_caps", "mixed"} or acronyms),
        "contains_acronyms": bool(acronyms),
        "acronyms": acronyms,
        "letter_count": len(letters),
        "word_count": len(words),
        "uppercase_ratio": round(sum(1 for c in letters if c.upper() == c) / max(1, len(letters)), 3),
    }


def _hash_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _pdf_date_to_iso(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.match(r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?", text)
    if not match:
        return text
    parts = match.groups()
    year = int(parts[0])
    month = int(parts[1] or 1)
    day = int(parts[2] or 1)
    hour = int(parts[3] or 0)
    minute = int(parts[4] or 0)
    second = int(parts[5] or 0)
    try:
        return datetime(year, month, day, hour, minute, second).isoformat()
    except Exception:
        return text


def _page_format(width: float, height: float) -> str:
    candidates = {
        "A4": (595.276, 841.89),
        "A5": (419.528, 595.276),
        "Letter": (612.0, 792.0),
        "Legal": (612.0, 1008.0),
    }
    w, h = sorted([float(width), float(height)])
    for name, dims in candidates.items():
        cw, ch = sorted(dims)
        if abs(w - cw) <= 8 and abs(h - ch) <= 8:
            return name
    return "custom"


def _orientation(width: float, height: float) -> str:
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return "square"


def _style_from_span(span: Dict[str, Any], embedded_font_path: Optional[str] = None) -> Dict[str, Any]:
    font = str(span.get("font") or "")
    flags = _font_flags_from_name(font, span.get("flags"))
    return {
        "font_family": font or None,
        "font_postscript_name": font or None,
        "font_key_normalized": _normalize_font_key(font) or None,
        "embedded_font_path": embedded_font_path,
        "size_pt": _round_float(span.get("size")),
        "color_rgb": _rgb_int_to_hex(span.get("color")),
        "fill_color": _rgb_int_to_hex(span.get("color")),
        "stroke_color": None,
        "opacity": None,
        "bold": flags["is_bold"],
        "italic": flags["is_italic"],
        "serif": flags["is_serif"],
        "monospace": flags["is_monospace"],
        "symbolic": flags["is_symbolic"],
        "superscript": flags["is_superscript"],
        "underline": flags["is_underline"],
        "strikeout": flags["is_strikeout"],
        "small_caps": flags["is_small_caps"],
        "flags": flags,
        "writing_mode": None,
        "text_rendering_mode": None,
        "baseline": None,
        "line_height": None,
        "letter_spacing": None,
        "word_spacing": None,
        "horizontal_scale": None,
    }


def _empty_doc_profile() -> Dict[str, Any]:
    return {
        "probable_type": None,
        "dominant_orientation": None,
        "dominant_style": None,
        "dominant_column_count": None,
        "has_tables": None,
        "has_figures": None,
        "has_charts": None,
        "has_formulas": None,
        "has_code": None,
        "has_notes": None,
        "has_references": None,
        "has_pagination": None,
        "has_headers": None,
        "has_footers": None,
        "has_watermark": None,
        "has_logo": None,
        "has_stamp": None,
        "has_signature": None,
        "has_handwriting": None,
        "visual_complexity": None,
        "linguistic_complexity": None,
        "document_quality": None,
        "ocr_quality": None,
        "reconstructibility_score": None,
    }


@dataclass
class ExtractionConfig:
    target_dpi: int = 150
    enable_ocr: bool = True
    enable_special_regions: bool = True
    extract_embedded_fonts: bool = True
    font_cache_dir: Optional[Path] = None
    max_chars_per_page: int = 200_000


class PerfectDocumentExtractor:
    """Extract a complete, explicit document model from source files."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self.font_cache_dir = Path(self.config.font_cache_dir or _project_embedded_font_dir())
        self._font_cache: Dict[Tuple[str, int], Optional[str]] = {}
        self._ocr_engine = None

    def extract(self, source: str | os.PathLike[str], pages: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(str(source_path))
        mime, _ = mimetypes.guess_type(str(source_path))
        if source_path.suffix.lower() != ".pdf":
            raise ValueError("perfect_document_extractor currently supports PDF sources.")
        with fitz.open(str(source_path)) as doc:
            page_indexes = self._normalize_pages(pages, doc.page_count)
            model = self._document_shell(source_path, doc, mime)
            model["pages"] = [self._extract_page(doc, page_index) for page_index in page_indexes]
            self._finalize_document_profile(model)
            model["quality_report"] = self._document_quality_report(model)
            return model

    def _normalize_pages(self, pages: Optional[Sequence[int]], page_count: int) -> List[int]:
        if not pages:
            return list(range(page_count))
        out = []
        for page in pages:
            index = int(page) - 1
            if index < 0 or index >= page_count:
                raise ValueError(f"Page {page} out of range 1..{page_count}")
            out.append(index)
        return sorted(set(out))

    def _document_shell(self, source_path: Path, doc: fitz.Document, mime: Optional[str]) -> Dict[str, Any]:
        stat = source_path.stat()
        meta = doc.metadata or {}
        return {
            "schema_version": SCHEMA_VERSION,
            "complete_field_policy": "present_null_when_unknown",
            "created_at": _now_iso(),
            "extractor": {
                "name": "PerfectDocumentExtractor",
                "version": "0.1.0",
                "philosophy": "native_pdf_first_ocr_when_needed_ai_assisted_special_regions",
                "target_dpi": self.config.target_dpi,
                "uses_current_pipeline_extraction_as_input": False,
            },
            "source": {
                "path": str(source_path),
                "file_name": source_path.name,
                "extension": source_path.suffix.lower() or None,
                "mime_type": mime,
                "detected_format": "pdf",
                "size_bytes": stat.st_size,
                "sha256": _hash_file(source_path),
                "created_at_fs": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at_fs": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            },
            "pdf": {
                "page_count": doc.page_count,
                "format_version": getattr(doc, "pdf_version", lambda: None)() if callable(getattr(doc, "pdf_version", None)) else None,
                "is_encrypted": bool(doc.is_encrypted),
                "needs_pass": bool(getattr(doc, "needs_pass", False)),
                "permissions": getattr(doc, "permissions", None),
                "has_xref_streams": None,
                "has_layers": None,
                "has_form": bool(getattr(doc, "is_form_pdf", False)),
                "has_annotations": None,
                "has_attachments": None,
                "has_tagged_structure": None,
                "has_accessibility": None,
                "has_preexisting_ocr": None,
                "has_invisible_text": None,
                "has_vectorized_text": None,
                "has_scanned_pages": None,
                "has_embedded_fonts": None,
                "has_substituted_fonts": None,
                "has_high_resolution_images": None,
                "has_transparency": None,
                "has_complex_vectors": None,
            },
            "metadata": {
                "title": meta.get("title") or None,
                "author": meta.get("author") or None,
                "subject": meta.get("subject") or None,
                "keywords": meta.get("keywords") or None,
                "declared_language": None,
                "detected_languages": [],
                "creation_date": _pdf_date_to_iso(meta.get("creationDate")),
                "modification_date": _pdf_date_to_iso(meta.get("modDate")),
                "producer": meta.get("producer") or None,
                "creator": meta.get("creator") or None,
                "application_source": meta.get("creator") or None,
                "application_source_version": None,
                "unique_identifier": None,
                "copyright": None,
                "classification": None,
                "status": None,
                "template": None,
                "revision_count": None,
                "history": None,
                "internal_comments": None,
            },
            "resources": {
                "fonts": [],
                "images": [],
                "color_spaces": [],
                "xobjects": [],
                "patterns": [],
            },
            "global_styles": [],
            "document_profile": _empty_doc_profile(),
            "pages": [],
            "quality_report": None,
        }

    def _extract_page(self, doc: fitz.Document, page_index: int) -> Dict[str, Any]:
        page = doc.load_page(page_index)
        width = float(page.rect.width)
        height = float(page.rect.height)
        zoom = self.config.target_dpi / POINTS_PER_INCH
        sx = sy = zoom
        page_image_info, page_image_for_ai = self._render_page_image(page, zoom)
        fonts = self._extract_page_fonts(doc, page, page_index)
        font_map = {f["font_name"]: f.get("embedded_font_path") for f in fonts if f.get("font_name")}
        native_text = self._extract_native_text(page, font_map)
        images = self._extract_images(page, page_index)
        drawings = self._extract_drawings(page)
        ocr_text = self._extract_ocr_if_needed(page, page_image_info, native_text)
        blocks = native_text["blocks"] if native_text["blocks"] else ocr_text["blocks"]
        regions = self._infer_regions(width, height, blocks, images, drawings)
        tables = self._detect_tables(blocks, drawings)
        special_regions, special_report = self._detect_special_regions(
            blocks=blocks,
            page=page,
            page_image=page_image_for_ai,
            sx=sx,
            sy=sy,
            width=width * sx,
            height=height * sy,
        )
        render_objects = self._build_render_objects(blocks, images, drawings, special_regions)
        page_model = {
            "page_index": page_index,
            "page_number_physical": page_index + 1,
            "page_number_logical": None,
            "page_label": None,
            "geometry": {
                "width": round(width, 3),
                "height": round(height, 3),
                "unit_source": "pt",
                "unit_normalized": "pt",
                "orientation": _orientation(width, height),
                "rotation": int(page.rotation or 0),
                "format_probable": _page_format(width, height),
                "mediabox": _bbox(page.mediabox),
                "cropbox": _bbox(page.cropbox),
                "bleedbox": None,
                "trimbox": None,
                "artbox": None,
                "printable_area": None,
                "visible_area": _bbox(page.rect),
                "content_area": _union_bbox([b.get("bbox") for b in blocks] + [i.get("bbox") for i in images] + [d.get("bbox") for d in drawings]),
                "origin": "top_left",
                "coordinate_system": "pymupdf_page_points",
                "global_transform_matrix": None,
                "target_dpi": self.config.target_dpi,
                "native_dpi": None,
                "aspect_ratio": round(width / max(1.0, height), 4),
                "visible_margins": self._estimate_margins(width, height, blocks, images, drawings),
                "probable_margins": None,
                "probable_grid": None,
            },
            "background": self._extract_background_profile(page_image_info, drawings, images),
            "margins": {
                "left": None,
                "right": None,
                "top": None,
                "bottom": None,
                "inner": None,
                "outer": None,
                "binding": None,
                "header": None,
                "footer": None,
                "even_odd_variation": None,
                "first_page_variation": None,
                "content_alignment": None,
            },
            "columns": self._infer_columns(width, blocks),
            "grid": {
                "vertical_grid": None,
                "horizontal_grid": None,
                "baseline_grid": self._estimate_baseline_grid(blocks),
                "dominant_line_step": self._estimate_line_step(blocks),
                "paragraphs_aligned_on_grid": None,
            },
            "regions": regions,
            "text": {
                "native": native_text,
                "ocr": ocr_text,
                "chosen_source": "native_pdf" if native_text["blocks"] else ("ocr" if ocr_text["blocks"] else None),
                "blocks": blocks,
                "paragraphs": self._infer_paragraphs(blocks),
                "headers": [],
                "footers": [],
                "page_numbers": [],
                "languages": [],
                "translatable_segments": self._infer_translatable_segments(blocks, special_regions),
                "non_translatable_segments": self._infer_non_translatable_segments(blocks, special_regions),
            },
            "images": images,
            "drawings": drawings,
            "tables": tables,
            "formulas_and_specials": special_regions,
            "annotations": self._extract_annotations(page),
            "render_objects": render_objects,
            "reading_order": self._build_reading_order(blocks, images, drawings, special_regions),
            "render_order": self._build_render_order(render_objects),
            "relationships": self._infer_relationships(blocks, images, tables, special_regions),
            "reconstruction_constraints": self._page_reconstruction_constraints(width, height, blocks, images, drawings, special_regions),
            "quality_report": None,
        }
        page_model["quality_report"] = self._page_quality_report(page_model, special_report)
        return page_model

    def _render_page_image(self, page: fitz.Page, zoom: float) -> Tuple[Dict[str, Any], Any]:
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            return (
                {
                    "width_px": pix.width,
                    "height_px": pix.height,
                    "colorspace": str(pix.colorspace),
                    "alpha": bool(pix.alpha),
                    "sample_count": len(pix.samples or b""),
                    "rendered": True,
                },
                self._pixmap_to_pil(pix),
            )
        except Exception as exc:
            return (
                {
                    "width_px": None,
                    "height_px": None,
                    "colorspace": None,
                    "alpha": None,
                    "sample_count": None,
                    "rendered": False,
                    "error": str(exc),
                },
                None,
            )

    def _pixmap_to_pil(self, pix: fitz.Pixmap) -> Any:
        try:
            from PIL import Image

            mode = "RGBA" if pix.alpha else "RGB"
            return Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        except Exception:
            return None

    def _extract_page_fonts(self, doc: fitz.Document, page: fitz.Page, page_index: int) -> List[Dict[str, Any]]:
        out = []
        try:
            fonts = page.get_fonts(full=True)
        except Exception:
            fonts = []
        for idx, item in enumerate(fonts):
            xref = item[0] if len(item) > 0 else None
            ext = item[1] if len(item) > 1 else None
            font_type = item[2] if len(item) > 2 else None
            base_font = item[3] if len(item) > 3 else None
            resource_name = item[4] if len(item) > 4 else None
            encoding = item[5] if len(item) > 5 else None
            embedded_path = self._extract_font_file(doc, page_index, xref, base_font) if self.config.extract_embedded_fonts else None
            out.append(
                {
                    "id": f"p{page_index + 1}_font_{idx}",
                    "xref": xref,
                    "extension": ext,
                    "font_type": font_type,
                    "font_name": base_font,
                    "font_key_normalized": _normalize_font_key(base_font or ""),
                    "resource_name": resource_name,
                    "encoding": encoding,
                    "embedded": bool(embedded_path),
                    "embedded_font_path": embedded_path,
                    "subset": bool(re.match(r"^[A-Z]{6}\+", str(base_font or ""))),
                    "fallback_used": None,
                }
            )
        return out

    def _extract_font_file(self, doc: fitz.Document, page_index: int, xref: Any, font_name: Any) -> Optional[str]:
        try:
            xref_int = int(xref)
        except Exception:
            return None
        if xref_int <= 0:
            return None
        cache_key = (str(doc.name), xref_int)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]
        try:
            extracted = doc.extract_font(xref_int)
        except Exception:
            self._font_cache[cache_key] = None
            return None
        if not extracted:
            self._font_cache[cache_key] = None
            return None
        ext = str(extracted[1] or "font").lower().strip(".")
        payload = extracted[3]
        if not payload:
            self._font_cache[cache_key] = None
            return None
        try:
            self.font_cache_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(font_name or f"font_{xref_int}"))
            out_path = self.font_cache_dir / f"{Path(doc.name).stem}_p{page_index + 1}_xref{xref_int}_{safe_name}.{ext}"
            if not out_path.exists():
                out_path.write_bytes(payload)
            self._font_cache[cache_key] = str(out_path)
            return str(out_path)
        except Exception:
            fallback = Path(tempfile.gettempdir()) / "docs_parser_embedded_fonts"
            try:
                fallback.mkdir(parents=True, exist_ok=True)
                out_path = fallback / f"xref{xref_int}.{ext}"
                if not out_path.exists():
                    out_path.write_bytes(payload)
                self._font_cache[cache_key] = str(out_path)
                return str(out_path)
            except Exception:
                self._font_cache[cache_key] = None
                return None

    def _resolve_embedded_font_path(self, font_map: Dict[str, Optional[str]], font_name: str) -> Optional[str]:
        if not font_name:
            return None
        if font_map.get(font_name):
            return font_map[font_name]
        key = _normalize_font_key(font_name)
        for mapped_name, path in font_map.items():
            if path and (_normalize_font_key(mapped_name) == key or key in _normalize_font_key(mapped_name) or _normalize_font_key(mapped_name) in key):
                return path
        return None

    def _extract_native_text(self, page: fitz.Page, font_map: Dict[str, Optional[str]]) -> Dict[str, Any]:
        text_dict = self._safe_page_text(page, "rawdict")
        words_raw = self._safe_page_words(page)
        blocks = []
        chars_total = 0
        for block_idx, raw_block in enumerate(text_dict.get("blocks", []) or []):
            if raw_block.get("type", 0) != 0:
                continue
            lines = []
            block_text_parts = []
            for line_idx, raw_line in enumerate(raw_block.get("lines", []) or []):
                spans = []
                line_text_parts = []
                for span_idx, raw_span in enumerate(raw_line.get("spans", []) or []):
                    font_name = str(raw_span.get("font") or "")
                    embedded_font_path = self._resolve_embedded_font_path(font_map, font_name)
                    chars = []
                    span_text_parts = []
                    for char_idx, raw_char in enumerate(raw_span.get("chars", []) or []):
                        char_text = str(raw_char.get("c") or "")
                        if chars_total >= self.config.max_chars_per_page:
                            continue
                        chars_total += 1
                        span_text_parts.append(char_text)
                        chars.append(
                            {
                                "id": f"ch_{block_idx}_{line_idx}_{span_idx}_{char_idx}",
                                "text": char_text,
                                "bbox": _bbox(raw_char.get("bbox") or raw_span.get("bbox")),
                                "origin": list(raw_char.get("origin")) if raw_char.get("origin") else None,
                                "style": _style_from_span(raw_span, embedded_font_path),
                                "unicode_category": None,
                                "is_whitespace": char_text.isspace(),
                                "is_math_symbol": self._is_math_symbol(char_text, font_name),
                                "confidence": 1.0,
                            }
                        )
                    span_text = "".join(span_text_parts)
                    line_text_parts.append(span_text)
                    spans.append(
                        {
                            "id": f"span_{block_idx}_{line_idx}_{span_idx}",
                            "text": span_text,
                            "bbox": _bbox(raw_span.get("bbox")),
                            "origin": list(raw_span.get("origin")) if raw_span.get("origin") else None,
                            "style": _style_from_span(raw_span, embedded_font_path),
                            "case_profile": _case_profile(span_text),
                            "language": None,
                            "script": None,
                            "direction": None,
                            "rotation": None,
                            "chars": chars,
                            "confidence": 1.0,
                        }
                    )
                line_text = "".join(line_text_parts)
                block_text_parts.append(line_text)
                lines.append(
                    {
                        "id": f"line_{block_idx}_{line_idx}",
                        "text": line_text,
                        "bbox": _bbox(raw_line.get("bbox")),
                        "wmode": raw_line.get("wmode"),
                        "dir": list(raw_line.get("dir")) if raw_line.get("dir") else None,
                        "spans": spans,
                        "words": self._words_in_bbox(words_raw, raw_line.get("bbox")),
                        "case_profile": _case_profile(line_text),
                        "baseline": None,
                        "alignment": None,
                        "line_height": self._line_height_from_spans(spans),
                        "confidence": 1.0,
                    }
                )
            block_text = "\n".join(t for t in block_text_parts if t is not None)
            block = {
                "id": f"native_block_{block_idx}",
                "source": "native_pdf",
                "role": self._classify_text_role(block_text, raw_block.get("bbox"), lines),
                "object_type": "text_block",
                "text": block_text,
                "bbox": _bbox(raw_block.get("bbox")),
                "lines": lines,
                "words": self._words_in_bbox(words_raw, raw_block.get("bbox")),
                "style_summary": self._summarize_styles(lines),
                "case_profile": _case_profile(block_text),
                "language": None,
                "script": None,
                "reading_order_index": None,
                "render_order_index": block_idx,
                "translatable": None,
                "non_translatable_reason": None,
                "semantic": {
                    "heading_level": None,
                    "list_type": None,
                    "list_marker": None,
                    "caption_for": None,
                    "table_membership": None,
                    "formula_membership": None,
                },
                "layout": {
                    "column_index": None,
                    "paragraph_index": None,
                    "alignment": self._infer_block_alignment(raw_block.get("bbox"), page.rect.width),
                    "indent_left": None,
                    "indent_right": None,
                    "space_before": None,
                    "space_after": None,
                    "keep_with_next": None,
                },
                "reconstruction_constraints": {
                    "preserve_bbox": True,
                    "preserve_font_size": True,
                    "preserve_case": _case_profile(block_text)["must_preserve_case"],
                    "preserve_color": True,
                    "preserve_style_flags": True,
                    "allow_reflow": True,
                    "allow_horizontal_expansion": True,
                    "allow_vertical_expansion": True,
                    "allow_overlap": False,
                    "must_render": True,
                },
                "confidence": 1.0,
            }
            blocks.append(block)
        self._assign_reading_order(blocks)
        return {
            "available": bool(blocks),
            "source": "native_pdf",
            "raw_block_count": len(text_dict.get("blocks", []) or []),
            "block_count": len(blocks),
            "line_count": sum(len(b.get("lines") or []) for b in blocks),
            "word_count": len(words_raw),
            "char_count": chars_total,
            "blocks": blocks,
            "words_flat": self._words_flat(words_raw),
            "confidence": 1.0 if blocks else None,
        }

    def _safe_page_text(self, page: fitz.Page, mode: str) -> Dict[str, Any]:
        try:
            return page.get_text(mode, sort=False) or {}
        except Exception:
            return {}

    def _safe_page_words(self, page: fitz.Page) -> List[Tuple[Any, ...]]:
        try:
            return page.get_text("words", sort=False) or []
        except Exception:
            return []

    def _words_flat(self, words_raw: List[Tuple[Any, ...]]) -> List[Dict[str, Any]]:
        out = []
        for idx, word in enumerate(words_raw):
            out.append(
                {
                    "id": f"word_{idx}",
                    "text": word[4] if len(word) > 4 else None,
                    "bbox": _bbox(word[:4]),
                    "block_index": word[5] if len(word) > 5 else None,
                    "line_index": word[6] if len(word) > 6 else None,
                    "word_index": word[7] if len(word) > 7 else None,
                    "case_profile": _case_profile(word[4] if len(word) > 4 else ""),
                    "confidence": 1.0,
                }
            )
        return out

    def _words_in_bbox(self, words_raw: List[Tuple[Any, ...]], bbox: Any) -> List[Dict[str, Any]]:
        rect = _rect(bbox)
        if rect.get_area() <= 0:
            return []
        out = []
        for idx, word in enumerate(words_raw):
            wrect = _rect(word[:4])
            if wrect.get_area() <= 0:
                continue
            if (rect & wrect).get_area() / max(1.0, wrect.get_area()) >= 0.45:
                out.append(
                    {
                        "id": f"word_{idx}",
                        "text": word[4] if len(word) > 4 else None,
                        "bbox": _bbox(wrect),
                        "case_profile": _case_profile(word[4] if len(word) > 4 else ""),
                        "confidence": 1.0,
                    }
                )
        return out

    def _line_height_from_spans(self, spans: List[Dict[str, Any]]) -> Optional[float]:
        heights = [_rect_area(span.get("bbox")) / max(1.0, _rect(span.get("bbox")).width) for span in spans if span.get("bbox")]
        if not heights:
            return None
        return round(statistics.median(heights), 3)

    def _summarize_styles(self, lines: List[Dict[str, Any]]) -> Dict[str, Any]:
        spans = [span for line in lines for span in (line.get("spans") or [])]
        sizes = [span.get("style", {}).get("size_pt") for span in spans if span.get("style", {}).get("size_pt")]
        fonts = [span.get("style", {}).get("font_family") for span in spans if span.get("style", {}).get("font_family")]
        colors = [span.get("style", {}).get("color_rgb") for span in spans if span.get("style", {}).get("color_rgb")]
        return {
            "dominant_font_family": self._mode(fonts),
            "dominant_font_key_normalized": _normalize_font_key(self._mode(fonts) or "") or None,
            "dominant_size_pt": round(statistics.median(sizes), 3) if sizes else None,
            "dominant_color_rgb": self._mode(colors),
            "has_bold": any(span.get("style", {}).get("bold") for span in spans),
            "has_italic": any(span.get("style", {}).get("italic") for span in spans),
            "has_serif": any(span.get("style", {}).get("serif") for span in spans),
            "has_monospace": any(span.get("style", {}).get("monospace") for span in spans),
            "has_symbolic": any(span.get("style", {}).get("symbolic") for span in spans),
            "span_style_count": len(spans),
        }

    def _mode(self, values: Sequence[Any]) -> Any:
        clean = [v for v in values if v is not None]
        if not clean:
            return None
        try:
            return statistics.mode(clean)
        except Exception:
            return clean[0]

    def _is_math_symbol(self, text: str, font_name: str = "") -> bool:
        if not text:
            return False
        low_font = str(font_name or "").lower()
        if any(v in low_font for v in ("math", "symbol", "cmmi", "cmsy", "stix", "mtex")):
            return True
        return bool(re.search(r"[∂∑∏∫√∞≈≠≤≥±×÷−∆Ω∗·δµμ=<>*/^_{}()[\]\\|¬→←↔⇒⇔∈∉⊂⊃⊆⊇∧∨∩∪]|[α-ωΑ-Ω]", text))

    def _classify_text_role(self, text: str, bbox: Any, lines: List[Dict[str, Any]]) -> str:
        stripped = re.sub(r"\s+", " ", text or "").strip()
        if not stripped:
            return "empty"
        if self._looks_like_code(stripped, lines):
            return "code"
        if self._looks_like_formula(stripped, lines):
            return "formula"
        if len(stripped) < 12 and stripped.upper() == stripped and re.search(r"[A-Z]", stripped):
            return "heading"
        if re.match(r"^\s*(\d+(\.\d+)*|[A-Z]\.|[-*•])\s+", stripped):
            return "list_item"
        return "paragraph"

    def _looks_like_formula(self, text: str, lines: List[Dict[str, Any]]) -> bool:
        if not text:
            return False
        if any(span.get("style", {}).get("symbolic") for line in lines for span in line.get("spans", [])):
            return True
        symbols = len(re.findall(r"[=∂∑∏∫√∞≈≠≤≥±×÷−∆Ω∗·δµμ^_{}<>]", text))
        words = len(re.findall(r"[A-Za-zÀ-ÿ]{3,}", text))
        return symbols >= 2 and words <= max(3, symbols)

    def _looks_like_code(self, text: str, lines: List[Dict[str, Any]]) -> bool:
        compact_lines = [line for line in re.split(r"\n+", text or "") if line.strip()]
        if not compact_lines:
            return False
        code_punct = len(re.findall(r"[{}();=<>]|==|!=|<=|>=|//|::|->|=>|\\[A-Za-z]+", text or ""))
        strong_keywords = self._code_syntax_signal_count(text or "")
        function_syntax = bool(re.search(r"\bfunction\s+[$A-Za-z_][$A-Za-z0-9_]*\s*\(|\bfunction\s*\(", text or ""))
        path_or_command = bool(re.search(r"(^|\s)(/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+|[A-Za-z]:\\|\.{0,2}/[A-Za-z0-9_.-]+|npm\s+|pip\s+|python\s+)", text or ""))
        monospace = any(span.get("style", {}).get("monospace") for line in lines for span in line.get("spans", []))
        punctuation_density = code_punct / max(1, len(re.sub(r"\s+", "", text or "")))
        if monospace and (code_punct >= 2 or strong_keywords >= 1 or path_or_command):
            return True
        if strong_keywords >= 2 and (code_punct >= 1 or len(compact_lines) >= 2):
            return True
        if strong_keywords >= 1 and code_punct >= 2:
            return True
        if function_syntax and code_punct >= 2:
            return True
        if path_or_command and (code_punct >= 1 or monospace):
            return True
        return punctuation_density >= 0.12 and code_punct >= 4 and strong_keywords >= 1

    def _infer_block_alignment(self, bbox: Any, page_width: float) -> Optional[str]:
        rect = _rect(bbox)
        if rect.get_area() <= 0:
            return None
        center = (rect.x0 + rect.x1) / 2.0
        if abs(center - page_width / 2.0) <= page_width * 0.06:
            return "center"
        if rect.x0 < page_width * 0.20:
            return "left"
        if rect.x1 > page_width * 0.80:
            return "right"
        return "floating"

    def _assign_reading_order(self, blocks: List[Dict[str, Any]]) -> None:
        indexed = sorted(enumerate(blocks), key=lambda item: (_rect(item[1].get("bbox")).y0, _rect(item[1].get("bbox")).x0))
        for order, (_idx, block) in enumerate(indexed):
            block["reading_order_index"] = order

    def _extract_images(self, page: fitz.Page, page_index: int) -> List[Dict[str, Any]]:
        text_dict = self._safe_page_text(page, "dict")
        out = []
        for block_idx, block in enumerate(text_dict.get("blocks", []) or []):
            if block.get("type") != 1:
                continue
            out.append(
                {
                    "id": f"p{page_index + 1}_image_{block_idx}",
                    "source": "pdf_image_block",
                    "bbox": _bbox(block.get("bbox")),
                    "width_px": block.get("width"),
                    "height_px": block.get("height"),
                    "ext": block.get("ext"),
                    "colorspace": block.get("colorspace"),
                    "xres": block.get("xres"),
                    "yres": block.get("yres"),
                    "transform": list(block.get("transform")) if block.get("transform") else None,
                    "xref": block.get("xref"),
                    "mask": block.get("mask"),
                    "alpha": None,
                    "role": "image",
                    "caption_id": None,
                    "must_preserve_visual": True,
                    "confidence": 1.0,
                }
            )
        return out

    def _extract_drawings(self, page: fitz.Page) -> List[Dict[str, Any]]:
        out = []
        try:
            drawings = page.get_drawings()
        except Exception:
            drawings = []
        for idx, drawing in enumerate(drawings or []):
            rect = drawing.get("rect")
            out.append(
                {
                    "id": f"drawing_{idx}",
                    "source": "pdf_vector",
                    "bbox": _bbox(rect),
                    "type": drawing.get("type"),
                    "items": self._summarize_path_items(drawing.get("items") or []),
                    "fill_color": _color_tuple_to_hex(drawing.get("fill")),
                    "stroke_color": _color_tuple_to_hex(drawing.get("color")),
                    "stroke_width": _round_float(drawing.get("width")),
                    "opacity": _round_float(drawing.get("fill_opacity") if drawing.get("fill_opacity") is not None else drawing.get("stroke_opacity")),
                    "even_odd": drawing.get("even_odd"),
                    "close_path": drawing.get("closePath"),
                    "line_cap": drawing.get("lineCap"),
                    "line_join": drawing.get("lineJoin"),
                    "dashes": drawing.get("dashes"),
                    "role": self._classify_drawing(drawing),
                    "must_preserve_visual": True,
                }
            )
        return out

    def _summarize_path_items(self, items: Sequence[Any]) -> List[Dict[str, Any]]:
        out = []
        for item in items[:80]:
            try:
                cmd = item[0]
                coords = []
                for part in item[1:]:
                    if isinstance(part, fitz.Point):
                        coords.append([round(part.x, 3), round(part.y, 3)])
                    elif isinstance(part, fitz.Rect):
                        coords.append(_bbox(part))
                    else:
                        coords.append(str(part))
                out.append({"cmd": cmd, "coords": coords})
            except Exception:
                out.append({"cmd": None, "coords": None})
        return out

    def _classify_drawing(self, drawing: Dict[str, Any]) -> str:
        rect = _rect(drawing.get("rect"))
        fill = drawing.get("fill")
        stroke = drawing.get("color")
        if rect.get_area() > 20_000 and fill and not stroke:
            return "background_rect"
        if rect.width > 20 and rect.height < 4:
            return "rule_line"
        if rect.height > 20 and rect.width < 4:
            return "vertical_rule"
        return "shape"

    def _extract_ocr_if_needed(self, page: fitz.Page, page_image_info: Dict[str, Any], native_text: Dict[str, Any]) -> Dict[str, Any]:
        if not self.config.enable_ocr:
            return self._empty_ocr_result("disabled")
        if native_text.get("char_count", 0) > 10:
            return self._empty_ocr_result("not_needed_native_text_available")
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(self.config.target_dpi / 72.0, self.config.target_dpi / 72.0), alpha=False)
            image = self._pixmap_to_pil(pix)
            if image is None:
                return self._empty_ocr_result("pil_unavailable")
            engine = self._get_ocr_engine()
            if engine is None:
                return self._empty_ocr_result("ocr_engine_unavailable")
            result = engine(image)
            return self._normalize_ocr_result(result, page.rect.width, page.rect.height, pix.width, pix.height)
        except Exception as exc:
            out = self._empty_ocr_result("ocr_error")
            out["error"] = str(exc)
            return out

    def _empty_ocr_result(self, reason: str) -> Dict[str, Any]:
        return {
            "available": False,
            "source": "ocr",
            "reason": reason,
            "engine": None,
            "block_count": 0,
            "line_count": 0,
            "word_count": 0,
            "char_count": 0,
            "blocks": [],
            "words_flat": [],
            "confidence": None,
        }

    def _get_ocr_engine(self) -> Any:
        if self._ocr_engine is not None:
            return self._ocr_engine
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._ocr_engine = RapidOCR()
            return self._ocr_engine
        except Exception:
            self._ocr_engine = None
            return None

    def _normalize_ocr_result(self, result: Any, page_w: float, page_h: float, img_w: int, img_h: int) -> Dict[str, Any]:
        entries = result[0] if isinstance(result, tuple) and result else result
        blocks = []
        words = []
        if not entries:
            return self._empty_ocr_result("empty")
        sx = page_w / max(1, img_w)
        sy = page_h / max(1, img_h)
        for idx, entry in enumerate(entries):
            try:
                points, text, conf = entry[0], entry[1], float(entry[2])
                xs = [float(p[0]) * sx for p in points]
                ys = [float(p[1]) * sy for p in points]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
            except Exception:
                continue
            word_items = []
            for widx, word in enumerate(re.findall(r"\S+", str(text or ""))):
                word_item = {
                    "id": f"ocr_word_{idx}_{widx}",
                    "text": word,
                    "bbox": None,
                    "case_profile": _case_profile(word),
                    "confidence": conf,
                }
                words.append(word_item)
                word_items.append(word_item)
            line = {
                "id": f"ocr_line_{idx}",
                "text": text,
                "bbox": _bbox(bbox),
                "spans": [],
                "words": word_items,
                "case_profile": _case_profile(text),
                "confidence": conf,
            }
            blocks.append(
                {
                    "id": f"ocr_block_{idx}",
                    "source": "ocr",
                    "role": self._classify_text_role(text, bbox, []),
                    "object_type": "text_block",
                    "text": text,
                    "bbox": _bbox(bbox),
                    "lines": [line],
                    "words": word_items,
                    "style_summary": {
                        "dominant_font_family": None,
                        "dominant_font_key_normalized": None,
                        "dominant_size_pt": None,
                        "dominant_color_rgb": None,
                        "has_bold": None,
                        "has_italic": None,
                        "has_serif": None,
                        "has_monospace": None,
                        "has_symbolic": None,
                        "span_style_count": 0,
                    },
                    "case_profile": _case_profile(text),
                    "language": None,
                    "script": None,
                    "reading_order_index": idx,
                    "render_order_index": idx,
                    "translatable": None,
                    "non_translatable_reason": None,
                    "semantic": {
                        "heading_level": None,
                        "list_type": None,
                        "list_marker": None,
                        "caption_for": None,
                        "table_membership": None,
                        "formula_membership": None,
                    },
                    "layout": {
                        "column_index": None,
                        "paragraph_index": None,
                        "alignment": self._infer_block_alignment(bbox, page_w),
                        "indent_left": None,
                        "indent_right": None,
                        "space_before": None,
                        "space_after": None,
                        "keep_with_next": None,
                    },
                    "reconstruction_constraints": {
                        "preserve_bbox": True,
                        "preserve_font_size": None,
                        "preserve_case": _case_profile(text)["must_preserve_case"],
                        "preserve_color": None,
                        "preserve_style_flags": None,
                        "allow_reflow": True,
                        "allow_horizontal_expansion": True,
                        "allow_vertical_expansion": True,
                        "allow_overlap": False,
                        "must_render": True,
                    },
                    "confidence": conf,
                }
            )
        return {
            "available": bool(blocks),
            "source": "ocr",
            "engine": "RapidOCR",
            "block_count": len(blocks),
            "line_count": len(blocks),
            "word_count": len(words),
            "char_count": sum(len(str(b.get("text") or "")) for b in blocks),
            "blocks": blocks,
            "words_flat": words,
            "confidence": round(statistics.mean([b["confidence"] for b in blocks]), 3) if blocks else None,
        }

    def _detect_special_regions(
        self,
        blocks: List[Dict[str, Any]],
        page: fitz.Page,
        page_image: Any,
        sx: float,
        sy: float,
        width: float,
        height: float,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.config.enable_special_regions:
            return [], {"detector": "disabled", "special_region_count": 0}
        page_data = {
            "dimensions": {"width": width, "height": height},
            "blocks": [self._block_for_special_detector(block, sx, sy) for block in blocks],
            "layout_ai_structure": {},
        }
        try:
            from special_region_detector import detect_special_regions

            enriched, report = detect_special_regions(page_data, page_image=page_image, pdf_page=page, sx=sx, sy=sy)
            regions = []
            for idx, item in enumerate(enriched.get("special_regions") or []):
                region = self._normalize_special_region(item, idx, sx, sy)
                if self._special_region_is_valid(region, page):
                    regions.append(region)
            regions.extend(self._detect_code_regions(blocks, sx, sy, start=len(regions)))
            report = dict(report or {})
            report["raw_special_region_count_before_filter"] = report.get("special_region_count")
            report["special_region_count_after_filter"] = len(regions)
            return self._merge_special_regions(regions), report
        except Exception as exc:
            regions = self._detect_code_regions(blocks, sx, sy, start=0)
            return regions, {"detector": "fallback_code_only", "special_region_count": len(regions), "error": str(exc)}

    def _special_region_is_valid(self, region: Dict[str, Any], page: fitz.Page) -> bool:
        if region.get("class") == "code":
            return self._region_text_looks_like_code(region, page)
        if region.get("class") != "formula":
            return True
        signature = self._region_text_signature(region, page)
        text = signature["text_compact"]
        fonts = signature["fonts_lower"]
        if not text:
            source = str(region.get("detection_source") or "").lower()
            return "ai" in source or "layout" in source
        if text in {"■", "•", "▪", "▫", ""}:
            return False
        if all(any(marker in font for marker in ("zapfdingbats", "wingdings")) for font in fonts) and len(text) <= 3:
            return False
        math_fonts = [
            font
            for font in fonts
            if any(marker in font for marker in ("math", "symbol", "cmmi", "cmsy", "cmex", "stix", "mtex"))
            and not any(marker in font for marker in ("zapfdingbats", "wingdings"))
        ]
        heavy_math = bool(re.search(r"[=∂∑∏∫√∞≈≠≤≥±×÷−∆Ω∗·δµμ^{}]|[α-ωΑ-Ω]", text))
        operator_run = len(re.findall(r"[+\-−*/=<>_^]", text))
        natural_words = re.findall(r"[A-Za-zÀ-ÿ]{3,}", text)
        if heavy_math and (operator_run >= 1 or math_fonts or len(natural_words) <= 3):
            return True
        if math_fonts and operator_run >= 1 and len(natural_words) <= 3:
            return True
        if re.fullmatch(r"[A-Za-z]{1,4}\d+[A-Za-z0-9]*", text or ""):
            return False
        return False

    def _region_text_looks_like_code(self, region: Dict[str, Any], page: fitz.Page) -> bool:
        signature = self._region_text_signature(region, page)
        text = signature["text"]
        if not text:
            return False
        code_punct = len(re.findall(r"[{}();=<>]|==|!=|<=|>=|//|::|->|=>", text))
        code_keywords = self._code_syntax_signal_count(text)
        function_syntax = bool(re.search(r"\bfunction\s+[$A-Za-z_][$A-Za-z0-9_]*\s*\(|\bfunction\s*\(", text, re.I))
        command = bool(re.search(r"(^|\s)(/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+|npm\s+|pip\s+|python\s+)", text))
        return bool((code_punct >= 2 and code_keywords >= 1) or code_keywords >= 2 or command or (function_syntax and code_punct >= 2))

    def _code_syntax_signal_count(self, text: str) -> int:
        text = text or ""
        signals = 0
        if re.search(r"^\s*(def|class)\s+[A-Za-z_][A-Za-z0-9_]*\s*(\(|:)", text, re.I | re.M):
            signals += 1
        if re.search(r"^\s*(from\s+[A-Za-z_][A-Za-z0-9_.]*\s+import|import\s+[A-Za-z_][A-Za-z0-9_.]*|return\b)", text, re.I | re.M):
            signals += 1
        sql_words = set(re.findall(r"\b(SELECT|FROM|WHERE|JOIN|INSERT|UPDATE|DELETE|GROUP\s+BY|ORDER\s+BY)\b", text, re.I))
        if len(sql_words) >= 2:
            signals += 1
        if re.search(r"^\s*(const|let|var)\s+[$A-Za-z_][$A-Za-z0-9_]*\s*=", text, re.I | re.M):
            signals += 1
        if re.search(r"^\s*(sudo|npm|pip|python)\s+\S+", text, re.I | re.M):
            signals += 1
        return signals

    def _region_text_signature(self, region: Dict[str, Any], page: fitz.Page) -> Dict[str, Any]:
        rect = _rect(region.get("bbox"))
        if rect.get_area() <= 0:
            return {"text": "", "text_compact": "", "fonts_lower": []}
        chars = []
        fonts = []
        try:
            raw = page.get_text("rawdict")
        except Exception:
            raw = {}
        for block in raw.get("blocks", []) or []:
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []) or []:
                for span in line.get("spans", []) or []:
                    font = str(span.get("font") or "")
                    for char in span.get("chars", []) or []:
                        cbbox = char.get("bbox") or span.get("bbox")
                        crect = _rect(cbbox)
                        if crect.get_area() <= 0:
                            continue
                        if (rect & crect).get_area() / max(1.0, min(rect.get_area(), crect.get_area())) > 0.10:
                            chars.append(str(char.get("c") or ""))
                            if font:
                                fonts.append(font.lower())
        text = "".join(chars)
        return {
            "text": text,
            "text_compact": re.sub(r"\s+", "", text),
            "fonts_lower": sorted(set(fonts)),
        }

    def _block_for_special_detector(self, block: Dict[str, Any], sx: float, sy: float) -> Dict[str, Any]:
        bbox = block.get("bbox")
        return {
            "id": block.get("id"),
            "bbox": self._scale_bbox(bbox, sx, sy),
            "text": block.get("text"),
            "role": block.get("role"),
            "object_type": block.get("object_type"),
            "lines": [
                {
                    "bbox": self._scale_bbox(line.get("bbox"), sx, sy),
                    "text": line.get("text"),
                    "spans": [
                        {
                            "text": span.get("text"),
                            "bbox": self._scale_bbox(span.get("bbox"), sx, sy),
                            "font": span.get("style", {}).get("font_family"),
                            "size": span.get("style", {}).get("size_pt"),
                        }
                        for span in (line.get("spans") or [])
                    ],
                }
                for line in (block.get("lines") or [])
            ],
        }

    def _scale_bbox(self, bbox: Any, sx: float, sy: float) -> Optional[List[float]]:
        bb = _bbox(bbox)
        if not bb:
            return None
        return [round(bb[0] * sx, 3), round(bb[1] * sy, 3), round(bb[2] * sx, 3), round(bb[3] * sy, 3)]

    def _unscale_bbox(self, bbox: Any, sx: float, sy: float) -> Optional[List[float]]:
        bb = _bbox(bbox)
        if not bb:
            return None
        return [round(bb[0] / max(1e-6, sx), 3), round(bb[1] / max(1e-6, sy), 3), round(bb[2] / max(1e-6, sx), 3), round(bb[3] / max(1e-6, sy), 3)]

    def _normalize_special_region(self, item: Dict[str, Any], idx: int, sx: float, sy: float) -> Dict[str, Any]:
        preserve = []
        for sidx, sub in enumerate(item.get("preserve_subregions") or []):
            preserve.append(
                {
                    "id": sub.get("id") or f"special_{idx}_preserve_{sidx}",
                    "bbox": self._unscale_bbox(sub.get("bbox"), sx, sy),
                    "bbox_px": _bbox(sub.get("bbox")),
                    "policy": sub.get("policy") or "preserve_visual",
                    "source": sub.get("source"),
                    "text_hint": sub.get("text_hint"),
                }
            )
        return {
            "id": item.get("id") or f"special_region_{idx}",
            "class": item.get("special_class") or "special",
            "role": item.get("special_class") or "special",
            "bbox": self._unscale_bbox(item.get("visual_bbox") or item.get("bbox"), sx, sy),
            "bbox_px": _bbox(item.get("visual_bbox") or item.get("bbox")),
            "preserve_subregions": preserve,
            "translatable_text_subregions": item.get("text_subregions") or [],
            "render_policy": item.get("render_policy") or "preserve_source_region",
            "translation_policy": item.get("translation_policy") or "preserve_visual_region",
            "detection_source": item.get("detection_source"),
            "confidence": item.get("confidence"),
            "must_preserve_visual": True,
            "must_exclude_from_translation_flow": True,
            "internal_discrimination": {
                "preserve_visual": preserve,
                "translate_text": item.get("text_subregions") or [],
                "protect_token": [],
                "ignore_background": [],
            },
        }

    def _detect_code_regions(self, blocks: List[Dict[str, Any]], sx: float, sy: float, start: int = 0) -> List[Dict[str, Any]]:
        out = []
        for block in blocks:
            if block.get("role") != "code":
                continue
            idx = start + len(out)
            out.append(
                {
                    "id": f"special_region_code_{idx}",
                    "class": "code",
                    "role": "code",
                    "bbox": block.get("bbox"),
                    "bbox_px": self._scale_bbox(block.get("bbox"), sx, sy),
                    "preserve_subregions": [
                        {
                            "id": f"code_preserve_{idx}",
                            "bbox": block.get("bbox"),
                            "bbox_px": self._scale_bbox(block.get("bbox"), sx, sy),
                            "policy": "preserve_visual",
                            "source": "monospace_code_heuristic",
                            "text_hint": block.get("text"),
                        }
                    ],
                    "translatable_text_subregions": [],
                    "render_policy": "preserve_source_region",
                    "translation_policy": "preserve_visual_region",
                    "detection_source": "monospace_code_heuristic",
                    "confidence": 0.78,
                    "must_preserve_visual": True,
                    "must_exclude_from_translation_flow": True,
                    "internal_discrimination": {
                        "preserve_visual": [],
                        "translate_text": [],
                        "protect_token": [],
                        "ignore_background": [],
                    },
                }
            )
        return out

    def _merge_special_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not regions:
            return []
        # Conservative de-duplication: keep the tightest duplicate for each class.
        kept: List[Dict[str, Any]] = []
        for region in sorted(regions, key=lambda r: (_rect_area(r.get("bbox")), r.get("id") or "")):
            rect = _rect(region.get("bbox"))
            duplicate = False
            for old in kept:
                old_rect = _rect(old.get("bbox"))
                inter = (rect & old_rect).get_area()
                if inter / max(1.0, min(rect.get_area(), old_rect.get_area())) > 0.82 and region.get("class") == old.get("class"):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(region)
        return sorted(kept, key=lambda r: (_rect(r.get("bbox")).y0, _rect(r.get("bbox")).x0))

    def _infer_regions(self, width: float, height: float, blocks: List[Dict[str, Any]], images: List[Dict[str, Any]], drawings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        content_bbox = _union_bbox([b.get("bbox") for b in blocks] + [i.get("bbox") for i in images] + [d.get("bbox") for d in drawings])
        return [
            {
                "id": "page_body",
                "type": "body",
                "bbox": content_bbox,
                "role": "main_content",
                "reading_order_index": 0,
                "column_count": self._infer_column_count(width, blocks),
                "background": None,
                "constraints": {"allow_overlap": False, "must_render_all_text": True},
                "confidence": 0.5 if content_bbox else None,
            }
        ]

    def _infer_columns(self, width: float, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        count = self._infer_column_count(width, blocks)
        return {
            "count": count,
            "widths": None,
            "gutters": None,
            "bounds": None,
            "reading_order": "left_to_right_top_to_bottom",
            "symmetric": None,
            "sidebar": None,
            "column_breaks": [],
            "spanning_blocks": [],
        }

    def _infer_column_count(self, width: float, blocks: List[Dict[str, Any]]) -> int:
        centers = sorted((_rect(b.get("bbox")).x0 + _rect(b.get("bbox")).x1) / 2.0 for b in blocks if _rect_area(b.get("bbox")) > 0)
        if len(centers) < 4:
            return 1
        left = sum(1 for c in centers if c < width * 0.46)
        right = sum(1 for c in centers if c > width * 0.54)
        middle = len(centers) - left - right
        if left >= 2 and right >= 2 and middle <= max(2, len(centers) * 0.25):
            return 2
        return 1

    def _detect_tables(self, blocks: List[Dict[str, Any]], drawings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Initial autonomous heuristic: identify dense aligned text areas plus
        # ruling lines. It remains conservative and exposes nulls when unsure.
        horizontal_rules = [d for d in drawings if d.get("role") == "rule_line"]
        vertical_rules = [d for d in drawings if d.get("role") == "vertical_rule"]
        tables = []
        if len(horizontal_rules) >= 2 and len(vertical_rules) >= 2:
            bbox = _union_bbox([d.get("bbox") for d in horizontal_rules + vertical_rules])
            tables.append(
                {
                    "id": "table_0",
                    "bbox": bbox,
                    "source": "vector_rules_heuristic",
                    "rows": None,
                    "columns": None,
                    "cells": [],
                    "header_rows": None,
                    "style": {
                        "border_color": None,
                        "border_width": None,
                        "fill_colors": [],
                        "grid_detected": True,
                    },
                    "confidence": 0.55,
                }
            )
        return tables

    def _infer_paragraphs(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        paragraphs = []
        for idx, block in enumerate(blocks):
            if block.get("role") in {"paragraph", "heading", "list_item"}:
                paragraphs.append(
                    {
                        "id": f"paragraph_{idx}",
                        "block_ids": [block.get("id")],
                        "text": block.get("text"),
                        "bbox": block.get("bbox"),
                        "role": block.get("role"),
                        "style_summary": block.get("style_summary"),
                        "case_profile": block.get("case_profile"),
                        "reading_order_index": block.get("reading_order_index"),
                        "translation_policy": "translate" if block.get("role") != "heading" else "translate_preserve_case",
                    }
                )
        return paragraphs

    def _infer_translatable_segments(self, blocks: List[Dict[str, Any]], special_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for block in blocks:
            if self._block_overlaps_special(block, special_regions):
                continue
            if block.get("role") in {"formula", "code", "empty"}:
                continue
            text = re.sub(r"\s+", " ", block.get("text") or "").strip()
            if not text:
                continue
            out.append(
                {
                    "id": f"translatable_{block.get('id')}",
                    "block_id": block.get("id"),
                    "text": text,
                    "bbox": block.get("bbox"),
                    "style_summary": block.get("style_summary"),
                    "case_profile": block.get("case_profile"),
                    "translation_policy": "translate_preserve_style",
                    "must_render": True,
                }
            )
        return out

    def _infer_non_translatable_segments(self, blocks: List[Dict[str, Any]], special_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for block in blocks:
            reason = None
            if self._block_overlaps_special(block, special_regions):
                reason = "inside_special_visual_region"
            elif block.get("role") in {"formula", "code", "empty"}:
                reason = block.get("role")
            if reason:
                out.append(
                    {
                        "id": f"non_translatable_{block.get('id')}",
                        "block_id": block.get("id"),
                        "text": block.get("text"),
                        "bbox": block.get("bbox"),
                        "reason": reason,
                        "policy": "preserve_visual" if reason != "empty" else "ignore_empty",
                    }
                )
        return out

    def _block_overlaps_special(self, block: Dict[str, Any], regions: List[Dict[str, Any]]) -> bool:
        rect = _rect(block.get("bbox"))
        if rect.get_area() <= 0:
            return False
        for region in regions:
            rrect = _rect(region.get("bbox"))
            if rrect.get_area() <= 0:
                continue
            if (rect & rrect).get_area() / max(1.0, min(rect.get_area(), rrect.get_area())) > 0.55:
                return True
        return False

    def _extract_annotations(self, page: fitz.Page) -> List[Dict[str, Any]]:
        out = []
        try:
            annot = page.first_annot
            idx = 0
            while annot:
                out.append(
                    {
                        "id": f"annotation_{idx}",
                        "type": annot.type[1] if annot.type else None,
                        "bbox": _bbox(annot.rect),
                        "content": annot.info.get("content") if annot.info else None,
                        "title": annot.info.get("title") if annot.info else None,
                        "subject": annot.info.get("subject") if annot.info else None,
                        "flags": annot.flags,
                        "colors": annot.colors,
                    }
                )
                annot = annot.next
                idx += 1
        except Exception:
            pass
        return out

    def _build_render_objects(self, blocks: List[Dict[str, Any]], images: List[Dict[str, Any]], drawings: List[Dict[str, Any]], special_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        objects = []
        for block in blocks:
            objects.append({"id": block.get("id"), "type": "text", "bbox": block.get("bbox"), "source_ref": block.get("id"), "z_index": block.get("render_order_index"), "must_render": True})
        offset = len(objects)
        for idx, drawing in enumerate(drawings):
            objects.append({"id": drawing.get("id"), "type": "drawing", "bbox": drawing.get("bbox"), "source_ref": drawing.get("id"), "z_index": offset + idx, "must_render": True})
        offset = len(objects)
        for idx, image in enumerate(images):
            objects.append({"id": image.get("id"), "type": "image", "bbox": image.get("bbox"), "source_ref": image.get("id"), "z_index": offset + idx, "must_render": True})
        offset = len(objects)
        for idx, region in enumerate(special_regions):
            objects.append({"id": region.get("id"), "type": "special_region", "bbox": region.get("bbox"), "source_ref": region.get("id"), "z_index": offset + idx, "must_render": True})
        return objects

    def _build_reading_order(self, blocks: List[Dict[str, Any]], images: List[Dict[str, Any]], drawings: List[Dict[str, Any]], special_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items = []
        for block in blocks:
            items.append({"id": block.get("id"), "type": "text", "bbox": block.get("bbox"), "order": block.get("reading_order_index")})
        for region in special_regions:
            items.append({"id": region.get("id"), "type": region.get("class"), "bbox": region.get("bbox"), "order": None})
        sorted_items = sorted(items, key=lambda item: (_rect(item.get("bbox")).y0, _rect(item.get("bbox")).x0))
        for idx, item in enumerate(sorted_items):
            item["order"] = idx
        return sorted_items

    def _build_render_order(self, render_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            [{"id": obj.get("id"), "type": obj.get("type"), "z_index": idx, "bbox": obj.get("bbox")} for idx, obj in enumerate(render_objects)],
            key=lambda item: item["z_index"],
        )

    def _infer_relationships(self, blocks: List[Dict[str, Any]], images: List[Dict[str, Any]], tables: List[Dict[str, Any]], special_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        relationships = []
        for image in images:
            caption = self._nearest_caption(image, blocks)
            if caption:
                relationships.append({"type": "caption_for_image", "from": caption.get("id"), "to": image.get("id"), "confidence": 0.45})
        for table in tables:
            caption = self._nearest_caption(table, blocks)
            if caption:
                relationships.append({"type": "caption_for_table", "from": caption.get("id"), "to": table.get("id"), "confidence": 0.45})
        for region in special_regions:
            nearby = self._nearest_caption(region, blocks)
            if nearby:
                relationships.append({"type": "text_context_for_special_region", "from": nearby.get("id"), "to": region.get("id"), "confidence": 0.35})
        return relationships

    def _nearest_caption(self, obj: Dict[str, Any], blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        rect = _rect(obj.get("bbox"))
        if rect.get_area() <= 0:
            return None
        candidates = []
        for block in blocks:
            text = re.sub(r"\s+", " ", block.get("text") or "").strip()
            if not text:
                continue
            brect = _rect(block.get("bbox"))
            distance = abs(brect.y0 - rect.y1)
            if distance < 60 or re.match(r"^(fig\\.?|figure|table|tab\\.)", text, re.I):
                candidates.append((distance, block))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _page_reconstruction_constraints(self, width: float, height: float, blocks: List[Dict[str, Any]], images: List[Dict[str, Any]], drawings: List[Dict[str, Any]], special_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "page_size_must_match_source": True,
            "all_text_must_render": True,
            "overlap_allowed": False,
            "formula_regions_fixed_as_visual": True,
            "image_regions_fixed_as_visual": True,
            "preserve_typography": True,
            "preserve_font_size": True,
            "preserve_colors": True,
            "preserve_case": True,
            "layout_reflow_allowed": True,
            "horizontal_expansion_allowed": True,
            "vertical_expansion_allowed": True,
            "cascade_reposition_required_on_collision": True,
            "page_bounds": [0, 0, round(width, 3), round(height, 3)],
            "fixed_obstacles": [
                {"id": obj.get("id"), "type": kind, "bbox": obj.get("bbox")}
                for kind, collection in (("image", images), ("drawing", drawings), ("special_region", special_regions))
                for obj in collection
            ],
        }

    def _estimate_margins(self, width: float, height: float, blocks: List[Dict[str, Any]], images: List[Dict[str, Any]], drawings: List[Dict[str, Any]]) -> Dict[str, Any]:
        bbox = _union_bbox([b.get("bbox") for b in blocks] + [i.get("bbox") for i in images] + [d.get("bbox") for d in drawings])
        if not bbox:
            return {"left": None, "right": None, "top": None, "bottom": None}
        return {
            "left": round(bbox[0], 3),
            "right": round(width - bbox[2], 3),
            "top": round(bbox[1], 3),
            "bottom": round(height - bbox[3], 3),
        }

    def _estimate_line_step(self, blocks: List[Dict[str, Any]]) -> Optional[float]:
        ys = []
        for block in blocks:
            for line in block.get("lines") or []:
                rect = _rect(line.get("bbox"))
                if rect.get_area() > 0:
                    ys.append(rect.y0)
        ys = sorted(set(round(y, 1) for y in ys))
        deltas = [b - a for a, b in zip(ys, ys[1:]) if 2 <= b - a <= 60]
        return round(statistics.median(deltas), 3) if deltas else None

    def _estimate_baseline_grid(self, blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        step = self._estimate_line_step(blocks)
        if step is None:
            return None
        return {"step": step, "offset": None, "confidence": 0.35}

    def _extract_background_profile(self, page_image_info: Dict[str, Any], drawings: List[Dict[str, Any]], images: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "color": None,
            "image": None,
            "texture": None,
            "scanned_paper": None,
            "noise": None,
            "shadow": None,
            "gradient": None,
            "watermark": None,
            "pattern": None,
            "page_border": None,
            "vector_background": any(d.get("role") == "background_rect" for d in drawings),
            "raster_background": bool(images and not drawings),
            "transparency": None,
            "layers": [],
            "rendered_page_image": page_image_info,
        }

    def _page_quality_report(self, page_model: Dict[str, Any], special_report: Dict[str, Any]) -> Dict[str, Any]:
        text_blocks = page_model.get("text", {}).get("blocks") or []
        missing_style = sum(1 for b in text_blocks if not (b.get("style_summary") or {}).get("dominant_font_family"))
        field_presence_score = self._field_presence_score(page_model)
        warnings = self._page_warnings(page_model)
        return {
            "field_presence_score": field_presence_score,
            "native_text_available": bool(page_model.get("text", {}).get("native", {}).get("available")),
            "ocr_used": page_model.get("text", {}).get("chosen_source") == "ocr",
            "text_block_count": len(text_blocks),
            "image_count": len(page_model.get("images") or []),
            "drawing_count": len(page_model.get("drawings") or []),
            "table_count": len(page_model.get("tables") or []),
            "special_region_count": len(page_model.get("formulas_and_specials") or []),
            "special_region_detector": special_report,
            "missing_style_block_count": missing_style,
            "warnings": warnings,
            "reconstructibility_score": self._page_reconstructibility_score(page_model, warnings),
        }

    def _field_presence_score(self, value: Any) -> float:
        total = 0
        filled = 0

        def walk(item: Any) -> None:
            nonlocal total, filled
            if isinstance(item, dict):
                for v in item.values():
                    walk(v)
            elif isinstance(item, list):
                for v in item:
                    walk(v)
            else:
                total += 1
                if item is not None:
                    filled += 1

        walk(value)
        return round(filled / max(1, total), 3)

    def _page_warnings(self, page_model: Dict[str, Any]) -> List[str]:
        warnings = []
        if not page_model.get("text", {}).get("blocks"):
            warnings.append("no_text_blocks_detected")
        if page_model.get("text", {}).get("native", {}).get("available") is False and page_model.get("text", {}).get("ocr", {}).get("available") is False:
            warnings.append("no_native_or_ocr_text")
        if any(r.get("bbox") is None for r in page_model.get("formulas_and_specials") or []):
            warnings.append("special_region_missing_bbox")
        return warnings

    def _page_reconstructibility_score(self, page_model: Dict[str, Any], warnings: Optional[List[str]] = None) -> float:
        score = 0.45
        if page_model.get("text", {}).get("native", {}).get("available"):
            score += 0.25
        if page_model.get("images") is not None:
            score += 0.05
        if page_model.get("drawings") is not None:
            score += 0.05
        if page_model.get("formulas_and_specials") is not None:
            score += 0.05
        if not (warnings or []):
            score += 0.10
        return round(min(1.0, score), 3)

    def _finalize_document_profile(self, model: Dict[str, Any]) -> None:
        pages = model.get("pages") or []
        orientations = [p.get("geometry", {}).get("orientation") for p in pages]
        profile = model["document_profile"]
        profile["dominant_orientation"] = self._mode(orientations)
        profile["dominant_column_count"] = self._mode([p.get("columns", {}).get("count") for p in pages])
        profile["has_tables"] = any(p.get("tables") for p in pages)
        profile["has_figures"] = any(p.get("images") for p in pages)
        profile["has_formulas"] = any(any(r.get("class") == "formula" for r in p.get("formulas_and_specials") or []) for p in pages)
        profile["has_code"] = any(any(r.get("class") == "code" for r in p.get("formulas_and_specials") or []) for p in pages)
        profile["has_pagination"] = None
        profile["visual_complexity"] = self._complexity_label(sum(len(p.get("render_objects") or []) for p in pages) / max(1, len(pages)))
        profile["document_quality"] = self._quality_label([p.get("quality_report", {}).get("reconstructibility_score") for p in pages])
        profile["reconstructibility_score"] = round(statistics.mean([p.get("quality_report", {}).get("reconstructibility_score") or 0 for p in pages]), 3) if pages else None
        font_resources = []
        seen = set()
        for page in pages:
            for block in page.get("text", {}).get("blocks") or []:
                for line in block.get("lines") or []:
                    for span in line.get("spans") or []:
                        style = span.get("style") or {}
                        key = (style.get("font_family"), style.get("embedded_font_path"))
                        if key in seen:
                            continue
                        seen.add(key)
                        font_resources.append(
                            {
                                "font_family": style.get("font_family"),
                                "font_key_normalized": style.get("font_key_normalized"),
                                "embedded_font_path": style.get("embedded_font_path"),
                                "bold": style.get("bold"),
                                "italic": style.get("italic"),
                                "serif": style.get("serif"),
                                "monospace": style.get("monospace"),
                            }
                        )
        model["resources"]["fonts"] = font_resources
        model["pdf"]["has_embedded_fonts"] = any(f.get("embedded_font_path") for f in font_resources)

    def _complexity_label(self, count: float) -> Optional[str]:
        if count <= 0:
            return None
        if count < 15:
            return "low"
        if count < 45:
            return "medium"
        return "high"

    def _quality_label(self, scores: Sequence[Any]) -> Optional[str]:
        clean = [float(s) for s in scores if s is not None]
        if not clean:
            return None
        avg = statistics.mean(clean)
        if avg >= 0.8:
            return "high"
        if avg >= 0.55:
            return "medium"
        return "low"

    def _document_quality_report(self, model: Dict[str, Any]) -> Dict[str, Any]:
        pages = model.get("pages") or []
        return {
            "page_count_extracted": len(pages),
            "field_presence_score": self._field_presence_score(model),
            "mean_page_reconstructibility_score": round(statistics.mean([p.get("quality_report", {}).get("reconstructibility_score") or 0 for p in pages]), 3) if pages else None,
            "total_text_blocks": sum(len(p.get("text", {}).get("blocks") or []) for p in pages),
            "total_images": sum(len(p.get("images") or []) for p in pages),
            "total_drawings": sum(len(p.get("drawings") or []) for p in pages),
            "total_tables": sum(len(p.get("tables") or []) for p in pages),
            "total_special_regions": sum(len(p.get("formulas_and_specials") or []) for p in pages),
            "warnings": sorted({w for p in pages for w in (p.get("quality_report", {}).get("warnings") or [])}),
        }


def _parse_pages(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    pages: List[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = [int(v.strip()) for v in chunk.split("-", 1)]
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(chunk))
    return pages


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Autonomous complete PDF extraction.")
    parser.add_argument("source", help="PDF source path")
    parser.add_argument("--out", required=True, help="Output JSON path or output directory")
    parser.add_argument("--pages", help="1-based pages, e.g. 1,3,5-8")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-special-regions", action="store_true")
    args = parser.parse_args(argv)

    config = ExtractionConfig(
        target_dpi=args.dpi,
        enable_ocr=not args.no_ocr,
        enable_special_regions=not args.no_special_regions,
    )
    extractor = PerfectDocumentExtractor(config)
    model = extractor.extract(args.source, pages=_parse_pages(args.pages))

    out_path = Path(args.out)
    if out_path.suffix.lower() != ".json":
        out_path.mkdir(parents=True, exist_ok=True)
        source_stem = Path(args.source).stem
        page_suffix = (args.pages or "all").replace(",", "_").replace("-", "_")
        out_path = out_path / f"{source_stem}_perfect_extraction_pages_{page_suffix}.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
