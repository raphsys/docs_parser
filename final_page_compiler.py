from __future__ import annotations

import argparse
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import fitz
from PIL import Image

from font_resolver import FontResolver
from special_region_detector import detect_special_regions


@dataclass
class FormulaItem:
    formula_id: str
    rect: fitz.Rect
    source_rect: fitz.Rect
    clips: list[fitz.Rect]
    text_subregions: list[fitz.Rect]
    linked_text_ids: list[str]


@dataclass
class TextItem:
    block_id: str
    role: str
    rect: fitz.Rect
    text: str
    style: dict[str, Any]
    fontfile: str | None = None
    fontname: str | None = None
    alignment: str = "left"
    color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    coverage_fallback: bool = False


@dataclass
class DrawOp:
    kind: str
    rect: fitz.Rect
    text: str = ""
    formula_id: str = ""
    source_rect: fitz.Rect | None = None
    source_clips: list[fitz.Rect] | None = None
    source_erase_rects: list[fitz.Rect] | None = None
    font_size: float = 10.0
    fontfile: str | None = None
    fontname: str = "Times-Roman"
    alignment: str = "left"
    color: tuple[float, float, float] = (0.0, 0.0, 0.0)


class ContinuousFinalPageCompiler:
    """Build a final page as a continuous text/formula flow.

    This is intentionally separate from reconstructor.py. It consumes already
    detected translated text blocks and formula regions, then composes them in
    reading order. Formulas are inserted when they occur in the flow instead of
    being pre-placed as global obstacles.
    """

    def __init__(self, page_data: dict[str, Any], *, pixel_to_point: float = 72.0 / 150.0):
        self.page_data = self._normalize_page_data(page_data)
        self.pixel_to_point = pixel_to_point
        self.page_width, self.page_height = self._page_size()
        dims = dict(self.page_data.get("dimensions") or {})
        px_width = float(dims.get("width") or self.page_data.get("page_width") or 0)
        px_height = float(dims.get("height") or self.page_data.get("page_height") or 0)
        self.pixel_to_point_x = self.page_width / px_width if px_width > 0 else self.pixel_to_point
        self.pixel_to_point_y = self.page_height / px_height if px_height > 0 else self.pixel_to_point
        self.margin = 34.0
        self.body_font = self._first_existing_font(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
            ]
        )
        self.bold_font = self._first_existing_font(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
            ]
        ) or self.body_font
        self._font_cache: dict[tuple[str | None, float], fitz.Font] = {}
        self._font_resolver = FontResolver()
        self._source_page_image: Image.Image | None = None
        self._source_span_cache: dict[str, dict[str, Any]] = {}
        self._formula_protected_rect_cache: list[fitz.Rect] | None = None
        self._block_policies = {
            str(p.get("block_id")): p
            for p in (self.page_data.get("positioning_policy") or {}).get("block_policies") or []
            if isinstance(p, dict) and p.get("block_id")
        }
        self._ensure_formula_regions()

    def _normalize_page_data(self, page_data: dict[str, Any]) -> dict[str, Any]:
        data = dict(page_data or {})
        document = data.get("document") if isinstance(data.get("document"), dict) else {}
        if document and "source_pdf_path" not in data:
            data["source_pdf_path"] = document.get("pdf")
        if document and "source_page_index" not in data:
            data["source_page_index"] = document.get("page_index", 0)
        if document and "dimensions" not in data:
            data["dimensions"] = document.get("dimensions") or {}

        # scripts/export_extraction_translation.py stores block geometry in
        # blocks and text/translation units in items. Convert that schema into
        # the canonical block shape consumed by this compiler.
        items_payload = data.get("items") if isinstance(data.get("items"), dict) else {}
        if document and items_payload:
            by_block: dict[str, list[dict[str, Any]]] = {}
            for group_name in ("phrases", "expressions", "mots"):
                for item in items_payload.get(group_name) or []:
                    if not isinstance(item, dict) or not item.get("block_id"):
                        continue
                    by_block.setdefault(str(item.get("block_id")), []).append(item)
            normalized_blocks = []
            for block in data.get("blocks") or []:
                if not isinstance(block, dict):
                    continue
                block_id = str(block.get("id") or block.get("block_id") or f"block_{len(normalized_blocks)}")
                units = by_block.get(block_id) or []
                source = self._clean_text(" ".join(str(item.get("source") or "") for item in units))
                translation = self._clean_text(" ".join(str(item.get("translation") or item.get("source") or "") for item in units))
                if not translation:
                    translation = self._clean_text(block.get("context") or "")
                    source = translation
                translation = self._fallback_translation(translation if translation != source else source)
                style_samples = [s for s in block.get("style_samples") or [] if isinstance(s, dict)]
                style_sample = next((s for s in style_samples if not self._is_math_font_name(str(s.get("font") or ""))), None)
                if style_sample is None and style_samples:
                    style_sample = style_samples[0]
                style_sample = style_sample or {}
                flags = {
                    "bold": bool(style_sample.get("bold")),
                    "italic": bool(style_sample.get("italic")),
                    "serif": bool(style_sample.get("serif", True)),
                }
                normalized_blocks.append(
                    {
                        "id": block_id,
                        "role": block.get("role") or "body",
                        "bbox": block.get("bbox"),
                        "source_bbox": block.get("bbox"),
                        "original_bbox": block.get("bbox"),
                        "text": source,
                        "translated_text": translation,
                        "object_class": "formula" if self._looks_like_formula_text(source) else "",
                        "style": {
                            "font": style_sample.get("font") or "Times-Roman",
                            "size": style_sample.get("size") or 10.0,
                            "color": style_sample.get("color") or "#000000",
                            "flags": flags,
                        },
                    }
                )
            data["blocks"] = normalized_blocks

        if "formula_regions" not in data and isinstance(data.get("special_regions"), list):
            data["formula_regions"] = data["special_regions"]
        return data

    def _ensure_formula_regions(self) -> None:
        if self.page_data.get("formula_regions"):
            return
        source_pdf = self._source_pdf_path()
        if source_pdf is None:
            return
        dims = dict(self.page_data.get("dimensions") or {})
        px_width = float(dims.get("width") or self.page_data.get("page_width") or 0)
        px_height = float(dims.get("height") or self.page_data.get("page_height") or 0)
        try:
            with fitz.open(source_pdf) as doc:
                page = doc[self._source_page_index()]
                sx = px_width / page.rect.width if px_width > 0 else self.pixel_to_point_x
                sy = px_height / page.rect.height if px_height > 0 else self.pixel_to_point_y
                page_image = None
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(sx, sy), alpha=False)
                    page_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                except Exception:
                    page_image = None
                enriched, _info = detect_special_regions(self.page_data, page_image=page_image, pdf_page=page, sx=sx, sy=sy)
                regions = enriched.get("formula_regions") or enriched.get("special_regions") or []
                if regions:
                    self.page_data["formula_regions"] = regions
        except Exception:
            return

    def _first_existing_font(self, candidates: list[str]) -> str | None:
        for candidate in candidates:
            if Path(candidate).exists():
                return candidate
        return None

    def _page_size(self) -> tuple[float, float]:
        source_pdf = Path(str(self.page_data.get("source_pdf_path") or ""))
        if not source_pdf.exists():
            source_pdf = Path.cwd() / source_pdf
        if source_pdf.exists():
            source_page_index = int(self.page_data.get("source_page_index") or self.page_data.get("page_index") or 0)
            try:
                with fitz.open(source_pdf) as source_doc:
                    rect = source_doc[source_page_index].rect
                    if rect.width > 0 and rect.height > 0:
                        return float(rect.width), float(rect.height)
            except Exception:
                pass
        dims = dict(self.page_data.get("dimensions") or {})
        width_pt = dims.get("width_pt") or dims.get("page_width_pt")
        height_pt = dims.get("height_pt") or dims.get("page_height_pt")
        if width_pt and height_pt:
            return float(width_pt), float(height_pt)
        width = float(dims.get("width") or self.page_data.get("page_width") or 879)
        height = float(dims.get("height") or self.page_data.get("page_height") or 1333)
        return width * self.pixel_to_point, height * self.pixel_to_point

    def _rect(self, bbox: Any) -> fitz.Rect | None:
        if isinstance(bbox, fitz.Rect):
            return fitz.Rect(bbox)
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            values = [float(v) for v in bbox]
        except Exception:
            return None
        rect = fitz.Rect(
            values[0] * self.pixel_to_point_x,
            values[1] * self.pixel_to_point_y,
            values[2] * self.pixel_to_point_x,
            values[3] * self.pixel_to_point_y,
        )
        return rect if rect.get_area() > 0 else None

    def _font(self, fontfile: str | None, size: float) -> fitz.Font:
        key = (fontfile, round(float(size), 2))
        if key in self._font_cache:
            return self._font_cache[key]
        try:
            font = fitz.Font(fontfile=fontfile) if fontfile else fitz.Font("Times-Roman")
        except Exception:
            font = fitz.Font("Times-Roman")
        self._font_cache[key] = font
        return font

    def _width(self, text: str, font_size: float, fontfile: str | None) -> float:
        return self._font(fontfile, font_size).text_length(text, fontsize=font_size)

    def _parse_color(self, color: Any) -> tuple[float, float, float]:
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            try:
                return tuple(max(0.0, min(1.0, float(c))) for c in color[:3])  # type: ignore[return-value]
            except Exception:
                return (0.0, 0.0, 0.0)
        if isinstance(color, str) and color.startswith("#") and len(color) == 7:
            try:
                return (int(color[1:3], 16) / 255.0, int(color[3:5], 16) / 255.0, int(color[5:7], 16) / 255.0)
            except Exception:
                return (0.0, 0.0, 0.0)
        return (0.0, 0.0, 0.0)

    def _source_pdf_path(self) -> Path | None:
        source_pdf = Path(str(self.page_data.get("source_pdf_path") or ""))
        if not source_pdf.exists():
            source_pdf = Path.cwd() / source_pdf
        return source_pdf if source_pdf.exists() else None

    def _normalize_font_key(self, font_name: str) -> str:
        name = re.sub(r"^[A-Z]{6}\+", "", str(font_name or ""))
        name = name.lower()
        name = re.sub(r"[^a-z0-9]+", "", name)
        return name

    def _is_math_font_name(self, font_name: str) -> bool:
        key = self._normalize_font_key(font_name)
        return any(marker in key for marker in ("mtmi", "mtsyn", "mtex", "symbol", "math"))

    def _looks_like_formula_text(self, text: str) -> bool:
        value = str(text or "")
        if not value.strip():
            return False
        natural = [
            tok
            for tok in re.findall(r"[A-Za-zÀ-ÿ]{3,}", value)
            if tok.lower() not in {"target", "softmax", "relu", "sigmoid"}
        ]
        math_chars = len(
            re.findall(
                r"[∂∑∏∫√∞≈≠≤≥±×÷−∆Ω∗·δµμ=<>*/^_{}()[\]\\|]|\x02|\x03|\x04|\x05|\x06|\x07",
                value,
            )
        )
        compact = re.sub(r"\s+", "", value)
        if math_chars >= 2 and len(natural) <= 2:
            return True
        if len(compact) <= 18 and re.search(r"[A-Za-z]\d|\d[A-Za-z]|[=∂∑∗*/^_]", compact) and len(natural) <= 1:
            return True
        return False

    def _span_weight(self, span: dict[str, Any], rect: fitz.Rect) -> float:
        try:
            sb = fitz.Rect(span.get("bbox"))
            inter = sb & rect
            if inter.width > 0 and inter.height > 0:
                return max(inter.width, 0.0) * max(inter.height, 0.0)
            return max(sb.width, 0.0)
        except Exception:
            return 1.0

    def _rect_coverage_ratio(self, a: fitz.Rect, b: fitz.Rect) -> float:
        inter = a & b
        if inter.width <= 0 or inter.height <= 0:
            return 0.0
        return inter.get_area() / max(1.0, min(a.get_area(), b.get_area()))

    def _is_covered_by_any(self, rect: fitz.Rect, others: list[fitz.Rect], *, threshold: float = 0.55) -> bool:
        return any(self._rect_coverage_ratio(rect, other) >= threshold for other in others)

    def _source_page_index(self) -> int:
        return int(self.page_data.get("source_page_index") or self.page_data.get("page_index") or 0)

    def _source_style_for_rect(self, rect: fitz.Rect, *, text_hint: str = "") -> dict[str, Any]:
        cache_key = f"{self._source_page_index()}:{round(rect.x0,1)}:{round(rect.y0,1)}:{round(rect.x1,1)}:{round(rect.y1,1)}:{text_hint[:32]}"
        if cache_key in self._source_span_cache:
            return self._source_span_cache[cache_key]
        source_pdf = self._source_pdf_path()
        result = {
            "size": 10.0,
            "font": "Times-Roman",
            "fontfile": None,
            "color": (0.0, 0.0, 0.0),
            "alignment": "left",
            "italic": False,
            "bold": False,
        }
        if source_pdf is None:
            self._source_span_cache[cache_key] = result
            return result
        try:
            with fitz.open(source_pdf) as doc:
                page = doc[self._source_page_index()]
                spans = []
                blocks = page.get_text("dict").get("blocks") or []
                for block in blocks:
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines") or []:
                        for span in line.get("spans") or []:
                            sb = fitz.Rect(span.get("bbox"))
                            if sb.intersects(rect) or (sb & rect).get_area() > 0:
                                spans.append(span)
                if spans:
                    text_spans = [s for s in spans if not self._is_math_font_name(str(s.get("font") or ""))]
                    if text_spans:
                        spans = text_spans
                    groups: dict[float, list[dict[str, Any]]] = {}
                    for s in spans:
                        try:
                            key = round(float(s.get("size") or 10.0), 2)
                        except Exception:
                            key = 10.0
                        groups.setdefault(key, []).append(s)
                    def weight(group: list[dict[str, Any]]) -> float:
                        return sum(self._span_weight(s, rect) for s in group)
                    size_key = max(groups.items(), key=lambda kv: (weight(kv[1]), kv[0]))[0]
                    chosen = groups[size_key]
                    result["size"] = float(size_key)
                    font_weights: dict[str, float] = {}
                    for s in chosen:
                        font_name = str(s.get("font") or "")
                        font_weights[font_name] = font_weights.get(font_name, 0.0) + self._span_weight(s, rect)
                    font = max(font_weights.items(), key=lambda kv: kv[1])[0] if font_weights else ""
                    result["font"] = font or self.body_font
                    flags = [int(s.get("flags") or 0) for s in chosen]
                    colors: dict[Any, float] = {}
                    bold_weight = 0.0
                    italic_weight = 0.0
                    total_weight = 0.0
                    for s, f in zip(chosen, flags):
                        w = self._span_weight(s, rect)
                        total_weight += w
                        if f & 16:
                            bold_weight += w
                        if (f & 2) or "italic" in str(s.get("font") or "").lower():
                            italic_weight += w
                        if s.get("color") is not None:
                            colors[s.get("color")] = colors.get(s.get("color"), 0.0) + w
                    result["fontfile"] = None
                    result["bold"] = total_weight > 0 and bold_weight / total_weight >= 0.65
                    result["italic"] = total_weight > 0 and italic_weight / total_weight >= 0.65
                    if colors:
                        color = max(colors.items(), key=lambda kv: kv[1])[0]
                        result["color"] = self._parse_color(color)
        except Exception:
            pass
        self._source_span_cache[cache_key] = result
        return result

    def _fontfile_for_source_font(self, font_name: str, flags: int = 0) -> str | None:
        name = (font_name or "").lower()
        if "bold" in name or (flags & 16):
            if "italic" in name or (flags & 2):
                return self._first_existing_font([
                    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf",
                    "/usr/share/fonts/truetype/liberation2/LiberationSerif-BoldItalic.ttf",
                ]) or self.bold_font
            return self.bold_font
        if "italic" in name or (flags & 2):
            return self._first_existing_font([
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf",
            ]) or self.body_font
        if "times" in name or "serif" in name:
            return self.body_font
        return self.body_font

    def _alignment_for_block(self, block: dict[str, Any]) -> str:
        block_id = str(block.get("id") or "")
        policy = self._block_policies.get(block_id) or {}
        phrase_policies = policy.get("phrase_policies") or []
        if phrase_policies:
            phrase = phrase_policies[0] or {}
            pref = str(((phrase.get("primary_position_reference") or {}).get("horizontal")) or "").lower()
            signal_align = str(((phrase.get("signals") or {}).get("alignment")) or "").lower()
            if pref in {"center", "middle"} or signal_align in {"center", "middle"}:
                return "center"
            if pref in {"right", "end"} or signal_align in {"right", "end"}:
                return "right"
        role = str(block.get("role") or "").lower()
        rect = self._rect(block.get("source_bbox") or block.get("original_bbox") or block.get("bbox"))
        if role in {"title", "heading", "section_heading"} and rect is not None:
            if rect.x0 > self.page_width * 0.35 and rect.width < self.page_width * 0.8:
                return "center"
        return "left"

    def _style_for_text(self, item: TextItem) -> tuple[float, str | None, str, tuple[float, float, float]]:
        style = item.style or {}
        try:
            size = float(style.get("size") or 10.0)
        except Exception:
            size = 10.0
        if size <= 0:
            size = 10.0
        resolved = self._font_resolver.resolve(style, text=item.text)
        fontfile = resolved.get("fontfile")
        style["_resolved_builtin"] = resolved.get("builtin")
        style["_resolved_fontfile"] = fontfile
        return size, fontfile, item.alignment, item.color

    def _pdf_builtin_fontname(self, builtin: str | None) -> str:
        mapping = {
            "tiro": "Times-Roman",
            "tibo": "Times-Bold",
            "tiit": "Times-Italic",
            "tibi": "Times-BoldItalic",
            "helv": "Helvetica",
            "hebo": "Helvetica-Bold",
            "heit": "Helvetica-Oblique",
            "hebi": "Helvetica-BoldOblique",
            "cour": "Courier",
            "cobo": "Courier-Bold",
            "coit": "Courier-Oblique",
            "cobi": "Courier-BoldOblique",
        }
        return mapping.get(str(builtin or ""), str(builtin or "Times-Roman"))

    def _render_fontname_for_text(self, item: TextItem) -> str:
        style = item.style or {}
        if style.get("_resolved_fontfile"):
            return ""
        if style.get("_resolved_builtin"):
            return self._pdf_builtin_fontname(str(style.get("_resolved_builtin")))
        flags = style.get("flags") if isinstance(style.get("flags"), dict) else {}
        font_name = str(style.get("font") or item.fontname or "").lower()
        is_bold = bool(flags.get("bold")) or "bold" in font_name
        is_italic = bool(flags.get("italic")) or "italic" in font_name
        if is_bold and is_italic:
            return "Times-BoldItalic"
        if is_bold:
            return "Times-Bold"
        if is_italic:
            return "Times-Italic"
        return "Times-Roman"

    def _formula_clip_should_merge(self, left: fitz.Rect, right: fitz.Rect) -> bool:
        if left.get_area() <= 0 or right.get_area() <= 0:
            return False
        if (left & right).get_area() > 0:
            return True
        horizontal_gap = max(0.0, max(left.x0, right.x0) - min(left.x1, right.x1))
        vertical_gap = max(0.0, max(left.y0, right.y0) - min(left.y1, right.y1))
        vertical_overlap = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0)) / max(1.0, min(left.height, right.height))
        horizontal_overlap = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0)) / max(1.0, min(left.width, right.width))
        same_line = vertical_overlap >= 0.10 and horizontal_gap <= max(8.5, 1.7 * min(left.height, right.height))
        same_stack = horizontal_overlap >= 0.10 and vertical_gap <= max(6.5, 1.15 * min(left.height, right.height))
        aligned_stack = (
            vertical_gap <= max(5.5, 0.34 * max(left.height, right.height))
            and abs((left.x0 + left.x1) * 0.5 - (right.x0 + right.x1) * 0.5) <= max(16.0, 0.42 * max(left.width, right.width))
        )
        return bool(same_line or same_stack or aligned_stack)

    def _merge_formula_clips(self, clips: list[fitz.Rect], source_rect: fitz.Rect) -> list[fitz.Rect]:
        clean: list[fitz.Rect] = []
        for clip in clips:
            if clip.get_area() <= 0:
                continue
            rect = fitz.Rect(
                max(source_rect.x0, clip.x0 - 0.35),
                max(source_rect.y0, clip.y0 - 0.35),
                min(source_rect.x1, clip.x1 + 0.35),
                min(source_rect.y1, clip.y1 + 0.35),
            )
            if rect.get_area() > 1.0:
                clean.append(rect)
        if not clean:
            return []

        components: list[list[fitz.Rect]] = [[rect] for rect in sorted(clean, key=lambda item: (item.y0, item.x0))]
        changed = True
        while changed:
            changed = False
            merged: list[list[fitz.Rect]] = []
            while components:
                current = components.pop(0)
                current_rect = fitz.Rect(current[0])
                for rect in current[1:]:
                    current_rect |= rect
                absorbed = []
                for idx, other in enumerate(components):
                    other_rect = fitz.Rect(other[0])
                    for rect in other[1:]:
                        other_rect |= rect
                    if self._formula_clip_should_merge(current_rect, other_rect):
                        current.extend(other)
                        current_rect |= other_rect
                        absorbed.append(idx)
                        changed = True
                components = [component for idx, component in enumerate(components) if idx not in absorbed]
                merged.append(current)
            components = merged

        out: list[fitz.Rect] = []
        for component in components:
            rect = fitz.Rect(component[0])
            for clip in component[1:]:
                rect |= clip
            rect = fitz.Rect(
                max(source_rect.x0, rect.x0 - 0.25),
                max(source_rect.y0, rect.y0 - 0.25),
                min(source_rect.x1, rect.x1 + 0.25),
                min(source_rect.y1, rect.y1 + 0.25),
            )
            if rect.get_area() > 1.0:
                out.append(rect)
        return sorted(out, key=lambda item: (item.y0, item.x0))

    def formulas(self) -> list[FormulaItem]:
        items: list[FormulaItem] = []
        seen: set[tuple[float, float, float, float]] = set()
        for region in self.page_data.get("formula_regions") or []:
            region_id = str(region.get("id") or f"formula_{len(items)}")
            source_rect = self._rect(region.get("visual_bbox") or region.get("bbox"))
            if source_rect is None:
                continue
            source_rect = fitz.Rect(
                max(0.0, source_rect.x0 - 1.5),
                max(0.0, source_rect.y0 - 1.2),
                min(self.page_width, source_rect.x1 + 1.5),
                min(self.page_height, source_rect.y1 + 1.2),
            )
            key = tuple(round(v, 2) for v in (source_rect.x0, source_rect.y0, source_rect.x1, source_rect.y1))
            if key in seen:
                continue
            seen.add(key)
            detected_clips = []
            for subregion in list(region.get("formula_subregions") or []) + list(region.get("preserve_subregions") or []):
                clip = self._rect((subregion or {}).get("bbox"))
                if clip is not None and clip.get_area() > 0:
                    detected_clips.append(clip)
            clips = self._merge_formula_clips(detected_clips, source_rect) or [source_rect]
            text_subregions = []
            for subregion in region.get("text_subregions") or []:
                rect = self._rect((subregion or {}).get("bbox"))
                if rect is not None and rect.get_area() > 0:
                    text_subregions.append(rect)
            linked_text_ids = [
                str(block_id)
                for block_id in (region.get("translatable_block_ids") or region.get("block_ids") or [])
                if str(block_id).strip()
            ]
            items.append(
                FormulaItem(
                    formula_id=region_id,
                    rect=source_rect,
                    source_rect=source_rect,
                    clips=clips,
                    text_subregions=text_subregions,
                    linked_text_ids=linked_text_ids,
                )
            )
        return sorted(items, key=lambda item: (item.rect.y0, item.rect.x0))

    def _formula_protected_rects(self) -> list[fitz.Rect]:
        if self._formula_protected_rect_cache is not None:
            return self._formula_protected_rect_cache
        rects: list[fitz.Rect] = []
        for formula in self.formulas():
            rects.append(formula.rect)
            rects.extend(formula.clips)
        self._formula_protected_rect_cache = rects
        return rects

    def _text_is_absorbed_by_formula(self, rect: fitz.Rect, text: str) -> bool:
        protected = self._formula_protected_rects()
        if not protected:
            return False
        compact = re.sub(r"\s+", "", text or "")
        words = re.findall(r"[A-Za-zÀ-ÿ]{3,}(?:['-][A-Za-zÀ-ÿ]{2,})?", text or "")
        short_or_symbolic = len(compact) <= 12 or self._looks_like_formula_text(text)
        best = 0.0
        for item in protected:
            inter = rect & item
            if inter.width <= 0 or inter.height <= 0:
                continue
            best = max(best, inter.get_area() / max(1.0, rect.get_area()))
        if best >= 0.72:
            return True
        if short_or_symbolic and best >= 0.38:
            return True
        if len(words) <= 1 and best >= 0.55:
            return True
        return False

    def text_blocks(self) -> list[TextItem]:
        out: list[TextItem] = []
        for block in self.page_data.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            if block.get("formula_region_id") or str(block.get("object_class") or "").lower() == "formula":
                continue
            text = self._clean_text(block.get("translated_text") or block.get("text") or block.get("line_text") or "")
            if not text:
                continue
            rect = self._rect(block.get("source_bbox") or block.get("original_bbox") or block.get("bbox"))
            if rect is None:
                continue
            if self._text_is_absorbed_by_formula(rect, text):
                continue
            style = dict(block.get("style") or {})
            source_style = self._source_style_for_rect(rect, text_hint=text)
            block_fontfile = None
            style["size"] = source_style.get("size", style.get("size"))
            style["font"] = source_style.get("font", style.get("font"))
            style["color"] = style.get("color") or "#000000"
            style_flags = dict(style.get("flags") or {})
            style_flags["bold"] = bool(source_style.get("bold"))
            style_flags["italic"] = bool(source_style.get("italic"))
            style["flags"] = style_flags
            out.append(
                TextItem(
                    block_id=str(block.get("id") or f"text_{len(out)}"),
                    role=str(block.get("role") or "body"),
                    rect=rect,
                    text=text,
                    style=style,
                    fontfile=block_fontfile,
                    fontname=str(source_style.get("font") or style.get("font") or ""),
                    alignment=self._alignment_for_block(block),
                    color=self._parse_color(source_style.get("color") or style.get("color")),
                )
            )
        return self._merge_text_fragments(sorted(out, key=lambda item: (item.rect.y0, item.rect.x0)))

    def _merge_text_fragments(self, items: list[TextItem]) -> list[TextItem]:
        if not items:
            return []
        merged: list[TextItem] = []
        for item in items:
            if not merged:
                merged.append(item)
                continue
            prev = merged[-1]
            same_band = item.rect.y0 <= prev.rect.y1 + 12.0
            same_family = str(prev.style.get("font") or "") == str(item.style.get("font") or "")
            same_size = abs(float(prev.style.get("size") or 10.0) - float(item.style.get("size") or 10.0)) <= 0.5
            overlaps_x = min(prev.rect.x1, item.rect.x1) - max(prev.rect.x0, item.rect.x0) > 20.0
            continuation = bool(re.match(r"^[a-zà-ÿ]", item.text)) or len(item.text) <= 18
            not_header = max(prev.rect.y1, item.rect.y1) > self.page_height * 0.18
            if same_band and same_family and same_size and overlaps_x and continuation and not_header:
                merged[-1] = TextItem(
                    block_id=f"{prev.block_id}+{item.block_id}",
                    role=prev.role,
                    rect=fitz.Rect(
                        min(prev.rect.x0, item.rect.x0),
                        min(prev.rect.y0, item.rect.y0),
                        max(prev.rect.x1, item.rect.x1),
                        max(prev.rect.y1, item.rect.y1),
                    ),
                    text=self._clean_text(f"{prev.text} {item.text}"),
                    style=prev.style,
                    fontfile=prev.fontfile,
                    fontname=prev.fontname,
                    alignment=prev.alignment,
                    color=prev.color,
                    coverage_fallback=prev.coverage_fallback and item.coverage_fallback,
                )
            else:
                merged.append(item)
        return merged

    def _fallback_translation(self, text: str) -> str:
        clean = self._clean_text(text)
        low = clean.lower()
        fixed = {
            "therefore,": "Par conséquent,",
            "therefore": "Par conséquent",
            "from eq.": "Depuis l'Eq.",
            "using the formula": "en utilisant la formule",
            "formula": "formule",
            "is represented by": "est représenté par",
            "which are given as": "qui sont donnés comme",
        }
        for src, dst in fixed.items():
            if low == src:
                return dst
        if low.startswith("from eq."):
            return re.sub(r"(?i)^from eq\\.", "Depuis l'Eq.", clean)
        return clean

    def _source_text_line_fallbacks(self, existing: list[TextItem], formulas: list[FormulaItem]) -> list[TextItem]:
        source_pdf = self._source_pdf_path()
        if source_pdf is None:
            return []
        existing_rects = [item.rect for item in existing]
        formula_rects = [item.rect for item in formulas]
        out: list[TextItem] = []
        try:
            with fitz.open(source_pdf) as doc:
                page = doc[self._source_page_index()]
                for block in page.get_text("dict").get("blocks") or []:
                    if block.get("type") != 0:
                        continue
                    for line in block.get("lines") or []:
                        spans = [s for s in line.get("spans") or [] if str(s.get("text") or "").strip()]
                        if not spans:
                            continue
                        text_spans = [s for s in spans if not self._is_math_font_name(str(s.get("font") or ""))]
                        if text_spans:
                            spans = text_spans
                        ordered = []
                        for span in spans:
                            try:
                                rect = fitz.Rect(span.get("bbox"))
                            except Exception:
                                continue
                            if rect.get_area() <= 0:
                                continue
                            ordered.append((rect, span))
                        ordered.sort(key=lambda item: item[0].x0)
                        clusters: list[list[tuple[fitz.Rect, dict[str, Any]]]] = []
                        for rect, span in ordered:
                            if not clusters:
                                clusters.append([(rect, span)])
                                continue
                            prev_rect = clusters[-1][-1][0]
                            size = float(span.get("size") or 10.0)
                            if rect.x0 - prev_rect.x1 > max(18.0, size * 2.2):
                                clusters.append([(rect, span)])
                            else:
                                clusters[-1].append((rect, span))
                        for cluster in clusters:
                            rect = fitz.Rect(cluster[0][0])
                            for r, _ in cluster[1:]:
                                rect |= r
                            if self._is_covered_by_any(rect, existing_rects, threshold=0.35):
                                continue
                            if self._is_covered_by_any(rect, formula_rects, threshold=0.35):
                                continue
                            parts: list[str] = []
                            previous: fitz.Rect | None = None
                            for span_rect, span in cluster:
                                value = str(span.get("text") or "")
                                if not value.strip():
                                    continue
                                if previous is not None:
                                    size = float(span.get("size") or 10.0)
                                    if span_rect.x0 - previous.x1 > max(1.5, size * 0.16):
                                        parts.append(" ")
                                parts.append(value)
                                previous = span_rect
                            text = self._clean_text("".join(parts))
                            if not text:
                                continue
                            if self._looks_like_formula_text(text):
                                continue
                            cluster_spans = [s for _, s in cluster]
                            sizes = [float(s.get("size") or 10.0) for s in cluster_spans]
                            size = median(sizes) if sizes else 10.0
                            font = str((cluster_spans[0].get("font") or "Times-Roman")) if cluster_spans else "Times-Roman"
                            flags = [int(s.get("flags") or 0) for s in cluster_spans]
                            style = {
                                "font": font,
                                "size": size,
                                "color": "#000000",
                                "flags": {
                                    "bold": bool(flags and sum(1 for f in flags if f & 16) / len(flags) >= 0.65),
                                    "italic": bool(flags and sum(1 for f in flags if f & 2) / len(flags) >= 0.65),
                                    "serif": True,
                                },
                            }
                            out.append(
                                TextItem(
                                    block_id=f"coverage_source_line_{len(out)}",
                                    role="coverage_fallback",
                                    rect=rect,
                                    text=self._fallback_translation(text),
                                    style=style,
                                    fontfile=None,
                                    fontname=font,
                                    alignment="left",
                                    color=(0.0, 0.0, 0.0),
                                    coverage_fallback=True,
                                )
                            )
        except Exception:
            return []
        return out

    def _clean_text(self, text: Any) -> str:
        text = str(text or "").replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _is_inline_formula_for_text(self, text: TextItem, formula: FormulaItem) -> bool:
        y_overlap = max(0.0, min(text.rect.y1, formula.rect.y1) - max(text.rect.y0, formula.rect.y0))
        text_link_overlap = any(
            max(0.0, min(text.rect.x1, sub.x1) - max(text.rect.x0, sub.x0))
            * max(0.0, min(text.rect.y1, sub.y1) - max(text.rect.y0, sub.y0))
            > 1.0
            for sub in formula.text_subregions
        )
        id_link = text.block_id in formula.linked_text_ids
        if y_overlap <= 0 and not text_link_overlap and not id_link:
            return False
        ratio = y_overlap / max(1.0, min(text.rect.height, formula.rect.height)) if y_overlap > 0 else 0.0
        if ratio < 0.18 and not text_link_overlap and not id_link:
            return False
        compact_formula = formula.rect.height <= max(18.0, text.rect.height * 1.4) and formula.rect.width <= 95.0
        if formula.rect.height > max(18.0, text.rect.height * 2.0) and not compact_formula:
            return False
        if compact_formula and (text_link_overlap or id_link or y_overlap > 0):
            return True
        # Wider inline derivative chains usually sit inside/right of the
        # running line. A wide formula starting near the left margin is a
        # display equation row, even if its bbox vertically overlaps text.
        return formula.rect.x0 >= text.rect.x0 + max(28.0, text.rect.width * 0.32)

    def _inline_formulas_for_text(self, text: TextItem, formulas: list[FormulaItem], used: set[str]) -> list[FormulaItem]:
        out = []
        for formula in formulas:
            if formula.formula_id in used:
                continue
            if self._is_inline_formula_for_text(text, formula):
                out.append(formula)
        return sorted(out, key=lambda item: (item.rect.y0, item.rect.x0))

    def _display_formulas_overlapping_text(self, text: TextItem, formulas: list[FormulaItem], used: set[str]) -> list[FormulaItem]:
        out = []
        for formula in formulas:
            if formula.formula_id in used or self._is_inline_formula_for_text(text, formula):
                continue
            y_overlap = max(0.0, min(text.rect.y1, formula.rect.y1) - max(text.rect.y0, formula.rect.y0))
            if y_overlap > 0.5:
                out.append(formula)
        return sorted(out, key=lambda item: (item.rect.y0, item.rect.x0))

    def _formulas_between(
        self,
        formulas: list[FormulaItem],
        used: set[str],
        y0: float,
        y1: float | None,
    ) -> list[FormulaItem]:
        out = []
        for formula in formulas:
            if formula.formula_id in used:
                continue
            if formula.rect.y0 + 0.5 < y0:
                continue
            if y1 is not None and formula.rect.y0 >= y1 - 0.5:
                continue
            out.append(formula)
        return sorted(out, key=lambda item: (item.rect.y0, item.rect.x0))

    def _is_inline_for_nearby_text(self, formula: FormulaItem, texts: list[TextItem], start_idx: int, window: int = 2) -> bool:
        for text in texts[start_idx : min(len(texts), start_idx + window)]:
            if self._is_inline_formula_for_text(text, formula):
                return True
        return False

    def _text_tokens_with_inline_formulas(self, text: TextItem, inline_formulas: list[FormulaItem]) -> list[Any]:
        if not inline_formulas:
            return text.text.split()
        # General heuristic: insert inline formulas at sentence boundaries when
        # possible; otherwise append them at the end of the current phrase.
        parts = re.split(r"(?<=[.!?])\s+", text.text, maxsplit=1)
        tokens: list[Any] = []
        head = parts[0] if parts else text.text
        tokens.extend(head.split())
        for formula in inline_formulas:
            tokens.append(formula)
        tail = " ".join(parts[1:]).strip()
        if tail:
            tokens.extend(tail.split())
        return tokens

    def _compose_text_item(
        self,
        text: TextItem,
        y: float,
        inline_formulas: list[FormulaItem],
        used: set[str],
    ) -> tuple[list[DrawOp], float]:
        font_size, fontfile, alignment, color = self._style_for_text(text)
        line_h = max(font_size * 1.25, 9.5)
        x0 = max(self.margin, text.rect.x0)
        right = self.page_width - self.margin
        if alignment == "center":
            x0 = max(self.margin, text.rect.x0)
        elif alignment == "right":
            x0 = max(self.margin, text.rect.x1 - max(80.0, text.rect.width))
        elif text.role.lower() in {"title", "heading", "section_heading"} and text.rect.x0 > self.page_width * 0.35:
            x0 = text.rect.x0
        max_width = max(80.0, right - x0)
        y = text.rect.y0 if text.coverage_fallback else max(y, text.rect.y0)
        baseline = y + font_size
        x = x0
        ops: list[DrawOp] = []
        tokens = self._text_tokens_with_inline_formulas(text, inline_formulas)
        anchor_y = y

        for token in tokens:
            if isinstance(token, FormulaItem):
                width = token.rect.width
                height = token.rect.height
                place_x = max(x, min(token.rect.x0, right - width))
                if place_x + width > right and x > x0:
                    baseline += max(line_h, height + 2.0)
                    x = x0
                    place_x = max(x, min(token.rect.x0, right - width))
                top = baseline - min(height * 0.72, max(height - 2.0, font_size))
                rect = fitz.Rect(place_x, top, place_x + width, top + height)
                ops.append(
                    DrawOp(
                        "formula",
                        rect,
                        formula_id=token.formula_id,
                        source_rect=token.source_rect,
                        source_clips=list(token.clips),
                        source_erase_rects=list(token.text_subregions),
                    )
                )
                used.add(token.formula_id)
                x = rect.x1 + 4.0
                baseline = max(baseline, rect.y1 + font_size * 0.25)
                if x > right - 24.0:
                    baseline += line_h
                    x = x0
                continue

            word = str(token)
            word_width = self._width(word, font_size, fontfile)
            space_width = self._width(" ", font_size, fontfile) * 0.96
            if x > x0 and x + word_width > x0 + max_width:
                baseline += line_h
                x = x0
            rect = fitz.Rect(x, baseline - font_size, x + word_width, baseline + font_size * 0.25)
            ops.append(
                DrawOp(
                    "text",
                    rect,
                    text=word,
                    font_size=font_size,
                    fontfile=fontfile,
                    fontname=self._render_fontname_for_text(text),
                    alignment=alignment,
                    color=color,
                )
            )
            x += word_width + space_width

        max_bottom = max((op.rect.y1 for op in ops), default=baseline)
        if text.coverage_fallback:
            return ops, anchor_y
        return ops, max_bottom + max(4.0, font_size * 0.35)

    def _compose_formula_group(self, group: list[FormulaItem], y: float, used: set[str]) -> tuple[list[DrawOp], float]:
        if not group:
            return [], y
        source_top = min(item.rect.y0 for item in group)
        source_bottom = max(item.rect.y1 for item in group)
        y = max(y, source_top)
        ops: list[DrawOp] = []
        for item in group:
            rect = fitz.Rect(item.rect.x0, y + (item.rect.y0 - source_top), item.rect.x1, y + (item.rect.y1 - source_top))
            ops.append(
                DrawOp(
                    "formula",
                    rect,
                    formula_id=item.formula_id,
                    source_rect=item.source_rect,
                    source_clips=list(item.clips),
                    source_erase_rects=list(item.text_subregions),
                )
            )
            used.add(item.formula_id)
        return ops, y + (source_bottom - source_top) + 8.0

    def _formula_groups(self, formulas: list[FormulaItem]) -> list[list[FormulaItem]]:
        groups: list[list[FormulaItem]] = []
        for formula in formulas:
            if not groups:
                groups.append([formula])
                continue
            current = groups[-1]
            current_top = min(item.rect.y0 for item in current)
            current_bottom = max(item.rect.y1 for item in current)
            if formula.rect.y0 <= current_bottom + 8.0 or abs(formula.rect.y0 - current_top) <= 8.0:
                current.append(formula)
            else:
                groups.append([formula])
        return groups

    def _overlaps(self, a: fitz.Rect, b: fitz.Rect, *, min_ratio: float = 0.02) -> bool:
        inter = a & b
        if inter.width <= 0.5 or inter.height <= 0.5:
            return False
        return inter.get_area() / max(1.0, min(a.get_area(), b.get_area())) >= min_ratio

    def _shift_op(self, op: DrawOp, dx: float = 0.0, dy: float = 0.0) -> None:
        op.rect = fitz.Rect(op.rect.x0 + dx, op.rect.y0 + dy, op.rect.x1 + dx, op.rect.y1 + dy)

    def _group_visual_bands(self, ops: list[DrawOp]) -> list[list[DrawOp]]:
        bands: list[list[DrawOp]] = []
        current: list[DrawOp] = []
        current_rect: fitz.Rect | None = None
        for op in sorted(ops, key=lambda item: (item.rect.y0, item.rect.x0)):
            if current_rect is None:
                current = [op]
                current_rect = fitz.Rect(op.rect)
                continue
            y_overlap = min(current_rect.y1, op.rect.y1) - max(current_rect.y0, op.rect.y0)
            same_band = y_overlap > 1.5 or abs(op.rect.y0 - current_rect.y0) <= 3.0
            if same_band:
                current.append(op)
                current_rect |= op.rect
            else:
                bands.append(current)
                current = [op]
                current_rect = fitz.Rect(op.rect)
        if current:
            bands.append(current)
        return bands

    def _band_rect(self, band: list[DrawOp]) -> fitz.Rect:
        rect = fitz.Rect(band[0].rect)
        for op in band[1:]:
            rect |= op.rect
        return rect

    def _band_required_gap(self, prev_band: list[DrawOp], band: list[DrawOp]) -> float:
        prev_has_formula = any(op.kind == "formula" for op in prev_band)
        band_has_formula = any(op.kind == "formula" for op in band)
        if prev_has_formula and not band_has_formula:
            return 7.0
        if prev_has_formula or band_has_formula:
            return 4.5
        return 1.8

    def _resolve_band_internal_overlaps(self, band: list[DrawOp]) -> list[DrawOp]:
        if len(band) < 2:
            return band
        ordered = sorted(band, key=lambda item: (item.rect.x0, item.rect.y0))
        placed: list[DrawOp] = []
        line_gap = 2.0
        for op in ordered:
            for prev in placed:
                if not self._overlaps(op.rect, prev.rect, min_ratio=0.04):
                    continue
                dx = prev.rect.x1 - op.rect.x0 + 2.0
                if op.rect.x1 + dx <= self.page_width - self.margin:
                    self._shift_op(op, dx=dx)
                else:
                    self._shift_op(op, dx=self.margin - op.rect.x0, dy=prev.rect.y1 - op.rect.y0 + line_gap)
            placed.append(op)
        return ordered

    def _enforce_no_overlap(self, ops: list[DrawOp]) -> list[DrawOp]:
        """Hard layout constraint: no visible draw operation may overlap.

        This pass preserves every text/formula operation and resolves
        collisions by shifting whole visual bands downward in cascade. It is
        deliberately page-level: a collision near the top can move all later
        material, instead of letting local overlap survive.
        """
        if not ops:
            return ops
        bands = [self._resolve_band_internal_overlaps(band) for band in self._group_visual_bands(ops)]
        placed_bands: list[list[DrawOp]] = []
        gap = 2.5
        for band in bands:
            for _ in range(200):
                band_rect = self._band_rect(band)
                dy = 0.0
                for prev_band in placed_bands:
                    prev_rect = self._band_rect(prev_band)
                    x_overlap = min(band_rect.x1, prev_rect.x1) - max(band_rect.x0, prev_rect.x0)
                    required_gap = self._band_required_gap(prev_band, band)
                    if self._overlaps(band_rect, prev_rect, min_ratio=0.01):
                        dy = max(dy, prev_rect.y1 - band_rect.y0 + gap)
                    elif x_overlap > 2.0 and band_rect.y0 < prev_rect.y1 + required_gap:
                        dy = max(dy, prev_rect.y1 + required_gap - band_rect.y0)
                if dy <= 0:
                    break
                for op in band:
                    self._shift_op(op, dy=dy)
            placed_bands.append(band)
        resolved: list[DrawOp] = []
        for band in placed_bands:
            resolved.extend(sorted(band, key=lambda item: (item.rect.y0, item.rect.x0)))
        return sorted(resolved, key=lambda item: (item.rect.y0, item.rect.x0))

    def _fit_ops_to_page_height(self, ops: list[DrawOp]) -> list[DrawOp]:
        """Keep the final page at source size by compacting vertical gaps.

        Font sizes and operation rectangles are preserved. Only whitespace
        between already formed visual bands is reduced, down to a small
        readable minimum, so the page size does not silently grow.
        """
        if not ops:
            return ops
        bottom_limit = self.page_height - min(18.0, self.margin * 0.55)
        bands = self._group_visual_bands(ops)
        if not bands:
            return ops
        bottom = max(self._band_rect(band).y1 for band in bands)
        overflow = bottom - bottom_limit
        if overflow <= 0:
            return ops

        band_rects = [self._band_rect(band) for band in bands]
        gaps: list[float] = []
        min_gaps: list[float] = []
        reducible: list[float] = []
        for idx in range(1, len(bands)):
            gap = band_rects[idx].y0 - band_rects[idx - 1].y1
            min_gap = 1.2
            if any(op.kind == "formula" for op in bands[idx - 1]) or any(op.kind == "formula" for op in bands[idx]):
                min_gap = 2.2
            gaps.append(gap)
            min_gaps.append(min_gap)
            reducible.append(max(0.0, gap - min_gap))

        remaining = overflow
        reductions = [0.0 for _ in gaps]
        for idx in sorted(range(len(gaps)), key=lambda i: reducible[i], reverse=True):
            if remaining <= 0:
                break
            take = min(reducible[idx], remaining)
            reductions[idx] = take
            remaining -= take

        shift = 0.0
        for band_idx, band in enumerate(bands):
            if band_idx > 0:
                shift += reductions[band_idx - 1]
            if shift > 0:
                for op in band:
                    self._shift_op(op, dy=-shift)

        # If whitespace alone was not enough, use the top white margin as the
        # final safe reservoir. This preserves all content and page size.
        bottom = max(op.rect.y1 for op in ops)
        remaining = bottom - bottom_limit
        if remaining > 0:
            top = min(op.rect.y0 for op in ops)
            top_room = max(0.0, top - 12.0)
            take = min(remaining, top_room)
            if take > 0:
                for op in ops:
                    self._shift_op(op, dy=-take)
        return sorted(ops, key=lambda item: (item.rect.y0, item.rect.x0))

    def compile_ops(self) -> tuple[list[DrawOp], float]:
        texts = self.text_blocks()
        formulas = self.formulas()
        fallbacks = self._source_text_line_fallbacks(texts, formulas)
        if fallbacks:
            texts = sorted(texts + fallbacks, key=lambda item: (item.rect.y0, item.rect.x0))
        used: set[str] = set()
        ops: list[DrawOp] = []
        y = self.margin

        for idx, text in enumerate(texts):
            next_y = texts[idx + 1].rect.y0 if idx + 1 < len(texts) else None
            before = self._formulas_between(formulas, used, y0=y, y1=text.rect.y0)
            before = [item for item in before if not self._is_inline_for_nearby_text(item, texts, idx)]
            for group in self._formula_groups(before):
                group_ops, y = self._compose_formula_group(group, y, used)
                ops.extend(group_ops)

            display_overlaps = self._display_formulas_overlapping_text(text, formulas, used)
            for group in self._formula_groups(display_overlaps):
                group_ops, y = self._compose_formula_group(group, y, used)
                ops.extend(group_ops)

            inline = self._inline_formulas_for_text(text, formulas, used)
            text_ops, y = self._compose_text_item(text, y, inline, used)
            ops.extend(text_ops)

            between = self._formulas_between(formulas, used, y0=text.rect.y0, y1=next_y)
            between = [item for item in between if item.formula_id not in used]
            between = [item for item in between if not self._is_inline_for_nearby_text(item, texts, idx + 1)]
            for group in self._formula_groups(between):
                group_ops, y = self._compose_formula_group(group, y, used)
                ops.extend(group_ops)

        remaining = [item for item in formulas if item.formula_id not in used]
        for group in self._formula_groups(remaining):
            group_ops, y = self._compose_formula_group(group, y, used)
            ops.extend(group_ops)

        ops = self._enforce_no_overlap(ops)
        ops = self._fit_ops_to_page_height(ops)
        ops = self._enforce_no_overlap(ops)
        ops = self._fit_ops_to_page_height(ops)
        final_height = self.page_height
        return ops, final_height

    def render(self, output_pdf: Path) -> Path:
        ops, final_height = self.compile_ops()
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        source_pdf = Path(str(self.page_data.get("source_pdf_path") or ""))
        if not source_pdf.exists():
            source_pdf = Path.cwd() / source_pdf
        source_page_index = int(self.page_data.get("source_page_index") or self.page_data.get("page_index") or 0)

        doc = fitz.open()
        page = doc.new_page(width=self.page_width, height=final_height)
        source_doc = fitz.open(source_pdf) if source_pdf.exists() else None
        try:
            source_image = None
            source_page_rect = None
            if source_doc is not None:
                try:
                    source_page_rect = source_doc[source_page_index].rect
                    pix = source_doc[source_page_index].get_pixmap(dpi=150, alpha=False)
                    source_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                except Exception:
                    source_image = None
            def formula_image_stream(op: DrawOp) -> bytes | None:
                if source_image is None or source_page_rect is None or op.source_rect is None:
                    return None
                scale_x = source_image.width / max(1.0, source_page_rect.width)
                scale_y = source_image.height / max(1.0, source_page_rect.height)
                canvas_w = max(1, int(round(op.source_rect.width * scale_x)))
                canvas_h = max(1, int(round(op.source_rect.height * scale_y)))
                canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
                clips = list(op.source_clips or [op.source_rect])
                for clip in clips:
                    px = [
                        int(round(clip.x0 * scale_x)),
                        int(round(clip.y0 * scale_y)),
                        int(round(clip.x1 * scale_x)),
                        int(round(clip.y1 * scale_y)),
                    ]
                    px[0] = max(0, min(source_image.width, px[0]))
                    px[2] = max(0, min(source_image.width, px[2]))
                    px[1] = max(0, min(source_image.height, px[1]))
                    px[3] = max(0, min(source_image.height, px[3]))
                    if px[2] <= px[0] or px[3] <= px[1]:
                        continue
                    crop = source_image.crop(tuple(px)).convert("RGB")
                    off_x = int(round((clip.x0 - op.source_rect.x0) * scale_x))
                    off_y = int(round((clip.y0 - op.source_rect.y0) * scale_y))
                    canvas.paste(crop, (off_x, off_y))
                for erase in op.source_erase_rects or []:
                    intersect = fitz.Rect(
                        max(op.source_rect.x0, erase.x0),
                        max(op.source_rect.y0, erase.y0),
                        min(op.source_rect.x1, erase.x1),
                        min(op.source_rect.y1, erase.y1),
                    )
                    if intersect.get_area() <= 0:
                        continue
                    if intersect.get_area() / max(1.0, op.source_rect.get_area()) > 0.35:
                        continue
                    box = [
                        int(round((intersect.x0 - op.source_rect.x0) * scale_x)),
                        int(round((intersect.y0 - op.source_rect.y0) * scale_y)),
                        int(round((intersect.x1 - op.source_rect.x0) * scale_x)),
                        int(round((intersect.y1 - op.source_rect.y0) * scale_y)),
                    ]
                    box[0] = max(0, min(canvas.width, box[0]))
                    box[2] = max(0, min(canvas.width, box[2]))
                    box[1] = max(0, min(canvas.height, box[1]))
                    box[3] = max(0, min(canvas.height, box[3]))
                    if box[2] > box[0] and box[3] > box[1]:
                        canvas.paste((255, 255, 255), tuple(box))
                stream = io.BytesIO()
                canvas.save(stream, format="PNG")
                return stream.getvalue()

            for op in ops:
                if op.kind == "formula" and source_doc is not None and op.source_rect is not None:
                    stream = formula_image_stream(op)
                    if stream:
                        page.insert_image(op.rect, stream=stream)
                    else:
                        page.show_pdf_page(op.rect, source_doc, source_page_index, clip=op.source_rect)
                elif op.kind == "text":
                    fontname = op.fontname or "Times-Roman"
                    if op.fontfile:
                        fontname = f"F{abs(hash(op.fontfile)) % 100000}"
                        try:
                            page.insert_font(fontname=fontname, fontfile=op.fontfile)
                        except Exception:
                            fontname = "Times-Roman"
                    page.insert_text(
                        (op.rect.x0, op.rect.y0 + op.font_size),
                        op.text,
                        fontsize=op.font_size,
                        fontname=fontname,
                        fill=op.color or (0, 0, 0),
                    )
                elif op.kind == "textbox":
                    fontname = op.fontname or "Times-Roman"
                    if op.fontfile:
                        fontname = f"F{abs(hash(op.fontfile)) % 100000}"
                        try:
                            page.insert_font(fontname=fontname, fontfile=op.fontfile)
                        except Exception:
                            fontname = "Times-Roman"
                    align = {
                        "left": fitz.TEXT_ALIGN_LEFT,
                        "center": fitz.TEXT_ALIGN_CENTER,
                        "right": fitz.TEXT_ALIGN_RIGHT,
                        "justify": fitz.TEXT_ALIGN_JUSTIFY,
                    }.get((op.alignment or "left").lower(), fitz.TEXT_ALIGN_LEFT)
                    remaining = page.insert_textbox(
                        op.rect,
                        op.text,
                        fontsize=op.font_size,
                        fontname=fontname,
                        align=align,
                        fill=op.color or (0, 0, 0),
                    )
                    if isinstance(remaining, (int, float)) and remaining < 0:
                        page.insert_text(
                            (op.rect.x0, op.rect.y0 + op.font_size),
                            op.text,
                            fontsize=op.font_size,
                            fontname=fontname,
                            fill=op.color or (0, 0, 0),
                        )
            doc.save(output_pdf)
        finally:
            if source_doc is not None:
                source_doc.close()
            doc.close()
        return output_pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-data", required=True)
    parser.add_argument("--formula-json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    page_data = json.loads(Path(args.page_data).read_text())
    if args.formula_json:
        formula_payload = json.loads(Path(args.formula_json).read_text())
        if isinstance(formula_payload.get("formula_regions"), list):
            page_data["formula_regions"] = formula_payload["formula_regions"]
        elif isinstance(formula_payload.get("special_regions"), list):
            page_data["formula_regions"] = formula_payload["special_regions"]
        if formula_payload.get("pdf"):
            page_data["source_pdf_path"] = formula_payload["pdf"]
        if formula_payload.get("page_index") is not None:
            page_data["source_page_index"] = formula_payload["page_index"]
    out = Path(args.out)
    pdf = ContinuousFinalPageCompiler(page_data).render(out)
    doc = fitz.open(pdf)
    try:
        png = pdf.with_suffix(".png")
        doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(png)
        print(pdf)
        print(png)
        print(doc[0].rect)
    finally:
        doc.close()


if __name__ == "__main__":
    main()
