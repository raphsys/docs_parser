"""Adapter from PerfectDocumentExtractor output to DocumentReconstructor input.

This file is experimental and intentionally does not modify the existing
pipeline. It converts the richer perfect_extraction document model into the
historical reconstruction contract, while carrying the full perfect metadata
alongside each page/block/line/span.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz
from PIL import Image, ImageDraw

from document_object_contract import build_document_object_contract
from perfect_document_extractor import ExtractionConfig, PerfectDocumentExtractor


POINTS_PER_INCH = 72.0
RECON_DPI = 150.0
PT_TO_RECON_PX = RECON_DPI / POINTS_PER_INCH


def _clean_text(text: Any) -> str:
    return re.sub(r"[ \t]+", " ", str(text or "")).strip()


def _sanitize_reconstructable_text(text: Any) -> str:
    text = str(text or "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
    return re.sub(r"[ \t]+", " ", text)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except Exception:
        return default


def _bbox_pt_to_px(bbox: Any) -> Optional[List[float]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        vals = [float(v) * PT_TO_RECON_PX for v in bbox]
    except Exception:
        return None
    if vals[2] <= vals[0] or vals[3] <= vals[1]:
        return None
    return [round(v, 3) for v in vals]


def _bbox_area(bbox: Any) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)
    except Exception:
        return 0.0


def _intersection_ratio(lhs: Any, rhs: Any) -> float:
    if not isinstance(lhs, (list, tuple)) or not isinstance(rhs, (list, tuple)) or len(lhs) != 4 or len(rhs) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in lhs]
    bx0, by0, bx1, by1 = [float(v) for v in rhs]
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    return inter / max(1.0, min(_bbox_area(lhs), _bbox_area(rhs)))


def _style_from_perfect(style: Dict[str, Any] | None, fallback: Dict[str, Any] | None = None) -> Dict[str, Any]:
    style = dict(style or {})
    fallback = dict(fallback or {})
    flags_raw = dict(style.get("flags") or fallback.get("flags") or {})
    flags = {
        "bold": bool(style.get("bold") if style.get("bold") is not None else flags_raw.get("is_bold") or flags_raw.get("bold")),
        "italic": bool(style.get("italic") if style.get("italic") is not None else flags_raw.get("is_italic") or flags_raw.get("italic")),
        "serif": bool(style.get("serif") if style.get("serif") is not None else flags_raw.get("is_serif") or flags_raw.get("serif")),
        "monospace": bool(style.get("monospace") if style.get("monospace") is not None else flags_raw.get("is_monospace") or flags_raw.get("monospace")),
        "symbolic": bool(style.get("symbolic") if style.get("symbolic") is not None else flags_raw.get("is_symbolic") or flags_raw.get("symbolic")),
        "superscript": bool(style.get("superscript") if style.get("superscript") is not None else flags_raw.get("is_superscript") or flags_raw.get("superscript")),
        "underline": style.get("underline") if style.get("underline") is not None else flags_raw.get("underline"),
        "strikeout": style.get("strikeout") if style.get("strikeout") is not None else flags_raw.get("strikeout"),
    }
    font = (
        style.get("font_family")
        or style.get("font_postscript_name")
        or fallback.get("font")
        or fallback.get("font_name")
        or "helv"
    )
    size = (
        style.get("size_pt")
        or style.get("font_size_pt")
        or fallback.get("size")
        or fallback.get("font_size_pt")
        or 12.0
    )
    color = style.get("color_rgb") or style.get("fill_color") or fallback.get("color") or "#000000"
    out = {
        "font": font,
        "font_name": font,
        "font_family": font,
        "font_key_normalized": style.get("font_key_normalized") or fallback.get("font_key_normalized"),
        "embedded_font_path": style.get("embedded_font_path") or fallback.get("embedded_font_path"),
        "size": float(_safe_float(size, 12.0)),
        "font_size_pt": float(_safe_float(size, 12.0)),
        "color": color,
        "fill_color": color,
        "flags": flags,
        "perfect_style": style,
    }
    return out


def _style_from_block_summary(block: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(block.get("style_summary") or {})
    flags = {
        "bold": bool(summary.get("has_bold")),
        "italic": bool(summary.get("has_italic")),
        "serif": bool(summary.get("has_serif")),
        "monospace": bool(summary.get("has_monospace")),
        "symbolic": bool(summary.get("has_symbolic")),
    }
    return {
        "font": summary.get("dominant_font_family") or "helv",
        "font_name": summary.get("dominant_font_family") or "helv",
        "font_family": summary.get("dominant_font_family") or "helv",
        "font_key_normalized": summary.get("dominant_font_key_normalized"),
        "embedded_font_path": summary.get("embedded_font_path"),
        "size": float(_safe_float(summary.get("dominant_size_pt"), 12.0)),
        "font_size_pt": float(_safe_float(summary.get("dominant_size_pt"), 12.0)),
        "color": summary.get("dominant_color_rgb") or "#000000",
        "fill_color": summary.get("dominant_color_rgb") or "#000000",
        "flags": flags,
    }


def _style_attributes_from_styles(styles: Iterable[Dict[str, Any]], text: Any = "") -> Dict[str, Any]:
    entries = [style for style in styles if isinstance(style, dict)]
    sizes: list[float] = []
    fonts: list[str] = []
    colors: list[str] = []
    embedded_paths: list[str] = []
    flag_counts = {"bold": 0, "italic": 0, "underline": 0, "highlight": 0, "uppercase": 0, "monospace": 0, "serif": 0}
    for style in entries:
        font = style.get("font") or style.get("font_name") or style.get("font_family")
        if font:
            fonts.append(str(font))
        color = style.get("color") or style.get("fill_color")
        if color:
            colors.append(str(color))
        embedded_path = style.get("embedded_font_path")
        if embedded_path:
            embedded_paths.append(str(embedded_path))
        try:
            size = style.get("size") if style.get("size") not in {None, ""} else style.get("font_size_pt")
            if size not in {None, ""}:
                sizes.append(float(size))
        except Exception:
            pass
        flags = style.get("flags") if isinstance(style.get("flags"), dict) else {}
        for key in flag_counts:
            if bool(flags.get(key)):
                flag_counts[key] += 1
    if not any(flag_counts.values()):
        letters = [char for char in str(text or "") if char.isalpha()]
        if letters and all(char.upper() == char for char in letters):
            flag_counts["uppercase"] = max(1, len(entries))
    size_values = sorted(sizes)
    median_size = size_values[len(size_values) // 2] if size_values else 0.0
    if len(size_values) and len(size_values) % 2 == 0:
        median_size = (size_values[len(size_values) // 2 - 1] + size_values[len(size_values) // 2]) / 2.0
    total = max(1, len(entries))
    return {
        "font_family_primary": max(fonts, key=fonts.count) if fonts else "",
        "font_size_pt_median": round(float(median_size), 2),
        "font_size_pt_min": round(min(sizes), 2) if sizes else 0.0,
        "font_size_pt_max": round(max(sizes), 2) if sizes else 0.0,
        "color_primary": max(colors, key=colors.count) if colors else "",
        "embedded_font_path_primary": max(embedded_paths, key=embedded_paths.count) if embedded_paths else "",
        "flags_any": {key: bool(value) for key, value in flag_counts.items()},
        "flags_ratio": {key: round(value / total, 4) for key, value in flag_counts.items()},
    }


def _empty_style() -> Dict[str, Any]:
    return {
        "font": "",
        "font_name": "",
        "font_family": "",
        "font_key_normalized": None,
        "embedded_font_path": None,
        "size": 0.0,
        "font_size_pt": 0.0,
        "color": "",
        "fill_color": "",
        "flags": {
            "bold": None,
            "italic": None,
            "serif": None,
            "monospace": None,
            "symbolic": None,
            "superscript": None,
            "underline": None,
            "strikeout": None,
        },
    }


def _complete_style(style: Dict[str, Any] | None) -> Dict[str, Any]:
    out = _empty_style()
    if isinstance(style, dict):
        flags = dict(out["flags"])
        flags.update(style.get("flags") if isinstance(style.get("flags"), dict) else {})
        out.update(style)
        out["flags"] = flags
    return out


def _empty_style_attributes() -> Dict[str, Any]:
    return {
        "font_family_primary": "",
        "font_size_pt_median": 0.0,
        "font_size_pt_min": 0.0,
        "font_size_pt_max": 0.0,
        "color_primary": "",
        "embedded_font_path_primary": "",
        "flags_any": {
            "bold": False,
            "italic": False,
            "underline": False,
            "highlight": False,
            "uppercase": False,
            "monospace": False,
            "serif": False,
        },
        "flags_ratio": {
            "bold": 0.0,
            "italic": 0.0,
            "underline": 0.0,
            "highlight": 0.0,
            "uppercase": 0.0,
            "monospace": 0.0,
            "serif": 0.0,
        },
    }


def _complete_style_attributes(attrs: Dict[str, Any] | None) -> Dict[str, Any]:
    out = _empty_style_attributes()
    if isinstance(attrs, dict):
        flags_any = dict(out["flags_any"])
        flags_ratio = dict(out["flags_ratio"])
        flags_any.update(attrs.get("flags_any") if isinstance(attrs.get("flags_any"), dict) else {})
        flags_ratio.update(attrs.get("flags_ratio") if isinstance(attrs.get("flags_ratio"), dict) else {})
        out.update(attrs)
        out["flags_any"] = flags_any
        out["flags_ratio"] = flags_ratio
    return out


def _default_layout_attributes() -> Dict[str, Any]:
    return {
        "alignment": None,
        "column_index": None,
        "paragraph_index": None,
        "rotation_deg": 0,
        "writing_mode": None,
        "direction": None,
        "indent_left": None,
        "indent_right": None,
        "space_before": None,
        "space_after": None,
        "padding_left_px": 0,
        "padding_right_px": 0,
        "padding_top_px": 0,
        "padding_bottom_px": 0,
        "perfect_layout": {},
    }


def _default_source_layout_mode() -> Dict[str, Any]:
    return {
        "line_flow": "",
        "render_contract": "",
        "source_line_flow": "",
        "source_render_contract": "",
        "can_reflow_within_paragraph": None,
        "alignment": None,
        "perfect_layout": {},
    }


def _default_structure_hints() -> Dict[str, Any]:
    return {
        "band_role_hint": "",
        "structural_role_hint": "",
        "layout_behavior_hint": "",
        "column_index_hint": None,
        "group_ids": {},
    }


def _default_quality() -> Dict[str, Any]:
    return {
        "confidence": None,
        "ocr_confidence": None,
        "native_confidence": None,
        "classification_confidence": None,
        "bbox_confidence": None,
        "warnings": [],
    }


def _default_relationships(parent_id: Any = None) -> Dict[str, Any]:
    return {
        "parent_id": parent_id,
        "block_id": None,
        "line_id": None,
        "phrase_id": None,
        "span_id": None,
        "children_ids": [],
        "sibling_previous_id": None,
        "sibling_next_id": None,
        "reading_order_previous_id": None,
        "reading_order_next_id": None,
    }


class PerfectExtractionReconstructionAdapter:
    """Builds a reconstruction structure from perfect_extraction."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.assets_dir = self.output_dir / "perfect_assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def _finalize_unit(
        self,
        unit: Dict[str, Any],
        *,
        level: str,
        parent_id: Any = None,
        text: Any = None,
    ) -> Dict[str, Any]:
        unit.setdefault("schema_version", "perfect_legacy_unit.v1")
        unit["level"] = level
        unit.setdefault("id", f"{level}_unknown")
        unit.setdefault("parent_id", parent_id)
        unit.setdefault("source", "perfect_extraction")
        unit.setdefault("source_kind", f"perfect_{level}")
        unit.setdefault("role", "")
        unit.setdefault("object_type", "plain_text" if level in {"word", "span", "phrase"} else f"{level}_object")
        unit.setdefault("object_class", "editorial")
        unit.setdefault("object_subtype", "")
        unit.setdefault("inline_object_type", "")
        unit.setdefault("inline_object_subtype", "")

        clean = _sanitize_reconstructable_text(text if text is not None else unit.get("translated_text") or unit.get("text") or unit.get("texte") or "")
        unit.setdefault("text", clean)
        unit.setdefault("texte", clean)
        unit.setdefault("raw_text", clean)
        unit.setdefault("translated_text", clean)
        unit.setdefault("bbox", None)
        unit.setdefault("bbox_source_unit", "px_150dpi")
        unit.setdefault("bbox_origin", "top_left")
        unit.setdefault("bbox_confidence", None)
        unit.setdefault("reading_order_index", None)
        unit.setdefault("render_order_index", None)
        unit["style"] = _complete_style(unit.get("style"))
        unit["style_attributes"] = _complete_style_attributes(unit.get("style_attributes"))

        layout = _default_layout_attributes()
        layout.update(unit.get("layout_attributes") if isinstance(unit.get("layout_attributes"), dict) else {})
        unit["layout_attributes"] = layout

        source_layout = _default_source_layout_mode()
        source_layout.update(unit.get("source_layout_mode") if isinstance(unit.get("source_layout_mode"), dict) else {})
        unit["source_layout_mode"] = source_layout

        hints = _default_structure_hints()
        hints.update(unit.get("structure_hints") if isinstance(unit.get("structure_hints"), dict) else {})
        if not isinstance(hints.get("group_ids"), dict):
            hints["group_ids"] = {}
        unit["structure_hints"] = hints

        comprehension = {
            "object_type": unit.get("object_type"),
            "object_class": unit.get("object_class"),
            "object_subtype": unit.get("object_subtype"),
            "inline_object_type": unit.get("inline_object_type"),
            "inline_object_subtype": unit.get("inline_object_subtype"),
            "role": unit.get("role"),
            "case_profile": unit.get("case_profile"),
            "style_summary": None,
            "layout": unit.get("layout_attributes"),
            "semantic": None,
            "classification_evidence": [],
        }
        if isinstance(unit.get("object_comprehension"), dict):
            comprehension.update(unit["object_comprehension"])
        unit["object_comprehension"] = comprehension

        detection = {
            "detected": True,
            "source": unit.get("source"),
            "source_kind": unit.get("source_kind"),
            "method": "perfect_extraction_adapter",
            "classifier": "legacy_contract_enrichment",
            "confidence": unit.get("confidence"),
            "evidence": [],
        }
        if isinstance(unit.get("detection"), dict):
            detection.update(unit["detection"])
        unit["detection"] = detection

        quality = _default_quality()
        quality.update(unit.get("quality") if isinstance(unit.get("quality"), dict) else {})
        if unit.get("confidence") is not None:
            quality["confidence"] = unit.get("confidence")
        unit["quality"] = quality

        relationships = _default_relationships(parent_id)
        relationships.update(unit.get("relationships") if isinstance(unit.get("relationships"), dict) else {})
        relationships["parent_id"] = parent_id if relationships.get("parent_id") is None else relationships.get("parent_id")
        unit["relationships"] = relationships

        unit.setdefault("protected_regions", [])
        unit.setdefault("translatable", None)
        unit.setdefault("render_policy", "")
        unit.setdefault("translation_strategy", "")
        unit.setdefault("coverage_required", "")
        unit.setdefault("translation_policy", {})
        unit.setdefault("children_counts", {})
        unit.setdefault("perfect_source", None)
        self._attach_contract_metadata(unit, level=level, text=clean)
        return unit

    def load_or_extract(self, source: str | Path, pages: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        path = Path(source)
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        extractor = PerfectDocumentExtractor(ExtractionConfig(target_dpi=150, enable_ocr=True, enable_special_regions=True))
        return extractor.extract(path, pages=pages)

    def adapt(self, perfect_model: Dict[str, Any], translated_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        source_path = Path((perfect_model.get("source") or {}).get("path") or "")
        pages = []
        for page_model in perfect_model.get("pages") or []:
            pages.append(self._adapt_page(perfect_model, page_model, source_path, translated_payload=translated_payload))
        document_text = "\n".join(str(block.get("text") or "") for page in pages for block in (page.get("blocks") or []) if block.get("text"))
        document_characteristics = self._finalize_unit(
            {
                "id": str((perfect_model.get("source") or {}).get("path") or "document"),
                "source": "perfect_extraction",
                "source_kind": "perfect_document",
                "role": "document",
                "object_type": "document",
                "object_class": "document",
                "text": document_text,
                "raw_text": document_text,
                "translated_text": document_text,
                "bbox": None,
                "render_policy": "document_container",
                "translatable": False,
                "metadata": perfect_model.get("metadata") or {},
                "pdf": perfect_model.get("pdf") or {},
                "resources": perfect_model.get("resources") or {},
                "document_profile": perfect_model.get("document_profile") or {},
                "quality_report": perfect_model.get("quality_report"),
                "children_counts": {
                    "pages": len(pages),
                    "blocks": sum(len(page.get("blocks") or []) for page in pages),
                    "regions": sum(len(page.get("regions") or []) for page in pages),
                    "images": sum(len(page.get("images") or []) for page in pages),
                    "tables": sum(len(page.get("tables") or []) for page in pages),
                    "special_regions": sum(len(page.get("special_regions") or []) for page in pages),
                },
                "perfect_source": perfect_model,
            },
            level="document",
            parent_id=None,
            text=document_text,
        )
        return {
            "schema_version": "perfect_reconstruction_adapter.v1",
            "source_schema_version": perfect_model.get("schema_version"),
            "complete_field_policy": "present_null_when_unknown_zero_for_counts_empty_string_for_text",
            "target_lang": (translated_payload or {}).get("target_lang") or "fr",
            "adapter": {
                "name": "PerfectExtractionReconstructionAdapter",
                "mode": "existing_reconstructor_contract_enriched",
                "does_not_modify_existing_pipeline": True,
                "coordinates": "historical_reconstructor_pixels_150dpi",
            },
            "source": perfect_model.get("source"),
            "document_profile": perfect_model.get("document_profile"),
            "document_characteristics": document_characteristics,
            "pages": pages,
        }

    def _adapt_page(
        self,
        perfect_model: Dict[str, Any],
        page_model: Dict[str, Any],
        source_path: Path,
        translated_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        geometry = dict(page_model.get("geometry") or {})
        width_pt = _safe_float(geometry.get("width"), 595.0)
        height_pt = _safe_float(geometry.get("height"), 842.0)
        width_px = width_pt * PT_TO_RECON_PX
        height_px = height_pt * PT_TO_RECON_PX
        page_number = int(page_model.get("page_number_physical") or page_model.get("page_index", 0) + 1)
        page_id = f"p{page_number:03d}"

        background_path = self._blank_background(page_id, int(round(width_px)), int(round(height_px)))
        immutable_overlays = self._build_immutable_overlays(source_path, page_model, page_id)
        special_regions_px = [self._special_region_to_recon(s) for s in page_model.get("formulas_and_specials") or []]
        special_regions_px = [s for s in special_regions_px if s]
        regions = [
            self._adapt_page_object(region, index, parent_id=page_id, level="region", object_type=str(region.get("type") or "page_region"), object_class="layout")
            for index, region in enumerate(page_model.get("regions") or [])
            if isinstance(region, dict)
        ]
        images = [
            self._adapt_page_object(image, index, parent_id=page_id, level="image", object_type="image_region", object_class="visual")
            for index, image in enumerate(page_model.get("images") or [])
            if isinstance(image, dict)
        ]
        tables = [
            self._adapt_page_object(table, index, parent_id=page_id, level="table", object_type="table_block", object_class="tabular")
            for index, table in enumerate(page_model.get("tables") or [])
            if isinstance(table, dict)
        ]
        drawings = [
            self._adapt_page_object(drawing, index, parent_id=page_id, level="drawing", object_type="drawing_region", object_class="visual")
            for index, drawing in enumerate(page_model.get("drawings") or [])
            if isinstance(drawing, dict)
        ]

        blocks = []
        for index, block in enumerate((page_model.get("text") or {}).get("blocks") or []):
            adapted = self._adapt_block(block, index, special_regions_px, translated_payload=translated_payload)
            if adapted:
                blocks.append(adapted)

        blocks.extend(self._synthetic_overlay_blocks(immutable_overlays, start=len(blocks)))

        page_role = self._infer_page_role(page_model, blocks)
        page_data = {
            "id": page_id,
            "page_index": page_model.get("page_index"),
            "page_number": page_number,
            "page_role": page_role,
            "document_type": (perfect_model.get("document_profile") or {}).get("probable_type"),
            "page_family": self._infer_page_family(page_model),
            "layout_type": self._infer_layout_type(page_model),
            "style_profile": "perfect_extraction_enriched",
            "target_lang": (translated_payload or {}).get("target_lang") or "fr",
            "dimensions": {
                "width": round(width_px, 3),
                "height": round(height_px, 3),
                "page_width": round(width_px, 3),
                "page_height": round(height_px, 3),
                "width_pt": round(width_pt, 3),
                "height_pt": round(height_pt, 3),
                "source_unit": "pt",
                "reconstructor_unit": "px_150dpi",
            },
            "background_path": str(background_path),
            "source_pdf_path": str(source_path),
            "source_page_number": page_number,
            "blocks": blocks,
            "regions": regions,
            "images": images,
            "tables": tables,
            "drawings": drawings,
            "page_objects": regions + images + tables + drawings + special_regions_px,
            "special_regions": special_regions_px,
            "immutable_overlays": immutable_overlays,
            "perfect_extraction_page": page_model,
            "perfect_extraction_quality": page_model.get("quality_report"),
            "reconstruction_constraints": page_model.get("reconstruction_constraints"),
            "layout_descriptor_v3": self._build_layout_descriptor(page_model, blocks, special_regions_px),
            "descriptor_v3_contract": self._build_descriptor_contract(blocks, special_regions_px),
            "children_counts": {
                "blocks": len(blocks),
                "regions": len(regions),
                "images": len(images),
                "tables": len(tables),
                "drawings": len(drawings),
                "special_regions": len(special_regions_px),
                "immutable_overlays": len(immutable_overlays),
            },
        }
        if page_role == "toc":
            page_data["toc"] = {
                "toc_rows": [],
                "perfect_toc_rows_candidate": self._build_toc_rows(blocks),
                "toc_rows_disabled_reason": "perfect_block_rows_need_explicit_validation",
            }
        page_text = "\n".join(str(block.get("text") or "") for block in blocks if block.get("text"))
        page_data.update(
            {
                "source": "perfect_extraction",
                "source_kind": "perfect_page",
                "role": page_role,
                "object_type": "page",
                "object_class": "document",
                "bbox": [0.0, 0.0, round(width_px, 3), round(height_px, 3)],
                "text": page_text,
                "raw_text": page_text,
                "translated_text": page_text,
                "render_policy": "page_container",
                "translatable": False,
                "perfect_source": page_model,
            }
        )
        return self._finalize_unit(page_data, level="page", parent_id=(perfect_model.get("source") or {}).get("path"), text=page_text)

    def _adapt_block(
        self,
        block: Dict[str, Any],
        index: int,
        special_regions_px: List[Dict[str, Any]],
        translated_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        bbox = _bbox_pt_to_px(block.get("bbox"))
        if not bbox:
            return None
        if self._block_should_be_overlay_only(block, bbox, special_regions_px):
            return None
        text = _sanitize_reconstructable_text(block.get("text") or "")
        translated_text = self._translated_text_for_block(block, translated_payload) or text
        style = _style_from_block_summary(block)
        protected = [
            {"id": s.get("id"), "bbox": s.get("bbox"), "class": s.get("class"), "render_policy": "source_overlay"}
            for s in special_regions_px
            if _intersection_ratio(bbox, s.get("bbox")) > 0.05
        ]
        role = str(block.get("role") or "paragraph").lower()
        object_type = self._object_type_for_block(block, protected)
        object_class = self._object_class_for_block(block, protected)
        render_policy = "translated_editorial"
        if protected and object_class in {"formula", "technical"}:
            first_protected_bbox = protected[0].get("bbox")
            if _intersection_ratio(bbox, first_protected_bbox) > 0.50:
                render_policy = "source_overlay"
        if protected and role in {"formula", "code"}:
            render_policy = "source_overlay"

        block_id = block.get("id") or f"perfect_block_{index}"
        lines = [
            self._adapt_line(line, line_index, style, parent_id=block_id, translated_payload=translated_payload)
            for line_index, line in enumerate(block.get("lines") or [])
        ]
        lines = [line for line in lines if line]
        style_attributes = self._style_attributes_for_block(style, lines, text)
        block_words = [
            self._adapt_word(word, word_index, parent_id=block_id, fallback_style=style)
            for word_index, word in enumerate(block.get("words") or [])
            if isinstance(word, dict)
        ]

        adapted = {
            "id": block_id,
            "source": block.get("source") or "perfect_extraction",
            "source_kind": self._source_kind_for_block(block),
            "role": self._role_for_reconstructor(role, object_type),
            "object_type": object_type,
            "object_class": object_class,
            "bbox": bbox,
            "text": text,
            "raw_text": text,
            "translated_text": translated_text,
            "lines": lines,
            "words": block_words,
            "style": style,
            "style_attributes": style_attributes,
            "reading_order_index": block.get("reading_order_index", index),
            "render_order_index": block.get("render_order_index", index),
            "translatable": not bool(render_policy == "source_overlay"),
            "translation_policy": {
                "translatable": not bool(render_policy == "source_overlay"),
                "render_policy": render_policy,
                "preserve_case": (block.get("case_profile") or {}).get("must_preserve_case"),
                "preserve_style": True,
                "preserve_font_size": True,
                "preserve_color": True,
            },
            "render_policy": render_policy,
            "protected_regions": protected,
            "layout_attributes": self._layout_attributes(block),
            "source_layout_mode": self._source_layout_mode_for_block(block, object_type, render_policy),
            "structure_hints": self._structure_hints_for_block(block, object_type, object_class, render_policy),
            "editorial_semantics": self._editorial_semantics(block, object_type, protected),
            "object_comprehension": {
                "object_type": object_type,
                "object_class": object_class,
                "role": block.get("role"),
                "case_profile": block.get("case_profile"),
                "style_summary": block.get("style_summary"),
                "layout": block.get("layout"),
                "semantic": block.get("semantic"),
            },
            "perfect_source": block,
        }
        adapted["semantic_phrases"] = self._semantic_phrases_for_block(adapted)
        adapted["semantic_phrase_count"] = len(adapted["semantic_phrases"])
        adapted["semantic_runs"] = self._semantic_runs_for_block(adapted)
        adapted["semantic_run_count"] = len(adapted["semantic_runs"])
        adapted["semantic_groups"] = self._semantic_groups_for_block(adapted)
        adapted["semantic_group_count"] = len(adapted["semantic_groups"])
        adapted["expressions"] = self._expressions_for_unit(adapted, parent_id=block_id, level="block")
        adapted["expression_count"] = len(adapted["expressions"])
        adapted["children_counts"] = {
            "lines": len(lines),
            "words": len(block_words),
            "semantic_phrases": len(adapted["semantic_phrases"]),
            "semantic_runs": len(adapted["semantic_runs"]),
            "semantic_groups": len(adapted["semantic_groups"]),
            "expressions": len(adapted["expressions"]),
        }
        return self._finalize_unit(adapted, level="block", parent_id=None, text=translated_text or text)

    def _adapt_line(
        self,
        line: Dict[str, Any],
        line_index: int,
        block_style: Dict[str, Any],
        parent_id: Any = None,
        translated_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        bbox = _bbox_pt_to_px(line.get("bbox"))
        if not bbox:
            return None
        text = _sanitize_reconstructable_text(line.get("text") or "")
        line_style = block_style
        line_id = line.get("id") or f"line_{line_index}"
        phrases = []
        for span_index, span in enumerate(line.get("spans") or []):
            span_bbox = _bbox_pt_to_px(span.get("bbox"))
            span_text = span.get("text") or ""
            span_text = _sanitize_reconstructable_text(span_text)
            if not span_bbox and not span_text:
                continue
            span_style = _style_from_perfect(span.get("style"), fallback=block_style)
            phrase_id = span.get("id") or f"{line_id}_phrase_{span_index}"
            legacy_span = self._adapt_span(span, span_index, span_text, span_bbox or bbox, span_style, parent_id=phrase_id)
            phrase = {
                "id": phrase_id,
                "parent_id": line_id,
                "source": line.get("source") or "perfect_extraction",
                "text": span_text,
                "texte": span_text,
                "raw_text": span_text,
                "translated_text": span_text,
                "bbox": span_bbox or bbox,
                "style": span_style,
                "style_attributes": _style_attributes_from_styles([span_style], span_text),
                "source_kind": "perfect_span_phrase",
                "role": "phrase",
                "object_type": "phrase",
                "object_class": "editorial",
                "case_profile": span.get("case_profile"),
                "spans": [legacy_span],
                "words": [],
                "perfect_source": span,
            }
            phrase["expressions"] = self._expressions_for_unit(phrase, parent_id=phrase_id, level="phrase")
            phrase["expression_count"] = len(phrase["expressions"])
            phrase["children_counts"] = {"spans": 1, "words": 0, "expressions": len(phrase["expressions"])}
            self._finalize_unit(phrase, level="phrase", parent_id=line_id, text=span_text)
            phrases.append(phrase)
            line_style = span_style
        line_words = [
            self._adapt_word(word, word_index, parent_id=line_id, fallback_style=line_style)
            for word_index, word in enumerate(line.get("words") or [])
            if isinstance(word, dict)
        ]
        line_payload = {
            "id": line_id,
            "parent_id": parent_id,
            "source": line.get("source") or "perfect_extraction",
            "role": "line",
            "object_type": "line",
            "object_class": "editorial",
            "line_text": text,
            "text": text,
            "texte": text,
            "raw_text": text,
            "translated_text": text,
            "bbox": bbox,
            "style": line_style,
            "style_attributes": _style_attributes_from_styles([phrase.get("style") for phrase in phrases] or [line_style], text),
            "source_kind": "perfect_line",
            "phrases": phrases,
            "words": line_words,
            "case_profile": line.get("case_profile"),
            "line_height": line.get("line_height"),
            "perfect_source": line,
        }
        line_payload["expressions"] = self._expressions_for_unit(line_payload, parent_id=line_id, level="line")
        line_payload["expression_count"] = len(line_payload["expressions"])
        line_payload["children_counts"] = {"phrases": len(phrases), "words": len(line_words), "expressions": len(line_payload["expressions"])}
        return self._finalize_unit(line_payload, level="line", parent_id=parent_id, text=text)

    def _adapt_span(
        self,
        span: Dict[str, Any],
        span_index: int,
        text: str,
        bbox: List[float],
        style: Dict[str, Any],
        parent_id: Any = None,
    ) -> Dict[str, Any]:
        chars = []
        for char in span.get("chars") or []:
            if not isinstance(char, dict):
                continue
            char_text = _sanitize_reconstructable_text(char.get("text") or char.get("c") or "")
            char_style = _style_from_perfect(char.get("style"), fallback=style)
            chars.append(
                self._adapt_char(char, len(chars), parent_id=span.get("id") or f"perfect_span_{span_index}", fallback_style=char_style)
            )
        span_id = span.get("id") or f"perfect_span_{span_index}"
        legacy_span = {
            "id": span_id,
            "parent_id": parent_id,
            "source": span.get("source") or "perfect_extraction",
            "role": "span",
            "object_type": "span",
            "object_class": "editorial",
            "text": text,
            "texte": text,
            "raw_text": text,
            "translated_text": text,
            "bbox": bbox,
            "style": style,
            "style_attributes": _style_attributes_from_styles([style], text),
            "source_kind": "perfect_span",
            "case_profile": span.get("case_profile"),
            "chars": chars,
            "perfect_source": span,
        }
        legacy_span["expressions"] = self._expressions_for_unit(legacy_span, parent_id=span_id, level="span")
        legacy_span["expression_count"] = len(legacy_span["expressions"])
        legacy_span["children_counts"] = {"chars": len(chars), "expressions": len(legacy_span["expressions"])}
        return self._finalize_unit(legacy_span, level="span", parent_id=parent_id, text=text)

    def _adapt_word(
        self,
        word: Dict[str, Any],
        word_index: int,
        *,
        parent_id: Any = None,
        fallback_style: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        text = _sanitize_reconstructable_text(word.get("text") or word.get("word") or "")
        style = _complete_style(word.get("style") if isinstance(word.get("style"), dict) else fallback_style)
        bbox = _bbox_pt_to_px(word.get("bbox"))
        payload = {
            "id": word.get("id") or f"{parent_id or 'unit'}_word_{word_index}",
            "parent_id": parent_id,
            "source": word.get("source") or "perfect_extraction",
            "source_kind": "perfect_word",
            "role": "word",
            "object_type": "word",
            "object_class": "editorial",
            "text": text,
            "texte": text,
            "raw_text": text,
            "translated_text": text,
            "bbox": bbox,
            "style": style,
            "style_attributes": _style_attributes_from_styles([style], text),
            "case_profile": word.get("case_profile"),
            "confidence": word.get("confidence"),
            "reading_order_index": word.get("reading_order_index", word_index),
            "render_order_index": word.get("render_order_index", word_index),
            "perfect_source": word,
            "children_counts": {"chars": 0, "expressions": 0},
        }
        return self._finalize_unit(payload, level="word", parent_id=parent_id, text=text)

    def _adapt_char(
        self,
        char: Dict[str, Any],
        char_index: int,
        *,
        parent_id: Any = None,
        fallback_style: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        text = _sanitize_reconstructable_text(char.get("text") or char.get("c") or "")
        style = _complete_style(char.get("style") if isinstance(char.get("style"), dict) else fallback_style)
        payload = {
            "id": char.get("id") or f"{parent_id or 'unit'}_char_{char_index}",
            "parent_id": parent_id,
            "source": char.get("source") or "perfect_extraction",
            "source_kind": "perfect_char",
            "role": "char",
            "object_type": "char",
            "object_class": "editorial",
            "text": text,
            "texte": text,
            "raw_text": text,
            "translated_text": text,
            "bbox": _bbox_pt_to_px(char.get("bbox")),
            "style": style,
            "style_attributes": _style_attributes_from_styles([style], text),
            "case_profile": char.get("case_profile"),
            "confidence": char.get("confidence"),
            "reading_order_index": char.get("reading_order_index", char_index),
            "render_order_index": char.get("render_order_index", char_index),
            "perfect_source": char,
            "children_counts": {"expressions": 0},
        }
        return self._finalize_unit(payload, level="char", parent_id=parent_id, text=text)

    def _expressions_for_unit(self, unit: Dict[str, Any], *, parent_id: Any, level: str) -> List[Dict[str, Any]]:
        text = _sanitize_reconstructable_text(unit.get("translated_text") or unit.get("text") or unit.get("texte") or "")
        if not text:
            return []
        contract = build_document_object_contract(unit, level=level, text=text)
        segments = list((contract.get("inline_structure") or {}).get("segments") or [])
        expressions = []
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            segment_text = _sanitize_reconstructable_text(segment.get("text") or "")
            if not segment_text:
                continue
            inline_type = str(segment.get("inline_object_type") or "plain_text")
            expression = {
                "id": f"{parent_id}:expr:{index}",
                "parent_id": parent_id,
                "source": unit.get("source") or "perfect_extraction",
                "source_kind": f"perfect_{level}_expression",
                "role": "expression",
                "object_type": "expression",
                "object_class": "technical" if inline_type not in {"", "plain_text"} else "editorial",
                "object_subtype": inline_type,
                "inline_object_type": inline_type,
                "inline_object_subtype": segment.get("inline_object_subtype") or "",
                "text": segment_text,
                "texte": segment_text,
                "raw_text": segment_text,
                "translated_text": segment_text,
                "bbox": unit.get("bbox"),
                "style": unit.get("style"),
                "style_attributes": unit.get("style_attributes"),
                "case_profile": None,
                "confidence": None,
                "reading_order_index": index,
                "render_order_index": index,
                "translation_hint": segment.get("translation_hint") or "",
                "preserve_exact_text": bool(segment.get("preserve_exact_text")),
                "segment_start": segment.get("start"),
                "segment_end": segment.get("end"),
                "perfect_source": segment,
                "children_counts": {"expressions": 0},
            }
            expressions.append(self._finalize_unit(expression, level="expression", parent_id=parent_id, text=segment_text))
        return expressions

    def _semantic_runs_for_block(self, block: Dict[str, Any]) -> List[Dict[str, Any]]:
        runs = []
        block_id = str(block.get("id") or "perfect_block")
        for line in block.get("lines") or []:
            for phrase in line.get("phrases") or []:
                for span in phrase.get("spans") or []:
                    if not isinstance(span, dict):
                        continue
                    text = _sanitize_reconstructable_text(span.get("translated_text") or span.get("text") or span.get("texte") or "")
                    if not text:
                        continue
                    run = {
                        "id": f"{block_id}:semantic_run:{len(runs)}",
                        "parent_id": block_id,
                        "source": span.get("source") or "perfect_extraction",
                        "source_kind": "perfect_semantic_run",
                        "role": "semantic_run",
                        "object_type": "semantic_run",
                        "object_class": span.get("object_class") or "editorial",
                        "text": text,
                        "texte": text,
                        "raw_text": text,
                        "translated_text": text,
                        "bbox": span.get("bbox"),
                        "style": span.get("style"),
                        "style_attributes": span.get("style_attributes"),
                        "spans": [span],
                        "children_counts": {"spans": 1, "expressions": 0},
                    }
                    run["expressions"] = self._expressions_for_unit(run, parent_id=run["id"], level="semantic_run")
                    run["expression_count"] = len(run["expressions"])
                    run["children_counts"]["expressions"] = len(run["expressions"])
                    runs.append(self._finalize_unit(run, level="semantic_run", parent_id=block_id, text=text))
        return runs

    def _semantic_groups_for_block(self, block: Dict[str, Any]) -> List[Dict[str, Any]]:
        groups = []
        block_id = str(block.get("id") or "perfect_block")
        for phrase in block.get("semantic_phrases") or []:
            if not isinstance(phrase, dict):
                continue
            text = _sanitize_reconstructable_text(phrase.get("translated_text") or phrase.get("text") or phrase.get("texte") or "")
            if not text:
                continue
            group = {
                "id": f"{block_id}:semantic_group:{len(groups)}",
                "parent_id": block_id,
                "source": phrase.get("source") or "perfect_extraction",
                "source_kind": "perfect_semantic_group",
                "role": "semantic_group",
                "object_type": "semantic_group",
                "object_class": phrase.get("object_class") or "editorial",
                "text": text,
                "texte": text,
                "raw_text": text,
                "translated_text": text,
                "bbox": phrase.get("bbox"),
                "style": phrase.get("style"),
                "style_attributes": phrase.get("style_attributes"),
                "semantic_phrase_ids": [phrase.get("id")],
                "children_counts": {"semantic_phrases": 1, "expressions": 0},
            }
            group["expressions"] = self._expressions_for_unit(group, parent_id=group["id"], level="semantic_group")
            group["expression_count"] = len(group["expressions"])
            group["children_counts"]["expressions"] = len(group["expressions"])
            groups.append(self._finalize_unit(group, level="semantic_group", parent_id=block_id, text=text))
        return groups

    def _style_attributes_for_block(self, style: Dict[str, Any], lines: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
        styles = [style]
        for line in lines or []:
            if isinstance(line.get("style"), dict):
                styles.append(line["style"])
            for phrase in line.get("phrases") or []:
                if isinstance(phrase.get("style"), dict):
                    styles.append(phrase["style"])
                for span in phrase.get("spans") or []:
                    if isinstance(span.get("style"), dict):
                        styles.append(span["style"])
        return _style_attributes_from_styles(styles, text)

    def _semantic_phrases_for_block(self, block: Dict[str, Any]) -> List[Dict[str, Any]]:
        semantic_phrases: list[dict[str, Any]] = []
        block_id = str(block.get("id") or "perfect_block")
        for line_index, line in enumerate(block.get("lines") or []):
            if not isinstance(line, dict):
                continue
            text = _sanitize_reconstructable_text(line.get("translated_text") or line.get("line_text") or line.get("text") or "")
            bbox = line.get("bbox")
            if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            spans = []
            for phrase in line.get("phrases") or []:
                for span in phrase.get("spans") or []:
                    if isinstance(span, dict):
                        spans.append(span)
            style = line.get("style") if isinstance(line.get("style"), dict) else block.get("style")
            semantic_phrase = {
                "id": f"{block_id}:semantic_phrase:{len(semantic_phrases)}",
                "parent_id": block_id,
                "source": line.get("source") or block.get("source") or "perfect_extraction",
                "sentence_id": f"{block_id}:semantic_phrase:{len(semantic_phrases)}",
                "sentence_index": len(semantic_phrases),
                "line_indices": [line_index],
                "role": "semantic_phrase",
                "object_type": "semantic_phrase",
                "object_class": "editorial",
                "text": text,
                "texte": text,
                "raw_text": text,
                "translated_text": text,
                "bbox": list(bbox),
                "style": style,
                "style_attributes": _style_attributes_from_styles([span.get("style") for span in spans if isinstance(span.get("style"), dict)] or [style], text),
                "spans": spans,
                "source_kind": "perfect_semantic_phrase",
                "render_policy": line.get("render_policy") or block.get("render_policy"),
                "sentence_end_reason": "line_preserve",
            }
            semantic_phrase["expressions"] = self._expressions_for_unit(semantic_phrase, parent_id=semantic_phrase["id"], level="semantic_phrase")
            semantic_phrase["expression_count"] = len(semantic_phrase["expressions"])
            semantic_phrase["children_counts"] = {"spans": len(spans), "expressions": len(semantic_phrase["expressions"])}
            semantic_phrases.append(self._finalize_unit(semantic_phrase, level="semantic_phrase", parent_id=block_id, text=text))
        return semantic_phrases

    def _attach_contract_metadata(self, unit: Dict[str, Any], *, level: str, text: str) -> None:
        """Attach the shared document-object contract without changing legacy decisions."""
        if not isinstance(unit, dict):
            return
        existing_render_policy = unit.get("render_policy")
        existing_translation_policy = dict(unit.get("translation_policy") or {})
        existing_translatable = unit.get("translatable")
        contract = build_document_object_contract(unit, level=level, text=text)
        unit["document_object_contract"] = contract
        unit["inline_structure"] = dict(contract.get("inline_structure") or {})
        unit["has_special_inline_objects"] = bool((contract.get("inline_structure") or {}).get("has_special_inline_objects"))
        reconstruction = dict(contract.get("reconstruction") or {})
        translation = dict(contract.get("translation") or {})
        merged_policy = {
            **existing_translation_policy,
            "translation_strategy": translation.get("strategy"),
            "coverage_required": translation.get("coverage_required"),
            "translation_protection": list(translation.get("protection") or []),
            "reinject_mode": reconstruction.get("reinject_mode"),
            "contract_key": reconstruction.get("contract_key"),
            "document_object_contract_schema": contract.get("schema_version"),
        }
        if "translatable" not in merged_policy:
            merged_policy["translatable"] = bool(translation.get("translatable"))
        if "render_policy" not in merged_policy or not merged_policy.get("render_policy"):
            merged_policy["render_policy"] = existing_render_policy or reconstruction.get("render_policy")
        unit["translation_policy"] = merged_policy
        if existing_render_policy:
            unit["render_policy"] = existing_render_policy
        else:
            unit["render_policy"] = reconstruction.get("render_policy")
        if existing_translatable is not None:
            unit["translatable"] = existing_translatable
        else:
            unit["translatable"] = bool(translation.get("translatable"))

    def _source_kind_for_block(self, block: Dict[str, Any]) -> str:
        source = str(block.get("source") or "").strip().lower()
        if source == "native_pdf":
            return "perfect_native_pdf_block"
        if source == "ocr":
            return "perfect_ocr_block"
        return "perfect_extraction_block"

    def _source_layout_mode_for_block(self, block: Dict[str, Any], object_type: str, render_policy: str) -> Dict[str, Any]:
        role = str(block.get("role") or "").strip().lower()
        layout = dict(block.get("layout") or {})
        if render_policy == "source_overlay":
            line_flow = "fixed_lines"
            render_contract = "source_overlay"
            can_reflow = False
        elif object_type in {"code_block", "formula_block", "table_block", "table_cell"} or role in {"code", "formula"}:
            line_flow = "fixed_lines"
            render_contract = "fixed_slots"
            can_reflow = False
        elif object_type in {"section_heading", "figure_caption", "table_caption", "caption"} or role in {"heading", "title", "caption"}:
            line_flow = "preserve_line_breaks"
            render_contract = "preserve_breaks"
            can_reflow = False
        else:
            line_flow = "inline_reflow"
            render_contract = "paragraph_reflow"
            can_reflow = True
        return {
            "line_flow": line_flow,
            "render_contract": render_contract,
            "source_line_flow": line_flow,
            "source_render_contract": render_contract,
            "can_reflow_within_paragraph": can_reflow,
            "alignment": layout.get("alignment"),
            "perfect_layout": layout,
        }

    def _structure_hints_for_block(
        self,
        block: Dict[str, Any],
        object_type: str,
        object_class: str,
        render_policy: str,
    ) -> Dict[str, Any]:
        role = str(block.get("role") or "").strip().lower()
        hints: dict[str, Any] = {}
        if render_policy == "source_overlay":
            hints.update(
                {
                    "band_role_hint": "visual_preserve_band",
                    "structural_role_hint": "source_overlay",
                    "layout_behavior_hint": "background_only",
                }
            )
        elif object_type == "section_heading" or role in {"heading", "title"}:
            hints.update(
                {
                    "band_role_hint": "title_band",
                    "structural_role_hint": "section_title",
                    "layout_behavior_hint": "anchored_text",
                }
            )
        elif object_type in {"figure_caption", "table_caption", "caption"} or role in {"caption", "figure_caption", "table_caption"}:
            hints.update(
                {
                    "band_role_hint": "caption_band",
                    "structural_role_hint": object_type,
                    "layout_behavior_hint": "anchored_text",
                }
            )
        elif object_type == "code_block" or object_class == "technical":
            hints.update(
                {
                    "band_role_hint": "code_band",
                    "structural_role_hint": "code_block",
                    "layout_behavior_hint": "fixed_lines",
                }
            )
        elif object_type == "formula_block" or object_class == "formula":
            hints.update(
                {
                    "band_role_hint": "formula_band",
                    "structural_role_hint": "formula_block",
                    "layout_behavior_hint": "fixed_visual",
                }
            )
        else:
            hints.update(
                {
                    "band_role_hint": "text_band",
                    "structural_role_hint": "editorial_body",
                    "layout_behavior_hint": "flow_in_band",
                }
            )
        layout = dict(block.get("layout") or {})
        if layout.get("column_index") is not None:
            hints["column_index_hint"] = layout.get("column_index")
        return hints

    def _block_should_be_overlay_only(self, block: Dict[str, Any], bbox_px: List[float], special_regions_px: List[Dict[str, Any]]) -> bool:
        if not special_regions_px:
            return False
        role = str(block.get("role") or "").strip().lower()
        text = str(block.get("text") or "")
        max_overlap = 0.0
        best_class = ""
        for region in special_regions_px:
            overlap = _intersection_ratio(bbox_px, region.get("bbox"))
            if overlap > max_overlap:
                max_overlap = overlap
                best_class = str(region.get("class") or "")
        if max_overlap >= 0.50:
            return True
        if role in {"formula", "code"} and max_overlap >= 0.20:
            return True
        control_chars = len(re.findall(r"[\x02-\x07]", text))
        math_symbols = len(re.findall(r"[∂∑∏∫√∞≈≠≤≥±×÷−∆Ω∗·δµμ^{}=]", text))
        natural_words = len(re.findall(r"[A-Za-zÀ-ÿ]{3,}", text))
        if best_class == "formula" and max_overlap >= 0.12 and (control_chars + math_symbols) >= max(2, natural_words):
            return True
        return False

    def _translated_text_for_block(self, block: Dict[str, Any], translated_payload: Optional[Dict[str, Any]]) -> Optional[str]:
        if not translated_payload:
            return None
        block_id = str(block.get("id") or "")
        by_block = translated_payload.get("blocks") or {}
        if isinstance(by_block, dict) and block_id in by_block:
            item = by_block[block_id]
            if isinstance(item, dict):
                return item.get("translated_text") or item.get("text")
            return str(item)
        return None

    def _object_type_for_block(self, block: Dict[str, Any], protected: List[Dict[str, Any]]) -> str:
        role = str(block.get("role") or "").lower()
        text = _clean_text(block.get("text") or "")
        semantic = dict(block.get("semantic") or {})
        if protected and any(p.get("class") == "formula" for p in protected) and role in {"formula", "code"}:
            return "formula_block"
        if role in {"formula", "equation"}:
            return "formula_block"
        if role == "code":
            return "code_block"
        if role in {"figure_caption", "table_caption", "caption"}:
            return role if role != "caption" else "caption"
        if role in {"heading", "title"}:
            return "section_heading"
        if semantic.get("caption_for") and re.match(r"^(fig(?:ure)?|table|tab\.)\s+\d", text, flags=re.IGNORECASE):
            return "figure_caption"
        if re.search(r"(?:\.\s*){4,}", text) and re.search(r"\d+\s*$", text):
            return "toc_entry"
        if role == "list_item":
            return "paragraph"
        return "text_block"

    def _object_class_for_block(self, block: Dict[str, Any], protected: List[Dict[str, Any]]) -> str:
        role = str(block.get("role") or "").lower()
        if protected and any(p.get("class") == "formula" for p in protected) and role in {"formula", "code"}:
            return "formula"
        if role in {"formula", "equation"}:
            return "formula"
        if role == "code":
            return "technical"
        return "text"

    def _role_for_reconstructor(self, role: str, object_type: str) -> str:
        if object_type == "section_heading":
            return "section_heading"
        if object_type == "formula_block":
            return "formula"
        if object_type == "code_block":
            return "code"
        if object_type in {"figure_caption", "table_caption", "caption"}:
            return object_type
        if object_type == "toc_entry":
            return "toc_entry"
        return role or "paragraph"

    def _layout_attributes(self, block: Dict[str, Any]) -> Dict[str, Any]:
        layout = dict(block.get("layout") or {})
        return {
            "alignment": layout.get("alignment"),
            "column_index": layout.get("column_index"),
            "paragraph_index": layout.get("paragraph_index"),
            "padding_left_px": 0,
            "padding_right_px": 0,
            "padding_top_px": 0,
            "padding_bottom_px": 0,
            "perfect_layout": layout,
        }

    def _editorial_semantics(self, block: Dict[str, Any], object_type: str, protected: List[Dict[str, Any]]) -> Dict[str, Any]:
        role = str(block.get("role") or "").lower()
        return {
            "heading_like": role in {"heading", "title"} or object_type == "section_heading",
            "caption_like": role in {"caption", "figure_caption"},
            "anchored_annotation": bool(protected),
            "flow_class": "technical" if object_type in {"formula_block", "code_block"} else "body",
            "perfect_semantic": block.get("semantic"),
        }

    def _adapt_page_object(
        self,
        source: Dict[str, Any],
        index: int,
        *,
        parent_id: Any,
        level: str,
        object_type: str,
        object_class: str,
    ) -> Dict[str, Any]:
        text = _sanitize_reconstructable_text(source.get("text") or source.get("label") or source.get("content") or "")
        bbox = _bbox_pt_to_px(source.get("bbox") or source.get("visual_bbox"))
        payload = {
            "id": source.get("id") or f"{level}_{index}",
            "parent_id": parent_id,
            "source": source.get("source") or "perfect_extraction",
            "source_kind": f"perfect_{level}",
            "role": source.get("role") or level,
            "object_type": object_type,
            "object_class": object_class,
            "object_subtype": source.get("type") or source.get("class") or "",
            "text": text,
            "texte": text,
            "raw_text": text,
            "translated_text": text,
            "bbox": bbox,
            "style": source.get("style") if isinstance(source.get("style"), dict) else {},
            "style_attributes": _style_attributes_from_styles([source.get("style")] if isinstance(source.get("style"), dict) else [], text),
            "confidence": source.get("confidence"),
            "reading_order_index": source.get("reading_order_index", index),
            "render_order_index": source.get("render_order_index", index),
            "render_policy": source.get("render_policy") or ("source_overlay" if object_class in {"visual", "formula"} else ""),
            "translatable": False if object_class in {"visual", "formula"} else None,
            "translation_policy": {
                "translatable": False if object_class in {"visual", "formula"} else None,
                "render_policy": source.get("render_policy") or ("source_overlay" if object_class in {"visual", "formula"} else ""),
                "preserve_visual": object_class in {"visual", "formula"},
            },
            "protected_regions": [],
            "perfect_source": source,
            "children_counts": {
                "cells": len(source.get("cells") or []) if isinstance(source.get("cells"), list) else 0,
                "rows": len(source.get("rows") or []) if isinstance(source.get("rows"), list) else 0,
                "columns": len(source.get("columns") or []) if isinstance(source.get("columns"), list) else 0,
                "expressions": 0,
            },
        }
        payload["expressions"] = self._expressions_for_unit(payload, parent_id=payload["id"], level=level)
        payload["expression_count"] = len(payload["expressions"])
        payload["children_counts"]["expressions"] = len(payload["expressions"])
        return self._finalize_unit(payload, level=level, parent_id=parent_id, text=text)

    def _special_region_to_recon(self, special: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        bbox = _bbox_pt_to_px(special.get("bbox"))
        if not bbox:
            return None
        object_class = str(special.get("class") or "special")
        object_type = "formula_block" if object_class == "formula" else "code_block" if object_class == "code" else "special_region"
        payload = {
            "id": special.get("id"),
            "source": special.get("source") or "perfect_extraction",
            "source_kind": "perfect_special_region",
            "role": special.get("role") or object_class,
            "class": special.get("class"),
            "object_type": object_type,
            "object_class": "formula" if object_class == "formula" else "technical" if object_class == "code" else "visual",
            "object_subtype": object_class,
            "text": _sanitize_reconstructable_text(special.get("text") or special.get("text_hint") or ""),
            "raw_text": _sanitize_reconstructable_text(special.get("text") or special.get("text_hint") or ""),
            "translated_text": _sanitize_reconstructable_text(special.get("text") or special.get("text_hint") or ""),
            "bbox": bbox,
            "render_policy": special.get("render_policy") or "preserve_source_region",
            "translation_policy": special.get("translation_policy") or "preserve_visual_region",
            "translatable": False,
            "must_preserve_visual": True,
            "must_exclude_from_translation_flow": True,
            "confidence": special.get("confidence"),
            "perfect_source": special,
        }
        if not isinstance(payload["translation_policy"], dict):
            payload["translation_policy"] = {
                "translatable": False,
                "render_policy": payload["render_policy"],
                "preserve_visual": True,
                "source_policy": payload["translation_policy"],
            }
        return self._finalize_unit(payload, level="special_region", parent_id=None, text=payload["text"])

    def _build_immutable_overlays(self, source_path: Path, page_model: Dict[str, Any], page_id: str) -> List[Dict[str, Any]]:
        if not source_path.exists():
            return []
        overlays = []
        overlay_sources = []
        for special in page_model.get("formulas_and_specials") or []:
            overlay_sources.append(("special", special.get("id"), special.get("bbox"), special))
        for image in page_model.get("images") or []:
            overlay_sources.append(("image", image.get("id"), image.get("bbox"), image))
        for table in page_model.get("tables") or []:
            overlay_sources.append(("table", table.get("id"), table.get("bbox"), table))
        if not overlay_sources:
            return []
        try:
            with fitz.open(str(source_path)) as doc:
                page = doc.load_page(int(page_model.get("page_number_physical") or 1) - 1)
                pix = page.get_pixmap(matrix=fitz.Matrix(PT_TO_RECON_PX, PT_TO_RECON_PX), alpha=False)
            full = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except Exception:
            return []
        for index, (kind, source_id, bbox_pt, source) in enumerate(overlay_sources):
            bbox_px = _bbox_pt_to_px(bbox_pt)
            if not bbox_px:
                continue
            crop_box = tuple(int(round(v)) for v in bbox_px)
            crop_box = (
                max(0, crop_box[0]),
                max(0, crop_box[1]),
                min(full.width, crop_box[2]),
                min(full.height, crop_box[3]),
            )
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                continue
            crop = full.crop(crop_box)
            path = self.assets_dir / f"{page_id}_{kind}_{index:03d}.png"
            crop.save(path)
            overlays.append(
                {
                    "id": f"{page_id}_{kind}_{index:03d}",
                    "kind": kind,
                    "source_id": source_id,
                    "path": str(path),
                    "bbox": bbox_px,
                    "render_policy": "source_overlay",
                    "perfect_source": source,
                }
            )
        return overlays

    def _synthetic_overlay_blocks(self, overlays: List[Dict[str, Any]], start: int) -> List[Dict[str, Any]]:
        blocks = []
        for offset, overlay in enumerate(overlays):
            kind = overlay.get("kind") or "overlay"
            object_type = "image_region" if kind == "image" else "formula_block" if kind == "special" else "table_block"
            object_class = "visual" if kind == "image" else "formula" if kind == "special" else "tabular"
            block_id = f"perfect_overlay_{start + offset:03d}"
            payload = {
                    "id": block_id,
                    "source": "perfect_extraction",
                    "source_kind": "perfect_overlay_block",
                    "role": kind,
                    "object_type": object_type,
                    "object_class": object_class,
                    "object_subtype": kind,
                    "bbox": overlay.get("bbox"),
                    "text": "",
                    "raw_text": "",
                    "texte": "",
                    "translated_text": "",
                    "lines": [],
                    "words": [],
                    "semantic_phrases": [],
                    "semantic_phrase_count": 0,
                    "semantic_runs": [],
                    "semantic_run_count": 0,
                    "semantic_groups": [],
                    "semantic_group_count": 0,
                    "expressions": [],
                    "expression_count": 0,
                    "style": {"font": "helv", "size": 1.0, "color": "#000000", "flags": {}},
                    "style_attributes": _style_attributes_from_styles([], ""),
                    "reading_order_index": 900000 + offset,
                    "render_order_index": 900000 + offset,
                    "translatable": False,
                    "render_policy": "source_overlay",
                    "translation_policy": {"translatable": False, "render_policy": "source_overlay", "preserve_visual": True},
                    "protected_regions": [{"id": overlay.get("id"), "bbox": overlay.get("bbox"), "class": kind}],
                    "immutable_overlay": overlay,
                    "perfect_source": overlay.get("perfect_source"),
                    "children_counts": {"lines": 0, "words": 0, "semantic_phrases": 0, "semantic_runs": 0, "semantic_groups": 0, "expressions": 0},
            }
            blocks.append(self._finalize_unit(payload, level="block", parent_id=None, text=""))
        return blocks

    def _blank_background(self, page_id: str, width_px: int, height_px: int) -> Path:
        path = self.assets_dir / f"{page_id}_blank_background.png"
        if not path.exists():
            img = Image.new("RGB", (max(1, width_px), max(1, height_px)), "white")
            ImageDraw.Draw(img).rectangle((0, 0, max(0, width_px - 1), max(0, height_px - 1)), outline=(255, 255, 255))
            img.save(path)
        return path

    def _infer_page_role(self, page_model: Dict[str, Any], blocks: List[Dict[str, Any]]) -> str:
        text = "\n".join(str(b.get("text") or "") for b in blocks[:8]).lower()
        if "contents" in text or "sommaire" in text:
            return "toc"
        return "body"

    def _infer_page_family(self, page_model: Dict[str, Any]) -> str:
        counts = page_model.get("quality_report") or {}
        if counts.get("special_region_count"):
            return "technical_formula"
        if len(page_model.get("images") or []) > 0:
            return "figure_or_image"
        if len(page_model.get("tables") or []) > 0:
            return "table"
        return "text"

    def _infer_layout_type(self, page_model: Dict[str, Any]) -> str:
        columns = (page_model.get("columns") or {}).get("count")
        if columns and int(columns) >= 2:
            return "multi_column"
        return "single_column"

    def _build_toc_rows(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for block in blocks:
            text = _clean_text(block.get("translated_text") or block.get("text"))
            if not text:
                continue
            m = re.search(r"(.*?)(\d{1,4}|[ivxlcdm]+)\s*$", text, re.I | re.S)
            label = _clean_text(m.group(1)) if m else text
            page = _clean_text(m.group(2)) if m else ""
            rows.append(
                {
                    "id": block.get("id"),
                    "label": label,
                    "translated_label": label,
                    "page": page,
                    "bbox": block.get("bbox"),
                    "label_bbox": block.get("bbox"),
                    "page_bbox": block.get("bbox"),
                    "style": block.get("style"),
                    "role": block.get("role"),
                }
            )
        return rows

    def _build_layout_descriptor(self, page_model: Dict[str, Any], blocks: List[Dict[str, Any]], special_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "source": "perfect_extraction_adapter",
            "page_geometry": page_model.get("geometry"),
            "regions": page_model.get("regions"),
            "reading_order": page_model.get("reading_order"),
            "render_order": page_model.get("render_order"),
            "special_regions": special_regions,
            "block_count": len(blocks),
        }

    def _build_descriptor_contract(self, blocks: List[Dict[str, Any]], special_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "source": "perfect_extraction_adapter",
            "constraints": {
                "all_text_must_render": True,
                "allow_overlap": False,
                "preserve_font_size": True,
                "preserve_style": True,
                "preserve_color": True,
                "formula_regions_fixed_as_visual": True,
            },
            "special_regions": special_regions,
            "blocks": [
                {
                    "block_id": block.get("id"),
                    "bbox": block.get("bbox"),
                    "render_policy": block.get("render_policy"),
                    "style": block.get("style"),
                    "protected_regions": block.get("protected_regions"),
                }
                for block in blocks
            ],
        }
