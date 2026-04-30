from __future__ import annotations

import importlib.machinery
import importlib.util
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import fitz

from font_resolver import FontResolver

_LEGACY_MODULE = None


def _load_legacy_module():
    global _LEGACY_MODULE
    if _LEGACY_MODULE is not None:
        return _LEGACY_MODULE
    backup_path = Path(__file__).with_suffix(".py.bak")
    if not backup_path.exists():
        raise FileNotFoundError(f"Legacy reconstructor backup not found: {backup_path}")
    loader = importlib.machinery.SourceFileLoader("_reconstructor_backup", str(backup_path))
    spec = importlib.util.spec_from_loader("_reconstructor_backup", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    _LEGACY_MODULE = module
    return module


@dataclass
class BlockGeometryContext:
    block_id: str
    block_bbox: tuple[float, float, float, float]
    container_bbox: tuple[float, float, float, float]
    padding_left: float
    padding_right: float
    padding_top: float
    padding_bottom: float
    protected_regions: list[dict] = field(default_factory=list)
    background_strategy: str = "preserve"
    background_color: tuple[int, int, int] | None = None
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class LineTemplate:
    line_id: str
    source_line_indices: list[int]
    bbox: tuple[float, float, float, float]
    baseline_y: float
    ascent: float
    descent: float
    left_x: float
    right_x: float
    usable_width: float
    indent_px: float
    first_line_indent_px: float
    alignment: str
    paragraph_id: str
    paragraph_index: int
    line_index_in_paragraph: int
    is_first_paragraph_line: bool
    is_last_paragraph_line_hint: bool


@dataclass
class PlacableUnit:
    unit_id: str
    unit_type: str
    source_kind: str
    parent_unit_id: str | None
    block_unit_id: str
    phrase_unit_id: str
    line_indices: list[int]
    text_source: str
    text_translated: str
    role: str
    inline_class: str | None = None
    group_class: str | None = None
    style: dict[str, Any] = field(default_factory=dict)
    layout_attributes: dict[str, Any] = field(default_factory=dict)
    text_attributes: dict[str, Any] = field(default_factory=dict)
    relative_bbox: tuple[float, float, float, float] | None = None
    anchor_horizontal: str | None = None
    anchor_vertical: str | None = None
    continuation_before: bool = False
    continuation_after: bool = False
    hard_break_before: bool = False
    hard_break_after: bool = False
    keep_with_previous: bool = False
    keep_with_next: bool = False
    reflowable: bool = True
    protected_inline: bool = False
    immutable: bool = False
    render_policy: str = "translated_editorial"
    justification_eligible: bool = True
    break_priority: int = 10
    paragraph_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relation: str
    hard: bool
    weight: float


@dataclass
class PlacementCursor:
    template_index: int = 0
    x: float = 0.0
    baseline_y: float = 0.0


@dataclass
class PlacementResult:
    ops: list["BlockRenderOp"] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BlockRenderOp:
    op_type: str
    block_id: str
    unit_id: str | None
    bbox: tuple[float, float, float, float] | None = None
    text: str | None = None
    style: dict[str, Any] | None = None
    z_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockReconstructionPlan:
    block_id: str
    page_index: int
    block_type: str
    block_role: str
    block_bbox: tuple[float, float, float, float]
    block_bbox_pt: tuple[float, float, float, float] | None
    container_bbox: tuple[float, float, float, float]
    writing_direction: str
    block_progression: str
    alignment: str
    paragraph_alignment: str
    padding_left: float
    padding_right: float
    padding_top: float
    padding_bottom: float
    protected_regions: list[dict]
    background_strategy: str
    background_color: tuple[int, int, int] | None
    line_templates: list[LineTemplate]
    units: list[PlacableUnit]
    graph_edges: list[GraphEdge]
    positioning_policy: dict[str, Any]
    relative_geometry: dict[str, Any]
    editorial_semantics: dict[str, Any]
    editorial_relations: dict[str, Any]
    source_layout_mode: dict[str, Any]
    adaptive_profile: dict[str, Any]
    constraints: dict[str, Any]
    source_block: dict[str, Any] = field(default_factory=dict)
    semantic_profile: "BlockSemanticProfile | None" = field(default=None)


@dataclass
class BlockSemanticProfile:
    block_id: str
    content_class: str
    render_strategy: str
    font_normalization: str
    allow_vertical_expansion: bool
    text_flow_mode: str
    unicode_safe_required: bool
    source_is_translated: bool
    estimated_text_expansion: float
    dominant_fontsize: float
    dominant_is_serif: bool
    dominant_is_bold: bool
    dominant_is_italic: bool
    dominant_is_mono: bool


class DocumentReconstructor:
    def __init__(self):
        self.pixel_to_point = 72.0 / 150.0
        self.hierarchical_reconstruction_mode = True
        self.layout_debug_overlay = os.getenv("LAYOUT_DEBUG_OVERLAY", "1") == "1"
        self._legacy = None
        self._rendered_signatures = set()
        self._debug_page_images: list[Path] = []
        self.font_resolver = FontResolver()
        self._font_objects: dict[str, fitz.Font] = {}
        self._page_font_aliases: dict[tuple, str] = {}
        self._font_truly_supports_cache: dict[tuple, bool] = {}

    # ------------------------------------------------------------------
    # Résolution de polices et mesure de texte (portées depuis le .bak)
    # ------------------------------------------------------------------

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

    def _resolve_text_color(self, style, item):
        try:
            c = (style or {}).get("color", "#000000").lstrip("#")
            if len(c) != 6:
                return (0, 0, 0)
            return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except Exception:
            return (0, 0, 0)

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

    # ------------------------------------------------------------------

    def _ensure_legacy(self):
        # Utiliser __dict__ directement pour éviter de déclencher __getattr__
        # (les tests qui utilisent __new__ sans __init__ n'ont pas _legacy dans __dict__)
        legacy = self.__dict__.get("_legacy")
        if legacy is None:
            legacy_module = _load_legacy_module()
            legacy = legacy_module.DocumentReconstructor()
            # Initialiser les attributs natifs s'ils sont absents (cas __new__ sans __init__)
            if "font_resolver" not in self.__dict__:
                self.__dict__["font_resolver"] = FontResolver()
            if "_font_objects" not in self.__dict__:
                self.__dict__["_font_objects"] = {}
            if "_page_font_aliases" not in self.__dict__:
                self.__dict__["_page_font_aliases"] = {}
            self.__dict__["_legacy"] = legacy
        return legacy

    def _sync_to_legacy(self):
        legacy = self._ensure_legacy()
        for key, value in self.__dict__.items():
            if key == "_legacy":
                continue
            setattr(legacy, key, value)
        return legacy

    def _sync_from_legacy(self, legacy=None):
        legacy = legacy or self._ensure_legacy()
        for key, value in legacy.__dict__.items():
            if key == "_legacy":
                continue
            self.__dict__[key] = value

    def _legacy_call(self, name, *args, **kwargs):
        legacy = self._sync_to_legacy()
        result = getattr(legacy, name)(*args, **kwargs)
        self._sync_from_legacy(legacy)
        return result

    def __getattr__(self, name):
        legacy = self._sync_to_legacy()
        attr = getattr(legacy, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                return self._legacy_call(name, *args, **kwargs)
            return wrapper
        return attr

    # Marqueurs annotés Unicode : cercles numérotés ➊-➓, ①-⑳, ❶-❿, parenthèses ⑴-⑼, etc.
    # Ces symboles sont des unités atomiques non-wrappables, à placer tels quels.
    _ANNOTATION_MARKER_RE = re.compile(
        r'^[\u2460-\u2473\u2474-\u2487\u2488-\u249b'  # ①-⑳ ⑴-⒇ ⒈-⒛
        r'\u24b6-\u24e9'                                # Ⓐ-ⓩ
        r'\u2776-\u2793'                                # ❶-❿ ➀-➉ (Zapf Dingbats)
        r'\u2780-\u2793'                                # ➀-➓
        r'\u278a-\u2793'                                # ➊-➓
        r']+$'
    )

    def _is_annotation_marker(self, text):
        s = (text or "").strip()
        return bool(s) and bool(self._ANNOTATION_MARKER_RE.match(s))

    def _normalize_alignment(self, value):
        value = str(value or "left").strip().lower()
        if value in {"left", "center", "right", "justify"}:
            return value
        return "left"

    def _clean_text_for_render(self, text):
        text = str(text or "")
        text = text.replace("\u00a0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        return text.strip()

    def _format_toc_label_for_render(self, label_type, text):
        text = self._clean_text_for_render(text or "")
        if not text:
            return ""
        if label_type == "part_title":
            text = re.sub(r'^(?:partie|part)\s+', '', text, flags=re.IGNORECASE)
            text = text.upper()
        return text

    def _merge_styles(self, preferred, fallback):
        pref = preferred if isinstance(preferred, dict) else {}
        fb = fallback if isinstance(fallback, dict) else {}
        out = {"font": "helv", "size": 12.0, "color": "#000000", "flags": {}}
        out.update(fb)
        out.update(pref)
        flags = {}
        if isinstance(fb.get("flags"), dict):
            flags.update(fb.get("flags") or {})
        if isinstance(pref.get("flags"), dict):
            flags.update(pref.get("flags") or {})
        out["flags"] = flags
        return out

    def _style_from_block(self, block):
        if isinstance((block or {}).get("style"), dict):
            return dict(block.get("style") or {})
        for line in (block or {}).get("lines") or []:
            if isinstance(line.get("style"), dict):
                return dict(line.get("style") or {})
            for phrase in line.get("phrases") or []:
                if isinstance((phrase or {}).get("style"), dict):
                    return dict((phrase or {}).get("style") or {})
        style_attrs = dict((block or {}).get("style_attributes") or {})
        color = style_attrs.get("color_primary") or "#000000"
        size = style_attrs.get("font_size_pt_median") or style_attrs.get("font_size_pt_max") or 12.0
        font = style_attrs.get("font_family_primary") or "helv"
        flags_any = dict(style_attrs.get("flags_any") or {})
        return {"font": font, "size": float(size), "color": color, "flags": flags_any}

    def _fitz_rect_from_bbox_like(self, bbox_like):
        if isinstance(bbox_like, fitz.Rect):
            return fitz.Rect(bbox_like)
        if isinstance(bbox_like, (list, tuple)) and len(bbox_like) == 4:
            try:
                values = [float(v) for v in bbox_like]
            except Exception:
                return None
            if max(abs(v) for v in values) > 2000:
                return fitz.Rect(values)
            return fitz.Rect([v * self.pixel_to_point for v in values])
        return None

    def _translated_text_from_block(self, block):
        translated = self._clean_text_for_render((block or {}).get("translated_text") or "")
        if translated:
            return translated
        parts = []
        for line in (block or {}).get("lines") or []:
            line_text = self._clean_text_for_render((line or {}).get("translated_text") or "")
            if line_text:
                parts.append(line_text)
                continue
            phrase_parts = []
            for phrase in (line or {}).get("phrases") or []:
                phrase_text = self._clean_text_for_render(
                    (phrase or {}).get("translated_text") or ""
                )
                if phrase_text:
                    phrase_parts.append(phrase_text)
            if phrase_parts:
                parts.append(" ".join(phrase_parts))
        return "\n".join(part for part in parts if part)

    def _has_translated_payload(self, block):
        if self._clean_text_for_render((block or {}).get("translated_text") or ""):
            return True
        for line in (block or {}).get("lines") or []:
            if self._clean_text_for_render((line or {}).get("translated_text") or ""):
                return True
            for phrase in (line or {}).get("phrases") or []:
                if self._clean_text_for_render((phrase or {}).get("translated_text") or ""):
                    return True
        for key in ("semantic_phrases", "semantic_groups", "semantic_runs", "semantic_spans"):
            for unit in (block or {}).get(key) or []:
                if self._clean_text_for_render((unit or {}).get("translated_text") or ""):
                    return True
        return False

    def _page_aux_translated_segments(self, page_data):
        cache_key = "__aux_translated_segments"
        if isinstance(page_data, dict) and isinstance(page_data.get(cache_key), list):
            return page_data.get(cache_key) or []
        segments = []
        seen = set()

        def add_segment(text, bbox, style=None, source_id="", source_text=None, segment_type="aux"):
            text = self._clean_text_for_render(text)
            rect = self._fitz_rect_from_bbox_like(bbox)
            if not text or not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                return
            source_text = self._clean_text_for_render(source_text or "")
            key = (round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2), text, segment_type)
            if key in seen:
                return
            seen.add(key)
            segments.append(
                {
                    "unit_id": source_id or f"aux:{len(segments)}",
                    "text": text,
                    "source_text": source_text,
                    "bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
                    "style": dict(style or {}),
                    "segment_type": str(segment_type or "aux"),
                }
            )

        def walk(node, path="root"):
            if isinstance(node, dict):
                add_segment(
                    node.get("translated_text"),
                    node.get("bbox"),
                    node.get("style"),
                    f"{path}:translated_text",
                    source_text=node.get("text") or node.get("texte"),
                    segment_type="translated_text",
                )
                add_segment(
                    node.get("translated_label"),
                    node.get("label_bbox"),
                    node.get("style"),
                    f"{path}:translated_label",
                    source_text=node.get("label"),
                    segment_type="label",
                )
                add_segment(
                    node.get("translated_page_number") or node.get("page"),
                    node.get("page_bbox"),
                    node.get("page_style"),
                    f"{path}:page",
                    source_text=node.get("page"),
                    segment_type="page",
                )
                for key, value in node.items():
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for idx, item in enumerate(node):
                    walk(item, f"{path}[{idx}]")

        walk(page_data or {})
        if isinstance(page_data, dict):
            page_data[cache_key] = segments
        return segments

    def _line_index_for_bbox(self, block, bbox):
        rect = self._fitz_rect_from_bbox_like(bbox)
        if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
            return 0
        lines = list((block or {}).get("lines") or [])
        best_idx = 0
        best_key = None
        for idx, line in enumerate(lines):
            line_rect = self._fitz_rect_from_bbox_like((line or {}).get("bbox"))
            if not isinstance(line_rect, fitz.Rect) or line_rect.get_area() <= 0:
                continue
            overlap = (rect & line_rect).get_area()
            cy = (line_rect.y0 + line_rect.y1) / 2.0
            ry = (rect.y0 + rect.y1) / 2.0
            key = (-overlap, abs(cy - ry), idx)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx
        return best_idx

    def _external_units_for_block(self, block, page_data, target_lang):
        if not isinstance(page_data, dict):
            return []
        block_rect = self._fitz_rect_from_bbox_like((block or {}).get("bbox"))
        if not isinstance(block_rect, fitz.Rect) or block_rect.get_area() <= 0:
            return []
        block_id = str((block or {}).get("id") or "")
        block_style = self._style_from_block(block)
        block_role = str((block or {}).get("role") or "body").strip().lower()
        candidates = []
        seen = set()
        for idx, seg in enumerate(self._page_aux_translated_segments(page_data)):
            seg_id = str(seg.get("unit_id") or "")
            if ".blocks[" in seg_id:
                continue
            bbox = seg.get("bbox")
            rect = fitz.Rect(bbox) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else self._fitz_rect_from_bbox_like(bbox)
            if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                continue
            if (rect & block_rect).get_area() <= 0.5:
                continue
            text = self._clean_text_for_render(seg.get("text") or "")
            if not text:
                continue
            bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
            key = (text, tuple(round(v, 2) for v in bbox))
            if key in seen:
                continue
            seen.add(key)
            segment_type = str(seg.get("segment_type") or "aux").strip().lower()
            candidates.append(
                {
                    "idx": idx,
                    "seg": seg,
                    "bbox": bbox,
                    "rect": rect,
                    "text": text,
                    "segment_type": segment_type,
                    "style": self._merge_styles(seg.get("style") or {}, block_style),
                    "line_idx": self._line_index_for_bbox(block, rect),
                }
            )
        candidates.sort(key=lambda item: (item["rect"].y0, item["rect"].x0, item["idx"]))
        rows = []
        for item in candidates:
            rect = item["rect"]
            assigned = None
            for row in rows:
                if abs(rect.y0 - row["top_y"]) <= 2.5:
                    assigned = row
                    break
            if assigned is None:
                assigned = {"row_idx": len(rows), "top_y": rect.y0, "bottom_y": rect.y1, "items": []}
                rows.append(assigned)
            else:
                assigned["top_y"] = min(assigned["top_y"], rect.y0)
                assigned["bottom_y"] = max(assigned["bottom_y"], rect.y1)
            assigned["items"].append(item)
        units = []
        for row_idx, row in enumerate(rows):
            row_items = sorted(row["items"], key=lambda item: (item["rect"].x0, item["rect"].y0, item["idx"]))
            for col_idx, item in enumerate(row_items):
                segment_type = item["segment_type"]
                is_first = col_idx == 0
                is_last = col_idx == len(row_items) - 1
                raw_unit = {
                    "unit_id": str(item["seg"].get("unit_id") or f"{block_id}:external:{item['idx']}"),
                    "unit_type": f"external_{segment_type}",
                    "layout_attributes": {
                        "horizontal_anchor": "end" if segment_type == "page" else "start",
                        "vertical_anchor": "top",
                    },
                    "editorial_semantics": {
                        "flow_class": "reference_run" if segment_type == "page" else "anchored_annotation",
                        "reflowable": bool(segment_type != "page"),
                    },
                    "bbox": item["bbox"],
                    "render_policy": "external_flow",
                }
                positioning = self._positioning_preferences_for_unit(
                    raw_unit,
                    text=item["text"],
                    child_units=None,
                    block=block,
                    page_data=page_data,
                    default_anchor_horizontal="end" if segment_type == "page" else "start",
                    default_anchor_vertical="top",
                    default_render_policy="external_flow",
                    default_keep_with_previous=not is_first,
                    default_keep_with_next=not is_last,
                    default_hard_break_before=is_first,
                    default_hard_break_after=is_last,
                    default_reflowable=(segment_type != "page"),
                    default_break_priority=20,
                )
                units.append(
                    PlacableUnit(
                        unit_id=raw_unit["unit_id"],
                        unit_type=raw_unit["unit_type"],
                        source_kind="page_external_segment",
                        parent_unit_id=block_id or None,
                        block_unit_id=block_id,
                        phrase_unit_id=f"{block_id}:external_phrase:{row_idx}",
                        line_indices=[row_idx],
                        text_source=self._clean_text_for_render(item["seg"].get("source_text") or item["text"]),
                        text_translated=item["text"],
                        role=block_role,
                        inline_class="reference" if segment_type == "page" else None,
                        group_class=segment_type if segment_type in {"label", "page"} else None,
                        style=item["style"],
                        layout_attributes=dict(raw_unit["layout_attributes"]),
                        text_attributes={},
                        relative_bbox=item["bbox"],
                        anchor_horizontal=positioning["anchor_horizontal"],
                        anchor_vertical=positioning["anchor_vertical"],
                        continuation_before=not is_first,
                        continuation_after=not is_last,
                        hard_break_before=positioning["hard_break_before"],
                        hard_break_after=positioning["hard_break_after"],
                        keep_with_previous=positioning["keep_with_previous"],
                        keep_with_next=positioning["keep_with_next"],
                        reflowable=positioning["reflowable"],
                        protected_inline=False,
                        immutable=False,
                        render_policy=positioning["render_policy"],
                        justification_eligible=(segment_type != "page") and not positioning["protected_inline"],
                        break_priority=positioning["break_priority"],
                        paragraph_id=f"{block_id}:external:{row_idx}",
                        metadata={"target_lang": target_lang, "segment_type": segment_type, "raw_unit": dict(item["seg"]), **positioning["metadata"]},
                    )
                )
        units.sort(key=lambda unit: ((unit.line_indices or [0])[0], (unit.relative_bbox or (0, 0, 0, 0))[0], unit.unit_id))
        return units

    def _aux_coverage_entries_for_block(self, page_data, block):
        block_rect = self._fitz_rect_from_bbox_like((block or {}).get("bbox"))
        if not isinstance(block_rect, fitz.Rect) or block_rect.get_area() <= 0:
            return []
        entries = []
        for seg in self._page_aux_translated_segments(page_data):
            seg_id = str(seg.get("unit_id") or "")
            if ".blocks[" in seg_id:
                continue
            bbox = seg.get("bbox")
            rect = fitz.Rect(bbox) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else self._fitz_rect_from_bbox_like(bbox)
            if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                continue
            if (rect & block_rect).get_area() <= 0.5:
                continue
            entries.append(
                {
                    "unit_id": seg.get("unit_id"),
                    "text": seg.get("text"),
                    "source_text": seg.get("source_text"),
                    "bbox": seg.get("bbox"),
                    "style": self._merge_styles(seg.get("style") or {}, self._style_from_block(block)),
                    "unit_type": "aux_segment",
                    "line_indices": [],
                    "render_policy": str((block or {}).get("render_policy") or ""),
                    "segment_type": seg.get("segment_type"),
                    "alignment": "right" if str(seg.get("segment_type") or "").strip().lower() == "page" else "left",
                }
            )
        entries.sort(key=lambda item: ((item.get("bbox") or [0, 0, 0, 0])[1], (item.get("bbox") or [0, 0, 0, 0])[0], item.get("unit_id") or ""))
        return entries

    def _source_text_from_block(self, block):
        source = self._clean_text_for_render((block or {}).get("text") or (block or {}).get("raw_text") or "")
        if source:
            return source
        parts = []
        for line in (block or {}).get("lines") or []:
            line_text = self._clean_text_for_render((line or {}).get("line_text") or "")
            if line_text:
                parts.append(line_text)
        return "\n".join(part for part in parts if part)

    def _is_translated_block(self, block):
        translated = self._translated_text_from_block(block)
        source = self._source_text_from_block(block)
        if not translated:
            return False
        if not source:
            return True
        return translated != source

    def _block_text_stats(self, block):
        text = self._clean_text_for_render(self._translated_text_from_block(block) or self._source_text_from_block(block))
        letters = sum(1 for ch in text if ch.isalpha())
        digits = sum(1 for ch in text if ch.isdigit())
        spaces = sum(1 for ch in text if ch.isspace())
        punctuation = max(0, len(text) - letters - digits - spaces)
        tokens = [token for token in re.findall(r"\S+", text) if token]
        alpha_ratio = float(letters) / max(1, letters + digits + punctuation)
        return {
            "text": text,
            "letters": letters,
            "digits": digits,
            "punctuation": punctuation,
            "token_count": len(tokens),
            "alpha_ratio": alpha_ratio,
        }

    def _is_symbolic_visual_block(self, block):
        render_policy = str((block or {}).get("render_policy") or "").strip().lower()
        if render_policy not in {"anchored_text", "fixed_preserve"}:
            return False
        stats = self._block_text_stats(block)
        text = stats["text"]
        if not text or len(text) > 96:
            return False
        if stats["token_count"] <= 4 and stats["alpha_ratio"] < 0.5:
            return True
        if stats["letters"] <= 6 and (stats["digits"] + stats["punctuation"]) >= stats["letters"]:
            return True
        tokens = [token for token in re.findall(r"\S+", text) if token]
        if not tokens:
            return False
        symbolic = 0
        for token in tokens:
            alpha = sum(1 for ch in token if ch.isalpha())
            digit = sum(1 for ch in token if ch.isdigit())
            punct = len(token) - alpha - digit
            if alpha <= 1 and (digit + punct) >= alpha:
                symbolic += 1
        return symbolic >= max(2, len(tokens) // 2)

    def _classify_block_for_reconstruction(self, block, page_data=None):
        role = str((block or {}).get("role") or "").strip().lower()
        unit_type = str((block or {}).get("unit_type") or "").strip().lower()
        descriptor_group_ids = dict((block or {}).get("descriptor_group_ids") or {})
        editorial_semantics = dict((block or {}).get("editorial_semantics") or {})
        flow_class = str(editorial_semantics.get("flow_class") or "").strip().lower()
        if (
            self._block_is_immutable_programming_code(block)
            or self._is_symbolic_visual_block(block)
            or self._block_looks_technical_structured(block)
        ):
            return "code"
        if role == "figure_caption" or bool(editorial_semantics.get("caption_like")):
            return "caption"
        if role in {"title", "section_heading", "header", "footer"} or bool(editorial_semantics.get("heading_like")):
            return "heading"
        if bool(editorial_semantics.get("anchored_annotation")):
            return "annotation"
        if (
            unit_type.startswith("table_")
            or flow_class in {"table", "tabular"}
            or str(descriptor_group_ids.get("cell_id") or "").strip()
            or str(descriptor_group_ids.get("table_row_group_id") or "").strip()
            or "table_" in str((block or {}).get("descriptor_structural_role") or "").strip().lower()
        ):
            return "table"
        if role == "body" or flow_class == "editorial_body" or bool(editorial_semantics.get("reflowable", True)):
            return "editorial"
        return "mixed"

    def compute_block_semantic_profile(self, block, page_data, translated_text=""):
        if not isinstance(block, dict):
            block = {}
        block_id = str(block.get("id") or "")
        # Etape A - recuperer les metadonnees IA existantes
        role = str(block.get("role") or "").strip().lower()
        editorial_semantics = dict(block.get("editorial_semantics") or {})
        flow_class = str(editorial_semantics.get("flow_class") or "").strip().lower()
        render_policy = str(block.get("render_policy") or "").strip().lower()
        target_lang = str((page_data or {}).get("target_lang") or "").strip().lower() if page_data else ""
        # Etape B - analyser les patterns de bbox des lignes
        lines = list(block.get("lines") or [])
        line_count = len(lines)
        block_bbox = block.get("bbox") or [0, 0, 0, 0]
        try:
            block_width = max(1.0, float(block_bbox[2]) - float(block_bbox[0]))
            block_height = max(1.0, float(block_bbox[3]) - float(block_bbox[1]))
        except Exception:
            block_width = 1.0
            block_height = 1.0
        block_aspect_ratio = block_height / block_width
        words_per_line = []
        for line in lines:
            line_text = self._clean_text_for_render(
                (line or {}).get("translated_text") or (line or {}).get("line_text") or ""
            )
            words = [w for w in re.findall(r"\S+", line_text) if w]
            words_per_line.append(len(words))
        avg_words_per_line = float(sum(words_per_line)) / max(1, len(words_per_line)) if words_per_line else 0.0
        short_line_ratio = float(sum(1 for w in words_per_line if w <= 5)) / max(1, len(words_per_line)) if words_per_line else 0.0
        is_column_shape = block_aspect_ratio > 1.5 and block_width < 200.0
        # Etape C - analyser le contenu textuel
        all_text = self._clean_text_for_render(
            self._translated_text_from_block(block) or self._source_text_from_block(block)
        )
        has_math_chars = bool(re.search(
            r'[Ͱ-Ͽ∀-⟿°-¿]',
            all_text
        ))
        has_code_pattern = bool(re.search(
            r'(?:\([^)]*\(|\[[^\]]*\[|->|:=|\w\s*=\s*\w)',
            all_text
        ))
        all_uppercase_lines = bool(lines) and all(
            self._clean_text_for_render(
                (line or {}).get("translated_text") or (line or {}).get("line_text") or ""
            ).isupper() or not self._clean_text_for_render(
                (line or {}).get("translated_text") or (line or {}).get("line_text") or ""
            )
            for line in lines
        )
        if len(words_per_line) >= 2 and avg_words_per_line <= 6:
            avg_wpl = avg_words_per_line
            repeated_structure = all(
                abs(w - avg_wpl) <= avg_wpl * 0.2 for w in words_per_line if w > 0
            ) and avg_wpl > 0
        else:
            repeated_structure = False
        # Etape D - analyser le style dominant
        fontsizes = []
        style_flags_list = []
        for line in lines:
            for phrase in (line or {}).get("phrases") or []:
                style = (phrase or {}).get("style") or {}
                if isinstance(style, dict):
                    try:
                        fontsizes.append(float(style.get("size") or 0.0))
                    except Exception:
                        pass
                    flags = style.get("flags") or {}
                    if isinstance(flags, dict):
                        style_flags_list.append(flags)
        if not fontsizes:
            block_style = self._style_from_block(block)
            try:
                fontsizes = [float(block_style.get("size") or 12.0)]
            except Exception:
                fontsizes = [12.0]
            flags = block_style.get("flags") or {}
            if isinstance(flags, dict):
                style_flags_list = [flags]
        sorted_fs = sorted(fontsizes)
        dominant_fontsize = sorted_fs[len(sorted_fs) // 2] if sorted_fs else 12.0
        dominant_flags = style_flags_list[len(style_flags_list) // 2] if style_flags_list else {}
        dominant_is_serif = bool(dominant_flags.get("serif"))
        dominant_is_bold = bool(dominant_flags.get("bold"))
        dominant_is_italic = bool(dominant_flags.get("italic"))
        dominant_is_mono = bool(dominant_flags.get("monospace"))
        # Etape E - estimer l'expansion du texte traduit
        source_text = self._clean_text_for_render(self._source_text_from_block(block))
        translated_text_clean = self._clean_text_for_render(translated_text or "")
        if translated_text_clean and source_text:
            estimated_text_expansion = len(translated_text_clean) / max(1, len(source_text))
        else:
            estimated_text_expansion = 1.15
        # Etape F - decider content_class et render_strategy
        _code_checker = None
        for _cls in type(self).__mro__:
            if '_block_is_immutable_programming_code' in _cls.__dict__:
                _code_checker = _cls.__dict__['_block_is_immutable_programming_code']
                break
        is_code_block = bool(_code_checker(self, block)) if _code_checker is not None else False
        if role in {"code", "code_block"} or (has_code_pattern and render_policy == "fixed_preserve") or is_code_block:
            content_class = "code"
            render_strategy = "code_preserve"
        elif has_math_chars and (role in {"formula", "equation"} or flow_class == "symbolic"):
            content_class = "formula"
            render_strategy = "bitmap_preserve"
        elif role in {"heading", "title", "section_title", "chapter_title"}:
            content_class = "heading"
            render_strategy = "heading_reflow"
        elif role in {"figure_caption", "table_caption", "caption"}:
            content_class = "caption"
            render_strategy = "prose_reflow"
        elif role in {"body", "paragraph", "text", "list_item"} or flow_class in {"prose", "editorial"}:
            # Si des semantic_groups existent ET l'heuristique label s'applique : label_stack
            _has_semantic_groups = bool((block or {}).get("semantic_groups"))
            _label_heuristic = is_column_shape or (short_line_ratio >= 0.8 and avg_words_per_line <= 4) or repeated_structure
            if _has_semantic_groups and _label_heuristic:
                content_class = "label"
                render_strategy = "label_stack"
            else:
                content_class = "prose"
                render_strategy = "prose_reflow"
        elif is_column_shape or (short_line_ratio >= 0.8 and avg_words_per_line <= 4) or repeated_structure:
            content_class = "label"
            render_strategy = "label_stack"
        elif line_count >= 2 and avg_words_per_line >= 6:
            content_class = "prose"
            render_strategy = "prose_reflow"
        else:
            content_class = "label"
            render_strategy = "label_stack"
        # Autres champs derives
        if render_strategy == "prose_reflow":
            font_normalization = "fit_to_bbox"
        elif render_strategy == "label_stack":
            font_normalization = "block_median"
        else:
            font_normalization = "span_original"
        allow_vertical_expansion = (
            render_strategy in {"prose_reflow", "heading_reflow"}
            and estimated_text_expansion > 1.0
        )
        if render_strategy == "prose_reflow":
            text_flow_mode = "continuous"
        elif render_strategy == "label_stack":
            text_flow_mode = "line_by_line"
        else:
            text_flow_mode = "atomic"
        unicode_safe_required = (
            any(ord(ch) > 127 for ch in (translated_text_clean or ""))
            or target_lang in {"fr", "es", "de", "it", "pt"}
        )
        source_is_translated = self._has_translated_payload(block)
        return BlockSemanticProfile(
            block_id=block_id,
            content_class=content_class,
            render_strategy=render_strategy,
            font_normalization=font_normalization,
            allow_vertical_expansion=allow_vertical_expansion,
            text_flow_mode=text_flow_mode,
            unicode_safe_required=unicode_safe_required,
            source_is_translated=source_is_translated,
            estimated_text_expansion=estimated_text_expansion,
            dominant_fontsize=dominant_fontsize,
            dominant_is_serif=dominant_is_serif,
            dominant_is_bold=dominant_is_bold,
            dominant_is_italic=dominant_is_italic,
            dominant_is_mono=dominant_is_mono,
        )

    # ------------------------------------------------------------------
    # Securite des polices unicode + helpers systeme
    # ------------------------------------------------------------------

    _SYSTEM_FONT_MAP = {
        # (serif, bold, italic, mono) -> ordered candidates
        (False, False, False, True):  [
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ],
        (False, True, False, True):   [
            "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        ],
        (True, False, False, False):  [
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ],
        (True, True, False, False):   [
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ],
        (True, False, True, False):   [
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
        ],
        (True, True, True, False):    [
            "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-BoldItalic.ttf",
        ],
        (False, False, False, False): [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
        (False, True, False, False):  [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ],
        (False, False, True, False):  [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Oblique.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        ],
        (False, True, True, False):   [
            "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
        ],
    }

    def _get_system_unicode_font(self, is_serif=False, is_bold=False, is_italic=False, is_mono=False):
        key = (bool(is_serif), bool(is_bold), bool(is_italic), bool(is_mono))
        candidates = self._SYSTEM_FONT_MAP.get(key) or self._SYSTEM_FONT_MAP.get((False, False, False, False), [])
        for path in candidates:
            if os.path.isfile(path):
                return path
        # Fallback absolu : parcourir tous les styles
        for paths in self._SYSTEM_FONT_MAP.values():
            for path in paths:
                if os.path.isfile(path):
                    return path
        return None

    def _font_truly_supports_text(self, fontfile, text):
        if not fontfile or not text:
            return True
        probe_chars = list(set(ch for ch in text if ord(ch) > 127 and not ch.isspace()))
        if not probe_chars:
            return True
        cache_key = (fontfile, "".join(sorted(probe_chars)))
        cached = self._font_truly_supports_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            font = fitz.Font(fontfile=fontfile)
            # Test 1 : has_glyph de base
            for ch in probe_chars:
                if not font.has_glyph(ord(ch)):
                    self._font_truly_supports_cache[cache_key] = False
                    return False
            # Test 2 : rendu reel dans un doc temporaire pour detecter les faux positifs CFF
            test_chars = probe_chars[:4]
            test_text = "".join(test_chars)
            try:
                tmp_doc = fitz.open()
                tmp_page = tmp_doc.new_page(width=200, height=50)
                alias = f"TMPCHK{abs(hash(fontfile)) % 9999}"
                tmp_page.insert_font(fontname=alias, fontfile=fontfile)
                tmp_page.insert_text((10, 35), test_text, fontname=alias, fontsize=14)
                rendered = tmp_page.get_text("text").strip()
                tmp_doc.close()
                ok = bool(rendered) and any(ch in rendered for ch in test_chars)
            except Exception:
                # Si le test de rendu echoue, se fier a has_glyph
                ok = True
            self._font_truly_supports_cache[cache_key] = ok
            return ok
        except Exception:
            self._font_truly_supports_cache[cache_key] = False
            return False

    def _resolve_unicode_safe_font(self, page, plan, text):
        profile = getattr(plan, "semantic_profile", None)
        base_style = self._style_from_block(plan.source_block or {})
        probe_chars = [ch for ch in (text or "") if ord(ch) > 127]
        if not probe_chars:
            _, fontfile, builtin, fontname = self._resolve_style_font(page, base_style, text=text)
            return fontfile, fontname
        # Tentative normale
        _, fontfile, builtin, fontname = self._resolve_style_font(page, base_style, text=text)
        # Verifier que la police supporte VRAIMENT les chars (anti faux-positif CFF)
        if fontfile and self._font_truly_supports_text(fontfile, text):
            return fontfile, fontname
        # Fallback explicite vers police systeme unicode-safe par style dominant
        is_serif = bool(profile.dominant_is_serif) if profile else bool((base_style.get("flags") or {}).get("serif"))
        is_bold = bool(profile.dominant_is_bold) if profile else bool((base_style.get("flags") or {}).get("bold"))
        is_italic = bool(profile.dominant_is_italic) if profile else bool((base_style.get("flags") or {}).get("italic"))
        is_mono = bool(profile.dominant_is_mono) if profile else bool((base_style.get("flags") or {}).get("monospace"))
        fallback_file = self._get_system_unicode_font(is_serif, is_bold, is_italic, is_mono)
        if fallback_file:
            fallback_name = self._resolve_page_fontname(page, fallback_file, None)
            return fallback_file, fallback_name
        # Dernier recours : helv (supporte Latin-1 dont tous les accents francais)
        return None, "helv"

    def _document_adaptive_profile(self, page_data):
        layout_type = str((page_data or {}).get("layout_type") or "").strip().lower()
        document_type = str((page_data or {}).get("document_type") or "").strip().lower()
        page_family = str((page_data or {}).get("page_family") or "").strip().lower()
        style_profile = str((page_data or {}).get("style_profile") or "").strip().lower()
        if layout_type in {"double_column", "reference_page"} or document_type in {"scientific_paper"} or style_profile == "academic_dense":
            return {
                "document_profile": "academic_dense",
                "dense_layout": True,
                "technical_bias": bool(style_profile in {"academic_dense", "tabular_structured"}),
                "visual_bias": False,
            }
        if layout_type in {"annotated_page", "image_dominant"} or page_family in {"illustrated_label_page", "chart_label_page"} or style_profile in {"editorial_visual", "marketing_visual"}:
            return {
                "document_profile": "visual_labels",
                "dense_layout": False,
                "technical_bias": False,
                "visual_bias": True,
            }
        if layout_type == "table_dominant" or style_profile == "tabular_structured":
            return {
                "document_profile": "technical_structured",
                "dense_layout": True,
                "technical_bias": True,
                "visual_bias": False,
            }
        return {
            "document_profile": "editorial_standard",
            "dense_layout": False,
            "technical_bias": False,
            "visual_bias": False,
        }

    def _page_adaptive_profile(self, page_data):
        doc_profile = self._document_adaptive_profile(page_data)
        layout_type = str((page_data or {}).get("layout_type") or "").strip().lower()
        page_family = str((page_data or {}).get("page_family") or "").strip().lower()
        style_profile = str((page_data or {}).get("style_profile") or "").strip().lower()
        page_role = str((page_data or {}).get("page_role") or "body").strip().lower()
        profile_name = doc_profile["document_profile"]
        if page_role == "toc":
            profile_name = "toc"
        elif layout_type == "table_dominant" or style_profile == "tabular_structured":
            profile_name = "technical_structured"
        elif layout_type in {"annotated_page", "image_dominant"} or page_family in {"illustrated_label_page", "chart_label_page"}:
            profile_name = "visual_labels"
        elif layout_type in {"double_column", "reference_page"}:
            profile_name = "academic_dense"

        if profile_name == "academic_dense":
            return {
                **doc_profile,
                "page_profile": profile_name,
                "fallback_scales": (1.0, 0.94, 0.88, 0.82, 0.76, 0.7, 0.64, 0.58),
                "editorial_scales": (1.0, 0.96, 0.92, 0.88, 0.84, 0.8, 0.76, 0.72, 0.68),
                "line_spacing_factor": 0.96,
                "allow_aggressive_reflow": False,
                "prefer_bbox_anchor": False,
                "prefer_atomic_short_units": True,
            }
        if profile_name == "technical_structured":
            return {
                **doc_profile,
                "page_profile": profile_name,
                "fallback_scales": (1.0, 0.92, 0.84, 0.76, 0.68, 0.6),
                "editorial_scales": (1.0, 0.94, 0.88, 0.82, 0.76),
                "line_spacing_factor": 0.93,
                "allow_aggressive_reflow": False,
                "prefer_bbox_anchor": True,
                "prefer_atomic_short_units": True,
            }
        if profile_name == "visual_labels":
            return {
                **doc_profile,
                "page_profile": profile_name,
                "fallback_scales": (1.0, 0.95, 0.9, 0.85, 0.8),
                "editorial_scales": (1.0, 0.97, 0.94, 0.91, 0.88, 0.85),
                "line_spacing_factor": 1.0,
                "allow_aggressive_reflow": False,
                "prefer_bbox_anchor": True,
                "prefer_atomic_short_units": True,
            }
        if profile_name == "toc":
            return {
                **doc_profile,
                "page_profile": profile_name,
                "fallback_scales": (1.0, 0.95, 0.9, 0.85),
                "editorial_scales": (1.0, 0.97, 0.94, 0.91),
                "line_spacing_factor": 0.98,
                "allow_aggressive_reflow": False,
                "prefer_bbox_anchor": True,
                "prefer_atomic_short_units": True,
            }
        return {
            **doc_profile,
            "page_profile": "editorial_standard",
            "fallback_scales": (1.0, 0.9, 0.8, 0.7, 0.6),
            "editorial_scales": (1.0, 0.96, 0.92, 0.88, 0.84, 0.8, 0.76, 0.72),
            "line_spacing_factor": 1.0,
            "allow_aggressive_reflow": True,
            "prefer_bbox_anchor": False,
            "prefer_atomic_short_units": False,
        }

    def _block_adaptive_profile(self, block, page_data=None, block_type=None):
        page_profile = self._page_adaptive_profile(page_data)
        block_type = block_type or self._classify_block_for_reconstruction(block, page_data)
        unit_type = str((block or {}).get("unit_type") or "").strip().lower()
        render_policy = str((block or {}).get("render_policy") or "").strip().lower()
        editorial_semantics = dict((block or {}).get("editorial_semantics") or {})
        flow_class = str(editorial_semantics.get("flow_class") or "").strip().lower()
        block_name = "editorial_flow"
        if block_type == "code" or self._block_looks_technical_structured(block):
            block_name = "technical_structured"
        elif block_type == "table":
            block_name = "tabular_dense"
        elif block_type in {"annotation", "caption"} or flow_class == "anchored_annotation":
            block_name = "visual_label_cluster"
        elif unit_type in {"short_label", "chart_label", "diagram_label"} or render_policy in {"anchored_text", "fixed_preserve"}:
            block_name = "anchored_microcopy"
        elif page_profile.get("page_profile") == "academic_dense":
            block_name = "dense_editorial"

        profile = {
            **page_profile,
            "block_profile": block_name,
            "block_type": block_type,
            "force_whiteout": bool(block_type in {"editorial", "heading", "caption", "annotation", "table"}),
            "line_spacing_factor": float(page_profile.get("line_spacing_factor") or 1.0),
            "prefer_bbox_anchor": bool(page_profile.get("prefer_bbox_anchor")),
            "allow_aggressive_reflow": bool(page_profile.get("allow_aggressive_reflow")),
            "allow_linewise_fallback": True,
            "presence_fallback_requires_progress": True,
        }
        if block_name in {"technical_structured", "tabular_dense"}:
            profile["line_spacing_factor"] = min(profile["line_spacing_factor"], 0.92)
            profile["prefer_bbox_anchor"] = True
            profile["allow_aggressive_reflow"] = False
        elif block_name in {"visual_label_cluster", "anchored_microcopy"}:
            profile["prefer_bbox_anchor"] = True
            profile["allow_aggressive_reflow"] = False
            profile["line_spacing_factor"] = max(profile["line_spacing_factor"], 0.98)
        elif block_name == "dense_editorial":
            profile["line_spacing_factor"] = min(profile["line_spacing_factor"], 0.95)
            profile["allow_aggressive_reflow"] = False
        return profile

    def _unit_adaptive_profile(self, raw_unit, *, text="", child_units=None, block=None, page_data=None):
        block_profile = self._block_adaptive_profile(block or {}, page_data=page_data, block_type=self._classify_block_for_reconstruction(block or {}, page_data))
        expression_semantics = dict((raw_unit or {}).get("expression_semantics") or {})
        inline_class = str(expression_semantics.get("inline_class") or "").strip().lower()
        unit_type = str((raw_unit or {}).get("unit_type") or "").strip().lower()
        editorial_semantics = dict((raw_unit or {}).get("editorial_semantics") or {})
        flow_class = str(editorial_semantics.get("flow_class") or "").strip().lower()
        text_clean = self._clean_text_for_render(text or "")
        short_text = bool(text_clean and len(text_clean) <= 64)
        child_summary = self._protected_fragment_summary(child_units)
        unit_profile = "editorial_phrase"
        if inline_class in {"formula", "reference", "code"} or child_summary["has_immutable"]:
            unit_profile = "protected_inline"
        elif unit_type in {"short_label", "chart_label", "diagram_label", "formula_label"} or flow_class == "anchored_annotation":
            unit_profile = "anchored_label"
        elif block_profile.get("block_profile") in {"technical_structured", "tabular_dense"}:
            unit_profile = "technical_inline_cluster" if short_text else "dense_editorial_phrase"
        elif block_profile.get("block_profile") in {"visual_label_cluster", "anchored_microcopy"}:
            unit_profile = "anchored_label"
        elif block_profile.get("page_profile") == "academic_dense":
            unit_profile = "dense_editorial_phrase"
        return {
            **block_profile,
            "unit_profile": unit_profile,
            "short_text": short_text,
            "inline_class": inline_class,
            "has_protected_fragments": bool(child_summary["has_protected"]),
            "has_immutable_fragments": bool(child_summary["has_immutable"]),
        }

    def _build_page_reconstruction_context(self, page_data, target_lang):
        adaptive_profile = self._page_adaptive_profile(page_data)
        return {
            "target_lang": target_lang,
            "writing_direction": "right_to_left" if target_lang in {"ar", "he", "fa"} else "left_to_right",
            "adaptive_profile": adaptive_profile,
        }

    def _iter_renderable_blocks(self, page_data):
        blocks = list((page_data or {}).get("blocks") or [])
        def sort_key(block):
            bbox = block.get("bbox") or [0, 0, 0, 0]
            return (
                int(block.get("reading_order_index") or 10**9),
                float(bbox[1] if len(bbox) == 4 else 0.0),
                float(bbox[0] if len(bbox) == 4 else 0.0),
                str(block.get("id") or ""),
            )
        return sorted([block for block in blocks if isinstance(block, dict)], key=sort_key)

    def _build_block_geometry_context(self, page, page_data, block):
        block_rect = self._fitz_rect_from_bbox_like((block or {}).get("bbox"))
        if not isinstance(block_rect, fitz.Rect) or block_rect.get_area() <= 0:
            block_rect = fitz.Rect(page.rect)
        layout_attrs = dict((block or {}).get("layout_attributes") or {})
        padding_left = float(layout_attrs.get("padding_left_px", 0.0) or 0.0) * self.pixel_to_point
        padding_right = float(layout_attrs.get("padding_right_px", 0.0) or 0.0) * self.pixel_to_point
        padding_top = float(layout_attrs.get("padding_top_px", 0.0) or 0.0) * self.pixel_to_point
        padding_bottom = float(layout_attrs.get("padding_bottom_px", 0.0) or 0.0) * self.pixel_to_point
        background_strategy = "preserve"
        block_type = self._classify_block_for_reconstruction(block, page_data)
        adaptive_profile = self._block_adaptive_profile(block, page_data=page_data, block_type=block_type)
        has_clean_background = bool(self._clean_page_background_path(page_data))
        if (
            self._is_translated_block(block)
            and bool(adaptive_profile.get("force_whiteout"))
            and not has_clean_background
        ):
            background_strategy = "whiteout"
        protected_regions = list((block or {}).get("protected_regions") or [])
        return BlockGeometryContext(
            block_id=str((block or {}).get("id") or ""),
            block_bbox=(block_rect.x0, block_rect.y0, block_rect.x1, block_rect.y1),
            container_bbox=(block_rect.x0, block_rect.y0, block_rect.x1, block_rect.y1),
            padding_left=padding_left,
            padding_right=padding_right,
            padding_top=padding_top,
            padding_bottom=padding_bottom,
            protected_regions=protected_regions,
            background_strategy=background_strategy,
            background_color=None,
            constraints={
                "page_role": str((page_data or {}).get("page_role") or "").strip().lower(),
                "adaptive_profile": adaptive_profile,
            },
        )

    def _build_line_templates(self, block, geometry_ctx):
        block_rect = fitz.Rect(geometry_ctx.block_bbox)
        adaptive_profile = dict((geometry_ctx.constraints or {}).get("adaptive_profile") or {})
        line_spacing_factor = max(0.86, min(1.08, float(adaptive_profile.get("line_spacing_factor") or 1.0)))
        lines = list((block or {}).get("lines") or [])
        templates = []
        paragraph_index = 0
        line_index_in_paragraph = 0
        alignment = self._normalize_alignment((block or {}).get("alignment") or "left")
        inner_left = block_rect.x0 + geometry_ctx.padding_left
        inner_right = max(inner_left + 8.0, block_rect.x1 - geometry_ctx.padding_right)
        inner_top = block_rect.y0 + geometry_ctx.padding_top
        inner_bottom = max(inner_top + 8.0, block_rect.y1 - geometry_ctx.padding_bottom)
        line_heights = []
        previous_bottom = inner_top
        for idx, line in enumerate(lines):
            line_rect = self._fitz_rect_from_bbox_like((line or {}).get("bbox"))
            if not isinstance(line_rect, fitz.Rect) or line_rect.get_area() <= 0:
                line_rect = fitz.Rect(block_rect.x0, block_rect.y0 + idx * 12.0, block_rect.x1, block_rect.y0 + (idx + 1) * 12.0)
            line_rect = fitz.Rect(inner_left, max(inner_top, line_rect.y0), inner_right, min(inner_bottom, line_rect.y1))
            if line_rect.height <= 0:
                fallback_top = min(inner_bottom - 6.0, inner_top + idx * 12.0)
                line_rect = fitz.Rect(inner_left, fallback_top, inner_right, min(inner_bottom, fallback_top + 12.0))
            hard_break_before = bool((line or {}).get("hard_break_before"))
            if idx > 0 and hard_break_before:
                paragraph_index += 1
                line_index_in_paragraph = 0
            indent_px = float((line or {}).get("indent_px", 0.0) or 0.0) * self.pixel_to_point
            line_h = max(6.0, line_rect.height * line_spacing_factor)
            top = max(line_rect.y0, previous_bottom if idx > 0 else inner_top)
            bottom = min(inner_bottom, max(top + line_h, line_rect.y1))
            if bottom - top < 4.0:
                bottom = min(inner_bottom, top + max(4.0, line_h))
            line_rect = fitz.Rect(inner_left, top, inner_right, bottom)
            previous_bottom = line_rect.y1
            line_heights.append(line_h)
            templates.append(
                LineTemplate(
                    line_id=f"{block.get('id') or 'block'}:line:{idx}",
                    source_line_indices=[idx],
                    bbox=(line_rect.x0, line_rect.y0, line_rect.x1, line_rect.y1),
                    baseline_y=line_rect.y0 + min(line_h * 0.82, line_h - 1.0),
                    ascent=line_h * 0.82,
                    descent=max(1.0, line_h * 0.18),
                    left_x=line_rect.x0,
                    right_x=line_rect.x1,
                    usable_width=max(8.0, line_rect.width - indent_px),
                    indent_px=indent_px,
                    first_line_indent_px=indent_px if line_index_in_paragraph == 0 else 0.0,
                    alignment=alignment,
                    paragraph_id=f"{block.get('id') or 'block'}:paragraph:{paragraph_index}",
                    paragraph_index=paragraph_index,
                    line_index_in_paragraph=line_index_in_paragraph,
                    is_first_paragraph_line=(line_index_in_paragraph == 0),
                    is_last_paragraph_line_hint=bool((line or {}).get("line_break_after")),
                )
            )
            line_index_in_paragraph += 1
        if not templates:
            line_h = max(8.0, block_rect.height)
            templates.append(
                LineTemplate(
                    line_id=f"{block.get('id') or 'block'}:line:0",
                    source_line_indices=[0],
                    bbox=(inner_left, inner_top, inner_right, inner_bottom),
                    baseline_y=inner_top + min(line_h * 0.82, line_h - 1.0),
                    ascent=line_h * 0.82,
                    descent=max(1.0, line_h * 0.18),
                    left_x=inner_left,
                    right_x=inner_right,
                    usable_width=max(8.0, inner_right - inner_left),
                    indent_px=0.0,
                    first_line_indent_px=0.0,
                    alignment=alignment,
                    paragraph_id=f"{block.get('id') or 'block'}:paragraph:0",
                    paragraph_index=0,
                    line_index_in_paragraph=0,
                    is_first_paragraph_line=True,
                    is_last_paragraph_line_hint=True,
                )
            )
        return templates

    def _collect_block_semantic_payload(self, block):
        return {
            "semantic_phrases": list((block or {}).get("semantic_phrases") or []),
            "semantic_groups": list((block or {}).get("semantic_groups") or []),
            "semantic_runs": list((block or {}).get("semantic_runs") or []),
            "semantic_spans": list((block or {}).get("semantic_spans") or []),
        }

    def _semantic_unit_sort_key(self, unit, default_idx):
        line_indices = unit.get("line_indices") or []
        first_line = line_indices[0] if line_indices else default_idx
        return (
            int(first_line),
            int(unit.get("fragment_index", default_idx) or default_idx),
            str(unit.get("unit_id") or ""),
        )

    def _unit_fitz_bbox(self, unit):
        rect = self._fitz_rect_from_bbox_like(unit.get("bbox"))
        if isinstance(rect, fitz.Rect) and rect.get_area() > 0:
            return (rect.x0, rect.y0, rect.x1, rect.y1)
        return None

    def _children_for_phrase(self, phrase_id, semantic_payload):
        selected = []
        for key in ("semantic_runs", "semantic_spans", "semantic_groups"):
            for unit in semantic_payload.get(key) or []:
                ctx = dict((unit or {}).get("structural_context") or {})
                candidate_phrase_id = str(
                    ctx.get("phrase_unit_id")
                    or unit.get("phrase_unit_id")
                    or unit.get("paragraph_id")
                    or ""
                )
                if candidate_phrase_id and candidate_phrase_id == phrase_id:
                    selected.append(dict(unit))
            if selected:
                return sorted(selected, key=lambda unit: self._semantic_unit_sort_key(unit, 0))
        return []

    def _translation_ruleset_for_unit(self, unit):
        if not isinstance(unit, dict):
            return {}
        for key in ("translation_ruleset", "element_ruleset"):
            value = unit.get(key)
            if isinstance(value, dict):
                return value
        return {}

    def _positioning_policy_for_unit(self, unit):
        if not isinstance(unit, dict):
            return {}
        value = unit.get("positioning_policy")
        return value if isinstance(value, dict) else {}

    def _protected_fragment_summary(self, child_units):
        fragments = [dict(unit) for unit in (child_units or []) if isinstance(unit, dict)]
        if not fragments:
            return {
                "count": 0,
                "protected_count": 0,
                "immutable_count": 0,
                "has_protected": False,
                "has_immutable": False,
                "dominant_inline_class": "",
            }
        protected_count = 0
        immutable_count = 0
        inline_classes = []
        for fragment in fragments:
            sem = dict(fragment.get("expression_semantics") or {})
            if bool(sem.get("protected_inline")):
                protected_count += 1
            if bool(sem.get("immutable_inline")):
                immutable_count += 1
            inline_class = str(sem.get("inline_class") or "").strip().lower()
            if inline_class:
                inline_classes.append(inline_class)
        dominant_inline_class = ""
        if inline_classes:
            dominant_inline_class = max(sorted(set(inline_classes)), key=inline_classes.count)
        return {
            "count": len(fragments),
            "protected_count": protected_count,
            "immutable_count": immutable_count,
            "has_protected": protected_count > 0,
            "has_immutable": immutable_count > 0,
            "dominant_inline_class": dominant_inline_class,
        }

    def _positioning_preferences_for_unit(
        self,
        raw_unit,
        *,
        text="",
        child_units=None,
        block=None,
        page_data=None,
        default_anchor_horizontal=None,
        default_anchor_vertical=None,
        default_render_policy="translated_editorial",
        default_keep_with_previous=False,
        default_keep_with_next=False,
        default_hard_break_before=False,
        default_hard_break_after=False,
        default_reflowable=True,
        default_break_priority=10,
    ):
        ruleset = self._translation_ruleset_for_unit(raw_unit)
        rules = dict(ruleset.get("rules") or {})
        constraints = dict(ruleset.get("constraints") or {})
        policy = self._positioning_policy_for_unit(raw_unit)
        primary_ref = dict(policy.get("primary_position_reference") or {})
        anchor_horizontal = (
            str(rules.get("preserve_horizontal_anchor") or "").strip().lower()
            or str((primary_ref or {}).get("horizontal") or "").strip().lower()
            or str(default_anchor_horizontal or "").strip().lower()
            or None
        )
        anchor_vertical = (
            str(rules.get("preserve_vertical_anchor") or "").strip().lower()
            or str((primary_ref or {}).get("vertical") or "").strip().lower()
            or str(default_anchor_vertical or "").strip().lower()
            or None
        )
        text_clean = self._clean_text_for_render(text or "")
        child_summary = self._protected_fragment_summary(child_units)
        adaptive_profile = self._unit_adaptive_profile(raw_unit, text=text_clean, child_units=child_units, block=block, page_data=page_data)
        anchor_confidence = float(primary_ref.get("confidence") or 0.0)
        semantic_role = str(rules.get("semantic_role") or "").strip().lower()
        horizontal_growth = str(rules.get("horizontal_growth") or "").strip().lower()
        vertical_growth = str(rules.get("vertical_growth") or "").strip().lower()
        allow_horizontal_reflow = bool(constraints.get("allow_horizontal_reflow", True))
        preserve_center = bool(constraints.get("preserve_center_if_possible", False))
        short_text = bool(text_clean and len(text_clean) <= 64)
        has_bbox = bool(self._unit_fitz_bbox(raw_unit))
        anchored_role = semantic_role in {"attached_label", "end_value", "centered_title"}
        force_bbox_anchor = False
        render_policy = str(default_render_policy or "translated_editorial")
        reflowable = bool(default_reflowable)
        # "external_flow" est une politique verrouillée qui ne doit pas être écrasée
        _locked_policy = render_policy in {"external_flow"}
        if not _locked_policy and has_bbox and (
            anchored_role
            or preserve_center
            or adaptive_profile.get("prefer_bbox_anchor")
            or (not allow_horizontal_reflow and short_text and anchor_confidence >= 0.55)
            or (child_summary["has_protected"] and short_text)
            or child_summary["dominant_inline_class"] in {"formula", "reference", "code"}
        ):
            force_bbox_anchor = True
            render_policy = "anchored_text"
            reflowable = False
        if not _locked_policy and child_summary["has_immutable"] and short_text:
            render_policy = "fixed_preserve"
            reflowable = False
            force_bbox_anchor = True
        unit_profile = str(adaptive_profile.get("unit_profile") or "")
        if not _locked_policy and unit_profile in {"protected_inline", "anchored_label", "technical_inline_cluster"}:
            render_policy = "fixed_preserve" if unit_profile == "protected_inline" and short_text else "anchored_text"
            reflowable = False
            force_bbox_anchor = force_bbox_anchor or has_bbox
        elif unit_profile == "dense_editorial_phrase" and not adaptive_profile.get("allow_aggressive_reflow", True):
            reflowable = False if short_text and has_bbox else reflowable
        keep_with_previous = bool(rules.get("keep_with_previous", default_keep_with_previous))
        keep_with_next = bool(rules.get("keep_with_next", default_keep_with_next))
        hard_break_before = bool(rules.get("hard_break_before", default_hard_break_before))
        hard_break_after = bool(rules.get("hard_break_after", default_hard_break_after))
        break_priority = int(default_break_priority)
        if anchored_role or child_summary["has_protected"]:
            break_priority = max(break_priority, 18)
        if unit_profile in {"protected_inline", "anchored_label", "technical_inline_cluster"}:
            break_priority = max(break_priority, 20)
        metadata = {
            "translation_positioning_mode": str(rules.get("translation_positioning_mode") or "").strip().lower(),
            "semantic_role": semantic_role,
            "horizontal_growth": horizontal_growth,
            "vertical_growth": vertical_growth,
            "anchor_confidence": anchor_confidence,
            "force_bbox_anchor": bool(force_bbox_anchor),
            "allow_horizontal_reflow": allow_horizontal_reflow,
            "preserve_center_if_possible": preserve_center,
            "has_protected_fragments": bool(child_summary["has_protected"]),
            "has_immutable_fragments": bool(child_summary["has_immutable"]),
            "dominant_inline_class": child_summary["dominant_inline_class"],
            "adaptive_profile": adaptive_profile,
        }
        return {
            "anchor_horizontal": anchor_horizontal,
            "anchor_vertical": anchor_vertical,
            "render_policy": render_policy,
            "reflowable": reflowable,
            "keep_with_previous": keep_with_previous,
            "keep_with_next": keep_with_next,
            "hard_break_before": hard_break_before,
            "hard_break_after": hard_break_after,
            "break_priority": break_priority,
            "protected_inline": bool(child_summary["has_protected"]),
            "immutable": bool(child_summary["has_immutable"]),
            "metadata": metadata,
        }

    def _line_looks_technical_structured(self, line, block=None):
        if not isinstance(line, dict):
            return False
        unit_type = str(line.get("unit_type") or "").strip().lower()
        if unit_type == "code_visible":
            return True
        text = self._clean_text_for_render(
            line.get("line_text")
            or line.get("translated_text")
            or ""
        )
        phrases = list(line.get("phrases") or [])
        if any(str((phrase or {}).get("unit_type") or "").strip().lower() == "code_visible" for phrase in phrases):
            return True
        if any(bool((((phrase or {}).get("style") or {}).get("flags") or {}).get("monospace")) for phrase in phrases):
            return True
        if not text:
            return False
        technical_patterns = (
            r"[A-Za-z_][A-Za-z0-9_]*\s*\(",
            r"\b(?:Conv\dD|Dense|MaxPool|AvgPool|BatchNorm|ReLU|Dropout)\b",
            r"\b(?:padding|stride|strides|filters|kernel_size|activation)\s*=",
            r"^#\s*\w+",
            r"\b[a-z]+_[a-z0-9_]+\b",
        )
        if any(re.search(pattern, text) for pattern in technical_patterns):
            return True
        punctuation = len(re.findall(r"[^A-Za-z0-9\s]", text))
        lexical = len(re.findall(r"[A-Za-z]+", text))
        if "_" in text and punctuation >= 2:
            return True
        if punctuation >= 6 and lexical <= 14 and ("=" in text or "(" in text or ")" in text):
            return True
        if bool((block or {}).get("immutable_code_block")):
            return True
        return False

    def _block_looks_technical_structured(self, block):
        if not isinstance(block, dict):
            return False
        descriptor_role = str((block or {}).get("descriptor_structural_role") or "").strip().lower()
        if descriptor_role in {"code_block", "listing", "table_code_listing"}:
            return True
        lines = [line for line in ((block or {}).get("lines") or []) if isinstance(line, dict)]
        if not lines:
            return False
        technical_lines = sum(1 for line in lines if self._line_looks_technical_structured(line, block=block))
        if technical_lines >= max(2, math.ceil(len(lines) * 0.35)):
            return True
        return False

    def _unit_horizontal_alignment(self, unit, fallback):
        alignment = self._normalize_alignment(fallback)
        anchor = str(getattr(unit, "anchor_horizontal", "") or "").strip().lower()
        metadata = dict(getattr(unit, "metadata", {}) or {})
        horizontal_growth = str(metadata.get("horizontal_growth") or "").strip().lower()
        if anchor in {"start", "left"}:
            return "left"
        if anchor in {"end", "right"}:
            return "right"
        if anchor == "center":
            return "center"
        if horizontal_growth == "grow_to_start":
            return "right"
        if horizontal_growth == "grow_symmetrically":
            return "center"
        return alignment

    def _unit_render_tuning(self, unit, plan=None):
        metadata = dict(getattr(unit, "metadata", {}) or {})
        adaptive = dict(metadata.get("adaptive_profile") or {})
        plan_profile = dict((getattr(plan, "adaptive_profile", {}) or {}))
        line_spacing_factor = float(adaptive.get("line_spacing_factor") or plan_profile.get("line_spacing_factor") or 1.0)
        unit_profile = str(adaptive.get("unit_profile") or "")
        page_profile = str(adaptive.get("page_profile") or plan_profile.get("page_profile") or "")
        min_fontsize = 5.5
        if page_profile in {"academic_dense", "technical_structured"} or unit_profile in {"protected_inline", "technical_inline_cluster", "dense_editorial_phrase"}:
            min_fontsize = 5.0
        if unit_profile in {"anchored_label", "protected_inline"}:
            line_spacing_factor = max(line_spacing_factor, 0.98)
        if unit_profile in {"technical_inline_cluster", "dense_editorial_phrase"}:
            line_spacing_factor = min(line_spacing_factor, 0.94)
        return {
            "adaptive_profile": adaptive,
            "unit_profile": unit_profile,
            "line_spacing_factor": max(0.84, min(1.08, line_spacing_factor)),
            "min_fontsize": min_fontsize,
            "prefer_bbox_anchor": bool(adaptive.get("prefer_bbox_anchor") or plan_profile.get("prefer_bbox_anchor")),
            "prefer_atomic_short_units": bool(adaptive.get("prefer_atomic_short_units") or plan_profile.get("prefer_atomic_short_units")),
        }

    def _anchored_line_baseline(self, rect, unit, fontsize, line_h, line_index, line_count):
        anchor = str(getattr(unit, "anchor_vertical", "") or "").strip().lower()
        if anchor == "middle":
            total_h = max(line_h, line_h * max(1, line_count))
            top_y = rect.y0 + max(0.0, (rect.height - total_h) / 2.0)
        elif anchor == "bottom":
            total_h = max(line_h, line_h * max(1, line_count))
            top_y = max(rect.y0, rect.y1 - total_h)
        else:
            top_y = rect.y0
        return top_y + min(rect.height - 1.0, (line_index + 1) * line_h * 0.82)

    def _line_translated_text(self, line):
        text = self._clean_text_for_render((line or {}).get("translated_text") or "")
        if text:
            return text
        parts = []
        for phrase in (line or {}).get("phrases") or []:
            phrase_text = self._clean_text_for_render((phrase or {}).get("translated_text") or "")
            if phrase_text:
                parts.append(phrase_text)
        return " ".join(parts).strip()

    def _line_source_text(self, line):
        text = self._clean_text_for_render((line or {}).get("line_text") or "")
        if text:
            return text
        parts = []
        for phrase in (line or {}).get("phrases") or []:
            phrase_text = self._clean_text_for_render((phrase or {}).get("text") or (phrase or {}).get("texte") or "")
            if phrase_text:
                parts.append(phrase_text)
        return " ".join(parts).strip()

    def _semantic_phrase_translated_text(self, block, phrase, fallback_text):
        direct = self._clean_text_for_render((phrase or {}).get("translated_text") or "")
        if direct:
            return direct
        line_indices = [
            int(v) for v in ((phrase or {}).get("line_indices") or [])
            if isinstance(v, (int, float))
        ]
        lines = list((block or {}).get("lines") or [])
        parts = []
        seen = set()
        for idx in line_indices:
            if idx < 0 or idx >= len(lines):
                continue
            text = self._line_translated_text(lines[idx])
            if not text:
                continue
            key = text.strip()
            if key and key not in seen:
                seen.add(key)
                parts.append(text)
        if parts:
            return self._clean_text_for_render(" ".join(parts))
        return fallback_text

    def _semantic_phrase_source_text(self, block, phrase, fallback_text):
        direct = self._clean_text_for_render((phrase or {}).get("text") or (phrase or {}).get("texte") or "")
        if direct:
            return direct
        line_indices = [
            int(v) for v in ((phrase or {}).get("line_indices") or [])
            if isinstance(v, (int, float))
        ]
        lines = list((block or {}).get("lines") or [])
        parts = []
        seen = set()
        for idx in line_indices:
            if idx < 0 or idx >= len(lines):
                continue
            text = self._line_source_text(lines[idx])
            if not text:
                continue
            key = text.strip()
            if key and key not in seen:
                seen.add(key)
                parts.append(text)
        if parts:
            return self._clean_text_for_render(" ".join(parts))
        return fallback_text

    def _phrase_units(self, block, semantic_payload, target_lang, page_data=None):
        block_id = str((block or {}).get("id") or "")
        block_role = str((block or {}).get("role") or "body").strip().lower()
        block_style = self._style_from_block(block)
        phrases = self._dedupe_semantic_phrases(sorted(
            [phrase for phrase in semantic_payload.get("semantic_phrases") or [] if isinstance(phrase, dict)],
            key=lambda unit: self._semantic_unit_sort_key(unit, 0),
        ))
        units = []
        for idx, phrase in enumerate(phrases):
            source_fallback = self._clean_text_for_render(phrase.get("text") or phrase.get("texte") or "")
            translated_text = self._semantic_phrase_translated_text(block, phrase, "")
            source_text = self._semantic_phrase_source_text(block, phrase, source_fallback)
            if not translated_text:
                continue
            ctx = dict(phrase.get("structural_context") or {})
            phrase_id = str(ctx.get("phrase_unit_id") or phrase.get("unit_id") or f"{block_id}:phrase:{idx}")
            editorial_rel = dict((phrase.get("editorial_relations") or {}).get("with_previous") or {})
            child_units = self._children_for_phrase(phrase_id, semantic_payload)
            positioning = self._positioning_preferences_for_unit(
                phrase,
                text=translated_text,
                child_units=child_units,
                default_anchor_horizontal=((phrase.get("layout_attributes") or {}).get("horizontal_anchor")),
                default_anchor_vertical=((phrase.get("layout_attributes") or {}).get("vertical_anchor")),
                block=block,
                page_data=page_data,
                default_render_policy=str(phrase.get("render_policy") or block.get("render_policy") or "translated_editorial"),
                default_keep_with_previous=bool(editorial_rel.get("relation") in {"keep_with_previous", "label_value"}),
                default_keep_with_next=bool(((phrase.get("editorial_relations") or {}).get("with_next") or {}).get("relation") in {"keep_with_next", "label_value"}),
                default_hard_break_before=bool(phrase.get("hard_break_before") or editorial_rel.get("relation") in {"paragraph_break", "new_line"}),
                default_hard_break_after=bool(phrase.get("hard_break_after")),
                default_reflowable=bool((phrase.get("editorial_semantics") or {}).get("reflowable", True)),
                default_break_priority=10,
            )
            units.append(
                PlacableUnit(
                    unit_id=str(phrase.get("unit_id") or phrase_id),
                    unit_type=str(phrase.get("unit_type") or "semantic_phrase"),
                    source_kind=str(phrase.get("source_kind") or "semantic_phrase"),
                    parent_unit_id=ctx.get("parent_unit_id"),
                    block_unit_id=str(ctx.get("block_unit_id") or block_id),
                    phrase_unit_id=phrase_id,
                    line_indices=[int(v) for v in (phrase.get("line_indices") or []) if isinstance(v, (int, float))] or [idx],
                    text_source=source_text,
                    text_translated=translated_text,
                    role=block_role,
                    inline_class=None,
                    group_class=str(phrase.get("group_class") or "").strip().lower() or None,
                    style=self._merge_styles(phrase.get("style") or {}, block_style),
                    layout_attributes=dict(phrase.get("layout_attributes") or {}),
                    text_attributes=dict(phrase.get("text_attributes") or {}),
                    relative_bbox=self._unit_fitz_bbox(phrase),
                    anchor_horizontal=positioning["anchor_horizontal"],
                    anchor_vertical=positioning["anchor_vertical"],
                    continuation_before=bool(editorial_rel.get("continuation")),
                    continuation_after=bool(((phrase.get("editorial_relations") or {}).get("with_next") or {}).get("continuation")),
                    hard_break_before=positioning["hard_break_before"],
                    hard_break_after=positioning["hard_break_after"],
                    keep_with_previous=positioning["keep_with_previous"],
                    keep_with_next=positioning["keep_with_next"],
                    reflowable=positioning["reflowable"],
                    protected_inline=positioning["protected_inline"],
                    immutable=positioning["immutable"],
                    render_policy=positioning["render_policy"],
                    justification_eligible=not positioning["protected_inline"],
                    break_priority=positioning["break_priority"],
                    paragraph_id=str(ctx.get("paragraph_id") or phrase_id),
                    metadata={
                        "target_lang": target_lang,
                        "raw_unit": dict(phrase),
                        "fragments": child_units,
                        **positioning["metadata"],
                    },
                )
            )
        return units

    def _dedupe_semantic_phrases(self, phrases):
        kept = []
        for phrase in phrases or []:
            text = self._clean_text_for_render(
                phrase.get("translated_text") or phrase.get("text") or phrase.get("texte") or ""
            )
            if not text:
                kept.append(phrase)
                continue
            line_indices = [
                int(v) for v in (phrase.get("line_indices") or [])
                if isinstance(v, (int, float))
            ]
            first_line = line_indices[0] if line_indices else -1
            rect = self._fitz_rect_from_bbox_like(phrase.get("bbox"))
            replaced = False
            skip = False
            for idx, existing in enumerate(list(kept)):
                existing_text = self._clean_text_for_render(
                    existing.get("translated_text") or existing.get("text") or existing.get("texte") or ""
                )
                existing_lines = [
                    int(v) for v in (existing.get("line_indices") or [])
                    if isinstance(v, (int, float))
                ]
                existing_first = existing_lines[0] if existing_lines else -1
                if first_line != existing_first:
                    continue
                nested = text == existing_text or text in existing_text or existing_text in text
                if not nested:
                    continue
                existing_rect = self._fitz_rect_from_bbox_like(existing.get("bbox"))
                current_score = (
                    len(line_indices),
                    len(text),
                    rect.get_area() if isinstance(rect, fitz.Rect) else 0.0,
                )
                existing_score = (
                    len(existing_lines),
                    len(existing_text),
                    existing_rect.get_area() if isinstance(existing_rect, fitz.Rect) else 0.0,
                )
                if current_score > existing_score:
                    kept[idx] = phrase
                    replaced = True
                else:
                    skip = True
                break
            if skip:
                continue
            if not replaced:
                kept.append(phrase)
        kept.sort(key=lambda unit: self._semantic_unit_sort_key(unit, 0))
        return kept

    def _semantic_phrases_are_overlapping(self, phrases):
        normalized = []
        for phrase in phrases or []:
            line_indices = [
                int(v) for v in (phrase.get("line_indices") or [])
                if isinstance(v, (int, float))
            ]
            if not line_indices:
                continue
            normalized.append((min(line_indices), max(line_indices), phrase))
        normalized.sort(key=lambda item: (item[0], item[1]))
        for idx in range(1, len(normalized)):
            prev_start, prev_end, prev_phrase = normalized[idx - 1]
            cur_start, cur_end, cur_phrase = normalized[idx]
            if cur_start <= prev_end and cur_end > prev_end:
                prev_text = self._clean_text_for_render(
                    (prev_phrase or {}).get("translated_text")
                    or (prev_phrase or {}).get("text")
                    or (prev_phrase or {}).get("texte")
                    or ""
                )
                cur_text = self._clean_text_for_render(
                    (cur_phrase or {}).get("translated_text")
                    or (cur_phrase or {}).get("text")
                    or (cur_phrase or {}).get("texte")
                    or ""
                )
                if prev_text and cur_text:
                    return True
        return False

    def _line_units(self, block, target_lang, page_data=None):
        block_id = str((block or {}).get("id") or "")
        block_role = str((block or {}).get("role") or "body").strip().lower()
        block_style = self._style_from_block(block)
        units = []
        for idx, line in enumerate((block or {}).get("lines") or []):
            translated_text = self._line_translated_text(line)
            if not translated_text:
                continue
            source_text = self._line_source_text(line) or translated_text
            line_rect = self._fitz_rect_from_bbox_like((line or {}).get("bbox"))
            bbox = (line_rect.x0, line_rect.y0, line_rect.x1, line_rect.y1) if isinstance(line_rect, fitz.Rect) else None
            positioning = self._positioning_preferences_for_unit(
                line,
                text=translated_text,
                child_units=None,
                block=block,
                page_data=page_data,
                default_anchor_horizontal=((line.get("layout_attributes") or {}).get("horizontal_anchor")),
                default_anchor_vertical=((line.get("layout_attributes") or {}).get("vertical_anchor")),
                default_render_policy=str((line or {}).get("render_policy") or (block or {}).get("render_policy") or "translated_editorial"),
                default_keep_with_previous=False,
                default_keep_with_next=False,
                default_hard_break_before=bool((line or {}).get("hard_break_before") or idx > 0),
                default_hard_break_after=bool((line or {}).get("line_break_after")),
                default_reflowable=False,
                default_break_priority=15,
            )
            units.append(
                PlacableUnit(
                    unit_id=f"{block_id}:line_unit:{idx}",
                    unit_type="translated_line",
                    source_kind="line",
                    parent_unit_id=block_id or None,
                    block_unit_id=block_id,
                    phrase_unit_id=f"{block_id}:line_phrase:{idx}",
                    line_indices=[idx],
                    text_source=source_text,
                    text_translated=translated_text,
                    role=block_role,
                    inline_class=None,
                    group_class="line",
                    style=self._merge_styles((line or {}).get("style") or {}, block_style),
                    layout_attributes=dict((line or {}).get("layout_attributes") or {}),
                    text_attributes={},
                    relative_bbox=bbox,
                    anchor_horizontal=positioning["anchor_horizontal"],
                    anchor_vertical=positioning["anchor_vertical"],
                    continuation_before=False,
                    continuation_after=False,
                    hard_break_before=positioning["hard_break_before"],
                    hard_break_after=positioning["hard_break_after"],
                    keep_with_previous=positioning["keep_with_previous"],
                    keep_with_next=positioning["keep_with_next"],
                    reflowable=positioning["reflowable"],
                    protected_inline=positioning["protected_inline"],
                    immutable=positioning["immutable"],
                    render_policy=positioning["render_policy"],
                    justification_eligible=not positioning["protected_inline"],
                    break_priority=positioning["break_priority"],
                    paragraph_id=f"{block_id}:line_paragraph:{idx}",
                    metadata={"target_lang": target_lang, "raw_unit": dict(line or {}), **positioning["metadata"]},
                )
            )
        return units

    def _nested_span_units(self, block, target_lang, page_data=None):
        block_id = str((block or {}).get("id") or "")
        block_role = str((block or {}).get("role") or "body").strip().lower()
        block_style = self._style_from_block(block)
        render_policy = str((block or {}).get("render_policy") or "translated_editorial")
        units = []
        for li, line in enumerate((block or {}).get("lines") or []):
            # Texte traduit de la ligne pour détecter les spans non-traduits
            line_translated = self._clean_text_for_render((line or {}).get("translated_text") or "")
            for pi, phrase in enumerate((line or {}).get("phrases") or []):
                phrase_id = str((phrase or {}).get("unit_id") or f"{block_id}:line:{li}:phrase:{pi}")
                phrase_translated = self._clean_text_for_render((phrase or {}).get("translated_text") or "")
                # Texte de référence pour détecter les doublon dans cette phrase
                seen_translated_in_phrase: set[str] = set()
                spans = list((phrase or {}).get("spans") or [])
                for si, span in enumerate(spans):
                    source_text = self._clean_text_for_render((span or {}).get("texte") or (span or {}).get("text") or "")
                    translated_text = self._clean_text_for_render((span or {}).get("translated_text") or "")
                    if not translated_text:
                        translated_text = source_text
                    if not translated_text:
                        continue
                    # Filtrer les spans dont translated_text == source_text (non traduit)
                    # quand la phrase ou la ligne dispose d'une vraie traduction.
                    # Exception : ne pas filtrer si le texte source est court (≤3 chars)
                    # et apparaît dans la traduction de la phrase (ex. "A" dans "ANNEXE A").
                    ref_translation = phrase_translated or line_translated
                    if (
                        source_text
                        and translated_text == source_text
                        and ref_translation
                        and ref_translation != source_text
                    ):
                        short_preserved = (
                            len(source_text) <= 3
                            and re.search(r"(?<!\w)" + re.escape(source_text) + r"(?!\w)", ref_translation)
                        )
                        if not short_preserved:
                            continue
                    # Filtrer les spans "gloutons" : la traduction du span est
                    # significativement plus longue que celle de la phrase parente.
                    # Ces spans ont absorbé plus que leur part et génèrent des overlaps.
                    if (
                        phrase_translated
                        and len(phrase_translated) > 10
                        and len(translated_text) > len(phrase_translated) * 1.3
                    ):
                        continue
                    # Supprimer le préfixe bullet PUA déplacé : quand la traduction commence
                    # par des caractères PUA (ex. \uf0a1) non présents dans le source, ces
                    # caractères appartiennent à un span voisin filtré et débordent de la bbox.
                    if translated_text and source_text:
                        leading_pua = re.match(r"^[\ue000-\uf8ff][\ue000-\uf8ff \t]*", translated_text)
                        if leading_pua and not re.match(r"^[\ue000-\uf8ff]", source_text):
                            translated_text = translated_text[len(leading_pua.group(0)):].lstrip()
                            if not translated_text:
                                continue
                    # Dédupliquer : si ce texte traduit a déjà été émis dans cette phrase,
                    # sauter ce span (cas dégénéré où plusieurs spans portent la même traduction).
                    dedup_key = re.sub(r"\s+", " ", translated_text).strip().lower()
                    if dedup_key and len(dedup_key.split()) >= 4:
                        if dedup_key in seen_translated_in_phrase:
                            continue
                        seen_translated_in_phrase.add(dedup_key)
                    bbox = self._unit_fitz_bbox(span)
                    if not bbox:
                        continue
                    expression_semantics = dict((span or {}).get("expression_semantics") or {})
                    inline_class = str(expression_semantics.get("inline_class") or "").strip().lower() or None
                    positioning = self._positioning_preferences_for_unit(
                        span,
                        text=translated_text,
                        child_units=None,
                        block=block,
                        page_data=page_data,
                        default_anchor_horizontal=(((span or {}).get("layout_attributes") or {}).get("horizontal_anchor")),
                        default_anchor_vertical=(((span or {}).get("layout_attributes") or {}).get("vertical_anchor")),
                        default_render_policy=render_policy,
                        default_keep_with_previous=False,
                        default_keep_with_next=False,
                        default_hard_break_before=bool(si == 0),
                        default_hard_break_after=bool(si == len((phrase or {}).get("spans") or []) - 1),
                        default_reflowable=False,
                        default_break_priority=20,
                    )
                    units.append(
                        PlacableUnit(
                            unit_id=str((span or {}).get("unit_id") or f"{phrase_id}:span:{si}"),
                            unit_type=str((span or {}).get("unit_type") or "span"),
                            source_kind=str((span or {}).get("source_kind") or "span"),
                            parent_unit_id=phrase_id,
                            block_unit_id=block_id,
                            phrase_unit_id=phrase_id,
                            line_indices=[li],
                            text_source=source_text or translated_text,
                            text_translated=translated_text,
                            role=block_role,
                            inline_class=inline_class,
                            group_class=None,
                            style=self._merge_styles((span or {}).get("style") or {}, block_style),
                            layout_attributes=dict((span or {}).get("layout_attributes") or {}),
                            text_attributes=dict((span or {}).get("text_attributes") or {}),
                            relative_bbox=bbox,
                            anchor_horizontal=positioning["anchor_horizontal"],
                            anchor_vertical=positioning["anchor_vertical"],
                            continuation_before=False,
                            continuation_after=False,
                            hard_break_before=positioning["hard_break_before"],
                            hard_break_after=positioning["hard_break_after"],
                            keep_with_previous=positioning["keep_with_previous"],
                            keep_with_next=positioning["keep_with_next"],
                            reflowable=positioning["reflowable"],
                            protected_inline=bool(expression_semantics.get("protected_inline", False)) or positioning["protected_inline"],
                            immutable=bool(expression_semantics.get("immutable_inline", False)) or positioning["immutable"],
                            render_policy=positioning["render_policy"],
                            justification_eligible=inline_class not in {"code", "formula", "reference"} and not positioning["protected_inline"],
                            break_priority=positioning["break_priority"],
                            paragraph_id=f"{block_id}:nested_span_line:{li}",
                            metadata={"target_lang": target_lang, "raw_unit": dict(span or {}), **positioning["metadata"]},
                        )
                    )
        return units

    def _fallback_units(self, block, semantic_payload, target_lang, page_data=None):
        block_id = str((block or {}).get("id") or "")
        block_role = str((block or {}).get("role") or "body").strip().lower()
        block_style = self._style_from_block(block)
        render_policy = str((block or {}).get("render_policy") or "").strip().lower()
        candidate_order = ("semantic_phrases", "semantic_spans", "semantic_runs", "semantic_groups") if render_policy in {"anchored_text", "fixed_preserve"} else ("semantic_groups", "semantic_runs", "semantic_spans")
        candidates = []
        for key in candidate_order:
            if semantic_payload.get(key):
                candidates = [unit for unit in semantic_payload.get(key) or [] if isinstance(unit, dict)]
                break
        if not candidates:
            text = self._translated_text_from_block(block)
            if not text:
                return []
            return [
                PlacableUnit(
                    unit_id=f"{block_id}:fallback:0",
                    unit_type="fallback",
                    source_kind="fallback",
                    parent_unit_id=None,
                    block_unit_id=block_id,
                    phrase_unit_id=f"{block_id}:paragraph:0",
                    line_indices=[0],
                    text_source=text,
                    text_translated=text,
                    role=block_role,
                    style=self._merge_styles(block_style, {}),
                    render_policy=render_policy or "translated_editorial",
                    paragraph_id=f"{block_id}:paragraph:0",
                    metadata={"target_lang": target_lang},
                )
            ]
        normalized = []
        ordered_candidates = [
            unit
            for _, unit in sorted(
                enumerate(candidates),
                key=lambda pair: self._semantic_unit_sort_key(pair[1], pair[0]),
            )
        ]
        for idx, unit in enumerate(ordered_candidates):
            text = self._clean_text_for_render(unit.get("translated_text") or "")
            if not text:
                continue
            ctx = dict(unit.get("structural_context") or {})
            editorial_rel = dict((unit.get("editorial_relations") or {}).get("with_previous") or {})
            expression_semantics = dict(unit.get("expression_semantics") or {})
            positioning = self._positioning_preferences_for_unit(
                unit,
                text=text,
                child_units=None,
                default_anchor_horizontal=((unit.get("layout_attributes") or {}).get("horizontal_anchor")),
                default_anchor_vertical=((unit.get("layout_attributes") or {}).get("vertical_anchor")),
                block=block,
                page_data=page_data,
                default_render_policy=str(unit.get("render_policy") or render_policy or "translated_editorial"),
                default_keep_with_previous=bool(editorial_rel.get("relation") in {"keep_with_previous", "label_value"}),
                default_keep_with_next=bool(((unit.get("editorial_relations") or {}).get("with_next") or {}).get("relation") in {"keep_with_next", "label_value"}),
                default_hard_break_before=bool(unit.get("hard_break_before") or editorial_rel.get("relation") in {"paragraph_break", "new_line"}),
                default_hard_break_after=bool(unit.get("hard_break_after")),
                default_reflowable=bool((unit.get("editorial_semantics") or {}).get("reflowable", True)),
                default_break_priority=10 if str(unit.get("group_class") or "").strip() else 5,
            )
            normalized.append(
                PlacableUnit(
                    unit_id=str(unit.get("unit_id") or f"{block_id}:{idx}"),
                    unit_type=str(unit.get("unit_type") or "semantic"),
                    source_kind=str(unit.get("source_kind") or "semantic"),
                    parent_unit_id=ctx.get("parent_unit_id"),
                    block_unit_id=str(ctx.get("block_unit_id") or block_id),
                    phrase_unit_id=str(ctx.get("phrase_unit_id") or unit.get("unit_id") or f"{block_id}:{idx}"),
                    line_indices=[int(v) for v in (unit.get("line_indices") or []) if isinstance(v, (int, float))] or [idx],
                    text_source=self._clean_text_for_render(unit.get("text") or unit.get("texte") or text),
                    text_translated=text,
                    role=block_role,
                    inline_class=str(expression_semantics.get("inline_class") or "").strip().lower() or None,
                    group_class=str(unit.get("group_class") or "").strip().lower() or None,
                    style=self._merge_styles(unit.get("style") or {}, block_style),
                    layout_attributes=dict(unit.get("layout_attributes") or {}),
                    text_attributes=dict(unit.get("text_attributes") or {}),
                    relative_bbox=self._unit_fitz_bbox(unit),
                    anchor_horizontal=positioning["anchor_horizontal"],
                    anchor_vertical=positioning["anchor_vertical"],
                    continuation_before=bool(editorial_rel.get("continuation")),
                    continuation_after=bool(((unit.get("editorial_relations") or {}).get("with_next") or {}).get("continuation")),
                    hard_break_before=positioning["hard_break_before"],
                    hard_break_after=positioning["hard_break_after"],
                    keep_with_previous=positioning["keep_with_previous"],
                    keep_with_next=positioning["keep_with_next"],
                    reflowable=positioning["reflowable"],
                    protected_inline=bool(expression_semantics.get("protected_inline", False)) or positioning["protected_inline"],
                    immutable=bool(expression_semantics.get("immutable_inline", False)) or positioning["immutable"],
                    render_policy=positioning["render_policy"],
                    justification_eligible=str(expression_semantics.get("inline_class") or "").strip().lower() not in {"code", "formula", "reference"},
                    break_priority=positioning["break_priority"],
                    paragraph_id=str(ctx.get("paragraph_id") or ctx.get("phrase_unit_id") or f"{block_id}:paragraph:0"),
                    metadata={"target_lang": target_lang, "raw_unit": dict(unit), **positioning["metadata"]},
                )
            )
        return normalized

    def _orphan_semantic_units(self, block, semantic_payload, target_lang, phrase_units, page_data=None):
        phrase_ids = {str(unit.phrase_unit_id or "") for unit in phrase_units or []}
        phrase_unit_ids = {str(unit.unit_id or "") for unit in phrase_units or []}
        extras = []
        seen_unit_ids = set()
        for unit in self._fallback_units(block, semantic_payload, target_lang, page_data=page_data):
            if unit.unit_id in phrase_unit_ids or unit.unit_id in seen_unit_ids:
                continue
            phrase_id = str(unit.phrase_unit_id or "")
            if phrase_id and phrase_id in phrase_ids:
                continue
            extras.append(unit)
            seen_unit_ids.add(unit.unit_id)
        return extras

    def _normalize_placable_units(self, block, semantic_payload, target_lang, page_data=None):
        render_policy = str((block or {}).get("render_policy") or "").strip().lower()
        if render_policy in {"anchored_text", "fixed_preserve"}:
            nested_span_units = self._nested_span_units(block, target_lang, page_data=page_data)
            if nested_span_units:
                return self._canonicalize_block_units(block, nested_span_units)
        # Pour les blocs non-ancrés, semantic_groups sont prioritaires sur semantic_phrases
        if render_policy not in {"anchored_text", "fixed_preserve"}:
            _group_candidates = [
                u for u in (semantic_payload.get("semantic_groups") or [])
                if isinstance(u, dict) and self._clean_text_for_render((u or {}).get("translated_text") or "")
            ]
            if _group_candidates:
                group_units = self._fallback_units(block, semantic_payload, target_lang, page_data=page_data)
                if group_units:
                    return self._canonicalize_block_units(block, group_units)
        phrase_units = self._phrase_units(block, semantic_payload, target_lang, page_data=page_data)
        if phrase_units and self._semantic_phrases_are_overlapping(semantic_payload.get("semantic_phrases") or []):
            line_units = self._line_units(block, target_lang, page_data=page_data)
            if line_units:
                phrase_units = line_units
        external_units = [] if self._has_translated_payload(block) else self._external_units_for_block(block, page_data, target_lang)
        if external_units and not phrase_units:
            return external_units
        if phrase_units:
            units = phrase_units + self._orphan_semantic_units(block, semantic_payload, target_lang, phrase_units, page_data=page_data)
            if external_units:
                seen = {
                    (
                        self._clean_text_for_render(unit.text_translated or ""),
                        tuple(round(float(v), 2) for v in (unit.relative_bbox or ())),
                    )
                    for unit in units
                }
                for unit in external_units:
                    key = (
                        self._clean_text_for_render(unit.text_translated or ""),
                        tuple(round(float(v), 2) for v in (unit.relative_bbox or ())),
                    )
                    if key not in seen:
                        units.append(unit)
                        seen.add(key)
            return self._canonicalize_block_units(block, units)
        fallback_units = self._fallback_units(block, semantic_payload, target_lang, page_data=page_data)
        if external_units:
            merged = external_units if not fallback_units else external_units + fallback_units
            return self._canonicalize_block_units(block, merged)
        return self._canonicalize_block_units(block, fallback_units)

    def _canonicalize_block_units(self, block, units):
        ordered = list(units or [])
        if not ordered:
            return ordered
        enriched = []
        for idx, unit in enumerate(ordered):
            rect = fitz.Rect(unit.relative_bbox) if unit.relative_bbox else None
            top = rect.y0 if isinstance(rect, fitz.Rect) else float(unit.line_indices[0] if unit.line_indices else idx) * 14.0
            left = rect.x0 if isinstance(rect, fitz.Rect) else 0.0
            line_idx = int(unit.line_indices[0]) if unit.line_indices else idx
            enriched.append({"idx": idx, "unit": unit, "rect": rect, "top": top, "left": left, "line_idx": line_idx})
        enriched.sort(key=lambda item: (item["line_idx"], item["top"], item["left"], item["idx"]))

        row_groups = []
        for item in enriched:
            assigned = None
            for row in row_groups:
                if item["line_idx"] != row["line_idx"]:
                    continue
                if abs(item["top"] - row["top"]) <= 3.0:
                    assigned = row
                    break
            if assigned is None:
                assigned = {
                    "line_idx": item["line_idx"],
                    "top": item["top"],
                    "items": [],
                }
                row_groups.append(assigned)
            assigned["items"].append(item)
            assigned["top"] = min(assigned["top"], item["top"])

        canonical = []
        final_row_idx = 0
        for row in row_groups:
            row_items = sorted(row["items"], key=lambda item: (item["left"], item["top"], item["idx"]))
            islands = []
            for item in row_items:
                if not islands:
                    islands.append([item])
                    continue
                prev = islands[-1][-1]
                prev_rect = prev["rect"]
                cur_rect = item["rect"]
                prev_unit = prev["unit"]
                cur_unit = item["unit"]
                gap = None
                if isinstance(prev_rect, fitz.Rect) and isinstance(cur_rect, fitz.Rect):
                    gap = cur_rect.x0 - prev_rect.x1
                prev_seg_type = str(((prev_unit.metadata or {}).get("segment_type") or "")).strip().lower()
                cur_seg_type = str(((cur_unit.metadata or {}).get("segment_type") or "")).strip().lower()
                large_gap = gap is not None and gap > 18.0
                repeated_cluster_start = prev_seg_type == "page" and cur_seg_type == "label"
                protected_edge = any(
                    bool(candidate.protected_inline or candidate.immutable)
                    for candidate in (prev_unit, cur_unit)
                )
                anchor_change = (
                    str(prev_unit.anchor_horizontal or "").strip().lower()
                    and str(cur_unit.anchor_horizontal or "").strip().lower()
                    and str(prev_unit.anchor_horizontal or "").strip().lower() != str(cur_unit.anchor_horizontal or "").strip().lower()
                )
                if large_gap and (repeated_cluster_start or protected_edge or anchor_change):
                    islands.append([item])
                else:
                    islands[-1].append(item)

            for island in islands:
                anchor_x = None
                if island and isinstance(island[0]["rect"], fitz.Rect):
                    anchor_x = island[0]["rect"].x0
                for pos, item in enumerate(island):
                    unit = item["unit"]
                    meta = dict(unit.metadata or {})
                    if anchor_x is not None:
                        meta["row_anchor_x"] = anchor_x
                    # Marqueurs annotés (➊➋➌ etc.) : atomiques, non-wrappables, protégés.
                    is_marker = self._is_annotation_marker(unit.text_source) or self._is_annotation_marker(unit.text_translated)
                    canonical.append(
                        replace(
                            unit,
                            line_indices=[final_row_idx],
                            continuation_before=(pos > 0),
                            continuation_after=(pos < len(island) - 1),
                            hard_break_before=(pos == 0),
                            hard_break_after=(pos == len(island) - 1),
                            keep_with_previous=(pos > 0),
                            keep_with_next=(pos < len(island) - 1),
                            paragraph_id=f"{str((block or {}).get('id') or 'block')}:final_row:{final_row_idx}",
                            metadata=meta,
                            # Forcer les attributs de protection sur les marqueurs
                            immutable=unit.immutable or is_marker,
                            protected_inline=unit.protected_inline or is_marker,
                            reflowable=(False if is_marker else unit.reflowable),
                            justification_eligible=(False if is_marker else unit.justification_eligible),
                            group_class=("annotation_marker" if is_marker else unit.group_class),
                        )
                    )
                final_row_idx += 1
        return canonical

    def _relation_from_units(self, prev_unit, unit):
        if unit.keep_with_previous:
            return "KEEP_WITH_PREVIOUS", True, 1.0
        if unit.hard_break_before:
            return "NEW_PARAGRAPH", True, 1.0
        if unit.paragraph_id and prev_unit.paragraph_id and unit.paragraph_id != prev_unit.paragraph_id:
            return "NEW_PARAGRAPH", True, 0.95
        prev_line = prev_unit.line_indices[-1] if prev_unit.line_indices else 0
        cur_line = unit.line_indices[0] if unit.line_indices else prev_line
        if unit.continuation_before:
            return "CONTINUE_INLINE", False, 0.8
        if cur_line > prev_line:
            return "NEW_LINE", True, 0.9
        return "CONTINUE_INLINE", False, 0.6

    def _build_reconstruction_graph(self, units):
        edges = []
        for prev_unit, unit in zip(units, units[1:]):
            relation, hard, weight = self._relation_from_units(prev_unit, unit)
            edges.append(GraphEdge(prev_unit.unit_id, unit.unit_id, relation, hard, weight))
        return edges

    def _source_layout_paragraph_for_line(self, source_layout_mode, line_index):
        try:
            target = int(line_index)
        except Exception:
            target = 0
        paragraph_index = 0
        for item in source_layout_mode.get("line_breaks") or []:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("line_index", 0) or 0)
            except Exception:
                idx = 0
            if idx >= target:
                break
            if str(item.get("after") or "").strip().lower() == "paragraph_break":
                paragraph_index += 1
        return paragraph_index

    def _apply_source_layout_mode_to_units(self, block, units):
        if not units:
            return units
        mode = dict((block or {}).get("source_layout_mode") or {})
        line_flow = str(mode.get("line_flow") or "").strip().lower()
        render_contract = str(mode.get("render_contract") or "").strip().lower()
        if not line_flow:
            return units

        block_id = str((block or {}).get("id") or "block")
        preserve_lines = bool(mode.get("preserve_line_breaks")) or line_flow in {"fixed_lines", "preserve_line_breaks"}
        can_reflow = bool(mode.get("can_reflow_within_paragraph", True))
        updated = []
        previous_paragraph = None
        for idx, unit in enumerate(units):
            line_indices = list(unit.line_indices or [idx])
            first_line = int(line_indices[0]) if line_indices else idx
            paragraph_index = (
                first_line
                if preserve_lines
                else self._source_layout_paragraph_for_line(mode, first_line)
            )
            paragraph_id = f"{block_id}:source_layout:{paragraph_index}"
            paragraph_changed = previous_paragraph is not None and paragraph_index != previous_paragraph
            metadata = dict(unit.metadata or {})
            metadata["source_layout_mode"] = {
                "mode": mode.get("mode"),
                "line_flow": line_flow,
                "render_contract": render_contract,
                "paragraph_index": paragraph_index,
            }
            if preserve_lines:
                updated_unit = replace(
                    unit,
                    paragraph_id=paragraph_id,
                    hard_break_before=bool(idx == 0 or first_line != (updated[-1].line_indices[-1] if updated and updated[-1].line_indices else first_line)),
                    continuation_before=False,
                    reflowable=bool(can_reflow and line_flow != "fixed_lines" and unit.reflowable),
                    metadata=metadata,
                )
            else:
                updated_unit = replace(
                    unit,
                    paragraph_id=paragraph_id,
                    hard_break_before=bool(idx == 0 or paragraph_changed),
                    continuation_before=bool(idx > 0 and not paragraph_changed),
                    reflowable=bool(can_reflow and unit.reflowable),
                    metadata=metadata,
                )
            updated.append(updated_unit)
            previous_paragraph = paragraph_index
        return updated

    def _build_block_reconstruction_plan(self, page, page_data, block, target_lang):
        block_type = self._classify_block_for_reconstruction(block, page_data)
        geometry_ctx = self._build_block_geometry_context(page, page_data, block)
        line_templates = self._build_line_templates(block, geometry_ctx)
        semantic_payload = self._collect_block_semantic_payload(block)
        units = self._normalize_placable_units(block, semantic_payload, target_lang, page_data=page_data)
        units = self._apply_source_layout_mode_to_units(block, units)
        graph_edges = self._build_reconstruction_graph(units)
        block_rect = self._fitz_rect_from_bbox_like((block or {}).get("bbox"))
        block_rect_tuple = (
            (block_rect.x0, block_rect.y0, block_rect.x1, block_rect.y1)
            if isinstance(block_rect, fitz.Rect) and block_rect.get_area() > 0
            else geometry_ctx.block_bbox
        )
        constraints = dict(geometry_ctx.constraints or {})
        descriptor_group_ids = dict((block or {}).get("descriptor_group_ids") or {})
        descriptor_page_organization = dict((block or {}).get("descriptor_page_organization") or {})
        row_id = str(descriptor_group_ids.get("table_row_group_id") or "").strip()
        cell_id = str(descriptor_group_ids.get("cell_id") or "").strip()
        if row_id:
            for row in descriptor_page_organization.get("table_row_groups") or []:
                if str((row or {}).get("id") or "") != row_id:
                    continue
                rect = None
                if cell_id:
                    for cell in (row or {}).get("cells") or []:
                        if str((cell or {}).get("id") or "") == cell_id or str((cell or {}).get("block_id") or "") == str((block or {}).get("id") or ""):
                            rect = self._fitz_rect_from_bbox_like((cell or {}).get("bbox"))
                            break
                if rect is None:
                    rect = self._fitz_rect_from_bbox_like((row or {}).get("bbox"))
                if isinstance(rect, fitz.Rect) and rect.get_area() > 0:
                    constraints["table_cell_bbox"] = (rect.x0, rect.y0, rect.x1, rect.y1)
                break
        overlay_matches = []
        for ov in (page_data or {}).get("immutable_overlays") or []:
            rect = self._fitz_rect_from_bbox_like((ov or {}).get("bbox")) if isinstance(ov, dict) else None
            if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                continue
            if (rect & fitz.Rect(block_rect_tuple)).get_area() <= 0:
                continue
            overlay_matches.append(dict(ov))
        if overlay_matches:
            constraints["matching_immutable_overlays"] = overlay_matches
        alignment = self._normalize_alignment((block or {}).get("alignment") or "left")
        return BlockReconstructionPlan(
            block_id=str((block or {}).get("id") or ""),
            page_index=int(getattr(page, "number", 0)),
            block_type=block_type,
            block_role=str((block or {}).get("role") or "body"),
            block_bbox=block_rect_tuple,
            block_bbox_pt=block_rect_tuple,
            container_bbox=geometry_ctx.container_bbox,
            writing_direction="right_to_left" if str(target_lang or "").strip().lower() in {"ar", "he", "fa"} else "left_to_right",
            block_progression="top_to_bottom",
            alignment=alignment,
            paragraph_alignment=alignment,
            padding_left=geometry_ctx.padding_left,
            padding_right=geometry_ctx.padding_right,
            padding_top=geometry_ctx.padding_top,
            padding_bottom=geometry_ctx.padding_bottom,
            protected_regions=list(geometry_ctx.protected_regions),
            background_strategy=geometry_ctx.background_strategy,
            background_color=geometry_ctx.background_color,
            line_templates=line_templates,
            units=units,
            graph_edges=graph_edges,
            positioning_policy=dict((block or {}).get("positioning_policy") or {}),
            relative_geometry=dict((block or {}).get("relative_geometry") or {}),
            editorial_semantics=dict((block or {}).get("editorial_semantics") or {}),
            editorial_relations=dict((block or {}).get("editorial_relations") or {}),
            source_layout_mode=dict((block or {}).get("source_layout_mode") or {}),
            adaptive_profile=dict((geometry_ctx.constraints or {}).get("adaptive_profile") or {}),
            constraints=constraints,
            source_block=dict(block or {}),
        )

    def _block_supported_by_hierarchical_engine(self, block, page_data):
        if not self.hierarchical_reconstruction_mode or not isinstance(block, dict):
            return False
        block_type = self._classify_block_for_reconstruction(block, page_data)
        if block_type == "code":
            return True
        if block_type in {"editorial", "heading", "caption", "annotation", "table"}:
            return self._has_translated_payload(block)
        return False

    def _style_for_bbox_from_blocks(self, page, page_data, bbox):
        target = self._fitz_rect_from_bbox_like(bbox)
        if not isinstance(target, fitz.Rect) or target.get_area() <= 0:
            return {"font": "helv", "size": 12.0, "color": "#000000", "flags": {}}
        best = None
        best_key = None
        for block in (page_data or {}).get("blocks") or []:
            block_rect = self._fitz_rect_from_bbox_like((block or {}).get("bbox"))
            if not isinstance(block_rect, fitz.Rect) or block_rect.get_area() <= 0:
                continue
            overlap = (target & block_rect).get_area()
            cx = (block_rect.x0 + block_rect.x1) / 2.0
            cy = (block_rect.y0 + block_rect.y1) / 2.0
            tx = (target.x0 + target.x1) / 2.0
            ty = (target.y0 + target.y1) / 2.0
            distance = math.hypot(cx - tx, cy - ty)
            key = (-overlap, distance, abs(block_rect.height - target.height))
            if best_key is None or key < best_key:
                best_key = key
                best = block
        return self._style_from_block(best or {})

    def _wrap_text_for_bbox(self, page, style, text, max_width):
        text = self._clean_text_for_render(text)
        if not text:
            return []
        tokens = [token for token in re.findall(r"\S+", text) if token]
        if not tokens:
            return [text]
        lines = []
        current = []
        for token in tokens:
            candidate = " ".join(current + [token]).strip()
            _, fontfile, _, fontname = self._resolve_style_font( page, style, text=candidate)
            fontsize = float(style.get("size") or 12.0)
            width = self._measure_text_width( candidate, fontsize, fontname, fontfile)
            if current and width > max_width:
                lines.append(" ".join(current))
                current = [token]
            else:
                current.append(token)
        if current:
            lines.append(" ".join(current))
        return lines or [text]

    def _select_block_renderer(self, plan):
        if plan.block_type == "editorial":
            return EditorialBlockRenderer(self)
        if plan.block_type == "heading":
            return HeadingBlockRenderer(self)
        if plan.block_type == "caption":
            return CaptionBlockRenderer(self)
        if plan.block_type == "annotation":
            return AnnotationBlockRenderer(self)
        if plan.block_type == "table":
            return TableBlockRenderer(self)
        if plan.block_type == "code":
            return CodeBlockRenderer(self)
        return None

    def _render_hierarchical_block_plan(self, page, plan):
        renderer = self._select_block_renderer(plan)
        if renderer is None:
            return []
        return renderer.render(page, plan)

    def _try_render_hierarchical_item_plan(self, page, item, hierarchical_plans, rendered_block_ids, forbidden_rects=None, debug_store=None):
        source_block_id = str((item or {}).get("source_block_id") or "")
        if not source_block_id or source_block_id not in hierarchical_plans:
            return False
        if source_block_id in rendered_block_ids:
            return True
        plan = hierarchical_plans[source_block_id]
        ops = self._render_hierarchical_block_plan(page, plan)
        findings = self._validate_block_layout(plan, ops)
        severe_types = {"overflow", "text_overlap", "protected_overlap"}
        if any(str((finding or {}).get("type") or "").strip().lower() in severe_types for finding in findings):
            if debug_store is not None:
                page_key = int(page.number)
                slot = debug_store.setdefault(page_key, {"blue": [], "red": [], "findings": []})
                for finding in findings:
                    slot.setdefault("findings", []).append({"kind": "hierarchical_layout_fallback", "finding": finding})
            return False
        self._commit_block_draw_ops(page, ops)
        rendered_block_ids.add(source_block_id)
        if forbidden_rects is not None:
            rect = self._fitz_rect_from_bbox_like(plan.block_bbox)
            if isinstance(rect, fitz.Rect) and rect.get_area() > 0:
                forbidden_rects.append(rect)
        return True

    def _validate_block_layout(self, plan, ops):
        findings = []
        block_rect = fitz.Rect(plan.block_bbox)
        tolerance = 2.0
        text_rects = []
        protected_rects = []
        for region in plan.protected_regions or []:
            rect = self._fitz_rect_from_bbox_like((region or {}).get("bbox"))
            if isinstance(rect, fitz.Rect) and rect.get_area() > 0:
                protected_rects.append(rect)
        for op in ops or []:
            rect = fitz.Rect(op.bbox) if isinstance(op.bbox, (list, tuple)) and len(op.bbox) == 4 else None
            if op.op_type.startswith("draw_text") and isinstance(rect, fitz.Rect) and rect.get_area() > 0:
                if (
                    rect.x0 < block_rect.x0 - tolerance
                    or rect.x1 > block_rect.x1 + tolerance
                    or rect.y0 < block_rect.y0 - tolerance
                    or rect.y1 > block_rect.y1 + tolerance
                ):
                    findings.append({"type": "overflow", "unit_id": op.unit_id, "bbox": tuple(rect)})
                for prev_rect in text_rects:
                    if (rect & prev_rect).get_area() > 0.5:
                        findings.append({"type": "text_overlap", "unit_id": op.unit_id, "bbox": tuple(rect)})
                        break
                for protected_rect in protected_rects:
                    if (rect & protected_rect).get_area() > 0.5:
                        findings.append({"type": "protected_overlap", "unit_id": op.unit_id, "bbox": tuple(rect)})
                        break
                text_rects.append(rect)
        return findings

    def _prune_block_draw_ops(self, plan, ops):
        block_rect = fitz.Rect(plan.block_bbox)
        tolerance = 2.0
        protected_rects = []
        for region in plan.protected_regions or []:
            rect = self._fitz_rect_from_bbox_like((region or {}).get("bbox"))
            if isinstance(rect, fitz.Rect) and rect.get_area() > 0:
                protected_rects.append(rect)

        ancillary_ops = []
        kept_text_ops = []
        kept_text_rects = []
        for op in ops or []:
            if op.op_type != "draw_text_run":
                ancillary_ops.append(op)
                continue
            rect = fitz.Rect(op.bbox) if isinstance(op.bbox, (list, tuple)) and len(op.bbox) == 4 else None
            text = self._clean_text_for_render(op.text or "")
            if not text or not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                continue
            if (
                rect.x0 < block_rect.x0 - tolerance
                or rect.x1 > block_rect.x1 + tolerance
                or rect.y0 < block_rect.y0 - tolerance
                or rect.y1 > block_rect.y1 + tolerance
            ):
                continue
            if any((rect & prev_rect).get_area() > 0.5 for prev_rect in kept_text_rects):
                continue
            if any((rect & protected_rect).get_area() > 0.5 for protected_rect in protected_rects):
                continue
            kept_text_ops.append(op)
            kept_text_rects.append(rect)

        overlay_ops = [op for op in ancillary_ops if op.op_type == "draw_overlay_image"]
        if not kept_text_ops:
            return overlay_ops

        pruned_ops = []
        text_op_ids = {id(op) for op in kept_text_ops}
        for op in ancillary_ops:
            if op.op_type == "erase_rect" or op.op_type == "draw_overlay_image":
                pruned_ops.append(op)
        for op in ops or []:
            if op.op_type == "draw_text_run" and id(op) in text_op_ids:
                pruned_ops.append(op)
        return pruned_ops

    def _normalized_rgb(self, rgb):
        if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
            return (0, 0, 0)
        values = []
        for channel in rgb:
            channel = float(channel)
            if channel > 1.0:
                channel = channel / 255.0
            values.append(max(0.0, min(1.0, channel)))
        return tuple(values)

    def _commit_block_draw_ops(self, page, ops):
        for op in ops or []:
            rect = fitz.Rect(op.bbox) if isinstance(op.bbox, (list, tuple)) and len(op.bbox) == 4 else None
            if op.op_type == "erase_rect" and isinstance(rect, fitz.Rect):
                page.draw_rect(rect, color=None, fill=(1, 1, 1), overlay=True)
                continue
            if op.op_type == "draw_overlay_image" and isinstance(rect, fitz.Rect):
                path = (op.metadata or {}).get("path")
                if path and os.path.exists(path):
                    page.insert_image(rect, filename=path, overlay=True)
                continue
            if op.op_type == "draw_text_run":
                if not (op.text or "").strip():
                    continue
                point = tuple((op.metadata or {}).get("point") or (rect.x0 if rect else 0.0, rect.y1 if rect else 0.0))
                style = dict(op.style or {})
                fontname = (op.metadata or {}).get("fontname") or style.get("font") or "helv"
                fontfile = (op.metadata or {}).get("fontfile")
                builtin = bool((op.metadata or {}).get("builtin"))
                fontsize = float((op.metadata or {}).get("fontsize") or style.get("size") or 12.0)
                rgb = self._normalized_rgb((op.metadata or {}).get("rgb") or self._resolve_text_color( style, None))
                insert_kwargs = {
                    "fontsize": fontsize,
                    "color": rgb,
                    "overlay": True,
                }
                try:
                    if fontfile and not builtin:
                        page.insert_text(point, op.text or "", fontname=fontname, fontfile=fontfile, **insert_kwargs)
                    else:
                        page.insert_text(point, op.text or "", fontname=fontname, **insert_kwargs)
                except Exception:
                    page.insert_text(point, op.text or "", fontname="helv", **insert_kwargs)

    def _clean_page_background_path(self, page_data):
        path = str((page_data or {}).get("background_path") or "").strip()
        if path and os.path.exists(path):
            return path
        return None

    def _page_background_path(self, page_data):
        for key in ("background_path", "source_image_path"):
            path = str((page_data or {}).get(key) or "").strip()
            if path and os.path.exists(path):
                return path
        return None

    def _page_size_pt(self, page_data):
        dims = dict((page_data or {}).get("dimensions") or {})
        width_px = float(dims.get("width") or dims.get("page_width") or 0.0)
        height_px = float(dims.get("height") or dims.get("page_height") or 0.0)
        bg = self._page_background_path(page_data)
        if bg:
            try:
                pix = fitz.Pixmap(bg)
                width_px = width_px or float(pix.width)
                height_px = height_px or float(pix.height)
            except Exception:
                pass
        width_px = width_px or 1240.0
        height_px = height_px or 1754.0
        return (width_px * self.pixel_to_point, height_px * self.pixel_to_point)

    def _insert_page_background(self, page, page_data):
        bg = self._page_background_path(page_data)
        if bg and os.path.exists(bg):
            page.insert_image(page.rect, filename=bg, overlay=False)

    def _render_page_debug_image(self, page, output_path, page_number):
        if not self.layout_debug_overlay:
            return
        debug_path = Path(output_path).with_name(f"{Path(output_path).stem}_layout_debug_p{page_number}.jpg")
        pix = page.get_pixmap(dpi=150, alpha=False)
        pix.save(str(debug_path))

    def _page_target_lang(self, structure, page_data):
        return str(
            (page_data or {}).get("target_lang")
            or (page_data or {}).get("translation_target_lang")
            or (structure or {}).get("target_lang")
            or "fr"
        ).strip().lower()

    def _render_plan_with_validation(self, page, plan):
        ops = self._render_hierarchical_block_plan(page, plan)
        findings = self._validate_block_layout(plan, ops)
        severe = {"overflow", "text_overlap", "protected_overlap"}
        if any(str((finding or {}).get("type") or "").strip().lower() in severe for finding in findings):
            pruned_ops = self._prune_block_draw_ops(plan, ops)
            if pruned_ops:
                pruned_findings = self._validate_block_layout(plan, pruned_ops)
                if not any(str((finding or {}).get("type") or "").strip().lower() in severe for finding in pruned_findings):
                    return pruned_ops, pruned_findings
            return [], findings
        return ops, findings

    def _translated_coverage_entries_for_block(self, block, target_lang, page_data=None):
        semantic_payload = self._collect_block_semantic_payload(block)
        phrase_units = self._phrase_units(block, semantic_payload, target_lang, page_data=page_data)
        orphan_units = self._orphan_semantic_units(block, semantic_payload, target_lang, phrase_units, page_data=page_data) if phrase_units else []
        entries = []
        used_ids = set()
        used_text_bbox = set()

        def _entry_from_unit(unit):
            text = self._clean_text_for_render(unit.text_translated or "")
            if not text:
                return None
            bbox = unit.relative_bbox
            if not bbox and unit.line_indices:
                lines = list((block or {}).get("lines") or [])
                idx = unit.line_indices[0]
                if 0 <= idx < len(lines):
                    bbox = (lines[idx] or {}).get("bbox")
            if not bbox:
                bbox = (block or {}).get("bbox")
            return {
                "unit_id": unit.unit_id,
                "text": text,
                "source_text": unit.text_source,
                "bbox": bbox,
                "style": dict(unit.style or {}),
                "unit_type": unit.unit_type,
                "line_indices": list(unit.line_indices or []),
                "render_policy": unit.render_policy,
                "segment_type": unit.group_class or unit.inline_class or unit.unit_type,
                "alignment": unit.anchor_horizontal or "left",
            }

        for unit in list(phrase_units or []) + list(orphan_units or []):
            if unit.unit_id in used_ids:
                continue
            entry = _entry_from_unit(unit)
            if entry:
                entries.append(entry)
                used_ids.add(unit.unit_id)
                bbox = entry.get("bbox") or ()
                used_text_bbox.add((entry.get("text"), tuple(round(float(v), 2) for v in bbox) if isinstance(bbox, (list, tuple)) else bbox))
        if page_data is not None and not self._has_translated_payload(block):
            aux_entries = self._aux_coverage_entries_for_block(page_data, block)
        else:
            aux_entries = []
        for entry in aux_entries:
            bbox = entry.get("bbox") or ()
            key = (entry.get("text"), tuple(round(float(v), 2) for v in bbox) if isinstance(bbox, (list, tuple)) else bbox)
            if key in used_text_bbox:
                continue
            entries.append(entry)
            used_text_bbox.add(key)
        if entries:
            return entries

        line_entries = []
        for idx, line in enumerate((block or {}).get("lines") or []):
            text = self._line_translated_text(line)
            if not text:
                continue
            line_entries.append(
                {
                    "unit_id": f"{(block or {}).get('id') or 'block'}:coverage:{idx}",
                    "text": text,
                    "bbox": (line or {}).get("bbox"),
                    "style": self._merge_styles((line or {}).get("style") or {}, self._style_from_block(block)),
                    "unit_type": "line",
                    "line_indices": [idx],
                    "render_policy": str((block or {}).get("render_policy") or ""),
                }
            )
        if line_entries:
            return line_entries

        block_text = self._clean_text_for_render((block or {}).get("translated_text") or "")
        if block_text:
            return [
                {
                    "unit_id": f"{(block or {}).get('id') or 'block'}:coverage:block",
                    "text": block_text,
                    "bbox": (block or {}).get("bbox"),
                    "style": self._style_from_block(block),
                    "unit_type": "block",
                    "line_indices": [],
                    "render_policy": str((block or {}).get("render_policy") or ""),
                }
            ]
        return []

    def _expected_block_text_units(self, block, target_lang, page_data=None):
        return len(self._translated_coverage_entries_for_block(block, target_lang, page_data=page_data))

    def _render_block_presence_fallback_ops(self, page, page_data, block, target_lang, font_scale=1.0):
        block_rect = self._fitz_rect_from_bbox_like((block or {}).get("bbox"))
        if not isinstance(block_rect, fitz.Rect) or block_rect.get_area() <= 0:
            return []
        adaptive_profile = self._block_adaptive_profile(block, page_data=page_data)
        entries = self._translated_coverage_entries_for_block(block, target_lang, page_data=page_data)
        if not entries:
            return []
        # N'efface que si le fond n'est pas déjà propre — un source_image_path brut
        # ne compte pas comme fond propre, sinon on réintroduit le texte source.
        has_clean_background = bool(self._clean_page_background_path(page_data))
        ops = []
        if not has_clean_background:
            ops.append(
                BlockRenderOp(
                    "erase_rect",
                    str((block or {}).get("id") or "coverage"),
                    None,
                    bbox=(block_rect.x0, block_rect.y0, block_rect.x1, block_rect.y1),
                    z_index=0,
                )
            )
        usable_left = block_rect.x0
        usable_right = block_rect.x1
        top = block_rect.y0
        bottom = block_rect.y1
        slot_h = max(6.0, (bottom - top) / max(1, len(entries)))
        for idx, entry in enumerate(entries):
            text = self._clean_text_for_render(entry.get("text") or "")
            if not text:
                continue
            bbox = entry.get("bbox")
            rect = None
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                raw_rect = fitz.Rect(bbox)
                if raw_rect.get_area() > 0 and (raw_rect & block_rect).get_area() > 0.5:
                    rect = raw_rect
            if rect is None:
                rect = self._fitz_rect_from_bbox_like(bbox)
            if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                y0 = min(bottom - slot_h, top + idx * slot_h)
                rect = fitz.Rect(usable_left, y0, usable_right, min(bottom, y0 + slot_h))
            rect = fitz.Rect(
                max(usable_left, rect.x0),
                max(top, rect.y0),
                min(usable_right, rect.x1),
                min(bottom, rect.y1),
            )
            if rect.height <= 0 or rect.width <= 0:
                continue
            style = self._merge_styles(entry.get("style") or {}, self._style_from_block(block))
            _, fontfile, builtin, fontname = self._resolve_style_font( page, style, text=text)
            fontsize = min(float(style.get("size") or 12.0) * max(0.4, float(font_scale)), max(4.5, rect.height * 0.72))
            wrapped = self._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            while fontsize > 4.5 and wrapped and (len(wrapped) * max(4.8, fontsize * 1.05)) > max(rect.height, fontsize * 1.15):
                fontsize -= 0.5
                wrapped = self._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            spacing_factor = max(0.84, min(1.05, float(adaptive_profile.get("line_spacing_factor") or 1.0)))
            line_h = max(4.8, min(rect.height / max(1, len(wrapped)), fontsize * 1.08 * spacing_factor))
            rgb = self._resolve_text_color( style, block)
            for line_idx, line_text in enumerate(wrapped):
                cur_size = fontsize
                while cur_size > 4.5:
                    width = self._measure_text_width( line_text, cur_size, fontname, fontfile)
                    if width <= max(8.0, rect.width):
                        break
                    cur_size -= 0.5
                baseline = rect.y0 + min(rect.height - 1.0, (line_idx + 1) * line_h * 0.82)
                width = self._measure_text_width( line_text, cur_size, fontname, fontfile)
                x = rect.x0
                align = self._normalize_alignment(entry.get("alignment") or (block or {}).get("alignment") or "left")
                if align == "center":
                    x = max(rect.x0, rect.x0 + max(0.0, (rect.width - width) / 2.0))
                elif align == "right":
                    x = max(rect.x0, rect.x1 - width)
                text_rect = fitz.Rect(x, baseline - max(1.0, cur_size * 0.82), min(rect.x1, x + width), baseline + max(1.0, cur_size * 0.18))
                ops.append(
                    BlockRenderOp(
                        op_type="draw_text_run",
                        block_id=str((block or {}).get("id") or "coverage"),
                        unit_id=f"{entry['unit_id']}:{line_idx}",
                        bbox=(text_rect.x0, text_rect.y0, text_rect.x1, text_rect.y1),
                        text=line_text,
                        style={**style, "size": cur_size},
                        z_index=10,
                        metadata={
                            "point": (x, baseline),
                            "fontname": fontname,
                            "fontfile": fontfile,
                            "builtin": builtin,
                            "fontsize": cur_size,
                            "rgb": rgb,
                        },
                    )
                )
        return ops

    def _validated_block_presence_fallback_ops(self, page, page_data, block, target_lang, plan=None):
        plan = plan or self._build_block_reconstruction_plan(page, page_data, block, target_lang)
        severe = {"overflow", "text_overlap", "protected_overlap"}
        best_ops = []
        best_text_ops = 0
        scale_ladder = tuple((plan.adaptive_profile or {}).get("fallback_scales") or (1.0, 0.9, 0.8, 0.7, 0.6))
        for scale in scale_ladder:
            ops = self._render_block_presence_fallback_ops(page, page_data, block, target_lang, font_scale=scale)
            if not ops:
                continue
            findings = self._validate_block_layout(plan, ops)
            if not any(str((finding or {}).get("type") or "").strip().lower() in severe for finding in findings):
                return ops
            pruned_ops = self._prune_block_draw_ops(plan, ops)
            text_ops = sum(1 for op in pruned_ops if op.op_type == "draw_text_run")
            if text_ops > best_text_ops:
                best_ops = pruned_ops
                best_text_ops = text_ops
        return best_ops

    def reconstruct(self, structure, output_path):
        pages = list((structure or {}).get("pages") or [])
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        try:
            for page_index, page_data in enumerate(pages):
                width_pt, height_pt = self._page_size_pt(page_data)
                page = doc.new_page(width=width_pt, height=height_pt)
                self._insert_page_background(page, page_data)
                target_lang = self._page_target_lang(structure, page_data)
                self._build_page_reconstruction_context(page_data, target_lang)
                rendered_block_ids = set()
                rendered_block_stats = {}
                for block in self._iter_renderable_blocks(page_data):
                    if not self._block_supported_by_hierarchical_engine(block, page_data):
                        rendered_block_stats[str((block or {}).get("id") or "")] = {
                            "committed": False,
                            "text_ops": 0,
                            "expected_units": self._expected_block_text_units(block, target_lang, page_data=page_data),
                            "block": block,
                        }
                        continue
                    plan = self._build_block_reconstruction_plan(page, page_data, block, target_lang)
                    ops, findings = self._render_plan_with_validation(page, plan)
                    text_ops = sum(1 for op in ops if op.op_type == "draw_text_run")
                    expected_units = self._expected_block_text_units(block, target_lang, page_data=page_data)
                    if (findings and not ops) or (expected_units > 0 and text_ops < expected_units):
                        fallback_ops = self._validated_block_presence_fallback_ops(page, page_data, block, target_lang, plan=plan)
                        fallback_text_ops = sum(1 for op in fallback_ops if op.op_type == "draw_text_run")
                        if fallback_text_ops > text_ops:
                            ops = fallback_ops
                            text_ops = fallback_text_ops
                            findings = []
                    if findings and not ops:
                        rendered_block_stats[plan.block_id] = {
                            "committed": False,
                            "text_ops": 0,
                            "expected_units": expected_units,
                            "block": block,
                        }
                        continue
                    self._commit_block_draw_ops(page, ops)
                    rendered_block_ids.add(plan.block_id)
                    rendered_block_stats[plan.block_id] = {
                        "committed": True,
                        "text_ops": text_ops,
                        "expected_units": expected_units,
                        "block": block,
                    }
                for block in self._iter_renderable_blocks(page_data):
                    block_id = str((block or {}).get("id") or "")
                    stats = rendered_block_stats.get(block_id) or {}
                    expected_units = int(stats.get("expected_units") or self._expected_block_text_units(block, target_lang, page_data=page_data))
                    committed = bool(stats.get("committed"))
                    text_ops = int(stats.get("text_ops") or 0)
                    if expected_units <= 0:
                        continue
                    if committed and text_ops >= expected_units:
                        continue
                    fallback_ops = self._validated_block_presence_fallback_ops(page, page_data, block, target_lang)
                    fallback_text_ops = sum(1 for op in fallback_ops if op.op_type == "draw_text_run")
                    if fallback_text_ops > text_ops:
                        self._commit_block_draw_ops(page, fallback_ops)
                self._render_page_debug_image(page, output_path, page_index + 1)
            doc.save(str(output_path))
        finally:
            doc.close()


class BaseBlockRenderer:
    def __init__(self, reconstructor: DocumentReconstructor):
        self.reconstructor = reconstructor

    def _block_rect(self, plan):
        return fitz.Rect(plan.block_bbox)

    def _resolve_style(self, page, plan, text, style_override=None):
        style = self.reconstructor._merge_styles(style_override or {}, self.reconstructor._style_from_block(plan.source_block))
        _, fontfile, builtin, fontname = self.reconstructor._resolve_style_font( page, style, text=text)
        fontsize = float(style.get("size") or 12.0)
        rgb = self.reconstructor._resolve_text_color( style, plan.source_block)
        return style, fontfile, builtin, fontname, fontsize, rgb

    def _fit_text_run_to_block(self, plan, rect, point, fontsize):
        block_rect = self._block_rect(plan)
        rect = fitz.Rect(rect)
        ascent = max(1.0, float(fontsize) * 0.82)
        descent = max(1.0, float(fontsize) * 0.18)
        baseline_min = block_rect.y0 + ascent
        baseline_max = block_rect.y1 - descent
        if baseline_max < baseline_min:
            return None, None
        x = min(max(float(point[0]), block_rect.x0), max(block_rect.x0, block_rect.x1 - 1.0))
        baseline = min(max(float(point[1]), baseline_min), baseline_max)
        fitted = fitz.Rect(
            max(block_rect.x0, rect.x0),
            max(block_rect.y0, baseline - ascent),
            min(block_rect.x1, rect.x1),
            min(block_rect.y1, baseline + descent),
        )
        if fitted.width <= 0 or fitted.height <= 0:
            return None, None
        return fitted, (x, baseline)

    def _emit_text_run(self, plan, text, rect, point, style, fontname, fontfile, builtin, fontsize, rgb, unit_id=None):
        fitted_rect, fitted_point = self._fit_text_run_to_block(plan, rect, point, fontsize)
        if fitted_rect is None or fitted_point is None:
            return BlockRenderOp(
                op_type="draw_text_run",
                block_id=plan.block_id,
                unit_id=unit_id,
                bbox=None,
                text="",
                style=style,
                z_index=10,
                metadata={},
            )
        return BlockRenderOp(
            op_type="draw_text_run",
            block_id=plan.block_id,
            unit_id=unit_id,
            bbox=(fitted_rect.x0, fitted_rect.y0, fitted_rect.x1, fitted_rect.y1),
            text=text,
            style=style,
            z_index=10,
            metadata={
                "point": fitted_point,
                "fontname": fontname,
                "fontfile": fontfile,
                "builtin": builtin,
                "fontsize": fontsize,
                "rgb": rgb,
            },
        )

    def _overlay_ops_for_matching_immutable_overlays(self, plan):
        ops = []
        for ov in plan.constraints.get("matching_immutable_overlays") or []:
            path = str((ov or {}).get("path") or "").strip()
            rect = self.reconstructor._fitz_rect_from_bbox_like((ov or {}).get("bbox"))
            if not path or not os.path.exists(path) or not isinstance(rect, fitz.Rect):
                continue
            ops.append(
                BlockRenderOp(
                    op_type="draw_overlay_image",
                    block_id=plan.block_id,
                    unit_id=None,
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    z_index=20,
                    metadata={"path": path},
                )
            )
        return ops


class EditorialBlockRenderer(BaseBlockRenderer):
    def _edge_map(self, plan):
        return {(edge.source_id, edge.target_id): edge for edge in (plan.graph_edges or [])}

    def _should_render_relative_slot_mode(self, plan):
        units = [unit for unit in (plan.units or []) if self.reconstructor._clean_text_for_render(unit.text_translated or unit.text_source)]
        if not units:
            return False
        if not all(unit.relative_bbox for unit in units):
            return False
        if any(self.reconstructor._unit_render_tuning(unit, plan).get("prefer_bbox_anchor") for unit in units):
            return False
        return all(str(unit.render_policy or "").strip().lower() == "external_flow" for unit in units)

    def _should_render_bbox_anchored(self, plan):
        units = [unit for unit in (plan.units or []) if self.reconstructor._clean_text_for_render(unit.text_translated or unit.text_source)]
        if not units:
            return False
        if any(
            str(unit.unit_type or "").strip().lower() == "translated_line"
            for unit in units
        ):
            return False
        if not all(unit.relative_bbox for unit in units):
            return False
        anchored_count = 0
        for unit in units:
            policy = str(unit.render_policy or "").strip().lower()
            tuning = self.reconstructor._unit_render_tuning(unit, plan)
            if policy in {"anchored_external", "anchored_text", "fixed_preserve"} or not unit.reflowable or tuning.get("prefer_bbox_anchor"):
                anchored_count += 1
        return anchored_count == len(units)

    def _prepare_templates(self, plan):
        templates = [LineTemplate(**template.__dict__) for template in (plan.line_templates or [])]
        if not templates:
            return templates
        prev_relation = str(((plan.editorial_relations or {}).get("with_previous") or {}).get("relation") or "").strip().lower()
        if prev_relation == "heading_to_body":
            block_rect = fitz.Rect(plan.block_bbox)
            first = templates[0]
            first_height = max(8.0, first.bbox[3] - first.bbox[1])
            new_top = block_rect.y0 + max(1.5, plan.padding_top * 0.35)
            new_bottom = min(block_rect.y1, new_top + first_height)
            templates[0] = LineTemplate(
                line_id=first.line_id,
                source_line_indices=list(first.source_line_indices),
                bbox=(first.bbox[0], new_top, first.bbox[2], new_bottom),
                baseline_y=new_top + min(first.ascent, max(1.0, first_height - 1.0)),
                ascent=first.ascent,
                descent=first.descent,
                left_x=first.left_x,
                right_x=first.right_x,
                usable_width=first.usable_width,
                indent_px=first.indent_px,
                first_line_indent_px=first.first_line_indent_px,
                alignment=first.alignment,
                paragraph_id=first.paragraph_id,
                paragraph_index=first.paragraph_index,
                line_index_in_paragraph=first.line_index_in_paragraph,
                is_first_paragraph_line=first.is_first_paragraph_line,
                is_last_paragraph_line_hint=first.is_last_paragraph_line_hint,
            )
        return templates

    def _tokenize_text(self, text):
        return [token for token in re.findall(r"\S+", text) if token]

    def _segments_for_unit(self, unit):
        raw_unit = dict((unit.metadata or {}).get("raw_unit") or {})
        raw_fragments = list((unit.metadata or {}).get("fragments") or raw_unit.get("fragments") or [])
        render_policy = str(unit.render_policy or "").strip().lower()
        unit_type = str(unit.unit_type or "").strip().lower()
        tuning = self.reconstructor._unit_render_tuning(unit)
        if unit_type == "translated_line":
            _tr = self.reconstructor._clean_text_for_render(unit.text_translated or "")
            _sr = self.reconstructor._clean_text_for_render(unit.text_source or "")
            text = _tr if _tr else _sr
            if not text:
                return []
            return [{"text": text, "style": dict(unit.style or {}), "unit": unit}]
        preserve_as_single = render_policy in {"anchored_text", "fixed_preserve"} or not unit.reflowable or tuning.get("prefer_atomic_short_units")
        if raw_fragments:
            segments = []
            protected_classes = {"technical_inline", "reference", "formula", "code"}
            fragment_classes = [
                str(((fragment.get("expression_semantics") or {}).get("inline_class") or "")).strip().lower()
                for fragment in raw_fragments
            ]
            preserve_fragment_runs = bool((unit.metadata or {}).get("has_protected_fragments"))
            atomic_fragments = preserve_as_single or preserve_fragment_runs or any(cls in protected_classes for cls in fragment_classes)
            for fragment in raw_fragments:
                text = self.reconstructor._clean_text_for_render(
                    fragment.get("translated_text") or fragment.get("text") or fragment.get("texte") or ""
                )
                if not text:
                    continue
                style = self.reconstructor._merge_styles(fragment.get("style") or {}, unit.style)
                if atomic_fragments:
                    tokens = [text]
                else:
                    inline_class = str(((fragment.get("expression_semantics") or {}).get("inline_class") or "")).strip().lower()
                    tokens = [text] if inline_class in protected_classes else self._tokenize_text(text)
                for token in tokens:
                    segments.append({"text": token, "style": style, "unit": unit})
            if segments:
                return segments
        _tr = self.reconstructor._clean_text_for_render(unit.text_translated or "")
        _sr = self.reconstructor._clean_text_for_render(unit.text_source or "")
        text = _tr if _tr else _sr
        if not text:
            return []
        tokens = [text] if preserve_as_single or unit.group_class else self._tokenize_text(text)
        return [{"text": token, "style": dict(unit.style or {}), "unit": unit} for token in tokens]

    def _measure_text(self, page, style, text):
        _, fontfile, builtin, fontname = self.reconstructor._resolve_style_font( page, style, text=text)
        fontsize = float(style.get("size") or 12.0)
        width = self.reconstructor._measure_text_width( text, fontsize, fontname, fontfile)
        rgb = self.reconstructor._resolve_text_color( style, None)
        return width, fontsize, fontname, fontfile, builtin, rgb

    def _scaled_style(self, style, scale):
        scaled = dict(style or {})
        scaled["size"] = max(5.5, float((style or {}).get("size") or 12.0) * scale)
        return scaled

    def _next_template(self, templates, template_index):
        next_index = template_index + 1
        if next_index < len(templates):
            return next_index, templates[next_index]
        return None, None

    def _wrap_text(self, page, style, text, max_width):
        if max_width <= 8.0:
            return [text]
        tokens = self._tokenize_text(text)
        if len(tokens) <= 1:
            return [text]
        lines = []
        current = []
        for token in tokens:
            candidate = " ".join(current + [token]).strip()
            width, *_ = self._measure_text(page, style, candidate)
            if current and width > max_width:
                lines.append(" ".join(current))
                current = [token]
            else:
                current.append(token)
        if current:
            lines.append(" ".join(current))
        return lines or [text]

    def _linewise_fallback(self, page, plan):
        block = plan.source_block or {}
        adaptive_profile = dict(plan.adaptive_profile or {})
        lines = list((block.get("lines") or []))
        if not lines:
            return []
        templates = self._prepare_templates(plan)
        if not templates:
            return []
        ops = []
        if plan.background_strategy == "whiteout":
            ops.append(BlockRenderOp("erase_rect", plan.block_id, None, bbox=plan.block_bbox, z_index=0))
        for idx, line in enumerate(lines):
            if idx >= len(templates):
                break
            text = self.reconstructor._line_translated_text(line)
            if not text:
                continue
            style = self.reconstructor._merge_styles((line or {}).get("style") or {}, self.reconstructor._style_from_block(block))
            template_index = idx
            remaining_lines = [text]
            fontsize = float(style.get("size") or 12.0)
            min_fontsize = 5.0 if str(adaptive_profile.get("page_profile") or "") in {"academic_dense", "technical_structured"} else 5.5
            while fontsize >= min_fontsize:
                probe_template = templates[min(template_index, len(templates) - 1)]
                wrapped = self._wrap_text(page, {**style, "size": fontsize}, text, max(8.0, probe_template.usable_width))
                if template_index + len(wrapped) <= len(templates):
                    remaining_lines = wrapped
                    break
                fontsize -= 0.5
            _, fontfile, builtin, fontname = self.reconstructor._resolve_style_font( page, {**style, "size": fontsize}, text=text)
            rgb = self.reconstructor._resolve_text_color( style, block)
            for wrapped_text in remaining_lines:
                if template_index >= len(templates):
                    break
                template = templates[template_index]
                width = self.reconstructor._measure_text_width( wrapped_text, fontsize, fontname, fontfile)
                align = self.reconstructor._normalize_alignment(template.alignment or plan.paragraph_alignment or plan.alignment)
                x = template.left_x + (template.first_line_indent_px if template.is_first_paragraph_line else template.indent_px)
                if align == "center":
                    x = max(x, x + max(0.0, (template.usable_width - width) / 2.0))
                elif align == "right":
                    x = max(x, template.right_x - width)
                baseline = template.baseline_y
                rect = fitz.Rect(x, baseline - max(1.0, fontsize * 0.82), min(template.right_x, x + width), baseline + max(1.0, fontsize * 0.18))
                ops.append(
                    self._emit_text_run(
                        plan,
                        wrapped_text,
                        rect,
                        (x, baseline),
                        {**style, "size": fontsize},
                        fontname,
                        fontfile,
                        builtin,
                        fontsize,
                        rgb,
                        unit_id=f"{plan.block_id}:line:{idx}",
                    )
                )
                template_index += 1
        return ops

    def _validate_fallback_ops(self, plan, ops):
        severe = {"overflow", "text_overlap", "protected_overlap"}
        findings = self.reconstructor._validate_block_layout(plan, ops)
        return not any(str((finding or {}).get("type") or "").strip().lower() in severe for finding in findings)

    def _render_bbox_anchored(self, page, plan):
        units = sorted(
            [unit for unit in (plan.units or []) if unit.relative_bbox],
            key=lambda unit: ((unit.relative_bbox or (0, 0, 0, 0))[1], (unit.relative_bbox or (0, 0, 0, 0))[0], unit.unit_id),
        )
        if not units:
            return []
        ops = []
        if plan.background_strategy == "whiteout":
            ops.append(BlockRenderOp("erase_rect", plan.block_id, None, bbox=plan.block_bbox, z_index=0))
        block_rect = fitz.Rect(plan.block_bbox)
        for unit in units:
            _tr = self.reconstructor._clean_text_for_render(unit.text_translated or "")
            _sr = self.reconstructor._clean_text_for_render(unit.text_source or "")
            text = _tr if _tr else _sr
            if not text:
                continue
            tuning = self.reconstructor._unit_render_tuning(unit, plan)
            rect = fitz.Rect(unit.relative_bbox)
            rect = fitz.Rect(
                max(block_rect.x0, rect.x0),
                max(block_rect.y0, rect.y0),
                min(block_rect.x1, rect.x1),
                min(block_rect.y1, rect.y1),
            )
            if rect.width <= 0 or rect.height <= 0:
                continue
            style = dict(unit.style or {})
            _, fontfile, builtin, fontname = self.reconstructor._resolve_style_font( page, style, text=text)
            fontsize = min(float(style.get("size") or 12.0), max(6.0, rect.height * 0.78))
            wrapped = self.reconstructor._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            while fontsize > tuning["min_fontsize"] and wrapped and (len(wrapped) * max(6.0, fontsize * 1.12 * tuning["line_spacing_factor"])) > max(rect.height, fontsize * 1.3):
                fontsize -= 0.5
                wrapped = self.reconstructor._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            rgb = self.reconstructor._resolve_text_color( style, plan.source_block)
            line_h = max(6.0, fontsize * 1.12 * tuning["line_spacing_factor"])
            align = self.reconstructor._unit_horizontal_alignment(unit, plan.paragraph_alignment or plan.alignment)
            for line_idx, line_text in enumerate(wrapped):
                width = self.reconstructor._measure_text_width( line_text, fontsize, fontname, fontfile)
                x = rect.x0
                if align == "center":
                    x = max(rect.x0, rect.x0 + max(0.0, (rect.width - width) / 2.0))
                elif align == "right":
                    x = max(rect.x0, rect.x1 - width)
                baseline = self.reconstructor._anchored_line_baseline(rect, unit, fontsize, line_h, line_idx, len(wrapped))
                text_rect = fitz.Rect(
                    x,
                    baseline - max(1.0, fontsize * 0.82),
                    min(block_rect.x1, x + width),
                    baseline + max(1.0, fontsize * 0.18),
                )
                ops.append(
                    self._emit_text_run(
                        plan,
                        line_text,
                        text_rect,
                        (x, baseline),
                        {**style, "size": fontsize},
                        fontname,
                        fontfile,
                        builtin,
                        fontsize,
                        rgb,
                        unit_id=unit.unit_id,
                    )
                )
        return ops

    def _render_relative_slots(self, page, plan):
        units = sorted(
            [unit for unit in (plan.units or []) if unit.relative_bbox],
            key=lambda unit: ((unit.line_indices or [0])[0], (unit.relative_bbox or (0, 0, 0, 0))[1], (unit.relative_bbox or (0, 0, 0, 0))[0], unit.unit_id),
        )
        if not units:
            return []
        ops = []
        block_rect = fitz.Rect(plan.block_bbox)
        for unit in units:
            _tr = self.reconstructor._clean_text_for_render(unit.text_translated or "")
            _sr = self.reconstructor._clean_text_for_render(unit.text_source or "")
            text = _tr if _tr else _sr
            if not text:
                continue
            tuning = self.reconstructor._unit_render_tuning(unit, plan)
            rect = fitz.Rect(unit.relative_bbox)
            rect = fitz.Rect(
                max(block_rect.x0, rect.x0),
                max(block_rect.y0, rect.y0),
                min(block_rect.x1, rect.x1),
                min(block_rect.y1, rect.y1),
            )
            if rect.width <= 0 or rect.height <= 0:
                continue
            style = dict(unit.style or {})
            _, fontfile, builtin, fontname = self.reconstructor._resolve_style_font( page, style, text=text)
            fontsize = min(float(style.get("size") or 12.0), max(6.0, rect.height * 0.78))
            wrapped = self.reconstructor._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            while fontsize > tuning["min_fontsize"] and wrapped and (len(wrapped) * max(6.0, fontsize * 1.12 * tuning["line_spacing_factor"])) > max(rect.height, fontsize * 1.3):
                fontsize -= 0.5
                wrapped = self.reconstructor._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            rgb = self.reconstructor._resolve_text_color( style, plan.source_block)
            align = self.reconstructor._unit_horizontal_alignment(unit, plan.paragraph_alignment or plan.alignment)
            if align == "end":
                align = "right"
            elif align == "start":
                align = "left"
            line_h = max(6.0, fontsize * 1.12 * tuning["line_spacing_factor"])
            for line_idx, line_text in enumerate(wrapped):
                width = self.reconstructor._measure_text_width( line_text, fontsize, fontname, fontfile)
                x = rect.x0
                if align == "center":
                    x = max(rect.x0, rect.x0 + max(0.0, (rect.width - width) / 2.0))
                elif align == "right":
                    x = max(rect.x0, rect.x1 - width)
                baseline = self.reconstructor._anchored_line_baseline(rect, unit, fontsize, line_h, line_idx, len(wrapped))
                text_rect = fitz.Rect(
                    x,
                    baseline - max(1.0, fontsize * 0.82),
                    min(block_rect.x1, x + width),
                    baseline + max(1.0, fontsize * 0.18),
                )
                ops.append(
                    self._emit_text_run(
                        plan,
                        line_text,
                        text_rect,
                        (x, baseline),
                        {**style, "size": fontsize},
                        fontname,
                        fontfile,
                        builtin,
                        fontsize,
                        rgb,
                        unit_id=unit.unit_id,
                    )
                )
                next_baseline = self.reconstructor._anchored_line_baseline(rect, unit, fontsize, line_h, line_idx + 1, len(wrapped))
                if next_baseline + line_h * 0.18 > rect.y1:
                    break
        return ops

    def _finalize_line(self, page, plan, template, segments, is_last_line):
        if not segments:
            return []
        alignment = self.reconstructor._normalize_alignment(template.alignment or plan.paragraph_alignment or plan.alignment)
        dense_profile = str((plan.adaptive_profile or {}).get("page_profile") or "") in {"academic_dense", "technical_structured"}
        gap_factor = 0.85 if dense_profile else 1.0
        default_gap = max(1.5, min(6.0, template.ascent * 0.22 * gap_factor))
        widths = []
        measurements = []
        for seg in segments:
            width, fontsize, fontname, fontfile, builtin, rgb = self._measure_text(page, seg["style"], seg["text"])
            widths.append(width)
            measurements.append((fontsize, fontname, fontfile, builtin, rgb))
        total_width = sum(widths) + default_gap * max(0, len(widths) - 1)
        start_x = template.left_x + (template.first_line_indent_px if template.is_first_paragraph_line else template.indent_px)
        first_unit = segments[0]["unit"] if segments else None
        first_meta = dict((first_unit.metadata or {})) if first_unit else {}
        row_anchor_x = first_meta.get("row_anchor_x")
        if isinstance(row_anchor_x, (int, float)):
            start_x = max(start_x, float(row_anchor_x))
        if alignment == "center":
            start_x = max(start_x, start_x + max(0.0, (template.usable_width - total_width) / 2.0))
        elif alignment == "right":
            start_x = max(start_x, template.right_x - total_width)
        gaps = [default_gap for _ in range(max(0, len(widths) - 1))]
        if alignment == "justify" and len(widths) > 1 and not is_last_line:
            extra = max(0.0, template.usable_width - total_width)
            if extra > 0.0:
                spread = extra / max(1, len(gaps))
                gaps = [gap + spread for gap in gaps]
        ops = []
        x = start_x
        baseline = template.baseline_y
        for idx, seg in enumerate(segments):
            width = widths[idx]
            fontsize, fontname, fontfile, builtin, rgb = measurements[idx]
            rect = fitz.Rect(x, baseline - max(1.0, fontsize * 0.82), x + width, baseline + max(1.0, fontsize * 0.18))
            ops.append(
                self._emit_text_run(
                    plan,
                    seg["text"],
                    rect,
                    (x, baseline),
                    seg["style"],
                    fontname,
                    fontfile,
                    builtin,
                    fontsize,
                    rgb,
                    unit_id=seg["unit"].unit_id,
                )
            )
            x += width
            if idx < len(gaps):
                x += gaps[idx]
        return ops

    def _render_with_scale(self, page, plan, scale):
        if not plan.units:
            return []
        templates = self._prepare_templates(plan)
        if not templates:
            return []
        ops = []
        if plan.background_strategy == "whiteout":
            ops.append(BlockRenderOp("erase_rect", plan.block_id, None, bbox=plan.block_bbox, z_index=0))
        edge_map = self._edge_map(plan)
        template_index = 0
        current_segments = []
        current_template = templates[template_index]
        previous_unit = None
        for unit in plan.units:
            relation = edge_map.get((previous_unit.unit_id, unit.unit_id)).relation if previous_unit and edge_map.get((previous_unit.unit_id, unit.unit_id)) else None
            if relation in {"NEW_PARAGRAPH", "NEW_LINE"} and current_segments:
                ops.extend(self._finalize_line(page, plan, current_template, current_segments, is_last_line=(relation == "NEW_PARAGRAPH")))
                current_segments = []
                next_index, next_template = self._next_template(templates, template_index)
                if next_template is not None:
                    template_index = next_index
                    current_template = next_template
            segments = self._segments_for_unit(unit)
            if not segments:
                previous_unit = unit
                continue
            preserve_as_single = str(unit.render_policy or "").strip().lower() in {"anchored_text", "fixed_preserve"} or not unit.reflowable
            tuning = self.reconstructor._unit_render_tuning(unit, plan)
            for seg in segments:
                scaled_style = self._scaled_style(seg["style"], scale)
                width, _, _, _, _, _ = self._measure_text(page, scaled_style, seg["text"])
                existing_width = 0.0
                if current_segments:
                    for prev_seg in current_segments:
                        prev_w, *_ = self._measure_text(page, prev_seg["style"], prev_seg["text"])
                        existing_width += prev_w
                    existing_width += max(2.0, min(6.0, current_template.ascent * 0.22)) * max(0, len(current_segments))
                projected = existing_width + width
                can_wrap_segment = (
                    ((unit.reflowable and not preserve_as_single) or str(unit.unit_type or "").strip().lower() == "translated_line")
                    and not tuning.get("prefer_atomic_short_units")
                )
                if current_segments and projected > current_template.usable_width and can_wrap_segment:
                    ops.extend(self._finalize_line(page, plan, current_template, current_segments, is_last_line=False))
                    current_segments = []
                    next_index, next_template = self._next_template(templates, template_index)
                    if next_template is not None:
                        template_index = next_index
                        current_template = next_template
                if not current_segments and width > current_template.usable_width and can_wrap_segment:
                    wrapped = self._wrap_text(page, scaled_style, seg["text"], current_template.usable_width)
                    for line_idx, wrapped_text in enumerate(wrapped):
                        current_segments.append({"text": wrapped_text, "style": scaled_style, "unit": unit})
                        ops.extend(self._finalize_line(page, plan, current_template, current_segments, is_last_line=False))
                        current_segments = []
                        if line_idx < len(wrapped) - 1:
                            next_index, next_template = self._next_template(templates, template_index)
                            if next_template is not None:
                                template_index = next_index
                                current_template = next_template
                    continue
                current_segments.append({"text": seg["text"], "style": scaled_style, "unit": unit})
            previous_unit = unit
        if current_segments:
            ops.extend(self._finalize_line(page, plan, current_template, current_segments, is_last_line=True))
        return ops

    # ------------------------------------------------------------------
    # Reflow prose : flux continu du texte traduit dans la bbox du bloc
    # ------------------------------------------------------------------

    def _collect_translated_text_stream(self, plan):
        """Assemble le texte traduit de toutes les unites en flux continu."""
        units = list(plan.units or [])
        parts = []
        for unit in units:
            translated = self.reconstructor._clean_text_for_render(unit.text_translated or "")
            source = self.reconstructor._clean_text_for_render(unit.text_source or "")
            text = translated if translated else source
            if not text:
                continue
            if unit.hard_break_before and parts:
                # Nouveau paragraphe : ajouter un saut de ligne
                parts.append("\n")
            elif parts:
                prev = parts[-1]
                if prev and prev[-1] == "-":
                    # Coupure de mot en fin de ligne : coller sans espace
                    parts[-1] = prev[:-1]
                else:
                    parts.append(" ")
            parts.append(text)
        return "".join(parts)

    def _render_prose_reflow(self, page, plan):
        """Reflow du texte traduit en flux continu dans la bbox du bloc."""
        full_text = self._collect_translated_text_stream(plan)
        if not full_text.strip():
            return []
        ops = []
        if plan.background_strategy == "whiteout":
            ops.append(BlockRenderOp("erase_rect", plan.block_id, None, bbox=plan.block_bbox, z_index=0))
        block_rect = fitz.Rect(plan.block_bbox)
        profile = getattr(plan, "semantic_profile", None)
        # Police unicode-safe garantie
        fontfile, fontname = self.reconstructor._resolve_unicode_safe_font(page, plan, full_text)
        # Style de base (couleur, flags)
        base_style = self.reconstructor._style_from_block(plan.source_block or {})
        rgb = self.reconstructor._resolve_text_color(base_style, plan.source_block)
        # Taille de police : partir de la taille dominante, reduire si debordement
        fontsize = float(profile.dominant_fontsize) if (profile and profile.dominant_fontsize > 0) else float(base_style.get("size") or 11.0)
        fontsize = max(7.0, min(fontsize, 36.0))
        usable_w = max(8.0, block_rect.width - 4.0)
        usable_h = max(8.0, block_rect.height - 4.0)
        # Calcul word-wrap et ajustement de taille
        style_for_wrap = {**base_style, "size": fontsize}
        wrapped = self._wrap_text(page, style_for_wrap, full_text, usable_w)
        line_h = fontsize * 1.18
        allow_expand = bool(profile.allow_vertical_expansion) if profile else False
        while fontsize > 7.0 and not allow_expand and len(wrapped) * line_h > usable_h:
            fontsize -= 0.5
            line_h = fontsize * 1.18
            style_for_wrap = {**base_style, "size": fontsize}
            wrapped = self._wrap_text(page, style_for_wrap, full_text, usable_w)
        # Alignement
        alignment = self.reconstructor._normalize_alignment(
            plan.paragraph_alignment or plan.alignment or "left"
        )
        # Utiliser les templates avec compaction heading_to_body si disponibles
        templates = self._prepare_templates(plan)
        y = block_rect.y0 + fontsize * 0.82
        for line_idx, line_text in enumerate(wrapped):
            if templates and line_idx < len(templates):
                baseline = templates[line_idx].baseline_y
                left_x = templates[line_idx].left_x
                right_x = templates[line_idx].right_x
            else:
                baseline = y
                left_x = block_rect.x0 + 2.0
                right_x = block_rect.x1 - 2.0
            width = self.reconstructor._measure_text_width(line_text, fontsize, fontname, fontfile)
            x = left_x
            if alignment == "center":
                x = max(left_x, left_x + max(0.0, (right_x - left_x - width) / 2.0))
            elif alignment == "right":
                x = max(left_x, right_x - width)
            text_rect = fitz.Rect(x, baseline - max(1.0, fontsize * 0.82), min(right_x, x + width), baseline + max(1.0, fontsize * 0.18))
            ops.append(self._emit_text_run(
                plan, line_text, text_rect, (x, baseline),
                {**base_style, "size": fontsize}, fontname, fontfile, None, fontsize, rgb,
                unit_id=f"{plan.block_id}:reflow:{line_idx}",
            ))
            y = baseline + line_h
        return ops

    def _render_label_stack(self, page, plan):
        """Rendu ligne par ligne pour les blocs de labels/colonnes/listes atomiques."""
        units = sorted(
            [u for u in (plan.units or []) if u.relative_bbox],
            key=lambda u: ((u.relative_bbox or (0, 0, 0, 0))[1], (u.relative_bbox or (0, 0, 0, 0))[0]),
        )
        if not units:
            # Pas de relative_bbox : fallback sur les templates
            units = list(plan.units or [])
        ops = []
        if plan.background_strategy == "whiteout":
            ops.append(BlockRenderOp("erase_rect", plan.block_id, None, bbox=plan.block_bbox, z_index=0))
        block_rect = fitz.Rect(plan.block_bbox)
        templates = list(plan.line_templates or [])
        for unit_idx, unit in enumerate(units):
            translated = self.reconstructor._clean_text_for_render(unit.text_translated or "")
            source = self.reconstructor._clean_text_for_render(unit.text_source or "")
            text = translated if translated else source
            if not text:
                continue
            # Bbox de l'unite ou fallback template
            if unit.relative_bbox:
                unit_rect = fitz.Rect(unit.relative_bbox)
                unit_rect = fitz.Rect(
                    max(block_rect.x0, unit_rect.x0),
                    max(block_rect.y0, unit_rect.y0),
                    min(block_rect.x1, unit_rect.x1),
                    min(block_rect.y1, unit_rect.y1),
                )
            elif unit_idx < len(templates):
                t = templates[unit_idx]
                unit_rect = fitz.Rect(t.bbox)
            else:
                continue
            if unit_rect.width <= 0 or unit_rect.height <= 0:
                continue
            fontfile, fontname = self.reconstructor._resolve_unicode_safe_font(page, plan, text)
            style = dict(unit.style or {})
            fontsize = min(float(style.get("size") or 11.0), max(6.0, unit_rect.height * 0.78))
            # Reduire si le texte ne tient pas en largeur
            available_w = max(4.0, unit_rect.width)
            while fontsize > 6.0 and self.reconstructor._measure_text_width(text, fontsize, fontname, fontfile) > available_w:
                fontsize -= 0.5
            rgb = self.reconstructor._resolve_text_color(style, plan.source_block)
            alignment = self.reconstructor._normalize_alignment(
                self.reconstructor._unit_horizontal_alignment(unit, plan.paragraph_alignment or plan.alignment)
            )
            width = self.reconstructor._measure_text_width(text, fontsize, fontname, fontfile)
            baseline = unit_rect.y1 - max(1.0, fontsize * 0.18)
            x = unit_rect.x0
            if alignment == "center":
                x = max(unit_rect.x0, unit_rect.x0 + max(0.0, (unit_rect.width - width) / 2.0))
            elif alignment == "right":
                x = max(unit_rect.x0, unit_rect.x1 - width)
            text_rect = fitz.Rect(x, baseline - max(1.0, fontsize * 0.82), min(block_rect.x1, x + width), baseline + max(1.0, fontsize * 0.18))
            ops.append(self._emit_text_run(
                plan, text, text_rect, (x, baseline),
                {**style, "size": fontsize}, fontname, fontfile, None, fontsize, rgb,
                unit_id=unit.unit_id,
            ))
        return ops

    def render(self, page, plan):
        _tr_parts = [
            self.reconstructor._clean_text_for_render(u.text_translated or "")
            for u in (plan.units or [])
            if self.reconstructor._clean_text_for_render(u.text_translated or "")
        ]
        profile = self.reconstructor.compute_block_semantic_profile(
            plan.source_block,
            getattr(plan, "page_data", None),
            translated_text=" ".join(_tr_parts),
        )
        if plan.semantic_profile is None:
            plan = replace(plan, semantic_profile=profile)
        # Dispatch base sur le profil semantique si le bloc est traduit
        _units_have_mixed_style_fragments = any(
            list((u.metadata or {}).get("fragments") or {}) or
            list((((u.metadata or {}).get("raw_unit") or {})).get("fragments") or [])
            for u in (plan.units or [])
        )
        if profile is not None and profile.source_is_translated and not _units_have_mixed_style_fragments:
            strategy = profile.render_strategy
            if strategy in ("prose_reflow", "heading_reflow", "caption_reflow"):
                return self._render_prose_reflow(page, plan)
            if strategy == "label_stack":
                return self._render_label_stack(page, plan)
            # bitmap_preserve et code_preserve : paths existants
        # Paths existants (blocs non traduits ou strategies speciales)
        if self._should_render_relative_slot_mode(plan):
            return self._render_relative_slots(page, plan)
        if self._should_render_bbox_anchored(plan):
            return self._render_bbox_anchored(page, plan)
        best_ops = []
        best_finding_count = None
        severe = {"overflow", "text_overlap", "protected_overlap"}
        scale_ladder = tuple((plan.adaptive_profile or {}).get("editorial_scales") or (1.0, 0.96, 0.92, 0.88, 0.84, 0.8, 0.76, 0.72))
        for scale in scale_ladder:
            ops = self._render_with_scale(page, plan, scale)
            findings = self.reconstructor._validate_block_layout(plan, ops)
            severe_findings = [
                finding for finding in findings
                if str((finding or {}).get("type") or "").strip().lower() in severe
            ]
            if best_finding_count is None or len(severe_findings) < best_finding_count:
                best_ops = ops
                best_finding_count = len(severe_findings)
            if not severe_findings:
                return ops
        fallback_ops = self._linewise_fallback(page, plan)
        if fallback_ops and self._validate_fallback_ops(plan, fallback_ops):
            return fallback_ops
        return best_ops


class HeadingBlockRenderer(EditorialBlockRenderer):
    pass


class CaptionBlockRenderer(EditorialBlockRenderer):
    pass


class AnnotationBlockRenderer(EditorialBlockRenderer):
    pass


class CodeBlockRenderer(BaseBlockRenderer):
    def render(self, page, plan):
        overlay_ops = self._overlay_ops_for_matching_immutable_overlays(plan)
        if overlay_ops:
            return overlay_ops
        # Pas d'overlay pré-capturé : rendu texte source ligne par ligne en monospace,
        # sans traduction, sans reflow. Garantit que le code n'est pas silencieusement absent.
        block = plan.source_block or {}
        lines = list(block.get("lines") or [])
        if not lines:
            return []
        ops = []
        block_rect = fitz.Rect(plan.block_bbox)
        base_style = self.reconstructor._style_from_block(block)
        mono_style = {**base_style, "font": base_style.get("font") or "courier", "flags": {**(base_style.get("flags") or {}), "monospace": True}}
        for idx, line in enumerate(lines):
            line_is_technical = self.reconstructor._line_looks_technical_structured(line, block=block)
            source_text = self.reconstructor._line_source_text(line)
            translated_text = self.reconstructor._line_translated_text(line)
            text = self.reconstructor._clean_text_for_render(source_text if line_is_technical else (translated_text or source_text))
            if not text:
                continue
            line_style = self.reconstructor._merge_styles((line or {}).get("style") or {}, base_style)
            effective_style = (
                {**line_style, "font": line_style.get("font") or "courier", "flags": {**(line_style.get("flags") or {}), "monospace": True}}
                if line_is_technical
                else line_style
            )
            template_lines = plan.line_templates
            if template_lines and idx < len(template_lines):
                tmpl = template_lines[idx]
                baseline = tmpl.baseline_y
                x = tmpl.left_x
                available_width = max(8.0, tmpl.right_x - tmpl.left_x)
            else:
                fontsize_probe = float(effective_style.get("size") or 10.0)
                line_h = max(fontsize_probe * 1.15, block_rect.height / max(1, len(lines)))
                baseline = block_rect.y0 + (idx + 0.82) * line_h
                x = block_rect.x0
                available_width = max(8.0, block_rect.x1 - x)
            _, fontfile, builtin, fontname = self.reconstructor._resolve_style_font(page, effective_style, text=text)
            fontsize = min(float(effective_style.get("size") or 10.0), max(5.5, block_rect.height / max(1, len(lines)) * 0.82))
            while fontsize > 5.5 and self.reconstructor._measure_text_width(text, fontsize, fontname, fontfile) > available_width:
                fontsize -= 0.5
            rgb = self.reconstructor._resolve_text_color(effective_style, block)
            text_rect = fitz.Rect(x, baseline - max(1.0, fontsize * 0.82), min(block_rect.x1, x + available_width), baseline + max(1.0, fontsize * 0.18))
            ops.append(
                self._emit_text_run(
                    plan, text, text_rect, (x, baseline),
                    {**effective_style, "size": fontsize},
                    fontname, fontfile, builtin, fontsize, rgb,
                    unit_id=f"{plan.block_id}:code:{idx}",
                )
            )
        return ops


class TableBlockRenderer(BaseBlockRenderer):
    def _line_text(self, line):
        text = self.reconstructor._clean_text_for_render((line or {}).get("translated_text") or "")
        if text:
            return text
        phrase_parts = []
        for phrase in (line or {}).get("phrases") or []:
            phrase_text = self.reconstructor._clean_text_for_render(
                (phrase or {}).get("translated_text") or (phrase or {}).get("texte") or ""
            )
            if phrase_text:
                phrase_parts.append(phrase_text)
        if phrase_parts:
            return " ".join(phrase_parts)
        return self.reconstructor._clean_text_for_render((line or {}).get("line_text") or "")

    def _wrap_text_to_lines(self, text, available_width, fontsize, fontname, fontfile):
        """Découpe `text` en segments qui tiennent dans `available_width`."""
        words = text.split()
        if not words:
            return [text]
        wrapped = []
        current_words = []
        for word in words:
            candidate = " ".join(current_words + [word])
            w = self.reconstructor._measure_text_width( candidate, fontsize, fontname, fontfile)
            if w <= max(8.0, available_width) or not current_words:
                current_words.append(word)
            else:
                wrapped.append(" ".join(current_words))
                current_words = [word]
        if current_words:
            wrapped.append(" ".join(current_words))
        return wrapped if wrapped else [text]

    def render(self, page, plan):
        block = plan.source_block or {}
        descriptor_group_ids = dict((block or {}).get("descriptor_group_ids") or {})
        has_explicit_cell = bool(
            plan.constraints.get("table_cell_bbox")
            or str(descriptor_group_ids.get("cell_id") or "").strip()
            or str(descriptor_group_ids.get("table_row_group_id") or "").strip()
        )
        if self.reconstructor._block_looks_technical_structured(block):
            return CodeBlockRenderer(self.reconstructor).render(page, plan)
        if not has_explicit_cell:
            return EditorialBlockRenderer(self.reconstructor).render(page, plan)
        cell_bbox = plan.constraints.get("table_cell_bbox") or plan.block_bbox
        cell_rect = fitz.Rect(cell_bbox)
        ops = []
        if plan.background_strategy == "whiteout":
            ops.append(BlockRenderOp("erase_rect", plan.block_id, None, bbox=(cell_rect.x0, cell_rect.y0, cell_rect.x1, cell_rect.y1), z_index=0))
        lines = list(block.get("lines") or [])
        if not lines:
            lines = [{"bbox": block.get("bbox"), "translated_text": self.reconstructor._translated_text_from_block(block)}]
        align = self.reconstructor._normalize_alignment((block or {}).get("alignment") or "left")
        template_lines = plan.line_templates or []
        run_index = 0
        for idx, line in enumerate(lines):
            text = self._line_text(line)
            if not text:
                continue
            style = self.reconstructor._merge_styles((line or {}).get("style") or {}, self.reconstructor._style_from_block(block))
            _, fontfile, builtin, fontname = self.reconstructor._resolve_style_font( page, style, text=text)
            fontsize = max(5.5, float(style.get("size") or 10.0))
            rgb = self.reconstructor._resolve_text_color( style, block)
            # Résoudre la zone de référence pour cette ligne
            if template_lines and idx < len(template_lines):
                tmpl = template_lines[idx]
                ref_x0 = tmpl.left_x
                ref_baseline = tmpl.baseline_y
                ref_x1 = cell_rect.x1
                # Wrapping sur la largeur de la cellule
                available_w = max(8.0, ref_x1 - ref_x0)
                wrapped = self._wrap_text_to_lines(text, available_w, fontsize, fontname, fontfile)
                line_h = max(fontsize * 1.2, 1.0)
                for wi, seg in enumerate(wrapped):
                    seg_w = self.reconstructor._measure_text_width( seg, fontsize, fontname, fontfile)
                    baseline = ref_baseline + wi * line_h
                    x = ref_x0
                    if align == "center":
                        x = max(ref_x0, ref_x0 + max(0.0, (available_w - seg_w) / 2.0))
                    elif align == "right":
                        x = max(ref_x0, ref_x1 - seg_w)
                    rect = fitz.Rect(x, baseline - max(1.0, fontsize * 0.82), min(ref_x1, x + seg_w), baseline + max(1.0, fontsize * 0.18))
                    ops.append(self._emit_text_run(plan, seg, rect, (x, baseline), {**style, "size": fontsize},
                                                   fontname, fontfile, builtin, fontsize, rgb,
                                                   unit_id=f"{plan.block_id}:table:{run_index}"))
                    run_index += 1
            else:
                # Placement proportionnel dans la cellule
                available_w = max(8.0, cell_rect.width)
                wrapped = self._wrap_text_to_lines(text, available_w, fontsize, fontname, fontfile)
                total_lines_in_block = max(1, len(lines))
                line_slot_h = cell_rect.height / total_lines_in_block
                line_h = max(fontsize * 1.2, 1.0)
                slot_top = cell_rect.y0 + idx * line_slot_h
                for wi, seg in enumerate(wrapped):
                    seg_w = self.reconstructor._measure_text_width( seg, fontsize, fontname, fontfile)
                    baseline = slot_top + (wi + 0.82) * line_h
                    baseline = min(baseline, cell_rect.y1 - 1.0)
                    x = cell_rect.x0
                    if align == "center":
                        x = max(cell_rect.x0, cell_rect.x0 + max(0.0, (available_w - seg_w) / 2.0))
                    elif align == "right":
                        x = max(cell_rect.x0, cell_rect.x1 - seg_w)
                    rect = fitz.Rect(x, baseline - max(1.0, fontsize * 0.82), min(cell_rect.x1, x + seg_w), baseline + max(1.0, fontsize * 0.18))
                    ops.append(self._emit_text_run(plan, seg, rect, (x, baseline), {**style, "size": fontsize},
                                                   fontname, fontfile, builtin, fontsize, rgb,
                                                   unit_id=f"{plan.block_id}:table:{run_index}"))
                    run_index += 1
        return ops
