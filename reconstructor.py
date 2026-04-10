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
    constraints: dict[str, Any]
    source_block: dict[str, Any] = field(default_factory=dict)


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
                units.append(
                    PlacableUnit(
                        unit_id=str(item["seg"].get("unit_id") or f"{block_id}:external:{item['idx']}"),
                        unit_type=f"external_{segment_type}",
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
                        layout_attributes={
                            "horizontal_anchor": "end" if segment_type == "page" else "start",
                            "vertical_anchor": "top",
                        },
                        text_attributes={},
                        relative_bbox=item["bbox"],
                        anchor_horizontal="end" if segment_type == "page" else "start",
                        anchor_vertical="top",
                        continuation_before=not is_first,
                        continuation_after=not is_last,
                        hard_break_before=is_first,
                        hard_break_after=is_last,
                        keep_with_previous=not is_first,
                        keep_with_next=not is_last,
                        reflowable=(segment_type != "page"),
                        protected_inline=False,
                        immutable=False,
                        render_policy="external_flow",
                        justification_eligible=(segment_type != "page"),
                        break_priority=20,
                        paragraph_id=f"{block_id}:external:{row_idx}",
                        metadata={"target_lang": target_lang, "segment_type": segment_type, "raw_unit": dict(item["seg"])},
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
        if self._block_is_immutable_programming_code(block) or self._is_symbolic_visual_block(block):
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

    def _build_page_reconstruction_context(self, page_data, target_lang):
        return {
            "target_lang": target_lang,
            "writing_direction": "right_to_left" if target_lang in {"ar", "he", "fa"} else "left_to_right",
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
        # Utilise _page_background_path (vérifie background_path ET source_image_path)
        # pour ne pas effacer un fond déjà propre — évite le double-whiteout.
        has_clean_background = bool(self._page_background_path(page_data))
        if (
            self._is_translated_block(block)
            and block_type in {"editorial", "heading", "caption", "annotation", "table"}
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
            constraints={"page_role": str((page_data or {}).get("page_role") or "").strip().lower()},
        )

    def _build_line_templates(self, block, geometry_ctx):
        block_rect = fitz.Rect(geometry_ctx.block_bbox)
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
            line_h = max(6.0, line_rect.height)
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
        if templates:
            sorted_heights = sorted(line_heights or [12.0])
            line_h = sorted_heights[len(sorted_heights) // 2]
            last = templates[-1]
            next_top = last.bbox[3]
            extra_idx = 0
            while next_top + max(6.0, line_h * 0.75) <= inner_bottom:
                next_bottom = min(inner_bottom, next_top + line_h)
                templates.append(
                    LineTemplate(
                        line_id=f"{block.get('id') or 'block'}:extra:{extra_idx}",
                        source_line_indices=list(last.source_line_indices),
                        bbox=(inner_left, next_top, inner_right, next_bottom),
                        baseline_y=next_top + min(line_h * 0.82, max(1.0, line_h - 1.0)),
                        ascent=line_h * 0.82,
                        descent=max(1.0, line_h * 0.18),
                        left_x=inner_left,
                        right_x=inner_right,
                        usable_width=max(8.0, inner_right - inner_left - last.indent_px),
                        indent_px=last.indent_px,
                        first_line_indent_px=0.0,
                        alignment=alignment,
                        paragraph_id=last.paragraph_id,
                        paragraph_index=last.paragraph_index,
                        line_index_in_paragraph=last.line_index_in_paragraph + 1 + extra_idx,
                        is_first_paragraph_line=False,
                        is_last_paragraph_line_hint=False,
                    )
                )
                next_top = next_bottom
                extra_idx += 1
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

    def _phrase_units(self, block, semantic_payload, target_lang):
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
                    anchor_horizontal=((phrase.get("layout_attributes") or {}).get("horizontal_anchor")),
                    anchor_vertical=((phrase.get("layout_attributes") or {}).get("vertical_anchor")),
                    continuation_before=bool(editorial_rel.get("continuation")),
                    continuation_after=bool(((phrase.get("editorial_relations") or {}).get("with_next") or {}).get("continuation")),
                    hard_break_before=bool(phrase.get("hard_break_before") or editorial_rel.get("relation") in {"paragraph_break", "new_line"}),
                    hard_break_after=bool(phrase.get("hard_break_after")),
                    keep_with_previous=bool(editorial_rel.get("relation") in {"keep_with_previous", "label_value"}),
                    keep_with_next=bool(((phrase.get("editorial_relations") or {}).get("with_next") or {}).get("relation") in {"keep_with_next", "label_value"}),
                    reflowable=bool((phrase.get("editorial_semantics") or {}).get("reflowable", True)),
                    protected_inline=False,
                    immutable=False,
                    render_policy=str(phrase.get("render_policy") or block.get("render_policy") or "translated_editorial"),
                    justification_eligible=True,
                    break_priority=10,
                    paragraph_id=str(ctx.get("paragraph_id") or phrase_id),
                    metadata={"target_lang": target_lang, "raw_unit": dict(phrase), "fragments": child_units},
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

    def _line_units(self, block, target_lang):
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
                    anchor_horizontal=((line.get("layout_attributes") or {}).get("horizontal_anchor")) if isinstance(line, dict) else None,
                    anchor_vertical=((line.get("layout_attributes") or {}).get("vertical_anchor")) if isinstance(line, dict) else None,
                    continuation_before=False,
                    continuation_after=False,
                    hard_break_before=bool((line or {}).get("hard_break_before") or idx > 0),
                    hard_break_after=bool((line or {}).get("line_break_after")),
                    keep_with_previous=False,
                    keep_with_next=False,
                    reflowable=False,
                    protected_inline=False,
                    immutable=False,
                    render_policy=str((block or {}).get("render_policy") or "translated_editorial"),
                    justification_eligible=True,
                    break_priority=15,
                    paragraph_id=f"{block_id}:line_paragraph:{idx}",
                    metadata={"target_lang": target_lang, "raw_unit": dict(line or {})},
                )
            )
        return units

    def _nested_span_units(self, block, target_lang):
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
                            anchor_horizontal=(((span or {}).get("layout_attributes") or {}).get("horizontal_anchor")),
                            anchor_vertical=(((span or {}).get("layout_attributes") or {}).get("vertical_anchor")),
                            continuation_before=False,
                            continuation_after=False,
                            hard_break_before=bool(si == 0),
                            hard_break_after=bool(si == len((phrase or {}).get("spans") or []) - 1),
                            keep_with_previous=False,
                            keep_with_next=False,
                            reflowable=False,
                            protected_inline=bool(expression_semantics.get("protected_inline", False)),
                            immutable=bool(expression_semantics.get("immutable_inline", False)),
                            render_policy=render_policy,
                            justification_eligible=inline_class not in {"code", "formula", "reference"},
                            break_priority=20,
                            paragraph_id=f"{block_id}:nested_span_line:{li}",
                            metadata={"target_lang": target_lang, "raw_unit": dict(span or {})},
                        )
                    )
        return units

    def _fallback_units(self, block, semantic_payload, target_lang):
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
                    anchor_horizontal=((unit.get("layout_attributes") or {}).get("horizontal_anchor")),
                    anchor_vertical=((unit.get("layout_attributes") or {}).get("vertical_anchor")),
                    continuation_before=bool(editorial_rel.get("continuation")),
                    continuation_after=bool(((unit.get("editorial_relations") or {}).get("with_next") or {}).get("continuation")),
                    hard_break_before=bool(unit.get("hard_break_before") or editorial_rel.get("relation") in {"paragraph_break", "new_line"}),
                    hard_break_after=bool(unit.get("hard_break_after")),
                    keep_with_previous=bool(editorial_rel.get("relation") in {"keep_with_previous", "label_value"}),
                    keep_with_next=bool(((unit.get("editorial_relations") or {}).get("with_next") or {}).get("relation") in {"keep_with_next", "label_value"}),
                    reflowable=bool((unit.get("editorial_semantics") or {}).get("reflowable", True)),
                    protected_inline=bool(expression_semantics.get("protected_inline", False)),
                    immutable=bool(expression_semantics.get("immutable_inline", False)),
                    render_policy=str(unit.get("render_policy") or render_policy or "translated_editorial"),
                    justification_eligible=str(expression_semantics.get("inline_class") or "").strip().lower() not in {"code", "formula", "reference"},
                    break_priority=10 if str(unit.get("group_class") or "").strip() else 5,
                    paragraph_id=str(ctx.get("paragraph_id") or ctx.get("phrase_unit_id") or f"{block_id}:paragraph:0"),
                    metadata={"target_lang": target_lang, "raw_unit": dict(unit)},
                )
            )
        return normalized

    def _orphan_semantic_units(self, block, semantic_payload, target_lang, phrase_units):
        phrase_ids = {str(unit.phrase_unit_id or "") for unit in phrase_units or []}
        phrase_unit_ids = {str(unit.unit_id or "") for unit in phrase_units or []}
        extras = []
        seen_unit_ids = set()
        for unit in self._fallback_units(block, semantic_payload, target_lang):
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
            nested_span_units = self._nested_span_units(block, target_lang)
            if nested_span_units:
                return self._canonicalize_block_units(block, nested_span_units)
        phrase_units = self._phrase_units(block, semantic_payload, target_lang)
        if phrase_units and self._semantic_phrases_are_overlapping(semantic_payload.get("semantic_phrases") or []):
            line_units = self._line_units(block, target_lang)
            if line_units:
                phrase_units = line_units
        external_units = [] if self._has_translated_payload(block) else self._external_units_for_block(block, page_data, target_lang)
        if external_units and not phrase_units:
            return external_units
        if phrase_units:
            units = phrase_units + self._orphan_semantic_units(block, semantic_payload, target_lang, phrase_units)
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
        fallback_units = self._fallback_units(block, semantic_payload, target_lang)
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
                if large_gap and repeated_cluster_start:
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

    def _build_block_reconstruction_plan(self, page, page_data, block, target_lang):
        block_type = self._classify_block_for_reconstruction(block, page_data)
        geometry_ctx = self._build_block_geometry_context(page, page_data, block)
        line_templates = self._build_line_templates(block, geometry_ctx)
        semantic_payload = self._collect_block_semantic_payload(block)
        units = self._normalize_placable_units(block, semantic_payload, target_lang, page_data=page_data)
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
            return [], findings
        return ops, findings

    def _translated_coverage_entries_for_block(self, block, target_lang, page_data=None):
        semantic_payload = self._collect_block_semantic_payload(block)
        phrase_units = self._phrase_units(block, semantic_payload, target_lang)
        orphan_units = self._orphan_semantic_units(block, semantic_payload, target_lang, phrase_units) if phrase_units else []
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

    def _render_block_presence_fallback_ops(self, page, page_data, block, target_lang):
        block_rect = self._fitz_rect_from_bbox_like((block or {}).get("bbox"))
        if not isinstance(block_rect, fitz.Rect) or block_rect.get_area() <= 0:
            return []
        entries = self._translated_coverage_entries_for_block(block, target_lang, page_data=page_data)
        if not entries:
            return []
        # N'efface que si le fond n'est pas déjà propre — évite le double-whiteout sur bg_master.
        has_clean_background = bool(self._page_background_path(page_data))
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
            fontsize = min(float(style.get("size") or 12.0), max(6.0, rect.height * 0.72))
            wrapped = self._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            line_h = max(6.0, rect.height / max(1, len(wrapped)))
            rgb = self._resolve_text_color( style, block)
            for line_idx, line_text in enumerate(wrapped):
                cur_size = fontsize
                while cur_size > 5.5:
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
                        fallback_ops = self._render_block_presence_fallback_ops(page, page_data, block, target_lang)
                        if fallback_ops:
                            ops = fallback_ops
                            text_ops = sum(1 for op in ops if op.op_type == "draw_text_run")
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
                    fallback_ops = self._render_block_presence_fallback_ops(page, page_data, block, target_lang)
                    if fallback_ops:
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
        return all(str(unit.render_policy or "").strip().lower() == "external_flow" for unit in units)

    def _should_render_bbox_anchored(self, plan):
        units = [unit for unit in (plan.units or []) if self.reconstructor._clean_text_for_render(unit.text_translated or unit.text_source)]
        if not units:
            return False
        if any(str(unit.unit_type or "").strip().lower() == "translated_line" for unit in units):
            return False
        if not all(unit.relative_bbox for unit in units):
            return False
        anchored_count = 0
        for unit in units:
            policy = str(unit.render_policy or "").strip().lower()
            if policy in {"anchored_external", "anchored_text", "fixed_preserve"} or not unit.reflowable:
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
        raw_fragments = list((unit.metadata or {}).get("fragments") or [])
        render_policy = str(unit.render_policy or "").strip().lower()
        unit_type = str(unit.unit_type or "").strip().lower()
        if unit_type == "translated_line":
            text = self.reconstructor._clean_text_for_render(unit.text_translated or unit.text_source)
            if not text:
                return []
            return [{"text": text, "style": dict(unit.style or {}), "unit": unit}]
        preserve_as_single = render_policy in {"anchored_text", "fixed_preserve"} or not unit.reflowable
        if raw_fragments:
            segments = []
            for fragment in raw_fragments:
                text = self.reconstructor._clean_text_for_render(
                    fragment.get("translated_text") or fragment.get("text") or fragment.get("texte") or ""
                )
                if not text:
                    continue
                style = self.reconstructor._merge_styles(fragment.get("style") or {}, unit.style)
                if preserve_as_single:
                    tokens = [text]
                else:
                    inline_class = str(((fragment.get("expression_semantics") or {}).get("inline_class") or "")).strip().lower()
                    tokens = [text] if inline_class in {"technical_inline", "reference", "formula", "code"} else self._tokenize_text(text)
                for token in tokens:
                    segments.append({"text": token, "style": style, "unit": unit})
            if segments:
                return segments
        text = self.reconstructor._clean_text_for_render(unit.text_translated or unit.text_source)
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
            while fontsize >= 5.5:
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
            text = self.reconstructor._clean_text_for_render(unit.text_translated or unit.text_source)
            if not text:
                continue
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
            while fontsize > 5.5 and wrapped and (len(wrapped) * max(6.0, fontsize * 1.12)) > max(rect.height, fontsize * 1.3):
                fontsize -= 0.5
                wrapped = self.reconstructor._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            rgb = self.reconstructor._resolve_text_color( style, plan.source_block)
            line_h = max(6.0, fontsize * 1.12)
            align = self.reconstructor._normalize_alignment(unit.anchor_horizontal or plan.paragraph_alignment or plan.alignment)
            for line_idx, line_text in enumerate(wrapped):
                width = self.reconstructor._measure_text_width( line_text, fontsize, fontname, fontfile)
                x = rect.x0
                if align == "center":
                    x = max(rect.x0, rect.x0 + max(0.0, (rect.width - width) / 2.0))
                elif align == "right":
                    x = max(rect.x0, rect.x1 - width)
                baseline = rect.y0 + min(rect.height - 1.0, (line_idx + 1) * line_h * 0.82)
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
            text = self.reconstructor._clean_text_for_render(unit.text_translated or unit.text_source)
            if not text:
                continue
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
            while fontsize > 5.5 and wrapped and (len(wrapped) * max(6.0, fontsize * 1.12)) > max(rect.height, fontsize * 1.3):
                fontsize -= 0.5
                wrapped = self.reconstructor._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            rgb = self.reconstructor._resolve_text_color( style, plan.source_block)
            align = self.reconstructor._normalize_alignment(unit.anchor_horizontal or plan.paragraph_alignment or plan.alignment)
            if align == "end":
                align = "right"
            elif align == "start":
                align = "left"
            line_h = max(6.0, fontsize * 1.12)
            top_y = rect.y0
            for line_text in wrapped:
                width = self.reconstructor._measure_text_width( line_text, fontsize, fontname, fontfile)
                x = rect.x0
                if align == "center":
                    x = max(rect.x0, rect.x0 + max(0.0, (rect.width - width) / 2.0))
                elif align == "right":
                    x = max(rect.x0, rect.x1 - width)
                baseline = top_y + min(rect.height - 1.0, fontsize * 0.82)
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
                top_y += line_h
                if top_y + line_h * 0.6 > rect.y1:
                    break
        return ops

    def _finalize_line(self, page, plan, template, segments, is_last_line):
        if not segments:
            return []
        alignment = self.reconstructor._normalize_alignment(template.alignment or plan.paragraph_alignment or plan.alignment)
        default_gap = max(2.0, min(6.0, template.ascent * 0.22))
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
                can_wrap_segment = (unit.reflowable and not preserve_as_single) or str(unit.unit_type or "").strip().lower() == "translated_line"
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

    def render(self, page, plan):
        if self._should_render_relative_slot_mode(plan):
            return self._render_relative_slots(page, plan)
        if self._should_render_bbox_anchored(plan):
            return self._render_bbox_anchored(page, plan)
        best_ops = []
        best_finding_count = None
        severe = {"overflow", "text_overlap", "protected_overlap"}
        for scale in (1.0, 0.96, 0.92, 0.88, 0.84, 0.8, 0.76, 0.72):
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
        if fallback_ops:
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
        # Forcer monospace pour le code
        mono_style = {**base_style, "font": base_style.get("font") or "courier", "flags": {**(base_style.get("flags") or {}), "monospace": True}}
        _, fontfile, builtin, fontname = self.reconstructor._resolve_style_font( page, mono_style, text="x")
        fontsize = min(float(mono_style.get("size") or 10.0), max(5.5, block_rect.height / max(1, len(lines)) * 0.82))
        rgb = self.reconstructor._resolve_text_color( mono_style, block)
        for idx, line in enumerate(lines):
            text = self.reconstructor._clean_text_for_render(
                self.reconstructor._line_source_text(line) or self.reconstructor._line_translated_text(line)
            )
            if not text:
                continue
            template_lines = plan.line_templates
            if template_lines and idx < len(template_lines):
                tmpl = template_lines[idx]
                baseline = tmpl.baseline_y
                x = tmpl.left_x
            else:
                line_h = max(fontsize * 1.15, block_rect.height / max(1, len(lines)))
                baseline = block_rect.y0 + (idx + 0.82) * line_h
                x = block_rect.x0
            text_rect = fitz.Rect(x, baseline - max(1.0, fontsize * 0.82), block_rect.x1, baseline + max(1.0, fontsize * 0.18))
            ops.append(
                self._emit_text_run(
                    plan, text, text_rect, (x, baseline),
                    {**mono_style, "size": fontsize},
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
