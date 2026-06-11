from __future__ import annotations

import importlib.machinery
import importlib.util
import copy
import hashlib
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageStat

from background_inpainter import get_background_inpainter
from font_resolver import FontResolver
from text_composer import ComposeOptions, TextComposer

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
    rotation_deg: int = 0


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
class BlockRenderVerdict:
    status: str
    ok: bool
    block_id: str
    causes: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    text_ops_expected: int = 0
    text_ops_rendered: int = 0
    recommended_strategy: str = ""


@dataclass
class CandidateScore:
    value: float
    status: str
    penalties: dict[str, float] = field(default_factory=dict)
    hard_failures: list[str] = field(default_factory=list)


@dataclass
class RenderCandidate:
    candidate_id: str
    strategy: str
    ops: list["BlockRenderOp"] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    score: CandidateScore | None = None


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
    page_data: dict[str, Any] = field(default_factory=dict)
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
        self.two_pass_reconstruction = os.getenv("RECONSTRUCTOR_TWO_PASS", "0").strip().lower() in {"1", "true", "yes", "on"}
        self._legacy = None
        self._rendered_signatures = set()
        self._debug_page_images: list[Path] = []
        self.font_resolver = FontResolver()
        self._font_objects: dict[str, fitz.Font] = {}
        self._page_font_aliases: dict[tuple, str] = {}
        self._document_font_fallbacks: dict[tuple, dict[str, Any]] = {}
        self._font_truly_supports_cache: dict[tuple, bool] = {}
        self._render_agent: Any = None
        self._render_agent_loaded: bool = False
        self._local_background_cache: dict[tuple, str] = {}
        self.background_inpainter = get_background_inpainter()
        self.text_composer = TextComposer()

    # ------------------------------------------------------------------
    # Rendu final déterministe
    # ------------------------------------------------------------------

    def _get_render_agent(self) -> Any:
        """
        Les agents ne participent plus au chemin critique du rendu final.

        On conserve cette méthode comme point de compatibilité API pour les
        tests et pour le code historique, mais elle ne charge jamais P5.
        """
        self._render_agent_loaded = True
        self._render_agent = None
        return None

    def _ai_refine_render_strategy(
        self,
        block: dict,
        heuristic_strategy: str,
        page_data: dict | None = None,
    ) -> str:
        """
        Chemin de compatibilité historique.

        Le choix de stratégie de rendu reste désormais strictement
        déterministe et purement Python. Cette méthode renvoie donc
        toujours la stratégie heuristique calculée localement.
        """
        return heuristic_strategy

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

    def _style_flags_for_font_fallback(self, style):
        style_dict = style if isinstance(style, dict) else {}
        flags = dict(style_dict.get("flags") or {})
        font_name = str(style_dict.get("font") or style_dict.get("font_name") or "").casefold()
        normalized = str(style_dict.get("font_key_normalized") or "").casefold()
        family_key = f"{font_name} {normalized}"
        is_mono = bool(flags.get("monospace")) or any(token in family_key for token in ("courier", "mono", "consolas"))
        is_serif = bool(flags.get("serif")) or any(
            token in family_key
            for token in (
                "times",
                "serif",
                "roman",
                "garamond",
                "georgia",
                "janson",
                "baskerville",
                "palatino",
                "minion",
                "caslon",
                "bookman",
            )
        )
        is_bold = bool(flags.get("bold")) or any(token in family_key for token in ("bold", "black", "semibold", "demibold"))
        is_italic = bool(flags.get("italic")) or any(token in family_key for token in ("italic", "oblique"))
        return {
            "serif": is_serif and not is_mono,
            "bold": is_bold,
            "italic": is_italic,
            "monospace": is_mono,
        }

    def _text_requires_unicode_safe_font(self, text):
        if not text:
            return False
        for ch in str(text):
            if ord(ch) > 127 and not unicodedata.category(ch).startswith("C"):
                return True
        return False

    def _is_risky_extracted_font(self, fontfile, style):
        style_dict = style if isinstance(style, dict) else {}
        if style_dict.get("embedded_font_path"):
            return True
        font_path = str(fontfile or "")
        if not font_path:
            return False
        try:
            real = os.path.realpath(font_path)
            temp_root = os.path.realpath(os.path.join(tempfile.gettempdir(), "docs_parser_embedded_fonts"))
            if real.startswith(temp_root + os.sep) or real == temp_root:
                return True
        except Exception:
            pass
        name = str(style_dict.get("font") or style_dict.get("font_name") or "")
        return bool(re.match(r"^[A-Z]{6}\+", name))

    def _resolve_compatible_font(self, page, style, text=""):
        style_dict = style if isinstance(style, dict) else {}
        probe_text = self._clean_text_for_render(text or "")
        resolved = self.font_resolver.resolve(style_dict, text=probe_text)
        fontfile = resolved.get("fontfile")
        builtin = resolved.get("builtin")
        flags = self._style_flags_for_font_fallback(style_dict)
        source_font_key = (
            str(style_dict.get("font") or style_dict.get("font_name") or "").casefold(),
            str(style_dict.get("font_key_normalized") or "").casefold(),
            bool(flags.get("serif")),
            bool(flags.get("bold")),
            bool(flags.get("italic")),
            bool(flags.get("monospace")),
        )
        needs_fallback = False
        if probe_text:
            if fontfile:
                needs_fallback = not self._font_truly_supports_text(fontfile, probe_text)
            elif builtin:
                needs_fallback = not self._builtin_font_truly_supports_text(builtin, probe_text)
            if self._text_requires_unicode_safe_font(probe_text) and self._is_risky_extracted_font(fontfile, style_dict):
                needs_fallback = True
        if needs_fallback:
            cached_fallback = dict(self._document_font_fallbacks.get(source_font_key) or {})
            fallback_file = cached_fallback.get("fontfile")
            fallback_builtin = cached_fallback.get("builtin")
            if not fallback_file and not fallback_builtin:
                fallback_file = self._get_system_unicode_font(
                    is_serif=bool(flags.get("serif")),
                    is_bold=bool(flags.get("bold")),
                    is_italic=bool(flags.get("italic")),
                    is_mono=bool(flags.get("monospace")),
                )
                if fallback_file:
                    self._document_font_fallbacks[source_font_key] = {
                        "fontfile": fallback_file,
                        "builtin": None,
                        "reason": "missing_target_language_glyphs",
                    }
                else:
                    fallback_builtin = self.font_resolver._builtin_font(flags)
                    self._document_font_fallbacks[source_font_key] = {
                        "fontfile": None,
                        "builtin": fallback_builtin,
                        "reason": "missing_target_language_glyphs",
                    }
            if fallback_file:
                fontfile = fallback_file
                builtin = None
                resolved = {
                    **resolved,
                    "fontfile": fallback_file,
                    "builtin": None,
                    "unicode_fallback": True,
                    "font_substitution_reason": "missing_target_language_glyphs",
                    "document_level_font_fallback": True,
                }
            else:
                builtin = fallback_builtin or self.font_resolver._builtin_font(flags)
                fontfile = None
                resolved = {
                    **resolved,
                    "fontfile": None,
                    "builtin": builtin,
                    "unicode_fallback": True,
                    "font_substitution_reason": "missing_target_language_glyphs",
                    "document_level_font_fallback": True,
                }
        if page is None:
            fontname = builtin or str(style_dict.get("font") or "helv")
        else:
            fontname = self._resolve_page_fontname(page, fontfile, builtin)
        return resolved, fontfile, builtin, fontname

    def _resolve_style_font(self, page, style, text=""):
        return self._resolve_compatible_font(page, style, text=text)

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
            if "_document_font_fallbacks" not in self.__dict__:
                self.__dict__["_document_font_fallbacks"] = {}
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

    def _render_contract_for_item(self, item):
        if not isinstance(item, dict):
            return {}
        value = item.get("render_contract")
        return value if isinstance(value, dict) else {}

    RECONSTRUCTION_CONTRACTS_VERSION = "reconstruction_contracts.v1"
    RECONSTRUCTION_CONTRACTS = {
        "paragraph": {
            "render_mode": "prose_reflow",
            "font_size_policy": "preserve_extracted",
            "style_policy": "preserve_visible_style",
            "shrink_max_pt": 1.2,
            "min_font_ratio": 0.90,
            "allow_expansion": True,
            "strict_non_reflow": False,
        },
        "caption": {
            "render_mode": "line_preserve",
            "font_size_policy": "bounded_shrink",
            "style_policy": "preserve_visible_style",
            "shrink_max_pt": 1.0,
            "min_font_ratio": 0.90,
            "allow_expansion": True,
            "strict_non_reflow": False,
        },
        "table_cell": {
            "render_mode": "cell_locked",
            "font_size_policy": "bounded_shrink",
            "style_policy": "preserve_visible_style",
            "shrink_max_pt": 0.8,
            "min_font_ratio": 0.95,
            "allow_expansion": False,
            "strict_non_reflow": True,
        },
        "table_cell_micro": {
            "render_mode": "cell_locked",
            "font_size_policy": "micro_label_preserve",
            "style_policy": "preserve_visible_style",
            "shrink_max_pt": 0.2,
            "min_font_ratio": 0.98,
            "allow_expansion": False,
            "strict_non_reflow": True,
            "preserve_if_translation_overflows": True,
        },
        "table_cell_symbolic": {
            "render_mode": "cell_locked",
            "font_size_policy": "micro_label_preserve",
            "style_policy": "preserve_visible_style",
            "shrink_max_pt": 0.0,
            "min_font_ratio": 1.0,
            "allow_expansion": False,
            "strict_non_reflow": True,
            "translation_policy": "preserve",
            "preserve_if_translation_overflows": True,
        },
        "table_cell_numeric": {
            "render_mode": "cell_locked",
            "font_size_policy": "micro_label_preserve",
            "style_policy": "preserve_visible_style",
            "shrink_max_pt": 0.0,
            "min_font_ratio": 1.0,
            "allow_expansion": False,
            "strict_non_reflow": True,
            "translation_policy": "preserve",
            "preserve_if_translation_overflows": True,
        },
        "code_block": {
            "render_mode": "line_preserve",
            "font_size_policy": "micro_label_preserve",
            "style_policy": "preserve_monospace_unicode",
            "shrink_max_pt": 0.0,
            "min_font_ratio": 1.0,
            "allow_expansion": False,
            "strict_non_reflow": True,
            "translation_policy": "preserve",
        },
        "formula_block": {
            "render_mode": "bbox_anchored",
            "font_size_policy": "micro_label_preserve",
            "style_policy": "preserve_source_visual",
            "shrink_max_pt": 0.0,
            "min_font_ratio": 1.0,
            "allow_expansion": False,
            "strict_non_reflow": True,
            "translation_policy": "preserve",
        },
        "inline_formula": {
            "render_mode": "bbox_anchored",
            "font_size_policy": "micro_label_preserve",
            "style_policy": "preserve_source_visual",
            "shrink_max_pt": 0.0,
            "min_font_ratio": 1.0,
            "allow_expansion": False,
            "strict_non_reflow": True,
            "translation_policy": "preserve",
        },
        "url_reference": {
            "render_mode": "bbox_anchored",
            "font_size_policy": "micro_label_preserve",
            "style_policy": "preserve_visible_style",
            "shrink_max_pt": 0.0,
            "min_font_ratio": 1.0,
            "allow_expansion": False,
            "strict_non_reflow": True,
            "translation_policy": "preserve",
        },
        "figure_label": {
            "render_mode": "bbox_anchored",
            "font_size_policy": "micro_label_preserve",
            "style_policy": "preserve_visible_style",
            "shrink_max_pt": 0.4,
            "min_font_ratio": 0.95,
            "allow_expansion": False,
            "strict_non_reflow": True,
        },
        "figure_region": {
            "render_mode": "bbox_anchored",
            "font_size_policy": "preserve_extracted",
            "style_policy": "preserve_source_visual",
            "shrink_max_pt": 0.0,
            "min_font_ratio": 1.0,
            "allow_expansion": False,
            "strict_non_reflow": True,
            "translation_policy": "preserve",
        },
        "note_box": {
            "render_mode": "prose_reflow",
            "font_size_policy": "footnote_allowed_shrink",
            "style_policy": "preserve_visible_style",
            "shrink_max_pt": 1.0,
            "min_font_ratio": 0.90,
            "allow_expansion": True,
            "strict_non_reflow": False,
        },
        "toc_entry": {
            "render_mode": "line_preserve",
            "font_size_policy": "bounded_shrink",
            "style_policy": "preserve_visible_style",
            "shrink_max_pt": 0.8,
            "min_font_ratio": 0.95,
            "allow_expansion": False,
            "strict_non_reflow": True,
        },
    }

    def _reconstruction_contract_key_for_block(self, block, page_data=None):
        if not isinstance(block, dict):
            return "paragraph"
        page_role = str((page_data or {}).get("page_role") or "").strip().lower() if isinstance(page_data, dict) else ""
        if page_role == "toc":
            object_type = str(block.get("object_type") or "").strip().lower()
            object_class = str(block.get("object_class") or "").strip().lower()
            if object_type not in {"figure_region", "image_region", "drawing_region"} and object_class not in {"visual"}:
                return "toc_entry"
        document_contract = block.get("document_object_contract") if isinstance(block.get("document_object_contract"), dict) else {}
        reconstruction_contract = document_contract.get("reconstruction") if isinstance(document_contract.get("reconstruction"), dict) else {}
        contract_key = str(reconstruction_contract.get("contract_key") or "").strip().lower()
        if contract_key in self.RECONSTRUCTION_CONTRACTS:
            return contract_key
        payload = dict(block.get("object_comprehension") or {})
        object_type = str(block.get("object_type") or payload.get("object_type") or "").strip().lower()
        object_class = str(block.get("object_class") or payload.get("object_class") or "").strip().lower()
        object_subtype = str(block.get("object_subtype") or payload.get("object_subtype") or "").strip().lower()
        role = str(block.get("role") or "").strip().lower()
        text = self._clean_text_for_render(self._translated_text_from_block(block) or self._source_text_from_block(block))
        if object_type in {"code_block", "code_line", "inline_code"} or role in {"code", "code_block"} or object_class == "technical":
            return "code_block"
        if object_type in {"formula_block", "formula_line", "formula_equation", "formula_symbol", "inline_formula_cluster"} or object_class == "formula":
            return "formula_block"
        if object_type in {"inline_formula", "chemical_formula"} or object_subtype in {"inline_formula", "formula_symbol", "formula_equation"}:
            return "inline_formula"
        line_count = len([line for line in (block.get("lines") or []) if isinstance(line, dict)])
        toc_like = (
            object_type in {"toc_entry", "toc_leader"}
            or role in {"toc", "toc_entry"}
            or (
                line_count >= 8
                and (
                    re.search(r"(?:\.\s*){6,}", text)
                    or sum(1 for line in (block.get("lines") or []) if re.search(r"(?:\.\s*){4,}", self._line_source_text(line) or self._line_translated_text(line))) >= 3
                )
            )
        )
        if toc_like:
            return "toc_entry"
        if object_type in {"reference_link", "url", "web_url", "email_address", "doi_reference", "arxiv_reference"} or re.search(r"(https?://|www\.|doi:|\b10\.\d{4,9}/)", text, flags=re.IGNORECASE):
            return "url_reference"
        if object_type in {"table_cell_micro", "micro_table_cell"} or object_subtype in {"table_cell_micro", "micro_cell"}:
            return "table_cell_micro"
        if object_type in {"table_cell_numeric", "numeric_cell"} or object_subtype in {"table_cell_numeric", "numeric_cell"}:
            return "table_cell_numeric"
        if object_type in {"table_cell_symbolic", "symbolic_cell"} or object_subtype in {"table_cell_symbolic", "symbolic_cell"}:
            return "table_cell_symbolic"
        if object_type in {"table_block", "table_cell", "table_row", "table_cell_text"} or object_class == "tabular":
            return "table_cell"
        if object_type in {"figure_region", "image_region", "drawing_region", "chart_region", "dense_diagram_region", "complex_vector_region", "clipping_region", "mask_region", "overlay_stack_region"} or object_class == "visual":
            return "figure_region"
        if object_type in {"figure_axis_label", "figure_label", "axis_label", "diagram_label", "chart_label", "legend_label", "short_label", "micro_label"} or object_subtype == "micro_label" or object_class == "visual_label":
            return "figure_label"
        if object_type in {"note_box", "footnote"} or role in {"note", "footnote"}:
            return "note_box"
        if object_type in {"figure_caption", "table_caption", "caption"} or role in {"figure_caption", "table_caption", "caption"}:
            return "caption"
        return "paragraph"

    def _reconstruction_contract_for_block(self, block, page_data=None):
        key = self._reconstruction_contract_key_for_block(block, page_data=page_data)
        contract = dict(self.RECONSTRUCTION_CONTRACTS.get(key) or self.RECONSTRUCTION_CONTRACTS["paragraph"])
        contract["contract_key"] = key
        contract["schema_version"] = self.RECONSTRUCTION_CONTRACTS_VERSION
        return contract

    def _contract_alignment_value(self, item, *, key="paragraph", fallback="left"):
        contract = self._render_contract_for_item(item)
        alignment = dict(contract.get("alignment") or {})
        if key == "inline":
            return self._normalize_alignment(alignment.get("inline") or item.get("alignment") or fallback)
        return self._normalize_alignment(alignment.get("paragraph") or item.get("alignment") or fallback)

    def _contract_background_mode(self, item):
        contract = self._render_contract_for_item(item)
        background = dict(contract.get("background") or {})
        mode = str(background.get("mode") or "").strip().lower()
        layout_mode = dict(contract.get("layout_mode") or {})
        source_layout_mode = dict((item or {}).get("source_layout_mode") or {}) if isinstance(item, dict) else {}
        source_render_contract = str(
            source_layout_mode.get("render_contract")
            or layout_mode.get("source_render_contract")
            or ""
        ).strip().lower()
        source_line_flow = str(
            source_layout_mode.get("line_flow")
            or layout_mode.get("source_line_flow")
            or ""
        ).strip().lower()
        can_reflow = bool(
            source_layout_mode.get("can_reflow_within_paragraph", layout_mode.get("can_reflow_within_paragraph", False))
        )
        if source_render_contract in {"reflow_block", "paragraph_reflow"} or (
            can_reflow and source_line_flow in {"inline_reflow", "preserve_paragraphs", "rewrap"}
        ):
            return "plain_whiteout"
        return mode

    def _contract_preserves_horizontal_slot(self, item):
        contract = self._render_contract_for_item(item)
        alignment = dict(contract.get("alignment") or {})
        family = str(contract.get("family") or "").strip().lower()
        layout_mode = dict(contract.get("layout_mode") or {})
        source_contract = str(layout_mode.get("source_render_contract") or "").strip().lower()
        line_flow = str(layout_mode.get("source_line_flow") or "").strip().lower()
        return bool(
            alignment.get("locked_position")
            or family in {"table_cell", "anchored", "fixed_lines", "code_line", "background_only"}
            or source_contract in {"fixed_slots", "preserve_breaks", "single_line_or_shrink"}
            or line_flow in {"fixed_lines", "preserve_line_breaks", "single_line"}
        )

    def _clean_text_for_render(self, text):
        text = str(text or "")
        text = text.replace("\u00a0", " ")
        text = text.replace("\ufffd", "")
        text = text.replace("\u25a0", "")
        text = re.sub(r"\s*'\s*", "'", text)
        text = re.sub(r"\s*’\s*", "’", text)
        text = re.sub(r"\s+([,.;!?])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]}])", r"\1", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        return text.strip()

    def _page_orientation(self, page_data):
        width_pt, height_pt = self._page_size_pt(page_data)
        if width_pt > height_pt * 1.05:
            return "landscape"
        return "portrait"

    def _rotation_deg_for_bbox_text(self, bbox_like, text="", fallback=0):
        rect = self._fitz_rect_from_bbox_like(bbox_like)
        if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
            return int(fallback or 0)
        text = self._clean_text_for_render(text or "")
        if not text:
            return int(fallback or 0)
        width = max(1.0, rect.width)
        height = max(1.0, rect.height)
        if height > width * 1.8:
            return 90
        return int(fallback or 0)

    def _rotation_deg_for_item(self, item, *, bbox_like=None, text="", fallback=0):
        if not isinstance(item, dict):
            return int(fallback or 0)
        layout_attributes = dict(item.get("layout_attributes") or {})
        render_contract = self._render_contract_for_item(item)
        positioning = dict(render_contract.get("positioning") or {})
        for candidate in (
            layout_attributes.get("rotation_deg"),
            layout_attributes.get("text_rotation_deg"),
            item.get("rotation_deg"),
            positioning.get("rotation_deg"),
        ):
            try:
                value = int(round(float(candidate)))
            except Exception:
                value = None
            if value is not None:
                value = value % 360
                if value in {0, 90, 180, 270}:
                    return value
        probe_text = (
            text
            or item.get("text")
            or item.get("translated_text")
            or item.get("texte")
            or ""
        )
        return self._rotation_deg_for_bbox_text(
            bbox_like or item.get("bbox"),
            probe_text,
            fallback=fallback,
        )

    def _format_toc_label_for_render(self, label_type, text):
        text = self._clean_text_for_render(text or "")
        if not text:
            return ""
        if label_type == "part_title":
            text = re.sub(r'^(?:partie|part)\s+', '', text, flags=re.IGNORECASE)
            text = text.upper()
        return text

    def _extract_leading_marker(self, label):
        text = self._clean_text_for_render(label or "")
        match = re.match(r"^([•■·\-\u2022])\s*(.*)$", text)
        if match:
            return match.group(1), self._clean_text_for_render(match.group(2))
        return "", text

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

    def _expand_text_rect_within_block(self, rect, block_rect, fontsize, line_count=1, line_height_factor=1.12):
        rect = fitz.Rect(rect)
        block_rect = fitz.Rect(block_rect)
        if rect.get_area() <= 0 or block_rect.get_area() <= 0:
            return rect
        line_count = max(1, int(line_count or 1))
        fontsize = max(1.0, float(fontsize or 1.0))
        target_h = max(rect.height, fontsize * max(1.0, float(line_height_factor or 1.12)) * line_count)
        target_h = min(target_h, block_rect.height)
        if target_h <= rect.height + 0.01:
            return fitz.Rect(
                max(block_rect.x0, rect.x0),
                max(block_rect.y0, rect.y0),
                min(block_rect.x1, rect.x1),
                min(block_rect.y1, rect.y1),
            )

        center_y = (rect.y0 + rect.y1) / 2.0
        y0 = center_y - target_h / 2.0
        y1 = center_y + target_h / 2.0
        if y0 < block_rect.y0:
            y1 += block_rect.y0 - y0
            y0 = block_rect.y0
        if y1 > block_rect.y1:
            y0 -= y1 - block_rect.y1
            y1 = block_rect.y1
        y0 = max(block_rect.y0, y0)
        y1 = min(block_rect.y1, y1)
        return fitz.Rect(
            max(block_rect.x0, rect.x0),
            y0,
            min(block_rect.x1, rect.x1),
            y1,
        )

    def _translated_text_from_block(self, block):
        translated = self._clean_text_for_render((block or {}).get("translated_text") or "")
        if translated:
            return translated
        parts = []
        for line in (block or {}).get("lines") or []:
            line_text = self._clean_text_for_render((line or {}).get("translated_text") or (line or {}).get("line_text") or (line or {}).get("text") or "")
            if line_text:
                parts.append(line_text)
                continue
            phrase_parts = []
            for phrase in (line or {}).get("phrases") or []:
                phrase_text = self._clean_text_for_render(
                    (phrase or {}).get("translated_text") or (phrase or {}).get("texte") or (phrase or {}).get("text") or ""
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
        object_type = str((block or {}).get("object_type") or ((block or {}).get("object_comprehension") or {}).get("object_type") or "").strip().lower()
        object_class = str((block or {}).get("object_class") or ((block or {}).get("object_comprehension") or {}).get("object_class") or "").strip().lower()
        descriptor_group_ids = dict((block or {}).get("descriptor_group_ids") or {})
        editorial_semantics = dict((block or {}).get("editorial_semantics") or {})
        flow_class = str(editorial_semantics.get("flow_class") or "").strip().lower()
        if object_type in {"table_block", "table_cell", "table_row"} or object_class == "tabular":
            return "table"
        if object_type in {"code_block", "code_line"} or object_class == "technical":
            return "code"
        if object_type in {"formula_block", "formula_line", "inline_formula_cluster"} or object_class == "formula":
            return "annotation"
        if object_type in {"figure_caption", "table_caption", "caption"}:
            return "caption"
        if object_type in {"title", "section_heading", "page_header", "page_footer", "page_number"}:
            return "heading"
        if object_type in {"diagram_label", "chart_label", "axis_label", "legend_label", "short_label"} or object_class == "visual_label":
            return "annotation"
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
        object_type = str(block.get("object_type") or ((block.get("object_comprehension") or {}).get("object_type")) or "").strip().lower()
        object_class = str(block.get("object_class") or ((block.get("object_comprehension") or {}).get("object_class")) or "").strip().lower()
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
        if object_type in {"code_block", "code_line"} or object_class == "technical" or role in {"code", "code_block"} or (has_code_pattern and render_policy == "fixed_preserve") or is_code_block:
            content_class = "code"
            render_strategy = "code_preserve"
        elif object_type in {"formula_block", "formula_line", "inline_formula_cluster"} or object_class == "formula" or (has_math_chars and (role in {"formula", "equation"} or flow_class == "symbolic")):
            content_class = "formula"
            render_strategy = "bitmap_preserve"
        elif object_type in {"title", "section_heading", "page_header", "page_footer"} or role in {"heading", "title", "section_title", "chapter_title"}:
            content_class = "heading"
            render_strategy = "heading_reflow"
        elif object_type in {"figure_caption", "table_caption", "caption"} or role in {"figure_caption", "table_caption", "caption"}:
            content_class = "caption"
            render_strategy = "prose_reflow"
        elif object_type in {"table_block", "table_cell", "table_row"} or object_class == "tabular":
            content_class = "table"
            render_strategy = "label_stack"
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
        # Etape G - le rendu final reste déterministe : pas d'affinage agent.
        if render_strategy in {"prose_reflow", "label_stack"}:
            ai_strategy = self._ai_refine_render_strategy(block, render_strategy, page_data)
            if ai_strategy != render_strategy:
                render_strategy = ai_strategy
                content_class = "prose" if render_strategy == "prose_reflow" else "label"
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
        target_contract = self._reconstruction_contract_for_block(block, page_data=page_data)
        contract_key = target_contract.get("contract_key")
        if contract_key in {"code_block"}:
            content_class = "code"
            render_strategy = "code_preserve"
        elif contract_key in {"formula_block", "inline_formula"}:
            content_class = "formula"
            render_strategy = "bitmap_preserve"
        elif contract_key == "table_cell":
            content_class = "table"
            render_strategy = "label_stack"
        elif contract_key == "figure_region":
            content_class = "visual"
            render_strategy = "source_overlay"
        elif contract_key in {"figure_label", "url_reference", "toc_entry"}:
            content_class = "label" if contract_key != "url_reference" else "reference"
            render_strategy = "label_stack"
        elif contract_key == "note_box":
            content_class = "note"
            render_strategy = "prose_reflow"
        if render_strategy == "prose_reflow":
            font_normalization = "fit_to_bbox"
            text_flow_mode = "continuous"
        elif render_strategy == "label_stack":
            font_normalization = "block_median"
            text_flow_mode = "line_by_line"
        else:
            font_normalization = "span_original"
            text_flow_mode = "atomic"
        return BlockSemanticProfile(
            block_id=block_id,
            content_class=content_class,
            render_strategy=render_strategy,
            font_normalization=font_normalization,
            allow_vertical_expansion=bool(target_contract.get("allow_expansion", allow_vertical_expansion)),
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
        probe_chars = []
        seen = set()
        chars = list(text)
        chars.sort(key=lambda ch: 0 if ord(ch) > 127 else 1)
        for ch in chars:
            if ch.isspace() or unicodedata.category(ch).startswith("C"):
                continue
            if ch in seen:
                continue
            seen.add(ch)
            probe_chars.append(ch)
            if len(probe_chars) >= 8:
                break
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
                rendered = self._clean_text_for_render(tmp_page.get_text("text").strip())
                tmp_doc.close()
                rendered_chars = {ch for ch in rendered if not ch.isspace()}
                expected_chars = {ch for ch in test_chars if not ch.isspace()}
                ok = bool(rendered_chars) and expected_chars.issubset(rendered_chars)
            except Exception:
                # Si le test de rendu echoue, se fier a has_glyph
                ok = True
            self._font_truly_supports_cache[cache_key] = ok
            return ok
        except Exception:
            self._font_truly_supports_cache[cache_key] = False
            return False

    def _builtin_font_truly_supports_text(self, builtin, text):
        builtin = str(builtin or "").strip()
        if not builtin or not text:
            return True
        probe_chars = []
        seen = set()
        chars = list(text)
        chars.sort(key=lambda ch: 0 if ord(ch) > 127 else 1)
        for ch in chars:
            if ch.isspace() or unicodedata.category(ch).startswith("C"):
                continue
            if ch in seen:
                continue
            seen.add(ch)
            probe_chars.append(ch)
            if len(probe_chars) >= 8:
                break
        if not probe_chars:
            return True
        cache_key = (f"builtin:{builtin}", "".join(sorted(probe_chars)))
        cached = self._font_truly_supports_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            tmp_doc = fitz.open()
            tmp_page = tmp_doc.new_page(width=240, height=60)
            tmp_page.insert_text((10, 35), "".join(probe_chars), fontname=builtin, fontsize=14)
            rendered = self._clean_text_for_render(tmp_page.get_text("text").strip())
            tmp_doc.close()
            rendered_chars = {ch for ch in rendered if not ch.isspace()}
            expected_chars = {ch for ch in probe_chars if not ch.isspace()}
            ok = bool(rendered_chars) and expected_chars.issubset(rendered_chars)
        except Exception:
            ok = False
        self._font_truly_supports_cache[cache_key] = ok
        return ok

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
        if builtin and self._builtin_font_truly_supports_text(builtin, text):
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
        object_payload = dict((raw_unit or {}).get("object_comprehension") or {})
        object_type = str((raw_unit or {}).get("object_type") or object_payload.get("object_type") or "").strip().lower()
        object_class = str((raw_unit or {}).get("object_class") or object_payload.get("object_class") or "").strip().lower()
        object_subtype = str((raw_unit or {}).get("object_subtype") or object_payload.get("object_subtype") or "").strip().lower()
        inline_object_type = str((raw_unit or {}).get("inline_object_type") or object_payload.get("inline_object_type") or "").strip().lower()
        inline_object_subtype = str((raw_unit or {}).get("inline_object_subtype") or object_payload.get("inline_object_subtype") or "").strip().lower()
        translation_policy = dict((raw_unit or {}).get("translation_policy") or object_payload.get("translation_policy") or {})
        inline_class = str(expression_semantics.get("inline_class") or "").strip().lower()
        unit_type = str((raw_unit or {}).get("unit_type") or "").strip().lower()
        editorial_semantics = dict((raw_unit or {}).get("editorial_semantics") or {})
        flow_class = str(editorial_semantics.get("flow_class") or "").strip().lower()
        text_clean = self._clean_text_for_render(text or "")
        short_text = bool(text_clean and len(text_clean) <= 64)
        child_summary = self._protected_fragment_summary(child_units)
        unit_profile = "editorial_phrase"
        preserve_inline_types = {
            "web_url", "email_address", "doi_reference", "arxiv_reference",
            "inline_formula", "chemical_formula", "function_call",
            "technical_identifier", "currency_amount", "percentage_value",
            "measurement_value", "date_value", "reference_link",
        }
        preserve_object_types = {
            "formula_block", "formula_line", "inline_formula_cluster",
            "code_block", "code_line", "inline_code",
            "figure_region", "image_region", "drawing_region", "chart_region", "seal_region",
        }
        if (
            inline_class in {"formula", "reference", "code"}
            or inline_object_type in preserve_inline_types
            or inline_object_subtype in preserve_inline_types
            or object_type in preserve_object_types
            or object_class in {"visual", "technical", "formula"}
            or str(translation_policy.get("render_policy") or "").strip().lower() in {"fixed_preserve", "source_overlay", "cell_locked"}
            or child_summary["has_immutable"]
        ):
            unit_profile = "protected_inline"
        elif (
            unit_type in {"short_label", "chart_label", "diagram_label", "formula_label"}
            or object_type in {"diagram_label", "chart_label", "axis_label", "legend_label", "short_label"}
            or object_class == "visual_label"
            or flow_class == "anchored_annotation"
        ):
            unit_profile = "anchored_label"
        elif (
            object_type in {"table_block", "table_cell", "table_row"}
            or object_class == "tabular"
            or block_profile.get("block_profile") in {"technical_structured", "tabular_dense"}
        ):
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
            "object_type": object_type,
            "object_class": object_class,
            "object_subtype": object_subtype,
            "inline_object_type": inline_object_type,
            "inline_object_subtype": inline_object_subtype,
            "translation_policy": translation_policy,
            "has_protected_fragments": bool(child_summary["has_protected"]),
            "has_immutable_fragments": bool(child_summary["has_immutable"]),
        }

    def _build_page_reconstruction_context(self, page_data, target_lang):
        adaptive_profile = self._page_adaptive_profile(page_data)
        context = {
            "target_lang": target_lang,
            "writing_direction": "right_to_left" if target_lang in {"ar", "he", "fa"} else "left_to_right",
            "page_orientation": self._page_orientation(page_data),
            "adaptive_profile": adaptive_profile,
            "reconstruction_contract": self._build_reconstruction_contract_context(page_data),
        }
        if isinstance(page_data, dict):
            page_data["_reconstruction_context"] = context
        self._attach_descriptor_v3_metadata(page_data, context["reconstruction_contract"])
        return context

    def _page_rebalance_bounds(self, page, page_data):
        if page is not None:
            try:
                rect = fitz.Rect(page.rect)
                if rect.get_area() > 0:
                    return rect
            except Exception:
                pass
        dims = dict((page_data or {}).get("dimensions") or {})
        width_px = float(dims.get("width") or dims.get("page_width") or 0.0)
        height_px = float(dims.get("height") or dims.get("page_height") or 0.0)
        if width_px > 0 and height_px > 0:
            return fitz.Rect(0.0, 0.0, width_px * self.pixel_to_point, height_px * self.pixel_to_point)
        return fitz.Rect(0.0, 0.0, 1240.0 * self.pixel_to_point, 1754.0 * self.pixel_to_point)

    def _bbox_overlap_ratio_xy(self, lhs, rhs):
        left = self._fitz_rect_from_bbox_like(lhs)
        right = self._fitz_rect_from_bbox_like(rhs)
        if not isinstance(left, fitz.Rect) or not isinstance(right, fitz.Rect):
            return 0.0
        if left.get_area() <= 0 or right.get_area() <= 0:
            return 0.0
        inter = left & right
        if inter.get_area() <= 0:
            return 0.0
        return float(inter.get_area()) / max(1.0, min(left.get_area(), right.get_area()))

    def _bbox_horizontal_overlap_ratio(self, lhs, rhs):
        left = self._fitz_rect_from_bbox_like(lhs)
        right = self._fitz_rect_from_bbox_like(rhs)
        if not isinstance(left, fitz.Rect) or not isinstance(right, fitz.Rect):
            return 0.0
        left_width = max(1.0, float(left.width))
        right_width = max(1.0, float(right.width))
        overlap = min(float(left.x1), float(right.x1)) - max(float(left.x0), float(right.x0))
        if overlap <= 0.0:
            return 0.0
        return float(overlap) / max(1.0, min(left_width, right_width))

    def _rebalance_block_is_eligible(self, block, page_data):
        if not isinstance(block, dict):
            return False
        contract_key = self._reconstruction_contract_key_for_block(block, page_data=page_data)
        if contract_key in {
            "toc_entry",
            "table_cell",
            "table_cell_micro",
            "table_cell_symbolic",
            "table_cell_numeric",
            "table_row",
            "table_block",
            "code_block",
            "code_line",
            "formula_block",
            "inline_formula",
            "figure_region",
            "image_region",
            "drawing_region",
            "diagram_label",
            "chart_label",
            "axis_label",
            "legend_label",
            "short_label",
            "url_reference",
        }:
            return False
        block_type = self._classify_block_for_reconstruction(block, page_data)
        if block_type not in {"editorial", "heading", "caption", "mixed"}:
            return False
        text = self._clean_text_for_render(self._translated_text_from_block(block) or self._source_text_from_block(block))
        if not text:
            return False
        if len(text) < 8 and block_type == "editorial":
            return False
        return True

    def _rebalance_candidate_size(self, block, rect, page_data):
        source_text = self._clean_text_for_render(self._source_text_from_block(block))
        translated_text = self._clean_text_for_render(self._translated_text_from_block(block))
        text = translated_text or source_text
        if not text:
            return rect.width, rect.height
        source_len = max(1, len(source_text or text))
        translated_len = max(1, len(translated_text or text))
        expansion = float(translated_len) / float(source_len)
        block_type = self._classify_block_for_reconstruction(block, page_data)
        line_count = max(1, len((block or {}).get("lines") or []))
        height_factor = 1.0
        if expansion > 1.0:
            height_factor += min(0.8, (expansion - 1.0) * (0.55 if block_type == "heading" else 0.75))
        if line_count > 1:
            height_factor += min(0.35, (line_count - 1) * 0.04)
        width_factor = 1.0
        if expansion > 1.0:
            width_factor += min(0.12, (expansion - 1.0) * 0.05)
        return rect.width * width_factor, rect.height * height_factor

    def _rebalance_safe_horizontal_span(self, ordered_blocks, current_index, rect, page_bounds):
        safe_left = float(page_bounds.x0)
        safe_right = float(page_bounds.x1)
        for idx, other in enumerate(ordered_blocks):
            if idx == current_index:
                continue
            other_rect = self._fitz_rect_from_bbox_like(self._rebalance_effective_bbox(other))
            if not isinstance(other_rect, fitz.Rect) or other_rect.get_area() <= 0:
                continue
            if self._bbox_overlap_ratio_xy(rect, other_rect) < 0.28:
                continue
            if other_rect.x1 <= rect.x0:
                safe_left = max(safe_left, float(other_rect.x1) + 2.0)
            elif other_rect.x0 >= rect.x1:
                safe_right = min(safe_right, float(other_rect.x0) - 2.0)
        if safe_right <= safe_left:
            return float(page_bounds.x0), float(page_bounds.x1)
        return safe_left, safe_right

    def _rebalance_safe_vertical_limit(self, ordered_blocks, current_index, rect, page_bounds):
        safe_bottom = float(page_bounds.y1)
        for idx in range(current_index + 1, len(ordered_blocks)):
            other = ordered_blocks[idx]
            other_rect = self._fitz_rect_from_bbox_like(self._rebalance_effective_bbox(other))
            if not isinstance(other_rect, fitz.Rect) or other_rect.get_area() <= 0:
                continue
            horizontal_overlap = self._bbox_horizontal_overlap_ratio(rect, other_rect)
            if horizontal_overlap < 0.20:
                continue
            if other_rect.y0 >= rect.y1:
                if self._rebalance_block_can_move(other):
                    continue
                safe_bottom = min(safe_bottom, float(other_rect.y0) - 2.0)
                break
        return max(float(rect.y1), safe_bottom)

    def _rebalance_block_can_move(self, block):
        if not isinstance(block, dict):
            return False
        contract_key = self._reconstruction_contract_key_for_block(block)
        if contract_key in {
            "table_cell",
            "table_cell_micro",
            "table_cell_symbolic",
            "table_cell_numeric",
            "code_block",
            "formula_block",
            "inline_formula",
            "figure_region",
            "url_reference",
            "toc_entry",
        }:
            return False
        block_type = str(self._classify_block_for_reconstruction(block, {}) or "").strip().lower()
        return block_type in {"editorial", "heading", "caption", "mixed"}

    def _set_rebalanced_bbox_pt(self, block, rect):
        bbox = (
            float(rect.x0 / self.pixel_to_point),
            float(rect.y0 / self.pixel_to_point),
            float(rect.x1 / self.pixel_to_point),
            float(rect.y1 / self.pixel_to_point),
        )
        block["rebalanced_bbox"] = bbox
        layout_attrs = block.get("layout_attributes")
        if isinstance(layout_attrs, dict):
            layout_attrs["rebalanced_bbox"] = bbox
        else:
            block["layout_attributes"] = {"rebalanced_bbox": bbox}
        return bbox

    def _rebalance_resolve_vertical_collisions(self, ordered_blocks, page_bounds):
        pushed = []
        sorted_blocks = sorted(
            [block for block in ordered_blocks if isinstance(block, dict)],
            key=lambda block: (
                (self._fitz_rect_from_bbox_like(self._rebalance_effective_bbox(block)) or fitz.Rect()).y0,
                (self._fitz_rect_from_bbox_like(self._rebalance_effective_bbox(block)) or fitz.Rect()).x0,
                str(block.get("id") or ""),
            ),
        )
        for idx, block in enumerate(sorted_blocks):
            rect = self._fitz_rect_from_bbox_like(self._rebalance_effective_bbox(block))
            if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                continue
            for other in sorted_blocks[idx + 1:]:
                other_rect = self._fitz_rect_from_bbox_like(self._rebalance_effective_bbox(other))
                if not isinstance(other_rect, fitz.Rect) or other_rect.get_area() <= 0:
                    continue
                if self._bbox_horizontal_overlap_ratio(rect, other_rect) < 0.20:
                    continue
                required_y0 = rect.y1 + 2.0
                if other_rect.y0 >= required_y0:
                    break
                if not self._rebalance_block_can_move(other):
                    continue
                delta = required_y0 - other_rect.y0
                if other_rect.y1 + delta > page_bounds.y1:
                    delta = max(0.0, page_bounds.y1 - other_rect.y1)
                if delta <= 0.1:
                    continue
                moved = fitz.Rect(other_rect.x0, other_rect.y0 + delta, other_rect.x1, other_rect.y1 + delta)
                self._set_rebalanced_bbox_pt(other, moved)
                pushed.append({"block_id": str(other.get("id") or ""), "delta_y_pt": float(delta)})
        return pushed

    def _rebalance_effective_bbox(self, block):
        if not isinstance(block, dict):
            return None
        bbox = block.get("rebalanced_bbox") or block.get("bbox")
        return bbox

    def _rebalance_page_layout(self, page, page_data, target_lang):
        if not isinstance(page_data, dict):
            return page_data
        blocks = [block for block in (page_data.get("blocks") or []) if isinstance(block, dict)]
        if not blocks:
            return page_data
        rebased = copy.deepcopy(page_data)
        rebased_blocks = [block for block in (rebased.get("blocks") or []) if isinstance(block, dict)]
        if not rebased_blocks:
            return rebased
        page_bounds = self._page_rebalance_bounds(page, rebased)
        ordered_blocks = self._iter_renderable_blocks(rebased)
        applied = []
        for idx, block in enumerate(ordered_blocks):
            if not self._rebalance_block_is_eligible(block, rebased):
                continue
            current = self._fitz_rect_from_bbox_like(block.get("bbox"))
            if not isinstance(current, fitz.Rect) or current.get_area() <= 0:
                continue
            desired_w, desired_h = self._rebalance_candidate_size(block, current, rebased)
            safe_left, safe_right = self._rebalance_safe_horizontal_span(ordered_blocks, idx, current, page_bounds)
            safe_bottom = self._rebalance_safe_vertical_limit(ordered_blocks, idx, current, page_bounds)
            new_x0 = max(float(page_bounds.x0), current.x0)
            new_x1 = min(safe_right, max(current.x1, new_x0 + desired_w))
            if new_x1 <= new_x0 + 4.0:
                new_x0 = max(float(page_bounds.x0), min(current.x0, safe_right - 8.0))
                new_x1 = min(safe_right, max(new_x0 + 8.0, current.x1))
            new_y0 = current.y0
            new_y1 = min(safe_bottom, max(current.y1, new_y0 + desired_h))
            if new_x1 <= new_x0 or new_y1 <= new_y0:
                continue
            if abs(new_x1 - current.x1) < 1.0 and abs(new_y1 - current.y1) < 1.0:
                continue
            bbox_pt = fitz.Rect(float(new_x0), float(new_y0), float(new_x1), float(new_y1))
            bbox = self._set_rebalanced_bbox_pt(block, bbox_pt)
            applied.append(
                {
                    "block_id": str(block.get("id") or ""),
                    "original_bbox": tuple(float(v) for v in (block.get("bbox") or current))[:4],
                    "rebalanced_bbox": bbox,
                    "rebalanced_bbox_pt": (float(bbox_pt.x0), float(bbox_pt.y0), float(bbox_pt.x1), float(bbox_pt.y1)),
                    "block_type": self._classify_block_for_reconstruction(block, rebased),
                }
            )
        pushed_blocks = self._rebalance_resolve_vertical_collisions(ordered_blocks, page_bounds)
        rebased["_page_rebalanced"] = {
            "applied": bool(applied),
            "applied_blocks": applied,
            "pushed_blocks": pushed_blocks,
            "block_count": len(ordered_blocks),
            "target_lang": target_lang,
        }
        return rebased

    def _reconstruction_contract_payload(self, page_data):
        if not isinstance(page_data, dict):
            return {}
        layout_descriptor_v3 = page_data.get("layout_descriptor_v3")
        if isinstance(layout_descriptor_v3, dict):
            contract = layout_descriptor_v3.get("reconstruction_contract")
            if isinstance(contract, dict):
                return contract
        contract = page_data.get("descriptor_v3_contract")
        return contract if isinstance(contract, dict) else {}

    def _contract_block_id_for_source_id(self, source_id):
        source_id = str(source_id or "").strip()
        if not source_id:
            return ""
        for marker in ("::line::", "::phrase::", "::span::"):
            if marker in source_id:
                return source_id.split(marker, 1)[0]
        return source_id

    def _line_index_for_source_id(self, source_id):
        match = re.match(r"^(.+)::line::(\d+)$", str(source_id or "").strip())
        if not match:
            return None
        try:
            return int(match.group(2))
        except Exception:
            return None

    def _line_source_id(self, block_id, line_index):
        return f"{str(block_id or '').strip()}::line::{int(line_index)}"

    def _build_reconstruction_contract_context(self, page_data):
        contract = self._reconstruction_contract_payload(page_data)
        if not isinstance(contract, dict):
            return {}
        render_units = [unit for unit in (contract.get("render_units") or []) if isinstance(unit, dict)]
        placement_constraints = [item for item in (contract.get("placement_constraints") or []) if isinstance(item, dict)]
        execution_edges = [edge for edge in (contract.get("execution_edges") or []) if isinstance(edge, dict)]
        containers = [container for container in (contract.get("containers") or []) if isinstance(container, dict)]
        if not any((render_units, placement_constraints, execution_edges, containers)):
            return {"contract": contract, "primary_structure_family": str(contract.get("primary_structure_family") or "").strip().lower()}

        by_block = {}

        def ensure_block_entry(block_id):
            block_id = str(block_id or "").strip()
            if not block_id:
                return None
            entry = by_block.get(block_id)
            if entry is None:
                entry = {
                    "block_id": block_id,
                    "block_render_unit": {},
                    "block_constraint": {},
                    "line_render_units": {},
                    "line_constraints": {},
                    "containers": [],
                    "execution_edges": [],
                    "primary_structure_family": str(contract.get("primary_structure_family") or "").strip().lower(),
                }
                by_block[block_id] = entry
            return entry

        for render_unit in render_units:
            source_id = str(render_unit.get("source_element_id") or "").strip()
            block_id = self._contract_block_id_for_source_id(source_id)
            entry = ensure_block_entry(block_id)
            if entry is None:
                continue
            line_index = self._line_index_for_source_id(source_id)
            if line_index is None:
                entry["block_render_unit"] = dict(render_unit)
            else:
                entry["line_render_units"][line_index] = dict(render_unit)

        for constraint in placement_constraints:
            source_id = str(constraint.get("source_element_id") or "").strip()
            block_id = self._contract_block_id_for_source_id(source_id)
            entry = ensure_block_entry(block_id)
            if entry is None:
                continue
            line_index = self._line_index_for_source_id(source_id)
            if line_index is None:
                entry["block_constraint"] = dict(constraint)
            else:
                entry["line_constraints"][line_index] = dict(constraint)

        for edge in execution_edges:
            source_id = str(edge.get("source") or "").strip()
            target_id = str(edge.get("target") or "").strip()
            block_ids = {
                self._contract_block_id_for_source_id(source_id),
                self._contract_block_id_for_source_id(target_id),
            }
            for block_id in block_ids:
                entry = ensure_block_entry(block_id)
                if entry is not None:
                    entry["execution_edges"].append(dict(edge))

        for container in containers:
            member_ids = {str(member_id or "").strip() for member_id in (container.get("member_ids") or []) if str(member_id or "").strip()}
            if not member_ids:
                continue
            for block_id, entry in by_block.items():
                if block_id in member_ids or any(self._line_source_id(block_id, idx) in member_ids for idx in entry["line_constraints"].keys() | entry["line_render_units"].keys()):
                    entry["containers"].append(dict(container))

        return {
            "contract": contract,
            "primary_structure_family": str(contract.get("primary_structure_family") or "").strip().lower(),
            "by_block": by_block,
        }

    def _reconstruction_contract_context(self, page_data):
        if not isinstance(page_data, dict):
            return {}
        cached = (page_data.get("_reconstruction_context") or {}).get("reconstruction_contract")
        if isinstance(cached, dict) and cached:
            return cached
        context = self._build_reconstruction_contract_context(page_data)
        page_ctx = dict(page_data.get("_reconstruction_context") or {})
        page_ctx["reconstruction_contract"] = context
        page_data["_reconstruction_context"] = page_ctx
        self._attach_descriptor_v3_metadata(page_data, context)
        return context

    def _contract_entry_for_block(self, page_data, block):
        contract_context = self._reconstruction_contract_context(page_data)
        if not isinstance(contract_context, dict):
            return {}
        block_id = str((block or {}).get("id") or "").strip()
        if not block_id:
            return {}
        return dict((contract_context.get("by_block") or {}).get(block_id) or {})

    def _descriptor_table_region_info(self, block):
        descriptor_group_ids = dict((block or {}).get("descriptor_group_ids") or {})
        descriptor_page_organization = dict((block or {}).get("descriptor_page_organization") or {})
        row_id = str(descriptor_group_ids.get("table_row_group_id") or "").strip()
        cell_id = str(descriptor_group_ids.get("cell_id") or "").strip()
        column_id = str(descriptor_group_ids.get("table_column_group_id") or "").strip()
        row_bbox = None
        cell_bbox = None
        if row_id:
            for row in descriptor_page_organization.get("table_row_groups") or []:
                if str((row or {}).get("id") or "") != row_id:
                    continue
                row_rect = self._fitz_rect_from_bbox_like((row or {}).get("bbox"))
                if isinstance(row_rect, fitz.Rect) and row_rect.get_area() > 0:
                    row_bbox = (row_rect.x0, row_rect.y0, row_rect.x1, row_rect.y1)
                if cell_id:
                    for cell in (row or {}).get("cells") or []:
                        if str((cell or {}).get("id") or "") == cell_id or str((cell or {}).get("block_id") or "") == str((block or {}).get("id") or ""):
                            cell_rect = self._fitz_rect_from_bbox_like((cell or {}).get("bbox"))
                            if isinstance(cell_rect, fitz.Rect) and cell_rect.get_area() > 0:
                                cell_bbox = (cell_rect.x0, cell_rect.y0, cell_rect.x1, cell_rect.y1)
                            break
                break
        return {
            "row_id": row_id,
            "cell_id": cell_id,
            "column_id": column_id,
            "row_bbox": row_bbox,
            "cell_bbox": cell_bbox,
        }

    def _select_primary_contract_container(self, block, contract_entry):
        containers = [dict(container) for container in (contract_entry or {}).get("containers") or [] if isinstance(container, dict)]
        if not containers:
            return {}
        block_rect = self._fitz_rect_from_bbox_like((block or {}).get("bbox"))
        best = None
        best_key = None
        for container in containers:
            rect = self._fitz_rect_from_bbox_like(container.get("bbox"))
            overlap = 0.0
            distance = 0.0
            if isinstance(block_rect, fitz.Rect) and block_rect.get_area() > 0 and isinstance(rect, fitz.Rect) and rect.get_area() > 0:
                overlap = (block_rect & rect).get_area()
                distance = abs(rect.get_area() - block_rect.get_area())
            priority = str(container.get("structure_priority") or "").strip().lower()
            key = (
                0 if bool(container.get("active")) else 1,
                0 if priority == "primary" else (1 if priority == "secondary" else 2),
                -overlap,
                distance,
                str(container.get("kind") or ""),
                str(container.get("id") or ""),
            )
            if best_key is None or key < best_key:
                best_key = key
                best = container
        return best or {}

    def _rect_tuple_or_none(self, bbox_like):
        rect = self._fitz_rect_from_bbox_like(bbox_like)
        if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
            return None
        return (rect.x0, rect.y0, rect.x1, rect.y1)

    def _structured_contract_plan_for_block(self, block, page_data, units, contract_entry, block_type, render_mode):
        if not contract_entry or not render_mode:
            return {}
        units = [unit for unit in (units or []) if isinstance(unit, PlacableUnit)]
        primary_container = self._select_primary_contract_container(block, contract_entry)
        table_region = self._descriptor_table_region_info(block)
        block_bbox = self._rect_tuple_or_none((block or {}).get("bbox"))
        container_bbox = self._rect_tuple_or_none(primary_container.get("bbox"))
        line_rotations = []
        for line in (block or {}).get("lines") or []:
            if not isinstance(line, dict):
                continue
            rotation = self._rotation_deg_for_item(
                line,
                bbox_like=(line or {}).get("bbox"),
                text=(line or {}).get("translated_text") or (line or {}).get("line_text") or "",
                fallback=0,
            )
            if rotation in {90, 180, 270}:
                line_rotations.append(rotation)
        unit_rotations = [
            int(round(float((unit.metadata or {}).get("rotation_deg") or 0.0))) % 360
            for unit in units
            if int(round(float((unit.metadata or {}).get("rotation_deg") or 0.0))) % 360 in {90, 180, 270}
        ]
        dominant_rotation = 0
        if unit_rotations or line_rotations:
            dominant_rotation = Counter(unit_rotations + line_rotations).most_common(1)[0][0]

        container_kind = str(primary_container.get("kind") or "").strip().lower()
        structure_family = str((contract_entry or {}).get("primary_structure_family") or "").strip().lower()
        background_region_bbox = block_bbox
        background_region_id = ""
        draw_bbox = table_region.get("cell_bbox") or block_bbox or container_bbox
        kind = "anchored_cluster"

        if block_type == "table" or render_mode == "cell_locked" or table_region.get("cell_id") or table_region.get("row_id"):
            kind = "rotated_grid" if dominant_rotation in {90, 270} else "grid"
            background_region_bbox = table_region.get("row_bbox") or container_bbox or block_bbox
            if table_region.get("row_id"):
                background_region_id = f"table_row::{table_region['row_id']}"
            elif table_region.get("cell_id"):
                background_region_id = f"table_cell::{table_region['cell_id']}"
            else:
                background_region_id = f"table_block::{str((block or {}).get('id') or '').strip()}"
        elif render_mode == "prose_reflow":
            # prose_reflow prend la priorité sur la rotation : on reflow le texte en flux continu
            # même si quelques lignes source sont en rotation (ex. numéro de page/chapitre rotatif).
            kind = "styled_paragraph"
            background_region_bbox = block_bbox
        elif dominant_rotation in {90, 270}:
            kind = "rotated_grid"
            background_region_bbox = container_bbox or block_bbox
            background_region_id = f"rotated::{str((block or {}).get('id') or '').strip()}"
        elif container_kind in {"chapter_opening", "key_value_pair", "toc_entry", "section"}:
            kind = "anchored_composite"
            background_region_bbox = container_bbox or block_bbox
            background_region_id = f"container::{str(primary_container.get('id') or (block or {}).get('id') or '').strip()}"
        elif render_mode in {"line_preserve", "bbox_anchored"}:
            kind = "anchored_composite" if (container_bbox and container_bbox != block_bbox and len(units) >= 2) else "line_locked_cluster"
            background_region_bbox = container_bbox or block_bbox
            if kind == "anchored_composite":
                background_region_id = f"container::{str(primary_container.get('id') or (block or {}).get('id') or '').strip()}"

        if structure_family in {"visual_labels", "key_value_pairs"} and kind == "anchored_cluster":
            kind = "anchored_composite"
            background_region_bbox = container_bbox or block_bbox
            background_region_id = background_region_id or f"container::{str(primary_container.get('id') or (block or {}).get('id') or '').strip()}"

        return {
            "enabled": bool(kind),
            "kind": kind,
            "render_mode": render_mode,
            "rotation_deg": dominant_rotation,
            "container_kind": container_kind,
            "container_id": str(primary_container.get("id") or "").strip(),
            "container_bbox": container_bbox,
            "draw_bbox": draw_bbox,
            "background_region_bbox": background_region_bbox,
            "background_region_id": background_region_id,
            "structure_family": structure_family,
            "unit_count": len(units),
            "table_region": table_region,
        }

    def _attach_descriptor_v3_metadata(self, page_data, contract_context):
        if not isinstance(page_data, dict) or not isinstance(contract_context, dict):
            return
        by_block = contract_context.get("by_block") or {}
        contract = contract_context.get("contract") or {}
        for block in page_data.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("id") or "").strip()
            entry = dict(by_block.get(block_id) or {})
            if contract:
                block["descriptor_v3_contract"] = dict(contract)
            if entry.get("block_render_unit"):
                block["descriptor_v3_render_unit"] = dict(entry.get("block_render_unit") or {})
            if entry.get("block_constraint"):
                block["descriptor_v3_placement_constraint"] = dict(entry.get("block_constraint") or {})
            for line_index, line in enumerate(block.get("lines") or []):
                if not isinstance(line, dict):
                    continue
                render_unit = dict((entry.get("line_render_units") or {}).get(line_index) or {})
                constraint = dict((entry.get("line_constraints") or {}).get(line_index) or {})
                if render_unit:
                    line["descriptor_v3_render_unit"] = render_unit
                if constraint:
                    line["descriptor_v3_placement_constraint"] = constraint

    def _page_background_audit(self, page_data):
        path = str((page_data or {}).get("background_path") or "").strip()
        if not path or not os.path.exists(path):
            return {"usable": False, "path": None, "reason": "missing_background"}

        audit = dict((page_data or {}).get("p6_bg_audit") or {})
        if audit:
            quality = float(audit.get("quality") or 0.0)
            artifacts = len(audit.get("artifacts") or [])
            if bool(audit.get("reprocess")):
                return {"usable": False, "path": path, "reason": "p6_reprocess", "quality": quality, "artifacts": artifacts}
            if audit.get("ok") is False:
                return {"usable": False, "path": path, "reason": "p6_not_ok", "quality": quality, "artifacts": artifacts}
            if quality and quality < 0.72:
                return {"usable": False, "path": path, "reason": "p6_low_quality", "quality": quality, "artifacts": artifacts}
            if artifacts >= 5:
                return {"usable": False, "path": path, "reason": "p6_artifact_cluster", "quality": quality, "artifacts": artifacts}

        debug = dict((page_data or {}).get("text_removal_debug") or {})
        mask_nonzero = int(debug.get("mask_nonzero") or 0)
        dims = dict((page_data or {}).get("dimensions") or {})
        width_px = float(dims.get("width") or dims.get("page_width") or 0.0)
        height_px = float(dims.get("height") or dims.get("page_height") or 0.0)
        if width_px <= 0.0 or height_px <= 0.0:
            fallback_bg = self._page_background_path(page_data)
            if fallback_bg:
                try:
                    pix = fitz.Pixmap(fallback_bg)
                    width_px = width_px or float(pix.width)
                    height_px = height_px or float(pix.height)
                except Exception:
                    pass
        page_area = max(1.0, width_px * height_px)
        if mask_nonzero > 0 and page_area > 1.0:
            mask_ratio = mask_nonzero / page_area
            if mask_ratio > 0.22 and not audit:
                return {"usable": False, "path": path, "reason": "mask_ratio_high", "mask_ratio": mask_ratio}
        return {"usable": True, "path": path, "reason": "clean_background"}

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
        contract_entry = self._contract_entry_for_block(page_data, block)
        block_constraint = dict(contract_entry.get("block_constraint") or {})
        bbox_lock = block_constraint.get("bbox_lock")
        effective_bbox = bbox_lock or (block or {}).get("rebalanced_bbox") or (block or {}).get("bbox")
        block_rect = self._fitz_rect_from_bbox_like(effective_bbox)
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
        background_audit = self._page_background_audit(page_data)
        has_clean_background = bool(background_audit.get("usable"))
        background_mode = self._contract_background_mode(block)
        if (
            self._is_translated_block(block)
            and not has_clean_background
            and (
                bool(adaptive_profile.get("force_whiteout"))
                or background_mode == "plain_whiteout"
            )
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
                "page_orientation": self._page_orientation(page_data),
                "adaptive_profile": adaptive_profile,
                "background_audit": background_audit,
                "background_mode": background_mode,
                "contract_block_constraint": block_constraint,
                "contract_entry": contract_entry,
            },
        )

    def _build_line_templates(self, block, geometry_ctx, page_data=None):
        block_rect = fitz.Rect(geometry_ctx.block_bbox)
        adaptive_profile = dict((geometry_ctx.constraints or {}).get("adaptive_profile") or {})
        line_spacing_factor = max(0.86, min(1.08, float(adaptive_profile.get("line_spacing_factor") or 1.0)))
        lines = list((block or {}).get("lines") or [])
        contract_entry = self._contract_entry_for_block(page_data, block)
        line_constraints = dict(contract_entry.get("line_constraints") or {})
        templates = []
        paragraph_index = 0
        line_index_in_paragraph = 0
        block_alignment = self._contract_alignment_value(block, key="paragraph", fallback=(block or {}).get("alignment") or "left")
        preserve_horizontal_slot = self._contract_preserves_horizontal_slot(block)
        inner_left = block_rect.x0 + geometry_ctx.padding_left
        inner_right = max(inner_left + 8.0, block_rect.x1 - geometry_ctx.padding_right)
        inner_top = block_rect.y0 + geometry_ctx.padding_top
        inner_bottom = max(inner_top + 8.0, block_rect.y1 - geometry_ctx.padding_bottom)
        line_heights = []
        previous_bottom = inner_top
        for idx, line in enumerate(lines):
            line_constraint = dict(line_constraints.get(idx) or {})
            line_bbox_lock = line_constraint.get("bbox_lock")
            line_rect = self._fitz_rect_from_bbox_like(line_bbox_lock) if line_bbox_lock else self._fitz_rect_from_bbox_like((line or {}).get("bbox"))
            if not isinstance(line_rect, fitz.Rect) or line_rect.get_area() <= 0:
                line_rect = fitz.Rect(block_rect.x0, block_rect.y0 + idx * 12.0, block_rect.x1, block_rect.y0 + (idx + 1) * 12.0)
            line_style_invariants = dict(line_constraint.get("style_invariants") or {})
            template_alignment = self._normalize_alignment(
                line_style_invariants.get("align")
                or self._contract_alignment_value(line, key="inline", fallback=block_alignment)
            )
            rotation_deg = self._rotation_deg_for_item(
                line,
                bbox_like=line_bbox_lock or (line or {}).get("bbox"),
                text=(line or {}).get("text") or (line or {}).get("line_text") or "",
                fallback=0,
            )
            linebreak_policy = dict(line_constraint.get("linebreak_policy") or {})
            anchor_policy = dict(line_constraint.get("anchor_policy") or {})
            line_slot_locked = (
                preserve_horizontal_slot
                or self._contract_preserves_horizontal_slot(line)
                or bool(line_bbox_lock)
                or bool(anchor_policy.get("source_y_locked"))
                or str(linebreak_policy.get("mode") or "").strip().lower() == "preserve_source_lines"
            )
            line_left = inner_left
            line_right = inner_right
            if line_slot_locked:
                line_left = max(inner_left, min(inner_right - 8.0, line_rect.x0))
                line_right = min(inner_right, max(line_left + 8.0, line_rect.x1))
            line_rect = fitz.Rect(line_left, max(inner_top, line_rect.y0), line_right, min(inner_bottom, line_rect.y1))
            if line_rect.height <= 0:
                fallback_top = min(inner_bottom - 6.0, inner_top + idx * 12.0)
                line_rect = fitz.Rect(line_left, fallback_top, line_right, min(inner_bottom, fallback_top + 12.0))
            hard_break_before = bool((line or {}).get("hard_break_before"))
            if idx > 0 and hard_break_before:
                paragraph_index += 1
                line_index_in_paragraph = 0
            indent_px = float((line or {}).get("indent_px", 0.0) or 0.0) * self.pixel_to_point
            # Pour les lignes slot-locked, left_x est déjà à line_rect.x0 ; indent_px encode
            # l'offset x depuis le bord gauche du bloc, pas une indentation de texte.
            template_indent_px = 0.0 if line_slot_locked else indent_px
            line_h = max(6.0, line_rect.height * line_spacing_factor)
            # Lignes côte à côte (même bande verticale, x différents) : ne pas les empiler
            _is_side_by_side = idx > 0 and line_rect.y0 < previous_bottom and line_rect.y1 <= previous_bottom + 2.0
            top = max(line_rect.y0, inner_top if (idx == 0 or _is_side_by_side) else previous_bottom)
            bottom = min(inner_bottom, max(top + line_h, line_rect.y1))
            if bottom - top < 4.0:
                bottom = min(inner_bottom, top + max(4.0, line_h))
            line_rect = fitz.Rect(line_left, top, line_right, bottom)
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
                    usable_width=max(8.0, line_rect.width - template_indent_px),
                    indent_px=template_indent_px,
                    first_line_indent_px=template_indent_px if line_index_in_paragraph == 0 else 0.0,
                    alignment=template_alignment,
                    paragraph_id=f"{block.get('id') or 'block'}:paragraph:{paragraph_index}",
                    paragraph_index=paragraph_index,
                    line_index_in_paragraph=line_index_in_paragraph,
                    is_first_paragraph_line=(line_index_in_paragraph == 0),
                    is_last_paragraph_line_hint=bool((line or {}).get("line_break_after")),
                    rotation_deg=rotation_deg,
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
                    alignment=block_alignment,
                    paragraph_id=f"{block.get('id') or 'block'}:paragraph:0",
                    paragraph_index=0,
                    line_index_in_paragraph=0,
                    is_first_paragraph_line=True,
                    is_last_paragraph_line_hint=True,
                    rotation_deg=0,
                )
            )
        return templates

    def _infer_paragraph_alignment_from_lines(self, block, fallback="left"):
        fallback = self._normalize_alignment(fallback)
        if fallback != "left":
            return fallback
        object_type = str((block or {}).get("object_type") or "").strip().lower()
        role = str((block or {}).get("role") or "").strip().lower()
        if object_type not in {"paragraph", "plain_text"} and role not in {"body", "paragraph"}:
            return fallback
        line_rects = [
            self._fitz_rect_from_bbox_like((line or {}).get("bbox"))
            for line in ((block or {}).get("lines") or [])
        ]
        line_rects = [rect for rect in line_rects if isinstance(rect, fitz.Rect) and rect.get_area() > 0]
        if len(line_rects) < 3:
            return fallback
        block_rect = self._fitz_rect_from_bbox_like((block or {}).get("bbox"))
        if not isinstance(block_rect, fitz.Rect) or block_rect.width <= 0:
            return fallback
        content_width = max(1.0, max(rect.x1 for rect in line_rects) - min(rect.x0 for rect in line_rects))
        body_lines = line_rects[:-1] if len(line_rects) > 3 else line_rects
        fill_ratios = [max(0.0, rect.width / content_width) for rect in body_lines]
        right_edges = [rect.x1 for rect in body_lines if rect.width / content_width >= 0.68]
        if not right_edges:
            return fallback
        right_spread = max(right_edges) - min(right_edges)
        if sum(1 for ratio in fill_ratios if ratio >= 0.82) >= max(2, int(len(body_lines) * 0.65)) and right_spread <= max(5.0, content_width * 0.035):
            return "justify"
        return fallback

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

    def _constraint_alignment(self, constraint, fallback="left"):
        if not isinstance(constraint, dict):
            return self._normalize_alignment(fallback)
        style_invariants = dict(constraint.get("style_invariants") or {})
        anchor_policy = dict(constraint.get("anchor_policy") or {})
        alignment = str(style_invariants.get("align") or "").strip().lower()
        if alignment in {"left", "center", "right", "justify"}:
            return alignment
        if bool(anchor_policy.get("force_end_anchor")):
            return "right"
        if bool(anchor_policy.get("force_start_anchor")):
            return "left"
        return self._normalize_alignment(fallback)

    def _contract_render_mode_for_block(self, block, page_data=None, block_type=None, contract_entry=None):
        entry = contract_entry if isinstance(contract_entry, dict) else self._contract_entry_for_block(page_data, block)
        block_type = str(block_type or self._classify_block_for_reconstruction(block, page_data)).strip().lower()
        target_contract = self._reconstruction_contract_for_block(block, page_data=page_data)
        target_render_mode = str(target_contract.get("render_mode") or "").strip().lower()
        if bool(target_contract.get("strict_non_reflow")) and target_render_mode:
            return target_render_mode
        block_constraint = dict(entry.get("block_constraint") or {})
        block_render_unit = dict(entry.get("block_render_unit") or {})
        line_constraints = dict(entry.get("line_constraints") or {})
        reflow_policy = str(block_constraint.get("reflow_policy") or block_render_unit.get("reflow_policy") or "").strip().lower()
        linebreak_policy = dict(block_constraint.get("linebreak_policy") or {})
        linebreak_mode = str(linebreak_policy.get("mode") or "").strip().lower()
        anchor_policy = dict(block_constraint.get("anchor_policy") or {})
        anchor_mode = str(anchor_policy.get("mode") or "").strip().lower()
        if block_type == "table":
            return "cell_locked"
        block_contract = self._render_contract_for_item(block)
        layout_mode = dict(block_contract.get("layout_mode") or {})
        alignment = dict(block_contract.get("alignment") or {})
        source_layout_mode = dict((block or {}).get("source_layout_mode") or {})
        family = str(block_contract.get("family") or "").strip().lower()
        source_render_contract = str(
            source_layout_mode.get("render_contract")
            or layout_mode.get("source_render_contract")
            or ""
        ).strip().lower()
        source_line_flow = str(
            source_layout_mode.get("line_flow")
            or layout_mode.get("source_line_flow")
            or ""
        ).strip().lower()
        preserve_line_breaks = bool(
            source_layout_mode.get("preserve_line_breaks", layout_mode.get("preserve_line_breaks", False))
        )
        can_reflow = bool(
            source_layout_mode.get("can_reflow_within_paragraph", layout_mode.get("can_reflow_within_paragraph", False))
        )
        line_count = int(source_layout_mode.get("line_count") or len((block or {}).get("lines") or []) or 0)
        locked_position = bool(alignment.get("locked_position"))
        has_rotated_lines = any(
            self._rotation_deg_for_item(line, bbox_like=(line or {}).get("bbox"), text=(line or {}).get("translated_text") or (line or {}).get("line_text") or "", fallback=0) in {90, 180, 270}
            for line in ((block or {}).get("lines") or [])
            if isinstance(line, dict)
        )
        if family == "table_cell":
            return "cell_locked"
        if source_render_contract in {"fixed_slots", "preserve_breaks"} or source_line_flow in {"fixed_lines", "preserve_line_breaks", "single_line"}:
            if line_count > 1 or preserve_line_breaks or has_rotated_lines or locked_position:
                return "line_preserve"
            return "bbox_anchored" if locked_position else "line_preserve"
        if entry:
            if line_constraints and linebreak_mode == "preserve_source_lines":
                return "line_preserve"
            if reflow_policy in {"toc_row_locked", "pair_locked"}:
                return "line_preserve"
            if reflow_policy == "anchored_locked" or anchor_mode == "attachment_locked":
                return "bbox_anchored"
            if line_constraints and anchor_policy.get("source_y_locked"):
                return "line_preserve"
            if reflow_policy == "paragraph_reflow":
                return "prose_reflow"
        if target_render_mode and not entry:
            return target_render_mode
        if source_render_contract in {"reflow_block", "paragraph_reflow"} or source_line_flow in {"inline_reflow", "preserve_paragraphs", "rewrap"}:
            if can_reflow:
                return "prose_reflow"
            return "line_preserve" if line_count > 1 else "bbox_anchored"
        if family == "anchored" and locked_position:
            return "line_preserve" if line_count > 1 else "bbox_anchored"
        return ""

    def _contract_render_contract(self, block_type, constraint, block, fallback_policy="translated_editorial"):
        constraint = dict(constraint or {})
        alignment = self._constraint_alignment(constraint, fallback=(block or {}).get("alignment") or "left")
        linebreak_policy = dict(constraint.get("linebreak_policy") or {})
        anchor_policy = dict(constraint.get("anchor_policy") or {})
        reflow_policy = str(constraint.get("reflow_policy") or "").strip().lower()
        if block_type == "table":
            family = "table_cell"
            policy = "cell_locked"
            background_mode = "local_bg_restore"
            source_render_contract = "fixed_slots"
            source_line_flow = "fixed_lines"
            locked_position = True
        elif reflow_policy in {"toc_row_locked", "pair_locked", "anchored_locked"} or str(anchor_policy.get("mode") or "").strip().lower() == "attachment_locked":
            family = "anchored"
            policy = "anchored_text"
            background_mode = self._contract_background_mode(block) or "plain_whiteout"
            source_render_contract = "fixed_slots"
            source_line_flow = "fixed_lines"
            locked_position = True
        elif str(linebreak_policy.get("mode") or "").strip().lower() == "preserve_source_lines":
            family = "fixed_lines"
            policy = "anchored_text"
            background_mode = self._contract_background_mode(block) or "plain_whiteout"
            source_render_contract = "preserve_breaks"
            source_line_flow = "fixed_lines"
            locked_position = True
        else:
            family = "paragraph"
            policy = fallback_policy
            background_mode = self._contract_background_mode(block) or "plain_whiteout"
            source_render_contract = "paragraph_reflow"
            source_line_flow = "rewrap"
            locked_position = False
        return {
            "schema_version": "render_contract.v1",
            "family": family,
            "canonical_render_policy": policy,
            "alignment": {"paragraph": alignment, "inline": alignment, "locked_position": locked_position},
            "background": {"mode": background_mode},
            "layout_mode": {
                "source_render_contract": source_render_contract,
                "source_line_flow": source_line_flow,
            },
        }

    def _apply_contract_constraints_to_units(self, block, page_data, units, target_lang, contract_entry, render_mode, block_type):
        entry = dict(contract_entry or {})
        block_constraint = dict(entry.get("block_constraint") or {})
        line_constraints = dict(entry.get("line_constraints") or {})
        if not units and block_constraint:
            text = self._translated_text_from_block(block) or self._source_text_from_block(block)
            text = self._clean_text_for_render(text)
            if text:
                style = self._style_from_block(block)
                bbox = self._fitz_rect_from_bbox_like(block_constraint.get("bbox_lock") or (block or {}).get("bbox"))
                units = [
                    PlacableUnit(
                        unit_id=f"{str((block or {}).get('id') or 'block')}::contract::0",
                        unit_type="contract_block",
                        source_kind=str((block or {}).get("source") or "native"),
                        parent_unit_id=None,
                        block_unit_id=str((block or {}).get("id") or ""),
                        phrase_unit_id=str((block or {}).get("id") or ""),
                        line_indices=[0],
                        text_source=self._source_text_from_block(block) or text,
                        text_translated=self._translated_text_from_block(block) or text,
                        role=str((block or {}).get("role") or "body"),
                        style=style,
                        relative_bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1) if isinstance(bbox, fitz.Rect) and bbox.get_area() > 0 else self._unit_fitz_bbox(block) or None,
                    )
                ]
        if not units:
            return units

        updated_units = []
        for order, unit in enumerate(units):
            line_index = int(unit.line_indices[0]) if unit.line_indices else order
            constraint = dict(line_constraints.get(line_index) or block_constraint or {})
            bbox_lock = constraint.get("bbox_lock")
            rect = self._fitz_rect_from_bbox_like(bbox_lock) if bbox_lock else self._fitz_rect_from_bbox_like(unit.relative_bbox)
            alignment = self._constraint_alignment(constraint, fallback=(block or {}).get("alignment") or "left")
            style = self._merge_styles(unit.style or {}, self._style_from_block(block))
            style["align"] = alignment
            render_contract = self._contract_render_contract(block_type, constraint, block, fallback_policy=str(unit.render_policy or "translated_editorial"))
            anchor_policy = dict(constraint.get("anchor_policy") or {})
            preferred_horizontal = str(anchor_policy.get("preferred_horizontal_anchor") or "").strip().lower()
            preferred_vertical = str(anchor_policy.get("preferred_vertical_anchor") or "").strip().lower()
            anchor_horizontal = unit.anchor_horizontal or preferred_horizontal or ("center" if alignment == "center" else ("right" if alignment == "right" else "left"))
            anchor_vertical = unit.anchor_vertical or preferred_vertical or "top"
            reflowable = bool(unit.reflowable)
            render_policy = str(unit.render_policy or "translated_editorial")
            hard_break_before = bool(unit.hard_break_before)
            if render_mode in {"line_preserve", "bbox_anchored", "cell_locked"}:
                reflowable = False
                hard_break_before = bool(order > 0)
            if render_mode == "cell_locked":
                render_policy = "cell_locked"
            elif render_mode == "bbox_anchored":
                render_policy = "anchored_text"
            metadata = dict(unit.metadata or {})
            metadata["reconstruction_contract"] = self._reconstruction_contract_for_block(block, page_data=page_data)
            metadata["render_contract"] = render_contract
            metadata["descriptor_v3_contract"] = dict((self._reconstruction_contract_payload(page_data) or {}))
            metadata["descriptor_v3_placement_constraint"] = constraint
            metadata["contract_render_mode"] = render_mode
            metadata["rotation_deg"] = self._rotation_deg_for_item(
                unit.metadata.get("raw_unit") if isinstance(unit.metadata.get("raw_unit"), dict) else {},
                bbox_like=(rect.x0, rect.y0, rect.x1, rect.y1) if isinstance(rect, fitz.Rect) and rect.get_area() > 0 else unit.relative_bbox,
                text=unit.text_translated or unit.text_source,
                fallback=metadata.get("rotation_deg") or 0,
            )
            updated_units.append(
                replace(
                    unit,
                    unit_type="translated_line" if render_mode == "line_preserve" else unit.unit_type,
                    style=style,
                    relative_bbox=(rect.x0, rect.y0, rect.x1, rect.y1) if isinstance(rect, fitz.Rect) and rect.get_area() > 0 else unit.relative_bbox,
                    anchor_horizontal=anchor_horizontal,
                    anchor_vertical=anchor_vertical,
                    hard_break_before=hard_break_before,
                    reflowable=reflowable,
                    render_policy=render_policy,
                    metadata=metadata,
                )
            )
        return updated_units

    def _contract_driven_units_for_block(self, block, semantic_payload, target_lang, page_data=None):
        contract_entry = self._contract_entry_for_block(page_data, block)
        if not contract_entry:
            return None
        block_type = self._classify_block_for_reconstruction(block, page_data)
        render_mode = self._contract_render_mode_for_block(block, page_data=page_data, block_type=block_type, contract_entry=contract_entry)
        if not render_mode:
            return None
        if render_mode in {"line_preserve", "bbox_anchored"}:
            units = self._line_units(block, target_lang, page_data=page_data)
            if not units:
                units = self._fallback_units(block, semantic_payload, target_lang, page_data=page_data)
        elif block_type == "table":
            units = self._phrase_units(block, semantic_payload, target_lang, page_data=page_data)
            if not units:
                units = self._line_units(block, target_lang, page_data=page_data)
            if not units:
                units = self._fallback_units(block, semantic_payload, target_lang, page_data=page_data)
        else:
            units = self._phrase_units(block, semantic_payload, target_lang, page_data=page_data)
            if not units:
                units = self._line_units(block, target_lang, page_data=page_data)
            if not units:
                units = self._fallback_units(block, semantic_payload, target_lang, page_data=page_data)
        return self._apply_contract_constraints_to_units(
            block,
            page_data,
            units,
            target_lang,
            contract_entry,
            render_mode,
            block_type,
        )

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
        object_payload = dict((raw_unit or {}).get("object_comprehension") or {})
        object_type = str((raw_unit or {}).get("object_type") or object_payload.get("object_type") or "").strip().lower()
        object_class = str((raw_unit or {}).get("object_class") or object_payload.get("object_class") or "").strip().lower()
        object_subtype = str((raw_unit or {}).get("object_subtype") or object_payload.get("object_subtype") or "").strip().lower()
        inline_object_type = str((raw_unit or {}).get("inline_object_type") or object_payload.get("inline_object_type") or "").strip().lower()
        inline_object_subtype = str((raw_unit or {}).get("inline_object_subtype") or object_payload.get("inline_object_subtype") or "").strip().lower()
        translation_policy = dict((raw_unit or {}).get("translation_policy") or object_payload.get("translation_policy") or {})
        reconstruction_contract = self._reconstruction_contract_for_block(raw_unit, page_data=page_data)
        # "external_flow" est une politique verrouillée qui ne doit pas être écrasée
        _locked_policy = render_policy in {"external_flow"}
        policy_render = str(translation_policy.get("render_policy") or "").strip().lower()
        policy_translatable = translation_policy.get("translatable")
        if not _locked_policy and has_bbox and (
            anchored_role
            or preserve_center
            or adaptive_profile.get("prefer_bbox_anchor")
            or (not allow_horizontal_reflow and short_text and anchor_confidence >= 0.55)
            or (child_summary["has_protected"] and short_text)
            or child_summary["dominant_inline_class"] in {"formula", "reference", "code"}
            or object_type in {"short_label", "diagram_label", "chart_label", "axis_label", "legend_label", "figure_axis_label", "micro_label"}
            or object_subtype in {"micro_label", "formula_symbol", "formula_equation"}
            or object_class == "visual_label"
            or policy_render in {"anchored_text", "fixed_preserve", "cell_locked"}
            or bool(reconstruction_contract.get("strict_non_reflow"))
        ):
            force_bbox_anchor = True
            render_policy = "anchored_text"
            reflowable = False
        if not _locked_policy and child_summary["has_immutable"] and short_text:
            render_policy = "fixed_preserve"
            reflowable = False
            force_bbox_anchor = True
        unit_profile = str(adaptive_profile.get("unit_profile") or "")
        if not _locked_policy and (
            unit_profile in {"protected_inline", "anchored_label", "technical_inline_cluster"}
            or object_class in {"visual", "technical", "formula", "metadata"}
            or object_type in {"code_block", "code_line", "inline_code", "formula_block", "formula_line", "inline_formula", "inline_formula_cluster"}
            or inline_object_type in {"web_url", "email_address", "doi_reference", "arxiv_reference", "inline_formula", "chemical_formula", "function_call", "technical_identifier"}
            or inline_object_subtype in {"web_url", "email_address", "doi_reference", "arxiv_reference", "inline_formula", "chemical_formula", "function_call", "technical_identifier"}
            or policy_render in {"fixed_preserve", "source_overlay", "cell_locked"}
        ):
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
        if unit_profile in {"protected_inline", "anchored_label", "technical_inline_cluster"} or object_class in {"visual", "technical", "formula"}:
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
            "object_type": object_type,
            "object_class": object_class,
            "object_subtype": object_subtype,
            "inline_object_type": inline_object_type,
            "inline_object_subtype": inline_object_subtype,
            "translation_policy": translation_policy,
            "translation_policy_translatable": policy_translatable,
            "reconstruction_contract": reconstruction_contract,
            "object_comprehension": dict(raw_unit.get("object_comprehension") or {}) if isinstance(raw_unit.get("object_comprehension"), dict) else {},
            "render_contract": dict(raw_unit.get("render_contract") or {}) if isinstance(raw_unit.get("render_contract"), dict) else {},
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
            "immutable": bool(child_summary["has_immutable"]) or policy_render == "source_overlay" or object_class in {"visual", "technical", "formula"},
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
        metadata = dict(getattr(unit, "metadata", {}) or {})
        render_contract = dict(metadata.get("render_contract") or {})
        contract_alignment = str(((render_contract.get("alignment") or {}).get("inline") or "")).strip().lower()
        if contract_alignment in {"left", "center", "right", "justify"}:
            return contract_alignment
        alignment = self._normalize_alignment(fallback)
        anchor = str(getattr(unit, "anchor_horizontal", "") or "").strip().lower()
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
        text = self._clean_text_for_render((line or {}).get("translated_text") or (line or {}).get("line_text") or (line or {}).get("text") or "")
        if text:
            return text
        parts = []
        for phrase in (line or {}).get("phrases") or []:
            phrase_text = self._clean_text_for_render((phrase or {}).get("translated_text") or (phrase or {}).get("texte") or (phrase or {}).get("text") or "")
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

    def _text_is_toc_leader_only(self, text):
        text = self._clean_text_for_render(text or "")
        if not text:
            return False
        return bool(re.fullmatch(r"[\.\s·•\-–—_]+", text)) and text.count(".") >= 3

    def _normalized_text_signature(self, text):
        text = self._clean_text_for_render(text or "").casefold()
        if not text:
            return ""
        return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)

    def _text_requires_visible_replacement(self, source_text, translated_text, *, item=None):
        source = self._clean_text_for_render(source_text or "")
        translated = self._clean_text_for_render(translated_text or "")
        if not translated:
            return False
        if not source:
            return True
        source_sig = self._normalized_text_signature(source)
        translated_sig = self._normalized_text_signature(translated)
        if source_sig and source_sig == translated_sig:
            return False
        source_alpha = sum(1 for ch in source if ch.isalpha())
        translated_alpha = sum(1 for ch in translated if ch.isalpha())
        if source_alpha == 0 and translated_alpha == 0:
            return False
        if item is not None and self._line_looks_technical_structured(item) and source_alpha == 0:
            return False
        return True

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
            source_text = self._line_source_text(line) or translated_text
            if self._reconstruction_contract_key_for_block(block, page_data=page_data) == "toc_entry" and self._text_is_toc_leader_only(translated_text) and any(ch.isalpha() for ch in source_text):
                translated_text = source_text
            if not translated_text:
                translated_text = source_text
            if not translated_text:
                continue
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
        if self._reconstruction_contract_key_for_block(block, page_data=page_data) == "toc_entry":
            line_units = self._line_units(block, target_lang, page_data=page_data)
            if line_units:
                return self._canonicalize_block_units(block, line_units)
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
            return self._canonicalize_block_units(block, external_units)
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

    def _canonical_render_contract(self, block, unit, page_data=None):
        block_type = self._classify_block_for_reconstruction(block, page_data)
        metadata = dict(getattr(unit, "metadata", {}) or {})
        render_contract = dict(metadata.get("render_contract") or {})
        adaptive = dict(metadata.get("adaptive_profile") or {})
        unit_profile = str(adaptive.get("unit_profile") or "").strip().lower()
        inline_class = str(getattr(unit, "inline_class", "") or "").strip().lower()
        render_policy = str(getattr(unit, "render_policy", "") or "").strip().lower()
        explicit_family = str(render_contract.get("family") or "").strip().lower()
        explicit_policy = str(render_contract.get("canonical_render_policy") or "").strip().lower()

        canonical_class = "paragraph_line"
        contract = render_policy or "translated_editorial"
        reflowable = bool(getattr(unit, "reflowable", True))
        protected_inline = bool(getattr(unit, "protected_inline", False))
        immutable = bool(getattr(unit, "immutable", False))
        justification_eligible = bool(getattr(unit, "justification_eligible", True))

        if immutable or render_policy == "source_overlay":
            canonical_class = "immutable_overlay"
            contract = "source_overlay"
            reflowable = False
            protected_inline = True
            immutable = True
            justification_eligible = False
        elif block_type == "code" or inline_class == "code":
            canonical_class = "code_line"
            contract = "source_overlay" if immutable else "code_line"
            reflowable = False
            protected_inline = True
            justification_eligible = False
        elif block_type == "table" or render_policy in {"cell_locked", "locked_in_cell", "locked_in_table"}:
            canonical_class = "table_cell_text"
            contract = "cell_locked"
            reflowable = False
            justification_eligible = False
        elif render_policy == "external_flow":
            canonical_class = "anchored_label" if unit_profile == "anchored_label" else "paragraph_line"
            contract = "external_flow"
            reflowable = False if unit_profile == "anchored_label" else reflowable
            justification_eligible = False if unit_profile == "anchored_label" else justification_eligible
        elif render_policy in {"anchored_text", "fixed_preserve", "anchored_external"} or unit_profile == "anchored_label":
            canonical_class = "anchored_label"
            contract = "anchored_text"
            reflowable = False
            justification_eligible = False
        elif protected_inline or inline_class in {"reference", "formula", "technical_inline"}:
            canonical_class = "inline_protected_token"
            reflowable = False
            protected_inline = True
            justification_eligible = False

        if explicit_family == "table_cell":
            canonical_class = "table_cell_text"
            contract = explicit_policy or "cell_locked"
            reflowable = False
            justification_eligible = False
        elif explicit_family in {"anchored", "fixed_lines"}:
            canonical_class = "anchored_label"
            contract = explicit_policy or "anchored_text"
            reflowable = False
            justification_eligible = False
        elif explicit_family in {"source_overlay", "background_only"}:
            canonical_class = "immutable_overlay"
            contract = "source_overlay"
            reflowable = False
            protected_inline = True
            immutable = True
            justification_eligible = False
        elif explicit_family == "code_line":
            canonical_class = "code_line"
            contract = explicit_policy or "code_line"
            reflowable = False
            protected_inline = True
            justification_eligible = False

        return {
            "canonical_unit_class": canonical_class,
            "canonical_render_contract": contract,
            "reflowable": reflowable,
            "protected_inline": protected_inline,
            "immutable": immutable,
            "justification_eligible": justification_eligible,
        }

    def _finalize_placable_units(self, block, units, page_data=None):
        finalized = []
        for unit in units or []:
            contract = self._canonical_render_contract(block, unit, page_data=page_data)
            metadata = dict(getattr(unit, "metadata", {}) or {})
            metadata.update({
                "canonical_unit_class": contract["canonical_unit_class"],
                "canonical_render_contract": contract["canonical_render_contract"],
                "rotation_deg": int(
                    metadata.get("rotation_deg")
                    or self._rotation_deg_for_bbox_text(
                        getattr(unit, "relative_bbox", None),
                        getattr(unit, "text_translated", "") or getattr(unit, "text_source", ""),
                        fallback=0,
                    )
                ),
            })
            finalized.append(
                replace(
                    unit,
                    render_policy=contract["canonical_render_contract"],
                    reflowable=contract["reflowable"],
                    protected_inline=contract["protected_inline"],
                    immutable=contract["immutable"],
                    justification_eligible=contract["justification_eligible"],
                    metadata=metadata,
                )
            )
        return finalized

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
        line_templates = self._build_line_templates(block, geometry_ctx, page_data=page_data)
        semantic_payload = self._collect_block_semantic_payload(block)
        contract_entry = self._contract_entry_for_block(page_data, block)
        contract_units = self._contract_driven_units_for_block(block, semantic_payload, target_lang, page_data=page_data)
        units = contract_units if contract_units is not None else self._normalize_placable_units(block, semantic_payload, target_lang, page_data=page_data)
        units = self._apply_source_layout_mode_to_units(block, units)
        units = self._finalize_placable_units(block, units, page_data=page_data)
        graph_edges = self._build_reconstruction_graph(units)
        block_rect = fitz.Rect(geometry_ctx.block_bbox)
        block_rect_tuple = (
            (block_rect.x0, block_rect.y0, block_rect.x1, block_rect.y1)
            if isinstance(block_rect, fitz.Rect) and block_rect.get_area() > 0
            else geometry_ctx.block_bbox
        )
        constraints = dict(geometry_ctx.constraints or {})
        descriptor_group_ids = dict((block or {}).get("descriptor_group_ids") or {})
        descriptor_page_organization = dict((block or {}).get("descriptor_page_organization") or {})
        table_region = self._descriptor_table_region_info(block)
        row_id = str(table_region.get("row_id") or "").strip()
        cell_id = str(table_region.get("cell_id") or "").strip()
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
        target_contract = self._reconstruction_contract_for_block(block, page_data=page_data)
        constraints["reconstruction_contract"] = target_contract
        constraints["reconstruction_contract_key"] = target_contract.get("contract_key")
        constraints["font_size_policy"] = target_contract.get("font_size_policy")
        constraints["style_policy"] = target_contract.get("style_policy")
        constraints["min_font_ratio"] = target_contract.get("min_font_ratio")
        constraints["shrink_max_pt"] = target_contract.get("shrink_max_pt")
        constraints["allow_contract_expansion"] = bool(target_contract.get("allow_expansion"))
        contract_render_mode = self._contract_render_mode_for_block(block, page_data=page_data, block_type=block_type, contract_entry=contract_entry)
        if contract_render_mode:
            constraints["contract_driven"] = True
            constraints["contract_render_mode"] = contract_render_mode
            constraints["contract_entry"] = contract_entry
            constraints["contract_structure_family"] = str((self._reconstruction_contract_payload(page_data) or {}).get("primary_structure_family") or "").strip().lower()
        structured_contract_plan = self._structured_contract_plan_for_block(
            block,
            page_data,
            units,
            contract_entry,
            block_type,
            contract_render_mode,
        )
        if structured_contract_plan:
            constraints["structured_contract_plan"] = structured_contract_plan
            constraints["structured_contract_kind"] = str(structured_contract_plan.get("kind") or "").strip().lower()
            if structured_contract_plan.get("background_region_id"):
                constraints["structured_background_region_id"] = structured_contract_plan.get("background_region_id")
            if structured_contract_plan.get("background_region_bbox"):
                constraints["structured_background_region_bbox"] = tuple(structured_contract_plan.get("background_region_bbox"))
        alignment = self._contract_alignment_value(block, key="inline", fallback=(block or {}).get("alignment") or "left")
        paragraph_alignment = self._contract_alignment_value(block, key="paragraph", fallback=alignment)
        paragraph_alignment = self._infer_paragraph_alignment_from_lines(block, paragraph_alignment)
        semantic_profile = None
        if contract_render_mode:
            render_strategy = {
                "line_preserve": "contract_line_preserve",
                "bbox_anchored": "contract_bbox_anchored",
                "cell_locked": "contract_cell_locked",
                "prose_reflow": "prose_reflow",
            }.get(contract_render_mode, "contract_line_preserve")
            base_style = self._style_from_block(block or {})
            flags = dict(base_style.get("flags") or {})
            semantic_profile = BlockSemanticProfile(
                block_id=str((block or {}).get("id") or ""),
                content_class="contract",
                render_strategy=render_strategy,
                font_normalization="fit_to_bbox",
                allow_vertical_expansion=bool(contract_render_mode == "prose_reflow"),
                text_flow_mode="line_by_line" if contract_render_mode in {"line_preserve", "bbox_anchored", "cell_locked"} else "continuous",
                unicode_safe_required=True,
                source_is_translated=self._has_translated_payload(block),
                estimated_text_expansion=1.0,
                dominant_fontsize=float(base_style.get("size") or 12.0),
                dominant_is_serif=bool(flags.get("serif")),
                dominant_is_bold=bool(flags.get("bold")),
                dominant_is_italic=bool(flags.get("italic")),
                dominant_is_mono=bool(flags.get("monospace")),
            )
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
            paragraph_alignment=paragraph_alignment,
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
            page_data=dict(page_data or {}),
            source_block=dict(block or {}),
            semantic_profile=semantic_profile,
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
        if plan.block_type == "code":
            return CodeBlockRenderer(self)
        structured_kind = str((plan.constraints or {}).get("structured_contract_kind") or "").strip().lower()
        if structured_kind:
            return StructuredContractRenderer(self)
        contract_render_mode = str((plan.constraints or {}).get("contract_render_mode") or "").strip().lower()
        if contract_render_mode == "cell_locked":
            return TableBlockRenderer(self)
        if contract_render_mode in {"line_preserve", "bbox_anchored", "relative_slots", "prose_reflow"}:
            if plan.block_type == "heading":
                return HeadingBlockRenderer(self)
            if plan.block_type == "caption":
                return CaptionBlockRenderer(self)
            if plan.block_type == "annotation":
                return AnnotationBlockRenderer(self)
            return EditorialBlockRenderer(self)
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
        return None

    def _render_hierarchical_block_plan(self, page, plan):
        renderer = self._select_block_renderer(plan)
        if renderer is None:
            return []
        return renderer.render(page, plan)

    def _background_region_signature(self, plan):
        region_id = str((plan.constraints or {}).get("structured_background_region_id") or "").strip()
        if region_id:
            return region_id
        return ""

    def _same_rect_bbox(self, lhs, rhs, tolerance=1.0):
        left = self._fitz_rect_from_bbox_like(lhs)
        right = self._fitz_rect_from_bbox_like(rhs)
        if not isinstance(left, fitz.Rect) or not isinstance(right, fitz.Rect):
            return False
        return (
            abs(left.x0 - right.x0) <= tolerance
            and abs(left.y0 - right.y0) <= tolerance
            and abs(left.x1 - right.x1) <= tolerance
            and abs(left.y1 - right.y1) <= tolerance
        )

    def _filter_redundant_background_region_ops(self, plan, ops, rendered_regions):
        region_sig = self._background_region_signature(plan)
        if not region_sig:
            return list(ops or [])
        region_bbox = (plan.constraints or {}).get("structured_background_region_bbox")
        if region_sig not in rendered_regions:
            rendered_regions.add(region_sig)
            return list(ops or [])
        filtered = []
        for op in ops or []:
            if (
                op.op_type in {"erase_rect", "draw_overlay_image"}
                and int(op.z_index or 0) == 0
                and self._same_rect_bbox(op.bbox, region_bbox)
            ):
                continue
            filtered.append(op)
        return filtered

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

    def _line_preserve_effective_rect(self, plan):
        """En mode line_preserve, étend la block_bbox pour couvrir toutes les positions de templates."""
        block_rect = fitz.Rect(plan.block_bbox)
        if str((plan.constraints or {}).get("contract_render_mode") or "").strip().lower() != "line_preserve":
            return block_rect
        fontsize = float(getattr(plan, "dominant_fontsize", None) or 12.0)
        if plan.semantic_profile is not None:
            fontsize = float(plan.semantic_profile.dominant_fontsize or fontsize)
        for tmpl in (plan.line_templates or []):
            try:
                baseline = float(tmpl.baseline_y)
                left_x = float(tmpl.left_x)
                right_x = float(tmpl.right_x)
                tmpl_rect = fitz.Rect(
                    min(block_rect.x0, left_x),
                    min(block_rect.y0, baseline - fontsize),
                    max(block_rect.x1, right_x),
                    max(block_rect.y1, baseline + fontsize * 0.3),
                )
                block_rect |= tmpl_rect
            except Exception:
                pass
        return block_rect

    def _validate_block_layout(self, plan, ops):
        findings = []
        block_rect = self._line_preserve_effective_rect(plan)
        tolerance = 2.0
        text_rects = []
        protected_rects = []
        for region in plan.protected_regions or []:
            rect = self._fitz_rect_from_bbox_like((region or {}).get("bbox"))
            if isinstance(rect, fitz.Rect) and rect.get_area() > 0:
                protected_rects.append(rect)
        for op in ops or []:
            for finding in ((getattr(op, "metadata", {}) or {}).get("render_findings") or []):
                if isinstance(finding, dict):
                    findings.append(dict(finding))
            rect = fitz.Rect(op.bbox) if isinstance(op.bbox, (list, tuple)) and len(op.bbox) == 4 else None
            if op.op_type.startswith("draw_text") and isinstance(rect, fitz.Rect) and rect.get_area() > 0:
                metadata = getattr(op, "metadata", {}) or {}
                intended_rect = None
                intended_bbox = metadata.get("intended_bbox")
                if isinstance(intended_bbox, (list, tuple)) and len(intended_bbox) == 4:
                    intended_rect = fitz.Rect(intended_bbox)
                try:
                    rendered_fontsize = float(metadata.get("fontsize") or 0.0)
                    source_fontsize = float(metadata.get("source_fontsize") or 0.0)
                except Exception:
                    rendered_fontsize = 0.0
                    source_fontsize = 0.0
                min_font_ratio = float(metadata.get("min_font_ratio") or 0.0)
                if source_fontsize > 0 and rendered_fontsize > 0 and min_font_ratio > 0:
                    ratio = rendered_fontsize / source_fontsize
                    if ratio + 1e-6 < min_font_ratio:
                        findings.append(
                            {
                                "type": "font_too_small",
                                "unit_id": op.unit_id,
                                "bbox": tuple(rect),
                                "source_fontsize": source_fontsize,
                                "rendered_fontsize": rendered_fontsize,
                                "font_size_ratio": ratio,
                                "min_font_ratio": min_font_ratio,
                            }
                        )
                if (
                    rect.x0 < block_rect.x0 - tolerance
                    or rect.x1 > block_rect.x1 + tolerance
                    or rect.y0 < block_rect.y0 - tolerance
                    or rect.y1 > block_rect.y1 + tolerance
                    or (
                        isinstance(intended_rect, fitz.Rect)
                        and (
                            intended_rect.x0 < block_rect.x0 - tolerance
                            or intended_rect.x1 > block_rect.x1 + tolerance
                            or intended_rect.y0 < block_rect.y0 - tolerance
                            or intended_rect.y1 > block_rect.y1 + tolerance
                        )
                    )
                ):
                    findings.append({"type": "overflow", "unit_id": op.unit_id, "bbox": tuple(intended_rect or rect)})
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
        block_rect = self._line_preserve_effective_rect(plan)
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
        align_map = {"left": 0, "center": 1, "right": 2, "justify": 3}
        for op in ops or []:
            rect = fitz.Rect(op.bbox) if isinstance(op.bbox, (list, tuple)) and len(op.bbox) == 4 else None
            if isinstance(rect, fitz.Rect):
                coords = (rect.x0, rect.y0, rect.x1, rect.y1)
                if not all(math.isfinite(float(v)) for v in coords):
                    rect = None
                elif rect.width <= 0 or rect.height <= 0:
                    rect = None
            if op.op_type == "erase_rect" and isinstance(rect, fitz.Rect):
                fill_rgb = self._normalized_rgb((op.metadata or {}).get("fill_rgb") or (1, 1, 1))
                page.draw_rect(rect, color=None, fill=fill_rgb, overlay=True)
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
                rotation_deg = int(round(float((op.metadata or {}).get("rotation_deg") or style.get("_rotation_deg") or 0.0))) % 360
                insert_kwargs = {
                    "fontsize": fontsize,
                    "color": rgb,
                    "overlay": True,
                }
                try:
                    if rotation_deg in {90, 180, 270} and isinstance(rect, fitz.Rect):
                        align_name = str((op.metadata or {}).get("textbox_align") or style.get("_textbox_align") or "left").strip().lower()
                        textbox_kwargs = {
                            **insert_kwargs,
                            "rotate": rotation_deg,
                            "align": align_map.get(align_name, 0),
                        }
                        if fontfile and not builtin:
                            page.insert_textbox(rect, op.text or "", fontname=fontname, fontfile=fontfile, **textbox_kwargs)
                        else:
                            page.insert_textbox(rect, op.text or "", fontname=fontname, **textbox_kwargs)
                    elif rotation_deg in {90, 180, 270}:
                        continue
                    elif fontfile and not builtin:
                        page.insert_text(point, op.text or "", fontname=fontname, fontfile=fontfile, **insert_kwargs)
                    else:
                        page.insert_text(point, op.text or "", fontname=fontname, **insert_kwargs)
                except Exception:
                    if rotation_deg in {90, 180, 270} and isinstance(rect, fitz.Rect):
                        page.insert_textbox(
                            rect,
                            op.text or "",
                            fontname="helv",
                            rotate=rotation_deg,
                            align=align_map.get(str((op.metadata or {}).get("textbox_align") or "left").strip().lower(), 0),
                            **insert_kwargs,
                        )
                    else:
                        page.insert_text(point, op.text or "", fontname="helv", **insert_kwargs)

    def _clean_page_background_path(self, page_data):
        # The cleaned background is the reconstruction substrate. We keep the
        # audit for diagnostics, but we do not veto the cleaned background here,
        # otherwise the renderer can silently fall back to the source image and
        # reintroduce the original text under the translated overlay.
        path = str((page_data or {}).get("background_path") or "").strip()
        return path if path and os.path.exists(path) else None

    def _page_background_path(self, page_data):
        for key in ("background_path", "source_image_path"):
            path = str((page_data or {}).get(key) or "").strip()
            if path and os.path.exists(path):
                return path
        return None

    def _page_background_crop_source_path(self, page_data, allow_unsafe=False):
        clean = self._clean_page_background_path(page_data)
        if clean and os.path.exists(clean):
            return clean
        if allow_unsafe:
            path = str((page_data or {}).get("background_path") or "").strip()
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

    def _local_background_crop_path(self, page_data, bbox, allow_unsafe=False, pad_px=2):
        source_path = self._page_background_crop_source_path(page_data, allow_unsafe=allow_unsafe)
        rect = self._fitz_rect_from_bbox_like(bbox)
        if not source_path or not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
            return None
        key = (
            source_path,
            bool(allow_unsafe),
            int(pad_px),
            tuple(round(float(v), 2) for v in (rect.x0, rect.y0, rect.x1, rect.y1)),
        )
        cached = self._local_background_cache.get(key)
        if cached and os.path.exists(cached):
            return cached
        page_w_pt, page_h_pt = self._page_size_pt(page_data)
        try:
            with Image.open(source_path) as img:
                scale_x = float(img.width) / max(1.0, page_w_pt)
                scale_y = float(img.height) / max(1.0, page_h_pt)
                left = max(0, int(math.floor(rect.x0 * scale_x)) - int(pad_px))
                top = max(0, int(math.floor(rect.y0 * scale_y)) - int(pad_px))
                right = min(img.width, int(math.ceil(rect.x1 * scale_x)) + int(pad_px))
                bottom = min(img.height, int(math.ceil(rect.y1 * scale_y)) + int(pad_px))
                if right <= left or bottom <= top:
                    return None
                crop = img.crop((left, top, right, bottom))
                cache_dir = Path("/tmp/docs_parser_local_bg")
                cache_dir.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()
                out_path = cache_dir / f"{digest}.png"
                if not out_path.exists():
                    crop.save(out_path)
        except Exception:
            return None
        out_str = str(out_path)
        self._local_background_cache[key] = out_str
        return out_str

    def _sample_region_fill_rgb(self, page_data, bbox, allow_unsafe=False):
        source_path = self._page_background_crop_source_path(page_data, allow_unsafe=allow_unsafe)
        rect = self._fitz_rect_from_bbox_like(bbox)
        if (not source_path or not os.path.exists(source_path) or not isinstance(rect, fitz.Rect) or rect.get_area() <= 0):
            source_path = str((page_data or {}).get("source_image_path") or "").strip()
        if not source_path or not os.path.exists(source_path) or not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
            return None
        page_w_pt, page_h_pt = self._page_size_pt(page_data)
        try:
            with Image.open(source_path) as img:
                rgb_img = img.convert("RGB")
                scale_x = float(rgb_img.width) / max(1.0, page_w_pt)
                scale_y = float(rgb_img.height) / max(1.0, page_h_pt)
                left = max(0, int(math.floor(rect.x0 * scale_x)))
                top = max(0, int(math.floor(rect.y0 * scale_y)))
                right = min(rgb_img.width, int(math.ceil(rect.x1 * scale_x)))
                bottom = min(rgb_img.height, int(math.ceil(rect.y1 * scale_y)))
                if right <= left or bottom <= top:
                    return None
                crop = rgb_img.crop((left, top, right, bottom))
                if crop.width <= 0 or crop.height <= 0:
                    return None
                stats = ImageStat.Stat(crop)
                mean = list(stats.mean or [255.0, 255.0, 255.0])[:3]
                stddev = list(stats.stddev or [0.0, 0.0, 0.0])[:3]
                rgb = tuple(max(0.0, min(1.0, float(channel) / 255.0)) for channel in mean)
                max_std = max((float(channel) / 255.0) for channel in stddev) if stddev else 0.0
                brightness = sum(rgb) / 3.0
                return {
                    "rgb": rgb,
                    "brightness": brightness,
                    "max_stddev": max_std,
                    "uniform": bool(max_std <= 0.055),
                    "light_uniform": bool(brightness >= 0.83 and max_std <= 0.08),
                }
        except Exception:
            return None

    def _sample_surrounding_region_fill_rgb(self, page_data, bbox, allow_unsafe=False):
        source_path = self._page_background_crop_source_path(page_data, allow_unsafe=allow_unsafe)
        rect = self._fitz_rect_from_bbox_like(bbox)
        if (not source_path or not os.path.exists(source_path) or not isinstance(rect, fitz.Rect) or rect.get_area() <= 0):
            source_path = str((page_data or {}).get("source_image_path") or "").strip()
        if not source_path or not os.path.exists(source_path) or not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
            return None
        page_w_pt, page_h_pt = self._page_size_pt(page_data)
        try:
            with Image.open(source_path) as img:
                rgb_img = img.convert("RGB")
                scale_x = float(rgb_img.width) / max(1.0, page_w_pt)
                scale_y = float(rgb_img.height) / max(1.0, page_h_pt)
                left = max(0, int(math.floor(rect.x0 * scale_x)))
                top = max(0, int(math.floor(rect.y0 * scale_y)))
                right = min(rgb_img.width, int(math.ceil(rect.x1 * scale_x)))
                bottom = min(rgb_img.height, int(math.ceil(rect.y1 * scale_y)))
                if right <= left or bottom <= top:
                    return None
                pad_x = max(3, min(40, int(math.ceil((right - left) * 0.45))))
                pad_y = max(3, min(30, int(math.ceil((bottom - top) * 0.65))))
                outer_left = max(0, left - pad_x)
                outer_top = max(0, top - pad_y)
                outer_right = min(rgb_img.width, right + pad_x)
                outer_bottom = min(rgb_img.height, bottom + pad_y)
                if outer_right <= outer_left or outer_bottom <= outer_top:
                    return None
                if outer_left == left and outer_top == top and outer_right == right and outer_bottom == bottom:
                    return None
                crop = rgb_img.crop((outer_left, outer_top, outer_right, outer_bottom))
                mask = Image.new("L", crop.size, 255)
                draw = ImageDraw.Draw(mask)
                inner = (
                    max(0, left - outer_left - 1),
                    max(0, top - outer_top - 1),
                    min(crop.width, right - outer_left + 1),
                    min(crop.height, bottom - outer_top + 1),
                )
                draw.rectangle(inner, fill=0)
                stats = ImageStat.Stat(crop, mask=mask)
                if not stats.count or max(stats.count) <= 0:
                    return None
                mean = list(stats.mean or [255.0, 255.0, 255.0])[:3]
                stddev = list(stats.stddev or [0.0, 0.0, 0.0])[:3]
                rgb = tuple(max(0.0, min(1.0, float(channel) / 255.0)) for channel in mean)
                max_std = max((float(channel) / 255.0) for channel in stddev) if stddev else 0.0
                brightness = sum(rgb) / 3.0
                return {
                    "rgb": rgb,
                    "brightness": brightness,
                    "max_stddev": max_std,
                    "uniform": bool(max_std <= 0.075),
                    "light_uniform": bool(brightness >= 0.83 and max_std <= 0.10),
                    "sample": "surrounding_ring",
                }
        except Exception:
            return None

    def _sample_local_background_fill_rgb(self, page_data, bbox, allow_unsafe=False):
        sampled = self._sample_surrounding_region_fill_rgb(page_data, bbox, allow_unsafe=allow_unsafe)
        if isinstance(sampled, dict) and sampled.get("rgb"):
            return sampled
        return self._sample_region_fill_rgb(page_data, bbox, allow_unsafe=allow_unsafe)

    def _expanded_text_mask_rect_px(self, rect_px, *, rotation_deg=0, role=""):
        if not isinstance(rect_px, fitz.Rect) or rect_px.get_area() <= 0:
            return rect_px
        role = str(role or "").strip().lower()
        table_like = role in {"table", "table_cell", "table_header_cell", "table_stub_cell", "table_value_cell"}
        if table_like and int(rotation_deg or 0) % 180 == 90:
            pad_x = max(0.5, min(2.0, rect_px.width * 0.12))
            pad_y = max(0.5, min(2.0, rect_px.height * 0.03))
        elif table_like:
            pad_x = max(0.5, min(2.5, rect_px.width * 0.03))
            pad_y = max(0.5, min(1.5, rect_px.height * 0.08))
        elif int(rotation_deg or 0) % 180 == 90:
            pad_x = max(1.0, min(4.0, rect_px.width * 0.30))
            pad_y = max(1.0, min(8.0, rect_px.height * 0.04))
        else:
            pad_x = max(1.0, min(8.0, rect_px.width * 0.08))
            pad_y = max(1.0, min(4.0, rect_px.height * 0.18))
        if role in {"header", "title", "figure_caption", "diagram_label"}:
            pad_x += 0.5
            pad_y += 0.5
        return fitz.Rect(rect_px.x0 - pad_x, rect_px.y0 - pad_y, rect_px.x1 + pad_x, rect_px.y1 + pad_y)

    def _source_text_mask_rects_for_plan(self, plan, rect_pt):
        if not isinstance(rect_pt, fitz.Rect) or rect_pt.get_area() <= 0:
            return []
        block = dict(plan.source_block or {})
        block_structural_role = str((block or {}).get("descriptor_structural_role") or "").strip().lower()
        block_role = block_structural_role or str((block or {}).get("role") or "").strip().lower()
        table_like = block_role in {"table", "table_cell", "table_header_cell", "table_stub_cell", "table_value_cell"}
        if str(getattr(plan, "block_type", "") or "").strip().lower() == "table" and not block_structural_role:
            block_role = "table_cell"
            table_like = True
        crop_px = fitz.Rect(
            rect_pt.x0 / self.pixel_to_point,
            rect_pt.y0 / self.pixel_to_point,
            rect_pt.x1 / self.pixel_to_point,
            rect_pt.y1 / self.pixel_to_point,
        )
        mask_rects = []
        candidates = []
        for line in block.get("lines") or []:
            if isinstance(line, dict):
                phrase_added = False
                for phrase in line.get("phrases") or []:
                    if isinstance(phrase, dict):
                        phrase_source = phrase.get("text") or phrase.get("texte") or ""
                        phrase_translated = phrase.get("translated_text") or ""
                        if not self._text_requires_visible_replacement(phrase_source, phrase_translated, item=phrase):
                            continue
                        candidates.append((phrase, phrase.get("bbox"), phrase_source, phrase_translated))
                        phrase_added = True
                line_source = self._line_source_text(line)
                line_translated = self._line_translated_text(line)
                if self._text_requires_visible_replacement(line_source, line_translated, item=line) and (not table_like or not phrase_added):
                    candidates.append((line, line.get("bbox"), line_source, line_translated))
        if not candidates and block.get("bbox"):
            block_source = block.get("text") or block.get("line_text") or ""
            block_translated = block.get("translated_text") or ""
            if self._text_requires_visible_replacement(block_source, block_translated, item=block):
                candidates.append((block, block.get("bbox"), block_source, block_translated))
        seen = set()
        for item, bbox, text, translated_text in candidates:
            if not self._clean_text_for_render(text) or not self._clean_text_for_render(translated_text):
                continue
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                rect_px = fitz.Rect([float(v) for v in bbox])
            except Exception:
                continue
            inter = rect_px & crop_px
            if inter.get_area() <= 0.5:
                continue
            rotation_deg = self._rotation_deg_for_item(item, bbox_like=bbox, text=text, fallback=0)
            expanded = self._expanded_text_mask_rect_px(
                inter,
                rotation_deg=rotation_deg,
                role=item.get("descriptor_structural_role") or item.get("role") or block_role,
            )
            expanded = expanded & crop_px
            if expanded.get_area() <= 0.5:
                continue
            key = tuple(round(float(v), 2) for v in (expanded.x0, expanded.y0, expanded.x1, expanded.y1))
            if key in seen:
                continue
            seen.add(key)
            mask_rects.append([expanded.x0, expanded.y0, expanded.x1, expanded.y1])
        return mask_rects

    def _local_inpaint_overlay_for_plan(self, plan, rect_pt):
        inpainter = getattr(self, "background_inpainter", None)
        if inpainter is None or not getattr(inpainter, "enabled", False):
            return None
        page_data = dict(plan.page_data or {})
        source_img = str(page_data.get("source_image_path") or "").strip()
        if not source_img or not os.path.exists(source_img):
            return None
        rect_pt = fitz.Rect(rect_pt)
        rect_pt = rect_pt & fitz.Rect(plan.block_bbox)
        if rect_pt.get_area() <= 0:
            return None
        mask_rects = self._source_text_mask_rects_for_plan(plan, rect_pt)
        if not mask_rects:
            return None
        crop_bbox = [
            rect_pt.x0 / self.pixel_to_point,
            rect_pt.y0 / self.pixel_to_point,
            rect_pt.x1 / self.pixel_to_point,
            rect_pt.y1 / self.pixel_to_point,
        ]
        out_dir = os.path.dirname(page_data.get("background_path", "")) or "ocr_results"
        return inpainter.save_inpaint_overlay(
            source_image_path=source_img,
            crop_bbox=crop_bbox,
            mask_rects=mask_rects,
            out_dir=out_dir,
            kind=f"{str(plan.block_type or 'block').strip().lower()}_local_bg_restore",
        )

    def _text_erase_ops_for_plan(self, plan, rect_pt):
        rect_pt = self._fitz_rect_from_bbox_like(rect_pt)
        if not isinstance(rect_pt, fitz.Rect) or rect_pt.get_area() <= 0:
            return []
        mask_rects = self._source_text_mask_rects_for_plan(plan, rect_pt)
        ops = []
        for mask_rect in mask_rects:
            try:
                mask_px = fitz.Rect([float(v) for v in mask_rect])
            except Exception:
                continue
            if mask_px.get_area() <= 0:
                continue
            rect = fitz.Rect(
                mask_px.x0 * self.pixel_to_point,
                mask_px.y0 * self.pixel_to_point,
                mask_px.x1 * self.pixel_to_point,
                mask_px.y1 * self.pixel_to_point,
            ) & rect_pt
            if rect.get_area() <= 0:
                continue
            sampled = self._sample_local_background_fill_rgb(plan.page_data, fitz.Rect(rect), allow_unsafe=True)
            ops.append(
                BlockRenderOp(
                    "erase_rect",
                    plan.block_id,
                    None,
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    z_index=0,
                    metadata=(
                        {"fill_rgb": sampled.get("rgb"), "fill_sample": sampled.get("sample") or "region"}
                        if isinstance(sampled, dict) and sampled.get("rgb")
                        else {}
                    ),
                )
            )
        return ops

    def _text_background_patch_ops_for_plan(self, plan, rect_pt, *, allow_unsafe=False, prefer_inpaint=False):
        rect_pt = self._fitz_rect_from_bbox_like(rect_pt)
        if not isinstance(rect_pt, fitz.Rect) or rect_pt.get_area() <= 0:
            return []
        mask_rects = self._source_text_mask_rects_for_plan(plan, rect_pt)
        ops = []
        for mask_rect in mask_rects:
            try:
                mask_px = fitz.Rect([float(v) for v in mask_rect])
            except Exception:
                continue
            if mask_px.get_area() <= 0:
                continue
            mask_pt = fitz.Rect(
                mask_px.x0 * self.pixel_to_point,
                mask_px.y0 * self.pixel_to_point,
                mask_px.x1 * self.pixel_to_point,
                mask_px.y1 * self.pixel_to_point,
            ) & rect_pt
            if mask_pt.get_area() <= 0:
                continue
            sampled = self._sample_local_background_fill_rgb(
                plan.page_data,
                fitz.Rect(mask_pt),
                allow_unsafe=True,
            )
            if isinstance(sampled, dict) and (sampled.get("uniform") or sampled.get("light_uniform")):
                ops.append(
                    BlockRenderOp(
                        "erase_rect",
                        plan.block_id,
                        None,
                        bbox=(mask_pt.x0, mask_pt.y0, mask_pt.x1, mask_pt.y1),
                        z_index=0,
                        metadata={"fill_rgb": sampled.get("rgb"), "fill_sample": sampled.get("sample") or "region"},
                    )
                )
                continue
            if prefer_inpaint:
                overlay = self._local_inpaint_overlay_for_plan(plan, mask_pt)
                if isinstance(overlay, dict):
                    ops.append(
                        BlockRenderOp(
                            op_type="draw_overlay_image",
                            block_id=plan.block_id,
                            unit_id=None,
                            bbox=(mask_pt.x0, mask_pt.y0, mask_pt.x1, mask_pt.y1),
                            z_index=0,
                            metadata={"path": overlay.get("path")},
                        )
                    )
                    continue
            crop_path = self._local_background_crop_path(
                plan.page_data,
                fitz.Rect(mask_pt),
                allow_unsafe=allow_unsafe,
                pad_px=0,
            )
            if crop_path and os.path.exists(crop_path):
                ops.append(
                    BlockRenderOp(
                        op_type="draw_overlay_image",
                        block_id=plan.block_id,
                        unit_id=None,
                        bbox=(mask_pt.x0, mask_pt.y0, mask_pt.x1, mask_pt.y1),
                        z_index=0,
                        metadata={"path": crop_path},
                    )
                )
                continue
            ops.append(
                BlockRenderOp(
                    "erase_rect",
                    plan.block_id,
                    None,
                    bbox=(mask_pt.x0, mask_pt.y0, mask_pt.x1, mask_pt.y1),
                    z_index=0,
                    metadata=(
                        {"fill_rgb": sampled.get("rgb"), "fill_sample": sampled.get("sample") or "region"}
                        if isinstance(sampled, dict) and sampled.get("rgb")
                        else {}
                    ),
                )
            )
        return ops

    def _insert_page_background(self, page, page_data):
        bg = self._clean_page_background_path(page_data) or str((page_data or {}).get("source_image_path") or "").strip()
        if bg and os.path.exists(bg):
            page.insert_image(page.rect, filename=bg, overlay=False)

    def _overlay_signature(self, overlay):
        if not isinstance(overlay, dict):
            return None
        bbox = overlay.get("bbox")
        rect = self._fitz_rect_from_bbox_like(bbox)
        if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
            return None
        path = str(overlay.get("path") or "").strip()
        return (
            path,
            tuple(round(float(v), 2) for v in (rect.x0, rect.y0, rect.x1, rect.y1)),
        )

    def _remaining_page_immutable_overlays(self, page_data, rendered_overlay_signatures):
        if not rendered_overlay_signatures:
            return list((page_data or {}).get("immutable_overlays") or [])
        remaining = []
        for overlay in (page_data or {}).get("immutable_overlays") or []:
            sig = self._overlay_signature(overlay)
            if sig and sig in rendered_overlay_signatures:
                continue
            remaining.append(overlay)
        return remaining

    def _render_toc_page_rows(self, page, page_data, target_lang):
        page_role = str((page_data or {}).get("page_role") or "").strip().lower()
        if page_role != "toc":
            return False
        toc = (page_data or {}).get("toc") or {}
        rows = list(toc.get("toc_rows") or [])
        if not rows:
            return False

        page_rect = fitz.Rect(page.rect)
        rendered_any = False
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = self._clean_text_for_render(
                row.get("translated_label")
                or row.get("translated_text")
                or row.get("label")
                or ""
            )
            page_value = self._clean_text_for_render(row.get("page") or "")
            if not label and not page_value:
                continue

            role = str(row.get("role") or "").strip().lower()
            style = {}
            if isinstance(row.get("style"), dict):
                style.update(row.get("style") or {})
            if isinstance(row.get("translation_style"), str):
                style.setdefault("_translation_style", row.get("translation_style"))
            if isinstance(row.get("translation_tone"), str):
                style.setdefault("_translation_tone", row.get("translation_tone"))
            page_style_base = dict(row.get("page_style") or {}) if isinstance(row.get("page_style"), dict) else dict(style)
            marker_style_base = dict(row.get("marker_style") or {}) if isinstance(row.get("marker_style"), dict) else {}

            label_rect = self._fitz_rect_from_bbox_like(row.get("label_bbox") or row.get("bbox") or row.get("page_bbox"))
            page_rect_row = self._fitz_rect_from_bbox_like(row.get("page_bbox") or row.get("bbox") or row.get("label_bbox"))
            if not isinstance(label_rect, fitz.Rect) or label_rect.get_area() <= 0:
                label_rect = self._fitz_rect_from_bbox_like(row.get("bbox"))
            if not isinstance(page_rect_row, fitz.Rect) or page_rect_row.get_area() <= 0:
                page_rect_row = self._fitz_rect_from_bbox_like(row.get("bbox"))

            if not isinstance(label_rect, fitz.Rect) or label_rect.get_area() <= 0:
                label_rect = fitz.Rect(page_rect.x0 + 24.0, page_rect.y0, max(page_rect.x0 + 24.0, page_rect.x1 - 24.0), page_rect.y1)
            if not isinstance(page_rect_row, fitz.Rect) or page_rect_row.get_area() <= 0:
                page_rect_row = fitz.Rect(max(page_rect.x1 - 72.0, label_rect.x1 + 12.0), label_rect.y0, page_rect.x1 - 24.0, label_rect.y1)

            # TOC source bboxes often bound only the English label glyphs.
            # French labels are longer; the real layout slot is the row area
            # up to the page-number column, not the original text width.
            if role != "toc_title":
                if page_value:
                    available_right = max(label_rect.x1, page_rect_row.x0 - 3.0)
                else:
                    available_right = page_rect.x1 - 24.0
                if available_right > label_rect.x1 + 2.0:
                    label_rect = fitz.Rect(label_rect.x0, label_rect.y0, available_right, label_rect.y1)

            row_bbox = fitz.Rect(
                min(label_rect.x0, page_rect_row.x0),
                min(label_rect.y0, page_rect_row.y0),
                max(label_rect.x1, page_rect_row.x1),
                max(label_rect.y1, page_rect_row.y1),
            )
            row_bbox = row_bbox & page_rect
            if row_bbox.get_area() <= 0:
                continue

            base_fontsize = float(style.get("size") or style.get("font_size_pt") or 12.0)
            min_fontsize = 6.0 if role == "toc_title" else 5.5
            page_base_fontsize = float(page_style_base.get("size") or base_fontsize)
            label_fontsize = min(base_fontsize, max(min_fontsize, min(label_rect.height * 0.88, label_rect.width * 0.24)))
            page_fontsize = min(page_base_fontsize, max(min_fontsize, min(page_rect_row.height * 0.88, page_rect_row.width * 0.38)))

            label_style = dict(style)
            label_style["size"] = label_fontsize
            page_style = dict(page_style_base or style)
            page_style.setdefault("_translation_style", style.get("_translation_style"))
            page_style.setdefault("_translation_tone", style.get("_translation_tone"))
            page_style["size"] = page_fontsize

            label_font_style, label_fontfile, label_builtin, label_fontname = self._resolve_style_font(page, label_style, text=label or page_value)
            page_font_style, page_fontfile, page_builtin, page_fontname = self._resolve_style_font(page, page_style, text=page_value or label)
            label_rgb = self._resolve_text_color(label_style, row)
            page_rgb = self._resolve_text_color(page_style, row)

            # Avoid wrapping TOC rows into fragments. We prefer shrinking
            # within the row geometry over creating extra lines.
            while label_fontsize > min_fontsize:
                if self._measure_text_width(label, label_fontsize, label_fontname, label_fontfile) <= max(8.0, label_rect.width * 0.98):
                    break
                label_fontsize -= 0.5
            while page_fontsize > min_fontsize:
                if self._measure_text_width(page_value, page_fontsize, page_fontname, page_fontfile) <= max(8.0, page_rect_row.width * 0.98):
                    break
                page_fontsize -= 0.5

            label_style["size"] = label_fontsize
            page_style["size"] = page_fontsize
            label_font_style, label_fontfile, label_builtin, label_fontname = self._resolve_style_font(page, label_style, text=label or page_value)
            page_font_style, page_fontfile, page_builtin, page_fontname = self._resolve_style_font(page, page_style, text=page_value or label)
            label_rgb = self._resolve_text_color(label_style, row)
            page_rgb = self._resolve_text_color(page_style, row)

            # Draw title rows centered when they are page headers, otherwise
            # keep the label/page split stable.
            if role == "toc_title":
                target_rect = row_bbox
                text = label or page_value
                if text:
                    try:
                        baseline = target_rect.y0 + min(target_rect.height - 1.0, max(label_fontsize * 0.82, target_rect.height * 0.62))
                        width = self._measure_text_width(text, label_fontsize, label_fontname, label_fontfile)
                        x = max(target_rect.x0, target_rect.x0 + max(0.0, (target_rect.width - width) / 2.0))
                        page.insert_text(
                            (x, baseline),
                            text,
                            fontname=label_fontname if label_builtin or not label_fontfile else label_fontname,
                            fontfile=label_fontfile if label_fontfile and not label_builtin else None,
                            fontsize=label_fontsize,
                            color=label_rgb,
                            overlay=True,
                        )
                    except Exception:
                        page.insert_text(
                            (target_rect.x0 + 4.0, target_rect.y1 - max(1.0, label_fontsize * 0.18)),
                            text,
                            fontname=label_fontname if label_builtin or not label_fontfile else label_fontname,
                            fontfile=label_fontfile if label_fontfile and not label_builtin else None,
                            fontsize=label_fontsize,
                            color=label_rgb,
                            overlay=True,
                        )
                    rendered_any = True
                continue

            if label:
                try:
                    written = page.insert_textbox(
                        label_rect,
                        label,
                        fontname=label_fontname if label_builtin or not label_fontfile else label_fontname,
                        fontfile=label_fontfile if label_fontfile and not label_builtin else None,
                        fontsize=label_fontsize,
                        color=label_rgb,
                        align=0,
                        overlay=True,
                    )
                    if isinstance(written, (int, float)) and written < 0:
                        raise RuntimeError("toc label did not fit textbox")
                except Exception:
                    page.insert_text(
                        (label_rect.x0, label_rect.y1 - max(1.0, label_fontsize * 0.18)),
                        label,
                        fontname=label_fontname if label_builtin or not label_fontfile else label_fontname,
                        fontfile=label_fontfile if label_fontfile and not label_builtin else None,
                        fontsize=label_fontsize,
                        color=label_rgb,
                        overlay=True,
                    )
                rendered_any = True

                source_label = self._clean_text_for_render(row.get("label") or "")
                translated_label = self._clean_text_for_render(row.get("translated_label") or row.get("translated_text") or "")
                marker_overlay_safe = bool(source_label and translated_label and source_label == translated_label)
                marker_positions = [match.start() for match in re.finditer(r"[■•▪◦·]", label or "")]
                if marker_positions and marker_style_base:
                    marker_style = dict(marker_style_base)
                    marker_size = float(marker_style.get("size") or max(3.0, label_fontsize * 0.45))
                    marker_style["size"] = marker_size
                    marker_font_style, marker_fontfile, marker_builtin, marker_fontname = self._resolve_style_font(page, marker_style, text="■")
                    marker_rgb = self._resolve_text_color(marker_style, row)
                    for marker_pos in marker_positions:
                        try:
                            if marker_overlay_safe:
                                marker_bboxes = row.get("marker_bboxes") if isinstance(row.get("marker_bboxes"), list) else []
                                marker_rect = self._fitz_rect_from_bbox_like(marker_bboxes[0]) if marker_bboxes else None
                                x = marker_rect.x0 if isinstance(marker_rect, fitz.Rect) else label_rect.x0
                                baseline = marker_rect.y1 - max(0.1, marker_size * 0.12) if isinstance(marker_rect, fitz.Rect) else label_rect.y1 - max(1.0, label_fontsize * 0.18)
                            else:
                                prefix = label[:marker_pos]
                                x = label_rect.x0 + self._measure_text_width(prefix, label_fontsize, label_fontname, label_fontfile)
                                baseline = label_rect.y1 - max(1.0, label_fontsize * 0.18)
                            if x < label_rect.x0 - 1.0 or x > label_rect.x1 + 1.0:
                                continue
                            page.insert_text(
                                (x, baseline),
                                "■",
                                fontname=marker_fontname if marker_builtin or not marker_fontfile else marker_fontname,
                                fontfile=marker_fontfile if marker_fontfile and not marker_builtin else None,
                                fontsize=marker_size,
                                color=marker_rgb,
                                overlay=True,
                            )
                        except Exception:
                            pass

            if label and page_value:
                gap_left = min(page_rect_row.x0, row_bbox.x1)
                gap_right = max(label_rect.x1, row_bbox.x0)
                gap_width = float(page_rect_row.x0 - label_rect.x1)
                if gap_width > 8.0:
                    dot_count = max(3, int(gap_width / max(2.0, self._measure_text_width(".", label_fontsize, label_fontname, label_fontfile))))
                    leader_text = "." * min(64, dot_count)
                    leader_rect = fitz.Rect(
                        max(label_rect.x1 + 2.0, label_rect.x1),
                        min(label_rect.y0, page_rect_row.y0),
                        min(page_rect_row.x0 - 2.0, page_rect.x0 + page_rect.width),
                        max(label_rect.y1, page_rect_row.y1),
                    )
                    if leader_rect.width > 4.0:
                        try:
                            page.insert_textbox(
                                leader_rect,
                                leader_text,
                                fontname=label_fontname if label_builtin or not label_fontfile else label_fontname,
                                fontfile=label_fontfile if label_fontfile and not label_builtin else None,
                                fontsize=max(4.5, label_fontsize * 0.92),
                                color=label_rgb,
                                align=0,
                                overlay=True,
                            )
                        except Exception:
                            pass

            if page_value:
                try:
                    page.insert_textbox(
                        page_rect_row,
                        page_value,
                        fontname=page_fontname if page_builtin or not page_fontfile else page_fontname,
                        fontfile=page_fontfile if page_fontfile and not page_builtin else None,
                        fontsize=page_fontsize,
                        color=page_rgb,
                        align=2,
                        overlay=True,
                    )
                except Exception:
                    page.insert_text(
                        (page_rect_row.x1 - 4.0, page_rect_row.y1 - max(1.0, page_fontsize * 0.18)),
                        page_value,
                        fontname=page_fontname if page_builtin or not page_fontfile else page_fontname,
                        fontfile=page_fontfile if page_fontfile and not page_builtin else None,
                        fontsize=page_fontsize,
                        color=page_rgb,
                        overlay=True,
                    )
                rendered_any = True
        return rendered_any

    def _looks_like_toc_page(self, page_data):
        page_role = str((page_data or {}).get("page_role") or "").strip().lower()
        if page_role == "toc":
            return True
        toc = (page_data or {}).get("toc") or {}
        if isinstance(toc.get("toc_rows"), list) and toc.get("toc_rows"):
            return True
        blocks = [block for block in (page_data or {}).get("blocks") or [] if isinstance(block, dict)]
        if len(blocks) < 4:
            return False
        toc_hits = 0
        for block in blocks[:30]:
            text = self._clean_text_for_render(self._translated_text_from_block(block) or self._source_text_from_block(block))
            if not text:
                continue
            lower = text.lower()
            if "contents" in lower or "sommaire" in lower or "table of contents" in lower:
                toc_hits += 2
            if re.match(r"^\s*(?:\d+\.\d+|\d+|[ivxlcdm]+)\s+\S+", text, flags=re.I):
                toc_hits += 1
            if re.search(r"\.{2,}\s*(?:\d{1,4}|[ivxlcdm]+)\s*$", text, flags=re.I):
                toc_hits += 1
            if re.search(r"\b(?:\d{1,4}|[ivxlcdm]+)\s*$", text, flags=re.I):
                toc_hits += 1
        return toc_hits >= 3

    def _synthesized_toc_rows_from_blocks(self, page_data):
        blocks = [block for block in (page_data or {}).get("blocks") or [] if isinstance(block, dict)]
        rows = []

        def _text_from_line(line):
            return self._clean_text_for_render(
                (line or {}).get("translated_text")
                or (line or {}).get("line_translated_text")
                or (line or {}).get("line_text")
                or self._line_translated_text(line)
                or self._line_source_text(line)
                or ""
            )

        def _bbox_from_line(line):
            return self._fitz_rect_from_bbox_like((line or {}).get("bbox"))

        def _merge_rects(rects):
            rects = [r for r in rects if isinstance(r, fitz.Rect) and r.get_area() > 0]
            if not rects:
                return None
            out = fitz.Rect(rects[0])
            for rect in rects[1:]:
                out |= rect
            return out

        for block in blocks:
            lines = [ln for ln in (block.get("lines") or []) if isinstance(ln, dict)]
            if not lines:
                continue
            ordered = []
            for ln in lines:
                text = _text_from_line(ln)
                if not text:
                    continue
                ordered.append((ln, text))
            if not ordered:
                continue
            title_text = self._clean_text_for_render(self._translated_text_from_block(block) or self._source_text_from_block(block))
            if not title_text:
                continue

            numeric_pages = []
            label_parts = []
            label_rects = []
            page_rects = []
            for ln, text in ordered:
                if re.fullmatch(r"\s*(?:\d{1,4}|[ivxlcdm]+)\s*", text, flags=re.I):
                    numeric_pages.append(text.strip())
                    rect = _bbox_from_line(ln)
                    if rect is not None:
                        page_rects.append(rect)
                    continue
                if re.fullmatch(r"[\.\s·•\-–—_]+", text):
                    continue
                if re.search(r"\b(?:\d{1,4}|[ivxlcdm]+)\s*$", text, flags=re.I) and len(ordered) > 1:
                    m = re.search(r"(.*?)(\s*(?:\d{1,4}|[ivxlcdm]+)\s*)$", text, flags=re.I)
                    if m:
                        label_candidate = self._clean_text_for_render(m.group(1))
                        page_candidate = self._clean_text_for_render(m.group(2))
                        if label_candidate:
                            label_parts.append(label_candidate)
                            rect = _bbox_from_line(ln)
                            if rect is not None:
                                label_rects.append(rect)
                        if page_candidate:
                            numeric_pages.append(page_candidate)
                            rect = _bbox_from_line(ln)
                            if rect is not None:
                                page_rects.append(rect)
                        continue
                label_parts.append(text)
                rect = _bbox_from_line(ln)
                if rect is not None:
                    label_rects.append(rect)

            label_text = self._clean_text_for_render(" ".join(label_parts))
            if not label_text and numeric_pages:
                continue
            page_value = self._clean_text_for_render(numeric_pages[-1] if numeric_pages else "")
            if not page_value:
                m = re.search(r"(?:\s|^)(\d{1,4}|[ivxlcdm]+)\s*$", title_text, flags=re.I)
                if m:
                    page_value = m.group(1)
            marker = ""
            if label_text.startswith(("■", "•", "·", "-", "*")):
                marker, label_text = self._extract_leading_marker(label_text)
            role = "section_heading"
            lower = label_text.lower()
            if lower in {"contents", "sommaire", "table of contents"}:
                role = "toc_title"
            elif re.match(r"^\d+\.\d+\b", label_text):
                role = "section_heading"
            elif label_text.startswith(marker) or marker:
                role = "subentry_marker"
            elif len(label_parts) == 1 and page_value:
                role = "subentry"
            label_rect = _merge_rects(label_rects) or _merge_rects(page_rects)
            page_rect = _merge_rects(page_rects) or label_rect
            if label_rect is None and page_rect is None:
                continue
            if role == "toc_title" and not label_text:
                label_text = title_text
            rows.append(
                {
                    "role": role,
                    "label": label_text or title_text,
                    "page": page_value,
                    "marker": marker,
                    "label_bbox": [label_rect.x0, label_rect.y0, label_rect.x1, label_rect.y1] if isinstance(label_rect, fitz.Rect) else None,
                    "page_bbox": [page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1] if isinstance(page_rect, fitz.Rect) else None,
                    "style": self._style_from_block(block),
                    "page_style": self._style_from_block(block),
                    "translation_style": "professionnel",
                    "translation_tone": "neutre",
                }
            )

        if not rows:
            return []
        rows.sort(key=lambda row: ((row.get("label_bbox") or [0, 0, 0, 0])[1], (row.get("label_bbox") or [0, 0, 0, 0])[0]))
        title_rows = [row for row in rows if str(row.get("role") or "").lower() == "toc_title"]
        body_rows = [row for row in rows if str(row.get("role") or "").lower() != "toc_title"]
        return title_rows + body_rows

    def _render_page_debug_image(self, page, output_path, page_number):
        if not self.layout_debug_overlay:
            return
        debug_path = Path(output_path).with_name(f"{Path(output_path).stem}_layout_debug_p{page_number}.jpg")
        pix = page.get_pixmap(dpi=150, alpha=False)
        pix.save(str(debug_path))

    def _page_has_text_layer(self, page):
        try:
            return bool(self._clean_text_for_render(page.get_text("text") or ""))
        except Exception:
            return False

    def _presence_signature(self, text):
        text = self._clean_text_for_render(text or "").casefold()
        if not text:
            return ""
        return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)

    def _visible_text_in_rect(self, page, rect):
        if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
            return self._clean_text_for_render(page.get_text("text") or "")
        query_rect = fitz.Rect(rect)
        query_rect.x0 = max(float(page.rect.x0), query_rect.x0 - 1.5)
        query_rect.y0 = max(float(page.rect.y0), query_rect.y0 - 1.5)
        query_rect.x1 = min(float(page.rect.x1), query_rect.x1 + 1.5)
        query_rect.y1 = min(float(page.rect.y1), query_rect.y1 + 1.5)
        try:
            text = self._clean_text_for_render(page.get_textbox(query_rect) or "")
            if text:
                return text
        except Exception:
            pass
        words = []
        try:
            for word in page.get_text("words") or []:
                if len(word) < 5:
                    continue
                word_rect = fitz.Rect(word[:4])
                center = ((word_rect.x0 + word_rect.x1) / 2.0, (word_rect.y0 + word_rect.y1) / 2.0)
                if (word_rect & query_rect).get_area() > 0.1 or query_rect.contains(center):
                    words.append(str(word[4] or ""))
        except Exception:
            return ""
        return self._clean_text_for_render(" ".join(words))

    def _entry_text_present(self, page, expected_text, rect):
        expected = self._clean_text_for_render(expected_text or "")
        if not expected:
            return True
        region_text = self._visible_text_in_rect(page, rect)
        expected_sig = self._presence_signature(expected)
        region_sig = self._presence_signature(region_text)
        if expected_sig and expected_sig in region_sig:
            return True
        expected_fold = expected.casefold()
        region_fold = region_text.casefold()
        if expected_fold and expected_fold in region_fold:
            return True
        tokens = [self._presence_signature(token) for token in re.findall(r"\w+", expected_fold, flags=re.UNICODE)]
        tokens = [token for token in tokens if len(token) >= 3]
        if tokens:
            required = tokens if len(tokens) <= 4 else tokens[:2] + tokens[-2:]
            return all(token in region_sig for token in required)
        compact_expected = re.sub(r"\s+", "", expected_fold)
        compact_region = re.sub(r"\s+", "", region_fold)
        return bool(compact_expected and compact_expected in compact_region)

    def _entry_text_present_on_page(self, page, expected_text):
        expected = self._clean_text_for_render(expected_text or "")
        if not expected:
            return True
        page_text = self._clean_text_for_render(page.get_text("text") or "")
        expected_sig = self._presence_signature(expected)
        page_sig = self._presence_signature(page_text)
        if expected_sig and expected_sig in page_sig:
            return True
        tokens = [self._presence_signature(token) for token in re.findall(r"\w+", expected.casefold(), flags=re.UNICODE)]
        tokens = [token for token in tokens if len(token) >= 3]
        if tokens:
            required = tokens if len(tokens) <= 4 else tokens[:2] + tokens[-2:]
            return all(token in page_sig for token in required)
        return bool(re.sub(r"\s+", "", expected.casefold()) in re.sub(r"\s+", "", page_text.casefold()))

    def _entry_rect_for_block(self, block, entry):
        block_rect = self._fitz_rect_from_bbox_like((block or {}).get("rebalanced_bbox") or (block or {}).get("bbox"))
        bbox = (entry or {}).get("bbox")
        rect = self._fitz_rect_from_bbox_like(bbox)
        if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
            rect = block_rect
        if isinstance(rect, fitz.Rect) and isinstance(block_rect, fitz.Rect) and block_rect.get_area() > 0:
            intersection = rect & block_rect
            # Minimum usable area to render text (~5pt × 6pt); smaller intersections fall back to block rect
            if intersection.get_area() >= 30.0:
                rect = intersection
            elif rect.get_area() <= 0:
                rect = block_rect
            else:
                rect = block_rect
        return rect

    def _contract_entry_render_text(self, page, block, entry, rect, target_lang):
        text = self._clean_text_for_render((entry or {}).get("text") or "")
        source_text = self._clean_text_for_render((entry or {}).get("source_text") or "")
        contract = self._reconstruction_contract_for_block(block, page_data=None)
        key = str(contract.get("contract_key") or "").strip().lower()
        style = self._merge_styles((entry or {}).get("style") or {}, self._style_from_block(block))
        try:
            _, fontfile, _, fontname = self._resolve_style_font(page, style, text=text or source_text)
        except Exception:
            fontfile, fontname = None, "helv"
        fontsize = float(style.get("size") or 12.0)
        strict = bool(contract.get("strict_non_reflow")) or key in {"table_cell_micro", "table_cell_symbolic", "table_cell_numeric", "url_reference", "formula_block", "inline_formula"}
        preserve_on_overflow = bool(contract.get("preserve_if_translation_overflows") or contract.get("translation_policy") == "preserve")
        if strict and source_text:
            translated_width = self._measure_text_width(text, fontsize, fontname, fontfile) if text else 0.0
            source_width = self._measure_text_width(source_text, fontsize, fontname, fontfile)
            if preserve_on_overflow or (translated_width > max(4.0, rect.width * 1.02) and source_width <= max(translated_width, rect.width * 1.08)):
                text = source_text
        return text or source_text

    def _direct_text_ops_for_entry(self, page, page_data, block, entry, target_lang):
        rect = self._entry_rect_for_block(block, entry)
        if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
            return []
        contract = self._reconstruction_contract_for_block(block, page_data=page_data)
        contract_key = str(contract.get("contract_key") or "").strip().lower()
        text = self._contract_entry_render_text(page, block, entry, rect, target_lang)
        text = self._clean_text_for_render(text)
        if not text:
            return []
        style = self._merge_styles((entry or {}).get("style") or {}, self._style_from_block(block))
        source_fontsize = float(style.get("size") or 12.0)
        min_ratio = float(contract.get("min_font_ratio") or 0.90)
        min_fontsize = max(4.0, source_fontsize * min_ratio)
        fontsize = min(source_fontsize, max(min_fontsize, rect.height * 0.86))
        try:
            _, fontfile, builtin, fontname = self._resolve_style_font(page, style, text=text)
        except Exception:
            fontfile, builtin, fontname = None, True, "helv"
        rgb = self._resolve_text_color(style, block)
        strict_single_line = bool(contract.get("strict_non_reflow")) or contract_key in {"toc_entry", "table_cell_micro", "table_cell_symbolic", "table_cell_numeric", "url_reference", "formula_block", "inline_formula", "figure_label"}
        lines = [text] if strict_single_line else self._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(6.0, rect.width))
        if not lines:
            lines = [text]
        while fontsize > min_fontsize and lines:
            too_wide = any(self._measure_text_width(line, fontsize, fontname, fontfile) > max(5.0, rect.width * 0.98) for line in lines)
            too_tall = len(lines) * max(4.5, fontsize * 1.05) > max(rect.height, fontsize * 1.1)
            if not too_wide and not too_tall:
                break
            fontsize -= 0.35
            if not strict_single_line:
                lines = self._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(6.0, rect.width)) or [text]
        line_h = max(4.5, min(rect.height / max(1, len(lines)), fontsize * 1.06))
        align = self._normalize_alignment((entry or {}).get("alignment") or (block or {}).get("alignment") or "left")
        ops = []
        for line_idx, line_text in enumerate(lines):
            line_text = self._clean_text_for_render(line_text)
            if not line_text:
                continue
            width = self._measure_text_width(line_text, fontsize, fontname, fontfile)
            x = rect.x0
            if align == "center":
                x = max(rect.x0, rect.x0 + max(0.0, (rect.width - width) / 2.0))
            elif align == "right":
                x = max(rect.x0, rect.x1 - width)
            baseline = min(rect.y1 - 0.8, rect.y0 + min(rect.height - 0.8, (line_idx + 1) * line_h * 0.82))
            text_rect = fitz.Rect(
                max(rect.x0, x),
                baseline - max(1.0, fontsize * 0.82),
                min(rect.x1, max(rect.x0, x) + max(1.0, min(width, rect.width))),
                baseline + max(1.0, fontsize * 0.18),
            )
            ops.append(
                BlockRenderOp(
                    op_type="draw_text_run",
                    block_id=str((block or {}).get("id") or "coverage"),
                    unit_id=f"{(entry or {}).get('unit_id') or 'coverage'}:direct:{line_idx}",
                    bbox=(text_rect.x0, text_rect.y0, text_rect.x1, text_rect.y1),
                    text=line_text,
                    style={**style, "size": fontsize},
                    z_index=120,
                    metadata={
                        "point": (max(rect.x0, x), baseline),
                        "fontname": fontname,
                        "fontfile": fontfile,
                        "builtin": builtin,
                        "fontsize": fontsize,
                        "source_fontsize": source_fontsize,
                        "min_font_ratio": min_ratio,
                        "rgb": rgb,
                    },
                )
            )
        return ops

    def _enforce_page_block_text_coverage(self, page, page_data, target_lang):
        missing_after_rescue = []
        rescue_ops = []
        for block in self._iter_renderable_blocks(page_data):
            entries = self._translated_coverage_entries_for_block(block, target_lang, page_data=page_data)
            if not entries:
                continue
            for entry in entries:
                expected = self._clean_text_for_render((entry or {}).get("text") or (entry or {}).get("source_text") or "")
                if not expected:
                    continue
                rect = self._entry_rect_for_block(block, entry)
                if self._entry_text_present(page, expected, rect):
                    continue
                if self._entry_text_present_on_page(page, expected):
                    continue
                rescue_ops.extend(self._direct_text_ops_for_entry(page, page_data, block, entry, target_lang))
        if rescue_ops:
            self._commit_block_draw_ops(page, rescue_ops)
        for block in self._iter_renderable_blocks(page_data):
            for entry in self._translated_coverage_entries_for_block(block, target_lang, page_data=page_data):
                expected = self._clean_text_for_render((entry or {}).get("text") or (entry or {}).get("source_text") or "")
                if not expected:
                    continue
                rect = self._entry_rect_for_block(block, entry)
                if not self._entry_text_present(page, expected, rect):
                    if self._entry_text_present_on_page(page, expected):
                        continue
                    missing_after_rescue.append(
                        {
                            "block_id": str((block or {}).get("id") or ""),
                            "unit_id": str((entry or {}).get("unit_id") or ""),
                            "text": expected[:120],
                        }
                    )
        if missing_after_rescue:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "block coverage incomplete after rescue: page=%s missing=%d entries=%s",
                int(getattr(page, "number", 0)) + 1,
                len(missing_after_rescue),
                [m.get("text", "")[:40] for m in missing_after_rescue[:3]],
            )
        return bool(rescue_ops)

    def _page_requires_text_layer(self, page_data, target_lang):
        for block in self._iter_renderable_blocks(page_data):
            text = self._clean_text_for_render(self._translated_text_from_block(block) or self._source_text_from_block(block))
            if text:
                return True
            try:
                if self._expected_block_text_units(block, target_lang, page_data=page_data) > 0:
                    return True
            except Exception:
                pass
        toc_rows = list(((page_data or {}).get("toc") or {}).get("toc_rows") or [])
        if not toc_rows and self._looks_like_toc_page(page_data):
            toc_rows = self._synthesized_toc_rows_from_blocks(page_data)
        for row in toc_rows:
            if self._clean_text_for_render((row or {}).get("label") or (row or {}).get("title") or (row or {}).get("page") or ""):
                return True
        return False

    def _render_page_text_rescue(self, page, page_data, target_lang):
        blocks = list(self._iter_renderable_blocks(page_data))
        if not blocks:
            return False
        ops = []
        for block in blocks:
            text = self._clean_text_for_render(self._translated_text_from_block(block) or self._source_text_from_block(block))
            if not text:
                continue
            rect = self._fitz_rect_from_bbox_like((block or {}).get("rebalanced_bbox") or (block or {}).get("bbox"))
            if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                continue
            style = self._style_from_block(block)
            style = dict(style or {})
            try:
                _, fontfile, builtin, fontname = self._resolve_style_font(page, style, text=text)
            except Exception:
                fontfile, builtin, fontname = None, True, "helv"
            source_fontsize = float(style.get("size") or 11.0)
            fontsize = min(source_fontsize, max(4.5, rect.height * 0.78))
            wrapped = self._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            while fontsize > 4.5 and wrapped and (len(wrapped) * max(5.0, fontsize * 1.05)) > max(rect.height, fontsize * 1.2):
                fontsize -= 0.5
                wrapped = self._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            if not wrapped:
                wrapped = [text]
            line_h = max(4.8, fontsize * 1.06)
            align = self._normalize_alignment((block or {}).get("alignment") or "left")
            rgb = self._resolve_text_color(style, block)
            for line_idx, line_text in enumerate(wrapped):
                width = self._measure_text_width(line_text, fontsize, fontname, fontfile)
                x = rect.x0
                if align == "center":
                    x = max(rect.x0, rect.x0 + max(0.0, (rect.width - width) / 2.0))
                elif align == "right":
                    x = max(rect.x0, rect.x1 - width)
                baseline = min(rect.y1 - 1.0, rect.y0 + min(rect.height - 1.0, (line_idx + 1) * line_h * 0.82))
                text_rect = fitz.Rect(
                    x,
                    baseline - max(1.0, fontsize * 0.82),
                    min(rect.x1, x + width),
                    baseline + max(1.0, fontsize * 0.18),
                )
                ops.append(
                    BlockRenderOp(
                        op_type="draw_text_run",
                        block_id=str((block or {}).get("id") or "rescue"),
                        unit_id=f"{(block or {}).get('id') or 'rescue'}:{line_idx}",
                        bbox=(text_rect.x0, text_rect.y0, text_rect.x1, text_rect.y1),
                        text=line_text,
                        style={**style, "size": fontsize, "_source_size": source_fontsize, "_min_font_ratio": 0.85},
                        z_index=100,
                        metadata={
                            "point": (x, baseline),
                            "fontname": fontname,
                            "fontfile": fontfile,
                            "builtin": builtin,
                            "fontsize": fontsize,
                            "source_fontsize": source_fontsize,
                            "min_font_ratio": 0.85,
                            "rgb": rgb,
                        },
                    )
                )
        if not ops:
            return False
        self._commit_block_draw_ops(page, ops)
        return self._page_has_text_layer(page)

    def _page_target_lang(self, structure, page_data):
        return str(
            (page_data or {}).get("target_lang")
            or (page_data or {}).get("translation_target_lang")
            or (structure or {}).get("target_lang")
            or "fr"
        ).strip().lower()

    _BLOCKING_RENDER_FINDINGS = {
        "overflow",
        "text_overlap",
        "protected_overlap",
        "composition_overflow",
        "font_too_small",
        "text_missing",
        "protected_missing",
        "source_overlay",
        "geometry_drift",
    }

    _RENDER_FIX_STRATEGY = {
        "overflow": "expanded_bbox_or_reflow",
        "composition_overflow": "expanded_bbox_or_reflow",
        "text_overlap": "rerender_with_collision_avoidance",
        "protected_overlap": "anchored_protected_token_fallback",
        "font_too_small": "expand_block_before_shrink",
        "text_missing": "rerender_missing_text_in_origin_block",
        "protected_missing": "exact_preserve_anchored",
        "source_overlay": "repatch_background_then_rerender",
        "geometry_drift": "restore_origin_geometry_contract",
    }

    def _render_finding_severity(self, finding_type):
        finding_type = str(finding_type or "").strip().lower()
        if finding_type in self._BLOCKING_RENDER_FINDINGS:
            return "failed"
        return "warning"

    def _block_render_verdict(self, plan, ops, findings=None):
        findings_out = []
        seen = set()

        def add_finding(finding):
            if not isinstance(finding, dict):
                return
            item = dict(finding)
            ftype = str(item.get("type") or "").strip().lower() or "unknown"
            item["type"] = ftype
            item.setdefault("severity", self._render_finding_severity(ftype))
            item.setdefault("recommended_strategy", self._RENDER_FIX_STRATEGY.get(ftype, "block_contract_fallback"))
            bbox_values = item.get("bbox") if isinstance(item.get("bbox"), (list, tuple)) else []
            key = (
                ftype,
                str(item.get("unit_id") or ""),
                tuple(round(float(v), 2) for v in bbox_values if isinstance(v, (int, float))),
            )
            if key in seen:
                return
            seen.add(key)
            findings_out.append(item)

        for finding in findings or []:
            add_finding(finding)

        expected_text_ops = self._expected_plan_text_ops(plan)
        rendered_text_ops = sum(
            1
            for op in (ops or [])
            if str(getattr(op, "op_type", "") or "").startswith("draw_text")
            and self._clean_text_for_render(getattr(op, "text", "") or "")
        )
        if expected_text_ops > 0 and rendered_text_ops < expected_text_ops:
            add_finding(
                {
                    "type": "text_missing",
                    "block_id": getattr(plan, "block_id", ""),
                    "text_ops_expected": expected_text_ops,
                    "text_ops_rendered": rendered_text_ops,
                }
            )

        causes = sorted({str(f.get("type") or "").strip().lower() for f in findings_out if f.get("type")})
        has_failed = any(str(f.get("severity") or "").strip().lower() == "failed" for f in findings_out)
        status = "failed" if has_failed else ("warning" if findings_out else "ok")
        recommended = ""
        for cause in causes:
            if cause in self._RENDER_FIX_STRATEGY:
                recommended = self._RENDER_FIX_STRATEGY[cause]
                break
        return BlockRenderVerdict(
            status=status,
            ok=status == "ok",
            block_id=str(getattr(plan, "block_id", "") or ""),
            causes=causes,
            findings=findings_out,
            text_ops_expected=expected_text_ops,
            text_ops_rendered=rendered_text_ops,
            recommended_strategy=recommended,
        )

    def _block_render_verdict_dict(self, plan, ops, findings=None):
        return asdict(self._block_render_verdict(plan, ops, findings=findings))

    _CANDIDATE_PENALTIES = {
        "text_missing": math.inf,
        "protected_missing": math.inf,
        "source_overlay": math.inf,
        "overflow": 1000.0,
        "composition_overflow": 1000.0,
        "font_too_small": 900.0,
        "text_overlap": 900.0,
        "protected_overlap": 900.0,
        "style_lost": 800.0,
        "geometry_drift": 500.0,
    }

    def _score_render_candidate(self, plan, candidate):
        if not isinstance(candidate, RenderCandidate):
            candidate = RenderCandidate(
                candidate_id=str(getattr(candidate, "candidate_id", "") or "candidate"),
                strategy=str(getattr(candidate, "strategy", "") or "unknown"),
                ops=list(getattr(candidate, "ops", []) or []),
                findings=list(getattr(candidate, "findings", []) or []),
                metadata=dict(getattr(candidate, "metadata", {}) or {}),
            )
        verdict = self._block_render_verdict(plan, candidate.ops, candidate.findings)
        penalties = {}
        hard_failures = []
        total = 0.0
        for cause in verdict.causes:
            penalty = self._CANDIDATE_PENALTIES.get(cause, 50.0)
            penalties[cause] = penalty
            if math.isinf(penalty):
                hard_failures.append(cause)
            else:
                total += penalty
        expected = max(1, int(verdict.text_ops_expected or 0))
        rendered = int(verdict.text_ops_rendered or 0)
        if rendered < expected:
            missing_penalty = (expected - rendered) * 1000.0
            penalties["missing_text_ops"] = missing_penalty
            total += missing_penalty
            if "text_missing" not in hard_failures:
                hard_failures.append("text_missing")
        font_ratios = []
        for op in candidate.ops or []:
            metadata = dict(getattr(op, "metadata", {}) or {})
            try:
                rendered_size = float(metadata.get("fontsize") or 0.0)
                source_size = float(metadata.get("source_fontsize") or 0.0)
            except Exception:
                continue
            if rendered_size > 0 and source_size > 0:
                font_ratios.append(rendered_size / source_size)
        if font_ratios:
            min_ratio = min(font_ratios)
            contract_min = float(((getattr(plan, "constraints", {}) or {}).get("min_font_ratio") or 0.0) or 0.0)
            if contract_min and min_ratio < contract_min:
                penalty = (contract_min - min_ratio) * 1000.0
                penalties["font_ratio_under_contract"] = penalty
                total += penalty
        status = "failed" if hard_failures or verdict.status == "failed" else ("warning" if verdict.status == "warning" else "ok")
        return CandidateScore(value=math.inf if hard_failures else total, status=status, penalties=penalties, hard_failures=hard_failures)

    def select_best_candidate(self, plan, candidates):
        scored = []
        for idx, candidate in enumerate(candidates or []):
            if not isinstance(candidate, RenderCandidate):
                continue
            candidate.score = self._score_render_candidate(plan, candidate)
            scored.append((candidate.score.value, idx, candidate))
        if not scored:
            return None
        scored.sort(key=lambda item: (math.isinf(item[0]), item[0], item[1]))
        return scored[0][2]

    def _render_plan_with_validation(self, page, plan):
        ops = self._render_hierarchical_block_plan(page, plan)
        findings = self._validate_block_layout(plan, ops)
        verdict = self._block_render_verdict(plan, ops, findings=findings)
        findings = verdict.findings
        if verdict.status == "failed":
            candidates = [
                RenderCandidate(
                    candidate_id=f"{getattr(plan, 'block_id', 'block')}::primary",
                    strategy=str((getattr(plan, "constraints", {}) or {}).get("contract_render_mode") or "primary"),
                    ops=ops,
                    findings=findings,
                    metadata={"candidate_source": "primary"},
                )
            ]
            pruned_ops = self._prune_block_draw_ops(plan, ops)
            if pruned_ops:
                pruned_findings = self._validate_block_layout(plan, pruned_ops)
                pruned_verdict = self._block_render_verdict(plan, pruned_ops, findings=pruned_findings)
                candidates.append(
                    RenderCandidate(
                        candidate_id=f"{getattr(plan, 'block_id', 'block')}::pruned",
                        strategy="pruned_overflow_candidate",
                        ops=pruned_ops,
                        findings=pruned_verdict.findings,
                        metadata={"candidate_source": "pruned"},
                    )
                )
            best = self.select_best_candidate(plan, candidates)
            if best is not None and best.score is not None and best.score.status != "failed":
                return best.ops, best.findings
            return [], findings
        return ops, findings

    def _translated_coverage_entries_for_block(self, block, target_lang, page_data=None):
        semantic_payload = self._collect_block_semantic_payload(block)
        units = self._normalize_placable_units(block, semantic_payload, target_lang, page_data=page_data)
        units = self._apply_source_layout_mode_to_units(block, units)
        units = self._finalize_placable_units(block, units, page_data=page_data)
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

        for unit in list(units or []):
            if unit.unit_id in used_ids:
                continue
            contract = str(((unit.metadata or {}).get("canonical_render_contract") or unit.render_policy or "")).strip().lower()
            unit_class = str(((unit.metadata or {}).get("canonical_unit_class") or "")).strip().lower()
            if contract == "source_overlay" or unit_class == "immutable_overlay":
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

        block_text = self._clean_text_for_render((block or {}).get("translated_text") or "")
        block_source_text = self._source_text_from_block(block) or block_text
        contract = self._reconstruction_contract_for_block(block, page_data=page_data)
        if block_text and not bool(contract.get("strict_non_reflow")):
            return [
                {
                    "unit_id": f"{(block or {}).get('id') or 'block'}:coverage:block",
                    "text": block_text,
                    "source_text": block_source_text,
                    "bbox": (block or {}).get("rebalanced_bbox") or (block or {}).get("bbox"),
                    "style": self._style_from_block(block),
                    "unit_type": "block",
                    "line_indices": [],
                    "render_policy": str((block or {}).get("render_policy") or ""),
                }
            ]

        line_entries = []
        for idx, line in enumerate((block or {}).get("lines") or []):
            text = self._line_translated_text(line)
            if not text:
                continue
            source_text = self._line_source_text(line) or text
            line_entries.append(
                {
                    "unit_id": f"{(block or {}).get('id') or 'block'}:coverage:{idx}",
                    "text": text,
                    "source_text": source_text,
                    "bbox": (line or {}).get("bbox"),
                    "style": self._merge_styles((line or {}).get("style") or {}, self._style_from_block(block)),
                    "unit_type": "line",
                    "line_indices": [idx],
                    "render_policy": str((block or {}).get("render_policy") or ""),
                }
            )
        if line_entries:
            return line_entries

        if block_text:
            return [
                {
                    "unit_id": f"{(block or {}).get('id') or 'block'}:coverage:block",
                    "text": block_text,
                    "source_text": block_source_text,
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

    def _expected_plan_text_ops(self, plan):
        text_units = [
            unit for unit in (getattr(plan, "units", None) or [])
            if self._clean_text_for_render(getattr(unit, "text_translated", "") or getattr(unit, "text_source", ""))
        ]
        if not text_units:
            return 0
        render_mode = str((getattr(plan, "constraints", {}) or {}).get("contract_render_mode") or "").strip().lower()
        if render_mode == "prose_reflow":
            return 1
        return len(text_units)

    def _render_block_presence_fallback_ops(self, page, page_data, block, target_lang, font_scale=1.0):
        block_rect = self._fitz_rect_from_bbox_like((block or {}).get("rebalanced_bbox") or (block or {}).get("bbox"))
        if not isinstance(block_rect, fitz.Rect) or block_rect.get_area() <= 0:
            return []
        adaptive_profile = self._block_adaptive_profile(block, page_data=page_data)
        contract = self._reconstruction_contract_for_block(block, page_data=page_data)
        contract_key = str(contract.get("contract_key") or "").strip().lower()
        min_font_ratio = float(contract.get("min_font_ratio") or 0.90)
        object_type = str((block or {}).get("object_type") or "").strip().lower()
        block_role = str((block or {}).get("role") or "").strip().lower()
        block_text = self._clean_text_for_render((block or {}).get("translated_text") or "")
        if not block_text:
            block_text = self._clean_text_for_render(self._translated_text_from_block(block))
        if object_type == "figure_caption" or block_role == "figure_caption":
            caption_lines = list((block or {}).get("lines") or [])
            line_rects = [
                self._fitz_rect_from_bbox_like((line or {}).get("bbox"))
                for line in caption_lines
                if self._fitz_rect_from_bbox_like((line or {}).get("bbox")) is not None
            ]
            same_visual_line = False
            if len(line_rects) >= 2:
                same_visual_line = (max(rect.y0 for rect in line_rects) - min(rect.y0 for rect in line_rects)) <= 1.5
            entries = []
            if same_visual_line and block_text:
                entries = [
                    {
                        "unit_id": f"{(block or {}).get('id') or 'block'}:coverage:caption",
                        "text": block_text,
                        "bbox": None,
                        "style": self._style_from_block(block),
                        "unit_type": "block",
                        "line_indices": [],
                        "render_policy": str((block or {}).get("render_policy") or ""),
                        "alignment": (block or {}).get("alignment") or "left",
                    }
                ]
            else:
                for idx, line in enumerate(caption_lines):
                    line_text = self._line_translated_text(line)
                    if not line_text:
                        continue
                    entries.append(
                        {
                            "unit_id": f"{(block or {}).get('id') or 'block'}:coverage:caption:{idx}",
                            "text": line_text,
                            "bbox": None,
                            "style": self._merge_styles((line or {}).get("style") or {}, self._style_from_block(block)),
                            "unit_type": "line",
                            "line_indices": [idx],
                            "render_policy": str((block or {}).get("render_policy") or ""),
                            "alignment": (block or {}).get("alignment") or "left",
                        }
                    )
                if not entries and block_text:
                    entries = [
                        {
                            "unit_id": f"{(block or {}).get('id') or 'block'}:coverage:caption",
                            "text": block_text,
                            "bbox": None,
                            "style": self._style_from_block(block),
                            "unit_type": "block",
                            "line_indices": [],
                            "render_policy": str((block or {}).get("render_policy") or ""),
                            "alignment": (block or {}).get("alignment") or "left",
                        }
                    ]
        else:
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
            if contract.get("translation_policy") == "preserve" and self._clean_text_for_render(entry.get("source_text") or ""):
                text = self._clean_text_for_render(entry.get("source_text") or "")
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
            source_fontsize = float(style.get("size") or 12.0)
            target_fontsize = source_fontsize * max(0.4, float(font_scale))
            if contract_key in {"table_cell_micro", "table_cell_symbolic", "table_cell_numeric", "figure_label", "url_reference", "formula_block", "inline_formula"}:
                source_text = self._clean_text_for_render(entry.get("source_text") or "")
                translated_width = self._measure_text_width(text, source_fontsize, fontname, fontfile)
                source_width = self._measure_text_width(source_text, source_fontsize, fontname, fontfile) if source_text else translated_width
                if source_text and (contract.get("preserve_if_translation_overflows") or translated_width > max(4.0, rect.width * 1.02)) and source_width <= max(translated_width, rect.width * 1.08):
                    text = source_text
            rect = self._expand_text_rect_within_block(rect, block_rect, target_fontsize, line_count=1, line_height_factor=1.12)
            fontsize = min(target_fontsize, max(4.5, rect.height * 0.90))
            if bool(contract.get("strict_non_reflow")):
                wrapped = [text]
            else:
                wrapped = self._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            if wrapped:
                rect = self._expand_text_rect_within_block(
                    rect,
                    block_rect,
                    fontsize,
                    line_count=len(wrapped),
                    line_height_factor=1.08 * max(0.84, min(1.05, float(adaptive_profile.get("line_spacing_factor") or 1.0))),
                )
            minimum_fontsize = max(4.5, source_fontsize * min_font_ratio)
            while fontsize > minimum_fontsize and wrapped and (len(wrapped) * max(4.8, fontsize * 1.05)) > max(rect.height, fontsize * 1.15):
                fontsize -= 0.5
                if bool(contract.get("strict_non_reflow")):
                    wrapped = [text]
                else:
                    wrapped = self._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            spacing_factor = max(0.84, min(1.05, float(adaptive_profile.get("line_spacing_factor") or 1.0)))
            line_h = max(4.8, min(rect.height / max(1, len(wrapped)), fontsize * 1.08 * spacing_factor))
            rgb = self._resolve_text_color( style, block)
            for line_idx, line_text in enumerate(wrapped):
                cur_size = fontsize
                while cur_size > minimum_fontsize:
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
                            "source_fontsize": source_fontsize,
                            "min_font_ratio": min_font_ratio,
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
                if isinstance(page_data, dict):
                    try:
                        self._augment_page_data_from_layout_xml(page_data)
                    except Exception:
                        pass
                    try:
                        self._inject_dynamic_immutable_overlays(page_data)
                    except Exception:
                        pass
                    try:
                        page_data = self._rebalance_page_layout(None, page_data, target_lang=self._page_target_lang(structure, page_data))
                        pages[page_index] = page_data
                    except Exception:
                        pass
                width_pt, height_pt = self._page_size_pt(page_data)
                page = doc.new_page(width=width_pt, height=height_pt)
                self._insert_page_background(page, page_data)
                target_lang = self._page_target_lang(structure, page_data)
                self._build_page_reconstruction_context(page_data, target_lang)
                toc_rows = list(((page_data or {}).get("toc") or {}).get("toc_rows") or [])
                if not toc_rows and self._looks_like_toc_page(page_data):
                    toc_rows = self._synthesized_toc_rows_from_blocks(page_data)
                if toc_rows:
                    try:
                        toc_page_data = dict(page_data or {})
                        toc_page_data["page_role"] = "toc"
                        toc_page_data["toc"] = dict((page_data or {}).get("toc") or {})
                        toc_page_data["toc"]["toc_rows"] = toc_rows
                        if self._render_toc_page_rows(page, toc_page_data, target_lang):
                            remaining_overlays = self._remaining_page_immutable_overlays(page_data, set())
                            if remaining_overlays:
                                final_page_data = dict(page_data or {})
                                final_page_data["immutable_overlays"] = remaining_overlays
                                try:
                                    self._insert_immutable_overlays(page, final_page_data)
                                except Exception:
                                    pass
                            if self._page_requires_text_layer(toc_page_data, target_lang) and not self._page_has_text_layer(page):
                                try:
                                    self._render_page_text_rescue(page, toc_page_data, target_lang)
                                except Exception:
                                    pass
                            if self._page_requires_text_layer(toc_page_data, target_lang):
                                self._enforce_page_block_text_coverage(page, toc_page_data, target_lang)
                            if self._page_requires_text_layer(toc_page_data, target_lang) and not self._page_has_text_layer(page):
                                raise RuntimeError(f"reconstruction produced image-only page with expected text: page={page_index + 1}")
                            self._render_page_debug_image(page, output_path, page_index + 1)
                            continue
                    except Exception:
                        pass
                rendered_block_ids = set()
                rendered_block_stats = {}
                rendered_overlay_signatures = set()
                rendered_background_regions = set()
                for block in self._iter_renderable_blocks(page_data):
                    if not self._block_supported_by_hierarchical_engine(block, page_data):
                        rendered_block_stats[str((block or {}).get("id") or "")] = {
                            "committed": False,
                            "text_ops": 0,
                            "expected_units": self._expected_block_text_units(block, target_lang, page_data=page_data),
                            "contract_driven": False,
                            "used_presence_fallback": False,
                            "block": block,
                        }
                        continue
                    plan = self._build_block_reconstruction_plan(page, page_data, block, target_lang)
                    ops, findings = self._render_plan_with_validation(page, plan)
                    text_ops = sum(1 for op in ops if op.op_type == "draw_text_run")
                    text_chars = sum(len((op.text or "").strip()) for op in ops if op.op_type == "draw_text_run")
                    contract_driven = bool((plan.constraints or {}).get("contract_driven"))
                    expected_units = self._expected_plan_text_ops(plan) if contract_driven else self._expected_block_text_units(block, target_lang, page_data=page_data)
                    used_presence_fallback = False
                    if (findings and not ops) or (expected_units > 0 and text_ops < expected_units):
                        fallback_ops = self._validated_block_presence_fallback_ops(page, page_data, block, target_lang, plan=plan)
                        fallback_text_ops = sum(1 for op in fallback_ops if op.op_type == "draw_text_run")
                        fallback_text_chars = sum(len((op.text or "").strip()) for op in fallback_ops if op.op_type == "draw_text_run")
                        if fallback_text_ops > text_ops or (fallback_text_ops == text_ops and fallback_text_chars > text_chars):
                            ops = fallback_ops
                            text_ops = fallback_text_ops
                            text_chars = fallback_text_chars
                            used_presence_fallback = True
                            findings = []
                    if findings and not ops:
                        rendered_block_stats[plan.block_id] = {
                            "committed": False,
                            "text_ops": 0,
                            "expected_units": expected_units,
                            "contract_driven": contract_driven,
                            "used_presence_fallback": used_presence_fallback,
                            "block": block,
                        }
                        continue
                    ops = self._filter_redundant_background_region_ops(plan, ops, rendered_background_regions)
                    for op in ops:
                        if op.op_type != "draw_overlay_image":
                            continue
                        sig = self._overlay_signature({
                            "path": (op.metadata or {}).get("path"),
                            "bbox": fitz.Rect(op.bbox) if isinstance(op.bbox, (list, tuple)) and len(op.bbox) == 4 else op.bbox,
                            "kind": "draw_overlay_image",
                        })
                        if sig:
                            rendered_overlay_signatures.add(sig)
                    self._commit_block_draw_ops(page, ops)
                    rendered_block_ids.add(plan.block_id)
                    rendered_block_stats[plan.block_id] = {
                        "committed": True,
                        "text_ops": text_ops,
                        "expected_units": expected_units,
                        "contract_driven": contract_driven,
                        "used_presence_fallback": used_presence_fallback,
                        "block": block,
                    }
                for block in self._iter_renderable_blocks(page_data):
                    block_id = str((block or {}).get("id") or "")
                    stats = rendered_block_stats.get(block_id) or {}
                    expected_units = int(stats.get("expected_units") or self._expected_block_text_units(block, target_lang, page_data=page_data))
                    committed = bool(stats.get("committed"))
                    text_ops = int(stats.get("text_ops") or 0)
                    contract_driven = bool(stats.get("contract_driven"))
                    used_presence_fallback = bool(stats.get("used_presence_fallback"))
                    if expected_units <= 0:
                        continue
                    if committed and used_presence_fallback:
                        continue
                    if committed and text_ops >= expected_units:
                        continue
                    fallback_ops = self._validated_block_presence_fallback_ops(page, page_data, block, target_lang)
                    fallback_text_ops = sum(1 for op in fallback_ops if op.op_type == "draw_text_run")
                    current_text_chars = 0
                    fallback_text_chars = sum(len((op.text or "").strip()) for op in fallback_ops if op.op_type == "draw_text_run")
                    if fallback_text_ops > text_ops or (fallback_text_ops == text_ops and fallback_text_chars > current_text_chars):
                        self._commit_block_draw_ops(page, fallback_ops)
                remaining_overlays = self._remaining_page_immutable_overlays(page_data, rendered_overlay_signatures)
                if remaining_overlays:
                    final_page_data = dict(page_data or {})
                    final_page_data["immutable_overlays"] = remaining_overlays
                    try:
                        self._insert_immutable_overlays(page, final_page_data)
                    except Exception:
                        pass
                requires_text_layer = self._page_requires_text_layer(page_data, target_lang)
                if requires_text_layer and not self._page_has_text_layer(page):
                    try:
                        self._render_page_text_rescue(page, page_data, target_lang)
                    except Exception:
                        pass
                if requires_text_layer:
                    self._enforce_page_block_text_coverage(page, page_data, target_lang)
                if requires_text_layer and not self._page_has_text_layer(page):
                    raise RuntimeError(f"reconstruction produced image-only page with expected text: page={page_index + 1}")
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
        resolved, fontfile, builtin, fontname = self.reconstructor._resolve_style_font( page, style, text=text)
        if isinstance(resolved, dict):
            style = {
                **style,
                "_resolved_fontname": fontname,
                "_resolved_fontfile": fontfile,
                "_font_substitution_reason": resolved.get("font_substitution_reason") or "",
                "_unicode_fallback": bool(resolved.get("unicode_fallback")),
                "_document_level_font_fallback": bool(resolved.get("document_level_font_fallback")),
            }
        fontsize = float(style.get("size") or 12.0)
        rgb = self.reconstructor._resolve_text_color( style, plan.source_block)
        return style, fontfile, builtin, fontname, fontsize, rgb

    def _unit_rotation_deg(self, unit=None, template=None):
        if template is not None:
            try:
                value = int(getattr(template, "rotation_deg", 0) or 0) % 360
            except Exception:
                value = 0
            if value in {90, 180, 270}:
                return value
        metadata = dict(getattr(unit, "metadata", {}) or {}) if unit is not None else {}
        try:
            value = int(round(float(metadata.get("rotation_deg") or 0.0))) % 360
        except Exception:
            value = 0
        if value in {90, 180, 270}:
            return value
        if unit is not None:
            return self.reconstructor._rotation_deg_for_bbox_text(
                getattr(unit, "relative_bbox", None),
                getattr(unit, "text_translated", "") or getattr(unit, "text_source", ""),
                fallback=0,
            )
        return 0

    def _emit_rotated_textbox_run(self, page, plan, text, rect, style, fontname, fontfile, builtin, rgb, *, rotation_deg, align, unit_id, min_fontsize=5.0):
        rect = fitz.Rect(rect)
        if rect.get_area() <= 0:
            return None
        fontsize = min(float(style.get("size") or 12.0), max(min_fontsize, rect.width * 0.92))
        primary_limit = max(8.0, rect.height - 1.0)
        while fontsize > min_fontsize and (
            self.reconstructor._measure_text_width(text, fontsize, fontname, fontfile) > primary_limit
            or fontsize > rect.width * 0.98
        ):
            fontsize -= 0.5
        if fontsize < min_fontsize:
            fontsize = min_fontsize
        rotated_style = {**style, "size": fontsize, "_rotation_deg": rotation_deg, "_textbox_align": align}
        return self._emit_text_run(
            plan,
            text,
            rect,
            (rect.x0, rect.y1),
            rotated_style,
            fontname,
            fontfile,
            builtin,
            fontsize,
            rgb,
            unit_id=unit_id,
        )

    def _fit_text_run_to_block(self, plan, rect, point, fontsize):
        block_rect = self.reconstructor._line_preserve_effective_rect(plan)
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
        rotation_deg = int(round(float((style or {}).get("_rotation_deg") or 0.0))) % 360
        textbox_align = str((style or {}).get("_textbox_align") or "left").strip().lower()
        if rotation_deg in {90, 180, 270}:
            block_rect = self._block_rect(plan)
            fitted_rect = fitz.Rect(rect) & block_rect
            fitted_point = tuple(point or (fitted_rect.x0, fitted_rect.y1))
        else:
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
                "font_substitution_reason": str((style or {}).get("_font_substitution_reason") or ""),
                "unicode_fallback": bool((style or {}).get("_unicode_fallback")),
                "document_level_font_fallback": bool((style or {}).get("_document_level_font_fallback")),
                "fontsize": fontsize,
                "source_fontsize": float((style or {}).get("_source_size") or (style or {}).get("source_size") or fontsize),
                "min_font_ratio": float((style or {}).get("_min_font_ratio") or 0.90),
                "rgb": rgb,
                "rotation_deg": rotation_deg,
                "textbox_align": textbox_align,
                "intended_bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
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

    def _background_prep_ops(self, plan, bbox=None):
        rect = fitz.Rect(bbox or plan.block_bbox)
        if rect.get_area() <= 0:
            return []
        background_mode = str((plan.constraints or {}).get("background_mode") or "").strip().lower()
        structured_kind = str((plan.constraints or {}).get("structured_contract_kind") or "").strip().lower()
        table_locked = bool(plan.constraints.get("table_cell_bbox")) or str(getattr(plan, "block_type", "") or "").strip().lower() == "table"
        patch_ops = self.reconstructor._text_background_patch_ops_for_plan(
            plan,
            rect,
            allow_unsafe=False,
            prefer_inpaint=True,
        )
        if patch_ops:
            return patch_ops
        if background_mode == "local_bg_restore" and not self.reconstructor._clean_page_background_path(plan.page_data):
            overlay = self.reconstructor._local_inpaint_overlay_for_plan(plan, rect)
            if isinstance(overlay, dict):
                return [
                    BlockRenderOp(
                        op_type="draw_overlay_image",
                        block_id=plan.block_id,
                        unit_id=None,
                        bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                        z_index=0,
                        metadata={"path": overlay.get("path")},
                    )
                ]
            if table_locked:
                crop_path = self.reconstructor._local_background_crop_path(
                    plan.page_data,
                    fitz.Rect(rect),
                    allow_unsafe=True,
                )
                if crop_path and os.path.exists(crop_path):
                    return [
                        BlockRenderOp(
                            op_type="draw_overlay_image",
                            block_id=plan.block_id,
                            unit_id=None,
                            bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                            z_index=0,
                            metadata={"path": crop_path},
                        )
                    ]
        if plan.background_strategy == "whiteout":
            sampled = self.reconstructor._sample_local_background_fill_rgb(
                plan.page_data,
                fitz.Rect(rect),
                allow_unsafe=True,
            )
            metadata = {}
            if isinstance(sampled, dict) and sampled.get("rgb"):
                metadata = {
                    "fill_rgb": sampled.get("rgb"),
                    "fill_sample": sampled.get("sample") or "region",
                }
            return [
                BlockRenderOp(
                    "erase_rect",
                    plan.block_id,
                    None,
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    z_index=0,
                    metadata=metadata,
                )
            ]
        return []


class StructuredContractRenderer(BaseBlockRenderer):
    def _structured_payload(self, plan):
        payload = dict((plan.constraints or {}).get("structured_contract_plan") or {})
        return payload if payload.get("enabled") else {}

    def _background_ops(self, plan, payload):
        bbox = payload.get("background_region_bbox") or payload.get("draw_bbox") or plan.block_bbox
        return self._background_prep_ops(plan, bbox=bbox)

    def _slot_units(self, plan):
        units = []
        for unit in (plan.units or []):
            if not isinstance(unit, PlacableUnit):
                continue
            text = self.reconstructor._clean_text_for_render(unit.text_translated or unit.text_source)
            if not text:
                continue
            rect = self.reconstructor._fitz_rect_from_bbox_like(unit.relative_bbox)
            if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                continue
            units.append(unit)
        return sorted(
            units,
            key=lambda unit: (
                (unit.relative_bbox or plan.block_bbox)[1],
                (unit.relative_bbox or plan.block_bbox)[0],
                (unit.line_indices or [10**9])[0],
                unit.unit_id,
            ),
        )

    def _render_units_in_slots(self, page, plan, payload, *, force_rotation=None):
        ops = []
        block_rect = fitz.Rect(plan.block_bbox)
        units = self._slot_units(plan)
        if not units:
            return ops
        for unit in units:
            text = self.reconstructor._clean_text_for_render(unit.text_translated or unit.text_source)
            if not text:
                continue
            rect = self.reconstructor._fitz_rect_from_bbox_like(unit.relative_bbox)
            if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                continue
            rect = rect & block_rect
            if rect.get_area() <= 0:
                continue
            style = self.reconstructor._merge_styles(unit.style or {}, self.reconstructor._style_from_block(plan.source_block))
            _, fontfile, builtin, fontname = self.reconstructor._resolve_style_font(page, style, text=text)
            rotation_deg = force_rotation if force_rotation in {90, 180, 270} else self._unit_rotation_deg(unit=unit)
            align = self.reconstructor._normalize_alignment(
                self.reconstructor._unit_horizontal_alignment(unit, plan.paragraph_alignment or plan.alignment)
            )
            rgb = self.reconstructor._resolve_text_color(style, plan.source_block)
            tuning = self.reconstructor._unit_render_tuning(unit, plan)
            if rotation_deg in {90, 180, 270}:
                op = self._emit_rotated_textbox_run(
                    page,
                    plan,
                    text,
                    rect,
                    style,
                    fontname,
                    fontfile,
                    builtin,
                    rgb,
                    rotation_deg=rotation_deg,
                    align="center" if align not in {"left", "right"} else align,
                    unit_id=unit.unit_id,
                    min_fontsize=tuning["min_fontsize"],
                )
                if op is not None:
                    ops.append(op)
                continue
            fontsize = min(float(style.get("size") or 11.0), max(tuning["min_fontsize"], rect.height * 0.78))
            wrapped = self.reconstructor._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            while fontsize > tuning["min_fontsize"] and wrapped and (len(wrapped) * max(6.0, fontsize * 1.08)) > max(rect.height, fontsize * 1.25):
                fontsize -= 0.5
                wrapped = self.reconstructor._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            line_h = max(5.5, min(rect.height / max(1, len(wrapped)), fontsize * 1.08))
            baseline = rect.y0 + min(rect.height - 1.0, max(fontsize * 0.82, (rect.height - line_h * max(0, len(wrapped) - 1)) * 0.5))
            for line_idx, line_text in enumerate(wrapped):
                width = self.reconstructor._measure_text_width(line_text, fontsize, fontname, fontfile)
                x = rect.x0
                if align == "center":
                    x = max(rect.x0, rect.x0 + max(0.0, (rect.width - width) / 2.0))
                elif align == "right":
                    x = max(rect.x0, rect.x1 - width)
                line_baseline = min(rect.y1 - 1.0, baseline + line_idx * line_h)
                text_rect = fitz.Rect(
                    x,
                    line_baseline - max(1.0, fontsize * 0.82),
                    min(rect.x1, x + width),
                    line_baseline + max(1.0, fontsize * 0.18),
                )
                ops.append(
                    self._emit_text_run(
                        plan,
                        line_text,
                        text_rect,
                        (x, line_baseline),
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

    def _render_grid(self, page, plan, payload):
        block = plan.source_block or {}
        descriptor_role = str((block or {}).get("descriptor_structural_role") or "").strip().lower()
        object_type = str((block or {}).get("object_type") or "").strip().lower()
        object_class = str((block or {}).get("object_class") or "").strip().lower()
        if (
            plan.constraints.get("table_cell_bbox")
            or descriptor_role.startswith("table")
            or object_type in {"table_cell", "table_header_cell", "table_stub_cell", "table_value_cell"}
            or object_class == "tabular"
        ):
            return TableBlockRenderer(self.reconstructor).render(page, plan)
        ops = self._background_ops(plan, payload)
        ops.extend(self._render_units_in_slots(page, plan, payload))
        return ops

    def _render_rotated_grid(self, page, plan, payload):
        block = plan.source_block or {}
        descriptor_role = str((block or {}).get("descriptor_structural_role") or "").strip().lower()
        object_type = str((block or {}).get("object_type") or "").strip().lower()
        object_class = str((block or {}).get("object_class") or "").strip().lower()
        if (
            plan.constraints.get("table_cell_bbox")
            or descriptor_role.startswith("table")
            or object_type in {"table_cell", "table_header_cell", "table_stub_cell", "table_value_cell"}
            or object_class == "tabular"
        ):
            return TableBlockRenderer(self.reconstructor).render(page, plan)
        ops = self._background_ops(plan, payload)
        rotation = int(payload.get("rotation_deg") or 90)
        ops.extend(self._render_units_in_slots(page, plan, payload, force_rotation=rotation))
        if any(op.op_type == "draw_text_run" for op in ops):
            return ops
        return ops + EditorialBlockRenderer(self.reconstructor)._linewise_fallback(page, plan)

    def _render_anchored_composite(self, page, plan, payload):
        ops = self._background_ops(plan, payload)
        block = plan.source_block or {}
        object_type = str((block or {}).get("object_type") or "").strip().lower()
        block_role = str((block or {}).get("role") or "").strip().lower()
        if object_type == "figure_caption" or block_role == "figure_caption":
            linewise = EditorialBlockRenderer(self.reconstructor)._linewise_fallback(page, plan)
            if linewise:
                severe = {"overflow", "text_overlap", "protected_overlap"}
                linewise_no_bg = [
                    op for op in linewise
                    if not (op.op_type in {"erase_rect", "draw_overlay_image"} and int(op.z_index or 0) == 0)
                ]
                combined_linewise = ops + linewise_no_bg
                findings = self.reconstructor._validate_block_layout(plan, combined_linewise)
                if not any(str((f or {}).get("type") or "").strip().lower() in severe for f in findings):
                    return combined_linewise
        slot_ops = self._render_units_in_slots(page, plan, payload)
        if slot_ops:
            combined = ops + slot_ops
            severe = {"overflow", "text_overlap", "protected_overlap"}
            findings = self.reconstructor._validate_block_layout(plan, combined)
            text_ops = sum(1 for op in combined if op.op_type == "draw_text_run" and (op.text or "").strip())
            expected_units = max(1, self.reconstructor._expected_plan_text_ops(plan))
            if not any(str((f or {}).get("type") or "").strip().lower() in severe for f in findings) and text_ops >= expected_units:
                return combined
        return EditorialBlockRenderer(self.reconstructor)._render_bbox_anchored(page, plan)

    def render(self, page, plan):
        payload = self._structured_payload(plan)
        if not payload:
            return EditorialBlockRenderer(self.reconstructor).render(page, plan)
        # Pour les blocs line_preserve, les templates définissent les positions de ligne.
        # On tente _render_with_scale (qui utilise templates + edges) avant le rendu structuré
        # par slots (qui ne tient pas compte des positions inter-lignes des templates).
        contract_render_mode = str((plan.constraints or {}).get("contract_render_mode") or "").strip().lower()
        if contract_render_mode == "line_preserve":
            editor = EditorialBlockRenderer(self.reconstructor)
            linewise = editor._linewise_fallback(page, plan)
            if linewise:
                return linewise
            scale_ops = editor._render_with_scale(page, plan, 1.0)
            if scale_ops:
                severe = {"overflow", "text_overlap", "protected_overlap"}
                findings = self.reconstructor._validate_block_layout(plan, scale_ops)
                if not any(str((f or {}).get("type") or "").strip().lower() in severe for f in findings):
                    return scale_ops
        overlay_ops = self._overlay_ops_for_matching_immutable_overlays(plan)
        kind = str(payload.get("kind") or "").strip().lower()
        if kind == "styled_paragraph":
            return EditorialBlockRenderer(self.reconstructor)._render_prose_reflow(page, plan)
        if kind == "line_locked_cluster":
            ops = self._background_ops(plan, payload)
            locked = EditorialBlockRenderer(self.reconstructor)._linewise_fallback(page, plan)
            if locked:
                if ops:
                    locked = [op for op in locked if not (op.op_type in {"erase_rect", "draw_overlay_image"} and int(op.z_index or 0) == 0)]
                return ops + locked + overlay_ops
            return EditorialBlockRenderer(self.reconstructor).render(page, plan)
        if kind == "grid":
            return self._render_grid(page, plan, payload) + overlay_ops
        if kind == "rotated_grid":
            return self._render_rotated_grid(page, plan, payload) + overlay_ops
        if kind == "anchored_composite":
            return self._render_anchored_composite(page, plan, payload) + overlay_ops
        slot_ops = self._render_units_in_slots(page, plan, payload)
        if slot_ops:
            return self._background_ops(plan, payload) + slot_ops + overlay_ops
        return EditorialBlockRenderer(self.reconstructor).render(page, plan)


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
        if all(str(unit.render_policy or "").strip().lower() == "external_flow" for unit in units):
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
                rotation_deg=first.rotation_deg,
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
        contract_key = self.reconstructor._reconstruction_contract_key_for_block(block, page_data=plan.page_data)
        lines = list((block.get("lines") or []))
        if not lines:
            return []
        templates = self._prepare_templates(plan)
        if not templates:
            return []
        ops = []
        ops.extend(self._background_prep_ops(plan))
        for idx, line in enumerate(lines):
            if idx >= len(templates):
                break
            text = self.reconstructor._line_translated_text(line)
            source_text = self.reconstructor._line_source_text(line)
            if (
                contract_key == "toc_entry"
                and source_text
                and any(ch.isalpha() for ch in source_text)
                and not any(ch.isalpha() for ch in text)
            ):
                text = source_text
            if not text and source_text:
                text = source_text
            if not text:
                continue
            style = self.reconstructor._merge_styles((line or {}).get("style") or {}, self.reconstructor._style_from_block(block))
            template_index = idx
            remaining_lines = [text]
            fontsize = float(style.get("size") or 12.0)
            min_fontsize = 4.0 if contract_key == "toc_entry" else (5.0 if str(adaptive_profile.get("page_profile") or "") in {"academic_dense", "technical_structured"} else 5.5)
            while fontsize >= min_fontsize:
                probe_template = templates[min(template_index, len(templates) - 1)]
                if contract_key == "toc_entry":
                    wrapped = [text]
                else:
                    wrapped = self._wrap_text(page, {**style, "size": fontsize}, text, max(8.0, probe_template.usable_width))
                if len(wrapped) <= 1:
                    remaining_lines = wrapped
                    break
                fontsize -= 0.5
            if contract_key == "toc_entry" and len(remaining_lines) > 1:
                remaining_lines = [" ".join(part for part in remaining_lines if part)]
            _, fontfile, builtin, fontname = self.reconstructor._resolve_style_font( page, {**style, "size": fontsize}, text=text)
            rgb = self.reconstructor._resolve_text_color( style, block)
            for wrapped_text in remaining_lines:
                if template_index >= len(templates):
                    break
                template = templates[template_index]
                rotation_deg = self._unit_rotation_deg(template=template)
                if rotation_deg in {90, 180, 270}:
                    rect = fitz.Rect(template.bbox)
                    op = self._emit_rotated_textbox_run(
                        page,
                        plan,
                        wrapped_text,
                        rect,
                        style,
                        fontname,
                        fontfile,
                        builtin,
                        rgb,
                        rotation_deg=rotation_deg,
                        align="center",
                        unit_id=f"{plan.block_id}:line:{idx}",
                        min_fontsize=min_fontsize,
                    )
                    if op is not None:
                        ops.append(op)
                    template_index += 1
                    continue
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
                if contract_key == "toc_entry":
                    break
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
        ops.extend(self._background_prep_ops(plan))
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
            rotation_deg = self._unit_rotation_deg(unit=unit)
            if rotation_deg in {90, 180, 270}:
                rgb = self.reconstructor._resolve_text_color( style, plan.source_block)
                op = self._emit_rotated_textbox_run(
                    page,
                    plan,
                    text,
                    rect,
                    style,
                    fontname,
                    fontfile,
                    builtin,
                    rgb,
                    rotation_deg=rotation_deg,
                    align="center",
                    unit_id=unit.unit_id,
                    min_fontsize=tuning["min_fontsize"],
                )
                if op is not None:
                    ops.append(op)
                continue
            source_fontsize = float(style.get("size") or 12.0)
            rect = self.reconstructor._expand_text_rect_within_block(rect, block_rect, source_fontsize, line_count=1, line_height_factor=1.14)
            fontsize = min(source_fontsize, max(6.0, rect.height * 0.90))
            wrapped = self.reconstructor._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            if wrapped:
                rect = self.reconstructor._expand_text_rect_within_block(
                    rect,
                    block_rect,
                    fontsize,
                    line_count=len(wrapped),
                    line_height_factor=1.12 * tuning["line_spacing_factor"],
                )
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
                        {**style, "size": fontsize, "_source_size": source_fontsize, "_min_font_ratio": 0.90},
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
        ops.extend(self._background_prep_ops(plan))
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
            rotation_deg = self._unit_rotation_deg(unit=unit)
            if rotation_deg in {90, 180, 270}:
                rgb = self.reconstructor._resolve_text_color( style, plan.source_block)
                op = self._emit_rotated_textbox_run(
                    page,
                    plan,
                    text,
                    rect,
                    style,
                    fontname,
                    fontfile,
                    builtin,
                    rgb,
                    rotation_deg=rotation_deg,
                    align="center",
                    unit_id=unit.unit_id,
                    min_fontsize=tuning["min_fontsize"],
                )
                if op is not None:
                    ops.append(op)
                continue
            source_fontsize = float(style.get("size") or 12.0)
            rect = self.reconstructor._expand_text_rect_within_block(rect, block_rect, source_fontsize, line_count=1, line_height_factor=1.14)
            fontsize = min(source_fontsize, max(6.0, rect.height * 0.90))
            wrapped = self.reconstructor._wrap_text_for_bbox(page, {**style, "size": fontsize}, text, max(8.0, rect.width))
            if wrapped:
                rect = self.reconstructor._expand_text_rect_within_block(
                    rect,
                    block_rect,
                    fontsize,
                    line_count=len(wrapped),
                    line_height_factor=1.12 * tuning["line_spacing_factor"],
                )
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
                        {**style, "size": fontsize, "_source_size": source_fontsize, "_min_font_ratio": 0.90},
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

        def _measure_segments(candidate_segments):
            widths = []
            measurements = []
            for seg in candidate_segments:
                width, fontsize, fontname, fontfile, builtin, rgb = self._measure_text(page, seg["style"], seg["text"])
                widths.append(width)
                measurements.append((fontsize, fontname, fontfile, builtin, rgb))
            return widths, measurements

        widths, measurements = _measure_segments(segments)
        if len(segments) > 1:
            first_style = dict(segments[0].get("style") or {})
            space_probe_size = float(first_style.get("size") or measurements[0][0] or template.ascent)
            _, space_fontfile, _, space_fontname = self.reconstructor._resolve_style_font(page, first_style, text="n n")
            space_width = max(
                0.0,
                self.reconstructor._measure_text_width("n n", space_probe_size, space_fontname, space_fontfile)
                - (2.0 * self.reconstructor._measure_text_width("n", space_probe_size, space_fontname, space_fontfile)),
            )
            default_gap = max(default_gap, min(7.0, max(1.8, space_width)))

        available_line_width = max(8.0, template.usable_width)
        min_fontsize = 5.0 if dense_profile else 5.5
        while segments and len(segments) > 1 and sum(widths) + default_gap * max(0, len(widths) - 1) > available_line_width:
            current_sizes = [float((seg.get("style") or {}).get("size") or measurements[idx][0] or 12.0) for idx, seg in enumerate(segments)]
            if min(current_sizes) <= min_fontsize:
                break
            segments = [
                {
                    **seg,
                    "style": {
                        **dict(seg.get("style") or {}),
                        "size": max(min_fontsize, float((seg.get("style") or {}).get("size") or measurements[idx][0] or 12.0) - 0.35),
                    },
                }
                for idx, seg in enumerate(segments)
            ]
            widths, measurements = _measure_segments(segments)
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
        ops = self._background_prep_ops(plan)
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
        block_text = self.reconstructor._clean_text_for_render(
            self.reconstructor._translated_text_from_block(plan.source_block or {})
        )
        if block_text:
            return block_text
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

    def _compose_paragraphs_in_box(
        self,
        page,
        plan,
        paragraphs,
        block_rect,
        base_style,
        fontname,
        fontfile,
        *,
        reflow_left=None,
        reflow_right=None,
        max_font_shrink_pt=None,
        min_font_ratio=0.90,
    ):
        base_fontsize = max(7.0, min(float(base_style.get("size") or 11.0), 36.0))
        if max_font_shrink_pt is None:
            max_font_shrink_pt = max(0.5, base_fontsize * 0.10)
        max_font_shrink_pt = max(0.0, float(max_font_shrink_pt or 0.0))
        min_font_ratio = max(0.50, min(1.0, float(min_font_ratio or 0.90)))
        options = ComposeOptions(
            enable_hyphenation=True,
            max_font_shrink=max_font_shrink_pt,
            step_pt=0.25,
            min_font_pt=max(5.5, base_fontsize * min_font_ratio),
        )
        x0 = max(block_rect.x0 + 2.0, float(reflow_left) if isinstance(reflow_left, (int, float)) else (block_rect.x0 + 2.0))
        x1 = min(block_rect.x1 - 2.0, float(reflow_right) if isinstance(reflow_right, (int, float)) else (block_rect.x1 - 2.0))
        usable_w = max(8.0, x1 - x0)
        remaining_h = max(8.0, block_rect.height - 4.0)
        composed = []
        for idx, paragraph in enumerate(paragraphs):
            if not paragraph:
                continue
            result = self.reconstructor.text_composer.compose_text_in_box(
                paragraph,
                usable_w,
                remaining_h,
                base_fontsize,
                1.18,
                lambda txt, fs: self.reconstructor._measure_text_width(txt, fs, fontname, fontfile),
                alignment=self.reconstructor._normalize_alignment(plan.paragraph_alignment or plan.alignment or "left"),
                lang=str((plan.page_data or {}).get("target_lang") or "fr"),
                options=options,
            )
            composed.append(result)
            used_h = max(0.0, len(result.get("lines") or []) * float(result.get("line_height") or (base_fontsize * 1.18)))
            remaining_h = max(8.0, remaining_h - used_h - (float(result.get("line_height") or (base_fontsize * 1.18)) * 0.55))
        return composed

    def _composed_paragraphs_have_overflow(self, composed_paragraphs):
        return any(not bool((item or {}).get("fits", True)) for item in composed_paragraphs or [])

    def _overflow_findings_from_composition(self, plan, composed_paragraphs):
        findings = []
        for idx, item in enumerate(composed_paragraphs or []):
            if bool((item or {}).get("fits", True)):
                continue
            findings.append(
                {
                    "type": "composition_overflow",
                    "block_id": plan.block_id,
                    "paragraph_index": idx,
                    "lost_text": str((item or {}).get("lost_text") or (item or {}).get("overflow") or "")[:500],
                    "line_count_required": int((item or {}).get("line_count_required") or 0),
                    "line_count_available": int((item or {}).get("line_count_available") or 0),
                    "font_size": float((item or {}).get("font_size") or 0.0),
                }
            )
        return findings

    def _expanded_reflow_rect(self, plan, block_rect, *, max_growth_factor=1.75):
        page_rect = fitz.Rect(0, 0, block_rect.x1, block_rect.y1)
        try:
            dims = dict((plan.page_data or {}).get("dimensions") or {})
            page_w = float(dims.get("width_pt") or dims.get("page_width_pt") or 0.0)
            page_h = float(dims.get("height_pt") or dims.get("page_height_pt") or 0.0)
            if page_w > 0 and page_h > 0:
                page_rect = fitz.Rect(0, 0, page_w, page_h)
        except Exception:
            pass
        if getattr(plan, "page_data", None):
            try:
                # Page objects are already in PDF points in the reconstruction stage.
                page_rect = fitz.Rect(0, 0, max(page_rect.x1, block_rect.x1), max(page_rect.y1, block_rect.y1))
            except Exception:
                pass
        max_bottom = page_rect.y1 - 8.0
        for other in (plan.page_data or {}).get("blocks") or []:
            if not isinstance(other, dict) or str(other.get("id") or "") == str(plan.block_id):
                continue
            rect = self.reconstructor._fitz_rect_from_bbox_like(other.get("bbox"))
            if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                continue
            horizontal_overlap = max(0.0, min(block_rect.x1, rect.x1) - max(block_rect.x0, rect.x0))
            if horizontal_overlap < min(block_rect.width, rect.width) * 0.18:
                continue
            if rect.y0 >= block_rect.y1 - 1.0:
                max_bottom = min(max_bottom, rect.y0 - 3.0)
        growth_limit = block_rect.y0 + max(block_rect.height, block_rect.height * max_growth_factor)
        new_bottom = min(max_bottom, growth_limit)
        if new_bottom <= block_rect.y1 + 2.0:
            return block_rect
        return fitz.Rect(block_rect.x0, block_rect.y0, block_rect.x1, new_bottom)

    def _render_prose_reflow(self, page, plan):
        """Reflow du texte traduit en flux continu dans la bbox du bloc."""
        full_text = self._collect_translated_text_stream(plan)
        if not full_text.strip():
            return []
        ops = []
        block_rect = fitz.Rect(plan.block_bbox)
        profile = getattr(plan, "semantic_profile", None)
        # Police unicode-safe garantie
        fontfile, fontname = self.reconstructor._resolve_unicode_safe_font(page, plan, full_text)
        # Style de base (couleur, flags)
        base_style = self.reconstructor._style_from_block(plan.source_block or {})
        rgb = self.reconstructor._resolve_text_color(base_style, plan.source_block)
        alignment = self.reconstructor._normalize_alignment(plan.paragraph_alignment or plan.alignment or "left")
        templates = self._prepare_templates(plan)
        non_rotated_templates = [template for template in templates if int(getattr(template, "rotation_deg", 0) or 0) % 360 not in {90, 180, 270}]
        reflow_left = block_rect.x0 + 2.0
        reflow_right = block_rect.x1 - 2.0
        if non_rotated_templates:
            reflow_left = max(reflow_left, min(float(template.left_x) for template in non_rotated_templates))
            reflow_right = min(reflow_right, max(float(template.right_x) for template in non_rotated_templates))
        if reflow_right - reflow_left < 16.0:
            reflow_left = block_rect.x0 + 2.0
            reflow_right = block_rect.x1 - 2.0
        paragraphs = [
            self.reconstructor._clean_text_for_render(part)
            for part in re.split(r"\n+", full_text)
            if self.reconstructor._clean_text_for_render(part)
        ]
        if not paragraphs:
            paragraphs = [full_text]
        composed_paragraphs = self._compose_paragraphs_in_box(
            page, plan, paragraphs, block_rect, base_style, fontname, fontfile,
            reflow_left=reflow_left, reflow_right=reflow_right,
        )
        if self._composed_paragraphs_have_overflow(composed_paragraphs):
            expanded_rect = self._expanded_reflow_rect(plan, block_rect, max_growth_factor=2.50)
            if expanded_rect.height > block_rect.height + 2.0:
                expanded_plan = replace(
                    plan,
                    block_bbox=(expanded_rect.x0, expanded_rect.y0, expanded_rect.x1, expanded_rect.y1),
                    container_bbox=(expanded_rect.x0, expanded_rect.y0, expanded_rect.x1, expanded_rect.y1),
                )
                expanded_composed = self._compose_paragraphs_in_box(
                    page, expanded_plan, paragraphs, expanded_rect, base_style, fontname, fontfile,
                    reflow_left=reflow_left, reflow_right=reflow_right,
                    max_font_shrink_pt=0.0,
                    min_font_ratio=1.0,
                )
                if self._composed_paragraphs_have_overflow(expanded_composed):
                    expanded_composed = self._compose_paragraphs_in_box(
                        page, expanded_plan, paragraphs, expanded_rect, base_style, fontname, fontfile,
                        reflow_left=reflow_left, reflow_right=reflow_right,
                    )
                if not self._composed_paragraphs_have_overflow(expanded_composed):
                    plan = expanded_plan
                    block_rect = expanded_rect
                    composed_paragraphs = expanded_composed
        overflow_findings = self._overflow_findings_from_composition(plan, composed_paragraphs)
        ops.extend(self._background_prep_ops(plan, bbox=(block_rect.x0, block_rect.y0, block_rect.x1, block_rect.y1)))
        for finding in overflow_findings:
            ops.append(
                BlockRenderOp(
                    op_type="render_warning",
                    block_id=plan.block_id,
                    unit_id=None,
                    bbox=(block_rect.x0, block_rect.y0, block_rect.x1, block_rect.y1),
                    z_index=0,
                    metadata={"render_findings": [finding]},
                )
            )
        current_y = block_rect.y0
        global_line_idx = 0
        for paragraph_idx, composed in enumerate(composed_paragraphs):
            fontsize = float(composed.get("font_size") or base_style.get("size") or 11.0)
            line_h = float(composed.get("line_height") or (fontsize * 1.18))
            lines = list(composed.get("lines") or [])
            for local_idx, line_text in enumerate(lines):
                template = None
                if templates and global_line_idx < len(templates) and templates[global_line_idx].rotation_deg not in {90, 180, 270}:
                    template = templates[global_line_idx]
                    baseline = templates[global_line_idx].baseline_y
                else:
                    baseline = current_y + fontsize * 0.82
                left_x = reflow_left
                if template is not None and local_idx == 0:
                    left_x = min(reflow_right - 8.0, left_x + max(0.0, float(template.first_line_indent_px or template.indent_px or 0.0)))
                right_x = reflow_right
                width = self.reconstructor._measure_text_width(line_text, fontsize, fontname, fontfile)
                is_last_line = local_idx == len(lines) - 1
                available_width = max(8.0, right_x - left_x)
                if alignment == "justify" and not is_last_line and width >= available_width * 0.72:
                    tokens = self._tokenize_text(line_text)
                    if len(tokens) > 1:
                        token_widths = [self.reconstructor._measure_text_width(token, fontsize, fontname, fontfile) for token in tokens]
                        total_token_width = sum(token_widths)
                        available = max(0.0, available_width - total_token_width)
                        gap = available / max(1, len(tokens) - 1)
                        cur_x = left_x
                        for token, token_width in zip(tokens, token_widths):
                            token_rect = fitz.Rect(
                                cur_x,
                                baseline - max(1.0, fontsize * 0.82),
                                min(right_x, cur_x + token_width),
                                baseline + max(1.0, fontsize * 0.18),
                            )
                            ops.append(self._emit_text_run(
                                plan, token, token_rect, (cur_x, baseline),
                                {**base_style, "size": fontsize, "_source_size": base_style.get("size") or fontsize, "_min_font_ratio": 0.90}, fontname, fontfile, None, fontsize, rgb,
                                unit_id=f"{plan.block_id}:reflow:{global_line_idx}:{token}",
                            ))
                            cur_x += token_width + gap
                        current_y = baseline + line_h
                        global_line_idx += 1
                        continue
                x = left_x
                if alignment == "center":
                    x = max(left_x, left_x + max(0.0, (right_x - left_x - width) / 2.0))
                elif alignment == "right":
                    x = max(left_x, right_x - width)
                text_rect = fitz.Rect(x, baseline - max(1.0, fontsize * 0.82), min(right_x, x + width), baseline + max(1.0, fontsize * 0.18))
                ops.append(self._emit_text_run(
                    plan, line_text, text_rect, (x, baseline),
                    {**base_style, "size": fontsize, "_source_size": base_style.get("size") or fontsize, "_min_font_ratio": 0.90}, fontname, fontfile, None, fontsize, rgb,
                    unit_id=f"{plan.block_id}:reflow:{global_line_idx}",
                ))
                current_y = baseline + line_h
                global_line_idx += 1
            current_y += line_h * 0.55 if lines else 0.0
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
        ops.extend(self._background_prep_ops(plan))
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
            source_fontsize = float(style.get("size") or 11.0)
            min_font_ratio = 0.95
            unit_rect = self.reconstructor._expand_text_rect_within_block(
                unit_rect,
                block_rect,
                source_fontsize,
                line_count=1,
                line_height_factor=1.08,
            )
            fontsize = min(source_fontsize, max(source_fontsize * min_font_ratio, unit_rect.height * 0.90))
            # Reduire si le texte ne tient pas en largeur
            available_w = max(4.0, unit_rect.width)
            min_fontsize = max(5.5, source_fontsize * min_font_ratio)
            while fontsize > min_fontsize and self.reconstructor._measure_text_width(text, fontsize, fontname, fontfile) > available_w:
                fontsize -= 0.5
            fontsize = max(min_fontsize, fontsize)
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
                {**style, "size": fontsize, "_source_size": source_fontsize, "_min_font_ratio": min_font_ratio}, fontname, fontfile, None, fontsize, rgb,
                unit_id=unit.unit_id,
            ))
        return ops

    def render(self, page, plan):
        contract_render_mode = str((plan.constraints or {}).get("contract_render_mode") or "").strip().lower()
        if contract_render_mode == "relative_slots":
            return self._render_relative_slots(page, plan)
        if contract_render_mode == "bbox_anchored":
            return self._render_bbox_anchored(page, plan)
        if contract_render_mode == "line_preserve":
            fallback_ops = self._linewise_fallback(page, plan)
            if fallback_ops:
                return fallback_ops
        if contract_render_mode == "prose_reflow":
            return self._render_prose_reflow(page, plan)

        profile = plan.semantic_profile
        if profile is None:
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
            plan = replace(plan, semantic_profile=profile)
        # Dispatch base sur le profil semantique si le bloc est traduit
        _units_have_mixed_style_fragments = any(
            list((u.metadata or {}).get("fragments") or {}) or
            list((((u.metadata or {}).get("raw_unit") or {})).get("fragments") or [])
            for u in (plan.units or [])
        )
        if profile is not None and profile.source_is_translated and not _units_have_mixed_style_fragments:
            strategy = profile.render_strategy
            # Pour les blocs positionnels (anchored_text / fixed_preserve), on ne reflow jamais :
            # le texte traduit doit rester à la position exacte du source (TOC, étiquettes, etc.)
            _src_rp = str((plan.source_block or {}).get("render_policy") or "").strip().lower()
            _is_anchored = _src_rp in {"anchored_text", "fixed_preserve"}
            if strategy in ("prose_reflow", "heading_reflow", "caption_reflow"):
                if _is_anchored:
                    return self._render_label_stack(page, plan)
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
        ops.extend(self._background_prep_ops(plan))
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
            source_rect = self.reconstructor._fitz_rect_from_bbox_like((line or {}).get("bbox"))
            if isinstance(source_rect, fitz.Rect) and source_rect.get_area() > 0:
                source_rect = source_rect & block_rect
                x = max(block_rect.x0, source_rect.x0)
                baseline_probe = float(effective_style.get("size") or 10.0)
                baseline = min(
                    block_rect.y1 - 1.0,
                    max(
                        block_rect.y0 + baseline_probe * 0.82,
                        source_rect.y0 + min(max(1.0, source_rect.height - 1.0), baseline_probe * 0.82),
                    ),
                )
                available_width = max(8.0, min(block_rect.x1, max(source_rect.x1, x + 8.0)) - x)
                available_height = max(5.0, source_rect.height)
            elif template_lines and idx < len(template_lines):
                tmpl = template_lines[idx]
                baseline = tmpl.baseline_y
                x = tmpl.left_x
                available_width = max(8.0, tmpl.right_x - tmpl.left_x)
                available_height = max(5.0, tmpl.height)
            else:
                fontsize_probe = float(effective_style.get("size") or 10.0)
                line_h = max(5.0, block_rect.height / max(1, len(lines)))
                baseline = block_rect.y0 + (idx + 0.82) * line_h
                x = block_rect.x0
                available_width = max(8.0, block_rect.x1 - x)
                available_height = max(5.0, line_h)
            _, fontfile, builtin, fontname = self.reconstructor._resolve_style_font(page, effective_style, text=text)
            fontsize = min(float(effective_style.get("size") or 10.0), max(4.5, available_height * 0.82))
            min_fontsize = 4.5 if line_is_technical else 5.0
            while fontsize > min_fontsize and self.reconstructor._measure_text_width(text, fontsize, fontname, fontfile) > available_width:
                fontsize -= 0.5
            rgb = self.reconstructor._resolve_text_color(effective_style, block)
            text_rect = fitz.Rect(
                x,
                baseline - max(1.0, fontsize * 0.82),
                min(block_rect.x1, x + available_width),
                min(block_rect.y1, baseline + max(1.0, fontsize * 0.18)),
            )
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

    def _table_unit_ops(self, page, plan, cell_rect):
        units = sorted(
            [
                unit for unit in (plan.units or [])
                if self.reconstructor._text_requires_visible_replacement(unit.text_source, unit.text_translated)
            ],
            key=lambda unit: (
                (unit.relative_bbox or plan.block_bbox)[1],
                (unit.relative_bbox or plan.block_bbox)[0],
                unit.unit_id,
            ),
        )
        if not units:
            return []
        ops = []
        run_index = 0
        for unit in units:
            text = self.reconstructor._clean_text_for_render(unit.text_translated or "")
            if not text:
                continue
            rect = self.reconstructor._fitz_rect_from_bbox_like(unit.relative_bbox) if unit.relative_bbox else fitz.Rect(cell_rect)
            if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                rect = fitz.Rect(cell_rect)
            rect = fitz.Rect(
                max(cell_rect.x0, rect.x0),
                max(cell_rect.y0, rect.y0),
                min(cell_rect.x1, rect.x1),
                min(cell_rect.y1, rect.y1),
            )
            if rect.width <= 0 or rect.height <= 0:
                continue
            style = self.reconstructor._merge_styles(unit.style or {}, self.reconstructor._style_from_block(plan.source_block))
            _, fontfile, builtin, fontname = self.reconstructor._resolve_style_font(page, style, text=text)
            rotation_deg = self._unit_rotation_deg(unit=unit)
            if rotation_deg in {90, 180, 270}:
                rgb = self.reconstructor._resolve_text_color(style, plan.source_block)
                op = self._emit_rotated_textbox_run(
                    page,
                    plan,
                    text,
                    rect,
                    style,
                    fontname,
                    fontfile,
                    builtin,
                    rgb,
                    rotation_deg=rotation_deg,
                    align="center",
                    unit_id=f"{plan.block_id}:table_unit:{run_index}",
                    min_fontsize=5.0,
                )
                if op is not None:
                    ops.append(op)
                    run_index += 1
                continue
            source_fontsize = float(style.get("size") or 10.0)
            min_font_ratio = 0.95
            rect = self.reconstructor._expand_text_rect_within_block(
                rect,
                cell_rect,
                source_fontsize,
                line_count=1,
                line_height_factor=1.08,
            )
            fontsize = min(source_fontsize, max(source_fontsize * min_font_ratio, rect.height * 0.90))
            wrapped = self._wrap_text_to_lines(text, max(8.0, rect.width), fontsize, fontname, fontfile)
            if wrapped:
                rect = self.reconstructor._expand_text_rect_within_block(
                    rect,
                    cell_rect,
                    fontsize,
                    line_count=len(wrapped),
                    line_height_factor=1.05,
                )
            min_fontsize = max(5.0, source_fontsize * min_font_ratio)
            hard_min_fontsize = 4.5
            while fontsize > min_fontsize and wrapped and (len(wrapped) * max(5.0, fontsize * 1.05)) > rect.height:
                fontsize -= 0.5
                wrapped = self._wrap_text_to_lines(text, max(8.0, rect.width), fontsize, fontname, fontfile)
            fontsize = max(min_fontsize, fontsize)
            rgb = self.reconstructor._resolve_text_color(style, plan.source_block)
            align = self.reconstructor._normalize_alignment(
                self.reconstructor._unit_horizontal_alignment(unit, (plan.source_block or {}).get("alignment") or "left")
            )
            line_h = max(5.0, min(rect.height / max(1, len(wrapped)), fontsize * 1.05))
            for line_idx, line_text in enumerate(wrapped):
                width = self.reconstructor._measure_text_width(line_text, fontsize, fontname, fontfile)
                baseline = rect.y0 + min(rect.height - 1.0, (line_idx + 1) * line_h * 0.82)
                x = rect.x0
                if align == "center":
                    x = max(rect.x0, rect.x0 + max(0.0, (rect.width - width) / 2.0))
                elif align == "right":
                    x = max(rect.x0, rect.x1 - width)
                text_rect = fitz.Rect(
                    x,
                    baseline - max(1.0, fontsize * 0.82),
                    min(rect.x1, x + width),
                    baseline + max(1.0, fontsize * 0.18),
                )
                ops.append(
                    self._emit_text_run(
                        plan,
                        line_text,
                        text_rect,
                        (x, baseline),
                        {**style, "size": fontsize, "_source_size": source_fontsize, "_min_font_ratio": min_font_ratio},
                        fontname,
                        fontfile,
                        builtin,
                        fontsize,
                        rgb,
                        unit_id=f"{plan.block_id}:table_unit:{run_index}",
                    )
                )
                run_index += 1
        return ops

    def render(self, page, plan):
        block = plan.source_block or {}
        descriptor_group_ids = dict((block or {}).get("descriptor_group_ids") or {})
        descriptor_role = str((block or {}).get("descriptor_structural_role") or "").strip().lower()
        object_type = str((block or {}).get("object_type") or "").strip().lower()
        object_class = str((block or {}).get("object_class") or "").strip().lower()
        has_explicit_cell = bool(
            plan.constraints.get("table_cell_bbox")
            or str(descriptor_group_ids.get("cell_id") or "").strip()
            or str(descriptor_group_ids.get("table_row_group_id") or "").strip()
            or descriptor_role.startswith("table")
            or object_type in {"table_cell", "table_header_cell", "table_stub_cell", "table_value_cell"}
            or object_class == "tabular"
        )
        if self.reconstructor._block_looks_technical_structured(block):
            return CodeBlockRenderer(self.reconstructor).render(page, plan)
        if not has_explicit_cell:
            return EditorialBlockRenderer(self.reconstructor).render(page, plan)
        cell_bbox = plan.constraints.get("table_cell_bbox") or plan.block_bbox
        cell_rect = fitz.Rect(cell_bbox)
        ops = self._background_prep_ops(plan, bbox=(cell_rect.x0, cell_rect.y0, cell_rect.x1, cell_rect.y1))
        # On reconstruit sur un fond maitre deja nettoye : toute ligne visible
        # doit etre re-emise, meme si la traduction est identique au source.
        lines = [
            line for line in (block.get("lines") or [])
            if self.reconstructor._clean_text_for_render(
                self.reconstructor._line_translated_text(line)
                or self.reconstructor._line_source_text(line)
            )
        ]
        if not lines:
            unit_ops = self._table_unit_ops(page, plan, cell_rect)
            if unit_ops:
                return ops + unit_ops
        if not lines:
            block_translated = self.reconstructor._translated_text_from_block(block)
            block_source = self.reconstructor._clean_text_for_render((block or {}).get("text") or "")
            if not block_source:
                block_source = "\n".join(
                    self.reconstructor._line_source_text(line)
                    for line in (block.get("lines") or [])
                    if self.reconstructor._line_source_text(line)
                ).strip()
            if self.reconstructor._text_requires_visible_replacement(block_source, block_translated, item=block):
                lines = [{"bbox": block.get("bbox"), "translated_text": block_translated, "line_text": block_source}]
        align = self.reconstructor._normalize_alignment((block or {}).get("alignment") or "left")
        template_lines = plan.line_templates or []
        run_index = 0
        for idx, line in enumerate(lines):
            text = self.reconstructor._line_translated_text(line) or self._line_text(line)
            if not text:
                continue
            style = self.reconstructor._merge_styles((line or {}).get("style") or {}, self.reconstructor._style_from_block(block))
            _, fontfile, builtin, fontname = self.reconstructor._resolve_style_font( page, style, text=text)
            rotation_deg = self.reconstructor._rotation_deg_for_item(
                line,
                bbox_like=(line or {}).get("bbox"),
                text=text,
                fallback=0,
            )
            if rotation_deg in {90, 180, 270}:
                rect = self.reconstructor._fitz_rect_from_bbox_like((line or {}).get("bbox"))
                if not isinstance(rect, fitz.Rect) or rect.get_area() <= 0:
                    if template_lines and idx < len(template_lines):
                        rect = fitz.Rect(template_lines[idx].bbox)
                    else:
                        total_lines_in_block = max(1, len(lines))
                        line_slot_h = cell_rect.height / total_lines_in_block
                        slot_top = cell_rect.y0 + idx * line_slot_h
                        rect = fitz.Rect(cell_rect.x0, slot_top, cell_rect.x1, min(cell_rect.y1, slot_top + line_slot_h))
                rect = rect & cell_rect
                rgb = self.reconstructor._resolve_text_color(style, block)
                op = self._emit_rotated_textbox_run(
                    page,
                    plan,
                    text,
                    rect,
                    style,
                    fontname,
                    fontfile,
                    builtin,
                    rgb,
                    rotation_deg=rotation_deg,
                    align="center",
                    unit_id=f"{plan.block_id}:table:{run_index}",
                    min_fontsize=5.5,
                )
                if op is not None:
                    ops.append(op)
                    run_index += 1
                continue
            fontsize = max(5.5, float(style.get("size") or 10.0))
            rgb = self.reconstructor._resolve_text_color( style, block)
            # Résoudre la zone de référence pour cette ligne
            source_rect = self.reconstructor._fitz_rect_from_bbox_like((line or {}).get("bbox"))
            if isinstance(source_rect, fitz.Rect) and source_rect.get_area() > 0:
                ref_x0 = max(cell_rect.x0, source_rect.x0)
                ref_baseline = min(cell_rect.y1 - 1.0, max(cell_rect.y0 + fontsize * 0.82, source_rect.y0 + min(source_rect.height - 1.0, fontsize * 0.82)))
                # Respecter d'abord la largeur reelle de la ligne source pour
                # eviter qu'une ligne de cellule n'empiète sur les autres
                # colonnes ou groupes voisins.
                source_width = max(8.0, source_rect.x1 - ref_x0)
                expandable_width = min(cell_rect.x1 - ref_x0, max(source_width, source_width * 1.35))
                ref_x1 = min(cell_rect.x1, max(ref_x0 + 8.0, ref_x0 + expandable_width))
                available_w = max(8.0, ref_x1 - ref_x0)
                source_fontsize = max(1.0, float(style.get("size") or fontsize or 10.0))
                min_ratio = 0.925 if len(lines) >= 2 else 0.95
                min_fontsize = max(4.5, source_fontsize * min_ratio)
                source_line_text = self.reconstructor._line_source_text(line)
                measured_w = self.reconstructor._measure_text_width(text, fontsize, fontname, fontfile)
                cell_available_w = max(8.0, cell_rect.x1 - ref_x0)
                if measured_w > available_w and measured_w <= cell_available_w:
                    ref_x1 = cell_rect.x1
                    available_w = cell_available_w
                translated_min_w = self.reconstructor._measure_text_width(text, min_fontsize, fontname, fontfile)
                if (
                    source_line_text
                    and source_line_text != text
                    and translated_min_w > available_w
                ):
                    source_w = self.reconstructor._measure_text_width(source_line_text, fontsize, fontname, fontfile)
                    source_w_min = self.reconstructor._measure_text_width(source_line_text, min_fontsize, fontname, fontfile)
                    if source_w <= cell_available_w or source_w_min <= cell_available_w:
                        text = source_line_text
                        ref_x1 = cell_rect.x1
                        available_w = cell_available_w
                        if source_w > available_w:
                            fontsize = min_fontsize
                            measured_w = source_w_min
                        else:
                            measured_w = source_w
                while fontsize > min_fontsize and measured_w > available_w:
                    fontsize -= 0.5
                    if fontsize < min_fontsize:
                        fontsize = min_fontsize
                    measured_w = self.reconstructor._measure_text_width(text, fontsize, fontname, fontfile)
                if measured_w <= available_w:
                    wrapped = [text]
                else:
                    wrapped = self._wrap_text_to_lines(text, available_w, fontsize, fontname, fontfile)
                    while fontsize > min_fontsize and wrapped and (len(wrapped) * max(fontsize * 1.04, 4.8)) > max(5.0, cell_rect.y1 - source_rect.y0):
                        fontsize -= 0.5
                        wrapped = self._wrap_text_to_lines(text, available_w, fontsize, fontname, fontfile)
                line_h = max(fontsize * 1.08, 1.0)
                for wi, seg in enumerate(wrapped):
                    seg_w = self.reconstructor._measure_text_width(seg, fontsize, fontname, fontfile)
                    baseline = min(cell_rect.y1 - 1.0, ref_baseline + wi * line_h)
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
            elif template_lines and idx < len(template_lines):
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
