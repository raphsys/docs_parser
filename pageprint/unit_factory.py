"""PAGEPRINT Unit Factory — transforme blocs/lignes/phrases/spans en unités canoniques.

Entrée : page_structure["blocks"] (structure legacy).
Sortie : units[] — liste plate d'unités canoniques respectant UNIT_REQUIRED,
chaque unité portant les trois couches séparées :
    extraction brute / compréhension / politique de traitement.
"""

from __future__ import annotations

import re

from .normalizer import (
    clamp_confidence,
    bbox_pt_to_px,
    normalize_bbox_to_pt,
    normalize_style,
)
from .schema import (
    CANONICAL_UNIT,
    MIN_VISUAL_AREA_PT2,
    MIN_VISUAL_DIM_PT,
    empty_unit,
)

TOC_BULLET_MARKERS = {
    "■",
    "•",
    "▪",
    "◦",
    "‣",
    "⁃",
    "·",
    "◆",
    "▶",
    "▷",
}
TOC_TITLE_RE = re.compile(r"^(?:contents?|table of contents|sommaire)$", re.IGNORECASE)
TOC_SECTION_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*[a-z]?$")
TOC_PAGE_REFERENCE_RE = re.compile(r"^(?:\d{1,4}|[ivxlcdm]{1,8})$", re.IGNORECASE)


def _span_style_signature(span: dict) -> tuple:
    """Signature de style pour la fusion de spans consécutifs identiques."""
    style = span.get("style") or {}
    flags = style.get("flags") or {}
    return (
        style.get("font_family") or style.get("font"),
        style.get("font_size_pt") or style.get("size_pt") or style.get("size"),
        str(style.get("color")),
        bool(style.get("bold") or flags.get("bold")),
        bool(style.get("italic") or flags.get("italic")),
        bool(style.get("underline") or flags.get("underline")),
        bool(style.get("monospace") or flags.get("monospace")),
    )


def _union_bbox(a, b):
    if not (isinstance(a, (list, tuple)) and len(a) == 4):
        return list(b) if b else None
    if not (isinstance(b, (list, tuple)) and len(b) == 4):
        return list(a)
    return [
        min(float(a[0]), float(b[0])),
        min(float(a[1]), float(b[1])),
        max(float(a[2]), float(b[2])),
        max(float(a[3]), float(b[3])),
    ]


def _page_bbox_pt(page_structure: dict, sx: float, sy: float) -> list[float] | None:
    dimensions = page_structure.get("dimensions") or {}
    if dimensions.get("unit") == CANONICAL_UNIT and dimensions.get("width") and dimensions.get("height"):
        return [0.0, 0.0, round(float(dimensions["width"]), 3), round(float(dimensions["height"]), 3)]
    if dimensions.get("width_pt") and dimensions.get("height_pt"):
        return [0.0, 0.0, round(float(dimensions["width_pt"]), 3), round(float(dimensions["height_pt"]), 3)]
    if dimensions.get("width") and dimensions.get("height"):
        return [
            0.0,
            0.0,
            round(float(dimensions["width"]) / max(1e-9, sx), 3),
            round(float(dimensions["height"]) / max(1e-9, sy), 3),
        ]
    return None


def _set_bbox_from_pt(unit: dict, bbox_pt, sx: float, sy: float, dpi: float,
                      reading_index: int | None = None) -> None:
    if not (isinstance(bbox_pt, (list, tuple)) and len(bbox_pt) == 4):
        return
    bbox = [round(float(v), 3) for v in bbox_pt]
    unit["geometry"].update({
        "bbox": bbox,
        "bbox_unit": CANONICAL_UNIT,
        "bbox_px": bbox_pt_to_px(bbox, sx, sy),
        "bbox_px_dpi": dpi,
    })
    if reading_index is not None:
        unit["geometry"]["reading_order_index"] = reading_index
        unit["geometry"]["render_order_index"] = reading_index


def _span_text(span: dict) -> str:
    return str(span.get("text") or span.get("texte") or "")


def merge_phrase_spans(spans: list) -> list:
    """Fusionne les spans consécutifs de style identique au sein d'une phrase.

    Les extracteurs produisent souvent un span par mot (voire par espace) :
    une page TOC peut générer des milliers de spans identiques en style.
    La fusion est sans perte : le style est préservé, le texte concaténé,
    la bbox unionnée. Un changement de style coupe toujours la fusion.
    """
    merged: list[dict] = []
    for span in spans or []:
        if not isinstance(span, dict):
            continue
        if merged and _span_style_signature(span) == _span_style_signature(merged[-1]):
            acc = merged[-1]
            prev_text = _span_text(acc)
            text = _span_text(span)
            separator = ""
            if prev_text and text and not prev_text[-1].isspace() and not text[0].isspace():
                # Espace implicite : trou horizontal notable entre les bboxes.
                acc_bbox, span_bbox = acc.get("bbox"), span.get("bbox")
                if (isinstance(acc_bbox, (list, tuple)) and isinstance(span_bbox, (list, tuple))
                        and len(acc_bbox) == 4 and len(span_bbox) == 4):
                    gap = float(span_bbox[0]) - float(acc_bbox[2])
                    height = max(1.0, float(acc_bbox[3]) - float(acc_bbox[1]))
                    if gap > 0.15 * height:
                        separator = " "
            acc["text"] = prev_text + separator + text
            acc.pop("texte", None)
            acc["bbox"] = _union_bbox(acc.get("bbox"), span.get("bbox"))
            acc["bbox_pt"] = _union_bbox(acc.get("bbox_pt"), span.get("bbox_pt"))
            acc.setdefault("merged_from", [acc.get("id")])
            acc["merged_from"].append(span.get("id"))
            scores = [s for s in (acc.get("score"), span.get("score")) if s is not None]
            if scores:
                acc["score"] = min(scores)
        else:
            merged.append(dict(span))
    return merged


def _visual_dims_pt(node: dict, sx: float, sy: float):
    """Dimensions (largeur, hauteur) en points d'un objet visuel legacy."""
    bbox_pt = node.get("bbox_pt")
    if isinstance(bbox_pt, (list, tuple)) and len(bbox_pt) == 4:
        return float(bbox_pt[2]) - float(bbox_pt[0]), float(bbox_pt[3]) - float(bbox_pt[1])
    bbox = node.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return (
            (float(bbox[2]) - float(bbox[0])) / max(1e-9, sx),
            (float(bbox[3]) - float(bbox[1])) / max(1e-9, sy),
        )
    return 0.0, 0.0


def is_significant_visual(node: dict, sx: float, sy: float) -> bool:
    """Un visuel est signifiant s'il n'est pas un trait décoratif.

    Filets de bordure, règles et points : min(dim) < MIN_VISUAL_DIM_PT ou
    surface < MIN_VISUAL_AREA_PT2 → pas d'unité canonique (le background
    et compatibility.legacy les préservent déjà).
    """
    width, height = _visual_dims_pt(node, sx, sy)
    return (
        min(width, height) >= MIN_VISUAL_DIM_PT
        and width * height >= MIN_VISUAL_AREA_PT2
    )


def _unit_geometry(unit: dict, node: dict, sx: float, sy: float, dpi: float,
                   reading_index: int) -> None:
    bbox_pt, bbox_px = normalize_bbox_to_pt(
        node.get("bbox_pt") or node.get("bbox"),
        node.get("bbox_unit") or node.get("bbox_source_unit")
        or ("pt" if node.get("bbox_pt") is not None else None),
        sx, sy
    )
    geo = unit["geometry"]
    geo["bbox"] = bbox_pt
    geo["bbox_unit"] = CANONICAL_UNIT
    geo["bbox_px"] = bbox_px
    geo["bbox_px_dpi"] = dpi if bbox_px is not None else None
    if bbox_pt is not None and (
        float(bbox_pt[2]) <= float(bbox_pt[0]) or float(bbox_pt[3]) <= float(bbox_pt[1])
    ):
        # Trait vectoriel fin ou point : géométrie dégénérée, à étudier en aval.
        geo["degenerate"] = True
    geo["rotation"] = node.get("rotation") or 0
    geo["reading_order_index"] = reading_index
    geo["render_order_index"] = reading_index


def _aggregate_lines_text(node: dict) -> str | None:
    """Texte d'un bloc agrégé depuis ses lignes quand il n'en porte pas."""
    parts = []
    for line in node.get("lines") or []:
        if isinstance(line, dict):
            text = line.get("line_text") or line.get("text")
            if text and str(text).strip():
                parts.append(str(text).strip())
    return " ".join(parts) or None


def _unit_content(unit: dict, node: dict, language: str | None) -> None:
    text = (
        node.get("text")
        or node.get("texte")
        or node.get("line_text")
        or node.get("label")
        or _aggregate_lines_text(node)
    )
    unit["content"].update({
        "text": text,
        "raw_text": node.get("raw_text") or text,
        "normalized_text": node.get("normalized_text") or text,
        "translated_text": node.get("translated_text"),
        "language": node.get("language") or language,
    })


def _token_semantic_kind(text: str, level: str) -> str:
    token = str(text or "")
    if not token.strip():
        return "space"
    if level == "char":
        if token.isspace():
            return "space"
        if token.isdigit():
            return "digit"
        if token.isalpha():
            return "letter"
        return "symbol"
    if re.fullmatch(r"\d+([.,]\d+)*", token):
        return "number"
    if re.fullmatch(r"[\[\(]?\d+[\]\)]?", token):
        return "reference_marker"
    if re.fullmatch(r"[A-Z]{2,}([-/][A-Z0-9]+)*", token):
        return "acronym"
    if re.search(r"[=∑∫√≤≥±×÷∞∂λµπσΔαβγ]", token):
        return "math_token"
    if re.fullmatch(r"\W+", token):
        return "symbol"
    return "word"


def _toc_unit_role(text: str, level: str) -> tuple[str | None, str | None, bool | None]:
    normalized = re.sub(r"\s+", " ", str(text or "").strip())
    if not normalized:
        return None, None, None
    if normalized in TOC_BULLET_MARKERS:
        return "toc_bullet_marker", "marker", False
    if TOC_TITLE_RE.fullmatch(normalized):
        return "toc_title", "title", True
    if "." in normalized and TOC_SECTION_NUMBER_RE.fullmatch(normalized):
        return "toc_section_number", "section_number", False
    if TOC_PAGE_REFERENCE_RE.fullmatch(normalized):
        return "toc_page_reference", "page_reference", False
    if level in {"block", "line", "phrase", "span"} and re.search(r"[A-Za-zÀ-ÿ]", normalized):
        return "toc_entry", "entry", True
    return None, None, None


def _split_words(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", str(text or ""))]


def _slice_bbox_pt(bbox_pt, start: int, end: int, total: int):
    if not (isinstance(bbox_pt, (list, tuple)) and len(bbox_pt) == 4 and total > 0):
        return None
    x0, y0, x1, y1 = [float(v) for v in bbox_pt]
    width = max(0.001, x1 - x0)
    left = x0 + width * max(0.0, min(1.0, start / total))
    right = x0 + width * max(0.0, min(1.0, end / total))
    if right <= left:
        right = left + max(0.25, width / max(1, total))
    return [round(left, 3), round(y0, 3), round(right, 3), round(y1, 3)]


def _char_text(char: dict) -> str:
    return str(char.get("c") or char.get("text") or "")


def _char_bbox_pt(char: dict):
    bbox_pt = char.get("bbox_pt")
    if isinstance(bbox_pt, (list, tuple)) and len(bbox_pt) == 4:
        return [float(v) for v in bbox_pt]
    return None


def _char_center_in_bbox(char: dict, bbox_pt, tolerance_pt: float = 1.0) -> bool:
    char_bbox = _char_bbox_pt(char)
    if not (
        isinstance(char_bbox, (list, tuple)) and len(char_bbox) == 4
        and isinstance(bbox_pt, (list, tuple)) and len(bbox_pt) == 4
    ):
        return False
    cx = (float(char_bbox[0]) + float(char_bbox[2])) / 2.0
    cy = (float(char_bbox[1]) + float(char_bbox[3])) / 2.0
    return (
        float(bbox_pt[0]) - tolerance_pt <= cx <= float(bbox_pt[2]) + tolerance_pt
        and float(bbox_pt[1]) - tolerance_pt <= cy <= float(bbox_pt[3]) + tolerance_pt
    )


def _chars_for_span(span: dict, line_chars: list[dict]) -> list[dict]:
    """Retourne les caractères natifs qui appartiennent au span courant."""
    if not line_chars:
        return []
    span_bbox = span.get("bbox_pt")
    chars = [ch for ch in line_chars if isinstance(ch, dict) and _char_center_in_bbox(ch, span_bbox)]
    if not chars:
        return []
    chars.sort(key=lambda ch: (
        (_char_bbox_pt(ch) or [0, 0, 0, 0])[1],
        (_char_bbox_pt(ch) or [0, 0, 0, 0])[0],
    ))
    span_text = _span_text(span)
    chars_text = "".join(_char_text(ch) for ch in chars)
    if span_text and chars_text and chars_text != span_text:
        # Les offsets word/char sont indexés dans span_text. Si la chaîne des
        # chars natifs diverge, utiliser ces bboxes produirait un audit faux.
        return []
    return chars


def _bbox_from_native_chars(chars: list[dict], start: int, end: int):
    selected = []
    for ch in chars[start:end]:
        if not isinstance(ch, dict) or not _char_text(ch).strip():
            continue
        bbox = _char_bbox_pt(ch)
        if bbox is not None:
            selected.append(bbox)
    if not selected:
        return None
    bbox = selected[0]
    for other in selected[1:]:
        bbox = _union_bbox(bbox, other)
    return [round(float(v), 3) for v in bbox]


def _native_char_bbox(chars: list[dict], offset: int):
    if 0 <= offset < len(chars):
        bbox = _char_bbox_pt(chars[offset])
        if bbox is not None:
            return [round(float(v), 3) for v in bbox]
    return None


def _make_text_child_unit(*, unit_id: str, level: str, parent_id: str, text: str,
                          bbox_pt, source_node: dict, sx: float, sy: float,
                          dpi: float, reading_index: int, page_context: dict,
                          language: str | None, created_by: str,
                          source_default: str | None) -> dict:
    node = dict(source_node or {})
    node["text"] = text
    node["bbox"] = bbox_pt
    node["bbox_pt"] = bbox_pt
    node["bbox_unit"] = CANONICAL_UNIT
    node["semantic_kind"] = _token_semantic_kind(text, level)
    unit = _make_unit(
        unit_id=unit_id, level=level, parent_id=parent_id, node=node,
        sx=sx, sy=sy, dpi=dpi, reading_index=reading_index,
        page_context=page_context, language=language,
        created_by=created_by, source_default=source_default,
    )
    unit["understanding"]["semantic_kind"] = node["semantic_kind"]
    unit["policy"].update({
        "translatable": False,
        "translation_strategy": "lexical_context",
        "render_policy": "inherit_parent",
        "coverage_required": "normal",
        "non_translatable_reason": f"{level}_granularity_context",
    })
    return unit


def _unit_extraction(unit: dict, node: dict, default_source: str | None) -> None:
    unit["extraction"].update({
        "source": node.get("source") or node.get("source_kind") or default_source,
        "source_kind": node.get("source_kind"),
        "confidence": clamp_confidence(
            node.get("confidence")
            or node.get("score")
            or node.get("ocr_confidence_mean")
        ),
        "ocr_confidence_mean": clamp_confidence(node.get("ocr_confidence_mean")),
        "native_confidence": clamp_confidence(node.get("native_confidence")),
        "dedupe_status": node.get("dedupe_status") or "kept",
    })


def _unit_understanding(unit: dict, node: dict, page_context: dict) -> None:
    role = node.get("role")
    object_type = node.get("object_type") or node.get("unit_type")
    object_class = node.get("object_class")
    semantic_kind = node.get("phrase_semantics") or node.get("semantic_kind")
    text = unit.get("content", {}).get("text")
    page_role = page_context.get("page_role")

    if page_role == "toc":
        toc_role, toc_kind, translatable = _toc_unit_role(text, unit.get("level") or "")
        if toc_role:
            role = toc_role
            object_type = toc_role
            object_class = toc_role
            semantic_kind = toc_kind or semantic_kind
            if translatable is False:
                unit["policy"].update({
                    "translatable": False,
                    "translation_strategy": "exact_preserve",
                    "coverage_required": "strict",
                    "render_policy": "anchored_text",
                })

    unit["understanding"].update({
        "role": role,
        "object_type": object_type,
        "object_class": object_class,
        "page_family": page_context.get("page_family"),
        "layout_type": page_context.get("layout_type"),
        "document_type": page_context.get("document_type"),
        "page_role": page_role,
        "region_memberships": list(node.get("region_memberships") or []),
        "structure_hints": dict(node.get("structure_hints") or {}),
        "semantic_kind": semantic_kind,
    })


def _unit_policy(unit: dict, node: dict) -> None:
    contract = node.get("translation_contract") or {}
    unit["policy"].update({
        "translatable": node.get("translatable", contract.get("translatable")),
        "translation_strategy": node.get("translation_strategy")
        or contract.get("translation_strategy"),
        "render_policy": node.get("render_policy") or contract.get("render_policy"),
        "coverage_required": node.get("coverage_required")
        or contract.get("coverage_required"),
        "preserve_exact_text": bool(
            node.get("preserve_exact_text") or contract.get("preserve_exact_text")
        ),
        "preserve_visual": bool(
            node.get("preserve_visual") or contract.get("preserve_visual")
        ),
    })
    if contract.get("unit_type") or node.get("unit_type"):
        unit["policy"]["unit_type"] = node.get("unit_type") or contract.get("unit_type")


def _apply_toc_policy(unit: dict, page_context: dict) -> None:
    if page_context.get("page_role") != "toc":
        return
    toc_role, toc_kind, translatable = _toc_unit_role(
        unit.get("content", {}).get("text") or "",
        unit.get("level") or "",
    )
    if not toc_role:
        return
    if toc_role == "toc_entry" and translatable is True:
        text = str(unit.get("content", {}).get("text") or "")
        stripped = re.sub(r"^[■•▪◦‣⁃·◆▶▷]\s*", "", text)
        if stripped != text and re.search(r"[A-Za-zÀ-ÿ]", stripped):
            unit["content"]["text"] = stripped
            unit["content"]["normalized_text"] = stripped
    unit["understanding"]["role"] = toc_role
    unit["understanding"]["object_type"] = toc_role
    unit["understanding"]["object_class"] = toc_role
    unit["understanding"]["semantic_kind"] = toc_kind or unit["understanding"].get("semantic_kind")
    unit["policy"]["unit_type"] = toc_role
    if translatable is False:
        unit["policy"].update({
            "translatable": False,
            "translation_strategy": "exact_preserve",
            "coverage_required": "strict",
            "render_policy": "anchored_text",
        })


def _make_unit(*, unit_id: str, level: str, parent_id: str | None, node: dict,
               sx: float, sy: float, dpi: float, reading_index: int,
               page_context: dict, language: str | None,
               created_by: str, source_default: str | None) -> dict:
    unit = empty_unit(unit_id, level, parent_id)
    _unit_content(unit, node, language)
    _unit_geometry(unit, node, sx, sy, dpi, reading_index)
    unit["visual"]["style"] = normalize_style(
        node.get("style"), sx=sx,
        source=node.get("source") or node.get("source_kind") or source_default,
    )
    unit["visual"]["style_class"] = node.get("style_class")
    unit["visual"]["style_confidence"] = clamp_confidence(node.get("style_confidence"))
    _unit_extraction(unit, node, source_default)
    _unit_understanding(unit, node, page_context)
    _unit_policy(unit, node)
    _apply_toc_policy(unit, page_context)
    unit["provenance"]["created_by"] = created_by
    unit["lifecycle"]["created_at_stage"] = created_by
    if node.get("merged_from"):
        unit["lifecycle"]["merged_from"] = list(node.get("merged_from"))
    legacy_id = node.get("id")
    if legacy_id is not None:
        unit["legacy_id"] = legacy_id
    return unit


def _region_level(region_type: str) -> str:
    mapping = {
        "formula": "region",
        "formula_region": "region",
        "formula_candidate_region": "region",
        "code": "region",
        "code_region": "region",
        "code_candidate_region": "region",
        "visual_candidate_region": "region",
        "protected_visual_region": "region",
        "table_region": "table",
        "table_cell": "cell",
    }
    return mapping.get(str(region_type or ""), "region")


def build_region_units(regions: list[dict], *, page_unit_id: str,
                       page_index: int = 0, sx: float = 1.0, sy: float = 1.0,
                       dpi: float = 150, language: str | None = None,
                       start_index: int = 0, page_context: dict | None = None) -> list[dict]:
    """Matérialise les régions/zones détectées comme unités canoniques.

    Les régions ne sont pas seulement des annotations spatiales : formules,
    code, tableaux, cellules, figures et zones non textuelles deviennent des
    unités de premier rang consommables par traduction/reconstruction/QA.
    """
    page_context = page_context or {}
    units: list[dict] = []
    counter = start_index
    for idx, region in enumerate(regions or []):
        if not isinstance(region, dict):
            continue
        region_type = region.get("region_type") or "body_region"
        level = _region_level(region_type)
        unit_id = f"{region.get('region_id') or f'p{page_index + 1:03d}_region_{idx + 1:03d}'}_unit"
        node = {
            "id": region.get("region_id"),
            "bbox": region.get("bbox"),
            "bbox_unit": CANONICAL_UNIT,
            "source": region.get("source"),
            "source_kind": "pageprint_region",
            "confidence": region.get("confidence"),
            "role": region.get("role") or region_type,
            "object_type": region.get("object_type") or region_type,
            "object_class": region.get("object_class") or level,
            "structure_hints": {
                "region_type": region_type,
                "region_id": region.get("region_id"),
                "materialized_from_region": True,
                "policy_pending": bool(region.get("policy_pending")),
                "observation_only": bool(region.get("observation_only")),
            },
        }
        region_policy = dict(region.get("policy") or {})
        node.update({
            "translatable": region_policy.get("translatable"),
            "translation_strategy": region_policy.get("translation_strategy"),
            "render_policy": region_policy.get("render_policy"),
            "coverage_required": "strict",
            "preserve_visual": region_policy.get("must_preserve_visual"),
            "preserve_original_pixels": region_policy.get("preserve_original_pixels"),
            "policy_pending": region_policy.get("policy_pending"),
        })
        unit = _make_unit(
            unit_id=unit_id,
            level=level,
            parent_id=page_unit_id,
            node=node,
            sx=sx,
            sy=sy,
            dpi=dpi,
            reading_index=counter,
            page_context=page_context,
            language=language,
            created_by="pageprint.region_materializer",
            source_default=region.get("source"),
        )
        unit["relations"]["parent_region_id"] = region.get("region_id")
        unit["understanding"]["region_memberships"] = [{
            "region_id": region.get("region_id"),
            "region_type": region_type,
            "overlap_ratio": 1.0,
            "membership_role": "materialized_region",
            "confidence": region.get("confidence"),
        }]
        unit["evidence"] = {
            "sources": [{
                "source": region.get("source") or "region_index",
                "claim": region_type,
                "confidence": region.get("confidence"),
            }],
            "resolved_as": level,
            "resolution_rule": "region_materialized_as_canonical_unit",
            "confidence": region.get("confidence"),
        }
        units.append(unit)
        counter += 1
    return units


def build_units(page_structure: dict, *, page_index: int = 0,
                sx: float = 1.0, sy: float = 1.0,
                dpi: float = 150, language: str | None = None,
                merge_spans: bool = True,
                stats: dict | None = None) -> list[dict]:
    """Construit la liste canonique d'unités depuis la structure legacy.

    Hiérarchie produite : page → block → line → phrase → span → word → char,
    plus les unités image/drawing signifiantes. Les spans consécutifs de
    style identique sont fusionnés (merge_spans=True), puis les mots et
    caractères sont synthétisés pour garantir la granularité fine du contrat.
    """
    if stats is not None:
        stats.setdefault("spans_before_merge", 0)
        stats.setdefault("spans_after_merge", 0)
        stats.setdefault("words_created", 0)
        stats.setdefault("chars_created", 0)
        stats.setdefault("visuals_total", 0)
        stats.setdefault("visuals_filtered_decorative", 0)
    page_context = {
        "page_family": page_structure.get("page_family"),
        "layout_type": page_structure.get("layout_type"),
        "document_type": page_structure.get("document_type"),
        "page_role": page_structure.get("page_role"),
    }
    prefix = f"p{page_index + 1:03d}"
    units: list[dict] = []
    counter = {"i": 0}

    def next_index() -> int:
        counter["i"] += 1
        return counter["i"] - 1

    page_unit_id = f"{prefix}_page"
    page_unit = empty_unit(page_unit_id, "page", None)
    page_unit["content"].update({
        "text": None,
        "raw_text": None,
        "normalized_text": None,
        "language": language,
    })
    _set_bbox_from_pt(page_unit, _page_bbox_pt(page_structure, sx, sy), sx, sy, dpi, next_index())
    page_unit["understanding"].update({
        "role": page_structure.get("page_role") or "page",
        "object_type": "page",
        "object_class": "page",
        "page_family": page_context.get("page_family"),
        "layout_type": page_context.get("layout_type"),
        "document_type": page_context.get("document_type"),
        "structure_hints": {
            "page_role": page_structure.get("page_role"),
            "format_probable": page_structure.get("format_probable"),
        },
    })
    page_unit["policy"].update({
        "translatable": False,
        "translation_strategy": "page_container",
        "render_policy": "page_container",
        "coverage_required": "normal",
        "non_translatable_reason": "page_container",
    })
    page_unit["extraction"].update({
        "source": "pageprint",
        "source_kind": "page_container",
        "confidence": 1.0,
    })
    page_unit["provenance"]["created_by"] = "pageprint.unit_factory"
    page_unit["lifecycle"]["created_at_stage"] = "pageprint.unit_factory"
    units.append(page_unit)

    for b_idx, block in enumerate(page_structure.get("blocks") or []):
        if not isinstance(block, dict):
            continue
        block_id = f"{prefix}_block_{b_idx + 1:03d}"
        block_unit = _make_unit(
            unit_id=block_id, level="block", parent_id=page_unit_id, node=block,
            sx=sx, sy=sy, dpi=dpi, reading_index=next_index(),
            page_context=page_context, language=language,
            created_by="pageprint.unit_factory", source_default=block.get("source_kind"),
        )
        page_unit["children_ids"].append(block_id)
        units.append(block_unit)

        for l_idx, line in enumerate(block.get("lines") or []):
            if not isinstance(line, dict):
                continue
            line_id = f"{block_id}_line_{l_idx + 1:03d}"
            line_unit = _make_unit(
                unit_id=line_id, level="line", parent_id=block_id, node=line,
                sx=sx, sy=sy, dpi=dpi, reading_index=next_index(),
                page_context=page_context, language=language,
                created_by="pageprint.unit_factory",
                source_default=block.get("source_kind"),
            )
            block_unit["children_ids"].append(line_id)
            units.append(line_unit)

            for p_idx, phrase in enumerate(line.get("phrases") or []):
                if not isinstance(phrase, dict):
                    continue
                phrase_id = f"{line_id}_phrase_{p_idx + 1:03d}"
                phrase_unit = _make_unit(
                    unit_id=phrase_id, level="phrase", parent_id=line_id, node=phrase,
                    sx=sx, sy=sy, dpi=dpi, reading_index=next_index(),
                    page_context=page_context, language=language,
                    created_by="pageprint.unit_factory",
                    source_default=block.get("source_kind"),
                )
                line_unit["children_ids"].append(phrase_id)
                units.append(phrase_unit)

                raw_spans = phrase.get("spans") or []
                if stats is not None:
                    stats["spans_before_merge"] += len(raw_spans)
                if merge_spans:
                    raw_spans = merge_phrase_spans(raw_spans)
                if stats is not None:
                    stats["spans_after_merge"] += len(raw_spans)
                line_chars = [
                    ch for ch in (line.get("chars") or [])
                    if isinstance(ch, dict) and _char_bbox_pt(ch) is not None
                ]

                for s_idx, span in enumerate(raw_spans):
                    if not isinstance(span, dict):
                        continue
                    span_id = f"{phrase_id}_span_{s_idx + 1:03d}"
                    span_unit = _make_unit(
                        unit_id=span_id, level="span", parent_id=phrase_id, node=span,
                        sx=sx, sy=sy, dpi=dpi, reading_index=next_index(),
                        page_context=page_context, language=language,
                        created_by="pageprint.unit_factory",
                        source_default=block.get("source_kind"),
                    )
                    phrase_unit["children_ids"].append(span_id)
                    units.append(span_unit)

                    span_text = _span_text(span)
                    span_bbox_pt = span_unit["geometry"].get("bbox")
                    span_chars = _chars_for_span(span, line_chars)
                    words = _split_words(span_text)
                    total_len = max(1, len(span_text))
                    for w_idx, (word_text, start, end) in enumerate(words):
                        word_id = f"{span_id}_word_{w_idx + 1:03d}"
                        word_bbox = _bbox_from_native_chars(span_chars, start, end)
                        if word_bbox is None:
                            word_bbox = _slice_bbox_pt(span_bbox_pt, start, end, total_len)
                        word_unit = _make_text_child_unit(
                            unit_id=word_id, level="word", parent_id=span_id,
                            text=word_text, bbox_pt=word_bbox, source_node=span,
                            sx=sx, sy=sy, dpi=dpi, reading_index=next_index(),
                            page_context=page_context, language=language,
                            created_by="pageprint.tokenizer",
                            source_default=block.get("source_kind"),
                        )
                        span_unit["children_ids"].append(word_id)
                        units.append(word_unit)
                        if stats is not None:
                            stats["words_created"] += 1

                        rel_start = start
                        for c_idx, char_offset in enumerate(range(start, end)):
                            char_text = span_text[char_offset]
                            char_id = f"{word_id}_char_{c_idx + 1:03d}"
                            char_bbox = _native_char_bbox(span_chars, char_offset)
                            if char_bbox is None:
                                char_bbox = _slice_bbox_pt(
                                    span_bbox_pt, char_offset, char_offset + 1, total_len
                                )
                            char_unit = _make_text_child_unit(
                                unit_id=char_id, level="char", parent_id=word_id,
                                text=char_text, bbox_pt=char_bbox, source_node=span,
                                sx=sx, sy=sy, dpi=dpi, reading_index=next_index(),
                                page_context=page_context, language=language,
                                created_by="pageprint.tokenizer",
                                source_default=block.get("source_kind"),
                            )
                            char_unit["relations"]["char_offset_in_span"] = char_offset
                            char_unit["relations"]["char_offset_in_word"] = char_offset - rel_start
                            word_unit["children_ids"].append(char_id)
                            units.append(char_unit)
                            if stats is not None:
                                stats["chars_created"] += 1

    # Unités visuelles signifiantes (images / drawings) ; les traits
    # décoratifs (filets, règles) sont filtrés.
    images = page_structure.get("images") or []
    drawings = page_structure.get("drawings") or []
    if stats is not None:
        stats["visuals_total"] += len(images) + len(drawings)

    for i_idx, image in enumerate(images):
        if not isinstance(image, dict):
            continue
        if not is_significant_visual(image, sx, sy):
            if stats is not None:
                stats["visuals_filtered_decorative"] += 1
            continue
        unit = _make_unit(
            unit_id=f"{prefix}_image_{i_idx + 1:03d}", level="image", parent_id=page_unit_id,
            node=image, sx=sx, sy=sy, dpi=dpi, reading_index=next_index(),
            page_context=page_context, language=language,
            created_by="pageprint.unit_factory", source_default="native_pdf",
        )
        unit["policy"].update({
            "translatable": False,
            "translation_strategy": "exact_preserve",
            "render_policy": "fixed_preserve",
            "coverage_required": "strict",
            "preserve_visual": True,
        })
        page_unit["children_ids"].append(unit["unit_id"])
        units.append(unit)

    for d_idx, drawing in enumerate(drawings):
        if not isinstance(drawing, dict):
            continue
        if not is_significant_visual(drawing, sx, sy):
            if stats is not None:
                stats["visuals_filtered_decorative"] += 1
            continue
        unit = _make_unit(
            unit_id=f"{prefix}_drawing_{d_idx + 1:03d}", level="drawing", parent_id=page_unit_id,
            node=drawing, sx=sx, sy=sy, dpi=dpi, reading_index=next_index(),
            page_context=page_context, language=language,
            created_by="pageprint.unit_factory", source_default="native_pdf",
        )
        unit["policy"].update({
            "translatable": False,
            "translation_strategy": "exact_preserve",
            "render_policy": "fixed_preserve",
            "coverage_required": "strict",
            "preserve_visual": True,
        })
        page_unit["children_ids"].append(unit["unit_id"])
        units.append(unit)

    # Chaînage previous/next dans l'ordre de lecture.
    ordered = sorted(
        (u for u in units if u["geometry"].get("reading_order_index") is not None),
        key=lambda u: u["geometry"]["reading_order_index"],
    )
    for prev, nxt in zip(ordered, ordered[1:]):
        prev["relations"]["next_unit_id"] = nxt["unit_id"]
        nxt["relations"]["previous_unit_id"] = prev["unit_id"]

    return units
