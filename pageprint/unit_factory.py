"""PAGEPRINT Unit Factory — transforme blocs/lignes/phrases/spans en unités canoniques.

Entrée : page_structure["blocks"] (structure legacy).
Sortie : units[] — liste plate d'unités canoniques respectant UNIT_REQUIRED,
chaque unité portant les trois couches séparées :
    extraction brute / compréhension / politique de traitement.
"""

from __future__ import annotations

from .normalizer import (
    clamp_confidence,
    normalize_bbox_to_pt,
    normalize_style,
)
from .schema import (
    CANONICAL_UNIT,
    MIN_VISUAL_AREA_PT2,
    MIN_VISUAL_DIM_PT,
    empty_unit,
)


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
        node.get("bbox"), node.get("bbox_unit") or node.get("bbox_source_unit"), sx, sy
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
    unit["understanding"].update({
        "role": node.get("role"),
        "object_type": node.get("object_type") or node.get("unit_type"),
        "object_class": node.get("object_class"),
        "page_family": page_context.get("page_family"),
        "layout_type": page_context.get("layout_type"),
        "document_type": page_context.get("document_type"),
        "region_memberships": list(node.get("region_memberships") or []),
        "structure_hints": dict(node.get("structure_hints") or {}),
        "semantic_kind": node.get("phrase_semantics") or node.get("semantic_kind"),
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
    unit["provenance"]["created_by"] = created_by
    unit["lifecycle"]["created_at_stage"] = created_by
    if node.get("merged_from"):
        unit["lifecycle"]["merged_from"] = list(node.get("merged_from"))
    legacy_id = node.get("id")
    if legacy_id is not None:
        unit["legacy_id"] = legacy_id
    return unit


def build_units(page_structure: dict, *, page_index: int = 0,
                sx: float = 1.0, sy: float = 1.0,
                dpi: float = 150, language: str | None = None,
                merge_spans: bool = True,
                stats: dict | None = None) -> list[dict]:
    """Construit la liste canonique d'unités depuis la structure legacy.

    Hiérarchie produite : block → line → phrase → span, plus les unités
    image/drawing signifiantes (les traits décoratifs sont filtrés et
    comptés dans `stats`). Les spans consécutifs de style identique sont
    fusionnés (merge_spans=True).
    """
    if stats is not None:
        stats.setdefault("spans_before_merge", 0)
        stats.setdefault("spans_after_merge", 0)
        stats.setdefault("visuals_total", 0)
        stats.setdefault("visuals_filtered_decorative", 0)
    page_context = {
        "page_family": page_structure.get("page_family"),
        "layout_type": page_structure.get("layout_type"),
        "document_type": page_structure.get("document_type"),
    }
    prefix = f"p{page_index + 1:03d}"
    units: list[dict] = []
    counter = {"i": 0}

    def next_index() -> int:
        counter["i"] += 1
        return counter["i"] - 1

    for b_idx, block in enumerate(page_structure.get("blocks") or []):
        if not isinstance(block, dict):
            continue
        block_id = f"{prefix}_block_{b_idx + 1:03d}"
        block_unit = _make_unit(
            unit_id=block_id, level="block", parent_id=None, node=block,
            sx=sx, sy=sy, dpi=dpi, reading_index=next_index(),
            page_context=page_context, language=language,
            created_by="pageprint.unit_factory", source_default=block.get("source_kind"),
        )
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
            unit_id=f"{prefix}_image_{i_idx + 1:03d}", level="image", parent_id=None,
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
        units.append(unit)

    for d_idx, drawing in enumerate(drawings):
        if not isinstance(drawing, dict):
            continue
        if not is_significant_visual(drawing, sx, sy):
            if stats is not None:
                stats["visuals_filtered_decorative"] += 1
            continue
        unit = _make_unit(
            unit_id=f"{prefix}_drawing_{d_idx + 1:03d}", level="drawing", parent_id=None,
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
