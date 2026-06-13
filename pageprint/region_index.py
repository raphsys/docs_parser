"""PAGEPRINT Region Index — indexe régions/zones/objets et calcule les appartenances.

Il fusionne : regions, special_regions, non_text_zones, images, drawings,
tables, charts, formulas, code regions, layout_ai regions.

Puis calcule les appartenances :
    unit → region
    region → unit

C'est ainsi que les régions influencent le pipeline : elles deviennent des
observations spatiales/claims. Les politiques finales sont compilées plus tard.
"""

from __future__ import annotations

from .normalizer import clamp_confidence, normalize_bbox_to_pt
from .schema import CANONICAL_UNIT, MIN_VISUAL_AREA_PT2, MIN_VISUAL_DIM_PT

MEMBERSHIP_OVERLAP_THRESHOLD = 0.5

# Régions purement visuelles soumises au filtre des traits décoratifs.
VISUAL_REGION_TYPES = {
    "image_region",
    "drawing_region",
    "non_text_zone",
    "formula_candidate_region",
    "code_candidate_region",
    "visual_candidate_region",
}
PROTECTED_VISUAL_TYPES = {
    "formula",
    "formula_region",
    "equation",
    "math_expression",
    "chemical_formula",
    "symbolic_expression",
    "code",
    "code_region",
    "code_block",
    "inline_code",
    "algorithm_block",
    "special_notation",
    "table_formula_cell",
    "diagram_label_non_linguistic",
    "protected_visual",
    "protected_visual_region",
}


def _bbox_area(bbox) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _overlap_ratio(unit_bbox, region_bbox) -> float:
    """Ratio de la surface de l'unité couverte par la région."""
    area = _bbox_area(unit_bbox)
    if area <= 0.0:
        return 0.0
    x0 = max(float(unit_bbox[0]), float(region_bbox[0]))
    y0 = max(float(unit_bbox[1]), float(region_bbox[1]))
    x1 = min(float(unit_bbox[2]), float(region_bbox[2]))
    y1 = min(float(unit_bbox[3]), float(region_bbox[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / area


def _normalize_region_type(raw_type: str | None) -> str:
    mapping = {
        "formula": "formula_candidate_region",
        "formula_region": "formula_candidate_region",
        "formula_candidate_region": "formula_candidate_region",
        "equation": "formula_candidate_region",
        "math_expression": "formula_candidate_region",
        "chemical_formula": "formula_candidate_region",
        "symbolic_expression": "formula_candidate_region",
        "code": "code_candidate_region",
        "code_region": "code_candidate_region",
        "code_candidate_region": "code_candidate_region",
        "code_block": "code_candidate_region",
        "inline_code": "code_candidate_region",
        "algorithm_block": "code_candidate_region",
        "special_notation": "formula_candidate_region",
        "table_formula_cell": "formula_candidate_region",
        "diagram_label_non_linguistic": "diagram_region",
        "protected_visual": "visual_candidate_region",
        "protected_visual_region": "visual_candidate_region",
        "visual_candidate_region": "visual_candidate_region",
        "table_candidate_region": "table_candidate_region",
        "table": "table_region",
        "table_region": "table_region",
        "table_cell": "table_cell",
        "chart": "chart_region",
        "chart_region": "chart_region",
        "chart_tick": "chart_tick",
        "figure": "figure_region",
        "image": "image_region",
        "drawing": "drawing_region",
        "non_text_zone": "non_text_zone",
        "toc": "toc_region",
        "caption": "caption_region",
        "annotation": "annotation_region",
        "header": "header_region",
        "footer": "footer_region",
        "text": "body_region",
        "body": "body_region",
    }
    if not raw_type:
        return "body_region"
    return mapping.get(str(raw_type), str(raw_type))


def _region_policy(region_type: str) -> dict:
    if region_type in {"formula_candidate_region", "code_candidate_region", "visual_candidate_region"}:
        return {
            "unit_type": region_type,
            "object_type": region_type.replace("_region", ""),
            "translatable": None,
            "translation_strategy": "claim_pending",
            "render_policy": "candidate_region",
            "coverage_required": "strict",
            "policy_pending": True,
            "must_preserve_visual": False,
            "must_exclude_from_translation_flow": False,
        }
    if region_type in {"image_region", "drawing_region", "non_text_zone"}:
        return {
            "translatable": False,
            "translation_strategy": "exact_preserve",
            "render_policy": "fixed_preserve",
            "must_preserve_visual": True,
            "must_exclude_from_translation_flow": True,
        }
    if region_type in {"chart_tick"}:
        return {
            "translatable": False,
            "translation_strategy": "exact_preserve",
            "render_policy": "anchored_text" if region_type == "code" else "fixed_preserve",
            "must_preserve_visual": region_type == "chart_tick",
            "must_exclude_from_translation_flow": True,
        }
    if region_type in {"table_region", "table_cell"}:
        return {
            "translatable": True,
            "translation_strategy": "layout_constrained",
            "render_policy": "anchored_text",
            "must_preserve_visual": False,
            "must_exclude_from_translation_flow": False,
        }
    return {
        "translatable": True,
        "translation_strategy": "layout_constrained",
        "render_policy": "anchored_text",
        "must_preserve_visual": False,
        "must_exclude_from_translation_flow": False,
    }


def _region_constraints(region_type: str) -> dict:
    fixed = region_type in {"image_region", "drawing_region", "non_text_zone", "chart_tick"}
    return {
        "preserve_bbox": fixed,
        "preserve_as_overlay": region_type in {"image_region", "drawing_region"},
        "allow_reflow": not fixed and region_type not in {"table_cell", "code"},
    }


def _collect_raw_regions(page_structure: dict) -> list[tuple[dict, str]]:
    """Retourne [(raw_region, source_collection)] depuis toutes les sources."""
    collected: list[tuple[dict, str]] = []
    layout = page_structure.get("layout") or {}

    for key in ("special_regions",):
        for raw in (page_structure.get(key) or layout.get(key) or []):
            if isinstance(raw, dict):
                collected.append((raw, "special_region_detector"))
    for raw in (page_structure.get("regions") or layout.get("regions") or []):
        if isinstance(raw, dict):
            collected.append((raw, raw.get("source") or "layout_ai"))
    for raw in (page_structure.get("zones") or layout.get("zones") or []):
        if isinstance(raw, dict):
            collected.append((raw, "zones"))
    for raw in (page_structure.get("non_text_zones") or []):
        if isinstance(raw, dict):
            collected.append(({**raw, "region_type": "non_text_zone"}, "native_pdf"))
        elif isinstance(raw, (list, tuple)) and len(raw) == 4:
            collected.append(({"bbox": list(raw), "region_type": "non_text_zone"}, "native_pdf"))
    for raw in (page_structure.get("images") or []):
        if isinstance(raw, dict):
            collected.append(({**raw, "region_type": "image"}, "native_pdf"))
    for raw in (page_structure.get("drawings") or []):
        if isinstance(raw, dict):
            collected.append(({**raw, "region_type": "drawing"}, "native_pdf"))
    return collected


def build_regions(page_structure: dict, *, page_index: int = 0,
                  sx: float = 1.0, sy: float = 1.0,
                  stats: dict | None = None) -> list[dict]:
    """Construit la liste canonique de régions (bbox en points).

    Deux réductions de bruit :
    - les régions visuelles décoratives (traits/filets sous les seuils
      MIN_VISUAL_DIM_PT / MIN_VISUAL_AREA_PT2) sont filtrées ;
    - les doublons exacts de bbox entre collections (l'extracteur natif
      enregistre chaque image à la fois comme image et comme non_text_zone)
      sont dédupliqués, la version typée gagnant sur non_text_zone.
    """
    if stats is not None:
        stats.setdefault("regions_filtered_decorative", 0)
        stats.setdefault("regions_deduplicated", 0)

    prefix = f"p{page_index + 1:03d}"
    regions: list[dict] = []
    type_counters: dict[str, int] = {}
    seen_visual_bboxes: dict[tuple, str] = {}

    collected = _collect_raw_regions(page_structure)
    # Les collections typées (image/drawing) passent avant non_text_zone
    # pour que le doublon générique soit celui qui est écarté.
    collected.sort(key=lambda item: 1 if (
        str((item[0].get("region_type") or item[0].get("type") or "")) == "non_text_zone"
    ) else 0)

    for raw, source in collected:
        region_type = _normalize_region_type(
            raw.get("region_type") or raw.get("type") or raw.get("kind")
            or raw.get("special_class")
        )
        bbox_pt, bbox_px = normalize_bbox_to_pt(
            raw.get("bbox"), raw.get("bbox_unit") or raw.get("bbox_source_unit"), sx, sy
        )
        if bbox_pt is None:
            continue

        if region_type in VISUAL_REGION_TYPES:
            width = float(bbox_pt[2]) - float(bbox_pt[0])
            height = float(bbox_pt[3]) - float(bbox_pt[1])
            if min(width, height) < MIN_VISUAL_DIM_PT or width * height < MIN_VISUAL_AREA_PT2:
                if stats is not None:
                    stats["regions_filtered_decorative"] += 1
                continue
            bbox_key = tuple(round(float(v), 1) for v in bbox_pt)
            if bbox_key in seen_visual_bboxes:
                if stats is not None:
                    stats["regions_deduplicated"] += 1
                continue
            seen_visual_bboxes[bbox_key] = region_type

        type_counters[region_type] = type_counters.get(region_type, 0) + 1
        region = {
            "region_id": f"{prefix}_region_{region_type}_{type_counters[region_type]:03d}",
            "region_type": region_type,
            "role": raw.get("role") or f"{region_type}",
            "bbox": bbox_pt,
            "bbox_unit": CANONICAL_UNIT,
            "bbox_px": bbox_px,
            "source": raw.get("source") or source,
            "confidence": clamp_confidence(raw.get("confidence") or raw.get("score")) or 0.5,
            "members": {
                "block_ids": [],
                "line_ids": [],
                "phrase_ids": [],
                "span_ids": [],
            },
            "policy": _region_policy(region_type),
            "constraints": _region_constraints(region_type),
        }
        if region_type in {"formula_candidate_region", "code_candidate_region", "visual_candidate_region"}:
            raw_type = raw.get("region_type") or raw.get("type") or raw.get("kind") or raw.get("special_class")
            object_type = raw.get("object_type") or raw.get("object_class") or raw_type or region_type
            region.update({
                "object_type": str(object_type),
                "object_class": raw.get("object_class") or raw.get("subtype") or str(object_type),
                "claim_type": raw.get("claim_type") or ("formula_candidate" if region_type == "formula_candidate_region" else "code_candidate" if region_type == "code_candidate_region" else "visual_candidate"),
                "policy_pending": True,
                "observation_only": True,
                "reason": raw.get("reason") or "candidate_region_observation",
                "detection_source": raw.get("detection_source") or raw.get("source") or source,
            })
        if raw.get("subtype"):
            region["subtype"] = raw.get("subtype")
        regions.append(region)

    return regions


def attach_region_memberships(units: list[dict], regions: list[dict], *,
                              threshold: float = MEMBERSHIP_OVERLAP_THRESHOLD) -> None:
    """Calcule unit → region et region → unit (in place).

    La sortie de classification devient ainsi disponible au niveau le plus fin
    (bloc, ligne, phrase, span).
    """
    member_keys = {
        "block": "block_ids",
        "line": "line_ids",
        "phrase": "phrase_ids",
        "span": "span_ids",
        "word": "word_ids",
        "char": "char_ids",
    }
    for region in regions:
        members = region.setdefault("members", {})
        members.setdefault("word_ids", [])
        members.setdefault("char_ids", [])
    for unit in units:
        bbox = (unit.get("geometry") or {}).get("bbox")
        if not bbox:
            continue
        memberships = []
        for region in regions:
            local_threshold = _membership_threshold(unit.get("level"), region.get("region_type"), threshold)
            ratio = _overlap_ratio(bbox, region["bbox"])
            if ratio >= local_threshold:
                coverage_mode = _coverage_mode(unit.get("level"), ratio)
                memberships.append({
                    "region_id": region["region_id"],
                    "region_type": region["region_type"],
                    "overlap_ratio": round(ratio, 3),
                    "coverage_mode": coverage_mode,
                    "membership_role": "inside" if coverage_mode == "full_coverage" else coverage_mode,
                    "action_hint": _action_hint(region.get("region_type"), coverage_mode),
                    "confidence": region.get("confidence"),
                })
                key = member_keys.get(unit.get("level"))
                if key:
                    region["members"][key].append(unit["unit_id"])
        memberships.sort(key=lambda m: m["overlap_ratio"], reverse=True)
        unit["understanding"]["region_memberships"] = memberships
        if memberships:
            unit["relations"]["parent_region_id"] = memberships[0]["region_id"]


def unit_has_region(unit: dict, region_type: str) -> bool:
    for membership in (unit.get("understanding") or {}).get("region_memberships") or []:
        if membership.get("region_type") == region_type:
            return True
    return False


def unit_has_protected_visual_region(unit: dict) -> bool:
    return unit_has_region(unit, "protected_visual_region")


def _membership_threshold(level: str | None, region_type: str | None, default: float) -> float:
    if region_type not in {"formula_candidate_region", "code_candidate_region", "visual_candidate_region"}:
        return default
    if level in {"block", "line"}:
        return 0.10
    return 0.55


def _coverage_mode(level: str | None, overlap_ratio: float) -> str:
    if overlap_ratio <= 0.0:
        return "none"
    if overlap_ratio < 0.10:
        return "incidental_overlap"
    thresholds = {
        "block": 0.90,
        "line": 0.85,
        "phrase": 0.80,
        "span": 0.75,
        "word": 0.75,
        "char": 0.75,
    }
    full_threshold = thresholds.get(level or "", 0.85)
    if overlap_ratio >= full_threshold:
        return "full_coverage"
    if overlap_ratio >= 0.10:
        return "partial_inline" if level in {"line", "phrase", "span", "word", "char"} else "dominant_overlap" if overlap_ratio >= 0.50 else "partial_inline"
    return "incidental_overlap"


def _action_hint(region_type: str | None, coverage_mode: str) -> str:
    if coverage_mode == "full_coverage":
        return "resolve_policy_for_unit"
    if region_type in {"formula_candidate_region", "code_candidate_region", "visual_candidate_region"}:
        return "protect_inline_token_not_parent"
    return "contextual_region_membership"
