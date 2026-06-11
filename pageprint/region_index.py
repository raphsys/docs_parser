"""PAGEPRINT Region Index — indexe régions/zones/objets et calcule les appartenances.

Il fusionne : regions, special_regions, non_text_zones, images, drawings,
tables, charts, formulas, code regions, layout_ai regions.

Puis calcule les appartenances :
    unit → region
    region → unit

C'est ainsi que les régions influencent vraiment le pipeline : la région
devient plus forte que le simple texte.
"""

from __future__ import annotations

from .normalizer import clamp_confidence, normalize_bbox_to_pt
from .schema import CANONICAL_UNIT, MIN_VISUAL_AREA_PT2, MIN_VISUAL_DIM_PT

MEMBERSHIP_OVERLAP_THRESHOLD = 0.5

# Régions purement visuelles soumises au filtre des traits décoratifs.
VISUAL_REGION_TYPES = {"image_region", "drawing_region", "non_text_zone"}


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
        "formula": "formula",
        "formula_region": "formula",
        "code": "code",
        "code_region": "code",
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
    }
    if not raw_type:
        return "body_region"
    return mapping.get(str(raw_type), str(raw_type))


def _region_policy(region_type: str) -> dict:
    if region_type in {"formula", "image_region", "drawing_region", "non_text_zone"}:
        return {
            "translatable": False,
            "translation_strategy": "exact_preserve",
            "render_policy": "fixed_preserve",
            "must_preserve_visual": True,
            "must_exclude_from_translation_flow": True,
        }
    if region_type in {"code", "chart_tick"}:
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
    fixed = region_type in {"formula", "image_region", "drawing_region", "non_text_zone", "chart_tick"}
    return {
        "preserve_bbox": fixed,
        "preserve_as_overlay": region_type in {"formula", "image_region", "drawing_region"},
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
    }
    for unit in units:
        bbox = (unit.get("geometry") or {}).get("bbox")
        if not bbox:
            continue
        memberships = []
        for region in regions:
            ratio = _overlap_ratio(bbox, region["bbox"])
            if ratio >= threshold:
                memberships.append({
                    "region_id": region["region_id"],
                    "region_type": region["region_type"],
                    "overlap_ratio": round(ratio, 3),
                    "membership_role": "inside" if ratio >= 0.9 else "partial",
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
