from __future__ import annotations

from .common import bbox_of, eligible_text_units, role_of, text_of


FIGURE_REGION_MARKERS = ("image_region", "drawing_region", "diagram", "chart", "figure")


def _inside_preserved_figure(unit: dict) -> bool:
    """Return True for text units that are internal labels of a figure.

    Formula extraction must not sweep short labels from diagrams/charts into
    logical formula units. Those labels are part of the preserved visual object;
    repainting them separately is one of the main sources of collisions in the
    reconstructed pages.
    """
    for membership in (unit.get("understanding") or {}).get("region_memberships") or []:
        region_type = str(membership.get("region_type") or "").lower()
        if not any(marker in region_type for marker in FIGURE_REGION_MARKERS):
            continue
        if membership.get("coverage_mode") in {"full_coverage", "dominant_overlap"}:
            return True
        try:
            if float(membership.get("overlap_ratio") or 0.0) >= 0.55:
                return True
        except Exception:
            pass
    return False


def build_formula_units(units: list[dict], *, page_intelligence: dict | None = None) -> list[dict]:
    by_id = {u.get("unit_id"): u for u in units if isinstance(u, dict) and u.get("unit_id")}
    candidates = [
        u for u in eligible_text_units(units)
        if role_of(u) == "formula_expression" and not _inside_preserved_figure(u)
    ]
    formula_units = _prefer_coarse_formula_units(candidates, by_id)
    return [
        {
            "logical_unit_id": f"formula_{idx:04d}",
            "type": "formula_expression",
            "text": text_of(unit),
            "source_unit_ids": [unit["unit_id"]],
            "preservation_mode": (unit.get("policy") or {}).get("preservation_mode"),
            "bbox": bbox_of(unit),
        }
        for idx, unit in enumerate(formula_units, start=1)
    ]


def _prefer_coarse_formula_units(candidates: list[dict], by_id: dict[str, dict]) -> list[dict]:
    """Keep one formula granularity per branch.

    PAGEPRINT may carry the same visual formula as block, line, phrase and span.
    Logical structures must emit one formula, not four duplicate protected
    regions. Prefer line-level formulas, then phrase, then block. Spans are only
    used when they have no text parent.
    """
    ids = {u.get("unit_id") for u in candidates}
    descendants: dict[str, set[str]] = {str(u.get("unit_id")): set() for u in candidates if u.get("unit_id")}
    for unit in candidates:
        cursor = unit
        while cursor.get("parent_id") in by_id:
            parent_id = cursor.get("parent_id")
            if parent_id in ids:
                descendants.setdefault(str(parent_id), set()).add(str(unit.get("unit_id")))
            cursor = by_id[parent_id]

    output_ids: set[str] = set()
    covered_ancestors: set[str] = set()
    for level in ["line", "phrase", "block", "span"]:
        for unit in candidates:
            uid = str(unit.get("unit_id") or "")
            if not uid or unit.get("level") != level or uid in output_ids or uid in covered_ancestors:
                continue
            if descendants.get(uid) and level in {"block", "line"}:
                continue
            output_ids.add(uid)
            cursor = unit
            while cursor.get("parent_id") in by_id:
                covered_ancestors.add(str(cursor.get("parent_id")))
                cursor = by_id[cursor.get("parent_id")]
    return [u for u in candidates if u.get("unit_id") in output_ids]
