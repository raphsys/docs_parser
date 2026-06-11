"""PAGEPRINT Validators — valide que INPUT_DATA est exploitable.

Règles de validation du contrat :
    - toutes les bboxes sont en points ;
    - chaque unit_id est unique ;
    - chaque parent_id existe ;
    - chaque region_id existe ;
    - chaque policy est complète ;
    - chaque unité traduisible a du texte ;
    - chaque unité non traduisible a une raison ;
    - chaque bbox a une surface positive.
"""

from __future__ import annotations

from .schema import (
    CANONICAL_UNIT,
    INPUT_DATA_V1_REQUIRED,
    UNIT_LEVELS,
    UNIT_REQUIRED,
)

POLICY_REQUIRED_FIELDS = (
    "translatable",
    "translation_strategy",
    "render_policy",
    "coverage_required",
)


def validate_input_data(input_data: dict) -> dict:
    """Valide le contrat INPUT_DATA. Retourne {valid, errors, warnings}."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(input_data, dict):
        return {"valid": False, "errors": ["input_data_is_not_a_dict"], "warnings": []}

    # Clés de premier niveau obligatoires.
    missing = INPUT_DATA_V1_REQUIRED - set(input_data.keys())
    if missing:
        errors.append(f"missing_top_level_keys: {sorted(missing)}")

    units = input_data.get("units") or []
    regions = input_data.get("regions") or []

    unit_ids: set[str] = set()
    region_ids = {r.get("region_id") for r in regions if isinstance(r, dict)}

    # Unicité des unit_id.
    for unit in units:
        if not isinstance(unit, dict):
            errors.append("unit_is_not_a_dict")
            continue
        unit_id = unit.get("unit_id")
        if not unit_id:
            errors.append("unit_without_unit_id")
            continue
        if unit_id in unit_ids:
            errors.append(f"duplicate_unit_id: {unit_id}")
        unit_ids.add(unit_id)

    for unit in units:
        if not isinstance(unit, dict):
            continue
        unit_id = unit.get("unit_id") or "<unknown>"

        # Contrat de l'unité.
        missing_fields = UNIT_REQUIRED - set(unit.keys())
        if missing_fields:
            errors.append(f"unit_missing_fields[{unit_id}]: {sorted(missing_fields)}")
            continue

        if unit.get("level") not in UNIT_LEVELS:
            warnings.append(f"unknown_unit_level[{unit_id}]: {unit.get('level')}")

        # parent_id doit exister.
        parent_id = unit.get("parent_id")
        if parent_id is not None and parent_id not in unit_ids:
            errors.append(f"unknown_parent_id[{unit_id}]: {parent_id}")

        # bbox en points, surface positive.
        geometry = unit.get("geometry") or {}
        bbox = geometry.get("bbox")
        if bbox is not None:
            if geometry.get("bbox_unit") != CANONICAL_UNIT:
                errors.append(f"bbox_not_in_points[{unit_id}]: {geometry.get('bbox_unit')}")
            if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                    or float(bbox[2]) <= float(bbox[0])
                    or float(bbox[3]) <= float(bbox[1])):
                # Les traits vectoriels fins (images/drawings de hauteur ou
                # largeur nulle) sont une caractéristique de la source, pas
                # une violation du contrat textuel.
                if unit.get("level") in {"image", "drawing"}:
                    warnings.append(f"degenerate_visual_bbox[{unit_id}]: {bbox}")
                else:
                    errors.append(f"bbox_non_positive_area[{unit_id}]: {bbox}")
        elif unit.get("level") in {"block", "line", "phrase", "span"}:
            warnings.append(f"unit_without_bbox[{unit_id}]")

        # Politique complète.
        policy = unit.get("policy") or {}
        for field in POLICY_REQUIRED_FIELDS:
            if policy.get(field) is None:
                errors.append(f"incomplete_policy[{unit_id}]: missing {field}")

        # Unité traduisible → texte ; non traduisible → raison.
        text = (unit.get("content") or {}).get("text")
        if policy.get("translatable") is True and not (text and str(text).strip()):
            errors.append(f"translatable_unit_without_text[{unit_id}]")
        if policy.get("translatable") is False and not (
            policy.get("non_translatable_reason") or policy.get("policy_source")
        ):
            errors.append(f"non_translatable_unit_without_reason[{unit_id}]")

        # region_memberships → région connue.
        for membership in (unit.get("understanding") or {}).get("region_memberships") or []:
            region_id = membership.get("region_id")
            if region_id and region_id not in region_ids:
                errors.append(f"unknown_region_id[{unit_id}]: {region_id}")

    # Régions : bbox en points, surface positive.
    visual_region_types = {"image_region", "drawing_region", "non_text_zone"}
    for region in regions:
        if not isinstance(region, dict):
            continue
        region_id = region.get("region_id") or "<unknown>"
        if region.get("bbox_unit") != CANONICAL_UNIT:
            errors.append(f"region_bbox_not_in_points[{region_id}]")
        bbox = region.get("bbox")
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or float(bbox[2]) <= float(bbox[0])
                or float(bbox[3]) <= float(bbox[1])):
            if region.get("region_type") in visual_region_types:
                warnings.append(f"degenerate_visual_region[{region_id}]: {bbox}")
            else:
                errors.append(f"region_bbox_non_positive_area[{region_id}]: {bbox}")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "unit_count": len(units),
        "region_count": len(regions),
    }
