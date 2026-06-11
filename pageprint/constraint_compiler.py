"""PAGEPRINT Constraint Engine — compile les contraintes WYSIWYG.

Produit par unité :
    constraints (preserve_bbox, allow_reflow, allow_wrap, allow_font_scaling,
    preserve_alignment, preserve_grid, preserve_anchor, preserve_visual),
    transformation_budget (jusqu'où l'unité peut changer),
    layout_freedom (degrés de liberté x/y/width/height/font/line_breaks/anchor),
    render_contract (contrat final de rendu).

Et au niveau page : reconstruction_constraints, consommées telles quelles
par le reconstructeur — il ne doit pas deviner, il lit.
"""

from __future__ import annotations

EN_FR_EXPANSION_RATIO = 1.22


def _is_fixed(policy: dict) -> bool:
    return policy.get("render_policy") == "fixed_preserve"


def _is_table_cell(policy: dict) -> bool:
    return policy.get("unit_type") in {"table_cell_text", "table_cell"}


def _is_prose(unit: dict, policy: dict) -> bool:
    if _is_fixed(policy) or _is_table_cell(policy):
        return False
    return (unit.get("understanding") or {}).get("role") in {
        "body", "paragraph", None
    } and bool(policy.get("translatable"))


def compile_unit_constraints(unit: dict) -> dict:
    """Compile constraints + transformation_budget + layout_freedom + render_contract."""
    policy = unit.get("policy") or {}
    fixed = _is_fixed(policy)
    table_cell = _is_table_cell(policy)
    prose = _is_prose(unit, policy)

    constraints = {
        "allow_reflow": prose,
        "allow_line_wrap": not fixed,
        "preserve_bbox": fixed or table_cell,
        "preserve_alignment": True,
        "preserve_grid": table_cell,
        "preserve_anchor": policy.get("render_policy") == "anchored_text",
        "preserve_font_size": fixed,
        "preserve_color": True,
        "preserve_visual": bool(policy.get("preserve_visual")) or fixed,
        "allow_font_scaling": not fixed,
        "allow_horizontal_expansion": prose,
        "allow_vertical_expansion": prose,
        "overflow_risk": 0.0,
    }

    if policy.get("translatable"):
        text = (unit.get("content") or {}).get("text") or ""
        # Risque d'overflow croissant avec la longueur et l'absence de reflow.
        length_factor = min(1.0, len(str(text)) / 200.0)
        constraints["overflow_risk"] = round(
            (EN_FR_EXPANSION_RATIO - 1.0) * (1.0 if not prose else 0.5)
            * (0.4 + 0.6 * length_factor), 3
        )

    transformation_budget = {
        "max_text_expansion_ratio": 1.0 if fixed else (1.15 if table_cell else 1.30),
        "max_font_reduction_ratio": 0.0 if fixed else (0.18 if table_cell else 0.12),
        "max_bbox_growth_x_pt": 0.0 if (fixed or table_cell) else 8.0,
        "max_bbox_growth_y_pt": 0.0 if fixed else (4.0 if table_cell else 12.0),
        "allow_line_wrap": constraints["allow_line_wrap"],
        "allow_reflow": constraints["allow_reflow"],
        "allow_reposition": False,
        "must_keep_alignment": constraints["preserve_alignment"],
    }

    if fixed:
        layout_freedom = {
            "x": "fixed", "y": "fixed", "width": "fixed", "height": "fixed",
            "font_size": "fixed", "line_breaks": "preserve", "anchor": "page",
        }
    elif table_cell:
        layout_freedom = {
            "x": "fixed", "y": "fixed", "width": "fixed", "height": "elastic",
            "font_size": "shrink_allowed", "line_breaks": "adaptive",
            "anchor": "region",
        }
    elif prose:
        layout_freedom = {
            "x": "fixed", "y": "elastic", "width": "fixed", "height": "elastic",
            "font_size": "adaptive", "line_breaks": "reflow", "anchor": "none",
        }
    else:
        layout_freedom = {
            "x": "fixed", "y": "fixed", "width": "elastic", "height": "elastic",
            "font_size": "shrink_allowed", "line_breaks": "preserve",
            "anchor": "visual_object" if constraints["preserve_anchor"] else "none",
        }

    render_contract = {
        "mode": "paragraph_flow" if prose else (policy.get("render_policy") or "anchored_text"),
        "target_layer": "overlay_layer" if constraints["preserve_visual"] else "text_layer",
        "background_handling": "preserve" if constraints["preserve_visual"] else "erase_and_redraw",
        "text_box": {
            "bbox": (unit.get("geometry") or {}).get("bbox"),
            "unit": "pt",
            "baseline_policy": "preserve_baseline_grid",
        },
        "font_policy": {
            "family": "preserve_or_substitute",
            "size": "fixed" if fixed else "adaptive_within_budget",
            "color": "preserve",
        },
        "overflow_policy": {
            "first": "wrap",
            "second": "shrink_font",
            "third": "expand_height",
            "final": "flag_review",
        },
    }

    translation_forecast = None
    if policy.get("translatable"):
        translation_forecast = {
            "expected_length_ratio": EN_FR_EXPANSION_RATIO,
            "overflow_probability": constraints["overflow_risk"],
            "recommended_strategy": "wrap_then_shrink" if not fixed else "manual_review",
            "fallback_strategy": "manual_review",
        }

    unit["constraints"] = constraints
    unit["transformation_budget"] = transformation_budget
    unit["layout_freedom"] = layout_freedom
    unit["render_contract"] = render_contract
    if translation_forecast:
        unit["translation_forecast"] = translation_forecast
    return constraints


def compile_page_constraints(units: list[dict],
                             page_intelligence: dict | None = None) -> dict:
    """Compile les contraintes au niveau page + reconstruction_constraints."""
    for unit in units:
        compile_unit_constraints(unit)

    intelligence = page_intelligence or {}
    decision_context = intelligence.get("decision_context") or {}
    reading_modes = intelligence.get("reading_modes") or {}

    reconstruction_constraints = {
        "page": {
            "preserve_page_size": True,
            "preserve_background": True,
            "preserve_margins": True,
            "coordinate_unit": "pt",
        },
        "layout": {
            "preserve_columns": True,
            "preserve_table_grid": bool(
                decision_context.get("preserve_grid")
                or float(reading_modes.get("tabular_grid_flow") or 0.0) > 0.65
            ),
            "preserve_anchors": bool(
                decision_context.get("preserve_anchors")
                or float(reading_modes.get("anchored_overlay_flow") or 0.0) > 0.55
            ),
            "allow_global_reflow": bool(decision_context.get("allow_paragraph_reflow")),
        },
        "overflow": {
            "strategy": "line_wrap",
            "max_font_reduction_ratio": 0.12,
            "allow_vertical_growth": True,
        },
        "visual_preservation": {
            "preserve_formulas_as_images": True,
            "preserve_code_blocks": True,
            "preserve_images": True,
            "preserve_drawings": True,
        },
    }
    return reconstruction_constraints
