"""PAGEPRINT Policy Compiler — compile les politiques finales par unité.

Rôle : prendre page_intelligence + regions + unit roles + source_kind +
style + risks, et produire une politique finale par unité.

La politique devient un ordre exécutable pour les modules aval.
Règle de priorité : la région est plus forte que le simple texte.
"""

from __future__ import annotations

from .region_index import unit_has_region

# Politiques par région prioritaire (la première qui matche gagne).
REGION_PRIORITY_POLICIES = [
    ("formula", {
        "unit_type": "formula",
        "translatable": False,
        "translation_strategy": "exact_preserve",
        "coverage_required": "strict",
        "render_policy": "fixed_preserve",
    }),
    ("code", {
        "unit_type": "code_visible",
        "translatable": False,
        "translation_strategy": "exact_preserve",
        "coverage_required": "strict",
        "render_policy": "anchored_text",
    }),
    ("table_cell", {
        "unit_type": "table_cell_text",
        "translatable": True,
        "translation_strategy": "layout_constrained",
        "coverage_required": "strict",
        "render_policy": "anchored_text",
    }),
    ("chart_tick", {
        "unit_type": "chart_tick_label",
        "translatable": False,
        "translation_strategy": "exact_preserve",
        "coverage_required": "strict",
        "render_policy": "fixed_preserve",
    }),
]

DEFAULT_POLICY = {
    "translation_strategy": "layout_constrained",
    "render_policy": "anchored_text",
    "coverage_required": "strict",
}


def _decision_context(page_intelligence: dict | None) -> dict:
    intelligence = page_intelligence or {}
    context = dict(intelligence.get("decision_context") or {})
    reading_modes = intelligence.get("reading_modes") or {}

    # Les modes de lecture pèsent sur les décisions par défaut.
    if float(reading_modes.get("tabular_grid_flow") or 0.0) > 0.65:
        context.setdefault("default_translation_strategy", "layout_constrained")
        context.setdefault("preserve_grid", True)
        context.setdefault("translation_mode", "cell_constrained")
    if float(reading_modes.get("anchored_overlay_flow") or 0.0) > 0.55:
        context.setdefault("preserve_anchors", True)
        context.setdefault("translation_mode", "anchored_labels")
        context.setdefault("default_render_policy", "anchored_text")
    if float(reading_modes.get("linear_flow") or 0.0) > 0.65:
        context.setdefault("translation_mode", "paragraph_reflow")
        context.setdefault("allow_paragraph_reflow", True)
    if float(reading_modes.get("toc_row_flow") or 0.0) > 0.80:
        context.setdefault("translation_mode", "toc_row_layout")

    context.setdefault("default_translation_strategy",
                       DEFAULT_POLICY["translation_strategy"])
    context.setdefault("default_render_policy", DEFAULT_POLICY["render_policy"])
    return context


def compile_unit_policy(unit: dict, *, decision_context: dict) -> dict:
    """Compile la politique finale d'une unité (région > contrat existant > défauts)."""
    existing = dict(unit.get("policy") or {})

    # 1. Régions prioritaires.
    for region_type, region_policy in REGION_PRIORITY_POLICIES:
        if unit_has_region(unit, region_type):
            compiled = {**existing, **region_policy}
            compiled["policy_source"] = f"region:{region_type}"
            return compiled

    # 2. Contrat existant (PagePolicyMatrix / translation_contract legacy).
    compiled = dict(existing)
    compiled.setdefault("unit_type", (unit.get("understanding") or {}).get("object_type"))
    if compiled.get("translatable") is None:
        text = (unit.get("content") or {}).get("text")
        compiled["translatable"] = bool(text and str(text).strip())
    if not compiled.get("translation_strategy"):
        compiled["translation_strategy"] = decision_context.get(
            "default_translation_strategy", DEFAULT_POLICY["translation_strategy"]
        )
    if not compiled.get("render_policy"):
        compiled["render_policy"] = decision_context.get(
            "default_render_policy", DEFAULT_POLICY["render_policy"]
        )
    if not compiled.get("coverage_required"):
        compiled["coverage_required"] = DEFAULT_POLICY["coverage_required"]
    compiled.setdefault("policy_source", "legacy_contract_or_default")
    if not compiled.get("translatable") and not compiled.get("non_translatable_reason"):
        compiled["non_translatable_reason"] = (
            "empty_text" if not (unit.get("content") or {}).get("text")
            else compiled.get("policy_source")
        )
    return compiled


def compile_policies(units: list[dict], page_intelligence: dict | None = None) -> dict:
    """Compile les politiques de toutes les unités et la couche policies globale."""
    decision_context = _decision_context(page_intelligence)
    unit_policies: dict[str, dict] = {}

    for unit in units:
        compiled = compile_unit_policy(unit, decision_context=decision_context)
        unit["policy"] = compiled
        unit["compiled_policy"] = {
            "translation": {
                "enabled": bool(compiled.get("translatable")),
                "strategy": compiled.get("translation_strategy"),
            },
            "rendering": {
                "mode": compiled.get("render_policy"),
                "preserve_bbox": compiled.get("render_policy") == "fixed_preserve",
            },
            "preservation": {
                "preserve_text_exactly": bool(compiled.get("preserve_exact_text"))
                or compiled.get("translation_strategy") == "exact_preserve",
                "preserve_visual_exactly": bool(compiled.get("preserve_visual"))
                or compiled.get("render_policy") == "fixed_preserve",
            },
        }
        if compiled.get("policy_source", "").startswith("region:"):
            unit["provenance"]["decision_trace"].append({
                "stage": "policy_compilation",
                "target_id": unit["unit_id"],
                "decision": f"unit_type={compiled.get('unit_type')}",
                "reason": compiled["policy_source"],
            })
        # Seules les exceptions sont indexées ici : politique imposée par
        # une région ou unité non traduisible. La politique complète de
        # chaque unité vit sur units[].policy (source de vérité).
        is_exception = (
            compiled.get("policy_source", "").startswith("region:")
            or compiled.get("translatable") is False
        )
        if is_exception:
            unit_policies[unit["unit_id"]] = {
                "unit_type": compiled.get("unit_type"),
                "translatable": compiled.get("translatable"),
                "translation_strategy": compiled.get("translation_strategy"),
                "render_policy": compiled.get("render_policy"),
                "coverage_required": compiled.get("coverage_required"),
                "policy_source": compiled.get("policy_source"),
            }

    return {
        "default_policy": {
            "translation_strategy": decision_context["default_translation_strategy"],
            "render_policy": decision_context["default_render_policy"],
            "coverage_required": DEFAULT_POLICY["coverage_required"],
        },
        "decision_context": decision_context,
        "unit_policies_location": "units[].policy",
        "unit_policy_exceptions": unit_policies,
    }
