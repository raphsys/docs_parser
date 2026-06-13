"""PAGEPRINT Policy Compiler.

Policies are executable downstream orders compiled after evidence, roles and
preservation. Candidate regions are observations only; they never directly set
``skip_translation`` or ``background_only``.
"""

from __future__ import annotations

import re

from .region_index import unit_has_region


REGION_PRIORITY_POLICIES = [
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
        "non_translatable_reason": "chart_tick",
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
    if str(intelligence.get("page_role") or "").lower() == "toc":
        context.setdefault("translation_mode", "toc_row_layout")
    context.setdefault("default_translation_strategy", DEFAULT_POLICY["translation_strategy"])
    context.setdefault("default_render_policy", DEFAULT_POLICY["render_policy"])
    return context


def compile_unit_policy(unit: dict, *, decision_context: dict) -> dict:
    existing = dict(unit.get("policy") or {})
    level = unit.get("level")
    text = (unit.get("content") or {}).get("text")
    understanding = unit.get("understanding") or {}
    preservation = unit.get("preservation_policy") or {}
    preservation_mode = existing.get("preservation_mode") or preservation.get("mode") or "none"

    if level in {"page", "region", "table", "cell", "overlay"} and not (text and str(text).strip()):
        compiled = {**existing}
        compiled.update({
            "translatable": False,
            "translation_strategy": compiled.get("translation_strategy") or f"{level}_container",
            "render_policy": compiled.get("render_policy") or f"{level}_container",
            "coverage_required": compiled.get("coverage_required") or "normal",
            "policy_source": "structural_container",
            "non_translatable_reason": f"{level}_container",
        })
        return compiled

    role = str(understanding.get("role") or "").lower()
    if role in {"publisher_mark", "watermark"}:
        return _with_source({**existing, **_artifact_policy("exclude_as_artifact")}, f"role:{role}_artifact", unit)

    if preservation_mode == "exclude_as_artifact":
        return _with_source({**existing, **_artifact_policy(preservation_mode)}, "preservation:exclude_as_artifact", unit)
    if preservation_mode in {"preserve_text_exactly", "preserve_as_visual_overlay"}:
        return _with_source({**existing, **_preserve_policy(preservation_mode)}, f"preservation:{preservation_mode}", unit)

    for region_type, region_policy in REGION_PRIORITY_POLICIES:
        if unit_has_region(unit, region_type):
            return _with_source({**existing, **region_policy}, f"region:{region_type}", unit)

    compiled = dict(existing)
    if _looks_like_preserve_unit(unit, text):
        role = str(understanding.get("role") or "").lower()
        mode = "preserve_text_exactly" if role in {"code_line", "command_name", "path"} else "preserve_as_visual_overlay"
        return _with_source({**compiled, **_preserve_policy(mode)}, f"text_score:{mode}", unit)

    compiled.setdefault("unit_type", understanding.get("object_type"))
    if compiled.get("translatable") is None:
        compiled["translatable"] = bool(text and str(text).strip())
    if str(understanding.get("role") or "") == "unknown" and text and str(text).strip():
        compiled["translatable"] = False
        compiled["translation_strategy"] = "needs_role_resolution"
        compiled["render_policy"] = "anchored_text"
        compiled["non_translatable_reason"] = "needs_role_resolution"
    if not compiled.get("translation_strategy"):
        compiled["translation_strategy"] = decision_context.get("default_translation_strategy", DEFAULT_POLICY["translation_strategy"])
    if not compiled.get("render_policy"):
        compiled["render_policy"] = decision_context.get("default_render_policy", DEFAULT_POLICY["render_policy"])
    if not compiled.get("coverage_required"):
        compiled["coverage_required"] = DEFAULT_POLICY["coverage_required"]
    compiled.setdefault("policy_source", "role_evidence_preservation_or_default")
    if not compiled.get("translatable") and not compiled.get("non_translatable_reason"):
        compiled["non_translatable_reason"] = "empty_text" if not text else compiled.get("policy_source")
    _mirror_policy_to_unit_contract(unit, compiled)
    return compiled


def compile_policies(units: list[dict], page_intelligence: dict | None = None) -> dict:
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
                "preserve_bbox": compiled.get("render_policy") in {"fixed_preserve", "preserve_overlay"},
            },
            "preservation": {
                "mode": compiled.get("preservation_mode") or (unit.get("preservation_policy") or {}).get("mode"),
                "preserve_text_exactly": bool(compiled.get("preserve_exact_text")),
                "preserve_visual_exactly": bool(compiled.get("preserve_visual")),
                "protected_tokens": compiled.get("protected_tokens") or [],
            },
        }
        if _is_exception_policy(compiled):
            unit_policies[unit["unit_id"]] = {
                "unit_type": compiled.get("unit_type"),
                "translatable": compiled.get("translatable"),
                "translation_strategy": compiled.get("translation_strategy"),
                "render_policy": compiled.get("render_policy"),
                "coverage_required": compiled.get("coverage_required"),
                "preservation_mode": compiled.get("preservation_mode"),
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


def _with_source(compiled: dict, source: str, unit: dict) -> dict:
    compiled["policy_source"] = source
    _mirror_policy_to_unit_contract(unit, compiled)
    return compiled


def _mirror_policy_to_unit_contract(unit: dict, compiled: dict) -> None:
    unit.setdefault("constraints", {})["skip_translation"] = bool(compiled.get("skip_translation"))
    unit.setdefault("constraints", {})["skip_text_reconstruction"] = bool(compiled.get("skip_text_reconstruction"))
    unit.setdefault("constraints", {})["preserve_original_pixels"] = bool(compiled.get("preserve_original_pixels"))


def _looks_like_preserve_unit(unit: dict, text: str | None) -> bool:
    understanding = unit.get("understanding") or {}
    tags = {
        str(understanding.get("role") or "").lower(),
        str(understanding.get("object_type") or "").lower(),
        str(understanding.get("semantic_kind") or "").lower(),
    }
    if tags & {"formula_expression", "formula", "code_line", "command_name", "path"}:
        return True
    if unit.get("level") in {"word", "char"}:
        return False
    s = str(text or "").strip()
    if not s:
        return False
    return _formula_like_text(s) or _code_like_text(s)


def _formula_like_text(text: str) -> bool:
    score = 0
    if re.search(r"[∑∫√∞≈≠≤≥±×÷∂∆λµπσΔαβγ]", text):
        score += 4
    if re.search(r"\w\s*(?:=|≈|<=|>=|≤|≥)\s*\w", text):
        score += 3
    if len(re.findall(r"[=+\*/^<>≤≥≈]", text)) >= 2:
        score += 2
    if len(re.findall(r"[A-Za-z]{3,}", text)) >= 5:
        score -= 3
    if re.fullmatch(r"\([^)]{1,40}\)", text):
        score -= 4
    return score >= 4


def _code_like_text(text: str) -> bool:
    score = 0
    if re.search(r"\b(def|class|import|return|function|SELECT|FROM|WHERE)\b", text):
        score += 2
    if re.search(r"(==|!=|:=|=>|->|[{};])", text) and len(text.split()) <= 12:
        score += 2
    if re.search(r"(?:[A-Za-z]:\\|/[\w.-]+/|\.{1,2}/)[^\s]+", text):
        score += 3
    if re.search(r"^(?:copy|dir|del|findstr|mkdir|rmdir|cd|ls|cat|grep|sudo|docker|npm|pip|python|git)\b", text, re.IGNORECASE):
        score += 3
    if len(re.findall(r"[A-Za-z]{3,}", text)) >= 8 and re.search(r"\s", text):
        score -= 3
    return score >= 4


def _preserve_policy(mode: str) -> dict:
    visual = mode == "preserve_as_visual_overlay"
    return {
        "unit_type": mode,
        "translatable": False,
        "translation_strategy": "exact_preserve" if mode == "preserve_text_exactly" else "visual_overlay",
        "coverage_required": "strict",
        "render_policy": "fixed_preserve" if mode == "preserve_text_exactly" else "preserve_overlay",
        "preserve_exact_text": mode == "preserve_text_exactly",
        "preserve_original_pixels": visual,
        "preserve_visual": visual,
        "skip_translation": True,
        "skip_text_reconstruction": visual,
        "non_translatable_reason": mode,
    }


def _artifact_policy(mode: str) -> dict:
    return {
        "unit_type": mode,
        "translatable": False,
        "translation_strategy": "skip",
        "coverage_required": "strict",
        "render_policy": "skip_artifact",
        "skip_translation": True,
        "skip_text_reconstruction": True,
        "non_translatable_reason": mode,
    }


def _is_exception_policy(compiled: dict) -> bool:
    return bool(
        str(compiled.get("policy_source") or "").startswith(("region:", "preservation:", "text_score:"))
        or compiled.get("translatable") is False
    )
