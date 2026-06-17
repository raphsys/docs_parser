"""Pont Rule Studio <-> vrais audits du pipeline (directive §16).

Rule Studio EXPLIQUE les règles ; les VRAIS audits PROUVENT. Ce pont relie les
règles fondamentales de couverture à leurs fonctions d'audit réelles, et
rapporte leur disponibilité + (si un plan/normalized est fourni) leur statut
réel. Il n'exécute jamais le pipeline lourd lui-même."""

from __future__ import annotations

import importlib
from typing import Any

# règle fondamentale -> (module, fonction d'audit) réels du pipeline.
AUDIT_TARGETS: dict[str, tuple[str, str]] = {
    "NO-DROP-001": ("pagetranslate.functional_validator", "audit_original_text_coverage"),
    "PT-COVERAGE-001": ("pagetranslate.functional_validator", "audit_original_text_coverage"),
    "PR-COVERAGE-001": ("pagereconstruct.source_text_lifecycle_ledger", "audit_source_text_lifecycle"),
    "RENDER-COVERAGE-001": ("pubready.stages.visual_image_audit", "audit_page"),
}


def _resolve(module: str, func: str):
    mod = importlib.import_module(module)
    return getattr(mod, func)


def audit_availability() -> dict[str, dict[str, Any]]:
    """Pour chaque règle critique : l'audit réel est-il importable ?"""
    out: dict[str, dict[str, Any]] = {}
    for rid, (module, func) in AUDIT_TARGETS.items():
        info = {"module": module, "function": func, "available": False, "error": None}
        try:
            _resolve(module, func)
            info["available"] = True
        except Exception as exc:  # pragma: no cover - dépend de l'environnement
            info["error"] = repr(exc)
        out[rid] = info
    return out


def run_audit_for(rule_id: str, plan: dict | None = None,
                  normalized: dict | None = None) -> dict[str, Any]:
    """Exécute l'audit réel lié à une règle (si plan/normalized fournis).

    Renvoie status / hard_blockers / missing_count quand l'audit le fournit ;
    sinon seulement la disponibilité. N'exécute pas l'extraction/traduction."""
    target = AUDIT_TARGETS.get(rule_id)
    if not target:
        return {"rule_id": rule_id, "status": "no_audit_bound"}
    module, func = target
    try:
        fn = _resolve(module, func)
    except Exception as exc:
        return {"rule_id": rule_id, "status": "audit_unavailable", "error": repr(exc)}
    if plan is None or normalized is None:
        return {"rule_id": rule_id, "status": "ready", "module": module, "function": func}
    try:
        res = fn(plan, normalized)
    except Exception as exc:  # signatures variables : on rapporte sans planter l'UI
        return {"rule_id": rule_id, "status": "audit_error", "error": repr(exc)}
    res = res if isinstance(res, dict) else getattr(res, "to_dict", lambda: {"result": str(res)})()
    return {
        "rule_id": rule_id,
        "status": res.get("status", "ran"),
        "hard_blockers": res.get("hard_blockers", []),
        "missing_count": res.get("missing_count"),
    }
