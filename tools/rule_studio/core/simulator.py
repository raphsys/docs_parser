"""Moteur DSL sûr pour simuler les règles gérées (when/then).

SÉCURITÉ (cf. cahier des charges §10) : aucune utilisation de `eval`/`exec`.
On interprète une structure de données déclarative. Les seuls effets possibles
sont des écritures dans un contexte simulé (jamais sur le système réel).

Grammaire WHEN :
    {"all": [cond, ...]} | {"any": [cond, ...]} | {"not": cond} | cond
    cond = {"field": "page.index", "op": "eq", "value": 0}

Grammaire THEN :
    {"set": {"page.role": "cover"}, "add_tag": ["x"], "set_policy": {...},
     "set_constraint": {...}, "raise_warning": "msg"}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

ALLOWED_OPS = {
    "eq",
    "neq",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "not_in",
    "contains",
    "regex",
    "exists",
    "missing",
    "between",
}

ALLOWED_ACTIONS = {
    "set",
    "add_tag",
    "set_policy",
    "set_constraint",
    "raise_warning",
}


class SimulationError(Exception):
    """DSL invalide (op inconnue, structure malformée)."""


@dataclass
class SimulationResult:
    matched: bool
    reads: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)


# --- accès champ pointé (page.index) ---------------------------------------

_MISSING = object()


def _get_path(ctx: dict[str, Any], path: str) -> Any:
    cur: Any = ctx
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _set_path(ctx: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = ctx
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
        if not isinstance(cur, dict):
            raise SimulationError(f"Chemin non assignable : {path}")
    cur[parts[-1]] = value


# --- évaluation des conditions ---------------------------------------------


def _eval_cond(cond: dict[str, Any], ctx: dict[str, Any], res: SimulationResult) -> bool:
    field_path = cond.get("field")
    op = cond.get("op")
    value = cond.get("value")
    if not field_path or not op:
        raise SimulationError(f"Condition incomplète : {cond}")
    if op not in ALLOWED_OPS:
        raise SimulationError(f"Opérateur interdit : {op}")

    actual = _get_path(ctx, field_path)
    present = actual is not _MISSING
    shown = None if not present else actual
    res.reads.append({"field": field_path, "value": shown, "op": op, "expected": value})

    if op == "exists":
        ok = present
    elif op == "missing":
        ok = not present
    elif not present:
        ok = False  # tout autre op sur champ absent => faux
    elif op == "eq":
        ok = actual == value
    elif op == "neq":
        ok = actual != value
    elif op == "gt":
        ok = _num(actual) > _num(value)
    elif op == "gte":
        ok = _num(actual) >= _num(value)
    elif op == "lt":
        ok = _num(actual) < _num(value)
    elif op == "lte":
        ok = _num(actual) <= _num(value)
    elif op == "in":
        ok = actual in (value or [])
    elif op == "not_in":
        ok = actual not in (value or [])
    elif op == "contains":
        ok = value in actual if hasattr(actual, "__contains__") else False
    elif op == "regex":
        ok = bool(re.search(str(value), str(actual)))
    elif op == "between":
        lo, hi = value
        ok = _num(lo) <= _num(actual) <= _num(hi)
    else:  # pragma: no cover — déjà filtré par ALLOWED_OPS
        ok = False

    res.trace.append(f"{field_path} {op} {value!r} -> {ok} (réel={shown!r})")
    return ok


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError) as exc:
        raise SimulationError(f"Valeur non numérique : {v!r}") from exc


def _eval_when(node: Any, ctx: dict[str, Any], res: SimulationResult) -> bool:
    if not node:
        return True
    if not isinstance(node, dict):
        raise SimulationError(f"Nœud WHEN invalide : {node!r}")
    if "all" in node:
        return all(_eval_when(c, ctx, res) for c in node["all"])
    if "any" in node:
        return any(_eval_when(c, ctx, res) for c in node["any"])
    if "not" in node:
        return not _eval_when(node["not"], ctx, res)
    return _eval_cond(node, ctx, res)


# --- application des actions (sur copie) ------------------------------------


def _apply_then(then: dict[str, Any], ctx: dict[str, Any], res: SimulationResult) -> None:
    if not then:
        return
    for action, payload in then.items():
        if action not in ALLOWED_ACTIONS:
            raise SimulationError(f"Action interdite : {action}")
        if action == "set":
            for path, val in (payload or {}).items():
                _set_path(ctx, path, val)
                res.actions.append({"action": "set", "field": path, "value": val})
        elif action == "set_policy":
            for path, val in (payload or {}).items():
                _set_path(ctx, f"policy.{path}", val)
                res.actions.append({"action": "set_policy", "field": path, "value": val})
        elif action == "set_constraint":
            for path, val in (payload or {}).items():
                _set_path(ctx, f"constraint.{path}", val)
                res.actions.append({"action": "set_constraint", "field": path, "value": val})
        elif action == "add_tag":
            tags = ctx.setdefault("tags", [])
            for t in payload if isinstance(payload, list) else [payload]:
                tags.append(t)
                res.actions.append({"action": "add_tag", "value": t})
        elif action == "raise_warning":
            res.warnings.append(str(payload))
            res.actions.append({"action": "raise_warning", "value": str(payload)})


# --- API --------------------------------------------------------------------


def simulate(dsl: dict[str, Any], context: dict[str, Any]) -> SimulationResult:
    """Simule une règle `{when, then}` sur un `context` (jamais muté)."""
    import copy

    res = SimulationResult(matched=False)
    res.before = copy.deepcopy(context)
    work = copy.deepcopy(context)

    when = dsl.get("when", {})
    res.matched = _eval_when(when, work, res)

    if res.matched:
        _apply_then(dsl.get("then", {}), work, res)
    res.after = work
    return res


def validate_dsl(dsl: dict[str, Any]) -> list[str]:
    """Valide statiquement un DSL ; renvoie la liste des erreurs (vide si OK)."""
    errors: list[str] = []

    def _check_when(node: Any) -> None:
        if not node:
            return
        if not isinstance(node, dict):
            errors.append(f"WHEN doit être un dict : {node!r}")
            return
        if "all" in node or "any" in node:
            for c in node.get("all", node.get("any", [])):
                _check_when(c)
        elif "not" in node:
            _check_when(node["not"])
        else:
            if "field" not in node:
                errors.append(f"Condition sans `field` : {node}")
            op = node.get("op")
            if op not in ALLOWED_OPS:
                errors.append(f"Opérateur interdit/absent : {op}")

    _check_when(dsl.get("when", {}))
    for action in (dsl.get("then") or {}):
        if action not in ALLOWED_ACTIONS:
            errors.append(f"Action interdite : {action}")
    return errors
