"""Registre des règles gérées explicitement (`managed_rules.yaml`).

Distingue clairement (cf. cahier des charges §7) :
    - règles détectées automatiquement (issues du scan, source_type != managed)
    - règles gérées explicitement (ce fichier)
Les règles gérées portent un DSL `when/then`, une cible (`target`) et des tests.
Elles sont simulables et constituent la base de gouvernance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import RuleRecord

try:
    import yaml  # type: ignore
except Exception:  # noqa: BLE001
    yaml = None  # type: ignore


def load_managed_rules(path: str | Path) -> list[RuleRecord]:
    """Charge les règles gérées depuis le YAML."""
    p = Path(path)
    if not p.exists() or yaml is None:
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: list[RuleRecord] = []
    for entry in data.get("rules", []):
        out.append(_entry_to_record(entry))
    return out


def _entry_to_record(entry: dict[str, Any]) -> RuleRecord:
    dsl = {"when": entry.get("when", {}), "then": entry.get("then", {})}
    rec = RuleRecord(
        id=entry["id"],
        title=entry.get("title", entry["id"]),
        description=entry.get("objective", ""),
        pipeline_unit=entry.get("pipeline_unit", "UNKNOWN"),
        secondary_units=entry.get("secondary_units", []),
        rule_type=entry.get("rule_type", "unknown"),
        source_type="managed",
        file_path=(entry.get("target", {}) or {}).get("file", ""),
        symbol_name=(entry.get("target", {}) or {}).get("function"),
        objective=entry.get("objective", ""),
        understanding=entry.get("understanding", ""),
        status=entry.get("status", "active"),
        usage_status="unknown",
        validation_status=entry.get("validation_status", "needs_review"),
        confidence=1.0,
        risk_level=entry.get("risk_level", "medium"),
        managed=True,
        editable=True,
        deletable=True,
        simulable=bool(entry.get("when")),
        tests_linked=entry.get("tests", []),
        dsl=dsl,
        target=entry.get("target", {}),
    )
    # Champs étendus (langage naturel, booléens de décision, agents, cycle de vie).
    from .models import BOOL_DECISION_FIELDS
    _NL = ("natural_language_title", "natural_language_rule", "impact",
           "risk_explanation", "human_decision", "recommended_decision",
           "implementation_notes", "proposed_modification", "lifecycle_status")
    for key in _NL:
        if entry.get(key) is not None:
            setattr(rec, key, entry[key])
    for key in BOOL_DECISION_FIELDS:
        if key in entry:
            setattr(rec, key, bool(entry[key]))
    rec.last_seen_hash = rec.compute_hash()
    rec.touch()
    return rec


def record_to_entry(rec: RuleRecord) -> dict[str, Any]:
    """Inverse : convertit un `RuleRecord` géré en entrée YAML sérialisable."""
    entry: dict[str, Any] = {
        "id": rec.id,
        "title": rec.title,
        "pipeline_unit": rec.pipeline_unit,
        "rule_type": rec.rule_type,
        "status": rec.status,
        "objective": rec.objective,
    }
    if rec.understanding:
        entry["understanding"] = rec.understanding
    if rec.dsl.get("when"):
        entry["when"] = rec.dsl["when"]
    if rec.dsl.get("then"):
        entry["then"] = rec.dsl["then"]
    if rec.target:
        entry["target"] = rec.target
    if rec.tests_linked:
        entry["tests"] = rec.tests_linked
    if rec.risk_level and rec.risk_level != "unknown":
        entry["risk_level"] = rec.risk_level
    # round-trip des champs étendus si renseignés
    from .models import BOOL_DECISION_FIELDS
    for key in ("natural_language_title", "natural_language_rule", "impact",
                "risk_explanation", "human_decision", "recommended_decision",
                "implementation_notes", "proposed_modification", "lifecycle_status"):
        val = getattr(rec, key, None)
        if val:
            entry[key] = val
    for key in BOOL_DECISION_FIELDS:
        if getattr(rec, key, False):
            entry[key] = True
    return entry


def save_managed_rules(path: str | Path, records: list[RuleRecord]) -> None:
    """Réécrit `managed_rules.yaml` à partir des règles gérées fournies."""
    if yaml is None:
        raise RuntimeError("PyYAML requis pour écrire managed_rules.yaml")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rules": [record_to_entry(r) for r in records if r.managed]}
    p.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
