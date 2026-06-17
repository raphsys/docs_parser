"""Extracteur de règles depuis fichiers de configuration (yaml/json/toml/ini).

On cherche les clés porteuses de sémantique de règle (seuils, politiques, types
autorisés, mappings, familles de pages, classes documentaires). Chaque entrée
pertinente devient un `RuleRecord` de `source_type=config`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..core.classifier import classify
from ..core.models import RuleRecord, make_rule_id

try:
    import yaml  # type: ignore
except Exception:  # noqa: BLE001
    yaml = None  # type: ignore

try:
    import tomllib  # py311+
except Exception:  # noqa: BLE001
    tomllib = None  # type: ignore


_KEY_HINT = re.compile(
    r"(threshold|policy|policies|rule|rules|constraint|allowed|"
    r"family|families|class|classes|role|roles|min|max|limit|"
    r"translat|reconstruct|font|layout|protect|quality|score)",
    re.IGNORECASE,
)


def _load(text: str, suffix: str) -> Any:
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            return None
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)
    if suffix == ".toml":
        if tomllib is None:
            return None
        return tomllib.loads(text)
    # .ini : non parsé structurellement ; on saute (peu de règles métier).
    return None


def _walk(obj: Any, path: str, out: list[tuple[str, Any]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{path}.{k}" if path else str(k)
            if _KEY_HINT.search(str(k)):
                out.append((kp, v))
            _walk(v, kp, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk(v, f"{path}[{i}]", out)


def extract_config_rules(text: str, file_path: str, suffix: str) -> list[RuleRecord]:
    try:
        data = _load(text, suffix)
    except Exception:  # noqa: BLE001
        return []
    if data is None:
        return []

    hits: list[tuple[str, Any]] = []
    _walk(data, "", hits)

    out: list[RuleRecord] = []
    for keypath, value in hits:
        val_repr = json.dumps(value, ensure_ascii=False, default=str)[:400]
        unit, secondary, conf = classify(file_path, keypath, val_repr)
        rid = make_rule_id("CFG", f"{file_path}:{keypath}")
        rec = RuleRecord(
            id=rid,
            title=f"[config] {keypath}",
            description=f"Entrée de configuration `{keypath}` dans {file_path}.",
            pipeline_unit=unit,
            secondary_units=secondary,
            rule_type="threshold" if re.search(r"min|max|threshold|limit", keypath, re.I) else "policy",
            source_type="config",
            file_path=file_path,
            symbol_name=keypath,
            raw_code=f"{keypath} = {val_repr}",
            normalized_expression=f"{keypath}={val_repr}",
            status="active",
            usage_status="unknown",
            validation_status="untested",
            confidence=max(conf, 0.4),
            risk_level="low",
            managed=False,
            editable=True,
            deletable=False,  # ne pas supprimer une clé de config par mégarde
            simulable=False,
        )
        rec.last_seen_hash = rec.compute_hash()
        rec.touch()
        out.append(rec)
    return out
