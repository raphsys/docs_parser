"""Analyse d'usage des règles.

§11.1 Usage statique : une règle est `used_static` si son symbole/fichier est
référencé depuis (ou atteignable depuis) un entrypoint connu. Heuristique
légère par grep d'imports et de références de symboles ; pas d'analyse de flot
complète, mais suffisant pour distinguer le mort du vivant.

§11.2 Usage dynamique : architecture posée via `RULE_HIT_COUNTER`. Une règle
gérée simulée/testée peut être marquée `used_dynamic`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ..core.scanner import iter_source_files
from .models import RuleRecord

# Points d'entrée probables du pipeline (cf. §11.1).
ENTRYPOINTS = [
    "ocr_server.py",
    "pipelines/orchestrator.py",
    "pageprint/builder.py",
    "pagetranslate/builder.py",
    "pagereconstruct/plan_compiler.py",
    "server/api.py",
]


# --- compteur dynamique (instrumentation légère) ---------------------------

RULE_HIT_COUNTER: dict[str, int] = defaultdict(int)
RULE_HIT_LOG: list[dict] = []


def record_hit(rule_id: str, unit_id: str, result: bool, input_id: str = "") -> None:
    """Enregistre un déclenchement de règle (simulation ou test)."""
    import time

    RULE_HIT_COUNTER[rule_id] += 1
    RULE_HIT_LOG.append(
        {
            "rule_id": rule_id,
            "unit_id": unit_id,
            "result": result,
            "input_id": input_id,
            "timestamp": time.time(),
        }
    )


# --- usage statique ---------------------------------------------------------


def _symbol_index(root: Path) -> dict[str, set[str]]:
    """Index inversé : symbole/nom-de-module -> fichiers qui le référencent."""
    index: dict[str, set[str]] = defaultdict(set)
    files = list(iter_source_files(root))
    token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    contents: dict[str, set[str]] = {}
    for p in files:
        if p.suffix != ".py":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(p.relative_to(root))
        contents[rel] = set(token_re.findall(text))
    for rel, tokens in contents.items():
        for tok in tokens:
            index[tok].add(rel)
    return index


def analyze_static_usage(records: list[RuleRecord], root: str | Path) -> dict[str, str]:
    """Renvoie {rule_id: usage_status} par analyse statique légère."""
    root = Path(root).resolve()
    index = _symbol_index(root)
    entry_set = {e for e in ENTRYPOINTS}

    out: dict[str, str] = {}
    for rec in records:
        if rec.source_type in {"documentation", "config"}:
            out[rec.id] = "unknown"
            continue

        # Symbole le plus spécifique : dernier segment de symbol_name, sinon
        # nom de module du fichier.
        symbol = None
        if rec.symbol_name:
            symbol = rec.symbol_name.split(".")[-1]
        module = Path(rec.file_path).stem if rec.file_path else None

        referencing: set[str] = set()
        if symbol and symbol in index:
            referencing |= index[symbol]
        if module and module in index:
            referencing |= index[module]

        # On retire l'auto-référence (le fichier qui se contient lui-même).
        referencing.discard(rec.file_path)

        if not referencing:
            # Si le fichier est lui-même un entrypoint, on considère utilisé.
            out[rec.id] = "used_static" if rec.file_path in entry_set else "not_referenced"
        else:
            # Référencé par au moins un autre fichier => statiquement utilisé.
            out[rec.id] = "used_static"
    return out


def apply_usage(records: list[RuleRecord], root: str | Path) -> None:
    """Met à jour `usage_status` in place (statique + dynamique connu)."""
    static = analyze_static_usage(records, root)
    for rec in records:
        if RULE_HIT_COUNTER.get(rec.id):
            rec.usage_status = "used_dynamic"
        else:
            rec.usage_status = static.get(rec.id, "unknown")
