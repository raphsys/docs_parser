"""Extracteur de règles rédigées en langage naturel dans la documentation.

Cible : README, AUDIT*, CONTRACT*, SPEC*, docs/*.md. On repère les phrases
normatives (doit / ne doit pas / toujours / jamais / si ... alors / interdit).
Ces règles sont documentaires : non éditables et non gérées par défaut (§6.3).
"""

from __future__ import annotations

import re

from ..core.classifier import classify
from ..core.models import RuleRecord, make_rule_id

# Déclencheurs normatifs (FR + EN).
_NORMATIVE = re.compile(
    r"\b(doit(?:\s+pas|\s+être|\s+rester|\s+toujours)?|ne\s+doit\s+pas|"
    r"toujours|jamais|interdit|interdiction|obligatoire|"
    r"must(?:\s+not)?|never|always|shall|should(?:\s+not)?|"
    r"si\s+.+\s+alors|if\s+.+\s+then|fallback|priorité|contrainte)\b",
    re.IGNORECASE,
)

# Fichiers documentaires « porteurs de contrat ».
_DOC_NAME = re.compile(
    r"(readme|audit|contract|contrat|spec|plan|policy|règle|rule)",
    re.IGNORECASE,
)


def _is_target_doc(file_path: str) -> bool:
    low = file_path.lower()
    return low.endswith(".md") and (
        "docs/" in low or "docs\\" in low or _DOC_NAME.search(file_path) is not None
    )


def extract_markdown_rules(text: str, file_path: str) -> list[RuleRecord]:
    if not _is_target_doc(file_path):
        return []

    out: list[RuleRecord] = []
    in_code = False
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or len(line) < 12:
            continue
        if not _NORMATIVE.search(line):
            continue

        # Nettoie les marqueurs markdown de tête.
        clean = re.sub(r"^[#>*\-\d.\s]+", "", line).strip()
        if len(clean) < 12:
            continue

        unit, secondary, conf = classify(file_path, None, clean)
        rid = make_rule_id("DOC", f"{file_path}:{i}:{clean[:40]}")
        rec = RuleRecord(
            id=rid,
            title=f"[doc] {clean[:100]}",
            description=clean[:500],
            pipeline_unit=unit,
            secondary_units=secondary,
            rule_type="contract",
            source_type="documentation",
            file_path=file_path,
            line_start=i,
            line_end=i,
            symbol_name=None,
            raw_code=clean[:1000],
            normalized_expression=clean[:200],
            objective=clean[:200],
            status="active",
            usage_status="unknown",
            validation_status="needs_review",
            confidence=max(conf, 0.35),
            risk_level="low",
            managed=False,
            editable=False,  # documentaire
            deletable=False,
            simulable=False,
        )
        rec.last_seen_hash = rec.compute_hash()
        rec.touch()
        out.append(rec)
    return out
