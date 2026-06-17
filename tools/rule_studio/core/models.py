"""Modèle canonique d'une règle du pipeline vSense / WYSIWYG.

Ce module définit `RuleRecord`, les énumérations de valeurs autorisées et les
helpers de (de)sérialisation. Aucun import lourd : il doit rester importable
sans dépendances externes (utilisé par les extracteurs, le store et l'UI).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Vocabulaires contrôlés
# ---------------------------------------------------------------------------

PIPELINE_UNITS = [
    "GLOBAL",
    "INPUT_DATA",
    "PAGE_DETECTION",
    "PAGE_CLASSIFICATION",
    "DOCUMENT_COMPREHENSION",
    "SEGMENTATION",
    "REGION_DETECTION",
    "UNIT_FACTORY",
    "STYLE_EXTRACTION",
    "SEMANTIC_ROLE_RESOLUTION",
    "POLICY_COMPILATION",
    "CONSTRAINT_COMPILATION",
    "TRANSLATION",
    "PROTECTED_CONTENT",
    "RECONSTRUCTION",
    "LAYOUT_SOLVER",
    "TYPOGRAPHY",
    "QA",
    "EXPORT",
    "LEGACY_COMPATIBILITY",
    "DEBUG",
    "UNKNOWN",
]

RULE_TYPES = [
    "contract",
    "policy",
    "constraint",
    "heuristic",
    "validator",
    "fallback",
    "exception",
    "threshold",
    "classification",
    "segmentation",
    "translation",
    "reconstruction",
    "typography",
    "layout",
    "quality_score",
    "protected_content",
    "legacy_compatibility",
    "unknown",
]

SOURCE_TYPES = [
    "python_ast",
    "config",
    "documentation",
    "comment_block",
    "schema",
    "managed",
    "manual",
]

STATUSES = [
    "active",
    "inactive",
    "candidate",
    "deprecated",
    "conflicting",
    "duplicate",
    "orphan",
    "unknown",
]

USAGE_STATUSES = [
    "used_static",
    "used_dynamic",
    "not_referenced",
    "not_triggered",
    "unknown",
]

VALIDATION_STATUSES = [
    "valid",
    "invalid",
    "needs_review",
    "untested",
    "conflict_detected",
]

KEEP_DECISIONS = ["keep", "modify", "delete", "needs_review"]

RISK_LEVELS = ["low", "medium", "high", "critical", "unknown"]

# Cycle de vie d'une règle (≠ active/valid/applied : états distincts).
LIFECYCLE_STATUSES = [
    "detected",
    "interpreted",
    "needs_human_review",
    "edited_by_human",
    "candidate",
    "ready_to_generate",
    "patch_generated",
    "tests_failed",
    "tests_passed",
    "ready_to_apply",
    "applied",
    "rejected",
    "deprecated",
]

# Cases à cocher booléennes exposées dans l'UID (vrais bool, jamais du texte).
BOOL_DECISION_FIELDS = (
    "keep", "active", "selected", "modify", "delete", "validate",
    "simulate", "implement", "blocking", "requires_tests", "apply_to_project",
)


# ---------------------------------------------------------------------------
# RuleRecord
# ---------------------------------------------------------------------------


@dataclass
class RuleRecord:
    """Représentation canonique d'une règle, détectée ou gérée."""

    id: str
    title: str = ""
    description: str = ""

    pipeline_unit: str = "UNKNOWN"
    secondary_units: list[str] = field(default_factory=list)

    rule_type: str = "unknown"
    source_type: str = "python_ast"

    file_path: str = ""
    line_start: int | None = None
    line_end: int | None = None
    symbol_name: str | None = None

    raw_code: str | None = None
    normalized_expression: str | None = None

    objective: str = ""
    understanding: str = ""

    # --- explication en langage naturel (RuleInterpreterAgent) ---
    natural_language_title: str | None = None
    natural_language_rule: str | None = None
    impact: str | None = None
    risk_explanation: str | None = None
    human_decision: str | None = None
    recommended_decision: str | None = None
    implementation_notes: str | None = None

    keep_decision: str = "needs_review"
    modification_decision: str = ""  # libre : "yes"/"no"/note
    proposed_modification: str | None = None

    # --- décisions humaines en cases à cocher (vrais booléens) ---
    keep: bool = False
    active: bool = True
    selected: bool = False
    modify: bool = False
    delete: bool = False
    validate: bool = False
    simulate: bool = False
    implement: bool = False
    blocking: bool = False
    requires_tests: bool = False
    apply_to_project: bool = False

    # --- état des agents IA ---
    ai_interpreted: bool = False
    ai_interpretation_version: str | None = None
    ai_interpretation_status: str | None = None
    ai_last_interpreted_at: str | None = None
    coding_agent_status: str | None = None
    coding_agent_patch_path: str | None = None
    coding_agent_report: str | None = None
    validation_agent_status: str | None = None
    validation_report: str | None = None

    # cycle de vie (distinct de status/usage/validation)
    lifecycle_status: str = "detected"

    status: str = "unknown"
    usage_status: str = "unknown"
    validation_status: str = "untested"

    confidence: float = 0.0
    risk_level: str = "unknown"

    managed: bool = False
    editable: bool = True
    deletable: bool = True
    simulable: bool = False

    tests_linked: list[str] = field(default_factory=list)
    last_seen_hash: str = ""
    created_at: str = ""
    updated_at: str = ""

    # Données structurées optionnelles pour les règles gérées / simulables
    # (DSL when/then). Stockées en JSON dans le store.
    dsl: dict[str, Any] = field(default_factory=dict)
    target: dict[str, Any] = field(default_factory=dict)

    # --- helpers -----------------------------------------------------------

    def touch(self) -> None:
        now = _now_iso()
        if not self.created_at:
            self.created_at = now
        self.updated_at = now

    def compute_hash(self) -> str:
        basis = "|".join(
            [
                self.file_path,
                self.symbol_name or "",
                self.raw_code or self.normalized_expression or self.title,
            ]
        )
        return hashlib.sha1(basis.encode("utf-8", "ignore")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_row(self) -> dict[str, Any]:
        """Ligne aplatie pour l'éditeur tabulaire Streamlit."""
        d = self.to_dict()
        d["secondary_units"] = ", ".join(self.secondary_units)
        d["tests_linked"] = ", ".join(self.tests_linked)
        d["dsl"] = json.dumps(self.dsl, ensure_ascii=False) if self.dsl else ""
        d["target"] = json.dumps(self.target, ensure_ascii=False) if self.target else ""
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleRecord":
        fields = {f.name for f in dataclasses.fields(cls)}
        clean = {k: v for k, v in data.items() if k in fields}
        # normalise les listes éventuellement passées en chaîne
        for key in ("secondary_units", "tests_linked"):
            val = clean.get(key)
            if isinstance(val, str):
                clean[key] = [x.strip() for x in val.split(",") if x.strip()]
        for key in ("dsl", "target"):
            val = clean.get(key)
            if isinstance(val, str) and val.strip():
                try:
                    clean[key] = json.loads(val)
                except json.JSONDecodeError:
                    clean[key] = {}
        return cls(**clean)


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def make_rule_id(prefix: str, basis: str) -> str:
    """ID stable et lisible : PREFIX-<hash8>."""
    h = hashlib.sha1(basis.encode("utf-8", "ignore")).hexdigest()[:8].upper()
    return f"{prefix}-{h}"


def slugify(text: str, max_len: int = 48) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return s[:max_len] or "rule"
