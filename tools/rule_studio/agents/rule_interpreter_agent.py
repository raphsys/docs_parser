"""RuleInterpreterAgent — code/contrat -> explication en langage naturel.

Remplit les champs naturels d'un RuleRecord via un RuleStudioModelClient et met
à jour l'état d'interprétation + le cycle de vie. N'altère JAMAIS le code projet.
"""

from __future__ import annotations

import time

from ..core.models import RuleRecord
from .model_client import RuleStudioModelClient, DummyModelClient


class RuleInterpreterAgent:
    def __init__(self, client: RuleStudioModelClient | None = None) -> None:
        self.client = client or DummyModelClient()

    def interpret(self, rule: RuleRecord, project_context: dict | None = None) -> RuleRecord:
        out = self.client.explain_rule(rule.to_dict(), project_context) or {}
        for key in ("natural_language_title", "natural_language_rule", "objective",
                    "understanding", "impact", "risk_explanation",
                    "recommended_decision", "proposed_modification"):
            val = out.get(key)
            if val:
                setattr(rule, key, val)
        rule.ai_interpreted = True
        rule.ai_interpretation_version = out.get("interpretation_version") or getattr(self.client, "version", "?")
        status = out.get("interpretation_status") or "ok"
        rule.ai_interpretation_status = status
        rule.ai_last_interpreted_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        # cycle de vie : interpreted, sauf si l'agent signale une revue nécessaire
        needs_review = (status != "ok") or (rule.recommended_decision in {"needs_review", "delete"})
        if rule.lifecycle_status in {"detected", "interpreted", "needs_human_review"}:
            rule.lifecycle_status = "needs_human_review" if needs_review else "interpreted"
        rule.touch()
        return rule

    def interpret_many(self, rules: list[RuleRecord], project_context: dict | None = None,
                       only_uninterpreted: bool = True) -> dict:
        done = review = 0
        for r in rules:
            if only_uninterpreted and r.ai_interpreted:
                continue
            self.interpret(r, project_context)
            done += 1
            if r.lifecycle_status == "needs_human_review":
                review += 1
        critical = sum(1 for r in rules if r.risk_level in {"high", "critical"}
                       and r.pipeline_unit in {"TRANSLATION", "RECONSTRUCTION"})
        return {"interpreted": done, "needs_human_review": review, "critical_units": critical,
                "total": len(rules)}
