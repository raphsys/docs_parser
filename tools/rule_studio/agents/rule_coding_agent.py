"""RuleCodingAgent — décision en langage naturel -> patch proposé (jamais appliqué).

Produit fichiers cibles, patch, tests, risque, statut. N'écrit pas dans le projet ;
le patch est écrit dans data/agent_runs/ et appliqué seulement plus tard, après
diff + tests + confirmation (directive §5, §12, §13)."""

from __future__ import annotations

from ..core.models import RuleRecord
from .model_client import RuleStudioModelClient, DummyModelClient


class RuleCodingAgent:
    def __init__(self, client: RuleStudioModelClient | None = None) -> None:
        self.client = client or DummyModelClient()

    def generate(self, rule: RuleRecord, project_context: dict | None = None) -> dict:
        out = self.client.natural_language_to_patch(rule.to_dict(), project_context) or {}
        status = out.get("status") or "needs_human_review"
        rule.coding_agent_status = status
        rule.coding_agent_report = out.get("explanation") or ""
        # le patch n'est jamais appliqué ici ; on mémorise le statut de cycle de vie
        if status == "patch_generated":
            rule.lifecycle_status = "patch_generated"
        elif status == "impossible_without_context":
            rule.lifecycle_status = "needs_human_review"
        else:
            rule.lifecycle_status = "needs_human_review"
        rule.touch()
        return {
            "status": status,
            "target_files": out.get("target_files") or [],
            "patch": out.get("patch") or "",
            "tests_to_run": out.get("tests_to_run") or [],
            "risk_level": out.get("risk_level") or rule.risk_level,
            "explanation": out.get("explanation") or "",
        }
