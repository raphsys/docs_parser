"""RuleValidationAgent — juge cohérence/patch/tests (l'IA ne prouve pas, les tests prouvent).

Distingue les types de validation : humaine, automatique, par tests, par
simulation, par audit pipeline (directive §6). Ne marque jamais une règle
'valide' sur la seule explication IA."""

from __future__ import annotations

from ..core.models import RuleRecord
from .model_client import RuleStudioModelClient, DummyModelClient

VALIDATION_KINDS = ["human", "automatic", "tests", "simulation", "pipeline_audit"]


class RuleValidationAgent:
    def __init__(self, client: RuleStudioModelClient | None = None) -> None:
        self.client = client or DummyModelClient()

    def validate(self, rule: RuleRecord, *, patch: str = "", test_output: str | None = None,
                 kind: str = "tests") -> dict:
        checks = {
            "has_natural_language": bool(rule.natural_language_rule),
            "has_test": bool(rule.tests_linked) or not rule.requires_tests,
            "has_patch": bool(patch) or rule.coding_agent_status == "patch_generated",
            "tests_passed": test_output is None or "fail" not in (test_output or "").lower(),
        }
        out = self.client.review_patch(rule.to_dict(), patch, test_output) or {}
        status = out.get("validation_agent_status")
        if not status:
            status = "tests_passed" if all(checks.values()) else "needs_human_review"
        rule.validation_agent_status = status
        rule.validation_report = out.get("validation_report") or "; ".join(
            f"{k}={'ok' if v else 'ko'}" for k, v in checks.items())
        # statut de validation distinct du cycle de vie
        if test_output is not None:
            rule.validation_status = "valid" if checks["tests_passed"] else "invalid"
            rule.lifecycle_status = "tests_passed" if checks["tests_passed"] else "tests_failed"
        rule.touch()
        return {"validation_kind": kind, "status": status, "checks": checks,
                "report": rule.validation_report}
