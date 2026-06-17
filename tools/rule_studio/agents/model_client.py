"""Interface IA branchable pour Rule Studio.

L'application ne doit jamais être verrouillée sur un fournisseur. Les agents
parlent à `RuleStudioModelClient` ; les tests utilisent `DummyModelClient`
(déterministe, sans dépendance externe). Des adaptateurs réels (OpenAI / Claude /
local) peuvent être branchés plus tard sans changer les agents.

Règle d'or (directive §21) : l'IA EXPLIQUE et PROPOSE. Les tests et audits
PROUVENT. Aucun client ne doit appliquer de patch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RuleStudioModelClient(ABC):
    """Contrat minimal attendu par les agents."""

    @abstractmethod
    def explain_rule(self, rule_record: dict, project_context: dict | None = None) -> dict:
        """Code/contrat -> explication en langage naturel (champs naturels)."""

    @abstractmethod
    def natural_language_to_patch(self, rule_record: dict, project_context: dict | None = None) -> dict:
        """Décision en langage naturel -> patch proposé + tests + risque (jamais appliqué)."""

    @abstractmethod
    def review_patch(self, rule_record: dict, patch: str, test_output: str | None = None) -> dict:
        """Patch (+ sortie de tests) -> revue/validation proposée."""


class DummyModelClient(RuleStudioModelClient):
    """Client déterministe sans IA réelle. Produit des explications structurées
    à partir des champs de la règle — suffisant pour les tests et un mode hors-ligne."""

    version = "dummy-1"

    def explain_rule(self, rule_record: dict, project_context: dict | None = None) -> dict:
        title = rule_record.get("title") or rule_record.get("symbol_name") or rule_record.get("id") or "Règle"
        unit = rule_record.get("pipeline_unit") or "UNKNOWN"
        rtype = rule_record.get("rule_type") or "unknown"
        raw = (rule_record.get("raw_code") or rule_record.get("normalized_expression")
               or rule_record.get("description") or "").strip()
        risk = rule_record.get("risk_level") or "unknown"
        rule_txt = (f"Dans l'unité {unit}, cette règle de type {rtype} s'applique : "
                    f"{raw[:200] or title}.")
        understanding = (
            f"Règle de type {rtype} rattachée à {unit}. "
            "À vérifier : ce qu'elle protège, ce qu'elle risque de casser, et si "
            "elle doit être gardée, modifiée ou supprimée."
        )
        reco = "modify" if risk in {"high", "critical"} else "keep"
        return {
            "natural_language_title": title,
            "natural_language_rule": rule_txt,
            "objective": f"Garantir le comportement attendu de l'étape {unit}.",
            "understanding": understanding,
            "impact": f"Affecte le rendu/contrat de {unit}" + (
                " + unités secondaires" if rule_record.get("secondary_units") else "."),
            "risk_explanation": f"Niveau de risque estimé : {risk}.",
            "recommended_decision": reco,
            "proposed_modification": (
                "Préciser les conditions et éviter les comportements par défaut implicites."
                if reco == "modify" else ""),
            "interpretation_version": self.version,
            "interpretation_status": "ok",
        }

    def natural_language_to_patch(self, rule_record: dict, project_context: dict | None = None) -> dict:
        decision = (rule_record.get("human_decision") or rule_record.get("proposed_modification")
                    or rule_record.get("natural_language_rule") or "").strip()
        target = rule_record.get("target") or {}
        files = []
        if target.get("file"):
            files.append(target["file"])
        if rule_record.get("file_path"):
            files.append(rule_record["file_path"])
        files = list(dict.fromkeys(files))
        status = "patch_generated" if (decision and files) else "needs_human_review"
        return {
            "status": status,
            "target_files": files,
            "patch": "" if status != "patch_generated" else (
                f"# PATCH PROPOSÉ (non appliqué) pour {rule_record.get('id')}\n"
                f"# Décision : {decision[:160]}\n"
                f"# Fichiers cibles : {', '.join(files)}\n"),
            "tests_to_run": list(rule_record.get("tests_linked") or []),
            "risk_level": rule_record.get("risk_level") or "unknown",
            "explanation": (
                f"Patch dérivé de la décision humaine pour {rule_record.get('id')}."
                if status == "patch_generated"
                else "Décision ou fichier cible manquant : revue humaine requise."),
        }

    def review_patch(self, rule_record: dict, patch: str, test_output: str | None = None) -> dict:
        ok = bool(patch) and (test_output is None or "fail" not in (test_output or "").lower())
        return {
            "validation_agent_status": "tests_passed" if ok else "needs_human_review",
            "validation_report": (
                "Patch présent" + (", tests OK" if test_output and ok else "")
                if ok else "Patch absent ou tests en échec — revue humaine requise."),
        }
