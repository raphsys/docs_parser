"""Façade orchestrant scan, store, registre, usage et exports.

Sert de point d'entrée unique à l'UI Streamlit comme à la CLI, pour que la
logique ne vive pas dans la couche présentation.
"""

from __future__ import annotations

from pathlib import Path

from ..storage.rule_store import RuleStore
from .exporters import audit_report, to_csv, to_json, to_markdown
from .models import RuleRecord
from .rule_registry import load_managed_rules, save_managed_rules
from .scanner import scan_project
from .usage_analyzer import apply_usage

# Emplacements par défaut, relatifs au package rule_studio.
_PKG = Path(__file__).resolve().parent.parent
DEFAULT_DB = _PKG / "data" / "rules.sqlite"
DEFAULT_MANAGED = _PKG / "data" / "managed_rules.yaml"


class Studio:
    def __init__(
        self,
        project_root: str | Path,
        db_path: str | Path = DEFAULT_DB,
        managed_path: str | Path = DEFAULT_MANAGED,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.managed_path = Path(managed_path)
        self.store = RuleStore(db_path)

    # -- inventaire ----------------------------------------------------------
    def scan(self, with_usage: bool = True) -> dict:
        """Scanne le projet + charge les règles gérées, fusionne dans le store."""
        scan_res = scan_project(self.project_root)
        managed = load_managed_rules(self.managed_path)
        all_scanned = scan_res.rules + managed

        if with_usage:
            apply_usage(all_scanned, self.project_root)

        stats = self.store.merge_scan(all_scanned)
        stats["files_scanned"] = scan_res.files_scanned
        stats["files_skipped"] = scan_res.files_skipped
        stats["errors"] = len(scan_res.errors)
        stats["total"] = self.store.count()
        return stats

    def rules(self) -> list[RuleRecord]:
        return self.store.all()

    def get(self, rule_id: str) -> RuleRecord | None:
        return self.store.get(rule_id)

    def save(self, rec: RuleRecord) -> None:
        rec.touch()
        self.store.upsert(rec)
        if rec.managed:
            self._sync_managed()

    def delete(self, rule_id: str) -> None:
        self.store.delete(rule_id)
        self._sync_managed()

    def _sync_managed(self) -> None:
        managed = [r for r in self.store.all() if r.managed]
        save_managed_rules(self.managed_path, managed)

    # -- exports -------------------------------------------------------------
    def export_csv(self) -> str:
        return to_csv(self.store.all())

    def export_json(self) -> str:
        return to_json(self.store.all())

    def export_markdown(self) -> str:
        return to_markdown(self.store.all())

    def export_audit(self) -> str:
        return audit_report(self.store.all())

    # -- agents IA -----------------------------------------------------------
    def _runner(self):
        if getattr(self, "_agent_runner", None) is None:
            from tools.rule_studio.agents import AgentRunner
            self._agent_runner = AgentRunner()
        return self._agent_runner

    def interpret_rules(self, ids: list[str] | None = None,
                        only_uninterpreted: bool = True) -> dict:
        """Interprète en langage naturel (RuleInterpreterAgent) puis persiste."""
        from tools.rule_studio.agents import RuleInterpreterAgent
        rules = self.store.all()
        if ids:
            keep = set(ids)
            rules = [r for r in rules if r.id in keep]
        summ = RuleInterpreterAgent(self._runner().client).interpret_many(
            rules, only_uninterpreted=only_uninterpreted)
        for r in rules:
            self.save(r)
        return summ

    def generate_patch(self, rule_id: str) -> dict | None:
        """Génère un patch PROPOSÉ (jamais appliqué) + persiste l'état."""
        rec = self.store.get(rule_id)
        if not rec:
            return None
        out = self._runner().generate_patch(rec)
        self.save(rec)
        return out

    def validate_rule(self, rule_id: str, patch: str = "",
                      test_output: str | None = None) -> dict | None:
        rec = self.store.get(rule_id)
        if not rec:
            return None
        out = self._runner().validate(rec, patch=patch, test_output=test_output)
        self.save(rec)
        return out
