"""AgentRunner — orchestre les agents et trace chaque exécution.

Chaque run écrit un rapport JSON dans data/agent_runs/ (directive §17.2). Le
runner n'applique jamais de patch (séparation explication/preuve/application)."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from ..core.models import RuleRecord
from .model_client import RuleStudioModelClient, DummyModelClient
from .rule_interpreter_agent import RuleInterpreterAgent
from .rule_coding_agent import RuleCodingAgent
from .rule_validation_agent import RuleValidationAgent

_DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "data" / "agent_runs"


class AgentRunner:
    def __init__(self, client: RuleStudioModelClient | None = None,
                 runs_dir: str | Path | None = None) -> None:
        self.client = client or DummyModelClient()
        self.interpreter = RuleInterpreterAgent(self.client)
        self.coder = RuleCodingAgent(self.client)
        self.validator = RuleValidationAgent(self.client)
        self.runs_dir = Path(runs_dir) if runs_dir else _DEFAULT_RUNS_DIR

    def _trace(self, agent_type: str, rule: RuleRecord, output: dict,
               patch_path: str | None = None, test_result: str | None = None) -> dict:
        rec = {
            "agent_run_id": uuid.uuid4().hex[:12],
            "agent_type": agent_type,
            "rule_id": rule.id,
            "input_summary": (rule.natural_language_rule or rule.title or rule.id)[:200],
            "output_summary": json.dumps(output, ensure_ascii=False)[:500],
            "patch_path": patch_path,
            "test_result": test_result,
            "status": output.get("status") or output.get("validation_agent_status")
                      or rule.ai_interpretation_status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        }
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            (self.runs_dir / f"{agent_type}_{rec['agent_run_id']}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return rec

    def interpret(self, rule: RuleRecord, context: dict | None = None) -> dict:
        self.interpreter.interpret(rule, context)
        return self._trace("interpreter", rule, {"status": rule.ai_interpretation_status})

    def generate_patch(self, rule: RuleRecord, context: dict | None = None) -> dict:
        out = self.coder.generate(rule, context)
        patch_path = None
        if out.get("patch"):
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            patch_path = str(self.runs_dir / f"patch_{rule.id}.diff")
            Path(patch_path).write_text(out["patch"], encoding="utf-8")
            rule.coding_agent_patch_path = patch_path
        self._trace("coding", rule, out, patch_path=patch_path)
        return out

    def validate(self, rule: RuleRecord, patch: str = "", test_output: str | None = None) -> dict:
        out = self.validator.validate(rule, patch=patch, test_output=test_output)
        self._trace("validation", rule, out, test_result=test_output)
        return out
