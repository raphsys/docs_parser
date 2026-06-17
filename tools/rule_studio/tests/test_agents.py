"""Tests des agents IA (DummyModelClient — aucun modèle externe requis)."""

from tools.rule_studio.core.models import RuleRecord
from tools.rule_studio.agents import (
    DummyModelClient, RuleInterpreterAgent, RuleCodingAgent, RuleValidationAgent, AgentRunner)


def _rule(**kw):
    base = dict(id="R-1", title="Rendu par défaut", pipeline_unit="RECONSTRUCTION",
                rule_type="heuristic", file_path="pagereconstruct/x.py",
                raw_code="if role is None: mode='paragraph_flow'", risk_level="high")
    base.update(kw)
    return RuleRecord(**base)


def test_interpreter_fills_natural_language():
    r = RuleInterpreterAgent(DummyModelClient()).interpret(_rule())
    assert r.ai_interpreted is True
    assert r.natural_language_rule and r.objective and r.understanding
    assert r.recommended_decision in {"keep", "modify", "delete", "needs_review"}
    assert r.lifecycle_status in {"interpreted", "needs_human_review"}


def test_interpreter_high_risk_recommends_modify_review():
    r = RuleInterpreterAgent().interpret(_rule(risk_level="critical"))
    assert r.recommended_decision == "modify"


def test_interpret_many_summary():
    rules = [_rule(id=f"R-{i}", risk_level="low", pipeline_unit="QA") for i in range(3)]
    summ = RuleInterpreterAgent().interpret_many(rules)
    assert summ["interpreted"] == 3 and summ["total"] == 3


def test_coding_agent_generates_patch_without_applying(tmp_path):
    r = _rule(human_decision="Ne jamais supprimer un texte original.",
              file_path="pagetranslate/projection.py")
    out = RuleCodingAgent().generate(r)
    assert out["status"] in {"patch_generated", "needs_human_review"}
    assert r.coding_agent_status == out["status"]
    # le projet n'est jamais modifié par l'agent
    assert any("projection.py" in f for f in out["target_files"])


def test_coding_agent_needs_review_without_target():
    r = _rule(file_path="", target={})
    r.human_decision = "fais quelque chose"
    out = RuleCodingAgent().generate(r)
    assert out["status"] == "needs_human_review"


def test_validation_agent_tests_prove_not_explanation():
    r = _rule()
    RuleInterpreterAgent().interpret(r)
    out = RuleValidationAgent().validate(r, patch="--- a\n+++ b\n", test_output="2 passed")
    assert out["status"] in {"tests_passed", "needs_human_review"}
    out2 = RuleValidationAgent().validate(r, patch="x", test_output="1 failed")
    assert r.validation_status == "invalid" and r.lifecycle_status == "tests_failed"


def test_agent_runner_writes_trace(tmp_path):
    runner = AgentRunner(runs_dir=tmp_path)
    r = _rule()
    runner.interpret(r)
    out = runner.generate_patch(r)
    files = list(tmp_path.glob("*.json"))
    assert files, "agent run trace written"
    if out.get("patch"):
        assert (tmp_path / f"patch_{r.id}.diff").is_file()
