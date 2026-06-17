"""Workflows Studio (Phases 4/5) testés sans UI ni modèle IA externe."""

from tools.rule_studio.core.models import RuleRecord
from tools.rule_studio.core.studio import Studio


def _studio(tmp_path):
    db = tmp_path / "rules.sqlite"
    managed = tmp_path / "managed_rules.yaml"
    return Studio(str(tmp_path), str(db), str(managed))


def _candidate(rid="C-1"):
    return RuleRecord(id=rid, title="Nouvelle règle", pipeline_unit="RECONSTRUCTION",
                      rule_type="constraint", source_type="manual", managed=True,
                      lifecycle_status="candidate", file_path="pagereconstruct/x.py",
                      human_decision="Ne jamais supprimer un texte original.")


def test_new_candidate_rule_persisted(tmp_path):
    s = _studio(tmp_path)
    s.save(_candidate())
    r = s.get("C-1")
    assert r and r.lifecycle_status == "candidate" and r.ai_interpreted is False


def test_interpret_rules_fills_nl_and_marks(tmp_path):
    s = _studio(tmp_path)
    s.save(_candidate())
    summ = s.interpret_rules(["C-1"], only_uninterpreted=False)
    assert summ["interpreted"] == 1
    r = s.get("C-1")
    assert r.ai_interpreted is True and r.natural_language_rule


def test_generate_patch_does_not_apply(tmp_path):
    s = _studio(tmp_path)
    s.save(_candidate())
    out = s.generate_patch("C-1")
    assert out["status"] in {"patch_generated", "needs_human_review"}
    r = s.get("C-1")
    assert r.coding_agent_status == out["status"]
    # le fichier projet n'est pas touché : l'agent ne renvoie qu'une proposition
    assert "apply" not in out  # pas de champ d'application


def test_edition_persisted_and_lifecycle(tmp_path):
    s = _studio(tmp_path)
    s.save(_candidate())
    r = s.get("C-1")
    r.objective = "Objectif édité par l'humain"
    r.lifecycle_status = "edited_by_human"
    s.save(r)
    assert s.get("C-1").objective == "Objectif édité par l'humain"
    assert s.get("C-1").lifecycle_status == "edited_by_human"


def test_validate_rule_tests_status(tmp_path):
    s = _studio(tmp_path)
    s.save(_candidate())
    s.generate_patch("C-1")
    out = s.validate_rule("C-1", patch="--- a\n+++ b\n", test_output="3 passed")
    assert out["status"] in {"tests_passed", "needs_human_review"}
