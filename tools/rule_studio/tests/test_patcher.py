"""Tests du patcher : diff généré, non-application sans confirmation, exports."""

from pathlib import Path

from tools.rule_studio.core import patcher
from tools.rule_studio.core.exporters import audit_report, to_csv, to_markdown
from tools.rule_studio.core.models import RuleRecord


def _rule(target: str) -> RuleRecord:
    return RuleRecord(
        id="PP-COVER-001",
        title="Detect cover",
        pipeline_unit="PAGE_CLASSIFICATION",
        rule_type="classification",
        objective="Detect visual cover",
        managed=True,
        target={"file": target},
    )


def test_diff_generated(tmp_path: Path):
    f = tmp_path / "mod.py"
    f.write_text("x = 1\n", encoding="utf-8")
    prop = patcher.propose_comment_patch(_rule("mod.py"), tmp_path)
    assert "RULE-ID: PP-COVER-001" in prop.proposed
    assert prop.diff  # diff non vide


def test_no_apply_without_confirm(tmp_path: Path):
    f = tmp_path / "mod.py"
    f.write_text("x = 1\n", encoding="utf-8")
    prop = patcher.propose_comment_patch(_rule("mod.py"), tmp_path)
    prop.safe = True
    out = patcher.apply_patch(prop, tmp_path, confirm=False, make_backup=False)
    assert out.applied is False
    assert f.read_text(encoding="utf-8") == "x = 1\n"


def test_apply_with_confirm(tmp_path: Path):
    f = tmp_path / "mod.py"
    f.write_text("x = 1\n", encoding="utf-8")
    prop = patcher.propose_comment_patch(_rule("mod.py"), tmp_path)
    prop.safe = True
    out = patcher.apply_patch(prop, tmp_path, confirm=True, make_backup=False)
    assert out.applied is True
    assert "RULE-ID: PP-COVER-001" in f.read_text(encoding="utf-8")
    # rollback
    assert patcher.rollback(out, tmp_path) is True
    assert f.read_text(encoding="utf-8") == "x = 1\n"


def test_exports():
    rules = [_rule("a.py")]
    csv = to_csv(rules)
    assert "PP-COVER-001" in csv
    md = to_markdown(rules)
    assert "PP-COVER-001" in md
    report = audit_report(rules)
    assert "Rapport d'audit" in report
