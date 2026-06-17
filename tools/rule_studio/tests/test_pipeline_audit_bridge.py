"""Pont vers les vrais audits pipeline (directive §16)."""

from tools.rule_studio.core.pipeline_audit_bridge import (
    AUDIT_TARGETS, audit_availability, run_audit_for)


def test_targets_cover_fundamentals():
    assert {"NO-DROP-001", "PT-COVERAGE-001", "PR-COVERAGE-001", "RENDER-COVERAGE-001"} <= set(AUDIT_TARGETS)


def test_availability_reports_real_audits():
    avail = audit_availability()
    assert set(avail) == set(AUDIT_TARGETS)
    # au moins l'audit de couverture reconstruction doit être importable
    assert avail["PR-COVERAGE-001"]["available"] is True


def test_run_without_context_is_ready_or_unavailable():
    out = run_audit_for("PR-COVERAGE-001")
    assert out["status"] in {"ready", "audit_unavailable"}


def test_unbound_rule_returns_no_audit():
    assert run_audit_for("UNKNOWN-999")["status"] == "no_audit_bound"
