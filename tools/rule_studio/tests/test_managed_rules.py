"""Présence et intégrité des règles fondamentales (directive §15, §18)."""

from pathlib import Path

from tools.rule_studio.core.rule_registry import load_managed_rules, record_to_entry

_YAML = Path(__file__).resolve().parents[1] / "data" / "managed_rules.yaml"
_FUNDAMENTALS = {"NO-DROP-001", "PT-COVERAGE-001", "PR-COVERAGE-001",
                 "RENDER-COVERAGE-001", "RULE-NL-001", "RULE-CODING-001"}


def _by_id():
    return {r.id: r for r in load_managed_rules(_YAML)}


def test_fundamentals_present():
    ids = set(_by_id())
    assert _FUNDAMENTALS <= ids


def test_no_drop_is_blocking_contract():
    r = _by_id()["NO-DROP-001"]
    assert r.rule_type == "contract" and r.blocking is True
    assert r.natural_language_rule and r.lifecycle_status == "candidate"


def test_coverage_rules_have_targets_and_nl():
    for rid in ("PT-COVERAGE-001", "PR-COVERAGE-001", "RENDER-COVERAGE-001"):
        r = _by_id()[rid]
        assert r.file_path, f"{rid} sans fichier cible"
        assert r.natural_language_rule, f"{rid} sans règle naturelle"


def test_markdown_export_includes_natural_language():
    from tools.rule_studio.core.exporters import to_markdown, to_csv
    recs = load_managed_rules(_YAML)
    md = to_markdown(recs)
    assert "natural_language_title" in md or "natural_language_rule" in md
    # le contenu naturel d'une règle fondamentale apparaît
    csv_out = to_csv(recs)
    assert "natural_language_rule" in csv_out


def test_extended_fields_round_trip():
    r = _by_id()["NO-DROP-001"]
    entry = record_to_entry(r)
    assert entry.get("blocking") is True
    assert entry.get("natural_language_rule")
    assert entry.get("lifecycle_status") == "candidate"
