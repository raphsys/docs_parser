"""Tests du moteur DSL de simulation (sûreté + all/any/not)."""

import pytest

from tools.rule_studio.core.simulator import (
    SimulationError,
    simulate,
    validate_dsl,
)


def test_simple_match():
    dsl = {
        "when": {"field": "page.index", "op": "eq", "value": 0},
        "then": {"set": {"page.role": "cover"}},
    }
    res = simulate(dsl, {"page": {"index": 0}})
    assert res.matched is True
    assert res.after["page"]["role"] == "cover"
    # contexte d'origine non muté
    assert res.before["page"].get("role") is None


def test_all_conditions():
    dsl = {
        "when": {
            "all": [
                {"field": "page.visual_density", "op": "gte", "value": 0.7},
                {"field": "page.text_density", "op": "lte", "value": 0.2},
            ]
        },
        "then": {"set": {"page.role": "cover"}},
    }
    ctx = {"page": {"visual_density": 0.8, "text_density": 0.1}}
    assert simulate(dsl, ctx).matched is True
    ctx2 = {"page": {"visual_density": 0.5, "text_density": 0.1}}
    assert simulate(dsl, ctx2).matched is False


def test_any_and_not():
    dsl = {"when": {"any": [
        {"field": "object.role", "op": "missing"},
        {"field": "object.role", "op": "eq", "value": None},
    ]}, "then": {"set": {"object.render_mode": "anchored_text"}}}
    assert simulate(dsl, {"object": {}}).matched is True
    assert simulate(dsl, {"object": {"role": "caption"}}).matched is False


def test_missing_field_is_false():
    dsl = {"when": {"field": "a.b.c", "op": "gt", "value": 1}}
    assert simulate(dsl, {}).matched is False


def test_forbidden_operator_raises():
    dsl = {"when": {"field": "x", "op": "exec", "value": 1}}
    with pytest.raises(SimulationError):
        simulate(dsl, {"x": 1})


def test_validate_dsl_reports_errors():
    errors = validate_dsl({"when": {"field": "x", "op": "bogus"}, "then": {"danger": 1}})
    assert any("Opérateur" in e for e in errors)
    assert any("Action" in e for e in errors)


def test_warning_action():
    dsl = {
        "when": {"field": "tokens.siblings_same_bbox", "op": "eq", "value": True},
        "then": {
            "set": {"fine_token_geometry.reliable": False},
            "raise_warning": "fallback span",
        },
    }
    res = simulate(dsl, {"tokens": {"siblings_same_bbox": True}})
    assert res.matched
    assert "fallback span" in res.warnings
    assert res.after["fine_token_geometry"]["reliable"] is False
