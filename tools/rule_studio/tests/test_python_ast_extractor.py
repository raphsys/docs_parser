"""Tests de l'extracteur Python AST."""

from tools.rule_studio.extractors.python_ast_extractor import extract_python_rules


def test_extracts_domain_branch():
    src = (
        "def classify_page(page):\n"
        "    if page.visual_density > 0.7 and page.text_density < 0.2:\n"
        "        page_role = 'cover'\n"
        "    return page_role\n"
    )
    rules = extract_python_rules(src, "pageprint/page_classifier.py")
    assert rules, "doit extraire au moins une règle métier"
    assert any(r.pipeline_unit == "PAGE_CLASSIFICATION" for r in rules)
    assert all(r.source_type == "python_ast" for r in rules)


def test_ignores_trivial_technical_if():
    src = "def f(x):\n    if x is None:\n        return 0\n    return x + 1\n"
    rules = extract_python_rules(src, "utils/math_helpers.py")
    # Pas de vocabulaire métier => pas de règle.
    assert rules == []


def test_extracts_enum_schema():
    src = (
        "from enum import Enum\n"
        "class RenderPolicy(Enum):\n"
        "    ANCHORED = 'anchored_text'\n"
        "    FLOW = 'paragraph_flow'\n"
    )
    rules = extract_python_rules(src, "pagereconstruct/policy.py")
    assert any(r.title.startswith("[schema]") for r in rules)


def test_confidence_within_bounds():
    src = (
        "def validate_bbox(obj):\n"
        "    assert obj.bbox is not None, 'bbox protected required'\n"
    )
    rules = extract_python_rules(src, "pagereconstruct/validator.py")
    for r in rules:
        assert 0.0 <= r.confidence <= 1.0
