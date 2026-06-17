"""Tests de classification par unité de pipeline + extracteurs config/markdown."""

from tools.rule_studio.core.classifier import classify
from tools.rule_studio.extractors.config_extractor import extract_config_rules
from tools.rule_studio.extractors.markdown_extractor import extract_markdown_rules
from tools.rule_studio.extractors.comment_rule_extractor import extract_comment_rules


def test_classify_role_resolver():
    unit, secondary, conf = classify("pageprint/role_resolver.py", "resolve_role", "role == None")
    assert unit == "SEMANTIC_ROLE_RESOLUTION"
    assert conf > 0


def test_classify_unknown():
    unit, _, conf = classify("random/file.py", None, "x = 1")
    assert unit == "UNKNOWN"
    assert conf == 0.0


def test_config_extractor_yaml():
    text = "thresholds:\n  cover_visual_density: 0.7\npolicy:\n  default_mode: anchored\n"
    rules = extract_config_rules(text, "config/rules.yaml", ".yaml")
    assert any("threshold" in r.title.lower() or "policy" in r.title.lower() for r in rules)
    assert all(r.source_type == "config" for r in rules)


def test_markdown_normative_extraction():
    text = "# Spec\n\nUne page de couverture ne doit pas utiliser paragraph_flow.\n"
    rules = extract_markdown_rules(text, "docs/CONTRACT.md")
    assert rules
    assert rules[0].source_type == "documentation"
    assert rules[0].editable is False


def test_comment_block_extraction():
    src = (
        "# RULE-ID: PP-COVER-001\n"
        "# RULE-UNIT: PAGE_CLASSIFICATION\n"
        "# RULE-TYPE: classification\n"
        "# RULE-OBJECTIVE: Detect cover\n"
        "# RULE-MANAGED: true\n"
        "if page_index == 0:\n"
        "    page_role = 'cover'\n"
        "# END-RULE\n"
    )
    rules = extract_comment_rules(src, "pageprint/page_classifier.py")
    assert len(rules) == 1
    assert rules[0].id == "PP-COVER-001"
    assert rules[0].managed is True
    assert rules[0].confidence >= 0.9
