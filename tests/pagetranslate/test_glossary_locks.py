from pagetranslate.terminology import explicit_protected_terms


def test_glossary_locks():
    terms = explicit_protected_terms({"terminology": {"locked_terms": ["PostGIS"]}})
    assert "MLP" in terms
    assert "ReLU" in terms
    assert "PostGIS" in terms
