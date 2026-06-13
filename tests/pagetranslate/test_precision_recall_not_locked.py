from pagetranslate.terminology import explicit_protected_terms


def test_precision_recall_not_locked():
    terms = explicit_protected_terms({"terminology": {"terms_path": "ai_models/translation/terminology_en_fr.json"}})
    assert "precision" not in terms
    assert "recall" not in terms

