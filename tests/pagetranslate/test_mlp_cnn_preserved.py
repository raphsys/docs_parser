from pagetranslate.terminology import explicit_protected_terms


def test_mlp_cnn_preserved():
    terms = explicit_protected_terms({"terminology": {"terms_path": "ai_models/translation/terminology_en_fr.json"}})
    assert "MLP" in terms
    assert "CNN" in terms

