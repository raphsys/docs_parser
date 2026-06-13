from pagetranslate.terminology import apply_post_translation_glossary, check_terminology_consistency, explicit_protected_terms


def test_typed_terminology_preserve_and_preferred():
    profile = {"terminology": {"terms_path": "ai_models/translation/terminology_en_fr.json"}}
    protected = explicit_protected_terms(profile)
    assert "MLP" in protected
    assert "OCR" in protected
    assert "precision" not in protected
    translated = apply_post_translation_glossary("precision and recall in the paper", profile)
    assert "précision" in translated
    assert "rappel" in translated
    consistency = check_terminology_consistency("precision and recall", "précision et rappel", profile)
    assert not consistency["terminology_issue"]

