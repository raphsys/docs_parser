from pagetranslate.quality import unit_quality


def _profile():
    return {"source_lang": "en", "target_lang": "fr"}


def test_qa_reason_repeated_output():
    q = unit_quality("the challenges", "les défis et les défis et les défis", {}, _profile())
    assert "repeated_output" in q["qa_reasons"]
    assert q["needs_review"] is True


def test_qa_reason_source_fragment_and_dehyphenation():
    q = unit_quality("improve the unsu-", "améliorer le non-", {}, _profile())
    assert "source_fragment" in q["qa_reasons"]
    assert "dehyphenation_needed" in q["qa_reasons"]


def test_clean_translation_has_no_blocking_qa_reason():
    q = unit_quality(
        "Deep learning architectures for character recognition",
        "Architectures d'apprentissage profond pour la reconnaissance des caractères",
        {},
        _profile(),
    )
    assert "repeated_output" not in q["qa_reasons"]
    assert "source_fragment" not in q["qa_reasons"]
