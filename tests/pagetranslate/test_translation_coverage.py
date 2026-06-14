from pagetranslate.quality import assess_translation_quality


def _unit(src, tr, status="translated"):
    return {"status": status, "source_text": src, "translated_text": tr, "quality": {}}


def test_truncated_translation_detected():
    long_src = "x" * 2000
    q = assess_translation_quality([_unit(long_src, "y" * 600)])
    assert q["translation_coverage_ratio"] < 0.85
    assert q["translation_truncated"] is True


def test_full_translation_not_truncated():
    q = assess_translation_quality([_unit("Hello world " * 40, "Bonjour le monde " * 40)])
    assert q["translation_truncated"] is False
    assert q["translation_coverage_ratio"] >= 0.85


def test_short_text_never_truncated():
    q = assess_translation_quality([_unit("Hi", "")])
    assert q["translation_truncated"] is False
