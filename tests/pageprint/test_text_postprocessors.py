from pageprint.text_postprocessors import (
    ends_with_break_hyphen,
    has_repeated_ngram,
    merge_hyphenated_segments,
    repair_hyphenation,
)


def test_repair_intra_text_hyphenation():
    assert repair_hyphenation("improve the unsu-\npervised network") == "improve the unsupervised network"
    assert repair_hyphenation("classifica- tion accuracy") == "classification accuracy"


def test_repair_keeps_lexical_hyphens():
    assert repair_hyphenation("state-of-the-art model") == "state-of-the-art model"
    assert repair_hyphenation("semi-supervised setup") == "semi-supervised setup"
    assert repair_hyphenation("VGG-16 backbone") == "VGG-16 backbone"
    assert repair_hyphenation("the F-score metric") == "the F-score metric"


def test_ends_with_break_hyphen():
    assert ends_with_break_hyphen("improve the unsu-") == "unsu"
    assert ends_with_break_hyphen("state-") is None  # lexical prefix
    assert ends_with_break_hyphen("plain text") is None


def test_merge_cross_segment_hyphenation():
    segs = [
        {"source_text": "improve the unsu-", "role": "table_body_cell", "source_unit_ids": ["a"], "translation_mode": "translate"},
        {"source_text": "pervised deep learning algorithms.", "role": "table_body_cell", "source_unit_ids": ["b"], "translation_mode": "translate"},
    ]
    merged = merge_hyphenated_segments(segs)
    assert len(merged) == 1
    assert merged[0]["source_text"] == "improve the unsupervised deep learning algorithms."
    assert merged[0]["source_unit_ids"] == ["a", "b"]
    assert "dehyphenation_merge" in merged[0]["normalization_applied"]


def test_repeated_ngram_detection():
    assert has_repeated_ngram("les défis et les défis et les défis") is True
    assert has_repeated_ngram("une phrase normale et complète sans répétition excessive") is False
