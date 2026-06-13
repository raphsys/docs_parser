from tests.pageprint.test_caption_split import _caption_page


def test_caption_prefix_not_translated():
    assert all(not item["source_text"].startswith("Figure 2.1") for item in _caption_page()["views"]["translation_plan"])
