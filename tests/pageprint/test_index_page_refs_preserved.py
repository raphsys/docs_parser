from tests.pageprint.test_index_page_detection import _index_page


def test_index_refs_not_in_translation_text():
    assert all("242" not in item["source_text"] for item in _index_page()["views"]["translation_plan"])
