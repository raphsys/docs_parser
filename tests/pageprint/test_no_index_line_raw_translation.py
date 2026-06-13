from tests.pageprint.test_index_page_detection import _index_page


def test_raw_index_line_absent():
    assert "PostGIS, 242" not in [item["source_text"] for item in _index_page()["views"]["translation_plan"]]
