from tests.pageprint.test_table_grid_detection import _table_page


def test_raw_table_row_absent_from_translation_plan():
    assert all("    " not in item["source_text"] for item in _table_page()["views"]["translation_plan"])
