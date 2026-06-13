from tests.pageprint.test_table_grid_detection import _table_page


def test_description_cell_is_translation_unit():
    assert "Create a directory" in [item["source_text"] for item in _table_page()["views"]["translation_plan"]]
