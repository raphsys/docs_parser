from tests.pageprint.test_table_grid_detection import _table_page


def test_command_cell_not_sent_to_translation_plan():
    texts = [item["source_text"] for item in _table_page()["views"]["translation_plan"]]
    assert "mkdir" not in texts
