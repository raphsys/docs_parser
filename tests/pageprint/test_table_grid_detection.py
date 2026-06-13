import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pageprint import build_pageprint_input_data


def _table_page():
    return build_pageprint_input_data(
        page_structure={
            "page_role": "table",
            "layout_type": "table_dominant",
            "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
            "blocks": [{
                "id": "tbl",
                "bbox": [40, 80, 560, 180],
                "role": "table",
                "lines": [
                    {"id": "r1", "bbox": [40, 80, 560, 105], "line_text": "Command    Description    Count", "phrases": [{"id": "r1p", "bbox": [40, 80, 560, 105], "texte": "Command    Description    Count"}]},
                    {"id": "r2", "bbox": [40, 110, 560, 135], "line_text": "mkdir    Create a directory    3", "phrases": [{"id": "r2p", "bbox": [40, 110, 560, 135], "texte": "mkdir    Create a directory    3"}]},
                ],
            }],
        },
        source_context={"language": {"source_lang": "en", "target_lang": "fr"}},
    )


def test_table_grid_detection():
    input_data = _table_page()
    assert input_data["logical_structures"]["tables"]
    assert len(input_data["logical_structures"]["tables"][0]["cells"]) == 6


def test_table_command_cells_preserved():
    table = _table_page()["logical_structures"]["tables"][0]
    command = next(cell for cell in table["cells"] if cell["text"] == "mkdir")
    assert command["translation_mode"] == "preserve_text_exactly"


def test_table_description_cells_translated():
    texts = [item["source_text"] for item in _table_page()["views"]["translation_plan"]]
    assert "Create a directory" in texts


def test_no_raw_table_row_in_translation_plan():
    texts = [item["source_text"] for item in _table_page()["views"]["translation_plan"]]
    assert "mkdir    Create a directory    3" not in texts
