import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pageprint import build_pageprint_input_data


def _index_page():
    return build_pageprint_input_data(
        page_structure={
            "page_role": "index",
            "layout_type": "two_column",
            "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
            "blocks": [{
                "id": "idx",
                "bbox": [50, 80, 550, 180],
                "role": "index",
                "lines": [
                    {"id": "i1", "bbox": [50, 80, 300, 105], "line_text": "PostGIS, 242", "phrases": [{"id": "i1p", "bbox": [50, 80, 300, 105], "texte": "PostGIS, 242"}]},
                    {"id": "i2", "bbox": [65, 110, 330, 135], "line_text": "  creating spatial database, 242-243", "phrases": [{"id": "i2p", "bbox": [65, 110, 330, 135], "texte": "  creating spatial database, 242-243"}]},
                ],
            }],
        },
        source_context={"language": {"source_lang": "en", "target_lang": "fr"}},
    )


def test_index_page_detection():
    assert _index_page()["logical_structures"]["index_entries"]


def test_index_entry_builder():
    entry = _index_page()["logical_structures"]["index_entries"][0]
    assert entry["head_term"] == "PostGIS"
    assert entry["page_refs"] == ["242"]


def test_index_page_refs_preserved():
    refs = {item["text"] for item in _index_page()["views"]["preservation_plan"]}
    assert "242" in refs or "242-243" in refs


def test_no_index_line_raw_translation():
    texts = [item["source_text"] for item in _index_page()["views"]["translation_plan"]]
    assert "PostGIS, 242" not in texts
    assert all("242" not in text for text in texts)
