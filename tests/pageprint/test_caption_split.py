import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pageprint import build_pageprint_input_data


def _caption_page():
    return build_pageprint_input_data(
        page_structure={
            "page_role": "body",
            "layout_type": "single_column",
            "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
            "blocks": [{
                "id": "cap",
                "bbox": [50, 400, 550, 440],
                "role": "caption",
                "lines": [{"id": "c1", "bbox": [50, 400, 550, 430], "line_text": "Figure 2.1 Traditional ML algorithms require features.", "phrases": [{"id": "c1p", "bbox": [50, 400, 550, 430], "texte": "Figure 2.1 Traditional ML algorithms require features."}]}],
            }],
        },
        source_context={"language": {"source_lang": "en", "target_lang": "fr"}},
    )


def test_caption_split():
    caption = _caption_page()["logical_structures"]["captions"][0]
    assert caption["label"] == "Figure"
    assert caption["number"] == "2.1"
    assert caption["caption_text"] == "Traditional ML algorithms require features."


def test_figure_number_preserved():
    assert ("2.1", "caption_number") in {(item["text"], item["reason"]) for item in _caption_page()["views"]["preservation_plan"]}


def test_no_raw_caption_block_translation():
    texts = [item["source_text"] for item in _caption_page()["views"]["translation_plan"]]
    assert "Figure 2.1 Traditional ML algorithms require features." not in texts
    assert "Traditional ML algorithms require features." in texts
