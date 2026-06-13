import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pageprint import build_pageprint_input_data


def _body_page():
    return build_pageprint_input_data(
        page_structure={
            "page_role": "body",
            "page_family": "body_text",
            "layout_type": "single_column",
            "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
            "blocks": [{
                "id": "b1",
                "bbox": [50, 80, 550, 150],
                "role": "body",
                "lines": [
                    {"id": "l1", "bbox": [50, 80, 550, 105], "line_text": "This paragraph starts with a real sentence.", "phrases": [{"id": "p1", "bbox": [50, 80, 550, 105], "texte": "This paragraph starts with a real sentence."}]},
                    {"id": "l2", "bbox": [50, 108, 550, 135], "line_text": "It continues on the next visual line.", "phrases": [{"id": "p2", "bbox": [50, 108, 550, 135], "texte": "It continues on the next visual line."}]},
                ],
            }],
        },
        source_context={"language": {"source_lang": "en", "target_lang": "fr"}},
    )


def test_body_paragraph_builder_groups_lines():
    input_data = _body_page()
    paragraphs = input_data["logical_structures"]["body_paragraphs"]
    assert len(paragraphs) == 1
    assert "It continues" in paragraphs[0]["text"]


def test_body_no_visual_fallback_when_paragraphs_exist():
    input_data = _body_page()
    assert input_data["semantic_system"]["segment_source"] == "logical_body_paragraphs"
    assert input_data["views"]["translation_plan"][0]["role"] == "body_paragraph"
