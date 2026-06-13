import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pageprint import build_pageprint_input_data


def test_uppercase_heading_is_not_protected_as_acronym_by_default():
    input_data = build_pageprint_input_data(
        page_structure={
            "page_role": "body",
            "page_family": "body_text",
            "document_type": "technical_book",
            "layout_type": "single_column",
            "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
            "blocks": [{
                "id": "b1",
                "bbox": [50, 50, 550, 100],
                "role": "section_heading",
                "lines": [{"id": "l1", "bbox": [50, 50, 550, 90], "line_text": "BACKGROUND", "phrases": [{"id": "p1", "bbox": [50, 50, 550, 90], "texte": "BACKGROUND"}]}],
            }],
        },
        source_context={"language": {"source_lang": "en", "target_lang": "fr"}},
    )
    item = input_data["views"]["translation_plan"][0]
    assert item["source_text"] == "BACKGROUND"
    assert "BACKGROUND" not in item["protected_tokens"]
