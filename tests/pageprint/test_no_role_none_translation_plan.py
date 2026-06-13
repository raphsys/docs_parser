import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pageprint import build_pageprint_input_data


def test_no_role_none_in_translation_plan():
    input_data = build_pageprint_input_data(
        page_structure={
            "page_role": "body",
            "page_family": "body_text",
            "document_type": "technical_book",
            "layout_type": "single_column",
            "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
            "blocks": [{
                "id": "b1",
                "bbox": [50, 50, 550, 120],
                "role": "body",
                "lines": [{"id": "l1", "bbox": [50, 50, 550, 90], "line_text": "Translate this body sentence.", "phrases": [{"id": "p1", "bbox": [50, 50, 550, 90], "texte": "Translate this body sentence."}]}],
            }],
        },
        source_context={"language": {"source_lang": "en", "target_lang": "fr"}},
    )
    assert input_data["views"]["translation_plan"]
    assert all(item.get("role") for item in input_data["views"]["translation_plan"])
    assert all(item.get("object_type") for item in input_data["views"]["translation_plan"])
    assert all(item.get("semantic_kind") for item in input_data["views"]["translation_plan"])
