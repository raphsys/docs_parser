import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pageprint import build_pageprint_input_data


def test_body_page_produces_translation_plan():
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
                "lines": [{
                    "id": "l1",
                    "bbox": [50, 50, 550, 90],
                    "line_text": "This is a real sentence that should be translated.",
                    "phrases": [{"id": "p1", "bbox": [50, 50, 550, 90], "texte": "This is a real sentence that should be translated."}],
                }],
            }],
        },
        source_context={"language": {"source_lang": "en", "target_lang": "fr"}},
    )
    assert input_data["views"]["translation_plan"]
    assert input_data["debug"]["audit_status"]["functional_status"] == "ok"
