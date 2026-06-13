import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pageprint import build_pageprint_input_data
from pageprint.functional_validator import validate_functional


def test_validator_ko_when_translatable_text_has_empty_translation_plan():
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
                "lines": [{"id": "l1", "bbox": [50, 50, 550, 90], "line_text": "This text must be planned.", "phrases": [{"id": "p1", "bbox": [50, 50, 550, 90], "texte": "This text must be planned."}]}],
            }],
        },
        source_context={"language": {"source_lang": "en", "target_lang": "fr"}},
    )
    input_data["views"]["translation_plan"] = []
    report = validate_functional(input_data)
    assert report["functional_status"] == "ko"
    assert "translation_plan_empty_but_translatable_text_exists" in report["errors"]
    assert "fallback_required_after_pageprint" in report["errors"]
