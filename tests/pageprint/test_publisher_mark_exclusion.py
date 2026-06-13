import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pageprint import build_pageprint_input_data


def _artifact_page(text="Manning eBook", bbox=None):
    bbox = bbox or [40, 745, 300, 770]
    return build_pageprint_input_data(
        page_structure={
            "page_role": "body",
            "layout_type": "single_column",
            "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
            "blocks": [{
                "id": "art",
                "bbox": bbox,
                "role": "body",
                "lines": [{"id": "a1", "bbox": bbox, "line_text": text, "phrases": [{"id": "a1p", "bbox": bbox, "texte": text}]}],
            }],
        },
        source_context={"language": {"source_lang": "en", "target_lang": "fr"}},
    )


def test_publisher_mark_exclusion():
    page = _artifact_page()
    assert page["logical_structures"]["publisher_marks"]
    assert not page["views"]["translation_plan"]
    assert page["views"]["exclusion_plan"]


def test_page_number_preserved():
    page = _artifact_page("42")
    assert any(item["text"] == "42" for item in page["views"]["preservation_plan"])
    assert not page["views"]["translation_plan"]


def test_watermark_excluded():
    page = _artifact_page("DRAFT", [150, 300, 450, 360])
    assert page["logical_structures"]["watermarks"]
    assert not page["views"]["translation_plan"]
