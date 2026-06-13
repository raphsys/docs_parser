import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pageprint import build_pageprint_input_data
from pagetranslate import build_page_translation


def _toc_input():
    return build_pageprint_input_data(
        page_structure={
            "page_role": "toc",
            "page_family": "toc",
            "document_type": "technical_book",
            "layout_type": "single_column",
            "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
            "blocks": [{
                "id": "toc_block",
                "bbox": [50, 50, 550, 180],
                "role": "toc",
                "lines": [
                    {"id": "l0", "bbox": [50, 50, 550, 80], "line_text": "CONTENTS", "phrases": [{"id": "p0", "bbox": [50, 50, 200, 80], "texte": "CONTENTS"}]},
                    {"id": "l1", "bbox": [50, 90, 550, 120], "line_text": "3.1 Image classification using MLP 93", "phrases": [{"id": "p1", "bbox": [50, 90, 550, 120], "texte": "3.1 Image classification using MLP 93"}]},
                    {"id": "l2", "bbox": [50, 130, 550, 160], "line_text": "■ Hidden layers 94", "phrases": [{"id": "p2", "bbox": [50, 130, 550, 160], "texte": "■ Hidden layers 94"}]},
                ],
            }],
        },
        source_context={"language": {"source_lang": "en", "target_lang": "fr"}},
    )


def test_toc_splits_section_title_page_and_builds_translation_plan():
    input_data = _toc_input()
    entries = input_data["logical_structures"]["toc_entries"]
    assert len(entries) == 3
    assert entries[1]["section_number"] == "3.1"
    assert entries[1]["title_text"] == "Image classification using MLP"
    assert entries[1]["page_reference"] == "93"
    assert entries[2]["title_text"] == "Hidden layers"

    plan_texts = [item["source_text"] for item in input_data["views"]["translation_plan"]]
    assert "Image classification using MLP" in plan_texts
    assert "3.1 Image classification using MLP 93" not in plan_texts
    target = next(item for item in input_data["views"]["translation_plan"] if item["source_text"] == "Image classification using MLP")
    assert target["protected_tokens"] == ["MLP"]

    preservation = {(item["text"], item["reason"]) for item in input_data["views"]["preservation_plan"]}
    assert ("93", "toc_page_reference") in preservation
    assert ("3.1", "toc_section_number") in preservation


def test_toc_pagetranslate_uses_translation_plan_without_fallback():
    result = build_page_translation(_toc_input(), dry_run=True, allow_fallback=False)
    assert result["debug"]["selection_mode"] == "translation_plan"
    assert result["debug"]["fallback_selector_used"] is False
    assert result["debug"]["generic_coalescer_used"] is False
    assert result["functional_validation"]["functional_status"] == "ok"
