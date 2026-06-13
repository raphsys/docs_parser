import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pageprint import build_pageprint_input_data


def _page():
    return {
        "page_role": "body",
        "page_family": "body_text",
        "document_type": "technical_book",
        "layout_type": "single_column",
        "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
        "special_regions": [
            {"special_class": "formula", "bbox": [200, 200, 360, 250], "confidence": 0.9}
        ],
        "blocks": [
            {
                "id": "b1",
                "bbox": [50, 50, 550, 120],
                "role": "body",
                "lines": [
                    {
                        "id": "l1",
                        "bbox": [50, 50, 550, 90],
                        "line_text": "This sentence should translate.",
                        "phrases": [{"id": "p1", "bbox": [50, 50, 550, 90], "texte": "This sentence should translate."}],
                    }
                ],
            },
            {
                "id": "formula",
                "bbox": [200, 200, 360, 250],
                "role": "equation_block",
                "lines": [{"id": "fl1", "bbox": [200, 200, 360, 250], "line_text": "E = mc^2", "phrases": []}],
            },
        ],
    }


def test_region_claim_not_direct_background_policy():
    input_data = build_pageprint_input_data(page_structure=_page(), source_context={"language": {"source_lang": "en", "target_lang": "fr"}})
    assert any(r["region_type"] == "formula_candidate_region" for r in input_data["regions"])
    assert not any(r.get("skip_translation") for r in input_data["regions"])
    assert not any(r.get("region_type") == "protected_visual_region" for r in input_data["regions"])


def test_translation_plan_has_roles_and_no_fine_tokens():
    input_data = build_pageprint_input_data(page_structure=_page(), source_context={"language": {"source_lang": "en", "target_lang": "fr"}})
    plan = input_data["views"]["translation_plan"]
    assert plan
    assert all(item["role"] for item in plan)
    assert all(item["object_type"] for item in plan)
    assert all(item["semantic_kind"] for item in plan)
    source_levels = {
        unit["unit_id"]: unit["level"]
        for unit in input_data["units"]
    }
    assert all(source_levels[sid] not in {"word", "char"} for item in plan for sid in item["source_unit_ids"])


def test_functional_status_is_separate_from_schema_status():
    input_data = build_pageprint_input_data(page_structure=_page(), source_context={"language": {"source_lang": "en", "target_lang": "fr"}})
    status = input_data["debug"]["audit_status"]
    assert status["schema_status"] == "ok"
    assert status["functional_status"] == "ok"
