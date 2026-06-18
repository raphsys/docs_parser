from pagereconstruct import compile_page_render_plan
from render_contract_audit import audit_render_contract
from tests.pagereconstruct._fixtures import translated_input_data


def test_render_contract_audit_accepts_formula_preservation_pipeline():
    data = translated_input_data()
    plan = compile_page_render_plan(data).to_dict()
    audit = audit_render_contract(plan, data)

    assert audit["status"] == "ok"
    assert audit["preserved_visual_count"] >= 1
    assert audit["hard_blockers"] == []
    formula_rows = [r for r in audit["rows"] if r["source_unit_id"] == "fml1"]
    assert formula_rows
    row = formula_rows[0]
    assert row["in_protected_regions"] is True
    assert row["in_preserved_layers"] is True
    assert row["in_preservation_ops"] is True
    assert row["in_text_ops"] is False
    assert row["in_translated_text_layer"] is False


def test_render_contract_audit_blocks_preserved_visual_text_and_patch_leak():
    data = translated_input_data()
    plan = {
        "layers": {
            "translated_text": [
                {
                    "id": "ru_bad",
                    "source_unit_ids": ["fml1"],
                    "layout_bbox": [60, 120, 180, 140],
                    "patch_bbox": [60, 120, 180, 140],
                }
            ],
            "patches": [
                {"unit_id": "patch_bad", "bbox": [60, 120, 180, 140], "reason": "text_removal"}
            ],
            "preserved_underlays": [],
            "preserved_overlays": [],
        },
        "protected_regions": [],
        "render_ops": [
            {
                "op_type": "text",
                "run_id": "run_bad",
                "source_unit_ids": ["fml1"],
                "lines": [{"text": "E = mc^2", "x": 60, "y_top": 120}],
            }
        ],
    }

    audit = audit_render_contract(plan, data)

    assert audit["status"] == "ko"
    blockers = set(audit["hard_blockers"])
    assert "preserved_visual_in_translated_text_layer" in blockers
    assert "preserved_visual_as_textop" in blockers
    assert "preserved_visual_missing_protected_region" in blockers
    assert "preserved_visual_missing_preserved_layer" in blockers
    assert "preserved_visual_missing_preservationop" in blockers
    assert "preserved_visual_covered_by_patch" in blockers
