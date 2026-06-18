from pagereconstruct.plan_compiler import _enforce_preserved_visual_render_contract, _box_intersection
from pagereconstruct.schema import PageRenderPlan, PatchZone
from source_ownership import build_source_ownership


def test_v21_enforces_preserved_visual_layers_and_clips_patches():
    data = {
        "page": {"geometry": {"width": 200, "height": 200}},
        "units": [
            {"unit_id": "line_formula", "level": "line", "geometry": {"bbox": [10, 10, 50, 20]}, "content": {"text": "E=mc^2"}},
        ],
        "regions": [
            {"region_id": "reg_formula", "region_type": "formula_region", "bbox": [10, 10, 50, 20], "members": {"line_ids": ["line_formula"]}},
        ],
    }
    ownership = build_source_ownership(data)
    assert ownership["line_formula"]["state"] == "preserved_visual"

    plan = PageRenderPlan(
        page={"page_index": 0, "width_pt": 200, "height_pt": 200},
        patches=[PatchZone(op_type="patch_text_zone", unit_id="txt1", bbox=[8, 8, 52, 22])],
    )
    findings = []
    _enforce_preserved_visual_render_contract(plan, data, ownership, findings)

    assert plan.protected_regions
    assert plan.preserved_underlays
    assert any("line_formula" in p.source_unit_ids for p in plan.preserved_underlays)
    assert all(_box_intersection(p.bbox, [10, 10, 50, 20]) == 0 for p in plan.patches)
    assert any(f.get("type") == "ownership_lifecycle_v2_1_render_contract_enforced" for f in findings)
