from pagereconstruct import choose_renderer, compile_page_render_plan
from tests.pagereconstruct._fixtures import translated_input_data


def test_compiles_translated_text_layer():
    plan = compile_page_render_plan(translated_input_data())
    assert len(plan.translated_text) == 1
    tu = plan.translated_text[0]
    assert tu.translated_text == "Bonjour le monde."
    assert tu.renderer == "paragraph"
    assert tu.bbox == [50, 50, 400, 90]


def test_consumed_includes_children():
    plan = compile_page_render_plan(translated_input_data())
    # seg1 consumes blk1 and its descendants ln1, ln2.
    assert "blk1" in plan.consumed_source_unit_ids
    assert "ln1" in plan.consumed_source_unit_ids
    assert "ln2" in plan.consumed_source_unit_ids


def test_excluded_from_exclusion_plan():
    plan = compile_page_render_plan(translated_input_data())
    assert "pub1" in plan.excluded_source_unit_ids


def test_protected_regions_built():
    plan = compile_page_render_plan(translated_input_data())
    assert len(plan.protected_regions) >= 2
    reasons = {r.reason for r in plan.protected_regions}
    assert any("formula" in str(x).lower() for x in reasons)


def test_page_geometry_mapped():
    plan = compile_page_render_plan(translated_input_data())
    assert plan.page["width_pt"] == 531.0 and plan.page["coordinate_origin"] == "top_left"


def test_unknown_role_not_paragraph():
    assert choose_renderer("totally_unknown_role", None) == "anchored_label_review"
    assert choose_renderer("section_heading", None) == "heading"


def test_plan_to_dict_roundtrip():
    plan = compile_page_render_plan(translated_input_data())
    d = plan.to_dict()
    assert d["schema_version"] == "pagereconstruct.plan.v1"
    assert d["layers"]["translated_text"][0]["translated_text"] == "Bonjour le monde."
