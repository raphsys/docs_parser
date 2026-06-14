"""Integration cases reproducing the observed failures (directive Lot 11)."""

from pagereconstruct import compile_page_render_plan, validate
from pagereconstruct.renderer_dispatcher import dispatch
from tests.pagereconstruct._fixtures import translated_input_data


def _multiline_paragraph_page():
    data = translated_input_data()
    # block spans 200pt; its render_target/layout must cover the whole block.
    data["units"][0]["geometry"]["bbox"] = [50, 50, 400, 250]
    data["views"]["reconstruction_units"][0].update({
        "bbox": [50, 50, 400, 250], "layout_bbox": [50, 50, 400, 250],
        "coverage_bbox": [50, 50, 400, 250], "patch_bbox": [50, 50, 400, 250],
        "render_target": {"bbox": [50, 50, 400, 60]},  # tempting first-line bbox
    })
    return data


def test_paragraph_not_rendered_in_first_line_box():
    plan = compile_page_render_plan(_multiline_paragraph_page())
    para = plan.translated_text[0]
    h = para.layout_bbox[3] - para.layout_bbox[1]
    assert h >= 150  # full block, not the ~10pt first line


def test_unknown_role_is_review_not_paragraph():
    assert dispatch(None, "weird_unknown_role").renderer_name == "anchored_label_review"


def test_every_text_unit_has_resolved_style():
    plan = compile_page_render_plan(translated_input_data())
    assert plan.translated_text
    assert all(t.style.get("font_size_pt") for t in plan.translated_text)


def test_validator_returns_a_status():
    result = validate(compile_page_render_plan(translated_input_data()).to_dict())
    assert result["status"] in {"ok", "review", "ko"}
    assert "quality" in result
