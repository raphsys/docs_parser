from pagereconstruct import compile_page_render_plan
from pagereconstruct.background_resolver import resolve_background
from tests.pagereconstruct._fixtures import translated_input_data


def test_source_background_with_text_is_high_leak():
    bg = resolve_background({"assets": {"source_image_path": "/x.png"}, "translated_units": [{"a": 1}]})
    assert bg["mode"] == "source_background"
    assert bg["source_text_leak_risk"] == "high"
    assert bg["findings"]


def test_clean_background_low_risk():
    bg = resolve_background({"assets": {"background_path": "/clean.png", "clean_background_verified": True}, "translated_units": [{"a": 1}]})
    assert bg["mode"] == "clean_background"
    assert bg["source_text_leak_risk"] == "low"


def test_no_background_is_blank_degraded():
    bg = resolve_background({"assets": {}, "translated_units": [{"a": 1}]})
    assert bg["mode"] == "blank_degraded"


def test_plan_carries_background():
    plan = compile_page_render_plan(translated_input_data())
    assert plan.background and plan.background[0]["mode"] in {"clean_background", "source_background", "blank_degraded"}


def test_patches_never_default_white():
    plan = compile_page_render_plan(translated_input_data())
    for p in plan.patches:
        assert p.method in {"sampled_color_patch", "clean_background_patch", "inpaint_patch"}
