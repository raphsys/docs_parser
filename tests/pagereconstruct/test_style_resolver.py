from pagereconstruct import compile_page_render_plan
from pagereconstruct.style_resolver import DEFAULT_STYLE, resolve_style
from tests.pagereconstruct._fixtures import translated_input_data


def test_no_translated_text_unit_has_empty_style():
    plan = compile_page_render_plan(translated_input_data())
    for t in plan.translated_text:
        assert t.style and t.style.get("font_size_pt"), "style must never be empty"


def test_style_resolved_from_source_unit():
    # seg1 consumes blk1 -> descends to ln1 which carries a serif 11pt style.
    plan = compile_page_render_plan(translated_input_data())
    t = plan.translated_text[0]
    assert t.style["font_size_pt"] == 11.0
    assert t.style["flags"]["serif"] is True
    assert t.style["confidence"] > 0.2


def test_default_style_is_low_confidence():
    s = resolve_style({"source_unit_ids": []}, None, {}, {})
    assert s["style_source"] == "default"
    assert s["confidence"] <= 0.2
    assert s["font_size_pt"] == DEFAULT_STYLE["font_size_pt"]
