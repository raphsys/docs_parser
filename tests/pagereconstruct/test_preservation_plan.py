from pagereconstruct import compile_page_render_plan
from tests.pagereconstruct._fixtures import translated_input_data


def test_preservation_split_overlay_vs_underlay():
    plan = compile_page_render_plan(translated_input_data())
    overlay_texts = [p.text for p in plan.preserved_overlays]
    underlay_modes = [p.preservation_mode for p in plan.preserved_underlays]
    # page reference (preserve_text_exactly) renders over the text
    assert "84" in overlay_texts
    # the formula (preserve_as_visual_overlay) is kept as original underlay
    assert "preserve_as_visual_overlay" in underlay_modes


def test_preservation_entries_all_kept():
    plan = compile_page_render_plan(translated_input_data())
    assert len(plan.preserved_overlays) + len(plan.preserved_underlays) == 2
