from pagereconstruct import compile_page_render_plan
from tests.pagereconstruct._fixtures import translated_input_data, with_duplicate_child_unit


def test_child_already_consumed_is_not_rendered_twice():
    plan = compile_page_render_plan(with_duplicate_child_unit(translated_input_data()))
    # seg1 renders blk1 (consuming ln1/ln2); the extra ln1 unit must be skipped.
    assert len(plan.translated_text) == 1
    assert any(f["type"] == "duplicate_render_skipped" for f in plan.findings)
