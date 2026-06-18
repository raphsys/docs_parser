from pagereconstruct.render_ops import TextOp, _resolve_textop_vertical_collisions


def _op(unit_id, y0, y1):
    return TextOp(
        lines=[{"text": unit_id, "x": 10, "y_top": y0, "x1": 100, "y_bottom": y1}],
        size_pt=9,
        font_path="font.ttf",
        color=[0, 0, 0],
        unit_id=unit_id,
    )


def test_collision_shift_is_cumulative_against_original_position():
    first = _op("first", 0, 10)
    second = _op("second", 9, 19)
    third = _op("third", 10, 20)

    _resolve_textop_vertical_collisions([first, second, third], [], 100, [])

    assert second.lines[0]["y_top"] >= first.lines[0]["y_bottom"] + 1.25
    assert third.lines[0]["y_top"] >= second.lines[0]["y_bottom"] + 1.25
