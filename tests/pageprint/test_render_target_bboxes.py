from pageprint.view_compiler import _render_target_for_segment


def _unit_map():
    return {
        "ln1": {"unit_id": "ln1", "geometry": {"bbox": [50, 50, 400, 60]}, "visual": {"style": {"font_size_pt": 11}}},
        "ln2": {"unit_id": "ln2", "geometry": {"bbox": [50, 60, 400, 70]}},
    }


def test_segment_layout_is_logical_block_not_first_line():
    entry = {"bbox": [50, 50, 400, 70]}
    rt = _render_target_for_segment(entry, _unit_map(), ["ln1", "ln2"], 1, "body_paragraph")
    assert rt["layout_bbox"] == [50, 50, 400, 70]          # full block
    assert rt["layout_bbox"] != [50, 50, 400, 60]          # not the first line
    assert rt["patch_bbox"] == [50, 50, 400, 70]           # union of source lines
    assert rt["anchor_bbox"] == [50, 50, 400, 60]          # first line is the anchor
    assert rt["style_source_unit_id"] == "ln1"             # the line that carries style


def test_layout_falls_back_to_coverage_without_entry_bbox():
    rt = _render_target_for_segment({}, _unit_map(), ["ln1", "ln2"], 2, "body_paragraph")
    assert rt["layout_bbox"] == [50, 50, 400, 70]
