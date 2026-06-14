from pagereconstruct.renderer_dispatcher import dispatch
from pagereconstruct.renderers import ParagraphRenderer


def _unit(text, bbox):
    return {"id": "ru1", "translated_text": text, "layout_bbox": bbox,
            "style": {"font_size_pt": 10.0, "flags": {"serif": True}}}


def test_measure_returns_actual_line_boxes():
    rr = ParagraphRenderer().measure(_unit("mot " * 60, [0, 0, 400, 200]), 1.0, 1.0, page_w_px=800)
    assert rr.line_count >= 1
    assert rr.actual_line_boxes
    assert rr.actual_text_bbox and len(rr.actual_text_bbox) == 4
    assert rr.font_size_used


def test_formula_measure_is_preserved_no_boxes():
    rr = dispatch("formula", None).measure(_unit("x", [0, 0, 50, 20]), 1.0, 1.0)
    assert rr.actual_line_boxes == []
    assert any(f["type"] == "formula_preserved" for f in rr.findings)


def test_overflow_flagged():
    rr = ParagraphRenderer().measure(_unit("mot " * 200, [0, 0, 100, 30]), 1.0, 1.0)
    assert rr.overflow is True
