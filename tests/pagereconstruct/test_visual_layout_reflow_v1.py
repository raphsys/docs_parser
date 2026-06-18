from types import SimpleNamespace

from pagereconstruct.composition.text_sanitizer import sanitize_render_text
from pagereconstruct.render_ops import TextOp, _resolve_textop_vertical_collisions


def test_render_text_sanitizer_removes_internal_id_artifacts():
    text, findings = sanitize_render_text('Bonjour [n\'est pas id="abc123"] Je garde la précision. monde')
    assert 'id=' not in text
    assert 'Je garde' not in text
    assert 'Bonjour' in text and 'monde' in text
    assert findings


def test_textop_collision_guard_pushes_second_line_down():
    a = TextOp(lines=[{"text": "A", "x": 10, "y_top": 10, "x1": 120, "y_bottom": 22}], size_pt=10, font_path="/tmp/f.ttf", color=[0,0,0], unit_id="a")
    b = TextOp(lines=[{"text": "B", "x": 10, "y_top": 18, "x1": 120, "y_bottom": 30}], size_pt=10, font_path="/tmp/f.ttf", color=[0,0,0], unit_id="b")
    findings = []
    _resolve_textop_vertical_collisions([a, b], [], 200.0, findings)
    assert b.lines[0]["y_top"] >= a.lines[0]["y_bottom"] + 1.0
    assert any(f["type"] == "visual_layout_reflow_v1_textop_collision_shift" for f in findings)
