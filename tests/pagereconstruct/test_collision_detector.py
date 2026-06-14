from pagereconstruct.collision_detector import detect, detect_text_collisions
from pagereconstruct.ops import RenderResult


def _rr(uid, bbox):
    return RenderResult(unit_id=uid, renderer="paragraph", actual_text_bbox=bbox)


def test_no_overlap_is_ok():
    res = detect_text_collisions([_rr("a", [0, 0, 100, 20]), _rr("b", [0, 30, 100, 50])])
    assert res["status"] == "ok" and not res["collisions"]


def test_text_text_overlap_blocks():
    res = detect_text_collisions([_rr("a", [0, 0, 100, 40]), _rr("b", [0, 5, 100, 45])])
    assert res["status"] == "ko"  # heavy overlap
    assert any(c["type"] == "text_text_overlap" for c in res["collisions"])


def test_text_protected_overlap_detected():
    res = detect([_rr("a", [0, 0, 100, 40])], protected_px=[[10, 10, 90, 30]])
    assert res["status"] in {"review", "ko"}
    assert any(c["type"] == "text_protected_overlap" for c in res["findings"])
