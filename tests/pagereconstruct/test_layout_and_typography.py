from pagereconstruct.font_resolver_bridge import apply_font_class, infer_font_class
from pagereconstruct.font_size_sanitizer import sanitize
from pagereconstruct.layout_box_resolver import resolve_layout


# --- layout (the absolute-priority fix) ---------------------------------------
def test_paragraph_layout_repaired_from_coverage():
    first_line = [50, 50, 400, 60]          # 10pt tall
    coverage = [50, 50, 400, 260]           # full paragraph, 210pt tall
    lb, patch, anchor, findings = resolve_layout("body_paragraph", first_line, coverage)
    assert lb == coverage  # not the first line
    assert patch == coverage
    assert any(f["type"] == "layout_bbox_repaired_from_coverage" for f in findings)


def test_table_cell_layout_locked():
    cell = [10, 10, 60, 24]
    lb, patch, _a, findings = resolve_layout("table_body_cell", cell, [10, 10, 60, 200])
    assert lb == cell  # locked, not expanded
    assert not findings


# --- font class inference -----------------------------------------------------
def test_font_class_inference():
    assert infer_font_class("FrctghDrdrdhXjdpbgTimes-") == "serif"
    assert infer_font_class("JansonTextLTStd-Roman") == "serif"
    assert infer_font_class("NewBaskerville-Roman") == "serif"
    assert infer_font_class("Courier") == "mono"
    assert infer_font_class("ArialMT") == "sans"


def test_apply_font_class_overrides_wrong_serif_flag():
    style = {"font_family": "ABCDEF+TimesNewRomanPSMT", "flags": {"serif": False}}
    apply_font_class(style)
    assert style["flags"]["serif"] is True
    assert style["font_class"] == "serif"


# --- font size repair ---------------------------------------------------------
def test_absurd_font_size_repaired():
    size, findings = sanitize(4.78, [0, 0, 100, 10], "body_paragraph")
    assert size >= 7.0
    assert any("repaired" in f["type"] or "inferred" in f["type"] for f in findings)


def test_legit_font_size_untouched():
    size, findings = sanitize(11.0, [0, 0, 100, 40], "body_paragraph")
    assert size == 11.0
    assert findings == []
