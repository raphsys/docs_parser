import copy
from pathlib import Path

from PIL import Image

from pagereconstruct import compile_page_render_plan
from pagereconstruct.render_backend import render_ops_to_png
from tests.pagereconstruct._fixtures import translated_input_data


def test_missing_line_gets_identity_fallback_textop():
    data = translated_input_data()
    # seg1 consumes blk1 and its children. Add a visible sibling line that is not
    # in pagetranslate/reconstruction_units. It must not disappear.
    data["units"].append({
        "unit_id": "orphan_line", "level": "line", "children_ids": [],
        "geometry": {"bbox": [50, 160, 400, 175]},
        "understanding": {"role": "body_paragraph", "object_type": "natural_text"},
        "text": "This line was missed by pagetranslate.",
        "visual": {"style": {"font_size_pt": 10.0, "color": "#000000"}},
    })
    plan = compile_page_render_plan(data).to_dict()
    rendered = "\n".join(
        ln["text"]
        for op in plan.get("render_ops") or []
        if op.get("op_type") == "text"
        for ln in op.get("lines") or []
    )
    assert "This line was missed by pagetranslate." in rendered
    assert any(f.get("type") == "identity_fallback_text_survival" for f in plan.get("findings") or [])


def test_expansion_solver_is_not_enabled_by_default(monkeypatch):
    data = translated_input_data()
    monkeypatch.delenv("RECON_ENABLE_EXPANSION", raising=False)
    plan = compile_page_render_plan(data).to_dict()
    assert "block_expansion_applied" not in (plan.get("render_policy") or {})


def test_draw_text_exact_preservation_survives_in_png(tmp_path):
    src = tmp_path / "src.png"
    Image.new("RGB", (120, 80), "white").save(src)
    plan = {
        "page": {"width_pt": 120, "height_pt": 80, "render_width_px": 120, "render_height_px": 80},
        "render_ops": [
            {"op_type": "background", "path": str(src)},
            {"op_type": "preservation", "method": "draw_text_exact", "bbox": [10, 10, 80, 25], "text": "84"},
        ],
    }
    img = render_ops_to_png(plan, str(src))
    assert img is not None
    # At least one dark pixel should have been drawn or copied in the bbox.
    crop = img.crop((10, 10, 80, 25))
    assert any(sum(px) < 720 for px in crop.getdata())
