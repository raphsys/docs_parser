"""Phase 6 — RenderOps gelées + backends exécuteurs (zéro dispatch backend)."""

from pagereconstruct import compile_page_render_plan
from tests.pagereconstruct._fixtures import translated_input_data


def _plan():
    return compile_page_render_plan(translated_input_data()).to_dict()


def test_plan_freezes_render_ops():
    p = _plan()
    ops = p.get("render_ops") or []
    assert ops, "render_ops must be frozen at compile time"
    kinds = {o["op_type"] for o in ops}
    assert "background" in kinds
    assert any(o["op_type"] == "text" for o in ops)
    assert "source_text_lifecycle_ledger" in p, "source text lifecycle ledger must be frozen in the plan"


def test_text_ops_match_translated_units():
    p = _plan()
    text_ops = [o for o in p["render_ops"] if o["op_type"] == "text"]
    # one text op per rendered translated unit that produced lines
    assert len(text_ops) <= len(p["layers"]["translated_text"])
    for o in text_ops:
        assert o["lines"] and o["size_pt"] and o["font_path"]


def test_pdf_backend_executes_ops_without_dispatch():
    import importlib, tempfile, os
    pv = importlib.import_module("pagereconstruct.backends.pdf_vector")
    if not pv.is_available():
        return
    p = _plan()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "o.pdf")
        res = pv.render(p, out)
        assert res["ok"] and res.get("mode") == "ops"
        assert os.path.isfile(out)


def test_layer_order_in_ops():
    p = _plan()
    order = [o["op_type"] for o in p["render_ops"]]
    # background first, text after patches/preservation
    assert order[0] == "background"
    if "text" in order and "patch" in order:
        assert order.index("patch") < order.index("text")
