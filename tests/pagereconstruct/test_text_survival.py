"""Contrôle de survie du texte PAGEPRINT -> PAGERECONSTRUCT (script minimal).

Garantit : 100% du texte original extrait par PAGEPRINT est rendu (traduit ou
préservé) ; sinon KO bloquant.
"""

import copy

from tools.audit_text_survival import audit_text_survival
from tests.pagereconstruct._fixtures import translated_input_data


def _tid_with_text():
    tid = translated_input_data()
    texts = {"blk1": "Hello world.", "ln1": "Hello", "ln2": "world."}
    for u in tid["units"]:
        if u["unit_id"] in texts:
            u["text"] = texts[u["unit_id"]]
    return tid


def test_all_text_rendered_is_100_percent():
    rep = audit_text_survival(_tid_with_text(), page_id="ok", mode="debug")
    assert rep["status"] == "ok"
    assert rep["coverage_ratio"] == 1.0
    assert rep["summary"]["ko"] == 0
    assert rep["summary"]["total_texts"] >= 3


def test_untranslated_unpreserved_text_blocks():
    tid = copy.deepcopy(_tid_with_text())
    tid["units"].append({
        "unit_id": "orphan1", "level": "block", "children_ids": [],
        "geometry": {"bbox": [50, 200, 400, 230]},
        "understanding": {"role": "body_paragraph"},
        "text": "Lonely untranslated paragraph.",
    })
    rep = audit_text_survival(tid, page_id="ko", mode="debug")
    assert rep["status"] == "ko"
    assert rep["coverage_ratio"] < 1.0
    assert "source_text_missing_pagetranslate_decision" in rep["hard_blockers"]
    ko_rows = [r for r in rep["rows"] if r["status"] != "ok"]
    assert any(r["source_unit_id"] == "orphan1" for r in ko_rows)


def _two_blocks_one_translated():
    return {
        "schema_version": "pageprint.input.v1",
        "page": {"page_index": 0, "page_role": "body",
                 "geometry": {"width": 500, "height": 700, "unit": "pt", "origin": "top_left",
                              "render_width_px": 1000, "render_height_px": 1400,
                              "scale_x_px_per_pt": 2.0, "scale_y_px_per_pt": 2.0}},
        "document": {}, "assets": {"source_image_path": ""}, "visual_layers": {},
        "units": [
            {"unit_id": "b1", "level": "block", "children_ids": [],
             "geometry": {"bbox": [50, 50, 400, 90], "reading_order_index": 0},
             "understanding": {"role": "body_paragraph", "object_type": "natural_text"},
             "content": {"text": "First paragraph translated."}, "policy": {}},
            {"unit_id": "b2", "level": "block", "children_ids": [],
             "geometry": {"bbox": [50, 120, 400, 160], "reading_order_index": 1},
             "understanding": {"role": "body_paragraph", "object_type": "natural_text"},
             "content": {"text": "Second paragraph NEVER translated."}, "policy": {}},
        ],
        "regions": [], "views": {},
    }


def test_untranslated_text_is_not_silently_rendered_raw():
    """Un texte non traduit et non préservé ne doit plus être redessiné brut.
    Il doit rester détectable par les audits comme décision manquante."""
    from pagetranslate.projection import project_translations
    from pagereconstruct import compile_page_render_plan

    ti = _two_blocks_one_translated()
    tu = [{"unit_id": "b1", "level": "block", "translation_unit_id": "t1", "translation_id": "t1",
           "status": "ok", "strategy": "mt", "source_text": "First paragraph translated.",
           "translated_text": "Premier paragraphe traduit.", "bbox": [50, 50, 400, 90],
           "source_unit_ids": ["b1"]}]
    project_translations(ti, tu)
    plan = compile_page_render_plan(ti).to_dict()
    rendered = " ".join(ln["text"] for o in plan.get("render_ops") or []
                        if o.get("op_type") == "text" for ln in o.get("lines") or [])
    assert "Premier paragraphe traduit." in rendered
    assert "Second paragraph NEVER translated." not in rendered


def test_report_has_one_row_per_source_text():
    rep = audit_text_survival(_tid_with_text(), page_id="rows", mode="debug")
    assert len(rep["rows"]) == rep["summary"]["total_texts"]
    for r in rep["rows"]:
        assert {"source_unit_id", "state", "kind", "status"} <= set(r)
