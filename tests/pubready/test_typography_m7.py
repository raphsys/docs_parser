"""M7 — TypographyPlanner + hiérarchie typo."""
from pagereconstruct import compile_page_render_plan
from pagereconstruct.input_adapter import PageReconstructInputAdapter
from pagereconstruct.final_contract import FinalReconstructionContract
from pagereconstruct.composition.typography_planner import plan_typography
from pubready.stages import typography_audit
from tests.pagereconstruct._fixtures import translated_input_data


def test_typography_planner_outputs_em_and_confidence():
    tid = translated_input_data()
    norm = PageReconstructInputAdapter().normalize(tid)
    plan = compile_page_render_plan(tid).to_dict()
    c = FinalReconstructionContract.from_pageprint_pagetranslate(norm, plan)
    tp = plan_typography(c)
    assert tp.block_style_plans and 0.0 <= tp.confidence <= 1.0
    assert tp.block_style_plans[0].font_size_pt_em_estimated


def test_typography_audit_flags_heading_smaller_than_body():
    plan = {"layers": {"translated_text": [
        {"id": "h", "role": "section_heading", "source_text": "Titre", "translated_text": "Titre",
         "source_unit_ids": [], "style": {"font_size_pt": 8, "color": "#000", "flags": {}}},
        {"id": "p", "role": "body_paragraph", "source_text": "x", "translated_text": "x",
         "source_unit_ids": [], "style": {"font_size_pt": 11, "color": "#000", "flags": {}}}]}}
    r = typography_audit.audit_page(plan, {"units": []})
    assert any(f.type == "heading_rendered_as_body" for f in r.findings)
