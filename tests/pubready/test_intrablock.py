"""M5 — composition intra-bloc + audit."""
from pagereconstruct import compile_page_render_plan
from pagereconstruct.input_adapter import PageReconstructInputAdapter
from pagereconstruct.final_contract import FinalReconstructionContract
from pagereconstruct.composition.intrablock_composer import compose_contract, compose_block
from pubready.stages import intrablock_audit
from tests.pagereconstruct._fixtures import translated_input_data


def _ctx():
    tid = translated_input_data()
    norm = PageReconstructInputAdapter().normalize(tid)
    plan = compile_page_render_plan(tid).to_dict()
    return plan, norm


def test_compose_produces_line_layouts():
    plan, norm = _ctx()
    c = FinalReconstructionContract.from_pageprint_pagetranslate(norm, plan)
    comps = compose_contract(c)
    assert comps and comps[0].lines
    assert comps[0].lines[0].runs and comps[0].lines[0].runs[0].font_path


def test_intrablock_audit_full_coverage_ok():
    plan, norm = _ctx()
    r = intrablock_audit.audit_page(plan, norm)
    assert r.stage_name == "intrablock"
    assert r.score >= 0.85


def test_intrablock_audit_detects_missing_text():
    # render_ops sans le texte attendu -> block_text_missing
    plan = {"layers": {"translated_text": [{"id": "a", "role": "body_paragraph",
            "translated_text": "alpha beta gamma delta epsilon zeta"}]},
            "render_ops": [{"op_type": "text", "unit_id": "a", "lines": [{"text": "alpha"}]}]}
    r = intrablock_audit.audit_page(plan, {"units": []})
    assert r.status == "ko" and "missing_translatable_text" in r.hard_blockers
