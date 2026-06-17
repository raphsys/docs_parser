"""M2/M3 — audits pageprint / contract / background + TextRemovalLedger."""

from pagereconstruct import compile_page_render_plan
from pagereconstruct.input_adapter import PageReconstructInputAdapter
from pagereconstruct.text_removal_ledger import build_ledger
from tests.pagereconstruct._fixtures import translated_input_data
from pubready.stages import pageprint_audit, contract_audit, background_audit


def _ctx():
    tid = translated_input_data()
    norm = PageReconstructInputAdapter().normalize(tid)
    plan = compile_page_render_plan(tid).to_dict()
    return plan, norm


def test_pageprint_audit_ok_on_valid_input():
    plan, norm = _ctx()
    r = pageprint_audit.audit_page(plan, norm)
    assert r.stage_name == "pageprint"
    assert "missing_units" not in r.hard_blockers
    assert r.score > 0.5


def test_pageprint_audit_blocks_missing_units():
    plan, _ = _ctx()
    r = pageprint_audit.audit_page(plan, {"units": [], "page": {}, "assets": {}})
    assert "missing_units" in r.hard_blockers and r.status == "ko"


def test_contract_audit_requires_block_contracts():
    plan, norm = _ctx()
    r = contract_audit.audit_page(plan, norm)
    assert r.stage_name == "contract"
    # le plan compilé porte un final_contract
    assert r.score > 0.0


def test_contract_audit_blocks_missing_final_contract():
    r = contract_audit.audit_page({}, {})
    assert "missing_block_contract" in r.hard_blockers and r.status == "ko"


def test_text_removal_ledger_covers_translated_blocks():
    plan, _ = _ctx()
    ledger = build_ledger(plan)
    tt = (plan.get("layers") or {}).get("translated_text") or []
    assert len(ledger) == len(tt)
    assert all(e.expected_action in {"clean_background_removed", "patch_removed", "preserve_exact", "not_translatable"} for e in ledger)


def test_background_audit_flags_source_background_in_publication():
    plan, norm = _ctx()  # fixture = source/blank background
    r = background_audit.audit_page(plan, norm, mode="publication")
    assert r.stage_name == "background"
    assert r.status in {"ko", "review"}


def test_block_planner_classifies_blocks():
    from pagereconstruct.composition.block_planner import plan_blocks
    from pagereconstruct.final_contract import FinalReconstructionContract
    plan, norm = _ctx()
    c = FinalReconstructionContract.from_pageprint_pagetranslate(norm, plan)
    bp = plan_blocks(c)
    assert bp.flow_regions and bp.placed_blocks
    assert set(bp.locked_blocks) | set(bp.movable_blocks) == set(bp.placed_blocks)


def test_block_layout_audit_detects_block_overlap():
    from pubready.stages import position_audit
    # deux blocs au MÊME bbox -> overlap critique
    plan = {"layers": {"translated_text": [
        {"id": "a", "role": "body_paragraph", "layout_bbox": [10, 10, 200, 30], "source_text": "x"},
        {"id": "b", "role": "body_paragraph", "layout_bbox": [10, 10, 200, 30], "source_text": "y"}]},
        "protected_regions": []}
    r = position_audit.audit_page(plan, {"units": []})
    assert "block_text_overlap_critical" in r.hard_blockers and r.status == "ko"
