"""M1 — pubready: évaluateur granulaire + gates + document auditor."""

from pagereconstruct import compile_page_render_plan
from pagereconstruct.input_adapter import PageReconstructInputAdapter
from tests.pagereconstruct._fixtures import translated_input_data
import pubready
from pubready.gates import page_decision, document_decision
from pubready.schema import StageAuditResult


def _eval():
    tid = translated_input_data()
    norm = PageReconstructInputAdapter().normalize(tid)
    plan = compile_page_render_plan(tid).to_dict()
    return pubready.evaluate_page(plan, norm, page_id="fix", mode="publication")


def test_page_report_has_explainable_stage_scores():
    rep = _eval()
    for k in ("pagetranslate", "typography", "block_layout", "background", "render_ops", "visual_image"):
        assert k in rep.stage_scores


def test_typography_is_granular_block_and_dimensions():
    rep = _eval()
    typo = [s for s in rep.stages if s.stage_name == "typography"][0]
    assert typo.elements, "audit typo par bloc"
    b = typo.elements[0]
    names = {d.name for d in b.dimensions}
    assert {"font_class", "font_size", "color", "bold", "italic", "alignment"} <= names


def test_source_leak_blocks_page():
    rep = _eval()  # fixture = source background -> leak high
    assert rep.publication_ready is False
    assert "source_text_leak" in rep.hard_blockers or rep.status == "ko"


def test_hard_blocker_forces_ko():
    sr = StageAuditResult(stage_name="background", score=0.5, hard_blockers=["missing_clean_background"], status="ko")
    status, ready, blockers = page_decision([sr], mode="debug")
    assert status == "ko" and ready is False and "missing_clean_background" in blockers


def test_one_ko_page_blocks_document():
    good = pubready.schema.PagePublicationReadyReport(page_id="a", status="ok", publication_ready=True, publication_ready_score=0.97)
    bad = pubready.schema.PagePublicationReadyReport(page_id="b", status="ko", publication_ready=False, publication_ready_score=0.6, hard_blockers=["source_text_leak"])
    status, ready, blocking = document_decision([good, bad])
    assert status == "ko" and ready is False and "b" in blocking


def test_document_ok_when_all_pages_ready():
    pages = [pubready.schema.PagePublicationReadyReport(page_id=f"p{i}", status="ok", publication_ready=True, publication_ready_score=0.97) for i in range(3)]
    doc = pubready.evaluate_document(pages, document_id="d")
    assert doc.publication_ready is True and doc.publication_ready_score >= 0.95
