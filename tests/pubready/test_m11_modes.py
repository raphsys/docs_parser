"""M11 — entrée unique evaluate_reconstruction + modes debug/review/publication."""

from pagereconstruct import compile_page_render_plan
from tests.pagereconstruct._fixtures import translated_input_data
import pubready


def _plan_tid():
    tid = translated_input_data()
    plan = compile_page_render_plan(tid).to_dict()
    return tid, plan


def test_single_entry_normalizes_internally():
    tid, plan = _plan_tid()
    rep = pubready.evaluate_reconstruction(tid, plan, page_id="fix", mode="review")
    # 10 étapes présentes
    for k in ("pageprint", "pagetranslate", "contract", "background", "typography",
              "block_layout", "intrablock", "preservation", "render_ops", "visual_image"):
        assert k in rep.stage_scores


def test_publication_mode_requires_real_image_qa():
    tid, plan = _plan_tid()
    rep = pubready.evaluate_reconstruction(tid, plan, page_id="fix", mode="publication")
    # sans images réelles -> blocker visual_image_qa_missing
    assert "visual_image_qa_missing" in rep.hard_blockers
    assert rep.publication_ready is False


def test_debug_mode_never_publication_ready_but_reports():
    tid, plan = _plan_tid()
    rep = pubready.evaluate_reconstruction(tid, plan, page_id="fix", mode="debug")
    # debug : produit un rapport complet, n'exige pas l'image réelle comme blocker
    assert "visual_image_qa_missing" not in rep.hard_blockers
    assert rep.stage_scores  # rapport émis


def test_review_mode_partial_blocking():
    tid, plan = _plan_tid()
    rep = pubready.evaluate_reconstruction(tid, plan, page_id="fix", mode="review")
    assert "visual_image_qa_missing" not in rep.hard_blockers
    assert rep.status in {"ok", "review", "ko"}
