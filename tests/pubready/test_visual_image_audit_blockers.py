from pubready.gates import page_decision
from pubready.schema import StageAuditResult
from pubready.stages.visual_image_audit import VisualImageAuditResult, result_to_stage


def test_visual_audit_required_in_publication():
    status, ready, blockers = page_decision([StageAuditResult(stage_name="render_ops", score=1.0)], mode="publication")

    assert status == "ko"
    assert not ready
    assert "visual_image_qa_missing" in blockers


def test_visual_audit_detects_old_text_visible():
    stage = result_to_stage(VisualImageAuditResult(image_qa_executed=True, old_text_visible=True, score=0.4))

    assert stage.status == "ko"
    assert "source_text_leak" in stage.hard_blockers
    assert stage.score <= 0.5


def test_visual_audit_detects_double_text_rendering():
    stage = result_to_stage(VisualImageAuditResult(image_qa_executed=True, double_text_rendering=True, score=0.45))

    assert stage.status == "ko"
    assert "double_text_rendering" in stage.hard_blockers
    assert stage.score <= 0.5
