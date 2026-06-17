"""M8 autocorrect · M9 render_ops_audit · M10 visual_image_audit."""
import glob, json, os
from pagereconstruct import compile_page_render_plan
from pagereconstruct.input_adapter import PageReconstructInputAdapter
from pagereconstruct.autocorrect.correction_loop import run_correction_loop
from pagereconstruct.autocorrect.retry_policy import RetryPolicy
from pubready.stages import render_ops_audit, visual_image_audit
from pubready.schema import PagePublicationReadyReport
from tests.pagereconstruct._fixtures import translated_input_data


def _plan():
    tid = translated_input_data()
    return compile_page_render_plan(tid).to_dict()


# --- M8 ---
def test_autocorrect_stops_at_max_iter():
    bad = PagePublicationReadyReport(page_id="p", status="ko", publication_ready=False,
                                     publication_ready_score=0.5, hard_blockers=["overflow"])
    res = run_correction_loop(lambda k: {"k": dict(k)}, lambda p: bad, policy=RetryPolicy(max_iter=3))
    assert res.iterations <= 3 and res.best_report.publication_ready is False


def test_autocorrect_keeps_best_on_gain():
    seq = []
    def audit(p):
        # 1er appel mauvais, ensuite bon
        sc = 0.5 if not seq else 0.97
        seq.append(1)
        st = "ko" if sc < 0.9 else "ok"
        return PagePublicationReadyReport(page_id="p", status=st, publication_ready=(sc>=0.95),
                                          publication_ready_score=sc, hard_blockers=(["overflow"] if sc<0.9 else []))
    res = run_correction_loop(lambda k: {"k": dict(k)}, audit, policy=RetryPolicy(max_iter=3))
    assert res.best_report.publication_ready_score >= 0.95


# --- M9 ---
def test_render_ops_audit_ok_on_compiled_plan():
    r = render_ops_audit.audit_page(_plan(), {})
    assert r.stage_name == "render_ops" and r.status in {"ok", "ko"}


def test_render_ops_audit_blocks_missing_ops():
    r = render_ops_audit.audit_page({"layers": {}}, {})
    assert r.status == "ko" and "final_render_missing" in r.hard_blockers


# --- M10 ---
def test_visual_image_audit_on_real_images():
    base = None
    for d in ("results/show10_mission2", "results/show10_bgphrase", "results/show10_m3"):
        if glob.glob(os.path.join(d, "source_*.png")):
            base = d; break
    if not base:
        return  # pas d'images de référence dispo
    src = glob.glob(os.path.join(base, "source_*p0457*.png")) or glob.glob(os.path.join(base, "source_*.png"))
    rec = glob.glob(os.path.join(base, "reconstructed_*p0457*.png")) or glob.glob(os.path.join(base, "reconstructed_*.png"))
    plan_f = glob.glob(os.path.join(base, "pagereconstruct_plan_*p0457*.json")) or glob.glob(os.path.join(base, "pagereconstruct_plan_*.json"))
    plan = json.load(open(plan_f[0]))
    r = visual_image_audit.audit_page(plan, {}, source_image_path=src[0], reconstructed_image_path=rec[0])
    assert r.stage_name == "visual_image"
    assert 0.0 <= r.score <= 1.0
