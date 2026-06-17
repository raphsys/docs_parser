import json
from pathlib import Path

from pubready.page_auditor import evaluate_page


ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = ROOT / "results/show3_pubready/pagereconstruct_plan_test_docintelligence_p0284.json"


def _load_plan():
    with PLAN_PATH.open() as f:
        return json.load(f)


def test_p0284_golden_forbids_source_image_background():
    plan = _load_plan()
    source = ((plan.get("final_contract") or {}).get("background") or {}).get("source_image_path")
    background_paths = [
        op.get("path")
        for op in plan.get("render_ops") or []
        if op.get("op_type") == "background"
    ]

    assert source
    assert background_paths
    assert source not in background_paths


def test_p0284_golden_visual_leak_cannot_be_marked_ok():
    plan = _load_plan()
    plan["visual_image_audit"] = {
        "image_qa_executed": True,
        "old_text_visible": True,
        "double_text_rendering": True,
        "score": 0.35,
    }

    report = evaluate_page(plan, {"page": {"page_index": 284}, "units": [], "translated_units": []}, mode="publication")

    assert report.status == "ko"
    assert not report.publication_ready
    assert "source_text_leak" in report.hard_blockers
    assert "double_text_rendering" in report.hard_blockers
    assert report.publication_ready_score <= 0.50


def test_no_business_logic_is_specific_to_p0284():
    code = "\n".join(
        path.read_text(errors="ignore")
        for base in ("pagereconstruct", "pubready", "pagetranslate")
        for path in (ROOT / base).rglob("*.py")
    )

    assert "test_docintelligence_p0284" not in code
    assert "p0284" not in code
