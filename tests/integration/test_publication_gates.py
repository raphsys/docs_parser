"""Publication-ready gates on the observed failure shapes (directive PR-Lot 10)."""

from pagereconstruct import compile_page_render_plan, validate
from tests.pagereconstruct._fixtures import translated_input_data


def _truncated(data):
    data["translation_result"] = {"linguistic_quality_status": "ko",
                                  "linguistic_quality_validation": {"translation_coverage_ratio": 0.39}}
    return data


def test_p0006_like_truncation_blocks_publication():
    plan = compile_page_render_plan(_truncated(translated_input_data()), reconstruction_mode="publication")
    res = validate(plan.to_dict())
    assert res["status"] == "ko"
    assert res["publication_ready"] is False


def test_source_background_leak_blocks_publication_ready():
    res = validate(compile_page_render_plan(translated_input_data()).to_dict())
    assert res["publication_ready"] is False  # source_text_leak high


def test_clean_background_allows_higher_score():
    data = translated_input_data()
    data["assets"]["background_path"] = "/clean/bg.png"  # clean background available
    res = validate(compile_page_render_plan(data).to_dict())
    assert res["quality"]["source_text_leak_risk"] == "low"
    assert res["publication_ready_score"] >= 0.6
