from pagereconstruct import compile_page_render_plan, validate
from tests.pagereconstruct._fixtures import translated_input_data


def _with_bad_translation(data):
    data["translation_result"] = {
        "linguistic_quality_status": "ko",
        "linguistic_quality_validation": {"translation_coverage_ratio": 0.39},
    }
    return data


def test_publication_mode_blocks_incomplete_translation():
    plan = compile_page_render_plan(_with_bad_translation(translated_input_data()),
                                    reconstruction_mode="publication")
    assert plan.render_policy["publication_blocked"] is True
    assert validate(plan.to_dict())["status"] == "ko"


def test_debug_mode_does_not_block():
    plan = compile_page_render_plan(_with_bad_translation(translated_input_data()),
                                    reconstruction_mode="debug")
    assert plan.render_policy["publication_blocked"] is False


def test_good_translation_not_blocked_in_publication():
    data = translated_input_data()
    data["translation_result"] = {"linguistic_quality_status": "ok",
                                  "linguistic_quality_validation": {"translation_coverage_ratio": 1.0}}
    plan = compile_page_render_plan(data, reconstruction_mode="publication")
    assert plan.render_policy["publication_blocked"] is False
