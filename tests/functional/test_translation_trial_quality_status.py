from tests.functional.test_batch_audit_no_fallback import _page
from tools.run_translation_trial import run_translation_trial


def test_translation_trial_quality_status():
    result = run_translation_trial([_page()], engine_name="rule", target_lang="fr")
    assert result["pipeline_status"] == "ok"
    assert result["translation_runtime_status"] == "ok"
    assert result["linguistic_quality_status"] in {"ok", "review"}
    assert result["publication_readiness_status"] in {"ok", "review"}


def test_needs_review_makes_quality_review():
    result = run_translation_trial([_page()], engine_name="echo", target_lang="fr")
    assert result["pipeline_status"] == "ok"
    assert result["linguistic_quality_status"] == "review"


def test_functional_ok_quality_ko_possible():
    result = run_translation_trial([_page()], engine_name="mock", target_lang="fr")
    assert result["functional_status"] == "ok"
    assert result["publication_readiness_status"] in {"ok", "review"}
