import json

from tests.functional.test_batch_audit_no_fallback import _page
from tools.run_translation_trial import main as trial_main, run_translation_trial


def test_translation_trial_runner_mock_engine():
    result = run_translation_trial([_page()], engine_name="mock", target_lang="fr")
    assert result["functional_status"] == "ok"
    assert result["pageprint_functional_status"] == "ok"
    assert result["pagetranslate_functional_status"] == "ok"
    assert result["page_results"][0]["selection_mode"] == "translation_plan"
    assert result["page_results"][0]["fallback_selector_used"] is False
    assert result["engine_calls"][0]["translated_text"].startswith("FR::")


def test_translation_trial_refuses_bad_page():
    result = run_translation_trial([{"units": [], "views": {"translation_plan": []}}], engine_name="mock")
    assert result["functional_status"] == "ko"
    assert result["pagetranslate_functional_status"] == "not_run"
    assert "pageprint_preflight_failed" in result["errors"]


def test_translation_trial_cli_writes_output(tmp_path, monkeypatch):
    input_path = tmp_path / "page.json"
    output_path = tmp_path / "trial_result.json"
    input_path.write_text(json.dumps(_page()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_translation_trial.py",
            str(input_path),
            "--engine",
            "mock",
            "--target-lang",
            "fr",
            "--output",
            str(output_path),
        ],
    )
    assert trial_main() == 0
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["engine"] == "mock"
    assert output["unit_count"] == 1
