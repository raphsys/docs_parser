import tools.run_translation_trial as trial_mod
from tests.functional.test_batch_audit_no_fallback import _page


class RaisingEngine:
    profile = "raising"
    supports_batch = True
    batch_size = 8

    def translate_batch(self, requests):
        raise RuntimeError("engine exploded")


def test_engine_failure_sets_runtime_ko_not_pipeline(monkeypatch):
    monkeypatch.setattr(trial_mod, "create_translation_engine", lambda *a, **k: RaisingEngine())
    result = trial_mod.run_translation_trial([_page()], engine_name="raising", target_lang="fr")

    # Pipeline / functional stay OK (selection worked); only the engine is KO.
    assert result["pipeline_status"] == "ok"
    assert result["translation_runtime_status"] == "ko"


def test_fail_on_runtime_exit_code(monkeypatch):
    monkeypatch.setattr(trial_mod, "create_translation_engine", lambda *a, **k: RaisingEngine())
    result = trial_mod.run_translation_trial([_page()], engine_name="raising", target_lang="fr")
    assert trial_mod._trial_passes(result, "runtime") is False
    assert trial_mod._trial_passes(result, "functional") is True
