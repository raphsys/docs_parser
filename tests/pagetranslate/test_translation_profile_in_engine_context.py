from pagetranslate import build_page_translation
from tests.pagetranslate.test_batch_translation_engine import _input_data


class RecordingEngine:
    profile = "recording"
    supports_batch = True

    def __init__(self):
        self.contexts = []

    def translate_batch(self, requests):
        self.contexts.extend(r.get("context") or {} for r in requests)
        return [{"translated_text": f"FR::{r['text']}", "raw_output": r["text"], "metadata": {}} for r in requests]


def test_translation_profile_reaches_engine_context_and_trace():
    engine = RecordingEngine()
    result = build_page_translation(_input_data(), translator=engine, target_lang="fr", source_lang="en")

    # The engine received a translation_profile in its request context.
    assert engine.contexts
    assert "translation_profile" in engine.contexts[0]
    assert engine.contexts[0]["translation_profile"].get("target_lang") == "fr"

    # And the unit trace carries the profile for auditing.
    trace = result["translation_units"][0]["engine_trace"]
    assert "translation_profile" in trace
    assert trace["translation_profile"].get("target_lang") == "fr"
