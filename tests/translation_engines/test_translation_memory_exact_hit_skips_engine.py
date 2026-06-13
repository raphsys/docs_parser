from pagetranslate import build_page_translation
from tests.pagetranslate.test_batch_translation_engine import _input_data
from translation_engines.translation_memory import TranslationMemory


class CountingEngine:
    profile = "counting"
    supports_batch = True

    def __init__(self):
        self.calls = []

    def translate_batch(self, requests):
        self.calls.append(requests)
        return [{"translated_text": f"FR::{r['text']}", "raw_output": r["text"], "metadata": {}} for r in requests]


def test_exact_memory_hit_skips_engine():
    memory = TranslationMemory()
    memory.add({
        "source": "Hello MLP world.",
        "target": "Bonjour, monde MLP.",
        "source_lang": "en",
        "target_lang": "fr",
        "validated": True,
    })
    engine = CountingEngine()
    result = build_page_translation(
        _input_data(),
        translator=engine,
        translation_memory=memory,
        target_lang="fr",
        source_lang="en",
    )

    unit = result["translation_units"][0]
    assert unit["translated_text"] == "Bonjour, monde MLP."
    assert unit["engine_trace"]["memory_hit"] is True
    assert unit["engine_trace"]["memory_source"] == "exact"
    # The model was never called.
    assert engine.calls == []
    assert result["runtime_validation"]["memory_hit_count"] == 1
    assert result["runtime_validation"]["model_call_count"] == 0
