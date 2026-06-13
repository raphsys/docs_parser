from tests.pagetranslate.test_translation_plan_mode import _input_data
from pagetranslate import build_page_translation


class BadNumberEngine:
    def translate(self, text, source_lang, target_lang, context):
        return "Bonjour MLP 43."


def test_translation_quality_numbers():
    data = _input_data()
    data["views"]["translation_plan"][0]["source_text"] = "Hello MLP 42"
    result = build_page_translation(data, translator=BadNumberEngine(), target_lang="fr")
    assert result["translation_quality"]["number_mismatch_count"] == 1
