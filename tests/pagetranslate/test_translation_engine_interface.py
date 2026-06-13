from tests.pagetranslate.test_translation_plan_mode import _input_data
from pagetranslate import build_page_translation


class EngineOnly:
    def translate(self, text, source_lang, target_lang, context):
        return f"Bonjour {text}"


def test_translation_engine_interface():
    result = build_page_translation(_input_data(), translator=EngineOnly(), target_lang="fr")
    assert result["debug"]["selection_mode"] == "translation_plan"
    assert result["translation_units"][0]["translated_text"].startswith("Bonjour")
