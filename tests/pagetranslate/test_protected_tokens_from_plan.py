from tests.pagetranslate.test_translation_plan_mode import _input_data
from pagetranslate import build_page_translation


class EchoEngine:
    def translate(self, text, source_lang, target_lang, context):
        return text


def test_protected_tokens_from_plan():
    result = build_page_translation(_input_data(), translator=EchoEngine(), target_lang="fr")
    unit = result["translation_units"][0]
    assert any(item["text"] == "MLP" for item in unit["protections"])
    assert result["translation_quality"]["protected_token_mismatch_count"] == 0
