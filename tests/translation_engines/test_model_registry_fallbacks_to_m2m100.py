from tests.translation_engines.test_model_registry_selects_opus_for_en_fr import _bucket_inventory
from translation_engines.model_registry import TranslationModelRegistry


def test_fallback_to_m2m100_when_opus_missing(tmp_path):
    inv = _bucket_inventory(tmp_path, opus_available=False)
    registry = TranslationModelRegistry(inventory_path=str(inv))
    selected = registry.select_model("en", "fr")
    assert selected is not None
    assert selected.name == "m2m100_418m"
