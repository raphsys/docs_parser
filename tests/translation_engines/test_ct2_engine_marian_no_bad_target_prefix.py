from tests.translation_engines._ct2_fakes import FakeTokenizer, FakeTranslator, install_fakes, write_inventory
from translation_engines.ct2_engine import CTranslate2Engine


def test_marian_does_not_send_language_target_prefix(tmp_path, monkeypatch):
    inv = write_inventory(tmp_path, [
        {"name": "opus_mt_tc_big_en_fr", "family": "marian", "source_langs": ["en"], "target_langs": ["fr"]},
    ])
    tokenizer = FakeTokenizer(family="marian")
    translator = FakeTranslator()
    install_fakes(monkeypatch, tokenizer, translator)

    engine = CTranslate2Engine(inventory_path=str(inv), model_name="opus_mt_tc_big_en_fr", source_lang="en", target_lang="fr")
    engine.translate_batch([{"text": "Hidden layers", "source_lang": "en", "target_lang": "fr"}])

    # Marian/OPUS bilingual models must not get a bare "fr" target prefix.
    assert translator.calls[0]["target_prefix"] is None
