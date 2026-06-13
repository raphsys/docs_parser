from tests.translation_engines._ct2_fakes import FakeTokenizer, FakeTranslator, install_fakes, write_inventory
from translation_engines.ct2_engine import CTranslate2Engine


def test_ct2_engine_makes_one_real_batch_call(tmp_path, monkeypatch):
    inv = write_inventory(tmp_path, [
        {"name": "opus_mt_tc_big_en_fr", "family": "marian", "source_langs": ["en"], "target_langs": ["fr"]},
    ])
    tokenizer = FakeTokenizer(family="marian")
    translator = FakeTranslator()
    install_fakes(monkeypatch, tokenizer, translator)

    engine = CTranslate2Engine(inventory_path=str(inv), model_name="opus_mt_tc_big_en_fr", source_lang="en", target_lang="fr")
    requests = [
        {"text": "Hidden layers", "source_lang": "en", "target_lang": "fr"},
        {"text": "Activation functions", "source_lang": "en", "target_lang": "fr"},
        {"text": "Vision systems", "source_lang": "en", "target_lang": "fr"},
    ]
    out = engine.translate_batch(requests)

    assert len(out) == 3
    # One single CT2 call for the whole batch, not one per request.
    assert len(translator.calls) == 1
    assert len(translator.calls[0]["batch_tokens"]) == 3
