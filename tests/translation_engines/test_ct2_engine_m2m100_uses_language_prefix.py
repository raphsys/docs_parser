from tests.translation_engines._ct2_fakes import FakeTokenizer, FakeTranslator, install_fakes, write_inventory
from translation_engines.ct2_engine import CTranslate2Engine


def test_m2m100_uses_target_language_prefix(tmp_path, monkeypatch):
    inv = write_inventory(tmp_path, [
        {"name": "m2m100_418m", "family": "m2m100", "source_langs": [], "target_langs": []},
    ])
    tokenizer = FakeTokenizer(family="m2m100")
    # m2m100 output starts with the forced language token, dropped before decode.
    translator = FakeTranslator(hypothesis_builder=lambda i, toks, prefix: ["__fr__", "▁Couches", "▁cachées"])
    install_fakes(monkeypatch, tokenizer, translator)

    engine = CTranslate2Engine(inventory_path=str(inv), model_name="m2m100_418m", source_lang="en", target_lang="fr")
    out = engine.translate_batch([{"text": "Hidden layers", "source_lang": "en", "target_lang": "fr"}])

    assert translator.calls[0]["target_prefix"] == [["__fr__"]]
    assert out[0]["translated_text"] == "Couches cachées"
