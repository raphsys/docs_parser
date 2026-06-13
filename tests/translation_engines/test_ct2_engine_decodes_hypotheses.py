from tests.translation_engines._ct2_fakes import FakeTokenizer, FakeTranslator, install_fakes, write_inventory
from translation_engines.ct2_engine import CTranslate2Engine


def test_ct2_engine_decodes_hypotheses(tmp_path, monkeypatch):
    inv = write_inventory(tmp_path, [
        {"name": "opus_mt_tc_big_en_fr", "family": "marian", "source_langs": ["en"], "target_langs": ["fr"]},
    ])
    tokenizer = FakeTokenizer(family="marian")
    translator = FakeTranslator(hypothesis_builder=lambda i, toks, prefix: ["▁Couches", "▁cachées"])
    install_fakes(monkeypatch, tokenizer, translator)

    engine = CTranslate2Engine(inventory_path=str(inv), model_name="opus_mt_tc_big_en_fr", source_lang="en", target_lang="fr")
    out = engine.translate_batch([{"text": "Hidden layers", "source_lang": "en", "target_lang": "fr"}])

    assert out[0]["translated_text"] == "Couches cachées"
    assert out[0]["metadata"]["output_token_count"] == 2
    assert out[0]["metadata"]["input_token_count"] >= 1
