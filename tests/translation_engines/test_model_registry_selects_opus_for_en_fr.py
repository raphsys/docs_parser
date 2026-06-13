import json

from translation_engines.model_registry import TranslationModelRegistry


def _bucket_inventory(tmp_path, *, opus_available=True):
    opus_dir = tmp_path / "opus_ct2"
    opus_tok = tmp_path / "opus_tok"
    m2m_dir = tmp_path / "m2m_ct2"
    m2m_tok = tmp_path / "m2m_tok"
    for path in (m2m_dir, m2m_tok):
        path.mkdir(parents=True, exist_ok=True)
    if opus_available:
        for path in (opus_dir, opus_tok):
            path.mkdir(parents=True, exist_ok=True)
    payload = {
        "primary": [
            {"name": "m2m100_418m", "model_dir": str(m2m_dir), "tokenizer_dir": str(m2m_tok), "family": "m2m100"},
        ],
        "fallback": [
            {"name": "m2m100_418m", "model_dir": str(m2m_dir), "tokenizer_dir": str(m2m_tok), "family": "m2m100"},
        ],
        "enfr": [
            {"name": "opus_mt_tc_big_en_fr", "model_dir": str(opus_dir), "tokenizer_dir": str(opus_tok), "family": "marian"},
        ],
    }
    inv = tmp_path / "model_inventory.json"
    inv.write_text(json.dumps(payload), encoding="utf-8")
    return inv


def test_select_opus_for_en_fr(tmp_path):
    inv = _bucket_inventory(tmp_path, opus_available=True)
    registry = TranslationModelRegistry(inventory_path=str(inv))
    selected = registry.select_model("en", "fr")
    assert selected is not None
    assert selected.name == "opus_mt_tc_big_en_fr"
