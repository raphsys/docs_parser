import sys

from tests.translation_engines._ct2_fakes import write_inventory
from translation_engines.ct2_engine import CTranslate2Engine


def test_missing_ctranslate2_reports_clean_error(tmp_path, monkeypatch):
    inv = write_inventory(tmp_path, [
        {"name": "opus_mt_tc_big_en_fr", "family": "marian", "source_langs": ["en"], "target_langs": ["fr"]},
    ])
    # Simulate an environment without ctranslate2 installed.
    monkeypatch.setitem(sys.modules, "ctranslate2", None)

    engine = CTranslate2Engine(inventory_path=str(inv), model_name="opus_mt_tc_big_en_fr", source_lang="en", target_lang="fr")
    health = engine.healthcheck()

    assert health["status"] == "missing"
    assert "unavailable" in (health.get("error") or "").lower()
