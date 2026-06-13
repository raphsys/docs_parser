import json

from translation_engines.model_registry import TranslationModelRegistry


def test_relative_paths_resolve_against_inventory_dir(tmp_path):
    # Model dirs live next to the inventory, referenced by relative paths.
    (tmp_path / "models" / "opus" / "ct2").mkdir(parents=True)
    (tmp_path / "models" / "opus" / "tok").mkdir(parents=True)
    payload = {
        "default_engine": "ct2",
        "models": [
            {
                "name": "opus_mt_tc_big_en_fr",
                "family": "marian",
                "model_dir": "models/opus/ct2",
                "tokenizer_dir": "models/opus/tok",
                "source_langs": ["en"],
                "target_langs": ["fr"],
            }
        ],
    }
    inv = tmp_path / "model_inventory.json"
    inv.write_text(json.dumps(payload), encoding="utf-8")

    registry = TranslationModelRegistry(inventory_path=str(inv))
    model = registry.list_models()[0]
    # Resolved to an absolute path under the inventory directory and available.
    assert model.available
    assert str(tmp_path) in model.model_dir
    selected = registry.select_model("en", "fr")
    assert selected is not None and selected.name == "opus_mt_tc_big_en_fr"
