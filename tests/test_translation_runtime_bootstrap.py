import json
import os
import tempfile
import unittest
from unittest.mock import patch

from translator import DocumentTranslator


class TranslationRuntimeBootstrapTests(unittest.TestCase):
    def _translator(self, inventory_path):
        tr = DocumentTranslator.__new__(DocumentTranslator)
        tr._model_inventory_path = inventory_path
        return tr

    def test_load_model_inventory_uses_defaults_when_file_missing(self):
        tr = self._translator("/tmp/nonexistent_translation_model_inventory.json")
        inventory = tr._load_model_inventory()
        self.assertIn("primary", inventory)
        self.assertTrue(any(entry["name"] == "m2m100_418m" for entry in inventory["primary"]))
        self.assertTrue(any(entry["name"] == "opus_mt_tc_big_en_fr" for entry in inventory["enfr"]))

    def test_resolve_primary_assets_picks_first_existing_inventory_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            present_model = os.path.join(tmp, "m2m_model")
            present_tokenizer = os.path.join(tmp, "m2m_tokenizer")
            os.makedirs(present_model, exist_ok=True)
            os.makedirs(present_tokenizer, exist_ok=True)
            inventory_path = os.path.join(tmp, "model_inventory.json")
            with open(inventory_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "primary": [
                            {
                                "name": "missing",
                                "model_dir": os.path.join(tmp, "missing_model"),
                                "tokenizer_dir": os.path.join(tmp, "missing_tokenizer"),
                                "family": "nllb",
                            },
                            {
                                "name": "present",
                                "model_dir": present_model,
                                "tokenizer_dir": present_tokenizer,
                                "family": "m2m100",
                            },
                        ]
                    },
                    f,
                )
            tr = self._translator(inventory_path)
            tr._model_inventory = tr._load_model_inventory()
            with patch.dict(
                os.environ,
                {"CT2_MODEL_DIR": "", "CT2_TOKENIZER_DIR": "", "TRANSLATOR_MODEL_FAMILY": ""},
                clear=False,
            ):
                assets = tr._resolve_ct2_assets("primary")
            self.assertEqual(assets["name"], "present")
            self.assertEqual(assets["family"], "m2m100")

    def test_resolve_primary_assets_respects_explicit_env_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "env_model")
            tokenizer_dir = os.path.join(tmp, "env_tokenizer")
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(tokenizer_dir, exist_ok=True)
            tr = self._translator("/tmp/nonexistent_translation_model_inventory.json")
            tr._model_inventory = tr._load_model_inventory()
            with patch.dict(
                os.environ,
                {
                    "CT2_MODEL_DIR": model_dir,
                    "CT2_TOKENIZER_DIR": tokenizer_dir,
                    "TRANSLATOR_MODEL_FAMILY": "m2m100",
                },
                clear=False,
            ):
                assets = tr._resolve_ct2_assets("primary")
            self.assertEqual(assets["name"], "env:primary")
            self.assertEqual(assets["model_dir"], model_dir)
            self.assertEqual(assets["tokenizer_dir"], tokenizer_dir)

    def test_resolve_primary_assets_rejects_incomplete_env_override(self):
        tr = self._translator("/tmp/nonexistent_translation_model_inventory.json")
        tr._model_inventory = tr._load_model_inventory()
        with patch.dict(
            os.environ,
            {"CT2_MODEL_DIR": "/tmp/model_only", "CT2_TOKENIZER_DIR": ""},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "doivent être définies ensemble"):
                tr._resolve_ct2_assets("primary")


if __name__ == "__main__":
    unittest.main()
