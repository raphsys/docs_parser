import os
import tempfile
import unittest

from layout_ai_enricher import LayoutAIEnricher


class LayoutAIEnricherTests(unittest.TestCase):
    def test_disabled_enricher_is_noop(self):
        old = os.environ.get("LAYOUT_AI_ENABLE")
        try:
            os.environ["LAYOUT_AI_ENABLE"] = "0"
            enricher = LayoutAIEnricher()
            page = {"blocks": [], "regions": []}
            out, info = enricher.enrich(page, pil_img=None)
            self.assertEqual(out, page)
            self.assertFalse(info["applied"])
            self.assertFalse(info["ready"])
        finally:
            if old is None:
                os.environ.pop("LAYOUT_AI_ENABLE", None)
            else:
                os.environ["LAYOUT_AI_ENABLE"] = old

    def test_extract_regions_from_simulated_predictions(self):
        enricher = LayoutAIEnricher()
        preds = [
            {"label": "table", "bbox": [10, 20, 110, 220], "score": 0.91},
            {"type": "figure", "box": [130, 40, 230, 180], "confidence": 0.77},
        ]
        regions = enricher._extract_regions(preds)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0]["type"], "table")
        self.assertEqual(regions[0]["bbox"], [10, 20, 110, 220])
        self.assertEqual(regions[1]["type"], "figure")
        self.assertEqual(regions[1]["bbox"], [130, 40, 230, 180])

    def test_auto_enable_when_minimal_local_models_are_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model_dir in LayoutAIEnricher.MINIMAL_MODELS.values():
                os.makedirs(os.path.join(tmp, model_dir), exist_ok=True)
            old_root = os.environ.get("LAYOUT_AI_MODELS_ROOT")
            old_enable = os.environ.get("LAYOUT_AI_ENABLE")
            old_profile = os.environ.get("LAYOUT_AI_PROFILE")
            try:
                os.environ["LAYOUT_AI_MODELS_ROOT"] = tmp
                os.environ.pop("LAYOUT_AI_ENABLE", None)
                os.environ.pop("LAYOUT_AI_PROFILE", None)
                enricher = LayoutAIEnricher()
                self.assertTrue(enricher.enabled)
                self.assertEqual(enricher.profile, "minimal")
            finally:
                if old_root is None:
                    os.environ.pop("LAYOUT_AI_MODELS_ROOT", None)
                else:
                    os.environ["LAYOUT_AI_MODELS_ROOT"] = old_root
                if old_enable is None:
                    os.environ.pop("LAYOUT_AI_ENABLE", None)
                else:
                    os.environ["LAYOUT_AI_ENABLE"] = old_enable
                if old_profile is None:
                    os.environ.pop("LAYOUT_AI_PROFILE", None)
                else:
                    os.environ["LAYOUT_AI_PROFILE"] = old_profile

    def test_auto_profile_upgrades_to_advanced_when_full_bundle_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model_dir in {**LayoutAIEnricher.MINIMAL_MODELS, **LayoutAIEnricher.ADVANCED_MODELS}.values():
                os.makedirs(os.path.join(tmp, model_dir), exist_ok=True)
            old_root = os.environ.get("LAYOUT_AI_MODELS_ROOT")
            old_enable = os.environ.get("LAYOUT_AI_ENABLE")
            old_profile = os.environ.get("LAYOUT_AI_PROFILE")
            try:
                os.environ["LAYOUT_AI_MODELS_ROOT"] = tmp
                os.environ.pop("LAYOUT_AI_ENABLE", None)
                os.environ.pop("LAYOUT_AI_PROFILE", None)
                enricher = LayoutAIEnricher()
                self.assertTrue(enricher.enabled)
                self.assertEqual(enricher.profile, "advanced")
                self.assertTrue(enricher.feature_flags["table_recognition"])
                self.assertTrue(enricher.feature_flags["formula_recognition"])
                self.assertTrue(enricher.feature_flags["seal_recognition"])
            finally:
                if old_root is None:
                    os.environ.pop("LAYOUT_AI_MODELS_ROOT", None)
                else:
                    os.environ["LAYOUT_AI_MODELS_ROOT"] = old_root
                if old_enable is None:
                    os.environ.pop("LAYOUT_AI_ENABLE", None)
                else:
                    os.environ["LAYOUT_AI_ENABLE"] = old_enable
                if old_profile is None:
                    os.environ.pop("LAYOUT_AI_PROFILE", None)
                else:
                    os.environ["LAYOUT_AI_PROFILE"] = old_profile


if __name__ == "__main__":
    unittest.main()
