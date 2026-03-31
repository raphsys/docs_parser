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
        self.assertEqual(regions[1]["type"], "image")
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

    def test_extract_structural_payload_keeps_parsing_and_ocr_details(self):
        enricher = LayoutAIEnricher()
        payload = enricher._extract_structural_payload(
            {
                "layout_det_res": {
                    "boxes": [
                        {"label": "header", "bbox": [10, 10, 190, 40], "score": 0.88},
                    ]
                },
                "parsing_res_list": [
                    {"label": "doc_title", "bbox": [20, 60, 220, 120], "content": "Chapter 1", "order_index": 1, "num_of_lines": 1},
                    {"label": "text", "bbox": [20, 140, 260, 220], "content": "Body paragraph", "order_index": 2, "num_of_lines": 3},
                ],
                "table_res_list": [
                    {"bbox": [40, 260, 280, 360], "score": 0.93},
                ],
                "formula_res_list": [
                    {"bbox": [300, 260, 420, 320], "score": 0.77},
                ],
                "overall_ocr_res": {
                    "rec_texts": ["Hello world"],
                    "rec_boxes": [[25, 145, 240, 170]],
                    "rec_scores": [0.99],
                },
            }
        )
        self.assertEqual(len(payload["parsing_blocks"]), 2)
        self.assertEqual(payload["parsing_blocks"][0]["label"], "doc_title")
        self.assertEqual(len(payload["ocr_lines"]), 1)
        self.assertEqual(payload["ocr_lines"][0]["text"], "Hello world")
        region_types = {region["type"] for region in payload["regions"]}
        self.assertIn("header", region_types)
        self.assertIn("title", region_types)
        self.assertIn("text", region_types)
        self.assertIn("table", region_types)
        self.assertIn("formula", region_types)

    def test_rescale_structural_payload_restores_original_coordinates(self):
        enricher = LayoutAIEnricher()
        payload = {
            "regions": [{"bbox": [10, 20, 30, 40]}],
            "parsing_blocks": [{"bbox": [15, 25, 35, 45], "text_line_height": 12, "text_line_width": 30, "block_height": 20, "block_width": 25}],
            "ocr_lines": [{"bbox": [12, 18, 24, 30]}],
        }
        scaled = enricher._rescale_structural_payload(payload, scale_x=2.0, scale_y=3.0)
        self.assertEqual(scaled["regions"][0]["bbox"], [20, 60, 60, 120])
        self.assertEqual(scaled["parsing_blocks"][0]["bbox"], [30, 75, 70, 135])
        self.assertEqual(scaled["ocr_lines"][0]["bbox"], [24, 54, 48, 90])
        self.assertEqual(scaled["parsing_blocks"][0]["text_line_height"], 36.0)
        self.assertEqual(scaled["parsing_blocks"][0]["text_line_width"], 60.0)
        self.assertEqual(scaled["parsing_blocks"][0]["block_height"], 60.0)
        self.assertEqual(scaled["parsing_blocks"][0]["block_width"], 50.0)


if __name__ == "__main__":
    unittest.main()
