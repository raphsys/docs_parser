import os
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


if __name__ == "__main__":
    unittest.main()
