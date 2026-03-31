import unittest

from page_case_classifier_v2 import PageCaseClassifierV2


class PageCaseClassifierV2Tests(unittest.TestCase):
    def setUp(self):
        self.classifier = PageCaseClassifierV2()

    def test_v2_exposes_gradual_signals_for_toc(self):
        page_data = {
            "dimensions": {"width": 420, "height": 600},
            "blocks": [
                {"role": "body", "source": "native", "text": "1.1 Intro 10"},
                {"role": "body", "source": "native", "text": "1.2 Overview 12"},
                {"role": "body", "source": "native", "text": "1.3 Details 15"},
            ],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
            "layout": {"columns": [{"x0": 0, "x1": 420}]},
        }
        lines = [{"bbox": [20, 20 + i * 20, 220, 34 + i * 20], "block": block, "line": {}} for i, block in enumerate(page_data["blocks"])]
        result = self.classifier.classify(page_data, lines, page_role="toc")
        self.assertEqual(result["version"], "page_case.v2")
        self.assertEqual(result["page_role"], "toc")
        self.assertGreaterEqual(result["reading_modes"]["toc_row_flow"], 0.95)
        self.assertTrue(any(flag["code"] == "toc_row_fragmentation" for flag in result["risk_flags"]))

    def test_v2_keeps_legacy_bridge_but_does_not_replace_legacy(self):
        page_data = {
            "dimensions": {"width": 420, "height": 600},
            "blocks": [
                {"role": "header", "source": "native", "text": "12 Deep Learning"},
                {"role": "section_heading", "source": "native", "text": "3.1 Convolution layers"},
                {"role": "body", "source": "native", "text": "This section explains the main idea of the model."},
                {"role": "body", "source": "native", "text": "We then compare the architecture with previous approaches."},
            ],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
            "layout": {"columns": [{"x0": 0, "x1": 200}, {"x0": 220, "x1": 420}]},
        }
        lines = [{"bbox": [0, 0, 100, 10], "block": b, "line": {}} for b in page_data["blocks"]]
        result = self.classifier.classify(page_data, lines, page_role="body")
        self.assertIn("legacy_bridge", result)
        self.assertEqual(result["legacy_bridge"]["page_family"], "body_text_two_column_sectioned")
        self.assertGreater(result["reading_modes"]["columnar_flow"], result["reading_modes"]["linear_flow"])

    def test_v2_does_not_mark_toc_as_glossary_like(self):
        page_data = {
            "dimensions": {"width": 420, "height": 600},
            "blocks": [
                {"role": "section_heading", "source": "native", "text": "4.5 Improving the network and tuning hyperparameters 162"},
                {"role": "body", "source": "native", "text": "Collecting more data vs. tuning hyperparameters 162"},
                {"role": "body", "source": "native", "text": "Parameters vs. hyperparameters 163"},
                {"role": "body", "source": "native", "text": "Neural network hyperparameters 163"},
            ],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
            "layout": {"columns": [{"x0": 0, "x1": 420}]},
        }
        lines = [{"bbox": [20, 20 + i * 20, 380, 34 + i * 20], "block": block, "line": {}} for i, block in enumerate(page_data["blocks"])]
        result = self.classifier.classify(page_data, lines, page_role="toc")
        self.assertEqual(result["page_archetype_signals"]["toc"], 1.0)
        self.assertLess(result["page_archetype_signals"]["glossary_like"], 0.2)
        self.assertEqual(result["reading_modes"]["glossary_pair_flow"], 0.0)

    def test_v2_detects_chapter_opening_signal(self):
        page_data = {
            "dimensions": {"width": 600, "height": 900},
            "blocks": [
                {"role": "title", "source": "native", "text": "1"},
                {"role": "title", "source": "native", "text": "Introduction to Deep Learning"},
                {"role": "section_heading", "source": "native", "text": "1.1 Background"},
                {"role": "body", "source": "native", "text": "Deep learning has transformed computer vision and speech processing."},
                {"role": "body", "source": "native", "text": "This chapter introduces the main concepts and training principles."},
            ],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
            "layout": {"columns": [{"x0": 0, "x1": 280}, {"x0": 320, "x1": 600}]},
        }
        lines = [{"bbox": [20, 40 + i * 30, 560, 60 + i * 30], "block": block, "line": {}} for i, block in enumerate(page_data["blocks"])]
        result = self.classifier.classify(page_data, lines, page_role="body")
        self.assertGreaterEqual(result["page_archetype_signals"]["chapter_opening"], 0.68)

    def test_v2_does_not_mark_table_diagram_page_as_chapter_opening(self):
        page_data = {
            "dimensions": {"width": 600, "height": 900},
            "blocks": [
                {"role": "title", "source": "native", "text": "Transfer learning"},
                {"role": "title", "source": "native", "text": "new_model.summary()"},
                {"role": "body", "source": "native", "text": "input_1 Output Shape Param #"},
                {"role": "equation_inline", "source": "native", "text": "Total params: 14,714,688"},
            ],
            "images": [{"bbox": [300, 140, 560, 420]}],
            "drawings": [],
            "non_text_zones": [],
            "layout": {"columns": [{"x0": 0, "x1": 280}, {"x0": 320, "x1": 600}]},
        }
        lines = [{"bbox": [20, 40 + i * 30, 560, 60 + i * 30], "block": block, "line": {}} for i, block in enumerate(page_data["blocks"])]
        result = self.classifier.classify(page_data, lines, page_role="body")
        self.assertLess(result["page_archetype_signals"]["chapter_opening"], 0.2)


if __name__ == "__main__":
    unittest.main()
