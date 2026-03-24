import unittest

from page_case_classifier import PageCaseClassifier
from page_family_registry import get_family_config, get_family_group


class PageCaseClassifierTests(unittest.TestCase):
    def setUp(self):
        self.classifier = PageCaseClassifier()

    def test_detects_known_table_page(self):
        page_data = {
            "dimensions": {"width": 300, "height": 400},
            "blocks": [
                {
                    "role": "body",
                    "source": "native",
                    "lines": [
                        {"line_text": "Layer | Size | Params"},
                        {"line_text": "conv1 | 32 | 3,456"},
                        {"line_text": "conv2 | 64 | 12,800"},
                        {"line_text": "dense1 | 128 | 65,536"},
                        {"line_text": "dense2 | 10 | 1,280"},
                        {"line_text": "total |  | 83,072"},
                    ],
                }
            ],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
            "layout": {"columns": [{"x0": 0, "x1": 300}]},
        }
        lines = [{"bbox": [0, 0, 100, 10], "block": page_data["blocks"][0], "line": {}} for _ in range(6)]
        result = self.classifier.classify(page_data, lines, page_role="body")
        self.assertEqual(result["page_family"], "table_page")
        self.assertTrue(result["is_known_family"])

    def test_marks_ambiguous_page_as_unknown(self):
        page_data = {
            "dimensions": {"width": 200, "height": 300},
            "blocks": [
                {"role": "body", "source": "native", "text": "Short note"},
                {"role": "title", "source": "native", "text": "System view"},
            ],
            "images": [{"bbox": [0, 0, 100, 100]}],
            "drawings": [],
            "non_text_zones": [{"bbox": [0, 0, 100, 100]}],
            "layout": {"columns": [{"x0": 0, "x1": 200}]},
        }
        lines = [{"bbox": [0, 0, 100, 10], "block": page_data["blocks"][0], "line": {}}]
        result = self.classifier.classify(page_data, lines, page_role="body")
        self.assertEqual(result["page_family"], "unknown")
        self.assertFalse(result["is_known_family"])
        self.assertEqual(result["fallback_policy"], "safe_mixed")
        self.assertTrue(result["unknown_signature"])

    def test_detects_two_column_sectioned_text_family(self):
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
        self.assertEqual(result["page_family"], "body_text_two_column_sectioned")
        self.assertEqual(result["page_family_group"], "body_text")
        self.assertEqual(result["document_type"], "book_page")
        self.assertEqual(result["layout_type"], "double_column")
        self.assertIn("style_profile", result)
        self.assertIn("confidence", result)

    def test_detects_two_column_equation_text_family(self):
        page_data = {
            "dimensions": {"width": 420, "height": 600},
            "blocks": [
                {"role": "header", "source": "native", "text": "84 Optimization"},
                {"role": "equation_inline", "source": "native", "text": "dW / dX"},
                {"role": "body", "source": "native", "text": "The derivative can be propagated through the network."},
                {"role": "body", "source": "native", "text": "This gives a compact formulation for the update rule."},
            ],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
            "layout": {"columns": [{"x0": 0, "x1": 200}, {"x0": 220, "x1": 420}]},
        }
        lines = [{"bbox": [0, 0, 100, 10], "block": b, "line": {}} for b in page_data["blocks"]]
        result = self.classifier.classify(page_data, lines, page_role="body")
        self.assertEqual(result["page_family"], "body_text_two_column_equations")
        self.assertEqual(result["page_family_group"], "body_text")

    def test_registry_resolves_existing_and_unknown_families(self):
        config = get_family_config("body_text_two_column")
        self.assertEqual(config["group"], "body_text")
        self.assertEqual(get_family_group("body_text_two_column_equations"), "body_text")
        self.assertEqual(get_family_group("future_family_not_registered"), "unknown")

    def test_detects_table_diagram_example_family(self):
        page_data = {
            "dimensions": {"width": 300, "height": 400},
            "blocks": [
                {"role": "title", "source": "native", "text": "Input image", "lines": [{"line_text": "Input image"}]},
                {"role": "title", "source": "native", "text": "Edge detection", "lines": [{"line_text": "Edge detection"}]},
                {"role": "title", "source": "native", "text": "kernel", "lines": [{"line_text": "kernel"}]},
                {"role": "equation_inline", "source": "native", "text": "0 x 120 + -1 x 140", "lines": [{"line_text": "0 x 120 + -1 x 140"}]},
                {"role": "body", "source": "native", "lines": [{"line_text": "100 110 170 225"}, {"line_text": "255 250 230"}]},
            ],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
            "layout": {"columns": [{"x0": 0, "x1": 300}]},
        }
        lines = [{"bbox": [0, 0, 100, 10], "block": b, "line": {}} for b in page_data["blocks"]]
        result = self.classifier.classify(page_data, lines, page_role="body")
        self.assertEqual(result["page_family"], "table_diagram_example")
        self.assertEqual(result["layout_type"], "table_dominant")
        self.assertEqual(result["style_profile"], "tabular_structured")

    def test_detects_mixed_dense_illustrated_family(self):
        page_data = {
            "dimensions": {"width": 420, "height": 600},
            "blocks": [
                {"role": "body", "source": "native", "text": "Main body explanation", "lines": [{"line_text": "Main body explanation"}]},
                {"role": "body", "source": "native", "text": "Another body paragraph", "lines": [{"line_text": "Another body paragraph"}]},
                {"role": "title", "source": "native", "text": "Goal weight", "lines": [{"line_text": "Goal weight"}]},
                {"role": "title", "source": "native", "text": "Current error", "lines": [{"line_text": "Current error"}]},
                {"role": "title", "source": "native", "text": "Direction", "lines": [{"line_text": "Direction"}]},
            ],
            "images": [{"bbox": [0, 0, 100, 100]}],
            "drawings": [],
            "non_text_zones": [{"bbox": [0, 0, 100, 100]}],
            "layout": {"columns": [{"x0": 0, "x1": 200}, {"x0": 220, "x1": 420}]},
        }
        lines = [{"bbox": [0, 0, 100, 10], "block": b, "line": {}} for b in page_data["blocks"]]
        result = self.classifier.classify(page_data, lines, page_role="body")
        self.assertEqual(result["page_family"], "mixed_dense_illustrated")

    def test_detects_narrative_reference_page_family(self):
        page_data = {
            "dimensions": {"width": 400, "height": 600},
            "blocks": [
                {"role": "body", "source": "native", "text": "Visit the book website at www.example.com/deep-learning"},
                {"role": "body", "source": "native", "text": "This chapter introduces the main concepts."},
            ],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
            "layout": {"columns": [{"x0": 0, "x1": 400}]},
        }
        lines = [{"bbox": [0, 0, 100, 10], "block": b, "line": {}} for b in page_data["blocks"]]
        result = self.classifier.classify(page_data, lines, page_role="body")
        self.assertEqual(result["page_family"], "narrative_reference_page")

    def test_detects_citation_heavy_body_page_family(self):
        page_data = {
            "dimensions": {"width": 400, "height": 600},
            "blocks": [
                {"role": "body", "source": "native", "text": "Alexander Mordvintsev et al., “Deepdream,” Google AI Blog, 2015."},
                {"role": "body", "source": "native", "text": "The following section explains the result."},
            ],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
            "layout": {"columns": [{"x0": 0, "x1": 400}]},
        }
        lines = [{"bbox": [0, 0, 100, 10], "block": b, "line": {}} for b in page_data["blocks"]]
        result = self.classifier.classify(page_data, lines, page_role="body")
        self.assertEqual(result["page_family"], "citation_heavy_body_page")

    def test_detects_illustrated_label_page_from_short_label_lines_and_drawings(self):
        page_data = {
            "dimensions": {"width": 420, "height": 600},
            "blocks": [
                {
                    "role": "body",
                    "source": "native",
                    "lines": [
                        {"line_text": "Human vision system"},
                        {"line_text": "Eye"},
                        {"line_text": "Brain"},
                        {"line_text": "Interpretation"},
                        {"line_text": "Dogs"},
                        {"line_text": "grass"},
                    ],
                },
                {
                    "role": "body",
                    "source": "native",
                    "lines": [
                        {"line_text": "This section explains how the illustration maps perception to understanding."},
                    ],
                },
            ],
            "images": [],
            "drawings": [{"bbox": [0, 0, 200, 200]}],
            "non_text_zones": [],
            "layout": {"columns": [{"x0": 0, "x1": 200}, {"x0": 220, "x1": 420}]},
        }
        lines = [{"bbox": [0, 0, 100, 10], "block": page_data["blocks"][0], "line": ln} for ln in page_data["blocks"][0]["lines"]]
        result = self.classifier.classify(page_data, lines, page_role="body")
        self.assertEqual(result["page_family"], "illustrated_label_page")
        self.assertEqual(result["layout_type"], "annotated_page")
        self.assertIn("regions", result)
        self.assertGreaterEqual(result["features"]["font_size_levels"], 0)


if __name__ == "__main__":
    unittest.main()
