import unittest

from page_extraction_postprocessors import apply_page_extraction_postprocessors


def _line_block(block_id, role, bbox, text, source="native", font="Times", size=11):
    return {
        "id": block_id,
        "role": role,
        "source": source,
        "bbox": list(bbox),
        "text": text,
        "lines": [
            {
                "bbox": list(bbox),
                "line_text": text,
                "phrases": [
                    {
                        "bbox": list(bbox),
                        "texte": text,
                        "spans": [
                            {
                                "bbox": list(bbox),
                                "texte": text,
                                "style": {
                                    "font": font,
                                    "size": size,
                                    "color": "#111111",
                                    "flags": {},
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }


class PageExtractionPostprocessorTests(unittest.TestCase):
    def test_table_postprocessor_merges_vertical_fragments(self):
        page = {
            "dimensions": {"width": 600, "height": 900},
            "layout_type": "table_dominant",
            "document_type": "form",
            "page_family": "table_diagram_example",
            "blocks": [
                _line_block("a", "body", [80, 100, 210, 120], "value is > 0"),
                _line_block("b", "body", [82, 123, 208, 144], "which means that a"),
                _line_block("c", "body", [320, 100, 430, 120], "kernel"),
            ],
        }
        out, info = apply_page_extraction_postprocessors(page)
        self.assertTrue(info["changed"])
        self.assertIn("table_dominant_merge", info["applied"])
        self.assertEqual(len(out["blocks"]), 2)
        merged_texts = [(blk.get("text") or "") for blk in out["blocks"]]
        self.assertTrue(any("value is > 0 which means that a" in txt for txt in merged_texts))
        native_table = out.get("native_structure", {}).get("table") or {}
        self.assertTrue(native_table.get("row_groups"))
        self.assertTrue(native_table.get("header_row_group_ids"))
        first_block_hints = (out["blocks"][0].get("structure_hints") or {})
        self.assertEqual(first_block_hints.get("band_role_hint"), "table_band")
        self.assertEqual((first_block_hints.get("group_ids") or {}).get("table_id"), "native_table_main")

    def test_table_postprocessor_merges_same_row_fragments(self):
        page = {
            "dimensions": {"width": 800, "height": 1000},
            "layout_type": "table_dominant",
            "document_type": "form",
            "page_family": "unknown",
            "blocks": [
                _line_block("a", "body", [80, 100, 145, 122], "Input"),
                _line_block("b", "body", [150, 100, 245, 122], "image"),
                _line_block("c", "body", [420, 100, 540, 122], "kernel"),
            ],
        }
        out, info = apply_page_extraction_postprocessors(page)
        self.assertTrue(info["changed"])
        self.assertEqual(len(out["blocks"]), 2)
        merged_texts = [(blk.get("text") or "") for blk in out["blocks"]]
        self.assertTrue(any(txt == "Input image" for txt in merged_texts))

    def test_table_native_structure_marks_stub_cells_on_body_rows(self):
        page = {
            "dimensions": {"width": 900, "height": 1200},
            "layout_type": "table_dominant",
            "document_type": "form",
            "page_family": "table_diagram_example",
            "blocks": [
                _line_block("h1", "title", [80, 100, 200, 122], "Layer name"),
                _line_block("h2", "title", [320, 100, 460, 122], "Output size"),
                _line_block("r1c1", "body", [80, 150, 160, 172], "conv1"),
                _line_block("r1c2", "body", [320, 150, 420, 172], "112x112"),
            ],
        }
        out, _ = apply_page_extraction_postprocessors(page)
        native_table = out.get("native_structure", {}).get("table") or {}
        self.assertEqual(native_table.get("stub_column_group_id"), "native_table_col_0")
        conv1 = next(blk for blk in out["blocks"] if blk.get("id") == "r1c1")
        self.assertEqual((conv1.get("structure_hints") or {}).get("structural_role_hint"), "table_stub_cell")

    def test_annotated_page_groups_multiline_labels(self):
        page = {
            "dimensions": {"width": 1000, "height": 1400},
            "layout_type": "annotated_page",
            "document_type": "manual_guide",
            "page_family": "illustrated_label_page",
            "images": [{"bbox": [300, 820, 760, 1160]}],
            "non_text_zones": [],
            "blocks": [
                _line_block("body1", "body", [150, 110, 920, 220], "Long explanatory paragraph about visual perception.", size=12),
                _line_block("l1", "title", [780, 920, 950, 946], "Eye", size=10),
                _line_block("l2", "title", [780, 950, 980, 978], "(sensing device)", size=10),
                _line_block("l3", "title", [760, 1040, 940, 1068], "Brain", size=10),
                _line_block("l4", "title", [760, 1070, 995, 1098], "(interpreting device)", size=10),
            ],
        }
        out, info = apply_page_extraction_postprocessors(page)
        self.assertTrue(info["changed"])
        self.assertIn("annotated_page_grouping", info["applied"])
        texts = [(blk.get("text") or "") for blk in out["blocks"]]
        self.assertIn("Eye (sensing device)", texts)
        self.assertIn("Brain (interpreting device)", texts)
        self.assertEqual(len(out["blocks"]), 3)
        native_annotations = out.get("native_structure", {}).get("annotations") or {}
        self.assertEqual(len(native_annotations.get("groups") or []), 2)
        eye_block = next(blk for blk in out["blocks"] if (blk.get("text") or "") == "Eye (sensing device)")
        self.assertEqual((eye_block.get("structure_hints") or {}).get("structural_role_hint"), "diagram_label")

    def test_annotated_page_does_not_merge_distinct_labels(self):
        page = {
            "dimensions": {"width": 1000, "height": 1400},
            "layout_type": "annotated_page",
            "document_type": "manual_guide",
            "page_family": "chart_label_page",
            "images": [{"bbox": [300, 820, 760, 1160]}],
            "non_text_zones": [],
            "blocks": [
                _line_block("l1", "title", [760, 920, 810, 946], "Eye", size=10),
                _line_block("l2", "title", [760, 950, 930, 978], "(sensing device)", size=10),
                _line_block("l3", "title", [760, 1000, 830, 1028], "Brain", size=10),
                _line_block("l4", "title", [760, 1032, 980, 1060], "(interpreting device)", size=10),
            ],
        }
        out, info = apply_page_extraction_postprocessors(page)
        self.assertTrue(info["changed"])
        texts = [(blk.get("text") or "") for blk in out["blocks"]]
        self.assertIn("Eye (sensing device)", texts)
        self.assertIn("Brain (interpreting device)", texts)
        self.assertEqual(len(out["blocks"]), 2)

    def test_chart_page_groups_same_row_title_fragments_but_not_ticks(self):
        page = {
            "dimensions": {"width": 1000, "height": 1400},
            "layout_type": "annotated_page",
            "document_type": "manual_guide",
            "page_family": "chart_label_page",
            "blocks": [
                _line_block("t1", "title", [260, 230, 380, 244], "Greyhound", size=10),
                _line_block("t2", "title", [392, 230, 520, 244], "Labrador", size=10),
                _line_block("n1", "title", [235, 751, 259, 766], "300", size=10),
                _line_block("n2", "title", [235, 798, 259, 813], "250", size=10),
            ],
        }
        out, info = apply_page_extraction_postprocessors(page)
        self.assertTrue(info["changed"])
        self.assertIn("chart_label_grouping", info["applied"])
        texts = [(blk.get("text") or "") for blk in out["blocks"]]
        self.assertIn("Greyhound Labrador", texts)
        self.assertIn("300", texts)
        self.assertIn("250", texts)
        self.assertEqual(len(out["blocks"]), 3)

    def test_chart_page_extracts_chart_structure(self):
        page = {
            "dimensions": {"width": 1000, "height": 1400},
            "layout_type": "annotated_page",
            "document_type": "manual_guide",
            "page_family": "chart_label_page",
            "blocks": [
                _line_block("t1", "title", [690, 762, 749, 776], "Labrador", size=10),
                _line_block("t2", "title", [690, 806, 763, 820], "Greyhound", size=10),
                _line_block("y1", "title", [235, 751, 259, 766], "300", size=10),
                _line_block("y2", "title", [235, 798, 259, 813], "250", size=10),
                _line_block("y3", "title", [235, 845, 259, 859], "200", size=10),
                _line_block("y4", "title", [235, 891, 259, 906], "150", size=10),
                _line_block("ylabel", "title", [213, 846, 228, 950], "Number of dogs", size=10),
                _line_block("xlabel", "title", [257, 1044, 633, 1079], "10 15 20 25 30 35 40 Height", size=10),
            ],
        }
        out, info = apply_page_extraction_postprocessors(page)
        self.assertIn("chart_structure", info["applied"])
        chart = out.get("chart_structure") or {}
        self.assertTrue(chart.get("chart_area_bbox"))
        self.assertTrue(chart.get("plot_area_bbox"))
        self.assertEqual(len(chart.get("y_tick_block_ids") or []), 4)
        self.assertIn("ylabel", chart.get("y_axis_label_ids") or [])
        self.assertIn("xlabel", chart.get("x_axis_label_ids") or [])
        native_chart = out.get("native_structure", {}).get("chart") or {}
        self.assertEqual((native_chart.get("y_tick_group") or {}).get("id"), "native_chart_ticks_y")
        self.assertEqual((native_chart.get("x_tick_group") or {}).get("id"), "native_chart_ticks_x")
        self.assertTrue(native_chart.get("series_groups"))
        ylabel_block = next(blk for blk in out["blocks"] if blk.get("id") == "ylabel")
        self.assertEqual((ylabel_block.get("structure_hints") or {}).get("structural_role_hint"), "chart_axis_label")

    def test_text_page_is_left_unchanged(self):
        page = {
            "dimensions": {"width": 700, "height": 1000},
            "layout_type": "double_column",
            "document_type": "book_page",
            "page_family": "body_text_two_column",
            "blocks": [
                _line_block("p1", "body", [80, 100, 310, 160], "This is a normal paragraph line that should remain alone.", size=11),
                _line_block("p2", "body", [360, 100, 620, 160], "Second column paragraph line that should also stay alone.", size=11),
            ],
        }
        out, info = apply_page_extraction_postprocessors(page)
        self.assertFalse(info["changed"])
        self.assertEqual(len(out["blocks"]), 2)

    def test_layout_ai_structure_adds_native_hints_without_merging(self):
        page = {
            "dimensions": {"width": 700, "height": 1000},
            "layout_type": "double_column",
            "document_type": "book_page",
            "page_family": "body_text_two_column_sectioned",
            "layout_ai_structure": {
                "regions": [
                    {"id": "ai_r0", "type": "title", "bbox": [70, 60, 330, 104], "source": "layout_ai_parsing"},
                    {"id": "ai_r1", "type": "text", "bbox": [70, 130, 330, 230], "source": "layout_ai_parsing"},
                ],
                "parsing_blocks": [
                    {"id": "pb0", "label": "doc_title", "bbox": [70, 60, 330, 104], "text": "Introduction"},
                    {"id": "pb1", "label": "text", "bbox": [70, 130, 330, 230], "text": "First body paragraph"},
                ],
                "table_regions": [],
                "formula_regions": [],
                "chart_regions": [],
                "seal_regions": [],
                "ocr_lines": [],
            },
            "blocks": [
                _line_block("h1", "title", [80, 64, 320, 100], "Introduction", size=13),
                _line_block("p1", "body", [82, 140, 318, 220], "First body paragraph", size=11),
            ],
        }
        out, info = apply_page_extraction_postprocessors(page)
        self.assertFalse(info["changed"])
        native_ai = out.get("native_structure", {}).get("layout_ai") or {}
        self.assertEqual(len(native_ai.get("parsing_groups") or []), 2)
        title_block = next(blk for blk in out["blocks"] if blk.get("id") == "h1")
        body_block = next(blk for blk in out["blocks"] if blk.get("id") == "p1")
        self.assertEqual((title_block.get("structure_hints") or {}).get("band_role_hint"), "title_band")
        self.assertEqual((title_block.get("structure_hints") or {}).get("structural_role_hint"), "section_title")
        self.assertEqual((body_block.get("structure_hints") or {}).get("band_role_hint"), "text_band")
        self.assertEqual((body_block.get("structure_hints") or {}).get("layout_behavior_hint"), "flow_in_band")


if __name__ == "__main__":
    unittest.main()
