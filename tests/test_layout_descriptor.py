import unittest

from layout_descriptor import LayoutDescriptorBuilder
from structure_extractor import LayoutV2Builder


class LayoutDescriptorTests(unittest.TestCase):
    def setUp(self):
        self.builder = LayoutDescriptorBuilder()
        self.layout_builder = LayoutV2Builder()

    def _sample_page(self):
        return {
            "page": 1,
            "dimensions": {"width": 400, "height": 600},
            "blocks": [
                {
                    "id": "b1",
                    "role": "title",
                    "source": "native",
                    "bbox": [40, 30, 360, 70],
                    "text": "Deep Learning for Vision",
                    "lines": [
                        {
                            "bbox": [40, 30, 360, 70],
                            "line_text": "Deep Learning for Vision",
                            "phrases": [
                                {
                                    "bbox": [40, 30, 360, 70],
                                    "text": "Deep Learning for Vision",
                                    "spans": [
                                        {
                                            "bbox": [40, 30, 360, 70],
                                            "texte": "Deep Learning for Vision",
                                            "style": {"font": "Times", "size": 24, "color": "#111111", "flags": {"bold": True}},
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
                {
                    "id": "b2",
                    "role": "body",
                    "source": "native",
                    "bbox": [40, 110, 180, 220],
                    "text": "The project started as a fun experiment.",
                    "translated_text": "Le projet a commencé comme une expérience amusante.",
                    "unit_type": "narrative_body",
                    "lines": [
                        {
                            "bbox": [40, 110, 180, 145],
                            "line_text": "The project started as a fun experiment.",
                            "translated_text": "Le projet a commencé comme une expérience amusante.",
                            "phrases": [
                                {
                                    "bbox": [40, 110, 180, 145],
                                    "text": "The project started as a fun experiment.",
                                    "translated_text": "Le projet a commencé comme une expérience amusante.",
                                    "spans": [
                                        {
                                            "bbox": [40, 110, 180, 145],
                                            "texte": "The project started as a fun experiment.",
                                            "translated_text": "Le projet a commencé comme une expérience amusante.",
                                            "style": {"font": "Times", "size": 11, "color": "#222222", "flags": {}},
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
                {
                    "id": "b3",
                    "role": "figure_caption",
                    "source": "native",
                    "bbox": [220, 250, 360, 285],
                    "text": "Figure 1. Example output.",
                    "lines": [
                        {
                            "bbox": [220, 250, 360, 285],
                            "line_text": "Figure 1. Example output.",
                            "phrases": [],
                        }
                    ],
                },
            ],
            "images": [{"id": "img1", "bbox": [220, 120, 360, 240]}],
            "drawings": [],
            "non_text_zones": [],
            "layout": {
                "columns": [{"id": 0, "x0": 30, "x1": 190}, {"id": 1, "x0": 210, "x1": 370}],
            },
            "document_type": "scientific_paper",
            "layout_type": "double_column",
            "style_profile": "academic",
            "page_role": "body",
            "regions": [{"id": "fig_region", "type": "figure", "bbox": [220, 120, 360, 285]}],
            "classification_confidence": {"document_type": 0.8, "layout_type": 0.9, "style_profile": 0.7},
            "page_case": {"features": {"scientific_pattern_score": 0.8}},
        }

    def test_builds_descriptor_with_regions_elements_relations_constraints(self):
        descriptor = self.builder.build(self._sample_page())
        self.assertEqual(descriptor["descriptor_version"], "layout_descriptor.v2")
        self.assertEqual(descriptor["document_type"], "scientific_paper")
        self.assertEqual(descriptor["layout_type"], "double_column")
        self.assertTrue(descriptor["regions"])
        self.assertTrue(descriptor["elements"])
        self.assertTrue(descriptor["relations"])
        self.assertTrue(descriptor["constraints"])
        self.assertIn("reading_order", descriptor)
        self.assertIn("features", descriptor)
        self.assertIn("ai_structure", descriptor)
        self.assertIn("native_structure", descriptor)
        self.assertIn("page_organization", descriptor)
        self.assertIn("reconstruction_plan", descriptor)

        paragraph_groups = [g for g in descriptor["groups"] if g.get("type") == "paragraph"]
        self.assertEqual(len(paragraph_groups), 1)
        self.assertTrue(paragraph_groups[0]["constraints"]["allow_vertical_expand"])

        caption_rels = [r for r in descriptor["relations"] if r["type"] == "caption_of"]
        self.assertEqual(len(caption_rels), 1)

        sentence_constraints = [c for c in descriptor["constraints"] if c["type"] == "no_internal_sentence_break"]
        self.assertTrue(sentence_constraints)
        b2 = next(el for el in descriptor["elements"] if el["id"] == "b2")
        self.assertEqual(b2["band_role"], "content_band")
        self.assertEqual(b2["structural_role"], "body_paragraph")
        self.assertEqual(b2["typographic_class"], "editorial_body")
        self.assertEqual(b2["layout_behavior"], "flow")
        self.assertIn("render_sequence", descriptor["reconstruction_plan"])
        self.assertIn("bands", descriptor["page_organization"])

    def test_layout_v2_builder_attaches_descriptor_to_page(self):
        page = self._sample_page()
        enriched = self.layout_builder.build(page)
        self.assertIn("layout_descriptor", enriched)
        self.assertEqual(enriched["layout_descriptor"]["descriptor_version"], "layout_descriptor.v2")
        self.assertIn("layout_descriptor", enriched["layout"])
        self.assertEqual(enriched["layout"]["descriptor_version"], "layout_descriptor.v2")

    def test_table_dominant_page_builds_table_regions_and_locked_constraints(self):
        page = {
            "page": 1,
            "dimensions": {"width": 400, "height": 600},
            "document_type": "form",
            "layout_type": "table_dominant",
            "style_profile": "tabular_structured",
            "page_role": "body",
            "layout": {"columns": [{"id": 0, "x0": 20, "x1": 380}]},
            "regions": [],
            "classification_confidence": {},
            "page_case": {"features": {}},
            "blocks": [
                {"id": "r1c1", "role": "title", "source": "native", "bbox": [30, 100, 150, 130], "text": "Input image", "lines": []},
                {"id": "r1c2", "role": "title", "source": "native", "bbox": [170, 100, 290, 130], "text": "Edge detection", "lines": []},
                {"id": "r2c1", "role": "body", "source": "native", "bbox": [30, 150, 150, 180], "text": "kernel", "lines": []},
                {"id": "r2c2", "role": "body", "source": "native", "bbox": [170, 150, 290, 180], "text": "value is > 0", "lines": []},
            ],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
        }
        descriptor = self.builder.build(page)
        region_types = [r.get("type") for r in descriptor.get("regions", [])]
        self.assertIn("table", region_types)
        self.assertIn("table_row", region_types)
        self.assertIn("table_cell", region_types)
        constraint_types = [c.get("type") for c in descriptor.get("constraints", [])]
        self.assertIn("table_cell_locked", constraint_types)

    def test_annotated_page_builds_illustration_region_and_anchor_relations(self):
        page = {
            "page": 1,
            "dimensions": {"width": 1000, "height": 1400},
            "document_type": "manual_guide",
            "layout_type": "annotated_page",
            "style_profile": "editorial_visual",
            "page_role": "body",
            "layout": {"columns": [{"id": 0, "x0": 120, "x1": 940}]},
            "regions": [],
            "classification_confidence": {},
            "page_case": {"features": {}},
            "blocks": [
                {"id": "head1", "role": "header", "source": "native", "bbox": [140, 40, 340, 74], "text": "Computer vision", "lines": []},
                {"id": "body1", "role": "body", "source": "native", "bbox": [180, 120, 920, 260], "text": "A long explanatory paragraph about the chart and what it means for the reader.", "lines": []},
                {"id": "label1", "role": "title", "source": "native", "bbox": [340, 1080, 590, 1160], "text": "Eye (sensing device responsible for capturing images of the environment)", "lines": []},
            ],
            "images": [{"id": "img1", "bbox": [300, 860, 760, 1160]}],
            "drawings": [],
            "non_text_zones": [[300, 860, 760, 1160], [320, 900, 340, 920]],
        }
        descriptor = self.builder.build(page)
        region_types = [r.get("type") for r in descriptor.get("regions", [])]
        self.assertIn("illustration", region_types)
        self.assertIn("header_band", region_types)
        self.assertIn("text_band", region_types)
        self.assertIn("annotation_band", region_types)
        rel_types = [r.get("type") for r in descriptor.get("relations", [])]
        self.assertIn("anchored_to", rel_types)
        self.assertTrue(any(r.get("type") in {"left_of", "right_of", "above", "below"} for r in descriptor.get("relations", [])))
        header_constraints = [c for c in descriptor.get("constraints", []) if c.get("type") == "anchored_bbox"]
        self.assertTrue(header_constraints)

    def test_descriptor_builds_chart_regions_from_chart_structure(self):
        page = {
            "page": 1,
            "dimensions": {"width": 1000, "height": 1400},
            "document_type": "manual_guide",
            "layout_type": "annotated_page",
            "style_profile": "editorial_visual",
            "page_role": "body",
            "layout": {"columns": [{"id": 0, "x0": 120, "x1": 940}]},
            "regions": [],
            "classification_confidence": {},
            "page_case": {"features": {}},
            "blocks": [
                {"id": "ylabel", "role": "title", "source": "native", "bbox": [213, 846, 228, 950], "text": "Number of dogs", "lines": []},
                {"id": "xlabel", "role": "title", "source": "native", "bbox": [257, 1044, 633, 1079], "text": "10 15 20 25 30 35 40 Height", "lines": []},
                {"id": "legend1", "role": "title", "source": "native", "bbox": [690, 762, 749, 776], "text": "Labrador", "lines": []},
                {"id": "legend2", "role": "title", "source": "native", "bbox": [690, 806, 763, 820], "text": "Greyhound", "lines": []},
                {"id": "y1", "role": "title", "source": "native", "bbox": [235, 751, 259, 766], "text": "300", "lines": []},
                {"id": "y2", "role": "title", "source": "native", "bbox": [235, 798, 259, 813], "text": "250", "lines": []},
                {"id": "y3", "role": "title", "source": "native", "bbox": [235, 845, 259, 859], "text": "200", "lines": []},
                {"id": "y4", "role": "title", "source": "native", "bbox": [235, 891, 259, 906], "text": "150", "lines": []},
            ],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
            "chart_structure": {
                "chart_area_bbox": [271, 727, 775, 918],
                "plot_area_bbox": [280, 736, 766, 906],
                "y_tick_block_ids": ["y1", "y2", "y3", "y4"],
                "y_axis_label_ids": ["ylabel"],
                "x_axis_label_ids": ["xlabel"],
                "legend_label_ids": ["legend1", "legend2"],
            },
        }
        descriptor = self.builder.build(page)
        region_types = [r.get("type") for r in descriptor.get("regions", [])]
        self.assertIn("chart_area", region_types)
        self.assertIn("chart_plot_area", region_types)
        self.assertIn("chart_y_ticks", region_types)
        self.assertIn("chart_y_axis", region_types)
        self.assertIn("chart_x_axis", region_types)
        self.assertIn("chart_legend", region_types)
        ylabel = next(el for el in descriptor["elements"] if el["id"] == "ylabel")
        self.assertEqual(ylabel["typographic_class"], "chart_axis_label")

    def test_visual_text_model_marks_embedded_visual_text_and_strategy(self):
        page = {
            "page": 1,
            "dimensions": {"width": 1000, "height": 1400},
            "document_type": "manual_guide",
            "layout_type": "annotated_page",
            "style_profile": "editorial_visual",
            "page_role": "body",
            "layout": {"columns": [{"id": 0, "x0": 120, "x1": 940}]},
            "regions": [],
            "classification_confidence": {},
            "page_case": {"features": {}},
            "blocks": [
                {"id": "label1", "role": "title", "source": "native", "bbox": [340, 1080, 590, 1160], "text": "Eye (sensing device)", "lines": []},
                {"id": "body1", "role": "body", "source": "native", "bbox": [160, 120, 920, 240], "text": "Body paragraph", "lines": []},
            ],
            "images": [{"id": "img1", "bbox": [300, 860, 760, 1160]}],
            "drawings": [],
            "non_text_zones": [[300, 860, 760, 1160]],
        }
        descriptor = self.builder.build(page)
        visual_model = descriptor.get("visual_text_model") or {}
        self.assertTrue(visual_model.get("objects"))
        label_obj = next(obj for obj in visual_model["objects"] if obj["source_element_id"] == "label1")
        self.assertEqual(label_obj["text_embedding_mode"], "embedded_in_visual")
        self.assertEqual(label_obj["background_replacement_strategy"], "text_erase_then_overlay")
        self.assertTrue(label_obj["must_not_duplicate_source_text"])

    def test_descriptor_links_native_elements_to_ai_regions(self):
        page = {
            "page": 1,
            "dimensions": {"width": 500, "height": 700},
            "document_type": "book_page",
            "layout_type": "double_column",
            "style_profile": "minimalist",
            "page_role": "body",
            "layout": {"columns": [{"id": 0, "x0": 40, "x1": 240}, {"id": 1, "x0": 260, "x1": 460}]},
            "regions": [],
            "ai_layout_regions": [
                {"id": "ai_region_0", "type": "header", "bbox": [35, 20, 460, 60], "source": "layout_ai"},
                {"id": "ai_region_1", "type": "paragraph_title", "bbox": [40, 80, 220, 120], "source": "layout_ai"},
                {"id": "ai_region_2", "type": "text", "bbox": [40, 130, 230, 320], "source": "layout_ai"},
            ],
            "classification_confidence": {},
            "page_case": {"features": {}},
            "blocks": [
                {"id": "h1", "role": "header", "source": "native", "bbox": [40, 24, 200, 52], "text": "CHAPTER 9", "lines": []},
                {"id": "t1", "role": "section_heading", "source": "native", "bbox": [44, 86, 180, 114], "text": "9.2 DeepDream", "lines": []},
                {"id": "b1", "role": "body", "source": "native", "bbox": [42, 138, 228, 250], "text": "The project started as a fun experiment.", "lines": []},
            ],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
        }
        descriptor = self.builder.build(page)
        elements = {el["id"]: el for el in descriptor["elements"]}
        self.assertEqual(elements["h1"]["ai_region_id"], "ai_region_0")
        self.assertEqual(elements["t1"]["ai_region_id"], "ai_region_1")
        self.assertEqual(elements["b1"]["ai_region_id"], "ai_region_2")
        rel_types = [r["type"] for r in descriptor["relations"]]
        self.assertIn("inside_ai_region", rel_types)
        self.assertIn("title_of_region", rel_types)
        self.assertIn("heads_content", rel_types)
        constraint_types = [c["type"] for c in descriptor["constraints"] if c["element_id"] in {"h1", "t1", "b1"}]
        self.assertIn("anchored_bbox", constraint_types)
        self.assertIn("flow_in_region", constraint_types)
        self.assertGreaterEqual(descriptor["features"]["ai_region_count"], 3)
        self.assertTrue(descriptor["ai_structure"]["enabled"])
        self.assertEqual(descriptor["ai_structure"]["useful_region_count"], 3)
        complementary_roles = [r["complementary_role"] for r in descriptor["ai_structure"]["regions"]]
        self.assertIn("header_band", complementary_roles)
        self.assertIn("title_band", complementary_roles)
        self.assertIn("text_band", complementary_roles)
        t1 = elements["t1"]
        self.assertEqual(t1["band_role"], "title_band")
        self.assertEqual(t1["structural_role"], "section_title")
        self.assertEqual(t1["layout_behavior"], "anchored")
        self.assertEqual(t1["section_id"], "section_t1")
        self.assertEqual(elements["b1"]["section_id"], "section_t1")

    def test_descriptor_uses_native_structure_hints_for_groups_and_roles(self):
        page = {
            "page": 1,
            "dimensions": {"width": 900, "height": 1200},
            "document_type": "manual_guide",
            "layout_type": "annotated_page",
            "style_profile": "editorial_visual",
            "page_role": "body",
            "layout": {"columns": [{"id": 0, "x0": 40, "x1": 860}]},
            "regions": [],
            "classification_confidence": {},
            "page_case": {"features": {}},
            "blocks": [
                {
                    "id": "ann1",
                    "role": "title",
                    "source": "native",
                    "bbox": [700, 820, 850, 860],
                    "text": "Eye (sensing device)",
                    "structure_hints": {
                        "band_role_hint": "annotation_band",
                        "structural_role_hint": "diagram_label",
                        "layout_behavior_hint": "anchored",
                        "attachment_target_hint": "illustration_main",
                        "group_ids": {"annotation_group_id": "native_annotation_group_0"},
                    },
                    "lines": [],
                }
            ],
            "images": [{"id": "img1", "bbox": [260, 760, 620, 1080]}],
            "drawings": [],
            "non_text_zones": [[260, 760, 620, 1080]],
            "native_structure": {
                "annotations": {
                    "illustration_bbox": [260, 760, 620, 1080],
                    "groups": [
                        {
                            "id": "native_annotation_group_0",
                            "side": "right",
                            "bbox": [700, 820, 850, 860],
                            "block_ids": ["ann1"],
                            "attachment_target_id": "illustration_main",
                        }
                    ]
                }
            },
        }
        descriptor = self.builder.build(page)
        ann1 = next(el for el in descriptor["elements"] if el["id"] == "ann1")
        self.assertEqual(ann1["band_role"], "annotation_band")
        self.assertEqual(ann1["structural_role"], "diagram_label")
        self.assertEqual(ann1["layout_behavior"], "anchored")
        self.assertEqual(ann1["attachment_target_id"], "illustration_main")
        self.assertEqual((ann1["group_ids"] or {}).get("annotation_group_id"), "native_annotation_group_0")
        self.assertEqual(len(descriptor["page_organization"]["annotation_groups"]), 1)

    def test_descriptor_exposes_table_stub_and_chart_groups(self):
        page = {
            "page": 1,
            "dimensions": {"width": 1000, "height": 1200},
            "document_type": "form",
            "layout_type": "table_dominant",
            "style_profile": "tabular_structured",
            "page_role": "body",
            "layout": {"columns": [{"id": 0, "x0": 20, "x1": 980}]},
            "regions": [],
            "classification_confidence": {},
            "page_case": {"features": {}},
            "blocks": [
                {"id": "h1", "role": "title", "source": "native", "bbox": [40, 100, 180, 130], "text": "Input", "structure_hints": {"band_role_hint": "table_band", "structural_role_hint": "table_header_cell", "layout_behavior_hint": "locked_in_cell", "group_ids": {"table_id": "native_table_main", "table_row_group_id": "native_table_row_0", "table_column_group_id": "native_table_col_0", "cell_id": "native_table_cell_h1"}}, "lines": []},
                {"id": "s1", "role": "body", "source": "native", "bbox": [40, 160, 180, 190], "text": "Kernel", "structure_hints": {"band_role_hint": "table_band", "structural_role_hint": "table_stub_cell", "layout_behavior_hint": "locked_in_cell", "group_ids": {"table_id": "native_table_main", "table_row_group_id": "native_table_row_1", "table_column_group_id": "native_table_col_0", "cell_id": "native_table_cell_s1"}}, "lines": []},
            ],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
            "native_structure": {
                "table": {
                    "table_id": "native_table_main",
                    "bbox": [40, 100, 500, 300],
                    "row_groups": [
                        {"id": "native_table_row_0", "bbox": [40, 100, 500, 130], "block_ids": ["h1"], "cells": [], "row_role": "header"},
                        {"id": "native_table_row_1", "bbox": [40, 160, 500, 190], "block_ids": ["s1"], "cells": [], "row_role": "body"},
                    ],
                    "column_groups": [{"id": "native_table_col_0", "center_x": 110.0}],
                    "header_row_group_ids": ["native_table_row_0"],
                    "stub_column_group_id": "native_table_col_0",
                },
                "chart": {
                    "chart_id": "native_chart_main",
                    "chart_area_bbox": [200, 500, 800, 900],
                    "plot_area_bbox": [240, 540, 760, 860],
                    "y_tick_group": {"id": "native_chart_ticks_y", "block_ids": [], "bbox": [210, 540, 235, 860]},
                    "x_tick_group": {"id": "native_chart_ticks_x", "block_ids": [], "bbox": [260, 860, 740, 890]},
                    "axis_groups": [{"id": "native_chart_axis_y", "block_ids": [], "bbox": [180, 620, 205, 760]}],
                    "legend_group": {"id": "native_chart_legend_0", "block_ids": [], "bbox": [700, 540, 780, 620]},
                    "series_groups": [{"id": "native_chart_series_0", "block_ids": [], "bbox": [700, 540, 780, 620]}],
                },
            },
        }
        descriptor = self.builder.build(page)
        elements = {el["id"]: el for el in descriptor["elements"]}
        self.assertEqual(elements["s1"]["structural_role"], "table_stub_cell")
        self.assertEqual(descriptor["page_organization"]["table_stub_column_group_id"], "native_table_col_0")
        self.assertEqual((descriptor["page_organization"]["chart_groups"]["x_tick_group"] or {}).get("id"), "native_chart_ticks_x")
        self.assertTrue(descriptor["page_organization"]["chart_groups"]["series_groups"])


if __name__ == "__main__":
    unittest.main()
