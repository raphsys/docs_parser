import unittest

from layout_descriptor_v3 import LayoutDescriptorBuilderV3


class LayoutDescriptorV3Tests(unittest.TestCase):
    def setUp(self):
        self.builder = LayoutDescriptorBuilderV3()

    def _sample_page(self):
        return {
            "page": 1,
            "dimensions": {"width": 400, "height": 600},
            "page_role": "body",
            "page_family": "body_text_two_column_sectioned",
            "page_family_group": "body_text",
            "document_type": "book_page",
            "layout_type": "double_column",
            "style_profile": "academic_dense",
            "layout": {
                "columns": [{"id": 0, "x0": 30, "x1": 190}, {"id": 1, "x0": 210, "x1": 370}],
                "margins": {"left": 30, "right": 30, "top": 20, "bottom": 20},
            },
            "blocks": [
                {
                    "id": "title1",
                    "role": "title",
                    "source": "native",
                    "bbox": [40, 30, 360, 70],
                    "text": "Deep Learning for Vision",
                    "style": {"font": "Times", "size": 24, "flags": {"bold": True}},
                    "lines": [],
                },
                {
                    "id": "body1",
                    "role": "body",
                    "source": "native",
                    "bbox": [40, 110, 180, 160],
                    "text": "The project started as a fun experiment.",
                    "style": {"font": "Times", "size": 11},
                    "lines": [],
                },
                {
                    "id": "body2",
                    "role": "body",
                    "source": "native",
                    "bbox": [40, 162, 180, 210],
                    "text": "It later grew into a larger production pipeline.",
                    "style": {"font": "Times", "size": 11},
                    "lines": [],
                },
                {
                    "id": "cap1",
                    "role": "figure_caption",
                    "source": "native",
                    "bbox": [220, 250, 360, 285],
                    "text": "Figure 1. Example output.",
                    "style": {"font": "Times", "size": 10},
                    "lines": [],
                },
            ],
            "images": [{"id": "img1", "bbox": [220, 120, 360, 240]}],
            "non_text_zones": [],
        }

    def test_v3_exposes_observed_inferred_and_render_graphs(self):
        descriptor = self.builder.build(self._sample_page())
        self.assertEqual(descriptor["descriptor_version"], "layout_descriptor.v3")
        self.assertIn("observed_structure", descriptor)
        self.assertIn("inferred_structure", descriptor)
        self.assertIn("synthetic_structure", descriptor)
        self.assertIn("hierarchy", descriptor)
        self.assertIn("dependency_graph", descriptor)
        self.assertIn("spatial_graph", descriptor)
        self.assertIn("typographic_graph", descriptor)
        self.assertIn("primary_structure_family", descriptor)
        self.assertIn("structure_arbitration", descriptor)
        self.assertIn("render_model", descriptor)
        self.assertIn("reconstruction_contract", descriptor)

    def test_v3_captures_hierarchy_and_dependencies(self):
        descriptor = self.builder.build(self._sample_page())
        dep_types = {edge["type"] for edge in descriptor["dependency_graph"]["edges"]}
        self.assertIn("belongs_to_section", dep_types)
        self.assertIn("continues_paragraph", dep_types)
        self.assertIn("caption_for", dep_types)
        hier_edges = descriptor["hierarchy"]["edges"]
        self.assertTrue(any(edge["type"] == "contains" for edge in hier_edges))

    def test_v3_builds_spatial_and_typographic_groups(self):
        descriptor = self.builder.build(self._sample_page())
        spatial_types = {edge["type"] for edge in descriptor["spatial_graph"]["edges"]}
        self.assertIn("same_column", spatial_types)
        self.assertIn("aligned_left", spatial_types)
        self.assertTrue(descriptor["typographic_graph"]["groups"])
        self.assertTrue(descriptor["render_model"]["render_units"])
        self.assertIn("row_clusters", descriptor["spatial_graph"])
        self.assertIn("baseline_clusters", descriptor["spatial_graph"])

    def test_v3_builds_reconstruction_contract_with_execution_edges(self):
        descriptor = self.builder.build(self._sample_page())
        contract = descriptor["reconstruction_contract"]
        self.assertEqual(contract["version"], "reconstruction_contract.v1")
        self.assertTrue(contract["render_units"])
        self.assertTrue(contract["containers"])
        self.assertTrue(contract["execution_edges"])
        self.assertTrue(contract["placement_constraints"])
        edge_types = {edge["type"] for edge in contract["execution_edges"]}
        self.assertIn("belongs_to_section", edge_types)
        self.assertIn("continues_paragraph", edge_types)
        self.assertIn("caption_for", edge_types)

    def test_v3_uses_toc_rows_to_build_toc_entries(self):
        page = self._sample_page()
        page["page_role"] = "toc"
        page["toc"] = {
            "toc_rows": [
                {
                    "role": "section_heading",
                    "label": "4.5 Improving the network and tuning hyperparameters",
                    "page": "162",
                    "label_bbox": [40, 110, 280, 132],
                    "page_bbox": [300, 110, 330, 132],
                },
                {
                    "role": "subentry_marker",
                    "label": "Collecting more data vs. tuning hyperparameters",
                    "page": "162",
                    "label_bbox": [60, 145, 280, 166],
                    "page_bbox": [300, 145, 330, 166],
                },
            ]
        }
        page["blocks"] = [
            {
                "id": "toc1",
                "role": "section_heading",
                "source": "native",
                "bbox": [40, 110, 330, 132],
                "text": "4.5 Improving the network and tuning hyperparameters 162",
                "style": {"font": "Times", "size": 11, "flags": {"bold": True}},
                "lines": [],
            },
            {
                "id": "toc2",
                "role": "body",
                "source": "native",
                "bbox": [60, 145, 330, 166],
                "text": "Collecting more data vs. tuning hyperparameters 162",
                "style": {"font": "Times", "size": 10},
                "lines": [],
            },
        ]

        descriptor = self.builder.build(page)
        inferred = descriptor["inferred_structure"]
        self.assertEqual(len(inferred["toc_entries"]), 2)
        self.assertEqual(len(inferred["toc_memberships"]), 2)
        contract = descriptor["reconstruction_contract"]
        toc_containers = [c for c in contract["containers"] if c.get("kind") == "toc_entry"]
        self.assertEqual(len(toc_containers), 2)
        self.assertTrue(any(edge["type"] == "member_of_toc_entry" for edge in contract["execution_edges"]))
        self.assertEqual(descriptor["primary_structure_family"], "toc")
        self.assertEqual(contract["primary_structure_family"], "toc")
        self.assertIn("section", descriptor["structure_arbitration"]["secondary_container_kinds"])

    def test_v3_adds_line_level_paragraph_structure_for_dense_body_block(self):
        page = self._sample_page()
        page["blocks"] = [
            {
                "id": "body_dense",
                "role": "body",
                "source": "native",
                "bbox": [40, 110, 180, 260],
                "text": "Dense body paragraph",
                "style": {"font": "Times", "size": 11},
                "lines": [
                    {"bbox": [40, 110, 180, 126], "line_text": "Line one of a dense paragraph.", "phrases": []},
                    {"bbox": [40, 128, 180, 144], "line_text": "Line two of a dense paragraph.", "phrases": []},
                    {"bbox": [40, 146, 180, 162], "line_text": "Line three of a dense paragraph.", "phrases": []},
                    {"bbox": [40, 164, 180, 180], "line_text": "Line four of a dense paragraph.", "phrases": []},
                    {"bbox": [40, 182, 180, 198], "line_text": "Line five of a dense paragraph.", "phrases": []},
                ],
            }
        ]
        page["images"] = []

        descriptor = self.builder.build(page)
        dep_types = {edge["type"] for edge in descriptor["dependency_graph"]["edges"]}
        self.assertIn("continues_paragraph", dep_types)
        containers = descriptor["render_model"]["containers"]
        self.assertTrue(any(container["kind"] == "paragraph_segment" for container in containers))
        line_units = [unit for unit in descriptor["render_model"]["render_units"] if unit["kind"] == "line_flow_member"]
        self.assertGreaterEqual(len(line_units), 4)
        self.assertEqual(descriptor["primary_structure_family"], "dense_paragraph_flow")
        self.assertTrue(any(container.get("active") for container in containers if container["kind"] == "paragraph_segment"))

    def test_v3_builds_chapter_opening_container_from_classifier_signal(self):
        page = self._sample_page()
        page["page_case_v2"] = {
            "page_archetype_signals": {"chapter_opening": 0.82},
            "reading_modes": {},
            "layout_tendencies": {},
            "risk_flags": [],
        }
        page["blocks"] = [
            {
                "id": "t1",
                "role": "title",
                "source": "native",
                "bbox": [40, 30, 360, 70],
                "text": "1",
                "style": {"font": "Times", "size": 28, "flags": {"bold": True}},
                "lines": [],
            },
            {
                "id": "t2",
                "role": "title",
                "source": "native",
                "bbox": [40, 80, 360, 120],
                "text": "Introduction to Deep Learning",
                "style": {"font": "Times", "size": 22, "flags": {"bold": True}},
                "lines": [],
            },
            {
                "id": "b1",
                "role": "body",
                "source": "native",
                "bbox": [40, 160, 180, 210],
                "text": "This chapter introduces the main concepts.",
                "style": {"font": "Times", "size": 11},
                "lines": [],
            },
        ]
        descriptor = self.builder.build(page)
        dep_types = {edge["type"] for edge in descriptor["dependency_graph"]["edges"]}
        self.assertIn("member_of_chapter_opening", dep_types)
        containers = descriptor["render_model"]["containers"]
        self.assertTrue(any(container["kind"] == "chapter_opening" for container in containers))
        self.assertEqual(descriptor["primary_structure_family"], "chapter_opening")

    def test_v3_spatial_graph_stays_local_for_same_column_blocks(self):
        page = self._sample_page()
        page["blocks"] = [
            {
                "id": f"b{i}",
                "role": "body",
                "source": "native",
                "bbox": [40, 40 + i * 30, 180, 60 + i * 30],
                "text": f"Block {i}",
                "style": {"font": "Times", "size": 11},
                "lines": [],
            }
            for i in range(8)
        ]
        descriptor = self.builder.build(page)
        same_column_edges = [edge for edge in descriptor["spatial_graph"]["edges"] if edge["type"] == "same_column"]
        self.assertLessEqual(len(same_column_edges), 7)

    def test_v3_glossary_pairs_become_primary_structure(self):
        page = self._sample_page()
        page["blocks"] = [
            {
                "id": "h1",
                "role": "section_heading",
                "source": "native",
                "bbox": [40, 40, 220, 70],
                "text": "Abbreviations",
                "style": {"font": "Times", "size": 16, "flags": {"bold": True}},
                "lines": [],
            },
            {
                "id": "k1",
                "role": "body",
                "source": "native",
                "bbox": [40, 100, 110, 122],
                "text": "CNN",
                "style": {"font": "Times", "size": 11},
                "lines": [],
            },
            {
                "id": "v1",
                "role": "body",
                "source": "native",
                "bbox": [130, 100, 320, 122],
                "text": "Convolutional Neural Network",
                "style": {"font": "Times", "size": 11},
                "lines": [],
            },
        ]
        descriptor = self.builder.build(page)
        self.assertEqual(descriptor["primary_structure_family"], "glossary_pairs")
        self.assertIn("sections", descriptor["structure_arbitration"]["suppressed_inferred_collections"])

    def test_v3_preserves_phrase_span_granularity_and_executable_policies(self):
        page = self._sample_page()
        page["blocks"] = [
            {
                "id": "body_fine",
                "role": "body",
                "source": "native",
                "bbox": [40, 110, 180, 156],
                "text": "Bold term\nRegular term",
                "style": {"font": "Times", "size": 11, "color": "#111111"},
                "lines": [
                    {
                        "bbox": [40, 110, 180, 126],
                        "line_text": "Bold term",
                        "phrases": [
                            {
                                "bbox": [40, 110, 180, 126],
                                "texte": "Bold term",
                                "spans": [
                                    {
                                        "bbox": [40, 110, 86, 126],
                                        "texte": "Bold",
                                        "style": {"font": "Times-Bold", "size": 11, "color": "#111111", "flags": {"bold": True}},
                                    },
                                    {
                                        "bbox": [88, 110, 140, 126],
                                        "texte": "term",
                                        "style": {"font": "Times", "size": 11, "color": "#111111", "flags": {}},
                                    },
                                ],
                            }
                        ],
                    },
                    {
                        "bbox": [40, 130, 180, 146],
                        "line_text": "Regular term",
                        "phrases": [
                            {
                                "bbox": [40, 130, 180, 146],
                                "texte": "Regular term",
                                "spans": [
                                    {
                                        "bbox": [40, 130, 124, 146],
                                        "texte": "Regular",
                                        "style": {"font": "Times", "size": 11, "color": "#111111", "flags": {}},
                                    },
                                    {
                                        "bbox": [126, 130, 180, 146],
                                        "texte": "term",
                                        "style": {"font": "Times-Italic", "size": 11, "color": "#111111", "flags": {"italic": True}},
                                    },
                                ],
                            }
                        ],
                    },
                ],
            }
        ]
        page["images"] = []

        descriptor = self.builder.build(page)
        observed = descriptor["observed_structure"]
        phrase_nodes = [node for node in observed["elements"] if node.get("type") == "phrase"]
        self.assertEqual(len(phrase_nodes), 2)
        self.assertEqual(len(observed["spans"]), 4)

        render_unit = next(unit for unit in descriptor["render_model"]["render_units"] if unit["source_element_id"] == "body_fine")
        self.assertEqual(render_unit["source_metrics"]["line_count"], 2)
        self.assertEqual(render_unit["source_metrics"]["span_count"], 4)
        self.assertEqual(len(render_unit["descendant_phrase_ids"]), 2)
        self.assertEqual(len(render_unit["descendant_span_ids"]), 4)

        constraint = next(
            constraint
            for constraint in descriptor["reconstruction_contract"]["placement_constraints"]
            if constraint["source_element_id"] == "body_fine"
        )
        self.assertEqual(constraint["font_size_policy"]["mode"], "lock")
        self.assertEqual(constraint["linebreak_policy"]["mode"], "preserve_source_lines")
        self.assertTrue(constraint["anchor_policy"]["source_y_locked"])
        self.assertEqual(constraint["overflow_policy"]["mode"], "paginate")
        self.assertTrue(constraint["style_invariants"]["preserve_span_variation"])


if __name__ == "__main__":
    unittest.main()
