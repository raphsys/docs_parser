import unittest
from unittest.mock import patch

from element_rulesets import enrich_element_rulesets
from positioning_policy import enrich_positioning_policy
from structure_extractor import LayoutV2Builder


class _FakeAIHelper:
    def __init__(self):
        self.calls = 0

    def _get_runtime(self):
        return {"fake": True}

    def score_hypotheses(self, premise, hypotheses):
        self.calls += 1
        text = str(premise or "")
        keys = tuple(hypotheses.keys())
        if keys == ("start", "end", "center"):
            if "Fragment text: Figure 1" in text:
                return {"start": 0.1, "end": 0.1, "center": 0.8}
            if "Fragment text: 42.1%" in text:
                return {"start": 0.05, "end": 0.85, "center": 0.1}
            return {"start": 0.75, "end": 0.1, "center": 0.15}
        if keys == ("top", "bottom", "middle"):
            if "Fragment text: Figure 1" in text:
                return {"top": 0.55, "bottom": 0.1, "middle": 0.35}
            if "Fragment text: 42.1%" in text:
                return {"top": 0.1, "bottom": 0.65, "middle": 0.25}
            return {"top": 0.8, "bottom": 0.05, "middle": 0.15}
        if keys == ("flow_text", "centered_title", "end_value", "attached_label"):
            if "Fragment text: Figure 1" in text:
                return {"flow_text": 0.05, "centered_title": 0.8, "end_value": 0.05, "attached_label": 0.1}
            if "Fragment text: 42.1%" in text:
                return {"flow_text": 0.05, "centered_title": 0.05, "end_value": 0.8, "attached_label": 0.1}
            return {"flow_text": 0.8, "centered_title": 0.05, "end_value": 0.05, "attached_label": 0.1}
        return {}


class PositioningPolicyTests(unittest.TestCase):
    def setUp(self):
        self.layout_builder = LayoutV2Builder()

    def test_flow_text_prefers_top_start_anchor(self):
        page = {
            "page": 1,
            "layout_direction": "ltr",
            "layout": {"columns": [{"id": 0, "x0": 30, "x1": 360}]},
            "blocks": [
                {
                    "id": "b1",
                    "role": "body",
                    "alignment": "justify",
                    "bbox": [40, 40, 300, 120],
                    "text": "This pipeline remains stable",
                    "lines": [
                        {
                            "bbox": [40, 40, 250, 62],
                            "alignment": "justify",
                            "phrases": [
                                {
                                    "id": "p1",
                                    "bbox": [40, 40, 180, 62],
                                    "text": "This pipeline",
                                    "flow_to_next_phrase": {"continuation": True, "logical_relation": "same_sentence_continuation"},
                                    "spans": [{"bbox": [40, 40, 180, 62], "texte": "This pipeline", "style": {"font": "Times", "size": 11}}],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        with patch("positioning_policy.get_element_relations_ai_enricher", return_value=_FakeAIHelper()):
            enrich_positioning_policy(page)
        policy = page["blocks"][0]["lines"][0]["phrases"][0]["positioning_policy"]
        self.assertEqual(policy["anchors"]["horizontal"]["primary"], "start")
        self.assertEqual(policy["anchors"]["vertical"]["primary"], "top")
        self.assertEqual(policy["primary_position_reference"]["mode"], "top_start")
        self.assertEqual(policy["expansion_policy"]["horizontal"], "grow_to_end")

    def test_centered_title_prefers_center_anchor(self):
        page = {
            "page": 1,
            "layout_direction": "ltr",
            "layout": {"columns": [{"id": 0, "x0": 20, "x1": 380}]},
            "blocks": [
                {
                    "id": "b2",
                    "role": "title",
                    "alignment": "center",
                    "bbox": [40, 40, 340, 120],
                    "text": "Figure 1",
                    "lines": [
                        {
                            "bbox": [120, 55, 260, 82],
                            "alignment": "center",
                            "phrases": [
                                {
                                    "id": "p_title",
                                    "bbox": [130, 55, 250, 82],
                                    "text": "Figure 1",
                                    "spans": [{"bbox": [130, 55, 250, 82], "texte": "Figure 1", "style": {"font": "Times", "size": 14}}],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        with patch("positioning_policy.get_element_relations_ai_enricher", return_value=_FakeAIHelper()):
            enrich_positioning_policy(page)
        policy = page["blocks"][0]["lines"][0]["phrases"][0]["positioning_policy"]
        self.assertEqual(policy["anchors"]["horizontal"]["primary"], "center")
        self.assertEqual(policy["expansion_policy"]["horizontal"], "grow_symmetrically")
        self.assertTrue(policy["semantic_context"]["model_used"])

    def test_end_aligned_value_prefers_end_anchor(self):
        page = {
            "page": 1,
            "dimensions": {"width": 420, "height": 320},
            "blocks": [
                {
                    "id": "b3",
                    "role": "body",
                    "alignment": "right",
                    "bbox": [40, 40, 320, 110],
                    "text": "Accuracy 42.1%",
                    "lines": [
                        {
                            "bbox": [220, 50, 300, 74],
                            "alignment": "right",
                            "phrases": [
                                {
                                    "id": "p_value",
                                    "bbox": [238, 50, 300, 74],
                                    "text": "42.1%",
                                    "spans": [{"bbox": [238, 50, 300, 74], "texte": "42.1%", "style": {"font": "Times", "size": 11}}],
                                }
                            ],
                        }
                    ],
                }
            ],
            "images": [],
            "drawings": [],
            "non_text_zones": [],
            "layout": {"columns": [{"id": 0, "x0": 30, "x1": 360}]},
        }
        with patch("positioning_policy.get_element_relations_ai_enricher", return_value=_FakeAIHelper()):
            enriched = self.layout_builder.build(page)
        policy = enriched["blocks"][0]["lines"][0]["phrases"][0]["positioning_policy"]
        self.assertEqual(policy["anchors"]["horizontal"]["primary"], "end")
        self.assertEqual(policy["expansion_policy"]["horizontal"], "grow_to_start")
        self.assertIn("positioning_policy", enriched)
        self.assertIn("positioning_policy", enriched["layout"])

    def test_code_fragment_skips_semantic_ai_scoring(self):
        helper = _FakeAIHelper()
        page = {
            "page": 1,
            "layout_direction": "ltr",
            "layout": {"columns": [{"id": 0, "x0": 20, "x1": 420}]},
            "blocks": [
                {
                    "id": "code_block",
                    "role": "body",
                    "unit_type": "code_visible",
                    "bbox": [40, 40, 360, 90],
                    "text": "x = inception_module(x, filters_1x1=64)",
                    "lines": [
                        {
                            "bbox": [40, 40, 360, 66],
                            "phrases": [
                                {
                                    "id": "code_phrase",
                                    "unit_type": "code_visible",
                                    "bbox": [40, 40, 360, 66],
                                    "text": "x = inception_module(x, filters_1x1=64)",
                                    "spans": [
                                        {
                                            "bbox": [40, 40, 360, 66],
                                            "texte": "x = inception_module(x, filters_1x1=64)",
                                            "style": {"font": "Courier", "size": 10, "flags": {"monospace": True}},
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        with patch("positioning_policy.get_element_relations_ai_enricher", return_value=helper):
            enrich_positioning_policy(page)

        policy = page["blocks"][0]["lines"][0]["phrases"][0]["positioning_policy"]
        self.assertFalse(policy["semantic_context"]["model_used"])
        self.assertEqual(helper.calls, 0)

    def test_element_ruleset_is_built_per_phrase(self):
        page = {
            "page": 1,
            "layout_direction": "ltr",
            "layout": {"columns": [{"id": 0, "x0": 20, "x1": 420}]},
            "blocks": [
                {
                    "id": "toc_row",
                    "role": "body",
                    "bbox": [40, 40, 360, 90],
                    "text": "Convolutional neural networks 92",
                    "lines": [
                        {
                            "bbox": [40, 40, 360, 66],
                            "phrases": [
                                {
                                    "id": "entry",
                                    "bbox": [40, 40, 260, 66],
                                    "text": "Convolutional neural networks",
                                    "spans": [{"bbox": [40, 40, 260, 66], "texte": "Convolutional neural networks", "style": {"font": "Times", "size": 11}}],
                                },
                                {
                                    "id": "page_no",
                                    "bbox": [330, 40, 360, 66],
                                    "text": "92",
                                    "spans": [{"bbox": [330, 40, 360, 66], "texte": "92", "style": {"font": "Times", "size": 11}}],
                                },
                            ],
                        }
                    ],
                }
            ],
        }
        with patch("positioning_policy.get_element_relations_ai_enricher", return_value=_FakeAIHelper()):
            enriched = self.layout_builder.build(page)
        phrase_entry = enriched["blocks"][0]["lines"][0]["phrases"][0]
        phrase_page_no = enriched["blocks"][0]["lines"][0]["phrases"][1]
        entry_ruleset = phrase_entry["element_ruleset"]
        page_no_ruleset = phrase_page_no["element_ruleset"]
        self.assertEqual(entry_ruleset["rules"]["preserve_horizontal_anchor"], "start")
        self.assertEqual(page_no_ruleset["rules"]["preserve_horizontal_anchor"], "end")
        self.assertNotEqual(entry_ruleset["ruleset_id"], page_no_ruleset["ruleset_id"])
        self.assertEqual(entry_ruleset["rules"]["semantic_role"], "flow_text")
        self.assertEqual(page_no_ruleset["rules"]["semantic_role"], "end_value")
        self.assertEqual(page_no_ruleset["position_reference_priority"]["horizontal"][0]["reference"], "end")
        self.assertIn("protect_value_alignment", page_no_ruleset["override_conditions"])
        self.assertIn("element_rulesets", enriched)
        self.assertIn("translation_rulesets", enriched["layout"])

    def test_element_ruleset_direct_enrichment_exposes_combined_modes(self):
        page = {
            "page": 1,
            "layout_direction": "ltr",
            "blocks": [
                {
                    "id": "b_rules",
                    "role": "title",
                    "alignment": "center",
                    "bbox": [40, 40, 340, 120],
                    "text": "Figure 1",
                    "lines": [
                        {
                            "bbox": [120, 55, 260, 82],
                            "alignment": "center",
                            "phrases": [
                                {
                                    "id": "p_rules",
                                    "bbox": [130, 55, 250, 82],
                                    "text": "Figure 1",
                                    "spans": [{"bbox": [130, 55, 250, 82], "texte": "Figure 1", "style": {"font": "Times", "size": 14}}],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        with patch("positioning_policy.get_element_relations_ai_enricher", return_value=_FakeAIHelper()):
            enrich_positioning_policy(page)
        enrich_element_rulesets(page)
        ruleset = page["blocks"][0]["lines"][0]["phrases"][0]["translation_ruleset"]
        self.assertEqual(ruleset["rules"]["preserve_horizontal_anchor"], "center")
        self.assertEqual(ruleset["rules"]["preserve_vertical_anchor"], "middle")
        self.assertEqual(ruleset["position_reference_priority"]["combined_modes"][0]["mode"], "middle_center")
        self.assertTrue(ruleset["constraints"]["preserve_center_if_possible"])

    def test_toc_specialized_roles_are_assigned_per_phrase(self):
        page = {
            "page": 1,
            "page_role": "toc",
            "layout_type": "toc_page",
            "layout_direction": "ltr",
            "toc": {
                "toc_rows": [
                    {
                        "role": "header",
                        "label": "CONTENTS",
                        "page": "vii",
                        "label_bbox": [120, 40, 220, 64],
                        "page_bbox": [320, 40, 350, 64],
                    },
                    {
                        "role": "section_heading",
                        "label": "3.1 Image classification using MLP",
                        "page": "93",
                        "label_bbox": [40, 100, 280, 126],
                        "page_bbox": [330, 100, 360, 126],
                    },
                ]
            },
            "blocks": [
                {
                    "id": "toc_header",
                    "role": "header",
                    "alignment": "center",
                    "bbox": [100, 40, 360, 68],
                    "text": "CONTENTS vii",
                    "lines": [
                        {
                            "bbox": [120, 40, 350, 64],
                            "alignment": "center",
                            "phrases": [
                                {
                                    "id": "toc_title",
                                    "bbox": [120, 40, 220, 64],
                                    "text": "CONTENTS",
                                    "spans": [{"bbox": [120, 40, 220, 64], "texte": "CONTENTS", "style": {"font": "Times", "size": 12}}],
                                },
                                {
                                    "id": "toc_front_page",
                                    "bbox": [320, 40, 350, 64],
                                    "text": "vii",
                                    "spans": [{"bbox": [320, 40, 350, 64], "texte": "vii", "style": {"font": "Times", "size": 10}}],
                                },
                            ],
                        }
                    ],
                },
                {
                    "id": "toc_entry",
                    "role": "body",
                    "bbox": [40, 100, 360, 126],
                    "text": "3.1 Image classification using MLP 93",
                    "lines": [
                        {
                            "bbox": [40, 100, 360, 126],
                            "phrases": [
                                {
                                    "id": "toc_section",
                                    "bbox": [40, 100, 74, 126],
                                    "text": "3.1",
                                    "spans": [{"bbox": [40, 100, 74, 126], "texte": "3.1", "style": {"font": "Times", "size": 10}}],
                                },
                                {
                                    "id": "toc_entry_title",
                                    "bbox": [82, 100, 280, 126],
                                    "text": "Image classification using MLP",
                                    "spans": [{"bbox": [82, 100, 280, 126], "texte": "Image classification using MLP", "style": {"font": "Times", "size": 10}}],
                                },
                                {
                                    "id": "toc_page_no",
                                    "bbox": [330, 100, 360, 126],
                                    "text": "93",
                                    "spans": [{"bbox": [330, 100, 360, 126], "texte": "93", "style": {"font": "Times", "size": 10}}],
                                },
                            ],
                        }
                    ],
                },
            ],
        }
        with patch("positioning_policy.get_element_relations_ai_enricher", return_value=_FakeAIHelper()):
            enriched = self.layout_builder.build(page)
        header_title = enriched["blocks"][0]["lines"][0]["phrases"][0]["element_ruleset"]
        front_page = enriched["blocks"][0]["lines"][0]["phrases"][1]["element_ruleset"]
        section_marker = enriched["blocks"][1]["lines"][0]["phrases"][0]["element_ruleset"]
        entry_title = enriched["blocks"][1]["lines"][0]["phrases"][1]["element_ruleset"]
        page_no = enriched["blocks"][1]["lines"][0]["phrases"][2]["element_ruleset"]
        toc_row = enriched["toc"]["toc_rows"][1]
        self.assertEqual(header_title["rules"]["semantic_role"], "toc_heading")
        self.assertEqual(front_page["rules"]["semantic_role"], "toc_page_number")
        self.assertEqual(section_marker["rules"]["semantic_role"], "toc_section_number")
        self.assertEqual(entry_title["rules"]["semantic_role"], "toc_entry_title")
        self.assertEqual(page_no["rules"]["semantic_role"], "toc_page_number")
        self.assertEqual(front_page["rules"]["preserve_horizontal_anchor"], "end")
        self.assertEqual(page_no["rules"]["preserve_horizontal_anchor"], "end")
        self.assertEqual(section_marker["rules"]["preserve_horizontal_anchor"], "start")
        self.assertIn("preserve_toc_row_pairing", entry_title["override_conditions"])
        self.assertIn("protect_section_marker_alignment", section_marker["override_conditions"])
        self.assertIn("protect_value_alignment", page_no["override_conditions"])
        label_roles = [
            str(((ruleset.get("rules") or {}).get("semantic_role")) or "")
            for ruleset in (toc_row.get("label_rulesets") or [])
            if isinstance(ruleset, dict)
        ]
        self.assertIn("toc_section_number", label_roles)
        self.assertIn("toc_entry_title", label_roles)
        self.assertEqual(((toc_row.get("page_ruleset") or {}).get("rules") or {}).get("semantic_role"), "toc_page_number")


if __name__ == "__main__":
    unittest.main()
