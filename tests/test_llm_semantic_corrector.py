import unittest
from unittest import mock

import llm_semantic_corrector


class LLMSemanticCorrectorTests(unittest.TestCase):
    def test_resolve_model_spec_supports_local_phi_alias(self):
        spec = llm_semantic_corrector._resolve_model_spec("phi35-mini-local")

        self.assertEqual(spec["family"], "phi")
        self.assertTrue(spec["trust_remote_code"])
        self.assertTrue(spec["model_id"].endswith("ai_models/phi35_mini_instruct"))

    def test_parse_response_accepts_intra_line_sentence_boundaries(self):
        parsed = llm_semantic_corrector._parse_response(
            '{"c":[],"bc":["sb:0:0"],"sb":[{"i":0,"s":["First segment","Second segment"]}],"lb":[1],"lm":{"line_flow":"preserve_line_breaks","breaks":[{"i":0,"after":"new_line"},{"i":1,"after":"block_end"}]},"hj":[]}'
        )

        self.assertEqual(parsed["c"], [])
        self.assertEqual(parsed["bc"], ["sb:0:0"])
        self.assertEqual(
            parsed["sb"],
            [{"i": 0, "s": ["First segment", "Second segment"]}],
        )
        self.assertEqual(parsed["lb"], [1])
        self.assertEqual(
            parsed["lm"],
            {
                "line_flow": "preserve_line_breaks",
                "breaks": [{"i": 0, "after": "new_line"}, {"i": 1, "after": "block_end"}],
            },
        )
        self.assertEqual(parsed["hj"], [])

    def test_parse_response_rejects_invalid_layout_mode(self):
        parsed = llm_semantic_corrector._parse_response(
            '{"c":[],"bc":[],"sb":[],"lb":[],"lm":{"line_flow":"teleport","breaks":[{"i":0,"after":"new_line"}]},"hj":[]}'
        )

        self.assertIsNone(parsed["lm"])

    def test_resolve_model_spec_prefers_ct2_for_qwen_fast_alias_when_available(self):
        with mock.patch.dict("os.environ", {"SEMANTIC_CORRECTOR_BACKEND": "auto"}, clear=False):
            spec = llm_semantic_corrector._resolve_model_spec("qwen-fast")

        self.assertEqual(spec["family"], "qwen")
        self.assertEqual(spec["backend"], "ct2")
        self.assertTrue(spec["ct2_model_dir"].endswith("ai_models/semantic_corrector/qwen2_5_0_5b_ct2_int8"))

    def test_runtime_model_info_exposes_backend_choice(self):
        with mock.patch.dict("os.environ", {"SEMANTIC_CORRECTOR_MODEL": "qwen-fast", "SEMANTIC_CORRECTOR_BACKEND": "auto"}, clear=False):
            info = llm_semantic_corrector.get_runtime_model_info()

        self.assertEqual(info["family"], "qwen")
        self.assertEqual(info["backend"], "ct2")

    def test_format_block_for_prompt_includes_heuristic_semantic_phrases(self):
        payload = llm_semantic_corrector.format_block_for_prompt(
            {
                "role": "body",
                "bbox": [0, 0, 100, 100],
                "lines": [
                    {"line_index": 0, "line_text": "First line. Second line", "bbox": [0, 0, 100, 10]},
                    {"line_index": 1, "line_text": "Third line", "bbox": [0, 12, 100, 22]},
                ],
                "semantic_phrases": [
                    {
                        "line_indices": [0, 1],
                        "text": "First line. Second line Third line",
                        "sentence_end_reason": "eof",
                    }
                ],
                "source_layout_mode": {
                    "mode": "continuous_paragraph",
                    "line_flow": "inline_reflow",
                    "render_contract": "reflow_block",
                },
            }
        )

        self.assertIn('"hs"', payload)
        self.assertIn('"lm0"', payload)
        self.assertIn('"bc"', payload)
        self.assertIn('"id":"sb:0:0"', payload)
        self.assertIn('"li":[0,1]', payload)
        self.assertIn('"r":"eof"', payload)

    def test_resolve_selected_boundary_candidates_maps_candidate_ids_to_boundaries(self):
        resolved = llm_semantic_corrector._resolve_selected_boundary_candidates(
            {"c": [], "bc": ["sb:1:0", "lb:0"], "sb": [], "lb": [], "lm": {"line_flow": "inline_reflow", "breaks": []}, "hj": []},
            candidate_map={
                "sb:1:0": {"id": "sb:1:0", "k": "sb", "i": 1, "s": ["and 3 for color images.", "In the later layers"]},
                "lb:0": {"id": "lb:0", "k": "lb", "after": 0},
            },
        )

        self.assertEqual(resolved["bc"], ["sb:1:0", "lb:0"])
        self.assertEqual(resolved["sb"], [{"i": 1, "s": ["and 3 for color images.", "In the later layers"]}])
        self.assertEqual(resolved["lb"], [0])
        self.assertEqual(resolved["lm"], {"line_flow": "inline_reflow", "breaks": []})

    def test_select_boundary_candidates_via_votes_accepts_yes_answers(self):
        block = {
            "lines": [
                {"line_index": 0, "line_text": "Depth is referred to as the color channel, where depth is 1 for grayscale images"},
                {"line_index": 1, "line_text": "and 3 for color images. In the later layers, the images still have depth"},
            ],
            "semantic_phrases": [
                {
                    "text": "Depth is referred to as the color channel, where depth is 1 for grayscale images and 3 for color images.",
                    "sentence_end_reason": "terminal_punctuation",
                },
                {
                    "text": "and 3 for color images. In the later layers, the images still have depth.",
                    "sentence_end_reason": "terminal_punctuation",
                },
            ],
        }
        candidate_map = {
            "sb:1:0": {"id": "sb:1:0", "k": "sb", "i": 1, "s": ["and 3 for color images.", "In the later layers, the images still have depth"]},
            "lb:0": {"id": "lb:0", "k": "lb", "after": 0, "prev": "Depth is referred to as the color channel", "next": "and 3 for color images. In the later layers"},
        }

        with mock.patch.object(llm_semantic_corrector, "_generate_raw_response", side_effect=["YES", "NO"]):
            selected = llm_semantic_corrector._select_boundary_candidates_via_votes(block, {"backend": "ct2"}, candidate_map, max_candidates=2)

        self.assertEqual(selected, ["sb:1:0"])

    def test_auto_select_boundary_candidates_keeps_high_confidence_boundaries(self):
        block = {
            "lines": [
                {"line_index": 3, "line_text": "you see that each image is a 3D object that has a height, width, and depth."},
                {"line_index": 4, "line_text": "Depth is referred to as the color channel, where depth is 1 for grayscale images"},
                {"line_index": 5, "line_text": "and 3 for color images. In the later layers, the images still have depth"},
            ],
            "semantic_phrases": [
                {
                    "text": "Depth is referred to as the color channel, where depth is 1 for grayscale images and 3 for color images.",
                    "sentence_end_reason": "terminal_punctuation",
                },
                {
                    "text": "and 3 for color images. In the later layers, the images still have depth.",
                    "sentence_end_reason": "terminal_punctuation",
                },
            ],
        }
        candidate_map = {
            "lb:3": {"id": "lb:3", "k": "lb", "after": 3, "prev": "you see that each image is a 3D object that has a height, width, and depth.", "next": "Depth is referred to as the color channel"},
            "sb:5:0": {"id": "sb:5:0", "k": "sb", "i": 5, "s": ["and 3 for color images.", "In the later layers, the images still have depth"]},
        }

        selected = llm_semantic_corrector._auto_select_boundary_candidates(block, candidate_map)

        self.assertEqual(selected, ["lb:3", "sb:5:0"])

    def test_auto_select_boundary_candidates_does_not_force_lb_on_carryover_line(self):
        block = {
            "lines": [
                {"line_index": 4, "line_text": "Depth is referred to as the color channel, where depth is 1 for grayscale images"},
                {"line_index": 5, "line_text": "and 3 for color images. In the later layers, the images still have depth"},
            ],
            "semantic_phrases": [
                {
                    "text": "Depth is referred to as the color channel, where depth is 1 for grayscale images and 3 for color images.",
                    "sentence_end_reason": "terminal_punctuation",
                },
                {
                    "text": "and 3 for color images. In the later layers, the images still have depth.",
                    "sentence_end_reason": "terminal_punctuation",
                },
            ],
        }
        candidate_map = {
            "lb:4": {
                "id": "lb:4",
                "k": "lb",
                "after": 4,
                "prev": "Depth is referred to as the color channel, where depth is 1 for grayscale images",
                "next": "and 3 for color images. In the later layers, the images still have depth",
            },
            "sb:5:0": {
                "id": "sb:5:0",
                "k": "sb",
                "i": 5,
                "s": ["and 3 for color images.", "In the later layers, the images still have depth"],
            },
        }

        selected = llm_semantic_corrector._auto_select_boundary_candidates(block, candidate_map)

        self.assertEqual(selected, ["sb:5:0"])

    def test_build_boundary_candidates_skips_lb_when_next_line_has_carryover_then_new_sentence(self):
        candidates, _ = llm_semantic_corrector._build_boundary_candidates(
            {
                "lines": [
                    {"line_index": 4, "line_text": "Depth is referred to as the color channel, where depth is 1 for grayscale images"},
                    {"line_index": 5, "line_text": "and 3 for color images. In the later layers, the images still have depth"},
                ]
            }
        )

        candidate_ids = [candidate["id"] for candidate in candidates]
        self.assertIn("sb:5:0", candidate_ids)
        self.assertNotIn("lb:4", candidate_ids)

    def test_score_block_ambiguity_flags_overlap_artifact_even_when_reasons_look_resolved(self):
        block = {
            "lines": [
                {"line_index": 0, "line_text": "Depth is referred to as the color channel, where depth is 1 for grayscale images"},
                {"line_index": 1, "line_text": "and 3 for color images. In the later layers, the images still have depth"},
            ],
            "semantic_phrases": [
                {
                    "text": "Depth is referred to as the color channel, where depth is 1 for grayscale images and 3 for color images.",
                    "sentence_end_reason": "terminal_punctuation",
                    "multi_line": True,
                },
                {
                    "text": "and 3 for color images. In the later layers, the images still have depth.",
                    "sentence_end_reason": "terminal_punctuation",
                    "multi_line": True,
                },
            ],
        }

        score = llm_semantic_corrector.score_block_ambiguity(block)

        self.assertGreaterEqual(score, 0.45)

    def test_score_block_ambiguity_bypasses_block_size_cap_for_overlap_artifact(self):
        repeated = " feature map" * 120
        block = {
            "lines": [
                {"line_index": 0, "line_text": f"Depth is referred to as the color channel{repeated}"},
                {"line_index": 1, "line_text": "and 3 for color images. In the later layers, the images still have depth"},
            ],
            "semantic_phrases": [
                {
                    "text": f"Depth is referred to as the color channel{repeated} and 3 for color images.",
                    "sentence_end_reason": "terminal_punctuation",
                    "multi_line": True,
                },
                {
                    "text": "and 3 for color images. In the later layers, the images still have depth.",
                    "sentence_end_reason": "terminal_punctuation",
                    "multi_line": True,
                },
            ],
        }

        score = llm_semantic_corrector.score_block_ambiguity(block)

        self.assertGreaterEqual(score, 0.45)

    def test_score_block_ambiguity_flags_dangling_eof_phrase_without_terminal_punctuation(self):
        block = {
            "lines": [
                {"line_index": 0, "line_text": "Most of these architectures are famous because they performed well"},
                {"line_index": 1, "line_text": "ImageNet is a famous benchmark that"},
            ],
            "semantic_phrases": [
                {
                    "text": "Most of these architectures are famous because they performed well.",
                    "sentence_end_reason": "terminal_punctuation",
                    "multi_line": False,
                },
                {
                    "text": "ImageNet is a famous benchmark that",
                    "sentence_end_reason": "eof",
                    "multi_line": False,
                },
            ],
        }

        score = llm_semantic_corrector.score_block_ambiguity(block)

        self.assertGreaterEqual(score, 0.30)

    def test_block_needs_strong_retry_on_empty_result_for_overlap_artifact(self):
        block = {
            "lines": [
                {"line_index": 0, "line_text": "Depth is referred to as the color channel, where depth is 1 for grayscale images"},
                {"line_index": 1, "line_text": "and 3 for color images. In the later layers, the images still have depth"},
            ],
            "semantic_phrases": [
                {
                    "text": "Depth is referred to as the color channel, where depth is 1 for grayscale images and 3 for color images.",
                    "sentence_end_reason": "terminal_punctuation",
                },
                {
                    "text": "and 3 for color images. In the later layers, the images still have depth.",
                    "sentence_end_reason": "terminal_punctuation",
                },
            ],
        }

        self.assertTrue(
            llm_semantic_corrector.block_needs_strong_retry(
                block,
                score=0.65,
                result={"c": [], "bc": [], "sb": [], "lb": [], "hj": []},
            )
        )


if __name__ == "__main__":
    unittest.main()
