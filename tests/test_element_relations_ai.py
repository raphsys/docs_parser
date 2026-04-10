import os
import tempfile
import unittest

from element_relations import enrich_element_relations
from element_relations_ai import ElementRelationsAIEnricher


class ElementRelationsAITests(unittest.TestCase):
    def _sample_page(self):
        return {
            "page": 1,
            "layout_direction": "ltr",
            "layout": {},
            "blocks": [
                {
                    "id": "b1",
                    "role": "body",
                    "source": "native",
                    "bbox": [40, 40, 240, 140],
                    "lines": [
                        {
                            "bbox": [40, 40, 200, 64],
                            "line_index": 0,
                            "line_text": "The experiment ended.",
                            "phrases": [
                                {
                                    "id": "p1",
                                    "bbox": [40, 40, 200, 64],
                                    "text": "The experiment ended.",
                                    "line_index": 0,
                                    "line_break_after": True,
                                    "spans": [{"bbox": [40, 40, 200, 64], "texte": "The experiment ended.", "style": {"font": "Times", "size": 11}}],
                                }
                            ],
                        },
                        {
                            "bbox": [42, 70, 220, 94],
                            "line_index": 1,
                            "line_text": "continued on the next line",
                            "phrases": [
                                {
                                    "id": "p2",
                                    "bbox": [42, 70, 220, 94],
                                    "text": "continued on the next line",
                                    "line_index": 1,
                                    "hard_break_before": False,
                                    "line_break_after": True,
                                    "spans": [{"bbox": [42, 70, 220, 94], "texte": "continued on the next line", "style": {"font": "Times", "size": 11}}],
                                }
                            ],
                        },
                    ],
                }
            ],
        }

    def test_disabled_enricher_is_noop_but_exposes_status(self):
        old_enable = os.environ.get("ELEMENT_RELATIONS_AI_ENABLE")
        try:
            os.environ["ELEMENT_RELATIONS_AI_ENABLE"] = "0"
            enricher = ElementRelationsAIEnricher()
            page = self._sample_page()
            enrich_element_relations(page)
            out, info = enricher.enrich(page)
            self.assertIs(out, page)
            self.assertFalse(info["enabled"])
            self.assertFalse(info["applied"])
            self.assertIn("element_relations_ai", out)
            self.assertIn("element_relations_ai", out["layout"])
        finally:
            if old_enable is None:
                os.environ.pop("ELEMENT_RELATIONS_AI_ENABLE", None)
            else:
                os.environ["ELEMENT_RELATIONS_AI_ENABLE"] = old_enable

    def test_ai_review_can_resolve_ambiguous_relation(self):
        class FakeAIEnricher(ElementRelationsAIEnricher):
            def __init__(self):
                super().__init__()
                self.enabled = True

            def _get_runtime(self):
                self._runtime = {"fake": True}
                self._load_error = None
                return self._runtime

            def _review_relation(self, relation):
                return {
                    "review_mode": "fake_nli",
                    "continuation_label": "continuation",
                    "continuation_confidence": 0.93,
                    "logical_label": "same_paragraph_continuation",
                    "logical_confidence": 0.89,
                    "continuation_scores": {"continuation": 0.93, "new_unit": 0.07},
                    "logical_scores": {
                        "same_paragraph_continuation": 0.89,
                        "new_sentence_or_unit": 0.05,
                        "new_structural_unit": 0.03,
                        "same_sentence_continuation": 0.03,
                    },
                }

        page = self._sample_page()
        enrich_element_relations(page)
        relation = page["element_relations"]["flat_relations"][0]
        relation["ai_review_required"] = True
        relation["logical_relation"] = "uncertain"
        relation["visual_relation"] = "new_structural_unit"
        relation["continuation"] = False
        relation["confidence"] = 0.35

        enricher = FakeAIEnricher()
        out, info = enricher.enrich(page)

        self.assertTrue(info["ready"])
        self.assertTrue(info["applied"])
        self.assertEqual(info["reviewed_relations"], 1)
        self.assertEqual(info["resolved_relations"], 1)
        self.assertTrue(relation["continuation"])
        self.assertEqual(relation["visual_relation"], "continues_wrapped_line")
        self.assertEqual(relation["logical_relation"], "same_paragraph_continuation")
        self.assertEqual(relation["resolved_by"], "semantic_ai")
        self.assertFalse(relation["ai_review_required"])
        self.assertIn("heuristic_decision", relation)
        self.assertEqual(relation["semantic_ai_review"]["review_mode"], "fake_nli")
        self.assertIs(out, page)

    def test_local_model_bundle_accepts_standard_quantized_onnx_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as fh:
                fh.write("{}")
            os.makedirs(os.path.join(tmp, "onnx"), exist_ok=True)
            with open(os.path.join(tmp, "onnx", "model_quint8_avx2.onnx"), "wb") as fh:
                fh.write(b"placeholder")

            old_root = os.environ.get("ELEMENT_RELATIONS_AI_MODEL_DIR")
            old_enable = os.environ.get("ELEMENT_RELATIONS_AI_ENABLE")
            try:
                os.environ["ELEMENT_RELATIONS_AI_MODEL_DIR"] = tmp
                os.environ.pop("ELEMENT_RELATIONS_AI_ENABLE", None)
                enricher = ElementRelationsAIEnricher()
                self.assertTrue(enricher.enabled)
                self.assertEqual(enricher._resolve_model_path(), os.path.join(tmp, "onnx", "model_quint8_avx2.onnx"))
            finally:
                if old_root is None:
                    os.environ.pop("ELEMENT_RELATIONS_AI_MODEL_DIR", None)
                else:
                    os.environ["ELEMENT_RELATIONS_AI_MODEL_DIR"] = old_root
                if old_enable is None:
                    os.environ.pop("ELEMENT_RELATIONS_AI_ENABLE", None)
                else:
                    os.environ["ELEMENT_RELATIONS_AI_ENABLE"] = old_enable

    def test_score_hypotheses_uses_cache_for_identical_requests(self):
        class CachedFakeEnricher(ElementRelationsAIEnricher):
            def __init__(self):
                super().__init__()
                self.enabled = True
                self.batch_calls = 0

            def _get_runtime(self):
                self._runtime = {"fake": True}
                self._load_error = None
                return self._runtime

            def _score_hypotheses_batch(self, premise, hypotheses, runtime):
                self.batch_calls += 1
                return {"continuation": 0.9, "new_unit": 0.1}

        enricher = CachedFakeEnricher()
        hypotheses = {
            "continuation": "the second fragment is a continuation of the previous text",
            "new_unit": "the second fragment starts a new textual unit",
        }

        score_1 = enricher.score_hypotheses("Previous fragment: A. Next fragment: B.", hypotheses)
        score_2 = enricher.score_hypotheses("Previous fragment: A. Next fragment: B.", hypotheses)

        self.assertEqual(enricher.batch_calls, 1)
        self.assertEqual(score_1, score_2)
        self.assertAlmostEqual(score_1["continuation"], 0.9)


if __name__ == "__main__":
    unittest.main()
