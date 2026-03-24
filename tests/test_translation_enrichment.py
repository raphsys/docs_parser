import unittest

from page_policy_matrix import PagePolicyMatrix
from translator import DocumentTranslator
from reconstructor import DocumentReconstructor


class TranslationEnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.translator = DocumentTranslator.__new__(DocumentTranslator)
        self.reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        self.page_policy = PagePolicyMatrix()

    def test_equation_role_preserves_true_formula(self):
        self.assertTrue(self.translator._should_preserve_equation_role_text("dW / dX"))
        self.assertTrue(self.translator._should_preserve_equation_role_text("H2SO4"))

    def test_equation_role_does_not_preserve_technical_label(self):
        self.assertFalse(self.translator._should_preserve_equation_role_text("Multi-scale feature layers"))
        self.assertFalse(self.translator._should_preserve_equation_role_text("Hidden layer outputs"))

    def test_equation_reference_remains_preserved(self):
        self.assertTrue(self.translator._should_preserve_equation_role_text("7.3.3"))
        self.assertTrue(self.translator._should_preserve_equation_role_text("(2)"))

    def test_mixed_url_sentence_is_not_fully_protected(self):
        self.assertFalse(
            self.translator._is_protected_segment(
                "Visit the book's website at www.manning.com/books/example", block_role="body"
            )
        )

    def test_standalone_url_stays_protected(self):
        self.assertTrue(
            self.translator._is_protected_segment(
                "www.manning.com/books/example", block_role="body"
            )
        )

    def test_long_sentence_with_et_al_is_not_fully_protected(self):
        self.assertFalse(
            self.translator._is_protected_segment(
                "DeepDream was developed by Google researchers Alexander Mordvintsev et al. in 2015.",
                block_role="body",
            )
        )

    def test_prose_with_comparator_is_not_protected_as_equation(self):
        self.assertFalse(
            self.translator._is_protected_segment(
                "value is > 0, which means that a",
                block_role="body",
            )
        )

    def test_equation_label_is_rendered_as_anchored_text(self):
        self.assertTrue(
            self.reconstructor._should_render_equation_as_anchored_text(
                {"role": "equation_inline", "text": "Multi-scale feature layers"}
            )
        )
        self.assertTrue(
            self.reconstructor._should_render_equation_as_anchored_text(
                {"role": "equation_inline", "text": "7.3.3"}
            )
        )

    def test_true_equation_is_not_rendered_as_anchored_text(self):
        self.assertFalse(
            self.reconstructor._should_render_equation_as_anchored_text(
                {"role": "equation_inline", "text": "dW / dX"}
            )
        )
        self.assertFalse(
            self.reconstructor._should_render_equation_as_anchored_text(
                {"role": "equation_inline", "text": "x = y + z"}
            )
        )

    def test_code_like_text_is_exact_preserve(self):
        policy = self.page_policy.classify_unit_policy(
            text="from keras.layers import Conv2D",
            role="title",
            source_kind="native_phrase",
            page_role="body",
            page_family="table_page",
            page_family_group="table_page",
        )
        self.assertFalse(policy["translatable"])
        self.assertEqual(policy["translation_strategy"], "exact_preserve")

    def test_prose_starting_with_from_is_not_code_like(self):
        policy = self.page_policy.classify_unit_policy(
            text="From the histogram, we can see that the dogs are separated by height.",
            role="body",
            source_kind="native_phrase",
            page_role="body",
            page_family="body_with_figure",
            page_family_group="body_with_figure",
            document_type="book_page",
            layout_type="annotated_page",
            style_profile="editorial_visual",
        )
        self.assertEqual(policy["unit_type"], "narrative_body")
        self.assertTrue(policy["translatable"])

    def test_short_chart_label_gets_dedicated_unit_type(self):
        policy = self.page_policy.classify_unit_policy(
            text="Number of dogs",
            role="title",
            source_kind="native_phrase",
            page_role="body",
            page_family="chart_label_page",
            page_family_group="body_with_figure",
        )
        self.assertEqual(policy["unit_type"], "chart_label")
        self.assertEqual(policy["translation_strategy"], "layout_constrained")

    def test_reference_link_gets_exact_preserve_policy(self):
        policy = self.page_policy.classify_unit_policy(
            text="www.example.com/deep-learning",
            role="body",
            source_kind="native_phrase",
            page_role="body",
            page_family="narrative_reference_page",
            page_family_group="body_text",
        )
        self.assertEqual(policy["unit_type"], "reference_link")
        self.assertFalse(policy["translatable"])

    def test_short_native_title_gets_short_label_policy_without_special_page_family(self):
        policy = self.page_policy.classify_unit_policy(
            text="Input image",
            role="title",
            source_kind="native_phrase",
            page_role="body",
            page_family="unknown",
            page_family_group="unknown",
        )
        self.assertEqual(policy["unit_type"], "short_label")
        self.assertEqual(policy["translation_strategy"], "layout_constrained")
        self.assertEqual(policy["render_policy"], "anchored_text")

    def test_short_native_body_label_gets_short_label_policy(self):
        policy = self.page_policy.classify_unit_policy(
            text="Dogs",
            role="body",
            source_kind="native_phrase",
            page_role="body",
            page_family="body_text_two_column",
            page_family_group="body_text",
        )
        self.assertEqual(policy["unit_type"], "short_label")
        self.assertEqual(policy["translation_strategy"], "layout_constrained")

    def test_annotated_layout_drives_short_label_policy_without_page_family(self):
        policy = self.page_policy.classify_unit_policy(
            text="Activation",
            role="body",
            source_kind="native_phrase",
            page_role="body",
            page_family="unknown",
            page_family_group="unknown",
            document_type="manual_guide",
            layout_type="annotated_page",
            style_profile="editorial_visual",
        )
        self.assertEqual(policy["unit_type"], "short_label")
        self.assertEqual(policy["render_policy"], "anchored_text")

    def test_reference_page_uses_reference_policy_from_layout_type(self):
        policy = self.page_policy.classify_unit_policy(
            text="www.example.com/paper",
            role="body",
            source_kind="native_phrase",
            page_role="body",
            page_family="unknown",
            page_family_group="unknown",
            document_type="web_print",
            layout_type="reference_page",
            style_profile="mixed_irregular",
        )
        self.assertEqual(policy["unit_type"], "reference_link")
        self.assertEqual(policy["translation_strategy"], "exact_preserve")

    def test_annotated_page_long_body_uses_paragraph_flow(self):
        policy = self.page_policy.classify_unit_policy(
            text="This explanatory paragraph sits next to the chart and describes how the visual evidence should be interpreted by the reader.",
            role="body",
            source_kind="native_phrase",
            page_role="body",
            page_family="illustrated_label_page",
            page_family_group="body_with_figure",
            document_type="manual_guide",
            layout_type="annotated_page",
            style_profile="editorial_visual",
        )
        self.assertEqual(policy["unit_type"], "narrative_body")
        self.assertEqual(policy["render_policy"], "paragraph_flow")

    def test_annotated_page_explanatory_label_gets_diagram_label_type(self):
        unit_type = self.page_policy.classify_unit_type(
            text="Eye (sensing device responsible for capturing images of the environment)",
            role="title",
            source_kind="native_phrase",
            page_family="illustrated_label_page",
            page_family_group="body_with_figure",
            document_type="manual_guide",
            layout_type="annotated_page",
            style_profile="editorial_visual",
        )
        self.assertEqual(unit_type, "diagram_label")

    def test_short_label_lexical_fallback_translates_human_parts(self):
        translated = self.translator._fr_short_label_lexical_fallback("Human head")
        self.assertEqual(translated, "tête humaine")

    def test_reference_like_sentence_with_url_stays_narrative_body(self):
        unit_type = self.page_policy.classify_unit_type(
            text="Visit the book's website at www.manning.com/books/example to download the notebook.",
            role="body",
            source_kind="native_phrase",
            page_family="unknown",
            page_family_group="unknown",
            document_type="book_page",
            layout_type="double_column",
            style_profile="minimalist",
        )
        self.assertNotEqual(unit_type, "reference_link")


if __name__ == "__main__":
    unittest.main()
