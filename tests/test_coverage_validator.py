import unittest
from pathlib import Path

import fitz

from coverage_validator import _classify_rendered_presence, _classify_unit_status, analyze_rendered_text_coverage
from publication_qa import _english_leak_count, _is_decorative_raster


class CoverageValidatorTests(unittest.TestCase):
    def setUp(self):
        self.pdf_path = Path("/tmp/coverage_validator_empty.pdf")
        doc = fitz.open()
        doc.new_page()
        doc.save(self.pdf_path)
        doc.close()

    def test_background_only_phrase_is_not_expected_in_rendered_text(self):
        translated_pages = [
            {
                "page": 0,
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "u1",
                                        "text": "Input image",
                                        "raw_text": "Input image",
                                        "translated_text": "Image d'entrée",
                                        "coverage_required": "strict",
                                        "translatable": True,
                                        "translation_strategy": "layout_constrained",
                                        "render_mode": "background_only",
                                    }
                                ]
                            }
                        ],
                    }
                ],
            }
        ]
        report = analyze_rendered_text_coverage([], translated_pages, str(self.pdf_path))
        self.assertEqual(report["summary"]["strict_units"], 0)
        self.assertEqual(report["summary"]["rendered_missing_units"], 0)

    def test_all_skip_render_spans_are_not_expected_in_rendered_text(self):
        translated_pages = [
            {
                "page": 0,
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "u2",
                                        "text": "0 x 120 +",
                                        "raw_text": "0 x 120 +",
                                        "translated_text": "0 x 120 +",
                                        "coverage_required": "strict",
                                        "translatable": False,
                                        "translation_strategy": "exact_preserve",
                                        "spans": [
                                            {"text": "0 x 120 +", "skip_render": True},
                                        ],
                                    }
                                ]
                            }
                        ],
                    }
                ],
            }
        ]
        report = analyze_rendered_text_coverage([], translated_pages, str(self.pdf_path))
        self.assertEqual(report["summary"]["strict_units"], 0)
        self.assertEqual(report["summary"]["rendered_missing_units"], 0)

    def test_partially_skipped_phrase_uses_visible_text_for_rendered_expectation(self):
        translated_pages = [
            {
                "page": 0,
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "u3",
                                        "text": "'1' 'same' ,",
                                        "raw_text": "model.add(Conv2D(filters=16, kernel_size=2, strides='1', padding='same',",
                                        "translated_text": "model.add(Conv2D(filters=16, kernel_size=2, strides='1', padding='same',",
                                        "coverage_required": "strict",
                                        "translatable": False,
                                        "translation_strategy": "exact_preserve",
                                        "spans": [
                                            {"text": "model.add(", "skip_render": True},
                                            {"text": "'1' 'same' ,", "skip_render": False},
                                        ],
                                    }
                                ]
                            }
                        ],
                    }
                ],
            }
        ]
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "'1' 'same' ,")
        doc.save(self.pdf_path)
        doc.close()
        report = analyze_rendered_text_coverage([], translated_pages, str(self.pdf_path))
        self.assertEqual(report["summary"]["rendered_missing_units"], 0)

    def test_english_leak_ignores_reference_and_citation_units(self):
        pages = [
            {
                "page": 0,
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "translated_text": "www.example.com/deep-learning",
                                        "texte_original": "www.example.com/deep-learning",
                                        "unit_type": "reference_link",
                                        "translation_strategy": "layout_constrained",
                                        "translatable": True,
                                    },
                                    {
                                        "translated_text": "Deepdream—A Code Example for Visualizing",
                                        "texte_original": "Deepdream—A Code Example for Visualizing",
                                        "unit_type": "citation",
                                        "translation_strategy": "layout_constrained",
                                        "translatable": True,
                                    },
                                ]
                            }
                        ],
                    }
                ],
            }
        ]
        report = _english_leak_count(pages, target_lang="fr")
        self.assertEqual(report["flagged_units"], 0)

    def test_short_chart_label_proper_noun_can_remain_unchanged(self):
        status, reason = _classify_unit_status(
            {
                "source_text": "Labrador",
                "translated_text": "Labrador",
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "translatable": True,
                "unit_type": "chart_label",
            },
            target_lang="fr",
        )
        self.assertEqual(status, "covered")
        self.assertEqual(reason, "preserved_short_label")

    def test_sentence_integrity_marks_partial_render_as_truncated(self):
        status, reason, expected = _classify_rendered_presence(
            {
                "translated_text": "Le projet a commencé comme une expérience amusante pour visualiser son fonctionnement.",
                "source_text": "The project started as a fun experiment to visualize its behavior.",
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "translatable": True,
                "preserve_sentence_integrity": True,
            },
            "Le projet a commencé comme une expérience amusante",
            full_text="",
        )
        self.assertEqual(status, "warning")
        self.assertEqual(reason, "sentence_truncated")
        self.assertTrue(expected.startswith("Le projet a commencé"))

    def test_sentence_integrity_marks_low_ratio_as_missing(self):
        status, reason, _ = _classify_rendered_presence(
            {
                "translated_text": "Le projet a commencé comme une expérience amusante pour visualiser son fonctionnement.",
                "source_text": "The project started as a fun experiment to visualize its behavior.",
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "translatable": True,
                "preserve_sentence_integrity": True,
            },
            "Le projet a commencé",
            full_text="",
        )
        self.assertEqual(status, "missing")
        self.assertEqual(reason, "sentence_not_fully_rendered")

    def test_header_compact_match_is_accepted(self):
        status, reason, expected = _classify_rendered_presence(
            {
                "translated_text": "CHAPITRE 6",
                "source_text": "CHAPTER 6",
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "translatable": True,
                "role": "header",
            },
            "CHAPITRE6Transfert de l'apprentissage",
            full_text="",
        )
        self.assertEqual(status, "covered")
        self.assertEqual(reason, "header_compact_match")
        self.assertEqual(expected, "CHAPITRE 6")

    def test_decorative_raster_is_not_treated_as_real_image_collision(self):
        page_area = float(600 * 800)
        decorative = fitz.Rect(90, 580, 340, 588)
        real_figure = fitz.Rect(90, 200, 340, 420)
        self.assertTrue(_is_decorative_raster(decorative, page_area))
        self.assertFalse(_is_decorative_raster(real_figure, page_area))

    def test_tiny_chart_tick_raster_is_treated_as_decorative(self):
        page_area = float(531.36 * 666.24)
        tick_mark = fitz.Rect(112.32, 405.12, 124.80, 412.80)
        self.assertTrue(_is_decorative_raster(tick_mark, page_area))

    def test_rendered_text_extraction_uses_words_for_compact_headers(self):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "CHAPITRE")
        page.insert_text((130, 72), "6")
        page.insert_text((160, 72), "Transfert")
        path = Path("/tmp/coverage_validator_words_header.pdf")
        doc.save(path)
        doc.close()
        from coverage_validator import _extract_rendered_page_texts

        texts = _extract_rendered_page_texts(str(path))
        self.assertTrue(texts)
        self.assertIn("CHAPITRE", texts[0])
        self.assertIn("Transfert", texts[0])

    def test_exact_preserve_visible_snippet_is_accepted(self):
        status, reason = _classify_unit_status(
            {
                "source_text": "'relu'",
                "visible_text": "'relu'",
                "translated_text": "activation='relu'))",
                "translation_strategy": "exact_preserve",
                "coverage_required": "strict",
                "translatable": False,
                "unit_type": "code_visible",
            },
            target_lang="fr",
        )
        self.assertEqual(status, "covered")
        self.assertEqual(reason, "preserved_visible_snippet")

    def test_short_symbolic_operator_clause_is_preserved_without_page_metadata(self):
        status, reason = _classify_unit_status(
            {
                "source_text": "value is > 0, which means that a",
                "translated_text": "value is > 0, which means that a",
                "translation_strategy": "layout_constrained",
                "coverage_required": "strict",
                "translatable": True,
                "unit_type": "narrative_body",
                "document_type": "",
                "layout_type": "",
            },
            target_lang="fr",
        )
        self.assertEqual(status, "covered")
        self.assertEqual(reason, "preserved_symbolic_clause")

    def test_paragraph_flow_uses_block_level_render_expectation(self):
        translated_pages = [
            {
                "page": 0,
                "blocks": [
                    {
                        "unit_id": "blk1",
                        "role": "body",
                        "text": "The histogram shows a strong separation between the two classes.",
                        "translated_text": "L'histogramme montre une forte séparation entre les deux classes.",
                        "render_policy": "paragraph_flow",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "u1",
                                        "texte": "The histogram shows a strong",
                                        "translated_text": "L'histogramme montre une forte",
                                        "translation_strategy": "layout_constrained",
                                        "coverage_required": "strict",
                                        "translatable": True,
                                        "render_policy": "paragraph_flow",
                                    }
                                ]
                            },
                            {
                                "phrases": [
                                    {
                                        "unit_id": "u2",
                                        "texte": "separation between the two classes.",
                                        "translated_text": "séparation entre les deux classes.",
                                        "translation_strategy": "layout_constrained",
                                        "coverage_required": "strict",
                                        "translatable": True,
                                        "render_policy": "paragraph_flow",
                                    }
                                ]
                            },
                        ],
                    }
                ],
            }
        ]
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "L'histogramme montre une forte séparation entre les deux classes.")
        doc.save(self.pdf_path)
        doc.close()
        report = analyze_rendered_text_coverage([], translated_pages, str(self.pdf_path))
        self.assertEqual(report["summary"]["rendered_missing_units"], 0)


if __name__ == "__main__":
    unittest.main()
