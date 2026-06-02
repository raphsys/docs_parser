import unittest
from pathlib import Path

import fitz

from coverage_validator import _classify_rendered_presence, _classify_unit_status, _page_index_for_unit, analyze_document_coverage, analyze_rendered_text_coverage
from scripts.run_reconstruction_validation import _source_overlay_findings_from_lines
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

    def test_middle_dot_ocr_noise_does_not_break_rendered_sentence_match(self):
        status, reason, expected = _classify_rendered_presence(
            {
                "translated_text": "Visit the book's website at www.manning.com/books/deep-learning-for-vision-",
                "source_text": "Visit the book's website at www.manning.com/books/deep-learning-for-vision-",
                "translation_strategy": "exact_preserve",
                "coverage_required": "strict",
                "translatable": False,
                "preserve_sentence_integrity": True,
            },
            "Visit the book·s website at www.manning.com/books/deep-learning-for-vision-",
            full_text="",
        )
        self.assertEqual(status, "covered")
        self.assertEqual(reason, "substring_match")
        self.assertTrue(expected.startswith("Visit the book"))

    def test_reference_like_caption_uses_token_match_when_punctuation_is_noisy(self):
        status, reason, expected = _classify_rendered_presence(
            {
                "translated_text": "An Update to Open Images Now with Bounding Boxes July 2017 http://mng.bz/yyVG",
                "source_text": "An Update to Open Images Now with Bounding Boxes July 2017 http://mng.bz/yyVG",
                "translation_strategy": "exact_preserve",
                "coverage_required": "strict",
                "translatable": False,
                "unit_type": "citation",
            },
            "·An Update to Open Images·Now with Bounding-Boxes,· July 2017 http://mng.bz/yyVG",
            full_text="",
        )
        self.assertEqual(status, "covered")
        self.assertEqual(reason, "reference_like_token_match")
        self.assertIn("Open Images", expected)

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

    def test_source_overlay_ignores_short_common_source_fragments(self):
        findings = _source_overlay_findings_from_lines(
            [
                {
                    "non_translated_expected": False,
                    "text": "explique comment",
                    "source_text": "and",
                    "present_in_region": True,
                },
                {
                    "non_translated_expected": False,
                    "text": "texte traduit",
                    "source_text": "Information",
                    "present_in_region": True,
                },
            ],
            region_text="texte traduit explique comment",
        )
        self.assertEqual(findings, [])

    def test_page_index_for_unit_prefers_explicit_page_index(self):
        self.assertEqual(_page_index_for_unit({"page_id": 25, "page_index": 4}, 25), 4)

    def test_rendered_coverage_uses_page_index_derived_from_page_number(self):
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text((72, 72), "first page")
        page2 = doc.new_page()
        page2.insert_text((72, 72), "Bonjour le monde")
        doc.save(self.pdf_path)
        doc.close()

        translated_pages = [
            {
                "page": 2,
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "u_page_refs",
                                        "text": "Hello world",
                                        "raw_text": "Hello world",
                                        "translated_text": "Bonjour le monde",
                                        "coverage_required": "strict",
                                        "translatable": True,
                                        "translation_strategy": "layout_constrained",
                                    }
                                ]
                            }
                        ],
                    }
                ],
            }
        ]
        report = analyze_rendered_text_coverage([], translated_pages, str(self.pdf_path))
        self.assertEqual(report["summary"]["rendered_missing_units"], 0)
        self.assertEqual(report["summary"]["rendered_warning_units"], 0)

    def test_document_coverage_matches_duplicate_unit_ids_by_page_index(self):
        source_pages = [
            {
                "page": 1,
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "dup",
                                        "text": "First source",
                                        "raw_text": "First source",
                                        "translated_text": "Premier",
                                        "coverage_required": "strict",
                                        "translatable": True,
                                        "translation_strategy": "layout_constrained",
                                    }
                                ]
                            }
                        ],
                    }
                ],
            },
            {
                "page": 2,
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "dup",
                                        "text": "Second source",
                                        "raw_text": "Second source",
                                        "translated_text": "Deuxieme",
                                        "coverage_required": "strict",
                                        "translatable": True,
                                        "translation_strategy": "layout_constrained",
                                    }
                                ]
                            }
                        ],
                    }
                ],
            },
        ]
        translated_pages = [
            {
                "page": 1,
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "dup",
                                        "text": "First source",
                                        "raw_text": "First source",
                                        "translated_text": "Premier",
                                        "coverage_required": "strict",
                                        "translatable": True,
                                        "translation_strategy": "layout_constrained",
                                    }
                                ]
                            }
                        ],
                    }
                ],
            },
            {
                "page": 2,
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "dup",
                                        "text": "Second source",
                                        "raw_text": "Second source",
                                        "translated_text": "Deuxieme",
                                        "coverage_required": "strict",
                                        "translatable": True,
                                        "translation_strategy": "layout_constrained",
                                    }
                                ]
                            }
                        ],
                    }
                ],
            },
        ]
        report = analyze_document_coverage(source_pages, translated_pages)
        self.assertEqual(report["summary"]["missing_units"], 0)
        self.assertEqual(report["summary"]["warning_units"], 0)

    def test_document_coverage_exposes_page_histogram(self):
        source_pages = [
            {
                "page": 3,
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "warn_hist",
                                        "text": "Deep Learning for",
                                        "raw_text": "Deep Learning for",
                                        "translated_text": "",
                                        "coverage_required": "strict",
                                        "translatable": True,
                                        "translation_strategy": "layout_constrained",
                                    }
                                ]
                            }
                        ],
                    }
                ],
            }
        ]
        translated_pages = [
            {
                "page": 3,
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "warn_hist",
                                        "text": "Deep Learning for",
                                        "raw_text": "Deep Learning for",
                                        "translated_text": "4",
                                        "coverage_required": "strict",
                                        "translatable": True,
                                        "translation_strategy": "layout_constrained",
                                    }
                                ]
                            }
                        ],
                    }
                ],
            }
        ]
        report = analyze_document_coverage(source_pages, translated_pages)
        self.assertEqual(report["findings_total"], 1)
        self.assertEqual(report["findings_by_page"], {3: 1})
        self.assertEqual(report["findings_by_reason"], {"overcompressed": 1})

    def test_document_coverage_prefers_fallback_index_over_stale_descriptor_page_id(self):
        source_pages = [
            {
                "layout_descriptor": {"page_id": 0},
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "stale_desc",
                                        "text": "Needs more translation",
                                        "raw_text": "Needs more translation",
                                        "translated_text": "",
                                        "coverage_required": "strict",
                                        "translatable": True,
                                        "translation_strategy": "layout_constrained",
                                    }
                                ]
                            }
                        ],
                    }
                ],
            },
            {
                "layout_descriptor": {"page_id": 0},
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "stale_desc",
                                        "text": "Needs more translation",
                                        "raw_text": "Needs more translation",
                                        "translated_text": "",
                                        "coverage_required": "strict",
                                        "translatable": True,
                                        "translation_strategy": "layout_constrained",
                                    }
                                ]
                            }
                        ],
                    }
                ],
            },
        ]
        translated_pages = [
            {
                "layout_descriptor": {"page_id": 0},
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "stale_desc",
                                        "text": "Needs more translation",
                                        "raw_text": "Needs more translation",
                                        "translated_text": "4",
                                        "coverage_required": "strict",
                                        "translatable": True,
                                        "translation_strategy": "layout_constrained",
                                    }
                                ]
                            }
                        ],
                    }
                ],
            },
            {
                "layout_descriptor": {"page_id": 0},
                "blocks": [
                    {
                        "role": "body",
                        "lines": [
                            {
                                "phrases": [
                                    {
                                        "unit_id": "stale_desc",
                                        "text": "Needs more translation",
                                        "raw_text": "Needs more translation",
                                        "translated_text": "4",
                                        "coverage_required": "strict",
                                        "translatable": True,
                                        "translation_strategy": "layout_constrained",
                                    }
                                ]
                            }
                        ],
                    }
                ],
            },
        ]
        report = analyze_document_coverage(source_pages, translated_pages)
        self.assertEqual(report["findings_total"], 2)
        self.assertEqual(report["findings_by_page"], {1: 1, 2: 1})


if __name__ == "__main__":
    unittest.main()
