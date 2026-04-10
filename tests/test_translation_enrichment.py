import unittest
import re
import os
import tempfile

import fitz
from PIL import Image

from block_typology import classify_block_typology
from page_policy_matrix import PagePolicyMatrix
from translator import DocumentTranslator
from reconstructor import DocumentReconstructor, BlockRenderOp, EditorialBlockRenderer
from reconstructor import PlacableUnit
from structure_extractor import LayoutV2Builder
from ocr_server import _is_immutable_inline_text, _is_equation_like_text


class TranslationEnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.translator = DocumentTranslator.__new__(DocumentTranslator)
        self.reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        self.reconstructor.pixel_to_point = 72.0 / 150.0
        self.layout_v2_builder = LayoutV2Builder.__new__(LayoutV2Builder)
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

    def test_toc_page_number_anchor_falls_back_to_right_gutter_when_detected_stop_is_too_left(self):
        tab_stops = {
            "page_num_right_x": 701.0,
            "column_left_x": 287.0,
            "column_right_x": 911.0,
        }

        resolved = self.reconstructor._resolve_toc_page_num_right_x(
            tab_stops,
            left=0.0,
            right=531.36,
        )

        self.assertGreater(resolved, 500.0)

    def test_select_richer_line_text_prefers_phrase_join_when_line_text_is_truncated(self):
        richer = self.reconstructor._select_richer_line_text(
            "same relu",
            "x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)",
        )

        self.assertEqual(richer, "x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu')(x)")

    def test_locked_tabular_row_detection_avoids_multiline_editorial_merge(self):
        line_entries = [
            {"bbox": [246, 403, 281, 419]},
            {"bbox": [361, 403, 411, 419]},
            {"bbox": [449, 403, 555, 419]},
            {"bbox": [586, 403, 635, 419]},
        ]

        self.assertTrue(self.reconstructor._line_entries_look_like_locked_tabular_row(line_entries))
        self.assertFalse(
            self.reconstructor._should_keep_multiline_locked_editorial_block(
                page_data={"layout_type": "table_dominant", "document_type": "form"},
                block={"role": "body", "render_policy": "anchored_text"},
                descriptor_layout_behavior="locked_in_cell",
                descriptor_structural_role="table_value_cell",
                descriptor_typographic_class="editorial_body",
                line_entries=line_entries,
                source="native",
                translated_block=True,
            )
        )

    def test_table_band_prefers_text_erased_overlay_instead_of_source_crop_restore(self):
        item = {
            "descriptor_band_role": "table_band",
            "descriptor_region_type": "table_row",
            "descriptor_ai_region_type": "table",
            "descriptor_visual_text": {},
        }

        self.assertTrue(self.reconstructor._prefer_text_erased_overlay(item))

    def test_translator_detects_immutable_programming_code_block(self):
        block = {
            "role": "body",
            "unit_type": "narrative_body",
            "lines": [
                {
                    "unit_type": "code_visible",
                    "line_text": "x = inception_module(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128,",
                    "phrases": [
                        {
                            "unit_type": "code_visible",
                            "texte": "x = inception_module(x, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128,",
                            "style": {"font": "Courier", "flags": {"monospace": True}},
                        }
                    ],
                },
                {
                    "unit_type": "code_visible",
                    "line_text": "filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32,",
                    "phrases": [
                        {
                            "unit_type": "code_visible",
                            "texte": "filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32,",
                            "style": {"font": "Courier", "flags": {"monospace": True}},
                        }
                    ],
                },
                {
                    "unit_type": "code_visible",
                    "line_text": "name='inception_3a')",
                    "phrases": [
                        {
                            "unit_type": "code_visible",
                            "texte": "name='inception_3a')",
                            "style": {"font": "Courier", "flags": {"monospace": True}},
                        }
                    ],
                },
            ],
        }

        self.assertTrue(self.translator._block_has_immutable_programming_code(block))
        self.assertTrue(self.reconstructor._block_is_immutable_programming_code(block))

    def test_translator_detects_simple_mixed_heading_body_block(self):
        block = {
            "role": "body",
            "lines": [
                {
                    "line_text": "PART B: BUILDING THE INCEPTION MODULES AND MAX-POOLING LAYERS",
                    "phrases": [
                        {
                            "texte": "PART B: BUILDING THE INCEPTION MODULES AND MAX-POOLING LAYERS",
                            "style": {"font": "FranklinGothic-Demi", "size": 12.0, "color": "#2F5D7E", "flags": {"uppercase": True}},
                        }
                    ],
                },
                {
                    "line_text": "To build inception modules 3a and 3b and the first max-pooling layer, we use table 5.2",
                    "phrases": [
                        {
                            "texte": "To build inception modules 3a and 3b and the first max-pooling layer, we use table 5.2",
                            "style": {"font": "NewBaskerville-Roman", "size": 11.0, "color": "#000000", "flags": {"uppercase": False}},
                        }
                    ],
                },
                {
                    "line_text": "to start. The code is as follows:",
                    "phrases": [
                        {
                            "texte": "to start. The code is as follows:",
                            "style": {"font": "NewBaskerville-Roman", "size": 11.0, "color": "#000000", "flags": {"uppercase": False}},
                        }
                    ],
                },
            ],
        }

        self.assertTrue(self.translator._should_translate_simple_mixed_heading_body_block(block))

    def test_heading_like_line_translation_preserves_structural_marker(self):
        self.translator._translate_unit_text = lambda text, **kwargs: "construction des modules inception et des couches de max-pooling"
        self.translator._direct_ct2_translate_chunks = lambda text, target_lang="fr": "construction des modules inception et des couches de max-pooling"
        self.translator._apply_cnn_glossary_fr = lambda text: text
        self.translator._normalize_technical_terms_fr = lambda text: text
        self.translator._fix_english_residuals_in_fr = lambda text: text

        translated = self.translator._translate_heading_like_line_fr(
            "PART B: BUILDING THE INCEPTION MODULES AND MAX-POOLING LAYERS"
        )

        self.assertEqual(translated, "PARTIE B: CONSTRUCTION DES MODULES INCEPTION ET DES COUCHES DE MAX-POOLING")

    def test_heading_like_line_repairs_cnn_heading_terminology_in_french(self):
        self.translator._translate_unit_text = lambda text, **kwargs: "la création des modules de création et des couches de mise en commun max"
        self.translator._direct_ct2_translate_chunks = lambda text, target_lang="fr": "la création des modules de création et des couches de mise en commun max"
        self.translator._apply_cnn_glossary_fr = lambda text: text
        self.translator._normalize_technical_terms_fr = lambda text: text
        self.translator._fix_english_residuals_in_fr = lambda text: text

        translated = self.translator._translate_heading_like_line_fr(
            "PART B: BUILDING THE INCEPTION MODULES AND MAX-POOLING LAYERS"
        )

        self.assertEqual(translated, "PARTIE B: LA CONSTRUCTION DES MODULES INCEPTION ET DES COUCHES DE MAX-POOLING")

    def test_backfill_phrase_span_translations_does_not_dump_long_phrase_into_tiny_span(self):
        phrase = {
            "bbox": [40, 100, 320, 160],
            "translated_text": "GoogleNet a ete introduit pour traiter les donnees visuelles avec beaucoup moins de parametres.",
            "spans": [
                {
                    "texte": "GoogleNet",
                    "bbox": [118.34, 140.85, 130.52, 150.81],
                    "style": {"font": "Times", "size": 11},
                }
            ],
        }

        self.translator._backfill_phrase_span_translations(phrase, phrase["translated_text"])

        self.assertEqual(phrase["spans"][0].get("translated_text", ""), "")

    def test_emit_text_run_clips_rect_and_point_to_block_bbox(self):
        reconstructor = DocumentReconstructor()
        renderer = EditorialBlockRenderer(reconstructor)
        plan = type(
            "Plan",
            (),
            {
                "block_id": "b_clip",
                "block_bbox": (20.0, 20.0, 120.0, 60.0),
                "source_block": {"style": {"font": "helv", "size": 12.0, "color": "#000000"}},
            },
        )()

        op = renderer._emit_text_run(
            plan,
            "Texte long",
            fitz.Rect(10.0, 5.0, 140.0, 90.0),
            (10.0, 90.0),
            {"font": "helv", "size": 12.0, "color": "#000000"},
            "helv",
            None,
            True,
            12.0,
            (0.0, 0.0, 0.0),
            unit_id="u0",
        )

        self.assertEqual(op.bbox, (20.0, 48.0, 120.0, 60.0))
        self.assertEqual(op.metadata["point"], (20.0, 57.84))

    def test_aux_segments_hydrate_leaf_translations_generically(self):
        page = {
            "blocks": [
                {
                    "id": "b1",
                    "bbox": [0, 0, 200, 40],
                    "text": "■ Hidden layers 96",
                    "lines": [
                        {
                            "bbox": [0, 0, 130, 16],
                            "line_text": "■ Hidden layers",
                            "phrases": [
                                {
                                    "bbox": [0, 0, 130, 16],
                                    "text": "■ Hidden layers",
                                    "texte": "■ Hidden layers",
                                    "spans": [
                                        {"bbox": [0, 0, 10, 16], "text": "■", "texte": "■"},
                                        {"bbox": [12, 0, 110, 16], "text": "Hidden layers", "texte": "Hidden layers"},
                                    ],
                                }
                            ],
                        },
                        {
                            "bbox": [140, 0, 170, 16],
                            "line_text": "96",
                            "phrases": [
                                {
                                    "bbox": [140, 0, 170, 16],
                                    "text": "96",
                                    "texte": "96",
                                    "spans": [
                                        {"bbox": [140, 0, 170, 16], "text": "96", "texte": "96"},
                                    ],
                                }
                            ],
                        },
                    ],
                    "semantic_phrases": [
                        {"bbox": [0, 0, 130, 16], "text": "■ Hidden layers", "line_indices": [0]},
                        {"bbox": [140, 0, 170, 16], "text": "96", "line_indices": [1]},
                    ],
                    "semantic_spans": [
                        {"bbox": [12, 0, 110, 16], "text": "Hidden layers"},
                        {"bbox": [140, 0, 170, 16], "text": "96"},
                    ],
                }
            ],
            "toc": {
                "toc_rows": [
                    {
                        "label": "Hidden layers",
                        "translated_label": "Couches cachees",
                        "page": "96",
                        "translated_text": "Couches cachees 96",
                        "label_bbox": [12, 0, 110, 16],
                        "page_bbox": [140, 0, 170, 16],
                    }
                ]
            },
        }

        self.translator._enrich_leaf_translations_from_aux_segments(page)

        block = page["blocks"][0]
        line0 = block["lines"][0]
        phrase0 = line0["phrases"][0]
        span_bullet, span_label = phrase0["spans"]
        line1 = block["lines"][1]

        self.assertIsNone(span_bullet.get("translated_text"))
        self.assertEqual(span_label.get("translated_text"), "Couches cachees")
        self.assertEqual(phrase0.get("translated_text"), "■ Couches cachees")
        self.assertEqual(line0.get("translated_text"), "■ Couches cachees")
        self.assertEqual(line1.get("translated_text"), "96")
        self.assertEqual(block.get("translated_text"), "■ Couches cachees 96")
        self.assertEqual(block["semantic_phrases"][0].get("translated_text"), "■ Couches cachees")
        self.assertEqual(block["semantic_spans"][0].get("translated_text"), "Couches cachees")

    def test_anchored_text_block_prefers_nested_translated_spans_for_reconstruction(self):
        block = {
            "id": "b1",
            "role": "body",
            "render_policy": "anchored_text",
            "lines": [
                {
                    "phrases": [
                        {
                            "unit_id": "p1",
                            "spans": [
                                {
                                    "unit_id": "s1",
                                    "bbox": [10, 10, 20, 20],
                                    "texte": "■",
                                    "translated_text": "■",
                                    "style": {"size": 10},
                                },
                                {
                                    "unit_id": "s2",
                                    "bbox": [22, 10, 80, 20],
                                    "texte": "Hidden layers",
                                    "translated_text": "Couches cachees",
                                    "style": {"size": 10},
                                },
                            ],
                        }
                    ]
                }
            ],
            "semantic_phrases": [
                {
                    "unit_id": "ph1",
                    "bbox": [10, 10, 80, 20],
                    "text": "■ Hidden layers",
                    "translated_text": "■ Couches cachees",
                    "line_indices": [0],
                }
            ],
        }

        units = self.reconstructor._normalize_placable_units(
            block,
            {"semantic_phrases": block["semantic_phrases"], "semantic_groups": [], "semantic_runs": [], "semantic_spans": []},
            target_lang="fr",
            page_data={"blocks": [block]},
        )

        self.assertEqual([u.unit_id for u in units], ["s1", "s2"])
        self.assertEqual([u.text_translated for u in units], ["■", "Couches cachees"])

    def test_backfill_phrase_span_translations_ignores_bullet_span_for_lexical_translation(self):
        phrase = {
            "texte": "■ Hidden layers",
            "translated_text": "Couches cachees",
            "spans": [
                {
                    "texte": "■",
                    "style": {"font": "A", "color": "#000000", "flags": {"bold": False, "italic": False, "monospace": False}},
                },
                {
                    "texte": "Hidden layers",
                    "style": {"font": "B", "color": "#000000", "flags": {"bold": False, "italic": False, "monospace": False}},
                },
            ],
        }

        self.translator._backfill_phrase_span_translations(phrase, "Couches cachees")

        self.assertIsNone(phrase["spans"][0].get("translated_text"))
        self.assertEqual(phrase["spans"][1].get("translated_text"), "Couches cachees")

    def test_rebalance_block_line_translations_moves_heading_text_off_marker_line(self):
        block = {
            "lines": [
                {
                    "line_text": "4.1",
                    "translated_text": "4.1 Définition des paramètres de performance",
                    "phrases": [
                        {
                            "texte": "4.1",
                            "translated_text": "4.1 Définition des paramètres de performance",
                            "spans": [
                                {"texte": "4.1", "translated_text": "4.1 Définition des paramètres de performance"}
                            ],
                        }
                    ],
                },
                {
                    "line_text": "Defining performance metrics",
                    "translated_text": "Defining performance metrics",
                    "phrases": [
                        {
                            "texte": "Defining performance metrics",
                            "translated_text": "Defining performance metrics",
                            "spans": [
                                {"texte": "Defining performance metrics"}
                            ],
                        }
                    ],
                },
            ],
            "semantic_phrases": [],
        }

        self.translator._rebalance_block_line_translations(block)

        self.assertEqual(block["lines"][0]["translated_text"], "4.1")
        self.assertEqual(
            block["lines"][0]["phrases"][0]["spans"][0]["translated_text"],
            "4.1",
        )
        self.assertEqual(
            block["lines"][1]["translated_text"],
            "Définition des paramètres de performance",
        )
        self.assertEqual(
            block["lines"][1]["phrases"][0]["spans"][0]["translated_text"],
            "Définition des paramètres de performance",
        )

    def test_rebalance_block_line_translations_redistributes_multiline_phrase(self):
        block = {
            "lines": [
                {
                    "line_text": "■ Plotting the",
                    "translated_text": "■ Tracer les courbes d'apprentissage",
                    "leading_marker": "■",
                    "phrases": [
                        {
                            "texte": "■ Plotting the",
                            "translated_text": "■ Tracer les courbes d'apprentissage",
                            "spans": [
                                {"texte": "■"},
                                {"texte": "Plotting the", "translated_text": "Tracer les courbes d'apprentissage"},
                            ],
                        }
                    ],
                },
                {
                    "line_text": "learning curves",
                    "translated_text": "learning curves",
                    "phrases": [
                        {
                            "texte": "learning curves",
                            "translated_text": "learning curves",
                            "spans": [
                                {"texte": "learning curves"},
                            ],
                        }
                    ],
                },
            ],
            "semantic_phrases": [
                {
                    "text": "■ Plotting the learning curves",
                    "translated_text": "■ Tracer les courbes d'apprentissage learning curves",
                    "line_indices": [0, 1],
                }
            ],
        }

        self.translator._rebalance_block_line_translations(block)

        self.assertNotEqual(block["lines"][1]["translated_text"], "learning curves")
        combined = block["lines"][0]["translated_text"] + " " + block["lines"][1]["translated_text"]
        self.assertIn("Tracer", combined)
        self.assertIn("courbes", combined)
        self.assertIn("d'apprentissage", block["lines"][1]["phrases"][0]["spans"][0]["translated_text"])

    def test_rebalance_block_line_translations_redistributes_lexical_run_after_marker_split(self):
        block = {
            "lines": [
                {
                    "line_text": "4.4",
                    "translated_text": "4.4 Évaluer le modèle et interpréter sa performance",
                    "phrases": [
                        {
                            "texte": "4.4",
                            "translated_text": "4.4 Évaluer le modèle et interpréter sa performance",
                            "spans": [
                                {"texte": "4.4", "translated_text": "4.4 Évaluer le modèle et interpréter sa performance"}
                            ],
                        }
                    ],
                },
                {
                    "line_text": "Evaluating the model and interpreting its",
                    "translated_text": "Evaluating the model and interpreting its",
                    "phrases": [
                        {
                            "texte": "Evaluating the model and interpreting its",
                            "translated_text": "Evaluating the model and interpreting its",
                            "spans": [{"texte": "Evaluating the model and interpreting its"}],
                        }
                    ],
                },
                {
                    "line_text": "performance",
                    "translated_text": "performance",
                    "phrases": [
                        {
                            "texte": "performance",
                            "translated_text": "performance",
                            "spans": [{"texte": "performance"}],
                        }
                    ],
                },
            ],
            "semantic_phrases": [],
        }

        self.translator._rebalance_block_line_translations(block)

        self.assertEqual(block["lines"][0]["translated_text"], "4.4")
        combined = block["lines"][1]["translated_text"] + " " + block["lines"][2]["translated_text"]
        self.assertIn("Évaluer", combined)
        self.assertIn("performance", combined)
        self.assertNotEqual(
            block["lines"][1]["translated_text"],
            "Evaluating the model and interpreting its",
        )

    def test_line_text_for_translation_prefers_phrase_text_when_line_text_is_degraded(self):
        line = {
            "line_text": "P ART B: B UILDING THE INCEPTION MODULES",
            "phrases": [
                {
                    "texte": "PART B: BUILDING THE INCEPTION MODULES",
                }
            ],
        }

        self.assertEqual(
            self.translator._line_text_for_translation(line),
            "PART B: BUILDING THE INCEPTION MODULES",
        )

    def test_shared_source_block_background_groups_multi_item_native_translation_block(self):
        items = [
            {
                "source_block_id": "n_0",
                "source": "native",
                "translated_block": True,
                "bbox": fitz.Rect(10, 10, 40, 20),
                "source_text": "CHAPTER 5",
                "text": "CHAPITRE 5",
                "descriptor_band_role": "header_band",
                "descriptor_region_type": "header",
            },
            {
                "source_block_id": "n_0",
                "source": "native",
                "translated_block": True,
                "bbox": fitz.Rect(42, 10, 110, 20),
                "source_text": "Advanced CNN architectures",
                "text": "Architectures CNN avancees",
                "descriptor_band_role": "header_band",
                "descriptor_region_type": "header",
            },
        ]

        groups = self.reconstructor._group_shared_source_block_background_items(items)

        self.assertIn("n_0", groups)
        self.assertEqual(len(groups["n_0"]["items"]), 2)
        self.assertAlmostEqual(groups["n_0"]["bbox"].x0, 10.0)
        self.assertAlmostEqual(groups["n_0"]["bbox"].x1, 110.0)

    def test_shared_source_block_background_groups_skip_items_with_hidden_locked_content(self):
        items = [
            {
                "source_block_id": "n_hidden",
                "source": "native",
                "translated_block": True,
                "contains_background_only_content": True,
                "bbox": fitz.Rect(10, 10, 60, 20),
                "source_text": "Visible intro",
                "text": "Intro visible",
            },
            {
                "source_block_id": "n_hidden",
                "source": "native",
                "translated_block": True,
                "bbox": fitz.Rect(10, 24, 90, 34),
                "source_text": "Visible outro",
                "text": "Outro visible",
            },
        ]

        groups = self.reconstructor._group_shared_source_block_background_items(items)

        self.assertNotIn("n_hidden", groups)

    def test_shared_background_preparation_disables_followup_item_cleanup(self):
        item = {
            "_shared_bg_prepared": True,
            "source": "native",
            "translated_block": True,
            "source_text": "CHAPTER 5",
            "text": "CHAPITRE 5",
            "descriptor_band_role": "header_band",
            "descriptor_region_type": "header",
            "role": "header",
        }

        self.assertFalse(self.reconstructor._should_restore_background_before_render(item))
        self.assertFalse(self.reconstructor._should_whiteout_before_render(item))
        self.assertFalse(self.reconstructor._should_whiteout_per_line(item))

    def test_shared_source_block_white_background_can_use_group_whiteout(self):
        items = [
            {
                "source_block_id": "n_1",
                "source": "native",
                "translated_block": True,
                "bbox": fitz.Rect(10, 10, 60, 20),
                "source_text": "line 1",
                "text": "ligne 1",
                "descriptor_band_role": "table_band",
                "descriptor_region_type": "table_row",
                "style": {},
            },
            {
                "source_block_id": "n_1",
                "source": "native",
                "translated_block": True,
                "bbox": fitz.Rect(10, 24, 80, 34),
                "source_text": "line 2",
                "text": "ligne 2",
                "descriptor_band_role": "table_band",
                "descriptor_region_type": "table_row",
                "style": {},
            },
        ]

        self.assertFalse(any(self.reconstructor._item_has_nonwhite_background(item) for item in items))

    def test_toc_chapter_title_does_not_reuse_previous_chapter_number(self):
        rows = [
            {"role": "section_heading", "label": "5.6 Existing section"},
            {"role": "chapter_title", "label": "New chapter title"},
            {"role": "chapter_title", "label": "Another chapter title"},
            {"role": "section_heading", "label": "6.1 First section of next chapter"},
        ]

        self.layout_v2_builder._annotate_toc_chapter_numbers(rows)

        self.assertNotIn("chapter_number", rows[1])
        self.assertEqual(rows[2].get("chapter_number"), "6")

    def test_toc_short_label_translation_preserves_numeric_prefix_and_technical_term(self):
        translated = self.translator._translate_toc_label_fr("5.4 AlexNet architecture", role="section_heading")

        self.assertEqual(translated, "5.4 Architecture d'AlexNet")

    def test_toc_translation_preserves_dataset_and_product_names(self):
        self.assertEqual(
            self.translator._translate_toc_label_fr("Fashion-MNIST", role="subentry_marker"),
            "Fashion-MNIST",
        )
        self.assertEqual(
            self.translator._translate_toc_label_fr("Google Open Images", role="subentry_marker"),
            "Google Open Images",
        )
        self.assertEqual(
            self.translator._translate_toc_label_fr("Kaggle", role="section_heading"),
            "Kaggle",
        )

    def test_toc_translation_repairs_source_aware_technical_terms(self):
        self.assertEqual(
            self.translator._postprocess_toc_label_fr("Novel features of Inception", "Nouvelles caractéristiques de l'accueil", role="section_heading"),
            "Nouvelles caractéristiques d'Inception",
        )
        self.assertEqual(
            self.translator._postprocess_toc_label_fr("Using a pretrained network as a feature extractor", "Utilisation d'un réseau pré-qualifié comme extracteur de fonctionnalités", role="section_heading"),
            "Utilisation d'un réseau préentraîné comme extracteur de caractéristiques",
        )
        self.assertEqual(
            self.translator._postprocess_toc_label_fr("Project 2: Fine-tuning", "Projet 2: Fin de réglage", role="section_heading"),
            "Projet 2: réglage fin",
        )

    def test_toc_translation_repairs_remaining_page_7_terms(self):
        self.assertEqual(
            self.translator._translate_toc_label_fr("Converting color images to grayscale to reduce computation complexity", role="subentry"),
            "Conversion des images couleur en niveaux de gris pour réduire la complexité de calcul",
        )
        self.assertEqual(
            self.translator._translate_toc_label_fr("What is a feature in computer vision?", role="subentry"),
            "Qu'est-ce qu'une caractéristique en vision par ordinateur ?",
        )
        self.assertEqual(
            self.translator._translate_toc_label_fr("Mini-batch gradient descent", role="subentry_marker"),
            "Descente de gradient par mini-lots",
        )
        self.assertEqual(
            self.translator._translate_toc_label_fr("What is backpropagation?", role="section_heading"),
            "Qu'est-ce que la rétropropagation ?",
        )

    def test_toc_translation_repairs_remaining_page_11_terms(self):
        self.assertEqual(
            self.translator._translate_toc_label_fr("High-level SSD architecture", role="section_heading"),
            "Architecture générale du SSD",
        )
        self.assertEqual(
            self.translator._translate_toc_label_fr("7.4 You only look once (YOLO)", role="section_heading"),
            "7.4 YOLO (You Only Look Once)",
        )
        self.assertEqual(
            self.translator._translate_toc_label_fr("10.1 Applications of visual embeddings", role="section_heading"),
            "10.1 Applications des embeddings visuels",
        )

    def test_toc_translation_generalizes_pattern_based_model_labels(self):
        self.assertEqual(
            self.translator._translate_toc_label_fr("Novel features of MobileNet", role="section_heading"),
            "Nouvelles caractéristiques de MobileNet",
        )
        self.assertEqual(
            self.translator._translate_toc_label_fr("Architecture of YOLOv4", role="section_heading"),
            "Architecture de YOLOv4",
        )
        self.assertEqual(
            self.translator._translate_toc_label_fr("How YOLOv4 works", role="section_heading"),
            "Fonctionnement de YOLOv4",
        )

    def test_toc_translation_generalizes_safe_technical_concepts(self):
        self.assertEqual(
            self.translator._translate_toc_label_fr("What is a perceptron?", role="subentry"),
            "Qu'est-ce qu'un perceptron ?",
        )
        self.assertEqual(
            self.translator._translate_toc_label_fr("What is the error function?", role="subentry"),
            "Qu'est-ce que la fonction d'erreur ?",
        )
        self.assertEqual(
            self.translator._translate_toc_label_fr("What is optimization?", role="subentry"),
            "Qu'est-ce que l'optimisation ?",
        )
        self.assertEqual(
            self.translator._translate_toc_label_fr("Multilayer perceptron architecture", role="subentry"),
            "Architecture du perceptron multicouche",
        )
        self.assertEqual(
            self.translator._translate_toc_label_fr("How the DeepDream algorithm works", role="section_heading"),
            "Fonctionnement de l'algorithme DeepDream",
        )

    def test_backfills_translated_text_on_mixed_style_spans(self):
        phrase = {
            "texte": "PRACTICAL SQL. Copyright © 2018 by Anthony DeBarros.",
            "translated_text": "SQL PRATIQUE. Copyright © 2018 par Anthony DeBarros.",
            "spans": [
                {
                    "texte": "PRACTICAL SQL.",
                    "style": {"font": "JansonTextLTStd-Bold", "flags": {"bold": True}, "color": "#000000"},
                },
                {
                    "texte": "Copyright © 2018 by Anthony DeBarros.",
                    "style": {"font": "JansonTextLTStd-Roman", "flags": {"bold": False}, "color": "#000000"},
                },
            ],
        }

        self.translator._backfill_phrase_span_translations(phrase)

        self.assertEqual(phrase["spans"][0]["translated_text"], "SQL PRATIQUE.")
        self.assertEqual(
            phrase["spans"][1]["translated_text"],
            "Copyright © 2018 par Anthony DeBarros.",
        )

    def test_extract_block_slot_items_prefers_translated_inline_segments(self):
        reconstructor = DocumentReconstructor()
        page_data = {
            "dimensions": {"width": 400, "height": 300},
            "language": "fr",
            "page_role": "body",
            "layout_type": "table_dominant",
            "document_type": "form",
            "page_family": "form_page",
            "blocks": [
                {
                    "id": "b1",
                    "bbox": [40, 40, 300, 64],
                    "role": "title",
                    "source": "native",
                    "translated_text": "SQL PRATIQUE. Copyright © 2018 par Anthony DeBarros.",
                    "lines": [
                        {
                            "bbox": [40, 40, 300, 64],
                            "line_text": "PRACTICAL SQL. Copyright © 2018 by Anthony DeBarros.",
                            "translated_text": "SQL PRATIQUE. Copyright © 2018 par Anthony DeBarros.",
                            "phrases": [
                                {
                                    "bbox": [40, 40, 300, 64],
                                    "texte": "PRACTICAL SQL. Copyright © 2018 by Anthony DeBarros.",
                                    "translated_text": "SQL PRATIQUE. Copyright © 2018 par Anthony DeBarros.",
                                    "spans": [
                                        {
                                            "bbox": [40, 40, 120, 64],
                                            "texte": "PRACTICAL SQL.",
                                            "translated_text": "SQL PRATIQUE.",
                                            "style": {"font": "JansonTextLTStd-Bold", "size": 11.25, "color": "#000000", "flags": {"bold": True}},
                                        },
                                        {
                                            "bbox": [122, 40, 300, 64],
                                            "texte": "Copyright © 2018 by Anthony DeBarros.",
                                            "translated_text": "Copyright © 2018 par Anthony DeBarros.",
                                            "style": {"font": "JansonTextLTStd-Roman", "size": 11.25, "color": "#000000", "flags": {"bold": False}},
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        items = reconstructor._extract_block_slot_items(page_data)
        title_item = next(item for item in items if item.get("role") == "title")
        self.assertEqual(
            [seg.get("text") for seg in (title_item.get("inline_style_segments") or [])],
            ["SQL PRATIQUE.", "Copyright © 2018 par Anthony DeBarros."],
        )

    def test_toc_part_title_translation_does_not_duplicate_part_number(self):
        translated = self.translator._translate_toc_label_fr(
            "PART 2 IMAGE CLASSIFICATION AND DETECTION",
            role="part_title",
        )

        self.assertEqual(translated, "Partie classification d'images et détection")

    def test_part_title_single_digit_page_marker_is_dropped(self):
        rows = [
            {"role": "part_title", "label": "PART IMAGE CLASSIFICATION AND DETECTION", "page": "5"},
        ]

        self.layout_v2_builder._normalize_toc_rows(rows)

        self.assertEqual(rows[0].get("page"), "")
        self.assertEqual(rows[0].get("chapter_marker"), "5")

    def test_toc_normalization_merges_continuation_row_and_drops_empty_marker(self):
        rows = [
            {"role": "subentry_marker", "label": "Plotting the", "page": ""},
            {"role": "subentry", "label": "learning curves", "page": "158", "page_bbox": [1, 2, 3, 4]},
            {"role": "subentry_marker", "label": "", "page": ""},
        ]

        self.layout_v2_builder._normalize_toc_rows(rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get("label"), "Plotting the learning curves")
        self.assertEqual(rows[0].get("page"), "158")
        self.assertEqual(rows[0].get("role"), "subentry")

    def test_fontsize_from_bbox_uses_source_height(self):
        fs = self.reconstructor._fontsize_from_bbox([0, 100, 10, 175])

        self.assertEqual(fs, 36.0)

    def test_page_aux_segments_capture_label_and_page_with_source_text(self):
        page_data = {
            "toc": {
                "toc_rows": [
                    {
                        "label": "Convolutional neural networks",
                        "translated_label": "Réseaux de neurones convolutionnels",
                        "label_bbox": [287, 111, 672, 138],
                        "style": {"font": "FranklinGothic-Demi", "size": 8.0, "color": "#000000", "flags": {}},
                        "page": "92",
                        "page_bbox": [969, 111, 988, 138],
                        "page_style": {"font": "NewBaskerville-Bold", "size": 9.0, "color": "#000000", "flags": {}},
                    }
                ]
            }
        }

        segments = self.reconstructor._page_aux_translated_segments(page_data)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["segment_type"], "label")
        self.assertEqual(segments[0]["source_text"], "Convolutional neural networks")
        self.assertEqual(segments[1]["segment_type"], "page")
        self.assertEqual(segments[1]["source_text"], "92")

    def test_external_units_for_block_builds_positioned_units_from_page_segments(self):
        block = {
            "id": "n_1",
            "role": "header",
            "bbox": [287, 111, 988, 138],
            "style": {"font": "FranklinGothic-Demi", "size": 8.0, "color": "#000000", "flags": {}},
            "lines": [
                {"bbox": [287, 111, 672, 138], "line_text": "Convolutional neural networks"},
                {"bbox": [969, 111, 988, 138], "line_text": "92"},
            ],
        }
        page_data = {
            "blocks": [block],
            "toc": {
                "toc_rows": [
                    {
                        "label": "Convolutional neural networks",
                        "translated_label": "Réseaux de neurones convolutionnels",
                        "label_bbox": [287, 111, 672, 138],
                        "style": {"font": "FranklinGothic-Demi", "size": 8.0, "color": "#000000", "flags": {}},
                        "page": "92",
                        "page_bbox": [969, 111, 988, 138],
                        "page_style": {"font": "NewBaskerville-Bold", "size": 9.0, "color": "#000000", "flags": {}},
                    }
                ]
            },
        }

        units = self.reconstructor._external_units_for_block(block, page_data, "fr")

        self.assertEqual(len(units), 2)
        self.assertEqual(units[0].text_translated, "Réseaux de neurones convolutionnels")
        self.assertEqual(units[0].render_policy, "external_flow")
        self.assertEqual(units[1].text_translated, "92")
        self.assertEqual(units[1].anchor_horizontal, "end")

    def test_editorial_renderer_uses_bbox_anchored_mode_for_external_units(self):
        reconstructor = DocumentReconstructor()
        page_doc = fitz.open()
        try:
            page = page_doc.new_page(width=400, height=200)
            plan = reconstructor._build_block_reconstruction_plan(
                page,
                {
                    "blocks": [
                        {
                            "id": "n_1",
                            "role": "header",
                            "bbox": [40, 20, 300, 60],
                            "style": {"font": "helv", "size": 10.0, "color": "#000000", "flags": {}},
                            "lines": [
                                {"bbox": [40, 20, 220, 40], "line_text": "Example heading"},
                                {"bbox": [260, 20, 300, 40], "line_text": "12"},
                            ],
                        }
                    ],
                    "toc": {
                        "toc_rows": [
                            {
                                "label": "Example heading",
                                "translated_label": "Exemple de titre",
                                "label_bbox": [40, 20, 220, 40],
                                "style": {"font": "helv", "size": 10.0, "color": "#000000", "flags": {}},
                                "page": "12",
                                "page_bbox": [260, 20, 300, 40],
                                "page_style": {"font": "helv", "size": 10.0, "color": "#000000", "flags": {}},
                            }
                        ]
                    },
                },
                {
                    "id": "n_1",
                    "role": "header",
                    "bbox": [40, 20, 300, 60],
                    "style": {"font": "helv", "size": 10.0, "color": "#000000", "flags": {}},
                    "lines": [
                        {"bbox": [40, 20, 220, 40], "line_text": "Example heading"},
                        {"bbox": [260, 20, 300, 40], "line_text": "12"},
                    ],
                },
                "fr",
            )
            renderer = reconstructor._select_block_renderer(plan)

            self.assertFalse(renderer._should_render_bbox_anchored(plan))
            ops = renderer.render(page, plan)
            text_ops = [op for op in ops if op.op_type == "draw_text_run"]
            self.assertEqual(len(text_ops), 2)
        finally:
            page_doc.close()

    def test_external_units_ignore_segments_originating_from_blocks(self):
        block = {
            "id": "n_1",
            "role": "body",
            "bbox": [40, 20, 300, 80],
            "style": {"font": "helv", "size": 10.0, "color": "#000000", "flags": {}},
            "lines": [{"bbox": [40, 20, 300, 40], "line_text": "Example"}],
        }
        page_data = {
            "blocks": [block],
            "__aux_translated_segments": [
                {"unit_id": "root.blocks[0]:translated_text", "text": "Exemple", "source_text": "Example", "bbox": (40.0, 20.0, 120.0, 40.0), "style": {}, "segment_type": "translated_text"},
                {"unit_id": "root.external[0]:translated_label", "text": "Exemple externe", "source_text": "External example", "bbox": (30.0, 22.0, 90.0, 34.0), "style": {}, "segment_type": "label"},
            ],
        }

        units = self.reconstructor._external_units_for_block(block, page_data, "fr")

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].text_translated, "Exemple externe")

    def test_external_units_are_grouped_by_local_rows_inside_block(self):
        block = {
            "id": "n_rows",
            "role": "body",
            "bbox": [40, 20, 320, 140],
            "style": {"font": "helv", "size": 10.0, "color": "#000000", "flags": {}},
            "lines": [
                {"bbox": [40, 20, 320, 40], "line_text": "row 0"},
                {"bbox": [40, 44, 320, 64], "line_text": "row 1"},
            ],
        }
        page_data = {
            "__aux_translated_segments": [
                {"unit_id": "root.external:r0:label", "text": "Premiere ligne", "source_text": "First row", "bbox": [50, 24, 170, 38], "style": {}, "segment_type": "label"},
                {"unit_id": "root.external:r0:page", "text": "10", "source_text": "10", "bbox": [250, 24, 275, 38], "style": {}, "segment_type": "page"},
                {"unit_id": "root.external:r1:label", "text": "Deuxieme ligne", "source_text": "Second row", "bbox": [50, 48, 180, 62], "style": {}, "segment_type": "label"},
                {"unit_id": "root.external:r1:page", "text": "11", "source_text": "11", "bbox": [250, 48, 275, 62], "style": {}, "segment_type": "page"},
            ],
        }

        units = self.reconstructor._external_units_for_block(block, page_data, "fr")

        self.assertGreaterEqual(len(units), 2)
        self.assertTrue(all(u.unit_type.startswith("external_") for u in units[:2]))
        self.assertEqual(units[0].render_policy, "external_flow")
        self.assertEqual(units[1].render_policy, "external_flow")

    def test_canonicalize_block_units_splits_external_flow_islands(self):
        block = {"id": "b_canon", "role": "body"}
        units = [
            PlacableUnit(
                unit_id="u0",
                unit_type="external_label",
                source_kind="page_external_segment",
                parent_unit_id=None,
                block_unit_id="b_canon",
                phrase_unit_id="p0",
                line_indices=[0],
                text_source="A",
                text_translated="Alpha",
                role="body",
                style={},
                relative_bbox=(20.0, 20.0, 80.0, 30.0),
                render_policy="external_flow",
                metadata={"segment_type": "label"},
            ),
            PlacableUnit(
                unit_id="u1",
                unit_type="external_page",
                source_kind="page_external_segment",
                parent_unit_id=None,
                block_unit_id="b_canon",
                phrase_unit_id="p0",
                line_indices=[0],
                text_source="1",
                text_translated="1",
                role="body",
                style={},
                relative_bbox=(90.0, 20.0, 105.0, 30.0),
                render_policy="external_flow",
                metadata={"segment_type": "page"},
            ),
            PlacableUnit(
                unit_id="u2",
                unit_type="external_label",
                source_kind="page_external_segment",
                parent_unit_id=None,
                block_unit_id="b_canon",
                phrase_unit_id="p1",
                line_indices=[0],
                text_source="B",
                text_translated="Beta",
                role="body",
                style={},
                relative_bbox=(180.0, 20.0, 240.0, 30.0),
                render_policy="external_flow",
                metadata={"segment_type": "label"},
            ),
            PlacableUnit(
                unit_id="u3",
                unit_type="external_page",
                source_kind="page_external_segment",
                parent_unit_id=None,
                block_unit_id="b_canon",
                phrase_unit_id="p1",
                line_indices=[0],
                text_source="2",
                text_translated="2",
                role="body",
                style={},
                relative_bbox=(250.0, 20.0, 265.0, 30.0),
                render_policy="external_flow",
                metadata={"segment_type": "page"},
            ),
        ]

        normalized = self.reconstructor._canonicalize_block_units(block, units)

        self.assertEqual([u.line_indices for u in normalized], [[0], [0], [1], [1]])
        self.assertTrue(normalized[0].hard_break_before)
        self.assertFalse(normalized[1].hard_break_before)
        self.assertTrue(normalized[2].hard_break_before)

    def test_presence_fallback_accepts_point_space_bbox_entries(self):
        reconstructor = DocumentReconstructor()
        page_doc = fitz.open()
        try:
            page = page_doc.new_page(width=400, height=200)
            block = {
                "id": "n_1",
                "role": "header",
                "bbox": [40, 20, 300, 60],
                "style": {"font": "helv", "size": 10.0, "color": "#000000", "flags": {}},
            }
            page_data = {
                "blocks": [block],
                "__aux_translated_segments": [
                    {"unit_id": "root.external:label", "text": "Exemple", "source_text": "Example", "bbox": (19.2, 9.6, 96.0, 19.2), "style": {"font": "helv", "size": 10.0, "color": "#000000", "flags": {}}, "segment_type": "label"},
                ],
            }

            ops = reconstructor._render_block_presence_fallback_ops(page, page_data, block, "fr")
            text_ops = [op for op in ops if op.op_type == "draw_text_run"]

            self.assertEqual(len(text_ops), 1)
            self.assertEqual(text_ops[0].text, "Exemple")
        finally:
            page_doc.close()

    def test_dedupe_semantic_phrases_keeps_longest_nested_phrase_per_start_line(self):
        phrases = [
            {"translated_text": "Dropout means mixing up", "line_indices": [0], "bbox": [10, 10, 80, 20]},
            {"translated_text": "Dropout means mixing up our workout a little", "line_indices": [0, 1], "bbox": [10, 10, 160, 32]},
            {"translated_text": "Then we tie the left arm", "line_indices": [1], "bbox": [10, 34, 90, 44]},
        ]

        deduped = self.reconstructor._dedupe_semantic_phrases(phrases)

        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0]["translated_text"], "Dropout means mixing up our workout a little")

    def test_internal_translated_payload_does_not_import_external_units(self):
        block = {
            "id": "n_1",
            "role": "header",
            "bbox": [40, 20, 300, 60],
            "translated_text": "Exemple interne",
            "lines": [{"bbox": [40, 20, 200, 40], "translated_text": "Exemple interne"}],
        }
        semantic_payload = {"semantic_phrases": [], "semantic_groups": [], "semantic_runs": [], "semantic_spans": []}
        page_data = {
            "__aux_translated_segments": [
                {"unit_id": "root.external[0]:translated_label", "text": "Exemple externe", "source_text": "External example", "bbox": (40.0, 20.0, 120.0, 34.0), "style": {}, "segment_type": "label"},
            ]
        }

        units = self.reconstructor._normalize_placable_units(block, semantic_payload, "fr", page_data=page_data)

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].text_translated, "Exemple interne")

    def test_part_title_render_label_strips_partie_prefix(self):
        rendered = self.reconstructor._format_toc_label_for_render(
            "part_title",
            "Partie classification et détection d'images",
        )

        self.assertEqual(rendered, "CLASSIFICATION ET DÉTECTION D'IMAGES")

    def test_extract_leading_marker_accepts_marker_without_space(self):
        marker, text = self.layout_v2_builder._extract_leading_marker("■LeNet-5 implementation in Keras")

        self.assertEqual(marker, "■")
        self.assertEqual(text, "LeNet-5 implementation in Keras")

    def test_code_visible_sentence_can_be_downgraded_from_exact_preserve(self):
        contract = self.translator._resolve_translation_contract(
            {
                "unit_type": "code_visible",
                "translation_strategy": "exact_preserve",
                "texte": "Prints the new_model summary",
            }
        )

        self.assertEqual(contract.get("strategy"), "layout_constrained")
        self.assertTrue(contract.get("translatable"))

    def test_code_visible_block_without_direct_text_uses_line_content_for_contract(self):
        contract = self.translator._resolve_translation_contract(
            {
                "unit_type": "code_visible",
                "translation_strategy": "exact_preserve",
                "lines": [
                    {
                        "unit_type": "code_visible",
                        "text": "Saves the output of base_model",
                    },
                    {
                        "unit_type": "narrative_body",
                        "text": "to be the input of the next layer",
                    },
                ],
            }
        )

        self.assertEqual(contract.get("strategy"), "layout_constrained")
        self.assertTrue(contract.get("translatable"))

    def test_abbreviation_key_contract_is_exact_preserve(self):
        contract = self.translator._resolve_translation_contract(
            {
                "translation_strategy": "layout_constrained",
                "texte": "LRN",
                "structure_hints": {"structural_role_hint": "abbreviation_key"},
            },
            context={"block_role": "body", "role": "body"},
        )

        self.assertEqual(contract.get("strategy"), "exact_preserve")
        self.assertFalse(contract.get("translatable"))

    def test_abbreviation_value_contract_is_layout_constrained(self):
        contract = self.translator._resolve_translation_contract(
            {
                "translation_strategy": "exact_preserve",
                "texte": "Local Response Normalization",
                "structure_hints": {"structural_role_hint": "abbreviation_value"},
            },
            context={"block_role": "body", "role": "body"},
        )

        self.assertEqual(contract.get("strategy"), "layout_constrained")
        self.assertTrue(contract.get("translatable"))

    def test_abbreviation_page_detection_and_key_text_heuristics(self):
        structure = {
            "blocks": [
                {"role": "header", "texte": "Abbreviations"},
                {"role": "body", "texte": "LRN"},
                {"role": "body", "texte": "Local response normalization"},
                {"role": "body", "texte": "MSE"},
                {"role": "body", "texte": "Mean squared error"},
                {"role": "body", "texte": "CNN"},
                {"role": "body", "texte": "Convolutional neural network"},
                {"role": "body", "texte": "RBF"},
                {"role": "body", "texte": "Radial basis function"},
            ]
        }

        self.assertTrue(self.translator._looks_like_abbreviation_page(structure))
        self.assertTrue(self.translator._looks_like_abbreviation_key_text("LRN"))
        self.assertTrue(self.translator._looks_like_abbreviation_key_text("M-DBNs"))
        self.assertFalse(self.translator._looks_like_abbreviation_key_text("Local response normalization"))

    def test_paragraph_line_redistribution_avoids_pathological_one_word_lines(self):
        source_lines = [
            "Spatial analysis helps journalists compare places and reveal patterns",
            "across neighborhoods, cities, and regions in a rigorous way",
            "that can be explained clearly to readers with maps and tables",
            "when the story needs both narrative and quantitative evidence",
        ]

        redistributed = self.translator._redistribute_translated_to_lines(
            "L'analyse spatiale aide les journalistes a comparer des lieux et a reveler des motifs entre des quartiers des villes et des regions avec une methode rigoureuse qui reste claire pour les lecteurs grace aux cartes et aux tableaux lorsque le recit exige a la fois une narration et des preuves quantitatives.",
            source_lines,
            ["", "", "", ""],
        )

        word_counts = [
            len(self.translator._normalize_spaces(line).split())
            for line in redistributed[:-1]
        ]
        self.assertTrue(all(count >= 2 for count in word_counts))

    def test_paragraph_line_redistribution_preserves_leading_marker(self):
        redistributed = self.translator._redistribute_translated_to_lines(
            "premier point plus detaille pour la traduction du paragraphe",
            ["1. First bullet line", "continuation line"],
            ["1.", ""],
        )

        self.assertTrue(redistributed[0].startswith("1. "))

    def test_editorial_exact_preserve_is_relaxed_on_double_column_prose(self):
        contract = self.translator._resolve_translation_contract(
            {
                "translation_strategy": "exact_preserve",
                "unit_type": "citation",
                "texte": "in U.S. library use based on annual surveys.",
            },
            default_strategy="semantic_reflow",
            default_translatable=True,
            context={
                "layout_type": "double_column",
                "document_type": "book_page",
                "page_family": "body_text_two_column_equations",
                "block_role": "body",
                "role": "body",
            },
        )

        self.assertEqual(contract.get("strategy"), "layout_constrained")
        self.assertTrue(contract.get("translatable"))

    def test_true_bibliographic_exact_preserve_stays_locked(self):
        contract = self.translator._resolve_translation_contract(
            {
                "translation_strategy": "exact_preserve",
                "unit_type": "citation",
                "texte": "(Smith, 2020)",
            },
            default_strategy="semantic_reflow",
            default_translatable=True,
            context={
                "layout_type": "double_column",
                "document_type": "book_page",
                "page_family": "body_text_two_column_equations",
                "block_role": "body",
                "role": "body",
            },
        )

        self.assertEqual(contract.get("strategy"), "exact_preserve")

    def test_contents_like_block_is_not_treated_as_paragraph(self):
        block = {
            "role": "body",
            "lines": [
                {"line_text": "USING POSTGRESQL FROM THE COMMAND LINE", "indent_px": 0.0},
                {"line_text": "Setting Up the Command Line for psql", "indent_px": 0.0},
                {"line_text": "Windows psql Setup", "indent_px": 121.0},
                {"line_text": "macOS psql Setup", "indent_px": 121.0},
                {"line_text": "Linux psql Setup", "indent_px": 121.0},
                {"line_text": "Working with psql", "indent_px": 0.0},
                {"line_text": "Launching psql and Connecting to a Database", "indent_px": 121.0},
                {"line_text": "Getting Help", "indent_px": 121.0},
            ],
        }

        self.assertTrue(self.translator._looks_like_contents_block(block))
        self.assertFalse(self.translator._should_translate_block_as_paragraph(block))
        self.assertFalse(self.translator._looks_like_editorial_narrative_block(block))

    def test_code_visible_table_block_stays_exact_preserve(self):
        contract = self.translator._resolve_translation_contract(
            {
                "unit_type": "code_visible",
                "translation_strategy": "exact_preserve",
                "lines": [
                    {"unit_type": "code_visible", "text": "new_model.summary()"},
                    {"unit_type": "short_label", "text": "Layer (type) Output Shape Param #"},
                    {"unit_type": "code_visible", "text": "================================================================="},
                    {"unit_type": "code_visible", "text": "input_1 (InputLayer) (None, 224, 224, 3) 0"},
                    {"unit_type": "code_visible", "text": "_________________________________________________________________"},
                ],
            }
        )

        self.assertEqual(contract.get("strategy"), "exact_preserve")

    def test_short_locked_table_heading_code_visible_relaxes_by_type(self):
        contract = self.translator._resolve_translation_contract(
            {
                "role": "title",
                "unit_type": "code_visible",
                "translation_strategy": "exact_preserve",
                "semantic": {"type": "heading"},
                "structure_hints": {
                    "band_role_hint": "table_band",
                    "structural_role_hint": "table_stub_cell",
                    "layout_behavior_hint": "locked_in_cell",
                },
                "lines": [
                    {"unit_type": "short_label", "text": "Instantiates"},
                    {"unit_type": "code_visible", "text": "a feature_model"},
                    {"unit_type": "short_label", "text": "using Keras’s"},
                    {"unit_type": "short_label", "text": "Model class"},
                ],
            },
            context={"layout_type": "table_dominant", "page_family_group": "table_page", "block_role": "title"},
        )

        self.assertEqual(contract.get("strategy"), "layout_constrained")
        self.assertTrue(contract.get("translatable"))

    def test_dense_locked_table_code_visible_stays_preserved_by_type(self):
        contract = self.translator._resolve_translation_contract(
            {
                "role": "body",
                "unit_type": "code_visible",
                "translation_strategy": "exact_preserve",
                "semantic": {"type": "body"},
                "structure_hints": {
                    "band_role_hint": "table_band",
                    "structural_role_hint": "table_value_cell",
                    "layout_behavior_hint": "locked_in_cell",
                },
                "lines": [
                    {"unit_type": "code_visible", "text": "feature_model.summary()"},
                    {"unit_type": "short_label", "text": "Layer (type) Output Shape Param #"},
                    {"unit_type": "code_visible", "text": "================================================================="},
                    {"unit_type": "code_visible", "text": "dense_1 (Dense) (None, 128) 1024"},
                    {"unit_type": "code_visible", "text": "_________________________________________________________________"},
                ],
            },
            context={"layout_type": "table_dominant", "page_family_group": "table_page", "block_role": "body"},
        )

        self.assertEqual(contract.get("strategy"), "exact_preserve")

    def test_block_typology_classifies_editorial_locked_callout(self):
        profile = classify_block_typology(
            {
                "role": "title",
                "semantic": {"type": "heading"},
                "structure_hints": {
                    "band_role_hint": "table_band",
                    "structural_role_hint": "table_stub_cell",
                    "layout_behavior_hint": "locked_in_cell",
                },
                "lines": [{"text": "Instantiates"}, {"text": "using Keras’s"}],
            },
            context={"layout_type": "table_dominant", "page_family_group": "table_page"},
        )

        self.assertEqual(profile.get("subtype"), "editorial_locked_callout")

    def test_block_typology_classifies_locked_code_table(self):
        profile = classify_block_typology(
            {
                "role": "body",
                "semantic": {"type": "body"},
                "structure_hints": {
                    "band_role_hint": "table_band",
                    "structural_role_hint": "table_value_cell",
                    "layout_behavior_hint": "locked_in_cell",
                },
                "lines": [{}, {}, {}, {}, {}],
            },
            context={"layout_type": "table_dominant", "page_family_group": "table_page"},
        )

        self.assertEqual(profile.get("subtype"), "locked_code_table")

    def test_short_label_regex_translation_preserves_identifier(self):
        translated = self.translator._translate_short_label_fr("Prints the new_model summary")

        self.assertEqual(translated, "Affiche le résumé de new_model")

    def test_short_label_regex_translation_handles_new_model_fragment(self):
        translated = self.translator._translate_short_label_fr("a new_model")

        self.assertEqual(translated, "un nouveau modèle")

    def test_programming_code_line_is_preserved(self):
        self.assertTrue(
            self.translator._looks_like_programming_code_line(
                "x = Dense(10, activation='softmax', name='softmax')(last_output)"
            )
        )
        self.assertTrue(
            self.translator._looks_like_programming_code_line(
                "new_model = Model(inputs=base_model.input, outputs=x)"
            )
        )
        self.assertFalse(
            self.translator._looks_like_programming_code_line(
                "Instantiates a new_model using Keras’s Model class"
            )
        )
        self.assertFalse(self.translator._looks_like_programming_code_line("Prints the new_model summary"))

    def test_programming_code_line_ignores_editorial_parenthetical_prose(self):
        self.assertFalse(
            self.translator._looks_like_programming_code_line(
                "and the target domain is similar to the source domain (scenario 1). As explained in"
            )
        )

    def test_programming_code_line_ignores_url_sentences(self):
        self.assertFalse(
            self.translator._looks_like_programming_code_line(
                "Visit the book's website at www.manning.com/books/deep-learning-for-vision-systems"
            )
        )

    def test_editorial_body_with_urls_is_not_marked_as_immutable_code(self):
        block = {
            "role": "body",
            "source": "native",
            "unit_type": "narrative_body",
            "lines": [
                {
                    "line_text": "In this project, we use a very small amount of data to train a classifier.",
                    "phrases": [
                        {
                            "texte": "In this project, we use a very small amount of data to train a classifier.",
                            "style": {"font": "Times-Roman", "flags": {}},
                        }
                    ],
                },
                {
                    "line_text": "and the target domain is similar to the source domain (scenario 1). As explained in",
                    "phrases": [
                        {
                            "texte": "and the target domain is similar to the source domain (scenario 1). As explained in",
                            "style": {"font": "Times-Roman", "flags": {}},
                        }
                    ],
                },
                {
                    "line_text": "Visit the book's website at www.manning.com/books/deep-learning-for-vision-systems",
                    "phrases": [
                        {
                            "texte": "Visit the book's website at www.manning.com/books/deep-learning-for-vision-systems",
                            "style": {"font": "Times-Roman", "flags": {}},
                        }
                    ],
                },
            ],
        }

        self.assertFalse(self.translator._block_has_immutable_programming_code(block))

    def test_layout_postprocess_preserves_model_identifier_labels(self):
        translated = self.translator._apply_layout_constraint_postprocess(
            "Imprime le résumé du nouveau_modèle",
            "Prints the new_model summary",
            target_lang="fr",
            block_role="title",
        )
        self.assertEqual(translated, "Affiche le résumé de new_model")

    def test_toc_builder_uses_raw_phrase_text_for_part_title_page(self):
        page_data = {
            "blocks": [
                {
                    "lines": [
                        {
                            "bbox": [119, 800, 219, 834],
                            "line_text": "P ART",
                            "phrases": [{"texte": "PART 2", "text": "P ART", "style": {"size": 15.96, "flags": {"bold": True}}}],
                            "indent_level": 0,
                            "indent_px": 0.0,
                        },
                        {
                            "bbox": [252, 800, 969, 834],
                            "line_text": "I MAGE CLASSIFICATION AND DETECTION",
                            "phrases": [{"texte": "IMAGE CLASSIFICATION AND DETECTION...........193", "text": "I MAGE CLASSIFICATION AND DETECTION", "style": {"size": 15.96, "flags": {"bold": True}}}],
                            "indent_level": 0,
                            "indent_px": 0.0,
                        },
                        {
                            "bbox": [211, 862, 263, 937],
                            "line_text": "5",
                            "phrases": [{"texte": "5", "text": "5", "style": {"size": 36.0, "flags": {"italic": True}}}],
                            "indent_level": 0,
                            "indent_px": 0.0,
                        },
                    ]
                }
            ]
        }

        rows, _ = self.layout_v2_builder._build_toc_rows(
            page_data,
            columns=[{"x0": 119, "x1": 969}],
            margins={"left": 119, "right": 969},
        )

        self.assertEqual(rows[0].get("role"), "part_title")
        self.assertEqual(rows[0].get("page"), "193")
        self.assertIn("PART 2", rows[0].get("label"))

    def test_toc_builder_assigns_same_source_band_to_parallel_entries(self):
        page_data = {
            "blocks": [
                {
                    "lines": [
                        {
                            "bbox": [369, 976, 514, 997],
                            "line_text": "LeNet architecture",
                            "phrases": [{"texte": "LeNet architecture", "text": "LeNet architecture", "style": {"size": 9.96, "flags": {"italic": True}}}],
                            "indent_level": 0,
                            "indent_px": 117.0,
                        },
                        {
                            "bbox": [535, 976, 568, 997],
                            "line_text": "199",
                            "phrases": [{"texte": "199", "text": "199", "style": {"size": 9.96, "flags": {"italic": True}}}],
                            "indent_level": 0,
                            "indent_px": 117.0,
                        },
                        {
                            "bbox": [578, 976, 862, 997],
                            "line_text": "■LeNet-5 implementation in Keras",
                            "phrases": [{"texte": "■LeNet-5 implementation in Keras", "text": "■LeNet-5 implementation in Keras", "style": {"size": 9.96, "flags": {"italic": True}}}],
                            "indent_level": 2,
                            "indent_px": 326.0,
                        },
                        {
                            "bbox": [883, 976, 916, 997],
                            "line_text": "200",
                            "phrases": [{"texte": "200", "text": "200", "style": {"size": 9.96, "flags": {"italic": True}}}],
                            "indent_level": 2,
                            "indent_px": 326.0,
                        },
                    ]
                }
            ]
        }

        rows, _ = self.layout_v2_builder._build_toc_rows(
            page_data,
            columns=[{"x0": 369, "x1": 916}],
            margins={"left": 369, "right": 916},
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].get("source_band_id"), rows[1].get("source_band_id"))
        self.assertNotEqual(rows[0].get("source_band_lane"), rows[1].get("source_band_lane"))

    def test_toc_builder_flushes_ocr_only_inline_page_rows(self):
        page_data = {
            "blocks": [
                {
                    "lines": [
                        {
                            "bbox": [503, 56, 585, 70],
                            "line_text": "CONTENTS",
                            "phrases": [{"texte": "CONTENTS", "text": "CONTENTS", "style": {"size": 8.49, "flags": {"bold": False, "uppercase": True}}}],
                            "indent_level": 0,
                            "indent_px": 0.0,
                        },
                        {
                            "bbox": [290, 112, 919, 135],
                            "line_text": "4.5 Improving the network and tuning hyperparameters 162",
                            "phrases": [{"texte": "4.5 Improving the network and tuning hyperparameters 162", "text": "4.5 Improving the network and tuning hyperparameters 162", "style": {"size": 10.98, "flags": {"bold": True}}}],
                            "indent_level": 0,
                            "indent_px": 38.0,
                        },
                        {
                            "bbox": [370, 145, 815, 167],
                            "line_text": "Collecting more data vs. tuning hyperparameters 162",
                            "phrases": [{"texte": "Collecting more data vs. tuning hyperparameters 162", "text": "Collecting more data vs. tuning hyperparameters 162", "style": {"size": 9.96, "flags": {"italic": True}}}],
                            "indent_level": 1,
                            "indent_px": 118.0,
                        },
                    ]
                }
            ]
        }

        rows, _ = self.layout_v2_builder._build_toc_rows(
            page_data,
            columns=[{"x0": 252, "x1": 916}],
            margins={"left": 252, "right": 916},
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].get("role"), "toc_title")
        self.assertEqual(rows[1].get("page"), "162")
        self.assertEqual(rows[2].get("page"), "162")
        self.assertEqual(rows[1].get("label"), "4.5 Improving the network and tuning hyperparameters")

    def test_toc_inline_segment_split_handles_multiple_entries_in_one_row(self):
        segments, remainder = self.layout_v2_builder._extract_toc_inline_segments(
            "MNIST 263 - Fashion-MNIST 264 - CIFAR 264"
        )

        self.assertEqual(
            segments,
            [("MNIST", "263"), ("Fashion-MNIST", "264"), ("CIFAR", "264")],
        )
        self.assertEqual(remainder, "")

    def test_toc_compound_rows_are_split_across_row_continuations(self):
        rows = [
            {
                "y": 264.0,
                "indent_level": 3,
                "indent_px": 102.0,
                "marker": "",
                "label": "Learning rate and decay schedule 166 - A systematic approach",
                "page": "",
                "role": "subentry",
                "style": {"font": "Arial", "size": 19.0, "flags": {"bold": True, "italic": True}},
                "label_bbox": [370.0, 264.0, 900.0, 284.0],
                "page_bbox": None,
                "page_style": {},
                "source_band_id": 6,
                "source_band_lane": 0,
            },
            {
                "y": 289.0,
                "indent_level": 3,
                "indent_px": 103.0,
                "marker": "",
                "label": "to find the optimal learning rate 169 - Learning rate decay and",
                "page": "",
                "role": "subentry",
                "style": {"font": "Arial", "size": 19.0, "flags": {"bold": True, "italic": True}},
                "label_bbox": [371.0, 289.0, 908.0, 311.0],
                "page_bbox": None,
                "page_style": {},
                "source_band_id": 7,
                "source_band_lane": 0,
            },
            {
                "y": 314.0,
                "indent_level": 3,
                "indent_px": 102.0,
                "marker": "",
                "label": "adaptive learming 170 - Mini-batch size",
                "page": "171",
                "role": "subentry",
                "style": {"font": "Arial", "size": 19.0, "flags": {"bold": True, "italic": True}},
                "label_bbox": [370.0, 314.0, 769.0, 332.0],
                "page_bbox": [737.16, 314.0, 769.0, 332.0],
                "page_style": {"font": "Arial", "size": 19.0, "flags": {"bold": True, "italic": True}},
                "source_band_id": 8,
                "source_band_lane": 0,
            },
        ]

        split_rows = self.layout_v2_builder._split_compound_toc_rows(rows)

        self.assertEqual(
            [(row.get("label"), row.get("page")) for row in split_rows],
            [
                ("Learning rate and decay schedule", "166"),
                ("A systematic approach to find the optimal learning rate", "169"),
                ("Learning rate decay and adaptive learming", "170"),
                ("Mini-batch size", "171"),
            ],
        )

    def test_toc_role_inference_promotes_numbered_heading(self):
        page_data = {
            "blocks": [
                {
                    "lines": [
                        {
                            "bbox": [290, 112, 919, 135],
                            "line_text": "5.5 Inception and GoogLeNet 217",
                            "phrases": [
                                {
                                    "texte": "5.5 Inception and GoogLeNet 217",
                                    "text": "5.5 Inception and GoogLeNet 217",
                                    "style": {"size": 22.0, "flags": {"bold": True}},
                                }
                            ],
                            "indent_level": 2,
                            "indent_px": 22.0,
                        }
                    ]
                }
            ]
        }

        rows, _ = self.layout_v2_builder._build_toc_rows(
            page_data,
            columns=[{"x0": 268, "x1": 919}],
            margins={"left": 268, "right": 919},
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get("role"), "section_heading")
        self.assertEqual(rows[0].get("label"), "5.5 Inception and GoogLeNet")
        self.assertEqual(rows[0].get("page"), "217")

    def test_symbolic_equation_is_locked_as_overlay(self):
        self.assertTrue(
            self.reconstructor._should_lock_equation_overlay(
                {"role": "equation_inline"},
                rendered_text="dE / dw",
                source_text="dE / dw",
            )
        )
        self.assertFalse(
            self.reconstructor._should_lock_equation_overlay(
                {"role": "equation_inline"},
                rendered_text="Multi-scale feature layers",
                source_text="Couches de caractéristiques multi-échelles",
            )
        )

    def test_layout_v2_non_toc_page_does_not_trigger_toc_heuristic(self):
        reconstructor = DocumentReconstructor()
        page_data = {
            "schema_version": "layout.v2",
            "page_role": "body",
            "blocks": [
                {
                    "bbox": [0, 0, 100, 20],
                    "lines": [
                        {"line_text": "Power Rule 23"},
                        {"line_text": "Chain Rule 24"},
                        {"line_text": "Constant Rule 25"},
                        {"line_text": "Difference Rule 26"},
                        {"line_text": "Product Rule 27"},
                        {"line_text": "Quotient Rule 28"},
                    ],
                }
            ],
            "dimensions": {"height": 1000},
        }
        self.assertFalse(reconstructor._looks_like_toc_page(page_data))

    def test_table_band_item_restores_background_before_render(self):
        reconstructor = DocumentReconstructor()
        item = {
            "source": "native",
            "source_text": "Power Rule",
            "text": "Règle de puissance",
            "descriptor_region_type": "table_cell",
            "descriptor_band_role": "table_band",
        }
        self.assertTrue(reconstructor._should_restore_background_before_render(item))

    def test_highlighted_panel_item_restores_background_before_render(self):
        reconstructor = DocumentReconstructor()
        item = {
            "source": "native",
            "source_text": "Other hyperparameters",
            "text": "Plus sur les hyperparamètres",
            "style": {"highlight_color": "#efe7cf"},
        }
        self.assertTrue(reconstructor._should_restore_background_before_render(item))

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

    def test_two_column_equation_page_long_body_uses_paragraph_flow(self):
        policy = self.page_policy.classify_unit_policy(
            text="This longer editorial paragraph explains the context around the equation while remaining ordinary body prose for the reader across the two column page layout.",
            role="body",
            source_kind="native_phrase",
            page_role="body",
            page_family="body_text_two_column_equations",
            page_family_group="body_text",
            document_type="book_page",
            layout_type="double_column",
            style_profile="academic_dense",
        )
        self.assertEqual(policy["unit_type"], "narrative_body")
        self.assertEqual(policy["render_policy"], "paragraph_flow")

    def test_two_column_equation_page_formula_label_stays_anchored(self):
        policy = self.page_policy.classify_unit_policy(
            text="Gradient update rule",
            role="equation_inline",
            source_kind="native_phrase",
            page_role="body",
            page_family="body_text_two_column_equations",
            page_family_group="body_text",
            document_type="book_page",
            layout_type="double_column",
            style_profile="academic_dense",
        )
        self.assertEqual(policy["unit_type"], "formula_label")
        self.assertEqual(policy["render_policy"], "anchored_text")

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

    def test_translated_figure_caption_stays_renderable_when_page_family_is_unknown(self):
        reconstructor = DocumentReconstructor()
        page_data = {
            "dimensions": {"width": 400, "height": 600},
            "layout_type": "double_column",
            "document_type": "web_print",
            "page_family": "unknown",
            "immutable_overlays": [],
            "images": [],
            "non_text_zones": [],
            "layout": {},
            "blocks": [
                {
                    "id": "cap1",
                    "role": "figure_caption",
                    "source": "native",
                    "bbox": [40, 250, 220, 285],
                    "text": "Figure 1. Example output.",
                    "translated_text": "Figure 1. Exemple de sortie.",
                    "lines": [
                        {
                            "bbox": [40, 250, 220, 285],
                            "line_text": "Figure 1. Example output.",
                            "translated_text": "Figure 1. Exemple de sortie.",
                            "phrases": [
                                {
                                    "bbox": [40, 250, 220, 285],
                                    "texte": "Figure 1. Example output.",
                                    "translated_text": "Figure 1. Exemple de sortie.",
                                    "spans": [
                                        {
                                            "bbox": [40, 250, 220, 285],
                                            "texte": "Figure 1. Example output.",
                                            "translated_text": "Figure 1. Exemple de sortie.",
                                            "style": {"font": "ArialMT", "size": 10, "color": "#111111", "flags": {}},
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        reconstructor._inject_dynamic_immutable_overlays(page_data)
        self.assertNotEqual(page_data["blocks"][0].get("render_mode"), "background_only")

    def test_extract_immutable_overlays_protects_code_visible_phrase(self):
        from ocr_server import _extract_immutable_overlays

        img = Image.new("RGB", (220, 120), "white")
        blocks = [
            {
                "id": "code1",
                "role": "body",
                "unit_type": "code_visible",
                "bbox": [20, 20, 200, 52],
                "lines": [
                    {
                        "bbox": [20, 20, 200, 52],
                        "unit_type": "code_visible",
                        "phrases": [
                            {
                                "bbox": [20, 20, 200, 52],
                                "unit_type": "code_visible",
                                "text": "x = inception_module(x, filters_1x1=64)",
                                "texte": "x = inception_module(x, filters_1x1=64)",
                                "spans": [
                                    {
                                        "bbox": [20, 20, 200, 52],
                                        "texte": "x = inception_module(x, filters_1x1=64)",
                                        "style": {"font": "Courier", "flags": {"monospace": True}},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

        overlays = _extract_immutable_overlays(blocks, img, "test_code_visible.pdf", 0)

        self.assertEqual(len(overlays), 1)
        self.assertEqual(overlays[0].get("reason"), "immutable_code")
        self.assertEqual(blocks[0]["lines"][0]["phrases"][0].get("render_mode"), "background_only")
        self.assertTrue(blocks[0]["lines"][0]["phrases"][0]["spans"][0].get("skip_render"))

    def test_translated_immutable_code_block_generates_source_overlay(self):
        reconstructor = DocumentReconstructor()
        reconstructor._overlay_exists = lambda overlays, bbox: False
        reconstructor._save_crop_overlay = lambda page_data, bbox, kind="dynamic": {"path": "/tmp/fake.png", "bbox": list(bbox), "kind": kind}
        page_data = {
            "dimensions": {"width": 400, "height": 600},
            "layout_type": "double_column",
            "document_type": "book_page",
            "immutable_overlays": [],
            "images": [],
            "non_text_zones": [],
            "layout": {},
            "blocks": [
                {
                    "id": "code_block",
                    "role": "body",
                    "source": "native",
                    "bbox": [40, 300, 360, 420],
                    "translated_text": "x = inception_module(...)",
                    "immutable_code_block": True,
                    "lines": [
                        {
                            "bbox": [40, 300, 360, 330],
                            "unit_type": "code_visible",
                            "phrases": [
                                {
                                    "bbox": [40, 300, 360, 330],
                                    "unit_type": "code_visible",
                                    "texte": "x = inception_module(...)",
                                    "translated_text": "x = inception_module(...)",
                                    "style": {"font": "Courier", "flags": {"monospace": True}},
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        reconstructor._inject_dynamic_immutable_overlays(page_data)

        self.assertEqual(page_data["blocks"][0].get("render_mode"), "background_only")
        self.assertEqual(page_data["immutable_overlays"][0].get("kind"), "code_block_locked")

    def test_insert_immutable_code_overlay_is_not_skipped_by_translated_overlap(self):
        reconstructor = DocumentReconstructor()
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        with tempfile.TemporaryDirectory() as tmp:
            overlay_path = os.path.join(tmp, "code_overlay.png")
            Image.new("RGB", (40, 20), "black").save(overlay_path)
            page_data = {
                "page_role": "body",
                "blocks": [
                    {
                        "id": "body1",
                        "role": "body",
                        "bbox": [40, 40, 120, 80],
                        "translated_text": "texte traduit voisin",
                        "lines": [],
                    }
                ],
                "immutable_overlays": [
                    {
                        "bbox": [40, 40, 120, 80],
                        "path": overlay_path,
                        "kind": "code_block_locked",
                        "text": "x = code()",
                    }
                ],
            }

            reconstructor._insert_immutable_overlays(page, page_data)

            self.assertEqual(len(page.get_images(full=True)), 1)

    def test_translated_editorial_body_misclassified_as_table_prefers_whiteout_over_source_overlay(self):
        reconstructor = DocumentReconstructor()
        item = {
            "source": "native",
            "translated_block": True,
            "role": "body",
            "source_text": "Similarly, let's create inception modules 4a, 4b, 4c, 4d, and 4e and the max pooling layer:",
            "text": "De même, créons les modules de démarrage 4a, 4b, 4c, 4d et 4e et le calque de mise en commun max:",
            "descriptor_band_role": "table_band",
            "descriptor_region_type": "table_row",
            "descriptor_visual_text": {
                "text_embedding_mode": "outside_visual",
                "background_kind": "plain",
                "background_replacement_strategy": "whiteout",
            },
            "style": {
                "font": "NewBaskerville-Roman",
                "size": 9.96,
                "color": "#262626",
                "flags": {"bold": False, "italic": False, "serif": True},
            },
        }

        self.assertTrue(reconstructor._prefer_whiteout_for_translated_editorial_body(item))
        self.assertTrue(reconstructor._should_relax_table_lock_for_translated_editorial_body(item))
        self.assertFalse(reconstructor._prefer_text_erased_overlay(item))
        self.assertFalse(reconstructor._should_restore_background_before_render(item))
        self.assertTrue(reconstructor._should_whiteout_before_render(item))

    def test_extract_block_slot_items_shrinks_body_bbox_to_visible_content_when_hidden_content_exists(self):
        reconstructor = DocumentReconstructor()
        page_data = {
            "dimensions": {"width": 400, "height": 600},
            "layout_type": "double_column",
            "document_type": "book_page",
            "blocks": [
                {
                    "id": "b_hidden",
                    "role": "body",
                    "source": "native",
                    "bbox": [20, 20, 220, 140],
                    "translated_text": "Texte visible",
                    "lines": [
                        {
                            "bbox": [20, 20, 140, 42],
                            "line_text": "Visible text",
                            "translated_text": "Texte visible",
                            "phrases": [
                                {
                                    "bbox": [20, 20, 140, 42],
                                    "texte": "Visible text",
                                    "translated_text": "Texte visible",
                                    "spans": [
                                        {
                                            "bbox": [20, 20, 140, 42],
                                            "texte": "Visible text",
                                            "translated_text": "Texte visible",
                                            "style": {"font": "NewBaskerville-Roman", "size": 11, "color": "#000000", "flags": {}},
                                        }
                                    ],
                                }
                            ],
                        },
                        {
                            "bbox": [20, 90, 220, 140],
                            "line_text": "x = code()",
                            "translated_text": "x = code()",
                            "render_mode": "background_only",
                            "phrases": [
                                {
                                    "bbox": [20, 90, 220, 140],
                                    "texte": "x = code()",
                                    "translated_text": "x = code()",
                                    "render_mode": "background_only",
                                    "spans": [
                                        {
                                            "bbox": [20, 90, 220, 140],
                                            "texte": "x = code()",
                                            "translated_text": "x = code()",
                                            "style": {"font": "Courier", "size": 10, "color": "#000000", "flags": {"monospace": True}},
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ],
        }

        items = reconstructor._extract_block_slot_items(page_data)
        body_items = [item for item in items if item.get("source_block_id") == "b_hidden" and item.get("role") == "body"]

        self.assertEqual(len(body_items), 1)
        self.assertTrue(body_items[0].get("contains_background_only_content"))
        self.assertLess(body_items[0]["bbox"].y1, 30.0)

    def test_header_page_number_split_replaces_stale_source_lines(self):
        reconstructor = DocumentReconstructor()
        page_data = {
            "dimensions": {"width": 400, "height": 600},
            "layout_type": "double_column",
            "document_type": "web_print",
            "page_role": "body",
            "blocks": [
                {
                    "id": "h1",
                    "role": "header",
                    "source": "native",
                    "bbox": [40, 20, 240, 44],
                    "text": "268 CHAPTER 6 Transfer learning",
                    "translated_text": "268 CHAPITRE 6 Transfert de l'apprentissage",
                    "lines": [
                        {
                            "bbox": [40, 20, 70, 44],
                            "line_text": "268",
                            "translated_text": "268",
                            "phrases": [
                                {
                                    "bbox": [40, 20, 70, 44],
                                    "texte": "268",
                                    "translated_text": "268",
                                    "spans": [{"bbox": [40, 20, 70, 44], "texte": "268", "translated_text": "268", "style": {"font": "ArialMT", "size": 9, "color": "#444444", "flags": {}}}],
                                }
                            ],
                        },
                        {
                            "bbox": [72, 20, 150, 44],
                            "line_text": "CHAPTER 6",
                            "translated_text": "CHAPITRE 6",
                            "phrases": [
                                {
                                    "bbox": [72, 20, 150, 44],
                                    "texte": "CHAPTER 6",
                                    "translated_text": "CHAPITRE 6",
                                    "spans": [{"bbox": [72, 20, 150, 44], "texte": "CHAPTER 6", "translated_text": "CHAPITRE 6", "style": {"font": "ArialMT", "size": 9, "color": "#444444", "flags": {}}}],
                                }
                            ],
                        },
                        {
                            "bbox": [152, 20, 240, 44],
                            "line_text": "Transfer learning",
                            "translated_text": "Transfert de l'apprentissage",
                            "phrases": [
                                {
                                    "bbox": [152, 20, 240, 44],
                                    "texte": "Transfer learning",
                                    "translated_text": "Transfert de l'apprentissage",
                                    "spans": [{"bbox": [152, 20, 240, 44], "texte": "Transfer learning", "translated_text": "Transfert de l'apprentissage", "style": {"font": "ArialMT", "size": 9, "color": "#444444", "flags": {}}}],
                                }
                            ],
                        },
                    ],
                }
            ],
        }
        items = reconstructor._extract_block_slot_items(page_data)
        header_items = [item for item in items if item.get("role") == "header"]
        header_by_source = {
            tuple(item.get("source_lines") or []): item
            for item in header_items
        }
        self.assertIn(("CHAPITRE 6",), header_by_source)
        self.assertIn(("Transfert de l'apprentissage",), header_by_source)
        self.assertIn(("268",), header_by_source)
        title_item = header_by_source[("CHAPITRE 6",)]
        subtitle_item = header_by_source[("Transfert de l'apprentissage",)]
        number_item = header_by_source[("268",)]
        self.assertEqual(title_item.get("source_lines"), ["CHAPITRE 6"])
        self.assertEqual(subtitle_item.get("source_lines"), ["Transfert de l'apprentissage"])
        self.assertTrue(title_item.get("preserve_linebreaks"))
        self.assertTrue(subtitle_item.get("preserve_linebreaks"))
        self.assertEqual(number_item.get("source_lines"), ["268"])
        self.assertEqual(title_item.get("text"), "CHAPITRE 6")
        self.assertEqual(subtitle_item.get("text"), "Transfert de l'apprentissage")

    def test_top_header_reorder_skips_chapter_markers(self):
        reconstructor = DocumentReconstructor()
        self.assertFalse(reconstructor._should_reorder_top_header_number("CHAPITRE 6"))
        self.assertFalse(reconstructor._should_reorder_top_header_number("CHAPTER 6"))
        self.assertTrue(reconstructor._should_reorder_top_header_number("Setting up your AWS EC2 environment 441"))

    def test_header_collision_resolution_pushes_later_header_item(self):
        reconstructor = DocumentReconstructor()
        items = [
            {
                "role": "header",
                "text": "CHAPITRE 6",
                "bbox": fitz.Rect(232.3, 26.9, 268.3, 35.0),
                "slots": [fitz.Rect(232.3, 26.9, 268.3, 35.0)],
                "row_start_x_pt": 232.3,
                "style": {"font": "Helvetica", "font_size_pt": 9.0},
            },
            {
                "role": "header",
                "text": "Transfert de l'apprentissage",
                "bbox": fitz.Rect(276.5, 26.9, 339.4, 36.0),
                "slots": [fitz.Rect(276.5, 26.9, 339.4, 36.0)],
                "row_start_x_pt": 276.5,
                "style": {"font": "Helvetica", "font_size_pt": 9.0},
            },
        ]
        reconstructor._resolve_header_item_collisions(items, page_w_pt=595.0)
        self.assertGreater(items[1]["row_start_x_pt"], 276.5)
        self.assertGreater(items[1]["slots"][0].x0, 276.5)

    def test_running_header_ignores_table_band_region_reclamp(self):
        reconstructor = DocumentReconstructor()
        doc = fitz.open()
        try:
            page = doc.new_page(width=595, height=842)
            item = {
                "text": "Transfert de l'apprentissage",
                "source_text": "Transfer learning",
                "translated_block": True,
                "source": "native",
                "role": "header",
                "bbox": fitz.Rect(324.32, 26.88, 387.2, 36.0),
                "slots": [fitz.Rect(324.32, 26.88, 387.2, 36.0)],
                "slot_w_pt": 62.88,
                "slot_h_pt": 9.12,
                "slot_gap_x_pt": 2.0,
                "slot_gap_y_pt": 2.0,
                "row_start_x_pt": 324.32,
                "style": {"font": "Times-BoldItalic", "font_size_pt": 9.0, "color": "#656565"},
                "alignment": "left",
                "descriptor_region_type": "table_row",
                "descriptor_band_role": "table_band",
                "descriptor_structural_role": "table_header_cell",
                "descriptor_layout_behavior": "locked_in_cell",
                "descriptor_typographic_class": "running_header",
                "descriptor_region_bbox": fitz.Rect(57.12, 26.88, 339.36, 36.0),
            }
            _, _, blue_rect, used_slots = reconstructor._render_block_slots(
                page=page,
                item=item,
                anchor_y=26.88,
                left=2.0,
                right=593.0,
                zone_top=2.0,
                zone_bottom=80.0,
                render=False,
            )
            self.assertIsNotNone(blue_rect)
            self.assertGreaterEqual(blue_rect.x0, 324.0)
            self.assertTrue(used_slots)
            self.assertGreaterEqual(used_slots[0].x0, 324.0)
        finally:
            doc.close()

    def test_translated_table_band_body_line_whiteouts_even_if_text_is_unchanged(self):
        reconstructor = DocumentReconstructor()
        item = {
            "source": "native",
            "role": "body",
            "translated_block": True,
            "descriptor_band_role": "table_band",
            "source_text": "Non-trainable params: 7,635,264",
            "text": "Non-trainable params: 7,635,264",
        }
        self.assertTrue(reconstructor._should_whiteout_per_line(item))

    def test_target_attached_title_band_label_is_force_anchored(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "title",
            "descriptor_layout_behavior": "anchored",
            "descriptor_band_role": "title_band",
            "descriptor_region_type": "title",
            "anchor_target_bbox": fitz.Rect(100, 100, 240, 220),
        }
        self.assertTrue(reconstructor._item_requires_anchored_render(item, anchored_figure_page=False))

    def test_anchor_target_span_expands_above_label_width(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "header",
            "descriptor_layout_behavior": "anchored",
            "descriptor_band_role": "header_band",
            "anchor_target_bbox": fitz.Rect(120, 80, 260, 220),
            "anchor_preferred_side": "above",
            "is_diagram_label": True,
        }
        x0, block_right = reconstructor._expand_anchor_target_span(item, left=0.0, right=300.0, x0=180.0, block_right=210.0)
        self.assertLessEqual(x0, 120.0)
        self.assertGreaterEqual(block_right, 260.0)

    def test_visual_group_bbox_includes_labels_outside_visual_object(self):
        reconstructor = DocumentReconstructor()
        items = [
            {
                "text": "CHAPITRE",
                "bbox": fitz.Rect(40, 20, 160, 36),
                "descriptor_group_render_mode": "annotation_group",
                "descriptor_group_ids": {"annotation_group_id": "g1"},
                "descriptor_visual_text_group": {"bbox": [250, 180, 540, 460]},
            },
            {
                "text": "Label",
                "bbox": fitz.Rect(140, 100, 180, 116),
                "descriptor_group_render_mode": "annotation_group",
                "descriptor_group_ids": {"annotation_group_id": "g1"},
                "descriptor_visual_text_group": {"bbox": [250, 180, 540, 460]},
            },
        ]
        groups, _ = reconstructor._group_visual_items(items)
        self.assertEqual(len(groups), 1)
        group_bbox = groups[0]["bbox"]
        self.assertLessEqual(group_bbox.y0, 20.0)
        self.assertGreaterEqual(group_bbox.y1, 220.0)

    def test_translated_annotation_label_requires_exact_slot_render(self):
        reconstructor = DocumentReconstructor()
        item = {
            "translated_block": True,
            "role": "diagram_text_label",
            "descriptor_band_role": "annotation_band",
            "descriptor_group_render_mode": "annotation_group",
            "descriptor_structural_role": "diagram_label",
        }
        self.assertTrue(reconstructor._item_requires_exact_slot_render(item))
        self.assertFalse(
            reconstructor._item_requires_exact_slot_render(
                {
                    "translated_block": True,
                    "role": "figure_caption",
                    "descriptor_band_role": "caption_band",
                    "descriptor_group_render_mode": "",
                    "descriptor_structural_role": "figure_caption",
                }
            )
        )

    def test_translated_short_locked_cell_callout_requires_exact_slot_render(self):
        reconstructor = DocumentReconstructor()
        self.assertTrue(
            reconstructor._item_requires_exact_slot_render(
                {
                    "translated_block": True,
                    "role": "title",
                    "text": "avec Keras",
                    "source_text": "using Keras’s",
                    "source_lines": ["avec Keras"],
                    "descriptor_band_role": "table_band",
                    "descriptor_structural_role": "table_stub_cell",
                    "descriptor_layout_behavior": "locked_in_cell",
                }
            )
        )
        self.assertFalse(
            reconstructor._item_requires_exact_slot_render(
                {
                    "translated_block": True,
                    "role": "body",
                    "text": "new_model.summary()",
                    "source_text": "new_model.summary()",
                    "source_lines": ["new_model.summary()"],
                    "descriptor_band_role": "table_band",
                    "descriptor_structural_role": "table_value_cell",
                    "descriptor_layout_behavior": "locked_in_cell",
                }
            )
        )
        self.assertFalse(
            reconstructor._item_requires_exact_slot_render(
                {
                    "translated_block": True,
                    "role": "body",
                    "text": "Enregistre la sortie de base_model",
                    "source_text": "Saves the output of base_model",
                    "source_lines": ["Enregistre la sortie de base_model"],
                    "descriptor_band_role": "table_band",
                    "descriptor_structural_role": "table_value_cell",
                    "descriptor_layout_behavior": "locked_in_cell",
                }
            )
        )

    def test_translated_diagram_phrase_item_preserves_exact_slot(self):
        reconstructor = DocumentReconstructor()
        page_data = {
            "dimensions": {"width": 400, "height": 300},
            "language": "en",
            "layout": {
                "layout_type": "annotated_page",
                "document_type": "manual_guide",
                "page_family": "body_with_figure",
            },
            "layout_descriptor": {
                "elements": [
                    {
                        "id": "b1",
                        "band_role": "annotation_band",
                        "structural_role": "diagram_label",
                        "layout_behavior": "anchored",
                        "group_render_mode": "annotation_group",
                        "group_ids": {"annotation_group_id": "g1"},
                        "page_region_id": "r1",
                    }
                ],
                "regions": [
                    {"id": "r1", "type": "annotation_band", "bbox": [20, 20, 160, 120]},
                ],
                "visual_text_model": {
                    "objects": [
                        {
                            "source_element_id": "b1",
                            "bbox": [120, 80, 280, 220],
                            "group_id": "vg1",
                        }
                    ],
                    "groups": [
                        {"id": "vg1", "bbox": [20, 20, 300, 240]},
                    ],
                },
            },
            "blocks": [
                {
                    "id": "b1",
                    "bbox": [40, 50, 100, 70],
                    "role": "diagram_text_label",
                    "source": "native",
                    "translated_text": "Visage humain",
                    "text": "Human face",
                    "lines": [
                        {
                            "bbox": [40, 50, 100, 70],
                            "line_text": "Human face",
                            "translated_text": "Visage humain",
                            "phrases": [
                                {
                                    "bbox": [40, 50, 100, 70],
                                    "texte": "Human face",
                                    "translated_text": "Visage humain",
                                    "spans": [
                                        {
                                            "bbox": [40, 50, 100, 70],
                                            "texte": "Human face",
                                            "translated_text": "Visage humain",
                                            "style": {"font": "ArialMT", "size": 9, "color": "#2a5db0", "flags": {}},
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        items = reconstructor._extract_block_slot_items(page_data)
        label_item = next(item for item in items if item.get("role") == "diagram_text_label")
        self.assertTrue(label_item.get("translated_block"))
        self.assertTrue(label_item.get("exact_slot_render"))
        self.assertTrue(label_item.get("strict_bbox_mode"))
        self.assertEqual(label_item.get("descriptor_band_role"), "annotation_band")
        self.assertEqual(label_item.get("descriptor_group_render_mode"), "annotation_group")
        self.assertEqual(label_item.get("source_text"), "Human face")
        self.assertEqual(label_item.get("alignment"), "left")

    def test_translated_body_line_with_locked_equation_splits_into_exact_slot_runs(self):
        reconstructor = DocumentReconstructor()
        page_data = {
            "dimensions": {"width": 400, "height": 300},
            "language": "en",
            "page_role": "body",
            "layout_type": "double_column",
            "document_type": "book_page",
            "page_family": "body_text_two_column_equations",
            "page_case": {"fallback_policy": "known::body_text_two_column_equations"},
            "blocks": [
                {
                    "id": "b1",
                    "bbox": [40, 80, 320, 102],
                    "role": "body",
                    "source": "native",
                    "text": "Find the direction of the dE / dw The algorithm computes the slope.",
                    "translated_text": "",
                    "lines": [
                        {
                            "bbox": [40, 80, 320, 102],
                            "line_text": "Find the direction of the dE / dw The algorithm computes the slope.",
                            "phrases": [
                                {
                                    "bbox": [40, 80, 150, 102],
                                    "texte": "Find the direction of the",
                                    "translated_text": "Trouvez la direction de la",
                                    "spans": [
                                        {
                                            "bbox": [40, 80, 150, 102],
                                            "texte": "Find the direction of the",
                                            "style": {"font": "ArialMT", "size": 10, "color": "#222222", "flags": {}},
                                        }
                                    ],
                                },
                                {
                                    "bbox": [154, 80, 198, 102],
                                    "texte": "dE / dw",
                                    "translated_text": "dE / dw",
                                    "render_mode": "background_only",
                                    "spans": [
                                        {
                                            "bbox": [154, 80, 198, 102],
                                            "texte": "dE / dw",
                                            "style": {"font": "TimesNewRomanPSMT", "size": 10, "color": "#222222", "flags": {}},
                                        }
                                    ],
                                },
                                {
                                    "bbox": [206, 80, 320, 102],
                                    "texte": "The algorithm computes the slope.",
                                    "translated_text": "L'algorithme calcule la pente.",
                                    "spans": [
                                        {
                                            "bbox": [206, 80, 320, 102],
                                            "texte": "The algorithm computes the slope.",
                                            "style": {"font": "ArialMT", "size": 10, "color": "#222222", "flags": {}},
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                }
            ],
        }
        items = reconstructor._extract_block_slot_items(page_data)
        split_items = [it for it in items if it.get("exact_slot_render") and it.get("role") == "body"]
        self.assertEqual(len(split_items), 2)
        self.assertTrue(any("Trouvez la direction" in it.get("text", "") for it in split_items))
        self.assertTrue(any("algorithme calcule" in it.get("text", "") for it in split_items))

    def test_toc_renderer_scales_dense_page_instead_of_dropping_last_rows(self):
        reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        reconstructor.pixel_to_point = 72.0 / 150.0
        reconstructor._rendered_signatures = set()

        doc = fitz.open()
        page = doc.new_page(width=531.36, height=180.0)
        rows = []
        for idx in range(6):
            y = 48.0 + (idx * 42.0)
            rows.append(
                {
                    "role": "section_heading",
                    "label": f"6.{idx + 1} Long translated heading for dense toc rendering regression coverage",
                    "page": str(240 + idx),
                    "y": y,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [308.0, y, 860.0, y + 23.0],
                    "page_bbox": [894.0, y, 930.0, y + 23.0],
                    "source_band_id": idx,
                    "source_band_lane": 0,
                }
            )

        reconstructor._render_toc_rows_v2(
            page,
            rows,
            tab_stops={"column_left_x": 287.0, "column_right_x": 911.0, "page_num_right_x": 930.0},
            zone_top=20.0,
            zone_bottom=170.0,
            left=0.0,
            right=531.36,
        )

        text = page.get_text("text")
        self.assertIn("6.6", text)
        self.assertIn("245", text)
        doc.close()

    def test_toc_renderer_uses_native_label_width_when_page_slot_is_incoherent(self):
        reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        reconstructor.pixel_to_point = 72.0 / 150.0
        reconstructor._rendered_signatures = set()

        doc = fitz.open()
        page = doc.new_page(width=531.36, height=220.0)
        row = {
            "role": "subentry_marker",
            "label": "Module d'accueil avec réduction de dimensionnalité",
            "page": "220",
            "y": 170.0,
            "style": {"size": 10.0, "source": "native"},
            "label_bbox": [388.0, 170.0, 835.0, 216.0],
            "page_bbox": [484.0, 195.0, 517.0, 216.0],
            "source_band_id": 3,
            "source_band_lane": 0,
        }

        reconstructor._render_toc_rows_v2(
            page,
            [row],
            tab_stops={"column_left_x": 287.0, "column_right_x": 911.0, "page_num_right_x": 930.0},
            zone_top=20.0,
            zone_bottom=210.0,
            left=0.0,
            right=531.36,
        )

        blocks = [b for b in page.get_text("blocks") if "dimensionnalité" in b[4]]
        self.assertTrue(blocks)
        widest = max((b[2] - b[0]) for b in blocks)
        self.assertGreater(widest, 120.0)
        doc.close()

    def test_toc_renderer_forces_end_anchor_for_toc_page_number_ruleset(self):
        reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        reconstructor.pixel_to_point = 72.0 / 150.0
        reconstructor._rendered_signatures = set()

        doc = fitz.open()
        page = doc.new_page(width=531.36, height=220.0)
        row = {
            "role": "subentry",
            "label": "Hidden layers",
            "page": "96",
            "y": 170.0,
            "style": {"size": 10.0, "source": "native"},
            "label_bbox": [388.0, 170.0, 760.0, 196.0],
            "page_bbox": [430.0, 170.0, 460.0, 196.0],
            "source_band_id": 3,
            "source_band_lane": 0,
            "page_ruleset": {
                "rules": {
                    "semantic_role": "toc_page_number",
                    "preserve_horizontal_anchor": "end",
                }
            },
        }

        reconstructor._render_toc_rows_v2(
            page,
            [row],
            tab_stops={"column_left_x": 287.0, "column_right_x": 911.0, "page_num_right_x": 930.0},
            zone_top=20.0,
            zone_bottom=210.0,
            left=0.0,
            right=531.36,
        )

        words = [w for w in page.get_text("words") if w[4] == "96"]
        self.assertTrue(words)
        rightmost = max(word[2] for word in words)
        self.assertGreater(rightmost, 440.0)
        doc.close()

    def test_toc_renderer_resolves_fonts_with_rendered_text_payload(self):
        class _RecordingResolver:
            def __init__(self):
                self.calls = []

            def resolve(self, style, text=""):
                self.calls.append(text)
                return {"fontfile": None, "builtin": "helv"}

        reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        reconstructor.pixel_to_point = 72.0 / 150.0
        reconstructor._rendered_signatures = set()
        reconstructor.font_resolver = _RecordingResolver()

        doc = fitz.open()
        page = doc.new_page(width=531.36, height=220.0)
        row = {
            "role": "subentry",
            "label": "Parameters vs. hyperparameters",
            "translated_label": "Paramètres vs hyperparamètres",
            "page": "163",
            "y": 84.0,
            "style": {"size": 10.0, "source": "native"},
            "page_style": {"size": 10.0, "source": "native"},
            "label_bbox": [388.0, 84.0, 760.0, 110.0],
            "page_bbox": [894.0, 84.0, 930.0, 110.0],
            "source_band_id": 3,
            "source_band_lane": 0,
        }

        reconstructor._render_toc_rows_v2(
            page,
            [row],
            tab_stops={"column_left_x": 287.0, "column_right_x": 911.0, "page_num_right_x": 930.0},
            zone_top=20.0,
            zone_bottom=210.0,
            left=0.0,
            right=531.36,
        )

        self.assertIn("Paramètres vs hyperparamètres", reconstructor.font_resolver.calls)
        self.assertIn("163", reconstructor.font_resolver.calls)
        doc.close()

    def test_toc_marker_is_normalized_for_subentry_render(self):
        reconstructor = DocumentReconstructor()
        self.assertEqual(reconstructor._normalize_toc_marker_for_render("■", row_role="subentry_marker"), "·")
        self.assertEqual(reconstructor._normalize_toc_marker_for_render("■", row_role="section_heading"), "■")

    def test_part_title_prefix_is_not_duplicated_when_source_marker_is_reinjected(self):
        translated = "Partie classification d'images et détection"
        source = "PART 2 IMAGE CLASSIFICATION AND DETECTION"
        source_marker_match = re.match(r"^\s*(?:partie|part)\s+([0-9ivxlcdm]+)\b", source, flags=re.IGNORECASE)
        self.assertIsNotNone(source_marker_match)
        cleaned = re.sub(r"^\s*(?:partie|part)\s+[0-9ivxlcdm]+\s+", "", translated, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*(?:partie|part)\s+", "", cleaned, flags=re.IGNORECASE)
        rebuilt = f"Partie {source_marker_match.group(1)} {cleaned}".strip()
        self.assertEqual(rebuilt, "Partie 2 classification d'images et détection")

    def test_toc_part_title_moves_below_expanded_previous_rows(self):
        reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        reconstructor.pixel_to_point = 72.0 / 150.0
        reconstructor._rendered_signatures = set()

        doc = fitz.open()
        page = doc.new_page(width=531.36, height=360.0)
        rows = [
            {
                "role": "section_heading",
                "label": "Longue ligne précédente qui doit prendre plusieurs lignes avant le titre de partie",
                "translated_label": "Longue ligne précédente qui doit prendre plusieurs lignes avant le titre de partie",
                "page": "181",
                "y": 120.0,
                "style": {"size": 10.0, "source": "native"},
                "label_bbox": [308.0, 120.0, 720.0, 168.0],
                "page_bbox": [894.0, 120.0, 930.0, 168.0],
                "source_band_id": 0,
                "source_band_lane": 0,
            },
            {
                "role": "subentry",
                "label": "Autre sous entrée très longue pour étendre encore la hauteur du groupe précédent",
                "translated_label": "Autre sous entrée très longue pour étendre encore la hauteur du groupe précédent",
                "page": "182",
                "y": 148.0,
                "style": {"size": 10.0, "source": "native"},
                "label_bbox": [388.0, 148.0, 820.0, 208.0],
                "page_bbox": [894.0, 148.0, 930.0, 208.0],
                "source_band_id": 1,
                "source_band_lane": 0,
            },
            {
                "role": "part_title",
                "label": "PART 2 IMAGE CLASSIFICATION AND DETECTION",
                "translated_label": "Partie classification d'images et détection",
                "page": "193",
                "y": 160.0,
                "style": {"size": 12.78, "source": "native", "flags": {"bold": True, "serif": True, "uppercase": True}},
                "page_style": {"size": 12.78, "source": "native", "flags": {"bold": True, "serif": True, "uppercase": True}},
                "label_bbox": [119.0, 160.0, 969.0, 194.0],
                "page_bbox": [252.0, 160.0, 969.0, 194.0],
                "source_band_id": 2,
                "source_band_lane": 0,
            },
        ]

        reconstructor._render_toc_rows_v2(
            page,
            rows,
            tab_stops={"column_left_x": 287.0, "column_right_x": 911.0, "page_num_right_x": 930.0},
            zone_top=20.0,
            zone_bottom=340.0,
            left=0.0,
            right=531.36,
        )

        words = page.get_text("words")
        prev_words = [fitz.Rect(word[:4]) for word in words if word[4] in {"Longue", "Autre", "précédent"}]
        part_words = [fitz.Rect(word[:4]) for word in words if word[4] in {"PARTIE", "CLASSIFICATION", "DÉTECTION"}]

        self.assertTrue(prev_words)
        self.assertTrue(part_words)
        self.assertGreaterEqual(min(rect.y0 for rect in part_words), max(rect.y1 for rect in prev_words) - 0.1)
        doc.close()

    def test_toc_chapter_marker_is_not_merged_into_previous_section_line(self):
        reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        reconstructor.pixel_to_point = 72.0 / 150.0
        reconstructor._rendered_signatures = set()

        doc = fitz.open()
        page = doc.new_page(width=531.36, height=666.24)
        rows = [
            {
                "role": "section_heading",
                "label": "6.8 Projet 2: réglage fin",
                "page": "274",
                "y": 1070.0,
                "style": {"size": 10.0, "source": "native"},
                "label_bbox": [308.0, 1070.0, 579.0, 1093.0],
                "page_bbox": [602.0, 1070.0, 637.0, 1093.0],
                "source_band_id": 29,
                "source_band_lane": 0,
            },
            {
                "role": "chapter_title",
                "label": "Détection d'objets avec R-CNN, SSD et YOLO",
                "page": "283",
                "y": 1120.0,
                "style": {"size": 12.5, "source": "native"},
                "label_bbox": [287.0, 1120.0, 789.0, 1147.0],
                "page_bbox": [816.0, 1120.0, 857.0, 1147.0],
                "chapter_number": "7",
                "source_band_id": 30,
                "source_band_lane": 0,
            },
        ]

        reconstructor._render_toc_rows_v2(
            page,
            rows,
            tab_stops={"column_left_x": 287.0, "column_right_x": 911.0, "page_num_right_x": 930.0},
            zone_top=20.0,
            zone_bottom=650.0,
            left=0.0,
            right=531.36,
        )

        texts = [" ".join(b[4].split()) for b in page.get_text("blocks") if b[4].strip()]
        self.assertFalse(any("274 7" in t for t in texts))
        self.assertTrue(any("Détection d'objets" in t for t in texts))
        doc.close()

    def test_dense_toc_renderer_keeps_trailing_rows_visible(self):
        reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        reconstructor.pixel_to_point = 72.0 / 150.0
        reconstructor._rendered_signatures = set()

        doc = fitz.open()
        page = doc.new_page(width=531.36, height=666.24)
        rows = []
        y = 120.0
        for idx in range(44):
            rows.append(
                {
                    "role": "section_heading" if idx % 3 == 0 else "subentry",
                    "label": f"{idx + 1}. Dense toc row {idx + 1}",
                    "page": str(200 + idx),
                    "y": y,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [308.0, y, 700.0, y + 22.0],
                    "page_bbox": [894.0, y, 930.0, y + 22.0],
                    "source_band_id": idx,
                    "source_band_lane": 0,
                }
            )
            y += 23.0
        rows.extend(
            [
                {
                    "role": "section_heading",
                    "label": "Network predictions",
                    "page": "287",
                    "y": y,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [308.0, y, 760.0, y + 22.0],
                    "page_bbox": [894.0, y, 930.0, y + 22.0],
                    "source_band_id": 44,
                    "source_band_lane": 0,
                },
                {
                    "role": "section_heading",
                    "label": "Non-maximum suppression (NMS)",
                    "page": "288",
                    "y": y + 23.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [308.0, y + 23.0, 760.0, y + 45.0],
                    "page_bbox": [894.0, y + 23.0, 930.0, y + 45.0],
                    "source_band_id": 45,
                    "source_band_lane": 0,
                },
                {
                    "role": "subentry_marker",
                    "label": "Object-detector evaluation metrics",
                    "page": "289",
                    "y": y + 46.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [308.0, y + 46.0, 820.0, y + 68.0],
                    "page_bbox": [894.0, y + 46.0, 930.0, y + 68.0],
                    "source_band_id": 46,
                    "source_band_lane": 0,
                },
            ]
        )

        reconstructor._render_toc_rows_v2(
            page,
            rows,
            tab_stops={"column_left_x": 287.0, "column_right_x": 911.0, "page_num_right_x": 930.0},
            zone_top=20.0,
            zone_bottom=650.0,
            left=0.0,
            right=531.36,
        )

        text = " ".join(page.get_text("text").split())
        self.assertIn("Network predictions", text)
        self.assertIn("287", text)
        doc.close()

    def test_dense_toc_renderer_keeps_realistic_last_row_page_number_visible(self):
        reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        reconstructor.pixel_to_point = 72.0 / 150.0
        reconstructor._rendered_signatures = set()

        doc = fitz.open()
        page = doc.new_page(width=531.36, height=666.24)
        rows = []
        y = 55.0
        for idx in range(32):
            rows.append(
                {
                    "role": "section_heading" if idx % 3 == 0 else "subentry",
                    "label": f"4.{idx} Dense TOC scaffold row {idx}",
                    "page": str(160 + idx),
                    "y": y,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [308.0, y, 760.0, y + 22.0],
                    "page_bbox": [894.0, y, 930.0, y + 22.0],
                    "source_band_id": idx,
                    "source_band_lane": 0,
                }
            )
            y += 29.0
        rows.extend(
            [
                {
                    "role": "subentry_marker",
                    "label": "Mise en œuvre de LeNet-5 à Keras",
                    "page": "200",
                    "y": 976.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [578.0, 976.0, 862.0, 997.0],
                    "page_bbox": [883.0, 976.0, 916.0, 997.0],
                    "source_band_id": 32,
                    "source_band_lane": 0,
                    "marker": "■",
                },
                {
                    "role": "section_heading",
                    "label": "Mise en place des hyperparamètres d'apprentissage",
                    "page": "202",
                    "y": 1001.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [369.0, 1001.0, 685.0, 1022.0],
                    "page_bbox": [706.0, 1001.0, 738.0, 1022.0],
                    "source_band_id": 33,
                    "source_band_lane": 0,
                },
                {
                    "role": "subentry_marker",
                    "label": "Performances LeNet sur l'ensemble de données MNIST",
                    "page": "203",
                    "y": 1001.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [369.0, 1001.0, 920.0, 1047.0],
                    "page_bbox": [570.0, 1026.0, 602.0, 1047.0],
                    "source_band_id": 34,
                    "source_band_lane": 0,
                    "marker": "■",
                },
                {
                    "role": "section_heading",
                    "label": "5.3 AlexNet",
                    "page": "203",
                    "y": 1061.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [289.0, 1061.0, 425.0, 1084.0],
                    "page_bbox": [448.0, 1061.0, 483.0, 1084.0],
                    "source_band_id": 35,
                    "source_band_lane": 0,
                },
                {
                    "role": "section_heading",
                    "label": "Architecture d'AlexNet",
                    "page": "205",
                    "y": 1095.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [369.0, 1095.0, 530.0, 1116.0],
                    "page_bbox": [551.0, 1095.0, 584.0, 1116.0],
                    "source_band_id": 36,
                    "source_band_lane": 0,
                },
                {
                    "role": "subentry_marker",
                    "label": "Nouvelles caractéristiques d'AlexNet",
                    "page": "205",
                    "y": 1095.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [594.0, 1095.0, 813.0, 1116.0],
                    "page_bbox": [833.0, 1095.0, 866.0, 1116.0],
                    "source_band_id": 37,
                    "source_band_lane": 0,
                    "marker": "■",
                },
                {
                    "role": "section_heading",
                    "label": "Mise en œuvre d'AlexNet dans Keras",
                    "page": "207",
                    "y": 1120.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [369.0, 1120.0, 637.0, 1141.0],
                    "page_bbox": [657.0, 1120.0, 690.0, 1141.0],
                    "source_band_id": 38,
                    "source_band_lane": 0,
                },
                {
                    "role": "subentry_marker",
                    "label": "Mise en place des hyperparamètres d'apprentissage",
                    "page": "210",
                    "y": 1120.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [369.0, 1120.0, 908.0, 1166.0],
                    "page_bbox": [519.0, 1145.0, 552.0, 1166.0],
                    "source_band_id": 39,
                    "source_band_lane": 0,
                    "marker": "■",
                },
                {
                    "role": "subentry_marker",
                    "label": "Performances AlexNet",
                    "page": "211",
                    "y": 1145.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [562.0, 1145.0, 745.0, 1166.0],
                    "page_bbox": [766.0, 1145.0, 798.0, 1166.0],
                    "source_band_id": 40,
                    "source_band_lane": 0,
                    "marker": "■",
                },
                {
                    "role": "section_heading",
                    "label": "5.4 VGGNet",
                    "page": "212",
                    "y": 1180.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [289.0, 1180.0, 428.0, 1203.0],
                    "page_bbox": [451.0, 1180.0, 487.0, 1203.0],
                    "source_band_id": 41,
                    "source_band_lane": 0,
                },
                {
                    "role": "section_heading",
                    "label": "Nouvelles caractéristiques de VGGNet",
                    "page": "212",
                    "y": 1214.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [369.0, 1214.0, 575.0, 1235.0],
                    "page_bbox": [596.0, 1214.0, 629.0, 1235.0],
                    "source_band_id": 42,
                    "source_band_lane": 0,
                },
                {
                    "role": "subentry_marker",
                    "label": "Configurations VGGNet",
                    "page": "213",
                    "y": 1214.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [639.0, 1214.0, 847.0, 1235.0],
                    "page_bbox": [867.0, 1214.0, 900.0, 1235.0],
                    "source_band_id": 43,
                    "source_band_lane": 0,
                    "marker": "■",
                },
                {
                    "role": "section_heading",
                    "label": "Hyperparamètres d'apprentissage",
                    "page": "216",
                    "y": 1239.0,
                    "style": {"size": 10.0, "source": "native"},
                    "label_bbox": [369.0, 1239.0, 579.0, 1260.0],
                    "page_bbox": [600.0, 1239.0, 633.0, 1260.0],
                    "source_band_id": 44,
                    "source_band_lane": 0,
                },
            ]
        )

        reconstructor._render_toc_rows_v2(
            page,
            rows,
            tab_stops={"column_left_x": 287.0, "column_right_x": 911.0, "page_num_right_x": 930.0},
            zone_top=2.0,
            zone_bottom=664.24,
            left=2.0,
            right=529.36,
        )

        text = " ".join(page.get_text("text").split())
        self.assertIn("5.4 VGGNet", text)
        self.assertIn("213", text)
        self.assertIn("216", text)
        doc.close()

    def test_semantic_phrase_extraction_merges_sentence_across_lines(self):
        from ocr_server import _build_semantic_phrases_for_block

        block = {
            "id": "b1",
            "bbox": [100, 100, 500, 180],
            "source": "ocr",
            "lines": [
                {
                    "line_index": 0,
                    "bbox": [100, 100, 420, 118],
                    "line_text": "This sentence starts on the first line",
                    "hard_break_before": True,
                    "raw_words": [
                        {"label": "This", "bbox": [100, 100, 130, 118], "score": 0.99},
                        {"label": "sentence", "bbox": [135, 100, 195, 118], "score": 0.99},
                        {"label": "starts", "bbox": [200, 100, 242, 118], "score": 0.99},
                        {"label": "on", "bbox": [247, 100, 263, 118], "score": 0.99},
                        {"label": "the", "bbox": [268, 100, 292, 118], "score": 0.99},
                        {"label": "first", "bbox": [297, 100, 330, 118], "score": 0.99},
                        {"label": "line", "bbox": [335, 100, 362, 118], "score": 0.99},
                    ],
                    "phrases": [{"bbox": [100, 100, 420, 118], "texte": "This sentence starts on the first line", "spans": []}],
                },
                {
                    "line_index": 1,
                    "bbox": [100, 126, 500, 144],
                    "line_text": "and ends on the second line.",
                    "hard_break_before": False,
                    "raw_words": [
                        {"label": "and", "bbox": [100, 126, 126, 144], "score": 0.99},
                        {"label": "ends", "bbox": [131, 126, 164, 144], "score": 0.99},
                        {"label": "on", "bbox": [169, 126, 185, 144], "score": 0.99},
                        {"label": "the", "bbox": [190, 126, 214, 144], "score": 0.99},
                        {"label": "second", "bbox": [219, 126, 266, 144], "score": 0.99},
                        {"label": "line.", "bbox": [271, 126, 304, 144], "score": 0.99},
                    ],
                    "phrases": [{"bbox": [100, 126, 304, 144], "texte": "and ends on the second line.", "spans": []}],
                },
            ],
        }

        _build_semantic_phrases_for_block(block)
        semantic_phrases = block.get("semantic_phrases", [])

        self.assertEqual(len(semantic_phrases), 1)
        self.assertEqual(
            semantic_phrases[0]["text"],
            "This sentence starts on the first line and ends on the second line.",
        )
        self.assertTrue(semantic_phrases[0]["multi_line"])
        self.assertEqual(semantic_phrases[0]["line_indices"], [0, 1])

    def test_semantic_phrase_extraction_splits_two_sentences_on_same_line(self):
        from ocr_server import _build_semantic_phrases_for_block

        block = {
            "id": "b2",
            "bbox": [100, 100, 520, 140],
            "source": "ocr",
            "lines": [
                {
                    "line_index": 0,
                    "bbox": [100, 100, 520, 118],
                    "line_text": "First sentence ends here. Second sentence starts here.",
                    "hard_break_before": True,
                    "raw_words": [
                        {"label": "First", "bbox": [100, 100, 132, 118], "score": 0.99},
                        {"label": "sentence", "bbox": [137, 100, 197, 118], "score": 0.99},
                        {"label": "ends", "bbox": [202, 100, 235, 118], "score": 0.99},
                        {"label": "here.", "bbox": [240, 100, 272, 118], "score": 0.99},
                        {"label": "Second", "bbox": [282, 100, 327, 118], "score": 0.99},
                        {"label": "sentence", "bbox": [332, 100, 392, 118], "score": 0.99},
                        {"label": "starts", "bbox": [397, 100, 439, 118], "score": 0.99},
                        {"label": "here.", "bbox": [444, 100, 476, 118], "score": 0.99},
                    ],
                    "phrases": [{"bbox": [100, 100, 476, 118], "texte": "First sentence ends here. Second sentence starts here.", "spans": []}],
                }
            ],
        }

        _build_semantic_phrases_for_block(block)
        semantic_phrases = block.get("semantic_phrases", [])

        self.assertEqual(
            [phrase["text"] for phrase in semantic_phrases],
            ["First sentence ends here.", "Second sentence starts here."],
        )
        self.assertEqual([phrase["fragment_count"] for phrase in semantic_phrases], [1, 1])

    def test_semantic_phrase_extraction_does_not_split_on_visual_hard_break_without_sentence_end(self):
        from ocr_server import _build_semantic_phrases_for_block

        block = {
            "id": "b3",
            "bbox": [80, 80, 520, 190],
            "source": "ocr",
            "lines": [
                {
                    "line_index": 0,
                    "bbox": [100, 100, 420, 118],
                    "line_text": "The goal of the pooling layer is to downsample the feature maps produced by the",
                    "hard_break_before": True,
                    "indent_px": 0.0,
                    "leading_marker": "",
                    "phrases": [{"bbox": [100, 100, 420, 118], "texte": "The goal of the pooling layer is to downsample the feature maps produced by the", "spans": []}],
                },
                {
                    "line_index": 1,
                    "bbox": [100, 126, 500, 144],
                    "line_text": "convolutional layer into a smaller number of parameters, thus reducing computational complexity.",
                    "hard_break_before": True,
                    "indent_px": 0.0,
                    "leading_marker": "",
                    "phrases": [{"bbox": [100, 126, 500, 144], "texte": "convolutional layer into a smaller number of parameters, thus reducing computational complexity.", "spans": []}],
                },
            ],
        }

        _build_semantic_phrases_for_block(block)
        semantic_phrases = block.get("semantic_phrases", [])

        self.assertEqual(len(semantic_phrases), 1)
        self.assertEqual(
            semantic_phrases[0]["text"],
            "The goal of the pooling layer is to downsample the feature maps produced by the convolutional layer into a smaller number of parameters, thus reducing computational complexity.",
        )
        self.assertEqual(semantic_phrases[0]["line_indices"], [0, 1])

    def test_span_characteristics_are_attached_for_expression_inspection(self):
        from ocr_server import _attach_textual_characteristics, _annotate_translation_contracts

        block = {
            "bbox": [50, 50, 300, 120],
            "source": "native",
            "role": "body",
            "alignment": "left",
            "indent_px": 0.0,
            "lines": [
                {
                    "line_index": 0,
                    "bbox": [60, 60, 280, 80],
                    "alignment": "left",
                    "indent_px": 0.0,
                    "phrases": [
                        {
                            "bbox": [60, 60, 280, 80],
                            "alignment": "left",
                            "indent_px": 0.0,
                            "texte": "Basic components of a CNN",
                            "spans": [
                                {
                                    "texte": "CNN",
                                    "bbox": [60, 60, 120, 80],
                                    "style": {
                                        "font": "Times-BoldItalic",
                                        "size": 12.0,
                                        "color": "#2F5D7E",
                                        "flags": {"bold": True, "italic": True},
                                    },
                                },
                                {
                                    "texte": "architecture",
                                    "bbox": [125, 60, 280, 80],
                                    "style": {
                                        "font": "Times-Roman",
                                        "size": 12.0,
                                        "color": "#000000",
                                        "flags": {"bold": False, "italic": False},
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
            "semantic_phrases": [],
        }

        _annotate_translation_contracts([block], page_context={"page_role": "body"})
        _attach_textual_characteristics([block], [0, 0, 400, 200])
        span = block["lines"][0]["phrases"][0]["spans"][0]

        self.assertEqual(span["text_attributes"]["word_count"], 1)
        self.assertEqual(span["style_attributes"]["font_family_primary"], "Times-BoldItalic")
        self.assertEqual(span["style_attributes"]["color_primary"], "#2F5D7E")
        self.assertTrue(span["style_attributes"]["flags_any"]["bold"])
        self.assertTrue(span["style_attributes"]["flags_any"]["italic"])
        self.assertEqual(span["layout_attributes"]["horizontal_alignment"], "left")
        self.assertEqual(span["expression_semantics"]["inline_class"], "technical_inline")
        self.assertEqual(span["expression_semantics"]["emphasis_level"], "strong")
        self.assertTrue(span["expression_relations"]["with_next"]["exists"])
        self.assertEqual(span["expression_relations"]["with_next"]["relation"], "semantic_shift")

    def test_line_editorial_relations_detect_paragraph_continuation_and_break(self):
        from ocr_server import _annotate_translation_contracts

        block = {
            "id": "b_line",
            "bbox": [50, 50, 350, 150],
            "source": "native",
            "role": "body",
            "lines": [
                {
                    "bbox": [60, 60, 300, 78],
                    "line_text": "First line of a paragraph",
                    "hard_break_before": True,
                    "phrases": [{"texte": "First line of a paragraph", "bbox": [60, 60, 300, 78], "spans": []}],
                },
                {
                    "bbox": [60, 82, 300, 100],
                    "line_text": "continues on the next line.",
                    "hard_break_before": False,
                    "phrases": [{"texte": "continues on the next line.", "bbox": [60, 82, 300, 100], "spans": []}],
                },
                {
                    "bbox": [60, 112, 300, 130],
                    "line_text": "New paragraph starts here.",
                    "hard_break_before": True,
                    "phrases": [{"texte": "New paragraph starts here.", "bbox": [60, 112, 300, 130], "spans": []}],
                },
            ],
            "semantic_phrases": [],
        }

        _annotate_translation_contracts([block], page_context={"page_role": "body"})
        lines = block["lines"]

        self.assertEqual(lines[1]["editorial_relations"]["with_previous"]["relation"], "paragraph_continuation")
        self.assertTrue(lines[1]["editorial_relations"]["with_previous"]["continuation"])
        self.assertEqual(lines[2]["editorial_relations"]["with_previous"]["relation"], "paragraph_break")
        self.assertFalse(lines[2]["editorial_relations"]["with_previous"]["continuation"])

    def test_block_editorial_relations_detect_heading_to_body(self):
        from ocr_server import _annotate_translation_contracts

        blocks = [
            {
                "id": "b_head",
                "bbox": [50, 50, 320, 82],
                "source": "native",
                "role": "section_heading",
                "text": "6.7 Project 1",
                "lines": [{"bbox": [50, 50, 320, 82], "line_text": "6.7 Project 1", "phrases": [{"texte": "6.7 Project 1", "bbox": [50, 50, 320, 82], "spans": []}]}],
                "semantic_phrases": [],
            },
            {
                "id": "b_body",
                "bbox": [50, 92, 420, 150],
                "source": "native",
                "role": "body",
                "text": "In this project we use a pretrained network.",
                "lines": [{"bbox": [50, 92, 420, 110], "line_text": "In this project we use a pretrained network.", "phrases": [{"texte": "In this project we use a pretrained network.", "bbox": [50, 92, 420, 110], "spans": []}]}],
                "semantic_phrases": [],
            },
        ]

        _annotate_translation_contracts(blocks, page_context={"page_role": "body"})

        self.assertEqual(blocks[1]["editorial_relations"]["with_previous"]["relation"], "heading_to_body")
        self.assertTrue(blocks[1]["editorial_relations"]["with_previous"]["continuation"])

    def test_semantic_phrases_get_editorial_relations_and_structural_context(self):
        from ocr_server import _build_semantic_phrases_for_block, _annotate_translation_contracts

        block = {
            "id": "b_phrase",
            "bbox": [50, 50, 420, 180],
            "source": "native",
            "role": "body",
            "lines": [
                {
                    "line_index": 0,
                    "bbox": [60, 60, 380, 78],
                    "line_text": "First sentence ends here.",
                    "hard_break_before": True,
                    "phrases": [{"texte": "First sentence ends here.", "bbox": [60, 60, 380, 78], "spans": []}],
                },
                {
                    "line_index": 1,
                    "bbox": [60, 86, 400, 104],
                    "line_text": "Second sentence continues the editorial flow.",
                    "hard_break_before": False,
                    "phrases": [{"texte": "Second sentence continues the editorial flow.", "bbox": [60, 86, 400, 104], "spans": []}],
                },
            ],
        }

        _build_semantic_phrases_for_block(block)
        _annotate_translation_contracts([block], page_context={"page_role": "body"})
        phrases = block["semantic_phrases"]

        self.assertEqual(len(phrases), 2)
        self.assertEqual(phrases[0]["structural_context"]["block_unit_id"], "b_phrase")
        self.assertEqual(phrases[0]["editorial_semantics"]["flow_class"], "editorial_body")
        self.assertEqual(phrases[1]["editorial_relations"]["with_previous"]["relation"], "paragraph_continuation")
        self.assertEqual(phrases[1]["editorial_relations"]["with_previous"]["neighbor_id"], phrases[0]["unit_id"])

    def test_semantic_spans_merge_same_expression_across_lines(self):
        from ocr_server import _annotate_translation_contracts, _build_semantic_spans_for_block

        block = {
            "id": "b_span",
            "bbox": [50, 50, 420, 160],
            "source": "native",
            "role": "body",
            "lines": [
                {
                    "line_index": 0,
                    "bbox": [60, 60, 220, 80],
                    "line_text": "Transfer",
                    "hard_break_before": True,
                    "phrases": [
                        {
                            "bbox": [60, 60, 220, 80],
                            "texte": "Transfer",
                            "spans": [
                                {
                                    "texte": "Transfer",
                                    "bbox": [60, 60, 120, 80],
                                    "style": {"font": "Times-Bold", "size": 12.0, "color": "#000000", "flags": {"bold": True}},
                                }
                            ],
                        }
                    ],
                },
                {
                    "line_index": 1,
                    "bbox": [60, 84, 260, 104],
                    "line_text": "learning",
                    "hard_break_before": False,
                    "phrases": [
                        {
                            "bbox": [60, 84, 260, 104],
                            "texte": "learning",
                            "spans": [
                                {
                                    "texte": "learning",
                                    "bbox": [60, 84, 140, 104],
                                    "style": {"font": "Times-Bold", "size": 12.0, "color": "#000000", "flags": {"bold": True}},
                                }
                            ],
                        }
                    ],
                },
            ],
            "semantic_phrases": [],
        }

        _annotate_translation_contracts([block], page_context={"page_role": "body"})
        _build_semantic_spans_for_block(block)
        semantic_spans = block.get("semantic_spans", [])

        self.assertEqual(len(semantic_spans), 1)
        self.assertEqual(semantic_spans[0]["text"], "Transfer learning")
        self.assertTrue(semantic_spans[0]["multi_line"])
        self.assertEqual(semantic_spans[0]["line_indices"], [0, 1])

    def test_semantic_runs_merge_mixed_style_spans_within_phrase(self):
        from ocr_server import _annotate_translation_contracts, _build_semantic_runs_for_block

        block = {
            "id": "b_run",
            "bbox": [50, 50, 420, 120],
            "source": "native",
            "role": "body",
            "semantic_phrases": [
                {
                    "unit_id": "b_run:semantic_phrase:0",
                    "text": "Transfer learning",
                    "texte": "Transfer learning",
                    "bbox": [60, 60, 220, 80],
                    "line_indices": [0],
                    "spans": [
                        {
                            "unit_id": "sp_a",
                            "text": "Transfer",
                            "texte": "Transfer",
                            "bbox": [60, 60, 120, 80],
                            "expression_semantics": {"inline_class": "plain_text", "protected_inline": False},
                            "style": {"font": "Times-Bold", "size": 12.0, "color": "#000000", "flags": {"bold": True}},
                        },
                        {
                            "unit_id": "sp_b",
                            "text": "learning",
                            "texte": "learning",
                            "bbox": [125, 60, 220, 80],
                            "expression_semantics": {"inline_class": "plain_text", "protected_inline": False},
                            "style": {"font": "Times-Roman", "size": 12.0, "color": "#000000", "flags": {"bold": False}},
                        },
                    ],
                }
            ],
        }

        _annotate_translation_contracts([block], page_context={"page_role": "body"})
        _build_semantic_runs_for_block(block)
        semantic_runs = block.get("semantic_runs", [])

        self.assertEqual(len(semantic_runs), 1)
        self.assertEqual(semantic_runs[0]["text"], "Transfer learning")
        self.assertTrue(semantic_runs[0]["mixed_style"])
        self.assertEqual(block["semantic_phrases"][0]["semantic_run_count"], 1)

    def test_semantic_groups_merge_label_value_runs(self):
        from ocr_server import _build_semantic_groups_for_block

        block = {
            "id": "b_group",
            "semantic_phrases": [
                {
                    "unit_id": "b_group:semantic_phrase:0",
                    "semantic_runs": [
                        {
                            "unit_id": "run_a",
                            "text": "Model:",
                            "texte": "Model:",
                            "bbox": [60, 60, 120, 80],
                            "line_indices": [0],
                            "expression_semantics": {"inline_class": "plain_text", "protected_inline": False},
                        },
                        {
                            "unit_id": "run_b",
                            "text": "ResNet50",
                            "texte": "ResNet50",
                            "bbox": [125, 60, 220, 80],
                            "line_indices": [0],
                            "expression_semantics": {"inline_class": "technical_inline", "protected_inline": False},
                        },
                    ],
                }
            ],
        }

        _build_semantic_groups_for_block(block)
        semantic_groups = block.get("semantic_groups", [])

        self.assertEqual(len(semantic_groups), 1)
        self.assertEqual(semantic_groups[0]["group_class"], "label_value")
        self.assertEqual(semantic_groups[0]["text"], "Model: ResNet50")
        self.assertEqual(block["semantic_phrases"][0]["semantic_group_count"], 1)

    def test_hierarchical_reconstruction_classifies_editorial_block(self):
        reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        reconstructor.pixel_to_point = 72.0 / 150.0

        block = {
            "id": "b_editorial",
            "role": "body",
            "editorial_semantics": {"flow_class": "editorial_body", "reflowable": True},
            "lines": [],
        }

        self.assertEqual(
            reconstructor._classify_block_for_reconstruction(block, page_data={"page_role": "body"}),
            "editorial",
        )

    def test_build_line_templates_uses_source_line_geometry(self):
        reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        reconstructor.pixel_to_point = 72.0 / 150.0
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        block = {
            "id": "b_lines",
            "bbox": [20, 20, 220, 120],
            "alignment": "left",
            "lines": [
                {"bbox": [20, 20, 220, 40], "indent_px": 0.0, "hard_break_before": False},
                {"bbox": [30, 45, 220, 65], "indent_px": 10.0, "hard_break_before": True},
            ],
        }

        geometry_ctx = reconstructor._build_block_geometry_context(page, {"page_role": "body"}, block)
        templates = reconstructor._build_line_templates(block, geometry_ctx)
        doc.close()

        self.assertEqual(len(templates), 2)
        self.assertTrue(templates[0].is_first_paragraph_line)
        self.assertTrue(templates[1].is_first_paragraph_line)
        self.assertEqual(templates[1].paragraph_index, 1)
        self.assertGreater(templates[1].indent_px, 0.0)

    def test_build_block_reconstruction_plan_prefers_semantic_groups(self):
        reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        reconstructor.pixel_to_point = 72.0 / 150.0
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        block = {
            "id": "b_plan",
            "role": "body",
            "bbox": [20, 20, 220, 120],
            "text": "Model: ResNet50",
            "translated_text": "Modele : ResNet50",
            "lines": [
                {"bbox": [20, 20, 220, 40], "line_text": "Model: ResNet50", "phrases": []},
            ],
            "semantic_groups": [
                {
                    "unit_id": "group_1",
                    "group_class": "label_value",
                    "text": "Model: ResNet50",
                    "translated_text": "Modele : ResNet50",
                    "bbox": [20, 20, 220, 40],
                    "line_indices": [0],
                    "editorial_semantics": {"reflowable": True},
                    "structural_context": {"block_unit_id": "b_plan", "phrase_unit_id": "phrase_1"},
                }
            ],
            "semantic_phrases": [
                {
                    "unit_id": "phrase_1",
                    "text": "Model: ResNet50",
                    "translated_text": "Modele : ResNet50",
                    "bbox": [20, 20, 220, 40],
                    "line_indices": [0],
                    "structural_context": {"block_unit_id": "b_plan", "phrase_unit_id": "phrase_1"},
                }
            ],
        }

        plan = reconstructor._build_block_reconstruction_plan(page, {"page_role": "body"}, block, target_lang="fr")
        doc.close()

        self.assertEqual(plan.block_type, "editorial")
        self.assertEqual(len(plan.units), 1)
        self.assertEqual(plan.units[0].group_class, "label_value")
        self.assertEqual(plan.units[0].text_translated, "Modele : ResNet50")

    def test_build_block_reconstruction_plan_prefers_semantic_phrases_for_anchored_text_body(self):
        reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        reconstructor.pixel_to_point = 72.0 / 150.0
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        block = {
            "id": "b_anchor_phrases",
            "role": "body",
            "bbox": [20, 20, 220, 120],
            "render_policy": "anchored_text",
            "text": "Sentence one. Sentence two.",
            "translated_text": "Phrase un. Phrase deux.",
            "lines": [
                {"bbox": [20, 20, 220, 40], "line_text": "Sentence one.", "phrases": []},
                {"bbox": [20, 44, 220, 64], "line_text": "Sentence two.", "phrases": []},
            ],
            "semantic_groups": [
                {
                    "unit_id": "g0",
                    "group_class": "editorial_group",
                    "text": "Phrase un.",
                    "translated_text": "Phrase un.",
                    "bbox": [20, 20, 120, 40],
                    "line_indices": [0],
                    "structural_context": {"block_unit_id": "b_anchor_phrases", "phrase_unit_id": "p0"},
                },
                {
                    "unit_id": "g1",
                    "group_class": "editorial_group",
                    "text": "Phrase deux.",
                    "translated_text": "Phrase deux.",
                    "bbox": [20, 44, 120, 64],
                    "line_indices": [1],
                    "structural_context": {"block_unit_id": "b_anchor_phrases", "phrase_unit_id": "p1"},
                },
            ],
            "semantic_phrases": [
                {
                    "unit_id": "p0",
                    "text": "Phrase un.",
                    "translated_text": "Phrase un.",
                    "bbox": [20, 20, 220, 40],
                    "line_indices": [0],
                    "editorial_semantics": {"reflowable": True},
                    "structural_context": {"block_unit_id": "b_anchor_phrases", "phrase_unit_id": "p0"},
                },
                {
                    "unit_id": "p1",
                    "text": "Phrase deux.",
                    "translated_text": "Phrase deux.",
                    "bbox": [20, 44, 220, 64],
                    "line_indices": [1],
                    "editorial_semantics": {"reflowable": True},
                    "structural_context": {"block_unit_id": "b_anchor_phrases", "phrase_unit_id": "p1"},
                },
            ],
        }

        plan = reconstructor._build_block_reconstruction_plan(page, {"page_role": "body"}, block, target_lang="fr")
        doc.close()

        self.assertEqual([unit.unit_id for unit in plan.units], ["p0", "p1"])

    def test_hierarchical_reconstruction_classifies_table_from_descriptor_group_ids(self):
        reconstructor = DocumentReconstructor.__new__(DocumentReconstructor)
        reconstructor.pixel_to_point = 72.0 / 150.0

        block = {
            "id": "b_tableish",
            "role": "body",
            "descriptor_group_ids": {"cell_id": "cell_7", "table_row_group_id": "row_2"},
            "lines": [],
        }

        self.assertEqual(
            reconstructor._classify_block_for_reconstruction(block, page_data={"page_role": "body"}),
            "table",
        )

    def test_symbolic_visual_block_is_routed_to_preserve_renderer(self):
        reconstructor = DocumentReconstructor()
        block = {
            "id": "b_symbolic",
            "role": "title",
            "render_policy": "anchored_text",
            "text": "W 2 W 2 W 2 11 12 13",
            "translated_text": "W 2 W 2 W 2 11 12 13",
            "semantic_phrases": [
                {
                    "unit_id": "sym0",
                    "text": "W 2 W 2 W 2 11 12 13",
                    "translated_text": "W 2 W 2 W 2 11 12 13",
                    "bbox": [20, 20, 220, 40],
                    "line_indices": [0],
                    "structural_context": {"block_unit_id": "b_symbolic", "phrase_unit_id": "sym0"},
                }
            ],
        }

        self.assertEqual(
            reconstructor._classify_block_for_reconstruction(block, {"page_role": "body"}),
            "code",
        )
        self.assertTrue(reconstructor._block_supported_by_hierarchical_engine(block, {"page_role": "body"}))

    def test_render_hierarchical_editorial_block_emits_draw_ops(self):
        reconstructor = DocumentReconstructor()
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        block = {
            "id": "b_render",
            "role": "body",
            "bbox": [20, 20, 220, 120],
            "text": "This is a source sentence. Another source sentence.",
            "translated_text": "Ceci est une phrase traduite. Une autre phrase traduite.",
            "style": {"font": "Times-Roman", "size": 12.0, "color": "#000000", "flags": {}},
            "lines": [
                {"bbox": [20, 20, 220, 40], "line_text": "This is a source sentence.", "phrases": []},
                {"bbox": [20, 44, 220, 64], "line_text": "Another source sentence.", "phrases": []},
            ],
            "semantic_phrases": [
                {
                    "unit_id": "b_render:p0",
                    "text": "This is a source sentence.",
                    "translated_text": "Ceci est une phrase traduite.",
                    "bbox": [20, 20, 220, 40],
                    "line_indices": [0],
                    "editorial_semantics": {"reflowable": True},
                    "editorial_relations": {"with_previous": {"relation": "separate", "continuation": False}},
                    "structural_context": {"block_unit_id": "b_render", "phrase_unit_id": "b_render:p0", "paragraph_id": "para0"},
                },
                {
                    "unit_id": "b_render:p1",
                    "text": "Another source sentence.",
                    "translated_text": "Une autre phrase traduite.",
                    "bbox": [20, 44, 220, 64],
                    "line_indices": [1],
                    "editorial_semantics": {"reflowable": True},
                    "editorial_relations": {"with_previous": {"relation": "paragraph_break", "continuation": False}},
                    "structural_context": {"block_unit_id": "b_render", "phrase_unit_id": "b_render:p1", "paragraph_id": "para1"},
                },
            ],
        }

        plan = reconstructor._build_block_reconstruction_plan(page, {"page_role": "body"}, block, target_lang="fr")
        ops = reconstructor._render_hierarchical_block_plan(page, plan)
        findings = reconstructor._validate_block_layout(plan, ops)
        doc.close()

        self.assertTrue(any(op.op_type == "erase_rect" for op in ops))
        self.assertTrue(any(op.op_type == "draw_text_run" for op in ops))
        self.assertFalse(any(finding["type"] == "overflow" for finding in findings))

    def test_render_hierarchical_code_block_prefers_matching_overlay(self):
        reconstructor = DocumentReconstructor()
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            overlay_path = tmp.name
        try:
            Image.new("RGB", (40, 20), "white").save(overlay_path)
            block = {
                "id": "b_code",
                "role": "body",
                "unit_type": "code_visible",
                "bbox": [20, 20, 220, 80],
                "text": "x = foo()",
                "lines": [
                    {
                        "bbox": [20, 20, 220, 40],
                        "unit_type": "code_visible",
                        "line_text": "x = foo()",
                        "phrases": [
                            {
                                "unit_type": "code_visible",
                                "texte": "x = foo()",
                                "style": {"font": "Courier", "size": 12.0, "flags": {"monospace": True}},
                            }
                        ],
                    }
                ],
            }
            page_data = {
                "page_role": "body",
                "immutable_overlays": [
                    {"bbox": [20, 20, 220, 80], "path": overlay_path, "kind": "code_block_locked"}
                ],
            }
            plan = reconstructor._build_block_reconstruction_plan(page, page_data, block, target_lang="fr")
            ops = reconstructor._render_hierarchical_block_plan(page, plan)
        finally:
            doc.close()
            if os.path.exists(overlay_path):
                os.unlink(overlay_path)

        self.assertEqual(plan.block_type, "code")
        self.assertTrue(any(op.op_type == "draw_overlay_image" for op in ops))

    def test_render_hierarchical_table_block_stays_inside_cell_bbox(self):
        reconstructor = DocumentReconstructor()
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        block = {
            "id": "b_table",
            "role": "body",
            "bbox": [20, 20, 160, 80],
            "text": "Header value",
            "translated_text": "Valeur d'en-tete",
            "alignment": "center",
            "unit_type": "table_value_cell",
            "descriptor_group_ids": {"cell_id": "cell_1", "table_row_group_id": "row_1"},
            "descriptor_page_organization": {
                "table_row_groups": [
                    {
                        "id": "row_1",
                        "bbox": [20, 20, 160, 80],
                        "cells": [
                            {"id": "cell_1", "bbox": [20, 20, 160, 80], "block_id": "b_table"}
                        ],
                    }
                ]
            },
            "lines": [
                {"bbox": [20, 20, 160, 40], "line_text": "Header value", "phrases": []},
            ],
            "semantic_phrases": [
                {
                    "unit_id": "b_table:p0",
                    "text": "Header value",
                    "translated_text": "Valeur d'en-tete",
                    "bbox": [20, 20, 160, 40],
                    "line_indices": [0],
                    "editorial_semantics": {"reflowable": True},
                    "structural_context": {"block_unit_id": "b_table", "phrase_unit_id": "b_table:p0"},
                }
            ],
        }

        plan = reconstructor._build_block_reconstruction_plan(page, {"page_role": "body"}, block, target_lang="fr")
        ops = reconstructor._render_hierarchical_block_plan(page, plan)
        findings = reconstructor._validate_block_layout(plan, ops)
        doc.close()

        self.assertEqual(plan.block_type, "table")
        self.assertTrue(any(op.op_type == "erase_rect" for op in ops))
        self.assertTrue(any(op.op_type == "draw_text_run" for op in ops))
        self.assertFalse(any(finding["type"] == "overflow" for finding in findings))

    def test_render_hierarchical_editorial_block_preserves_mixed_style_fragments(self):
        reconstructor = DocumentReconstructor()
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        block = {
            "id": "b_mixed",
            "role": "body",
            "bbox": [20, 20, 220, 80],
            "text": "Transfer learning",
            "translated_text": "Apprentissage profond",
            "style": {"font": "Times-Roman", "size": 12.0, "color": "#000000", "flags": {}},
            "lines": [{"bbox": [20, 20, 220, 40], "line_text": "Transfer learning", "phrases": []}],
            "semantic_runs": [
                {
                    "unit_id": "run_mixed",
                    "text": "Apprentissage profond",
                    "translated_text": "Apprentissage profond",
                    "bbox": [20, 20, 220, 40],
                    "line_indices": [0],
                    "fragments": [
                        {"unit_id": "frag_a", "text": "Apprentissage", "translated_text": "Apprentissage", "style": {"font": "Times-Bold", "size": 12.0, "color": "#000000", "flags": {"bold": True}}},
                        {"unit_id": "frag_b", "text": "profond", "translated_text": "profond", "style": {"font": "Times-Italic", "size": 12.0, "color": "#000000", "flags": {"italic": True}}},
                    ],
                    "editorial_semantics": {"reflowable": True},
                    "structural_context": {"block_unit_id": "b_mixed", "phrase_unit_id": "b_mixed:p0"},
                }
            ],
        }

        plan = reconstructor._build_block_reconstruction_plan(page, {"page_role": "body"}, block, target_lang="fr")
        ops = reconstructor._render_hierarchical_block_plan(page, plan)
        doc.close()

        draw_ops = [op for op in ops if op.op_type == "draw_text_run"]
        self.assertGreaterEqual(len(draw_ops), 2)
        fonts = [str((op.style or {}).get("font") or "") for op in draw_ops]
        self.assertTrue(any("Bold" in font for font in fonts))
        self.assertTrue(any("Italic" in font for font in fonts))

    def test_translated_line_units_do_not_use_bbox_anchored_mode(self):
        reconstructor = DocumentReconstructor()
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        block = {
            "id": "b_line_units",
            "role": "body",
            "bbox": [20, 20, 220, 90],
            "render_policy": "anchored_text",
            "lines": [
                {
                    "bbox": [20, 20, 220, 35],
                    "line_text": "A long source line",
                    "translated_text": "Une ligne traduite assez longue pour forcer un repli propre",
                    "layout_attributes": {"horizontal_anchor": "left", "vertical_anchor": "top"},
                },
                {
                    "bbox": [20, 40, 220, 55],
                    "line_text": "Another source line",
                    "translated_text": "Une autre ligne traduite",
                    "layout_attributes": {"horizontal_anchor": "left", "vertical_anchor": "top"},
                },
            ],
            "semantic_phrases": [
                {"unit_id": "p0", "text": "A long source line", "translated_text": "Une ligne traduite assez longue", "bbox": [20, 20, 220, 35], "line_indices": [0, 1]},
                {"unit_id": "p1", "text": "Another source line", "translated_text": "Une autre ligne traduite", "bbox": [20, 40, 220, 55], "line_indices": [1, 2]},
            ],
        }

        plan = reconstructor._build_block_reconstruction_plan(page, {"page_role": "body"}, block, target_lang="fr")
        renderer = reconstructor._select_block_renderer(plan)
        self.assertGreaterEqual(sum(1 for unit in plan.units if unit.unit_type == "translated_line"), 1)
        self.assertFalse(renderer._should_render_bbox_anchored(plan))
        doc.close()

    def test_external_flow_units_use_relative_slot_mode(self):
        reconstructor = DocumentReconstructor()
        doc = fitz.open()
        page = doc.new_page(width=300, height=200)
        block = {
            "id": "b_ext_slot",
            "role": "body",
            "bbox": [20, 20, 260, 80],
            "style": {"font": "helv", "size": 10.0, "color": "#000000", "flags": {}},
        }
        units = [
            PlacableUnit(
                unit_id="u0",
                unit_type="external_label",
                source_kind="page_external_segment",
                parent_unit_id=None,
                block_unit_id="b_ext_slot",
                phrase_unit_id="p0",
                line_indices=[0],
                text_source="Alpha",
                text_translated="Alpha",
                role="body",
                style={"font": "helv", "size": 10.0, "color": "#000000", "flags": {}},
                relative_bbox=(20.0, 20.0, 120.0, 34.0),
                render_policy="external_flow",
                anchor_horizontal="start",
                metadata={"segment_type": "label"},
            ),
            PlacableUnit(
                unit_id="u1",
                unit_type="external_page",
                source_kind="page_external_segment",
                parent_unit_id=None,
                block_unit_id="b_ext_slot",
                phrase_unit_id="p0",
                line_indices=[0],
                text_source="10",
                text_translated="10",
                role="body",
                style={"font": "helv", "size": 10.0, "color": "#000000", "flags": {}},
                relative_bbox=(200.0, 20.0, 220.0, 34.0),
                render_policy="external_flow",
                anchor_horizontal="end",
                metadata={"segment_type": "page"},
            ),
        ]
        plan = reconstructor._build_block_reconstruction_plan(page, {"page_role": "body"}, {"id": "b_ext_slot", "role": "body", "bbox": [20, 20, 260, 80]}, target_lang="fr")
        plan = plan.__class__(**{**plan.__dict__, "units": units})
        renderer = reconstructor._select_block_renderer(plan)
        self.assertTrue(renderer._should_render_relative_slot_mode(plan))
        ops = renderer.render(page, plan)
        doc.close()

        text_ops = [op for op in ops if op.op_type == "draw_text_run"]
        self.assertGreaterEqual(len(text_ops), 1)

    def test_linewise_fallback_wraps_translated_lines_across_templates(self):
        reconstructor = DocumentReconstructor()
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        block = {
            "id": "b_line_wrap",
            "role": "body",
            "bbox": [20, 20, 180, 90],
            "style": {"font": "Times-Roman", "size": 12.0, "color": "#000000", "flags": {}},
            "lines": [
                {
                    "bbox": [20, 20, 120, 34],
                    "line_text": "source",
                    "translated_text": "Une ligne traduite tres tres longue qui doit se replier sur plusieurs gabarits",
                    "style": {"font": "Times-Roman", "size": 12.0, "color": "#000000", "flags": {}},
                },
                {
                    "bbox": [20, 38, 120, 52],
                    "line_text": "source 2",
                    "translated_text": "Ligne suivante",
                    "style": {"font": "Times-Roman", "size": 12.0, "color": "#000000", "flags": {}},
                },
                {
                    "bbox": [20, 56, 120, 70],
                    "line_text": "source 3",
                    "translated_text": "Troisieme ligne",
                    "style": {"font": "Times-Roman", "size": 12.0, "color": "#000000", "flags": {}},
                },
            ],
            "semantic_phrases": [
                {"unit_id": "p0", "text": "source", "translated_text": "", "bbox": [20, 20, 120, 34], "line_indices": [0, 1]},
                {"unit_id": "p1", "text": "source 2", "translated_text": "", "bbox": [20, 38, 120, 52], "line_indices": [1, 2]},
            ],
        }

        plan = reconstructor._build_block_reconstruction_plan(page, {"page_role": "body"}, block, target_lang="fr")
        renderer = reconstructor._select_block_renderer(plan)
        ops = renderer._linewise_fallback(page, plan)
        doc.close()

        draw_ops = [op for op in ops if op.op_type == "draw_text_run"]
        self.assertGreaterEqual(len(draw_ops), 3)

    def test_heading_caption_and_annotation_blocks_are_supported_by_hierarchical_engine(self):
        reconstructor = DocumentReconstructor()
        cases = [
            (
                {
                    "id": "b_heading",
                    "role": "section_heading",
                    "bbox": [20, 20, 220, 60],
                    "text": "PART B",
                    "translated_text": "PARTIE B",
                    "editorial_semantics": {"heading_like": True},
                    "semantic_phrases": [{"unit_id": "h0", "text": "PARTIE B", "translated_text": "PARTIE B", "bbox": [20, 20, 220, 60], "line_indices": [0], "structural_context": {"block_unit_id": "b_heading", "phrase_unit_id": "h0"}}],
                },
                "heading",
            ),
            (
                {
                    "id": "b_caption",
                    "role": "figure_caption",
                    "bbox": [20, 70, 220, 110],
                    "text": "Figure 1",
                    "translated_text": "Figure 1",
                    "editorial_semantics": {"caption_like": True},
                    "semantic_phrases": [{"unit_id": "c0", "text": "Figure 1", "translated_text": "Figure 1", "bbox": [20, 70, 220, 110], "line_indices": [0], "structural_context": {"block_unit_id": "b_caption", "phrase_unit_id": "c0"}}],
                },
                "caption",
            ),
            (
                {
                    "id": "b_annotation",
                    "role": "body",
                    "bbox": [20, 120, 220, 160],
                    "text": "Pool proj",
                    "translated_text": "Projet pool",
                    "editorial_semantics": {"anchored_annotation": True},
                    "semantic_phrases": [{"unit_id": "a0", "text": "Projet pool", "translated_text": "Projet pool", "bbox": [20, 120, 220, 160], "line_indices": [0], "structural_context": {"block_unit_id": "b_annotation", "phrase_unit_id": "a0"}}],
                },
                "annotation",
            ),
        ]
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        try:
            for block, expected_type in cases:
                plan = reconstructor._build_block_reconstruction_plan(page, {"page_role": "body"}, block, target_lang="fr")
                ops = reconstructor._render_hierarchical_block_plan(page, plan)
                self.assertEqual(plan.block_type, expected_type)
                self.assertTrue(reconstructor._block_supported_by_hierarchical_engine(block, {"page_role": "body"}))
                self.assertTrue(any(op.op_type == "draw_text_run" for op in ops))
        finally:
            doc.close()

    def test_editorial_renderer_compacts_first_line_after_heading_to_body_relation(self):
        reconstructor = DocumentReconstructor()
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        block = {
            "id": "b_body_after_heading",
            "role": "body",
            "bbox": [20, 80, 220, 160],
            "text": "Source paragraph text.",
            "translated_text": "Texte de paragraphe traduit.",
            "editorial_relations": {"with_previous": {"relation": "heading_to_body", "neighbor_id": "heading_1"}},
            "lines": [
                {"bbox": [20, 120, 220, 140], "line_text": "Source paragraph text.", "phrases": []},
            ],
            "semantic_phrases": [
                {
                    "unit_id": "body_p0",
                    "text": "Texte de paragraphe traduit.",
                    "translated_text": "Texte de paragraphe traduit.",
                    "bbox": [20, 120, 220, 140],
                    "line_indices": [0],
                    "editorial_semantics": {"reflowable": True},
                    "structural_context": {"block_unit_id": "b_body_after_heading", "phrase_unit_id": "body_p0"},
                }
            ],
        }
        plan = reconstructor._build_block_reconstruction_plan(page, {"page_role": "body"}, block, target_lang="fr")
        ops = reconstructor._render_hierarchical_block_plan(page, plan)
        doc.close()

        draw_ops = [op for op in ops if op.op_type == "draw_text_run"]
        self.assertTrue(draw_ops)
        first_y0 = min(op.bbox[1] for op in draw_ops)
        self.assertLess(first_y0, 50.0)

    def test_validate_block_layout_reports_protected_overlap(self):
        reconstructor = DocumentReconstructor()
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        plan = reconstructor._build_block_reconstruction_plan(
            page,
            {"page_role": "body"},
            {
                "id": "b_overlap",
                "role": "body",
                "bbox": [20, 20, 220, 100],
                "text": "abc",
                "translated_text": "abc",
                "protected_regions": [{"bbox": [40, 30, 120, 60]}],
                "semantic_phrases": [
                    {
                        "unit_id": "p0",
                        "text": "abc",
                        "translated_text": "abc",
                        "bbox": [20, 20, 220, 40],
                        "line_indices": [0],
                        "structural_context": {"block_unit_id": "b_overlap", "phrase_unit_id": "p0"},
                    }
                ],
                "lines": [{"bbox": [20, 20, 220, 40], "line_text": "abc", "phrases": []}],
            },
            target_lang="fr",
        )
        findings = reconstructor._validate_block_layout(
            plan,
            [
                BlockRenderOp(
                    op_type="draw_text_run",
                    block_id="b_overlap",
                    unit_id="p0",
                    bbox=(24.0, 16.0, 50.0, 27.0),
                    text="abc",
                    style={},
                )
            ],
        )
        doc.close()
        self.assertTrue(any(finding["type"] == "protected_overlap" for finding in findings))

    def test_justify_expands_only_eligible_gaps(self):
        reconstructor = DocumentReconstructor()
        doc = fitz.open()
        page = doc.new_page(width=400, height=200)
        block = {
            "id": "b_justify",
            "role": "body",
            "bbox": [20, 20, 350, 70],
            "alignment": "justify",
            "text": "Label: value more words",
            "translated_text": "Label: valeur plus mots",
            "lines": [{"bbox": [20, 20, 350, 45], "line_text": "Label: value more words", "phrases": []}],
            "semantic_groups": [
                {
                    "unit_id": "g1",
                    "group_class": "label_value",
                    "text": "Label: valeur",
                    "translated_text": "Label: valeur",
                    "bbox": [20, 20, 160, 45],
                    "line_indices": [0],
                    "editorial_semantics": {"reflowable": True},
                    "structural_context": {"block_unit_id": "b_justify", "phrase_unit_id": "p0"},
                },
                {
                    "unit_id": "g2",
                    "text": "plus mots",
                    "translated_text": "plus mots",
                    "bbox": [170, 20, 280, 45],
                    "line_indices": [0],
                    "editorial_semantics": {"reflowable": True},
                    "structural_context": {"block_unit_id": "b_justify", "phrase_unit_id": "p0"},
                },
            ],
        }
        plan = reconstructor._build_block_reconstruction_plan(page, {"page_role": "body"}, block, target_lang="fr")
        ops = reconstructor._render_hierarchical_block_plan(page, plan)
        doc.close()

        draw_ops = [op for op in ops if op.op_type == "draw_text_run"]
        self.assertGreaterEqual(len(draw_ops), 2)
        xs = [op.metadata["point"][0] for op in draw_ops]
        self.assertGreater(max(xs) - min(xs), 40.0)

    def test_try_render_hierarchical_item_plan_renders_once_and_marks_block(self):
        reconstructor = DocumentReconstructor()
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        block = {
            "id": "b_once",
            "role": "section_heading",
            "bbox": [20, 20, 220, 60],
            "text": "PART B",
            "translated_text": "PARTIE B",
            "editorial_semantics": {"heading_like": True},
            "semantic_phrases": [
                {
                    "unit_id": "u0",
                    "text": "PARTIE B",
                    "translated_text": "PARTIE B",
                    "bbox": [20, 20, 220, 60],
                    "line_indices": [0],
                    "structural_context": {"block_unit_id": "b_once", "phrase_unit_id": "u0"},
                }
            ],
        }
        plan = reconstructor._build_block_reconstruction_plan(page, {"page_role": "body"}, block, target_lang="fr")
        hierarchical_plans = {"b_once": plan}
        rendered_ids = set()
        forbidden = []
        item = {"source_block_id": "b_once"}

        first = reconstructor._try_render_hierarchical_item_plan(page, item, hierarchical_plans, rendered_ids, forbidden_rects=forbidden, debug_store={})
        second = reconstructor._try_render_hierarchical_item_plan(page, item, hierarchical_plans, rendered_ids, forbidden_rects=forbidden, debug_store={})
        doc.close()

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertIn("b_once", rendered_ids)
        self.assertTrue(forbidden)

    def test_try_render_hierarchical_item_plan_falls_back_on_severe_findings(self):
        reconstructor = DocumentReconstructor()
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        block = {
            "id": "b_fallback",
            "role": "section_heading",
            "bbox": [20, 20, 220, 60],
            "text": "PART B",
            "translated_text": "PARTIE B",
            "editorial_semantics": {"heading_like": True},
            "semantic_phrases": [
                {
                    "unit_id": "u0",
                    "text": "PARTIE B",
                    "translated_text": "PARTIE B",
                    "bbox": [20, 20, 220, 60],
                    "line_indices": [0],
                    "structural_context": {"block_unit_id": "b_fallback", "phrase_unit_id": "u0"},
                }
            ],
        }
        plan = reconstructor._build_block_reconstruction_plan(page, {"page_role": "body"}, block, target_lang="fr")
        hierarchical_plans = {"b_fallback": plan}
        rendered_ids = set()
        item = {"source_block_id": "b_fallback"}
        committed = []

        reconstructor._render_hierarchical_block_plan = lambda page, plan: [
            BlockRenderOp(
                op_type="draw_text_run",
                block_id="b_fallback",
                unit_id="u0",
                bbox=(20.0, 20.0, 90.0, 40.0),
                text="PARTIE B",
                style={},
            )
        ]
        reconstructor._validate_block_layout = lambda plan, ops: [{"type": "text_overlap", "unit_id": "u0"}]
        reconstructor._commit_block_draw_ops = lambda page, ops: committed.append(list(ops))

        rendered = reconstructor._try_render_hierarchical_item_plan(
            page,
            item,
            hierarchical_plans,
            rendered_ids,
            forbidden_rects=[],
            debug_store={},
        )
        doc.close()

        self.assertFalse(rendered)
        self.assertFalse(committed)
        self.assertNotIn("b_fallback", rendered_ids)

    def test_plain_page_number_is_not_immutable_inline_overlay(self):
        self.assertFalse(_is_immutable_inline_text("3"))
        self.assertFalse(_is_immutable_inline_text("92"))
        self.assertFalse(_is_immutable_inline_text("4"))

    def test_parenthetical_editorial_word_is_not_equation_or_immutable_overlay(self):
        self.assertFalse(_is_equation_like_text("(weights)"))
        self.assertFalse(_is_immutable_inline_text("(weights)"))


if __name__ == "__main__":
    unittest.main()
