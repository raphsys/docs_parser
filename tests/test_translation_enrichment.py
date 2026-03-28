import unittest

import fitz

from block_typology import classify_block_typology
from page_policy_matrix import PagePolicyMatrix
from translator import DocumentTranslator
from reconstructor import DocumentReconstructor
from structure_extractor import LayoutV2Builder


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

    def test_fontsize_from_bbox_uses_source_height(self):
        fs = self.reconstructor._fontsize_from_bbox([0, 100, 10, 175])

        self.assertEqual(fs, 36.0)

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


if __name__ == "__main__":
    unittest.main()
