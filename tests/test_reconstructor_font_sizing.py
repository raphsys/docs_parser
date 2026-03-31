import unittest

import fitz

from reconstructor import DocumentReconstructor


class ReconstructorFontSizingTests(unittest.TestCase):
    class _FakePage:
        def __init__(self):
            self.number = 0
            self.rect = fitz.Rect(0, 0, 400, 600)

    def test_prefers_layout_ai_line_height_for_sensitive_item_types(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "title",
            "descriptor_typographic_class": "section_title",
            "layout_ai_text_line_height_pt": 24.0,
            "layout_ai_block_height_pt": 52.0,
            "source_lines": ["Chapter title"],
            "source_line_count": 1,
        }
        fs = reconstructor._preferred_fontsize_for_item(item, {}, 26.0, "ocr")
        self.assertGreaterEqual(fs, 21.0)

    def test_keeps_native_font_size_when_pdf_style_is_available(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "descriptor_typographic_class": "editorial_body",
            "layout_ai_text_line_height_pt": 18.0,
            "source_lines": ["Body line"],
            "source_line_count": 1,
        }
        fs = reconstructor._preferred_fontsize_for_item(item, {"size": 11.0}, 14.0, "native")
        self.assertEqual(fs, 11.0)

    def test_primary_v3_toc_item_preserves_extracted_fontsize(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "source": "native",
            "descriptor_typographic_class": "editorial_body",
            "descriptor_v3_contract": {"primary_structure_family": "toc"},
            "descriptor_v3_render_unit": {"structure_priority": "primary"},
        }
        fs = reconstructor._normalized_fontsize_for_item(item, {"size": 13.0}, 16.0, "native")
        self.assertEqual(fs, 13.0)

    def test_primary_v3_toc_item_preserves_own_font_without_cache_flattening(self):
        reconstructor = DocumentReconstructor()
        item_a = {
            "role": "body",
            "source": "native",
            "descriptor_typographic_class": "editorial_body",
            "descriptor_group_ids": {"toc_entry_group_id": "toc_1"},
            "descriptor_v3_contract": {"primary_structure_family": "toc"},
            "descriptor_v3_render_unit": {"structure_priority": "primary"},
            "style": {"font": "FontA", "size": 12.0, "color": "#111111"},
        }
        item_b = {
            "role": "body",
            "source": "native",
            "descriptor_typographic_class": "editorial_body",
            "descriptor_group_ids": {"toc_entry_group_id": "toc_1"},
            "descriptor_v3_contract": {"primary_structure_family": "toc"},
            "descriptor_v3_render_unit": {"structure_priority": "primary"},
            "style": {"font": "FontB", "size": 12.0, "color": "#222222"},
        }
        style_a = reconstructor._normalized_style_for_item(item_a)
        style_b = reconstructor._normalized_style_for_item(item_b)
        self.assertEqual(style_a["font"], "FontA")
        self.assertEqual(style_b["font"], "FontB")

    def test_abbreviation_pair_preserves_extracted_fontsize_even_in_group(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "source": "native",
            "descriptor_typographic_class": "abbreviation_value",
            "descriptor_structural_role": "abbreviation_value",
            "descriptor_group_ids": {"same_row_group_id": "row_1"},
            "descriptor_v3_contract": {"primary_structure_family": "glossary_pairs"},
            "descriptor_v3_render_unit": {"structure_priority": "primary"},
        }
        fs = reconstructor._normalized_fontsize_for_item(item, {"size": 10.5}, 14.0, "native")
        self.assertEqual(fs, 10.5)

    def test_native_dense_paragraph_flow_has_high_min_font_floor(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "source": "native",
            "descriptor_v3_contract": {"primary_structure_family": "dense_paragraph_flow"},
            "descriptor_v3_render_unit": {"structure_priority": "secondary"},
        }
        min_fs = reconstructor._min_fontsize_for_item(item, 9.96, strict=False)
        self.assertGreaterEqual(min_fs, 9.3)

    def test_native_title_in_primary_structure_has_strict_font_floor(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "title",
            "source": "native",
            "descriptor_v3_contract": {"primary_structure_family": "chapter_opening"},
            "descriptor_v3_render_unit": {"structure_priority": "primary"},
        }
        min_fs = reconstructor._min_fontsize_for_item(item, 12.48, strict=True)
        self.assertGreaterEqual(min_fs, 11.9)

    def test_native_dense_paragraph_flow_keeps_right_padding_margin(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "source": "native",
            "descriptor_v3_contract": {"primary_structure_family": "dense_paragraph_flow"},
            "descriptor_v3_render_unit": {"structure_priority": "secondary"},
        }
        pad = reconstructor._line_right_padding_for_item(item, 9.96, strict=False)
        self.assertGreaterEqual(pad, 6.0)

    def test_non_fidelity_item_keeps_small_right_padding_margin(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "source": "ocr",
            "descriptor_v3_contract": {"primary_structure_family": "freeform_blocks"},
            "descriptor_v3_render_unit": {"structure_priority": "secondary"},
        }
        pad = reconstructor._line_right_padding_for_item(item, 10.0, strict=False)
        self.assertLessEqual(pad, 4.5)

    def test_exact_line_left_relief_allowed_for_translated_editorial_body(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "translated_block": True,
            "keep_exact_line": True,
            "descriptor_typographic_class": "editorial_body",
            "descriptor_structural_role": "body_paragraph",
        }
        self.assertTrue(reconstructor._allow_exact_line_left_relief(item))
        self.assertGreaterEqual(
            reconstructor._line_right_padding_for_item(item, 9.96, strict=False),
            10.0,
        )

    def test_exact_line_left_relief_not_allowed_for_abbreviation_entries(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "translated_block": True,
            "keep_exact_line": True,
            "descriptor_typographic_class": "editorial_body",
            "descriptor_structural_role": "abbreviation_value",
        }
        self.assertFalse(reconstructor._allow_exact_line_left_relief(item))

    def test_anchored_text_editorial_body_does_not_force_strict_line_items(self):
        reconstructor = DocumentReconstructor()
        self.assertFalse(
            reconstructor._allow_strict_line_items_for_anchored_text_body(
                "body",
                "editorial_body",
                "body_paragraph",
            )
        )

    def test_anchored_text_non_editorial_body_can_keep_strict_line_items(self):
        reconstructor = DocumentReconstructor()
        self.assertTrue(
            reconstructor._allow_strict_line_items_for_anchored_text_body(
                "body",
                "diagram_label",
                "diagram_label",
            )
        )

    def test_translated_body_lines_fit_source_slots_detects_overflow(self):
        reconstructor = DocumentReconstructor()
        line_entries = [
            {
                "text": "A very long translated body line that should not fit its narrow source slot at all",
                "bbox": [0, 0, 80, 20],
                "style": {"font": "helv", "size": 12.0, "color": "#000000"},
            }
        ]
        self.assertFalse(reconstructor._translated_body_lines_fit_source_slots(line_entries, "native"))

    def test_translated_body_lines_fit_source_slots_accepts_short_line(self):
        reconstructor = DocumentReconstructor()
        line_entries = [
            {
                "text": "Short line",
                "bbox": [0, 0, 180, 20],
                "style": {"font": "helv", "size": 12.0, "color": "#000000"},
            }
        ]
        self.assertTrue(reconstructor._translated_body_lines_fit_source_slots(line_entries, "native"))

    def test_preserves_double_column_lineation_for_narrative_body(self):
        reconstructor = DocumentReconstructor()
        page_data = {"layout_type": "double_column", "document_type": "book_page"}
        block = {
            "role": "body",
            "unit_type": "narrative_body",
            "render_policy": "paragraph_flow",
            "lines": [
                {
                    "line_text": (
                        "This is the original source paragraph with enough words to represent a real narrative block "
                        "laid out across a two column page for the reconstruction test while keeping a stable editorial "
                        "tone and a realistic amount of source material for the renderer."
                    )
                }
            ],
        }
        lines = [
            "Ceci est un paragraphe narratif de test",
            "traduit sur plusieurs lignes afin de",
            "conserver la geometrie editoriale de",
            "la colonne originale sans reflow brutal",
        ]
        self.assertTrue(
            reconstructor._should_preserve_double_column_lineation(
                page_data,
                block,
                "text_band",
                " ".join(lines),
                lines,
            )
        )

    def test_does_not_preserve_double_column_lineation_for_short_labels(self):
        reconstructor = DocumentReconstructor()
        page_data = {"layout_type": "double_column", "document_type": "book_page"}
        block = {
            "role": "body",
            "unit_type": "short_label",
            "render_policy": "paragraph_flow",
            "lines": [{"line_text": "Short label source text only."}],
        }
        lines = ["Etiquette courte", "sur deux lignes", "mais pas narrative", "donc non"]
        self.assertFalse(
            reconstructor._should_preserve_double_column_lineation(
                page_data,
                block,
                "text_band",
                " ".join(lines),
                lines,
            )
        )

    def test_keeps_local_source_slot_geometry_for_anchored_editorial_body_overflow(self):
        reconstructor = DocumentReconstructor()
        page_data = {"layout_type": "double_column", "document_type": "book_page"}
        block = {
            "role": "body",
            "render_policy": "anchored_text",
        }
        line_entries = [
            {
                "text": "Une ligne traduite plus longue que la ligne source mais encore reflowable localement",
                "bbox": [0, 0, 340, 20],
                "style": {"font": "helv", "size": 10.0, "color": "#000000"},
            },
            {
                "text": "Une seconde ligne en debordement modere qui doit rester dans le bloc source",
                "bbox": [0, 22, 340, 42],
                "style": {"font": "helv", "size": 10.0, "color": "#000000"},
            },
        ]
        self.assertTrue(
            reconstructor._should_keep_local_source_slot_geometry_for_anchored_body(
                page_data,
                block,
                "text_band",
                "editorial_body",
                "body_paragraph",
                line_entries,
                "native",
            )
        )

    def test_detects_meaningful_line_style_variation(self):
        reconstructor = DocumentReconstructor()
        line_entries = [
            {"text": "Regular line", "style": {"font": "JansonTextLTStd-Roman", "font_key_normalized": "jansontextltstdroman", "flags": {}}},
            {"text": "Bold line", "style": {"font": "JansonTextLTStd-Bold", "font_key_normalized": "jansontextltstdbold", "flags": {"bold": True}}},
        ]
        self.assertTrue(reconstructor._has_meaningful_line_style_variation(line_entries))

    def test_structured_source_lines_with_styles_preserves_nonempty_alignment(self):
        reconstructor = DocumentReconstructor()
        item = {
            "source_lines": ["Bold line", "", "Roman line"],
            "source_line_styles": [
                {"font": "JansonTextLTStd-Bold", "size": 15.0, "color": "#000000"},
                {"font": "Ignored", "size": 15.0, "color": "#000000"},
                {"font": "JansonTextLTStd-Roman", "size": 15.0, "color": "#000000"},
            ],
        }
        lines, styles = reconstructor._structured_source_lines_with_styles(item)
        self.assertEqual(lines, ["Bold line", "Roman line"])
        self.assertEqual(styles[0]["font"], "JansonTextLTStd-Bold")
        self.assertEqual(styles[1]["font"], "JansonTextLTStd-Roman")

    def test_partitions_translated_line_to_inline_segments(self):
        reconstructor = DocumentReconstructor()
        parts = reconstructor._partition_translated_line_to_segments(
            "SQL PRATIQUE. Copyright © 2018 par Anthony DeBarros.",
            [
                {"text": "PRACTICAL SQL.", "bbox": fitz.Rect(0, 0, 100, 10), "style": {"font": "Bold"}},
                {"text": "Copyright © 2018 by Anthony DeBarros.", "bbox": fitz.Rect(100, 0, 300, 10), "style": {"font": "Roman"}},
            ],
        )
        self.assertEqual(parts, ["SQL PRATIQUE.", "Copyright © 2018 par Anthony DeBarros."])

    def test_should_render_inline_style_segments_for_mixed_title_line(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "title",
            "translated_block": True,
            "descriptor_structural_role": "table_header_cell",
        }
        segments = [
            {"text": "PRACTICAL SQL.", "bbox": fitz.Rect(0, 0, 100, 10), "style": {"font": "Bold", "flags": {"bold": True}}},
            {"text": "Copyright © 2018 by Anthony DeBarros.", "bbox": fitz.Rect(100, 0, 300, 10), "style": {"font": "Roman", "flags": {"bold": False}}},
        ]
        self.assertTrue(reconstructor._should_render_inline_style_segments(item, segments))

    def test_conservative_right_padding_for_translated_form_body(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "translated_block": True,
            "page_data": {"layout_type": "table_dominant", "document_type": "form"},
        }
        self.assertGreaterEqual(reconstructor._line_right_padding_for_item(item, 11.25, strict=False), 8.0)

    def test_prefers_multiline_locked_editorial_block_for_overflowing_form_page(self):
        reconstructor = DocumentReconstructor()
        page_data = {"layout_type": "table_dominant", "document_type": "form"}
        block = {
            "role": "body",
            "render_policy": "anchored_text",
        }
        line_entries = [
            {
                "text": "Tous droits réservés. Aucune partie de cette oeuvre ne peut être reproduite ou transmise sous quelque forme que ce soit.",
                "bbox": [0, 0, 260, 20],
                "style": {"font": "helv", "size": 11.25, "color": "#000000"},
            },
            {
                "text": "moyens électroniques ou mécaniques, y compris photocopies, enregistrement ou système de stockage.",
                "bbox": [0, 24, 260, 44],
                "style": {"font": "helv", "size": 11.25, "color": "#000000"},
            },
        ]
        self.assertTrue(
            reconstructor._should_keep_multiline_locked_editorial_block(
                page_data=page_data,
                block=block,
                descriptor_layout_behavior="locked_in_cell",
                descriptor_structural_role="table_value_cell",
                descriptor_typographic_class="editorial_body",
                line_entries=line_entries,
                source="native",
                translated_block=True,
            )
        )

    def test_does_not_use_multiline_locked_editorial_block_for_short_fitting_form_lines(self):
        reconstructor = DocumentReconstructor()
        page_data = {"layout_type": "table_dominant", "document_type": "form"}
        block = {
            "role": "body",
            "render_policy": "anchored_text",
        }
        line_entries = [
            {
                "text": "ISBN-10: 1-59327-827-6",
                "bbox": [0, 0, 340, 20],
                "style": {"font": "helv", "size": 11.25, "color": "#000000"},
            },
            {
                "text": "ISBN-13: 978-1-59327-827-4",
                "bbox": [0, 24, 340, 44],
                "style": {"font": "helv", "size": 11.25, "color": "#000000"},
            },
        ]
        self.assertFalse(
            reconstructor._should_keep_multiline_locked_editorial_block(
                page_data=page_data,
                block=block,
                descriptor_layout_behavior="locked_in_cell",
                descriptor_structural_role="table_value_cell",
                descriptor_typographic_class="editorial_body",
                line_entries=line_entries,
                source="native",
                translated_block=True,
            )
        )

    def test_does_not_keep_local_source_slot_geometry_for_non_editorial_anchored_body(self):
        reconstructor = DocumentReconstructor()
        page_data = {"layout_type": "double_column", "document_type": "book_page"}
        block = {
            "role": "body",
            "render_policy": "anchored_text",
        }
        line_entries = [
            {
                "text": "Short label overflow",
                "bbox": [0, 0, 80, 20],
                "style": {"font": "helv", "size": 10.0, "color": "#000000"},
            },
            {
                "text": "Second label",
                "bbox": [0, 22, 80, 42],
                "style": {"font": "helv", "size": 10.0, "color": "#000000"},
            },
        ]
        self.assertFalse(
            reconstructor._should_keep_local_source_slot_geometry_for_anchored_body(
                page_data,
                block,
                "annotation_band",
                "diagram_label",
                "diagram_label",
                line_entries,
                "native",
            )
        )

    def test_uniform_preserved_line_fontsize_fits_all_lines_with_shared_size(self):
        reconstructor = DocumentReconstructor()
        lines = [
            "En ajoutant l'extension PostGIS a la base de donnees vous pouvez creer",
            "des donnees spatiales exportables en GeoJSON ou en shapefile",
            "un format facile a cartographier pour raconter une histoire",
            "avec une geometrie editoriale stable sur toute la colonne",
        ]
        widths = [410.0, 410.0, 410.0, 410.0]
        fs = reconstructor._fit_uniform_preserved_line_fontsize(
            lines,
            widths,
            15.0,
            "helv",
            None,
            overflow_limit=1.03,
            min_font_pt=11.0,
        )

        self.assertLess(fs, 15.0)
        self.assertGreaterEqual(fs, 11.0)
        for line, width in zip(lines, widths):
            self.assertLessEqual(
                reconstructor._measure_text_width(line, fs, "helv", None),
                width * 1.03 + 1e-6,
            )

    def test_preserve_line_style_variation_disables_uniform_line_fontsize(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "translated_block": True,
            "preserve_linebreaks": True,
            "keep_source_slot_geometry": True,
            "preserve_line_style_variation": True,
            "page_data": {"layout_type": "double_column", "document_type": "book_page"},
            "descriptor_region_type": "text",
            "source_line_count": 12,
        }
        self.assertFalse(reconstructor._should_use_uniform_preserved_line_fontsize(item))

    def test_typography_key_prefers_relation_group_ids(self):
        reconstructor = DocumentReconstructor()
        item = {
            "descriptor_typographic_class": "editorial_body",
            "descriptor_group_ids": {
                "paragraph_chain_group_id": "rel_paragraph_chain_3",
                "same_band_group_id": "rel_same_band_2",
            },
            "descriptor_region_id": "region_text_1",
            "source_block_id": "b1",
            "text": "Texte",
        }
        self.assertEqual(
            reconstructor._item_typography_key(item),
            ("editorial_body", "rel_paragraph_chain_3"),
        )

    def test_safe_mixed_allows_relation_driven_body_flow(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "translated_block": True,
            "descriptor_typographic_class": "editorial_body",
            "descriptor_group_ids": {"paragraph_chain_group_id": "rel_paragraph_chain_0"},
            "descriptor_region_type": "text_band",
            "descriptor_band_role": "content_band",
        }
        page_data = {"layout_type": "double_column", "document_type": "book_page"}
        self.assertTrue(
            reconstructor._should_allow_relation_flow_override(
                item,
                page_data,
                "safe_mixed",
                anchored_figure_page=False,
                table_locked_block=False,
            )
        )

    def test_safe_mixed_does_not_override_for_same_row_label_pairs(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "translated_block": True,
            "descriptor_typographic_class": "editorial_body",
            "descriptor_group_ids": {
                "same_band_group_id": "rel_same_band_1",
                "same_row_group_id": "rel_same_row_1",
            },
            "descriptor_region_type": "text_band",
            "descriptor_band_role": "content_band",
        }
        page_data = {"layout_type": "double_column", "document_type": "book_page"}
        self.assertFalse(
            reconstructor._should_allow_relation_flow_override(
                item,
                page_data,
                "safe_mixed",
                anchored_figure_page=False,
                table_locked_block=False,
            )
        )

    def test_relation_group_bbox_reads_descriptor_reconstruction_plan(self):
        reconstructor = DocumentReconstructor()
        item = {
            "descriptor_group_ids": {"paragraph_chain_group_id": "rel_paragraph_chain_0"},
            "descriptor_reconstruction_plan": {
                "relation_groups": {
                    "continues_paragraph": [
                        {
                            "id": "rel_paragraph_chain_0",
                            "member_ids": ["p1", "p2"],
                            "bbox": [320.0, 140.0, 560.0, 220.0],
                        }
                    ]
                }
            },
        }
        bbox = reconstructor._relation_group_bbox(item, "continues_paragraph")
        self.assertIsInstance(bbox, fitz.Rect)
        self.assertAlmostEqual(bbox.x0, 320.0 * reconstructor.pixel_to_point)
        self.assertAlmostEqual(bbox.y1, 220.0 * reconstructor.pixel_to_point)

    def test_abbreviation_key_requires_exact_slot_and_anchored_render(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "translated_block": True,
            "text": "LRN",
            "source_text": "LRN",
            "source_lines": ["LRN"],
            "descriptor_band_role": "content_band",
            "descriptor_group_render_mode": "flow_group",
            "descriptor_structural_role": "abbreviation_key",
            "descriptor_layout_behavior": "flow",
            "descriptor_typographic_class": "abbreviation_key",
            "style": {"font": "Times", "size": 11},
        }
        self.assertTrue(reconstructor._item_requires_exact_slot_render(item))
        self.assertTrue(reconstructor._item_requires_anchored_render(item, anchored_figure_page=False))

    def test_abbreviation_value_requires_anchored_render_but_not_exact_slot(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "translated_block": True,
            "text": "Normalisation des réponses locales",
            "source_text": "Local Response Normalization",
            "source_lines": ["Local Response Normalization"],
            "descriptor_band_role": "content_band",
            "descriptor_group_render_mode": "flow_group",
            "descriptor_structural_role": "abbreviation_value",
            "descriptor_layout_behavior": "flow",
            "descriptor_typographic_class": "abbreviation_value",
            "style": {"font": "Times", "size": 11},
        }
        self.assertFalse(reconstructor._item_requires_exact_slot_render(item))
        self.assertTrue(reconstructor._item_requires_anchored_render(item, anchored_figure_page=False))

if __name__ == "__main__":
    unittest.main()
