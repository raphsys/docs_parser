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

    def test_native_style_fidelity_locks_min_fontsize_to_original(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "source": "native",
            "descriptor_v3_contract": {"primary_structure_family": "dense_paragraph_flow"},
            "descriptor_v3_render_unit": {"structure_priority": "secondary"},
        }
        self.assertEqual(reconstructor._min_fontsize_for_item(item, 15.0, strict=False), 15.0)
        self.assertEqual(reconstructor._min_fontsize_for_item(item, 15.0, strict=True), 15.0)

    def test_native_structured_editorial_body_locks_fontsize(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "source": "native",
            "translated_block": True,
            "preserve_linebreaks": True,
            "use_structured_source_lines": True,
            "descriptor_typographic_class": "editorial_body",
            "descriptor_structural_role": "body_paragraph",
        }
        self.assertTrue(reconstructor._item_native_style_fidelity_mode(item))
        self.assertTrue(reconstructor._should_lock_fontsize_for_item(item))

    def test_native_structured_editorial_body_keeps_source_anchor(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "source": "native",
            "translated_block": True,
            "preserve_linebreaks": True,
            "use_structured_source_lines": True,
            "strict_bbox_mode": True,
            "descriptor_typographic_class": "editorial_body",
            "descriptor_structural_role": "body_paragraph",
            "source_line_count": 8,
            "page_data": {
                "layout_type": "double_column",
                "document_type": "mixed_unknown",
            },
        }
        self.assertTrue(reconstructor._should_keep_source_anchor_for_item(item))

    def test_translated_structured_table_band_body_keeps_source_anchor(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "source": "native",
            "translated_block": True,
            "preserve_linebreaks": True,
            "use_structured_source_lines": True,
            "strict_bbox_mode": True,
            "descriptor_typographic_class": "editorial_body",
            "descriptor_structural_role": "table_value_cell",
            "descriptor_band_role": "table_band",
            "source_line_count": 8,
            "page_data": {
                "layout_type": "table_dominant",
                "document_type": "form",
            },
        }
        self.assertTrue(reconstructor._should_keep_source_anchor_for_item(item))

    def test_reconstruct_translated_anchored_keeps_source_anchor_for_table_band_body(self):
        reconstructor = DocumentReconstructor()
        item = {
            "text": "Noms: DeBarros, Anthony, auteur.\nTitre: SQL pratique",
            "bbox": fitz.Rect(73.44, 487.2, 506.4, 633.6),
            "slots": [fitz.Rect(73.44, 487.2, 506.4, 498.24)],
            "slot_w_pt": 432.96,
            "slot_h_pt": 11.04,
            "slot_gap_x_pt": 0.0,
            "slot_gap_y_pt": 3.0,
            "row_start_x_pt": 73.44,
            "role": "body",
            "source": "native",
            "translated_block": True,
            "strict_bbox_mode": True,
            "preserve_linebreaks": True,
            "use_structured_source_lines": True,
            "source_lines": [
                "Noms: DeBarros, Anthony, auteur.",
                "Titre: SQL pratique",
            ],
            "source_line_count": 13,
            "descriptor_band_role": "table_band",
            "descriptor_structural_role": "table_value_cell",
            "descriptor_typographic_class": "editorial_body",
            "style": {"font": "helv", "size": 11.25, "color": "#000000", "flags": {}},
        }
        page_data = {
            "page": 3,
            "layout_type": "table_dominant",
            "document_type": "form",
            "blocks": [],
        }

        captured = []

        def fake_extract(_page_data):
            return [dict(item, page_data=page_data)]

        def fake_render_block_slots(*, page, item, anchor_y, left, right, zone_top, zone_bottom, **kwargs):
            captured.append((anchor_y, zone_bottom))
            return "", anchor_y + 20.0, fitz.Rect(item["bbox"]), [fitz.Rect(item["bbox"])]

        reconstructor._extract_block_slot_items = fake_extract
        reconstructor._item_requires_anchored_render = lambda *args, **kwargs: True
        reconstructor._group_visual_items = lambda items: ([], items)
        reconstructor._looks_like_toc_page = lambda _page_data: False
        reconstructor._render_block_slots = fake_render_block_slots

        doc = fitz.open()
        page = doc.new_page(width=600, height=800)
        reconstructor._reconstruct_translated_anchored(doc, page, page_data, debug_store=None, forbidden_rects=[])

        self.assertTrue(captured)
        self.assertAlmostEqual(captured[0][0], item["bbox"].y0)

    def test_translated_table_band_item_can_shift_to_avoid_overlap(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "translated_block": True,
            "strict_bbox_mode": True,
            "preserve_linebreaks": True,
            "descriptor_band_role": "table_band",
            "descriptor_structural_role": "table_value_cell",
        }
        self.assertTrue(reconstructor._should_shift_anchored_item_for_overlap(item))

    def test_structured_strict_translated_body_can_paginate_on_overflow(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "translated_block": True,
            "strict_bbox_mode": True,
            "preserve_linebreaks": True,
            "use_structured_source_lines": True,
            "descriptor_structural_role": "body_paragraph",
            "descriptor_typographic_class": "editorial_body",
        }
        self.assertTrue(reconstructor._should_paginate_strict_translated_item(item))

    def test_overflow_continuation_item_relaxes_strict_bbox(self):
        reconstructor = DocumentReconstructor()
        item = {
            "role": "body",
            "translated_block": True,
            "strict_bbox_mode": True,
            "preserve_sentence_integrity": True,
            "keep_source_slot_geometry": True,
        }
        continued = reconstructor._overflow_continuation_item(item)
        self.assertFalse(continued["strict_bbox_mode"])
        self.assertTrue(continued["allow_vertical_expand"])
        self.assertFalse(continued["preserve_sentence_integrity"])
        self.assertFalse(continued["keep_source_slot_geometry"])

    def test_split_text_to_preserve_fontsize_returns_multiple_lines_without_shrink(self):
        reconstructor = DocumentReconstructor()
        head, tail = reconstructor._split_text_to_preserve_fontsize(
            "This is a long translated sentence that must wrap locally while preserving the original font size",
            90.0,
            12.0,
            "helv",
            None,
        )
        self.assertTrue(head)
        self.assertTrue(tail)
        self.assertLessEqual(
            reconstructor._measure_text_width(head, 12.0, "helv", None),
            90.0,
        )

    def test_descriptor_v3_constraint_can_lock_fontsize(self):
        reconstructor = DocumentReconstructor()
        item = {
            "descriptor_v3_placement_constraints": [
                {
                    "font_size_policy": {
                        "mode": "lock",
                    }
                }
            ]
        }
        self.assertTrue(reconstructor._should_lock_fontsize_for_item(item))
        self.assertEqual(reconstructor._min_fontsize_for_item(item, 13.5), 13.5)

    def test_descriptor_v3_constraint_can_lock_source_anchor(self):
        reconstructor = DocumentReconstructor()
        item = {
            "translated_block": True,
            "descriptor_v3_placement_constraints": [
                {
                    "anchor_policy": {
                        "source_y_locked": True,
                    }
                }
            ],
        }
        self.assertTrue(reconstructor._should_keep_source_anchor_for_item(item))

    def test_descriptor_v3_constraint_can_request_pagination(self):
        reconstructor = DocumentReconstructor()
        item = {
            "translated_block": True,
            "strict_bbox_mode": True,
            "descriptor_v3_placement_constraints": [
                {
                    "overflow_policy": {
                        "mode": "paginate",
                    }
                }
            ],
        }
        self.assertTrue(reconstructor._should_paginate_strict_translated_item(item))

    def test_descriptor_v3_alignment_lock_preserves_alignment_when_line_is_wider_than_slot(self):
        reconstructor = DocumentReconstructor()
        item = {
            "descriptor_v3_placement_constraints": [
                {
                    "style_invariants": {
                        "preserve_alignment": True,
                    }
                }
            ]
        }
        self.assertTrue(reconstructor._should_preserve_alignment_for_item(item))
        applied, reason = reconstructor._resolve_applied_alignment(
            expected_alignment="right",
            line_w=140.0,
            left=20.0,
            right=100.0,
            preserve_alignment=reconstructor._should_preserve_alignment_for_item(item),
        )
        self.assertEqual(applied, "right")
        self.assertEqual(reason, "")
        x = reconstructor._compute_aligned_x(
            alignment="right",
            line_w=140.0,
            left=20.0,
            right=100.0,
            preferred_x=20.0,
            preserve_alignment=True,
        )
        self.assertLess(x, 20.0)

    def test_preserved_line_inline_segments_render_for_structured_body(self):
        reconstructor = DocumentReconstructor()
        reconstructor._rendered_signatures = set()
        reconstructor._style_audit_records = []
        captured = []
        reconstructor._safe_insert_text_dedup = lambda page, point, text, fontsize, fontname, color: captured.append((text, fontsize, fontname))
        doc = fitz.open()
        page = doc.new_page(width=300, height=200)
        item = {
            "text": "Alpha Beta",
            "bbox": fitz.Rect(20, 20, 180, 36),
            "slots": [fitz.Rect(20, 20, 180, 36)],
            "slot_h_pt": 16.0,
            "slot_gap_y_pt": 2.0,
            "slot_w_pt": 160.0,
            "row_start_x_pt": 20.0,
            "style": {"font": "helv", "size": 11.0, "color": "#000000"},
            "source": "native",
            "role": "body",
            "translated_block": True,
            "preserve_linebreaks": True,
            "use_structured_source_lines": True,
            "source_lines": ["Alpha Beta"],
            "source_line_styles": [{"font": "helv", "size": 11.0, "color": "#000000"}],
            "source_line_inline_segments": [[
                {
                    "text": "Alpha",
                    "bbox": fitz.Rect(20, 20, 68, 36),
                    "style": {"font": "helv", "size": 11.0, "color": "#000000", "flags": {"bold": True}},
                },
                {
                    "text": "Beta",
                    "bbox": fitz.Rect(72, 20, 112, 36),
                    "style": {"font": "courier", "size": 11.0, "color": "#000000", "flags": {"monospace": True}},
                },
            ]],
            "descriptor_v3_placement_constraints": [
                {
                    "style_invariants": {"preserve_span_variation": True},
                }
            ],
            "alignment": "left",
        }
        remaining, _, _, _ = reconstructor._render_block_slots(
            page=page,
            item=item,
            anchor_y=20.0,
            left=0.0,
            right=300.0,
            zone_top=0.0,
            zone_bottom=200.0,
            render=True,
            forbidden_rects=[],
        )
        self.assertEqual(remaining, "")
        self.assertEqual([entry[0] for entry in captured], ["Alpha", "Beta"])
        doc.close()

    def test_preserved_structured_line_uses_native_baseline_offset(self):
        reconstructor = DocumentReconstructor()
        reconstructor._rendered_signatures = set()
        reconstructor._style_audit_records = []
        captured = []
        reconstructor._safe_insert_text_dedup = lambda page, point, text, fontsize, fontname, color: captured.append((text, point[1]))
        doc = fitz.open()
        page = doc.new_page(width=300, height=200)
        item = {
            "text": "Alpha",
            "bbox": fitz.Rect(20, 20, 180, 36),
            "slots": [fitz.Rect(20, 20, 180, 36)],
            "slot_h_pt": 16.0,
            "slot_gap_y_pt": 2.0,
            "slot_w_pt": 160.0,
            "row_start_x_pt": 20.0,
            "style": {"font": "helv", "size": 11.0, "color": "#000000"},
            "source": "native",
            "role": "body",
            "translated_block": True,
            "preserve_linebreaks": True,
            "use_structured_source_lines": True,
            "source_lines": ["Alpha"],
            "source_line_metrics": [{"baseline_offset_pt": 5.5, "line_height_pt": 16.0}],
            "alignment": "left",
        }
        remaining, _, _, _ = reconstructor._render_block_slots(
            page=page,
            item=item,
            anchor_y=20.0,
            left=0.0,
            right=300.0,
            zone_top=0.0,
            zone_bottom=200.0,
            render=True,
            forbidden_rects=[],
        )
        self.assertEqual(remaining, "")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], "Alpha")
        self.assertAlmostEqual(captured[0][1], 25.5, places=3)
        doc.close()

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

    def test_place_item_in_frames_preserves_native_fontsize_for_wrapped_body(self):
        reconstructor = DocumentReconstructor()
        captured = []
        reconstructor._safe_insert_text_dedup = lambda page, point, text, fontsize, fontname, color: captured.append((text, fontsize))
        doc = fitz.open()
        page = doc.new_page(width=400, height=600)
        item = {
            "kind": "body",
            "text": "This translated paragraph should wrap across several lines while keeping the original native font size unchanged in the renderer.",
            "bbox": fitz.Rect(40, 40, 220, 80),
            "style": {"font": "helv", "size": 15.0, "color": "#000000"},
            "source": "native",
            "role": "body",
            "descriptor_v3_contract": {"primary_structure_family": "dense_paragraph_flow"},
            "descriptor_v3_render_unit": {"structure_priority": "secondary"},
        }
        frames = [fitz.Rect(40, 40, 170, 220)]
        reconstructor._place_item_in_frames(page, item, frames, 0, 40, [], [])
        self.assertTrue(captured)
        self.assertTrue(all(abs(fontsize - 15.0) < 1e-6 for _, fontsize in captured))
        doc.close()

    def test_structured_lines_stack_from_rendered_rect_without_overlap(self):
        reconstructor = DocumentReconstructor()
        reconstructor._rendered_signatures = set()
        reconstructor._style_audit_records = []
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        item = {
            "text": "First line\nSecond line",
            "bbox": fitz.Rect(20, 20, 220, 42),
            "slots": [
                fitz.Rect(20, 20, 220, 28),
                fitz.Rect(20, 29, 220, 37),
            ],
            "slot_h_pt": 8.0,
            "slot_gap_y_pt": 1.0,
            "slot_w_pt": 200.0,
            "row_start_x_pt": 20.0,
            "style": {"font": "helv", "size": 10.0, "color": "#000000"},
            "source": "native",
            "role": "body",
            "translated_block": True,
            "preserve_linebreaks": True,
            "use_structured_source_lines": True,
            "source_lines": ["First line", "Second line"],
            "alignment": "left",
            "descriptor_structural_role": "body_paragraph",
            "descriptor_band_role": "text_band",
        }
        remaining, _, _, used_slots = reconstructor._render_block_slots(
            page=page,
            item=item,
            anchor_y=20.0,
            left=0.0,
            right=300.0,
            zone_top=0.0,
            zone_bottom=300.0,
            render=True,
            forbidden_rects=[],
        )
        self.assertEqual(remaining, "")
        self.assertEqual(len(used_slots), 2)
        self.assertLessEqual(used_slots[0].y1, used_slots[1].y0)
        doc.close()

    def test_structured_wrapped_line_keeps_trailing_numeric_token(self):
        reconstructor = DocumentReconstructor()
        reconstructor._rendered_signatures = set()
        reconstructor._style_audit_records = []
        doc = fitz.open()
        page = doc.new_page(width=600, height=400)
        item = {
            "text": "9781593278458 (pub) - ISBN 1593278454 (pub) - ISBN 9781593278274\n(fiche papier) - ISBN 1593278276 (fiche papier)",
            "bbox": fitz.Rect(70, 100, 500, 150),
            "slots": [
                fitz.Rect(70, 100, 320, 111),
                fitz.Rect(70, 111, 320, 122),
                fitz.Rect(70, 122, 320, 133),
            ],
            "slot_h_pt": 11.0,
            "slot_gap_y_pt": 0.5,
            "slot_w_pt": 250.0,
            "row_start_x_pt": 70.0,
            "style": {"font": "UbuntuMono-Regular", "size": 11.25, "color": "#000000"},
            "source": "native",
            "role": "body",
            "translated_block": True,
            "preserve_linebreaks": True,
            "use_structured_source_lines": True,
            "source_lines": [
                "9781593278458 (pub) - ISBN 1593278454 (pub) - ISBN 9781593278274",
                "(fiche papier) - ISBN 1593278276 (fiche papier)",
            ],
            "descriptor_structural_role": "body_paragraph",
            "descriptor_typographic_class": "editorial_body",
            "alignment": "left",
        }
        reconstructor._render_block_slots(
            page=page,
            item=item,
            anchor_y=100.0,
            left=0.0,
            right=600.0,
            zone_top=0.0,
            zone_bottom=400.0,
            render=True,
            forbidden_rects=[],
        )
        text = " ".join(page.get_text("text").split())
        self.assertIn("9781593278274", text)
        doc.close()

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
