"""Synthetic translated_input_data matching the real PAGEPRINT/PAGETRANSLATE shapes."""

from __future__ import annotations


def translated_input_data() -> dict:
    return {
        "schema_version": "pageprint.input.v1",
        "page": {"page_index": 0, "page_role": "body", "rotation": 0,
                 "geometry": {"width": 531.0, "height": 666.0, "unit": "pt", "origin": "top_left",
                              "render_width_px": 1106, "render_height_px": 1387,
                              "scale_x_px_per_pt": 2.083, "scale_y_px_per_pt": 2.083}},
        "document": {}, "assets": {"source_image_path": ""}, "visual_layers": {},
        "units": [
            {"unit_id": "blk1", "level": "block", "children_ids": ["ln1", "ln2"],
             "geometry": {"bbox": [50, 50, 400, 90]},
             "understanding": {"role": "body_paragraph", "object_type": "natural_text"}, "policy": {}},
            {"unit_id": "ln1", "level": "line", "children_ids": [], "geometry": {"bbox": [50, 50, 400, 70]},
             "understanding": {"role": "body_paragraph"}, "policy": {},
             "visual": {"style": {"font_family": "Times", "font_size_pt": 11.0, "color": "#000000",
                                  "flags": {"bold": False, "italic": False, "serif": True, "monospace": False}}}},
            {"unit_id": "ln2", "level": "line", "children_ids": [], "geometry": {"bbox": [50, 70, 400, 90]},
             "understanding": {"role": "body_paragraph"}, "policy": {}},
            {"unit_id": "fml1", "level": "line", "children_ids": [], "geometry": {"bbox": [60, 120, 180, 140]},
             "understanding": {"role": "formula_expression", "object_type": "formula_expression"},
             "policy": {"render_policy": "fixed_preserve", "preservation_mode": "preserve_as_visual_overlay"}},
        ],
        "regions": [{"region_id": "r1", "region_type": "formula_candidate_region",
                     "object_type": "formula", "bbox": [60, 120, 180, 140]}],
        "views": {
            "reconstruction_units": [
                {"unit_id": "seg1", "translation_unit_id": "tp1", "level": "semantic_phrase",
                 "role": "body_paragraph", "object_type": "natural_text", "semantic_kind": "prose",
                 "text": "Hello world.", "translated_text": "Bonjour le monde.", "bbox": [50, 50, 400, 90],
                 "source_unit_ids": ["blk1"], "consume_source_units": True, "preferred_over_children": True,
                 "render_target": {"bbox": [50, 50, 400, 90]}, "layout_budget": {"bbox_reliable": True}},
            ],
            "reconstruction_plan": [],
            "preservation_plan": [
                {"preservation_id": "pres_0001", "source_unit_ids": ["pg1"], "text": "84",
                 "preservation_mode": "preserve_text_exactly", "render_mode": "fixed_preserve",
                 "reason": "index_page_reference", "bbox": [500, 640, 520, 655]},
                {"preservation_id": "pres_0002", "source_unit_ids": ["fml1"], "text": "E=mc^2",
                 "preservation_mode": "preserve_as_visual_overlay", "reason": "formula", "bbox": [60, 120, 180, 140]},
            ],
            "exclusion_plan": [
                {"exclusion_id": "exc_0001", "source_unit_ids": ["pub1"], "text": "Estadisticos",
                 "reason": "publisher_mark", "bbox": [40, 650, 200, 662]},
            ],
        },
    }


def with_duplicate_child_unit(data: dict) -> dict:
    """Add a reconstruction unit re-rendering a child already consumed by seg1."""
    data["views"]["reconstruction_units"].append(
        {"unit_id": "ln1", "translation_unit_id": "tp2", "level": "line", "role": "body_paragraph",
         "object_type": "natural_text", "text": "Hello", "translated_text": "Bonjour", "bbox": [50, 50, 400, 70],
         "source_unit_ids": ["ln1"], "consume_source_units": False, "preferred_over_children": True,
         "render_target": {}, "layout_budget": {"bbox_reliable": True}}
    )
    return data
