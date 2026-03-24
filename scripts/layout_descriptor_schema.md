# Layout Descriptor Schema

## Objective

Provide a canonical, relation-aware page description that is usable by:

- classification
- translation policy selection
- reconstruction planning
- QA / fidelity checks

The descriptor must describe not only each element's own geometry, but also its
structural and spatial relations to the other elements on the page.

## Canonical Page Contract

```json
{
  "page_id": 12,
  "page_number": 13,
  "page_size": {"width": 2480, "height": 3508, "unit": "px"},
  "document_type": "scientific_paper",
  "layout_type": "double_column",
  "style_profile": "academic",
  "page_role": "body_page",
  "classification_confidence": {
    "document_type": 0.87,
    "layout_type": 0.92,
    "style_profile": 0.76
  },
  "regions": [],
  "elements": [],
  "relations": [],
  "constraints": [],
  "features": {},
  "reading_order": [],
  "descriptor_version": "layout_descriptor.v1"
}
```

## 1. Regions

Regions are high-level page zones used by classification and reconstruction.

Allowed region types:

- `title`
- `text`
- `table`
- `picture`
- `caption`
- `header`
- `footer`
- `list_item`
- `section_header`
- `footnote`
- `formula`
- `sidebar`
- `column`
- `margin_note`
- `unknown`

Region schema:

```json
{
  "id": "region_col_1",
  "type": "column",
  "bbox": [120, 310, 1160, 3320],
  "column_index": 0,
  "parent_region_id": null,
  "reading_order": 2,
  "coverage_ratio": 0.28,
  "dominant_element_type": "text_block"
}
```

## 2. Elements

Elements are renderable or structural units extracted from the page.

Allowed element types:

- `text_block`
- `text_line`
- `text_phrase`
- `text_span`
- `title`
- `section_header`
- `header`
- `footer`
- `list_marker`
- `figure`
- `caption`
- `table`
- `table_row`
- `table_cell`
- `formula`
- `footnote`
- `page_number`
- `separator`
- `unknown`

Each element carries absolute geometry, hierarchy, and style.

```json
{
  "id": "blk_17_ln_3_ph_1",
  "type": "text_phrase",
  "role": "body",
  "source": "native",
  "bbox": [210, 1088, 1032, 1138],
  "polygon": null,
  "baseline": [214, 1126, 1027, 1126],
  "center": [621, 1113],
  "z_index": 3,
  "page_region_id": "region_col_1",
  "column_index": 0,
  "parent_id": "blk_17_ln_3",
  "children_ids": ["blk_17_ln_3_sp_1", "blk_17_ln_3_sp_2"],
  "reading_order": 37,
  "paragraph_id": "para_8",
  "sentence_id": "sent_21",
  "sentence_index_in_paragraph": 1,
  "line_index_in_block": 3,
  "style": {
    "font_family": "TimesNewRomanPSMT",
    "font_size_px": 23.4,
    "font_weight": 400,
    "italic": false,
    "underline": false,
    "color": "#111111",
    "align": "justify",
    "line_height_px": 31.2
  },
  "text": {
    "source_text": "The overall idea with DeepDream is that we pass an input image...",
    "visible_text": "The overall idea with DeepDream is that we pass an input image...",
    "translated_text": null,
    "language": "en",
    "tokens": 17,
    "is_truncated_source": false
  },
  "semantic": {
    "unit_type": "narrative_body",
    "is_translatable": true,
    "is_reference_like": false,
    "is_code_like": false,
    "is_formula_like": false
  }
}
```

## 3. Relations

Relations turn isolated boxes into a structural graph.

Allowed relation types:

- `inside`
- `contains`
- `above`
- `below`
- `left_of`
- `right_of`
- `same_row`
- `same_column`
- `aligned_left`
- `aligned_right`
- `aligned_center`
- `continues_as`
- `follows_in_reading_order`
- `caption_of`
- `has_caption`
- `cell_of`
- `row_of`
- `header_of_table`
- `anchored_to`
- `overlaps`
- `avoids`
- `references`

Relation schema:

```json
{
  "id": "rel_182",
  "type": "same_column",
  "source_id": "blk_17_ln_3_ph_1",
  "target_id": "para_8",
  "weight": 0.98,
  "metadata": {
    "dx": 3.0,
    "dy": 28.0
  }
}
```

## 4. Constraints

Constraints are the render contract derived from elements plus relations.

Allowed constraint types:

- `fixed_bbox`
- `anchored_bbox`
- `flow_in_region`
- `table_cell_locked`
- `caption_attached`
- `keep_same_column`
- `keep_same_row`
- `avoid_region`
- `avoid_element`
- `no_internal_sentence_break`
- `preserve_visible_text`
- `allow_vertical_expand`
- `allow_horizontal_expand`
- `preserve_baseline_rhythm`

Constraint schema:

```json
{
  "id": "c_91",
  "type": "flow_in_region",
  "element_id": "para_8",
  "region_id": "region_col_1",
  "params": {
    "max_width_px": 1020,
    "max_height_px": 1900,
    "allow_vertical_expand": true,
    "allow_horizontal_expand": false,
    "preserve_alignment": "justify"
  },
  "priority": 70
}
```

## 5. Reading Order

Reading order must be explicit and independent from raw OCR block order.

```json
{
  "reading_order": [
    "title_1",
    "author_1",
    "abstract_1",
    "blk_1",
    "blk_2",
    "fig_1",
    "cap_1"
  ]
}
```

## 6. Features

Features remain the classifier-facing summary of the page.

Required features:

- `num_columns`
- `text_coverage_ratio`
- `table_coverage_ratio`
- `image_coverage_ratio`
- `formula_coverage_ratio`
- `whitespace_ratio`
- `header_present`
- `footer_present`
- `title_count`
- `text_block_count`
- `table_count`
- `figure_count`
- `caption_count`
- `footnote_count`
- `font_size_levels`
- `dominant_font_size`
- `alignment_entropy`
- `column_balance_score`
- `toc_pattern_score`
- `form_pattern_score`
- `scientific_pattern_score`
- `invoice_pattern_score`

## 7. Derived Grouping Objects

Some render decisions must be made at paragraph or table level, not per phrase.

Derived grouping objects:

- `paragraph`
- `table_group`
- `figure_group`
- `reference_group`

Example paragraph object:

```json
{
  "id": "para_8",
  "type": "paragraph",
  "element_ids": [
    "blk_17_ln_1_ph_0",
    "blk_17_ln_2_ph_0",
    "blk_17_ln_3_ph_1"
  ],
  "region_id": "region_col_1",
  "column_index": 0,
  "sentence_ids": ["sent_19", "sent_20", "sent_21"],
  "constraints": {
    "render_mode": "flow_in_region",
    "can_break_inside_sentence": false,
    "allow_vertical_expand": true
  }
}
```

## 8. Reconstruction Rules Derived From Descriptor

The renderer must follow this order:

1. render fixed elements
2. reserve forbidden zones
3. render anchored elements
4. render table structures
5. render flowing paragraphs inside their assigned regions
6. validate sentence integrity and collision-free layout

## 9. Sentence Integrity Rules

To prevent broken phrases:

- every phrase belongs to a `sentence_id`
- sentence text must be renderable as a whole
- `can_break_inside_sentence` defaults to `false`
- line breaks are allowed only as normal composition breaks
- if a sentence does not fit, the paragraph must be replanned, not truncated

## 10. Implementation Mapping In This Repository

### Stage 1

Create a new module:

- `layout_descriptor.py`

Responsibilities:

- normalize extracted blocks into elements
- infer regions
- build paragraph/table/figure groups
- compute relations
- derive constraints

### Stage 2

Integrate after structural extraction:

- `structure_extractor.py`
- `ocr_server.py`

Expected output per page:

- `page["layout_descriptor"]`

### Stage 3

Consume in reconstruction:

- `reconstructor.py`

Priority migration order:

1. body paragraphs in columns
2. captions and diagram labels
3. tables and cells
4. references and citations
5. formulas

### Stage 4

Use in QA:

- `coverage_validator.py`
- `publication_qa.py`

New checks:

- paragraph stayed in assigned region
- sentence not truncated
- caption still attached
- table cells still aligned
- no forbidden overlap

## 11. Non-Goals For V1

V1 should not attempt:

- full graph optimization for every page
- deep learning based relation inference
- perfect semantic understanding of unknown diagrams

V1 should instead provide:

- stable canonical objects
- stable relations
- stable render constraints
- deterministic reconstruction behavior
