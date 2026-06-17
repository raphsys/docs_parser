# Taxonomie Object Comprehension

Objectif:

- introduire une etape explicite de comprehension / typage des objets de page;
- guider l'extraction, la traduction et la reconstruction avec une taxonomie stable;
- eviter de traiter de la meme facon un paragraphe, un code, une formule, un tableau ou un label visuel.

## Schema cible

Chaque unite (`block`, `line`, `phrase`, `span`, region visuelle) peut porter:

- `object_family`
- `object_class`
- `object_type`
- `object_subtype`
- `object_confidence`
- `object_comprehension`

`object_comprehension` contient:

- `schema_version`
- `engine`
- `level`
- `family`
- `object_class`
- `object_type`
- `object_subtype`
- `translation_hint`
- `reconstruction_hint`
- `preserve_exact_text`
- `preserve_geometry`
- `preserve_style`
- `confidence`
- `reasons`
- `model_capabilities`
- `preferred_open_source_models`

## Families

- `textual`
- `visual`
- `structural`
- `composite`

## Classes

- `editorial`
- `technical`
- `formula`
- `tabular`
- `reference`
- `navigational`
- `visual`
- `visual_label`
- `metadata`
- `mixed`

## Types principaux

Blocs:

- `title`
- `section_heading`
- `paragraph`
- `list_item`
- `footnote`
- `figure_caption`
- `table_caption`
- `caption`
- `page_header`
- `page_footer`
- `page_number`
- `toc_entry`
- `reference_entry`
- `table_block`
- `table_cell`
- `code_block`
- `formula_block`
- `diagram_label`
- `chart_label`
- `axis_label`
- `legend_label`
- `short_label`
- `figure_region`
- `image_region`
- `drawing_region`

Lignes / phrases:

- `code_line`
- `formula_line`
- `inline_formula_cluster`
- `plain_text`

Spans:

- `plain_text`
- `emphasis_span`
- `inline_code`
- `inline_formula`
- `reference_link`
- `citation`
- `abbreviation`
- `technical_identifier`

## Traduction par type

- `paragraph`, `title`, `section_heading`, `caption`, `figure_caption`, `table_caption`:
  traduire
- `reference_link`, `page_number`, `inline_code`, `code_block`, `formula_block`, `inline_formula`:
  preserver
- `table_block`, `table_cell`:
  traduire cellule par cellule, sans reflow global
- `diagram_label`, `chart_label`, `axis_label`, `legend_label`, `page_header`, `page_footer`:
  traduction contrainte, ancree si necessaire

## Reconstruction par type

- `paragraph`: `paragraph_reflow`
- `list_item`, `caption`, `footnote`: `preserve_breaks`
- `title`, `section_heading`, `page_header`, `page_footer`, `diagram_label`: `anchored_text`
- `table_block`, `table_cell`: `cell_locked`
- `code_block`, `code_line`: `code_preserve`
- `formula_block`, `formula_line`: `formula_preserve`
- `figure_region`, `image_region`, `drawing_region`: `source_overlay`

## Capacites modele attendues

- `layout_detection`
- `reading_order`
- `table_structure`
- `formula_ocr`

## Principe d'architecture

Pipeline vise:

1. PDF / image
2. detection d'objets
3. object comprehension / taxonomy
4. extraction structurelle pilotee par type
5. traduction pilotee par type
6. reconstruction pilotee par type

## Remarque critique

Il n'existe pas aujourd'hui un unique modele open source robuste qui couvre
toutes les classes documentaires avec la meme qualite. La bonne architecture
est hybride:

- un detecteur de layout generaliste
- un specialiste tableau
- un specialiste formule
- des heuristiques fortes pour code / references / inline techniques
