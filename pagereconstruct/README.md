# PAGERECONSTRUCT

Compile a translated page (`translated_input_data`) into a controlled set of
render operations, then render it. **Not a text drawer — a document recomposition
solver.**

## Philosophie

```
line        = evidence (jamais une unité de rendu de paragraphe)
paragraph   = renderable flow unit (layout = bloc logique complet)
table_cell  = locked unit
code/formula = preserve by default
unknown role ≠ paragraph  (-> anchored_label_review / review)
```

On ne reconstruit pas une page traduite en dessinant du texte ; on respecte un
plan de couches, de protections, de patchs, de styles et de contraintes.

## Pipeline

```
translated_input_data
  -> input_adapter            (4 vues: reconstruction_units + reconstruction/
                               preservation/exclusion_plan)
  -> plan_compiler            (fusion, consumed/excluded, anti double-rendu)
       layout_box_resolver    (layout/patch/coverage/anchor par rôle)
       style_resolver         (+ font_resolver_bridge, font_size_sanitizer)
       background_resolver     (clean / source / blank_degraded + leak risk)
       patch_planner          (patches déclarés, sampled color, protections)
  -> PageRenderPlan
  -> renderer_dispatcher + renderers/  (paragraph/heading/list/table/code/
                               formula/anchored_label/preservation)
  -> backends/ (pdf_vector | raster_debug)
  -> validator/quality        (status ok/review/ko gouverné par findings)
```

## Hiérarchie des bboxes

```
source_bbox / coverage_bbox / patch_bbox / layout_bbox / anchor_bbox
```
Le texte de flux (paragraphe/liste) se met en page dans `layout_bbox` = bloc
logique complet ; `patch_bbox` couvre les lignes source ; `anchor_bbox` = 1re ligne.

## Backends

- `backends/pdf_vector.py` — sortie vectorielle PyMuPDF (finale, V1).
- `backends/raster_debug.py` / `render_backend.py` — PNG de **debug** (overlays,
  contact sheets), pas la sortie WYSIWYG finale.

## Statut

Findings critiques empêchent `status = ok`. `validate(plan)` -> `ok|review|ko`.
