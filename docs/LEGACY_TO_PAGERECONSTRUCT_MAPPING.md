# LEGACY_TO_PAGERECONSTRUCT_MAPPING

Phase 1. Correspondance ancien moteur → architecture moderne `pagereconstruct/` (exécuteur de contrat). Chaque concept ancien utile doit avoir une destination. Action : `adapt` (reprendre la logique, nettoyée), `wrap` (envelopper l'existant), `new` (créer), `drop` (abandonner).

## Table principale

| Ancien (fichier · symbole) | Nouveau (module · symbole) | Action | Statut |
|---|---|---|---|
| `final_blocks` (ocr_server/reconstructor) | `FinalReconstructionContract.block_contracts[]` + `text_units` | adapt | à faire |
| `BlockReconstructionPlan` (reconstructor 179) | `block_contract.BlockReconstructionContract` | adapt | à faire |
| `PlacableUnit` (reconstructor 79) | `block_contract.PlacableUnit` (sous-unité) | adapt | à faire |
| `LineTemplate` (reconstructor 57) | `layout_contract.LineTemplate` | adapt | à faire |
| `BlockGeometryContext` (42) | `layout_contract` + `background_contract` | adapt | à faire |
| `BlockSemanticProfile` (214) | `style_contract.StyleContract` + strategy | adapt | à faire |
| `BlockRenderOp` (167) / `DrawOp` (final_page_compiler 43) | `render_ops` : `TextOp/PatchOp/PreservationOp/BackgroundOp` | adapt | à faire |
| `RenderCandidate`+`CandidateScore` (157/149) | `candidate_engine.RenderCandidate/CandidateScore` | adapt | à faire |
| `PlacementResult`+`PlacementCursor` (131/124) | `placement_solver.*` | adapt | à faire |
| `BlockRenderVerdict` (137) | `quality_contract` / `quality.py` verdict | adapt | à faire |
| `immutable_overlays` (ocr_server/reconstructor) | `preservation_contract` + `PreservationOp` (`overlay_manager`) | adapt | à faire |
| `FormulaItem` (final_page_compiler 19) source_rect+clips | `PreservationOp(method=copy_source_region)` | adapt | à faire |
| `background_path`/`clean_background`/background master | `background_contract.BackgroundContract` | adapt | partiel (pipeline fond propre fait) |
| `background_inpainter.BackgroundInpainter` | `background_resolver` (inpaint local) + `text_removal_contract` | wrap | à faire |
| `text_removal_strategy` / `_collect_text_regions_for_inpainting` | `text_removal_contract` + `PatchOp` ; déjà dans `pipelines/background_cleaner.py` | wrap | fait (pipeline) |
| `semantic_phrases` | `pagetranslate.views.reconstruction_units` (déjà) | drop (remplacé) | ok |
| `document_object_contract.build_document_object_contract` | `object_contract.ObjectContract` (object_type/role/policy/inline) | adapt | à faire |
| `reconstruction_contracts` / `_render_contract_for_item` | `layout_contract` + `renderer_contract` + `quality_contract` | adapt | à faire |
| `_resolve_compatible_font` / `FontResolver` | `font_resolver_bridge` (existe) — compléter compat glyphes | wrap | partiel |
| `_measure_text(_width)` | `text_measure.py` | adapt | à faire |
| `_render_plan_with_validation` / `_score_render_candidate` | `candidate_engine` + `placement_solver` | adapt | à faire |
| `_compose_paragraphs_in_box` / `_render_with_scale` | `placement_solver` (reflow/shrink) | adapt | à faire |
| `_emit_text_run` / `_emit_rotated_textbox_run` | `backends/pdf_vector.execute_ops` (TextOp) | adapt | à faire |
| `_insert_page_background` / `_insert_immutable_overlays` | `backends/*.execute_ops` (BackgroundOp/PreservationOp) | adapt | à faire |
| `_render_page_text_rescue` / `_enforce_page_block_text_coverage` | `validator` : interdit page image-only si texte attendu | adapt | à faire |
| renderers `BaseBlockRenderer`+sous-classes | `renderers/*` (existent) → `to_ops()` à partir du `BlockReconstructionContract` | adapt | à faire |
| Agents IA rendu (`_get_render_agent`, `_ai_refine_render_strategy`) | — | drop | ok |
| TOC special-path monolithique | renderer index/toc + contrat | drop/adapt | à faire |
| couplage renderer↔fitz | RenderOps + backends | drop | à faire |

## Concepts anciens sans équivalent encore (findings)
- `GraphEdge` (keep_with relations) → `layout_contract.graph_edges` : **à créer**, sinon perte du « keep_with_next/previous » (anti-veuve/orpheline).
- `DrawOp.source_erase_rects` (erase ciblé région source) → utile pour patch chirurgical sous overlay : **à intégrer dans PatchOp**.
- `coverage_fallback` (TextItem) → drapeau d'audit fallback : **à porter dans QualityContract**.
- `_render_page_text_rescue` (garantie présence texte) → **gate validator obligatoire**.

## Règles de priorité (consommation)
1. pageprint/pagetranslate modernes = source principale.
2. contrats legacy = complément (background/overlays/style/layout si moderne incomplet).
3. legacy ne réécrit jamais une traduction moderne validée.
4. legacy ne réintroduit jamais le texte source (fond propre obligatoire).

## Conclusion Phase 1
Tout concept ancien utile a une destination. Quatre éléments à ne pas oublier (findings ci-dessus). Suite : Phase 2 = `FinalReconstructionContract` + sous-contrats, alimenté par `from_pageprint_pagetranslate()` et `from_legacy_contract()`.
