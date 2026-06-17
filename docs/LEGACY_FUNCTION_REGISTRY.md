# LEGACY_FUNCTION_REGISTRY

Migration fonction par fonction de l'ancien moteur → moderne. Source unique: `pagereconstruct/legacy/function_registry.py`. Décisions: KEEP_AS_IS / ADAPT / WRAP / MERGE / DROP / REPLACE_TESTED.

| Ancien fichier | Symbole | Rôle | Décision | Destination | Statut | Test |
|---|---|---|---|---|---|---|
| reconstructor.py | _measure_text / _measure_text_width | Mesure texte/largeur | ADAPT | pagereconstruct/text_measure.py | done | test_render_ops |
| reconstructor.py | _resolve_compatible_font / FontResolver | Police compatible + fallback glyphes | WRAP | pagereconstruct/font_resolver_bridge.py | partial |  |
| reconstructor.py | BlockSemanticProfile (dominant font) | Profil typo dominant | ADAPT | pagereconstruct/ocr_typography_engine.py | done | test_pubready_core |
| reconstructor.py | BlockReconstructionPlan | Contrat de bloc | ADAPT | pagereconstruct/block_contract.py | done | test_final_contract |
| reconstructor.py | PlacableUnit / LineTemplate | Unité à placer / gabarit ligne | ADAPT | pagereconstruct/layout_contract.py | partial |  |
| reconstructor.py | _build_block_reconstruction_plan | Construction plan bloc | ADAPT | pagereconstruct/composition/block_planner.py | todo | test_block_planner |
| reconstructor.py | RenderCandidate / CandidateScore | Candidats + score | ADAPT | pagereconstruct/candidate_engine.py | done | test_render_ops |
| reconstructor.py | PlacementResult / PlacementCursor | Résultat/curseur placement | ADAPT | pagereconstruct/placement_solver.py | done | test_render_ops |
| reconstructor.py | _score_render_candidate / _render_plan_with_validation | Scoring + validation candidats | ADAPT | pagereconstruct/candidate_engine.py + autocorrect | partial |  |
| reconstructor.py | _compose_paragraphs_in_box / _render_prose_reflow | Composition paragraphe / reflow | ADAPT | pagereconstruct/composition/intrablock_composer.py | todo | test_intrablock_composer |
| reconstructor.py | _render_with_scale / _composed_paragraphs_have_overflow | Shrink-to-fit / overflow | ADAPT | pagereconstruct/composition/text_fitter.py | todo | test_legacy_text_fit_equivalence |
| reconstructor.py | _render_label_stack / _render_relative_slots | Stacks labels / slots relatifs | ADAPT | pagereconstruct/composition/line_layout_engine.py | todo |  |
| ocr_server.py | _collect_text_regions_for_inpainting | Zones texte à inpaint (protège non-texte 25%) | WRAP | pipelines/background_cleaner.py | done | test_clean_background |
| text_removal_strategy.py | TextRemovalStrategy.remove | Inpaint Telea bbox | WRAP | pipelines/background_cleaner.py | done | test_clean_background |
| ocr_server.py | _erase_uncovered_pdf_words | Effacer mots PDF non couverts (en-têtes/pieds) | ADAPT | pipelines/background_cleaner.py | todo | test_erase_uncovered_words |
| background_inpainter.py | BackgroundInpainter (LaMa/opencv) | Inpaint local par crop | WRAP | pipelines/background_cleaner.py (option) | available |  |
| reconstructor.py | _clean_page_background_path / _insert_page_background | Fond propre | ADAPT | pagereconstruct/background_resolver.py + backends | done | test_render_ops |
| ocr_server.py / reconstructor.py | immutable_overlays / _insert_immutable_overlays | Overlays immuables | ADAPT | pagereconstruct/overlay_manager.py + preservation_contract.py | done | test_legacy_contracts |
| final_page_compiler.py | FormulaItem (source_rect + clips) | Formule = copie région source | ADAPT | pagereconstruct/preservation_contract.py (copy_source_region) | partial |  |
| final_page_compiler.py | DrawOp (text/source_rect/erase_rects) | Op de dessin unifiée | ADAPT | pagereconstruct/render_ops.py | done | test_render_ops |
| document_object_contract.py | build_document_object_contract | Contrat objet + inline protection | ADAPT | pagereconstruct/object_contract.py | partial | test_legacy_contracts |
| reconstructor.py | _emit_text_run / _emit_rotated_textbox_run | Insertion texte PyMuPDF (rotation/fontfile) | ADAPT | pagereconstruct/backends/pdf_vector.py (execute_ops) | partial | test_render_ops |
| reconstructor.py | _render_page_text_rescue / _enforce_page_block_text_coverage | Jamais page image-only si texte attendu | ADAPT | pubready/stages/* (gate) + validator | todo | test_text_presence_gate |
| reconstructor.py | _get_render_agent / _ai_refine_render_strategy | Agents IA dans le rendu | DROP | — | dropped |  |

**Bilan**: 10/24 done · 6 partial · 6 todo · 1 dropped.

## Reste à porter (todo/partial) — ordre de priorité
1. background/inpaint: `_erase_uncovered_pdf_words` (en-têtes/pieds).
2. composition intra-bloc: `_compose_paragraphs_in_box`, `_render_with_scale` (overflow/shrink).
3. backend PDF: maturité `_emit_rotated_textbox_run` (rotation/clipping).
4. préservation: `FormulaItem` copy_source_region exact.
5. garde-fou: `_render_page_text_rescue` → gate présence texte.
