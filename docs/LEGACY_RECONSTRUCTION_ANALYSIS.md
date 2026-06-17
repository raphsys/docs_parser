# LEGACY_RECONSTRUCTION_ANALYSIS

Audit de l'ancien moteur de reconstruction (Phase 0). Source de vérité avant tout codage profond de `pagereconstruct/`. Fichiers audités : `reconstructor.py` (8645 l.), `ocr_server.py` (5636 l.), `document_object_contract.py` (473 l.), `final_page_compiler.py` (1411 l.), `background_inpainter.py` (273 l.).

---

## 1. Anciennes classes / dataclasses utiles

### `reconstructor.py` — dataclasses de contrat (cœur réutilisable)
| Dataclass | Rôle | Équivalent moderne cible |
|---|---|---|
| `BlockReconstructionPlan` (l.179) | Contrat de bloc complet : géométrie, line_templates, units, graph_edges, policies, semantics, source_layout_mode, adaptive_profile, constraints | `BlockReconstructionContract` |
| `PlacableUnit` (l.79) | Unité à placer : style, layout_attributes, text_attributes, reflowable, immutable, render_policy, break rules, anchors, keep_with | `ObjectContract` / unité du block contract |
| `LineTemplate` (l.57) | Gabarit de ligne : bbox, baseline_y, ascent/descent, usable_width, indent, first_line_indent, alignment, rotation | `LayoutContract.line_templates` |
| `BlockGeometryContext` (l.42) | Padding L/R/T/B, protected_regions, background_strategy, background_color | `LayoutContract` + `BackgroundContract` |
| `BlockSemanticProfile` (l.214) | content_class, render_strategy, font_normalization, allow_vertical_expansion, text_flow_mode, unicode_safe, estimated_text_expansion, dominant font (size/serif/bold/italic/mono) | `StyleContract` + `renderer_contract.strategy` |
| `BlockRenderOp` (l.167) | **Op de rendu** : op_type, block_id, unit_id, bbox, text, style, z_index, metadata | `RenderOps` (Text/Patch/Preservation/Background) |
| `RenderCandidate` (l.157) | candidate_id, strategy, ops, findings, score | `CandidateEngine` |
| `CandidateScore` (l.149) | value, status, penalties{}, hard_failures[] | `CandidateScore` |
| `PlacementResult` (l.131) / `PlacementCursor` (l.124) | ops + findings ; curseur (template_index, x, baseline_y) | `PlacementSolver` |
| `BlockRenderVerdict` (l.137) | status, ok, causes, text_ops_expected/rendered, recommended_strategy | `QualityContract` / verdict QA |
| `GraphEdge` (l.115) | relations dures entre units (keep_with) | `LayoutContract.graph_edges` |

### `reconstructor.py` — renderers (hiérarchie par type d'objet)
`BaseBlockRenderer` (l.6718) → `StructuredContractRenderer` (6948), `EditorialBlockRenderer` (7166) → `HeadingBlockRenderer` (8200), `CaptionBlockRenderer` (8204), `AnnotationBlockRenderer` (8208) ; `CodeBlockRenderer` (8212), `TableBlockRenderer` (8292). **Rendu par type d'objet = bonne idée à garder.**

### `final_page_compiler.py` — compilation d'items en ops
- `FormulaItem` (l.19) : formula_id, rect, source_rect, clips, text_subregions, linked_text_ids — **formule = copie de région source + clips** (préservation pixel exacte).
- `TextItem` (l.29) : block_id, role, rect, text, style, fontfile, fontname, alignment, color, coverage_fallback.
- `DrawOp` (l.43) : kind, rect, text, formula_id, source_rect, **source_clips, source_erase_rects**, font_size, fontfile, fontname, alignment, color — **op unifiée incluant copie/erase de régions source**.
- `ContinuousFinalPageCompiler` (l.59) : compile items → DrawOps.

### `document_object_contract.py` — contrat d'objet par unité
- `build_document_object_contract(unit)` (l.364) → object_class/object_type/policy + inline segments + translation protection + visual structure.
- `apply_contract_to_unit` (l.451), `_base_policy` (l.309), `extract_inline_segments` (l.158), `inline_structure_for_text` (l.228), `parse_toc_line`/`looks_like_toc`. **= couche ObjectContract + protection inline (code/math/refs).**

### `background_inpainter.py`
- `BackgroundInpainter` (l.14) : backends opencv (Telea) + LaMa (onnx), `save_inpaint_overlay` (l.52), `_inpaint_crop` (l.189). **Inpaint local par crop+mask.**

---

## 2. Anciennes fonctions utiles (`DocumentReconstructor`)

| Fonction | l. | Catégorie | À porter vers |
|---|---|---|---|
| `_resolve_compatible_font` | 358 | style/police (résout police compatible + fallback) | `font_resolver_bridge.py` |
| `_measure_text_width` | 445 | mesure texte (fontfile/builtin) | `text_measure.py` |
| `_measure_text` | 7284 | mesure multi-ligne | `text_measure.py` |
| `_render_contract_for_item` | 529 | contrat de rendu par item | `renderer_contract` |
| `_build_block_reconstruction_plan` | 4315 | **construit le BlockReconstructionPlan** | `block_contract.py` |
| `_render_hierarchical_block_plan` | 4539 | rend un plan de bloc | renderers |
| `_clean_page_background_path` | 4825 | résout le fond propre | `background_resolver.py` |
| `_insert_page_background` | 5249 | insère le fond | backend |
| `_overlay_signature` / `_overlay_ops_for_matching_immutable_overlays` | 5254/6859 | overlays immuables | `overlay_manager.py` |
| `_background_prep_ops` / `_background_ops` | 6878/6953 | ops de fond/patch local | `patch_planner.py` |
| `_score_render_candidate` | 6123 | **scoring candidat** | `candidate_engine.py` |
| `_render_plan_with_validation` | 6183 | **candidats + validation avant rendu** | `candidate_engine.py`+`placement_solver.py` |
| `_block_render_verdict` | 6045 | verdict QA bloc | `quality.py` |
| `_render_block_presence_fallback_ops` | 6346 | fallback présence texte | `quality.py` / rescue |
| `_emit_text_run` / `_emit_rotated_textbox_run` | 6813/6764 | **insertion texte PyMuPDF** (textbox, rotation, fontfile) | `backends/pdf_vector.py` |
| `_render_units_in_slots` / `_render_grid` / `_render_anchored_composite` / `_render_relative_slots` / `_render_prose_reflow` / `_render_label_stack` | 6979… | stratégies de rendu par layout | renderers + placement |
| `_compose_paragraphs_in_box` / `_composed_paragraphs_have_overflow` | 7820/7870 | reflow paragraphe + overflow | `placement_solver.py` |
| `_render_with_scale` | 7722 | candidat shrink | `candidate_engine.py` |
| `_render_page_text_rescue` / `_enforce_page_block_text_coverage` | 5938/(flux) | **garantie : jamais page image-only si texte attendu** | `validator.py` |

---

## 3. Politiques anciennes (ce qui marchait)

- **Reconstruction par contrat** : aucun rendu libre ; chaque bloc = `BlockReconstructionPlan` (géométrie + style + policy + units + line_templates).
- **Ordre des couches** (flux `reconstruct` l.6554) : overlays dynamiques injectés → **fond propre inséré** (`_insert_page_background`) → contexte → blocs (plan→candidats→validation→rendu) → **immutable overlays** réinsérés → **text rescue** (interdit page image-only avec texte attendu) → debug image.
- **Background** : `_clean_page_background_path` privilégie le fond nettoyé (`background_path` master de l'OCR) ; inpaint local possible (`background_inpainter`). Jamais « redessiner sur la source ».
- **Overlays immuables** : logos/figures/formules/watermark/page numbers réinsérés via `immutable_overlays` + signatures anti-doublon (`_overlay_signature`).
- **Style depuis la source** : `BlockSemanticProfile` capte la police dominante (size/serif/bold/italic/mono) ; police résolue + compatibilité glyphes (`_resolve_compatible_font`), pas inventée.
- **Overflow / candidats** : `_render_plan_with_validation` teste plusieurs candidats (scale/reflow), `_score_render_candidate` choisit ; `BlockRenderVerdict` dit ok/échec avec causes.
- **Fallback contrôlé** : `_render_block_presence_fallback_ops` + `coverage_fallback` (TextItem) ; substitution police auditée.
- **Formules = pixels** : `FormulaItem`/`DrawOp.source_rect+source_clips+source_erase_rects` → copie exacte de la région source (pas de re-rendu).

---

## 4. Ce qui ne marchait pas / fragile (à améliorer)

- Monolithe `reconstructor.py` 8645 l. : logique de décision + rendu + PyMuPDF mêlés, intestable finement.
- Renderers couplés à PyMuPDF (`page`, `fitz`) → pas de backend PNG/PDF symétrique.
- Beaucoup d'heuristiques de rôle/layout dans le reconstructeur (doublon avec ce que pageprint fait maintenant).
- Agents IA dans le chemin de rendu (désactivés `_get_render_agent`) — bruit.
- QA dispersée (verdicts par bloc) sans score image-réel global.

---

## 5. À EXTRAIRE (réutiliser, adapté)

1. Dataclasses contrat : `BlockReconstructionPlan`→`BlockReconstructionContract`, `PlacableUnit`, `LineTemplate`, `BlockSemanticProfile`, `BlockRenderOp`→RenderOps, `RenderCandidate`/`CandidateScore`, `PlacementResult/Cursor`, `BlockRenderVerdict`.
2. `DrawOp` unifiée (text + source_rect/clips/erase) — modèle d'op idéal pour backends.
3. `FormulaItem` (formule = copie région source) → `PreservationOp`.
4. Mesure/police : `_measure_text(_width)`, `_resolve_compatible_font`, `FontResolver`.
5. Candidats/placement : `_render_plan_with_validation`, `_score_render_candidate`, `_compose_paragraphs_in_box`, `_render_with_scale`.
6. Background : `_clean_page_background_path`, `_insert_page_background`, `background_inpainter`.
7. Overlays : `_overlay_ops_for_matching_immutable_overlays`, `_insert_immutable_overlays`, signatures.
8. Garde-fou : `_render_page_text_rescue` / `_enforce_page_block_text_coverage`.
9. ObjectContract : `build_document_object_contract` + inline protection.
10. PyMuPDF draw : `_emit_text_run`, `_emit_rotated_textbox_run`.

## 6. À ABANDONNER
- Agents IA de rendu (`_get_render_agent`, `_ai_refine_render_strategy`).
- Décisions de rôle/structure dans le reconstructeur (pageprint les fait).
- Couplage direct renderer↔fitz (passer par RenderOps).
- TOC special-path monolithique (remplacé par renderer index/toc + contrat).

---

## 7. Champs contractuels anciens (référence)
`bbox` (block/line/phrase/source/coverage/patch/layout/anchor) · `style{font,size,color,flags(bold/italic/mono),line_height,alignment,indent}` · `translated_text`/`source_text` · `background_path`/`source_image_path`/`mask_master_path` · `immutable_overlays[{bbox,kind,...}]` · `final_blocks[{lines[{phrases[{spans}]}],role,alignment,indent_px}]` · `semantic_phrases` · `render_policy`/`overflow_policy`/`layout_mode` · `object_type`/`role` · `text_removal_debug`/`p6_bg_audit`.

## 8. Source du fond propre (ocr_server) — déjà rebranché Phase 0 pipeline
`process_page` l.5087 : `_collect_text_regions_for_inpainting(final_blocks, non_text_zones, immutable_overlays)` → `text_removal_strategy.remove(img, text_regions)` → `clean_bgr` → l.5198 `background_path`. Préserve figures/diagram_labels/non_text_zones (≥25% overlap). **C'est ce savoir que `pipelines/background_cleaner.py` réplique déjà.**

---

## Conclusion Phase 0
L'ancien moteur avait DÉJÀ l'architecture cible : contrat de bloc + ops + candidats + placement + verdict + fond propre + overlays + fallback audité. La refonte = **extraire ces dataclasses/fonctions en modules propres, branchés sur pageprint/pagetranslate, exécutés par des backends symétriques via RenderOps**. Pas un nouveau renderer. Suite : `docs/LEGACY_TO_PAGERECONSTRUCT_MAPPING.md` (Phase 1).
