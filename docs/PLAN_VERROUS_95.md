# Plan d'implémentation — Verrous 95% (typographie OCR em + placement multi-blocs)

_Objectif: passer show10 de 0.779 → ≥0.95, sans casser les pages déjà OK, via 2 moteurs spécialisés EXTERNES + durcissement contract-first + QA image-réelle._

État de départ (vérifié): contrat + ops + backends exécuteurs + placement local + QA stricte en place (146 tests). 1 ok / 6 review / 3 ko. Plafonds: typo OCR 0.80, collisions denses, index.

Le squelette `verrous_95_impl.zip` (`pagereconstruct_ext/`) se couple **directement** au `FinalReconstructionContract` actuel (lit `contract.blocks`, `block.style.{font_size_pt,font_class,line_height,source}`, `block.layout.{source_bbox,layout_bbox,safe_bbox}`, `contract.findings`). Pas de réécriture du contrat nécessaire.

---

## Principe directeur (inchangé)
```
pageprint décide la structure ; pagetranslate décide le texte ;
pagereconstruct décide le CONTRAT ; les moteurs externes OPTIMISENT
typographie + placement SOUS CONTRAINTE, sans redécider la structure.
```
Moteurs externes = optionnels, désactivables, **fail-safe** (échec → review, jamais crash). Point d'injection unique:
```
translated_input_data
  → FinalReconstructionContract (vérité)
  → ContractEnhancementPipeline [OcrTypography, MultiBlock]   ← NOUVEAU
  → build_render_ops
  → backends exécuteurs
  → VisualQA (image-réelle)
```

---

## MISSION 0 — Vendoring + intégration fail-safe (prérequis, ~0.5j)

### 0.1 Vendorer le package
- [ ] Copier `pagereconstruct_ext/` (depuis `verrous_95_impl.zip`) → `pagereconstruct_ext/` à la racine du repo (package frère, pas dans `pagereconstruct/` pour rester découplé).
- [ ] `__init__.py` exporte `enhance_contract_for_publication`.

### 0.2 Câbler dans plan_compiler (après contrat, avant ops)
- [ ] Dans `compile_page_render_plan`, juste après `contract = FinalReconstructionContract.from_pageprint_pagetranslate(...)`:
```python
from pagereconstruct_ext.integration_adapter import enhance_contract_for_publication
contract, ext_report = enhance_contract_for_publication(
    contract, pageprint_data=normalized,
    page_image_path=contract.background.source_image_path,
    enable_typography=RECON_FLAGS.typography, enable_multiblock=RECON_FLAGS.multiblock, mutate=True)
plan_dict_after = ... ; plan.render_ops = build_render_ops(contract, plan.to_dict())
plan.final_contract = contract.to_dict()
plan.render_policy["external_enhancements"] = ext_report
```
- [ ] Flags d'activation (env/var): `RECON_ENABLE_TYPOGRAPHY`, `RECON_ENABLE_MULTIBLOCK` (défaut ON, désactivables).
- [ ] Fail-safe déjà dans l'adapter (try/except → findings). Vérifier qu'un échec n'empêche pas `build_render_ops`.

### 0.3 Adapter le solver multi-blocs au LayoutContract
- [ ] `_is_locked(block)` doit lire `block.layout.bbox_locked` (table/code/formule) en plus du rôle.
- [ ] Après patch layout, **re-mesurer** dans `build_render_ops` (le PlacementSolver local re-tourne sur la nouvelle `layout_bbox`) — vérifier la cohérence des 2 niveaux (multi-bloc grossier → local fin).

### Tests Mission 0
- [ ] `test_external_enhancers_fail_safe` (moteur qui lève → contrat intact + finding)
- [ ] `test_enhancers_disabled_by_flag`
- [ ] `test_render_ops_after_enhancement_have_no_overlap`

---

## MISSION 1 — Contract-first réel (~1j)

Inverser la dépendance: contrat = 1ère vérité, plan = snapshot/debug.

- [ ] Créer `pagereconstruct/final_contract_builder.py` : `FinalContractBuilder.build(translated_input_data) -> FinalReconstructionContract` (déplace la logique de décision: style/layout/patch/background/protected/preservation depuis `plan_compiler`).
- [ ] `compile_page_render_plan` devient un **wrapper**:
```
contract = FinalContractBuilder.build(tid)
contract = ContractEnhancementPipeline([...]).apply(contract)
ops = build_render_ops(contract)
plan = PageRenderPlan.from_contract_and_ops(contract, ops)   # snapshot debug
```
- [ ] Renderers consomment directement `BlockReconstructionContract` (ajouter `BaseRenderer.measure_contract(block)`), plus de dict fabriqué dans `build_render_ops`.
- [ ] Retirer le dispatch legacy résiduel du backend raster debug.
- [ ] `ContractEnhancementPipeline` : liste d'enhancers (`OcrTypographyEnhancer`, `MultiBlockLayoutEnhancer`) appliqués en ordre, chacun fail-safe.

### Tests Mission 1
- [ ] `test_contract_is_first_class` (build sans passer par un plan)
- [ ] `test_no_renderer_receives_raw_translated_text_dict`
- [ ] `test_plan_is_snapshot_of_contract`
- [ ] `test_enhancement_pipeline_order_and_failsafe`

---

## MISSION 2 — QA image-réelle branchée + débloquer les 3 KO (~2j)

### 2.1 VisualQA image-réelle dans le flux
- [ ] `tools/run_pipeline_full_demo.py` (et orchestrateur QA) : appeler `visual_qa.assess(plan, source_image_path=src, reconstructed_image_path=recon_png)`.
- [ ] Gate dur: une page ne peut être `ok` que si la QA image-réelle a tourné (sinon cap review).
- [ ] Comparaison crops: zones protégées (objet altéré) + zones patch (texte source résiduel). Exports: `visual_qa_image_findings.json`, `crops_failed/`, `overlay_failed_regions.png`.

### 2.2 p0505 — IndexRenderer complet (vrai modèle d'index)
- [ ] `IndexLayoutModel` : `index_entry / index_subentry / page_reference / indentation_level / column_id / reference_alignment_x`.
- [ ] `IndexRenderer` ≠ BaseRenderer compact : colonnes, hanging/indent, références de pages alignées à droite, pas d'overlap entre entrées, termes techniques non sur-traduits.
- [ ] Durcir rôle index (déjà noté): `index_head_term` seulement si page index OU ≥ plusieurs `index_entry`.

### 2.3 p0133 / p0180 — MultiBlockPlacementSolver (le moteur externe)
- [ ] `build_flow_regions` : détecter colonnes (2 colonnes → col_left/col_right ; sinon page_flow ; + figure bands).
- [ ] Obstacles = zones protégées (figure/formule/table grid/logo). Blocs verrouillés = table/code/formule/labels.
- [ ] `solve_region` : tri reading_order → candidats (normal/shrink/shift/expansion verticale) → packing vertical évitant obstacles + text/text → préserver ordre → minimiser déplacement → sinon review/ko.

### Tests Mission 2
- [ ] `test_visual_qa_uses_reconstructed_image`, `test_source_text_leak_crop_detected`, `test_non_text_crop_changed_detected`
- [ ] `test_index_renderer_columns_and_page_refs`, `test_isolated_index_term_rejected`
- [ ] `test_multiblock_removes_text_text_overlap`, `test_multiblock_avoids_protected`, `test_multiblock_preserves_reading_order`, `test_multiblock_locked_cells_not_moved`, `test_multiblock_ko_if_no_solution`
- **Cible Mission 2: show10 avg ≥ 0.88** (sans toucher l'extraction OCR).

---

## MISSION 3 — Typographie OCR em-size (~2-3j, transversal)

### 3.1 Côté extraction amont (pageprint) — la VRAIE source
- [ ] `raw_extractors` / `normalizer._normalize_style` : distinguer `glyph_bbox_height`, `line_bbox_height`, `font_em_size_pt`, `rendered_font_size_estimate`.
- [ ] Stocker `font_size_pt_raw`, `font_size_pt_em`, `font_size_source`, `font_size_confidence` (ne plus écraser avec `line_h_px/sx`).
- [ ] OCR: estimer em depuis hauteur cap/x-height moyennes + baseline + line-height + métriques police si reconnue.

### 3.2 Côté pagereconstruct — moteur externe `ocr_typography_engine`
- [ ] Le moteur (déjà squeletté) consomme l'évidence (line/glyph height, cap/x ratios par classe serif/sans/mono) → em-size + **style ladder de page** (body/caption/heading/index/table/diagram cohérents) + confidence.
- [ ] Remplacer le fallback `glyph_height = line_h*0.78` par analyse composantes connexes sur l'image source (quand `page_image_path` fourni) pour cap/x-height réels.
- [ ] `StyleContract` consomme `font_size_pt_em` si dispo (`source="ocr_em_estimator"`).
- [ ] Gate honnête: si seulement métrique ligne → review cap 0.80 ; si em-size confiance ≥ seuil → typo > 0.95 autorisé.

### 3.3 VisualQA — StyleSimilarityScorer
- [ ] Brancher `style_similarity.py` dans le score typo: famille/taille±8%/graisse/italique/couleur/interligne/alignement/indent (rendu vs source).

### Tests Mission 3
- [ ] `test_ocr_typography_estimates_em_size`, `test_detects_raw_metric_not_em`, `test_ladder_stabilizes_body`, `test_heading_larger_than_body`, `test_outputs_confidence`, `test_low_confidence_stays_review`, `test_patch_updates_style_contract`
- [ ] `test_em_size_lifts_typography_cap`, `test_style_similarity_gate`
- **Cible Mission 3: show10 avg ≥ 0.95, aucune page < 0.90.**

---

## Séparation des responsabilités (acté)
```
pagereconstruct PEUT corriger: exécution contractuelle, placement, collision,
  backend, overlays, VisualQA, rendu typographique avec les données reçues.
pagereconstruct NE PEUT PAS inventer: vraie taille em absente, police source
  jamais extraite, style mixte non fourni, baseline jamais mesurée, clean bg
  non produit. → ces manques se traitent à l'EXTRACTION (Mission 3.1).
```

## Rollout (3 versions du couplage)
1. **v1 — externe optionnel fail-safe** (Mission 0) : `pagereconstruct_ext/` appelé par l'adapter, désactivable. Comparer show10 avec/sans.
2. **v2 — ContractEnhancementPipeline** (Mission 1) : enhancers intégrés comme étape standard après le contrat.
3. **v3 — fusion** : si stable sur N docs, fondre dans `pagereconstruct/` (optionnel).

## Garde-fous / risques
- Le solver multi-blocs ne doit JAMAIS déplacer un bloc verrouillé (table/formule/code) ni un objet préservé → tests dédiés.
- Le moteur typo ne doit pas forcer `ok` sous confiance faible → reste review.
- Jitter typographique: stabiliser par **style ladder de page**, pas bloc-par-bloc.
- Re-mesure obligatoire après patch layout (la géométrie change → re-scorer collisions).
- Comparer systématiquement show10 avant/après chaque mission (non-régression des pages OK: p0457 reste 1.0).

## Cibles chiffrées par étape
| Étape | Cible avg show10 |
|---|---|
| Actuel | 0.779 |
| Après Mission 0 (branchement) | ~0.80-0.82 |
| Après Mission 2 (multiblock + index + QA image) | **≥ 0.88** |
| Après Mission 3 (em-size OCR fiable) | **≥ 0.95**, aucune < 0.90 |

## Ordre d'exécution
0 (vendoring+wire) → 1 (contract-first) → 2 (QA image + index + multiblock) → 3 (em-size). Mission 3.1 (extraction) peut démarrer en parallèle car transversale.

## Définition de DONE
- [ ] `pagereconstruct_ext/` vendoré, câblé, fail-safe, désactivable.
- [ ] FinalReconstructionContract = 1ère vérité ; renderers consomment le contrat ; plan = snapshot.
- [ ] VisualQA image-réelle branchée dans le pipeline ; page non `ok` sans elle.
- [ ] IndexRenderer complet + MultiBlockPlacementSolver opérationnels.
- [ ] em-size OCR fournie par l'extraction + consommée par StyleContract.
- [ ] show10 ≥ 0.95 moyen, aucune page < 0.90, p0457 reste ok, 0 régression tests.
