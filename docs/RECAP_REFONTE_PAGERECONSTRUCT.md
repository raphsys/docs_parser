# Récapitulatif — Refonte `pagereconstruct/` en exécuteur de contrat

_Document de synthèse : demande, plan, exécution, observations, reste à faire._
_Session 2026-06-14._

---

## 1. Ce qui a été DEMANDÉ

### 1.1 Demande initiale (diagnostic utilisateur)
Les pages reconstruites n'étaient **pas publication-ready** (0/10), malgré une meilleure architecture. Diagnostic acté avec l'utilisateur :

> « `pagereconstruct/` ne se comporte pas comme un consommateur fidèle. Il réinterprète, simplifie et redessine. Il devrait : pageprint + pagetranslate + contrats anciens → exécution fidèle du contrat de reconstruction → page reconstruite. »

**Principe central imposé :**
```
pagereconstruct/ doit ARRÊTER de réinterpréter.
Il doit EXÉCUTER un contrat final.
```
Interdits : nouveau moteur qui redécide la page ; renderer isolé ; backend PNG/PDF qui improvise ; patcher approximatif ; correcteur typographique heuristique.

### 1.2 Objectif chiffré
- show10 : moyenne publication-ready **≥ 95 %**, aucune page < 90 %.
- 0 leak texte source, 0 patch destructeur, 0 collision critique, typographie fidèle.

### 1.3 Méthode imposée (ordre strict)
Plan en 12 phases (réf. `PLAN_RECONSTRUCT_CONTRAT.md`), à implémenter **en entier** d'abord, puis revisiter les particularités. Audit du legacy AVANT tout code. Reprendre les acquis de : `ocr_server.py`, `reconstructor.py`, `document_object_contract.py`, `final_page_compiler.py`, `background_inpainter.py`, anciens contrats / final_blocks / immutable_overlays / background masters.

---

## 2. Ce qui a été FAIT — plan + exécution

### Phase 0 — Audit ancien moteur ✅
- `docs/LEGACY_RECONSTRUCTION_ANALYSIS.md` : inventaire complet. **Constat clé** : l'ancien moteur avait DÉJÀ l'architecture cible — `BlockReconstructionPlan` (contrat bloc), `BlockRenderOp`/`DrawOp` (ops), `RenderCandidate`+`CandidateScore` (candidats), `PlacementResult` (placement), fond propre + immutable overlays + text-rescue.

### Phase 1 — Mapping ancien→nouveau ✅
- `docs/LEGACY_TO_PAGERECONSTRUCT_MAPPING.md` : chaque concept legacy → destination moderne. 4 findings (GraphEdge keep_with, source_erase_rects, coverage_fallback, text_rescue gate).

### Phase 2 — FinalReconstructionContract ✅
- `final_contract.py` (`FinalReconstructionContract` + `PageInfo` + `LAYER_ORDER`) + 7 sous-contrats : `background_contract`, `block_contract`, `object_contract`, `style_contract`, `layout_contract`, `preservation_contract`, `quality_contract`.
- `from_pageprint_pagetranslate()`, `from_legacy_contract()`, `merge_legacy_and_new()`, `validate(mode)`.

### Phase 3 — LegacyContractBridge ✅
- `legacy_contract_bridge.py` : `extract_legacy_background / immutable_overlays / final_blocks / style / layout / render_policies / inpaint_masks / quality_hints` + `convert_legacy_to_final_contract`.
- Règles de priorité : moderne > legacy ; legacy ne réécrit pas une traduction validée ; legacy ne réintroduit pas le texte source.

### Phase 4 — Port reconstructor ✅
- `text_measure.py` (mesure canonique, ex-`_measure_text`), `candidate_engine.py` (`RenderCandidate`/`CandidateScore`, 6 scores : text_fit / style_similarity / position / collision / readability / preservation).

### Phase 5 — BlockReconstructionContract ✅
- `block_contract.py` (géométrie + style + render policy + préservation + qualité). Typographie **honnête** : taille inférée par géométrie ≠ échec ; réparation > 30 % → `page_style_unreliable` (review, cap 0.80).

### Phase 6 — Background + overlays ✅
- Fond propre : `pipelines/background_cleaner.py` (inpaint cv2 Telea du texte traduisible) câblé dans l'orchestrateur → `assets.background_path`.
- `overlay_manager.py` : classe underlays/overlays, z_index, PreservationOps, patches non destructeurs.

### Phase 7 — CandidateEngine / PlacementSolver ✅
- `placement_solver.py` : candidats (normal → shrink ≤14% → shift local), scoring 6 dims, **collisions évitées AVANT émission TextOp**. Intégré dans `build_render_ops`.

### Phase 8 — Backend PDF héritier ✅
- `backends/pdf_vector.py` : `execute_ops` **ops-only (zéro dispatch)**, vraie police TTF (`fontfile`, match PNG), audit substitution.
- `render_ops.py` : `BackgroundOp / PatchOp / PreservationOp / TextOp` + `build_render_ops` (dispatch+measure UNE fois). Plan gèle `render_ops` + `final_contract`.

### Phase 9 — VisualQA stricte ✅
- `visual_qa.py` mesure les **RenderOps placées** (pas le plan brut). Score `source_text_leak` + `hard_blockers` + hook image-réelle (`_image_leak_score`). Gates durs : non_text≥0.99, leak≥0.98, typo≥0.95, overlap≥0.99.
- `validator.py` durci.

### Phase 10 — Tests legacy ✅
- `tests/legacy_reconstruction/test_legacy_contracts.py` (8 tests non-régression contrats anciens).

### Phase 11 — show10 ✅
- Rejoué. Résultats §3.

### Patch non destructeur (Phase 7 directive)
- `patch_planner.py` : un patch chevauchant une zone protégée est **découpé autour** (soustraction de rectangles), jamais peint dessus.

### Corrections de sur-détection amont (en chemin)
- Sur-détection formule corrigée (`role_resolver`: `_is_strong_math`, `_has_math_evidence` durcis, court-circuit legacy "formula" rejeté sans évidence math).
- Code SQL : propagation bloc (`_propagate_block_code`) → listing cohérent.
- Labels de figure → `diagram_label` non-traduisible.
- Garde PUA (puces Symbol) restreint : ` prose…` n'est plus formule.
- Invariant cohérence : une source traduite n'est jamais protégée ; régions candidates ne hard-protègent pas.

---

## 3. RÉSULTATS (show10, 10 pages)

| Métrique | Début session | Fin (12 phases) |
|---|---|---|
| avg publication_ready | 0.629 | **0.779** |
| pages `ok` (publiables) | 0 | **1** (p0457 = 1.0) |
| ok / review / ko | 0 / 4 / 6 | **1 / 6 / 3** |
| KO total | 28 (1er jet) | **4** |
| leak texte source high | 10/10 | **0/10** |
| source_text_leak score | — | **1.0 / 10** |
| non_text_presence | partiel | **1.0 / 10** |

**Tests : 146/146 verts** (dont 8 non-régression legacy).

Détail pages (fin) : p0457 `ok` 1.0 ; p0051 review 0.99 ; p0140/p0192/p0337/p0406/p0463 review 0.80 ; p0133/p0180/p0505 `ko`.

Fichiers : `results/show10_all/` (PDF + PNG + overlay + audit JSON + montages).

---

## 4. OBSERVATIONS / COMMENTAIRES

1. **Le vrai blocage #1 n'était pas pagereconstruct** : l'orchestrateur ne produisait pas de **fond nettoyé** (texte source inpainté). Sans lui, leak high partout → publication impossible *par construction*. C'est le gain le plus structurant (mémoire `cleanbg_gap`).

2. **L'ancien moteur avait déjà la bonne architecture** (contrat/ops/candidats/placement). La refonte = *extraire* proprement, pas réinventer. Confirmé en Phase 0.

3. **Sur-détection formule = source récurrente de collisions** : puces de liste, prose à symbole inline, équations entrelacées étaient taggées formule → protégées → collisionnaient le texte traduit. Plusieurs gardes ajoutés.

4. **Plafond typographique honnête** : sur les pages **OCR/scannées**, `font_size_pt` est une métrique (~4.32 = line_h_px/sx), pas la vraie taille em. Le rendu est visuellement correct (inférence géométrique) mais ce n'est pas la fidélité source → plafond review **0.80**. Les **PDF natifs** (SQL) ont la vraie taille → atteignent `ok`. **Le scoring ne ment plus** (ni faux 0.0, ni faux 1.0).

5. **Backends = exécuteurs purs** : `pdf_vector` ne dispatche plus, n'improvise plus. dispatch+measure ont lieu UNE fois dans `build_render_ops`.

6. **Limite atteignable sans extraction amont** : ~0.80 sur docs scannés, `ok` sur natifs. Le ≥95 % global exige la vraie taille em (extraction).

---

## 5. RESTE À FAIRE

### 5.1 Particularités par phase (affinages convenus)
- **Phase 9** : brancher la VisualQA **image-réelle** dans le pipeline (le hook `_image_leak_score` existe mais n'est pas appelé par le demo).
- **Phase 8** : maturité PDF (rotation, clipping, gestion couleurs avancée) du legacy non entièrement portée.
- **Renderers** : doivent consommer directement le `BlockReconstructionContract` (aujourd'hui `build_render_ops` leur fabrique un dict).
- **Phase 3** : remonter `compile_page_render_plan` pour passer *exclusivement* par le contrat (le plan reste construit puis le contrat dérivé ; supprimer le dispatch legacy résiduel du backend raster debug).

### 5.2 Pages KO restantes (collisions)
- **p0505** (index, 2 KO) : durcir le rôle index + rendu `index` (indentation, page refs).
- **p0180** (overlap 0.31), **p0133** (0.73) : collisions denses → placement plus poussé (expansion verticale, reflow multi-bloc).
- **p0051** review 0.99 → `ok` : overlap 0.945, marginal.

### 5.3 Chantier futur acté — taille em (extraction amont)
- Pages OCR : fournir la **vraie taille em** par `pageprint` / extraction (`raw_extractors` / `normalizer._normalize_style`) au lieu de la métrique OCR. Lève le plafond 0.80 sur docs scannés. (Tâche #26, mémoire `font_size_em_extraction`.)

### 5.4 Cibles DONE non encore atteintes
- show10 ≥ 95 % moyen (actuel 0.779) — bloqué par §5.2 + §5.3.
- Aucune page < 90 % (3 pages ko à 0.60).

---

## 6. Fichiers clés créés/modifiés

**Contrat / ops :** `pagereconstruct/final_contract.py`, `block_contract.py`, `object_contract.py`, `style_contract.py`, `layout_contract.py`, `background_contract.py`, `preservation_contract.py`, `quality_contract.py`, `render_ops.py`, `legacy_contract_bridge.py`, `candidate_engine.py`, `placement_solver.py`, `text_measure.py`, `overlay_manager.py`.

**Modifiés :** `plan_compiler.py` (gèle contrat+ops), `backends/pdf_vector.py` (ops-only+fontfile), `render_backend.py` (render_ops_to_png), `visual_qa.py` (QA sur ops + leak score + gates), `validator.py` (gates stricts), `patch_planner.py` (découpe), `schema.py` (render_ops), `pageprint/role_resolver.py` (sur-détection), `pipelines/orchestrator.py` + `pipelines/background_cleaner.py` (fond propre).

**Docs :** `LEGACY_RECONSTRUCTION_ANALYSIS.md`, `LEGACY_TO_PAGERECONSTRUCT_MAPPING.md`, `PLAN_RECONSTRUCT_CONTRAT.md`, ce récap.

**Tests :** `tests/pagereconstruct/test_final_contract.py`, `test_render_ops.py`, `tests/legacy_reconstruction/test_legacy_contracts.py` (+ suite existante) = 146 verts.
