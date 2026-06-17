# Plan de correction — `pagereconstruct/` = exécuteur de contrat

**Principe directeur:** `pagereconstruct/` ne décide plus. Il fusionne `pageprint` + `pagetranslate` + savoir legacy en **un** `FinalReconstructionContract`, puis compile ce contrat en **RenderOps** plats, exécutés à l'identique par les backends PNG/PDF. Zéro décision dans le backend, zéro rendu direct depuis les vues brutes.

```
PAGEPRINT + PAGETRANSLATE + contrats legacy
   → FinalReconstructionContract (décisions gelées)
   → RenderOps (BackgroundOp / PatchOp / PreservationOp / TextOp)
   → backend PNG | backend PDF (exécution pure)
   → VisualQA stricte
```

---

## Phase 0 — Fond nettoyé ✅ FAIT
- [x] `pipelines/background_cleaner.py` — inpaint texte traduisible → `background_path`
- [x] Câblé dans `orchestrator` → `assets.background_path`
- [x] Vérifié: leak high→low 10/10, avg 0.629→0.692, 1ère page `ok` (p0457 0.99)

---

## Phase 1 — `FinalReconstructionContract` + RenderOps + backends exécuteurs (CŒUR)

### 1A. RenderOps (`pagereconstruct/render_ops.py`)
- [ ] `BackgroundOp(path)` — peindre l'image de fond pleine page
- [ ] `PatchOp(bbox, color, reason)` — effacer un rect (jamais sur zone protégée dure)
- [ ] `PreservationOp(bbox, method, source_path, z)` — `copy_source_region` / `keep_pixels` / `draw_text_exact`
- [ ] `TextOp(lines=[(text,x,y_baseline,font,size,color)], role, z)` — lignes déjà mesurées/résolues
- [ ] `LayerOrder` — ordre dur: background → patches → preserved_underlays → text → preserved_overlays

### 1B. `FinalReconstructionContract` (`pagereconstruct/final_contract.py`)
- [ ] dataclass avec sections nommées:
  - [ ] `page_info` (w/h/dpi/scale)
  - [ ] `background_contract` (path, mode, text_removed)
  - [ ] `text_removal_contract` (zones inpaintées, mask)
  - [ ] `preservation_contract` (zones gardées pixels + overlays exacts)
  - [ ] `translated_text_contract` (unités à rendre: texte FR, bbox, renderer obligatoire)
  - [ ] `style_contract` (police/taille/graisse/couleur/interligne/alignement/indent par unité)
  - [ ] `layout_contract` (bbox exactes + libertés de déplacement par unité)
  - [ ] `renderer_contract` (renderer figé par unité — plus de dispatch backend)
  - [ ] `layer_order_contract`
  - [ ] `quality_contract` (seuils publication attendus)
  - [ ] `legacy_compatibility_contract` (drapeaux issus du bridge)
- [ ] `FinalReconstructionContract.from_plan(normalized, plan)` — fusionne (réutilise les resolvers existants: style/layout/patch/background/protected — ils restent la logique de décision, mais centralisée UNE fois ici)
- [ ] `to_ops()` — compile le contrat en liste ordonnée de RenderOps; **c'est ici que dispatch+measure se font (UNE fois)**, plus dans les backends

### 1C. Backends = exécuteurs purs
- [ ] `backends/raster_debug.py` → `execute_ops(ops, ...) -> PNG` (aucun dispatch/measure)
- [ ] `backends/pdf_vector.py` → `execute_ops(ops, ...) -> PDF` (aucun dispatch/measure)
- [ ] Supprimer tout `from ..renderer_dispatcher import dispatch` des backends
- [ ] Renderers: ajouter `to_ops(unit) -> [TextOp]` (mesure → ops) à `BaseRenderer`
- [ ] Tests: `test_backends_execute_same_ops`, `test_pdf_and_png_same_textop_count`, `test_backend_does_not_dispatch`

---

## Phase 2 — `legacy_contract_bridge.py` (reprendre le savoir ancien)
- [ ] `pagereconstruct/legacy_contract_bridge.py`
- [ ] Porter de `reconstructor.py` / `ocr_server.py`: contrats de bloc, immutable overlays, rendu par type d'objet, fallbacks contrôlés
- [ ] Mapper `document_object_contract` / `FinalDocument` / `final_page_compiler` → sections du `FinalReconstructionContract`
- [ ] Tests: `test_legacy_block_contract_mapped`, `test_immutable_overlay_preserved`

---

## Phase 3 — Interdire le rendu direct depuis les vues brutes
- [ ] `compile_page_render_plan` ne retourne plus un plan « libre » mais **toujours** via `FinalReconstructionContract`
- [ ] Supprimer les chemins qui lisent `reconstruction_units` / `preservation_plan` directement au rendu
- [ ] Garde-fou: rendu impossible sans contrat complet (sinon `publication_blocked`)
- [ ] Test: `test_no_render_without_contract`

---

## Phase 4 — `PlacementSolver` (anti-collision AVANT rendu)
- [ ] `pagereconstruct/placement_solver.py`
- [ ] Entrée: candidats RenderResult, protected_regions, voisins, layout_bbox, libertés
- [ ] Candidats: style normal → shrink léger (≤14%) → interligne compact → expansion verticale si autorisée → shift local → fail
- [ ] Règles par rôle: paragraph reflow OK; heading shrink doux; table_cell verrouillé; formula/code preserve
- [ ] Intégré dans `to_ops()` (résoudre avant d'émettre TextOp)
- [ ] Tests: `test_solver_avoids_text_text`, `test_solver_avoids_protected`, `test_solver_fails_no_safe_candidate`, `test_table_cell_locked`

---

## Phase 5 — Typographie fidèle (stop réparations massives)
- [ ] `StyleResolver`: distinguer `extracted` / `inferred` / `rendered` font size
- [ ] `FontSizeSanitizer`: si réparations/page > 30% → finding `page_style_unreliable`, status ≥ review, pub ≤ 0.80
- [ ] Lire prioritairement: `units[].visual.style`, dominant span, `style_system.body_style`, métriques PDF
- [ ] `StyleSimilarityScorer` dans VisualQA (famille/taille±8%/graisse/italique/couleur/interligne/alignement/indent)
- [ ] Tests: `test_font_size_from_dominant_span`, `test_mass_repair_caps_score`, `test_typography_similarity_gate`

---

## Phase 6 — Durcissement rôles
- [x] Labels de figure/graphe → `diagram_label` non-traduisible (FAIT — à vérifier sur p0140/p0357)
- [ ] `index_head_term` seulement si `page_role==index` OU ≥ plusieurs `index_entry` (stop isolés p0337/p0406)
- [ ] `table_body_cell`: exiger ≥3 lignes + colonnes stables OU evidence cellule native (stop split-espaces)
- [ ] `heading` interdit dans figure/chart/table/code region
- [ ] Tests: `test_chart_label_not_heading`, `test_isolated_index_term_rejected`, `test_space_split_no_table`, `test_heading_in_figure_becomes_label`

---

## Phase 7 — Patchs durs + preservation ops + gates stricts
- [ ] `PatchPlanner`: overlap protégé dur > 0.01 → découper le patch autour OU bloquer (pas seulement signaler)
- [ ] `PreservationOp` réellement exécuté par les deux backends (overlays page_number/formula/logo réinsérés)
- [ ] `validator`: publication-ready seulement si leak≠high, patch_protected_overlap==0, overlap≥0.99, typo≥0.95, position≥0.95, translation ok
- [ ] VisualQA image-réelle (optionnelle): comparer crops source/reconstruit, détecter texte source résiduel
- [ ] Tests: `test_patch_blocked_on_hard_overlap`, `test_preserved_overlay_reinserted`, `test_strict_publication_gates`

---

## Phase 8 — Re-run `show10` + objectifs
- [ ] Rejouer 10 pages
- [ ] Cibles: 0 leak high, 0 patch_protected_overlap, 0 text_text ko, 0 text_protected ko, 0 faux index isolé, 0 label graphe en heading, PNG/PDF alignés, avg > 0.95, aucune page < 0.90

---

## Ordre d'exécution recommandé
1. **Phase 1** (contrat + ops + backends exécuteurs) — fondation, débloque tout le reste
2. **Phase 6 + 7** (rôles + patchs durs + gates) — tue la majorité des KO restants
3. **Phase 4** (PlacementSolver) — collisions résiduelles denses
4. **Phase 5** (typo) — passe de review→ok
5. **Phase 2** (bridge legacy) — fidélité fine
6. **Phase 8** (validation)

---

## CHANTIER FUTUR (acté) — Taille em à l'extraction amont
Pages OCR/scannées: `font_size_pt` = métrique OCR (~4.32pt = line_h_px/sx), pas la vraie taille em. Sanitizer infère (rendu correct) mais → `page_style_unreliable` → plafond **review 0.80**. PDF natifs OK (typo 1.0). **À traiter après la refonte contrat**: faire fournir la vraie taille em par pageprint/extraction (raw_extractors/normalizer `_normalize_style`). C'est le dernier plafond vers >95% sur docs scannés. Voir mémoire `font_size_em_extraction`.
