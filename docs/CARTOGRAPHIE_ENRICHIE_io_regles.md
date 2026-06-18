# Cartographie enrichie — Entrées / Sorties / Règles internes

Complément du document `CARTOGRAPHIE_docs_parser_15.md` (qui donne les rôles). Ici, pour chaque fichier du **pipeline WYSIWYG** : entrées (type + origine), sorties (type + destinations), et **règles internes numérotées**.

## Note sur la numérotation des règles

Le code n'annote pas les règles avec des `# RULE-ID:`. Les identifiants de règles réels du projet sont de deux natures :

1. **Règles gérées** (13), formalisées dans `tools/rule_studio/data/managed_rules.yaml` avec un ID stable (ex. `RC-MODE-001`, `NO-DROP-001`). Listées au §0 ci-dessous.
2. **Directives numérotées** citées dans le code : `Lot N`, `PR-Lot N`, `§N`, `NA-N`, `rev_04`. Ce sont les vraies « références de règle » que les fichiers implémentent.

Quand je donne un numéro `[Lot 3]`, `[§14]`, `[PR-Lot 5]`, il est **extrait du code**. Quand une règle n'a pas de numéro d'origine, je la note `[interne]` et je décris la logique réelle (seuils lus dans le code). Aucun numéro n'est inventé.

---
## §0 — Les 13 règles gérées (governance, `managed_rules.yaml`)

| ID | Unité | Libellé | Fichier cible |
|---|---|---|---|
| `PP-COVER-001` | PAGE_CLASSIFICATION | Page 0 à forte densité visuelle + ≥1 image + faible densité texte ⇒ `role=cover`, `family=image_dominant_cover` | `pageprint/detection` |
| `PP-ROLE-001` | SEMANTIC_ROLE | Texte court en bas de couverture matchant un éditeur connu ⇒ `publisher_mark` | `pageprint/role_resolver.py` |
| `PP-ROLE-002` | SEMANTIC_ROLE | Texte court de couverture façon « Prénom Nom » ⇒ `author_name` | `pageprint/role_resolver.py` |
| `RC-MODE-001` | CONSTRAINT | **Rôle inconnu ne devient jamais `paragraph_flow`** ⇒ repli `anchored_text` + warning | `pagereconstruct/plan_compiler.py` |
| `RC-MODE-002` | RECONSTRUCTION | Layouts visuels gardent les textes ancrés | `pagereconstruct/plan_compiler.py` |
| `PP-OCR-001` | PAGE_DETECTION | OCR ciblé sur grandes images en couverture | `pipelines/orchestrator.py` |
| `QA-BBOX-001` | QA | Bbox de fratrie identiques ⇒ géométrie peu fiable | `pagereconstruct/text_measure.py` |
| `NO-DROP-001` | GLOBAL | **Aucun élément original ne disparaît silencieusement** | `pipelines/orchestrator.py` |
| `PT-COVERAGE-001` | TRANSLATION | Couverture complète des textes PagePrint dans PageTranslate | `pagetranslate/functional_validator.py::audit_original_text_coverage` |
| `PR-COVERAGE-001` | RECONSTRUCTION | Toute entrée texte de PageReconstruct produit une sortie | `pagereconstruct/source_text_lifecycle_ledger.py` |
| `RENDER-COVERAGE-001` | QA | Tout texte attendu doit être visible dans le rendu final | `pubready/stages/visual_image_audit.py` |
| `RULE-NL-001` | GLOBAL | Toute règle doit avoir une explication en langage naturel | `tools/rule_studio/agents/rule_interpreter_agent.py` |
| `RULE-CODING-001` | GLOBAL | Toute modif de règle doit pouvoir générer un patch | `tools/rule_studio/agents/rule_coding_agent.py` |

---

## 1. `pipelines/` — orchestration

| Fichier | Entrées (type · origine) | Sorties (type · destinations) | Règles internes |
|---|---|---|---|
| `orchestrator.py` | chemin PDF/image/Office · API/CLI | `INPUT_DATA` par page (`dict`, `pageprint.input.v1`) · `__init__`, `run_document_trial`, audits | `[NO-DROP-001]` aucun élément perdu silencieusement ; `[PP-OCR-001]` OCR ciblé couverture ; enchaîne les 5 unités dans l'ordre fixe |
| `source_loader.py` | chemin fichier · orchestrator | `dict` source (bytes PDF, hash, metadata) · orchestrator | `[interne]` Office→PDF via LibreOffice headless ; calcul hash de provenance |
| `page_renderer.py` | page PDF · orchestrator | image + `scale` (px/pt), dims pt & px · orchestrator | `[interne]` le pixel ne sert qu'à OCR/masque/debug ; le modèle reste en points |
| `raw_extractors.py` | page rendue · orchestrator | extraction brute (texte natif, images, dessins, zones) · orchestrator | `[interne]` NativePDF prioritaire ; OCR paresseux/complémentaire |
| `ocr_router.py` | densités texte/image, claims · orchestrator | décision `full_page \| regions \| none` (`dict`) · orchestrator, tests | `[interne]` texte natif présent ≠ OCR inutile ; route selon image-dominance |
| `page_understanding.py` | extraction brute · orchestrator | page_role, family, document_type, style, relations (`dict`) · orchestrator | `[interne]` hub via `LayoutV2Builder` avant PAGEPRINT |
| `pageprint_stage.py` | compréhension de page · orchestrator | `INPUT_DATA` (`dict`) · orchestrator | `[interne]` séquence normalise→conflits→units→regions→graph→policies→constraints→quality |
| `background_cleaner.py` | image source + bbox texte · orchestrator, démos | image fond « text-removed » · pagereconstruct, démos | `[interne]` inpainting du texte source pour fond propre |
| `background_cover.py` | bbox lignes à redessiner · background_cleaner | fond « couvre-texte » déterministe · background_cleaner | `[interne]` chaque ligne source redessinée est recouverte ; couleur estimée localement |
| `document_context.py` | pages du document · — | en-têtes/pieds répétés, marques (`dict`) · tests | `[interne]` détection de répétition inter-pages |

---

## 2. `pageprint/` — PAGEPRINT (entrée : page comprise → sortie : `INPUT_DATA`)

| Fichier | Entrées (type · origine) | Sorties (type · destinations) | Règles internes |
|---|---|---|---|
| `builder.py` | `page_structure` + intelligence de page · pipeline | `INPUT_DATA` (`dict`) · tout l'aval | orchestre tous les sous-modules ci-dessous |
| `normalizer.py` | bbox/styles/dimensions · builder & co | bbox en **points**, scale px↔pt · builder, evidence, quality | `[interne]` **invariant** : modèle en points, pixels seulement aux frontières image/OCR |
| `unit_factory.py` | `page_structure["blocks"]` (`list[dict]`) · builder | `units[]` plates canoniques · builder, tests | `[interne]` chaque unité porte `UNIT_REQUIRED` ; fusion des spans de phrase |
| `region_index.py` | `page_structure` · builder, policy_compiler | `regions[]` + appartenances (`list[dict]`) · builder | `[interne]` fusionne regions/special/non_text/images/drawings/tables/charts/formulas/code |
| `role_resolver.py` | `units[]` · builder, semantic_builder | rôle + confiance par unité `tuple[str,str,float]` · builder | `[PR-Lot 2]` ; `[PP-ROLE-001/002]` publisher_mark / author_name sur couverture ; `infer_page_role` |
| `evidence_resolver.py` / `evidence/*` | claims concurrentes (native/ocr/layout_ai/detector/heuristics/LLM) · builder | compréhension d'unité résolue (`dict`) · builder | `[interne]` arbitrage : quelle source gagne par type de claim |
| `semantic_builder.py` | rôles + structures logiques · builder | `semantic_system` (`dict`) · builder, audits | `[interne]` unités sémantiques à partir des rôles |
| `graph_builder.py` | `units[]`, `regions[]` · builder | graphe page⊃region⊃block⊃line⊃phrase⊃span (`dict`) · builder | `[interne]` + `build_relations` (ordre de lecture) |
| `policy_compiler.py` | `units[]` + intelligence · builder | policies exécutables (skip_translation, background_only) · builder | `[interne]` les régions candidates sont des **observations**, ne fixent jamais directement les policies |
| `preservation_compiler.py` | rôles/evidence · builder | modes de préservation par unité · builder | `[interne]` mappe rôle → mode de préservation |
| `constraint_compiler.py` | `units[]` · builder | contraintes WYSIWYG (`preserve_bbox`, `allow_reflow`, `allow_wrap`, `preserve_alignment`, `preserve_grid`, `preserve_anchor`) · builder | `[interne]` une contrainte par unité |
| `quality_assessor.py` | `units`, `regions`, structure · builder | score confiance + risques + `downstream_risks` `tuple[dict,dict]` · builder | `[interne]` l'extraction prépare déjà les risques aval |
| `view_compiler.py` | `units[]` · builder | **vues** (`reconstruction_plan`, `preservation_plan`, `exclusion_plan`, `translation_plan`) · pagetranslate, pagereconstruct | `[interne]` les vues SONT le contrat de sortie consommé par l'aval |
| `validators.py` / `functional_validator.py` | `INPUT_DATA` · builder, tests | rapport de validité (`dict`) · builder, audits | `[interne]` bbox en points, `unit_id` uniques, `parent_id`/`region_id` existants, policies cohérentes |
| `text_postprocessors.py` | texte brut (`str`) · semantic_builder, quality | texte dé-césuré (`str`) · semantic_builder | `[interne]` répare césure intra-segment (`unsu- pervised`) et inter-segments |
| `provenance.py` | décisions · builder, source_loader | traces de décision + hashs · builder | `[interne]` chaque décision importante est traçable/rejouable |
| `structure_builders/*.py` | `units[]` · `__init__` | unités logiques typées (`list[dict]`) · semantic_builder | un builder par type (heading, body, list, table, toc `[PR-Lot 2]`, index, caption, figure, formula, code, author_bio, publisher_mark) ; chacun évite la duplication block/line/phrase |

---

## 3. `pagetranslate/` — PAGETRANSLATE (entrée : `INPUT_DATA` → sortie : `translated_input_data`)

| Fichier | Entrées (type · origine) | Sorties (type · destinations) | Règles internes |
|---|---|---|---|
| `builder.py` | `INPUT_DATA` (`dict`) · pipeline | `translated_input_data` + `pagetranslate.output.v1` · pagereconstruct | enchaîne sélection→protection→trad→QA→projection ; fallback **par bloc**, jamais page entière |
| `selector.py` | `INPUT_DATA` · builder | unités traduisibles (`list[dict]`) · builder | `[interne]` priorité `semantic_phrase > group > phrase > line > block` ; exclut `word/char` ; `strip_running_header_page_number` |
| `coalescer.py` | unités visuelles · builder | phrases sémantiques (`list[dict]`) · builder | `[interne]` fusion quand les unités visuelles cassent une phrase |
| `sentence_boundary.py` | unités (`list[dict]`) · builder | unités annotées (début/fin/continuation) · builder | `[interne]` abréviations, multi-ligne, hard break vs soft wrap |
| `protection.py` | texte (`str`) · builder, tests | `(texte_placeholderisé, protections[])` + `restore_text` + `audit_placeholders` · builder | `[interne]` placeholderise URL/DOI/email/nombre/unité/formule/chemin/référence puis restaure |
| `technical_protection.py` | texte + rôle/type · builder | tokens techniques (`list[str]`) · builder | `[Lot 9]` protège `None/True/False`, `Conv2D`, shapes, SQL, chemins, appels de fonction en contexte code/table |
| `context_builder.py` | `INPUT_DATA` · builder | `profile` + unités avec contexte avant/après · builder | `[interne]` page, domaine, style, ton, contraintes WYSIWYG |
| `terminology.py` | `profile` · builder, quality | termes protégés + glossaire post-trad · builder | `[interne]` verrous de glossaire ; cohérence terminologique |
| `translator_bridge.py` | unités · builder | unités traduites (`list[dict]`) · builder | `[rev_04]` appel `DocumentTranslator.translate_text` ; **retry** si sortie vide ou identique |
| `quality.py` | source+trad+profile · audits, dashboard | `unit_quality` + `needs_review` (`dict`) · audits | `[interne]` contrôle expansion, tokens protégés, nombres, unités |
| `text_survival.py` | `INPUT_DATA` + unités · builder, projection | unités split/réparées + fallbacks lignes non couvertes · builder, projection | `[PT-COVERAGE-001]` **invariant dur** : toute ligne source visible a un chemin de sortie ; privilégie présence du texte sur beauté layout |
| `projection.py` | `translated_input` + unités traduites · builder, tests | `INPUT_DATA` traduit + vue reconstruction · pagereconstruct | `[interne]` réinjecte `content.translated_text` ; conserve rôles, bboxes, tous les originaux |
| `functional_validator.py` | résultat traduit (`dict`) · builder, tests | rapport + `audit_original_text_coverage` · builder | `[PT-COVERAGE-001]` couverture des textes PagePrint |
| `translation_plan_reader.py` | `INPUT_DATA.views.translation_plan` · builder | plan de traduction (`list[dict]`) · builder | `[interne]` lit le plan compilé par PAGEPRINT |

---

## 4. `pagereconstruct/` — PAGERECONSTRUCT ⭐ (entrée : `translated_input_data` → sortie : PDF/PNG)

> Pipeline : `input_adapter → plan_compiler (layout/style/background/patch) → PageRenderPlan → renderer_dispatcher + renderers → backends → validator`.
> **Hiérarchie de bboxes :** `source_bbox / coverage_bbox / patch_bbox / layout_bbox / anchor_bbox`.

### Entrée & compilation

| Fichier | Entrées (type · origine) | Sorties (type · destinations) | Règles internes (numéro + détail lu dans le code) |
|---|---|---|---|
| `input_adapter.py` | `translated_input_data` (`dict`) · `__init__`, plan_compiler, evaluator | objet normalisé en 4 plans · plan_compiler | `[§5][§20]` source de vérité : texte=`views.reconstruction_units` (pagetranslate), `reconstruction/preservation/exclusion_plan` (pageprint) |
| `plan_compiler.py` | `translated_input_data` · `__init__` | `PageRenderPlan` (couches+protections+consumed/excluded) · backends | `[RC-MODE-001][§8]` rôle inconnu ⇒ `anchored_label_review`, **jamais paragraph** ; `[§6]` anti double-rendu (consumed/excluded) ; `[§9]` aucune unité sans style ; `[§17.1]` invariants ; `[PR-Lot 1]` |
| `schema.py` | — | DTOs `PageRenderPlan`, `TranslatedTextUnit`, `ProtectedRegion`, `PreservedUnit`, `PatchZone` · tout le module | `[interne]` le pipeline ne rend jamais les vues directement : il compile un plan intermédiaire |

### Résolveurs (⚠️ siège de tes deux bugs)

| Fichier | Entrées | Sorties · destinations | Règles internes |
|---|---|---|---|
| **`layout_box_resolver.py`** 🔴 | `role, layout_bbox, coverage_bbox, anchor_bbox` · plan_compiler | `(layout_bbox, patch_bbox, anchor_bbox, findings)` · plan_compiler | `[Lot 3 — PRIORITÉ ABSOLUE]` **Bug n°1** : pour rôles de **flux** (`body_paragraph, paragraph, body, list_item, author_bio, index_subentry, formula_explanation`), si `hauteur(layout) < 0.5 × hauteur(coverage)` ⇒ `layout = coverage` (le texte se met en page dans le **bloc complet**, pas la 1ʳᵉ ligne). Rôles **titres** : peut rester compact mais doit contenir toute la zone source. Rôles **verrouillés** (`table_*_cell, diagram_label, axis_label, legend_label, code_line, formula_expression`) : `patch = layout`, **pas d'expansion** |
| **`background_resolver.py`** 🔴 | `normalized` (`dict`) · plan_compiler, tests | `{mode, path, source_text_leak_risk, findings}` · plan_compiler | `[Lot 6]` priorité `clean_background > source_background > blank_degraded`. **Bug n°2 (fuite)** : un fichier clean-bg peut encore contenir le texte source ⇒ considéré sûr **seulement** si `clean_background_verified` **ET** `text_removed`. Sinon `risk=high`, patches obligatoires, publication bloquée. `source_background` avec texte traduit ⇒ `risk=high`. Pas de fond ⇒ `blank_degraded`, `risk=high` |
| **`patch_planner.py`** 🔴 | `translated_units, protected_index` · plan_compiler | `(patches[], findings[])` · plan_compiler | `[§14]` chaque unité traduite reçoit un patch sur sa zone source. Bbox PDF souvent trop serrée ⇒ **padding** `x_pad=1.8, y_pad=0.9` (descenders, halos anti-aliasing). `[Phase 7]` un patch chevauchant une région **dure** protégée est **découpé** autour (soustraction guillotine en sous-rects), jamais peint dessus. Méthode `sampled_color_patch` |
| **`font_size_sanitizer.py`** 🔴 | `font_size_pt, line_bbox, role` · style_resolver | `(taille_pt, findings)` · style_resolver | `[Lot 5/A3]` **Bug n°2 (illisible)** : ne répare que `< 6 pt` (sinon intact). Reconstruit depuis `hauteur_ligne × 0.85`. Puis **clamp** par rôle : body/list/bio `7–14`, titres `9–32`, sous-titres `9–28`, cellules table `5.5–12`, diagram_label `4.5–12` ; défaut `5–32` |
| `style_resolver.py` | `reconstruction_unit, recon_plan_item, unit_index, style_system` · plan_compiler, tests | style résolu (`dict`, jamais `{}`) · plan_compiler | `[§9]` chaîne de repli (fiable→moins) : `unit.style → plan_item.style → unit_index → style_system → défaut` ; appelle `font_resolver_bridge` + `font_size_sanitizer` |
| `font_resolver_bridge.py` | nom de police subset PDF (`str`), flags · style_resolver, similarity | classe `serif/sans/mono` (`str`) · style_resolver | `[Lot 5/A2]` les noms subset sont mutilés (`FrctghDrdrdhXjdpbgTimes-`) et le flag serif souvent faux ⇒ inférer la classe par motifs |

### Contrats

| Fichier | Entrées | Sorties · destinations | Règles internes |
|---|---|---|---|
| `final_contract.py` | pageprint + pagetranslate (+legacy) · plan_compiler, bridge | `FinalReconstructionContract` (figé) · plan_compiler | `[interne]` source **unique et non ambiguë** ; `from_pageprint_pagetranslate` = source principale, legacy = complément ; `validate()` |
| `block_contract.py` | `reconstruction_unit` · final_contract, bridge | `BlockReconstructionContract` (géom+style+policy+préservation+qualité) · renderers | `[interne]` les renderers reçoivent CE contrat, pas un dict vague |
| `style_contract.py` | style résolu · block_contract, bridge | `StyleContract` figé · renderers | `[interne]` provenance `extracted/inferred/repaired` pour audit ; `reliable()` |
| `layout_contract.py` | `unit` · block_contract, bridge | `LayoutContract` (LineTemplate, padding, keep_with) · solveurs | `[interne]` reprend `LineTemplate`+`BlockGeometryContext`+`GraphEdge` legacy |
| `background_contract.py` | résolution fond · final_contract, bridge, tests | `BackgroundContract` + `publication_blockers()` · plan | `[interne]` fond non nettoyé = **bloquant** en publication |
| `preservation_contract.py` | `plan` · final_contract, bridge | `PreservationContract.underlays()/overlays()` · overlay_manager | `[interne]` copie région source (pixels) + texte exact |
| `object_contract.py` | `region` · final_contract, bridge | `ObjectContract` (type/role/policy) · final_contract | `[interne]` identité/politique d'objet |
| `quality_contract.py` | — · final_contract | `QualityContract` (gates durs) · final_contract | `[interne]` `must_render` / no clip / no overlap au niveau page |
| `page_level_contracts.py` | `units` · plan_compiler, tests | `PageHeaderContract`, `PageNumberContract`, `SectionHeadingContract` · plan_compiler | `[interne]` ces objets ne passent pas par l'heuristique paragraphe ordinaire |

### Composition & solveurs

| Fichier | Entrées | Sorties · destinations | Règles internes |
|---|---|---|---|
| `composition/block_planner.py` | `contract` · tests | `BlockPlacementPlan` · — | `[interne]` place les grands blocs avant l'intra-bloc (net-improvement gardé) |
| `composition/intrablock_composer.py` | `contract` · block_expansion, plan_compiler, render_ops | `LineLayout`, `TextRunPlacement`, `InlineObjectPlacement` · render_ops | `[interne]` le renderer ne compose **plus** ; compose lignes/runs/inline AVANT rendu |
| `composition/typography_planner.py` | `contract` · tests | `TypographyPlan` (échelle page + em + confiance) · — | `[interne]` réutilise `ocr_typography_engine` (cap/x-height) |
| `composition/special_zone_preserver.py` | `plan` · preservation_audit, tests | `SpecialZone[]` (4 niveaux block/inline/page/background) · audits | `[interne]` réutilise protected_regions + PreservationContract |
| `candidate_engine.py` | `RenderCandidate` · placement_solver | `CandidateScore` · placement_solver | `[Phase 7]` 6 scores nommés : text_fit / style_similarity / position / collision / readability / preservation |
| `placement_solver.py` | `block, unit, renderer` · render_ops | meilleur candidat sans collision · render_ops | `[interne]` candidats : normal → shrink → interligne compact → shift local (selon libertés du LayoutContract) |
| `collision_detector.py` | `RenderResults` (géométrie **réelle**) · visual_qa, tests | `{status, max_overlap, collisions}` · visual_qa | `[PR-Lot 4]` texte/texte : `>0.02 → review`, `>0.10 → ko` ; texte/protégé : `>0.01 → review`, `>0.10 → ko` ; tolérance `0.20` si zone code/formule (légende au-dessus = normal) |
| `block_expansion_solver.py` | `contract` · plan_compiler, tests | `MultiBlockSolveResult` · plan_compiler | `[interne]` étend la bbox d'un bloc + reflow voisins (texte traduit plus long) |
| `multiblock_layout_solver.py` | groupes de blocs · expansion, planner, integration, plan_compiler | `MultiBlockSolveResult`, `FlowRegion`, `LayoutPatch` · solveurs | `[interne]` résout collisions de page dense par optimisation de groupes dans les régions de flux |
| `layout_reflow_solver.py` | `contract` · plan_compiler | `SpacingReflowResult`, `ReflowPatch` · plan_compiler | `[interne]` conservateur CPU-only ; préserve les garanties (texte survit, fond propre, objets non déplacés) |

### Rendu : ops, dispatch, backends, renderers

| Fichier | Entrées | Sorties · destinations | Règles internes |
|---|---|---|---|
| `ops.py` / `render_ops.py` | plan · backends, overlay_manager, plan_compiler | `BackgroundOp`, `PatchOp`, `TextOp`, `PreservationOp` (plates, résolues) · backends | `[interne]` dispatch + measure UNE fois ici (espace PT, scale 1.0) ; le backend n'improvise pas ; `assert_publication_background_allowed` |
| `renderer_dispatcher.py` | `renderer_name, role` · render_backend, render_ops, visual_qa | instance renderer · backends | `[Lot 8]` priorité `_BY_RENDERER` puis `_BY_ROLE`, sinon `AnchoredLabelReviewRenderer` — **jamais paragraph** pour un rôle inconnu |
| `text_measure.py` | `text, px, style` · base, render_ops | largeur/wrap/ajustement (pur, sans dessin) · renderers | `[QA-BBOX-001]` bbox de fratrie identiques ⇒ géométrie peu fiable ; **une seule porte de mesure** pour tout le module |
| `backends/pdf_vector.py` | `PageRenderPlan` + `output_path` · final | PDF vectoriel (`dict` résultat) · sortie **finale V1** | `[interne]` fond → rectangles patch → texte via `insert_textbox` (police base-14, taille pt, couleur, alignement) |
| `backends/raster_debug.py` | `plan` + `output_path` · — | PNG **debug** (overlays, contact sheets) · debug | `[interne]` PAS la sortie WYSIWYG |
| `render_backend.py` | `plan` + image source · démos, tests | `PIL.Image` reconstruite · démos, runners | `[interne]` patches effacent le texte source (en respectant protégés) puis dessin du texte traduit avec style résolu |
| `renderers/base.py` | `text, px, style` · render_ops, text_measure | `RenderResult` (géométrie réelle par ligne) · QA | `[PR-Lot 3]` **mesure d'abord, peint ensuite** (QA détecte overlap/overflow sur géométrie réelle) |
| `renderers/{paragraph,heading,list,table_cell,code,formula,index,bibliography,anchored_label,preservation}.py` | `BlockReconstructionContract` · dispatcher | `RenderResult` + ops · backends | `[interne]` un renderer par rôle ; `anchored_label_review` = repli sûr du rôle inconnu |

### Préservation, traçabilité, QA, autocorrection

| Fichier | Entrées | Sorties · destinations | Règles internes |
|---|---|---|---|
| `overlay_manager.py` | `contract` · render_ops | underlays/overlays + z_index + PreservationOps · render_ops | `[interne]` underlay sous le texte (figures/fonds/formules), overlay au-dessus (numéros de page, labels exacts) |
| `protected_region_index.py` | `preservation_plan, exclusion_plan, visual_layers` · plan_compiler, tests | `ProtectedRegionIndex.overlap_ratio()/intersections()` · patch_planner, collision | `[§15]` zones où le renderer n'écrit pas et le patch n'efface pas |
| `text_removal_ledger.py` | textes source remplacés · — | registre (1 ligne/texte : action attendue + vérifiée) · audits | `[interne]` garantit qu'aucun texte source traduit ne fuit |
| `source_text_lifecycle_ledger.py` | `plan, normalized` · plan_compiler, render_ops_audit, tests | `SourceTextLifecycleEntry[]` + `audit_source_text_lifecycle` · audits | `[PR-COVERAGE-001]` 1 ligne/unité texte PagePrint : décision trad → entrée reconstruction → op rendu → statut visuel |
| **`source_text_leak_detector.py`** 🔴 | image source, image reconstruite, patches px · visual_qa, tests | `{leak_count, findings}` · visual_qa | `[PR-Lot 5]` un patch dont le **changement moyen de gris < 12.0** entre source et reconstruit = **ancien texte non effacé = leak** |
| `invariant_guard.py` | `normalized, contract, render_ops, background` · plan_compiler | résumé d'invariants (`dict`) · plan_compiler | `[interne]` verrouille : (1) toute ligne PAGEPRINT visible a un owner ; (2) fond propre vérifié/text_removed ; (3) pas de fallback silencieux |
| `quality.py` | `plan` · validator, visual_qa, builder | métriques (`dict`) · validator | `[Lot 10]` dérive text_units, styled_units, overflow, leak_risk… du plan |
| `validator.py` | `plan` · — | `{status: ok\|review\|ko, findings, publication_ready}` · QA finale | `[Lot 10]` **ko** : texte attendu absent / fond manquant avec texte / `styled_units==0` / collision ko. **review** : `unresolved_style, layout_repaired, overflow, patch_protected_overlap, unknown_renderer, leak_risk=high`. `publication_ready` exige text_presence≥1.0, non_text≥0.99, overlap≥0.99, typography≥0.95, position≥0.95, leak≥0.98, score≥0.95 |
| `style_similarity.py` | `source_style, resolved_style` · — | `{score, status, components}` · QA | `[PR-Lot 6]` pondération font_class .30 / size .25 / bold .15 / italic .10 / color .10 / alignment .10 ; **≥0.95 ok**, ≥0.85 review, **<0.85 ko** |
| `visual_qa.py` | `plan` + géométrie mesurée · validator | scores 6 critères + `publication_ready` · validator | `[PR-Lot 8/9]` text_presence, non_text_presence, overlap, position, typography |
| `autocorrect/correction_loop.py` | `compile_fn, audit_fn` · tests | `LoopResult` (meilleur score gardé) · — | `[interne]` la validation **corrige** : compile→audit→corrige(knob)→recompile→garde net-improvement, `max_iter`, anti-jam |
| `autocorrect/correction_plan.py` | `page_report` · correction_loop | `CorrectionAction[]` · correction_loop | `[interne]` findings d'audit → actions |
| `legacy_contract_bridge.py` / `legacy/function_registry.py` | `page_data` legacy · final_contract | contrats modernes + registre de migration · final_contract | `[interne]` décisions KEEP_AS_IS/ADAPT/WRAP/MERGE/DROP/REPLACE_TESTED |

---

## 5. `pubready/` — QA publication-ready (entrée : plan rendu → sortie : rapport scoré)

> Hiérarchie : **page → blocs → phrases → dimensions**, chaque dimension comparée à l'**origine PAGEPRINT**.

| Fichier | Entrées · origine | Sorties · destinations | Règles internes |
|---|---|---|---|
| `evaluator.py` | input traduit · API | `evaluate_page` (rapport) · reports | `[interne]` normalise via `PageReconstructInputAdapter` puis évalue ; **additif**, ne touche pas le rendu |
| `page_auditor.py` | plan + origine · evaluator | `PagePublicationReadyReport` · document_auditor | `[interne]` lance audits granulaires + QA visuelle, combine selon gates |
| `document_auditor.py` | rapports page · — | score document · reports | `[interne]` **score ≠ moyenne** : une page ko / texte manquant / leak / objet détruit **bloque** le document |
| `gates.py` | scores par dimension · auditors | décision page/document · auditors | `[interne]` seuils stricts par étape + hard blockers explicites |
| `stages/typography_audit.py` | bloc→phrase vs origine · page_auditor | score typo granulaire · page_auditor | `[interne]` compare classe police/taille/couleur/gras/italique/alignement, individuellement puis combiné |
| `stages/translation_audit.py` | blocs traduits · page_auditor | score trad granulaire · page_auditor | `[interne]` présence, non-vide, troncature, tokens protégés conservés, code/formule non traduits |
| `stages/position_audit.py` | bbox rendu vs source · page_auditor | score position · page_auditor | `[interne]` la bbox de rendu reste ancrée sur la zone source, sans dérive ni collision |
| `stages/background_audit.py` | TextRemovalLedger + QA visuelle · page_auditor | audit fond · page_auditor | `[interne]` trame vraiment nettoyée + registre complet + patches non destructeurs |
| `stages/preservation_audit.py` | SpecialZone[] · page_auditor | audit préservation · page_auditor | `[interne]` formule/code/image/table/logo protégés et non écrasés |
| `stages/intrablock_audit.py` | composition intra-bloc · tests | audit intra-bloc · page_auditor | `[interne]` aucun mot perdu, pas d'overflow/clipping, inline présents |
| `stages/contract_audit.py` | `FinalReconstructionContract` · tests | audit contrat · page_auditor | `[interne]` chaque bloc a layout/style/renderer ; layer_order ; préservation |
| `stages/render_ops_audit.py` | `plan.render_ops` · tests | audit ops · page_auditor | `[interne]` BackgroundOp présent, TextOps couvrants, PatchOps non destructrices, **parité PDF/PNG** (même plan) |
| `stages/pageprint_audit.py` | `INPUT_DATA` · page_auditor | audit amont · page_auditor | `[interne]` PAGEPRINT fournit-il assez pour reconstruire |
| **`stages/visual_image_audit.py`** 🔴 | image finale (cv2) · tests | blockers (objet détruit / **leak**) · page_auditor | `[RENDER-COVERAGE-001]` regarde **réellement** l'image : zones protégées ~identiques (sinon objet détruit) ; zones texte remplacé ayant changé (sinon **leak = ancien texte visible**) |
| `evidence.py` | origine pageprint · auditors | style/texte/bbox source par unité · auditors | `[interne]` socle de comparaison reconstruit↔source |
| `reports.py` | rapports · auditors | JSON + Markdown · sortie | `[interne]` export lisible |

---

## 6. `translation_engines/` — moteurs (entrée : `TranslationRequest` → sortie : `TranslationResult`)

| Fichier | Entrées · origine | Sorties · destinations | Règles internes |
|---|---|---|---|
| `base.py` | — | `TranslationEngine` (interface), `TranslationResult` · tout le paquet, renderers | `[interne]` contrat commun |
| `request.py` | — | `TranslationRequest` (DTO) · moteurs | `[interne]` |
| `factory.py` | nom moteur · `__init__` | instance moteur · `__init__` | `[interne]` `create_translation_engine` |
| `model_registry.py` | paire de langues · ct2_engine, tests | `ModelEntry` choisi · ct2_engine | `[interne]` sélection : **Opus pour en→fr**, **fallback M2M100**, chemins relatifs à l'inventaire |
| `ct2_engine.py` | requête · tests | hypothèses décodées · — | `[interne]` batching ; **préfixe de langue** M2M100/NLLB ; pas de mauvais préfixe cible pour Marian |
| `translation_memory.py` | `(source, target, langs)` · builder, tests | hit mémoire (`dict`) · builder | `[interne]` un **hit exact valide** fait sauter le modèle ; **ignore** les entrées non validées |
| `placeholder_policy.py` | contexte · builder, protection, tests | style placeholder choisi · builder | `[interne]` `choose_placeholder_style` + variantes ASCII/XML |
| `profile_store.py` | `translation_profiles.json`, `style_tone_profiles.json` · context_builder | seuils qualité, post-édition, glossaire, ton · context_builder | `[interne]` fichiers manquants ne lèvent **jamais** d'erreur |
| `local_model_engine.py` / `external_model_engine.py` / `rule_engine.py` / `mock_engine.py` | requête · factory, tests | `TranslationResult` · factory | `[interne]` implémentations interchangeables (local / API / règles / factice) |
| `engine_health.py` | moteur · CLI | rapport de santé · check_translation_engine | `[interne]` |

---

## 7. `pipeline_agents/` — agents LLM P1–P7 (interface uniforme)

| Fichier | Entrées · origine | Sorties · destinations | Règles internes |
|---|---|---|---|
| `base.py` | features de bloc/page · agents | interface commune · tests | `[interne]` indépendant du backend (transformers/ct2/gguf) |
| `registry.py` | nom agent + modèle · `__init__`, tests | instance agent · `__init__` | `[interne]` `get_agent("p5_render", model=...)` provider-agnostique (env ou phi35 défaut) |
| `heuristics.py` | features · p1/p3/p5/p6 | estimations déterministes · ces agents | `[interne]` reproduit la logique LLM par règles (rôle, comptage de mots, ratios) — **sans modèle** |
| `p1_extraction.py` | blocs OCR · ocr_server, registry | titres/formules/légendes + frontières sémantiques · aval | `[interne]` segmentation sémantique intra-ligne |
| `p2_structure.py` | blocs de page · registry | hiérarchie parent/enfant, regroupement · aval | `[interne]` |
| `p3_layout.py` | bloc · registry, structure_extractor | mode `inline_reflow / preserve_line_breaks / preserve_paragraphs` · aval | `[interne]` |
| `p4_translation.py` | bloc traduit · registry, translator | score [0,1] + problèmes + version corrigée · aval | `[interne]` post-édition |
| `p4_qe_estimator.py` | bloc traduit · p4_translation | score QE sans LLM · p4_translation | `[interne]` HeuristicQE / CometKiwi/ChrF |
| `p5_render.py` | bloc traduit · registry | stratégie `prose_reflow / label_stack…` · aval | `[interne]` |
| `p6_background.py` | bg_master · registry | zones à re-inpainter, artéfacts résiduels · aval | `[interne]` |
| `p7_publication_layout.py` | page complète · CLI | placements globaux publication-ready · sortie | `[interne]` niveau page, pas bloc |

---

## Périphérie (I/O au niveau module ; règles implicites)

- **Racine / moteur legacy** (`ocr_server.py`, `translator.py`, `reconstructor.py`, extracteurs, classifieurs…) : entrées = PDF/structures legacy, sorties = structures legacy consommées par le pipeline moderne via `legacy_contract_bridge`. Règles = nombreuses heuristiques internes **non numérotées** ; `coverage_validator.py` et `publication_qa.py` portent les invariants de couverture côté legacy. À traiter à la demande, fichier par fichier (lecture nécessaire).
- **`tools/`** : `audit_text_survival.py` `[NA-01→07]` (100% du texte PAGEPRINT retrouvé dans le rendu), `run_functional_audit.py` / `run_batch_functional_audit.py` `[rev_04]`. Le reste = orchestrateurs d'audit/démo/patch sans règles métier propres. `rule_studio/` applique `[RULE-NL-001]` et `[RULE-CODING-001]`.
- **`scripts/`** : runners d'expérimentation (entrée = PDF/dossiers, sortie = rapports/PNG/JSON dans `results/`). Pas de règle métier.
- **`tests/`** : ~125 fichiers qui **encodent les règles en assertions** — ce sont la spécification exécutable. Pour une règle donnée, le test du même nom est la définition la plus précise (ex. `test_source_text_leak.py` ↔ `[PR-Lot 5]`, `test_reconstructor_font_sizing.py` ↔ `[Lot 5/A3]`, `test_layout_and_typography.py` ↔ `[Lot 3]`).

---
