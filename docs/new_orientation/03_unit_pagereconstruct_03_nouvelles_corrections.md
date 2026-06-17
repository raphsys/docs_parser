J’ai inspecté `page.zip` et `show10.zip`. Le nouveau niveau est meilleur architecturalement : `pagereconstruct/` contient maintenant `visual_qa.py`, `collision_detector.py`, `font_resolver_bridge.py`, `font_size_sanitizer.py`, `renderer_dispatcher.py`, des renderers spécialisés, `pdf_vector.py`, etc. Donc Claude a bien commencé à implémenter les corrections demandées.

Mais le résultat visuel n’est toujours **pas publication-ready**.

Planche complète générée ici :

[Voir la comparaison complète source / reconstruit / overlay](sandbox:/mnt/data/show10_full_contact_sheet.png)

---

# 1. Verdict global

Sur les 10 pages de `show10` :

```text
Pages analysées              : 10
Pages publication-ready       : 0 / 10
Pages status ok              : 0 / 10
Pages status review          : 4 / 10
Pages status ko              : 6 / 10
Score moyen déclaré           : 0.629
Score réel publication-ready  : plutôt 35–55 % selon les pages
Objectif                      : >95 %
```

Le pipeline a progressé sur trois points :

```text
1. Les styles ne sont plus vides.
2. Les renderers spécialisés existent.
3. VisualQA et CollisionDetector existent.
```

Mais les pages restent bloquées par quatre défauts fondamentaux :

```text
1. source_background partout → ancien texte encore potentiellement visible ;
2. collisions texte/texte et texte/zones protégées ;
3. typographie encore très approximative ;
4. rôles documentaires encore faux dans plusieurs cas.
```

Donc : **on a maintenant une meilleure architecture de diagnostic, mais pas encore une reconstruction publiable.**

---

# 2. Analyse page par page

## 2.1 `Advances in Deep Learnin_p0140`

```text
Status audit       : ko
Ready score        : 0.60
Text units         : 18
Font repaired      : 18 / 18
Patch overlap      : 1
Leak risk          : high
```

Problèmes visibles :

```text
- Le graphe est présent, mais la reconstruction mélange labels de figure et texte.
- Beaucoup de petits labels du graphique sont classés section_heading.
- Le texte traduit est présent, mais typographiquement très faible.
- Les éléments hors texte sont visibles surtout parce que la page source sert de fond.
- Le moteur signale des chevauchements texte/texte et texte/protected.
```

Cause profonde :

```text
PAGEPRINT confond encore certains labels de graphe avec des headings.
PAGERECONSTRUCT applique HeadingRenderer à des micro-labels de figure.
Le background reste source_background, donc pas de vraie page nettoyée.
```

Publication-ready réel : **~45–55 %**.

---

## 2.2 `Practical SQL A Beginner_p0051`

```text
Status audit       : review
Ready score        : 0.70
Text units         : 7
Font repaired      : 1
Patch overlap      : 0
Leak risk          : high
```

Problèmes visibles :

```text
- Page plus simple, donc meilleur rendu relatif.
- Le texte reconstruit existe, mais la mise en page paraît trop légère et pas assez fidèle.
- Les listes/bullets ne sont pas parfaitement conservées.
- Typographie encore éloignée du livre source.
- Source text leak high empêche toute validation publication.
```

Cause profonde :

```text
ListItemRenderer n’est pas réellement utilisé : les list_item sont reconvertis en paragraphes dans le plan.
Le moteur ne respecte pas encore assez indentation, bullet, line-height et densité éditoriale.
```

Publication-ready réel : **~55–65 %**, mais pas publiable à cause du fond source.

---

## 2.3 `Practical SQL A Beginner_p0133`

```text
Status audit       : ko
Ready score        : 0.60
Text units         : 7
Font repaired      : 2
Patch overlap      : 2
Leak risk          : high
```

Problèmes visibles :

```text
- Des blocs protégés ou encadrés sont touchés.
- Les zones techniques/code ne sont pas assez isolées.
- Le texte se rapproche de l’original, mais les patchs cassent certaines zones.
- Deux patchs chevauchent des zones protégées.
```

Cause profonde :

```text
PatchPlanner ne bloque pas assez strictement les patchs.
Les régions protégées existent mais le backend accepte encore trop d’overlap.
```

Publication-ready réel : **~45–55 %**.

---

## 2.4 `Practical SQL A Beginner_p0180`

```text
Status audit       : ko
Ready score        : 0.60
Text units         : 9
Font repaired      : 2
Patch overlap      : 2
Leak risk          : high
```

Problèmes visibles :

```text
- Page technique avec code/commandes.
- Les éléments code sont partiellement conservés, mais la hiérarchie visuelle est faible.
- Certains blocs se rapprochent trop des zones protégées.
- Le rendu global reste inférieur au document source.
```

Cause profonde :

```text
CodeRenderer existe mais n’est pas encore assez dominant.
PAGEPRINT laisse encore trop de contenu technique passer comme body_paragraph.
```

Publication-ready réel : **~45–55 %**.

---

## 2.5 `Practical SQL A Beginner_p0457`

```text
Status audit       : review
Ready score        : 0.70
Text units         : 5
Font repaired      : 0
Patch overlap      : 0
Leak risk          : high
```

Problèmes visibles :

```text
- C’est une des meilleures pages.
- Les blocs sont globalement conservés.
- Mais la typographie reste simplifiée.
- Le fond source empêche toute garantie que l’ancien texte est retiré.
```

Cause profonde :

```text
Le moteur bénéficie d’une page simple.
Ce n’est pas encore une preuve de robustesse.
```

Publication-ready réel : **~60–70 %**, mais pas 95 %.

---

## 2.6 `Practical SQL A Beginner_p0505`

```text
Status audit       : ko
Ready score        : 0.60
Text units         : 21
Font repaired      : 1
Patch overlap      : 1
Leak risk          : high
```

Problèmes visibles :

```text
- Page d’index.
- IndexRenderer existe, mais la page reste structurellement fragile.
- Beaucoup d’entrées sont classées index_head_term.
- Les niveaux d’indentation et références de page ne sont pas encore bien modélisés.
```

Cause profonde :

```text
IndexBuilder détecte partiellement l’index, mais ne structure pas encore :
  index_entry,
  index_subentry,
  page_reference,
  indentation hierarchy.
IndexRenderer est encore un simple BaseRenderer compact.
```

Publication-ready réel : **~35–45 %**.

---

## 2.7 `test_docintelligence_p0192`

```text
Status audit       : ko
Ready score        : 0.562
Text units         : 8
Font repaired      : 8 / 8
Patch overlap      : 2
Leak risk          : high
```

Problèmes visibles :

```text
- Figure/graphique présent, mais texte traduit se rapproche trop de zones protégées.
- Typographie réparée partout, donc style source non fiable.
- Certains textes sont repositionnés, mais l’équilibre page/figure/paragraphes est faible.
```

Cause profonde :

```text
FontSizeSanitizer travaille beaucoup trop : 8 unités sur 8 réparées.
Cela veut dire que PAGEPRINT ou StyleResolver ne fournissent pas encore une taille fiable.
```

Publication-ready réel : **~35–45 %**.

---

## 2.8 `test_docintelligence_p0337`

```text
Status audit       : review
Ready score        : 0.70
Text units         : 7
Font repaired      : 7 / 7
Patch overlap      : 1
Leak risk          : high
```

Problèmes visibles :

```text
- Image conservée.
- Texte traduit lisible.
- Mais typographie presque entièrement réparée artificiellement.
- Un rôle index_head_term apparaît sur une page qui ne semble pas être un index.
```

Cause profonde :

```text
RoleResolver / IndexBuilder produit encore de faux index terms.
StyleResolver compense trop au lieu de recevoir un style fiable.
```

Publication-ready réel : **~50–60 %**.

---

## 2.9 `test_docintelligence_p0406`

```text
Status audit       : ko
Ready score        : 0.53
Text units         : 9
Font repaired      : 9 / 9
Patch overlap      : 4
Leak risk          : high
```

Problèmes visibles :

```text
- Page fortement problématique.
- Plusieurs textes chevauchent des zones protégées.
- Un chevauchement atteint 100 % dans l’audit.
- Le rendu n’est pas publiable.
```

Cause profonde :

```text
ProtectedRegionIndex est trop large ou mal synchronisé avec les unités à rendre.
Certaines unités textuelles devraient être exclues ou rendues autour des zones protégées, pas dessus.
```

Publication-ready réel : **~25–35 %**.

---

## 2.10 `test_docintelligence_p0463`

```text
Status audit       : review
Ready score        : 0.70
Text units         : 6
Font repaired      : 6 / 6
Patch overlap      : 0
Leak risk          : high
```

Problèmes visibles :

```text
- Page plutôt stable.
- Pas de patch overlap.
- Mais toute la typographie est réparée, donc pas réellement fidèle.
- Fond source encore utilisé.
```

Cause profonde :

```text
La page passe parce qu’elle est simple, pas parce que le moteur est robuste.
```

Publication-ready réel : **~55–65 %**.

---

# 3. Réponse aux 6 critères demandés

## 1. Tous les textes sont-ils présents ? Objectif 100 %

Réponse : **presque dans les plans/PDF, mais pas encore garanti visuellement.**

Dans les PDFs reconstruits, l’extraction texte donne souvent un ratio proche de 1.0 par rapport au texte attendu dans le plan. Mais il y a trois réserves graves :

```text
1. Le statut pagetranslate est encore souvent ko/review.
2. Plusieurs unités sont unchanged_suspect ou preserved alors qu’elles devraient être traitées.
3. Texte présent dans le PDF ≠ texte lisible et bien placé visuellement.
```

Exemple :

```text
Advances_p0140 :
linguistic_quality_status = ko
publication_readiness_status = ko
31 unités needs_review sur 37
```

Conclusion : **non, le critère 100 % n’est pas validé.**

---

## 2. Tous les éléments hors texte sont-ils présents ?

Réponse : **visuellement souvent oui, techniquement non garanti.**

Les images, tableaux, graphiques et fonds restent visibles parce que le backend utilise encore :

```text
source_background
```

Mais cela signifie :

```text
ancien texte encore dans le fond ;
préservation non explicitement prouvée ;
patchs pouvant effacer des zones protégées ;
pas de vraie reconstruction des overlays.
```

Conclusion : **présence apparente, mais pas conservation robuste.**

---

## 3. Les textes se chevauchent-ils ?

Réponse : **oui.**

L’audit détecte encore :

```text
text_text_overlap
text_protected_overlap
patch_protected_overlap
```

Exemples :

```text
Advances_p0140       : text_text_overlap ko
test_docintelligence_p0406 : text_protected_overlap ratio 1.0
Practical_SQL_p0133  : text_protected_overlap ratio 0.209
Practical_SQL_p0505  : text_protected_overlap ratio 0.394
```

Conclusion : **critère non validé.**

---

## 4. Les positions sont-elles bien remises ?

Réponse : **partiellement.**

Le gros bug précédent `layout_bbox première ligne` semble corrigé :

```text
bbox_issues détectés : 0
```

C’est une vraie amélioration.

Mais il reste :

```text
pas de solveur de déplacement ;
pas de reflow multi-blocs ;
pas de stratégie autour des figures ;
pas de repositionnement après collision ;
pas de redistribution verticale ;
index/bibliographie encore faibles.
```

Conclusion : **positions améliorées, mais pas publication-ready.**

---

## 5. La typographie est-elle respectée à 100 % ?

Réponse : **non.**

Le style existe maintenant, mais il est souvent réparé :

```text
Font sizes réparées : 54 / 97 unités textuelles environ dans les pages auditées
DocIntelligence_p0192 : 8 / 8 réparées
DocIntelligence_p0337 : 7 / 7 réparées
DocIntelligence_p0406 : 9 / 9 réparées
DocIntelligence_p0463 : 6 / 6 réparées
Advances_p0140        : 18 / 18 réparées
```

Quand un style est “réparé” partout, cela signifie :

```text
le style source n’est pas suffisamment fiable ;
la typographie reconstruite est approximée ;
la fidélité n’est pas prouvée.
```

Conclusion : **typographie non validée.**

---

## 6. La mise en page est-elle publication-ready ? Combien de pourcent ?

Le système donne :

```text
Score moyen déclaré : 62.9 %
```

Mais ce score est trop optimiste, car `VisualQA` reste très simplifiée.

Évaluation réaliste :

```text
Practical_SQL_p0457        : 60–70 %
Practical_SQL_p0051        : 55–65 %
DocIntelligence_p0463      : 55–65 %
DocIntelligence_p0337      : 50–60 %
Advances_p0140             : 45–55 %
Practical_SQL_p0133        : 45–55 %
Practical_SQL_p0180        : 45–55 %
Practical_SQL_p0505        : 35–45 %
DocIntelligence_p0192      : 35–45 %
DocIntelligence_p0406      : 25–35 %
```

Moyenne réaliste : **45–55 % publication-ready**, pas 95 %.

---

# 4. Ce que Claude a corrigé

Claude a bien implémenté plusieurs fondations demandées :

```text
✓ VisualQA existe.
✓ CollisionDetector existe.
✓ RenderResult existe.
✓ Renderers spécialisés existent.
✓ IndexRenderer et BibliographyRenderer existent.
✓ PDFVectorBackend existe.
✓ RasterDebugBackend est séparé.
✓ FontResolverBridge existe.
✓ FontSizeSanitizer existe.
✓ PatchPlanner existe.
✓ BackgroundResolver existe.
✓ Translation coverage gate existe partiellement.
✓ layout_bbox / coverage_bbox semblent mieux séparés.
```

Donc le travail n’est pas nul. Mais il est encore **trop superficiel**.

---

# 5. Les causes profondes restantes

## Cause A — `source_background` partout

Tous les audits indiquent :

```text
background_mode = source_background
source_text_leak_risk = high
```

C’est incompatible avec 95 % publication-ready.

Décision :

```text
En mode publication, source_background + leak high doit bloquer la reconstruction.
```

---

## Cause B — `VisualQA` ne regarde pas encore réellement l’image

`visual_qa.py` calcule surtout à partir :

```text
du plan ;
des RenderResults simulés ;
des collisions géométriques ;
des métriques de qualité.
```

Mais il ne vérifie pas vraiment :

```text
l’image reconstruite ;
l’ancien texte visible ;
la présence visuelle des objets ;
la similarité typographique pixel/forme ;
les différences source/reconstruit par zones.
```

Donc le score est utile, mais insuffisant.

---

## Cause C — `PatchPlanner` signale mais ne bloque pas assez

Dans `patch_planner.py` :

```python
if overlap > 0.05:
    findings.append(...)
patches.append(...)
```

Donc même avec overlap, le patch est ajouté.

Il faut changer la philosophie :

```text
overlap protégé dur → patch bloqué ou découpé
pas seulement finding.
```

---

## Cause D — le backend PDF ne réinsère pas les preserved overlays

Dans `pdf_vector.py`, on voit :

```text
background
patches
translated_text
```

Mais pas de vraie exécution de :

```text
preserved_underlays
preserved_overlays
preservation_ops
copy source region
```

Donc la préservation hors texte reste dépendante du background source.

---

## Cause E — les renderers mesurent, mais ne résolvent pas

Les renderers actuels font :

```text
mesure ;
wrap ;
shrink ;
draw.
```

Mais ils ne font pas :

```text
chercher une position alternative ;
éviter protected region ;
réduire patch ;
déplacer autour d’une figure ;
rééquilibrer blocs voisins ;
réessayer avec plusieurs candidats.
```

Il manque encore un **PlacementSolver**.

---

## Cause F — typographie réparée au lieu d’être fidèle

FontSizeSanitizer répare beaucoup trop.

Cela indique :

```text
PAGEPRINT ne fournit pas encore les bons font sizes ;
ou StyleResolver lit les mauvais objets ;
ou les line_bbox utilisées pour réparer ne sont pas les bonnes.
```

---

## Cause G — rôles encore faux

Exemples :

```text
Advances_p0140 : beaucoup de labels de graphe classés section_heading
DocIntelligence_p0337 : index_head_term sur une page non index
DocIntelligence_p0406 : index_head_term sur une page non index
```

Donc `pageprint/role_resolver.py` et les builders restent trop permissifs.

---

# 6. Directives de correction pour Claude

Voici le plan concret à donner.

OBJECTIF

Après analyse de page.zip et show10.zip, le pipeline est meilleur mais pas publication-ready.

État actuel :
0 page publication-ready
6 pages ko
4 pages review
score moyen déclaré ≈ 0.629
score réel visuel ≈ 45–55 %

Il faut corriger les points restants pour viser >95 %.

============================================================
LOT 1 — BLOQUER SOURCE_BACKGROUND EN MODE PUBLICATION
=====================================================

Problème :
Toutes les pages ont :
background_mode = source_background
source_text_leak_risk = high

C’est incompatible avec publication-ready.

À faire :

[ ] Dans validator.py :
si reconstruction_mode == "publication"
et background_mode == "source_background"
et source_text_leak_risk == "high"
alors status = ko
publication_ready = False
publication_ready_score <= 0.50

[ ] Dans visual_qa.py :
appliquer gate dur :
source_text_leak_risk high → publication_ready max 0.50
actuellement max 0.70 est trop permissif.

[ ] Dans BackgroundResolver :
chercher explicitement :
visual_layers.clean_background_path
assets.background_clean_path
assets.background_path sans texte
inpainted_background_path
text_removed_background_path

[ ] Si clean_background absent :
plan doit déclarer :
publication_blocker = missing_clean_background

Tests :
[ ] test_source_background_high_leak_blocks_publication
[ ] test_clean_background_required_for_publication
[ ] test_visual_qa_caps_score_at_50_for_source_leak_high

============================================================
LOT 2 — PATCHS : BLOQUER OU DÉCOUPER, PAS SEULEMENT SIGNALER
============================================================

Problème actuel :
patch_planner.py ajoute quand même le patch même s’il chevauche une région protégée.

À corriger :

Actuel :
if overlap > 0.05:
findings.append(...)
patches.append(...)

Nouveau :
if hard_overlap > 0.01:
try split patch around protected bbox
if split possible:
emit split patches
finding patch_split_around_protected
else:
patch.status = blocked
finding patch_blocked_protected_overlap
do not paint patch

Règles :
overlap > 0.01 hard protected → pas de patch direct
overlap > 0.10 hard protected → status page ko
patch blocked sur unité traduite → renderer doit soit repositionner, soit page review/ko

Tests :
[ ] test_patch_overlap_hard_region_is_blocked
[ ] test_patch_split_around_protected_region
[ ] test_blocked_patch_forces_review_or_ko
[ ] test_patch_not_added_after_critical_overlap

============================================================
LOT 3 — RÉINSÉRER LES ÉLÉMENTS PRÉSERVÉS
========================================

Problème :
preserved_underlays / preserved_overlays existent dans le plan,
mais les backends ne les exécutent pas vraiment.

À faire :

[ ] Créer PreservationOp dans ops.py :
op_type
unit_id
bbox
method:
copy_source_region
draw_text_exact
preserve_from_background
skip_already_in_clean_background
z_index

[ ] Dans plan_compiler.py :
transformer preserved_underlays / preserved_overlays en PreservationOps.

[ ] Dans raster_debug.py :
copier les régions préservées depuis source image après patch si z_policy == over_text.

[ ] Dans pdf_vector.py :
insérer ces régions comme images ou redessiner texte exact.

[ ] Si source image nécessaire pour preservation :
source_image_path doit être déclaré dans plan.assets ou plan.layers, pas passé caché.

Tests :
[ ] test_preserved_overlay_reinserted_after_patch
[ ] test_page_number_preserved
[ ] test_formula_region_preserved
[ ] test_logo_region_preserved
[ ] test_preservation_op_exists_for_preservation_plan

============================================================
LOT 4 — VISUAL_QA IMAGE-RÉELLE
==============================

Problème :
VisualQA ne regarde pas assez les images.
Il mesure surtout le plan.

À faire :

[ ] Étendre visual_qa.py avec image-level QA optionnelle.

Entrées :
source_image_path
reconstructed_image_path
plan

Mesures :

1. text_presence_score :

   * texte attendu dans RenderResults ;
   * texte extrait du PDF ;
   * unités absentes ;
   * unités vides ;
   * unité avec rendering failed.

2. non_text_presence_score :

   * comparer crops des protected_regions source/reconstruit ;
   * si crop changé fortement sans PreservationOp → erreur.

3. overlap_score :

   * actual_line_boxes ;
   * protected boxes ;
   * image/protected boxes ;
   * pas seulement planned bbox.

4. position_score :

   * comparer centre actual_text_bbox vs layout_bbox ;
   * comparer ordre vertical ;
   * comparer distances inter-blocs.

5. typography_score :

   * StyleSimilarityScorer ;
   * font_class ;
   * font_size ;
   * bold/italic ;
   * color ;
   * line_height ;
   * alignment ;
   * indentation.

6. source_text_leak_score :

   * comparer patch_bbox source/reconstruit ;
   * détecter si le crop reconstruit contient encore structures sombres de l’ancien texte hors nouveau texte.

Tests :
[ ] test_visual_qa_uses_reconstructed_image
[ ] test_non_text_crop_changed_detected
[ ] test_source_text_leak_crop_detected
[ ] test_position_drift_detected
[ ] test_typography_score_uses_style_similarity

============================================================
LOT 5 — RENDERERS : MESURER, PUIS RÉSOUDRE
==========================================

Problème :
Les renderers mesurent mais ne résolvent pas les collisions.
Ils dessinent dans la bbox donnée.

À faire :

[ ] Créer PlacementSolver.

Entrée :
RenderResult candidates
protected_regions
neighboring_results
layout_bbox
allowed_movement

Sortie :
best RenderResult or ko/review.

Candidats :
1. style source normal
2. shrink léger
3. line-height compact
4. bbox expansion si autorisée
5. vertical shift local si autorisé
6. fail review

Règles :
body_paragraph :
peut reflow dans layout_bbox
peut shrink max 14 %
peut expansion verticale seulement si safe

```
heading :
    pas de shrink violent
    pas de déplacement fort

table_cell :
    bbox locked
    si impossible → review/ko

index :
    doit gérer indentation et page refs
    pas de superposition

formula/code :
    preserve par défaut
```

Tests :
[ ] test_solver_avoids_text_text_overlap
[ ] test_solver_avoids_protected_region
[ ] test_solver_fails_when_no_safe_candidate
[ ] test_table_cell_locked_no_shift
[ ] test_paragraph_can_reflow_safely

============================================================
LOT 6 — CORRIGER LES RÔLES RESTANTS
===================================

Problèmes :

* labels de graphe classés section_heading ;
* index_head_term sur pages non index ;
* table/header cell encore faux.

À faire :

[ ] Dans role_resolver.py :
Un micro-texte dans une figure/chart ne doit pas être section_heading.
Si bbox dans figure/chart/plot region :
role = diagram_label / axis_label / legend_label

[ ] Dans index_builder.py :
index_head_term seulement si page_role == index
ou si plusieurs index_entry détectés.
Une seule occurrence isolée ne suffit pas.

[ ] Dans table_builder.py :
La règle len(parts)>=2 sur split par espaces est encore trop permissive.
Exiger :
au moins 3 lignes candidates
colonnes x stables
ou page_role table avec preuve visuelle
ou native cell/table evidence

[ ] Dans heading_builder.py :
vérifier que heading n’est pas dans figure/chart/table/code region.

Tests :
[ ] test_chart_label_not_section_heading
[ ] test_isolated_index_head_term_rejected
[ ] test_space_split_lines_do_not_create_table_without_column_stability
[ ] test_heading_inside_figure_becomes_diagram_label

============================================================
LOT 7 — TYPOGRAPHIE : ARRÊTER LES RÉPARATIONS MASSIVES
======================================================

Problème :
Beaucoup de pages ont font_size_repaired sur presque toutes les unités.
Cela veut dire que la typographie n’est pas fidèle.

À faire :

[ ] Dans StyleResolver :
distinguer :
extracted_font_size_pt
inferred_font_size_pt
rendered_font_size_pt

[ ] Dans FontSizeSanitizer :
ne pas réparer automatiquement toutes les unités.
Si repair_count/page > 30 % :
finding = page_style_unreliable
status minimum = review
publication_ready max 0.80

[ ] Améliorer extraction depuis :
units[].visual.style
source line styles
dominant span style
style_system.body_style
pdf font metrics si disponible

[ ] Ajouter StyleSimilarityScorer dans VisualQA :
font_class match
font_size ratio within ±8 %
bold/italic match
color delta
line_height ratio
alignment
indentation

Tests :
[ ] test_page_many_font_repairs_caps_publication_score
[ ] test_font_size_from_dominant_span_preferred
[ ] test_typography_similarity_under_95_blocks_ready
[ ] test_heading_bold_serif_preserved

============================================================
LOT 8 — BACKEND PDF ET PNG DOIVENT EXÉCUTER LES MÊMES OPS
=========================================================

Problème :
PDFVectorBackend et RasterDebugBackend ne sont pas encore strictement alignés.

À faire :

[ ] Créer RenderOps complets :
BackgroundOp
PatchOp
TextOp
PreservationOp

[ ] Renderer.measure produit RenderResult.
[ ] Renderer.to_ops produit TextOps.
[ ] Backend.execute_ops exécute uniquement les ops.

[ ] RasterDebugBackend :
execute_ops_to_png

[ ] PDFVectorBackend :
execute_ops_to_pdf

[ ] Interdiction :
backend ne doit pas relire layers.translated_text et improviser.
backend ne doit pas choisir renderer lui-même.

Tests :
[ ] test_backends_execute_same_ops
[ ] test_pdf_and_png_have_same_textop_count
[ ] test_pdf_backend_does_not_dispatch_renderers
[ ] test_raster_backend_does_not_dispatch_renderers

============================================================
LOT 9 — DURCIR PUBLICATION_READY
================================

À faire :

Publication-ready seulement si :

```
text_presence_score == 1.0
non_text_presence_score >= 0.99
overlap_score >= 0.99
position_score >= 0.95
typography_score >= 0.95
source_text_leak_score >= 0.98
source_text_leak_risk != high
patch_protected_overlap == 0
critical_findings == 0
translation linguistic_quality_status == ok
publication_readiness_status == ok
```

Tests :
[ ] test_any_ko_translation_blocks_ready
[ ] test_source_leak_high_blocks_ready
[ ] test_patch_overlap_blocks_ready
[ ] test_font_repair_mass_blocks_ready
[ ] test_visual_score_below_threshold_blocks_ready

============================================================
LOT 10 — REJOUER SHOW10 APRÈS CORRECTION
========================================

Après corrections, relancer show10.

Objectifs minimaux :

[ ] 0 page source_text_leak_risk high en publication mode.
[ ] 0 patch_protected_overlap.
[ ] 0 text_text_overlap ko.
[ ] 0 text_protected_overlap ko.
[ ] 0 page avec font_size_repaired > 30 %.
[ ] 0 faux index_head_term isolé.
[ ] 0 chart label classé section_heading.
[ ] 0 backend divergence PNG/PDF.
[ ] score moyen publication-ready > 0.95.
[ ] aucune page < 0.90.

---

# 7. Conclusion nette

Le nouveau code montre que Claude a implémenté une grande partie des **structures demandées**. Mais il n’a pas encore transformé ces structures en vraie reconstruction fiable.

État actuel :

```text
Architecture : 65–70 %
Diagnostic   : 60 %
Rendu réel   : 45–55 %
Publication  : 0 %
```

Le prochain vrai saut ne viendra pas d’un nouveau renderer isolé. Il viendra de ces corrections prioritaires :

```text
1. clean background / suppression source text leak ;
2. patchs bloqués ou découpés, jamais destructeurs ;
3. preservation ops réellement exécutés ;
4. VisualQA image-réelle ;
5. PlacementSolver anti-collisions ;
6. rôles PAGEPRINT durcis ;
7. typographie évaluée par similarité, pas seulement réparée.
```

À ce stade, aucune page ne doit être considérée comme sortie finale. Ce sont encore des sorties de debug avancé.


Je suis fatigué !!! Pageprint/ C'est OK Pagetranslate/ OK Pagereconstruct/ doit juste consommer les éléments des 2 premiers, s'inspirer de ocr_server.py (la partie reconstruction), du contrat de reconstruction, des contrats FinalDocument etc... Logiquement on doit avoir des pages reconstruites publication-ready, au moins à 95%. Qu'est ce qui ne va pas alors ?


Je comprends. Et la réponse courte est celle-ci :

**Le problème n’est probablement plus `pageprint/` ni `pagetranslate/`. Le problème est que `pagereconstruct/` ne se comporte pas comme un consommateur fidèle. Il réinterprète, simplifie et redessine.**

Il devrait faire :

```text
pageprint + pagetranslate + contrats anciens
→ exécution fidèle du contrat de reconstruction
→ page reconstruite
```

Mais il fait encore trop souvent :

```text
données pageprint/pagetranslate
→ nouveau plan simplifié
→ patchs approximatifs
→ texte redessiné
→ QA après coup
```

C’est là que ça casse.

---

# 1. Le vrai problème

Même si `pageprint/` est OK et `pagetranslate/` est OK, `pagereconstruct/` peut échouer si :

```text
il ne consomme pas les bons champs ;
il consomme les champs mais les réinterprète mal ;
il ignore les anciens contrats de reconstruction ;
il n’utilise pas le vrai moteur de reconstruction de ocr_server.py ;
il rend directement au lieu d’exécuter un contrat final.
```

C’est exactement ce que montrent les résultats.

`pagereconstruct/` a maintenant beaucoup de modules, mais la philosophie reste mauvaise par endroits :

```text
il reconstruit comme un renderer,
pas comme un exécuteur de contrat documentaire.
```

---

# 2. Ce qui ne va pas fondamentalement

## 2.1 `pagereconstruct/` ne doit pas “décider” la page

Il ne doit pas redécider :

```text
quel est le rôle du bloc ;
quel est le style ;
quelle est la zone ;
quel élément est protégé ;
quel texte doit être rendu ;
quel objet doit être préservé.
```

Ces décisions doivent venir de :

```text
pageprint/
pagetranslate/
anciens contrats de reconstruction
FinalDocument / document_object_contract
```

`pagereconstruct/` doit seulement faire :

```text
exécuter fidèlement.
```

Aujourd’hui il y a encore trop de logique de “réinterprétation” dans :

```text
plan_compiler.py
style_resolver.py
patch_planner.py
renderer_dispatcher.py
visual_qa.py
```

---

## 2.2 Il manque un vrai `FinalReconstructionContract`

C’est probablement le point central.

Il faut un objet unique, final, non ambigu :

```text
FinalReconstructionContract
```

qui dit pour chaque page :

```text
fond à utiliser ;
zones à nettoyer ;
zones à préserver ;
unités de texte à rendre ;
styles exacts ;
bboxes exactes ;
libertés de déplacement ;
renderer obligatoire ;
ordre des couches ;
contrôles qualité attendus.
```

Actuellement, `pagereconstruct/` consomme plusieurs morceaux :

```text
reconstruction_units
reconstruction_plan
preservation_plan
exclusion_plan
visual_layers
assets
units
regions
```

Mais il manque une étape qui fusionne tout cela en contrat final dur.

Résultat : le reconstructeur improvise.

---

## 2.3 Le moteur ancien savait déjà certaines choses

Dans `ocr_server.py`, `reconstructor.py`, les anciens contrats `FinalDocument`, `document_object_contract`, etc., il y avait des idées importantes :

```text
background nettoyé ;
immutable overlays ;
contrats de bloc ;
contrats de style ;
zones à ne pas toucher ;
rendu par type d’objet ;
reconstruction par couches ;
fallbacks contrôlés ;
audit de rendu.
```

Le nouveau `pagereconstruct/` semble avoir recréé une partie de cela, mais pas encore avec la même maturité.

Erreur stratégique :

```text
on a reconstruit un nouveau moteur au lieu d’extraire proprement l’ancien moteur de reconstruction.
```

Il fallait plutôt faire :

```text
pagereconstruct/
  = adaptateur moderne vers l’ancien savoir de reconstruction
  + nettoyage architectural
  + contrats pageprint/pagetranslate
```

Pas :

```text
pagereconstruct/
  = nouveau renderer expérimental
```

---

# 3. Pourquoi les pages ne sont pas publication-ready

## 3.1 Le fond est mauvais

Tant que la page reconstruite part de :

```text
source_background
```

avec l’ancien texte encore dedans, on ne peut pas garantir une sortie publiable.

Il faut :

```text
clean_background
```

ou :

```text
background sans texte source
```

ou :

```text
inpainting fiable par zone
```

Sinon on cache l’ancien texte avec des patchs. C’est fragile.

---

## 3.2 Les patchs ne sont pas une reconstruction

Un patch blanc ou coloré n’est pas une vraie reconstruction.

Pour être publication-ready, il faut :

```text
retirer proprement l’ancien texte ;
préserver les trames ;
préserver les images ;
préserver les filets ;
préserver les fonds colorés ;
réinsérer les overlays.
```

Si le patch efface une zone protégée ou laisse une trace, la page est non publiable.

---

## 3.3 Les renderers ne doivent pas inventer le layout

Un renderer doit recevoir :

```text
voici la bbox ;
voici le style ;
voici le mode de rendu ;
voici les contraintes ;
voici les zones interdites ;
voici le texte traduit ;
exécute.
```

S’il doit encore résoudre trop de choses, c’est que le contrat amont est insuffisant ou mal consommé.

---

## 3.4 La typographie est encore traitée comme “approximation”

La typographie publication-ready exige :

```text
famille de police ;
classe serif/sans/mono ;
taille ;
graisse ;
italique ;
interligne ;
alignement ;
indentation ;
couleur ;
style mixte ;
code inline ;
indices/exposants ;
petites capitales éventuellement ;
espacement.
```

Si `pagereconstruct/` répare massivement les font sizes, cela veut dire :

```text
il ne reçoit pas ou n’exploite pas la typographie source de manière fiable.
```

Une typographie “réparée” n’est pas une typographie fidèle.

---

## 3.5 La QA arrive trop tard

La QA actuelle dit après coup :

```text
il y a collision ;
il y a leak ;
il y a overlap ;
il y a style repair.
```

Mais pour publication-ready, ces contrôles doivent être intégrés avant le rendu final :

```text
mesurer ;
tester les candidats ;
choisir une solution sans collision ;
bloquer si impossible ;
rendre seulement si valide.
```

Autrement dit : il manque encore un vrai **solveur de placement**.

---

# 4. Ce que devrait faire `pagereconstruct/`

Il doit avoir cette philosophie :

```text
Je ne comprends pas le document.
PAGEPRINT l’a déjà compris.

Je ne traduis pas.
PAGETRANSLATE l’a déjà fait.

Je ne décide pas librement.
J’exécute un contrat final.

Je ne dessine pas approximativement.
Je reconstruis par couches avec contraintes.
```

Architecture correcte :

```text
PAGEPRINT output
+
PAGETRANSLATE output
+
legacy reconstruction contracts
        ↓
FinalReconstructionContract
        ↓
Legacy-aware Reconstruction Engine
        ↓
RenderOps
        ↓
PDF backend / PNG debug backend
        ↓
VisualQA stricte
```

---

# 5. La correction à faire maintenant

Il faut arrêter d’ajouter des modules isolés. Il faut recentrer.

## Étape 1 — créer le contrat final

Créer :

```text
pagereconstruct/final_contract.py
```

Objet :

```text
FinalReconstructionContract
```

Il doit contenir :

```text
page_info
background_contract
text_removal_contract
preservation_contract
translated_text_contract
style_contract
layout_contract
renderer_contract
layer_order_contract
quality_contract
legacy_compatibility_contract
```

C’est cet objet qui devient la seule source du reconstructeur.

---

## Étape 2 — créer un bridge depuis les anciens contrats

Créer :

```text
pagereconstruct/legacy_contract_bridge.py
```

Il doit relire / reproduire la logique utile de :

```text
ocr_server.py
reconstructor.py
document_object_contract.py
FinalDocument
final_page_compiler.py
background_inpainter.py
```

But :

```text
ne pas perdre l’ancien savoir de reconstruction.
```

---

## Étape 3 — interdire la reconstruction directe depuis les vues brutes

`pagereconstruct/` ne doit plus rendre directement depuis :

```text
views.reconstruction_units
views.reconstruction_plan
preservation_plan
exclusion_plan
```

Il doit faire :

```text
views + contracts → FinalReconstructionContract → rendu
```

Sinon on restera dans l’improvisation.

---

## Étape 4 — backend = exécuteur, pas décideur

Le backend ne doit pas choisir :

```text
quelle bbox ;
quelle police ;
quel renderer ;
quel fond ;
quel patch.
```

Il doit seulement exécuter :

```text
BackgroundOp
PatchOp
TextOp
PreservationOp
```

---

# 6. Ce que je dirais à Claude maintenant

```text
Stop adding isolated fixes.

The issue is not that pagereconstruct lacks modules.
The issue is that pagereconstruct does not yet execute a single final reconstruction contract.

Implement FinalReconstructionContract.

Inputs:
- pageprint views
- pagetranslate views
- legacy reconstruction contracts from ocr_server/reconstructor/FinalDocument
- assets/backgrounds/overlays

Output:
- one unambiguous page reconstruction contract.

Then make renderers and backends execute only this contract.
No direct rendering from reconstruction_units.
No hidden source background.
No patch unless declared.
No style repair without audit.
No renderer decision inside backend.
No publication-ready without VisualQA.
```

---

# 7. Réponse directe à ta question

**Qu’est-ce qui ne va pas ?**

Ceci :

```text
pagereconstruct/ n’est pas encore un vrai consommateur contractuel.
C’est encore un reconstructeur partiellement autonome qui réinterprète les données.
```

Donc même si :

```text
pageprint/ = OK
pagetranslate/ = OK
```

on obtient une mauvaise reconstruction parce que :

```text
la consommation n’est pas fidèle ;
le contrat final manque ;
l’ancien savoir de reconstruction n’a pas été correctement repris ;
les backends rendent encore trop directement ;
le fond propre n’est pas garanti ;
les patchs ne sont pas une méthode suffisante ;
la typographie est approximée ;
la QA constate les erreurs au lieu de les empêcher.
```

---

# 8. Décision ferme

Pour obtenir 95 %, il faut maintenant imposer cette règle :

```text
Aucune page ne passe au rendu final sans FinalReconstructionContract complet.
```

Et :

```text
pagereconstruct/ doit devenir un exécuteur de contrat,
pas un nouveau moteur qui redécide la page.
```

C’est probablement la pièce manquante depuis le début.

