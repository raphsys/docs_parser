J’ai compris. Cette fois il faut traiter la reconstruction comme un **contrôle qualité visuel de publication**, pas seulement comme une sortie technique.

J’ai analysé les 10 pages de `essai_final_10p`, comparé les `source_*.png` et `reconstructed_*.png`, puis relu les plans `pagereconstruct_plan_*.json`, les audits et le code `rev_11`.

Planche utile pour suivre les écarts :

[Voir la planche de comparaison source / reconstruit / overlay](sandbox:/mnt/data/rev11_10p_contact_sheet.png)

Verdict : **on est encore très loin de 95 % publication-ready.** La chaîne a progressé depuis `rev_10`, mais elle est actuellement autour de **25–30 % publication-ready en moyenne**, avec certaines pages presque inutilisables.

---

# 1. Résultat global actuel

Sur les 10 pages :

```text
Pages analysées                         : 10
Pages status ok                         : 0
Pages status review                     : 10
Pages status ko                         : 0, mais plusieurs devraient être ko
Unités de texte reconstruites            : 154
Unités stylées                           : 154
Font sizes réparées                      : 109+
Patchs générés                           : 154
Régions protégées                        : 2214
Patchs chevauchant des zones protégées   : 78
Background mode                          : source_background sur toutes les pages
Source text leak risk                    : high sur toutes les pages
```

Point critique : **toutes les pages sont encore reconstruites sur l’image source originale**, donc le texte anglais reste potentiellement sous le texte français. Les patchs essaient de masquer, mais ils ne nettoient pas proprement la page.

---

# 2. Analyse visuelle page par page

Les pourcentages ci-dessous sont des estimations de QA visuelle, pas des mesures mathématiques exactes. Ils indiquent la proximité avec une page publiable.

| Page                    | Textes présents ? | Hors-texte présent ? | Chevauchements | Positions | Typographie | Publication-ready |
| ----------------------- | ----------------: | -------------------: | -------------: | --------: | ----------: | ----------------: |
| `Advances_p0006`        |             ~40 % |                ~95 % |          ~90 % |     ~45 % |       ~25 % |          **25 %** |
| `Advances_p0094`        |             ~65 % |                ~80 % |          ~45 % |     ~45 % |       ~35 % |          **30 %** |
| `Advances_p0103`        |     ~60 % lisible |                ~95 % |          ~10 % |     ~25 % |       ~30 % |          **15 %** |
| `Advances_p0130`        |             ~80 % |                ~95 % |          ~40 % |     ~65 % |       ~40 % |          **45 %** |
| `Practical SQL_p0069`   |             ~70 % |                ~85 % |          ~35 % |     ~45 % |       ~25 % |          **25 %** |
| `Practical SQL_p0432`   |             ~80 % |                ~90 % |          ~75 % |     ~50 % |       ~25 % |          **35 %** |
| `Practical SQL_p0509`   |     ~30 % lisible |                ~80 % |           ~5 % |     ~10 % |       ~20 % |           **5 %** |
| `DocIntelligence_p0213` |             ~80 % |                ~90 % |          ~70 % |     ~55 % |       ~45 % |          **45 %** |
| `DocIntelligence_p0242` |             ~50 % |                ~70 % |          ~15 % |     ~35 % |       ~35 % |          **15 %** |
| `DocIntelligence_p0320` |             ~50 % |                ~85 % |          ~10 % |     ~35 % |       ~40 % |          **15 %** |

Moyenne réaliste actuelle :

```text
Publication-ready visuel : 25–30 %
Objectif demandé         : > 95 %
Écart à corriger         : très important
```

---

# 3. Problèmes majeurs observés

## 3.1 Tous les textes ne sont pas présents à 100 %

Non. Exemple très net : `Advances_p0006`.

Le texte source du grand bloc fait environ :

```text
2175 caractères
```

La traduction reconstruite fait environ :

```text
848 caractères
```

Ratio :

```text
39 %
```

Visuellement, la page originale contient plusieurs paragraphes, y compris la liste des chapitres. La reconstruction ne garde qu’un résumé court jusqu’à :

```text
Divers défis sont présentés à la fin de chaque chapitre.
```

Donc le problème n’est pas seulement `pagereconstruct`. Il y a aussi un problème dans `pagetranslate` :

```text
traduction tronquée
contrôle de couverture textuelle insuffisant
linguistic_quality_status = ko ignoré par la reconstruction
```

Décision : **pagereconstruct ne doit pas reconstruire une page si `pagetranslate` n’a pas une couverture textuelle proche de 100 %.**

---

## 3.2 Les éléments hors texte sont souvent présents, mais parce que l’image source reste dessous

Les figures, tableaux, schémas, fonds colorés, images et numéros de page restent souvent visibles parce que le backend utilise :

```text
source_background
```

Cela donne une illusion de conservation.

Mais ce n’est pas propre :

```text
ancien texte anglais encore présent ;
patchs approximatifs ;
zones non textuelles parfois recouvertes ;
figures contaminées par texte traduit ;
captions superposées.
```

Exemple : `test_docintelligence_p0242`.

Le fond beige et les schémas sont visibles, mais le texte traduit est superposé à l’ancien texte dans la zone beige. Résultat : non publiable.

---

## 3.3 Les textes se chevauchent fortement

Oui, surtout sur :

```text
Advances_p0103
Practical SQL_p0509
test_docintelligence_p0242
test_docintelligence_p0320
Practical SQL_p0069
```

Exemples :

* `Practical SQL_p0509` : la page d’index devient un amas de lignes superposées. Les entrées source, les traductions, les sous-entrées et les références se mélangent.
* `test_docintelligence_p0320` : les formules mathématiques, les phrases explicatives et les traductions se superposent au centre de la page.
* `Advances_p0103` : la bibliographie est très perturbée, avec des lignes qui se touchent ou se recouvrent.

Cause profonde : le moteur ne fait pas encore une vraie **résolution de collisions** après rendu. Il applique des bboxes et des renderers, mais il ne calcule pas correctement les `actual_text_line_boxes`.

---

## 3.4 Les positions sont partiellement remises, mais sans solveur de mise en page

Les blocs sont souvent remis “au bon voisinage”, mais pas assez précisément.

Problèmes :

```text
gros blancs artificiels ;
blocs trop hauts ou trop bas ;
texte décalé dans les figures ;
figures captions mal placées ;
bibliographie rendue ligne par ligne ;
index non géré comme index ;
header/footer mélangés au contenu ;
formules et explications fusionnées.
```

Exemple `Advances_p0094` :

* le schéma est présent ;
* mais la légende est superposée ;
* le texte avant schéma est déplacé ;
* l’en-tête devient `84 5 Unsupervised Deep Learning Architectures` au lieu d’être correctement séparé.

---

## 3.5 La typographie n’est pas respectée à 100 %

Non. C’est même le second plus gros problème après la présence complète du texte.

Problèmes visibles :

```text
police source non respectée ;
serif/sans parfois faux ;
titres pas assez gras ;
tailles incorrectes ;
corps de texte trop petit ou trop grand ;
justification absente ;
indentations absentes ;
bullets perdus ;
italiques perdus ;
code et prose mélangés ;
caption style non respecté ;
espacement vertical non respecté.
```

Exemple `Practical SQL_p0432` :

Original : texte éditorial grand, typographie forte, hiérarchie claire.

Reconstruit : texte beaucoup trop petit, lignes trop longues, blocs espacés artificiellement.

---

# 4. Causes profondes dans le pipeline

## Cause 1 — `pagetranslate` laisse passer des traductions incomplètes

Exemple `Advances_p0006` :

```text
source chars      : 2175
translated chars  : 848
ratio             : 0.39
```

Pour une traduction EN → FR, le ratio attendu est souvent autour de 0.9 à 1.3, pas 0.39.

Le pipeline sait déjà que certaines traductions sont mauvaises :

```text
linguistic_quality_status = ko
publication_readiness_status = ko/review
needs_review = True
```

Mais `pagereconstruct` reconstruit quand même.

Décision :

```text
Si translation coverage < 0.98 ou linguistic_quality_status != ok :
    reconstruction publication-ready interdite.
    produire seulement debug/review.
```

---

## Cause 2 — `pageprint` classe encore trop mal certaines pages

Cas graves :

```text
Advances_p0103
test_docintelligence_p0242
Practical SQL_p0509
```

Exemple `Advances_p0103` :

```text
roles : 42 table_body_cell
renderers : 42 table
```

Mais la page source est une page de livre : titre + paragraphes + bibliographie.

Donc `pageprint` transforme du texte éditorial en cellule de table. Ensuite `pagereconstruct` utilise `TableCellRenderer`, qui verrouille les bboxes ligne par ligne. Résultat : fragmentation, chevauchements, absence de reflow.

Décision :

```text
Une page de texte justifié ne doit jamais devenir table_page/table_body_cell sans preuve forte de grille.
```

---

## Cause 3 — l’index n’est pas un TOC

Cas `Practical SQL_p0509`.

Les rôles détectés :

```text
toc_entry_title : 29
index_head_term : 1
table_header_cell : 1
```

Mais la page est un **index**, pas une table des matières.

Résultat : les entrées d’index sont rendues comme labels ancrés, avec mauvaises largeurs et mauvaises positions.

Décision :

```text
Créer un IndexRenderer.
Ne pas utiliser AnchoredLabelRenderer pour les pages index.
```

---

## Cause 4 — le backend PNG et le backend PDF ne rendent pas la même chose

Le script de démo produit :

```text
reconstructed_*.png  → via pagereconstruct.render_backend / RasterDebugBackend
reconstructed_*.pdf  → via pagereconstruct.backends.pdf_vector
```

Mais les deux backends ne partagent pas le même moteur.

Pire : dans `pdf_vector.py`, on trouve :

```python
bg = (plan.get("background") or [{}])[0]
```

Or le plan stocke le fond ici :

```python
plan["layers"]["background"]
```

Donc le PDF backend ne lit pas correctement le background.

Autre problème : `pdf_vector.py` ne passe pas par les renderers spécialisés. Il fait directement :

```python
page.insert_textbox(...)
```

Donc les règles `ParagraphRenderer`, `HeadingRenderer`, `CodeRenderer`, `FormulaRenderer`, etc. ne sont pas utilisées dans le PDF final.

Décision :

```text
Un seul moteur de layout doit produire les mêmes ops pour PNG debug et PDF final.
Le backend ne doit pas choisir sa propre logique.
```

---

## Cause 5 — les renderers dessinent mais ne retournent pas de géométrie réelle

Les renderers dessinent le texte, mais ils ne produisent pas :

```text
actual_line_boxes
actual_text_bbox
overflow
clipping
collision report
baseline positions
line count
font used
```

Donc `quality.py` ne peut pas savoir que :

```text
les textes se chevauchent ;
des textes sont sous d’autres ;
un texte est sorti de sa bbox ;
un texte a été réduit excessivement ;
un texte est visuellement illisible.
```

La QA actuelle lit surtout le plan, pas le rendu réel.

Décision :

```text
Le renderer doit simuler/mesurer avant de dessiner.
Puis retourner des RenderResult détaillés.
```

---

## Cause 6 — les patchs ne nettoient pas vraiment la page

Tous les plans signalent :

```text
background_mode = source_background
source_text_leak_risk = high
```

Cela veut dire :

```text
le fond contient encore l’ancien texte.
```

Les patchs sont calculés, mais :

```text
ils chevauchent souvent des régions protégées ;
ils ne sont pas assez strictement bloqués ;
ils ne réinsèrent pas les objets préservés effacés ;
ils ne garantissent pas que l’ancien texte source a disparu.
```

Décision :

```text
Pour publication-ready, source_background + patchs approximatifs est insuffisant.
Il faut clean_background ou text-removal fiable.
```

---

## Cause 7 — la typographie est résolue mais pas fidèle

`style_resolver` existe, mais il reste trop approximatif.

Problèmes :

```text
font_family inconnue → mauvais fallback ;
font_class unknown → parfois rendu sans-serif ;
gras/italique perdus ;
font size corrigée mais pas fidèle ;
line-height approximé ;
pas de styles mixtes dans une même ligne ;
pas de small caps ;
pas de tracking ;
pas de justification réelle.
```

Exemple `Advances_p0006` :

Original : titre “Preface” en serif gras.

Reconstruit : “Préface” beaucoup plus faible, sans style éditorial.

Décision :

```text
La typographie doit être évaluée par classe de police, taille, graisse, italique, couleur, line-height, alignement, indentation et hiérarchie.
```

---

# 5. Conclusion technique

Le système a progressé architecturalement, mais **la philosophie reste trop “dessiner du texte traduit”**.

Pour atteindre 95 % publication-ready, il faut passer à une vraie logique :

```text
PAGEPRINT
  extrait objets, rôles, style, fonds, contraintes

PAGETRANSLATE
  garantit traduction complète et tokens protégés

PAGERECONSTRUCT
  reconstruit par unités logiques, avec solveur visuel réel

VISUAL_QA
  compare source/reconstruit et bloque si non publiable
```

Règle dure :

```text
Aucune page ne doit être considérée publication-ready sans QA visuelle automatique.
```

---

# 6. Plan concret pour Claude CLI

Voici le plan à donner à Claude. Il est orienté correction profonde pour viser **>95 % publication-ready**.

OBJECTIF

Corriger le pipeline PAGEPRINT → PAGETRANSLATE → PAGERECONSTRUCT pour atteindre >95 % publication-ready sur les reconstructions.

Critères obligatoires :

1. 100 % des textes source traduisibles doivent être présents dans la reconstruction.
2. 100 % des éléments hors texte doivent être présents ou explicitement préservés.
3. Aucun chevauchement texte/texte ou texte/objet protégé.
4. Les blocs doivent être replacés dans des zones cohérentes, sans grands blancs artificiels.
5. La typographie source doit être respectée à >95 %.
6. La page finale doit être publication-ready à >95 %.

============================================================
LOT 1 — BLOQUER LES TRADUCTIONS INCOMPLÈTES
===========================================

Problème observé :
Advances_p0006 :
source chars      = 2175
translated chars  = 848
coverage ratio    = 0.39

Le pipeline reconstruit malgré linguistic_quality_status = ko.

À faire :

[ ] Dans pagetranslate, ajouter une métrique translation_coverage_ratio.

Calcul minimal :
translated_chars / source_chars
+ contrôle protected_tokens
+ contrôle segments terminés

[ ] Règle :
Si source_chars > 300 et translated_chars/source_chars < 0.85 :
status = ko
finding = translation_truncated
publication_readiness_status = ko

[ ] Dans pagereconstruct/input_adapter.py ou plan_compiler.py :
refuser mode publication si :
linguistic_quality_status != ok
ou publication_readiness_status == ko
ou translation_coverage_ratio < 0.98

[ ] Ajouter un mode :
reconstruction_mode = "debug" | "publication"

En mode publication :
traduction incomplète = stop.

En mode debug :
produire sortie mais status = ko/review.

Tests :

[ ] test_translation_truncation_blocks_publication_reconstruct
[ ] test_advances_p0006_translation_coverage_detected
[ ] test_pagereconstruct_refuses_publication_when_translation_ko

Critère de validation :
Aucune page avec texte tronqué ne peut sortir en publication-ready.

============================================================
LOT 2 — CORRIGER PAGEPRINT : FAUX TABLES, FAUX TOC, FAUX HEADINGS
=================================================================

Problèmes observés :

Advances_p0103 :
42 table_body_cell
alors que la page est une page livre classique.

Practical_SQL_p0509 :
index traité comme toc_entry_title/anchored_label.

test_docintelligence_p0242 :
texte éditorial et zone figure traités comme table_body_cell.

À faire :

[ ] Dans pageprint/structure_builders/table_builder.py :
renforcer _detect_rows_from_lines.

Une table ne doit être détectée que si au moins 2 signaux forts :
- plusieurs colonnes stables ;
- alignement vertical répété ;
- lignes/bordures ;
- cellules natives ;
- densité tabulaire ;
- largeur de colonnes cohérente ;
- répétition x positions.

Interdire :
une ligne contenant plusieurs espaces ne suffit pas à créer une table.

[ ] Dans pageprint/role_resolver.py :
ne pas faire :
if layout_type == table_dominant: return table_body_cell
sans preuve de table logique réelle.

[ ] Ajouter une correction :
Si page contient beaucoup de paragraphes continus + peu/pas de grille :
page_role = body
roles = body_paragraph / bibliography_entry / heading

[ ] Créer ou renforcer IndexBuilder :
page d’index = entrées alphabétiques + références de pages + indentation.
Ne pas la traiter comme TOC.

[ ] Ajouter rôles :
index_entry
index_subentry
index_page_reference
bibliography_entry
bibliography_heading

[ ] Créer tests réels :

```
test_book_page_not_table_page
test_bibliography_not_table_cells
test_index_page_not_toc_entries
test_figure_text_not_table_cell_without_grid
test_short_heading_not_table_header_cell
```

Critère de validation :
Advances_p0103 ne doit plus produire 42 table_body_cell.
Practical_SQL_p0509 doit produire index_entry/index_subentry.
test_docintelligence_p0242 ne doit pas classer le bloc beige comme table simple.

============================================================
LOT 3 — CRÉER UN INDEX_RENDERER ET UN BIBLIOGRAPHY_RENDERER
===========================================================

Problèmes observés :

Practical_SQL_p0509 :
page d’index totalement illisible.
Les entrées se chevauchent.

Advances_p0103 :
bibliographie rendue comme lignes/table cells, non comme liste structurée.

À faire :

[ ] Créer :
pagereconstruct/renderers/index.py
pagereconstruct/renderers/bibliography.py

[ ] Modifier renderer_dispatcher.py :

```
index_entry        → IndexRenderer
index_subentry     → IndexRenderer
index_page_reference → IndexRenderer
bibliography_entry → BibliographyRenderer
bibliography_heading → HeadingRenderer
```

[ ] IndexRenderer doit gérer :
- indentation ;
- sous-entrées ;
- page references ;
- maintien des termes techniques ;
- pas de traduction automatique des termes d’index techniques sauf si politique claire ;
- pas de superposition.

[ ] BibliographyRenderer doit gérer :
- une entrée = bloc logique ;
- hanging indent ;
- références numériques préservées ;
- URLs/arXiv/DOI préservés ;
- noms propres préservés ;
- reflow contrôlé.

Tests :

[ ] test_index_renderer_no_overlap
[ ] test_index_subentries_keep_indentation
[ ] test_bibliography_entry_hanging_indent
[ ] test_arxiv_doi_preserved
[ ] test_index_terms_not_translated_when_technical

Critère de validation :
Practical_SQL_p0509 doit devenir lisible à >90 %.
Advances_p0103 bibliographie sans chevauchement.

============================================================
LOT 4 — PASSER D’UN RENDERER QUI DESSINE À UN RENDERER QUI MESURE
=================================================================

Problème :
Les renderers dessinent mais ne retournent pas actual_line_boxes.
La QA ne sait donc pas détecter les chevauchements réels.

À faire :

[ ] Créer une structure RenderResult :

RenderResult:
unit_id
renderer
status: ok | review | ko
planned_bbox
actual_text_bbox
actual_line_boxes
line_count
font_used
font_size_used
overflow
clipping
findings

[ ] Modifier BaseRenderer :

```
measure(unit, context) -> RenderResult
render(draw/backend, render_result) -> None
```

[ ] Tous les renderers doivent d’abord mesurer.

[ ] Ajouter CollisionDetector :

```
input: list[RenderResult]
checks:
    text/text overlap
    text/protected overlap
    text/image overlap
    text/formula overlap
    patch/protected overlap
```

[ ] Règle :
overlap_ratio > 0.01 sur hard protected region = ko/review.
overlap texte/texte > 0.02 = review.
overlap texte/texte > 0.10 = ko.

Tests :

[ ] test_renderer_returns_actual_line_boxes
[ ] test_collision_detector_text_text_overlap
[ ] test_collision_detector_text_formula_overlap
[ ] test_collision_blocks_publication_ready

Critère de validation :
Aucun chevauchement majeur ne passe status ok.

============================================================
LOT 5 — CORRIGER LES PATCHS ET LE BACKGROUND
============================================

Problème :
Toutes les pages sont en source_background avec source_text_leak_risk = high.

À faire :

[ ] Brancher un vrai clean_background si disponible :
visual_layers.clean_background_path
assets.background_clean_path
text_removal_strategy output
background_inpainter output

[ ] Dans BackgroundResolver :
priorité :
1. clean_background fiable
2. source_background + patchs validés
3. blank_degraded

[ ] En mode publication :
source_background + source_text_leak_risk high = interdit
sauf si chaque patch zone a été validée comme nettoyée.

[ ] PatchPlanner :
- sampled_color_patch réel ;
- pas de blanc par défaut ;
- inpaint_patch pour zones complexes ;
- patch bloqué si hard protected overlap ;
- patch découpé si possible ;
- réinsertion des preserved_overlays après patch.

[ ] Ajouter SourceTextLeakDetector :

```
Pour chaque patch_bbox :
    comparer crop source vs crop reconstruit.
    si ancien texte encore visible :
        finding = source_text_leak_detected
        status = review/ko
```

Tests :

[ ] test_clean_background_preferred
[ ] test_source_background_forbidden_in_publication_if_leak_high
[ ] test_patch_does_not_cover_figure
[ ] test_patch_does_not_cover_formula
[ ] test_source_text_leak_detector
[ ] test_preserved_overlay_reinserted_after_patch

Critère de validation :
Aucun ancien texte source visible sous texte traduit.

============================================================
LOT 6 — CORRIGER LA TYPOGRAPHIE À 95 %
======================================

Problèmes :
Police, taille, graisse, italique, justification, bullets, hiérarchie non respectés.

À faire :

[ ] Étendre ResolvedTextStyle :

```
font_family_raw
font_family_normalized
font_class
font_size_pt_raw
font_size_pt_resolved
font_weight
bold
italic
color
background_color
line_height_pt
alignment
justification
first_line_indent_pt
hanging_indent_pt
bullet_style
tracking_pt
baseline_ratio
style_confidence
```

[ ] FontResolverBridge :
- détecter familles PDF subset ;
- mapper Times/Janson/Baskerville/Garamond vers serif ;
- mapper Courier/Mono vers mono ;
- mapper Helvetica/Arial/Calibri vers sans ;
- si font_class unknown sur page livre : ne pas forcer sans-serif.

[ ] Ajouter StyleSimilarityScorer :

```
compare source style vs resolved style :
    font_class match
    font_size ratio
    bold match
    italic match
    color delta
    line_height ratio
    alignment match
    indentation match
```

[ ] Règle publication :
style_similarity_score < 0.95 = review
style_similarity_score < 0.85 = ko

[ ] Ajouter support des styles mixtes :
italique dans paragraphe
code inline
noms techniques mono
bullets colorés
bold labels

Tests :

[ ] test_preface_title_keeps_serif_bold
[ ] test_body_paragraph_keeps_serif_and_size
[ ] test_sql_heading_keeps_heavy_bold_style
[ ] test_docintelligence_blue_heading_keeps_color
[ ] test_bullets_preserved
[ ] test_inline_code_preserved_monospace
[ ] test_style_similarity_score_blocks_bad_typography

Critère de validation :
Typographie >=95 % sur pages simples.

============================================================
LOT 7 — UNIFIER PNG DEBUG ET PDF FINAL
======================================

Problèmes :
PNG et PDF ne suivent pas le même chemin.
pdf_vector lit mal le background.

Bug actuel :
pdf_vector.py lit plan.get("background")
alors que le plan contient :
plan["layers"]["background"]

À faire :

[ ] Corriger pdf_vector.py :

```
layers = plan.get("layers") or {}
bg = (layers.get("background") or [{}])[0]
```

[ ] PDFVectorBackend ne doit plus rendre directement depuis translated_text.
Il doit exécuter des RenderOps produits par les renderers.

[ ] Créer une chaîne unique :

```
PageRenderPlan
    → Renderer.measure()
    → RenderOps
    → VisualQA
    → Backend.execute_ops()
```

[ ] RasterDebugBackend et PDFVectorBackend doivent exécuter les mêmes ops.

[ ] Le PNG final de comparaison doit venir du PDF rendu, pas d’un autre chemin.

Tests :

[ ] test_pdf_vector_reads_layers_background
[ ] test_png_and_pdf_use_same_render_ops
[ ] test_pdf_contains_expected_text
[ ] test_pdf_render_matches_debug_png_with_tolerance

Critère de validation :
Plus de divergence PNG/PDF.

============================================================
LOT 8 — AJOUTER VISUAL_QA OBLIGATOIRE
=====================================

But :
Comparer automatiquement source et reconstruction selon les 6 critères demandés.

Créer :
pagereconstruct/visual_qa.py

VisualQA doit produire :

VisualPageAudit:
text_presence_score
non_text_presence_score
overlap_score
position_score
typography_score
publication_ready_score
findings

Méthodes :

1. Text presence :

   * comparer source traduisible attendu vs texte rendu ;
   * contrôler translation_coverage_ratio ;
   * contrôler PDF text extraction ;
   * contrôler RenderResult status.

2. Non-text presence :

   * objets protégés présents ;
   * figures/tableaux/logos/page numbers ;
   * pas de patch destructeur.

3. Overlap :

   * RenderResult actual_line_boxes ;
   * protected region overlap ;
   * image/object overlap.

4. Position :

   * layout_bbox vs source logical bbox ;
   * drift x/y ;
   * distances entre blocs ;
   * conservation de l’ordre de lecture.

5. Typography :

   * StyleSimilarityScorer.

6. Publication-ready :
   pondération stricte :
   text presence 30 %
   non-text presence 15 %
   no overlap 20 %
   positions 15 %
   typography 20 %

Gates :
Si text_presence < 1.0 → publication_ready max 80 %
Si overlap critique → publication_ready max 60 %
Si typographie < 0.95 → publication_ready max 90 %
Si source_text_leak high → publication_ready max 70 %

Tests :

[ ] test_visual_qa_text_presence_100_required
[ ] test_visual_qa_overlap_blocks_ready
[ ] test_visual_qa_typography_blocks_ready
[ ] test_visual_qa_non_text_missing_blocks_ready
[ ] test_visual_qa_publication_ready_score

Critère de validation :
Une page mauvaise ne peut plus être marquée review vague.
Elle reçoit un score précis et des causes.

============================================================
LOT 9 — RÈGLES STRICTES POUR PUBLICATION_READY
==============================================

À faire :

[ ] Ajouter dans validator.py :

Publication-ready seulement si :

```
text_presence_score == 1.0
non_text_presence_score >= 0.99
overlap_score >= 0.99
position_score >= 0.95
typography_score >= 0.95
source_text_leak_risk != high
no critical findings
translation status ok
```

[ ] Si une page échoue :
status = review ou ko
publication_ready_score explicite

[ ] Ne jamais utiliser seulement plan.summary() comme preuve de qualité.

Tests :

[ ] test_bad_page_never_publication_ready
[ ] test_review_page_has_publication_ready_score
[ ] test_critical_finding_forces_ko
[ ] test_all_green_page_publication_ready_above_95

============================================================
LOT 10 — CAS RÉELS À FAIRE PASSER
=================================

Créer fixtures/tests sur les pages observées :

[ ] Advances_p0006
doit détecter traduction tronquée.
ne doit pas sortir publication-ready.

[ ] Advances_p0094
doit préserver figure.
caption ne doit pas chevaucher la figure.
header séparé du titre.

[ ] Advances_p0103
ne doit pas être table_page.
bibliographie sans chevauchement.

[ ] Advances_p0130
figures présentes.
captions propres.
texte autour des figures sans chevauchement.

[ ] Practical_SQL_p0069
code SQL préservé.
prose traduite.
heading style conservé.

[ ] Practical_SQL_p0432
sortie terminal préservée.
texte éditorial garde hiérarchie et taille.

[ ] Practical_SQL_p0509
index lisible.
entrées et sous-entrées indentées.
pas de chevauchement.

[ ] DocIntelligence_p0213
bullets présents.
titres bleus/italiques conservés.
header non fusionné avec body.

[ ] DocIntelligence_p0242
bloc beige propre.
figures intactes.
pas de texte superposé aux schémas.

[ ] DocIntelligence_p0320
formules préservées.
texte traduit autour sans chevauchement.
heading bleu conservé.

Critère final :
après correction, moyenne publication-ready >95 %
aucune page <90 %
aucune page avec texte manquant.

============================================================
ORDRE DE TRAVAIL
================

1. Translation coverage gate.
2. Page role correction : table/index/bibliography/headings.
3. RenderResult + actual_line_boxes.
4. CollisionDetector.
5. BackgroundResolver + SourceTextLeakDetector.
6. StyleSimilarityScorer.
7. IndexRenderer + BibliographyRenderer.
8. PDF/PNG shared RenderOps.
9. VisualQA.
10. Tests réels sur les 10 pages.

Ne pas optimiser le rendu avant que les gates soient en place.
Sinon le système continuera à produire des pages visuellement fausses mais techniquement “review”.

---

# 7. Décision finale

Pour atteindre >95 %, il faut arrêter de considérer `pagereconstruct` comme terminé dès qu’il produit une image.

Le nouveau principe doit être :

```text
Une reconstruction n’existe pas tant qu’elle n’a pas passé VISUAL_QA.
```

Et VISUAL_QA doit répondre explicitement aux 6 questions :

```text
1. Tous les textes sont-ils présents ? 100 % obligatoire.
2. Tous les éléments hors texte sont-ils présents ? 99–100 % obligatoire.
3. Aucun chevauchement ? 99–100 % obligatoire.
4. Positions fidèles ? >=95 %.
5. Typographie fidèle ? >=95 %.
6. Publication-ready ? >=95 %.
```

À l’état actuel, les reconstructions sont utiles pour le debug, mais **elles ne doivent pas être considérées comme sorties finales**. Le plus grand saut de qualité viendra de trois corrections : **gates de traduction complète, correction des rôles pageprint, puis VisualQA avec RenderResult réel**.

