# Audit WYSIWYG random20 - 2026-06-01

Source de l'audit: `results/reconstruction_validation_random20_20260601_afterfix4`.

Objectif verifie: pipeline WYSIWYG complet, de la comprehension/classification jusqu'au PDF reconstruit: extraction, traduction, protection des tokens, fond/trame et inpainting, bboxes, recalcul des blocs adjacents, rendu texte 100%, styles, glyphes, couleurs, surimpressions, figures, tableaux et dispositions.

## Verdict global

- Texte present global: 91.2%. Objectif 100% non atteint.
- Texte dans le bloc d'origine: 83.7%. Objectif WYSIWYG non atteint.
- Ordre de lecture: 99.98%. Bon globalement, mais insuffisant car plusieurs pages restent visuellement mal disposees.
- Styles visibles: 59.3%. La conservation taille/famille/flags est le chantier transversal principal.
- Rendu interne: 27 findings, concentres surtout sur TOC et bibliographie.
- Surimpressions source: 13 blocs detectes par validation bloc, meme si le compteur page-level n'en remonte que 5.
- Glyphes: 44 findings, surtout accents, tirets longs et puces carrees.
- Tableaux/cellules: 0 echec, mais aucun vrai `table_cell` n'est expose au validateur dans ce random20. Les grilles/diagrammes sont encore classes en `body`, `title` ou `equation_inline`; le score table est donc trop optimiste.
- Orientation: toutes les pages du run sont portrait, rotation 0, flux principal haut-vers-bas et gauche-vers-droite. Ce run ne valide pas encore paysage, texte vertical, bas-vers-haut, droite-vers-gauche ou texte courbe.

## Pages et conclusions

| Page | Verdict WYSIWYG | Defauts principaux | Correction prioritaire |
|---|---|---|---|
| `doc1_p00` | Partiel | couverture lisible mais titres courts/deplaces, style couleur non audite strictement | renforcer bboxes/titres courts et audit couleur |
| `doc1_p07` | Echec majeur | TOC dense, texte 72.3%, origine 38.1%, 24 findings rendu, 16 verdicts failed, overlays, glyphes, blocs disparus | correction prioritaire 1: renderer TOC traduit par slots ancres |
| `doc1_p15` | Bon avec overlay | texte 100%, style OK, mais source overlay dans le corps | nettoyage overlay paragraphes |
| `doc1_p19` | Bon avec style faible | texte 100%, titre trop petit, overlay source | font floor titres courts + overlay cleanup |
| `doc1_p20` | Bon avec style faible | texte 100%, titre/sous-titre trop petits | seuils de taille par role |
| `doc1_p26` | Partiel | texte 92.5%, origine 60%, encadre conserve mais lignes manquantes et shrink | rebalance bloc principal + encadre |
| `doc1_p27` | Texte OK, WYSIWYG faible | image/schema conserve, texte 100%, style 11.1%, rendu trop petit | ne pas compresser toute la page autour figure+caption |
| `doc2_p04` | Partiel | texte 97.6%, overlays auteurs/legal, legal dense compacte | overlay cleanup + renderer reference/legal |
| `doc2_p06` | Corrige cible | baseline: preface compressee en haut, signatures fusionnees, texte 83.3%; apres correction ciblee: texte/origine/styles 100%, overlays 0 | garder en regression preface/signatures |
| `doc2_p22` | Echec majeur | bibliographie effacee/non reconstruite, texte 19.6%, arXiv perdus, glyphes | renderer bibliographie multi-reference avec texte 100% obligatoire |
| `doc2_p23` | Bon | texte/style/origine OK; figure/formule peu semantisee | baseline sparse/formule |
| `doc2_p27` | Partiel | diagramme conserve mais labels courts IoU faible, styles faibles, warnings ReLU/ResNet | graphe/diagramme avec labels fixes et tokens preserves |
| `doc2_p33` | Bon avec overlay/style | texte 100%, header trop petit, overlay corps | overlay cleanup + font floor header |
| `doc2_p37` | Partiel | grille couleur non classee table, bloc disparu, styles faibles | classifier grid/table et reflow autour figure |
| `doc3_p11` | Partiel | sommaire Practical SQL, origine 60%, mots joints, overlays | mode TOC multi-niveaux + indentation |
| `doc3_p12` | Corrige cote texte/placement cible | baseline: TOC haut compresse avec chapitre bas, tokens `count()/max()/min()` perdus; apres correction ciblee: texte/origine/proteges 100%, style 80% | reste chantier style/font floor |
| `doc3_p15` | Bon avec style faible | texte 100%, bboxes/style moyens | ameliorer TOC long et footer |
| `doc3_p18` | Corrige cote texte/placement cible | baseline: 4 lignes manquantes, tokens `postgresql/pg_*` perdus; apres correction ciblee: texte/origine/proteges 100%, style 80% | reste chantier style/font floor |
| `doc3_p25` | Bon | texte 100%, illustration/fond OK, style footer moyen | baseline image simple |
| `doc3_p27` | Texte OK, style KO | texte 100%, origine OK, style 0% car police trop petite | relever font floor Practical SQL |

## Priorites

1. `doc1_p07`: corriger le chemin TOC traduit, le rendu slot-by-slot, les collisions, overlays et glyphes. C'est la correction prioritaire car elle touche comprehension TOC, bboxes, texte 100%, rendu PDF et style.
2. `doc2_p22`: bibliographie longue, inpainting et texte 100% obligatoire.
3. `doc2_p06`: recalcul des blocs adjacents et separation preface/signatures corriges en validation PDF ciblee.
4. `doc3_p12`, `doc3_p18`: texte/placement/proteges corriges en validation ciblee; conserver comme regression tests de TOC technique Practical SQL.
5. Styles: font floor par document/role, micro-labels, headers, footers.
6. Classification: renseigner `object_type/object_class`, ajouter contrats `toc_entry`, `bibliography_entry`, `diagram_graph`, `grid_table`.
7. Validation: ajouter audit couleur/fond reel; `background_finding_count=0` ne suffit pas pour affirmer le WYSIWYG.

## Correction prioritaire 1 - `doc1_p07`

Correction implementee:

- Desactivation du renderer page-level TOC quand les `toc_rows` n'ont pas de traduction explicite, afin d'eviter de redessiner le sommaire avec les libelles source.
- Passage des blocs `toc_entry` en rendu par slots ancres quand le contrat est `line_preserve` + `anchored_composite`.
- Preservation du glyphe `■` dans le nettoyage texte.
- Taille de police des slots TOC basee sur la hauteur reelle du slot, au meme titre que les references strictes.
- Correction du faux positif `source_overlay` lorsque le texte source est seulement une sous-chaine de la traduction.

Validation ciblee:

| Mesure `doc1_p07` | Avant random20 afterfix4 | Apres correction prioritaire 1 |
|---|---:|---:|
| Texte present | 72.3% | 100.0% |
| Texte dans le bloc d'origine | 38.1% | 100.0% |
| Styles OK | 42.9% random20 / 4.8% cible courant avant fix style | 81.0% |
| Non-traduit reinsere | 93.9% | 100.0% |
| Tokens proteges | 80.0% | 100.0% |
| Findings rendu | 24 | 0 |
| Verdicts rendu failed | 16 | 0 |
| Source overlays | 10 | 0 |
| Glyphes | 13 | 0 |
| Blocs disparus | 1 | 0 |

Artefacts:

- Avant correction mesuree avec le code courant: `results/reconstruction_validation_doc1_p07_priority1_current/doc1_p07`.
- Apres correction: `results/reconstruction_validation_doc1_p07_priority1_style_glyph_fix/doc1_p07`.

## Correction prioritaire 2 - `doc2_p22`

Diagnostic:

- La page etait bien comprise comme une bibliographie longue (`flow_class=reference_run`, 37 lignes attendues), mais le bloc `n_3` etait reclasse en `code` a cause de termes techniques visibles dans les references.
- Le renderer `CodeBlockRenderer` privilegiait alors les overlays immuables et n'emettait aucun vrai texte pour le bloc bibliographique.
- Une premiere bascule vers le renderer editorial a restaure des operations texte, mais en `prose_reflow` le bloc dense debordait et etait rejete par la validation interne.
- Le bon contrat pour ce cas est `line_preserve`: conserver les lignes source, leurs bboxes et leur ordre, avec une taille de police bornee par le pas vertical reel des lignes.

Correction implementee:

- Priorite donnee aux flux bibliographiques (`reference_run`, `bibliography`, `bibliography_run`, `citation_run`) avant la detection `code/technical`.
- Passage automatique des bibliographies multi-lignes en `line_preserve`.
- Autorisation de `line_preserve` pour les entrees dont la politique de linebreak demande `preserve_source_lines`, meme sans contraintes ligne explicites.
- Ajustement de la taille de police dense pour les bibliographies: minimum plus bas, mais maximum borne par la hauteur de slot et par l'espacement de baseline pour eviter les collisions.

Validation ciblee:

| Mesure `doc2_p22` | Avant correction prioritaire 2 | Apres correction prioritaire 2 |
|---|---:|---:|
| Texte present | 19.57% | 100.00% |
| Texte dans le bloc d'origine | 75.00% | 100.00% |
| Ordre de lecture | 100.00% | 100.00% |
| Geometrie moyenne IoU | 0.647 | 0.881 |
| Retours ligne | 100.00% | 100.00% |
| Styles OK | 75.00% | 25.00% |
| Non-traduit reinsere | 50.00% | 100.00% |
| Tokens proteges | 0.00% | 100.00% |
| Findings rendu | 1 | 0 |
| Verdicts rendu failed | 1 | 0 |
| Source overlays | 0 | 0 |
| Glyphes | 31 | 0 |
| Blocs disparus | 1 | 0 |

Note style:

- La baisse du score style vient du fait que le validateur penalise encore trois blocs pour une taille legerement sous le seuil: `n_0` ratio 0.80, `n_1` ratio 0.77, `n_3` ratio 0.89.
- Pour `n_3`, le choix est volontairement conservateur: la bibliographie complete est maintenant visible, sans overlap, sans glyphes manquants et avec 37/37 lignes presentes. Le prochain chantier style devra relever prudemment les polices sans casser la densite.

Artefacts:

- Avant correction mesuree avec le code courant: `results/reconstruction_validation_doc2_p22_priority2_current/doc2_p22`.
- Apres correction: `results/reconstruction_validation_doc2_p22_priority2_reference_run_fix/doc2_p22`.

Verification tests:

- Cibles reconstruction/traduction/validation: `286 passed, 1 warning`.
- Suite complete: `556 passed, 3 warnings`.

## Correction prioritaire 3 - `doc2_p06`

Diagnostic:

- Le bloc principal `n_0` melange corps de preface, lieu et signatures/auteurs.
- Le rendu prose traitait tout le bloc comme un flux continu, ce qui comprimait le corps en haut et fusionnait les signatures en fin de paragraphe.
- La page source contient une grande respiration verticale avant le lieu/signatures; cette rupture doit etre interpretee comme une queue structurelle ancree, pas comme une suite de prose.

Correction implementee:

- Detection dans le renderer prose des fins de bloc structurelles: grand espace vertical en fin de bloc, lignes courtes, signatures/lieu, ou lignes alignees a droite.
- Reflow du corps uniquement sur la zone prose.
- Rendu de la queue lieu/signatures en linewise fallback sur les ancres source.
- Test unitaire ajoute pour verrouiller la detection `body + signature tail`.

Validation:

- Test cible: `tests/test_reconstructor_font_sizing.py -q` -> OK.
- Validation PDF ciblee finale: `results/reconstruction_validation_doc2_p06_priority3_signature_preserve_final/doc2_p06`.
- Texte present: 83.33% -> 100.00%.
- Texte dans bloc origine: 50.00% -> 100.00%.
- Styles: 50.00% -> 100.00%.
- Source overlays: 1 -> 0.
- Render findings/verdicts failed/glyphes/blocs disparus: 0.

Complement de correction traduction:

- Les lignes preservees de preface/signatures pouvaient recevoir des morceaux de traduction de la ligne suivante (`Srinagar`, noms d'auteurs) ou des expansions tres longues issues du paragraphe complet.
- Ajout d'une reparation finale des lignes `translation_compose_mode=preserved`: expansions pathologiques retraduites localement, signatures/noms propres preserves exactement, localisation retraduite localement (`Srinagar, India` -> `Srinagar, Inde`).

## Correction prioritaire 4 - `doc3_p12` / `doc3_p18`

Diagnostic:

- Les blocs de sommaire Practical SQL sont classes comme `body/prose_reflow`, mais contiennent des lignes structurees et des tokens techniques a conserver.
- Sur `doc3_p12`, les lignes `count()`, `max()`, `min()` disparaissaient du PDF reconstruit; tokens proteges 25%.
- Sur `doc3_p18`, les lignes `postgresql.conf`, `pg_ctl`, `pg_dump`, `pg_restore` disparaissaient; tokens proteges 0%.
- Le rendu editorial evitait le chemin prose quand des `semantic_groups` etaient presents, puis tombait sur un rendu scale/fallback qui pouvait retourner `0` text op pour le bloc.

Correction implementee:

- Ajout d'un garde general pour les blocs `prose_reflow` structures: si le bloc contient des fragments proteges, `technical_inline`, appels `count()`, ou tokens `pg_*` / `postgresql`, le renderer tente d'abord un rendu ligne par ligne valide.
- Ce mode conserve les ancres source et evite que les lignes techniques soient absorbees ou perdues dans un fallback de reflow.
- Test unitaire ajoute pour verrouiller le cas `count()/max()/min()` dans un bloc prose structure.
- Correction du routage renderer: un bloc classe `code` mais sans contrat code explicite et avec contrat `line_preserve` reste dans le renderer editorial, afin que les TOC techniques soient rendus ligne par ligne au lieu de tomber sur le renderer code.
- Correction de la rotation heuristique des micro-marqueurs: un caractere court comme `9` ne doit pas etre tourne a 90 degres uniquement parce que sa bbox est haute/etroite; une rotation explicite reste respectee.

Validation:

- Test cible: `tests/test_reconstructor_font_sizing.py -q` -> `55 passed`.
- Suite complete apres correction: `561 passed, 3 warnings`.
- `doc3_p18`, validation ciblee `results/reconstruction_validation_doc3_toc_technical_linewise_renderer_fix/doc3_p18`: texte 87.10% -> 100.00%, origine 80.00% -> 100.00%, tokens proteges 0.00% -> 100.00%, non-traduits 75.00% -> 100.00%, render findings 1 -> 0, verdicts failed 1 -> 0, blocs disparus 0.
- `doc3_p12`, validation ciblee `results/reconstruction_validation_doc3_p12_rotation_marker_fix/doc3_p12`: texte 90.32% -> 100.00%, origine 60.00% -> 100.00%, tokens proteges 25.00% -> 100.00%, non-traduits 75.00% -> 100.00%, render findings 1 -> 0, verdicts failed 1 -> 0, blocs disparus 1 -> 0.
- Limite restante: styles a 80% sur `doc3_p12` et `doc3_p18`; c'est maintenant le chantier transversal `font floor / famille / flags`, plus un probleme de disparition texte.
