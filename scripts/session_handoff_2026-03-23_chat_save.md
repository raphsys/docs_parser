# Session Handoff 2026-03-23

## Contexte

Cette session a poursuivi le travail de généralisation du pipeline de traduction/reconstruction PDF, avec un focus sur :

- la classification stricte des pages/documents
- le `layout_descriptor`
- la reconstruction pilotée par type de page + descripteur
- la correction progressive des familles de pages sentinelles

Les artefacts de test sont dans :

- `/home/raphael/Mes_Projets/docs_parser/results/saved_regressions`

## Etat Stable Obtenu

Pages actuellement nettoyées / très proches du niveau attendu :

- `26`
- `50`
- `131`
- `148`

Etat validé récemment :

- `26`
  - `publication_ready = true`
  - `content_coverage_score = 1.0`
  - `rendered_text_coverage_score = 1.0`
  - `word_overlaps = 0`
  - `text_img_collisions = 0`
- `50`
  - `publication_ready = true`
  - `content_coverage_score = 1.0`
  - `rendered_text_coverage_score = 1.0`
  - `word_overlaps = 0`
  - `text_img_collisions = 0`
- `131`
  - `publication_ready = true`
- `148`
  - `publication_ready = true`

## Corrections Générales Déjà Intégrées

### Classification / structure

- modèle canonique multi-axes en place :
  - `document_type`
  - `layout_type`
  - `page_role`
  - `style_profile`
  - `regions`
  - `features`
  - `confidence`
- `page_family` conservé seulement comme compatibilité dérivée

Fichiers principaux :

- `page_case_classifier.py`
- `page_profile_registry.py`
- `page_policy_matrix.py`
- `structure_extractor.py`

### Layout descriptor

- `layout_descriptor.v2` en place
- régions synthétiques :
  - `header_band`
  - `text_band`
  - `annotation_band`
  - `caption_band`
  - `table`
  - `table_row`
  - `table_cell`
- contraintes consommées par le reconstructeur

Fichiers principaux :

- `layout_descriptor.py`
- `reconstructor.py`

### Annotated / chart pages

Famille corrigée de façon générale :

- `annotated_page`
- `chart_page`
- labels courts
- captions
- paragraphes explicatifs en `text_band`

Résultat :

- `26` et `50` ne sont plus bloquées

### QA publication

Corrections générales déjà intégrées :

- meilleure gestion des références de figure
- meilleure tolérance sur micro-headings numériques
- meilleure détection `sentence_truncated`
- exclusion des rasters décoratifs minuscules dans les collisions texte/image
- extraction texte rendu via `words` côté QA

Fichiers principaux :

- `coverage_validator.py`
- `publication_qa.py`

## Diagnostic Important Trouvé en Fin de Session

### Pages encore problématiques

Focus final de la session :

- `289`
- `405`

### Cause profonde 1 : mauvais typage de bloc narratif

Sur `289`, le gros bloc `n_2` était classé :

- `role = body`
- `unit_type = reference_link`

Conséquence :

- le bloc entier partait en logique de préservation / non-traduction
- une grande partie du corps restait en anglais

Correction posée :

- dans `page_policy_matrix.py`, une phrase ou un bloc narratif contenant une URL n’est plus automatiquement classé `reference_link`
- `reference_link` reste réservé aux vraies unités dominées par le lien

### Cause profonde 2 : gel abusif des petits titres natifs

Sur `289`, les labels :

- `Human head`
- `Human face`
- `Human nose`

étaient bien typés `short_label`, mais restaient inchangés car ils étaient gelés plus haut dans `translator.py` par `is_likely_figure_label`.

Correction posée :

- suppression du gel abusif des `title` natifs courts
- on ne garde plus automatiquement ces blocs en texte source

### Cause profonde 3 : fallback lexical humain

Fallback court enrichi dans `translator.py` :

- `Human head -> Tête humaine`
- `Human face -> Visage humain`
- `Human nose -> Nez humain`
- plus quelques tokens courts utiles (`human`, `head`, `face`, `nose`, `clothing`)

## Traces Directes Obtenues

Le traçage correct a fini par utiliser `data["structure"]` et non le wrapper retourné par `process_page()`.

Constat confirmé :

- `289 / n_2`
  - bloc corps mal typé avant correction
- `289 / n_11 n_12 n_13 n_16`
  - labels courts encore figés avant correction
- `405`
  - la plupart des blocs narratifs sont déjà traduits
  - le résidu principal concerne la citation/référence finale et la QA sur les headers

## Fichiers Modifiés en Fin de Session

- `page_policy_matrix.py`
- `translator.py`
- `coverage_validator.py`
- `publication_qa.py`
- `tests/test_translation_enrichment.py`
- `tests/test_coverage_validator.py`

## Tests Unitaires

Etat validé en fin de session :

- `python3 -m py_compile page_policy_matrix.py translator.py coverage_validator.py publication_qa.py`
- `.docs-parser/bin/python -m unittest tests/test_translation_enrichment.py tests/test_coverage_validator.py`

Dernier état vu :

- `35 tests OK`

## Point de Blocage Restant

Le rerun de confirmation complet sur `289` et `405` a été relancé après les deux corrections source majeures :

- correction du typage `reference_link`
- suppression du gel abusif des `title` natifs

Mais le dernier tour s’est arrêté pendant les runs de confirmation successifs. Il faut donc reprendre par :

1. relancer `289` et `405` depuis les originaux
2. réécrire :
   - `test_docintelligence-289-strict-classifier-fr-reconstructed.pdf`
   - `test_docintelligence-289-strict-classifier-fr-report.json`
   - `test_docintelligence-405-strict-classifier-fr-reconstructed.pdf`
   - `test_docintelligence-405-strict-classifier-fr-report.json`
3. vérifier si :
   - `n_2` de `289` est désormais traduit
   - `Human head / face / nose` sont maintenant en français
   - les headers de `405` sont toujours faussement vus comme `missing`

## Commande de Reprise Recommandée

Repartir directement sur un rerun réel `289/405` via `.docs-parser/bin/python`, en recopiant les sorties depuis `ocr_results/reconstructed_output.pdf` vers `results/saved_regressions`.

## Résumé Exécutif

La session a consolidé la base générale :

- classification stricte
- descripteur de layout
- reconstruction pilotée par type de page
- correction générale des pages annotées/graphiques

Le dernier gain important est d’avoir enfin trouvé les deux vraies causes profondes de `289` :

- bloc corps mal typé `reference_link`
- labels courts gelés trop tôt dans le traducteur

Ces deux corrections sont posées dans le code. La prochaine session doit surtout confirmer leur effet sur les reruns réels `289/405`.
