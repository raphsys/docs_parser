# Session Handoff - 2026-03-18 (suite)

## Objectif produit

Objectif non negociable:

- tous les textes utiles du document original doivent etre presents dans le document final
- tous les textes traduisibles doivent etre traduits
- ces textes doivent etre sur la vraie page reconstruite
- pas d'annexe de rattrapage
- le document final doit etre publication-ready

## Document de reference

- original: `tests/doc_pdf/test_docintelligence-8.pdf`
- sortie courante: `ocr_results/reconstructed_output.pdf`

## Etat courant mesure

Sur le document de reference:

- `content_coverage_score = 1.0`
- `rendered_text_coverage_score = 1.0`
- `english_leak_score = 1.0`
- `word_overlaps = 0`
- `text_img_collisions = 0`
- `visual_similarity_score = 0.7614`
- `publication_ready = false`

Raison bloquante restante:

- `visual_similarity_below_target`

## Corrections faites pendant cette suite de session

### 1. Renderer TOC stabilise geometriquement

Fichier:

- `reconstructor.py`

Corrections gardees:

- leading TOC augmente pour supprimer les chevauchements verticaux
- suppression des overlays immuables sur page `toc`
- conservation d'un rendu sans collisions

Etat obtenu:

- `word_overlaps = 0`
- `text_img_collisions = 0`

### 2. Propagation du vrai style source dans les `toc_rows`

Fichiers:

- `structure_extractor.py`
- `native_pdf_extractor.py`

Corrections:

- les `toc_rows` recuperent maintenant le style source depuis `block.resolved_style` quand les `phrases` n'ont pas de style
- correction du flag italique pour les fontes du type `BoldItali`

Impact:

- amelioration de la fidelite visuelle
- le renderer dispose enfin d'un vrai `style` par entree TOC

### 3. Structuration semantique des entrees TOC

Fichier:

- `structure_extractor.py`

Ajouts:

- role explicite par ligne TOC:
  - `toc_title`
  - `chapter_title`
  - `section_heading`
  - `subentry`
  - `subentry_marker`
- propagation d'un `chapter_number` pour les grands marqueurs de chapitre

Exemple de roles observes:

- `CONTENTS` -> `toc_title`
- `Convolutional neural networks` -> `chapter_title`
- `3.1 Image classification using MLP` -> `section_heading`
- `Input layer` -> `subentry`
- `Hidden layers` -> `subentry_marker`

### 4. Reinjection des grands numeros de chapitre

Fichier:

- `reconstructor.py`

Correction:

- rendu des grands marqueurs visuels `3` et `4` a gauche des chapitres TOC

Impact mesure:

- `visual_similarity_score` est monte jusqu'a `0.7614`

## Resultat important

Le vrai probleme de completude textuelle est regle sur le document de reference:

- tous les textes attendus sont presents
- tous les textes traduisibles sont traduits
- aucun overlap
- aucune collision texte/image

Le seul verrou restant est maintenant la fidelite visuelle fine du sommaire.

## Ce qui a ete essaye puis rejete

### Effacement blanc de la zone TOC avant rerendu

Decision:

- rejete

Raison:

- degradait la similarite visuelle

### Heuristiques typographiques agressives sans base semantique

Decision:

- rejetees

Raison:

- faisaient baisser le score global

## Hypothese technique actuelle

Le renderer TOC n'est plus bloque par:

- la traduction
- la couverture
- la collision
- les overlays

Il reste bloque par:

- une hierarchie typographique encore trop approximative
- un ecart de placement visuel entre:
  - `chapter_title`
  - `section_heading`
  - `subentry`
  - `subentry_marker`

## Prochaine action recommandee

Ne plus toucher:

- a la couverture
- a la QA de completude
- aux overlays TOC

Faire ensuite uniquement:

1. rendre le TOC par `role` avec un mapping explicite base sur le style source:
   - `chapter_title`
   - `section_heading`
   - `subentry`
   - `subentry_marker`
2. mieux caler les marqueurs/puces des `subentry_marker`
3. conserver strictement:
   - `word_overlaps = 0`
   - `text_img_collisions = 0`
   - `coverage = 1.0`

## Fichiers touches dans cette suite

- `native_pdf_extractor.py`
- `structure_extractor.py`
- `reconstructor.py`

