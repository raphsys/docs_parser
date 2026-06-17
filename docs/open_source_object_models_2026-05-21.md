# Modeles open source recommandes pour l'etape Object Comprehension

Date de cadrage: 2026-05-21

## Position

Ne pas chercher un modele unique magique.

Pour `docs_parser`, la meilleure architecture est modulaire:

1. layout detector generaliste
2. OCR / reading order
3. specialiste tableau
4. specialiste formule
5. heuristiques deterministes pour code, references, abbreviations et objets inline techniques

## Profil recommande

### 1. Base generaliste: Surya

Usage conseille:

- detection de lignes
- layout
- reading order
- table recognition
- latex OCR pour les formules recadrees

Forces:

- couvre deja OCR + layout + ordering + tables + latex OCR
- tres bon candidat pour un backend unifie rapide a brancher en POC
- utile pour les pages mixtes et multilingues

Limites:

- pour les tableaux tres complexes, un specialiste table reste preferable
- la taxonomie metier fine doit rester dans `docs_parser`

## 2. Detecteur layout pur: DocLayout-YOLO

Usage conseille:

- detection de regions documentaires a grande vitesse
- pages complexes / riches visuellement
- pre-etape de segmentation d'objets

Forces:

- modele dedie layout analysis
- bon compromis vitesse / precision

Limites:

- ce n'est pas un pipeline complet OCR + tables + formules
- a combiner avec OCR, reading order et specialistes

## 3. Alternative layout/OCR industrialisable: PaddleOCR Layout / PP-Structure

Usage conseille:

- alternative robuste si vous voulez rester dans l'ecosysteme PaddleOCR
- scenarios ou `title`, `table`, `figure`, `header`, `footer`, `reference`, `equation` sont utiles nativement

Forces:

- categories documentaires deja proches du besoin
- ecosysteme structure document reconnu

Limites:

- taxonomie anglaise par defaut parfois plus pauvre que le besoin metier final
- il faut garder votre couche de normalisation / comprehension metier au-dessus

## 4. Specialiste tableau: Table Transformer

Usage conseille:

- detection de tables
- structure rows / columns / cells
- reconstruction cellule par cellule

Forces:

- modele officiel tres adapte a l'analyse tabulaire
- aligne avec un rendu HTML/CSV/cellules

Limites:

- ne remplace ni le layout general ni l'OCR du reste de la page

## 5. Specialiste formule: Surya LaTeX OCR

Usage conseille:

- formules scientifiques / mathematiques / chimiques recadrees
- preservation stricte des expressions non traduisibles

Forces:

- aujourd'hui c'est la voie la plus simple si vous utilisez deja Surya
- remplace avantageusement l'ancien `texify` deprecie

Limites:

- necessite un bon crop / une bonne region formule amont

## Architecture recommandee pour docs_parser

Option A - pragmatique:

- `Surya` pour layout + reading order + OCR + table + latex OCR
- taxonomie `object_comprehension` dans `docs_parser`

Option B - plus specialisee:

- `DocLayout-YOLO` pour layout
- OCR existant / natif actuel
- `Table Transformer` pour structure tableau
- `Surya LaTeX OCR` pour formules
- taxonomie `object_comprehension` dans `docs_parser`

## Decision recommandee

Pour le projet actuel:

1. integrer d'abord la taxonomie et les contrats par type dans `docs_parser`
2. brancher ensuite `Surya` comme backend experimental prioritaire
3. ajouter `Table Transformer` seulement pour les pages `table_block`
4. ajouter `Surya LaTeX OCR` seulement pour `formula_block` / `inline_formula_cluster`
