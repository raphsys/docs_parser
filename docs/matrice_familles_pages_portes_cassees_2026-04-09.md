# Matrice Familles de Pages -> Première Porte Cassée -> Correction de Code

Date: `2026-04-09`

Contexte:
- Audit multi-documents sur `10` pages randoms
- Source: [summary.md](/home/raphael/Mes_Projets/docs_parser/results/random10_multi_docs_audit_20260409/summary.md)
- Objectif: identifier, par famille de pages, la première porte qui casse, puis en déduire les corrections de code à prioriser.

## Règle de lecture

On ne part pas d’un symptôme visuel.
On part de la première porte qui s’effondre dans la chaîne:

1. `P1 Extraction`
2. `P2 Appartenance bloc`
3. `P3 Qualité blocs / cadres / éléments`
4. `P4 Traduction`
5. `P5 Reconstruction / placement`
6. `P6 Fond`

La première porte cassée est le premier endroit où le pipeline cesse d’être fiable.

## Familles observées

### 1. Pages académiques denses

Exemple directeur:
- `Advances in Deep Learning.pdf`, page `101`

Scores typiques observés:
- `P1 = 0.0`
- `P3 = 0.6734`
- `P5 = 0.2996`
- `P6 = 0.7368`

Symptômes visibles:
- blocs éditoriaux longs
- nombreuses lignes serrées
- formules inline, symboles, indices, petites lettres isolées
- spans démesurés ou au contraire micro-fragments perdus

Première porte cassée:
- `P1 Extraction`

Pourquoi:
- les unités textuelles visibles ne sont pas extraites de façon exploitable
- la granularité est mauvaise:
  - micro-symboles isolés perdus
  - spans anormalement longs
  - mélange texte continu / symboles / indices

Conséquence en chaîne:
- `P3` casse ensuite, car les cadres rouges n’ont plus une sémantique propre
- `P5` casse ensuite, car le reconstructor travaille sur un payload déjà corrompu

Correction de code à prioriser:
- `ocr_server.py`
- `structure_extractor.py`

Travaux:
- extraire séparément:
  - texte éditorial continu
  - symboles isolés
  - indices / exposants
  - variables courtes
  - fragments de formule inline
- éviter les spans “gloutons” qui avalent un paragraphe entier en une seule unité
- améliorer la segmentation des phrases et spans dans les blocs très denses
- fiabiliser les bboxes des micro-unités

Critère de succès:
- `P1` doit remonter avant toute correction `P5`

---

### 2. Pages SQL / code / listes annotées

Exemple directeur:
- `Practical SQL A Beginner’s Guide to Storytelling with Data.pdf`, page `99`

Scores typiques observés:
- `P1 = 1.0`
- `P3 = 0.9217`
- `P5 = 0.4348`
- `P6 = 0.9251`

Symptômes visibles:
- structure source bien extraite
- texte bien traduit
- symboles de renvoi `➊➋➌➍➎`
- lignes SQL longues
- placement local faux

Première porte cassée:
- `P5 Reconstruction / placement`

Pourquoi:
- le payload est globalement bon
- le moteur reconstruit mal les unités contraintes:
  - symboles annotatifs
  - longues lignes de code/DDL
  - relations label -> valeur -> continuation

Conséquence en chaîne:
- contenu présent, mais placé dans de mauvais slots
- collisions locales et perte de fidélité visuelle

Correction de code à prioriser:
- `reconstructor.py`

Travaux:
- ajouter un mode local plus strict pour:
  - longues lignes techniques
  - code SQL / DDL
  - listes annotées avec marqueurs graphiques
- traiter les marqueurs `➊➋➌...` comme unités protégées, non absorbables
- imposer un placement par sous-rangées locales dans les blocs techniques
- mieux respecter la relation:
  - marqueur
  - champ
  - valeur
  - continuation

Critère de succès:
- `P1` ne doit pas bouger
- `P5` doit monter sans perte de couverture

---

### 3. Pages tutoriel technique mixtes

Exemple directeur:
- `test_docintelligence.pdf`, page `155`

Scores typiques observés:
- `P1 = 0.0`
- `P3 = 0.9372`
- `P5 = 0.4730`
- `P6 = 0.1111`

Symptômes visibles:
- titres, code, labels, mini-annotations
- fragments comme `LOAD`, `PREPROCESSING`, `F1`, `F2`
- fond encore pollué
- texte finement fragmenté ou perdu

Premières portes cassées:
- `P1 Extraction`
- `P6 Fond`

Pourquoi:
- l’extraction ne tient pas sur les petites unités techniques
- le fond reconstruit garde des résidus ou retire mal certaines zones

Conséquence en chaîne:
- `P5` casse, mais secondairement
- le placement final n’est pas la cause racine

Correction de code à prioriser:
- `ocr_server.py`
- logique de génération du fond maître
- audit / nettoyage du background

Travaux:
- mieux extraire les mini-labels techniques
- empêcher le découpage absurde en fragments partiels (`L`, `OAD`, etc.)
- améliorer le retrait des éléments source pour produire un vrai fond propre
- vérifier que les zones nettoyées correspondent exactement aux unités extraites

Critère de succès:
- `P6` doit monter fortement en même temps que `P1`

---

## Portes stables

### `P2 Appartenance bloc`

Constat:
- stable sur les `10` pages du lot
- `1.0` partout

Conclusion:
- ce n’est pas la priorité actuelle
- ne pas réécrire cette couche sans raison

### `P4 Traduction`

Constat:
- stable sur les `10` pages du lot
- `1.0` partout

Conclusion:
- la traduction n’est pas actuellement le goulot principal
- ne pas perturber cette couche en corrigeant `P1/P5/P6`

---

## Matrice synthétique

| Famille | Exemple | Première porte cassée | Cause racine probable | Correction prioritaire |
|---|---|---:|---|---|
| Académique dense | `Advances...` p.101 | `P1` | segmentation et granularité OCR/extraction | `ocr_server.py`, segmentation fine |
| SQL / code annoté | `Practical SQL...` p.99 | `P5` | solveur de placement local des unités contraintes | `reconstructor.py`, slots locaux techniques |
| Tutoriel technique mixte | `test_docintelligence` p.155 | `P1` + `P6` | micro-unités mal extraites + fond sale | extraction + fond maître |

---

## Ordre global recommandé

### Axe A: Extraction et fond

À appliquer d’abord sur:
- pages académiques denses
- pages techniques mixtes

Objectif:
- faire remonter `P1`
- faire remonter `P6`

Fichiers cibles:
- [ocr_server.py](/home/raphael/Mes_Projets/docs_parser/ocr_server.py)

### Axe B: Reconstruction locale spécialisée

À appliquer ensuite sur:
- pages SQL / code annoté
- blocs techniques à relations serrées

Objectif:
- faire remonter `P5`
- sans toucher `P1`, `P2`, `P4`

Fichier cible:
- [reconstructor.py](/home/raphael/Mes_Projets/docs_parser/reconstructor.py)

---

## Principe de pilotage

À partir de maintenant:
- on choisit une famille de page
- on choisit la première porte cassée
- on corrige seulement cette couche
- on mesure avant/après
- on refuse toute correction qui dégrade une porte déjà stable

Principe d’acceptation:
- correction monotone
- pas de perte des acquis

---

## Décision pratique

La prochaine vraie décision de développement ne doit pas être:
- “corriger la pire page”

Elle doit être:
- “choisir quelle famille on traite d’abord”

Choix possibles:
1. famille `académique dense`
2. famille `SQL / code annoté`
3. famille `technique mixte`

Recommandation:
- commencer par `académique dense`

Raison:
- c’est là que la chaîne casse le plus tôt
- tant que `P1` est à `0`, le reste du pipeline ne peut pas devenir fiable
