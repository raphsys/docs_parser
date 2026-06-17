# Verdict brut

Sur cette page précise — une **table des matières** — l’ancien `ocr_server.py` fait **mieux que la nouvelle chaîne `PAGEPRINT + PAGETRANSLATE` pour la sélection du texte naturel à traduire**.

Mais il fait aussi **beaucoup plus sale**.

Donc le verdict exact est :

```text id="69ttf3"
Ancien ocr_server.py :
meilleur instinct métier sur cette page TOC,
mais pipeline sale, doublonnant, peu contrôlable.

Nouveau PAGEPRINT + PAGETRANSLATE :
meilleure architecture,
meilleur contrat futur WYSIWYG,
mais mauvais comportement actuel sur cette page TOC.
```

Si on juge uniquement le résultat visible du test : **ocr_server.py gagne**.

Si on juge la base technique pour construire un vrai pipeline maintenable : **PAGEPRINT + PAGETRANSLATE gagne**, mais il doit récupérer plusieurs heuristiques de `ocr_server.py`.

---

# 1. Comparaison brute des deux résultats

## Ancien `ocr_server.py`

Le rapport ancien indique :

```text id="dyyaua"
unit_count: 114
translated_count: 100
preserved_count: 14
levels:
  block: 16
  line: 49
  phrase: 49
translated_with_protected_glyph_count: 23
```

Donc l’ancien pipeline traduit beaucoup.

Il traduit correctement plusieurs entrées naturelles :

```text id="t21tzo"
Convolutional neural networks
→ Réseaux neuronaux convolutionnels

Image classification using MLP
→ Classification de l'image en utilisant le MLP

Input layer
→ Couche d'entrée

CNN architecture
→ Architecture CNN

The big picture
→ La situation générale

Basic components of a CNN
→ Composants de base d'un CNN

Convolutional layers
→ Couches convolutionnelles

Fully connected layers
→ Couches entièrement connectées

Confusion matrix
→ Matrice de confusion

Precision and recall
→ Précision et rappel
```

Il ne bloque pas abusivement :

```text id="9dufd9"
(weights)
(3D images)
train/validation/test
Number of parameters
Data preprocessing
```

C’est exactement ce qui manquait à `PAGEPRINT`.

---

## Nouvelle chaîne `PAGEPRINT + PAGETRANSLATE`

Le rapport nouveau indique :

```text id="8lxft8"
PAGEPRINT units: 1857
protected_visual units: 14
protected_visual regions: 14

PAGETRANSLATE units: 34
translated: 34
levels:
  block: 1
  phrase: 28
  semantic_phrase: 5
protected sent to translation: 0
```

Le chiffre `protected sent to translation: 0` paraît bon, mais il est trompeur.

Le vrai problème est inverse :

```text id="6h2zud"
PAGEPRINT a trop protégé.
```

Il a marqué comme `protected_visual/formula/code` des morceaux qui sont en réalité du texte naturel.

Exemples :

```text id="8k1onq"
■Hidden layers
■Output layer
(weights)
(3D images)
Splitting your data for train/validation/test
Data preprocessing
```

Résultat : `PAGETRANSLATE` ne reçoit pas certaines bonnes unités, ou les reçoit dans un état structurel faussé.

---

# 2. Pourquoi `ocr_server.py` fait mieux ici

L’ancien `ocr_server.py` contient déjà des heuristiques spécialisées que la nouvelle modularisation n’a pas encore récupérées correctement.

## 2.1 Il reconnaît mieux la table des matières native

Dans `ocr_server.py`, il existe une extraction native des lignes de table des matières : elle calcule un score TOC avec `contents/table of contents`, numéros de section, numéros de page et bullets, puis injecte des `toc_rows` et force `page_role = toc`, `layout_type = toc`, `page_family = toc_page` quand les lignes TOC sont détectées.  

Il va même plus loin : il attribue des rôles comme :

```text id="ulsozf"
toc_title
chapter_heading
section_heading
subentry
toc_entry
```

et sépare :

```text id="z2n6f5"
label_bbox
page_bbox
marker_bboxes
```

C’est exactement ce qu’il faut pour une table des matières. 

La nouvelle chaîne détecte bien :

```text id="84m2q9"
page_role: toc
page_family: toc
layout_type: toc_page
```

mais elle ne transforme pas encore cette compréhension en rôles fins d’unités. Dans le nouveau résultat :

```text id="12n0zy"
role_counts_top:
unknown: 34
```

C’est le cœur du problème.

---

## 2.2 Il évite mieux les faux `formula`

L’ancien code contient déjà une règle importante : une parenthèse purement lexicale comme `(weights)` n’est pas une formule, et un fragment naturel avec plusieurs mots ne doit pas être classé équation s’il ne contient pas de vrais signes mathématiques. 

Il précise aussi que les numéros simples ne doivent pas être figés dans le fond visuel, alors que les vrais bullets/symboles peuvent être préservés. 

C’est exactement la correction qu’il faut reporter dans `PAGEPRINT`.

Sur ce test, `PAGEPRINT` a régressé : il classe trop vite des morceaux ordinaires en `protected_visual`.

---

## 2.3 Il traduit plus de texte naturel utile

L’ancien pipeline traduit beaucoup d’entrées que le nouveau protège, ignore ou fusionne mal.

Exemple :

```text id="bdifq3"
ancien :
Putting it all together → Tout mettre en place
images → preserved

nouveau :
Putting it all together images
```

Ici, le nouveau fabrique une unité sémantique fausse. L’ancien sépare correctement les deux lignes.

Autre exemple :

```text id="6p0qqj"
ancien :
A closer look at feature extraction
A closer look at classification

nouveau :
■A closer look at feature extraction A closer look at classification
```

L’ancien est meilleur sur cette partie : il ne fusionne pas deux entrées TOC indépendantes.

---

# 3. Là où l’ancien `ocr_server.py` est mauvais

Il ne faut pas idéaliser l’ancien pipeline. Il gagne sur certains instincts, mais il est très sale.

## 3.1 Il traduit trop de niveaux à la fois

L’ancien pipeline traduit souvent :

```text id="3ofr8x"
block
line
phrase
```

pour le même contenu.

Exemple :

```text id="3hcpn5"
Convolutional neural networks 92    → block
Convolutional neural networks       → line
Convolutional neural networks       → phrase
```

Donc il crée naturellement des doublons.

Le nouveau `PAGETRANSLATE`, lui, sélectionne seulement 34 unités et évite beaucoup mieux le doublon `block + line + phrase`.

Sur ce critère, le nouveau est meilleur.

---

## 3.2 Il envoie des numéros au traducteur

L’ancien traduit parfois :

```text id="lpe05c"
3.1 → 3.1.
3.2 → 3.2.
4.1 → 4.1.
4.4 → 4,4
```

C’est mauvais.

Ces éléments devraient être :

```text id="y1ei4e"
toc_section_number
exact_preserve
```

Le nouveau pipeline est meilleur en intention, car il cherche à protéger les tokens, mais comme les rôles TOC fins manquent encore, il ne résout pas complètement le problème.

---

## 3.3 Il garde les bullets dans le texte envoyé au traducteur

L’ancien traduit :

```text id="y3si4t"
■Hidden layers
→ ■Couches de protection
```

La traduction est mauvaise : `Hidden layers` devient `Couches de protection`.

Mais structurellement, il aurait fallu envoyer seulement :

```text id="5xvu29"
Hidden layers
```

et préserver séparément :

```text id="ud87rx"
■
```

Donc l’ancien ne sépare pas assez :

```text id="9hqo24"
marker / label / page_reference
```

Il traduit une chaîne mélangée.

---

## 3.4 Sa traduction réelle est parfois mauvaise

Quelques exemples :

```text id="g24zzx"
Hidden layers
→ Couches de protection
```

Mauvais. Il faut :

```text id="sfxxl3"
couches cachées
```

Autre exemple :

```text id="ckfq5p"
Pooling layers
→ Couches de polissage
```

Mauvais. Il faut :

```text id="p9ijr7"
couches de pooling
ou couches de sous-échantillonnage
```

Autre exemple :

```text id="dj93nm"
Where does the dropout layer go in the CNN architecture?
```

L’ancien le coupe en deux lignes :

```text id="p9g91o"
■Where does the dropout
layer go in the CNN architecture?
```

et produit :

```text id="mnjpl4"
■Où est l'abandon scolaire?
La couche va dans l'architecture CNN?
```

C’est très mauvais.

Sur ce point, le nouveau est meilleur, parce qu’il fusionne cette question complète :

```text id="1melau"
■Where does the dropout layer go in the CNN architecture?
```

Même si le test utilise un faux traducteur, la sélection de cette unité est meilleure.

---

# 4. Là où le nouveau pipeline est meilleur

## 4.1 Meilleure architecture

La nouvelle chaîne est plus saine :

```text id="wqx4gv"
PAGEPRINT
→ INPUT_DATA
→ PAGETRANSLATE
→ reconstruction_units
→ QA
```

L’ancien `ocr_server.py` mélange :

```text id="wrh6ap"
extraction
classification
segmentation
traduction
rendu
debug
politique
post-traitement
```

Donc pour maintenir, corriger et tester, le nouveau pipeline est supérieur.

---

## 4.2 Moins de doublons

Le nouveau sélectionne :

```text id="dtqttr"
34 unités
```

contre :

```text id="n1gd69"
114 unités
```

dans l’ancien.

C’est mieux pour une future reconstruction WYSIWYG.

L’ancien a beaucoup de doublons :

```text id="8ieteg"
line + phrase
block + line + phrase
```

Le nouveau évite mieux cette explosion.

---

## 4.3 Meilleure notion de contrat WYSIWYG

Le nouveau résultat contient :

```text id="4ltbwa"
translation_profile
translation_units
projection
translated_input_data
quality
views.reconstruction_units
wysiwyg_constraints
translation_forecast
render_contract
```

L’ancien ne produit pas un contrat propre de reconstruction. Il traduit, mais il ne donne pas une vraie structure propre à consommer par le reconstructeur.

Donc l’ancien est meilleur pour “voir une traduction apparaître”, mais le nouveau est meilleur pour construire un système robuste.

---

# 5. Là où le nouveau pipeline est actuellement moins bon

## 5.1 Faux protected_visual

C’est le plus grave.

Le nouveau produit :

```text id="rbgkae"
14 protected_visual regions
14 protected_visual units
176 éléments dans views.protected_visual_units
```

alors que cette page TOC n’a presque pas de vraie zone visuelle à protéger.

Il faut réduire cela drastiquement.

Sur cette page, les `protected_visual` devraient être :

```text id="0eu5vl"
0 à 2 au maximum,
et seulement pour des symboles/décors très précis.
```

Pas pour des fragments textuels.

---

## 5.2 Rôles TOC absents

La page est reconnue comme `toc`, mais les unités sélectionnées restent :

```text id="0muod0"
role: unknown
```

C’est incohérent.

Pour une TOC, il faut produire :

```text id="gc9lro"
toc_title
toc_chapter_number
toc_chapter_title
toc_section_number
toc_entry_title
toc_page_reference
toc_bullet_marker
toc_subentry_title
```

Sans ça, `PAGETRANSLATE` devine à l’aveugle.

---

## 5.3 Mauvaise fusion de certaines lignes

Le nouveau crée :

```text id="vc6yqa"
Putting it all together images
```

C’est faux.

Il crée aussi :

```text id="7xjd7w"
■A closer look at feature extraction A closer look at classification
```

C’est faux aussi.

Sur une page TOC, il ne faut pas utiliser la fusion normale de paragraphes. Il faut construire des `toc_entry_units`.

---

# 6. Comparaison synthétique

| Critère                        |          Ancien `ocr_server.py` |  Nouveau `PAGEPRINT + PAGETRANSLATE` | Meilleur          |
| ------------------------------ | ------------------------------: | -----------------------------------: | ----------------- |
| Détection TOC globale          |                           bonne |                                bonne | égal              |
| Rôles TOC fins                 |    présents dans le code ancien |    absents dans les unités actuelles | ancien            |
| Faux `protected_visual`        |              faible sur ce test |                                élevé | ancien            |
| Couverture du texte naturel    |                       meilleure |                  pertes / exclusions | ancien            |
| Fusion des lignes TOC          |           globalement meilleure |            plusieurs fusions fausses | ancien            |
| Doublons de traduction         |                    très mauvais |                        bien meilleur | nouveau           |
| Sélection unique par unité     |                         mauvais |                             meilleur | nouveau           |
| Protection des numéros         |                        instable | meilleure intention, mais incomplète | nouveau potentiel |
| Qualité traduction réelle      |       inégale, parfois mauvaise |        non testée, mock `FR_AUDIT::` | indécidable       |
| Contrat reconstruction WYSIWYG |                          faible |                   nettement meilleur | nouveau           |
| Maintenabilité                 |                        mauvaise |                                bonne | nouveau           |
| Potentiel production           | faible si conservé monolithique |           meilleur après corrections | nouveau           |

---

# 7. Lequel est meilleur ?

## Pour ce test précis

```text id="ewiswy"
ocr_server.py est meilleur.
```

Il fait moins d’erreurs de protection et couvre mieux les entrées naturelles de la table des matières.

Mais il ne faut pas revenir au monolithe.

## Pour le projet global

```text id="3v426o"
PAGEPRINT + PAGETRANSLATE est meilleur comme architecture.
```

Mais il doit récupérer les bonnes heuristiques métier de `ocr_server.py`.

Le bon choix n’est donc pas :

```text id="l7x6ub"
ancien OU nouveau
```

Le bon choix est :

```text id="w0r9ab"
nouvelle architecture
+
heuristiques TOC/protection de l’ancien ocr_server.py
```

---

# 8. Corrections à faire maintenant

## P0 — Porter la logique TOC de `ocr_server.py` dans PAGEPRINT

Créer dans `pageprint/` :

```text id="051nqg"
toc_extractor.py
toc_role_resolver.py
```

Avec sortie :

```json id="xabmiz"
{
  "role": "toc_entry",
  "section_number": "3.1",
  "title_text": "Image classification using MLP",
  "page_reference": "93",
  "marker": null,
  "translatable_text": "Image classification using MLP",
  "preserve": ["3.1", "93"]
}
```

Puis `PAGETRANSLATE` ne traduit que :

```text id="c127c8"
title_text
```

Jamais :

```text id="l1fb1d"
section_number
page_reference
marker
```

---

## P0 — Porter les règles anti-faux-formula de `ocr_server.py`

Dans `pageprint/policy_compiler.py` ou équivalent, reprendre l’idée :

```text id="lhz11q"
(weights) = texte éditorial, pas formule
(3D images) = texte éditorial, pas formule
train/validation/test = texte technique naturel, pas code
phrase avec plusieurs mots sans symbole mathématique = pas formule
```

L’ancien code contient déjà cette logique. Il faut l’intégrer dans la nouvelle brique.

---

## P0 — Séparer bullet et texte

Au lieu de :

```text id="2l74s6"
■Hidden layers
```

produire :

```json id="ck4ct5"
{
  "marker": "■",
  "marker_policy": "exact_preserve",
  "text": "Hidden layers",
  "text_policy": "translate"
}
```

Même chose pour :

```text id="l78qdi"
■Precision and recall
■Pooling layers or subsampling
■Plotting the learning curves
```

---

## P0 — Désactiver la fusion paragraphique normale sur TOC

Règle :

```python id="wavcy3"
if page_role == "toc":
    do_not_use_generic_sentence_coalescer()
    use_toc_entry_builder()
```

Sinon tu auras toujours des erreurs du type :

```text id="wep699"
Putting it all together images
```

---

## P1 — Ne plus traduire `block` dans TOC

La première unité nouvelle est encore :

```text id="hj6xnk"
CONTENTS vii 3
```

C’est un mélange.

Il faut découper :

```text id="9xk6wg"
CONTENTS → toc_title → traduisible
vii      → page marker → preserve
3        → chapter decorative number → preserve
```

---

# 9. Conclusion finale

Le résultat de comparaison est très instructif.

L’ancien `ocr_server.py` n’est pas “meilleur” au sens architectural. Il est trop monolithique, il traduit trop d’unités, il duplique, il envoie des numéros au traducteur, et sa traduction technique peut être mauvaise.

Mais sur cette page TOC, il avait déjà des heuristiques importantes que la nouvelle chaîne n’a pas encore réintégrées correctement.

Donc mon verdict final :

```text id="vctqty"
Ancien ocr_server.py :
meilleur comportement empirique sur cette page.

Nouveau PAGEPRINT + PAGETRANSLATE :
meilleure fondation système, mais comportement actuel inférieur sur TOC.

Décision :
ne pas revenir à ocr_server.py ;
extraire ses bonnes heuristiques TOC/protection
et les porter proprement dans PAGEPRINT/PAGETRANSLATE.
```

La priorité immédiate est donc claire :

```text id="4xyc7f"
faire une brique TOC dédiée,
réduire les faux protected_visual,
séparer marker / titre / numéro de page,
et interdire la fusion générique sur les tables des matières.
```

