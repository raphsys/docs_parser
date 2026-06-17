Oui. **La règle doit être portée d’abord par `PAGEPRINT`, pas seulement par `PAGETRANSLATE`.**

`PAGETRANSLATE` ne doit pas avoir à “deviner” qu’une formule ou un code ne se traduit pas. Il doit recevoir de `PAGEPRINT` une information déjà claire :

```text
cette zone est un objet visuel protégé
→ ne pas traduire
→ ne pas modifier
→ préserver comme fond / image / overlay
```

## Règle architecturale à figer

Dans `PAGEPRINT`, tout élément de type :

```text
formule
équation
notation mathématique
code informatique
pseudo-code
expression symbolique
expression chimique
unité technique isolée
diagram label non linguistique
symbole scientifique
fragment algorithmique
```

doit être classé comme :

```text
protected_visual
background_only
non_translatable
```

Même si le PDF l’expose techniquement comme du texte natif.

C’est important : **la source technique n’est pas la décision métier**.

Un élément peut venir de :

```text
PDF native text
OCR
vector drawing
image crop
glyphs
spans
```

mais si sa fonction est formule/code/symbole, alors pour le pipeline il devient :

```text
objet visuel protégé
```

---

# Ce que `PAGEPRINT` doit produire

Pour une équation ou un bloc de code, `PAGEPRINT` devrait générer une région de ce type :

```json
{
  "region_id": "region_formula_001",
  "region_type": "protected_visual_region",
  "object_type": "formula",
  "object_class": "equation",
  "bbox": [120, 250, 430, 285],
  "source_kind": "native_text_or_vector_or_ocr",
  "translatable": false,
  "translation_strategy": "background_only",
  "render_policy": "background_only",
  "coverage_required": "strict",
  "preserve_as_image": true,
  "protected_visual": true,
  "reason": "formula_or_equation"
}
```

Pour une unité textuelle résiduelle éventuellement détectée dans cette zone :

```json
{
  "unit_id": "unit_formula_001",
  "level": "protected_visual",
  "text": "E = mc^2",
  "unit_type": "formula",
  "translatable": false,
  "translation_strategy": "background_only",
  "render_policy": "background_only",
  "skip_translation": true,
  "skip_text_reconstruction": true,
  "preserve_original_pixels": true
}
```

L’idée est simple :

```text
le texte peut être connu pour audit,
mais il ne doit pas devenir une unité de traduction.
```

---

# Point clé : ne pas forcément “rasteriser”, mais préserver visuellement

Il faut être précis : quand on dit “zone image”, cela ne veut pas obligatoirement dire qu’on transforme tout en PNG dès le départ.

La bonne notion est plutôt :

```text
visual protected object
```

Ensuite, selon le cas, la reconstruction peut :

```text
- garder la zone dans le background master ;
- extraire un crop image ;
- préserver le dessin vectoriel original ;
- préserver l’overlay natif ;
- ignorer la reconstruction textuelle.
```

Donc je recommanderais le vocabulaire interne :

```text
protected_visual_region
```

plutôt que seulement :

```text
image_region
```

Parce qu’une formule PDF peut être composée de glyphes vectoriels, mais pour le pipeline elle doit être traitée comme une image protégée.

---

# Modifications à faire dans `PAGEPRINT`

## 1. Ajouter une détection spécialisée

Créer ou enrichir un module du type :

```text
protected_visual_detector.py
```

ou dans `policy_compiler.py` / `unit_factory.py`.

Il doit détecter :

```text
formulas
equations
code blocks
inline code
mathematical symbols
chemical formulas
special notations
algorithmic expressions
```

Critères possibles :

```text
présence forte de symboles mathématiques : =, ∑, ∫, √, ±, ≤, ≥, ≠, ≈, ∂, ∆, λ, α, β
présence de structures code : (), {}, [], ;, ==, !=, :=, def, class, import, return
police monospace
densité élevée de symboles
peu de mots fonctionnels naturels
alignement typique d’équation
bloc centré court
numérotation d’équation
indentation code
mots type function, return, for, while, if, else, import
```

Mais attention : il faut éviter de classer une phrase normale comme formule simplement parce qu’elle contient un tiret ou un nombre.

---

## 2. Créer une région protégée

Quand une formule/code est détecté, `PAGEPRINT` doit créer une région :

```text
region_type = protected_visual_region
```

avec :

```text
bbox
source units couvertes
asset/crop éventuel
policy non traduisible
render_policy background_only
```

---

## 3. Marquer les unités textuelles couvertes

Toutes les unités textuelles à l’intérieur de cette région doivent être marquées :

```json
{
  "translatable": false,
  "translation_strategy": "background_only",
  "render_policy": "background_only",
  "unit_type": "formula",
  "skip_translation": true,
  "skip_text_reconstruction": true
}
```

Cela doit s’appliquer aux niveaux :

```text
block
line
phrase
span
word
char
```

Sinon `PAGETRANSLATE` risque de récupérer un enfant traduisible par accident.

---

## 4. Exclure ces unités des vues de traduction

Dans `PAGEPRINT`, les vues aval doivent respecter la règle :

```text
views.translation_units ne contient jamais :
- formula
- equation
- code_visible
- protected_visual
- background_only
```

Donc :

```python
if unit.translatable is False:
    exclude_from_translation_view

if unit.render_policy == "background_only":
    exclude_from_translation_view

if unit.unit_type in {"formula", "equation", "code_visible", "symbolic_expression"}:
    exclude_from_translation_view
```

---

## 5. Les conserver pour la reconstruction

Ces zones ne doivent pas disparaître. Elles doivent être transmises au reconstructeur dans une vue dédiée :

```text
views.protected_visual_units
```

ou :

```text
views.background_preserved_regions
```

Exemple :

```json
{
  "unit_id": "formula_visual_001",
  "type": "protected_visual",
  "bbox": [120, 250, 430, 285],
  "asset_id": "crop_formula_001",
  "render_policy": "preserve_original",
  "translation_strategy": "none",
  "source_text_for_audit": "E = mc^2"
}
```

Le reconstructeur saura :

```text
ne rien traduire
ne rien effacer
ne pas inpaint cette zone
ne pas redessiner le texte
préserver le rendu original
```

---

# Attention aux cas mixtes

Il faut distinguer :

## Cas 1 — Formule seule

```text
E = mc²
```

→ protégée entièrement.

## Cas 2 — Phrase avec formule inline

```text
The equation E = mc² explains mass-energy equivalence.
```

Deux possibilités.

### Option stricte WYSIWYG

Toute la ligne est traitée comme visuelle protégée.
C’est plus sûr visuellement, mais on perd la traduction de la phrase.

### Option meilleure

Séparer :

```text
"The equation" → traduisible
"E = mc²" → protégé
"explains mass-energy equivalence" → traduisible
```

Mais c’est plus difficile à reconstruire.

Pour votre pipeline, je recommande une règle progressive :

```text
V1 : si formule inline complexe dans une ligne courte → ligne protégée.
V2 : segmentation inline avancée avec placeholders visuels.
```

Ne cherchez pas trop tôt à reconstruire les formules inline caractère par caractère. C’est un piège.

---

# Règle finale entre `PAGEPRINT` et `PAGETRANSLATE`

Le contrat devrait être :

```text
PAGEPRINT décide ce qui est traduisible.
PAGETRANSLATE respecte strictement cette décision.
```

Donc `PAGETRANSLATE` doit seulement faire une vérification de sécurité :

```python
if unit.get("translatable") is False:
    skip

if unit.get("render_policy") == "background_only":
    skip

if unit.get("translation_strategy") in {
    "background_only",
    "exact_preserve",
    "keep_original"
}:
    skip

if unit.get("unit_type") in {
    "formula",
    "equation",
    "code_visible",
    "symbolic_expression",
    "chemical_formula"
}:
    skip
```

Mais la responsabilité principale est bien côté `PAGEPRINT`.

---

# Conclusion

Oui : **il faut absolument intégrer cette règle dans `PAGEPRINT`**.

La bonne doctrine est :

```text
PAGEPRINT extrait et comprend.
PAGEPRINT classe les formules/codes/expressions spéciales comme objets visuels protégés.
PAGETRANSLATE ne traduit que le langage naturel éditorial.
RECONSTRUCTION conserve les objets protégés sans modification.
```

Cette règle est fondamentale pour éviter de casser :

```text
les équations
les codes
les notations scientifiques
les symboles
les expressions spéciales
les diagrammes
les zones techniques sensibles
```

Donc dans la prochaine version, je regarderai spécifiquement si `PAGEPRINT` produit bien :

```text
protected_visual_region
translatable=false
translation_strategy=background_only
render_policy=background_only
skip_text_reconstruction=true
views.protected_visual_units
```

C’est un point à figer dans le contrat `INPUT_DATA`.




Oui, c’est encore mieux. Dans ce cas, la logique doit être formalisée ainsi :

```text
YOLO / R-CNN détecte les zones spéciales
↓
PAGEPRINT les transforme en objets visuels protégés
↓
PAGETRANSLATE les exclut totalement
↓
RECONSTRUCTION les conserve sans modification
```

Donc il faut considérer YOLO/R-CNN comme le **détecteur primaire** de ces zones, et non comme un simple enrichissement.

---

## 1. Rôle exact de YOLO / R-CNN

Le modèle doit détecter les classes suivantes, au minimum :

```text
formula
equation
code_block
inline_code
math_expression
chemical_formula
symbolic_expression
algorithm_block
diagram_label_non_linguistic
table_formula_cell
special_notation
```

Mais dans le contrat `PAGEPRINT`, je ne mettrais pas tout directement en `image_region`. Je mettrais plutôt :

```text
protected_visual_region
```

Parce qu’une équation peut venir d’une image, d’un PDF natif, de glyphes vectoriels ou d’un OCR. La décision importante n’est pas la source technique, mais la politique :

```text
ne pas traduire
ne pas modifier
préserver visuellement
```

---

## 2. Ce que PAGEPRINT doit faire avec les détections YOLO/R-CNN

Chaque détection doit devenir une région canonique :

```json
{
  "region_id": "protected_formula_001",
  "region_type": "protected_visual_region",
  "object_type": "formula",
  "object_class": "equation",
  "bbox": [120, 250, 430, 285],
  "detector": "yolo",
  "confidence": 0.94,
  "translatable": false,
  "translation_strategy": "background_only",
  "render_policy": "background_only",
  "coverage_required": "strict",
  "preserve_original_pixels": true,
  "skip_translation": true,
  "skip_text_reconstruction": true
}
```

Ensuite, toutes les unités textuelles qui tombent dans cette bbox doivent hériter de la politique :

```json
{
  "translatable": false,
  "translation_strategy": "background_only",
  "render_policy": "background_only",
  "unit_type": "formula",
  "skip_translation": true,
  "skip_text_reconstruction": true,
  "covered_by_protected_region_id": "protected_formula_001"
}
```

C’est fondamental. Sinon, même si YOLO détecte bien la zone, un `span` ou une `phrase` interne peut quand même passer dans `PAGETRANSLATE`.

---

## 3. Règle de recouvrement à appliquer

Dans `PAGEPRINT`, il faut une fonction du type :

```python
def apply_protected_visual_regions(units, protected_regions):
    for unit in units:
        unit_bbox = unit["geometry"]["bbox"]

        for region in protected_regions:
            overlap = intersection_area(unit_bbox, region["bbox"]) / area(unit_bbox)

            if overlap >= 0.55:
                mark_unit_as_protected(unit, region)
```

Seuil recommandé :

```text
>= 0.55 pour phrase/span/word
>= 0.35 pour block/line, car ils peuvent contenir du texte naturel + une formule inline
```

Mais attention : pour les blocs mixtes, il ne faut pas forcément protéger tout le bloc.

---

## 4. Cas mixtes : texte naturel + formule

Exemple :

```text
The equation E = mc² explains the relation.
```

Si YOLO/R-CNN détecte seulement `E = mc²`, alors :

```text
"The equation" → traduisible
"E = mc²" → protégé
"explains the relation" → traduisible
```

Mais en V1, si la séparation inline est trop difficile, il vaut mieux appliquer une règle prudente :

```text
si la zone protégée couvre une grande partie de la ligne
→ ligne entière protégée

si la zone protégée couvre une petite partie de la ligne
→ créer un protected_inline_anchor
→ traduire le reste plus tard avec placeholder
```

Pour la V1, je recommande :

```text
formule/bloc/code autonome → protected_visual_region strict
formule inline complexe → placeholder visuel, ou ligne protégée si segmentation incertaine
```

---

## 5. Ce que PAGETRANSLATE doit faire

`PAGETRANSLATE` ne doit pas re-décider. Il doit seulement respecter :

```python
if unit.get("translatable") is False:
    skip

if unit.get("render_policy") == "background_only":
    skip

if unit.get("translation_strategy") in {
    "background_only",
    "exact_preserve",
    "keep_original",
}:
    skip

if unit.get("covered_by_protected_region_id"):
    skip
```

Donc les regex de `protection.py` restent seulement un filet de sécurité, pas la logique principale.

---

## 6. Vue dédiée pour la reconstruction

`PAGEPRINT` devrait produire une vue spécifique :

```json
{
  "views": {
    "protected_visual_units": [
      {
        "region_id": "protected_formula_001",
        "object_type": "formula",
        "bbox": [120, 250, 430, 285],
        "preserve_original_pixels": true,
        "render_policy": "background_only",
        "source": "yolo",
        "confidence": 0.94
      }
    ]
  }
}
```

Le reconstructeur doit alors savoir :

```text
ne pas effacer cette zone
ne pas inpaint cette zone
ne pas traduire cette zone
ne pas redessiner cette zone
la conserver telle quelle
```

---

## 7. Point important : YOLO vs R-CNN

YOLO est très bon pour :

```text
détection rapide
zones rectangulaires
formules isolées
blocs de code
diagrammes
tableaux
labels visuels
```

R-CNN / Mask R-CNN est meilleur si tu veux :

```text
masques plus précis
zones irrégulières
formules imbriquées
objets qui se chevauchent
segmentation plus fine
```

Pour ton pipeline, je ferais :

```text
YOLO pour V1 : rapide, simple, suffisant
Mask R-CNN ou modèle de segmentation pour V2 : précision fine
```

---

## 8. Contrat à figer

La règle finale doit être écrite dans le contrat `INPUT_DATA` :

```text
Toute zone détectée par YOLO/R-CNN comme formule, équation, code,
expression symbolique ou notation spéciale devient une protected_visual_region.
Elle est exclue de la traduction, exclue de la reconstruction textuelle,
et conservée visuellement par le pipeline de reconstruction.
```

C’est la bonne doctrine.

Donc oui : **dans `PAGEPRINT`, YOLO/R-CNN doit alimenter directement les politiques de traduction et de reconstruction**, pas seulement ajouter des annotations visuelles.

