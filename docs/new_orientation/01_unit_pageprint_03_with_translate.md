Oui. Il faut distinguer trois niveaux :

```text id="67frft"
1. Analyse pure de pageprint/ tel qu’il est.
2. Analyse pure de pagetranslate/ tel qu’il est.
3. Corrections nécessaires au regard du rôle réel de special_region_detector.py / YOLO-ONNX.
```

J’ai pris `page.zip` comme version de référence, car elle contient les versions complètes actuelles de `pageprint/` et `pagetranslate/`.

---

# 1. Analyse pure de `PAGEPRINT`

## Verdict sur `pageprint/`

`PAGEPRINT` est maintenant **bien avancé comme unité INPUT_DATA**.

Il fait déjà correctement plusieurs choses importantes :

```text id="66ga8p"
- création d’unités canoniques page/block/line/phrase/span/word/char ;
- création d’unités region/image/drawing/protected_visual ;
- normalisation bbox en pt ;
- indexation des régions ;
- association unit → region ;
- compilation des politiques ;
- exclusion des protected_visual de la traduction ;
- production de views.translation_units ;
- production de views.protected_visual_units ;
- transmission style_system, semantic_system, translation_context.
```

Donc, structurellement, `PAGEPRINT` joue bien son rôle de **première tête INPUT_DATA**.

Mais il reste des points à corriger.

---

## 1.1 Point fort : `PAGEPRINT` sait consommer des régions spéciales

Dans `pageprint/region_index.py`, les types suivants sont bien normalisés comme :

```text id="s3e806"
protected_visual_region
```

notamment :

```text id="ydg60v"
formula
formula_region
equation
math_expression
chemical_formula
symbolic_expression
code
code_region
code_block
inline_code
algorithm_block
special_notation
table_formula_cell
diagram_label_non_linguistic
protected_visual
```

Et une `protected_visual_region` reçoit bien la politique :

```json id="rf1z4w"
{
  "translatable": false,
  "translation_strategy": "background_only",
  "render_policy": "background_only",
  "preserve_original_pixels": true,
  "protected_visual": true,
  "skip_translation": true,
  "skip_text_reconstruction": true
}
```

C’est exactement ce qu’il faut.

---

## 1.2 Point fort : les unités couvertes par région protégée héritent de la politique

`PAGEPRINT` calcule les memberships :

```text id="asohyt"
unit → protected_visual_region
```

avec des seuils adaptés :

```text id="dtugev"
block/line : overlap >= 0.35
phrase/span/word/char : overlap >= 0.55
```

Ensuite `policy_compiler.py` impose :

```text id="9ntoz0"
translatable = False
translation_strategy = background_only
render_policy = background_only
skip_translation = True
skip_text_reconstruction = True
```

Donc si la région spéciale est correctement typée, `PAGEPRINT` protège bien le texte interne.

---

## 1.3 Point fort : `PAGEPRINT` produit une vue dédiée aux objets protégés

La vue :

```text id="6v2h7s"
views.protected_visual_units
```

existe bien.

Elle contient :

```text id="kv4bms"
unit_id
level
bbox
source_text_for_audit
unit_type
render_policy
translation_strategy
preserve_original_pixels
skip_translation
skip_text_reconstruction
covered_by_protected_region_id
```

C’est bon pour la reconstruction.

---

## 1.4 Problème : `PAGEPRINT` n’appelle pas le détecteur spécial

C’est le problème central.

Dans `pageprint/`, il n’y a pas d’appel à :

```python id="s829ul"
detect_special_regions(...)
```

Donc `PAGEPRINT` ne fait pas lui-même la détection YOLO/ONNX.

Il fait seulement ceci :

```text id="1lqykb"
si page_structure["special_regions"] existe déjà
→ alors PAGEPRINT les consomme
```

Donc `PAGEPRINT` est un **consommateur de régions spéciales**, pas encore un **orchestrateur de détection spéciale**.

Ce n’est pas forcément mauvais, mais il faut le figer clairement.

---

## 1.5 Problème critique : `special_class` n’est pas lu comme type de région

Le détecteur `special_region_detector.py` produit actuellement des sorties du type :

```json id="b34t26"
{
  "id": "special_region_0",
  "special_class": "formula",
  "bbox": [...]
}
```

Mais `pageprint/region_index.py` lit :

```python id="3zux2m"
raw.get("region_type") or raw.get("type") or raw.get("kind")
```

Il ne lit pas :

```python id="czr3sm"
raw.get("special_class")
```

Donc une région spéciale issue du détecteur peut être mal classée comme :

```text id="a04hzt"
body_region
```

au lieu de :

```text id="gv8gjo"
protected_visual_region
```

C’est un vrai défaut d’intégration.

Dans certains cas, `policy_compiler.py` rattrape l’erreur par heuristique textuelle. Par exemple `E = mc^2` est détecté comme formule même si la région est mal typée. Mais il ne faut pas dépendre de cette chance. Si la bbox contient du code ou une zone spéciale mal OCRisée, la protection peut échouer.

### Correction obligatoire

Dans `region_index.py`, remplacer :

```python id="eosho3"
raw.get("region_type") or raw.get("type") or raw.get("kind")
```

par :

```python id="3gt0av"
raw.get("region_type")
or raw.get("type")
or raw.get("kind")
or raw.get("special_class")
```

Et côté `special_region_detector.py`, ajouter aussi :

```python id="90q3oi"
"region_type": special_class,
"object_type": special_class,
"object_class": special_class,
"protected_visual": True,
"preserve_original_pixels": True,
"skip_translation": True,
"skip_text_reconstruction": True
```

Il faut faire les deux.

---

## 1.6 Problème : `views.translation_units` contient trop de niveaux

Dans mon test simple, `PAGEPRINT` produit :

```text id="72j79j"
block
line
phrase
span
```

dans `views.translation_units`.

`PAGETRANSLATE` choisit ensuite la `phrase`, donc il ne traduit pas tout. Mais conceptuellement, la vue `translation_units` de `PAGEPRINT` est trop large.

Elle devrait déjà être plus stricte :

```text id="mxjei3"
phrase
line
block
semantic_phrase si matérialisée
semantic_group si matérialisé
```

Mais pas :

```text id="mqn04w"
span
word
char
page
region
```

Le `span` doit rester disponible pour :

```text id="ou7y6f"
style
alignement
audit
projection
reconstruction fine
```

mais pas comme candidat de traduction principal.

### Correction recommandée

Dans `PAGEPRINT`, la vue :

```text id="5o1imn"
views.translation_units
```

devrait exclure `span`.

Actuellement `PAGETRANSLATE` compense, mais `PAGEPRINT` devrait produire une vue plus propre.

---

## 1.7 Problème important : `semantic_phrases` sont copiées, pas réellement canonisées

`PAGEPRINT` fait maintenant :

```python id="r96c6f"
semantic_system = {
    "semantic_phrases": [
        phrase for block in page_structure["blocks"]
        for phrase in block.get("semantic_phrases") or []
    ],
    "semantic_groups": [...]
}
```

C’est bien mieux qu’avant.

Mais attention : `PAGEPRINT` ne convertit pas automatiquement les `source_unit_ids` legacy vers les `unit_id` canoniques générés par `PAGEPRINT`.

Or `PAGETRANSLATE` a besoin de IDs canoniques du type :

```text id="6zaxck"
p001_block_001_line_001_phrase_001
```

Si `semantic_phrase.source_unit_ids` contient encore des IDs amont du type :

```text id="yeg0k7"
legacy_phrase_1
ocr_phrase_17
block_0_line_1_phrase_0
```

alors `PAGETRANSLATE` ne pourra pas projeter proprement.

J’ai testé ce cas : une `semantic_phrase` avec des `source_unit_ids` non canoniques est sélectionnée, mais comme son `block_id` n’est pas résolu, `PAGETRANSLATE` peut aussi sélectionner une unité visuelle du même bloc. Résultat :

```text id="ehg93u"
semantic_phrase traduite
+
synthetic_semantic_phrase traduite
=
doublon
```

Donc il faut absolument canoniser les liens.

### Correction obligatoire

Dans `PAGEPRINT`, après création des unités, il faut résoudre les `semantic_phrases` :

```text id="fp3ra7"
semantic_phrase.source_unit_ids legacy
→ canonical source_unit_ids PagePrint
```

Ou au minimum ajouter :

```json id="49djo3"
"structural_context": {
  "block_unit_id": "p001_block_001"
}
```

Mais le mieux est :

```json id="2xg5qx"
{
  "unit_id": "sem_p001_001",
  "text": "...",
  "source_unit_ids": [
    "p001_block_001_line_001_phrase_001",
    "p001_block_001_line_002_phrase_001"
  ],
  "structural_context": {
    "block_unit_id": "p001_block_001"
  }
}
```

---

# 2. Analyse pure de `PAGETRANSLATE`

## Verdict sur `pagetranslate/`

`PAGETRANSLATE` est maintenant **beaucoup plus solide** que la version précédente.

La structure est correcte :

```text id="s4c0qs"
builder.py
selector.py
sentence_boundary.py
coalescer.py
context_builder.py
protection.py
translator_bridge.py
quality.py
projection.py
schema.py
```

C’est une vraie unité métier, pas seulement un wrapper.

---

## 2.1 Point fort : sélection hiérarchique correcte

La priorité est maintenant :

```text id="44lam4"
semantic_phrase > semantic_group > phrase > line > block
```

Et le fallback est par bloc.

Le code bloque correctement :

```text id="pz90ms"
semantic_phrase + line du même bloc
semantic_phrase + phrase du même bloc
semantic_phrase + semantic_group du même bloc
```

à condition que la `semantic_phrase` ait un `block_id` ou des `source_unit_ids` résolubles.

Donc la logique est bonne, mais elle dépend d’un `PAGEPRINT` propre.

---

## 2.2 Point fort : coalescer utile

`coalescer.py` est une bonne addition.

Il permet de fusionner des unités visuelles ouvertes quand une phrase est coupée :

```text id="5wdmq8"
Line 1 : This method improves the performance
Line 2 : of the model.
```

en :

```text id="qxtpmz"
synthetic_semantic_phrase
```

C’est indispensable si `PAGEPRINT` ne fournit pas encore toutes les `semantic_phrases`.

---

## 2.3 Point fort : protection inline correcte comme filet de sécurité

`protection.py` protège :

```text id="7updmw"
URL
email
DOI
chemins
références
nombres + unités
formules simples
nombres
tokens explicites
```

C’est utile.

Mais dans ta doctrine actuelle, il faut bien rappeler :

```text id="uv2nrd"
les formules/codes doivent être bloqués par PAGEPRINT comme protected_visual ;
protection.py n’est qu’un filet de sécurité.
```

Donc il ne faut pas faire porter à `PAGETRANSLATE` la responsabilité principale des formules.

---

## 2.4 Point fort : `TranslatorBridge` transmet un vrai contexte

Le bridge transmet maintenant :

```text id="j7bv63"
source_lang
target_lang
domain
subdomain
document_type
page_role
page_family
layout_type
style
tone
terminology
context_before
context_after
section_title
protected_tokens
wysiwyg_constraints
object_class
object_type
phrase_semantics
```

C’est une bonne évolution.

---

## 2.5 Point fort : projection plus saine

La vue :

```text id="fsmuzu"
views.reconstruction_units
```

est maintenant construite depuis `translated_units`, pas depuis toutes les unités ayant `content.translated_text`.

Le bug précédent :

```text id="ox01xl"
page + block + line + phrase + span dupliqués
```

est corrigé dans le cas simple.

J’ai testé une page simple : la reconstruction ne contient qu’une unité `phrase`. C’est bon.

---

## 2.6 Problème important : `PAGETRANSLATE` reste dépendant de la qualité des IDs sémantiques

Si `semantic_phrase.source_unit_ids` ne pointe pas vers les IDs canoniques de `PAGEPRINT`, alors :

```text id="zkm5x2"
projection vers les unités sources impossible ;
skip_individual_render impossible ;
consume_source_units incomplet ;
risque de doublon avec coalescer/fallback.
```

Donc `PAGETRANSLATE` doit devenir plus défensif.

### Correction dans `selector.py`

Si une entrée sémantique a des `source_unit_ids` mais aucun ne se résout dans `units_by_id`, il faut :

```text id="i50qmd"
soit la rejeter ;
soit exiger structural_context.block_unit_id ;
soit tenter une résolution par legacy_id.
```

Règle recommandée :

```python id="3xatcp"
if source_ids and not source_units and not block_id:
    blocked_source_ids.update(source_ids)
    continue
```

Ou mieux :

```python id="6c9v5b"
source_units = resolve_source_units(source_ids, units_by_id, legacy_id_index)
```

---

## 2.7 Problème : statut `preserved` ambigu

Dans `builder.py`, si la traduction est identique :

```python id="dg8zn5"
status = "preserved"
```

Mais `preserved` peut vouloir dire deux choses très différentes :

```text id="6ak4ya"
1. texte volontairement préservé ;
2. échec de traduction car sortie identique.
```

La QA marque normalement `unchanged_problem`, mais le statut reste ambigu.

### Correction recommandée

Utiliser :

```text id="mmn6e6"
translated
dry_run
error
unchanged_suspect
preserved_by_policy
```

Ou garder `preserved`, mais ajouter :

```json id="nszufa"
"preserve_reason": "identical_output_suspected"
```

---

## 2.8 Problème : QA encore légère pour une vraie traduction

`quality.py` contrôle déjà :

```text id="0vsbow"
nombres
unités
tokens protégés
placeholders restants
fuite anglais vers français
ratio expansion
overflow WYSIWYG
unchanged
```

C’est bien.

Mais pour une traduction réelle de documents, il manque encore :

```text id="5x2lbn"
négation inversée
ponctuation terminale
parenthèses/guillemets
capitalisation des titres
terminologie obligatoire absente
glossaire interdit
préservation des noms propres
préservation des références légales
qualité langue cible
cohérence inter-unités
```

Ces contrôles peuvent venir en V2, pas forcément maintenant.

---

# 3. Analyse au regard de YOLO / `special_region_detector.py`

## 3.1 Chaîne actuelle réelle

Aujourd’hui, la chaîne est :

```text id="kz0n9b"
special_region_detector.py existe
mais n’est pas appelé par pageprint/
pageprint/ consomme seulement page_structure["special_regions"]
pagetranslate/ respecte les politiques de pageprint/
```

Donc la chaîne complète attendue n’est pas encore garantie :

```text id="3denug"
YOLO/ONNX
→ special_regions
→ PAGEPRINT protected_visual_region
→ PAGETRANSLATE skip
→ RECONSTRUCTION preserve_original_pixels
```

Elle ne marche que si l’orchestrateur appelle le détecteur avant `PAGEPRINT`.

---

## 3.2 Ce qui doit être fait côté orchestration

Il faut ajouter une étape avant `PagePrintBuilder` :

```python id="yo3qb4"
from special_region_detector import detect_special_regions

page_structure, special_region_info = detect_special_regions(
    page_structure,
    page_image=pil_image,
    pdf_page=pdf_page,
    sx=sx,
    sy=sy,
)

page_structure.setdefault("debug", {})["special_region_detector"] = special_region_info
```

Puis :

```python id="uxj1vu"
input_data = build_pageprint_input_data(
    page_structure=page_structure,
    source_context=source_context,
    extraction_result=extraction_result,
    assets=assets,
)
```

Cette étape ne doit pas être dans `PAGETRANSLATE`.

Elle peut être :

```text id="44qa9u"
- dans ocr_server.py ;
- ou dans un orchestrateur pipeline ;
- ou dans une unité avant PAGEPRINT.
```

Mais elle doit exister.

---

# 4. Ce qui doit être corrigé / amélioré / restreint

## P0 — Corrections bloquantes

### P0.1 — Corriger `special_class` → `region_type`

À faire dans `special_region_detector.py` :

```python id="jm1xan"
"region_type": special_class,
"object_type": special_class,
"object_class": special_class,
```

À faire aussi dans `pageprint/region_index.py` :

```python id="bsex3a"
raw.get("region_type")
or raw.get("type")
or raw.get("kind")
or raw.get("special_class")
```

C’est obligatoire.

---

### P0.2 — Ajouter l’appel réel à `detect_special_regions(...)`

Sans cela, on ne peut pas dire :

```text id="cdaqni"
PAGEPRINT détecte les formules/codes par YOLO.
```

La phrase correcte serait seulement :

```text id="om942h"
PAGEPRINT peut consommer des régions spéciales déjà détectées.
```

Donc il faut intégrer l’appel dans l’orchestrateur.

---

### P0.3 — Canoniser les `semantic_phrase.source_unit_ids`

`PAGEPRINT` doit convertir les références sémantiques vers les IDs canoniques.

Sinon `PAGETRANSLATE` peut dupliquer :

```text id="w26tt6"
semantic_phrase legacy
+
synthetic_semantic_phrase coalesced
```

Il faut produire :

```json id="w0zvns"
{
  "source_unit_ids": [
    "p001_block_001_line_001_phrase_001",
    "p001_block_001_line_002_phrase_001"
  ],
  "structural_context": {
    "block_unit_id": "p001_block_001"
  }
}
```

---

### P0.4 — Restreindre `PAGEPRINT.views.translation_units`

`views.translation_units` ne doit pas contenir `span`.

Restreindre à :

```text id="mdmxe1"
semantic_phrase
semantic_group
phrase
line
block
```

En pratique, côté `PAGEPRINT`, comme `semantic_phrase` n’est pas matérialisée comme unité normale, la vue devrait contenir seulement :

```text id="uahms9"
phrase
line
block
```

et le `semantic_system` gère les unités sémantiques.

---

### P0.5 — Ajouter un test contractuel complet

Test obligatoire :

```text id="ax2vd3"
YOLO/ONNX détecte une formule
→ special_regions contient formula
→ PAGEPRINT la transforme en protected_visual_region
→ views.translation_units ne contient rien de cette zone
→ views.protected_visual_units contient la zone
→ PAGETRANSLATE ne traduit rien dans cette zone
→ reconstruction_units ne contient pas cette zone en texte traduit
```

---

## P1 — Améliorations fortes

### P1.1 — Ajouter un `legacy_id_index` dans `PAGEPRINT`

Chaque unité générée par `PAGEPRINT` devrait garder :

```text id="8f1pmy"
legacy_id
source_node_id
legacy_path
```

Puis créer un index :

```json id="3nwgdm"
"indexes": {
  "legacy_id_to_unit_id": {
    "legacy_phrase_1": "p001_block_001_line_001_phrase_001"
  }
}
```

Cela permettra à `PAGETRANSLATE` de résoudre les `semantic_phrase.source_unit_ids` amont.

---

### P1.2 — Ajouter `views.translation_candidates_debug`

Il faut garder une vue d’audit :

```text id="a05sgl"
toutes les unités candidates
+
raison d’inclusion/exclusion
```

Exemple :

```json id="bi75tu"
{
  "unit_id": "p001_block_001_line_001_phrase_001",
  "candidate": false,
  "reason": "covered_by_protected_visual_region"
}
```

Cela rendra les erreurs faciles à diagnostiquer.

---

### P1.3 — Renommer clairement les policies

Actuellement on a :

```text id="jysp40"
render_policy = background_only
translation_strategy = background_only
translation_policy = preserve_visual_region
render_policy = preserve_source_region
```

Il y a une petite confusion entre les vocabulaires amont et PAGEPRINT.

Je recommanderais de figer :

```text id="be68d3"
translation_strategy:
- layout_constrained
- paragraph_reflow
- semantic_reflow
- exact_preserve
- background_only

render_policy:
- anchored_text
- fixed_preserve
- background_only
- preserve_original_region
```

Et éviter d’avoir plusieurs noms pour le même concept.

---

### P1.4 — Renforcer `PAGETRANSLATE.selector`

Ajouter des garde-fous :

```python id="t8sepi"
if item.get("render_policy") == "background_only":
    skip

if item.get("covered_by_protected_region_id"):
    skip

if item.get("constraints", {}).get("skip_translation"):
    skip

if semantic_entry has unresolved source ids and no block_id:
    skip or resolve legacy ids
```

Même si `PAGEPRINT` fait bien son travail, `PAGETRANSLATE` doit rester défensif.

---

### P1.5 — Ajouter des statuts de traduction plus précis

Au lieu de :

```text id="8qh54t"
translated / preserved / dry_run / error
```

je recommande :

```text id="sxg7ke"
translated
dry_run
error
unchanged_suspect
preserved_by_policy
skipped_non_translatable
```

Cela évite les ambiguïtés.

---

## P2 — Améliorations de robustesse

### P2.1 — QA traduction plus intelligente

Ajouter :

```text id="s41g8z"
- contrôle négations ;
- contrôle ponctuation ;
- contrôle parenthèses/guillemets ;
- contrôle noms propres ;
- contrôle références légales/bibliographiques ;
- contrôle terminologie obligatoire ;
- contrôle cohérence entre unités voisines.
```

---

### P2.2 — Meilleure gestion des inline formulas

Doctrine actuelle :

```text id="654f3t"
formule/code/équation = zone protégée visuelle
```

C’est bon.

Mais pour les cas inline :

```text id="m20p5k"
The equation E = mc² explains...
```

il faut décider :

```text id="z8v5or"
V1 : ligne entière protégée si segmentation incertaine.
V2 : découpage avec protected_inline_anchor.
```

Je conseille pour V1 :

```text id="p3n5st"
ne pas tenter de reconstruire inline formula.
protéger la ligne si la formule est dominante.
sinon utiliser placeholder visuel plus tard.
```

---

# 5. Synthèse stricte

## Ce qui est bon

```text id="zc3dsz"
PAGEPRINT sait représenter INPUT_DATA.
PAGEPRINT sait consommer des special_regions.
PAGEPRINT sait protéger les unités couvertes par protected_visual_region.
PAGEPRINT produit protected_visual_units.
PAGETRANSLATE sélectionne mieux les unités.
PAGETRANSLATE coalesce les lignes ouvertes.
PAGETRANSLATE ne duplique plus reconstruction_units dans le cas simple.
PAGETRANSLATE transmet un contexte riche au traducteur.
```

## Ce qui est faux ou incomplet actuellement

```text id="57u6lj"
PAGEPRINT ne lance pas YOLO/ONNX.
special_region_detector.py n’est pas intégré directement.
special_class n’est pas lu comme region_type.
semantic_phrase.source_unit_ids peut ne pas correspondre aux IDs canoniques.
views.translation_units contient encore span.
La chaîne protected_visual complète dépend trop de l’orchestrateur.
```

## Ce qu’il faut faire maintenant

Priorité absolue :

```text id="vi4k39"
1. Intégrer detect_special_regions(...) avant PAGEPRINT.
2. Corriger special_class → region_type.
3. Canoniser semantic_phrase.source_unit_ids.
4. Restreindre views.translation_units.
5. Ajouter tests end-to-end PAGEPRINT → PAGETRANSLATE sur formule/code détecté.
```

Après ces corrections, la chaîne deviendra cohérente :

```text id="ufdoxr"
YOLO/ONNX ou heuristique spéciale
→ special_regions typées
→ PAGEPRINT protected_visual_region
→ unités internes non traduisibles
→ PAGETRANSLATE skip total
→ RECONSTRUCTION préserve pixels/région originale
```

Mon classement actuel :

```text id="i6y8do"
PAGEPRINT : V1-alpha avancée, architecture solide, intégration détecteur à finaliser.
PAGETRANSLATE : V1-alpha+ solide, dépend encore trop des contrats PAGEPRINT.
Pipeline complet : pas encore V1-beta tant que special_region_detector n’est pas orchestré et que les IDs sémantiques ne sont pas canonisés.
```

