J’ai inspecté le nouveau `page.zip`. Il contient maintenant deux paquets :

```text id="9wtqxk"
pageprint/
pagetranslate/
```

Donc l’analyse ne porte plus seulement sur `pagetranslate/`, mais aussi sur son interaction réelle avec `PAGEPRINT`.

# Verdict général

Cette version est **beaucoup plus sérieuse que la précédente**. Plusieurs défauts P0 signalés avant ont été corrigés :

```text id="5w8r1k"
✓ semantic_phrase ne se double plus avec line/phrase du même bloc
✓ semantic_phrase gagne bien sur semantic_group
✓ translatable=False / background_only sont mieux filtrés
✓ state-of-the-art / well-known / pre-trained ne sont plus protégés comme formules
✓ unchanged translation devient needs_review
✓ le contexte riche est mieux transmis au TranslatorBridge
✓ TypeError n’est plus masqué grossièrement
✓ les erreurs traducteur sont maintenant locales par unité
✓ la projection semantic_phrase ne met plus toute la traduction dans le premier fragment
```

Mais il reste des points profonds qui peuvent encore casser le fonctionnement WYSIWYG réel.

Je classerais cette version ainsi :

```text id="ebenvt"
PAGETRANSLATE V1-alpha solide
proche V1-beta
mais pas encore stable pour documents complexes
```

Le cœur est bon. Les fragilités restantes sont surtout dans :

```text id="e9la6g"
1. la couverture réelle du texte ;
2. la fusion de phrases visuelles ;
3. la distinction acronyme / titre en majuscules ;
4. la consommation des unités source ;
5. le contrat avec le reconstructeur ;
6. l’application réelle de la terminologie ;
7. la QA sémantique encore trop superficielle.
```

---

# 1. Fonctionnement global réel

Le pipeline actuel fonctionne comme ceci :

```text id="6ujpsh"
PAGEPRINT
  ↓
INPUT_DATA canonique
  ↓
PAGETRANSLATE
  ↓
select_translation_units()
  ↓
annotate_sentence_boundaries()
  ↓
coalesce_translation_units()
  ↓
attach_unit_context()
  ↓
protect_text()
  ↓
TranslatorBridge.translate()
  ↓
restore_text()
  ↓
unit_quality()
  ↓
project_translations()
  ↓
translated_input_data + views.reconstruction_units
```

C’est une bonne architecture.

La séparation est maintenant claire :

```text id="zx9fux"
PAGEPRINT = comprendre, structurer, décrire, contraindre
PAGETRANSLATE = traduire selon cette compréhension
futur reconstructeur = consommer la vue de reconstruction traduite
```

C’est exactement la bonne méthode pour un pipeline WYSIWYG.

---

# 2. Interaction avec PAGEPRINT

## 2.1 `PAGEPRINT` produit maintenant un vrai contrat consommable

Dans `pageprint/schema.py`, `INPUT_DATA` contient bien :

```text id="ejfnr6"
units
regions
graph
relations
page_intelligence
document_comprehension
style_system
semantic_system
translation_context
reconstruction_constraints
views.translation_units
views.render_units
```

C’est bon.

`PAGETRANSLATE` consomme principalement :

```text id="yuwaf4"
input_data["semantic_system"]
input_data["views"]["translation_units"]
input_data["units"]
input_data["translation_context"]
input_data["page_intelligence"]
```

Méthodologiquement, c’est correct.

## 2.2 `semantic_system` est maintenant pris en compte

`PAGEPRINT` construit :

```text id="n60fuz"
semantic_system.semantic_phrases
semantic_system.semantic_groups
```

à partir de `page_structure["blocks"][...]["semantic_phrases"]` et `semantic_groups`.

C’est important, parce que la traduction doit privilégier le sens, pas la ligne visuelle.

Donc le chemin idéal devient :

```text id="th8xft"
semantic_phrase si disponible
sinon semantic_group
sinon phrase visuelle
sinon line
sinon block
```

Ce principe est maintenant implémenté.

---

# 3. Sélection des unités : progrès réel, mais encore dangereux

## 3.1 Ce qui est bon

Dans `selector.py`, la priorité est bien :

```text id="fhtsxt"
semantic_phrase > semantic_group > phrase > line > block
```

Et la sélection par bloc évite maintenant la première erreur grave :

```text id="bnzln7"
semantic_phrase + phrase/line du même bloc
```

J’ai testé un cas :

```text id="u8cszf"
p1 = "This is a long sentence"
p2 = "continued here."
semantic_phrase = "This is a long sentence continued here."
```

Résultat actuel :

```text id="39ihns"
sp1 semantic_phrase uniquement
```

Donc cette correction est bonne.

## 3.2 `semantic_phrase` gagne bien contre `semantic_group`

Test avec :

```text id="ctym7x"
semantic_phrase = sp1
semantic_group  = sg1 couvrant le même bloc
```

Résultat :

```text id="7isj3i"
sp1 uniquement
```

Donc la double traduction `semantic_phrase + semantic_group` est corrigée.

---

# 4. Défaut important : risque de perte de texte si la couverture phrase est incomplète

La sélection par bloc choisit le premier niveau disponible :

```text id="r1ixhd"
si le bloc contient au moins une phrase → traduire les phrases
sinon les lignes
sinon le bloc
```

Problème : si un bloc contient plusieurs lignes, mais que seulement certaines lignes ont des `phrase` units, les lignes sans phrase peuvent disparaître de la traduction.

Exemple testé :

```text id="c3ouls"
block b1
  line l1 = "First line text"
    phrase p1 = "First line text"
  line l2 = "Second line text"
    aucune phrase
```

Résultat actuel :

```text id="n0qkyl"
p1 uniquement sélectionnée
l2 ignorée
```

Donc la phrase `Second line text` est perdue.

### Correction attendue

Le fallback ne doit pas être seulement par bloc. Il doit être par **couverture**.

Il faut calculer les zones ou descendants couverts :

```text id="wvpfdo"
texte total du bloc
texte couvert par phrases
texte non couvert
```

Puis :

```text id="2hcwac"
si phrase couvre tout le bloc → traduire phrases
si phrase couvre partiellement → traduire phrases + fallback line pour les lignes non couvertes
si aucune phrase → traduire lignes
```

Il faudrait une fonction du type :

```python id="0k1z0k"
def coverage_complete(block_unit, selected_units, unit_map) -> bool:
    ...
```

ou plus simple :

```text id="0e67py"
pour chaque ligne :
    si elle a des phrases candidates → sélectionner phrases
    sinon sélectionner la ligne
```

Le bon niveau de fallback est donc plutôt :

```text id="6o3d6y"
bloc → ligne → phrase
```

pas seulement bloc entier.

---

# 5. Défaut important : les titres en majuscules sont exclus comme acronymes

Dans `selector.py`, la regex :

```python id="cb7873"
ACRONYM_RE = re.compile(r"^[A-Z0-9][A-Z0-9&./+-]{1,12}$")
```

exclut beaucoup de vrais titres :

```text id="v5owm4"
INAM          → exclu, normal
PDF           → exclu, normal
INTRODUCTION  → exclu, mauvais
BACKGROUND    → exclu, mauvais
CONCLUSION    → exclu, mauvais
ABSTRACT      → exclu, mauvais
```

C’est un problème sérieux.

Un document scientifique ou administratif contient souvent des titres en majuscules. Certains doivent être traduits :

```text id="lhmokl"
BACKGROUND → CONTEXTE
SUMMARY → RÉSUMÉ
FINDINGS → CONSTATS
RECOMMENDATIONS → RECOMMANDATIONS
```

### Correction attendue

Ne pas décider “acronyme” uniquement par majuscules.

Une meilleure logique :

```text id="k8fcbe"
- acronyme court : 2 à 6 caractères, peu ou pas de voyelles, ou connu dans protected_tokens ;
- mot majuscule long : probablement titre, donc traduisible ;
- si role == title / section_heading : traduire même en majuscules ;
- si terme explicitement protégé : préserver.
```

Exemple :

```python id="il4zi1"
def _is_probable_acronym(text: str, role: str | None = None) -> bool:
    s = normalize_spaces(text)
    if role in {"title", "section_heading"} and len(s) > 5:
        return False
    if not re.fullmatch(r"[A-Z0-9&./+-]{2,8}", s):
        return False
    if len(s) > 6 and re.search(r"[AEIOUY]{2,}", s):
        return False
    return True
```

---

# 6. Défaut important : fusion abusive possible des lignes sans ponctuation

Le module `coalescer.py` fusionne des unités visuelles quand une phrase semble continuer.

C’est une bonne idée pour gérer :

```text id="vbhxba"
This is a long sentence
continued on the next line.
```

Mais la logique actuelle peut aussi fusionner des listes ou des blocs séparés.

Test réel :

```text id="jsz6r8"
Item one
Item two
Item three
```

Le système produit :

```text id="w3mbhx"
synthetic_semantic_phrase = "Item one Item two Item three"
```

C’est mauvais si ce sont trois éléments de liste.

La cause est dans `sentence_boundary.py` :

```python id="fpg5oi"
continues_after = bool(next_item and not ends_sentence and not atomic_label)
```

Même si `_break_type()` détecte `soft_wrap`, cela n’empêche pas la continuation.

Donc le système sait que la rupture est verticale, mais continue quand même à fusionner.

### Correction attendue

`continues_to_next` doit tenir compte de `break_type`.

Actuellement :

```text id="ligk9b"
pas de ponctuation = continue
```

Il faudrait :

```text id="vukg4w"
pas de ponctuation + alignement compatible + interligne compatible + même paragraphe = continue
```

Concrètement :

```python id="v9ndhw"
continues_after = bool(
    next_item
    and not ends_sentence
    and not atomic_label
    and break_type in {"same_line_continuation", "soft_wrap"}
    and _same_paragraph_like(item, next_item)
)
```

Mais `soft_wrap` seul ne suffit pas. Il faut regarder :

```text id="atb325"
indentation proche
x0 proche
line spacing normal
absence de bullet
absence de numérotation
absence de changement de style
absence de rôle list_item / heading / table_cell
relation graph same_paragraph ou continues
```

Il faut aussi détecter :

```text id="smsdtt"
• Item one
- Item two
1. Item three
a) Item four
```

et empêcher la fusion.

---

# 7. Défaut important : `semantic_phrase` sans source ids peut encore doubler

Cas testé :

```json id="oi6p8k"
{
  "semantic_phrases": [
    {
      "unit_id": "sp1",
      "text": "Same sentence."
    }
  ]
}
```

sans `source_unit_ids`, sans `block_unit_id`.

Résultat :

```text id="w7d8au"
sp1 semantic_phrase
p1 phrase
```

Donc double sélection.

Dans la sortie normale de `PAGEPRINT`, ça devrait souvent être évité si `semantic_phrases` vient d’un bloc correctement identifié. Mais le code reste fragile si une entrée sémantique est incomplète.

### Correction attendue

Une `semantic_phrase` ne devrait être sélectionnée que si elle a au moins :

```text id="h7n3nk"
source_unit_ids
ou block_unit_id
ou bbox fiable
ou relation explicite dans le graphe
```

Sinon elle doit être considérée comme `unsafe_semantic_unit` et non comme unité prioritaire.

Exemple :

```python id="0v2v74"
if level in {"semantic_phrase", "semantic_group"}:
    if not source_ids and not block_id and not bbox:
        continue
```

ou :

```text id="ht5q90"
la traduire, mais ne pas bloquer/réinjecter automatiquement
status = needs_alignment
```

---

# 8. Projection : nettement améliorée, mais le contrat reconstructeur doit être figé

La projection ne met plus toute une `semantic_phrase` dans le premier fragment source. C’est une bonne correction.

Maintenant, pour une `semantic_phrase` couvrant `p1` et `p2`, le système fait :

```text id="e5iwka"
p1.translation.skip_individual_render = True
p2.translation.skip_individual_render = True
views.reconstruction_units contient sp1
```

C’est la bonne direction.

Mais il faut figer une règle importante :

```text id="ra38qd"
Le reconstructeur ne doit pas lire naïvement units[].content.translated_text.
Il doit lire views.reconstruction_units.
```

Sinon il risque de :

```text id="ohgfwf"
- ignorer les semantic_phrases ;
- rendre les anciennes phrases visuelles ;
- rendre des parents agrégés ;
- dupliquer des textes ;
- perdre les unités synthétiques.
```

## Point critique

`semantic_phrase` n’est pas un niveau canonique dans `pageprint/schema.py`.

`PAGEPRINT` connaît :

```text id="mltbb7"
page, region, block, line, phrase, span, word, char, image, drawing, table, cell, formula, code, protected_visual, overlay
```

Mais pas :

```text id="42w2cg"
semantic_phrase
semantic_group
```

Donc `semantic_phrase` existe dans :

```text id="i02fwq"
semantic_system
views.reconstruction_units
translation_units
```

mais pas dans `units`.

C’est acceptable, mais seulement si le reconstructeur est conçu pour ça.

### Deux options possibles

## Option A — Semantic phrase reste une vue

Dans ce cas :

```text id="ihm7pf"
semantic_phrase n’est pas une unité canonique PAGEPRINT
elle est une unité logique de traduction/reconstruction
le reconstructeur consomme views.reconstruction_units
```

C’est propre.

Mais il faut documenter fortement :

```text id="2fb25v"
views.reconstruction_units est la source de vérité après traduction
```

## Option B — Ajouter semantic_phrase dans `units`

Dans ce cas il faut étendre `pageprint/schema.py` :

```text id="8qplke"
UNIT_LEVELS += semantic_phrase, semantic_group
```

Puis créer de vraies unités :

```text id="gp0eav"
semantic_phrase contains phrase/line fragments
semantic_group contains semantic_phrase
```

C’est plus robuste, mais plus lourd.

Pour l’instant, je conseille **Option A** : garder `semantic_phrase` comme vue logique, mais imposer `views.reconstruction_units` au reconstructeur.

---

# 9. Problème méthodologique : la traduction n’applique pas encore vraiment la terminologie

Le profil contient :

```text id="cb2i3b"
terminology
protected_tokens
domain
subdomain
style
tone
```

Le `TranslatorBridge` les transmet maintenant au traducteur. C’est bien.

Mais côté `PAGETRANSLATE`, il n’y a pas encore de module autonome :

```text id="8k38lc"
terminology.py
```

qui ferait :

```text id="w4l7ch"
- verrouillage des termes ;
- application des preferred_terms ;
- vérification des reserved_terms ;
- normalisation post-traduction ;
- contrôle cohérence terminologique page/document.
```

Actuellement, si `DocumentTranslator` ne respecte pas la terminologie, `PAGETRANSLATE` ne corrige pas vraiment.

### À ajouter

Un module :

```text id="lpemgy"
terminology.py
```

avec :

```python id="fobftw"
apply_pre_translation_locks()
apply_post_translation_glossary()
check_terminology_consistency()
```

Exemple de structure :

```json id="p0d0y2"
{
  "locked_terms": ["INAM", "AMU", "RAMO"],
  "preferred_terms": {
    "medical audit": "contrôle médical",
    "claim": "facture/prestation selon contexte"
  },
  "reserved_terms": {
    "fraud": "fraude uniquement si intention caractérisée"
  }
}
```

---

# 10. Protection : meilleure, mais les placeholders restent fragiles

`protection.py` protège :

```text id="5tgvjh"
URL
emails
DOI
chemins
références
nombres + unités
formules
nombres isolés
tokens explicites
```

C’est bien.

Le bug des mots composés est corrigé :

```text id="8bjddn"
state-of-the-art method → non protégé
well-known algorithm    → non protégé
pre-trained model       → non protégé
```

Très bon.

Mais les placeholders :

```text id="2djv2f"
__PT_0001__
__PT_0002__
```

peuvent être altérés par certains traducteurs.

Exemples possibles :

```text id="1gu6oz"
__PT_0001__ → __ PT_0001 __
__PT_0001__ → PT_0001
__PT_0001__ → __PT_0001__.
```

Si le placeholder est modifié, `restore_text()` ne restaure rien.

### Correction recommandée

Utiliser des marqueurs plus résistants :

```text id="sk4cy2"
⟦PT0001⟧
<nt id="PT0001"/>
[[[PT0001]]]
```

Et ajouter une restauration tolérante :

```text id="1m445t"
- exact match
- match avec espaces parasites
- match sans underscores
- match XML-like
```

Il faut aussi ajouter dans le prompt traducteur :

```text id="j2sxi1"
Ne jamais modifier les marqueurs ⟦PT0001⟧.
```

---

# 11. QA : utile, mais encore trop syntaxique

`quality.py` contrôle :

```text id="tk2q2j"
empty_translation
unchanged
number_mismatch
unit_mismatch
protected_token_mismatch
source_language_leak
wysiwyg_overflow_risk
```

C’est une bonne base.

Mais ce n’est pas encore une QA de traduction complète.

## Limites actuelles

### 11.1 Détection de fuite anglaise trop simple

Le système cherche des marqueurs comme :

```text id="agbrw4"
the, and, with, from, this, that...
```

C’est utile mais faible.

Une phrase anglaise courte peut passer :

```text id="ayg8ug"
Deep learning model
Training loss
Vision system
```

Et une phrase française contenant “and” dans un nom propre peut être faussement signalée.

### 11.2 Nombres trop stricts

Le contrôle compare les listes :

```text id="1mtkh1"
source_numbers == translated_numbers
```

Mais la traduction peut légitimement changer la forme :

```text id="elqllg"
1,000 → 1 000
3.5 → 3,5
```

Il faut normaliser les nombres avant comparaison.

### 11.3 Unités trop limitées

Les unités couvertes sont utiles, mais insuffisantes :

```text id="riie7e"
mol/L
mg/dL
m²
m³
km/h
FCFA
XOF
USD
EUR
µg
UI
mmHg
bpm
```

Pour les documents médicaux, administratifs et scientifiques, il faut enrichir.

### 11.4 Pas encore de QA sémantique

Il manque :

```text id="j13i6m"
- inversion de négation ;
- omission ;
- ajout ;
- changement de modalité : must / may / should ;
- changement de relation logique : cause, opposition, condition ;
- changement de terme technique ;
- mauvais niveau de langue ;
- contradiction avec le contexte.
```

À terme, il faudra probablement un module :

```text id="2u1121"
semantic_qa.py
```

même simple au départ.

---

# 12. WYSIWYG : le risque overflow est trop approximatif

Actuellement, `_overflow_risk()` regarde surtout :

```text id="4ia8b7"
ratio caractères
bbox width < 180
strategy layout_constrained
```

C’est une approximation.

Mais `PAGEPRINT` produit déjà des informations plus riches :

```text id="xm1j18"
unit["constraints"]
unit["transformation_budget"]
unit["layout_freedom"]
unit["render_contract"]
unit["translation_forecast"]
```

`PAGETRANSLATE` ne les consomme pas encore assez.

### Ce qu’il faudrait consommer

Depuis `PAGEPRINT` :

```text id="i8ukm4"
transformation_budget.max_text_expansion_ratio
transformation_budget.max_font_reduction_ratio
transformation_budget.max_bbox_growth_x_pt
transformation_budget.max_bbox_growth_y_pt
layout_freedom.allow_reflow
layout_freedom.allow_line_wrap
render_contract.overflow_policy
translation_forecast.overflow_probability
```

Actuellement, `views.reconstruction_units.layout_budget` contient surtout :

```text id="cfidpu"
bbox width
bbox height
area
```

C’est trop pauvre.

### Correction attendue

Dans `context_builder.py` ou `selector.py`, il faut attacher :

```python id="m8sp21"
"wysiwyg_constraints": {
    "bbox": item.get("bbox"),
    "transformation_budget": source_unit.get("transformation_budget"),
    "layout_freedom": source_unit.get("layout_freedom"),
    "render_contract": source_unit.get("render_contract"),
    "translation_forecast": source_unit.get("translation_forecast"),
}
```

Et dans `quality.py`, comparer la traduction au budget réel :

```text id="j03c1o"
si ratio > max_text_expansion_ratio → overflow_risk high
si allow_reflow=False et traduction plus longue → high
si table_cell et ratio > 1.15 → medium/high
```

---

# 13. Relation `sentence_boundary → coalescer` : interaction à renforcer

La relation actuelle est :

```text id="cl678o"
sentence_boundary détecte continues_to_next
coalescer fusionne les unités ouvertes
```

C’est bon conceptuellement.

Mais il manque une couche de décision :

```text id="xfahc7"
Est-ce une vraie phrase multi-ligne ou une liste / tableau / titre / label ?
```

La décision doit utiliser plus que la ponctuation :

```text id="5guy09"
geometry
indentation
baseline
line spacing
style
role
object_type
graph relations
relations.flow_to_next
relations.edges
```

Actuellement, `PAGEPRINT` produit déjà un graphe :

```text id="cu53m4"
contains
belongs_to
flows_to
continues
same_paragraph
caption_of
label_of
```

Mais `PAGETRANSLATE` exploite très peu `input_data["graph"]` et `input_data["relations"]`.

C’est dommage.

### Amélioration clé

Créer :

```text id="gk2dlu"
flow_resolver.py
```

qui répond :

```python id="whv6by"
is_same_sentence_continuation(prev, current, input_data) -> bool
is_same_paragraph(prev, current, input_data) -> bool
is_list_boundary(prev, current, input_data) -> bool
is_table_boundary(prev, current, input_data) -> bool
```

Puis `sentence_boundary.py` ne décide plus seul.

---

# 14. Consommation des relations : actuellement insuffisante

`PAGEPRINT` construit un graphe documentaire. C’est une richesse importante.

Mais `PAGETRANSLATE` consomme surtout :

```text id="y2d2s8"
parent_id
block_id
source_unit_ids
reading_order_index
bbox
```

Il devrait aussi consommer :

```text id="i7yg4i"
relations.previous_unit_id
relations.next_unit_id
relations.flow_to_next
graph.edges relation=continues
graph.edges relation=same_paragraph
graph.edges relation=caption_of
graph.edges relation=label_of
region_memberships
```

Pourquoi ?

Parce qu’une traduction correcte dépend souvent de ces relations :

```text id="oz4n00"
caption → doit garder le lien avec image/table
label → souvent court, contraint, parfois non traduisible
same_paragraph → peut être fusionné
different_region → ne doit pas être fusionné
protected_visual_region → ne pas traduire/reconstruire
table_cell → traduction contrainte cellule par cellule
```

Actuellement, ce niveau relationnel n’est pas assez utilisé.

---

# 15. Statuts : bonne base, mais il manque des statuts intermédiaires

Actuellement :

```text id="rpvkvh"
translated
preserved
dry_run
error
```

C’est bien mais insuffisant.

Pour un pipeline WYSIWYG, il faudrait aussi :

```text id="3ntbj1"
skipped_policy
skipped_empty
skipped_protected
needs_alignment
needs_review
overflow_risk
terminology_conflict
projection_failed
```

Sinon des cas très différents seront mélangés.

Exemple :

```text id="w4o384"
preserved
```

peut signifier :

```text id="r3dydw"
- traduction identique légitime ;
- traduction échouée ;
- acronyme ;
- code ;
- formule ;
- erreur du modèle ;
- dry-run déguisé.
```

Il faut être plus explicite.

---

# 16. Scénarios où ça marchera bien

Cette version devrait fonctionner correctement sur :

```text id="at4rdr"
- pages textuelles simples ;
- paragraphes bien segmentés ;
- semantic_phrases disponibles avec source_unit_ids ;
- textes sans listes complexes ;
- peu de tableaux ;
- peu de labels/diagrammes ;
- documents où le reconstructeur consomme views.reconstruction_units ;
- traductions avec expansion modérée.
```

Exemple :

```text id="ui80s3"
bloc paragraphe anglais → semantic_phrase → traduction française → reconstruction semantic_phrase
```

Là, le pipeline est bon.

---

# 17. Scénarios où ça peut encore échouer

## 17.1 Document avec titres en majuscules

```text id="603yo6"
BACKGROUND
METHODOLOGY
FINDINGS
RECOMMENDATIONS
```

Risque :

```text id="vocr4x"
titres exclus comme acronymes
```

## 17.2 Bloc avec couverture phrase incomplète

```text id="uc9fix"
line 1 avec phrase
line 2 sans phrase
```

Risque :

```text id="g12sfg"
line 2 non traduite
```

## 17.3 Liste sans ponctuation

```text id="u5fb8u"
Patient information
Medical history
Treatment plan
```

Risque :

```text id="9d8lfr"
fusion en une seule phrase artificielle
```

## 17.4 Semantic phrase sans ancrage

```text id="h1euy4"
semantic_phrase sans source_unit_ids ni block_id
```

Risque :

```text id="pvjem6"
double traduction ou reconstruction sans bbox fiable
```

## 17.5 Traduction longue dans zone étroite

```text id="acrsr0"
court label anglais → long label français
```

Risque :

```text id="obnwqw"
overflow sous-estimé
```

## 17.6 Tableau

Risque :

```text id="xchzda"
cellules mal contraintes
fusion inter-cellules possible si mauvais block_id/reading_order
```

---

# 18. Ce que je changerais maintenant

## P0 — Corrections avant V1-beta

```text id="p0j96e"
1. Corriger la détection acronyme pour ne pas exclure INTRODUCTION/BACKGROUND/etc.
2. Corriger la sélection incomplète : fallback par ligne si couverture phrase partielle.
3. Empêcher la fusion de lignes/listes simplement parce qu’il n’y a pas de ponctuation.
4. Refuser ou marquer unsafe les semantic_phrases sans source_unit_ids/block_id/bbox.
5. Imposer officiellement views.reconstruction_units comme source du reconstructeur après traduction.
```

## P1 — Robustesse WYSIWYG

```text id="kc12zi"
6. Consommer transformation_budget/layout_freedom/render_contract de PAGEPRINT.
7. Enrichir reconstruction_units avec les vrais budgets de mise en page.
8. Ajouter flow_resolver.py pour exploiter graph/relations.
9. Ajouter terminology.py.
10. Ajouter restauration robuste des placeholders.
```

## P2 — QA traduction

```text id="jjw3rp"
11. Normaliser les nombres avant comparaison.
12. Enrichir les unités médicales/scientifiques/financières.
13. Ajouter QA négation/modalité/omission.
14. Ajouter statut needs_alignment/projection_failed/terminology_conflict.
15. Ajouter tests de contrats inter-modules.
```

---

# 19. Tests indispensables à ajouter

Il faut maintenant des tests orientés fonctionnement réel :

```text id="bnbrns"
test_uppercase_heading_is_translatable
test_acronym_stays_protected
test_partial_phrase_coverage_falls_back_to_line
test_list_items_are_not_coalesced
test_soft_wrap_paragraph_is_coalesced
test_semantic_phrase_without_anchor_is_unsafe
test_reconstruction_units_are_source_of_truth
test_table_cells_do_not_merge
test_layout_budget_uses_pageprint_transformation_budget
test_placeholder_restoration_after_translator_noise
test_terminology_locked_terms_preserved
test_numbers_normalized_before_comparison
```

Ces tests sont plus importants que de simples tests de compilation.

---

# 20. Conclusion technique

La méthodologie générale est maintenant bonne :

```text id="ipx27d"
comprendre avant traduire,
traduire par unité sémantique,
protéger les éléments intouchables,
contrôler la sortie,
réinjecter dans une vue reconstruisible.
```

L’implémentation commence à respecter cette méthodologie. On n’est plus dans un simple wrapper. On a une vraie unité pipeline.

Mais pour que cela marche correctement sur des documents réels, il faut renforcer trois contrats :

```text id="eh35pq"
1. contrat de couverture :
   aucune portion textuelle traduisible ne doit disparaître ;

2. contrat de relation :
   les unités ne doivent être fusionnées que si le graphe/geometry confirme une continuité réelle ;

3. contrat de reconstruction :
   le reconstructeur doit consommer views.reconstruction_units, pas deviner dans units[].
```

Le système est donc sur la bonne voie, mais la prochaine étape doit être très stricte : **tester les cas de rupture**, pas seulement les cas propres.

Le vrai critère V1-beta devrait être :

```text id="to0m0q"
zéro doublon,
zéro perte de texte,
zéro fusion abusive de listes,
zéro titre traduisible exclu,
projection reconstruisible sans ambiguïté.
```

