# Verdict

**Non, ce n’est pas encore assez bon pour passer directement à l’évaluation sérieuse du moteur de traduction.**

La v3 est **nettement meilleure que la version précédente**, mais l’audit montre encore des erreurs P0 sur la sélection des unités. Si on branche maintenant un vrai moteur de traduction, on ne saura pas si les mauvais résultats viennent :

```text
du moteur de traduction
ou d’une mauvaise sélection PAGEPRINT/PAGETRANSLATE
```

Donc la bonne conclusion est :

```text
PAGEPRINT/PAGETRANSLATE v3 = progrès réel
mais pas encore validé pour tester proprement le moteur de traduction.
```

On peut faire un **pré-test du moteur** sur les unités propres, mais pas encore un vrai test global.

---

# 1. Ce qui s’est clairement amélioré

## 1.1 PAGEPRINT détecte mieux la nature de la page

La page est correctement reconnue comme :

```text
page_role: toc
page_family: toc
layout_type: toc_page
document_type: book_page
```

C’est bon.

L’audit compact montre maintenant des rôles TOC :

```text
toc_entry: 156
toc_page_reference: 129
toc_section_number: 33
toc_bullet_marker: 13
toc_title: 3
```

C’est un progrès majeur par rapport à la version précédente où presque tout était `unknown`.

## 1.2 Les bullets sont mieux séparés

Avant, on avait des textes comme :

```text
■Hidden layers
■Output layer
■Precision and recall
```

Maintenant, `PAGETRANSLATE` reçoit plutôt :

```text
Hidden layers
Output layer
Precision and recall
```

C’est une bonne correction.

## 1.3 Le nombre de protected_visual a fortement baissé

Avant, il y avait beaucoup trop de zones rouges/protégées.

Maintenant :

```text
protected_visual_regions: 3
protected_visual_units: 3
```

C’est mieux.

Mais attention : **les 3 restantes sont encore fausses ou discutables**.

## 1.4 PAGETRANSLATE ne sélectionne plus word/char

La sélection finale est :

```text
34 unités de traduction
28 phrases
5 semantic_phrases synthétiques
1 block
0 word
0 char
```

C’est bon sur le principe.

---

# 2. Problème P0 : PAGEPRINT protège encore du texte naturel

Même après correction, `PAGEPRINT` marque encore certains textes naturels comme `background_only` / `protected_visual`.

Les cas problématiques :

```text
(weights)
(3D images)
Splitting your data for train/validation/test
Data preprocessing
Project: Image classification for color images
Is accuracy the best metric for evaluating a model?
Getting your data ready for training
Drawbacks of MLPs for processing
```

Ces éléments ne doivent pas être `protected_visual`.

## Ce que PAGEPRINT fait actuellement

Exemple :

```text
p001_block_020
Splitting your data for train/validation/test 151 Data preprocessing 153

policy:
translation_strategy = background_only
render_policy = background_only
translatable = False
protected_visual = True
```

C’est faux.

Cette zone est du texte naturel de table des matières. Elle doit être traduite, sauf les numéros de page.

## Correction nécessaire

Dans `PAGEPRINT`, il faut une règle forte :

```python
if page_role == "toc" and role in {"toc_entry", "toc_title"}:
    protected_visual = False
    translatable = True
    translation_strategy = "layout_constrained"
    render_policy = "anchored_text"
```

Sauf si l’unité est réellement :

```text
toc_page_reference
toc_section_number
toc_bullet_marker
logo
vrai code
vraie formule
```

Le problème vient probablement d’un ancien signal legacy `background_only/protected_visual` qui continue à contaminer les unités TOC. La fonction `_apply_toc_policy()` ne force pas assez la politique finale. Elle définit le rôle, mais ne réinitialise pas toujours :

```text
translatable
translation_strategy
render_policy
protected_visual
skip_translation
skip_text_reconstruction
```

Il faut le faire explicitement.

---

# 3. Problème P0 : faux protected_visual sur `(weights)` et `(3D images)`

Dans l’audit visuel, les zones magenta restent sur :

```text
(weights)
(3D images)
```

Ce ne sont pas des formules.

Pour cette page :

```text
Number of parameters (weights)
```

doit devenir une seule unité traduisible ou au minimum :

```text
Number of parameters → traduisible
(weights)            → traduisible ou terme technique contextualisé
```

Pas :

```text
(weights) → protected_visual
```

Même chose :

```text
Convolution over color images (3D images)
```

doit rester traduisible.

Correction dans la détection spéciale :

```text
si page_role == toc :
    ne pas classer comme formula/code les parenthèses lexicales
```

Règles concrètes :

```text
(weights)       ≠ formula
(3D images)     ≠ formula
train/validation/test ≠ code
Data preprocessing ≠ code
```

Ces éléments peuvent être des termes techniques, pas des objets visuels à préserver.

---

# 4. Problème P0 : `CONTENTS vii 3` est encore traduit comme bloc

Première unité sélectionnée :

```text
tu_0001
level: block
role: toc_entry
source_text: CONTENTS vii 3
```

C’est mauvais.

Cette unité mélange trois choses différentes :

```text
CONTENTS → titre TOC, traduisible
vii      → numéro de page, exact_preserve
3        → numéro de chapitre décoratif, exact_preserve
```

Le pipeline ne doit jamais envoyer ça comme une phrase au traducteur.

La cause est claire :

```text
p001_block_001_line_001 CONTENTS → toc_title mais exact_preserve / translatable False
p001_block_001_line_002 vii      → toc_page_reference
p001_block_001_line_003 3        → toc_page_reference
```

Comme les lignes ne sont pas sélectionnées, `PAGETRANSLATE` retombe sur le bloc entier.

## Correction

Dans `PAGEPRINT` :

```text
toc_title doit être translatable=True
```

Dans `PAGETRANSLATE` :

```text
si page_role == toc :
    ne jamais sélectionner block brut
```

Sauf cas extrêmement rare où le bloc est une seule entrée complète sans enfants exploitables.

---

# 5. Problème P0 : fusion TOC encore incorrecte

Il reste des fusions fausses.

## Exemple 1

Sélection actuelle :

```text
Putting it all together images
```

Mais la page contient :

```text
Putting it all together
Drawbacks of MLPs for processing images
```

Le système a fusionné :

```text
Putting it all together
+
images
```

alors que `images` appartient à l’entrée précédente protégée à tort :

```text
Drawbacks of MLPs for processing images
```

Donc le système fabrique une unité qui n’existe pas.

## Exemple 2

Sélection actuelle :

```text
A closer look at feature extraction A closer look at classification
```

Mais ce sont deux entrées TOC distinctes :

```text
A closer look at feature extraction
A closer look at classification
```

Elles ne doivent pas être fusionnées.

## Correction

Sur une page `toc`, il faut désactiver le coalescer générique :

```python
if page_role == "toc":
    do_not_use_generic_sentence_coalescer()
```

Ou plus proprement :

```text
coalescer autorisé seulement si PAGEPRINT dit explicitement :
same_toc_entry = true
same_row = true
wrapped_title = true
```

La ponctuation ne suffit pas pour fusionner des entrées de table des matières.

---

# 6. Problème P0 : plusieurs entrées naturelles manquent encore

En comparant avec le contenu réel de la page, il manque encore plusieurs entrées traduisibles.

## Manquantes

```text
Drawbacks of MLPs for processing images
Project: Image classification for color images
Is accuracy the best metric for evaluating a model?
Getting your data ready for training
Splitting your data for train/validation/test
Data preprocessing
```

## Partiellement déformées

```text
CONTENTS
→ devient CONTENTS vii 3

Putting it all together
→ devient Putting it all together images

A closer look at feature extraction
+
A closer look at classification
→ fusionnées en une seule unité

Number of parameters (weights)
→ devient Number of parameters
```

Donc la couverture du texte naturel n’est pas encore complète.

Pour valider `PAGEPRINT + PAGETRANSLATE`, il faut :

```text
0 entrée naturelle manquante
0 entrée naturelle protégée à tort
0 fusion TOC abusive
0 bloc mixte envoyé au traducteur
```

On n’y est pas encore.

---

# 7. Problème structurel : `semantic_system` est vide

Dans `input_data_p001.json` :

```text
semantic_system.semantic_phrases: 0
semantic_system.semantic_groups: 0
```

Donc les `semantic_phrase` visibles dans `PAGETRANSLATE` sont créées par `coalescer.py`, pas par `PAGEPRINT`.

C’est important.

Le contrat idéal était :

```text
PAGEPRINT comprend la page
PAGEPRINT produit les unités sémantiques
PAGETRANSLATE traduit ces unités
```

Actuellement, pour cette page :

```text
PAGEPRINT ne produit pas de semantic_phrase
PAGETRANSLATE tente de reconstruire des semantic_phrase
```

Sur une table des matières, ce n’est pas fiable.

Correction recommandée : pour `page_role = toc`, PAGEPRINT doit produire une vue dédiée :

```json
{
  "toc_entries": [
    {
      "entry_id": "toc_001",
      "section_number": "3.1",
      "title_text": "Image classification using MLP",
      "page_reference": "93",
      "marker": null,
      "title_unit_ids": ["..."],
      "bbox": [...]
    }
  ]
}
```

Puis `PAGETRANSLATE` consomme uniquement :

```text
toc_entries[].title_text
```

---

# 8. Problème de projection : les rôles disparaissent dans `reconstruction_units`

Dans les `translation_units`, le rôle est bien :

```text
role: toc_entry
```

Mais dans `translated_input_data.views.reconstruction_units`, les rôles apparaissent comme :

```text
role: None
```

C’est dangereux pour le reconstructeur.

Le reconstructeur doit savoir si une unité est :

```text
toc_title
toc_entry
toc_page_reference
toc_bullet_marker
toc_section_number
```

Même si on ne traduit que `toc_entry`, la reconstruction a besoin du rôle pour gérer :

```text
style
alignement
page reference
leader dots éventuels
positionnement
ancrage
préservation des numéros
```

Correction dans `projection.py` :

```python
"role": item.get("role"),
"object_type": item.get("object_type"),
"semantic_kind": item.get("semantic_kind"),
"page_role": item.get("context", {}).get("page_role")
```

à ajouter dans `_direct_reconstruction_unit()` et `_semantic_reconstruction_unit()`.

---

# 9. Problème d’audit : le résumé dit “ok”, mais fonctionnellement ce n’est pas OK

Le fichier `README_AUDIT_P001.md` contient seulement :

```text
ok
```

C’est insuffisant et trompeur.

L’audit devrait conclure :

```text
KO fonctionnel
```

Même si la validation JSON est correcte.

La validation actuelle dit :

```text
validation.valid = True
error_count = 0
```

Mais cela veut seulement dire :

```text
le JSON respecte le schéma
```

Pas :

```text
la sélection est bonne
```

Il faut séparer :

```text
schema_valid = true
functional_valid = false
```

Dans cette page, le verdict fonctionnel devrait être :

```json
{
  "functional_verdict": "KO",
  "critical_issues": [
    "natural_text_marked_protected",
    "toc_block_selected",
    "wrong_toc_coalescence",
    "missing_toc_entries"
  ]
}
```

---

# 10. Problème de performance

L’audit indique :

```text
duration_s: 130.144
page_understanding: 129.372
pageprint: 0.514
```

Donc la construction `PAGEPRINT` elle-même est rapide, mais la compréhension amont prend plus de deux minutes pour une page.

Ce n’est pas bloquant pour la qualité conceptuelle, mais pour un pipeline réel il faudra comprendre ce qui consomme 129 secondes :

```text
LLM ?
OCR ?
page understanding ?
PDF native extraction ?
post-traitement ?
```

Pour l’instant, ce n’est pas la priorité absolue, mais c’est à surveiller.

---

# 11. Comparaison avec l’état précédent

La v3 est meilleure que la version précédente sur :

```text
rôles TOC
baisse des faux protected_visual
suppression partielle des bullets dans le texte envoyé
meilleure sélection phrase/semantic_phrase
meilleure lisibilité de l’audit visuel
```

Mais elle n’a pas encore corrigé les points décisifs :

```text
faux background_only sur texte naturel
fusion TOC abusive
bloc mixte CONTENTS vii 3
absence de toc_entries structurés
perte d’entrées naturelles
```

Donc on a progressé, mais le test n’est pas encore réussi.

---

# 12. Corrections prioritaires avant le moteur de traduction

## P0.1 — Forcer la politique TOC dans PAGEPRINT

Dans `unit_factory.py`, `_apply_toc_policy()` doit faire plus que poser le rôle.

Pour :

```text
toc_entry
toc_title
```

il faut forcer :

```python
unit["policy"].update({
    "translatable": True,
    "translation_strategy": "layout_constrained",
    "render_policy": "anchored_text",
    "coverage_required": "strict",
    "preserve_exact_text": False,
    "preserve_visual": False,
    "unit_type": toc_role,
    "protected_visual": False,
    "skip_translation": False,
    "skip_text_reconstruction": False,
})
unit["understanding"].pop("protected_visual", None)
unit["constraints"]["skip_translation"] = False
unit["constraints"]["skip_text_reconstruction"] = False
```

Pour :

```text
toc_page_reference
toc_section_number
toc_bullet_marker
```

il faut :

```python
translatable = False
translation_strategy = "exact_preserve"
render_policy = "anchored_text"
```

## P0.2 — Interdire `protected_visual_detector` sur texte TOC naturel

Dans `policy_compiler.py`, avant `_looks_like_protected_visual_unit()` :

```python
if page_role == "toc" and role in {"toc_entry", "toc_title"}:
    do_not_apply_text_protected_visual_detector
```

Une région spéciale peut encore gagner, mais seulement si elle est réellement confirmée comme image/formule/code.

## P0.3 — Filtrer les fausses régions spéciales sur TOC

Dans `PageRegionDetectBuilder` ou juste après détection :

```text
si page_role == toc
et object_type in {formula, code}
et le texte recouvert contient plusieurs mots naturels
alors dégrader en body_region ou ignorer
```

Exemples à ignorer :

```text
(weights)
(3D images)
Splitting your data for train/validation/test
Data preprocessing
```

## P0.4 — Ne jamais sélectionner `block` brut dans une TOC

Dans `selector.py` :

```python
if page_role == "toc":
    allowed_levels = {"phrase", "line"}
    block = forbidden
```

Surtout quand le bloc contient des enfants.

## P0.5 — Désactiver le coalescer générique sur TOC

Dans `builder.py` ou `coalescer.py` :

```python
if translation_profile["page_role"] == "toc":
    skip coalesce_translation_units()
```

Ou alors coalescer uniquement les unités explicitement marquées comme fragments d’une même entrée TOC.

## P0.6 — Ajouter `views.toc_entries`

C’est la meilleure correction structurelle.

PAGEPRINT devrait produire :

```json
{
  "entry_id": "toc_003",
  "section_number": "3.1",
  "marker": null,
  "title_text": "Image classification using MLP",
  "page_reference": "93",
  "title_unit_ids": ["..."],
  "preserve_unit_ids": ["..."],
  "bbox": [...]
}
```

Puis PAGETRANSLATE consomme :

```text
views.toc_entries[].title_text
```

Pas les blocs/lignes bruts.

---

# 13. Peut-on avancer sur le moteur de traduction ?

## Pour un test global sérieux : non

Pas encore.

Si tu branches maintenant un vrai moteur, il va traduire des unités fausses comme :

```text
CONTENTS vii 3
Putting it all together images
A closer look at feature extraction A closer look at classification
```

Et il ne verra même pas :

```text
Project: Image classification for color images
Getting your data ready for training
Splitting your data for train/validation/test
Data preprocessing
```

Donc le test du moteur sera pollué.

## Pour un pré-test limité : oui

Tu peux commencer à tester le moteur uniquement sur les unités propres, par exemple :

```text
Convolutional neural networks
Image classification using MLP
Input layer
Hidden layers
Output layer
CNN architecture
The big picture
Basic components of a CNN
Convolutional layers
Pooling layers or subsampling
Fully connected layers
Image classification using CNNs
Adding dropout layers to avoid overfitting
Where does the dropout layer go in the CNN architecture?
Convolution over color images (3D images)
How do we perform a convolution on a color image?
Structuring DL projects and hyperparameter tuning
Defining performance metrics
Confusion matrix
Precision and recall
F-score
Designing a baseline model
Evaluating the model and interpreting its performance
Diagnosing overfitting and underfitting
```

Mais ce sera un test du moteur, pas une validation du pipeline.

---

# 14. Critère de passage avant vrai moteur

Pour dire “oui, on passe au moteur de traduction”, il faut que l’audit affiche :

```text
protected_visual_false_positive_count = 0
natural_text_marked_background_only = 0
toc_block_translation_count = 0
wrong_toc_coalescence_count = 0
missing_toc_entry_count = 0
selected_word_char_count = 0
protected_sent_to_translation_count = 0
reconstruction_units_have_roles = true
```

Sur cette v3 :

```text
protected_visual_false_positive_count > 0
natural_text_marked_background_only > 0
toc_block_translation_count = 1
wrong_toc_coalescence_count >= 2
missing_toc_entry_count >= 6
reconstruction_units_have_roles = false
```

Donc le feu est encore orange/rouge.

---

# Conclusion

La v3 est une vraie amélioration. Le pipeline est plus propre, les rôles TOC arrivent, les bullets sont mieux traités, et la sélection est plus proche de ce qu’il faut.

Mais le test n’est pas encore réussi.

Le problème principal restant est :

```text
PAGEPRINT laisse encore des signaux protected_visual/background_only contaminer des entrées TOC naturelles.
```

Le deuxième problème est :

```text
PAGETRANSLATE applique encore une fusion générique à une table des matières.
```

Décision recommandée :

```text
1. Corriger les P0 PAGEPRINT/PAGETRANSLATE ci-dessus.
2. Relancer le même audit v4 sur cette page.
3. Si v4 donne zéro perte / zéro fusion abusive / zéro faux protected natural text,
   alors passer au moteur de traduction.
```

Tu peux avancer en parallèle sur le moteur, mais **pas encore l’utiliser comme test de validation globale**.

