# Verdict

La refonte est **engagée**, mais elle n’est **pas terminée fonctionnellement**.

Le point central est celui-ci :

```text
PAGETRANSLATE sait maintenant lire views.translation_plan.
Donc côté PAGETRANSLATE, le changement de philosophie est amorcé.

Mais PAGEPRINT ne produit pas encore un translation_plan fiable dans tous les cas.
Donc le pipeline peut encore retomber sur selector.py + coalescer.py.
```

Conclusion stricte :

```text
Refonte architecturale : partiellement faite.
Refonte fonctionnelle : non validée.
Critère “PAGETRANSLATE ne dépend plus principalement de selector/coalescer” : pas encore garanti.
```

J’ai inspecté `rev_02.zip`, vérifié la structure, lu les fichiers clés, et lancé une compilation Python. La compilation passe. Mais il y a des incohérences importantes entre le rapport `REFACTOR_EVALUATION.md`, le code réel et le comportement observable.

---

# 1. Ce qui est réellement fait

## 1.1 Les nouveaux modules existent

La refonte a bien ajouté :

```text
pageprint/role_resolver.py
pageprint/semantic_builder.py
pageprint/preservation_compiler.py
pageprint/view_compiler.py
pageprint/functional_validator.py
pageprint/graph_query.py
pageprint/evidence/
pageprint/structure_builders/
pagetranslate/translation_plan_reader.py
pagetranslate/functional_validator.py
pipelines/ocr_router.py
pipelines/document_context.py
tools/run_functional_audit.py
```

C’est conforme aux directives structurelles.

## 1.2 `PAGETRANSLATE` utilise bien `translation_plan` si présent

Dans `pagetranslate/builder.py`, le comportement est maintenant :

```python
plan = input_data.get("views", {}).get("translation_plan") or []

if plan:
    units = read_translation_plan(input_data)
    selection_mode = "translation_plan"
else:
    units = select_translation_units(input_data)
    units = annotate_sentence_boundaries(units)
    units = coalesce_translation_units(units)
    selection_mode = "fallback_selector"
```

Donc oui, **le chemin prioritaire est maintenant `translation_plan`**.

C’est un vrai progrès.

## 1.3 `selector.py` et `coalescer.py` sont devenus fallback côté code

Sur le papier, oui :

```text
translation_plan présent → pas de selector/coalescer
translation_plan absent → fallback selector/coalescer
```

Donc la phrase :

```text
Tant que PAGETRANSLATE dépend encore principalement de selector.py + coalescer.py,
la refonte n’est pas terminée.
```

est maintenant presque satisfaite **à condition que `PAGEPRINT` produise toujours un `translation_plan` correct**.

Et c’est justement là que ça casse encore.

---

# 2. Ce qui n’est pas encore bon

## 2.1 PAGEPRINT ne produit pas toujours `translation_plan`

J’ai fait un test synthétique simple.

### Page body simple

Avec une phrase normale :

```text
This is a real sentence that should be translated.
```

`PAGEPRINT` produit bien :

```text
translation_plan length = 1
selection_mode = translation_plan
```

Donc pour une page body simple, ça fonctionne.

### Page TOC simple

Avec une page `toc` contenant :

```text
CONTENTS
3.1 Image classification using MLP 93
■Hidden layers 94
```

`PAGEPRINT` produit :

```text
semantic_phrases: 0
translation_segments: 0
translation_plan: []
```

Donc `PAGETRANSLATE` retombe sur :

```text
selection_mode = fallback_selector
```

C’est un KO sur le critère central.

---

# 3. Problème fondamental : les `structure_builders` existent, mais ne pilotent pas encore la traduction

Les modules existent :

```text
toc_builder.py
index_builder.py
table_builder.py
caption_builder.py
list_builder.py
code_builder.py
formula_builder.py
```

Mais dans l’implémentation actuelle, ils construisent surtout des objets dans :

```text
logical_structures
```

Puis `semantic_builder.py` ne les exploite pas vraiment pour produire les `translation_segments`.

Actuellement, `semantic_builder.py` fait surtout :

```text
prendre les unités texte visuelles
filtrer par rôle
construire semantic_phrases
construire translation_segments
```

Il ne fait pas encore :

```text
toc_entries → translation_segments propres
index_entries → translation_segments propres
table_cells → translation_segments propres
caption_parts → translation_segments propres
list_items → translation_segments propres
```

Donc la nouvelle architecture est présente, mais **pas encore branchée fonctionnellement**.

C’est le problème principal.

---

# 4. Problème critique : `toc_builder.py` produit des doublons

Sur mon test TOC, `toc_builder.py` a produit des `toc_entries` pour :

```text
block
line
phrase
span
```

Donc pour la même information, il crée plusieurs entrées logiques :

```text
toc_entry depuis le block entier
toc_entry depuis la line
toc_entry depuis la phrase
toc_entry depuis le span
```

C’est exactement ce qu’on voulait éviter.

Exemple mauvais :

```text
CONTENTS 3.1 Image classification using MLP 93 ■Hidden layers 94
```

est devenu une entrée TOC au niveau bloc.

Cela veut dire que le `toc_builder` n’applique pas encore une règle de granularité :

```text
TOC = construire à partir des lignes ou rows natives, pas depuis block+line+phrase+span.
```

Correction nécessaire :

```text
toc_builder doit ignorer block si ses lignes/phrases existent.
toc_builder doit ignorer span si phrase existe.
toc_builder doit produire une seule logical_unit par ligne/row TOC réelle.
```

---

# 5. Problème critique : le TOC n’est pas découpé en section/title/page

Pour :

```text
3.1 Image classification using MLP 93
```

le builder devrait produire :

```json
{
  "section_number": "3.1",
  "title_text": "Image classification using MLP",
  "page_reference": "93"
}
```

Mais dans le test, il garde :

```text
title_text = "3.1 Image classification using MLP 93"
page_reference = null
section_number = null
```

La regex actuelle de `toc_builder.py` exige :

```text
points leaders "....."
ou plusieurs espaces
```

Mais beaucoup de TOC PDF extraites n’ont qu’un espace normal entre le titre et le numéro de page.

Correction :

```text
toc_builder doit parser les lignes TOC avec plusieurs stratégies :
1. section + title + page avec leaders
2. section + title + page sans leaders
3. bullet + title + page
4. title + page
5. title seul si page ref absente
```

---

# 6. Problème critique : `semantic_builder.py` ignore les rôles `toc_entry`

Dans `semantic_builder.py`, les rôles traduisibles incluent :

```text
toc_entry_title
```

mais pas :

```text
toc_entry
```

Or `role_resolver.py` classe la plupart des textes TOC comme :

```text
toc_entry
```

Résultat :

```text
aucun semantic_phrase
aucun translation_segment
aucun translation_plan
fallback selector
```

Il y a deux corrections possibles.

## Correction courte

Ajouter `toc_entry` aux rôles traduisibles, mais ce n’est pas idéal, car ça risque d’envoyer les numéros de page au traducteur.

## Correction propre

Faire en sorte que :

```text
toc_builder.py produit title_text propre
semantic_builder.py consomme toc_entries[].title_text
view_compiler.py produit translation_plan depuis title_text seulement
```

Donc il ne faut pas traduire `toc_entry` brut. Il faut traduire `toc_entry_title`.

---

# 7. Problème critique : `functional_validator.py` est trop permissif

Dans mon test TOC, le pipeline a produit :

```text
translation_plan = []
semantic_system = vide
logical_structures.toc_entries = non vide
functional_status = ok
```

C’est faux.

Une page TOC avec des `toc_entries` contenant du texte naturel doit produire un `translation_plan`.

Le validateur doit signaler :

```text
toc_entries_exist_but_translation_plan_empty
translation_plan_empty_with_translatable_logical_units
semantic_system_empty_with_logical_text_units
```

Actuellement, il ne le fait pas.

Donc l’audit peut encore dire :

```text
functional_status = ok
```

alors que `PAGETRANSLATE` va retomber sur fallback.

C’est une erreur de validation.

---

# 8. Problème critique : le rapport `REFACTOR_EVALUATION.md` est inexact

Le rapport affirme :

```text
tests/pageprint/test_refactor_contract.py
tests/pagetranslate/test_translation_plan_mode.py
```

Mais dans l’archive inspectée, je ne trouve pas de dossier `tests/`.

J’ai lancé :

```bash
python3 -m pytest -q
```

Résultat :

```text
no tests ran
```

Donc les tests annoncés dans `REFACTOR_EVALUATION.md` ne sont pas présents dans l’archive.

Ce n’est pas un détail. Cela veut dire que la refonte n’a pas encore de garde-fous automatiques.

---

# 9. Problème : les plans existent, mais restent trop dérivés du système visuel

`view_compiler.py` construit `translation_plan` à partir de :

```text
semantic_system.translation_segments
```

Mais comme `semantic_system` lui-même est construit depuis les unités visuelles, on reste proche de l’ancien modèle.

Le vrai modèle cible devrait être :

```text
logical_structures
→ semantic_builder
→ translation_segments
→ translation_plan
```

Actuellement :

```text
visual units
→ semantic_builder
→ translation_segments
→ translation_plan
```

et `logical_structures` sont surtout attachées comme information parallèle.

Ce n’est pas encore un vrai compilateur documentaire.

---

# 10. Problème : `preservation_compiler.py` protège trop les majuscules

Dans mon test, `CONTENTS` est traité avec :

```text
preservation_mode = protect_token_inside_translation
```

probablement à cause de la regex acronymes :

```python
ACRONYM_RE = r"^[A-Z0-9][A-Z0-9&./+-]{1,12}$"
```

Cela risque de reproduire une ancienne erreur :

```text
INTRODUCTION
BACKGROUND
CONTENTS
SUMMARY
CONCLUSION
```

peuvent être pris pour des acronymes.

Correction :

```text
Un mot majuscule long dans un rôle title/toc_title/section_heading
ne doit pas être protégé comme acronyme.
```

Règle :

```python
if role in {"title", "section_heading", "toc_title", "toc_entry_title"}:
    do_not_auto_protect_uppercase_word_as_acronym
```

Sauf si le terme est dans un glossaire d’acronymes confirmés.

---

# 11. Problème : les builders sont encore trop heuristiques / contractuels

Ils existent, mais plusieurs sont encore très faibles.

## `table_builder.py`

Il ne détecte pas vraiment les tables. Il prend seulement :

```text
units avec role table_
ou level == cell
```

Donc si PAGEPRINT n’a pas déjà des cellules, il ne construit pas de table.

Il manque encore la vraie détection :

```text
alignement colonnes
lignes vectorielles
régularité x/y
grille
fonds alternés
headers
native PDF words alignés
```

## `index_builder.py`

Il détecte surtout :

```text
head, refs
```

mais ne reconstruit pas encore proprement les sous-entrées, indentations, fonctions techniques, références multiples.

## `figure_builder.py`

Il retourne encore :

```python
return []
```

Donc figure/diagramme n’est pas encore traité comme structure logique réelle.

---

# 12. Réponse précise à ta question

## Est-ce que la refonte est faite ?

Réponse :

```text
Non, pas complètement.
```

## Est-ce que le cœur côté PAGETRANSLATE est modifié ?

Réponse :

```text
Oui, partiellement.
```

`PAGETRANSLATE` donne priorité à `translation_plan`. C’est bien.

## Est-ce que PAGETRANSLATE ne dépend plus de selector/coalescer ?

Réponse :

```text
Seulement si PAGEPRINT produit translation_plan.
```

Or `PAGEPRINT` ne le produit pas encore de façon fiable, notamment sur TOC.

Donc :

```text
Le code a déplacé la dépendance.
Mais il ne l’a pas éliminée fonctionnellement.
```

Avant :

```text
PAGETRANSLATE dépendait directement de selector/coalescer.
```

Maintenant :

```text
PAGETRANSLATE dépend de PAGEPRINT.translation_plan.
Mais si PAGEPRINT échoue, il dépend encore de selector/coalescer.
```

Et comme PAGEPRINT échoue encore dans certains cas, la refonte n’est pas terminée.

---

# 13. Ce qu’il faut corriger maintenant

## P0.1 — Brancher `logical_structures` dans `semantic_builder.py`

`semantic_builder.py` doit produire des `translation_segments` depuis :

```text
toc_entries
index_entries
tables.cells
captions
list_items
code_blocks
formula_units
author_bios
```

Pas seulement depuis les unités visuelles.

Exemple attendu pour TOC :

```python
for entry in logical_structures["toc_entries"]:
    if entry["title_text"]:
        translation_segments.append({
            "translation_segment_id": ...,
            "logical_unit_id": entry["logical_unit_id"],
            "source_unit_ids": entry["title_unit_ids"],
            "source_text": entry["title_text"],
            "role": "toc_entry_title",
            "object_type": "natural_text",
            "semantic_kind": "toc_entry_title",
            "protected": [entry["section_number"], entry["page_reference"], entry["marker"]],
            "translation_mode": "translate",
        })
```

## P0.2 — Corriger `toc_builder.py`

Règles obligatoires :

```text
1. Construire TOC à partir des lignes/rows, pas block+line+phrase+span.
2. Ignorer les parents si les enfants textuels existent.
3. Parser section_number.
4. Parser page_reference.
5. Séparer bullet marker.
6. Nettoyer title_text.
7. Produire title_unit_ids seulement pour le titre.
8. Produire preserve_unit_ids pour section/page/bullet si disponibles.
```

## P0.3 — Corriger `functional_validator.py`

Ajouter erreurs bloquantes :

```text
translation_plan_empty_but_logical_units_translatable
toc_entries_exist_but_no_translation_segments
toc_entries_exist_but_no_translation_plan
fallback_required_on_pageprint_output
semantic_system_empty_but_page_has_translatable_text
```

Le validateur doit empêcher le faux OK observé.

## P0.4 — Mettre les tests réellement dans l’archive

Créer réellement :

```text
tests/pageprint/test_refactor_contract.py
tests/pagetranslate/test_translation_plan_mode.py
tests/pageprint/test_toc_translation_plan.py
tests/pageprint/test_body_translation_plan.py
tests/pageprint/test_no_role_none_translation_plan.py
```

Et vérifier que :

```bash
python3 -m pytest -q
```

ne retourne pas :

```text
no tests ran
```

## P0.5 — Ajouter un audit obligatoire `fallback_selector_usage`

Après génération d’un audit, il faut obtenir :

```json
{
  "selection_mode": "translation_plan",
  "fallback_selector_usage": 0
}
```

Si une page générée par PAGEPRINT tombe en fallback :

```text
functional_status = ko
```

---

# 14. Critère minimal pour dire “refonte terminée”

La refonte du cœur sera terminée quand, sur un lot multi-pages :

```text
PAGEPRINT.views.translation_plan existe et n’est pas vide pour toutes les pages avec texte traduisible.
PAGETRANSLATE.debug.selection_mode = "translation_plan" partout sauf pages sans texte.
fallback_selector_usage = 0.
generic_coalesced_units = 0.
translation_plan_role_missing = 0.
translation_plan_object_type_missing = 0.
word_char_translation_units = 0.
reconstruction_units_missing_roles = 0.
```

Là, on pourra dire :

```text
PAGETRANSLATE ne dépend plus principalement de selector/coalescer.
```

Pour l’instant, non.

---

# 15. Verdict final

```text
rev_02 est une bonne refonte structurelle.
Mais rev_02 n’est pas encore une refonte fonctionnelle complète.
```

Le meilleur résumé :

```text
Le chemin translation_plan existe.
Mais la production du translation_plan est encore fragile.
Les logical_structures existent.
Mais elles ne pilotent pas encore vraiment semantic_builder.
Les validators existent.
Mais ils valident trop facilement.
Les tests sont annoncés.
Mais ils ne sont pas présents dans l’archive.
```

Donc la prochaine tâche Codex doit être très précise :

```text
Ne pas ajouter de nouveaux modules.
Brancher les modules existants.
Faire de logical_structures la source de translation_segments.
Faire échouer functional_validator si translation_plan manque.
Ajouter les tests réels.
```

Phrase finale :

```text
La refonte est au bon endroit dans le code,
mais elle n’est pas encore au bon endroit dans le comportement.
```

