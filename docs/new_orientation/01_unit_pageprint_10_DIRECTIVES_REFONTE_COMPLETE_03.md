Oui. Voici le **plan de continuation de la refonte**, orienté exécution Codex. L’objectif est de passer de `rev_02` à une `rev_03` où `PAGETRANSLATE` ne dépend plus fonctionnellement de `selector/coalescer` parce que `PAGEPRINT` produit réellement un `translation_plan` exploitable.

---

# PLAN DE CONTINUATION — `rev_02` → `rev_03`

## Objectif principal

```text
Faire de PAGEPRINT la source réelle du translation_plan.
Faire de PAGETRANSLATE un consommateur du translation_plan.
Empêcher les faux “functional_status: ok” quand le plan manque ou est vide.
```

La priorité n’est pas d’ajouter encore des modules. La priorité est de **brancher correctement les modules déjà créés**.

---

# SPRINT 0 — Stabiliser la base et établir les métriques

## But

Avant de modifier profondément, il faut rendre mesurable le problème actuel.

## Todo

* [ ] Ajouter dans `pagetranslate/builder.py` un champ debug obligatoire :

```json
{
  "selection_mode": "translation_plan | fallback_selector",
  "translation_plan_input_count": 0,
  "fallback_selector_used": true,
  "generic_coalescer_used": true
}
```

* [ ] Ajouter dans le résultat global `pagetranslate_result` :

```json
{
  "debug": {
    "selection_mode": "...",
    "fallback_selector_used": false,
    "generic_coalescer_used": false
  }
}
```

* [ ] Ajouter dans `pageprint/views` ou `quality.metrics` :

```json
{
  "translation_plan_count": 0,
  "preservation_plan_count": 0,
  "reconstruction_plan_count": 0,
  "logical_unit_count": 0,
  "semantic_segment_count": 0
}
```

* [ ] Ajouter dans l’audit compact :

```json
{
  "pageprint_translation_plan_count": 0,
  "pagetranslate_selection_mode": "...",
  "fallback_selector_used": true,
  "generic_coalescer_used": true
}
```

## Critère d’acceptation

Sur chaque page auditée, on doit savoir clairement :

```text
PAGEPRINT a-t-il produit un translation_plan ?
PAGETRANSLATE l’a-t-il utilisé ?
Le fallback selector/coalescer a-t-il été utilisé ?
```

---

# SPRINT 1 — Brancher `logical_structures` dans `semantic_builder.py`

## But

Actuellement, les `structure_builders` existent, mais `semantic_builder.py` ne les consomme pas assez. Il faut faire de `logical_structures` la source principale des `translation_segments`.

## Fichiers à modifier

```text
pageprint/semantic_builder.py
pageprint/view_compiler.py
pageprint/builder.py
```

## Todo

* [ ] Dans `semantic_builder.py`, créer une fonction principale :

```python
def build_semantic_system_from_logical_structures(input_data: dict) -> dict:
    ...
```

* [ ] Cette fonction doit lire :

```python
input_data.get("logical_structures", {})
```

ou l’emplacement réel actuel où `toc_entries`, `tables`, `index_entries`, etc. sont stockés.

* [ ] Produire des `translation_segments` depuis :

```text
toc_entries
index_entries
tables.cells
captions
list_items
author_entries
body_paragraphs
```

* [ ] Garder le mode visuel/heuristique seulement comme fallback :

```python
if logical_translation_segments:
    use them
else:
    fallback_to_visual_units()
```

* [ ] Ajouter dans chaque `translation_segment` :

```json
{
  "translation_segment_id": "...",
  "logical_unit_id": "...",
  "source_unit_ids": [...],
  "source_text": "...",
  "role": "...",
  "object_type": "...",
  "semantic_kind": "...",
  "translation_mode": "translate",
  "protected_tokens": [...],
  "render_target": {...}
}
```

## Critère d’acceptation

Pour une page TOC, index ou table, les `translation_segments` doivent venir des structures logiques, pas des unités visuelles brutes.

---

# SPRINT 2 — Corriger le pipeline TOC en premier

## But

La TOC est le meilleur test de refonte, car elle révèle immédiatement si le système traduit encore des blocs visuels ou s’il comprend les structures documentaires.

## Fichiers à modifier

```text
pageprint/structure_builders/toc_builder.py
pageprint/semantic_builder.py
pageprint/view_compiler.py
pageprint/functional_validator.py
```

## Todo `toc_builder.py`

* [ ] Ne construire les TOC entries qu’à partir de lignes/rows, pas depuis `block + line + phrase + span` en même temps.

* [ ] Ajouter une règle de granularité :

```text
si une line exploitable existe, ignorer block pour construire toc_entry
si une phrase exploitable existe dans la line, ne pas créer une deuxième entrée depuis le span
```

* [ ] Parser les formes suivantes :

```text
3.1 Image classification using MLP 93
3.1 Image classification using MLP ..... 93
■ Hidden layers 94
Hidden layers 94
CONTENTS
```

* [ ] Produire :

```json
{
  "logical_unit_id": "toc_entry_001",
  "type": "toc_entry",
  "section_number": "3.1",
  "marker": "■",
  "title_text": "Image classification using MLP",
  "page_reference": "93",
  "source_unit_ids": [...],
  "title_unit_ids": [...],
  "preserve_unit_ids": [...],
  "bbox": [...]
}
```

* [ ] Ne jamais produire une entrée TOC brute comme :

```text
CONTENTS vii 3
```

## Todo `semantic_builder.py`

* [ ] Pour chaque `toc_entry`, créer un segment uniquement pour `title_text`.

* [ ] Ne pas traduire :

```text
section_number
page_reference
marker
```

* [ ] Créer un segment :

```json
{
  "role": "toc_entry_title",
  "object_type": "natural_text",
  "semantic_kind": "toc_entry_title",
  "source_text": "Image classification using MLP",
  "protected_tokens": ["MLP"]
}
```

## Todo `view_compiler.py`

* [ ] Transformer ce segment en `views.translation_plan`.

## Critère d’acceptation

Sur la page TOC test :

```text
translation_plan_count > 0
selection_mode = translation_plan
fallback_selector_used = false
generic_coalescer_used = false
aucun "CONTENTS vii 3" dans translation_plan
aucun numéro de page dans source_text à traduire
```

---

# SPRINT 3 — Corriger `functional_validator.py`

## But

Le validateur doit empêcher les faux `functional_status: ok`.

## Fichier à modifier

```text
pageprint/functional_validator.py
```

## Todo

Ajouter des erreurs bloquantes :

* [ ] `translation_plan_empty_but_translatable_text_exists`

* [ ] `toc_entries_exist_but_no_translation_plan`

* [ ] `index_entries_exist_but_no_translation_plan`

* [ ] `tables_exist_but_no_cell_translation_plan`

* [ ] `logical_units_exist_but_no_translation_segments`

* [ ] `translation_plan_item_missing_role`

* [ ] `translation_plan_item_missing_object_type`

* [ ] `translation_plan_item_missing_render_target`

* [ ] `word_or_char_in_translation_plan`

* [ ] `mixed_block_in_translation_plan`

* [ ] `fallback_required_after_pageprint`

## Règle

Si `PAGEPRINT` sort une page avec du texte naturel traduisible mais sans `translation_plan`, alors :

```json
{
  "functional_status": "ko"
}
```

## Critère d’acceptation

Le cas suivant doit être KO :

```text
page_role = toc
toc_entries_count > 0
translation_plan_count = 0
```

Le cas suivant doit être KO :

```text
body text page
semantic_system empty
translation_plan empty
```

---

# SPRINT 4 — Rendre `view_compiler.py` strict

## But

`view_compiler.py` doit compiler les plans aval et refuser les unités incomplètes.

## Fichier à modifier

```text
pageprint/view_compiler.py
```

## Todo

* [ ] Compiler `views.translation_plan`.

* [ ] Compiler `views.preservation_plan`.

* [ ] Compiler `views.reconstruction_plan`.

* [ ] Compiler `views.exclusion_plan`.

* [ ] Pour chaque item de `translation_plan`, exiger :

```text
translation_unit_id
source_unit_ids
source_text
role
object_type
semantic_kind
translation_mode
render_target
qa_requirements
```

* [ ] Si un champ manque, ne pas ignorer silencieusement. Ajouter une erreur ou warning fonctionnel.

* [ ] Ne jamais inclure :

```text
word
char
page_reference
section_number
command_name
path
file_name
watermark
publisher_mark
```

comme texte normal à traduire.

## Critère d’acceptation

Un `translation_plan` valide doit avoir :

```text
role_missing = 0
object_type_missing = 0
render_target_missing = 0
word_char_items = 0
```

---

# SPRINT 5 — Corriger `preservation_compiler.py`

## But

Empêcher les protections excessives du type `CONTENTS` traité comme acronyme.

## Fichier à modifier

```text
pageprint/preservation_compiler.py
```

## Todo

* [ ] Modifier la détection acronymes.

* [ ] Ne pas protéger automatiquement un mot en majuscules si son rôle est :

```text
title
toc_title
section_heading
chapter_heading
body_heading
```

* [ ] Protéger les acronymes seulement si :

```text
- le terme est dans un glossaire ;
- ou contient plusieurs majuscules techniques courtes : MLP, CNN, OCR, SQL ;
- ou apparaît dans un contexte technique confirmé.
```

* [ ] Ajouter règles :

```text
CONTENTS → traduisible si toc_title
SUMMARY → traduisible si title
INTRODUCTION → traduisible si heading/title
CONCLUSION → traduisible si heading/title
```

* [ ] Préserver :

```text
MLP
CNN
SQL
ReLU
Softmax
API
OCR
```

## Critère d’acceptation

Dans une TOC :

```text
CONTENTS ne doit pas être protect_token_inside_translation par défaut.
MLP doit être protégé comme token technique.
```

---

# SPRINT 6 — Rendre PAGETRANSLATE strict sur le fallback

## But

Le fallback doit rester disponible pour compatibilité, mais il ne doit plus masquer les échecs de PAGEPRINT.

## Fichiers à modifier

```text
pagetranslate/builder.py
pagetranslate/translation_plan_reader.py
pagetranslate/functional_validator.py
```

## Todo

* [ ] Si `input_data["views"]["translation_plan"]` existe, utiliser exclusivement ce plan.

* [ ] Si le plan existe mais est vide, ne pas tomber silencieusement en fallback.

Faire :

```python
if "translation_plan" in views and not views["translation_plan"]:
    selection_mode = "translation_plan_empty"
    add functional error
```

Pas :

```python
if not plan:
    fallback_selector()
```

* [ ] Le fallback ne doit s’activer que si le champ `translation_plan` est absent.

```python
if "translation_plan" not in views:
    fallback_selector()
```

* [ ] Ajouter une option explicite :

```python
allow_fallback=True
```

et permettre :

```python
allow_fallback=False
```

pour les tests stricts.

## Critère d’acceptation

Cas 1 :

```text
views.translation_plan existe et contient 10 items
→ selection_mode = translation_plan
```

Cas 2 :

```text
views.translation_plan existe mais vide
→ selection_mode = translation_plan_empty
→ functional_status = ko
→ aucun fallback
```

Cas 3 :

```text
views.translation_plan absent
→ fallback_selector autorisé si allow_fallback=True
```

---

# SPRINT 7 — Corriger `projection.py`

## But

Les unités de reconstruction doivent conserver toute l’intelligence documentaire.

## Fichier à modifier

```text
pagetranslate/projection.py
```

## Todo

* [ ] Chaque `reconstruction_unit` doit conserver :

```text
role
object_type
semantic_kind
page_role
logical_unit_id
source_unit_ids
translation_unit_id
render_target
render_contract
preservation_mode
```

* [ ] Ne jamais produire :

```json
{
  "role": null
}
```

pour une unité issue de `translation_plan`.

* [ ] Si rôle absent, ajouter erreur dans `pagetranslate/functional_validator.py`.

## Critère d’acceptation

Après traduction :

```text
reconstruction_units_missing_roles = 0
reconstruction_units_missing_render_target = 0
```

---

# SPRINT 8 — Ajouter les tests réels

## But

Le rapport ne doit plus annoncer des tests absents.

## Dossiers à créer

```text
tests/pageprint/
tests/pagetranslate/
tests/functional/
```

## Tests minimum

### `tests/pageprint/test_body_translation_plan.py`

* [ ] Une page body avec une phrase naturelle produit un `translation_plan` non vide.

### `tests/pageprint/test_toc_translation_plan.py`

* [ ] Une TOC avec `3.1 Image classification using MLP 93` produit :

```text
source_text = Image classification using MLP
protected_tokens = ["MLP"]
page_reference = 93 dans preservation_plan
```

### `tests/pageprint/test_no_role_none_translation_plan.py`

* [ ] Aucun item de `translation_plan` n’a `role=None`.

### `tests/pageprint/test_no_word_char_translation_plan.py`

* [ ] Aucun item de niveau `word` ou `char` dans `translation_plan`.

### `tests/pagetranslate/test_translation_plan_mode.py`

* [ ] Si `translation_plan` existe, `PAGETRANSLATE` utilise `selection_mode=translation_plan`.

### `tests/pagetranslate/test_no_fallback_when_empty_plan.py`

* [ ] Si `translation_plan=[]`, pas de fallback silencieux.

### `tests/pagetranslate/test_projection_keeps_roles.py`

* [ ] Les `reconstruction_units` gardent les rôles.

## Critère d’acceptation

La commande suivante doit lancer de vrais tests :

```bash
python3 -m pytest -q
```

Elle ne doit plus répondre :

```text
no tests ran
```

---

# SPRINT 9 — Audit strict multi-pages

## But

Valider que la refonte fonctionne sur plusieurs types de pages.

## Fichier / outil à modifier

```text
tools/run_functional_audit.py
```

## Todo

* [ ] Ajouter métriques globales :

```json
{
  "pages_total": 0,
  "pages_with_translation_plan": 0,
  "pages_using_translation_plan": 0,
  "pages_using_fallback": 0,
  "translation_plan_items": 0,
  "role_none_translation_items": 0,
  "word_char_translation_items": 0,
  "reconstruction_units_missing_roles": 0,
  "functional_status": "ok|ko"
}
```

* [ ] Si `pages_using_fallback > 0`, alors audit KO sauf si page explicitement legacy.

* [ ] Si `role_none_translation_items > 0`, audit KO.

* [ ] Si `word_char_translation_items > 0`, audit KO.

* [ ] Si `reconstruction_units_missing_roles > 0`, audit KO.

## Critère d’acceptation

Sur un lot de pages variées :

```text
pages_using_fallback = 0
role_none_translation_items = 0
word_char_translation_items = 0
reconstruction_units_missing_roles = 0
functional_status = ok
```

---

# SPRINT 10 — Rapport Codex final

## But

Codex doit produire un rapport qui prouve ce qui est fait.

## Fichier à créer / mettre à jour

```text
REFACTOR_EVALUATION.md
```

## Contenu obligatoire

* [ ] Fichiers créés.

* [ ] Fichiers modifiés.

* [ ] Chemin exact du pipeline :

```text
PAGEPRINT logical_structures
→ semantic_builder.translation_segments
→ view_compiler.translation_plan
→ PAGETRANSLATE translation_plan_reader
→ projection.reconstruction_units
```

* [ ] Résultats de compilation :

```bash
python3 -m py_compile pageprint/*.py pagetranslate/*.py
```

* [ ] Résultats tests :

```bash
python3 -m pytest -q
```

* [ ] Résultats audit fonctionnel.

* [ ] Limites restantes.

## Critère d’acceptation

Le rapport ne doit pas déclarer des tests qui n’existent pas.

---

# Ordre d’exécution recommandé

Ne pas faire tout en parallèle. Exécuter dans cet ordre :

```text
1. Sprint 0 — métriques/debug
2. Sprint 1 — brancher logical_structures → semantic_builder
3. Sprint 2 — TOC complet
4. Sprint 3 — functional_validator strict
5. Sprint 4 — view_compiler strict
6. Sprint 6 — PAGETRANSLATE fallback strict
7. Sprint 7 — projection enrichie
8. Sprint 8 — tests réels
9. Sprint 9 — audit multi-pages
10. Sprint 10 — rapport final
```

Les sprints 5, table/index/caption/list/code peuvent venir ensuite, mais le TOC doit d’abord prouver que la chaîne est correcte.

---

# Critère minimal de succès pour `rev_03`

`rev_03` est acceptable seulement si :

```text
PAGEPRINT produit views.translation_plan pour une page body.
PAGEPRINT produit views.translation_plan pour une page TOC.
PAGETRANSLATE utilise selection_mode=translation_plan.
PAGETRANSLATE ne tombe pas en fallback si translation_plan existe mais est vide.
Aucun item translation_plan avec role=None.
Aucun word/char dans translation_plan.
Aucun bloc mixte type CONTENTS vii 3 dans translation_plan.
reconstruction_units gardent role/object_type/semantic_kind.
pytest lance de vrais tests.
functional_status peut être ko même si schema_status est ok.
```

---

# Instruction courte à donner à Codex

```text
La refonte rev_02 a créé les bons modules, mais ils ne sont pas encore suffisamment branchés.

Objectif rev_03 :
faire de logical_structures la source principale de semantic_builder,
faire de semantic_builder la source de translation_segments,
faire de view_compiler la source stricte de views.translation_plan,
faire de PAGETRANSLATE un lecteur strict de translation_plan,
interdire le fallback silencieux,
ajouter des tests réels et un functional_validator sévère.

Ne pas ajouter de nouveaux modules tant que ceux de rev_02 ne sont pas correctement branchés.
```

C’est le plan prioritaire. Après `rev_03`, on pourra attaquer proprement les structures plus lourdes : tables, index, captions, code, formules, figures, document context.

