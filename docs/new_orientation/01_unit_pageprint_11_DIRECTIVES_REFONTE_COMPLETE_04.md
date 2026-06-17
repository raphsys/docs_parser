Oui. On ne doit pas continuer en mode séquentiel lent. Il faut passer en **mode refonte parallèle contrôlée**.

La bonne décision est :

```text
rev_03 est le socle validé.
On ne touche plus au principe translation_plan.
On lance rev_04 en plusieurs chantiers parallèles.
Objectif : rendre PAGEPRINT robuste sur les types documentaires réels,
puis brancher le moteur de traduction sans polluer le test.
```

---

# 1. Décision stratégique

## Décision 1 — Figer le cœur `translation_plan`

Ne plus remettre en cause ceci :

```text
PAGEPRINT → views.translation_plan
PAGETRANSLATE → lit translation_plan
selector/coalescer → fallback seulement
```

C’est acquis.

À partir de maintenant, toute amélioration doit respecter ce contrat.

---

## Décision 2 — Ne pas lancer directement le moteur de traduction global

Pas encore sur tous les documents.

On peut commencer le moteur en parallèle, mais il doit être testé sur :

```text
translation_plan propre
segments validés
tokens protégés
glossaire
mode dry-run / mock / vrai traducteur
```

Pas sur des unités sales.

---

## Décision 3 — Priorité à la robustesse documentaire

Les prochains gros risques sont :

```text
tables
index
figures/diagrammes
body paragraphs
publisher marks / headers / footers
OCR ciblé
document context
```

Donc `rev_04` doit être la version :

```text
structures documentaires robustes + audit multi-pages strict
```

---

# 2. Objectif `rev_04`

`rev_04` doit réussir ceci :

```text
Sur un lot multi-pages varié :
- aucune page traduisible sans translation_plan ;
- aucune utilisation fallback selector/coalescer ;
- aucun role=None dans translation_plan ;
- aucun word/char dans translation_plan ;
- tables reconnues ;
- index reconnu ;
- captions séparées ;
- publisher marks exclus ;
- body paragraphs structurés ;
- reconstruction_units avec rôles ;
- audit fonctionnel strict.
```

Critère minimal :

```text
pages_using_fallback = 0
role_none_translation_items = 0
word_char_translation_items = 0
reconstruction_units_missing_roles = 0
functional_status = ok
```

---

# 3. Organisation parallèle recommandée

Il faut lancer **8 chantiers en parallèle**, mais avec un contrat commun :

```text
Aucun chantier ne modifie le format de translation_plan sans accord.
Aucun chantier ne casse les tests rev_03.
Chaque chantier ajoute ses tests.
Chaque chantier ajoute ses métriques audit.
```

---

# 4. Chantiers parallèles `rev_04`

# CHANTIER A — Audit, tests, garde-fous

## But

Empêcher les fausses validations.

## Tâches

* [ ] Étendre `tools/run_functional_audit.py` pour lancer aussi `PAGETRANSLATE` en `dry_run`.

* [ ] Ajouter option :

```bash
python3 tools/run_functional_audit.py input_data.json --run-pagetranslate --dry-run
```

* [ ] Mesurer réellement :

```text
selection_mode
fallback_selector_used
generic_coalescer_used
translation_plan_count
reconstruction_units_missing_roles
protected_tokens_missing
```

* [ ] Ajouter un audit multi-pages :

```bash
python3 tools/run_batch_functional_audit.py audit_folder/
```

* [ ] Ajouter `functional_status = ko` si :

```text
translation_plan absent sur page traduisible
fallback utilisé
role=None
word/char dans translation_plan
table détectable mais non structurée
index détectable mais non structuré
caption brute envoyée
publisher mark envoyé à traduction
```

## Tests à ajouter

```text
tests/functional/test_batch_audit_no_fallback.py
tests/functional/test_batch_audit_detects_missing_roles.py
tests/functional/test_batch_audit_detects_empty_plan.py
tests/functional/test_batch_audit_runs_pagetranslate.py
```

## Critère d’acceptation

```text
L’audit doit pouvoir dire KO même si le JSON est valide.
```

---

# CHANTIER B — Body paragraphs réels

## But

Arrêter le `visual_fallback` pour les pages prose.

## Modules

Créer :

```text
pageprint/structure_builders/body_paragraph_builder.py
```

Brancher dans :

```text
pageprint/structure_builders/__init__.py
pageprint/semantic_builder.py
```

## Tâches

* [ ] Détecter les paragraphes à partir de :

```text
reading_order
same block
alignement gauche
interligne
indentation
style homogène
continuité syntaxique
colonnes
```

* [ ] Produire :

```json
{
  "logical_unit_id": "body_para_001",
  "type": "body_paragraph",
  "text": "...",
  "source_unit_ids": [...],
  "line_unit_ids": [...],
  "bbox": [...],
  "role": "body_paragraph"
}
```

* [ ] `semantic_builder` doit consommer `body_paragraphs`.

* [ ] Si `body_paragraphs` existe, ne pas utiliser `visual_fallback`.

## Tests

```text
tests/pageprint/test_body_paragraph_builder.py
tests/pageprint/test_body_no_visual_fallback_when_paragraphs_exist.py
```

## Critère

```text
body page → segment_source = logical_body_paragraphs
```

---

# CHANTIER C — Tables robustes

## But

Les pages tableau ne doivent plus être traitées comme phrases.

## Module

Renforcer :

```text
pageprint/structure_builders/table_builder.py
```

## Tâches

* [ ] Détecter tables par :

```text
alignements x répétés
alignements y répétés
lignes vectorielles
fonds alternés
colonnes régulières
headers
densité cellulaire
espacement horizontal régulier
```

* [ ] Créer :

```json
{
  "table_id": "tbl_001",
  "caption": "...",
  "columns": [...],
  "rows": [...],
  "cells": [
    {
      "cell_id": "tbl_001_r2_c3",
      "text": "...",
      "role": "table_body_cell",
      "cell_kind": "natural_text | numeric | command | path | code | formula",
      "translation_mode": "translate | preserve_text_exactly"
    }
  ]
}
```

* [ ] Séparer dans les tableaux :

```text
command/path/code → preserve_text_exactly
description/function/action → translate
numeric/formula → preserve
header text → translate
```

* [ ] Interdire l’envoi d’une ligne/table brute au `translation_plan`.

## Tests

```text
tests/pageprint/test_table_grid_detection.py
tests/pageprint/test_table_command_cells_preserved.py
tests/pageprint/test_table_description_cells_translated.py
tests/pageprint/test_no_raw_table_row_in_translation_plan.py
```

## Critère

```text
page avec table → logical_structures.tables non vide
translation_plan cellule par cellule
aucun command/path dans translation normale
```

---

# CHANTIER D — Index robuste

## But

Les pages index doivent produire des `index_entries`.

## Module

Renforcer :

```text
pageprint/structure_builders/index_builder.py
```

## Tâches

* [ ] Détecter page index par :

```text
ordre alphabétique
lignes courtes
virgule + page refs
indentation sous-entrée
beaucoup de nombres de page
peu de phrases complètes
deux colonnes fréquentes
```

* [ ] Produire :

```json
{
  "logical_unit_id": "index_entry_001",
  "head_term": "PostGIS",
  "page_refs": ["242"],
  "subentries": [
    {
      "text": "creating spatial database",
      "page_refs": ["242–243"]
    }
  ],
  "source_unit_ids": [...]
}
```

* [ ] Préserver :

```text
page_refs
noms de fonctions
commandes
termes techniques verrouillés
```

* [ ] Traduire uniquement les sous-entrées naturelles.

## Tests

```text
tests/pageprint/test_index_page_detection.py
tests/pageprint/test_index_entry_builder.py
tests/pageprint/test_index_page_refs_preserved.py
tests/pageprint/test_no_index_line_raw_translation.py
```

## Critère

```text
index page → index_entries > 0
fallback coalescer interdit
page refs absentes du texte à traduire
```

---

# CHANTIER E — Figures, diagrammes, captions

## But

Ne pas envoyer les figures/captions/labels comme texte brut.

## Modules

Renforcer :

```text
pageprint/structure_builders/caption_builder.py
pageprint/structure_builders/figure_builder.py
```

## Tâches

* [ ] Split caption :

```text
Figure 2.1 Traditional ML algorithms require...
```

en :

```json
{
  "label": "Figure",
  "number": "2.1",
  "caption_text": "Traditional ML algorithms require..."
}
```

* [ ] Préserver numéro.

* [ ] Traduire `caption_text`.

* [ ] Identifier les `diagram_label`.

* [ ] Politique diagramme :

```text
ReLU, Softmax, Conv, FC → preserve/protect
Input, Output, Hidden layer → translate selon glossaire
```

* [ ] Ne pas traduire labels techniques courts sans contexte.

## Tests

```text
tests/pageprint/test_caption_split.py
tests/pageprint/test_figure_number_preserved.py
tests/pageprint/test_diagram_label_policy.py
tests/pageprint/test_no_raw_caption_block_translation.py
```

## Critère

```text
caption brute absente du translation_plan
caption_text seul traduit
figure number préservé
```

---

# CHANTIER F — Publisher marks, headers, footers, artifacts

## But

Éviter d’envoyer les éléments non documentaires au traducteur.

## Modules

Créer/renforcer :

```text
pageprint/structure_builders/publisher_mark_builder.py
pipelines/document_context.py
```

## Tâches

* [ ] Détecter répétitions multi-pages :

```text
headers
footers
watermarks
publisher marks
page numbers
running titles
```

* [ ] Ajouter dans `document_context` :

```json
{
  "repeated_headers": [...],
  "repeated_footers": [...],
  "publisher_marks": [...],
  "watermarks": [...]
}
```

* [ ] Classer :

```text
publisher_mark → exclude_as_artifact ou preserve_as_visual_overlay
page_number → preserve_text_exactly
running_header → translate/preserve selon stratégie
watermark pirate → exclude_as_artifact
```

* [ ] Empêcher ces éléments d’entrer dans `translation_plan`.

## Tests

```text
tests/pageprint/test_publisher_mark_exclusion.py
tests/pageprint/test_repeated_footer_detection.py
tests/pageprint/test_page_number_preserved.py
tests/pageprint/test_watermark_excluded.py
```

## Critère

```text
publisher_mark_sent_to_translation = 0
watermark_sent_to_translation = 0
```

---

# CHANTIER G — OCR routing ciblé

## But

Ne pas rater le texte dans les images, couvertures, schémas, figures.

## Modules

Renforcer :

```text
pipelines/ocr_router.py
pipelines/raw_extractors.py
pipelines/page_understanding.py
```

## Tâches

* [ ] Ajouter politique :

```text
native text exists ≠ OCR inutile
```

* [ ] Lancer OCR ciblé si :

```text
image_dominant
cover page
diagram/image regions avec texte probable
low native text density
large image region
caption rasterisée
```

* [ ] Sortir des claims OCR, pas des décisions finales.

* [ ] Marquer chaque OCR text avec :

```text
source = ocr_targeted_region
confidence
bbox
region_id
```

* [ ] Fusionner via `evidence_resolver`.

## Tests

```text
tests/pipelines/test_ocr_router_image_dominant.py
tests/pipelines/test_ocr_router_native_text_not_enough.py
tests/pageprint/test_ocr_claims_not_direct_policy.py
```

## Critère

```text
image_dominant + peu texte natif → OCR ciblé proposé
```

---

# CHANTIER H — Moteur de traduction en parallèle

## But

Commencer le vrai moteur sans attendre la perfection documentaire.

Mais il doit consommer seulement un `translation_plan` propre.

## Modules

Renforcer :

```text
pagetranslate/translator_bridge.py
pagetranslate/terminology.py
pagetranslate/protection.py
pagetranslate/quality.py
```

## Tâches

* [ ] Ajouter profils :

```text
mock
dry_run
local_model
external_model
```

* [ ] Ajouter `TranslationEngineInterface` :

```python
class TranslationEngine:
    def translate(self, text: str, source_lang: str, target_lang: str, context: dict) -> str:
        ...
```

* [ ] Ajouter glossaire technique :

```text
MLP
CNN
ReLU
Softmax
dropout
pooling
precision
recall
F-score
SQL
```

* [ ] Protéger tokens depuis `translation_plan.protected_tokens`.

* [ ] Ajouter QA :

```text
protected tokens restored
numbers preserved
units preserved
not empty
not identical unless allowed
length expansion acceptable
terminology consistent
```

* [ ] Produire :

```json
{
  "translation_quality": {
    "protected_token_mismatch_count": 0,
    "number_mismatch_count": 0,
    "terminology_warning_count": 0,
    "needs_review_count": 0
  }
}
```

## Tests

```text
tests/pagetranslate/test_translation_engine_interface.py
tests/pagetranslate/test_glossary_locks.py
tests/pagetranslate/test_protected_tokens_from_plan.py
tests/pagetranslate/test_translation_quality_numbers.py
```

## Critère

```text
Le moteur ne reçoit jamais une unité sans role/render_target/protected_tokens.
```

---

# 5. Tâches transversales obligatoires

## Tous les chantiers doivent respecter

* [ ] Ne jamais ajouter une règle `if page_number == ...`.

* [ ] Ne jamais patcher un texte spécifique comme `(weights)` sauf via une classe générale.

* [ ] Toute correction doit devenir :

```text
rôle
claim
structure
policy
test
```

* [ ] Chaque nouveau module doit ajouter :

```text
tests
métriques audit
entrée dans REFACTOR_EVALUATION.md
```

---

# 6. Ordre d’intégration malgré le travail parallèle

Même si on travaille tout en même temps, l’intégration doit suivre cet ordre :

```text
1. Chantier A — audit/métriques
2. Chantier B — body paragraphs
3. Chantier C — tables
4. Chantier D — index
5. Chantier E — figures/captions
6. Chantier F — publisher/header/footer
7. Chantier G — OCR routing
8. Chantier H — moteur traduction
```

Mais Codex peut coder en parallèle sur des branches séparées :

```text
rev04-audit
rev04-body
rev04-table
rev04-index
rev04-caption
rev04-artifacts
rev04-ocr
rev04-translation-engine
```

Fusionner seulement si :

```text
py_compile OK
tests du chantier OK
tests rev_03 OK
audit fonctionnel OK
```

---

# 7. Plan de sortie `rev_04`

## `rev_04-alpha`

Objectif :

```text
audit strict + body_paragraph_builder + table_builder initial
```

Critères :

```text
body no visual_fallback
tables simples reconnues
batch audit fonctionne
```

## `rev_04-beta`

Objectif :

```text
index + captions + publisher marks
```

Critères :

```text
index page reconnue
captions splittées
publisher marks exclus
```

## `rev_04-final`

Objectif :

```text
OCR routing + moteur traduction dry_run/local prêt
```

Critères :

```text
10 pages variées passent sans fallback
translation_engine testé sur translation_plan propre
```

---

# 8. Critères d’acceptation finale `rev_04`

`rev_04` est validée si :

```text
python3 -m py_compile ... OK
python3 -m pytest -q tests/pageprint tests/pagetranslate tests/functional OK

batch audit :
pages_using_fallback = 0
generic_coalesced_units = 0
role_none_translation_items = 0
word_char_translation_items = 0
reconstruction_units_missing_roles = 0
publisher_mark_sent_to_translation = 0
caption_raw_block_translation = 0
table_pages_without_tables = 0
index_pages_without_index_entries = 0
functional_status = ok
```

---

# 9. Décision immédiate

Il ne faut pas refaire encore une grande refonte conceptuelle. Elle est faite.

Maintenant, on fait :

```text
implémentation spécialisée massive
+ audit strict
+ tests
+ moteur de traduction en parallèle
```

La priorité est :

```text
1. Audit batch strict
2. Body paragraphs
3. Tables
4. Index
5. Captions/figures
6. Publisher/header/footer
7. OCR routing
8. Traduction moteur
```

---

# 10. Instruction courte à donner à Codex

```text
Rev_03 valide le cœur translation_plan.

Pour rev_04, travailler en parallèle sur :
- audit batch strict,
- body_paragraph_builder,
- table_builder robuste,
- index_builder robuste,
- caption/figure builders,
- publisher/header/footer detection,
- OCR routing ciblé,
- moteur de traduction branché sur translation_plan.

Ne jamais revenir à selector/coalescer comme chemin normal.
Chaque nouveau builder doit produire logical_structures,
semantic_builder doit les consommer,
view_compiler doit produire translation_plan/preservation_plan/reconstruction_plan,
functional_audit doit vérifier que PAGETRANSLATE n’utilise aucun fallback.

Objectif rev_04 :
10 pages variées sans fallback, sans role=None, sans word/char, sans table/index/caption brute, avec audit fonctionnel OK.
```

