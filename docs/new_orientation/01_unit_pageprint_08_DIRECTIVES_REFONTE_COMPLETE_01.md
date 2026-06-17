Voici le **cahier de directives de refonte** à donner à Codex. Il est orienté implémentation, avec objectifs, modules à créer/modifier, règles de codage, critères d’acceptation et cadre d’évaluation final.

# DIRECTIVES DE REFONTE — docs_parser / PAGEPRINT / PAGETRANSLATE

## 0. Mission générale

Refondre le pipeline documentaire pour passer d’un système d’annotations heuristiques à un système de **compilation documentaire**.

Le principe cible est :

```text
PAGEPRINT = compilateur documentaire
PAGETRANSLATE = exécuteur contrôlé du plan de traduction
RECONSTRUCTOR = exécuteur du plan de reconstruction
```

Le pipeline ne doit plus décider directement :

```text
région spéciale → background_only → skip_translation
```

Il doit passer par :

```text
observations → claims/preuves → résolution → rôles → structures logiques → politiques → plans aval → validation fonctionnelle
```

Objectif principal : produire un `INPUT_DATA` fiable, explicable, exploitable par traduction et reconstruction WYSIWYG.

---

# 1. Principes non négociables

## 1.1 PAGEPRINT ne doit pas seulement décrire la page

PAGEPRINT doit produire :

```text
- unités visuelles neutres ;
- preuves concurrentes ;
- rôles documentaires ;
- structures logiques ;
- unités sémantiques ;
- politiques de traduction/préservation/rendu ;
- plans aval : translation_plan, preservation_plan, reconstruction_plan, exclusion_plan ;
- validation fonctionnelle.
```

## 1.2 PAGETRANSLATE ne doit pas comprendre le document

PAGETRANSLATE doit lire :

```text
input_data["views"]["translation_plan"]
```

et ne doit utiliser `selector.py` / `coalescer.py` qu’en fallback.

En mode normal :

```text
PAGETRANSLATE ne choisit pas phrase/line/block.
PAGETRANSLATE ne fusionne pas les lignes.
PAGETRANSLATE ne devine pas TOC/index/table/code/list.
```

## 1.3 Une région spéciale n’est pas une décision

Une région spéciale est une **preuve**, pas une politique finale.

Interdit :

```python
region_type == "protected_visual_region" → skip_translation=True directement
```

Obligatoire :

```text
region → claim → evidence_resolver → resolved_understanding → policy_compiler
```

## 1.4 Remplacer `protected_visual` par `preservation_mode`

Ne plus utiliser `protected_visual/background_only` comme catégorie générale.

Introduire :

```text
preservation_mode:
  none
  protect_token_inside_translation
  preserve_text_exactly
  preserve_as_visual_overlay
  exclude_as_artifact
```

Exemples :

```text
MLP                         → protect_token_inside_translation
CNN                         → protect_token_inside_translation
copy / dir / del            → preserve_text_exactly
C:\Music\song.mp3           → preserve_text_exactly
équation complexe           → preserve_as_visual_overlay
logo / signature            → preserve_as_visual_overlay
watermark / footer pirate   → exclude_as_artifact
texte naturel               → none
```

## 1.5 Aucun `role=None` dans `translation_plan`

Toute unité envoyée en traduction doit avoir au minimum :

```text
role
object_type
semantic_kind
translation_mode
source_unit_ids
render_target
qa_requirements
```

Si le rôle est inconnu :

```text
ne pas envoyer directement à la traduction ;
marquer needs_role_resolution ;
ajouter au functional_validator.
```

---

# 2. Architecture cible

## 2.1 PAGEPRINT cible

Créer ou refondre la structure suivante :

```text
pageprint/
├── schema.py
├── builder.py
├── normalizer.py
├── unit_factory.py
├── detection/
│   ├── builder.py
│   ├── claims.py
│   └── normalizer.py
├── region_index.py
├── evidence/
│   ├── claim_model.py
│   ├── collector.py
│   └── resolver.py
├── role_resolver.py
├── graph_builder.py
├── graph_query.py
├── structure_builders/
│   ├── toc_builder.py
│   ├── index_builder.py
│   ├── table_builder.py
│   ├── list_builder.py
│   ├── caption_builder.py
│   ├── code_builder.py
│   ├── formula_builder.py
│   ├── figure_builder.py
│   └── author_bio_builder.py
├── semantic_builder.py
├── policy_compiler.py
├── preservation_compiler.py
├── constraint_compiler.py
├── view_compiler.py
├── quality_assessor.py
├── functional_validator.py
├── validators.py
└── serializers.py
```

## 2.2 PAGETRANSLATE cible

Créer ou refondre :

```text
pagetranslate/
├── builder.py
├── translation_plan_reader.py
├── fallback_selector.py
├── fallback_coalescer.py
├── protection.py
├── terminology.py
├── translator_bridge.py
├── quality.py
├── projection.py
├── functional_validator.py
└── schema.py
```

`selector.py` et `coalescer.py` peuvent rester, mais ils doivent être renommés ou réorientés comme fallback.

---

# 3. Ordre cible du pipeline PAGEPRINT

Modifier `PagePrintBuilder.build()` pour suivre cet ordre logique :

```text
1. Normaliser page/source/geometry.
2. Construire les unités visuelles neutres.
3. Construire les régions neutres.
4. Calculer les memberships unit↔region.
5. Collecter toutes les preuves/claims.
6. Résoudre les rôles documentaires.
7. Construire les structures logiques.
8. Construire les unités sémantiques.
9. Résoudre les conflits de compréhension.
10. Compiler les politiques de traduction.
11. Compiler les politiques de préservation.
12. Compiler les contraintes WYSIWYG.
13. Compiler les vues/plans aval.
14. Valider schéma.
15. Valider fonctionnellement.
16. Retourner INPUT_DATA final.
```

Important : `policy_compiler` doit venir **après** `role_resolver`, `structure_builders` et `semantic_builder`.

---

# 4. Refondre la détection de régions

## 4.1 Ne plus écrire de politique finale dans `detection/builder.py`

Modifier `pageprint/detection/builder.py`.

Actuellement, une région spéciale peut produire directement :

```json
{
  "protected_visual": true,
  "translatable": false,
  "translation_strategy": "background_only",
  "render_policy": "background_only",
  "skip_translation": true,
  "skip_text_reconstruction": true
}
```

Cela doit être supprimé.

À la place, produire des claims :

```json
{
  "claim_id": "claim_region_001",
  "source": "special_region_detector",
  "claim_type": "possible_formula_or_code_or_visual",
  "bbox": [x0, y0, x1, y1],
  "confidence": 0.0,
  "reason": "...",
  "evidence": {
    "region_class": "...",
    "detector_score": 0.0,
    "text_overlap": "...",
    "coverage_ratio": 0.0
  }
}
```

## 4.2 Éviter les doubles appels au détecteur

Le détecteur de régions spéciales ne doit pas être appelé deux fois.

Règle :

```text
PageUnderstanding exécute les détecteurs coûteux.
PAGEPRINT normalise les résultats reçus.
```

Modifier `PageRegionDetectBuilder.build()` :

```python
def build(..., run_detector: bool = False, normalize_existing: bool = True):
    ...
```

Si `page_structure["special_regions"]` existe :

```text
ne pas relancer le détecteur sauf force_detect=True.
```

## 4.3 Typologie des régions

Les régions doivent être neutres :

```text
image_region
body_region
table_region
figure_region
formula_candidate_region
code_candidate_region
diagram_region
header_region
footer_region
watermark_candidate_region
background_region
```

Une région candidate n’est pas encore une politique.

---

# 5. Refondre `region_index.py`

## 5.1 Ne plus contaminer les parents

Actuellement, une région protégée partielle peut faire basculer une ligne ou un bloc entier.

Introduire `coverage_mode` :

```text
none
incidental_overlap
partial_inline
dominant_overlap
full_coverage
```

Règles :

```text
block full protection     : overlap >= 0.90
line full protection      : overlap >= 0.85
phrase full protection    : overlap >= 0.80
span full protection      : overlap >= 0.75
partial inline protection : 0.10 <= overlap < full threshold
incidental overlap        : overlap < 0.10
```

Le résultat doit être stocké dans :

```json
{
  "region_memberships": [
    {
      "region_id": "...",
      "region_type": "...",
      "overlap_ratio": 0.32,
      "coverage_mode": "partial_inline",
      "action_hint": "protect_inline_token_not_parent"
    }
  ]
}
```

Interdit :

```text
partial overlap → parent background_only
```

---

# 6. Créer le modèle de claims/preuves

## 6.1 Créer `pageprint/evidence/claim_model.py`

Définir des helpers simples, sans dépendance lourde :

```python
def make_claim(
    *,
    source: str,
    target_unit_id: str | None,
    claim_type: str,
    value: str,
    confidence: float,
    reason: str,
    evidence: dict | None = None,
) -> dict:
    ...
```

Types de claims minimaux :

```text
natural_text
formula_candidate
formula_confirmed
code_candidate
code_confirmed
table_candidate
table_confirmed
toc_candidate
toc_confirmed
index_candidate
index_confirmed
caption_candidate
caption_confirmed
publisher_mark_candidate
publisher_mark_confirmed
author_name_candidate
page_reference
section_number
command_name
file_path
url
email
acronym
proper_name
watermark
```

## 6.2 Créer `pageprint/evidence/collector.py`

Collecter des claims depuis :

```text
- unités natives PDF ;
- OCR ;
- régions ;
- style ;
- police monospace ;
- géométrie ;
- page_intelligence ;
- graph relations ;
- page number / header / footer ;
- motifs TOC/index/table ;
- captions ;
- code/path/command.
```

## 6.3 Refondre `evidence_resolver.py` ou créer `evidence/resolver.py`

Le resolver doit produire :

```json
{
  "resolved_understanding": {
    "role": "...",
    "object_type": "...",
    "semantic_kind": "...",
    "confidence": 0.0,
    "reason": "...",
    "winning_claims": [...],
    "rejected_claims": [...]
  }
}
```

Ne jamais faire gagner une région faible contre :

```text
- texte natif clair ;
- rôle TOC/index/table confirmé ;
- phrase naturelle longue ;
- caption reconnue ;
- table cell reconnue.
```

---

# 7. Créer `role_resolver.py`

## 7.1 But

Attribuer un rôle documentaire à chaque unité pertinente.

Rôles minimaux :

```text
body_paragraph
title
subtitle
section_heading
subsection_heading
list_item
list_marker
figure_caption
figure_label
table_caption
table_header_cell
table_body_cell
table_numeric_cell
formula_expression
formula_explanation
code_block
code_line
code_token
command_name
path
file_name
url
email
page_header
page_footer
publisher_mark
watermark
author_name
author_bio
index_entry
index_head_term
index_subentry
index_page_reference
toc_title
toc_entry
toc_entry_title
toc_section_number
toc_page_reference
toc_bullet_marker
diagram_label
diagram_text_label
```

## 7.2 Règles

Le rôle doit dépendre de :

```text
page_role
page_family
layout_type
position
style
text pattern
region claims
graph relations
native PDF data
document context si disponible
```

## 7.3 Principe de prudence

Si rôle incertain :

```text
role = "unknown"
translation_mode = "needs_role_resolution"
```

Ne pas envoyer en traduction normale.

---

# 8. Créer les structure_builders

## 8.1 `toc_builder.py`

Entrée :

```text
page_role == toc
ou forte preuve TOC
```

Sortie :

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

Règles :

```text
section_number → preserve_text_exactly
page_reference → preserve_text_exactly
bullet_marker → preserve_text_exactly
title_text → translate
```

Interdit :

```text
envoyer "CONTENTS vii 3" comme bloc à traduire.
```

## 8.2 `index_builder.py`

Détecter :

```text
entrée, sous-entrée, page refs, termes techniques, fonctions.
```

Sortie :

```json
{
  "logical_unit_id": "index_entry_001",
  "type": "index_entry",
  "head_term": "PostGIS",
  "subentries": [
    {
      "text": "creating spatial database",
      "page_refs": ["242–243"]
    }
  ],
  "source_unit_ids": [...]
}
```

Politique :

```text
page_refs → preserve
fonction technique → preserve
texte naturel de sous-entrée → translate
```

## 8.3 `table_builder.py`

Détecter tableaux par :

```text
alignement colonnes
lignes vectorielles
fonds alternés
colonnes récurrentes
headers
densité cellulaire
mots PDF natifs alignés
```

Sortie :

```json
{
  "table_id": "tbl_001",
  "caption": "...",
  "columns": [...],
  "cells": [
    {
      "cell_id": "tbl_001_r1_c1",
      "role": "command_name",
      "text": "copy",
      "translation_mode": "preserve_text_exactly"
    }
  ]
}
```

Règle :

```text
ne jamais traiter une table comme un bloc de phrases normal.
```

## 8.4 `list_builder.py`

Séparer :

```text
marker
text
nested level
continuation lines
```

Sortie :

```json
{
  "logical_unit_id": "list_item_001",
  "marker": "▪",
  "text": "...",
  "continuation_unit_ids": [...],
  "marker_policy": "preserve_text_exactly",
  "text_policy": "translate"
}
```

## 8.5 `caption_builder.py`

Pour :

```text
Figure 2.1 Traditional ML algorithms...
Table 16-1: Useful Windows Commands
```

Sortie :

```json
{
  "caption_id": "cap_001",
  "label": "Figure",
  "number": "2.1",
  "caption_text": "Traditional ML algorithms...",
  "preserve": ["2.1"],
  "translatable_text": "Traditional ML algorithms..."
}
```

## 8.6 `code_builder.py`

Identifier :

```text
code_block
code_line
command_name
sql_keyword
path
file_name
function_call
```

Politique :

```text
code/commands/paths → preserve_text_exactly
commentaires/descriptions → translate
```

## 8.7 `formula_builder.py`

Différencier :

```text
formula_expression → preserve_as_visual_overlay ou preserve_text_exactly
formula_explanation → translate
```

Ne pas classifier comme formule :

```text
(weights)
(3D images)
True positives (TP)
Figure 2.1
Table 16-1
```

## 8.8 `author_bio_builder.py`

Détecter :

```text
author_name
affiliation
biography_text
```

Protéger les noms propres.

## 8.9 `publisher_mark_builder.py`

Détecter :

```text
watermark
footer pirate
publisher mark
logo text
```

Politique :

```text
exclude_as_artifact ou preserve_as_visual_overlay selon cas.
```

---

# 9. Créer `semantic_builder.py`

## 9.1 But

Construire les unités sémantiques à partir des rôles et structures logiques.

PAGEPRINT doit produire :

```text
semantic_phrases
semantic_groups
logical_text_units
translation_segments
```

## 9.2 Règles par type de page

```text
body_text       → phrases / paragraphes
toc             → toc_entries
index           → index_entries
table           → table_cells
figure          → labels + captions
code_page       → code blocks + explanatory text
author_bio      → author entries
summary         → list_items
cover           → title/subtitle/author/publisher
```

## 9.3 Ne pas utiliser une logique unique de phrase

Interdit :

```text
absence de ponctuation → fusion automatique
```

Obligatoire :

```text
fusion seulement si relation logique confirmée.
```

---

# 10. Créer `graph_query.py`

Créer des fonctions :

```python
is_same_paragraph(unit_a, unit_b)
is_same_list_item(unit_a, unit_b)
is_same_table_cell(unit_a, unit_b)
is_same_toc_entry(unit_a, unit_b)
is_same_index_entry(unit_a, unit_b)
is_caption_of(unit_a, figure_unit)
is_inside_region(unit, region_type)
has_partial_protected_overlap(unit)
nearest_heading(unit)
reading_predecessor(unit)
reading_successor(unit)
can_merge_for_translation(unit_a, unit_b)
```

`semantic_builder` et fallback coalescer doivent utiliser ce module.

---

# 11. Refondre `policy_compiler.py`

## 11.1 Entrées

Le compiler doit utiliser :

```text
resolved_understanding
role
object_type
semantic_kind
claims
logical_units
region_memberships
page_role
page_family
layout_type
```

## 11.2 Sorties

Séparer :

```json
{
  "translation_policy": {
    "mode": "translate | preserve_text_exactly | skip | needs_review",
    "strategy": "...",
    "protected_tokens": [],
    "reason": "..."
  },
  "preservation_policy": {
    "mode": "none | protect_token_inside_translation | preserve_text_exactly | preserve_as_visual_overlay | exclude_as_artifact",
    "reason": "..."
  },
  "render_policy": {
    "mode": "redraw_text | preserve_overlay | skip_artifact | anchored_text | paragraph_flow",
    "reason": "..."
  }
}
```

## 11.3 Remplacer les regex booléennes par scores

Ne plus faire :

```python
if "if" in text:
    code = True
```

Faire :

```python
score = 0
# style, role, syntaxe, ponctuation, path, monospace, contexte
return score >= threshold
```

Code-like doit exiger preuves fortes :

```text
monospace
ou rôle code
ou ponctuation dense
ou function_call
ou path
ou commande
ou SQL keyword dans contexte code/table
```

Formula-like doit exiger :

```text
symboles mathématiques réels
équation structurée
ratio symbolique élevé
opérateurs mathématiques
```

Parenthèses seules ne suffisent pas.

---

# 12. Créer `preservation_compiler.py`

Compiler `preservation_mode`.

Règles minimales :

```text
page_reference              → preserve_text_exactly
section_number              → preserve_text_exactly
list_marker                 → preserve_text_exactly
command_name                → preserve_text_exactly
path/file_name              → preserve_text_exactly
url/email                   → preserve_text_exactly
acronym/model_name          → protect_token_inside_translation
proper_name                 → protect_token_inside_translation
formula_expression          → preserve_as_visual_overlay ou preserve_text_exactly
logo/signature              → preserve_as_visual_overlay
watermark/publisher artifact→ exclude_as_artifact
natural text                → none
```

---

# 13. Refondre `constraint_compiler.py`

## 13.1 Interdire `role=None` comme prose

Modifier `_is_prose()` :

```python
role in {"body", "paragraph", None}
```

doit devenir :

```python
role in {"body_paragraph", "paragraph", "body"}
```

Et refuser prose libre si :

```text
page_role in {toc, index, cover}
layout_type in {image_dominant, table_dominant, annotated_page}
```

Principe :

```text
unknown = anchored_text prudent
pas paragraph_flow.
```

## 13.2 Contraintes selon rôle

```text
body_paragraph → paragraph_flow
title/heading → anchored_text or title_fit
toc_entry → fixed_row_reflow
index_entry → fixed_entry_reflow
table_cell → cell_fit
caption → caption_reflow
diagram_label → fixed_label
code_line/path/command → fixed_text_exact
formula → overlay_preserve
```

---

# 14. Créer `view_compiler.py`

Compiler quatre vues principales.

## 14.1 `views.translation_plan`

Chaque item :

```json
{
  "translation_unit_id": "...",
  "source_unit_ids": [...],
  "logical_unit_id": "...",
  "source_text": "...",
  "role": "...",
  "object_type": "...",
  "semantic_kind": "...",
  "translation_mode": "translate",
  "translation_strategy": "...",
  "protected_tokens": [...],
  "context": {...},
  "render_target": {...},
  "qa_requirements": {...},
  "reason_included": "..."
}
```

## 14.2 `views.preservation_plan`

Chaque item :

```json
{
  "preservation_id": "...",
  "source_unit_ids": [...],
  "text": "...",
  "preservation_mode": "...",
  "render_mode": "...",
  "reason": "..."
}
```

## 14.3 `views.reconstruction_plan`

Chaque item :

```json
{
  "reconstruction_unit_id": "...",
  "source_unit_ids": [...],
  "role": "...",
  "object_type": "...",
  "bbox": [...],
  "style_source_unit_id": "...",
  "render_contract": {...},
  "text_source": "translation_plan | preservation_plan | original",
  "consume_source_unit_ids": [...]
}
```

## 14.4 `views.exclusion_plan`

Pour watermarks, artifacts, éléments non documentaires.

```json
{
  "exclusion_id": "...",
  "source_unit_ids": [...],
  "reason": "publisher_mark | watermark | artifact",
  "bbox": [...]
}
```

---

# 15. Refondre PAGETRANSLATE

## 15.1 `translation_plan_reader.py`

Lire `views.translation_plan`.

Normaliser vers les items internes :

```python
def read_translation_plan(input_data: dict) -> list[dict]:
    ...
```

Chaque item doit contenir :

```text
translation_unit_id
source_text
role
object_type
semantic_kind
protected_tokens
context
render_target
qa_requirements
```

## 15.2 Fallback seulement si pas de plan

Dans `pagetranslate/builder.py` :

```python
plan = input_data["views"].get("translation_plan")
if plan:
    units = read_translation_plan(input_data)
    selection_mode = "translation_plan"
else:
    units = fallback_selector(...)
    selection_mode = "fallback_selector"
```

Ajouter dans le résultat :

```json
{
  "selection_mode": "translation_plan"
}
```

ou

```json
{
  "selection_mode": "fallback_selector",
  "warning": "PAGEPRINT did not provide translation_plan"
}
```

## 15.3 Coalescer fallback contextuel

Le coalescer ne peut fusionner que si :

```text
graph_query.can_merge_for_translation(prev, current) == true
```

Interdire fusion pour :

```text
toc
index
table
code
formula
diagram
list
caption structurée
```

## 15.4 Projection

`projection.py` doit conserver :

```text
role
object_type
semantic_kind
page_role
render_target
source_unit_ids
logical_unit_id
render_contract
```

dans `reconstruction_units`.

---

# 16. Refondre `protection.py`

## 16.1 Placeholders robustes

Remplacer :

```text
__PT_0001__
```

par :

```text
⟦PT0001⟧
```

Restaurer aussi les variantes :

```text
⟦ PT0001 ⟧
[[PT0001]]
<nt id="PT0001"/>
PT0001
```

## 16.2 Protéger depuis le plan

`protect_text()` doit accepter :

```python
explicit_tokens: list[str]
token_classes: list[dict]
```

Tokens à protéger :

```text
acronym
proper_name
organization_name
model_name
library_name
command_name
sql_keyword
file_path
url
email
page_reference
equation_number
figure_number
table_number
numeric_value
unit
```

---

# 17. Créer `terminology.py`

But :

```text
appliquer glossaire, termes verrouillés, termes préférés.
```

Fonctions :

```python
prepare_terminology_context(item, profile)
apply_pre_translation_locks(text, terminology)
apply_post_translation_glossary(text, terminology)
check_terminology_consistency(source, translation, terminology)
```

Ne pas traduire :

```text
MLP, CNN, ReLU, Softmax, SQL keywords, noms propres, chemins.
```

Sauf règle explicite contraire.

---

# 18. Créer validateurs fonctionnels

## 18.1 `pageprint/functional_validator.py`

Retourner :

```json
{
  "functional_valid": false,
  "errors": [],
  "warnings": [],
  "metrics": {...}
}
```

Erreurs bloquantes :

```text
translation_plan item with role=None
translation_plan item with object_type=None
word/char in translation_plan
mixed block sent as translation unit
partial protected region made parent background_only
toc page without toc_entries
index page without index_entries
table-like page without tables
body prose page with empty semantic_system
reconstruction_plan item without role
preserve_as_visual_overlay on majority natural text
```

## 18.2 `pagetranslate/functional_validator.py`

Erreurs bloquantes :

```text
selection_mode=fallback_selector when translation_plan exists
translation item role=None
protected token missing after restore
command/path/page_reference translated
unchanged translation not flagged
reconstruction_unit role missing
translation without render_target
duplicate source_unit_ids rendered twice
```

---

# 19. Audit final

L’audit doit séparer :

```json
{
  "schema_status": "ok",
  "functional_status": "ko",
  "blocking_reasons": [...]
}
```

Ne jamais afficher seulement :

```text
status: ok
```

si le fonctionnement est faux.

Ajouter métriques :

```text
role_none_translation_units
word_char_translation_units
fallback_selector_usage
generic_coalesced_units
natural_text_marked_preserve_visual
table_false_negative_pages
index_false_negative_pages
toc_without_entries
publisher_mark_sent_to_translation
code_path_command_sent_to_translation
reconstruction_units_missing_roles
```

---

# 20. Tests obligatoires

Créer un dossier :

```text
tests/pageprint/
tests/pagetranslate/
tests/functional/
```

## 20.1 Tests PAGEPRINT

Créer au minimum :

```text
test_region_claim_not_direct_policy.py
test_partial_protected_region_not_parent_background.py
test_role_none_not_prose.py
test_toc_builder_entries.py
test_index_builder_entries.py
test_table_builder_cells.py
test_caption_split_label_number_text.py
test_code_detection_requires_strong_evidence.py
test_formula_detection_not_parentheses_only.py
test_publisher_mark_exclusion.py
test_semantic_system_non_empty_for_body.py
test_translation_plan_no_role_none.py
```

## 20.2 Tests PAGETRANSLATE

```text
test_reads_translation_plan.py
test_fallback_only_when_plan_missing.py
test_no_generic_coalescing_on_toc_index_table.py
test_protected_tokens_restored.py
test_command_path_not_translated.py
test_projection_keeps_roles.py
test_translation_unit_requires_render_target.py
test_unchanged_translation_needs_review.py
```

## 20.3 Tests audit

```text
test_functional_status_ko_when_role_missing.py
test_functional_status_ko_when_semantic_system_empty.py
test_functional_status_ko_when_table_not_detected.py
test_functional_status_ko_when_natural_text_preserved_as_visual.py
```

---

# 21. Cadre d’évaluation final Codex

À la fin de l’implémentation, produire un rapport :

```text
REFACTOR_EVALUATION.md
```

Ce rapport doit contenir :

## 21.1 Résumé des fichiers modifiés

```text
- fichiers créés
- fichiers modifiés
- fichiers supprimés
- fonctions principales ajoutées
```

## 21.2 Compatibilité

Répondre :

```text
Le schéma pageprint.input.v1 est-il encore compatible ?
Les anciens champs sont-ils conservés ou migrés ?
Les anciens tests passent-ils ?
PAGETRANSLATE fonctionne-t-il avec et sans translation_plan ?
```

## 21.3 Validation technique

Exécuter :

```bash
python -m py_compile pageprint/*.py pagetranslate/*.py
python -m py_compile pageprint/detection/*.py
python -m py_compile pageprint/evidence/*.py
python -m py_compile pageprint/structure_builders/*.py
pytest -q
```

Si `pytest` n’est pas installé, fournir au moins :

```bash
python -m unittest discover tests
```

## 21.4 Validation fonctionnelle

Créer ou mettre à jour un script :

```text
tools/run_functional_audit.py
```

Il doit produire :

```json
{
  "schema_status": "ok",
  "functional_status": "ok|ko",
  "metrics": {
    "role_none_translation_units": 0,
    "word_char_translation_units": 0,
    "fallback_selector_usage": 0,
    "generic_coalesced_units": 0,
    "reconstruction_units_missing_roles": 0
  }
}
```

## 21.5 Critères d’acceptation

La refonte est acceptée seulement si :

```text
1. Compilation OK.
2. Tests unitaires OK.
3. PAGEPRINT produit views.translation_plan.
4. PAGEPRINT produit views.preservation_plan.
5. PAGEPRINT produit views.reconstruction_plan.
6. PAGETRANSLATE lit translation_plan en mode normal.
7. selector/coalescer ne sont utilisés qu’en fallback.
8. Aucun item translation_plan avec role=None.
9. Aucun word/char dans translation_plan.
10. Aucun command/path/page_reference envoyé en traduction normale.
11. reconstruction_units conservent role/object_type/semantic_kind.
12. functional_status existe et peut être ko même si schema_status est ok.
```

## 21.6 Rapport des limites restantes

Le rapport doit finir par :

```text
LIMITES RESTANTES
- ce qui n’est pas encore implémenté ;
- ce qui reste heuristique ;
- ce qui nécessite un modèle AI externe ;
- ce qui nécessite un corpus golden ;
- risques de régression.
```

---

# 22. Priorité d’implémentation recommandée

Ne pas tout faire dans le désordre.

Ordre obligatoire :

```text
SPRINT 0 — Sécurisation
1. Région → claim, plus de policy directe.
2. Pas de double appel detector.
3. role None ≠ prose.
4. functional_status dans audit.

SPRINT 1 — Rôles et préservation
5. role_resolver.py.
6. preservation_mode.
7. preservation_compiler.py.
8. policy_compiler refondu.

SPRINT 2 — Structures
9. toc_builder.py.
10. index_builder.py.
11. table_builder.py.
12. caption_builder.py.
13. list_builder.py.
14. code_builder.py.
15. formula_builder.py.

SPRINT 3 — Plans aval
16. semantic_builder.py.
17. view_compiler.py.
18. translation_plan.
19. preservation_plan.
20. reconstruction_plan.
21. exclusion_plan.

SPRINT 4 — PAGETRANSLATE
22. translation_plan_reader.py.
23. fallback selector/coalescer.
24. projection enrichie.
25. protection robuste.
26. terminology.py.

SPRINT 5 — Validation
27. functional validators.
28. tests.
29. REFACTOR_EVALUATION.md.
```

---

# 23. Règle finale de codage

Ne pas créer de rustines spécifiques du type :

```python
if page_number == 501:
    ...
```

Ne pas créer de règles trop spécifiques :

```python
if text == "(weights)":
    ...
```

Toujours généraliser par rôle, claim, structure ou politique :

```text
parenthetical_technical_note
toc_entry_title
formula_candidate_rejected
table_cell_command
index_page_reference
```

Chaque correction doit appartenir à une règle générale testable.

---

# 24. Résultat attendu

À la fin, le comportement cible doit être :

```text
PAGEPRINT produit une compréhension documentaire actionnable.
PAGETRANSLATE exécute un plan.
Le reconstructeur reçoit un plan de rendu.
L’audit distingue JSON valide et pipeline fonctionnel.
Les erreurs ne sont plus corrigées page par page, mais par principes structurels.
```

La refonte est réussie quand le pipeline peut dire clairement :

```text
ceci est du texte naturel à traduire ;
ceci est un token à préserver ;
ceci est une cellule de tableau ;
ceci est une entrée d’index ;
ceci est une caption ;
ceci est un code ;
ceci est un artefact ;
voici comment traduire ;
voici comment reconstruire ;
voici pourquoi.
```

## Résumé opérationnel

La directive la plus importante pour Codex est celle-ci :

```text
Ne pas continuer à patcher les erreurs page par page.
Refondre la chaîne de décision :
claims → rôles → structures → politiques → plans → validation.
```

Le premier jalon à exiger est minimal mais décisif :

```text
PAGEPRINT produit views.translation_plan
et PAGETRANSLATE l’utilise en mode normal.
```

Tant que `PAGETRANSLATE` dépend encore principalement de `selector.py` + `coalescer.py`, la refonte n’est pas terminée.

