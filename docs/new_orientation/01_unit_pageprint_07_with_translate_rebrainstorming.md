Non. Ce que j’ai dit avant était **juste**, mais **pas complet**. Après inspection du dossier projet complet, je confirme l’idée principale, mais je la durcis :

```text
Le problème n’est pas seulement PAGEPRINT ou PAGETRANSLATE.
Le problème est la philosophie globale de décision du pipeline.
```

Le projet compile. Le découpage existe. Les unités sont bien engagées. Mais le système reste encore trop dépendant de décisions locales, de flags hérités, de regex, de régions “fortes”, et de modules legacy qui influencent le flux sans contrat suffisamment strict.

Le vrai objectif maintenant doit être :

```text
passer d’un pipeline d’annotations heuristiques
à un pipeline de compilation documentaire.
```

---

# 1. Rétro-brainstorming : ce que je confirme, infirme, approfondis

## Je confirme

Je confirme fortement ces corrections de fond :

```text
1. PAGEPRINT doit produire une compréhension documentaire exécutable.
2. PAGETRANSLATE ne doit plus inventer les unités sémantiques principales.
3. Il faut un translation_plan produit par PAGEPRINT.
4. protected_visual est trop brutal et doit être remplacé par preservation_mode.
5. Il faut un role_resolver avant policy_compiler.
6. Il faut un semantic_builder côté PAGEPRINT.
7. Le coalescer de PAGETRANSLATE doit devenir un fallback, pas un moteur principal.
8. Il faut des validateurs fonctionnels, pas seulement des validateurs de schéma.
```

## J’infirme / je nuance

Je nuance un point important : **il ne suffit pas d’ajouter `role_resolver.py`, `semantic_builder.py` et `translation_plan`**.

Si on les ajoute simplement au-dessus du système actuel sans changer la logique de décision, on aura seulement une couche de plus. Le vrai problème est que certaines étapes actuelles prennent des décisions définitives trop tôt.

Exemple actuel :

```text
special_region_detector
→ protected_visual_region
→ background_only
→ skip_translation
→ skip_text_reconstruction
```

Cette chaîne est trop directe. Il faut la casser.

Une région spéciale ne doit pas décider. Elle doit seulement déclarer une hypothèse.

## J’approfondis

Le concept central à introduire n’est pas seulement `translation_plan`.

Il faut introduire une architecture complète en **plans** :

```text
extraction_claims
understanding_plan
preservation_plan
translation_plan
reconstruction_plan
qa_plan
```

Autrement dit :

```text
PAGEPRINT ne doit pas seulement sortir units[].
PAGEPRINT doit sortir les décisions aval compilées.
```

---

# 2. Diagnostic profond du projet complet

Le projet a aujourd’hui trois centres de gravité.

## Centre 1 — ancien monolithe

```text
ocr_server.py        ~5637 lignes
translator.py        ~7399 lignes
reconstructor.py     ~8646 lignes
```

Ces fichiers contiennent encore beaucoup de logique précieuse, mais sous forme monolithique.

Ils savent faire beaucoup de choses, parfois mieux que les nouvelles unités, mais ils mélangent :

```text
extraction
classification
segmentation
traduction
post-édition
reconstruction
debug
politiques
fallbacks
```

Il ne faut pas revenir à ce modèle.

## Centre 2 — nouveau pipeline objectif

```text
pipelines/
├── source_loader.py
├── page_renderer.py
├── raw_extractors.py
├── page_understanding.py
├── pageprint_stage.py
└── orchestrator.py
```

C’est la bonne direction. Mais il reste trop léger. Il orchestre des modules, mais ne gouverne pas encore clairement les responsabilités.

## Centre 3 — unités nouvelles

```text
pageprint/
pagetranslate/
```

Elles sont bien séparées, mais `PAGEPRINT` ne comprend pas encore assez, donc `PAGETRANSLATE` compense trop.

---

# 3. Problème fondamental n°1 : le pipeline décide trop tôt

Dans `pageprint/detection/builder.py`, une région détectée comme classe protégée devient directement :

```python
region_type = "protected_visual_region"
protected_visual = True
translatable = False
translation_strategy = "background_only"
render_policy = "background_only"
preserve_original_pixels = True
preserve_as_image = True
skip_translation = True
skip_text_reconstruction = True
```

C’est trop violent.

À ce stade, on ne devrait pas encore décider :

```text
ne pas traduire
ne pas reconstruire
préserver en image
```

On devrait seulement dire :

```json
{
  "claim_type": "possible_formula_or_code_or_visual",
  "source": "special_region_detector",
  "confidence": 0.72,
  "bbox": [...]
}
```

La décision finale doit venir plus tard, après confrontation avec :

```text
texte natif
OCR
style
rôle de page
rôle d’unité
graph relations
table/index/toc/caption context
densité symbolique
preuve de monospace/code
preuve de formule réelle
```

## Correction structurelle

Remplacer la logique actuelle :

```text
detected_region → policy
```

par :

```text
detected_region → claim → evidence_resolver → resolved_understanding → policy
```

---

# 4. Problème fondamental n°2 : `protected_visual` est un concept trop large

Aujourd’hui, `protected_visual` couvre trop de choses :

```text
formule
code
image
logo
symbole
numéro
commande
chemin
table formula cell
notation
diagram label non linguistique
```

Ces objets n’ont pas la même politique.

Il faut remplacer `protected_visual` par une taxonomie plus fine.

## Nouvelle taxonomie

```text
preservation_mode =

none
→ texte naturel traduisible

protect_token_inside_translation
→ l’unité est traduite, mais certains tokens sont verrouillés

preserve_text_exactly
→ texte redessiné tel quel, pas traduit

preserve_as_visual_overlay
→ zone préservée comme image/pixels

exclude_as_artifact
→ élément non documentaire, watermark/footer pirate/pub
```

Exemples :

```text
MLP                         → protect_token_inside_translation
C:\Music\song.mp3           → preserve_text_exactly
copy / del / findstr        → preserve_text_exactly
équation complexe           → preserve_as_visual_overlay
logo Manning                → preserve_as_visual_overlay ou publisher_mark
numéro de page              → preserve_text_exactly
Estadísticos e-Books...     → exclude_as_artifact
texte de paragraphe         → none
```

Tant que tout est ramené à `protected_visual/background_only`, les faux positifs resteront structurels.

---

# 5. Problème fondamental n°3 : `evidence_resolver.py` existe mais ne gouverne pas vraiment

Le fichier `pageprint/evidence_resolver.py` est trop faible.

Il fait actuellement :

```text
source extraction par défaut
puis si région spéciale overlap > 0.65
→ la région gagne
```

C’est exactement le problème.

Une région spéciale ne doit pas “gagner” par simple overlap. Elle doit être pondérée.

## Nouvelle logique attendue

Créer un vrai modèle de preuves concurrentes :

```json
{
  "claims": [
    {
      "source": "native_pdf",
      "claim": "natural_text",
      "confidence": 0.96
    },
    {
      "source": "special_region_detector",
      "claim": "formula_region",
      "confidence": 0.68,
      "coverage_ratio": 0.31
    },
    {
      "source": "role_resolver",
      "claim": "toc_entry_title",
      "confidence": 0.91
    }
  ],
  "resolved": {
    "object_type": "toc_entry_title",
    "preservation_mode": "none",
    "translation_mode": "translate",
    "confidence": 0.91,
    "reason": "toc_role_and_native_text_override_weak_formula_claim"
  }
}
```

Le résultat final doit être une décision raisonnée, pas un flag hérité.

---

# 6. Problème fondamental n°4 : `unit_factory.py` fait trop de choses

`pageprint/unit_factory.py` devrait fabriquer des unités canoniques.

Mais il fait déjà de la compréhension :

```text
toc_role
toc_section_number
toc_page_reference
toc_bullet_marker
toc_entry
```

et applique même certaines politiques TOC.

C’est utile à court terme, mais méthodologiquement ce n’est pas propre.

Le rôle de `unit_factory` devrait être :

```text
transformer page_structure en units visuelles propres
normaliser geometry/style/content/extraction
conserver les ids et relations parent-enfant
```

Pas :

```text
décider qu’une unité est toc_entry
décider qu’elle est translatable
décider exact_preserve
```

## Correction structurelle

Déplacer cette logique vers :

```text
pageprint/role_resolver.py
pageprint/structure_builders/toc_builder.py
pageprint/policy_compiler.py
```

Donc :

```text
unit_factory = construction neutre
role_resolver = compréhension
policy_compiler = décision actionnable
```

---

# 7. Problème fondamental n°5 : l’ordre du pipeline est incorrect

L’ordre actuel dans `PagePrintBuilder` est approximativement :

```text
detection région
units
regions + memberships
region_units
evidence
graph
page_intelligence
semantic_system
policies
constraints
translation_context
quality
document_comprehension
views
validation
```

Le problème est que les politiques sont compilées **avant** une vraie résolution de rôles, avant une vraie construction logique, et avant une vraie résolution sémantique.

## Ordre cible

Je recommande cet ordre :

```text
1. Normaliser page/source/geometry
2. Construire unités visuelles neutres
3. Construire régions neutres
4. Calculer memberships unit↔region
5. Collecter toutes les preuves/claims
6. Résoudre les rôles documentaires
7. Construire les structures logiques
8. Construire les unités sémantiques
9. Résoudre les conflits de compréhension
10. Compiler les politiques
11. Compiler les contraintes WYSIWYG
12. Compiler les plans aval
13. Valider fonctionnellement
14. Produire INPUT_DATA final
```

Le changement principal est :

```text
policy_compiler doit venir après role_resolver + structure_builder + semantic_builder.
```

---

# 8. Problème fondamental n°6 : `semantic_system` est encore passif

Dans `pageprint/builder.py`, `_build_semantic_system()` récupère surtout les `semantic_phrases` et `semantic_groups` déjà présents dans `page_structure`.

Donc si l’amont ne fournit rien, le semantic system reste vide.

Ce n’est pas acceptable pour PAGEPRINT.

## Nouvelle responsabilité de PAGEPRINT

`PAGEPRINT` doit construire lui-même :

```text
semantic_phrases
semantic_groups
list_items
toc_entries
index_entries
table_cells
caption_units
code_blocks
formula_units
author_entries
publisher_marks
```

Il ne doit pas attendre que `ocr_server.py` ou `LayoutV2Builder` lui donne tout.

## Module à créer

```text
pageprint/semantic_builder.py
```

Mais attention : il ne doit pas être un simple coalescer. Il doit être piloté par les rôles et le graphe.

Exemple :

```text
body_paragraph → phrases
toc_page       → toc_entries
index_page     → index_entries
table_page     → table_cells
code_page      → code_blocks + explanations
figure_page    → labels + captions
author_page    → author_entries
```

---

# 9. Problème fondamental n°7 : `PAGETRANSLATE` choisit trop

Dans `pagetranslate/builder.py`, le pipeline fait :

```text
select_translation_units
annotate_sentence_boundaries
coalesce_translation_units
attach_unit_context
translate
project
```

Cette logique signifie que `PAGETRANSLATE` prend encore des décisions de compréhension.

Ce n’est pas son rôle.

Le rôle de `PAGETRANSLATE` devrait être :

```text
lire un translation_plan
protéger les tokens
appeler le traducteur
contrôler la qualité
projeter selon render_target
```

Pas :

```text
décider phrase vs line vs block
fusionner des lignes
deviner table/index/toc/list
compenser semantic_system vide
```

## Correction structurelle

Créer :

```text
pagetranslate/translation_plan_reader.py
```

Puis :

```python
if input_data["views"].get("translation_plan"):
    units = read_translation_plan(input_data)
else:
    units = fallback_selector_and_coalescer(input_data)
```

Et le debug doit signaler :

```json
{
  "selection_mode": "translation_plan"
}
```

ou :

```json
{
  "selection_mode": "fallback_selector",
  "warning": "PAGEPRINT did not provide translation_plan"
}
```

---

# 10. Problème fondamental n°8 : le coalescer est trop puissant

`pagetranslate/coalescer.py` fusionne des unités visuelles en `semantic_phrase`.

Même s’il a été amélioré, sa philosophie reste dangereuse :

```text
absence de ponctuation + continuité apparente
→ fusion
```

Mais dans les documents réels, l’absence de ponctuation signifie souvent :

```text
titre
label
liste
tableau
index
toc
cellule
caption courte
ligne de commande
entrée technique
```

## Correction

Le coalescer doit devenir relationnel :

```python
can_join = graph_query.can_merge_for_translation(prev, current)
```

Pas seulement géométrique ou ponctuationnel.

Et il doit être interdit par défaut pour :

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

---

# 11. Problème fondamental n°9 : les vues aval sont trop faibles

`views.translation_units` contient encore trop peu d’informations.

Il faut remplacer ou compléter par :

```text
views.translation_plan
views.reconstruction_plan
views.preservation_plan
views.exclusion_plan
```

## Exemple de `translation_plan`

```json
{
  "translation_unit_id": "tp_p001_0007",
  "source_unit_ids": ["p001_line_012", "p001_phrase_012"],
  "logical_unit_id": "toc_entry_007",
  "source_text": "Image classification using MLP",
  "role": "toc_entry_title",
  "object_type": "natural_text",
  "semantic_kind": "title_fragment",
  "translation_mode": "translate",
  "translation_strategy": "layout_constrained",
  "protected_tokens": ["MLP"],
  "context": {
    "page_role": "toc",
    "section_context": "Convolutional neural networks"
  },
  "render_target": {
    "reconstruction_unit_id": "ru_p001_0007",
    "bbox": [100, 120, 300, 135],
    "style_source_unit_id": "p001_phrase_012",
    "consume_source_unit_ids": ["p001_phrase_012"]
  },
  "qa_requirements": {
    "preserve_numbers": true,
    "preserve_protected_tokens": true,
    "check_overflow": true
  }
}
```

Avec ça, `PAGETRANSLATE` devient beaucoup plus simple et fiable.

---

# 12. Problème fondamental n°10 : OCR trop binaire

Dans `pipelines/raw_extractors.py`, l’OCR est utilisé surtout si le natif est absent :

```text
if enable_ocr and image is not None and not native_available:
    OCR
```

C’est insuffisant.

Une page peut avoir du texte natif mais aussi du texte important dans les images :

```text
couverture
schéma
figure annotée
scan partiel
titre intégré dans image
logo
caption rasterisée
```

## Correction

Il faut un `OCRRoutingPolicy`.

```text
native text available ≠ OCR inutile
```

Règle :

```text
si image_dominant
ou text_density faible
ou zones image larges
ou page cover
ou diagram/charts
→ OCR ciblé sur régions image
```

Pas forcément OCR plein page. Plutôt :

```text
OCR ciblé par zones candidates
```

Module :

```text
pipelines/ocr_router.py
```

---

# 13. Problème fondamental n°11 : special_region_detector est appelé à deux endroits

Il est appelé dans :

```text
pipelines/page_understanding.py
```

et aussi dans :

```text
pageprint/detection/builder.py
```

Cela crée un risque de :

```text
double détection
latence inutile
résultats divergents
duplication de régions
effets de bord
```

## Correction

Il faut un seul propriétaire.

Je recommande :

```text
PageUnderstanding = exécute les détecteurs coûteux
PAGEPRINT = normalise les claims reçus
```

Donc `PageRegionDetectBuilder` ne devrait pas relancer le détecteur sauf si :

```text
force_detect=True
ou aucune special_region fournie
```

Signature cible :

```python
PageRegionDetectBuilder().build(
    page_structure=page_structure,
    page_image=page_image,
    pdf_page=pdf_page,
    run_detector=False,
    normalize_existing=True,
)
```

---

# 14. Problème fondamental n°12 : `policy_compiler.py` applique des règles regex trop faibles

Exemple :

```python
if re.search(r"\b(def|class|import|return|for|while|if|else|elif|function)\b", text):
    return True
```

C’est mauvais comme philosophie.

Le mot `if` dans une phrase normale ne fait pas un code block.

## Correction

Remplacer par un score multi-preuves :

```python
score = 0

if monospace:
    score += 3
if role in {"code_block", "code_line", "command_name"}:
    score += 4
if has_code_punctuation:
    score += 2
if has_function_call:
    score += 2
if has_assignment:
    score += 2
if has_path_or_file_pattern:
    score += 3
if has_many_natural_words:
    score -= 3
if page_role in {"toc", "index", "author_bio"}:
    score -= 2

return score >= 4
```

Même chose pour formule :

```text
parenthèses ≠ formule
slash ≠ code
mot technique ≠ protected_visual
```

---

# 15. Problème fondamental n°13 : les contraintes WYSIWYG sont encore trop génériques

Dans `constraint_compiler.py`, `_is_prose()` considère :

```python
role in {"body", "paragraph", None}
```

Donc une unité sans rôle peut devenir prose.

C’est dangereux.

Une unité sans rôle ne doit pas avoir une grande liberté de reflow.

## Correction

```python
def _is_prose(unit, policy):
    if _is_fixed(policy) or _is_table_cell(policy):
        return False

    role = (unit.get("understanding") or {}).get("role")
    page_role = (unit.get("understanding") or {}).get("page_role")
    layout_type = (unit.get("understanding") or {}).get("layout_type")

    if role not in {"body_paragraph", "paragraph", "body"}:
        return False

    if page_role in {"toc", "index", "cover"}:
        return False

    if layout_type in {"image_dominant", "annotated_page", "table_dominant"}:
        return False

    return bool(policy.get("translatable"))
```

Principe :

```text
inconnu = prudent / anchored
pas prose libre
```

---

# 16. Problème fondamental n°14 : pas assez de niveau document

Beaucoup de décisions ne peuvent pas être prises page par page.

Exemples :

```text
header/footer répété
watermark
publisher mark
numéro de page
running title
index
TOC
style de chapitre
terminologie récurrente
```

Actuellement, le pipeline travaille surtout page par page.

Il faut une couche :

```text
DocumentContextBuilder
```

Elle collecte sur plusieurs pages :

```text
textes répétés en haut/bas
styles récurrents
positions récurrentes
numéros de page
publisher marks
chapitre courant
table des matières
index
glossaire
terminologie
```

Puis chaque `PAGEPRINT` reçoit :

```json
{
  "document_context": {
    "repeated_headers": [...],
    "repeated_footers": [...],
    "publisher_marks": [...],
    "known_terms": [...],
    "toc_detected": true,
    "index_pages": [...]
  }
}
```

Sans contexte document, tu vas continuer à confondre footer, watermark, titre, index, etc.

---

# 17. Problème fondamental n°15 : il manque une vraie validation fonctionnelle

`pageprint/validators.py` valide surtout :

```text
clés présentes
bbox
parent_id
policy complète
unité translatable avec texte
```

C’est utile, mais pas suffisant.

Il faut des invariants métier.

## Exemples d’invariants PAGEPRINT

```text
1. role=None interdit dans translation_plan.
2. word/char interdits dans translation_plan.
3. une unité preserve_as_visual_overlay ne doit pas être majoritairement texte naturel.
4. une page table doit produire table_units.
5. une page index doit produire index_entries.
6. une page toc doit produire toc_entries.
7. un bloc mixte ne doit pas devenir une unité de traduction brute.
8. une région protégée partielle ne doit pas rendre un parent background_only.
9. semantic_system ne doit pas être vide pour une page prose.
10. reconstruction_plan doit avoir des render_targets.
```

## Exemples d’invariants PAGETRANSLATE

```text
1. aucune unité role=None.
2. aucune unité object_type=None.
3. aucune traduction de command/path/page_reference.
4. aucune fusion à travers table/index/toc/list.
5. chaque translation_unit doit avoir render_target.
6. chaque reconstruction_unit doit garder role/object_type.
7. chaque protected_token doit être restauré.
8. unchanged doit être suspect si source_lang != target_lang.
```

---

# 18. Refonte cible pragmatique

## Nouvelle architecture PAGEPRINT

```text
pageprint/
├── schema.py
├── builder.py
├── normalizer.py
├── unit_factory.py
├── detection/
│   ├── builder.py
│   └── claims.py
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
├── functional_validator.py
└── serializers.py
```

## Nouvelle architecture PAGETRANSLATE

```text
pagetranslate/
├── builder.py
├── translation_plan_reader.py
├── fallback_selector.py
├── protection.py
├── terminology.py
├── translator_bridge.py
├── quality.py
├── projection.py
├── functional_validator.py
└── schema.py
```

---

# 19. Nouveau pipeline cible

```text
SourceLoader
→ PageRenderer
→ RawExtractors
→ OCRRouter
→ PageUnderstanding
→ PageRegionClaims
→ PAGEPRINT
    1. units visuelles
    2. régions
    3. memberships
    4. evidence claims
    5. role resolver
    6. structure builders
    7. semantic builder
    8. policy compiler
    9. preservation compiler
    10. constraints
    11. plans aval
    12. validation fonctionnelle
→ PAGETRANSLATE
    1. read translation_plan
    2. protect tokens
    3. translate
    4. QA
    5. project to reconstruction_plan
→ Reconstructor
    1. read reconstruction_plan
    2. preserve overlays
    3. redraw translated text
    4. QA visual
```

---

# 20. Actions de refonte prioritaires

## Sprint 0 — Sécurisation immédiate

Objectif : éviter les mauvaises décisions irréversibles.

À faire :

```text
1. Empêcher PageRegionDetectBuilder de relancer le détecteur si special_regions existe.
2. Changer normalize_detected_region : ne plus écrire skip_translation directement.
3. Remplacer protected_visual direct par claim.
4. Corriger _is_prose : role None ne doit pas être prose.
5. Modifier policy_compiler : région spéciale ≠ background_only automatique.
6. Ajouter functional_valid dans l’audit.
```

Livrable :

```text
Même pipeline, mais moins destructeur.
```

---

## Sprint 1 — Rôles et préservation

Objectif : arrêter de traduire à l’aveugle.

À faire :

```text
1. role_resolver.py
2. preservation_mode
3. policy_compiler basé sur role + resolved evidence
4. exclusion_plan
5. reconstruction_units conservent role/object_type
```

Rôles minimaux :

```text
body_paragraph
title
section_heading
list_item
list_marker
figure_caption
table_caption
table_cell
code_block
code_line
command_name
path
formula_expression
toc_entry
toc_page_reference
index_entry
index_page_reference
author_name
publisher_mark
page_header
page_footer
```

---

## Sprint 2 — Structures logiques

Objectif : ne plus traiter toutes les pages comme phrases.

À faire :

```text
toc_builder
index_builder
table_builder
caption_builder
list_builder
code_builder
```

Sorties :

```text
logical_units[]
toc_entries[]
index_entries[]
tables[]
captions[]
code_blocks[]
list_items[]
```

---

## Sprint 3 — Translation plan

Objectif : rendre PAGETRANSLATE consommateur.

À faire :

```text
1. view_compiler.py
2. views.translation_plan
3. views.preservation_plan
4. views.reconstruction_plan
5. pagetranslate/translation_plan_reader.py
6. selector/coalescer en fallback uniquement
```

Critère :

```text
PAGETRANSLATE ne doit plus choisir phrase/line/block en mode normal.
```

---

## Sprint 4 — OCR routing + document context

Objectif : gérer pages image/couvertures/headers/footers.

À faire :

```text
1. OCRRouter
2. OCR ciblé sur image regions
3. DocumentContextBuilder
4. repeated header/footer detector
5. publisher mark detector multi-pages
6. cover_page detector
```

---

## Sprint 5 — Validation et tests golden

Objectif : arrêter les corrections au hasard.

Créer un corpus golden :

```text
TOC page
index page
table command page
body paragraph page
figure/caption page
diagram label page
author bio page
cover page
code page
formula page
```

Pour chaque page :

```text
expected page_role
expected logical_units
expected translation_plan
expected preservation_plan
expected reconstruction_plan
expected exclusions
```

---

# 21. Critères de succès avant moteur de traduction

On pourra passer sérieusement au moteur de traduction seulement quand :

```text
role_none_translation_units = 0
word_char_in_translation_plan = 0
block_mixed_translation_units = 0
generic_coalescer_usage < 5 %
natural_text_preserve_as_visual_false_positive ≈ 0
tables detected when present
index detected when present
toc entries structured
captions split label/number/text
commands/paths/code preserved
publisher marks excluded
reconstruction_units keep role/object_type/render_contract
functional_valid = true
```

---

# 22. Conclusion stratégique

La refonte de fond doit changer le centre du système.

Aujourd’hui, le pipeline ressemble encore à :

```text
extraction visuelle
+ heuristiques
+ politique directe
+ traduction corrective
```

Il doit devenir :

```text
observations
+ preuves
+ rôles
+ structures logiques
+ plans aval
+ validation fonctionnelle
```

La phrase clé :

```text
PAGEPRINT ne doit pas seulement décrire la page.
PAGEPRINT doit compiler la page en instructions fiables pour les modules aval.
```

Et :

```text
PAGETRANSLATE ne doit pas comprendre le document.
PAGETRANSLATE doit exécuter un plan de traduction.
```

Donc non, mon analyse précédente n’était pas complète. La vraie correction de fond n’est pas seulement d’ajouter quelques modules. C’est de transformer `PAGEPRINT` en **compilateur documentaire** et `PAGETRANSLATE` en **exécuteur contrôlé de plan de traduction**.

