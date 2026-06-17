Oui, je comprends exactement. Et le bon diagnostic est le suivant :

**`pagereconstruct/` doit devenir l’unité qui remplace progressivement le cœur de `reconstructor.py`, mais sans jeter ses bonnes briques.**
L’ancien moteur contient beaucoup d’intelligence utile : fonds nettoyés, inpainting, overlays immuables, contrats de rendu, mesure typographique, fallback police, validation overflow/collision, rendu PyMuPDF. Mais tout cela est mélangé dans un fichier énorme.

J’ai inspecté `rev_08` : la base est saine. Les tests ciblés `pageprint`, `pagetranslate`, `functional`, `pipelines` passent : **90 passed**. Donc on peut avancer sur `pagereconstruct/` sans réouvrir une refonte générale.

---

# 1. Verdict net

Il faut créer :

```text
pagereconstruct/
```

comme **troisième unité canonique** :

```text
pageprint/
  → produit INPUT_DATA canonique

pagetranslate/
  → produit translated_input_data + views.reconstruction_units

pagereconstruct/
  → produit une page reconstruite WYSIWYG + audit de fidélité
```

La règle fondamentale doit être :

```text
pagereconstruct ne devine pas la traduction.
pagereconstruct ne relit pas naïvement units[].content.translated_text.
pagereconstruct consomme views.reconstruction_units comme source de vérité textuelle après traduction.
```

C’est critique. Sinon on réintroduit les anciens bugs : doublons, parents rendus plusieurs fois, enfants rendus en même temps que les semantic phrases, blocs traduits partiellement, perte de rôle, perte de style.

---

# 2. Ce que faisait l’ancien système

Avant, le chemin était à peu près :

```text
ocr_server.py
  → extraction native PDF / OCR
  → fusion final_blocks
  → enrichissement sémantique
  → background master / mask / inpainting
  → immutable_overlays
  → traduction
  → DocumentReconstructor.reconstruct()
  → PDF final
```

Les gros objets anciens étaient :

```text
final_blocks
background_path
source_image_path
immutable_overlays
semantic_phrases
semantic_groups
semantic_runs
semantic_spans
document_object_contract
descriptor_v3
layout_attributes
style_attributes
reconstruction_contracts
```

Dans `reconstructor.py`, on trouve déjà une architecture embryonnaire de reconstruction moderne :

```text
BlockGeometryContext
LineTemplate
PlacableUnit
GraphEdge
PlacementCursor
PlacementResult
BlockRenderOp
BlockReconstructionPlan
BlockSemanticProfile
```

C’est précisément ce qu’il faut extraire vers `pagereconstruct/`.

---

# 3. Ce qu’il faut garder de l’ancien moteur

À conserver, car ce sont des actifs techniques réels :

```text
1. Insertion du background nettoyé.
2. Gestion des overlays immuables.
3. Restauration locale du fond / inpainting.
4. Whiteout local quand le fond propre n’est pas fiable.
5. Résolution des polices.
6. Fallback Unicode.
7. Mesure typographique.
8. Shrink borné.
9. Validation overflow / collision / protected overlap.
10. Contrats de rendu : paragraph, caption, table_cell, code_block, formula_block, toc_entry.
11. Renderers spécialisés : EditorialBlockRenderer, TableBlockRenderer, CodeBlockRenderer.
12. Candidats de rendu et fallback contrôlé.
```

À ne pas garder comme cœur :

```text
1. Lecture directe de final_blocks comme source principale.
2. Rendu depuis blocks[] sans passer par reconstruction_units.
3. Agrégation aveugle translated_text depuis page/block/line/phrase/span.
4. Fallbacks silencieux qui masquent les erreurs de contrat.
5. Mélange serveur HTTP / extraction / traduction / reconstruction.
```

---

# 4. Rôle exact de `pagereconstruct/`

`pagereconstruct/` doit prendre en entrée :

```text
translated_input_data
```

c’est-à-dire la sortie de `pagetranslate`.

Il doit lire :

```text
translated_input_data.views.reconstruction_units
translated_input_data.views.reconstruction_plan
translated_input_data.views.preservation_plan
translated_input_data.views.exclusion_plan
translated_input_data.units
translated_input_data.regions
translated_input_data.graph
translated_input_data.assets
translated_input_data.visual_layers
translated_input_data.reconstruction_constraints
translated_input_data.page_intelligence
translated_input_data.style_system
translated_input_data.semantic_system
```

Et produire :

```text
PAGE_RECONSTRUCT_RESULT
```

avec :

```text
pdf_page ou page_ops
render_ops
background_ops
text_ops
overlay_ops
preservation_ops
audit
quality
risks
debug
```

---

# 5. Architecture proposée

Structure cible :

```text
pagereconstruct/
├── __init__.py
├── schema.py
├── builder.py
├── input_adapter.py
├── plan_compiler.py
├── background_resolver.py
├── style_resolver.py
├── font_resolver_bridge.py
├── geometry.py
├── flow_resolver.py
├── block_planner.py
├── line_template_builder.py
├── placement_engine.py
├── renderers/
│   ├── __init__.py
│   ├── base.py
│   ├── editorial.py
│   ├── heading.py
│   ├── caption.py
│   ├── table.py
│   ├── code.py
│   ├── formula.py
│   ├── anchored_label.py
│   └── preservation.py
├── ops.py
├── pdf_backend.py
├── validator.py
├── quality.py
├── fallback.py
├── debug_exporter.py
└── legacy_bridge.py
```

---

# 6. Le contrat central : `ReconstructionUnit`

Il faut figer un objet minimal consommable par le reconstructeur :

```python
{
    "reconstruction_unit_id": "...",
    "unit_id": "...",
    "translation_unit_id": "...",
    "source_unit_ids": [...],

    "role": "body_paragraph | title | table_body_cell | figure_caption | toc_entry_title | ...",
    "object_type": "...",
    "semantic_kind": "...",

    "source_text": "...",
    "translated_text": "...",

    "bbox": [x0, y0, x1, y1],
    "bbox_unit": "pt",

    "style": {...},
    "style_source_unit_id": "...",

    "render_contract": {
        "mode": "translated_text | fixed_preserve | background_only | cell_locked | anchored_text",
        "strategy": "semantic_reflow | layout_constrained | paragraph_flow | toc_row_layout",
        "overflow_policy": "shrink_or_reflow | shrink_only | preserve_if_overflows | fail",
        "line_break_policy": "semantic_reflow | preserve_source_lines | single_line_or_shrink",
        "bbox_policy": "locked | expandable | rebalanced",
        "background_policy": "clean_background | local_inpaint | whiteout | none"
    },

    "layout_budget": {
        "width": ...,
        "height": ...,
        "area": ...,
        "max_lines": ...,
        "min_font_size": ...,
        "max_font_size": ...,
        "allow_width_expansion": false,
        "allow_height_expansion": true
    },

    "quality": {...}
}
```

Ce contrat doit être plus riche que l’actuel `views.reconstruction_units`, qui reste encore trop léger.

---

# 7. Le principe de rendu

Le pipeline `pagereconstruct` doit être strict :

```text
1. Charger translated_input_data.
2. Valider que views.reconstruction_units existe.
3. Résoudre le fond de page.
4. Insérer le background propre.
5. Construire un PageReconstructionPlan.
6. Construire des BlockReconstructionPlan.
7. Pour chaque bloc :
   - choisir renderer
   - préparer fond local
   - placer texte traduit
   - préserver zones spéciales
   - valider overflow/collisions
   - sélectionner meilleur candidat
8. Réinsérer overlays immuables restants.
9. Faire QA visuelle/textuelle.
10. Retourner PAGE_RECONSTRUCT_RESULT.
```

En pseudo-pipeline :

```text
translated_input_data
  ↓
PageReconstructBuilder
  ↓
InputAdapter
  ↓
PagePlanCompiler
  ↓
BackgroundResolver
  ↓
BlockPlanner
  ↓
RendererDispatcher
  ↓
PDFBackend
  ↓
ReconstructionValidator
  ↓
PAGE_RECONSTRUCT_RESULT
```

---

# 8. Décision importante : ne pas reconstruire depuis `blocks[]`

Dans `rev_08`, `reconstructor.py` lit encore principalement :

```text
page_data["blocks"]
```

Pour `pagereconstruct`, il faut changer la hiérarchie de vérité :

```text
1. views.reconstruction_units          source textuelle post-traduction
2. views.reconstruction_plan           intention de rendu
3. views.preservation_plan             éléments à préserver
4. views.exclusion_plan                éléments à exclure
5. units[]                             géométrie/style/source détaillée
6. blocks legacy                       fallback seulement
```

Donc :

```text
blocks[] ne doit plus être la source principale.
blocks[] devient une vue legacy ou un fallback temporaire.
```

---

# 9. Les familles de rendu à gérer dès V1

## 9.1 `editorial`

Pour paragraphes classiques :

```text
body_paragraph
list_item
author_bio
normal prose
```

Renderer :

```text
EditorialRenderer
```

Politique :

```text
semantic_reflow
respect bbox
respect style dominant
shrink borné
reflow dans le bloc
```

## 9.2 `heading`

Pour titres :

```text
title
section_heading
chapter_heading
toc_heading
```

Renderer :

```text
HeadingRenderer
```

Politique :

```text
single/multi-line controlled
alignement source
fort respect typographique
pas de compression agressive
```

## 9.3 `caption`

Pour légendes :

```text
figure_caption
table_caption
```

Renderer :

```text
CaptionRenderer
```

Politique :

```text
préserver label/numéro
traduire seulement la description
recomposer label + texte traduit
```

## 9.4 `table`

Pour cellules :

```text
table_header_cell
table_body_cell
table_numeric_cell
table_command_cell
```

Renderer :

```text
TableRenderer
```

Politique :

```text
cell_locked
aucun débordement hors cellule
shrink borné
préserver numériques/commandes
traduire descriptions seulement
```

## 9.5 `code`

Pour code, commandes, chemins :

```text
code_block
command_name
path
file_name
technical_identifier
```

Renderer :

```text
CodeRenderer
```

Politique :

```text
fixed_preserve
monospace
pas de traduction
pas de reflow sémantique
```

## 9.6 `formula`

Pour formules :

```text
formula_block
inline_formula
equation
chemical_formula
```

Renderer :

```text
FormulaRenderer
```

Politique :

```text
background_only ou fixed_preserve
ne jamais traduire
préserver exactement
```

## 9.7 `anchored_label`

Pour labels de figures/diagrammes :

```text
diagram_label
axis_label
legend_label
short_label
```

Renderer :

```text
AnchoredLabelRenderer
```

Politique :

```text
bbox locked
single-line if possible
shrink only
pas de déplacement libre
```

## 9.8 `preservation`

Pour logos, watermark, publisher mark, page number :

```text
publisher_mark
watermark
page_number
logo_text
```

Renderer :

```text
PreservationRenderer
```

Politique :

```text
exclude_as_artifact
fixed_preserve
ou preserve_as_visual_overlay
selon plan
```

---

# 10. Gestion des fonds et trames

C’est central pour ton WYSIWYG.

Il faut séparer 4 couches :

```text
Layer 0 — source render original
Layer 1 — clean background / background master
Layer 2 — local patches / inpainting / whiteout
Layer 3 — text translated
Layer 4 — immutable overlays
```

`pagereconstruct/background_resolver.py` doit décider :

```text
si background_path fiable :
    insérer background_path comme fond de page

si zone texte à remplacer et fond local propre disponible :
    appliquer local_inpaint

si fond local non fiable mais zone simple :
    whiteout avec couleur locale échantillonnée

si zone spéciale non traduite :
    conserver overlay source

si image/figure/table visuelle :
    préserver en background/overlay, ne pas effacer
```

Erreur à éviter :

```text
ne jamais réutiliser une zone source contenant l’ancien texte anglais
si cette zone doit recevoir le texte traduit.
```

Donc il faut un audit :

```text
background_text_leak_check
```

qui détecte si du texte source reste visible dans les zones traduites.

---

# 11. Gestion des zones spéciales non traduites

`pagereconstruct` doit appliquer explicitement :

```text
views.preservation_plan
views.exclusion_plan
```

Exemples :

```text
MANNING                       → publisher_mark → exclure ou préserver visuellement
numéros de page                → fixed_preserve
formules                       → fixed_preserve/background_only
code                           → fixed_preserve
URL/email/DOI                  → fixed_preserve
watermark                      → exclusion/preservation selon politique
image                          → background_only
diagram labels traduisibles    → anchored_text
```

Règle :

```text
Une zone spéciale non traduite ne doit jamais être effacée par le patch de fond d’un bloc voisin.
```

Il faut donc construire un :

```text
ProtectedRegionIndex
```

à partir de :

```text
regions
preservation_plan
exclusion_plan
visual_layers.overlays
assets.immutable_overlays
```

---

# 12. Ajustement des blocs et positions

Il faut distinguer 4 niveaux de liberté :

```text
bbox_locked
    aucune modification de position/taille

height_expandable
    hauteur du bloc peut augmenter dans une zone sûre

width_expandable
    largeur peut augmenter si pas de collision

page_reflow
    déplacement coordonné de plusieurs blocs

absolute_preserve
    aucune modification, seulement shrink ou fail
```

Pour V1, je recommande :

```text
par défaut : bbox_locked
body_paragraph : height_expandable si safe
heading : bbox_locked + shrink modéré
caption : bbox_locked ou height_expandable léger
table_cell : bbox_locked strict
figure_label : bbox_locked strict
code/formula : absolute_preserve
```

Ne pas faire de grand reflow de page au début. C’est trop dangereux. D’abord réussir :

```text
rendu fidèle bloc par bloc
```

Puis seulement ensuite :

```text
layout rebalance multi-blocs
```

---

# 13. Le solveur de placement

Il faut que `placement_engine.py` fonctionne par candidats :

```text
Candidate A : taille source, reflow normal
Candidate B : shrink léger
Candidate C : reflow plus compact
Candidate D : line preserve
Candidate E : fallback anchored
Candidate F : fail needs_review
```

Chaque candidat reçoit un score :

```text
+ fidélité style
+ fidélité position
+ couverture texte
+ absence overflow
+ absence collision
+ lisibilité police
+ absence source text leak
```

Pénalités :

```text
overflow
font too small
line overlap
protected overlap
text missing
background leak
style drift
anchor drift
```

Décision :

```text
prendre le meilleur candidat valide
sinon ne pas masquer l’échec : status = needs_manual_review
```

Il ne faut pas que le fallback “répare” silencieusement en produisant une page visuellement fausse.

---

# 14. Contrat de sortie

Créer dans `pagereconstruct/schema.py` :

```python
PAGERECONSTRUCT_SCHEMA_VERSION = "pagereconstruct.output.v1"
```

Sortie :

```python
{
    "schema_version": "pagereconstruct.output.v1",
    "source_schema_version": "pagetranslate.output.v1",
    "page_index": 0,
    "status": "ok | review | ko",

    "output": {
        "pdf_path": "...",
        "page_image_path": "...",
        "debug_overlay_path": "..."
    },

    "render_summary": {
        "background_inserted": true,
        "translated_units_rendered": 42,
        "preserved_units_rendered": 12,
        "excluded_units": 3,
        "fallback_count": 1
    },

    "ops": {
        "background_ops": [...],
        "patch_ops": [...],
        "text_ops": [...],
        "overlay_ops": [...]
    },

    "quality": {
        "text_coverage_ratio": 1.0,
        "overflow_count": 0,
        "collision_count": 0,
        "font_too_small_count": 0,
        "background_leak_risk": "low",
        "wysiwyg_score": 0.94
    },

    "findings": [],
    "debug": {}
}
```

---

# 15. Ce qu’il faut extraire de `reconstructor.py`

À déplacer progressivement :

```text
vers pagereconstruct/schema.py
    BlockGeometryContext
    LineTemplate
    PlacableUnit
    GraphEdge
    PlacementCursor
    PlacementResult
    BlockRenderOp
    BlockReconstructionPlan
    BlockSemanticProfile
    CandidateScore
    RenderCandidate
    BlockRenderVerdict

vers pagereconstruct/plan_compiler.py
    _build_page_reconstruction_context
    _build_block_reconstruction_plan
    _build_block_geometry_context
    _build_line_templates
    _collect_block_semantic_payload
    _normalize_placable_units
    _build_reconstruction_graph

vers pagereconstruct/renderers/
    BaseBlockRenderer
    StructuredContractRenderer
    EditorialBlockRenderer
    HeadingBlockRenderer
    CaptionBlockRenderer
    AnnotationBlockRenderer
    CodeBlockRenderer
    TableBlockRenderer

vers pagereconstruct/validator.py
    _validate_block_layout
    _block_render_verdict
    select_best_candidate

vers pagereconstruct/pdf_backend.py
    _commit_block_draw_ops
    page.insert_text / insert_textbox / insert_image wrappers

vers pagereconstruct/background_resolver.py
    _insert_page_background
    _text_background_patch_ops_for_plan
    local inpaint
    whiteout
    background audit
```

---

# 16. Ce qu’il faut corriger avant ou pendant l’extraction

## P0 — Source de vérité reconstruction

`pagereconstruct` doit consommer :

```text
translated_input_data.views.reconstruction_units
```

Pas :

```text
units[].content.translated_text
```

Les parents peuvent garder un résumé traduit pour audit, mais pas pour rendu.

## P0 — Style dans reconstruction_units

S’assurer que chaque reconstruction unit porte ou résout :

```text
visual.style
font
font_size
color
bold/italic
line_height
alignment
```

Actuellement, `pagetranslate/projection.py` fait déjà mieux, mais il faut durcir le contrat : si le style est absent, `pagereconstruct` doit le récupérer par `style_source_unit_id`.

## P0 — Pas de doublons

Une traduction sélectionnée doit produire une seule unité de rendu principale.

À tester :

```text
semantic_phrase couvrant 3 phrases visuelles
→ 1 reconstruction_unit
→ les 3 source_unit_ids sont marqués consumed
→ aucun enfant n’est rendu séparément
```

## P0 — Zones protégées

Avant tout rendu de texte, construire :

```text
protected_region_index
```

Sinon risque d’effacer formules, logos, images, watermark ou labels non traduits.

## P0 — Background leak

Après patch/inpainting/whiteout, vérifier que la zone traduite ne contient pas encore l’ancien texte source.

---

# 17. Stratégie d’implémentation

Ne pas faire un “big bang”. Faire une extraction contrôlée.

## Lot 1 — Squelette `pagereconstruct/`

Créer :

```text
pagereconstruct/schema.py
pagereconstruct/builder.py
pagereconstruct/input_adapter.py
pagereconstruct/ops.py
pagereconstruct/pdf_backend.py
pagereconstruct/validator.py
```

Objectif :

```text
prendre translated_input_data
lire reconstruction_units
produire un plan
ne rien rendre encore de complexe
```

Tests :

```text
test_pagereconstruct_requires_reconstruction_units
test_pagereconstruct_reads_translated_input_data
test_pagereconstruct_does_not_read_parent_translated_text
```

## Lot 2 — Background et overlays

Créer :

```text
background_resolver.py
protected_region_index.py
```

Objectif :

```text
fond propre inséré
overlays immuables préservés
zones exclues non effacées
```

Tests :

```text
test_background_inserted_once
test_immutable_overlay_preserved
test_translated_text_patch_does_not_cover_protected_region
```

## Lot 3 — Rendu texte simple

Créer :

```text
renderers/editorial.py
style_resolver.py
geometry.py
```

Objectif :

```text
rendre body_paragraph / title / section_heading
avec bbox, style, shrink simple
```

Tests :

```text
test_body_paragraph_rendered_once
test_heading_keeps_alignment
test_style_resolved_from_style_source_unit_id
```

## Lot 4 — Validation locale

Créer :

```text
validator.py
quality.py
fallback.py
```

Objectif :

```text
détecter overflow, collision, protected overlap, font too small
```

Tests :

```text
test_overflow_detected
test_protected_overlap_detected
test_best_candidate_selected
test_failed_candidate_not_silently_committed
```

## Lot 5 — Tables / captions / labels

Créer :

```text
renderers/table.py
renderers/caption.py
renderers/anchored_label.py
```

Objectif :

```text
cell_locked pour tables
caption recomposée proprement
labels de diagramme ancrés
```

Tests :

```text
test_table_cell_locked
test_table_numeric_cell_preserved
test_caption_label_preserved_text_translated
test_diagram_label_anchored
```

## Lot 6 — Code / formules / préservation

Créer :

```text
renderers/code.py
renderers/formula.py
renderers/preservation.py
```

Objectif :

```text
ne jamais traduire ni déformer les objets immuables
```

Tests :

```text
test_code_not_translated
test_formula_preserved
test_url_preserved
test_publisher_mark_excluded_or_preserved_by_policy
```

## Lot 7 — Bridge legacy

Créer :

```text
legacy_bridge.py
```

Objectif :

```text
convertir ancien page_data vers translated_input_data minimal
ou permettre à ocr_server.py d’appeler pagereconstruct
```

Mais attention :

```text
legacy_bridge = transition
pas source de vérité permanente
```

---

# 18. Signature API recommandée

Dans `pagereconstruct/__init__.py` :

```python
from .builder import PageReconstructBuilder, reconstruct_page
```

Dans `builder.py` :

```python
class PageReconstructBuilder:
    def build(
        self,
        translated_input_data: dict,
        *,
        output_path: str | None = None,
        render_pdf: bool = True,
        debug: bool = False,
    ) -> dict:
        ...
```

Fonction courte :

```python
def reconstruct_page(
    translated_input_data: dict,
    *,
    output_path: str | None = None,
    render_pdf: bool = True,
    debug: bool = False,
) -> dict:
    return PageReconstructBuilder().build(
        translated_input_data,
        output_path=output_path,
        render_pdf=render_pdf,
        debug=debug,
    )
```

Pour document complet :

```python
def reconstruct_document(
    translated_pages: list[dict],
    *,
    output_path: str,
    debug: bool = False,
) -> dict:
    ...
```

---

# 19. Nouvelle chaîne cible

À terme :

```text
PipelineOrchestrator
  → PAGEPRINT
  → PAGETRANSLATE
  → PAGERECONSTRUCT
  → QA / export
```

Ancien `/reconstruct` dans `ocr_server.py` doit devenir un simple wrapper :

```python
@app.post("/reconstruct")
async def reconstruct_document(data: dict, ...):
    result = pagereconstruct.reconstruct_document(...)
    return JSONResponse(...)
```

`ocr_server.py` ne doit plus contenir la logique métier.

---

# 20. Décision finale

Il faut avancer avec cette décision :

```text
pagereconstruct/ sera le reconstructeur canonique V1.
reconstructor.py devient legacy + réserve de code à extraire.
views.reconstruction_units devient la source de vérité post-traduction.
background + overlays + protected regions deviennent des couches explicites.
chaque texte rendu doit venir d’un contrat, pas d’une heuristique libre.
```

La bonne philosophie :

```text
PAGEPRINT comprend la page.
PAGETRANSLATE transforme le texte.
PAGERECONSTRUCT remet le texte traduit dans la page sans trahir la géométrie, les fonds, les styles, ni les objets intouchables.
```

---

# 21. TODO list directe pour Claude CLI / implémentation

```text
[ ] Créer le dossier pagereconstruct/
[ ] Créer pagereconstruct/schema.py avec les dataclasses extraites de reconstructor.py.
[ ] Créer pagereconstruct/builder.py avec PageReconstructBuilder.
[ ] Créer pagereconstruct/input_adapter.py qui lit translated_input_data.views.reconstruction_units.
[ ] Interdire le rendu depuis units[].content.translated_text sauf fallback explicitement activé.
[ ] Créer pagereconstruct/ops.py avec RenderOp, TextOp, BackgroundOp, OverlayOp.
[ ] Créer pagereconstruct/pdf_backend.py pour isoler PyMuPDF.
[ ] Créer pagereconstruct/background_resolver.py.
[ ] Créer pagereconstruct/protected_region_index.py.
[ ] Créer pagereconstruct/style_resolver.py.
[ ] Créer pagereconstruct/plan_compiler.py.
[ ] Créer pagereconstruct/block_planner.py.
[ ] Créer pagereconstruct/line_template_builder.py.
[ ] Créer pagereconstruct/placement_engine.py.
[ ] Créer pagereconstruct/renderers/base.py.
[ ] Migrer BaseBlockRenderer depuis reconstructor.py.
[ ] Migrer EditorialBlockRenderer.
[ ] Migrer HeadingBlockRenderer.
[ ] Migrer CaptionBlockRenderer.
[ ] Migrer TableBlockRenderer.
[ ] Migrer CodeBlockRenderer.
[ ] Ajouter FormulaRenderer.
[ ] Ajouter PreservationRenderer.
[ ] Créer pagereconstruct/validator.py.
[ ] Créer pagereconstruct/quality.py.
[ ] Créer pagereconstruct/fallback.py.
[ ] Créer pagereconstruct/debug_exporter.py.
[ ] Créer pagereconstruct/legacy_bridge.py.
[ ] Ajouter tests/pagereconstruct/.
[ ] Ajouter test : reconstruction_units obligatoire.
[ ] Ajouter test : aucune duplication parent/enfant.
[ ] Ajouter test : style récupéré depuis visual.style.
[ ] Ajouter test : source_unit_ids consumed non rendus séparément.
[ ] Ajouter test : protected region non effacée.
[ ] Ajouter test : table_cell reste cell_locked.
[ ] Ajouter test : formula/code/url préservés.
[ ] Ajouter test : publisher_mark exclu/preservé selon policy.
[ ] Ajouter test : overflow détecté.
[ ] Ajouter test : fallback non silencieux.
[ ] Brancher pagereconstruct dans pipelines/orchestrator.py.
[ ] Réduire ocr_server.py à un wrapper API.
```

Conclusion dure : **on est prêt à créer `pagereconstruct/`, mais il faut résister à la tentation de juste déplacer `reconstructor.py`.** Il faut extraire les bonnes briques, puis imposer le nouveau contrat. Sinon on déplacera le monolithe sans résoudre le problème.

