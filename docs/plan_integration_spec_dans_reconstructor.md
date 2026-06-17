# Integration De La Spec Hierarchique Dans Le Reconstructor Actuel

## Objectif

Implementer la specification de reconstruction hierarchique dans le moteur actuel sans faire un big bang.

Principe:

- on garde l'orchestrateur actuel
- on garde les briques utiles de fond, police, mesure, overlays
- on ajoute un nouveau pipeline par bloc
- on migre progressivement les familles de rendu


## Etat De Depart

Le moteur actuel repose surtout sur:

- `reconstruct()`
- `_reconstruct_translated_anchored()`
- `_extract_block_slot_items()`
- `_render_block_slots()`

Le point fort actuel:

- beaucoup d'heuristiques utiles existent deja
- la gestion du fond et des overlays est riche
- le code immuable et plusieurs cas speciaux sont deja proteges

Le point faible:

- la reconstruction est encore pilotee par des items plats et des flags heuristiques
- les donnees semantiques extraites ne pilotent pas encore vraiment le rendu


## Ce Qu'il Faut Garder

- gestion du fond maitre
- overlays immuables
- selection police / font fallback
- mesure typographique
- alignement
- whiteout / restore background
- verrouillage du code immuable


## Ce Qu'il Faut Remplacer Progressivement

- `_extract_block_slot_items()` comme coeur logique
- `_render_block_slots()` comme moteur principal unique

Ils doivent devenir:

- soit des chemins legacy
- soit des helpers reutilises par les nouveaux renderers


## Architecture A Introduire

Nouveaux objets:

- `BlockGeometryContext`
- `LineTemplate`
- `PlacableUnit`
- `GraphEdge`
- `PlacementCursor`
- `PlacementResult`
- `BlockRenderOp`
- `BlockReconstructionPlan`

Nouvelles fonctions:

- `_build_page_reconstruction_context()`
- `_iter_renderable_blocks()`
- `_classify_block_for_reconstruction()`
- `_build_block_geometry_context()`
- `_build_line_templates()`
- `_collect_block_semantic_payload()`
- `_normalize_placable_units()`
- `_build_reconstruction_graph()`
- `_build_block_reconstruction_plan()`
- `_select_block_renderer()`
- `_render_hierarchical_block_plan()`
- `_validate_block_layout()`
- `_commit_block_draw_ops()`


## Renderers Cibles

- `EditorialBlockRenderer`
- `CodeBlockRenderer`
- `TableBlockRenderer`
- puis plus tard:
  - `HeadingBlockRenderer`
  - `CaptionBlockRenderer`
  - `AnnotationBlockRenderer`


## Strategie D'Integration

### Phase 1

Introduire l'ossature sans casser le moteur:

- dataclasses
- plan par bloc
- classification
- line templates
- units normalisees
- graphe minimal

### Phase 2

Brancher le nouveau moteur seulement pour:

- blocs `body`
- traduits
- editoriaux
- supportes par les donnees semantiques

Tous les autres blocs restent sur le moteur legacy.

### Phase 3

Utiliser vraiment:

- `semantic_phrases`
- `semantic_groups`
- `semantic_runs`
- `semantic_spans`

comme coeur du rendu editorial.

### Phase 4

Migrer:

- code
- tableau
- caption
- annotation
- heading


## Regle De Compatibilite

Le dispatcher doit toujours pouvoir faire:

- nouveau moteur si bloc supporte
- sinon fallback legacy

Donc la migration doit rester bloc par bloc.


## Ce Qui A Ete Branche Dans Cette Premiere Tranche

- mode `LAYOUT_HIERARCHICAL_RECONSTRUCTION`
- plan de reconstruction par bloc
- classification des blocs
- generation des `LineTemplate`
- normalisation des `PlacableUnit`
- graphe minimal des relations
- `EditorialBlockRenderer`
- `BlockRenderOp`
- validation locale simple
- branchement progressif dans `_reconstruct_translated_anchored()`


## Limitations Actuelles De Cette Premiere Tranche

- seul le rendu editorial traduit simple est branche
- `CodeBlockRenderer` et `TableBlockRenderer` sont encore des stubs
- la justification reste basique
- les `semantic_groups/runs/spans` sont consommes de facon incrementale, pas encore exhaustive
- les draw ops sont minimaux


## Decoupage En Plan D'Implementation

### Lot 1

- stabiliser `BlockReconstructionPlan`
- enrichir `_normalize_placable_units()`
- fiabiliser les textes traduits au niveau phrase/groupe/run/span

### Lot 2

- introduire les vraies transitions de paragraphe
- mieux utiliser `editorial_relations`
- mieux utiliser `expression_relations`
- gerer `CONTINUE_INLINE`, `NEW_LINE`, `NEW_PARAGRAPH`, `HEADING_TO_BODY`

### Lot 3

- factoriser le layout editorial dans `EditorialBlockRenderer`
- ajouter justification robuste
- ajouter gestion fine du centrage et du right alignment fiable

### Lot 4

- implementer `CodeBlockRenderer`
- brancher les overlays exacts comme draw ops
- verrouiller totalement le code non traduit

### Lot 5

- implementer `TableBlockRenderer`
- reconstruire cellule par cellule
- arreter le reflow paragraphe des tableaux

### Lot 6

- ajouter `HeadingBlockRenderer`
- ajouter `CaptionBlockRenderer`
- ajouter `AnnotationBlockRenderer`

### Lot 7

- renforcer la validation locale
- produire des findings lisibles
- brancher le QA bloc par bloc


## Resultat Attendu

Le reconstructor doit progressivement passer de:

- heuristiques par items et slots

a:

- reconstruction hierarchique bloc par bloc
- pilotage par structure semantique
- renderers specialises
- draw ops explicites
- validation locale avant commit
