# Specification Detailee De Reconstruction Hierarchique

## But

Passer d'une reconstruction basee sur un texte global par bloc a une reconstruction contrainte, pilotee par les unites extraites:

- bloc
- lignes
- phrases semantiques
- spans semantiques
- runs semantiques
- groupes semantiques

Le principe central:

- le bloc reste fixe
- la page reste fixe
- la reconstruction se fait bloc par bloc
- a l'interieur du bloc, le rendu se fait unite par unite
- la logique de lecture de la langue cible est respectee
- la logique visuelle de l'original est preservee autant que possible

Cette specification est pensee pour etre implementee dans [reconstructor.py](/home/raphael/Mes_Projets/docs_parser/reconstructor.py).


## Objectifs Fonctionnels

Le moteur final doit:

- preserver la geometrie globale de la page
- preserver la geometrie du bloc
- preserver les styles source: police, taille, couleur, bold, italic, underline, uppercase, monospace
- preserver les elements immuables: code, formules, references protegees, overlays exacts
- reconstruire le texte traduit selon la structure extraite
- suivre les alignements reels du bloc: left, center, right, justify
- gerer explicitement:
  - continuation inline
  - retour a la ligne
  - rupture de paragraphe
  - heading suivi de body
  - groupes courts non cassables
  - ancrages locaux dans le bloc
- produire des primitives de dessin explicites et verifiables


## Anti-Objectifs

Le moteur ne doit pas:

- traduire ou modifier le code
- reconstruire un bloc en partant d'une seule chaine concatenee
- inferer librement un layout entier sans tenir compte des lignes d'origine
- casser une unite logique juste pour la faire rentrer
- reutiliser un fond source qui reintroduit l'anglais si le bloc doit etre rerendu
- pousser du texte hors de son bloc sans politique explicite


## Invariants

Ces regles sont prioritaires:

1. la bbox du bloc ne change pas par defaut
2. les zones protegees et overlays immuables passent avant tout
3. le renderer doit suivre la hierarchie semantique, pas seulement les lignes OCR
4. les policies de placement doivent etre explicites et traceables
5. toute compression typographique doit rester bornee et justifiee
6. les collisions volontaires sont interdites


## Langue Cible Et Sens D'Ecriture

Le plan de reconstruction doit derivar du `target_lang`.

Exemple pour le francais:

- inline progression: `left_to_right`
- block progression: `top_to_bottom`
- default paragraph alignment:
  - reprendre l'alignement source si fiable
  - sinon `left`

Le moteur doit etre pret pour d'autres langues:

- `right_to_left`
- `vertical`
- combinations mixtes a terme

Donc la logique de curseur doit etre abstraite et ne pas etre codee en dur pour le francais.


## Niveaux De Donnees A Utiliser

La reconstruction doit reposer sur les donnees extraites deja presentes ou a completer:

- `relative_geometry`
- `layout_attributes`
- `style_attributes`
- `text_attributes`
- `positioning_policy`
- `editorial_semantics`
- `editorial_relations`
- `expression_semantics`
- `expression_relations`
- `structural_context`
- `semantic_spans`
- `semantic_runs`
- `semantic_groups`
- `semantic_phrases`


## Hierarchie Cible

La hierarchie logique de reconstruction doit etre:

1. `page`
2. `block`
3. `line_template`
4. `semantic_phrase`
5. `semantic_group`
6. `semantic_run`
7. `semantic_span`
8. `draw_fragment`

Notes:

- `line_template` est un gabarit physique derive de l'original
- `semantic_phrase` est une unite linguistique, pas une ligne
- `semantic_group` sert a garder ensemble une unite logique compacte
- `semantic_run` sert a garder ensemble une unite logique meme si le style varie
- `semantic_span` sert aux expressions inline continues, y compris multi-lignes


## Strategie Globale

Le bloc est l'unite de reconstruction fondamentale.

Chaque bloc suit le pipeline:

1. classification du bloc
2. preparation geometrique du bloc
3. construction du plan de reconstruction
4. construction du graphe de relations
5. generation des line templates
6. placement unite par unite
7. ajustement local de ligne
8. emission des draw ops
9. verification locale


## Classification Du Bloc

Il faut dispatcher chaque bloc dans une famille de rendu.

Familles minimales:

- `editorial`
- `code`
- `table`
- `caption`
- `annotation`
- `heading`
- `mixed`
- `protected_visual`

Proposition de selection:

- `code` si `protected_visual` ou `immutable_programming_code`
- `table` si `table_*` ou structure tabulaire fiable
- `heading` si `heading_like` et peu de contenu
- `caption` si `caption_like`
- `annotation` si `anchored_annotation`
- `editorial` si `editorial_body` ou `reflowable`
- `mixed` sinon

Cette classification doit etre centralisee dans une fonction unique:

```python
def classify_block_for_reconstruction(block: dict) -> str:
    ...
```


## Contrat De Donnees A Construire

Avant toute reconstruction, construire un `BlockReconstructionPlan`.

### Structure Recommandee

```python
@dataclass
class BlockReconstructionPlan:
    block_id: str
    page_index: int
    block_type: str
    block_role: str
    block_bbox: tuple[float, float, float, float]
    block_bbox_pt: tuple[float, float, float, float] | None
    container_bbox: tuple[float, float, float, float]
    writing_direction: str
    block_progression: str
    alignment: str
    paragraph_alignment: str
    padding_left: float
    padding_right: float
    padding_top: float
    padding_bottom: float
    protected_regions: list[dict]
    background_strategy: str
    background_color: tuple[int, int, int] | None
    line_templates: list["LineTemplate"]
    units: list["PlacableUnit"]
    graph_edges: list["GraphEdge"]
    positioning_policy: dict
    relative_geometry: dict
    editorial_semantics: dict
    editorial_relations: dict
    constraints: dict
```

### `LineTemplate`

```python
@dataclass
class LineTemplate:
    line_id: str
    source_line_indices: list[int]
    bbox: tuple[float, float, float, float]
    baseline_y: float
    ascent: float
    descent: float
    left_x: float
    right_x: float
    usable_width: float
    indent_px: float
    first_line_indent_px: float
    alignment: str
    paragraph_id: str | None
    paragraph_index: int
    line_index_in_paragraph: int
    is_first_paragraph_line: bool
    is_last_paragraph_line_hint: bool
```

### `PlacableUnit`

```python
@dataclass
class PlacableUnit:
    unit_id: str
    unit_type: str
    source_kind: str
    parent_unit_id: str | None
    block_unit_id: str
    phrase_unit_id: str | None
    line_indices: list[int]
    text_source: str
    text_translated: str
    role: str
    inline_class: str | None
    group_class: str | None
    style: dict
    layout_attributes: dict
    text_attributes: dict
    relative_bbox: tuple[float, float, float, float] | None
    anchor_horizontal: str | None
    anchor_vertical: str | None
    continuation_before: bool
    continuation_after: bool
    hard_break_before: bool
    hard_break_after: bool
    keep_with_previous: bool
    keep_with_next: bool
    reflowable: bool
    protected_inline: bool
    immutable: bool
    render_policy: str
    justification_eligible: bool
    break_priority: int
```

### `GraphEdge`

```python
@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relation: str
    hard: bool
    weight: float
```


## Regles De Construction Des Unites Placables

Le renderer ne doit pas travailler directement sur la structure brute.
Il faut normaliser les donnees extraites en une liste ordonnee de `PlacableUnit`.

Ordre recommande de generation:

1. `semantic_phrase`
2. `semantic_group`
3. `semantic_run`
4. `semantic_span`

Regles:

- si un `semantic_group` existe et couvre plusieurs runs, il devient l'unite de pilotage
- sinon si un `semantic_run` existe, il devient l'unite de pilotage
- sinon utiliser le `semantic_span`
- la `semantic_phrase` fournit la structure paragraphique et les points de rupture

Autrement dit:

- `semantic_phrase` structure
- `group/run/span` dessinent


## Politique Par Type D'Unite

### 1. Semantic Phrase

Role:

- fournir les delimitations editoriales
- indiquer les transitions:
  - inline continuation
  - newline
  - paragraph break
- fournir les paragraph ids

Elle ne doit pas necessairement etre dessinee comme un seul objet.

### 2. Semantic Group

Role:

- empecher de casser des unites compactes:
  - `Model: ResNet50`
  - `Transfer learning`
  - `YOLO v4`
  - `Table 5.2`

Regle:

- non cassable par defaut
- cassable seulement si `reflowable` et aucune autre solution locale n'existe

### 3. Semantic Run

Role:

- conserver la coherence d'une unite logique a style mixte

Exemple:

- un mot en gras suivi d'un mot en romain, mais faisant partie du meme label

Regle:

- se place comme une unite logique
- rend plusieurs fragments stylises en interne

### 4. Semantic Span

Role:

- exprimer une continuite inline locale
- peut traverser plusieurs lignes source

Regle:

- si `multi_line=True`, sa geometrie cible doit etre determinee par le moteur, pas recopier brutalement les lignes source


## Construction Du Graphe

Le graphe ne doit pas etre optionnel. Il doit piloter l'ordre et les contraintes de placement.

### Relations Minimales

- `CONTINUE_INLINE`
- `NEW_LINE`
- `NEW_PARAGRAPH`
- `KEEP_WITH_NEXT`
- `KEEP_WITH_PREVIOUS`
- `HEADING_TO_BODY`
- `LABEL_VALUE`
- `ANNOTATION_CLUSTER`
- `LOCK_TO_TEMPLATE_LINE`
- `LOCK_TO_PROTECTED_REGION`

### Regles De Construction

Pour chaque unite ordonnee:

- comparer avec l'unite precedente
- derivar la relation principale depuis:
  - `editorial_relations`
  - `expression_relations`
  - `structural_context`
  - `hard_break_before`
  - `continuation_before`
  - `same paragraph`

Pseudo-code:

```python
def build_reconstruction_graph(units: list[PlacableUnit]) -> list[GraphEdge]:
    edges = []
    for prev_unit, unit in pairwise(units):
        relation = derive_relation(prev_unit, unit)
        edges.append(GraphEdge(prev_unit.unit_id, unit.unit_id, relation.name, relation.hard, relation.weight))
    return edges
```

### Usage

Le graphe sert a:

- definir l'ordre de placement
- savoir quand ouvrir une ligne
- savoir quand ouvrir un paragraphe
- savoir quand garder un groupe intact
- controler les fallbacks


## Generation Des Line Templates

Il ne faut pas regenerer des lignes from scratch tant que le bloc n'est pas en echec.

Il faut reutiliser les lignes source comme gabarits.

### Sources Des Line Templates

- lignes OCR du bloc
- `relative_geometry`
- `layout_attributes`
- `style_attributes`
- `editorial_relations`

### Ce Que Chaque Template Doit Capturer

- bbox originale de la ligne
- indentation originale
- baseline estimee
- hauteur typographique
- largeur utile
- alignement local
- appartenance a un paragraphe

### Politique

- pour les blocs editoriaux: reemployer les lignes source comme base
- pour les blocs tres deformes: permettre la creation de lignes supplementaires dans le meme bloc
- pour le code: conserver exactement les lignes source
- pour un tableau: les `line_templates` ne sont pas le moteur principal, la cellule l'est


## Preparation Geometrique Du Bloc

Avant de dessiner, preparer un `BlockGeometryContext`.

### Contenu

- bbox du bloc
- padding interne
- surface utile
- zones protegees
- zones deja occupees par des overlays immuables
- masques de fond
- politique de nettoyage

### Strategie De Fond

Choisir explicitement:

- `background_restore`
- `whiteout`
- `solid_fill`
- `preserve`

Regles:

- si contenu source a rerendre: ne pas recoller l'anglais
- si bloc editorial simple: privilegier `whiteout` ou `solid_fill`
- si bloc protege ou texture non blanche: `background_restore`


## Algorithme De Placement

Le placement se fait a l'interieur d'un bloc, unite par unite.

### Boucle Principale

```python
def render_editorial_block(plan: BlockReconstructionPlan) -> list[BlockRenderOp]:
    cursor = init_cursor(plan)
    ops: list[BlockRenderOp] = []

    for unit in ordered_units(plan):
        relation = relation_with_previous(plan, unit)
        cursor = apply_relation_transition(plan, cursor, relation, unit)

        placement = place_unit(plan, cursor, unit)
        if not placement.success:
            placement = resolve_overflow(plan, cursor, unit, placement)

        if not placement.success:
            placement = emit_layout_failure_placeholder(plan, cursor, unit, placement)

        ops.extend(placement.draw_ops)
        cursor = placement.next_cursor

    ops.extend(finalize_line_alignment(plan, ops))
    return ops
```

### `apply_relation_transition`

Selon la relation precedente:

- `CONTINUE_INLINE`
  - rester sur la ligne
  - avancer avec l'espacement inline
- `NEW_LINE`
  - passer a la ligne suivante
  - conserver le paragraphe
- `NEW_PARAGRAPH`
  - passer a la ligne suivante
  - appliquer le spacing de paragraphe
  - reinitialiser l'indentation
- `HEADING_TO_BODY`
  - finaliser la hauteur du heading
  - ouvrir le body apres lui

### `place_unit`

Etapes:

1. choisir le texte cible
2. choisir le style cible
3. mesurer le texte
4. verifier si la bbox inline disponible suffit
5. sinon lancer la strategie locale d'overflow


## Choix Du Texte Cible

Le texte a dessiner depend du type:

- `immutable`: utiliser le source exact ou overlay exact
- `code`: source exact, jamais traduit
- `reference` protegee: source exact ou traduction selon policy stricte
- `editorial`: traduction
- `caption`: traduction
- `heading`: traduction avec style preserve

Fonction proposee:

```python
def resolve_render_text(unit: PlacableUnit, target_lang: str) -> str:
    ...
```


## Choix Du Style Cible

Le style cible doit partir du style source, puis appliquer des ajustements bornes.

### Style Source A Preserver

- police
- taille
- couleur
- bold
- italic
- underline
- uppercase mode
- monospace
- leading
- tracking

### Ajustements Autorises

- leger ajustement de taille
- leger ajustement de tracking
- leger ajustement de leading
- jamais changement arbitraire de famille de police

Fonction proposee:

```python
def resolve_target_style(unit: PlacableUnit, plan: BlockReconstructionPlan) -> dict:
    ...
```


## Mesure Typographique

Le moteur doit mesurer:

- largeur inline
- hauteur de boite
- ascent
- descent
- baseline

Cache recommande:

- cle = `(font_family, font_size, font_flags, text)`

API recommandee:

```python
def measure_text_run(text: str, style: dict) -> TextMetrics:
    ...
```


## Remplissage D'Une Ligne

Chaque ligne doit etre construite avant d'etre finalisee.

### Etapes

1. collecter les unites de la ligne
2. mesurer la largeur totale
3. appliquer l'alignement:
   - left
   - center
   - right
   - justify
4. emettre les draw ops avec les x/y definitifs

### Regle De Justification

Justifier uniquement:

- les unites editoriales
- les unites lexicales compatibles

Ne jamais justifier:

- code
- formules
- references techniques compactes
- groupes non cassables


## Politique D'Overflow

Ordre obligatoire de resolution:

1. wrapping local si `reflowable`
2. utilisation des lignes suivantes du meme paragraphe
3. creation d'une ligne supplementaire dans le meme bloc si policy autorisee
4. ajustement leger de taille/tracking/leading
5. extension locale bornée selon `positioning_policy`
6. echec explicite

### Ce Qui Est Interdit

- casser un groupe technique compact sans seuil explicite
- superposer deux textes
- mordre sur une zone protegee
- traduire du code pour le faire rentrer

### Signature

```python
def resolve_overflow(
    plan: BlockReconstructionPlan,
    cursor: PlacementCursor,
    unit: PlacableUnit,
    failed_placement: PlacementResult,
) -> PlacementResult:
    ...
```


## Regles De Continuite

### Continuite Inline

Si:

- meme phrase
- pas de hard break
- pas de rupture editoriale
- meme cluster logique

Alors:

- placer a la suite
- appliquer l'espacement inline source ou normalise

### Retour A La Ligne

Si:

- `NEW_LINE`
- ou overflow resolu par wrapping

Alors:

- passer au template de ligne suivant
- conserver le paragraphe
- recalculer l'indentation inline

### Rupture De Paragraphe

Si:

- `NEW_PARAGRAPH`
- ou vrai signal editorial de rupture

Alors:

- incrementer `paragraph_index`
- appliquer l'espacement vertical de paragraphe
- reinitialiser l'indentation de premiere ligne


## Alignement Par Bloc

### Left

- point de depart sur `left_x + indent`
- normal pour prose standard

### Center

- mesurer la ligne entiere
- recentrer le groupe

### Right

- n'utiliser que si confirme par les metadonnees fiables
- ne pas reutiliser aveuglement un `alignment=right` suspect sur un paragraphe editorial long

### Justify

- distribuer l'espace entre unites justifiables
- ne pas justifier la derniere ligne du paragraphe
- conserver les groupes non cassables


## Cas Speciaux

### Code

Contrat:

- pas de traduction
- pas de reflow editorial
- lignes source exactes
- style monospace exact
- overlay si necessaire

Renderer:

```python
class CodeBlockRenderer(BaseBlockRenderer):
    ...
```

### Tableau

Contrat:

- le conteneur principal est le tableau
- le moteur travaille cellule par cellule
- pas de reflow paragraphe global
- chaque cellule a son alignement local

Renderer:

```python
class TableBlockRenderer(BaseBlockRenderer):
    ...
```

### Heading

Contrat:

- priorite au style et a la geometrie
- texte souvent compact
- centrage/uppercase/couleur a conserver

### Caption

Contrat:

- suit la figure
- peut etre multiline
- mais reste un bloc editorial compact

### Annotation

Contrat:

- ancrage a un objet ou une zone
- reflow tres limite
- pas de fuite dans la colonne voisine


## Draw Ops A Produire

Le renderer doit produire des operations explicites.

Types minimaux:

- `erase_rect`
- `restore_background_patch`
- `draw_text_run`
- `draw_text_group`
- `draw_overlay_image`
- `reserve_region`
- `debug_bbox`

Structure recommandee:

```python
@dataclass
class BlockRenderOp:
    op_type: str
    block_id: str
    unit_id: str | None
    bbox: tuple[float, float, float, float] | None
    text: str | None
    style: dict | None
    z_index: int
    metadata: dict
```


## Verification Locale Post-Layout

Apres reconstruction de chaque bloc, verifier:

- aucun overlap texte/texte
- aucun overlap texte/zone protegee
- couverture des phrases critiques
- respect du bloc bbox
- respect du nombre minimal de lignes attendues
- preservation des unites immuables

API recommandee:

```python
def validate_block_layout(plan: BlockReconstructionPlan, ops: list[BlockRenderOp]) -> list[dict]:
    ...
```

Si validation echoue:

- essayer un fallback borné
- sinon remonter un diagnostic precis dans le QA output


## Architecture A Introduire Dans `reconstructor.py`

### Nouvelles Fonctions

- `classify_block_for_reconstruction(block)`
- `build_block_geometry_context(block, page_context)`
- `build_line_templates(block, geometry_context)`
- `build_block_reconstruction_plan(block, page_context, target_lang)`
- `build_reconstruction_graph(plan)`
- `normalize_placable_units(plan)`
- `select_block_renderer(plan)`
- `validate_block_layout(plan, ops)`
- `commit_block_draw_ops(page_canvas, ops)`

### Renderers

```python
class BaseBlockRenderer:
    def render(self, plan: BlockReconstructionPlan) -> list[BlockRenderOp]:
        raise NotImplementedError


class EditorialBlockRenderer(BaseBlockRenderer):
    ...


class HeadingBlockRenderer(BaseBlockRenderer):
    ...


class CaptionBlockRenderer(BaseBlockRenderer):
    ...


class AnnotationBlockRenderer(BaseBlockRenderer):
    ...


class CodeBlockRenderer(BaseBlockRenderer):
    ...


class TableBlockRenderer(BaseBlockRenderer):
    ...
```

### Objets Intermediaires

- `BlockGeometryContext`
- `BlockReconstructionPlan`
- `LineTemplate`
- `PlacableUnit`
- `GraphEdge`
- `PlacementCursor`
- `PlacementResult`
- `BlockRenderOp`


## Integration Progressive Dans Le Code Existant

La refonte doit etre incrementale.

### Phase 1

But:

- introduire les structures de donnees
- ne pas tout recoder d'un coup

Actions:

- ajouter les dataclasses
- ajouter `build_block_reconstruction_plan`
- logger les plans sans encore changer tous les renderers

### Phase 2

But:

- rerouter les blocs editoriaux vers `EditorialBlockRenderer`

Actions:

- utiliser `semantic_phrases`
- utiliser `line_templates`
- gerer `CONTINUE_INLINE`, `NEW_LINE`, `NEW_PARAGRAPH`

### Phase 3

But:

- introduire `semantic_groups` et `semantic_runs`

Actions:

- empecher la casse des groupes compacts
- stabiliser les titres, labels, references techniques

### Phase 4

But:

- brancher les renderers specialises

Actions:

- `CodeBlockRenderer`
- `TableBlockRenderer`
- `AnnotationBlockRenderer`

### Phase 5

But:

- remplacer les heuristiques globales de text fitting

Actions:

- deprecier les chemins trop globaux
- basculer sur les draw ops structures


## Strategie De Tests

### Tests Unitaires

Il faut couvrir:

- phrase multi-ligne
- plusieurs phrases sur une ligne
- groupe non cassable
- run a style mixte
- span multi-ligne
- heading suivi de body
- paragraphe justifie
- bloc center
- bloc right fiable
- faux `alignment=right` a normaliser
- code immuable
- tableau cellule par cellule
- annotation ancree

### Tests D'Integration

Jeux recommandes:

- page TOC
- page code
- page tableau
- page figure + caption + body
- page web_print
- page annotation technique

### Assertions

- `content_coverage_score`
- `rendered_text_coverage_score`
- `visual_similarity_score`
- `word_overlaps == 0` sur cas propres
- `text_img_collisions == 0` ou borne stricte
- presence des unites critiques attendues


## Criteres D'Acceptation

La refonte peut etre consideree exploitable si:

- les blocs editoriaux longs ne disparaissent plus
- les titres ne sont plus dupliques
- les blocs de code sont preserves intacts
- les tableaux ne sont plus reflowes comme des paragraphes
- l'alignement des paragraphes longs depend du bloc reel, pas de metadonnees erronees
- les renderers produisent des draw ops inspectables
- la validation locale explique les echecs de layout


## Pseudo-Code Global

```python
def reconstruct_page(page, target_lang):
    page_ops = []

    for block in iter_blocks_in_reading_order(page):
        block_type = classify_block_for_reconstruction(block)
        geometry_context = build_block_geometry_context(block, page)
        line_templates = build_line_templates(block, geometry_context)
        plan = build_block_reconstruction_plan(
            block=block,
            page_context=page,
            target_lang=target_lang,
        )
        plan.block_type = block_type
        plan.line_templates = line_templates
        plan.units = normalize_placable_units(plan)
        plan.graph_edges = build_reconstruction_graph(plan.units)

        renderer = select_block_renderer(plan)
        block_ops = renderer.render(plan)

        findings = validate_block_layout(plan, block_ops)
        block_ops = maybe_apply_last_chance_fallback(plan, block_ops, findings)

        page_ops.extend(block_ops)

    commit_block_draw_ops(page, page_ops)
    return page_ops
```


## Decision Finale

Le moteur cible n'est pas un moteur de `fit text in box`.

C'est un moteur de reconstruction contrainte:

- bloc fixe
- lignes gabarits
- phrases semantiques
- groupes/runs/spans comme unites de dessin
- graphe de relations
- placement sequentiel
- validation locale

La bonne question n'est donc plus:

- "comment faire rentrer le texte traduit dans la boite ?"

La bonne question devient:

- "comment reconstruire, dans cette boite fixe, la meme logique visuelle que l'original avec les unites extraites et les contraintes de la langue cible ?"


## Plan D'Implementation Recommande

Ordre de travail concret:

1. introduire les dataclasses et le plan de reconstruction
2. construire les `line_templates`
3. implementer `EditorialBlockRenderer`
4. brancher la validation locale
5. migrer `code` vers `CodeBlockRenderer`
6. migrer `table` vers `TableBlockRenderer`
7. migrer `annotation` et `caption`
8. supprimer progressivement les anciens chemins de reflow global

Ce plan permet une migration reelle, testable, sans big bang.
