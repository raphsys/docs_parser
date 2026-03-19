# Protocole General De Traduction Et Reconstruction Fideles

## Objectif

Obtenir un document traduit aussi proche que possible de l'original, quel que soit le type de page, avec les invariants suivants:

- tout texte utile du document original est present dans le document final
- tout texte traduisible est traduit
- aucune unite critique ne disparait
- aucune collision texte/image
- aucun chevauchement de mots
- aucune sortie hors page
- rejet automatique si la qualite n'atteint pas le seuil requis

## Principe directeur

Le systeme ne doit jamais etre optimise pour un fichier de test particulier.

Il doit:

- utiliser une representation canonique unique pour tous les documents
- typer chaque unite de document
- traduire chaque unite selon son role
- rendre chaque unite selon une strategie de rendu explicite
- valider le resultat avec une QA unifiee

Les specialisations sont autorisees:

- par role documentaire
- par famille de page

Elles sont interdites:

- par nom de fichier
- par corpus de test
- par coordonnees hard-codees propres a une page de benchmark

## 1. Format canonique `layout.v3`

### 1.1 Objet document

Le document doit etre normalise sous la forme:

```json
{
  "schema_version": "layout.v3",
  "doc_id": "string",
  "source_lang": "en",
  "target_lang": "fr",
  "pages": []
}
```

### 1.2 Objet page

Chaque page doit porter au minimum:

```json
{
  "page_id": 0,
  "page_number": 1,
  "width_pt": 595.0,
  "height_pt": 842.0,
  "page_family": "body_with_figure",
  "page_confidence": 0.0,
  "zones": [],
  "units": [],
  "assets": []
}
```

### 1.3 Objet unite

Chaque unite doit porter au minimum:

```json
{
  "unit_id": "p0:u17",
  "page_id": 0,
  "bbox_pt": [0.0, 0.0, 0.0, 0.0],
  "bbox_px": [0, 0, 0, 0],
  "text_source": "string",
  "text_normalized": "string",
  "source_kind": "native_phrase",
  "doc_role": "figure_caption",
  "style_source": {
    "font_name_raw": "string",
    "font_family_guess": "string",
    "font_size_pt": 10.0,
    "color": "#000000",
    "flags": {
      "bold": false,
      "italic": false,
      "serif": false,
      "uppercase": false
    }
  },
  "translatable": true,
  "translation_strategy": "layout_constrained",
  "coverage_required": "strict",
  "render_strategy": "render_caption",
  "anchor_policy": "hard_anchor",
  "z_index": 100,
  "parent_unit_id": null,
  "children_unit_ids": [],
  "metadata": {}
}
```

### 1.4 Champs obligatoires

Champs obligatoires pour chaque unite:

- `unit_id`
- `page_id`
- `bbox_pt`
- `text_source`
- `source_kind`
- `doc_role`
- `translatable`
- `translation_strategy`
- `coverage_required`
- `render_strategy`
- `anchor_policy`
- `z_index`

### 1.5 Invariants `layout.v3`

Invariants obligatoires:

- tout `unit_id` est unique dans le document
- toute unite appartient a exactement une page
- `bbox_pt` est dans le repere de la page
- `doc_role` appartient a la taxonomie definie plus bas
- `translation_strategy`, `render_strategy` et `anchor_policy` sont toujours definis
- aucune unite strictement utile n'a `text_source == ""`

## 2. Taxonomie stricte des roles documentaires

### 2.1 Roles autorises

La premiere version de `layout.v3` doit restreindre `doc_role` a:

- `running_header`
- `running_footer`
- `page_number`
- `title`
- `subtitle`
- `section_heading`
- `paragraph`
- `list_item`
- `list_marker`
- `table_cell`
- `toc_entry`
- `figure_caption`
- `diagram_label`
- `axis_label`
- `legend_label`
- `equation`
- `code_inline`
- `stamp_or_seal`
- `decorative_non_text`

### 2.2 Regles de priorite de typage

Quand plusieurs roles sont possibles, appliquer cet ordre:

1. `page_number`
2. `running_header` / `running_footer`
3. `equation`
4. `figure_caption`
5. `diagram_label` / `axis_label` / `legend_label`
6. `toc_entry`
7. `table_cell`
8. `section_heading`
9. `title` / `subtitle`
10. `list_item`
11. `paragraph`

### 2.3 Regles minimales de decision

Exemples de regles:

- si une unite est dans une zone de figure et courte, preferer `diagram_label`
- si une unite commence par `Figure`, `Fig.`, `Table`, preferer `figure_caption`
- si une unite ressemble a une entree de sommaire avec numero de page, preferer `toc_entry`
- si une unite est dans une grille tabulaire, preferer `table_cell`
- si une unite est numerotee `1.2`, `3.4.1`, preferer `section_heading`

## 3. Taxonomie stricte des familles de page

### 3.1 Familles autorisees

La premiere version doit limiter `page_family` a:

- `toc`
- `body_text`
- `body_with_figure`
- `body_with_diagram`
- `table_page`
- `form_page`
- `mixed_page`

### 3.2 Regle de classification

Calculer pour chaque page:

- ratio surface texte
- ratio surface image
- nombre de captions
- nombre de labels courts en zone non textuelle
- presence de grille tabulaire
- presence de lignes TOC

Decision:

- `toc` si >= 6 entrees TOC coherentes
- `table_page` si grille tabulaire dominante
- `body_with_figure` si au moins une grande figure + caption
- `body_with_diagram` si grande zone non textuelle + >= 3 labels courts
- `body_text` si > 80% du texte est narratif sans grande figure
- sinon `mixed_page`

### 3.3 Invariant

Chaque page doit avoir exactement une `page_family`.

## 4. Matrice de strategies

### 4.1 Strategies de traduction autorisees

- `exact_preserve`
- `layout_constrained`
- `semantic_reflow`

### 4.2 Strategies de rendu autorisees

- `render_exact_anchor`
- `render_fixed_label`
- `render_inline_flow`
- `render_grid_cell`
- `render_caption`
- `render_toc_row`
- `render_overlay_preserve`
- `background_only`

### 4.3 Politiques d'ancrage autorisees

- `hard_anchor`
- `soft_anchor`
- `flow_anchor`
- `grid_anchor`

### 4.4 Matrice obligatoire `doc_role -> strategies`

| doc_role | translation_strategy | render_strategy | anchor_policy |
|---|---|---|---|
| `page_number` | `exact_preserve` | `render_exact_anchor` | `hard_anchor` |
| `running_header` | `layout_constrained` | `render_exact_anchor` | `hard_anchor` |
| `running_footer` | `layout_constrained` | `render_exact_anchor` | `hard_anchor` |
| `title` | `layout_constrained` | `render_fixed_label` | `hard_anchor` |
| `subtitle` | `layout_constrained` | `render_fixed_label` | `soft_anchor` |
| `section_heading` | `layout_constrained` | `render_fixed_label` | `soft_anchor` |
| `paragraph` | `semantic_reflow` | `render_inline_flow` | `flow_anchor` |
| `list_item` | `semantic_reflow` | `render_inline_flow` | `flow_anchor` |
| `list_marker` | `exact_preserve` | `render_exact_anchor` | `hard_anchor` |
| `table_cell` | `layout_constrained` | `render_grid_cell` | `grid_anchor` |
| `toc_entry` | `layout_constrained` | `render_toc_row` | `grid_anchor` |
| `figure_caption` | `layout_constrained` | `render_caption` | `hard_anchor` |
| `diagram_label` | `layout_constrained` | `render_fixed_label` | `hard_anchor` |
| `axis_label` | `layout_constrained` | `render_fixed_label` | `hard_anchor` |
| `legend_label` | `layout_constrained` | `render_fixed_label` | `hard_anchor` |
| `equation` | `exact_preserve` | `render_overlay_preserve` | `hard_anchor` |
| `code_inline` | `exact_preserve` | `render_exact_anchor` | `hard_anchor` |
| `stamp_or_seal` | `exact_preserve` | `render_overlay_preserve` | `hard_anchor` |
| `decorative_non_text` | `exact_preserve` | `background_only` | `hard_anchor` |

Regle:

- toute unite hors matrice est invalide

## 5. Planification de page

### 5.1 Objet `page_render_plan`

Avant tout rendu, construire:

```json
{
  "page_id": 0,
  "page_family": "body_with_figure",
  "reserved_zones": [],
  "render_order": [],
  "planned_units": [],
  "conflicts": [],
  "fallback_budget": {}
}
```

### 5.2 Passes obligatoires

La planification de page doit suivre cet ordre:

1. reserver les zones non textuelles
2. reserver les captions et labels a `hard_anchor`
3. construire les conteneurs de flow pour les paragraphes
4. allouer les grilles specialisees:
   - TOC
   - tableaux
5. produire les unites planifiees avec leur position cible

### 5.3 Interdits

Interdits:

- rendre une unite avant que le plan de page n'existe
- laisser deux unites strictes partager la meme zone sans conflit explicite

## 6. Solveur de contraintes

### 6.1 Contraintes dures

Contraintes qui ne doivent jamais etre violees:

- aucune unite `coverage_required = strict` ne disparait
- aucune unite ne sort de la page
- aucune collision texte/image
- aucune collision texte/texte au-dela du seuil
- aucune unite `hard_anchor` n'est deplacee hors de sa zone autorisee

### 6.2 Contraintes souples

Contraintes a minimiser:

- variation de taille de police
- variation de line-height
- variation d'alignement
- variation de colonne
- variation de hierarchie visuelle
- wrap supplementaire
- ecart de couleur

### 6.3 Ordre de fallback obligatoire

Si une unite ne tient pas:

1. essayer police metrique proche
2. resserrer legerement tracking
3. reduire legerement taille
4. rewrapper
5. deplacer selon `anchor_policy`
6. marquer `render_failed`

Regle:

- chaque fallback applique doit etre journalise

## 7. QA unifiee

### 7.1 Coverage QA

Conditions de succes:

- `missing_units = 0`
- `warning_units = 0` pour toutes les unites `strict`
- toute unite `translatable = true` doit avoir une traduction non vide

### 7.2 Render QA

Conditions de succes:

- `rendered_missing_units = 0`
- `rendered_warning_units = 0`
- `word_overlaps = 0`
- `text_img_collisions = 0`
- `off_page_words = 0`

### 7.3 Fidelity QA

Mesures:

- `hierarchy_consistency`
- `spacing_consistency`
- `alignment_consistency`
- `color_distance`
- `table_fidelity`
- `overall`

### 7.4 Seuils par famille de page

Premiere version des seuils:

| page_family | overall_min | hierarchy_min | spacing_min | alignment_min | table_min |
|---|---:|---:|---:|---:|---:|
| `toc` | 0.88 | 0.82 | 0.82 | 0.90 | 0.70 |
| `body_text` | 0.90 | 0.85 | 0.85 | 0.88 | 0.50 |
| `body_with_figure` | 0.87 | 0.80 | 0.80 | 0.85 | 0.70 |
| `body_with_diagram` | 0.87 | 0.80 | 0.80 | 0.85 | 0.70 |
| `table_page` | 0.90 | 0.82 | 0.82 | 0.88 | 0.88 |
| `form_page` | 0.90 | 0.82 | 0.82 | 0.90 | 0.80 |
| `mixed_page` | 0.86 | 0.78 | 0.78 | 0.84 | 0.65 |

### 7.5 Regle finale de publication

`publication_ready = true` seulement si:

- Coverage QA passe
- Render QA passe
- Fidelity QA passe pour la famille de page

## 8. Journalisation minimale obligatoire

Chaque run doit produire:

- `coverage_report.json`
- `render_report.json`
- `fidelity_report.json`
- `page_render_plan.json`
- `fallback_log.json`

Chaque unite doit laisser une trace:

- role
- strategie de traduction
- strategie de rendu
- fallbacks appliques
- statut final

## 9. Plan de migration du repo actuel

### Phase 0. Stabilisation

Objectif:

- corriger les erreurs de robustesse bloquantes
- garder le pipeline actuel fonctionnel

Definition de fini:

- plus de crash sur les marqueurs multi-caracteres
- les tests actuels tournent jusqu'au bout

### Phase 1. Introduction de `layout.v3`

Fichiers cibles:

- `structure_extractor.py`
- `native_pdf_extractor.py`
- nouveau module `layout_v3.py`

Travail:

- definir le schema Python de `layout.v3`
- ajouter un export `layout.v2 -> layout.v3`
- typer chaque unite avec `doc_role`

Definition de fini:

- tout document parse sort aussi en `layout.v3`

### Phase 2. Translation planning

Fichiers cibles:

- `translator.py`
- nouveau module `translation_planner.py`

Travail:

- appliquer la matrice `doc_role -> translation_strategy`
- interdire les sorties anglaises non justifiees pour les roles courts

Definition de fini:

- toute unite traduisible passe par une strategie explicite

### Phase 3. Render planning

Fichiers cibles:

- nouveau module `render_planner.py`
- `reconstructor.py`

Travail:

- construire `page_render_plan`
- separer planification et rendu

Definition de fini:

- plus aucun rendu direct sans plan prealable

### Phase 4. Renderers specialises

Ordre obligatoire:

1. `render_body_with_figure_page`
2. `render_toc_page`
3. `render_table_page`
4. `render_mixed_page`

Definition de fini:

- chaque famille de page utilise un renderer dedie

### Phase 5. QA unifiee stricte

Fichiers cibles:

- `coverage_validator.py`
- `publication_qa.py`
- nouveau module `fidelity_gate.py`

Travail:

- appliquer les seuils par famille de page
- produire les rapports normalises

Definition de fini:

- `publication_ready` derive uniquement de la QA unifiee

## 10. Regles d'acceptation des changements

Un changement est accepte seulement si:

- il n'introduit aucune condition par nom de fichier
- il met a jour `layout.v3` ou un renderer par famille de page
- il garde les invariants Coverage QA et Render QA
- il laisse des traces exploitables dans les rapports

Un changement est refuse si:

- il corrige uniquement un benchmark
- il baisse la couverture rendue
- il baisse la robustesse sur une autre famille de page

## 11. Regle finale

Une solution est acceptee seulement si elle est:

- generale
- applicable a tout type de page via le meme protocole
- specialisee uniquement par role documentaire ou famille de page
- jamais par fichier de test

