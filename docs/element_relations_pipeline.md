# Pipeline de relations d'elements et de positionnement

## Objet du document

Ce document decrit cinq modules introduits dans le pipeline de layout:

- `relative_geometry.py`
- `element_relations.py`
- `element_relations_ai.py`
- `positioning_policy.py`
- `element_rulesets.py`

L'objectif global est de transformer une page OCR/native en representation exploitable pour la traduction et la reconstruction visuelle, en preservant:

- l'ordre de lecture
- les continuations textuelles
- les ancrages de position
- les roles semantiques des fragments
- les contraintes de reflow et d'alignement


## Vue d'ensemble

Dans `LayoutV2Builder.build()` le chainage est le suivant:

```text
page_data
  -> enrich_page_relative_geometry()
  -> enrich_element_relations()
  -> get_element_relations_ai_enricher().enrich()
  -> enrich_positioning_policy()
  -> enrich_element_rulesets()
```

Dependances fonctionnelles:

```text
relative_geometry
  -> fournit bbox relatives, ordre de lecture, colonnes, direction LTR/RTL

element_relations
  -> construit le graphe de flux entre phrases

element_relations_ai
  -> revoit les aretes ambigues du graphe de flux

positioning_policy
  -> combine geometrie + flux + signaux semantiques pour choisir un ancrage

element_rulesets
  -> convertit les politiques en regles de traduction/reconstruction
```

En pratique, la maille de base est la `phrase`. Chaque phrase devient:

- un noeud geometrique
- un noeud de lecture
- un noeud de flux textuel
- un candidat d'ancrage
- un ruleset de traduction/reconstruction


## 1. `relative_geometry.py`

### Objectif

Construire une representation geometrique hierarchique et relative de la page. Ce module normalise la geometrie dans un repere stable:

- page
- block
- line
- phrase
- span

Il sert de socle a tout ce qui suit.

### Ce que le module fait

`enrich_page_relative_geometry(page_data)`:

- lit `dimensions.width`, `dimensions.height`, `dimensions.dpi`
- determine `layout_direction` en utilisant la direction explicite ou une detection LTR/RTL par caracteres
- normalise les colonnes existantes dans `layout.columns`
- calcule les `page_features`
- ordonne les blocks selon l'ordre de lecture
- annote recursivement blocks, lines, phrases et spans avec leur geometrie relative
- produit un arbre `relative_layout` et un aplatissement `relative_layout_flat`

### Mecanismes principaux

1. Detection de direction:

- explicite si `layout_direction` ou `reading_direction` existe
- sinon heuristique sur un echantillon de texte
- plus de caracteres hebreu/arabe -> `rtl`, sinon `ltr`

2. Ordre de lecture:

- tri lineaire par bandes horizontales
- gestion speciale des colonnes si `layout.columns` contient au moins deux colonnes
- les blocs trans-colonnes sont traites a part

3. Geometrie relative:

- `bbox_relative_to_parent`
- `bbox_relative_to_container_block`
- distances des bords vers le parent ou le block conteneur

### Algorithme graphe / structure

Ce module ne construit pas un graphe semantique, mais un arbre de layout:

```text
page
  -> block
    -> line
      -> phrase
        -> span
```

Chaque noeud recoit:

- un `reading_order_index`
- un `reading_order_path`
- une `bbox`
- des `borders`
- des attributs de role/source/alignment

Le `reading_order_path` joue le role de chemin canonique dans l'arbre.

### Code cle

- `enrich_page_relative_geometry()`
- `_resolve_layout_direction()`
- `_reading_order()`
- `_reading_order_linear()`
- `_annotate_node()`
- `_flatten_nodes()`

### Sorties attendues

Au niveau page:

- `page_data["relative_layout"]`
- `page_data["relative_layout_flat"]`
- `page_data["layout"]["relative_layout"]`

Au niveau noeud:

- `node["relative_geometry"]`
- `node["reading_order_index"]`
- `node["reading_order_path"]`

### Comportement attendu

Les tests verifient notamment:

- schema `relative_geometry.v1`
- orientation et `column_count` correctement exposes
- ordre de lecture des blocks stable
- `bbox_relative_to_container_block` correct pour une phrase
- chemins de lecture coherents dans `relative_layout_flat`


## 2. `element_relations.py`

### Objectif

Construire le graphe de flux entre phrases a l'interieur d'un block. Le probleme traite ici est: "la phrase suivante continue-t-elle la precedente, ou ouvre-t-elle une nouvelle unite structurale ?"

### Ce que le module fait

`enrich_element_relations(page_data)`:

- parcourt les blocks
- extrait les phrases dans l'ordre de lecture
- annote chaque phrase avec un noeud de graphe `element_relation_node`
- relie chaque phrase a la suivante par une arete orientee
- attache les relations au block, a la page et a chaque phrase

### Mecanismes principaux

Les decisions reposent sur des heuristiques geometriques et textuelles:

- meme ligne ou ligne suivante
- ecart horizontal `inline_gap`
- ecart vertical `vertical_gap`
- difference d'indentation `indent_delta`
- ponctuation terminale de la phrase precedente
- mot coupe par tiret
- minuscule/majuscule au debut de la suivante
- detection de puce ou marqueur de liste
- hard break avant la suivante
- proximite de style typographique

### Algorithme graphe

Le graphe est simple et oriente, local a chaque block:

```text
p1 -> p2 -> p3 -> p4
```

Chaque arete relie deux phrases consecutives dans l'ordre de lecture. Pour chaque paire `(previous, current)`, `_infer_pair_relation()` produit:

- `visual_relation`
- `logical_relation`
- `continuation`
- `confidence`
- `signals`

Les classes produites sont:

- `continues_inline`
- `continues_wrapped_line`
- `new_structural_unit`

et cote logique:

- `same_token_continuation`
- `same_sentence_continuation`
- `same_paragraph_continuation`
- `new_list_item`
- `new_sentence_or_unit`
- `new_structural_unit`
- `uncertain`

### Heuristique de confiance

`_relation_confidence()` part d'un score de base puis:

- augmente si continuation, meme ligne, wrap plausible, meme style
- baisse si ponctuation terminale, marqueur de liste, hard break, grands ecarts

Une revue IA est demandee si:

- `confidence < 0.72`
- ou `logical_relation == "uncertain"`

### Code cle

- `enrich_element_relations()`
- `_enrich_block_relations()`
- `_ordered_phrases_in_block()`
- `_infer_pair_relation()`
- `_relation_confidence()`

### Sorties attendues

Au niveau phrase:

- `element_relation_node`
- `flow_to_next_phrase`
- `flow_from_previous_phrase`

Au niveau block:

- `block["element_relations"]["reading_order"]`
- `block["element_relations"]["pair_relations"]`

Au niveau page:

- `page_data["element_relations"]`
- `page_data["layout"]["element_relations"]`

### Comportement attendu

Les tests couvrent trois cas majeurs:

- continuation inline -> `continues_inline` + `same_sentence_continuation`
- continuation sur retour a la ligne -> `continues_wrapped_line` + `same_paragraph_continuation`
- rupture structurelle -> `new_structural_unit` et revue IA requise

Le module doit aussi renseigner des snapshots exploitables par les phases suivantes, pas juste une etiquette finale.


## 3. `element_relations_ai.py`

### Objectif

Revoir semantiquement les relations ambigues produites par `element_relations.py`, sans rendre le pipeline dependant d'un service distant.

### Ce que le module fait

`ElementRelationsAIEnricher.enrich(page_data)`:

- recupere `page_data["element_relations"]["flat_relations"]`
- filtre les aretes marquees `ai_review_required` ou `uncertain`
- charge un modele local ONNX de NLI si disponible
- score plusieurs hypotheses de continuation et de role logique
- fusionne la decision IA avec la decision heuristique

### Mecanisme

Le module utilise un modele local de Natural Language Inference:

- backend par defaut: `onnx_nli`
- bundle local attendu dans `ai_models/element_relations_nli`
- chargement via `onnxruntime` et `transformers`

Le "premise" donne au modele combine:

- texte source
- texte cible
- signaux geometriques
- ponctuation
- hard break
- position relative

Les hypotheses evaluees sont de deux types:

1. continuation binaire:

- `continuation`
- `new_unit`

2. relation logique:

- `same_token_continuation`
- `same_sentence_continuation`
- `same_paragraph_continuation`
- `new_list_item`
- `new_sentence_or_unit`
- `new_structural_unit`

### Algorithme de revue

Pour chaque arete candidate:

```text
relation heuristique
  -> construire un premise textuel
  -> scorer les hypotheses
  -> prendre le meilleur label continuation
  -> prendre le meilleur label logique
  -> appliquer un seuil min_confidence
  -> ecrire la decision finale dans la relation
```

Fusion:

- si la confiance continuation est suffisante, la relation visuelle est reecrite
- si la confiance logique est suffisante, `logical_relation` est reecrit
- la confiance finale devient le max entre heuristique et IA
- `resolved_by` passe a `semantic_ai` si l'IA a tranche
- l'etat heuristique initial est preserve dans `heuristic_decision`

### Configuration

Variables d'environnement:

- `ELEMENT_RELATIONS_AI_ENABLE`
- `ELEMENT_RELATIONS_AI_MODEL_DIR`
- `ELEMENT_RELATIONS_AI_BACKEND`
- `ELEMENT_RELATIONS_AI_MIN_CONFIDENCE`

### Code cle

- `ElementRelationsAIEnricher.enrich()`
- `_review_relation()`
- `_merge_relation_review()`
- `_build_premise()`
- `_get_runtime()`
- `get_element_relations_ai_enricher()`

### Sorties attendues

Au niveau page:

- `page_data["element_relations_ai"]`
- `page_data["layout"]["element_relations_ai"]`

Au niveau relation:

- `heuristic_decision`
- `semantic_ai_review`
- `resolved_by`
- `ai_review_required` mis a jour

### Comportement attendu

Les tests imposent:

- mode desactive -> no-op propre, avec status expose
- bundle local ONNX standard detecte automatiquement
- une relation ambigue peut etre resolue par l'IA en continuation de paragraphe


## 4. `positioning_policy.py`

### Objectif

Choisir, pour chaque phrase, le point d'ancrage a conserver lors de la traduction et de la reconstruction. Ce module repond a la question: "si le texte traduit change de taille, de quel cote faut-il proteger sa position ?"

### Ce que le module fait

`enrich_positioning_policy(page_data)`:

- parcourt les phrases block par block
- mesure l'espace libre autour de chaque phrase dans son block
- recupere le flux texte precedent/suivant
- calcule des scores d'ancrage horizontal et vertical
- integre, si disponible, des scores semantiques issus du helper IA
- publie une `positioning_policy` par phrase

### Mecanismes principaux

Signaux geometriques:

- espace libre gauche/droite/haut/bas
- ratios dans le block
- ecart au centre du block
- largeur/hauteur relatives du fragment

Signaux structurels:

- continuation avec phrase precedente/suivante
- ordre dans le block
- alignment du block/line/phrase

Signaux semantiques:

- `flow_text`
- `centered_title`
- `end_value`
- `attached_label`

### Algorithme

Le module calcule trois scores horizontaux:

- `start`
- `end`
- `center`

et trois scores verticaux:

- `top`
- `bottom`
- `middle`

Chaque score est une combinaison ponderee de signaux. Les formules sont publiees dans la sortie:

- horizontal `start`: proximite du bord start + espace libre apres + semantique + flow + alignment
- horizontal `end`: proximite du bord end + espace libre avant + numeric-like + alignment end + role end_value
- horizontal `center`: proximite du centre + symetrie des marges + alignment center + role centered_title

Puis:

- on prend les deux meilleurs scores par axe
- on construit `primary_position_reference`
- on derive `expansion_policy`

### Lien avec le graphe de relations

`positioning_policy.py` ne construit pas un graphe explicite, mais consomme les aretes du graphe de flux:

- `flow_from_previous_phrase`
- `flow_to_next_phrase`

Ces aretes determinent notamment:

- `in_flow`
- la croissance verticale
- certaines preferances d'ancrage

### Code cle

- `enrich_positioning_policy()`
- `_enrich_block_policy()`
- `_semantic_scores()`
- `_top_two()`
- `_horizontal_expansion()`
- `_vertical_expansion()`

### Sorties attendues

Au niveau phrase:

- `phrase["positioning_policy"]`

Contenu cle:

- `anchors.horizontal.primary|secondary`
- `anchors.vertical.primary|secondary`
- `primary_position_reference`
- `expansion_policy`
- `space_metrics`
- `semantic_context`
- `signals`
- `formula`

### Comportement attendu

Les tests couvrent:

- texte de flux -> ancrage `top_start`
- titre centre -> ancrage horizontal `center`
- valeur alignee a droite et numeric-like -> ancrage `end`

Le helper IA peut influencer les scores, mais le module doit rester stable avec des scores neutres si le runtime IA n'est pas disponible.


## 5. `element_rulesets.py`

### Objectif

Transformer la `positioning_policy` en regles de traduction/reconstruction directement exploitables par les etapes aval.

Autrement dit, `positioning_policy` dit "ou ancrer", et `element_rulesets` dit "quelles regles appliquer".

### Ce que le module fait

`enrich_element_rulesets(page_data)`:

- parcourt toutes les phrases
- convertit `positioning_policy` + `relative_geometry` + flux + role semantique en `element_ruleset`
- attache un ruleset a chaque phrase
- produit un resume par block
- specialise certains roles, notamment pour les pages de sommaire
- recopie aussi les resultats sous `translation_ruleset(s)`

### Mecanismes principaux

Le module combine:

- ancrages primaires/secondaires issus de `positioning_policy`
- continuite issue de `element_relations`
- geometrie relative issue de `relative_geometry`
- role semantique generique issu du scoring
- specialisation metier, surtout TOC

### Specialisation semantique

Le role semantique generique peut etre specialise en:

- `toc_heading`
- `toc_page_number`
- `toc_section_number`
- `toc_entry_title`

Cette specialisation croise:

- le contexte `page_role == toc`
- les `toc_rows`
- l'overlap geometrique avec `label_bbox` et `page_bbox`
- la nature du texte: numeral romain, nombre, numero de section, heading
- le voisinage dans la ligne

### Algorithme / logique de decision

Pour chaque phrase:

```text
positioning_policy
  + relative_geometry
  + flow_prev / flow_next
  + semantic_context
  + toc context eventuel
    -> semantic_role final
    -> anchors effectifs
    -> growth rules
    -> constraints
    -> override_conditions
    -> confidence globale
```

La confiance globale est une combinaison ponderee de:

- confiance d'ancrage
- confiance de continuite
- confiance de role semantique

Le module produit ensuite des regles concretes:

- `preserve_horizontal_anchor`
- `preserve_vertical_anchor`
- `translation_positioning_mode`
- `horizontal_growth`
- `vertical_growth`
- `keep_with_previous`
- `keep_with_next`
- `hard_break_before`
- `hard_break_after`
- `continuity_class`
- `semantic_role`

### Graphe / structure

Le module ne construit pas un nouveau graphe, mais il compile plusieurs structures:

- arbre geometrique
- graphe de flux entre phrases
- matrice de scores d'ancrage

En sortie, chaque phrase dispose d'un "contrat de placement" autonome.

### Code cle

- `enrich_element_rulesets()`
- `_build_phrase_ruleset()`
- `_resolved_role_scores()`
- `_apply_semantic_anchor_overrides()`
- `_resolve_specialized_role()`
- `_resolve_toc_role()`
- `_override_conditions()`
- `_annotate_toc_rows_with_rulesets()`

### Sorties attendues

Au niveau phrase:

- `element_ruleset`
- `translation_ruleset`

Au niveau block:

- `element_rulesets`
- `translation_rulesets`
- `translation_ruleset_summary`

Au niveau page:

- `page_data["element_rulesets"]`
- `page_data["translation_rulesets"]`

### Comportement attendu

Les tests verifient notamment:

- un ruleset distinct par phrase
- un texte de flux garde un ancrage `start`
- une valeur de page garde un ancrage `end`
- les `combined_modes` sont exposes
- les contraintes de centre sont preservees pour un titre centre
- les pages TOC recoivent des roles specialises corrects
- les `toc_rows` sont re-annotees avec les rulesets des phrases correspondantes


## Contrat global du systeme

### Ce qui est attendu de la chaine complete

1. La page doit d'abord etre convertie en structure geometrique stable.
2. Les phrases doivent ensuite etre reliees par un flux de lecture local.
3. Les transitions ambigues doivent pouvoir etre revisees par un modele local.
4. Chaque phrase doit recevoir un ancrage de position robuste a l'expansion du texte traduit.
5. Chaque phrase doit enfin recevoir un ruleset autonome, exploitable par reconstruction/translation.

### Ce que la chaine produit concretement

Pour une phrase donnee, le pipeline doit permettre de repondre a ces questions:

- Ou se trouve-t-elle relativement a son block et a la page ?
- Dans quel ordre se lit-elle ?
- Continue-t-elle le texte precedent ?
- Doit-elle rester collee au debut, a la fin ou au centre ?
- Si le texte traduit grandit, dans quelle direction doit-il s'etendre ?
- Faut-il la garder attachee a la phrase precedente ou suivante ?
- Joue-t-elle le role d'un titre, d'une valeur, d'un marqueur TOC, d'un texte courant ?

### Usage aval attendu

Ces modules preparent surtout:

- la traduction sans casser la structure
- la reconstruction visuelle
- les strategies de placement pour titres, valeurs, listes et sommaires
- la generation de descripteurs de layout plus riches


## Resume court par module

### `relative_geometry.py`

Fournit le repere geometrique canonique.

### `element_relations.py`

Construit le graphe heuristique de continuation entre phrases.

### `element_relations_ai.py`

Corrige les aretes ambigues via un NLI local ONNX.

### `positioning_policy.py`

Choisit l'ancrage de position et le mode d'expansion d'une phrase.

### `element_rulesets.py`

Compile tous les signaux precedents en regles de reconstruction/traduction.
