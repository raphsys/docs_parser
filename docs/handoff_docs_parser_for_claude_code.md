# Docs Parser - Handoff de reprise

Date: 2026-05-27

Ce document resume l'etat courant du projet `docs_parser`, les objectifs, le pipeline, les modules principaux, ce qui a ete corrige, ce qui reste a faire, et la logique de developpement a suivre pour la suite.

## 1. Objectif du projet

Le projet vise une chaine complete pour des documents PDF:

1. Extraction de toutes les donnees utiles.
2. Comprension / classification des objets.
3. Traduction intelligente selon le type d'objet.
4. Reconstruction WYSIWYG du document traduit.
5. Validation automatee de la qualite du rendu.

La priorite actuelle n'est pas seulement de traduire le texte, mais de conserver:

- la geometrie,
- la structure de lecture,
- le style visible,
- les objets visuels,
- les tableaux et cellules,
- les formules et le code,
- les sommaires TOC,
- les caracteres speciaux de la langue cible,
- le fond local du document apres suppression du texte.

## 2. Idee centrale de conception

Le point de bascule du projet est le suivant:

- l'extraction seule ne suffit pas;
- la traduction seule ne suffit pas;
- la reconstruction seule ne suffit pas.

Il faut un contrat explicite qui dit:

- quoi traduire,
- quoi preserver,
- comment reconstruire,
- quelle geometrie conserver,
- quels tokens sont proteges,
- quel type de rendu appliquer.

Ce contrat a ete formalise dans `document_object_contract.py` et est maintenant consomme par la comprehension, la traduction et la reconstruction.

## 3. Pipeline global

Le pipeline reel tourne dans `ocr_server.py` et suit cet ordre:

1. Chargement du document / image.
2. OCR / extraction native / parsing de page.
3. Construction de la structure de page.
4. Enrichissement layout V2.
5. Post-processing semantique.
6. Annotation des contrats de traduction.
7. Construction des semantic phrases / spans / runs / groups.
8. P1 enrichment.
9. Object comprehension.
10. Generation du fond maitre et du mask.
11. Traduction.
12. Reconstitution PDF.
13. Analyse couverture / QA / comparaison visuelle.
14. Export HTML si demande.

Les endpoints principaux sont:

- `/pipeline/run`
- `/translate`
- `/reconstruct`
- `/export_html`
- `/debug/visual-compare`
- `/healthz`

Le frontend HTML est servi via FastAPI sur `/ui/`, avec redirection depuis `/`.

## 4. Modules principaux et role de chacun

### `ocr_server.py`

Role:

- point d'entree API FastAPI,
- orchestration du pipeline,
- construction de la structure de page,
- appel a la comprehension,
- appel a la traduction,
- appel au reconstructeur,
- calcul des rapports QA.

Fonctions importantes:

- `_extract_selected_pages_from_upload(...)`
- `_annotate_translation_contracts(...)`
- `_build_semantic_phrases_for_blocks(...)`
- `_postprocess_blocks_semantic(...)`
- `_build_semantic_spans_for_blocks(...)`
- `_build_semantic_runs_for_blocks(...)`
- `_build_semantic_groups_for_blocks(...)`
- `run_full_pipeline_for_pages(...)`

### `object_comprehension.py`

Role:

- classifier les objets de document,
- typer les blocs / lignes / phrases / spans,
- produire `object_comprehension`,
- enrichir la structure avec les infos de famille, classe, type, sous-type,
- injecter le contrat document objet.

On y trouve:

- classification des blocs,
- classification des lignes,
- classification des phrases,
- classification des spans inline,
- classification visuelle,
- classification table / code / formula / reference / navigational / editorial.

### `document_object_contract.py`

Role:

- couche canonique de decision,
- formalise le contrat par objet,
- normalise:
  - `translation.translatable`,
  - `translation.strategy`,
  - `reconstruction.contract_key`,
  - `reconstruction.render_policy`,
  - `visual_structure`,
  - `translation.protection`.

Cas couverts:

- `toc_entry`
- `table_cell`
- `figure_region`
- `figure_label`
- `code_block`
- `formula_block`
- `url_reference`
- `paragraph`

### `translator.py`

Role:

- traduire en tenant compte du contrat,
- respecter les tokens proteges,
- appliquer les restrictions de traduction selon classe/type,
- traduire les TOC structurés sans detruire les leaders `........`,
- preserver le code, les formules, les URLs, les identifiants techniques,
- relire / corriger les objets qui ne doivent pas perdre de structure.

Fonctions importantes:

- `_resolve_translation_contract(...)`
- `_unit_translation_policy(...)`
- `_postfill_block_leaf_translations(...)`
- `_translate_toc_line_text(...)`
- `_translate_toc_block(...)`
- `translate_page(...)`
- `translate_layout_v2(...)`

### `reconstructor.py`

Role:

- reconstruire le PDF final,
- choisir le mode de rendu selon le contrat,
- faire le reflow,
- faire le rendu anchored / fixed / cell_locked / source_overlay,
- conserver les styles visibles,
- corriger les polices pour la langue cible,
- rendre les TOC en ligne preservee,
- rendre les cellules dense / verrouillees,
- gerer les zones visuelles et les fonds.

Fonctions importantes:

- `_reconstruction_contract_key_for_block(...)`
- `_line_units(...)`
- `_normalize_placable_units(...)`
- `_linewise_fallback(...)`
- `_render_with_scale(...)`
- `_render_prose_reflow(...)`
- `TableBlockRenderer.render(...)`
- `EditorialBlockRenderer.render(...)`
- `StructuredContractRenderer.render(...)`

### `coverage_validator.py`

Role:

- mesurer ce qui a effectivement ete rendu,
- detecter:
  - texte manquant,
  - texte hors bloc,
  - style perdu,
  - source overlay,
  - glyph loss,
  - cellules tableau en echec,
  - fragments trop faibles,
  - surimpression.

### `background_master.py`

Role:

- produire un fond document nettoye,
- faire du char-local inpainting sur les zones ou le texte a ete retire,
- conserver la texture locale,
- fournir le support visuel pour la reconstruction.

### `text_removal_strategy.py`

Role:

- definir la strategie de suppression du texte,
- supporter le mode `char_local`,
- eviter le blanc uniforme quand le fond d'origine est colore ou texture.

### `layout_ai_enricher.py`

Role:

- enrichir les regions layout avec de l'IA quand disponible,
- ameliorer la detection de tableaux, diagrammes, regions visuelles et structures denses.

### `page_policy_matrix.py`

Role:

- matrice de politique par classe / type / semantics,
- decide:
  - translatable ou non,
  - strategie de traduction,
  - politique de rendu,
  - mode de reinjection.

## 5. Ce qui a ete implemente recemment

### Comprension / classification

Le projet ne s'arrete plus a de gros blocs: il descend a:

- block,
- line,
- phrase,
- span,
- inline special content.

On detecte et classe:

- texte courant,
- code,
- formules,
- URLs,
- DOI,
- emails,
- references,
- labels de figure / diagramme / table,
- micro-labels,
- tableaux et cellules,
- regions visuelles,
- TOC.

Derniere extension ajoutee:

- `inline_structure` dans le contrat objet,
- segmentation inline des fragments speciaux a l'interieur d'un meme bloc,
- fallback de traduction segmentee pour les titres TOC et les morceaux mixtes,
- protection explicite des segments techniques / URL / DOI / formules / identifiants.

### Contrat document objet

On a ajoute un contrat explicite:

- `document_object_contract`

Il dit pour chaque objet:

- `translatable` ou non,
- `translation_strategy`,
- `render_policy`,
- `reinject_mode`,
- `contract_key`,
- `geometry_mode`,
- `visual_structure`.

### TOC

Les TOC ont ete traites comme cas special:

- ligne structuree,
- titre traduisible,
- leaders `........` preserves,
- numero de page preserve,
- positionnement verrouille.

Etat actuel:

- les titres TOC courants passent par une traduction courte explicite,
- les lignes TOC mixtes utilisent aussi un fallback inline structure,
- mais certains fragments anglais residuels restent visibles sur les pages TOC tres denses,
- il faut encore durcir la traduction phrase/par ligne avant rendu final pour viser 100% de couverture visible.

### Cellules / tableaux denses

Les cellules ne doivent plus:

- reduire la police trop aggressivement,
- perdre le texte attendu,
- surimprimer l'original,
- casser le rendu.

La politique actuelle prefere:

- traduire si ca tient,
- sinon preserver le source exact dans les cellules verrouillees.

Le `page_rebalanced` est en place, mais reste conservateur. Il ne pousse pas encore les blocs voisins.

### Polices

Le rendu essaie de choisir une police compatible avec:

- accents,
- cedilles,
- apostrophes typographiques,
- caracteres speciaux de la langue cible.

### Fond / inpainting

La suppression du texte s'est alignee vers:

- inpainting char-local,
- pas de repaint regional global,
- preservation de la texture locale autour du caractere.

## 6. Resultats de validation deja observes

Sur `random5` de validation, on a obtenu globalement:

- couverture texte elevee,
- ordre de lecture tres bon,
- glyphes langue cible OK,
- pas de surimpression source/traduction sur les cas retestes,
- mais styles et cellules table encore faibles sur certains cas denses.

Exemples de cas:

- `doc1_p16`: tres bon.
- `doc3_p29`: tres bon.
- `doc3_p14`: tres bon.
- `doc2_p04`: encore des pertes dans les blocs denses / cellules.
- `doc1_p29`: encore le cas le plus difficile.

Dernier smoke cible sur `tests/doc_pdf/test_docintelligence-7.pdf`:

- sortie disponible dans `results/doc7_smoke_final_check/`,
- le PDF reconstruit est genere correctement,
- le TOC est bien traduit en grande partie,
- mais le comparatif visuel montre encore des fragments anglais et des artefacts noirs sur certaines lignes TOC denses.

## 7. Ce qui reste a faire

### Priorite 1: page_rebalanced

On doit aller plus loin sur:

- rebalancement de page,
- repartition inter-blocs,
- repositionnement local quand la traduction est plus longue,
- maintien de la police extraites quand possible,
- adaptation du volume du bloc sans degradations.

Le rebalancement actuel est une base conservative, pas encore un vrai moteur de reflow global.

### Priorite 2: tableaux / cellules tres denses

Il faut encore traiter:

- cellules avec micro-bboxes,
- cellules multi-lignes tres compactes,
- labels internes de schemas / tableaux,
- cellules ou la traduction devient plus large que l'espace source.

### Priorite 3: TOC et structures lineaires riches

Il faut encore durcir:

- lignes avec points leaders,
- sections hierarchiques,
- numerotation page,
- preservation de la structure visuelle.

Le point a reprendre ensuite est le chemin `TOC -> traduction phrase -> reconstruction ligne`, pour supprimer les derniers restes anglais dans les entrées denses.

### Priorite 4: visual regions complexes

Il faut continuer sur:

- vectoriels complexes,
- clipping,
- masks,
- z-order fins,
- figures denses,
- diagrammes complexes,
- schémas scientifiques.

### Priorite 5: QA / validation massive

Le `random20` complet a ete trop lourd a executer en monolithe.
Il faut:

- un runner resumable,
- sauvegarde page par page,
- timeout par page,
- resume sur dossier deja produit.

## 8. Logique de developpement a suivre

Le bon ordre de travail est:

1. Fixer la comprehension.
2. Traduire en fonction du contrat.
3. Reconstituer en fonction du contrat.
4. Valider sur cas cibles.
5. Corriger les regressions.
6. Rejouer les randoms.

Il ne faut pas:

- coder la traduction sans comprendre l'objet,
- coder la reconstruction sans contrat,
- generaliser un reflow unique a toutes les classes,
- laisser la traduction casser les objets techniques.

La bonne discipline est:

- un type / une classe doit avoir une politique claire,
- un bloc dense doit avoir un comportement specialise,
- un objet proteges doit rester protege,
- un fond doit etre reconstruit localement,
- la validation doit mesurer le rendu reel.

## 9. Etat du frontend

Le frontend est maintenant une UI HTML simple servie par FastAPI:

- fichier: `frontend/index.html`
- acces: `/`
- backend attendu: `/pipeline/run`

Le serveur FastAPI redirige maintenant `/` vers `/ui/`.

Lancement:

```bash
.docs-parser/bin/python ocr_server.py
```

Puis:

```text
http://127.0.0.1:8001/
```

## 10. Fichiers de travail et rapports utiles

### Pipeline / validation

- `scripts/run_reconstruction_validation.py`
- `scripts/run_reconstruction_validation_random20.py`
- `scripts/export_full_document_results.py`

### Contrat / comprehension / rendu

- `document_object_contract.py`
- `object_comprehension.py`
- `translator.py`
- `reconstructor.py`
- `page_policy_matrix.py`
- `coverage_validator.py`

### Fond / suppression texte

- `background_master.py`
- `text_removal_strategy.py`

### UI

- `frontend/index.html`

## 11. Ce qu'il faut reprendre en premier dans la suite

Ordre recommande:

1. Stabiliser `page_rebalanced`.
2. Durcir `table_cell` et les micro-labels internes.
3. Rejouer un `random20` resumable.
4. Inspecter les pages les plus faibles (`doc1_p29`, tableaux denses, schemas complexes).
5. Ajuster les policies dans `page_policy_matrix.py` si une classe reste ambiguë.

## 12. Resume brutal

On a maintenant:

- une comprehension plus profonde,
- un contrat explicite par objet,
- une traduction pilotée par ce contrat,
- une reconstruction pilotée par ce contrat,
- une UI unique pour lancer le pipeline,
- des validations sur des cas cibles.

Ce qui manque encore:

- fiabiliser les cas denses,
- rebalancer les pages,
- faire tenir le rendu sur les tableaux/diagrammes compliques,
- rendre la validation massive plus resumable.

Le prochain assistant doit reprendre avec ce principe: toute decision de traduction ou de rendu doit venir du contrat objet, pas d'un heuristique locale isolee.

Derniere consigne pratique:

- reprendre par petits smokes cibles,
- verifier visuellement `doc7` avant de lancer un gros random,
- ne pas toucher aux cas proteges si la couverture visible baisse.

## 13. Analyse comparative finale du dernier smoke

Dernier smoke cible:

- source: `tests/doc_pdf/test_docintelligence-7.pdf`
- rendu: `results/doc7_smoke_final_check/reconstructed_fr.pdf`
- comparatif: `results/doc7_smoke_final_check/compare_original_reconstructed.png`

Constat global:

- la page n'est plus vide,
- la base de reconstruction tient,
- mais le rendu final reste loin d'un WYSIWYG propre sur le sommaire dense.

Problemes visibles et textuels encore presentes:

- plusieurs entrees TOC restent partiellement ou totalement en anglais,
- certaines traductions sont lexicalement mauvaises ou tronquees,
- plusieurs lignes sont coupees au mauvais endroit,
- des fragments sont dupliques ou désordonnés,
- des `■` parasites apparaissent et parfois se multiplient,
- la ponctuation et la numerotation sont degradees (`1,7`, `2,5`, `2,6`),
- des apostrophes et césures bizarres apparaissent (`Qu'est­ce`, `l'propagation`),
- certains titres TOC sont encore reemergents en anglais dans le rendu.

Liste des entrées encore problématiques observées:

- `What is a feature in computer vision?`
- `What makes a good (useful) feature?`
- `Extracting features (handcrafted vs. automatic extracting)`
- `What are hidden layers?`
- `How many layers, and how many nodes in each layer?`
- `Some takeaways from this section`
- `Sigmoid/logistic function`
- `Why do we need an error function?`
- `Batch gradient descent`
- `Mini-batch gradient descent`
- `Gradient descent takeaways`
- `What is backpropagation?`
- `Backpropagation takeaways`

Traductions visibles encore mauvaises:

- `Calculs de l'propagation avant`
- `Enseignement de la matière`
- `Pourquoi nous avons fonction?`
- `carré erreur`
- `Entropie croisée` éclaté
- `Déformation du gradient`
- `Qu'est-ce que la propagande de dos?`
- `Prises de propagande arrière`

Symptomes structurels:

- l'original a 111 lignes extraites sur cette page; le rendu en a 119,
- cela indique une sur-segmentation dans la reconstruction,
- le probleme principal n'est plus l'extraction brute, mais le réassemblage et la réinjection des sous-fragments TOC.

Priorite de reprise:

1. Refaire le chemin `toc_entry` dense ligne par ligne, avec réassemblage strict.
2. Supprimer les fragments anglais residuels avant rendu final.
3. Eliminer les `■` parasites et la sur-segmentation.
4. Corriger la ponctuation, les apostrophes et la numerotation TOC.
5. Rejouer un smoke sur `doc7` avant tout random large.
