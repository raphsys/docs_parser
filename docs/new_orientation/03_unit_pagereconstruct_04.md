Oui. J’ai repris `pagereconstruct.zip`, `pageprint.zip`, `pagetranslate.zip` et `show10_bgphrase.zip`.

La nouvelle implémentation a bien intégré les deux modules :

```text
pagereconstruct/ocr_typography_engine.py
pagereconstruct/multiblock_layout_solver.py
pagereconstruct/integration_adapter.py
```

Mais il y a un problème important : **les scores annoncent quasiment 95 %, alors que les pages ne sont pas visuellement publication-ready.**

J’ai généré une nouvelle planche de contrôle :

[Voir la comparaison show10_bgphrase source / reconstruit / overlay](sandbox:/mnt/data/show10_bgphrase_contact_sheet.png)

---

# 1. Verdict immédiat

Le pipeline annonce maintenant :

```text
Score moyen audit      : 0.96
Pages publication-ready: 8 / 10
Pages à 1.0            : 8 / 10
Pages à 0.8            : 2 / 10
```

Mais visuellement, ce n’est pas vrai.

Exemples évidents :

```text
Practical SQL p0505 : index illisible, textes superposés, entrées écrasées.
Practical SQL p0180 : code / texte source / texte traduit mélangés.
Practical SQL p0133 : ancien texte anglais encore visible, grosse zone non traduite.
Advances p0140      : tableau, graphique, labels et paragraphes très perturbés.
```

Donc la nouvelle situation est dangereuse :

```text
Avant : le moteur disait review/ko quand la page était mauvaise.
Maintenant : le moteur peut dire ok/1.0 alors que la page est mauvaise.
```

Ce n’est pas seulement une erreur de rendu. C’est un problème de **QA qui devient aveugle**.

---

# 2. Ce qui a réellement été implémenté

## 2.1 Le moteur typographique OCR existe

Le fichier `ocr_typography_engine.py` fait bien :

```text
mesure cap-height / x-height via composants connexes ;
estimation em-size ;
échelle typographique par rôle ;
patch de style par block_id ;
confidence score.
```

C’est une bonne base.

Mais actuellement, il ne corrige pas vraiment le rendu. Il ajoute surtout :

```text
font_size_pt_em
font_class
confidence
typo_method
```

Puis dans `apply_typography_patches_in_place`, il y a cette logique :

```python
# NE PAS écraser font_size_pt
setattr(style, "font_size_pt_em", ...)
```

Donc le moteur typographique **ne modifie pas la taille rendue**. Il modifie surtout les métadonnées de confiance typographique.

Ensuite `plan_compiler.py` retire certains findings `font_size_repaired_from_line_geometry` si l’em-size image semble fiable.

Résultat :

```text
Le score typographique monte.
Mais le rendu typographique n’est pas nécessairement meilleur.
```

C’est une correction de **notation**, pas encore une correction visuelle complète.

---

## 2.2 Le solveur multi-blocs existe

Le fichier `multiblock_layout_solver.py` existe aussi.

Mais dans les plans `show10_bgphrase`, j’ai vérifié :

```text
multiblock_applied = None sur les 10 pages
```

Donc le solveur multi-blocs n’a pratiquement pas corrigé les pages. Il est appelé, mais il n’applique rien, parce que sa logique est trop prudente :

```text
déplacer seulement si overlap détecté dans les bboxes du contrat ;
déplacement limité à ±36 pt ;
net-improvement only ;
pas de vraie recomposition verticale ;
pas de vrai packing ;
pas de modèle index.
```

Le cas `p0505` prouve que ce solveur ne voit pas le vrai problème : visuellement l’index est détruit, mais les `TextOps` ne se chevauchent pas selon ses boîtes internes.

---

# 3. Le vrai problème actuel

Le moteur a maintenant deux couches :

```text
rendu réel mauvais
+
audit trop optimiste
```

C’est plus grave que juste “il manque une amélioration”.

Le système pense avoir atteint 0.96 parce que :

```text
1. la typographie OCR est considérée résolue par métadonnées ;
2. les collisions sont mesurées sur les TextOps, pas sur l’image réelle ;
3. la QA image-réelle n’est pas réellement obligatoire ;
4. le clean background est déclaré low leak sans vérification suffisante ;
5. les anciens textes conservés dans le fond ne sont pas comptés comme collisions ;
6. certains objets textualisés sont préservés comme pixels au mauvais endroit.
```

---

# 4. Analyse profonde des deux verrous

## Verrou A — Typographie OCR

### Ce qui a été fait

Le moteur estime une taille typographique OCR à partir de :

```text
cap-height ;
x-height ;
glyph height ;
line height ;
raw font size si fiable ;
rôle du bloc ;
échelle de page.
```

C’est conceptuellement bon.

### Ce qui ne va pas

Le moteur ne doit pas seulement produire :

```text
font_size_pt_em = 9.8
confidence = 0.82
```

Il doit prouver que le rendu final ressemble à la source.

Actuellement, il y a confusion entre :

```text
typographie comprise
```

et :

```text
typographie rendue fidèlement
```

Ce sont deux choses différentes.

Le bon pipeline doit être :

```text
OCRTypographyEngine
→ estime em-size source
→ propose une taille de rendu candidate
→ renderer rend avec cette taille
→ VisualQA mesure le rendu final
→ compare source vs reconstruit
→ valide ou refuse
```

Aujourd’hui, on a plutôt :

```text
OCRTypographyEngine
→ estime em-size
→ ajoute métadonnée
→ retire findings
→ score typographie monte
```

C’est insuffisant.

### Correction conceptuelle

Il faut séparer trois tailles :

```text
font_size_pt_source_metric    : taille brute extraite/OCR
font_size_pt_em_estimated     : taille typographique estimée
font_size_pt_rendered         : taille réellement utilisée par le backend
```

Et il faut comparer visuellement :

```text
source ink height
source line height
source interline
source density
source glyph proportions

vs

reconstructed ink height
reconstructed line height
reconstructed interline
reconstructed density
reconstructed glyph proportions
```

Tant que cette comparaison n’est pas faite, on ne peut pas donner 1.0 en typographie.

---

## Verrou B — Placement multi-blocs

### Ce qui a été fait

Le solveur actuel construit des régions :

```text
page_flow
col_left
col_right
```

Puis il tente des déplacements verticaux locaux.

### Ce qui ne va pas

Il ne résout pas une page. Il corrige seulement quelques bboxes.

Or une page dense doit être traitée comme un système :

```text
bloc A pousse bloc B ;
bloc B pousse bloc C ;
une figure bloque une zone ;
un index impose colonnes + indentation ;
un code block est verrouillé ;
une sortie terminal doit rester monospacée ;
un caption dépend d’une figure ;
un tableau impose sa grille.
```

Le cas `Practical SQL p0505` est l’exemple parfait.

La page est un index. Elle ne peut pas être sauvée par :

```text
shift_y ±36 pt
```

Il lui faut un vrai modèle :

```text
index_entry
index_subentry
index_page_reference
indentation_level
column_id
reference_alignment_x
```

Autre problème : le solveur mesure les bboxes contractuelles / TextOps, mais pas les **pixels réellement visibles**.

Sur `p0505`, le rendu est visuellement superposé, mais le solveur ne voit pas de collision car les boîtes déclarées ne représentent pas les vieux pixels encore dans le fond.

---

# 5. Le nouveau verrou caché : QA faussement positive

Avant de continuer les deux moteurs, il faut corriger la QA. Sinon chaque amélioration sera sur-notée.

## 5.1 `validator.py` appelle VisualQA sans images

Dans `validator.py`, on a :

```python
from .visual_qa import assess as visual_assess
vqa = visual_assess(plan)
```

Donc `_image_leak_score()` n’est pas appelé.

Il faudrait :

```python
vqa = visual_assess(
    plan,
    source_image_path=...,
    reconstructed_image_path=...
)
```

En mode publication, si ces images ne sont pas fournies :

```text
status = review/ko
publication_ready = False
```

Actuellement, si l’image-réelle n’est pas fournie, le score source leak peut être déduit seulement de :

```text
background.source_text_leak_risk = low
```

C’est trop faible.

---

## 5.2 `source_text_leak_detector` est trop simple

Il regarde si les pixels changent entre source et reconstruit dans les zones patch.

Mais dans ce run :

```text
background_mode = clean_background
render_ops patch = 0
```

Donc les patchs ne sont pas exécutés. Le système suppose que le fond propre a déjà tout nettoyé.

Problème : visuellement, le fond propre garde encore des textes source dans certaines zones.

Il faut donc vérifier le **clean background lui-même**, pas seulement les patchs.

Nouvelle règle :

```text
clean_background ne peut pas être déclaré low leak
sans comparaison source vs clean_background dans toutes les zones textuelles à remplacer.
```

---

## 5.3 Les bboxes de préservation sont trop larges

Sur `Practical SQL p0505`, les `PreservationOp` pour les références d’index ont des bboxes de ligne entière.

Exemple conceptuel :

```text
reason = index_page_reference
text = 114
bbox = [72.0, 75.675, 252.098, 90.675]
```

Mais cette bbox couvre toute l’entrée :

```text
Public Libraries Survey, 114
```

Donc le moteur préserve potentiellement l’ancien texte anglais, pas seulement le numéro `114`.

Erreur de philosophie :

```text
index_page_reference ne doit pas être une PreservationOp full-line.
```

Une référence de page dans un index est du texte structuré. Elle doit être rendue par `IndexRenderer`, pas copiée comme pixel source, sauf si on dispose d’une bbox exacte du numéro seulement.

C’est une cause directe des superpositions dans l’index.

---

# 6. Correction profonde à implémenter maintenant

## Phase 1 — Corriger la QA avant de croire les scores

Objectif : empêcher les faux `1.0`.

### Tâches

```text
[ ] Dans validator.py, interdire publication_ready si VisualQA image-réelle non exécutée.

[ ] visual_qa.assess doit retourner :
    image_qa_executed: true/false
    clean_background_checked: true/false
    source_text_leak_checked: true/false

[ ] Si reconstruction_mode == publication et image_qa_executed == false :
    publication_ready = false
    publication_ready_score <= 0.70
    finding = image_real_qa_missing

[ ] Ajouter dans audit JSON :
    visual_scores.text_presence
    visual_scores.non_text_presence
    visual_scores.overlap
    visual_scores.position
    visual_scores.typography
    visual_scores.source_text_leak
    image_qa_executed
```

### Tests

```text
test_publication_blocks_without_image_real_qa
test_visual_qa_requires_source_and_reconstructed_images
test_audit_json_exposes_image_qa_executed
```

---

## Phase 2 — Créer un `TextRemovalLedger`

Objectif : vérifier que chaque texte source remplacé a réellement disparu du fond propre.

Créer :

```text
pagereconstruct/text_removal_ledger.py
```

### Entrées

```text
pageprint units
pagetranslate reconstruction_units
FinalReconstructionContract blocks
source image
clean background image
reconstructed image
```

### Pour chaque bloc traduisible

Créer une entrée :

```text
TextRemovalEntry:
    source_unit_ids
    source_text
    translated_text
    source_bbox
    removal_bbox
    expected_action:
        clean_background_removed
        patch_removed
        preserve_exact
        not_translatable
    removal_verified: bool
    residual_ink_score
    findings
```

### Vérification

Comparer :

```text
source crop
clean_background crop
reconstructed crop
```

Mesures :

```text
ink_density_source
ink_density_clean_background
ink_density_reconstructed
difference_score
residual_source_text_score
```

Règles :

```text
Si texte remplacé :
    clean_background doit avoir une forte baisse d’encre source.

Si clean_background garde encore l’ancien texte :
    source_text_leak_detected
    publication_ready = false

Si texte préservé :
    il doit être explicitement déclaré preserved.
```

### Tests

```text
test_text_removal_ledger_contains_all_translatable_blocks
test_clean_background_must_remove_source_text
test_residual_source_text_blocks_publication
test_untranslated_source_text_in_background_detected
```

---

## Phase 3 — Corriger les références d’index

Objectif : empêcher `p0505` d’être marqué OK alors qu’il est illisible.

### Règle

```text
index_page_reference n’est pas un overlay pixel full-line.
```

### Tâches

```text
[ ] Supprimer index_page_reference des PreservationOps sauf bbox exacte du numéro.

[ ] Créer IndexLineModel :
    term_text
    subterm_text
    page_refs
    term_bbox
    refs_bbox
    indentation_level
    column_id

[ ] IndexRenderer doit rendre :
    terme à gauche
    page_refs à droite ou après virgule selon source
    sous-entrées indentées
    interligne constant
    pas de superposition
```

### Si pageprint ne donne pas la bbox exacte du numéro

Faire estimation :

```text
refs_bbox = partie droite de la ligne après dernière virgule numérique
```

Mais ne jamais préserver toute la ligne comme `index_page_reference`.

### Tests

```text
test_index_page_reference_not_full_line_preservation
test_index_renderer_renders_page_refs
test_index_renderer_preserves_indentation
test_p0505_index_no_visual_overlap
```

---

## Phase 4 — Corriger le moteur typographique OCR

Objectif : la typographie doit améliorer le rendu, pas seulement les scores.

### Nouvelle règle

```text
font_size_pt_em ne suffit pas à valider la typographie.
```

### Tâches

```text
[ ] OCRTypographyEngine doit produire :
    source_em_size
    proposed_render_size
    line_height_target
    confidence
    visual_expected_ink_height

[ ] Ne pas supprimer font_size_repaired findings seulement parce que em-size existe.

[ ] Créer TypographyVisualVerifier :
    compare source image crop vs reconstructed image crop.

[ ] typography_score doit être basé sur :
    font_class_match
    rendered_ink_height_ratio
    rendered_line_height_ratio
    visual_density_ratio
    bold/italic approximation
    color similarity
    alignment/indentation
```

### Règle de score

```text
Si em-size estimée mais non utilisée dans rendu :
    typography_score max = 0.85

Si em-size utilisée mais rendu visuel ne correspond pas :
    typography_score max = 0.85

Si source/reconstruit visuellement cohérents :
    typography_score peut monter > 0.95
```

### Tests

```text
test_em_size_metadata_alone_does_not_lift_typography_to_1
test_typography_score_uses_reconstructed_image
test_rendered_ink_height_matches_source
test_typography_score_blocks_visual_mismatch
```

---

## Phase 5 — Corriger le solveur multi-blocs

Objectif : passer de correction locale à composition de région.

### Actuel

```text
détection bbox contractuelle
shift_y limité
pas de repacking
pas de modèle index
pas de vérification raster
```

### Cible

Créer :

```text
pagereconstruct/multiblock_flow_solver.py
```

### Algorithme

```text
1. Construire les FlowRegions :
    page_flow
    col_left
    col_right
    index_column
    figure_text_band
    code_region
    bibliography_region

2. Construire les RenderBoxes réels :
    actual_line_boxes
    actual_text_bbox
    residual_background_text_boxes
    protected_boxes

3. Détecter les clusters de collision :
    text/text
    text/background_residual
    text/protected
    text/preservation

4. Pour chaque cluster :
    identifier blocs mobiles
    identifier blocs verrouillés
    générer candidats :
        original
        shrink léger
        line-height compact
        vertical repack
        region expansion
        local reflow
        fail

5. Résoudre par région :
    préserver ordre de lecture
    interdire protected overlap
    minimiser déplacement
    maximiser typographie
    éviter collision

6. Re-render.
7. Re-QA.
8. Itérer max 2 ou 3 fois.
```

### Important

Le solveur ne doit pas seulement traiter :

```text
block.layout_bbox
```

Il doit traiter :

```text
actual rendered line boxes
+
residual old text boxes
+
preserved object boxes
```

### Tests

```text
test_multiblock_solver_uses_actual_render_boxes
test_multiblock_solver_detects_background_residual_collision
test_multiblock_solver_reflows_dense_region
test_multiblock_solver_rejects_false_ok_page
test_p0180_collision_not_marked_ok
test_p0505_index_not_marked_ok
```

---

## Phase 6 — Corriger les pages code / SQL

`Practical SQL p0180` montre que le code/source/texte traduit se mélangent.

Règles :

```text
SQL code block = préserver exact par défaut.
Caption/prose = traduire.
Output terminal = préserver exact ou traduire seulement prose explicative.
```

Tâches :

```text
[ ] CodeRenderer ne doit pas traiter SQL comme body_paragraph.
[ ] Les lignes EXPLAIN ANALYZE doivent être code_block/code_line.
[ ] Si code est conservé dans clean_background, ne pas redessiner traduction dessus.
[ ] Si code est rendu par CodeRenderer, clean_background doit avoir supprimé l’ancien code.
```

Tests :

```text
test_sql_code_not_body_paragraph
test_code_block_not_translated_as_prose
test_code_background_and_textop_not_double_rendered
test_p0180_code_area_no_overlap
```

---

# 7. Point critique : les scores actuels ne doivent plus être acceptés

Aujourd’hui, `Practical SQL p0505` reçoit :

```text
publication_ready_score = 1.0
publication_ready = true
```

Mais visuellement, la page est non publiable.

Donc il faut ajouter une règle de sécurité :

```text
Si VisualQA image-réelle absente ou incomplète :
    score maximal = 0.70
```

Et une règle spécifique :

```text
Si une page index contient PreservationOp index_page_reference avec bbox large :
    score maximal = 0.60
```

---

# 8. Plan très concret pour Claude

À donner tel quel :

```text
MISSION — Verrous 95 %, passe 2

Le run show10_bgphrase annonce avg 0.96 et 8/10 publication-ready.
C’est un faux positif. Visuellement p0505, p0180, p0133 et p0140 ne sont pas publiables.

Objectif de cette passe :
corriger la QA et rendre les deux verrous réellement actifs.

1. Corriger VisualQA :
   - validator doit appeler visual_qa avec source_image_path et reconstructed_image_path.
   - en mode publication, absence d’image QA = pas publication-ready.
   - ajouter image_qa_executed dans audit.

2. Créer TextRemovalLedger :
   - vérifier que chaque source text remplacé est absent du clean_background.
   - détecter ancien texte visible.
   - bloquer publication si résidu.

3. Corriger index preservation :
   - index_page_reference ne doit pas être keep_pixels sur bbox de ligne entière.
   - les références de page doivent être rendues par IndexRenderer.
   - bbox page refs exacte ou reconstruction textuelle.

4. Corriger OCRTypographyEngine :
   - font_size_pt_em seul ne valide pas la typo.
   - typography_score doit comparer source/reconstructed image crops.
   - si em-size non utilisé dans rendu, score max typo 0.85.

5. Corriger MultiBlockLayoutSolver :
   - utiliser actual render boxes, residual text boxes et protected boxes.
   - résoudre par flow region.
   - ajouter repacking vertical.
   - traiter index comme layout spécialisé.

6. Corriger code/SQL :
   - code block préservé exact par défaut.
   - pas double rendu source code + traduction.
   - p0180 ne doit plus être marqué OK si code chevauché.

Objectif réaliste de cette passe :
- ne plus avoir de faux 1.0 ;
- p0505/p0180/p0133 doivent être review/ko tant qu’elles sont visuellement mauvaises ;
- si corrigées, elles peuvent passer.
```

---

# 9. Conclusion nette

Les deux verrous ont été **implémentés nominalement**, mais pas encore **résolus réellement**.

## Typographie OCR

Actuel :

```text
em-size estimé → score amélioré
```

Cible :

```text
em-size estimé → rendu ajusté → image vérifiée → score validé
```

## Placement multi-blocs

Actuel :

```text
shift local si overlap bbox détecté
```

Cible :

```text
composition par région, avec actual render boxes et repacking
```

## Nouveau verrou à corriger immédiatement

```text
VisualQA doit redevenir fiable.
```

Parce qu’un score à 1.0 sur `p0505` est la preuve que le système ne voit pas encore ce que l’œil voit.

Donc la prochaine passe doit corriger dans cet ordre :

```text
1. QA image-réelle obligatoire.
2. TextRemovalLedger.
3. IndexRenderer + suppression des index_page_reference full-line.
4. Typography visual verifier.
5. MultiBlockFlowSolver.
6. Code/SQL no double-render.
```

Là, on pourra reparler sérieusement du 95 %.

