Oui. J’ai inspecté `rev_09`, le dossier `pagereconstruct/`, les sorties de `pipeline_full_demo_10p_fix_210337`, puis les couples **source / reconstructed / overlay**.

J’ai aussi généré deux planches visuelles pour l’analyse :

* [Comparaison globale source / reconstruit / overlay](sandbox:/mnt/data/reconstruction_contact_sheet.png)
* [Crops typographiques détaillés source / reconstruit](sandbox:/mnt/data/reconstruction_typography_crops.png)

Verdict direct : **le pipeline consomme déjà une partie de `pageprint/` et `pagetranslate/`, mais il ne les consomme pas encore suffisamment pour reconstruire correctement le style et la typographie.** Le problème principal n’est plus la traduction. Le problème est maintenant la **restitution typographique et graphique**.

---

# 1. État réel de `pagereconstruct/`

Dans `rev_09`, `pagereconstruct/` contient seulement :

```text
pagereconstruct/
├── __init__.py
├── errors.py
├── input_adapter.py
├── plan_compiler.py
├── protected_region_index.py
├── render_backend.py
└── schema.py
```

Donc on n’a pas encore un vrai reconstructeur WYSIWYG complet. On a surtout :

```text
1. un adaptateur d’entrée ;
2. un compilateur de plan ;
3. un index de régions protégées ;
4. un backend raster PIL très simple ;
5. un schéma PageRenderPlan.
```

Les tests passent :

```text
93 passed
```

C’est positif, mais les tests actuels vérifient surtout le **contrat structurel**, pas encore la fidélité typographique.

---

# 2. Résultat global sur les 10 pages de démo

J’ai mesuré les plans `pagereconstruct_plan_*.json`.

Résultat :

```text
Nombre de pages analysées        : 10
Unités de texte traduites         : 152
Unités avec style non vide        : 0
Patch zones déclarées dans plan   : 0
Régions protégées                 : 2214
Éléments préservés                : 133
Findings                          : 34
```

Le chiffre critique est celui-ci :

```text
style_nonempty = 0 / 152
```

Cela signifie que **toutes les unités reconstruites sont rendues sans style exploitable**.

C’est la cause principale des échecs visuels.

---

# 3. Le problème typographique est massif

Sur les images reconstruites, on observe :

```text
police perdue ;
graisse perdue ;
italique perdu ;
taille perdue ;
couleur perdue ;
alignement perdu ;
interlignage perdu ;
justification perdue ;
indentation perdue ;
baseline perdue ;
espacement source perdu ;
hiérarchie titre / corps perdue ;
texte parfois tronqué ou mal réparti.
```

Exemple très clair : page `Advances in Deep Learnin_p0006`.

Original :

```text
Titre "Preface" en serif gras.
Corps en serif, taille modérée, justifié, avec indentation et césures.
```

Reconstruit :

```text
"Préface" en sans-serif régulier, plus petit, sans graisse.
Corps en sans-serif noir, beaucoup trop gros, non justifié, sans indentation.
```

Donc le style n’est pas seulement “un peu approximatif”. Il est **largement perdu**.

---

# 4. Pourquoi le style est perdu

Il y a trois causes principales.

## 4.1 `pagetranslate/projection.py` perd le style sur les unités sémantiques

Dans `pagetranslate/projection.py`, les unités directes passent par :

```python
_direct_reconstruction_unit(...)
```

et récupèrent un style via :

```python
"style": _dominant_style(unit, unit_map)
```

Mais les unités sémantiques passent par :

```python
_semantic_reconstruction_unit(...)
```

et là, il n’y a pas de vrai style. On a seulement :

```python
"style_source": "dominant_source_span"
```

Mais aucun `style_source_unit_id` exploitable, aucun `style` réel, aucun snapshot typographique.

Résultat :

```text
PAGETRANSLATE produit des reconstruction_units textuellement valides,
mais typographiquement pauvres.
```

## 4.2 `pagereconstruct/plan_compiler.py` ne résout pas le style

Dans `plan_compiler.py`, chaque `TranslatedTextUnit` est construit avec :

```python
style=item.get("style") or {}
```

Donc si `pagetranslate` ne fournit pas de style, `pagereconstruct` garde `{}`.

Il ne va pas chercher :

```text
render_target.style_source_unit_id
views.reconstruction_plan[].style_source_unit_id
units[].visual.style
children spans
style_system
dominant_body_style
```

C’est une lacune majeure.

## 4.3 `render_backend.py` ignore complètement la typographie source

Le backend actuel est un rendu raster PIL très simple :

```python
draw.rectangle(..., fill=(255, 255, 255))
draw.text(..., fill=(15, 15, 15), font=DejaVuSans)
```

Il utilise :

```text
DejaVuSans
noir
taille calculée depuis la hauteur de bbox
whiteout blanc fixe
wrap simple par mots
aucun alignement
aucune justification
aucune police PDF source
aucun gras/italique
aucune couleur
aucun interlignage source
aucune baseline source
```

Donc même si `pageprint` avait extrait le style, le backend actuel ne saurait pas encore l’utiliser correctement.

---

# 5. Est-ce que `pageprint/` et `pagetranslate/` sont bien consommés dans `pagereconstruct/` ?

Réponse courte :

```text
Partiellement oui.
Correctement pour le texte brut traduit.
Insuffisamment pour le style, les contraintes, les fonds, les patchs et la reconstruction WYSIWYG.
```

## 5.1 Ce qui est consommé correctement

`pagereconstruct/input_adapter.py` lit bien :

```text
views.reconstruction_units
views.reconstruction_plan
views.preservation_plan
views.exclusion_plan
units
regions
assets
visual_layers
```

`plan_compiler.py` consomme réellement :

```text
views.reconstruction_units
views.preservation_plan
views.exclusion_plan
units
regions
unit policies
```

Donc le texte traduit venant de `pagetranslate` est bien utilisé.

Le système évite aussi une partie des doublons parent/enfant via :

```python
duplicate_render_skipped
```

C’est positif.

## 5.2 Ce qui est mal ou pas consommé

Le point critique :

```text
views.reconstruction_plan est exposé par l’adapter, mais pratiquement ignoré par plan_compiler.py.
```

Dans `plan_compiler.py`, on lit :

```python
translated_units = normalized["translated_units"]
preservation_plan = normalized["preservation_plan"]
exclusion_plan = normalized["exclusion_plan"]
```

Mais `normalized["reconstruction_plan"]` n’est pas réellement fusionné dans le plan final.

Conséquence :

```text
style_source_unit_id
render_contract détaillé
bbox_policy
overflow_policy
source de style
contraintes de reconstruction
text_source
mode translated_text / fixed_preserve
```

ne sont pas pleinement exploités.

Autres non-consommations importantes :

```text
style_system                 non consommé
reconstruction_constraints   non consommé
visual_layers.background     non consommé
assets.background_path       non consommé
patches du plan              non produits
protected_regions            produits mais non appliqués par le backend
preserved_overlays           produits mais non redessinés réellement
font_resolver.py             non utilisé
ancien savoir typographique  non migré depuis reconstructor.py.bak
```

Donc la réponse nette est :

```text
PAGERECONSTRUCT consomme les bons objets, mais pas encore les bonnes profondeurs de ces objets.
```

---

# 6. Les erreurs visuelles constatées

## 6.1 Perte des polices

Les pages source utilisent beaucoup de serif, souvent proches de Times/serif éditorial.

Les pages reconstruites utilisent visuellement du sans-serif, type DejaVuSans.

Exemples :

```text
Preface       → serif bold dans l’original, sans-serif regular dans le reconstruit
Bibliography  → serif bold dans l’original, sans-serif regular dans le reconstruit
corps de texte → serif justifié, reconstruit en sans-serif non justifié
```

## 6.2 Taille de police fausse

`render_backend.py` calcule la taille ainsi :

```python
max_size = max(7, min(h * 0.95, 28))
```

Donc la taille dépend surtout de la hauteur du bloc en pixels.

C’est faux pour un rendu WYSIWYG. La taille doit venir de :

```text
font_size_pt source
style_source_unit_id
line_height source
bbox source
contrat de shrink
```

Sur les pages reconstruites, les paragraphes sont souvent trop gros.

## 6.3 Graisse et italique perdus

Original :

```text
titres gras
headers italiques
références avec gras partiel
termes techniques en italique
```

Reconstruit :

```text
texte régulier uniforme
```

C’est particulièrement visible sur les pages `test_docintelligence_p0242` et `Advances_p0103`.

## 6.4 Couleurs perdues

Exemple : page `test_docintelligence_p0242`.

Original :

```text
titre de bloc bleu dans une zone beige
labels et figures conservés avec couleurs
```

Reconstruit :

```text
texte traduit noir
whiteout blanc dans une zone beige
mélange ancien texte / nouveau texte
```

Le fond coloré n’est pas respecté.

## 6.5 Whiteout destructeur

Le backend actuel efface en blanc :

```python
draw.rectangle(..., fill=(255, 255, 255))
```

C’est acceptable uniquement sur une page blanche.

Mais dans les figures ou blocs colorés, c’est mauvais :

```text
bloc beige → rectangle blanc visible
figure → rectangle blanc qui casse la trame
tableau coloré → patch blanc artificiel
```

Il faut remplacer cela par :

```text
clean background
inpainting local
couleur échantillonnée
patch local contrôlé
```

## 6.6 Protected regions non appliquées au rendu

Le plan contient beaucoup de régions protégées :

```text
2214 régions protégées sur 10 pages
```

Mais `render_backend.py` n’utilise pas `protected_regions`.

Donc actuellement :

```text
les régions protégées existent dans le plan,
mais elles ne protègent pas réellement le rendu.
```

C’est dangereux.

## 6.7 `patch_count = 0`

Les plans contiennent :

```text
patch_count = 0
```

alors que le backend applique des whiteouts implicites.

C’est une erreur architecturale :

```text
Le plan doit déclarer les patchs.
Le backend doit exécuter les patchs.
Le backend ne doit pas inventer des patchs non audités.
```

Actuellement, le rendu ne respecte pas encore le principe :

```text
plan explicite → opérations explicites → validation
```

## 6.8 Mauvaise classification de certaines pages

Exemple très grave : `Advances in Deep Learnin_p0103`.

La page source est une page classique de livre avec titre, paragraphe et bibliographie.

Mais `pageprint` la classe comme :

```text
page_role: table_page
```

et beaucoup de blocs deviennent :

```text
table_body_cell
```

Conséquence :

```text
paragraphes traduits ligne par ligne ;
renderer = table ;
fragmentation du texte ;
mauvaise logique de style ;
mauvaise logique de reconstruction.
```

Donc il y a un problème upstream dans `pageprint` :

```text
détection table trop agressive
page_role table_page faux positif
rôle table_body_cell attribué à du texte éditorial
```

---

# 7. Analyse de la chaîne complète

## 7.1 `pageprint/`

`pageprint` est globalement solide comme base structurelle. Il produit :

```text
translation_plan
reconstruction_plan
preservation_plan
exclusion_plan
logical_structures
units
regions
policies
constraints
```

Mais plusieurs faiblesses impactent directement `pagereconstruct`.

### Problème A — rôle sémantique parfois dégradé

Exemple `Preface`.

Dans le résumé `pageprint`, le bloc est bien identifié :

```text
role: title
```

Mais dans `pagetranslate`, il devient :

```text
role: body_paragraph
```

Donc quelque part entre :

```text
units[]
semantic_system.translation_segments
translation_plan
pagetranslate
reconstruction_units
```

le rôle est perdu ou remplacé.

Directive :

```text
Un titre source ne doit jamais devenir body_paragraph dans reconstruction_units.
```

### Problème B — faux positifs table

Des pages narratives deviennent `table_page` ou `table_body_cell`.

Directive :

```text
Ne classer en table que s’il existe une preuve forte de grille :
- lignes verticales/horizontales ;
- alignement cellulaire régulier ;
- colonnes répétées ;
- structure tabulaire stable ;
- densité cellulaire cohérente.
```

Un simple alignement de lignes de texte ne doit pas suffire.

### Problème C — reconstruction_plan trop peu exploité

`pageprint/view_compiler.py` produit `reconstruction_plan`, mais celui-ci doit contenir davantage :

```text
style_source_unit_id
style_snapshot
source_line_metrics
source_baseline
source_alignment
line_height_pt
font_size_policy
bbox_policy
overflow_policy
renderer_hint
patch_policy
```

Aujourd’hui il contient une bonne base, mais `pagereconstruct` ne l’utilise pas assez.

---

## 7.2 `pagetranslate/`

La traduction fonctionne suffisamment pour tester la reconstruction.

Le problème est la projection.

Pour les unités sémantiques :

```python
_semantic_reconstruction_unit(item)
```

il faut ajouter un vrai style.

Actuellement :

```python
"style_source": "dominant_source_span"
```

est trop vague.

Il faut plutôt :

```python
"style_source_unit_id": "...",
"style": {...},
"source_line_styles": [...],
"source_line_metrics": [...],
"dominant_style_confidence": ...
```

Directive :

```text
PAGETRANSLATE ne doit pas seulement réinjecter du texte.
Il doit produire des unités reconstructibles typographiquement.
```

---

## 7.3 `pagereconstruct/`

`pagereconstruct` est au stade :

```text
Pass 1 — plan compilation
Pass 1.5 — raster demo backend
```

Ce n’est pas encore :

```text
vrai reconstructeur WYSIWYG
```

Il manque encore :

```text
StyleResolver
FontResolverBridge
TextMeasurer
PatchPlanner
BackgroundResolver
RendererDispatcher
ParagraphRenderer
HeadingRenderer
TableCellRenderer
CaptionRenderer
AnchoredLabelRenderer
PreservationRenderer
Validator
QualityAssessor
PDFBackend vectoriel
```

---

# 8. Décision architecturale

Il ne faut pas améliorer le rendu en bricolant seulement `render_backend.py`.

Le rendu actuel est un prototype visuel.

La bonne décision :

```text
render_backend.py doit rester un backend de debug raster.
Le vrai backend WYSIWYG doit être un PDFBackend vectoriel basé sur PyMuPDF.
```

Pourquoi ?

Parce que le WYSIWYG exige :

```text
fontfile
font size pt
baseline
insert_textbox / insert_text
couleur
gras/italique
alignement
justification
line height
contrôle overflow
mesure largeur texte
audit glyphes manquants
```

PIL peut aider pour debug, mais il ne doit pas être le backend final.

---

# 9. Directives d’implémentation prioritaires

## P0 — Consommer réellement `views.reconstruction_plan`

Modifier `pagereconstruct/plan_compiler.py`.

Actuellement, il faut passer de :

```python
translated_units = normalized["translated_units"]
preservation_plan = normalized["preservation_plan"]
exclusion_plan = normalized["exclusion_plan"]
```

à :

```python
translated_units = normalized["translated_units"]
reconstruction_plan = normalized["reconstruction_plan"]
preservation_plan = normalized["preservation_plan"]
exclusion_plan = normalized["exclusion_plan"]
```

Puis construire un index :

```python
reconstruction_plan_by_translation_unit_id
reconstruction_plan_by_reconstruction_unit_id
reconstruction_plan_by_source_unit_id
```

Et enrichir chaque `TranslatedTextUnit` avec :

```text
style_source_unit_id
render_contract complet
bbox_policy
overflow_policy
font_size_policy
line_break_policy
patch_policy
text_source
consume_source_unit_ids
```

Règle :

```text
Si reconstruction_units et reconstruction_plan divergent, finding obligatoire.
```

---

## P0 — Ajouter un `StyleResolver`

Créer :

```text
pagereconstruct/style_resolver.py
```

Résolution recommandée :

```python
def resolve_style(reconstruction_unit, reconstruction_plan_item, unit_index):
    # 1. style explicite dans reconstruction_unit
    # 2. style_snapshot dans reconstruction_plan
    # 3. style_source_unit_id
    # 4. units[source_id].visual.style
    # 5. enfants spans dominants
    # 6. parent block
    # 7. style système page/document
    # 8. fallback contrôlé
```

Contrat de sortie :

```python
{
    "font_family": "TimesNewRomanPSMT",
    "font_size_pt": 10.0,
    "color": "#000000",
    "fill_color": "#000000",
    "flags": {
        "bold": False,
        "italic": False,
        "serif": True,
        "monospace": False
    },
    "line_height_pt": 11.8,
    "baseline_ratio": 0.78,
    "alignment": "justify",
    "style_source": "source_unit_visual_style",
    "style_source_unit_id": "...",
    "confidence": 0.92
}
```

Règle stricte :

```text
Aucune unité translated_text ne doit arriver au backend avec style = {}.
```

Si le style est absent :

```text
status = review
finding = unresolved_style
```

Pas `ok`.

---

## P0 — Corriger `pagetranslate/projection.py`

Dans `_semantic_reconstruction_unit`, ajouter :

```text
style_source_unit_id
style
source_line_styles
source_line_metrics
```

Pseudo-correction :

```python
def _semantic_reconstruction_unit(item, translated_input=None):
    render_target = item.get("render_target") or {}
    style_source_unit_id = render_target.get("style_source_unit_id")
    style = resolve_dominant_style_from_source_ids(
        translated_input,
        item.get("source_unit_ids") or [],
        preferred_id=style_source_unit_id,
    )

    return {
        ...
        "style_source_unit_id": style_source_unit_id,
        "style": style,
        "style_source": "resolved_from_pageprint_source_units",
        ...
    }
```

Actuellement, la fonction ne reçoit pas assez de contexte. Il faut lui passer `translated_input` ou `unit_map`.

---

## P0 — Séparer `layout_bbox` et `patch_bbox`

Aujourd’hui, dans `render_backend.py` :

```python
bbox = t.get("coverage_bbox") or t.get("bbox")
```

Le même rectangle sert à :

```text
effacer
placer le texte
wrapper
calculer la taille
```

C’est faux.

Il faut séparer :

```text
patch_bbox       = zone où l’ancien texte doit être retiré
layout_bbox      = zone où le texte traduit doit être placé
anchor_bbox      = bbox source de référence
safe_bbox        = zone utilisable après exclusions/protections
```

Pour un paragraphe :

```text
patch_bbox  = union des lignes source
layout_bbox = bloc source ou flow region
```

Pour un titre :

```text
patch_bbox  = bbox titre source
layout_bbox = bbox titre source, éventuellement élargie
```

Pour une cellule :

```text
patch_bbox = cellule texte
layout_bbox = cellule verrouillée
```

---

## P0 — Ne plus calculer la taille depuis la hauteur de bbox

À remplacer :

```python
max_size = max(7, min(h * 0.95, 28))
```

par :

```python
font_size_px = resolved_style.font_size_pt * scale_y
```

Puis appliquer un shrink contrôlé :

```text
font_size_min = source_font_size * 0.86 pour body
font_size_min = source_font_size * 0.90 pour heading
font_size_min = source_font_size * 0.75 pour table cell
```

Si ça ne rentre pas :

```text
finding = overflow_unresolved
status = review
```

Pas de shrink violent silencieux.

---

## P0 — Remplacer le whiteout blanc fixe

Actuellement :

```python
draw.rectangle(..., fill=(255, 255, 255))
```

À remplacer par un `PatchPlanner`.

Créer :

```text
pagereconstruct/patch_planner.py
```

Méthodes :

```text
clean_background_patch
sampled_color_patch
local_inpaint_patch
no_patch_for_transparent
```

Règles :

```text
page blanche → sampled whiteout acceptable
fond coloré → couleur locale dominante
figure/table/image → inpainting ou patch interdit si zone protégée
```

Chaque patch doit être dans le plan :

```json
{
  "op_type": "patch_text_zone",
  "unit_id": "ru_0001",
  "bbox": [...],
  "method": "sampled_color_patch",
  "background_color": "#f3f0dd",
  "protected_overlap_ratio": 0.0
}
```

---

## P0 — Appliquer réellement `ProtectedRegionIndex`

Aujourd’hui l’index existe mais le backend ne l’applique pas.

Avant tout patch :

```python
if protected_index.overlaps(patch_bbox):
    shrink_patch_or_review()
```

Avant tout texte :

```python
if protected_index.overlaps(text_bbox):
    adjust_or_review()
```

Directive :

```text
Aucun patch ne doit effacer une formule, un diagramme, une image, un logo, une page reference ou une zone préservée.
```

---

## P0 — Corriger les rôles `title` → `body_paragraph`

Il faut un test spécifique :

```text
source unit role = title
translation/reconstruction role ne doit pas devenir body_paragraph
```

À ajouter dans :

```text
tests/pagetranslate/test_projection_keeps_roles.py
tests/pagereconstruct/test_style_resolver.py
```

Règle :

```text
Le rôle de reconstruction doit être le rôle le plus spécifique disponible.
Priorité :
1. reconstruction_plan.role
2. translation_plan.role
3. semantic_segment.role
4. source unit understanding.role
5. fallback anchored_label_review
```

Pas l’inverse.

---

## P1 — Corriger les faux positifs table dans `pageprint`

Le cas `Advances_p0103` est important.

La page a été classée comme :

```text
table_page
table_body_cell
```

alors que visuellement c’est :

```text
book_page
body paragraph
section heading
bibliography
```

Il faut renforcer la détection table.

Critères minimaux pour table :

```text
présence d’une grille réelle
ou alignement multi-colonnes stable
ou cellules avec bordures
ou colonnes répétées avec mêmes x positions
ou relation ligne/colonne vérifiable
```

Interdiction :

```text
ne jamais classer un paragraphe justifié comme table_body_cell uniquement parce que les lignes sont alignées.
```

Ajouter un test :

```text
test_book_page_with_justified_paragraph_is_not_table_page
```

---

## P1 — Migrer les savoirs utiles de `reconstructor.py.bak`

L’ancien `reconstructor.py.bak` contient des fonctions importantes :

```text
FontResolver
_normalized_style_for_item
_normalized_fontsize_for_item
_resolve_style_font
_merge_styles
_measure_text_width
alignment handling
baseline ratio
style audit
inline style segments
overflow sizing
```

Il faut les extraire proprement, pas recopier le monolithe.

Cibles :

```text
pagereconstruct/font_resolver_bridge.py
pagereconstruct/text_measure.py
pagereconstruct/style_resolver.py
pagereconstruct/placement_engine.py
pagereconstruct/quality.py
```

---

# 10. Nouveau pipeline interne recommandé

Le pipeline `pagereconstruct` doit devenir :

```text
translated_input_data
  ↓
InputAdapter
  ↓
ReconstructionPlanMerger
  ↓
StyleResolver
  ↓
BackgroundResolver
  ↓
ProtectedRegionIndex
  ↓
PatchPlanner
  ↓
PageRenderPlan
  ↓
RendererDispatcher
  ↓
PDFBackend / RasterDebugBackend
  ↓
ReconstructionValidator
  ↓
PAGE_RECONSTRUCT_RESULT
```

Le point important :

```text
PageRenderPlan doit être complet avant le rendu.
```

Le backend ne doit pas improviser.

---

# 11. TODO list priorisée

## Lot 1 — consommation correcte des contrats

```text
[ ] Modifier pagereconstruct/plan_compiler.py pour utiliser views.reconstruction_plan.
[ ] Créer un index reconstruction_plan_by_translation_unit_id.
[ ] Fusionner reconstruction_unit + reconstruction_plan_item.
[ ] Remonter style_source_unit_id dans TranslatedTextUnit.
[ ] Remonter render_contract complet.
[ ] Remonter bbox_policy, overflow_policy, line_break_policy.
[ ] Ajouter finding si reconstruction_unit n’a pas d’entrée reconstruction_plan correspondante.
[ ] Ajouter finding si reconstruction_plan a une unité non rendue.
```

## Lot 2 — style resolver

```text
[ ] Créer pagereconstruct/style_resolver.py.
[ ] Résoudre style depuis reconstruction_unit.style.
[ ] Sinon depuis reconstruction_plan.style_source_unit_id.
[ ] Sinon depuis units[].visual.style.
[ ] Sinon depuis descendants span.
[ ] Sinon depuis parent block.
[ ] Sinon depuis style_system.
[ ] Ajouter ResolvedTextStyle.
[ ] Ajouter test : aucune translated_text unit avec style vide.
[ ] Ajouter test : title conserve serif/bold/font_size.
[ ] Ajouter test : body conserve serif/font_size/line_height.
```

## Lot 3 — correction `pagetranslate/projection.py`

```text
[ ] Ajouter style_source_unit_id dans _semantic_reconstruction_unit.
[ ] Ajouter style snapshot pour semantic reconstruction units.
[ ] Ajouter source_line_styles.
[ ] Ajouter source_line_metrics.
[ ] Ajouter dominant_style_confidence.
[ ] Ajouter test : semantic_phrase récupère le style du span dominant.
[ ] Ajouter test : Preface reste title/heading.
```

## Lot 4 — patch planner

```text
[ ] Créer pagereconstruct/patch_planner.py.
[ ] Générer patch zones dans PageRenderPlan.
[ ] Séparer patch_bbox et layout_bbox.
[ ] Interdire patch sur protected region.
[ ] Remplacer whiteout blanc par sampled_color_patch.
[ ] Ajouter clean_background_patch si background_path existe.
[ ] Ajouter finding source_text_leak_risk si source background sans patch.
```

## Lot 5 — backend vectoriel

```text
[ ] Créer pagereconstruct/pdf_backend.py basé PyMuPDF.
[ ] Garder render_backend.py comme RasterDebugBackend.
[ ] Brancher font_resolver.py via FontResolverBridge.
[ ] Supporter fontfile, builtin fallback, Unicode fallback.
[ ] Supporter couleur.
[ ] Supporter bold/italic via flags.
[ ] Supporter alignement left/center/right/justify.
[ ] Supporter line_height_pt.
[ ] Supporter baseline_ratio.
[ ] Ajouter audit font_substitution.
```

## Lot 6 — renderers spécialisés

```text
[ ] Créer renderers/base.py.
[ ] Créer renderers/paragraph.py.
[ ] Créer renderers/heading.py.
[ ] Créer renderers/table.py.
[ ] Créer renderers/caption.py.
[ ] Créer renderers/anchored_label.py.
[ ] Créer renderers/preservation.py.
[ ] Ne pas utiliser paragraph renderer pour role inconnu.
[ ] role inconnu => anchored_label_review ou review.
```

## Lot 7 — validation typographique

```text
[ ] Créer pagereconstruct/quality.py.
[ ] Vérifier font_family_class_match : serif/sans/mono.
[ ] Vérifier font_size_ratio.
[ ] Vérifier bold/italic match.
[ ] Vérifier color_delta.
[ ] Vérifier line_count_delta.
[ ] Vérifier alignment_match.
[ ] Vérifier text_coverage_ratio.
[ ] Vérifier clipping.
[ ] Vérifier protected_overlap.
[ ] Vérifier patch_coverage.
[ ] Sortir status ok/review/ko.
```

## Lot 8 — correction pageprint/table

```text
[ ] Ajouter test anti-faux-positif table_page sur page livre.
[ ] Exiger grille ou structure tabulaire forte.
[ ] Empêcher body paragraph justifié de devenir table_body_cell.
[ ] Ajouter role confidence.
[ ] Si table confidence faible, rester body_paragraph.
```

---

# 12. Conclusion ferme

Le niveau actuel est bon pour une **preuve de chaîne complète** :

```text
PAGEPRINT → PAGETRANSLATE → PAGERECONSTRUCT plan → rendu raster de debug
```

Mais ce n’est pas encore une reconstruction WYSIWYG.

Le problème principal est clair :

```text
Le texte traduit arrive.
La géométrie arrive partiellement.
La préservation arrive partiellement.
Mais la typographie n’arrive pas au rendu.
```

La phrase décisive :

```text
pagereconstruct/ consomme pagetranslate pour le texte, mais ne consomme pas encore pageprint pour le style.
```

Donc la prochaine implémentation doit viser une chose avant tout :

```text
faire passer le style source complet depuis PAGEPRINT jusqu’au backend de reconstruction.
```

Sans cela, aucune amélioration de rendu ne sera réellement WYSIWYG.

