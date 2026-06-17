Oui. En re-regardant `rev_08`, je corrige et précise plusieurs points. Le risque principal n’est pas seulement “créer `pagereconstruct/`”. Le vrai risque est de **mal brancher trois vues déjà existantes** :

```text
views.reconstruction_plan      ← vient de PAGEPRINT
views.preservation_plan        ← vient de PAGEPRINT
views.exclusion_plan           ← vient de PAGEPRINT
views.reconstruction_units     ← vient de PAGETRANSLATE
```

`pagereconstruct/` devra **fusionner ces quatre vues**, pas seulement lire `reconstruction_units`.

C’est le point le plus important que je précise maintenant.

---

# 1. Diagnostic plus précis

Dans `rev_08`, la chaîne est déjà avancée :

```text
PAGEPRINT
  produit :
    translation_plan
    reconstruction_plan
    preservation_plan
    exclusion_plan

PAGETRANSLATE
  produit :
    translated_input_data.views.reconstruction_units
```

Mais attention :

```text
views.reconstruction_units = unités traduites seulement
```

Ce n’est pas encore le plan complet de rendu de page.

Donc `pagereconstruct/` doit construire son propre plan final :

```text
PageRenderPlan =
    translated_text_units
  + preserved_visual_units
  + excluded_artifacts
  + background_layers
  + protected_regions
  + overlays
  + patch_zones
```

Erreur à éviter absolument :

```text
pagereconstruct lit uniquement reconstruction_units
→ il rend le texte traduit
→ mais oublie formules, logos, images, zones exclues, overlays, fonds
→ résultat incomplet.
```

Il faut donc un **PageReconstructPlanCompiler**.

---

# 2. Architecture affinée

Je modifierais la proposition précédente comme suit :

```text
pagereconstruct/
├── __init__.py
├── schema.py
├── builder.py
├── input_adapter.py
├── plan_compiler.py
├── protected_region_index.py
├── layer_model.py
├── background_resolver.py
├── patch_planner.py
├── style_resolver.py
├── font_resolver_bridge.py
├── text_measure.py
├── geometry.py
├── placement_engine.py
├── candidate_engine.py
├── renderers/
│   ├── base.py
│   ├── text_box.py
│   ├── paragraph.py
│   ├── heading.py
│   ├── caption.py
│   ├── table.py
│   ├── code.py
│   ├── formula.py
│   ├── anchored_label.py
│   └── preservation.py
├── pdf_backend.py
├── validator.py
├── quality.py
├── debug_exporter.py
├── legacy_bridge.py
└── errors.py
```

La différence importante : j’ajoute explicitement :

```text
layer_model.py
patch_planner.py
candidate_engine.py
protected_region_index.py
errors.py
```

Ces modules sont nécessaires pour éviter une reconstruction approximative.

---

# 3. Le vrai modèle mental : reconstruction par couches

`pagereconstruct/` ne doit pas “dessiner du texte sur une page”. Il doit reconstruire une page par **strates**.

```text
Layer 0 — page vierge
Layer 1 — fond propre / clean background
Layer 2 — patchs locaux / inpainting / whiteout
Layer 3 — objets préservés sous le texte
Layer 4 — texte traduit
Layer 5 — objets préservés au-dessus du texte
Layer 6 — debug overlays optionnels
```

L’ordre de rendu doit être strict.

## Ordre recommandé

```text
1. Créer la page PDF vide.
2. Insérer le clean background si fiable.
3. Appliquer les patchs uniquement sur les zones textuelles traduites.
4. Réinsérer les éléments visuels de fond nécessaires.
5. Rendre les textes traduits.
6. Rendre les overlays immuables restants.
7. Produire debug/audit.
```

Erreur classique :

```text
insérer overlays avant les patchs
→ les patchs effacent les overlays.
```

Autre erreur classique :

```text
utiliser l’image source originale comme background
→ l’ancien texte reste visible sous le texte traduit.
```

Donc la règle doit être :

```text
background original autorisé seulement pour les pages sans texte remplacé.
background propre obligatoire pour les zones contenant du texte remplacé.
```

---

# 4. Nouveau contrat interne : `PageRenderPlan`

Il ne faut pas directement rendre les vues d’entrée. Il faut compiler un contrat intermédiaire.

```python
{
    "schema_version": "pagereconstruct.plan.v1",
    "page": {
        "page_index": 0,
        "width_pt": 531.0,
        "height_pt": 666.0,
        "rotation": 0,
        "coordinate_unit": "pt",
        "coordinate_origin": "top_left"
    },
    "layers": {
        "background": [],
        "patches": [],
        "preserved_underlays": [],
        "translated_text": [],
        "preserved_overlays": []
    },
    "protected_regions": [],
    "consumed_source_unit_ids": [],
    "excluded_source_unit_ids": [],
    "render_policy": {
        "fail_on_missing_reconstruction_units": True,
        "fail_on_unresolved_style": False,
        "allow_legacy_blocks_fallback": False
    },
    "quality_expectations": {
        "require_text_coverage": True,
        "require_no_protected_overlap": True,
        "require_no_source_text_leak": True
    }
}
```

Puis seulement après :

```text
PageRenderPlan → PDFBackend → PAGE_RECONSTRUCT_RESULT
```

---

# 5. Fusion correcte des plans

Le cœur de `plan_compiler.py` doit faire ceci :

```text
INPUT:
  translated_input_data

READ:
  A = views.reconstruction_units       # texte traduit réel
  B = views.reconstruction_plan        # intention complète de reconstruction
  C = views.preservation_plan          # éléments à préserver
  D = views.exclusion_plan             # éléments à exclure
  E = units[]                          # géométrie/style/rôles source
  F = visual_layers/assets             # fonds/images/overlays

OUTPUT:
  PageRenderPlan
```

## Algorithme de fusion

```python
def compile_page_render_plan(translated_input_data):
    unit_index = build_unit_index(translated_input_data["units"])

    translated_units = read_reconstruction_units(translated_input_data)
    reconstruction_plan = read_pageprint_reconstruction_plan(translated_input_data)
    preservation_plan = read_preservation_plan(translated_input_data)
    exclusion_plan = read_exclusion_plan(translated_input_data)

    consumed = collect_consumed_source_ids(translated_units)
    excluded = collect_excluded_source_ids(exclusion_plan)

    protected_index = build_protected_region_index(
        units=unit_index,
        preservation_plan=preservation_plan,
        exclusion_plan=exclusion_plan,
        visual_layers=translated_input_data.get("visual_layers"),
        assets=translated_input_data.get("assets"),
    )

    text_layers = compile_translated_text_layers(
        translated_units,
        unit_index,
        protected_index,
    )

    preservation_layers = compile_preservation_layers(
        preservation_plan,
        exclusion_plan,
        unit_index,
        consumed,
    )

    patch_layers = compile_patch_layers(
        translated_units,
        protected_index,
    )

    return PageRenderPlan(...)
```

Important :

```text
consumed_source_unit_ids ne veut pas dire supprimer visuellement tout.
```

Cela veut dire :

```text
ne pas rendre ce texte source comme texte séparé.
```

Mais si la source est intégrée dans un background image, il faut encore la nettoyer par patch.

---

# 6. Point très dangereux : `consumed` ≠ `erased`

Il faut distinguer :

```text
consumed_text_unit
  = unité source couverte par une traduction sémantique ; ne pas rendre l’enfant séparément.

erased_visual_area
  = zone où l’ancien texte doit être retiré du fond.

preserved_visual_area
  = zone à ne jamais effacer.

excluded_artifact
  = zone à ignorer/traduire non, mais peut rester visuellement.
```

Exemple :

```text
phrase source en anglais → consumed
bbox de cette phrase     → erased/patchée
logo MANNING             → excluded + preserved
formule E=mc²            → preserved, non erased
```

Erreur possible :

```text
consumed_source_unit_ids envoyé à patch_planner sans filtre
→ le patch efface des éléments non textuels.
```

Donc `patch_planner.py` doit patcher seulement :

```text
unit.level in {"block", "line", "phrase", "span"}
ET policy.render_contract.mode == translated_text
ET role non protégé
ET bbox fiable
```

---

# 7. Contrat `ReconstructableTextUnit` plus strict

Le `reconstruction_unit` actuel de `pagetranslate/projection.py` est bon, mais pas assez contraignant.

Il faut normaliser vers ceci dans `input_adapter.py` :

```python
{
    "id": "ru_0001",
    "kind": "translated_text",

    "source": {
        "translation_unit_id": "tp_0001",
        "source_unit_ids": ["line1", "line2"],
        "source_text": "...",
        "translated_text": "..."
    },

    "semantics": {
        "role": "body_paragraph",
        "object_type": "natural_text",
        "semantic_kind": "paragraph",
        "page_role": "body"
    },

    "geometry": {
        "bbox": [x0, y0, x1, y1],
        "bbox_reliable": True,
        "rotation": 0,
        "anchor": "source_bbox",
        "allowed_movement": "none | local | vertical_only | reflow_region"
    },

    "style": {
        "font": "...",
        "size": 11.0,
        "color": "#000000",
        "bold": False,
        "italic": False,
        "alignment": "left",
        "line_height": 13.2
    },

    "render_contract": {
        "renderer": "paragraph",
        "mode": "translated_text",
        "strategy": "semantic_reflow",
        "bbox_policy": "locked",
        "overflow_policy": "shrink_or_reflow",
        "min_font_size_ratio": 0.86,
        "min_font_size_abs": 7.0,
        "allow_line_count_change": True,
        "allow_hyphenation": False,
        "preserve_case_pattern": False
    },

    "qa": {
        "must_render": True,
        "must_not_overlap_protected_regions": True,
        "must_not_clip": True
    }
}
```

Surtout, ne pas laisser `pagereconstruct` travailler avec des champs vagues.

---

# 8. Règles de sélection du renderer

Créer dans `renderers/base.py` ou `plan_compiler.py` une table déterministe :

```python
RENDERER_BY_ROLE = {
    "body_paragraph": "paragraph",
    "list_item": "paragraph",
    "title": "heading",
    "section_heading": "heading",
    "chapter_heading": "heading",
    "figure_caption_text": "caption",
    "table_caption_text": "caption",
    "table_header_cell": "table",
    "table_body_cell": "table",
    "table_numeric_cell": "table",
    "toc_entry_title": "anchored_label",
    "index_entry_term": "anchored_label",
    "diagram_label": "anchored_label",
    "axis_label": "anchored_label",
    "legend_label": "anchored_label",
    "code": "code",
    "formula": "formula",
}
```

Fallback strict :

```python
def choose_renderer(unit):
    role = unit.semantics.role
    if role in RENDERER_BY_ROLE:
        return RENDERER_BY_ROLE[role]

    object_type = unit.semantics.object_type
    if object_type in {"code_block", "inline_code"}:
        return "code"
    if object_type in {"formula_block", "equation"}:
        return "formula"
    if object_type in {"table_cell"}:
        return "table"

    return "anchored_label_review"
```

Erreur à éviter :

```text
role inconnu → paragraph
```

Non. C’est dangereux.

La bonne règle :

```text
role inconnu → anchored_label_review ou fail_soft
```

---

# 9. Règles d’ajustement typographique

Il faut formaliser le shrink. Sinon le reconstructeur va produire du texte illisible.

## Pour les paragraphes

```text
font_size_min = max(7.0 pt, source_font_size * 0.86)
font_size_max = source_font_size * 1.05
line_height_min = font_size * 1.05
line_height_target = source_line_height ou font_size * 1.2
```

Autorisé :

```text
reflow
changement du nombre de lignes
shrink léger
augmentation verticale si zone sûre
```

Interdit :

```text
font < 7 pt sauf micro-label
texte qui sort du bloc
texte sur image/formule/table
```

## Pour titres

```text
font_size_min = source_font_size * 0.90
max_lines = source_lines + 1
bbox locked par défaut
```

Si impossible :

```text
status = review
ne pas compresser à 50 %
```

## Pour tables

```text
font_size_min = max(5.5 pt, source_font_size * 0.75)
bbox locked strict
pas de déplacement hors cellule
```

Mais attention :

```text
une table traduite peut devenir impossible si les cellules sont trop petites.
```

Dans ce cas :

```text
status = review_table_overflow
```

Pas de bricolage.

## Pour labels de diagramme

```text
font_size_min = max(4.5 pt, source_font_size * 0.70)
bbox locked
single line prioritaire
rotation préservée
```

---

# 10. Gestion des rotations

À ne pas oublier. L’ancien `reconstructor.py` contient déjà des logiques de rotation.

`pagereconstruct` doit supporter dès le début :

```text
rotation = 0
rotation = 90
rotation = 180
rotation = 270
```

Mais pour V1 :

```text
rotation libre non orthogonale → preserve_as_overlay ou review
```

Dans `geometry.py` :

```python
def normalize_rotation(value):
    if value in (0, 90, 180, 270):
        return value
    if abs(value) < 1:
        return 0
    return "unsupported"
```

Erreur possible :

```text
bbox top-left interprétée comme bbox bottom-left PyMuPDF
```

Heureusement le contrat PAGEPRINT est en `top_left`. Mais PyMuPDF travaille avec coordonnées PDF visuelles compatibles top-left pour `page.rect` dans beaucoup d’opérations pratiques. Il faut malgré tout centraliser toutes conversions dans `pdf_backend.py`, jamais dans les renderers.

---

# 11. Gestion des polices

Tu dois éviter que chaque renderer résolve les polices à sa manière.

Créer :

```text
font_resolver_bridge.py
```

Responsabilité :

```text
style canonique PAGEPRINT
  → fontname PyMuPDF
  → fontfile si disponible
  → fallback Unicode
  → metrics disponibles
```

Règle :

```text
si texte contient caractères non supportés par la police source
→ fallback Unicode compatible
→ garder taille/couleur/style au mieux
→ noter font_substitution dans audit
```

Erreurs fréquentes :

```text
1. police source absente dans l’environnement ;
2. caractères accentués français non supportés ;
3. symbole mathématique cassé ;
4. emoji ou glyph spécial ;
5. faux gras/faux italique non disponible.
```

Donc chaque `TextOp` doit inclure :

```python
{
    "font_source": "source_pdf | embedded | fallback",
    "font_substitution": False,
    "missing_glyphs": [],
}
```

---

# 12. Texte source restant visible : test obligatoire

C’est un piège majeur.

Même si le texte traduit est rendu correctement, l’ancien texte peut rester dessous si le background n’a pas été nettoyé.

Il faut un contrôle :

```text
source_text_leak_risk
```

Version simple V1 :

```text
si background utilisé = source_image_path
et translated_text_units non vide
→ warning high: source_background_contains_original_text_possible
```

Version meilleure :

```text
pour chaque zone traduite :
  vérifier si un patch/whiteout/inpaint couvre >= 95 % de sa bbox
sinon warning high
```

Test obligatoire :

```python
def test_translated_text_area_requires_patch_or_clean_background():
    ...
```

---

# 13. `background_resolver.py` : décision stricte

Pseudo-règle :

```python
def resolve_background(input_data, page_plan):
    assets = input_data.get("assets") or {}
    visual_layers = input_data.get("visual_layers") or {}

    clean = visual_layers.get("clean_background_path") or assets.get("background_clean_path")
    source = assets.get("source_image_path")

    if clean:
        return BackgroundDecision(
            mode="clean_background",
            path=clean,
            source_text_leak_risk="low",
        )

    if page_plan.has_translated_text:
        return BackgroundDecision(
            mode="source_with_required_patches",
            path=source,
            source_text_leak_risk="high_until_patched",
        )

    return BackgroundDecision(
        mode="source_background",
        path=source,
        source_text_leak_risk="none_expected",
    )
```

Règle :

```text
Si aucune image de fond n’est disponible :
  - page blanche
  - rendu texte uniquement
  - status = degraded
```

Ne pas faire semblant que le WYSIWYG est réussi.

---

# 14. `patch_planner.py` : nettoyage local

Chaque texte traduit doit générer une zone de nettoyage :

```python
{
    "op_type": "patch_text_zone",
    "unit_id": "ru_0001",
    "bbox": [x0, y0, x1, y1],
    "method": "inpaint | sampled_whiteout | transparent_none",
    "must_not_overlap": ["protected_region_id"],
    "padding": [1.0, 0.5, 1.0, 0.5],
}
```

Mais il faut refuser les patchs dangereux :

```text
si bbox intersecte une région protégée > 5 %
→ réduire patch à la partie sûre
→ sinon status = review_patch_conflict
```

Ne pas patcher :

```text
formules
codes préservés
images
logos
watermark
page number
publisher mark
```

---

# 15. `protected_region_index.py`

Créer un index spatial simple. Pas besoin de R-tree au début, une liste suffit.

```python
class ProtectedRegionIndex:
    def __init__(self, regions):
        self.regions = regions

    def intersections(self, bbox, min_ratio=0.0):
        ...

    def overlaps(self, bbox, min_ratio=0.01):
        ...
```

Chaque région doit contenir :

```python
{
    "id": "prot_0001",
    "source": "preservation_plan | exclusion_plan | unit_policy | visual_layer",
    "reason": "formula | image | publisher_mark | watermark | code | table_grid",
    "bbox": [...],
    "hard": True,
    "z_policy": "under_text | over_text | preserve_original",
}
```

---

# 16. Validation finale obligatoire

`validator.py` doit vérifier :

```text
1. Tous les translated_text_units must_render ont produit au moins un TextOp.
2. Aucun TextOp ne dépasse sa bbox sauf autorisation.
3. Aucun TextOp ne chevauche une protected_region hard.
4. Aucun patch ne chevauche une protected_region hard.
5. Aucun source_unit consommé n’est rendu deux fois.
6. Aucun role interdit n’a été traduit/rendu.
7. La taille de police finale reste dans les limites.
8. Les overlays nécessaires sont présents.
9. Le nombre de textes rendus correspond au nombre attendu.
10. Le résultat n’est pas image-only si du texte traduit était attendu.
```

Le statut final :

```text
ok
  tout est cohérent

review
  page rendue mais risques détectés

ko
  page non fiable ou texte attendu absent
```

---

# 17. Les erreurs inapercues possibles

Voici les vrais pièges que je vois maintenant.

## 17.1 Double rendu parent/enfant

Cas :

```text
semantic_phrase traduite
+ lignes enfants ayant translated_text agrégé
```

Risque :

```text
la phrase traduite est rendue
puis les lignes enfants aussi
```

Solution :

```text
si source_unit_id est consommé par une reconstruction_unit préférée
→ bloquer tout rendu séparé de ce source_unit_id et de ses descendants.
```

## 17.2 Style absent dans les semantic reconstruction units

Dans `pagetranslate`, les unités sémantiques peuvent contenir :

```text
style_source = dominant_source_span
```

mais pas toujours un style complet.

Solution :

```text
StyleResolver doit être obligatoire.
Il doit chercher :
1. reconstruction_unit.style
2. render_target.style_source_unit_id
3. premier source_unit_id
4. descendant span avec style réel
5. style par défaut document
```

## 17.3 Confusion entre `render_target` et `bbox`

Certaines unités ont :

```text
bbox
render_target.bbox
```

Règle :

```text
render_target.bbox prime toujours.
bbox est fallback.
```

## 17.4 `preservation_plan` absent mais politiques présentes

Si `preservation_plan` est absent ou incomplet, il faut reconstruire depuis :

```text
units[].policy.preservation_mode
units[].policy.render_policy
views.protected_visual_units
regions[]
```

Pas de fail immédiat.

## 17.5 `exclusion_plan` duplique `preservation_plan`

Un publisher mark peut apparaître dans les deux.

Règle :

```text
exclusion_plan signifie : ne pas traduire/ne pas reconstruire comme texte.
preservation_plan signifie : préserver visuellement.
```

Donc :

```text
exclusion + preservation = préserver comme image/overlay, ne pas rendre texte.
exclusion seul = ne rien faire sauf si déjà dans background.
preservation seul = préserver selon mode.
```

## 17.6 Ancien texte dans les images natives

Si un titre est dans une image, il peut ne pas être dans `reconstruction_units`.

Dans ce cas, `pagereconstruct` ne peut pas le traduire.

Il doit signaler :

```text
image_text_not_translated_possible
```

Mais ne pas inventer une traduction.

## 17.7 Tables complexes

Ne pas tenter trop tôt de reconstruire les grilles.

V1 :

```text
préserver la grille/table en background
remplacer seulement les textes de cellules traduisibles
respecter les bboxes de cellules
```

Ne pas redessiner la table entière sauf si `PAGEPRINT` fournit un contrat de table complet.

## 17.8 PDF natif avec transparence / clipping

Le rendu background peut perdre des transparences vectorielles.

Solution V1 :

```text
rendre page source en image de fond propre
puis texte au-dessus
```

C’est moins pur, mais plus WYSIWYG.

## 17.9 Fonts trop petites

Un moteur WYSIWYG qui réduit tout pour faire tenir le texte peut produire un document illisible.

Solution :

```text
si shrink au-delà du seuil → review, pas ok.
```

## 17.10 Reflow global trop tôt

Le reflow global de page est séduisant, mais dangereux.

Pour V1 :

```text
pas de déplacement global automatique.
pas de redistribution multi-blocs sauf cas explicitement safe.
```

---

# 18. Implémentation en 5 passes

## Passe 1 — Plan sans rendu PDF

Objectif : compiler le plan.

```text
[ ] pagereconstruct/schema.py
[ ] pagereconstruct/input_adapter.py
[ ] pagereconstruct/protected_region_index.py
[ ] pagereconstruct/plan_compiler.py
[ ] tests/pagereconstruct/test_plan_compiler.py
```

Tests essentiels :

```text
[ ] reconstruction_units obligatoire si texte traduit attendu
[ ] preservation_plan fusionné
[ ] exclusion_plan fusionné
[ ] consumed_source_unit_ids calculé
[ ] protected_regions construites
[ ] aucun parent/enfant dupliqué
```

## Passe 2 — Background et patchs

```text
[ ] background_resolver.py
[ ] patch_planner.py
[ ] layer_model.py
```

Tests :

```text
[ ] fond propre prioritaire
[ ] source background + texte traduit => patch obligatoire
[ ] patch ne chevauche pas formule/logo/image
[ ] absence de fond => status degraded
```

## Passe 3 — Rendu texte minimal

```text
[ ] pdf_backend.py
[ ] renderers/text_box.py
[ ] style_resolver.py
[ ] font_resolver_bridge.py
[ ] text_measure.py
```

Rendu minimal :

```text
paragraph
heading
anchored_label
```

Tests :

```text
[ ] un texte traduit produit un TextOp
[ ] style récupéré depuis style_source_unit_id
[ ] font fallback auditée
[ ] overflow détecté
```

## Passe 4 — Renderers spécialisés

```text
[ ] table.py
[ ] caption.py
[ ] code.py
[ ] formula.py
[ ] preservation.py
```

Tests :

```text
[ ] table_cell locked
[ ] caption label préservé
[ ] code non traduit
[ ] formula non traduite
[ ] publisher_mark préservé/exclu
```

## Passe 5 — QA et branchement pipeline

```text
[ ] validator.py
[ ] quality.py
[ ] debug_exporter.py
[ ] brancher dans pipelines/orchestrator.py ou nouveau pipeline complet
[ ] ocr_server.py appelle pagereconstruct en wrapper seulement
```

---

# 19. Skeleton d’implémentation recommandé

## `builder.py`

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
        adapter = PageReconstructInputAdapter()
        normalized = adapter.normalize(translated_input_data)

        plan = PageRenderPlanCompiler().compile(normalized)

        if not render_pdf:
            return {
                "schema_version": "pagereconstruct.output.v1",
                "status": "plan_only",
                "plan": plan.to_dict(),
            }

        backend = PDFBackend()
        render_result = backend.render(plan, output_path=output_path)

        validation = ReconstructionValidator().validate(plan, render_result)

        return {
            "schema_version": "pagereconstruct.output.v1",
            "source_schema_version": translated_input_data.get("schema_version"),
            "status": validation["status"],
            "output": render_result,
            "plan_summary": plan.summary(),
            "quality": validation["quality"],
            "findings": validation["findings"],
            "debug": plan.debug if debug else {},
        }
```

## `input_adapter.py`

```python
class PageReconstructInputAdapter:
    def normalize(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise TypeError("translated_input_data must be a dict")

        views = data.get("views") or {}
        reconstruction_units = views.get("reconstruction_units")

        if reconstruction_units is None:
            raise PageReconstructInputError(
                "translated_input_data.views.reconstruction_units is missing"
            )

        return {
            "page": data.get("page") or {},
            "document": data.get("document") or {},
            "assets": data.get("assets") or {},
            "visual_layers": data.get("visual_layers") or {},
            "units": data.get("units") or [],
            "regions": data.get("regions") or [],
            "translated_units": reconstruction_units or [],
            "reconstruction_plan": views.get("reconstruction_plan") or [],
            "preservation_plan": views.get("preservation_plan") or [],
            "exclusion_plan": views.get("exclusion_plan") or [],
            "quality": data.get("quality") or {},
            "translation_result": data.get("translation_result") or {},
        }
```

## `errors.py`

```python
class PageReconstructError(Exception):
    pass

class PageReconstructInputError(PageReconstructError):
    pass

class PageRenderPlanError(PageReconstructError):
    pass

class PageRenderBackendError(PageReconstructError):
    pass
```

---

# 20. Directives claires pour Claude CLI

Voici les règles que je donnerais maintenant à Claude :

```text
1. Ne pas déplacer reconstructor.py tel quel dans pagereconstruct/.
2. Créer un nouveau pagereconstruct/ propre.
3. Utiliser reconstructor.py uniquement comme réserve d’algorithmes à extraire.
4. Ne jamais rendre depuis page_data["blocks"] dans le nouveau module.
5. Ne jamais utiliser units[].content.translated_text comme source principale.
6. Source texte principale = views.reconstruction_units.
7. Source préservation = views.preservation_plan + views.exclusion_plan + units[].policy.
8. Compiler un PageRenderPlan avant tout rendu PDF.
9. Toute opération de rendu doit devenir une RenderOp explicite.
10. Toute opération de patch doit vérifier ProtectedRegionIndex.
11. Tout fallback doit être visible dans findings/audit.
12. Aucun fallback silencieux.
13. Role inconnu ne devient jamais paragraph.
14. Style manquant passe par StyleResolver.
15. Font manquante passe par FontResolverBridge.
16. Overflow non résolu => status review/ko, pas ok.
17. Source background avec texte traduit => patch obligatoire.
18. Page image-only alors que texte attendu => erreur.
19. Ajouter tests avant migration des renderers complexes.
20. Garder ocr_server.py comme wrapper, pas comme moteur.
```

---

# 21. Priorité absolue : les tests avant le rendu avancé

Je recommande de commencer par ces tests exacts :

```text
tests/pagereconstruct/test_input_adapter.py
tests/pagereconstruct/test_plan_compiler.py
tests/pagereconstruct/test_protected_region_index.py
tests/pagereconstruct/test_patch_planner.py
tests/pagereconstruct/test_style_resolver.py
tests/pagereconstruct/test_no_duplicate_render.py
tests/pagereconstruct/test_preservation_plan.py
tests/pagereconstruct/test_background_source_text_leak.py
```

Le premier lot doit passer sans produire de PDF. C’est volontaire.

Si le plan est faux, le PDF sera faux.

---

# 22. Conclusion actualisée

La décision devient plus précise :

```text
pagereconstruct/ ne doit pas être “le nouveau reconstructor.py”.
pagereconstruct/ doit être un compilateur de page traduite vers opérations de rendu contrôlées.
```

La meilleure architecture est :

```text
translated_input_data
  ↓
InputAdapter
  ↓
PageRenderPlanCompiler
  ↓
ProtectedRegionIndex + PatchPlanner + StyleResolver
  ↓
RendererDispatcher
  ↓
PDFBackend
  ↓
ReconstructionValidator
  ↓
PAGE_RECONSTRUCT_RESULT
```

Et la règle fondatrice :

```text
On ne reconstruit pas une page traduite en dessinant du texte.
On reconstruit une page traduite en respectant un plan de couches, de protections, de patchs, de styles et de contraintes.
```

C’est cette discipline qui évitera de retomber dans le monolithe ancien.

