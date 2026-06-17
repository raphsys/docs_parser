Oui. Et je recommande une décision précise :

```text id="flcv7o"
La détection de zones doit devenir la première sous-unité officielle de PAGEPRINT.
```

Pas dans `PAGETRANSLATE`. Pas après `PAGEPRINT`. Elle doit être **avant la création finale de `INPUT_DATA`**, parce qu’elle influence tout :

```text id="m31c4m"
unités textuelles
régions protégées
zones images
zones formules
zones code
zones tableaux
politiques de traduction
politiques de reconstruction
vues translation_units / protected_visual_units
```

---

# 1. Décision d’architecture recommandée

Je ferais ceci :

```text id="9zgpxj"
PAGEPRINT
├── 0. PAGE_REGION_DETECT
├── 1. UNIT_FACTORY
├── 2. REGION_INDEX
├── 3. POLICY_COMPILER
├── 4. CONSTRAINT_COMPILER
├── 5. VIEWS_BUILDER
└── 6. INPUT_DATA_VALIDATOR
```

Donc la détection devient une **tête interne** de `PAGEPRINT`.

Elle peut avoir son propre nom :

```text id="s5jvw3"
PAGE_REGION_DETECT
```

ou plus court :

```text id="2mc0ox"
REGIONMAP
```

ou plus métier :

```text id="a4lkm3"
VISUAL_GUARD
```

Mon choix : **`PAGE_REGION_DETECT`** pour le code, parce que c’est clair et non ambigu.

---

# 2. Pourquoi dans `PAGEPRINT` et pas séparé totalement ?

Tu pourrais en faire une unité totalement séparée :

```text id="l5688v"
PAGEDETECT → PAGEPRINT → PAGETRANSLATE
```

Mais à ce stade, je conseille plutôt :

```text id="zoqlal"
PAGEPRINT inclut PAGE_REGION_DETECT comme première phase interne.
```

Raison : la détection de zones n’est pas une fin en soi. Elle sert à construire `INPUT_DATA`. Donc elle appartient naturellement à la tête `PAGEPRINT`.

Mais il faut la coder comme un module autonome, pour pouvoir plus tard l’extraire si nécessaire.

Doctrine :

```text id="gxoa3a"
PAGE_REGION_DETECT est interne à PAGEPRINT,
mais avec un contrat d’entrée/sortie indépendant.
```

---

# 3. Nouveau flux recommandé

Le pipeline doit devenir :

```text id="hknb3k"
page_image + pdf_page + page_structure brut
        ↓
PAGEPRINT.PAGE_REGION_DETECT
        ↓
page_structure enrichi avec special_regions / visual_regions / protected_regions
        ↓
PAGEPRINT.UNIT_FACTORY
        ↓
unités canoniques page/block/line/phrase/span/word/char
        ↓
PAGEPRINT.REGION_INDEX
        ↓
attachement unit → region
        ↓
PAGEPRINT.POLICY_COMPILER
        ↓
translatable=false pour zones protégées
        ↓
PAGEPRINT.VIEWS_BUILDER
        ↓
INPUT_DATA final
        ↓
PAGETRANSLATE
```

Donc la détection devient réellement :

```text id="n2r6yo"
la porte d’entrée de PAGEPRINT.
```

---

# 4. Ce qu’il faut mettre dans `pageprint/`

Je recommande cette structure :

```text id="eqq2zt"
pageprint/
├── __init__.py
├── builder.py
├── schema.py
├── unit_factory.py
├── region_index.py
├── policy_compiler.py
├── constraint_compiler.py
├── views_builder.py
│
├── detection/
│   ├── __init__.py
│   ├── schema.py
│   ├── builder.py
│   ├── special_region_detector.py
│   ├── yolo_backend.py
│   ├── pdf_glyph_detector.py
│   ├── heuristic_detector.py
│   ├── region_normalizer.py
│   ├── region_merger.py
│   ├── membership.py
│   └── debug.py
```

Le fichier actuel :

```text id="gvkx6z"
special_region_detector.py
```

doit être déplacé ou intégré dans :

```text id="2kkdho"
pageprint/detection/special_region_detector.py
```

ou découpé en plusieurs modules.

---

# 5. Contrat de sortie de `PAGE_REGION_DETECT`

Cette sous-unité doit produire une sortie claire :

```json id="o7hbp0"
{
  "schema_version": "pageprint.region_detect.v1",
  "regions": [],
  "special_regions": [],
  "protected_visual_regions": [],
  "debug": {
    "detectors": [],
    "warnings": []
  }
}
```

Pour les formules/codes, la sortie doit être typée comme ceci :

```json id="d3i387"
{
  "region_id": "special_region_001",
  "region_type": "protected_visual_region",
  "special_class": "formula",
  "object_type": "formula",
  "object_class": "equation",
  "bbox": [120, 250, 430, 285],
  "visual_bbox": [120, 250, 430, 285],
  "detection_source": "yolo_onnx",
  "confidence": 0.94,
  "translatable": false,
  "translation_strategy": "background_only",
  "render_policy": "background_only",
  "preserve_original_pixels": true,
  "protected_visual": true,
  "skip_translation": true,
  "skip_text_reconstruction": true
}
```

Point essentiel : **ne plus produire seulement `special_class=formula`**. Il faut aussi produire :

```text id="hphd6d"
region_type = protected_visual_region
object_type = formula
```

Sinon `region_index.py` peut mal classer.

---

# 6. Ce que `PAGE_REGION_DETECT` doit détecter

Il doit détecter au minimum :

```text id="hx5r1j"
formule
équation
code block
inline code
expression mathématique
expression chimique
notation spéciale
diagramme
image
table
graphique
signature
logo
cachet
fond décoratif
zone non textuelle
```

Mais les classes doivent être séparées en deux familles.

## Famille A — zones protégées non traduisibles

```text id="ue88nl"
formula
equation
code_block
inline_code
math_expression
chemical_formula
symbolic_expression
special_notation
logo
signature
stamp
barcode
qr_code
```

Politique :

```text id="zg54j1"
translatable = false
render_policy = background_only / preserve_original_region
skip_text_reconstruction = true
```

## Famille B — zones visuelles utiles mais pas forcément protégées

```text id="c5sbhn"
image
figure
diagram
chart
table
drawing
separator
background_shape
```

Politique selon type :

```text id="dkaz3k"
image/figure → préserver visuellement
table → structure à extraire si possible
diagram → labels éventuellement traduisibles, dessin préservé
chart → axes/légendes parfois traduisibles, graphique préservé
```

---

# 7. Correction à faire dans `PagePrintBuilder`

Dans `pageprint/builder.py`, au début du `build(...)`, ajouter une étape du type :

```python id="5j0lis"
from pageprint.detection.builder import PageRegionDetectBuilder

class PagePrintBuilder:
    def build(
        self,
        page_structure,
        source_context=None,
        extraction_result=None,
        assets=None,
        page_image=None,
        pdf_page=None,
        sx=1.0,
        sy=1.0,
        run_region_detection=True,
    ):
        if run_region_detection:
            page_structure, region_detection_info = PageRegionDetectBuilder().build(
                page_structure=page_structure,
                page_image=page_image,
                pdf_page=pdf_page,
                sx=sx,
                sy=sy,
            )
            page_structure.setdefault("debug", {})["page_region_detect"] = region_detection_info

        ...
```

Il faut permettre de désactiver :

```python id="0et7oj"
run_region_detection=False
```

pour les tests ou si l’orchestrateur a déjà fait la détection.

---

# 8. Attention : ne pas rendre `PAGEPRINT` dépendant de YOLO obligatoirement

Il faut éviter que `PAGEPRINT` plante si le modèle YOLO n’est pas disponible.

La logique doit être :

```text id="hiab8r"
1. Si modèle YOLO/ONNX configuré → utiliser.
2. Sinon → heuristiques PDF/glyphes.
3. Sinon → continuer sans special_regions.
4. Toujours écrire debug.detectors.
```

Exemple de debug :

```json id="euv74f"
{
  "page_region_detect": {
    "changed": true,
    "detectors": {
      "onnx_yolo": {
        "available": false,
        "reason": "no_model_configured"
      },
      "pdf_glyph_formula": {
        "available": true,
        "candidate_count": 3
      },
      "block_heuristic": {
        "available": true,
        "candidate_count": 1
      }
    },
    "special_region_count": 4
  }
}
```

C’est très important pour diagnostiquer.

---

# 9. Ce que `PAGETRANSLATE` doit faire ensuite

`PAGETRANSLATE` ne doit pas connaître YOLO.

Il doit seulement lire `INPUT_DATA`.

Donc sa règle reste :

```python id="lyf2eb"
if unit.policy.translatable is False:
    skip

if unit.policy.render_policy == "background_only":
    skip

if unit.policy.translation_strategy in {"background_only", "exact_preserve", "keep_original"}:
    skip

if unit.constraints.skip_translation:
    skip

if unit.covered_by_protected_region_id:
    skip
```

Donc :

```text id="o7kyhn"
YOLO appartient à PAGE_REGION_DETECT.
La décision canonique appartient à PAGEPRINT.
L’exclusion appartient à PAGETRANSLATE.
```

---

# 10. Ce qu’il faut corriger dans `pageprint/` au regard de cette décision

## P0 — Obligatoire

```text id="wcsqpe"
1. Créer pageprint/detection/.
2. Déplacer ou intégrer special_region_detector.py dans pageprint/detection/.
3. Appeler PAGE_REGION_DETECT au début de PagePrintBuilder.
4. Normaliser special_class → region_type/object_type/object_class.
5. Faire lire special_class par region_index.py en fallback.
6. Garantir que protected_visual_region donne translatable=false.
7. Garantir que views.translation_units exclut toute unité couverte.
8. Garantir que views.protected_visual_units contient les régions détectées.
```

## P1 — Fortement recommandé

```text id="3csvqg"
9. Ajouter debug.page_region_detect.
10. Ajouter detector_source/confidence sur chaque région.
11. Ajouter preserve_subregions.
12. Ajouter source_text_for_audit si disponible.
13. Ajouter tests end-to-end avec formule/code détecté.
14. Ajouter mode run_region_detection=False.
```

## P2 — Plus tard

```text id="d4yb6u"
15. Ajouter Mask R-CNN ou segmentation fine.
16. Ajouter inline protected anchors.
17. Ajouter table structure detector.
18. Ajouter diagram label classifier.
19. Ajouter modèle spécialisé code/formule.
```

---

# 11. Nom recommandé

Je propose ces noms :

## Pour le module interne

```text id="haf7vu"
PAGE_REGION_DETECT
```

## Pour le dossier

```text id="bf8tmn"
pageprint/detection/
```

## Pour la classe

```python id="n5ty1s"
PageRegionDetectBuilder
```

## Pour la sortie

```text id="k5lqug"
region_detect_result
```

## Pour la vue dans `INPUT_DATA`

```text id="41n7m5"
views.protected_visual_units
views.detected_regions
views.region_memberships
```

---

# 12. Doctrine finale à figer

La règle doit être écrite noir sur blanc :

```text id="riix3l"
PAGE_REGION_DETECT détecte les zones visuelles, spéciales et protégées.
PAGEPRINT transforme ces zones en régions canoniques et politiques.
PAGETRANSLATE ne traduit que les unités textuelles autorisées par PAGEPRINT.
RECONSTRUCTION préserve les régions protégées sans modification.
```

Donc oui : **la détection de zones doit être la première partie de `PAGEPRINT`**, mais codée comme une sous-unité autonome.

Architecture finale :

```text id="i5rvpp"
PAGEPRINT
└── PAGE_REGION_DETECT
    ↓
    REGION_INDEX
    ↓
    UNIT_FACTORY
    ↓
    POLICY_COMPILER
    ↓
    INPUT_DATA

PAGETRANSLATE
└── ne traduit que ce que PAGEPRINT autorise
```

C’est la meilleure solution pour garder le pipeline propre, robuste et évolutif.

