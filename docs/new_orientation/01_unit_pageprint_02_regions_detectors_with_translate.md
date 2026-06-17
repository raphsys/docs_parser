Oui, j’ai vérifié. Réponse directe : **non, la détection YOLO n’est pas réellement faite dans `pageprint/` lui-même.**
Ce qui existe actuellement est plutôt ceci :

```text
special_region_detector.py peut détecter des zones spéciales
↓
mais pageprint/ ne l’appelle pas directement
↓
pageprint/ sait seulement consommer page_structure["special_regions"] si elles existent déjà
↓
pagetranslate/ respecte ensuite les politiques produites par pageprint
```

Donc la chaîne n’est pas encore complète.

---

# 1. `special_region_detector.py` fait-il vraiment du YOLO ?

**Partiellement, mais pas explicitement comme “YOLO intégré”.**

Le fichier `special_region_detector.py` contient bien une détection IA par ONNX Runtime. Il lit le modèle depuis :

```python
DOCS_PARSER_SPECIAL_REGION_MODEL
```

Puis il charge :

```python
ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
```

Il accepte des sorties de type détection avec coordonnées, score et classe, et le commentaire dit qu’il accepte les formats `xyxy` et `cxcywh` de type YOLO. Il filtre les classes contenant `formula`, `equation`, `math`, `code`, `technical`. 

Donc :

```text
Oui : il peut exploiter un modèle ONNX de type YOLO.
Non : il n’y a pas de dépendance YOLO explicite.
Non : il n’y a pas de R-CNN / Mask R-CNN réel dans ce fichier.
```

C’est un détecteur ONNX générique, compatible avec des sorties YOLO-like.

Le même fichier a aussi une grosse partie **heuristique CPU/PDF glyphs** : il analyse les glyphes PDF, polices mathématiques, symboles, équations, blocs code, etc. Le retour de `detect_special_regions(...)` indique même :

```python
"detector": "cpu_pdf_glyph_heuristic_v3"
```

avec un champ `ai` séparé pour les résultats ONNX.  

Donc le nom exact serait plutôt :

```text
special_region_detector = détecteur hybride :
- heuristique PDF/glyphes
- signaux blocs formule/code
- layout_ai formula_regions
- ONNX optionnel, compatible YOLO-like
```

Pas encore :

```text
YOLO obligatoire intégré au pipeline PAGEPRINT
```

---

# 2. Est-ce que `special_region_detector.py` est pris en compte par `pageprint/` ?

**Non, pas directement.**

Dans `page.zip`, `pageprint/` ne contient pas `special_region_detector.py`, et je n’ai trouvé aucun import du type :

```python
from special_region_detector import detect_special_regions
```

ou :

```python
detect_special_regions(...)
```

dans `pageprint/`.

Ce que `pageprint/` fait en revanche est important : il **consomme** les régions spéciales si elles existent déjà dans `page_structure`.

Dans `pageprint/region_index.py`, la collecte lit :

```python
page_structure["special_regions"]
layout["special_regions"]
```

et les ajoute avec la source `"special_region_detector"`. Le module est explicitement conçu pour fusionner `regions`, `special_regions`, `non_text_zones`, `images`, `drawings`, formules, code regions, etc. Il dit aussi que la région devient plus forte que le simple texte. 

Donc :

```text
pageprint/ ne lance pas le détecteur.
pageprint/ sait intégrer le résultat du détecteur s’il est déjà injecté dans page_structure["special_regions"].
```

C’est une différence capitale.

---

# 3. Est-ce que `pageprint/` transforme bien ces régions en zones protégées ?

**Oui, si `special_regions` arrive déjà dans `page_structure`.**

Dans `pageprint/region_index.py`, les types suivants sont normalisés en :

```text
protected_visual_region
```

notamment :

```text
formula
formula_region
equation
math_expression
chemical_formula
symbolic_expression
code
code_region
code_block
inline_code
algorithm_block
special_notation
table_formula_cell
diagram_label_non_linguistic
protected_visual
```

Le même module donne aux `protected_visual_region` une politique claire :

```text
translatable = False
translation_strategy = background_only
render_policy = background_only
preserve_original_pixels = True
protected_visual = True
skip_translation = True
skip_text_reconstruction = True
must_exclude_from_translation_flow = True
```

C’est exactement la logique attendue.

Ensuite, `attach_region_memberships(...)` attache ces régions aux unités `block`, `line`, `phrase`, `span`, `word`, `char` selon le recouvrement spatial. Pour les `protected_visual_region`, les seuils sont plus souples : `0.35` pour `block/line`, `0.55` pour les niveaux plus fins.

Donc si une formule détectée recouvre une ligne ou une phrase, `pageprint/` peut bien imposer la politique protégée.

---

# 4. Est-ce que `pagetranslate/` en tient compte ?

**Oui, si `pageprint/` a bien marqué les unités.**

Dans `pagetranslate/selector.py`, les exclusions sont solides :

```python
EXCLUDED_CLASSES = {
    "publisher_mark",
    "author_name",
    "code",
    "formula",
    "url",
    "doi",
    "acronym",
    "reference",
    "references",
    "bibliography",
    "citation",
    "reference_link",
    "page_number",
}
```

Et il exclut aussi les stratégies :

```python
EXCLUDED_STRATEGIES = {"exact_preserve", "keep_original", "background_only"}
```

La fonction `_is_excluded_unit(...)` rejette une unité si :

```text
policy.translatable != True
translation_strategy in exact_preserve / keep_original / background_only
unit_type ou object_type indique formula/code/protected
```

Le `coalescer.py` protège aussi les unités dont le type est :

```text
protected_visual
formula
equation
code
code_visible
symbolic_expression
chemical_formula
```

Donc `pagetranslate/` ne devrait pas traduire ces zones **si `pageprint/` les a correctement marquées**.

---

# 5. Les autres fichiers sont-ils pris en compte ?

## `structure_extractor.py`

Il est **en amont**, pas dans `pageprint/`.

Il sert à construire la structure de page : blocs, lignes, phrases, styles OCR, layout v2, classification de page, colonnes, TOC, etc.

Mais dans le fichier fourni, je n’ai pas vu d’appel à :

```python
detect_special_regions(...)
```

Donc `structure_extractor.py` ne semble pas lancer le détecteur spécial.

Conclusion :

```text
structure_extractor.py alimente page_structure.
Mais il ne semble pas intégrer special_region_detector.py lui-même.
```

## `style_profiler.py`

Il est aussi **en amont**.

`pageprint/` consomme éventuellement :

```python
page_structure["visual_style_profile"]
page_structure["style_profile"]
```

Mais il ne lance pas lui-même `build_page_style_profile(...)`.

Donc :

```text
style_profiler.py est utile si ocr_server/pipeline l’exécute avant PAGEPRINT.
PAGEPRINT ne l’exécute pas directement.
```

## `style_tone_classifier.py`

Il n’est pas appelé directement par `pageprint/` ni par `pagetranslate/`.

Mais si un module amont remplit :

```python
page_structure["translation_style"]
page_structure["translation_tone"]
```

alors `pageprint/` les met dans `translation_context`, et `pagetranslate/` les utilise.

Donc :

```text
pris en compte indirectement seulement si ses résultats sont injectés avant.
```

## `terminology_manager.py`

Même logique.

`pagetranslate/` ne semble pas appeler `TerminologyManager` directement.

Mais `pageprint/` transmet :

```python
source_context["terminology"]
```

vers :

```python
input_data["translation_context"]["terminology"]
```

Puis `pagetranslate/` passe cette terminologie au `TranslatorBridge`.

Donc :

```text
TerminologyManager n’est pas appelé par PAGETRANSLATE.
Mais ses résultats peuvent être transmis si source_context["terminology"] est déjà préparé.
```

---

# 6. Chaîne réelle actuelle

La chaîne actuelle ressemble à ceci :

```text
structure_extractor.py / ocr_server / pipeline amont
    ↓
page_structure
    ↓
PagePrintBuilder
    ↓
INPUT_DATA
    ↓
PageTranslateBuilder
```

Mais le détecteur spécial n’est pris en compte que si, avant `PagePrintBuilder`, quelqu’un fait :

```python
from special_region_detector import detect_special_regions

page_structure, special_info = detect_special_regions(
    page_structure,
    page_image=page_image,
    pdf_page=pdf_page,
    sx=sx,
    sy=sy,
)
```

Puis transmet ce `page_structure` enrichi à :

```python
PagePrintBuilder().build(page_structure=page_structure, ...)
```

S’il n’y a pas cet appel, alors :

```text
pas de détection YOLO effective ;
pas de special_regions issues du détecteur ;
pageprint/ ne peut intégrer que les régions déjà présentes ;
pagetranslate/ ne verra rien de spécial sauf les heuristiques texte/policy.
```

---

# 7. Ce qu’il faut corriger

Il faut intégrer explicitement `special_region_detector.py` **avant `pageprint/`**, dans l’orchestrateur principal.

Pas dans `pagetranslate/`.

Pas forcément dans `pageprint/` lui-même non plus, si tu veux garder `PAGEPRINT` comme assembleur canonique.

La bonne intégration :

```python
# pipeline/orchestrator.py ou ocr_server.py, avant PagePrintBuilder

try:
    from special_region_detector import detect_special_regions

    page_structure, special_region_info = detect_special_regions(
        page_structure,
        page_image=pil_img,
        pdf_page=pdf_page,
        sx=sx,
        sy=sy,
    )

    page_structure.setdefault("debug", {})["special_region_detector"] = special_region_info

except Exception as exc:
    page_structure.setdefault("debug", {})["special_region_detector"] = {
        "changed": False,
        "error": str(exc),
    }
```

Ensuite seulement :

```python
input_data = PagePrintBuilder().build(
    page_structure=page_structure,
    source_context=source_context,
    extraction_result=extraction_result,
    assets=assets,
)
```

---

# 8. Point important : il faut normaliser la sortie de `special_region_detector.py`

Aujourd’hui `special_region_detector.py` produit :

```python
{
    "id": "special_region_0",
    "special_class": "formula",
    "bbox": [...],
    "render_policy": "preserve_source_region",
    "translation_policy": "preserve_visual_region",
    ...
}
```

`pageprint/region_index.py` comprend les types comme `formula`, `code`, `protected_visual_region`, etc.

Mais dans les `special_regions`, il manque parfois un champ explicite :

```python
"region_type": "formula"
```

ou :

```python
"region_type": "protected_visual_region"
```

Actuellement, `region_index.py` lit :

```python
raw.get("region_type") or raw.get("type") or raw.get("kind")
```

Il ne lit pas directement `special_class`.

Donc si `special_region_detector.py` donne seulement :

```python
"special_class": "formula"
```

mais pas :

```python
"region_type": "formula"
```

alors `pageprint/` risque de classer la région comme :

```text
body_region
```

car `raw_type` est absent.

C’est un vrai bug d’intégration.

## Correction minimale

Dans `special_region_detector.py`, au moment de créer `special_regions`, ajouter :

```python
"region_type": special_class,
"object_type": special_class,
"object_class": special_class,
"protected_visual": True,
"preserve_original_pixels": True,
"skip_translation": True,
"skip_text_reconstruction": True,
```

Exemple :

```python
special_regions.append({
    "id": f"special_region_{region_index}",
    "region_type": special_class,          # IMPORTANT
    "special_class": special_class,
    "object_type": special_class,
    "object_class": special_class,
    "visual_bbox": _bbox_from_rect(visual_rect),
    "bbox": _bbox_from_rect(visual_rect),
    "preserve_subregions": preserve_subregions,
    "formula_block_ids": sorted(set(formula_block_ids)),
    "code_block_ids": sorted(set(code_block_ids)),
    "translatable_block_ids": [],
    "text_subregions": [],
    "render_policy": "preserve_source_region",
    "translation_policy": "preserve_visual_region",
    "protected_visual": True,
    "preserve_original_pixels": True,
    "skip_translation": True,
    "skip_text_reconstruction": True,
    "detection_source": "+".join(sorted(s for s in sources if s)) or "cpu_heuristic",
    "confidence": round(confidence, 3),
})
```

Ou corriger côté `pageprint/region_index.py` :

```python
region_type = _normalize_region_type(
    raw.get("region_type")
    or raw.get("type")
    or raw.get("kind")
    or raw.get("special_class")   # IMPORTANT
)
```

Je recommande de faire **les deux**, pour être robuste.

---

# 9. Conclusion

État réel :

```text
special_region_detector.py existe.
Il peut utiliser un modèle ONNX compatible YOLO-like si DOCS_PARSER_SPECIAL_REGION_MODEL est configuré.
Il fait aussi une détection CPU/PDF glyphs.
Mais pageprint/ ne l’appelle pas.
pageprint/ consomme seulement special_regions si elles existent déjà.
pagetranslate/ respecte correctement les unités protégées si pageprint les a produites.
```

Donc la réponse stricte est :

```text
Non, la détection YOLO n’est pas actuellement faite par pageprint/.
Oui, pageprint/ est prêt à consommer ces régions.
Oui, pagetranslate/ est prêt à les exclure.
Mais il manque l’appel orchestrateur + une normalisation region_type/special_class.
```

La correction prioritaire est donc :

```text
1. Appeler detect_special_regions(...) avant PagePrintBuilder.
2. Ajouter region_type = special_class dans les special_regions.
3. Faire lire special_class par pageprint.region_index en fallback.
4. Vérifier que views.translation_units n’inclut aucune unité couverte par protected_visual_region.
5. Vérifier que views.protected_visual_units contient bien les formules/codes détectés.
```

Après ça, la chaîne sera cohérente :

```text
YOLO/ONNX ou heuristique spéciale
→ special_regions
→ PAGEPRINT protected_visual_region
→ PAGETRANSLATE skip
→ RECONSTRUCTION preserve_original_pixels
```

