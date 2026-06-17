## Verdict

**Ça va s’exécuter sur des cas simples, mais ça ne marchera pas correctement sur des documents réels WYSIWYG complexes.**

La nouvelle version est **nettement meilleure que la première base** : le découpage en modules est bon, la compilation passe, la logique générale est lisible, et on voit clairement les briques attendues :

```text id="xd284d"
selector.py
sentence_boundary.py
protection.py
context_builder.py
translator_bridge.py
quality.py
projection.py
schema.py
builder.py
```

Mais il reste des défauts bloquants. Les plus graves sont :

```text id="c88xnx"
1. risque de traduire deux fois le même texte ;
2. sélection simultanée semantic_phrase + line/phrase du même bloc ;
3. sélection possible d’unités semantic_system pourtant non traduisibles ;
4. projection dangereuse des semantic_phrases sur seulement le premier fragment source ;
5. protections trop agressives sur les mots composés anglais ;
6. unchanged translation non signalée comme problème ;
7. contexte riche construit mais encore trop peu transmis au traducteur.
```

Donc : **bonne V1-alpha avancée**, mais **pas encore V1-beta stable**.

---

# 1. Vérifications techniques

J’ai extrait l’archive et vérifié la compilation :

```text id="yjr9z6"
python -m py_compile pagetranslate/*.py
→ OK
```

Aucun `TODO`, `pass`, `NotImplemented` visible dans les fichiers Python.

Mais l’archive contient encore :

```text id="qzl3q6"
__pycache__/
*.pyc
```

À enlever du ZIP et du dépôt.

Autre point : **aucun test n’est inclus dans l’archive**. Donc on ne peut pas valider automatiquement le comportement annoncé.

---

# 2. Ce qui est bon

## 2.1 Le découpage logiciel est bon

La structure est maintenant cohérente :

```text id="h8ttv6"
builder.py             orchestration
selector.py            choix des unités
sentence_boundary.py   début/fin de phrase
protection.py          tokens intouchables
context_builder.py     contexte documentaire
translator_bridge.py   appel DocumentTranslator
quality.py             contrôle qualité
projection.py          réinjection
schema.py              contrat runtime
```

C’est exactement la bonne direction pour sortir du monolithe `ocr_server.py`.

## 2.2 Le pipeline métier est clair

`builder.py` suit une bonne séquence :

```text id="dqf1ih"
INPUT_DATA
→ build_translation_profile()
→ select_translation_units()
→ annotate_sentence_boundaries()
→ attach_unit_context()
→ protect_text()
→ translate()
→ restore_text()
→ unit_quality()
→ project_translations()
```

Conceptuellement, c’est propre.

## 2.3 Le fallback par bloc est commencé

La première version faisait un fallback global par page. Ici, `selector.py` tente bien une sélection par bloc.

C’est une amélioration importante.

## 2.4 La détection des abréviations est meilleure qu’avant

L’ancien bug du type :

```text id="olji7w"
"This is a test." considéré comme abréviation
```

est corrigé. Dans cette version :

```text id="obqar3"
"This is a test." → sentence_end = True
"U.S."            → sentence_end = False
```

C’est mieux.

---

# 3. Défaut bloquant n°1 : duplication des unités traduites

Le problème majeur est dans `selector.py`.

La logique actuelle fait :

```python id="4gdt2d"
semantic_units = _select_semantic_system_units(...)
selected.extend(semantic_units)

semantic_source_ids = {...}

selected.extend(
    _select_pageprint_units_by_block(..., selected_blocks, semantic_source_ids)
)
```

Mais `selected_blocks` ne contient que les blocs des semantic units **sans `source_unit_ids`** :

```python id="g7lh38"
selected_blocks = {
    item.get("block_id")
    for item in semantic_units
    if item.get("block_id") and not item.get("source_unit_ids")
}
```

C’est faux.

Si une `semantic_phrase` couvre les phrases `p1` et `p2`, alors `source_unit_ids = ["p1", "p2"]`. Donc le bloc n’est pas marqué comme déjà couvert. Ensuite le fallback peut sélectionner la `line` entière du même bloc.

Résultat réel obtenu en smoke test :

```text id="kpo65n"
tu_0001  l1   line             This is a long sentence continued here.
tu_0002  sp1  semantic_phrase  This is a long sentence continued here.
```

Donc **le même texte est sélectionné deux fois**.

C’est bloquant, parce que la traduction va produire :

```text id="t64wxx"
une traduction au niveau line
+
une traduction au niveau semantic_phrase
```

Puis la projection peut créer du texte doublé ou contradictoire selon le niveau lu par le reconstructeur.

### Correction nécessaire

Dans une V1 simple, dès qu’une `semantic_phrase` existe pour un bloc, on ne doit plus sélectionner `phrase/line/block` dans ce même bloc.

Remplacer la logique par :

```python id="bqruw7"
selected_blocks = {
    item.get("block_id")
    for item in semantic_units
    if item.get("block_id")
}
```

Puis dans `_select_pageprint_units_by_block()` :

```python id="plcx0r"
if block_key in selected_blocks:
    continue
```

C’est moins fin, mais beaucoup plus sûr.

---

# 4. Défaut bloquant n°2 : `semantic_phrase` et `semantic_group` peuvent être sélectionnés ensemble

Dans `_select_semantic_system_units()`, le code parcourt :

```python id="0mc38c"
("semantic_phrase", "semantic_phrases"),
("semantic_group", "semantic_groups")
```

et ajoute tout ce qui passe les filtres.

Mais une `semantic_group` peut recouvrir une ou plusieurs `semantic_phrase`.

Donc tu peux obtenir :

```text id="d1d95k"
semantic_phrase : "The model performs well."
semantic_group  : "The model performs well."
```

ou :

```text id="7zcqzq"
semantic_phrase 1 : partie A
semantic_phrase 2 : partie B
semantic_group    : partie A + partie B
```

Cela traduit encore deux fois la même matière textuelle.

### Correction nécessaire

Respecter réellement la priorité :

```text id="e1tmjd"
semantic_phrase > semantic_group > phrase > line > block
```

Donc, par bloc :

```python id="3jlm5m"
si semantic_phrases existent pour ce bloc :
    sélectionner semantic_phrases uniquement
sinon si semantic_groups existent :
    sélectionner semantic_groups
sinon fallback phrase/line/block
```

---

# 5. Défaut bloquant n°3 : unités non traduisibles du `semantic_system` quand même sélectionnées

Test réel :

```python id="ktxqj2"
{
  "unit_id": "sp",
  "text": "Do not translate",
  "translatable": False,
  "translation_strategy": "background_only"
}
```

Résultat actuel :

```text id="truerx"
l’unité est quand même sélectionnée
strategy = background_only
translatable = True
```

C’est mauvais.

La cause est dans `_select_semantic_system_units()` : le code vérifie le texte, les source units, mais **ne vérifie pas correctement la politique portée par l’entrée `semantic_system` elle-même**.

Puis `_make_item()` force :

```python id="f04dzw"
"translatable": True
```

Même si l’entrée semantic disait le contraire.

### Correction nécessaire

Dans `_select_semantic_system_units()` :

```python id="bhlvk9"
if entry.get("translatable") is False:
    continue

if str(entry.get("translation_strategy") or "").lower() in EXCLUDED_STRATEGIES:
    continue

if str(entry.get("render_policy") or "").lower() == "background_only":
    continue
```

Et dans `_make_item()` :

```python id="2bf02r"
"translatable": bool(entry.get("translatable", policy.get("translatable", True)))
```

Pas `True` en dur.

---

# 6. Défaut bloquant n°4 : projection dangereuse des semantic phrases

Quand une unité traduite n’existe pas directement dans `input_data["units"]`, le code appelle :

```python id="cq5ha2"
_project_to_source_units(...)
```

Et là :

```python id="83go4b"
if idx == 0:
    unit["content"]["translated_text"] = item["translated_text"]
```

Donc si une `semantic_phrase` couvre :

```text id="ohd5q4"
p1 = "This is a long sentence"
p2 = "continued here."
```

la traduction complète est injectée seulement dans `p1`.

Résultat observé :

```text id="78delq"
p1 reçoit toute la traduction
p2 ne reçoit pas de translated_text
```

C’est dangereux pour la reconstruction.

Dans un WYSIWYG, si le reconstructeur lit les phrases visuelles, il risque :

```text id="4wgyd8"
- d’afficher toute la traduction dans la bbox de p1 ;
- de laisser p2 en texte original ;
- ou de créer un chevauchement ;
- ou de perdre une partie du flux.
```

### Correction nécessaire

Il faut une vraie stratégie de projection :

```text id="j9y14r"
1. semantic_phrase traduite = unité de rendu principale si elle a une bbox globale fiable ;
2. les source_unit_ids doivent être marqués skip_render / consumed_by_translation ;
3. ne pas injecter toute la traduction dans le premier fragment ;
4. créer une reconstruction_unit de niveau semantic_phrase ;
5. le reconstructeur doit préférer semantic_phrase à phrase/line quand elle existe.
```

Exemple de projection attendue :

```json id="y7qxef"
{
  "unit_id": "sp1",
  "level": "semantic_phrase",
  "translated_text": "...",
  "bbox": [x0, y0, x1, y1],
  "source_unit_ids": ["p1", "p2"],
  "render_as": "semantic_phrase",
  "source_units_consumed": true
}
```

Et sur `p1`, `p2` :

```json id="7czzi0"
{
  "translation": {
    "consumed_by_translation_unit_id": "tu_0001",
    "skip_individual_render": true
  }
}
```

---

# 7. Défaut important : protection trop agressive des mots composés

`protection.py` protège trop de choses comme `formula`.

Test réel :

```text id="7y4dvq"
state-of-the-art method
→ __PT_0001__ method
→ token protégé = state-of-the-art, kind=formula

well-known algorithm
→ __PT_0001__ algorithm
→ token protégé = well-known, kind=formula

pre-trained model
→ __PT_0001__ model
→ token protégé = pre-trained, kind=formula
```

C’est mauvais.

Ces expressions doivent être traduites :

```text id="lljqlp"
state-of-the-art → de pointe / à l’état de l’art
well-known       → bien connu
pre-trained      → préentraîné
```

Actuellement, le système les fige. Cela va produire des traductions hybrides et artificielles.

### Cause

La regex `formula` est trop permissive :

```python id="1kzic1"
\b[A-Za-z]\w*\s*(?:=|≈|<=|>=|<|>|\+|-|\*|/|\^)\s*[-\w.()]+
```

Le simple tiret `-` fait considérer beaucoup de mots composés comme des formules.

### Correction nécessaire

Ne pas traiter `word-word` comme formule. Réserver `formula` à :

```text id="r8jv4e"
x = y
a + b
dx/dy
P(x)
E = mc²
m/s²
α + β
```

Une regex plus prudente :

```python id="m7mqzg"
FORMULA_PATTERN = re.compile(
    r"""
    (?:
        \b[A-Za-z]\w*\s*(?:=|≈|<=|>=|<|>|\+|\*|/|\^)\s*[-\w.()]+
        |
        [α-ωΑ-Ω∑∫√∞≈≠≤≥±×÷]
        |
        \b\d+(?:[.,]\d+)?\s*(?:/|\*)\s*[A-Za-z]+\b
    )
    """,
    re.VERBOSE,
)
```

Et surtout : **retirer `-` seul des opérateurs déclenchant une formule**, sauf contexte mathématique évident.

---

# 8. Défaut important : les protections déclarées par `PAGEPRINT` ne sont pas utilisées

Dans `selector.py`, l’item contient :

```python id="m2x8gq"
"protected": list(policy.get("translation_protection") or entry.get("protected") or [])
```

Mais dans `builder.py`, la protection réelle fait seulement :

```python id="5jy86j"
protected_text, protections = protect_text(source_text)
```

Donc les protections déjà calculées par `PAGEPRINT` ou par `translation_context.protected_tokens` ne sont pas injectées.

`TranslationProfile` contient bien :

```python id="5v7zo7"
protected_tokens=list(context.get("protected_tokens") or [])
```

mais `protection.py` ne les reçoit pas.

### Correction nécessaire

Modifier :

```python id="96tq8k"
protect_text(source_text)
```

en :

```python id="ov66ao"
protect_text(
    source_text,
    explicit_tokens=[
        *translation_profile.get("protected_tokens", []),
        *item.get("protected", []),
    ],
)
```

Puis dans `protection.py`, protéger d’abord les tokens explicites avec `re.escape()` avant les regex générales.

---

# 9. Défaut important : traduction inchangée non signalée comme problème

Test réel :

```python id="75mhtu"
unit_quality(
    "This is a test.",
    "This is a test.",
    target_lang="fr"
)
```

Résultat actuel :

```text id="c1bfui"
unchanged = True
needs_review = False
```

C’est faux.

Si on traduit de l’anglais vers le français et que la sortie reste identique, c’est presque toujours un échec, sauf cas protégé : acronymes, formules, noms propres, codes, URL, etc.

### Correction nécessaire

Dans `quality.py` :

```python id="829r9p"
unchanged_problem = (
    unchanged
    and item.get("strategy") not in {"exact_preserve", "keep_original", "background_only"}
    and item.get("object_type") not in {"formula", "code", "url", "reference_link", "citation"}
)
```

Puis :

```python id="xmm70j"
needs_review = any([
    empty,
    unchanged_problem,
    number_mismatch,
    unit_mismatch,
    protected_mismatch,
    source_leak,
    overflow_risk == "high",
])
```

---

# 10. Défaut important : `TranslatorBridge` transmet trop peu de contexte

`context_builder.py` construit un profil riche :

```text id="5mmnqb"
source_lang
target_lang
target_variant
domain
subdomain
document_type
page_role
page_family
layout_type
style
tone
terminology
protected_tokens
```

Mais `translator_bridge.py` transmet surtout :

```python id="3jqowr"
target_lang
block_role
strategy
style
tone
object_class
object_type
phrase_semantics
```

Il manque :

```text id="pog9j5"
source_lang
domain
subdomain
document_type
page_role
page_family
layout_type
terminology
context_before
context_after
section_title
protected_tokens
wysiwyg_constraints
```

Donc l’unité prétend suivre une méthodologie riche, mais l’appel réel au traducteur reste assez pauvre.

### Correction nécessaire

Préparer un `translation_context` complet :

```python id="h69tew"
kwargs = {
    "source_lang": profile.get("source_lang"),
    "target_lang": profile.get("target_lang"),
    "target_variant": profile.get("target_variant"),
    "domain": profile.get("domain"),
    "subdomain": profile.get("subdomain"),
    "document_type": profile.get("document_type"),
    "page_role": profile.get("page_role"),
    "page_family": profile.get("page_family"),
    "layout_type": profile.get("layout_type"),
    "style": profile.get("style"),
    "tone": profile.get("tone"),
    "terminology": profile.get("terminology") or {},
    "context_before": (item.get("context") or {}).get("previous_text"),
    "context_after": (item.get("context") or {}).get("next_text"),
    "section_title": (item.get("context") or {}).get("section_title"),
    "wysiwyg_constraints": (item.get("context") or {}).get("wysiwyg_constraints"),
}
```

Si `DocumentTranslator.translate_text()` ne supporte pas encore ces paramètres, il faut adapter `translator.py`.

---

# 11. Défaut important : `TypeError` est masqué trop largement

Dans `translator_bridge.py` :

```python id="9d49d9"
try:
    return translator.translate_text(text, **kwargs)
except TypeError:
    return translator.translate_text(text)
```

Cela peut masquer un vrai bug interne dans `translate_text()`.

Exemple :

```text id="grhw5i"
translate_text() accepte bien kwargs,
mais lève TypeError à cause d’un bug interne
→ le bridge rappelle translate_text(text)
→ le bug est caché
```

### Correction recommandée

Inspecter la signature au lieu de catcher tout `TypeError` :

```python id="f3r0gd"
import inspect

sig = inspect.signature(translator.translate_text)
accepted = set(sig.parameters)
filtered_kwargs = {
    k: v for k, v in kwargs.items()
    if k in accepted or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
}
return translator.translate_text(text, **filtered_kwargs)
```

---

# 12. Défaut WYSIWYG : `reconstruction_units` est trop naïf

`projection.py` construit :

```python id="6njgw2"
translated_input["views"]["reconstruction_units"] = _reconstruction_units(translated_input)
```

Mais cette vue inclut simplement toutes les unités ayant `content.translated_text`.

Il manque des informations critiques :

```text id="t9uur1"
render_level
source_units_consumed
preferred_over_children
overflow_policy
layout_budget
bbox source fiable ou non
style inheritance
baseline
line_break_policy
reflow_mode
skip_original_units
```

Pour le reconstructeur, il faut une vue plus contractuelle :

```json id="qsnbj8"
{
  "unit_id": "sp1",
  "render_level": "semantic_phrase",
  "translated_text": "...",
  "bbox": [...],
  "style_source": "dominant_source_span",
  "source_unit_ids": ["p1", "p2"],
  "consume_source_units": true,
  "overflow_policy": "shrink_or_reflow",
  "line_break_policy": "semantic_reflow",
  "render_contract": {...}
}
```

Sinon le reconstructeur risque de choisir le mauvais niveau.

---

# 13. Défaut documentaire : `_section_title()` prend le premier titre de la page

Dans `context_builder.py` :

```python id="2g01aa"
for unit in units:
    if role in {"section_heading", "title"}:
        return text
```

Cela prend le premier titre trouvé, pas le titre hiérarchiquement ou spatialement lié à l’unité traduite.

Pour une page avec plusieurs sections, le contexte sera faux.

### Correction attendue

Pour chaque unité :

```text id="g91byk"
chercher le titre précédent le plus proche dans l’ordre de lecture
ou utiliser les relations du graphe PAGEPRINT
ou utiliser structural_context.parent_section_id si disponible
```

---

# 14. Défaut de robustesse : une erreur traducteur peut casser toute la page

Dans `builder.py`, cette liste :

```python id="p17nti"
translated_units = [
    self._translate_item(...)
    for item in units
]
```

Si une seule unité lève une exception, toute la traduction de page échoue.

Pour une vraie unité pipeline, il faut que l’échec soit local :

```json id="tvz24f"
{
  "status": "error",
  "error_type": "...",
  "source_text": "...",
  "translated_text": null,
  "needs_review": true
}
```

Et la page continue.

---

# 15. Ce qui va marcher correctement

Sur des cas simples :

```text id="2pkte1"
- document textuel simple ;
- unités phrase propres ;
- pas de semantic_system complexe ;
- pas de chevauchement phrase/line ;
- traducteur disponible ;
- peu de tokens techniques ;
- reconstruction au même niveau que la traduction ;
```

ça devrait marcher.

Le pipeline va :

```text id="lwjiui"
sélectionner les phrases
protéger les nombres/URLs
appeler le traducteur
restaurer les tokens
réinjecter translated_text
produire translation_result
```

Donc ce n’est pas cassé.

---

# 16. Ce qui ne marchera pas correctement

Sur des cas réels WYSIWYG, surtout :

```text id="7wxrv7"
- phrases sur plusieurs lignes ;
- semantic_phrases couvrant plusieurs phrases visuelles ;
- documents techniques avec mots composés ;
- blocs avec formules + texte ;
- tableaux ;
- couvertures ;
- diagrammes ;
- textes ancrés sur images ;
- documents avec semantic_group + semantic_phrase ;
- reconstruction au niveau phrase/span ;
```

il y aura des erreurs probables :

```text id="1v4c7k"
doublons de traduction
fragments non traduits
traductions injectées dans une bbox trop petite
mots composés non traduits
unités background_only traduites par erreur
traduction inchangée non signalée
mauvais contexte de section
```

---

# 17. Priorités de correction

## P0 — À corriger avant toute suite

```text id="gplz1a"
1. Empêcher double sélection semantic_phrase + line/phrase/block.
2. Empêcher double sélection semantic_phrase + semantic_group.
3. Respecter translatable=False et background_only dans semantic_system.
4. Ne jamais injecter toute une semantic_phrase dans le premier fragment source.
5. Marquer unchanged comme needs_review si source_lang ≠ target_lang.
```

## P1 — À corriger ensuite

```text id="xm91r5"
6. Corriger la protection des mots composés.
7. Utiliser profile.protected_tokens et item.protected.
8. Transmettre le contexte riche au traducteur.
9. Ne pas masquer TypeError trop largement.
10. Ajouter try/except par unité traduite.
```

## P2 — Pour la vraie compatibilité WYSIWYG

```text id="3zm4x3"
11. Créer une vraie vue reconstruction_units contractuelle.
12. Ajouter consume_source_units / skip_individual_render.
13. Ajouter layout_budget et overflow_policy.
14. Utiliser le graphe PAGEPRINT pour section_title/context.
15. Ajouter des tests contractuels.
```

---

# 18. Tests minimaux à ajouter

Il faut au moins ces tests :

```text id="dzdnsc"
test_no_duplicate_semantic_phrase_and_line
test_no_duplicate_semantic_phrase_and_semantic_group
test_skip_non_translatable_semantic_entry
test_background_only_not_translated
test_hyphenated_words_not_protected_as_formula
test_profile_protected_tokens_are_preserved
test_unchanged_translation_needs_review
test_semantic_phrase_projection_consumes_source_units
test_translator_exception_is_local
test_reconstruction_units_prefer_semantic_level
```

Ces tests sont indispensables avant de dire que l’unité est stable.

---

# 19. Conclusion

La nouvelle archive montre un vrai progrès. On est passé de :

```text id="iqyr55"
wrapper minimal
```

à :

```text id="bi3s0k"
unité structurée avec sélection, contexte, protection, QA et projection
```

Mais il faut être strict : **ça ne marchera pas correctement en production WYSIWYG dans l’état actuel**.

Le problème n’est plus l’architecture générale. Elle est bonne.
Le problème est maintenant dans les **règles de sélection, de protection et de projection**.

Je classerais cette version comme :

```text id="v4zsp4"
PAGETRANSLATE V1-alpha avancée
```

Pas encore :

```text id="whq3qf"
PAGETRANSLATE V1-beta
```

Le prochain objectif doit être clair :

```text id="0q4j0p"
zéro doublon,
zéro traduction d’unité protégée,
projection sémantique fiable,
QA qui bloque les sorties suspectes.
```

Après correction des P0, l’unité commencera réellement à pouvoir soutenir la reconstruction WYSIWYG.

