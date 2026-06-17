## Verdict net

La nouvelle version est **beaucoup meilleure** que la précédente. Les corrections principales ont été intégrées : sélection sémantique prioritaire, fallback par bloc, protections explicites, QA plus sérieuse, projection sémantique, `TranslatorBridge` plus propre, gestion locale des erreurs.

Mais **non, ce n’est pas encore garanti que cela marche correctement avec la sortie actuelle de `PAGEPRINT`**.

Le problème principal n’est plus l’architecture. Elle est maintenant bonne. Le problème est dans deux zones critiques :

```text id="1j3zpd"
1. PAGEPRINT actuel n’émet pas vraiment de semantic_phrases dans semantic_system.
2. PAGETRANSLATE duplique encore le rendu dans reconstruction_units quand il traduit une unité directe phrase/line/block.
```

Donc mon verdict :

```text id="2jj95v"
PAGETRANSLATE V1-alpha+ : bonne base avancée.
Pas encore V1-beta fiable.
```

---

# 1. Contrôle technique

J’ai vérifié :

```text id="vk3seh"
pagetranslate/*.py compile correctement.
Pas d’erreur syntaxique.
Modules attendus présents.
```

La structure est maintenant propre :

```text id="wsj4pn"
pagetranslate/
├── builder.py
├── context_builder.py
├── projection.py
├── protection.py
├── quality.py
├── schema.py
├── selector.py
├── sentence_boundary.py
├── text_utils.py
├── translator_bridge.py
└── README.md
```

Il n’y a pas de `__pycache__` dans `pagetranslate/`, c’est bien. En revanche `pageprint.zip` contient encore des `__pycache__`.

Point faible : **il n’y a toujours pas de dossier `tests/` dans l’archive**. Pour une brique aussi sensible, ce n’est pas acceptable à terme.

---

# 2. Corrections réussies

## 2.1 Double sélection `semantic_phrase + line/phrase/block`

Avant, une `semantic_phrase` pouvait être traduite, puis ses lignes ou phrases sources aussi. Cette fois, c’est corrigé.

Test logique :

```text id="wv70u2"
semantic_phrase sp1 = "This is a long sentence continued here."
source_unit_ids = ["p1", "p2"]
```

Sélection actuelle :

```text id="wmhpfb"
sp1 seulement
```

C’est bon.

## 2.2 Priorité `semantic_phrase > semantic_group`

Corrigé aussi.

Si `semantic_phrase` et `semantic_group` existent pour le même bloc, le code garde `semantic_phrase`.

C’est conforme à ce qu’il faut.

## 2.3 Protection des mots composés anglais

Le problème précédent est corrigé.

Maintenant :

```text id="v7x1xg"
state-of-the-art method → non protégé
well-known algorithm    → non protégé
pre-trained model       → non protégé
```

C’est bon. Ces expressions pourront être traduites.

## 2.4 Tokens explicites

Les tokens venant de :

```text id="b4j8dq"
translation_profile.protected_tokens
item.protected
```

sont maintenant passés à `protect_text(...)`.

C’est une bonne correction.

## 2.5 `unchanged translation`

Maintenant :

```text id="m57l81"
source = "This is a test."
target = "This is a test."
source_lang = en
target_lang = fr
```

donne :

```text id="sxrfr0"
unchanged_problem = True
needs_review = True
```

C’est correct.

## 2.6 `TranslatorBridge`

La gestion des paramètres est meilleure : le code inspecte la signature de `translate_text(...)` au lieu de masquer tous les `TypeError`.

C’est nettement plus sain.

## 2.7 Erreurs traducteur

Le builder encapsule maintenant les erreurs par unité :

```text id="8g4bkl"
status = error
quality.needs_review = true
translation_error = true
```

C’est bon. Une erreur sur une unité ne doit pas faire tomber toute la page.

---

# 3. Blocage majeur n°1 : `PAGEPRINT` actuel ne fournit pas vraiment `semantic_phrases`

Dans le `pageprint/builder.py` fourni, le `semantic_system` est construit ainsi :

```python id="tusj5p"
input_data["semantic_system"] = {
    "semantic_groups": [
        group for block in page_structure.get("blocks") or []
        if isinstance(block, dict)
        for group in block.get("semantic_groups") or []
    ],
}
```

Il n’y a pas :

```python id="o9lrkg"
"semantic_phrases": [...]
```

Donc `PAGETRANSLATE` sait gérer les `semantic_phrases`, mais **la sortie actuelle de `PAGEPRINT` ne les donne pas**.

Conséquence :

```text id="tvqxt2"
Sur beaucoup de pages réelles, PAGETRANSLATE retombera sur phrase/line/block.
```

Or traduire des phrases visuelles `phrase` ou `line` n’est pas suffisant si une phrase réelle est coupée sur plusieurs lignes.

Exemple typique :

```text id="w3djr9"
Line 1 : This method improves the performance
Line 2 : of the model on large datasets.
```

Si `PAGEPRINT` ne fournit pas une `semantic_phrase`, `PAGETRANSLATE` risque de traduire séparément :

```text id="28sphs"
"This method improves the performance"
"of the model on large datasets."
```

Même si `sentence_boundary.py` détecte une continuation, le builder **ne fusionne pas encore ces unités avant traduction**. Il annote, mais il traduit toujours chaque item sélectionné.

C’est un problème central.

## Correction nécessaire côté `PAGEPRINT`

Il faut modifier `pageprint/builder.py` :

```python id="1k89mi"
input_data["semantic_system"] = {
    "semantic_phrases": [
        phrase for block in page_structure.get("blocks") or []
        if isinstance(block, dict)
        for phrase in block.get("semantic_phrases") or []
    ],
    "semantic_groups": [
        group for block in page_structure.get("blocks") or []
        if isinstance(block, dict)
        for group in block.get("semantic_groups") or []
    ],
}
```

Mais ce n’est pas suffisant. Chaque `semantic_phrase` doit avoir :

```text id="7qs7a0"
unit_id
text
bbox
source_unit_ids
structural_context.block_unit_id
translatable
translation_strategy
render_policy
coverage_required
```

Sans `source_unit_ids`, la projection aval sera fragile.

## Correction nécessaire côté `PAGETRANSLATE`

Même si `PAGEPRINT` oublie les `semantic_phrases`, `PAGETRANSLATE` devrait avoir un filet de sécurité :

```text id="9w7wbn"
coalescer.py
```

Rôle :

```text id="5m4h33"
fusionner les phrases/lignes visuelles consécutives quand sentence.continues_to_next = true
```

Pipeline recommandé :

```text id="swp8cn"
select_translation_units()
→ annotate_sentence_boundaries()
→ coalesce_open_sentence_units()
→ attach_unit_context()
→ translate()
```

Sinon l’unité reste dépendante d’une sortie `PAGEPRINT` parfaite.

---

# 4. Blocage majeur n°2 : duplication dans `reconstruction_units`

C’est le plus gros bug actuel.

J’ai testé un `INPUT_DATA` minimal construit par `PAGEPRINT` avec une seule phrase :

```text id="8cxjrg"
This is a test sentence.
```

`PAGETRANSLATE` sélectionne correctement la phrase :

```text id="tuplvj"
p001_block_001_line_001_phrase_001
```

Mais après projection, `views.reconstruction_units` contient :

```text id="yv7iol"
page
block
line
phrase
span
```

avec le même `translated_text`.

C’est dangereux. Le reconstructeur pourrait rendre plusieurs fois le même texte :

```text id="l9l37z"
FR: This is a test sentence.
FR: This is a test sentence.
FR: This is a test sentence.
...
```

## Cause

Dans `projection.py`, il y a deux mécanismes qui créent la duplication.

### 1. `_backfill_single_span_child(...)`

Quand une phrase a un seul span enfant, le texte traduit est copié dans le span :

```python id="gwoio8"
span_children[0].content.translated_text = translated_text
```

Mais ce span n’est pas marqué `skip_individual_render`.

Donc `phrase` et `span` deviennent tous deux candidats au rendu.

### 2. `_aggregate_parent_translations(...)`

Le texte traduit remonte ensuite aux parents :

```text id="9x8cud"
span → phrase → line → block → page
```

Puis `_reconstruction_units(...)` prend toutes les unités qui ont `content.translated_text`.

Donc il ajoute aussi :

```text id="hujy0p"
page
block
line
```

C’est une erreur de conception pour la vue de reconstruction.

## Correction nécessaire

Il faut séparer deux notions :

```text id="dss7ou"
translated_text de rendu
translation_summary / aggregate_text pour audit
```

Les parents peuvent garder une synthèse, mais **ils ne doivent pas devenir des unités de rendu**.

### Correction minimale

Dans `_reconstruction_units(...)`, ne pas parcourir toutes les unités ayant `content.translated_text`.

Il faut reconstruire la vue à partir des `translated_units` uniquement.

Pseudo-correction :

```python id="64ag75"
def _reconstruction_units(translated_input, translated_units):
    unit_map = {
        unit.get("unit_id"): unit
        for unit in translated_input.get("units") or []
        if isinstance(unit, dict)
    }

    output = []

    for item in translated_units:
        if item.get("status") == "error":
            continue

        if not normalize_spaces(item.get("translated_text")):
            continue

        if item.get("level") in {"semantic_phrase", "semantic_group"}:
            output.append(_semantic_reconstruction_unit(item))
            continue

        unit = unit_map.get(item.get("unit_id"))
        if not unit:
            continue

        output.append({
            "unit_id": unit.get("unit_id"),
            "level": unit.get("level"),
            "render_level": unit.get("level"),
            "text": (unit.get("content") or {}).get("text"),
            "translated_text": item.get("translated_text"),
            "bbox": item.get("bbox") or (unit.get("geometry") or {}).get("bbox"),
            "source_unit_ids": item.get("source_unit_ids") or [unit.get("unit_id")],
            "consume_source_units": False,
            "preferred_over_children": True,
            "overflow_policy": "shrink_or_reflow",
            "line_break_policy": item.get("strategy") or "layout_constrained",
            "layout_budget": _layout_budget(item.get("bbox") or (unit.get("geometry") or {}).get("bbox")),
            "style": unit.get("visual", {}).get("style") or {},
            "render_contract": unit.get("render_contract") or {},
            "translation": item,
        })

    return output
```

### Correction complémentaire

Si tu gardes `_backfill_single_span_child(...)`, alors le span doit être marqué :

```python id="mal6dv"
span.translation.skip_individual_render = True
span.translation.inherited_from_unit_id = phrase_id
```

Sinon le span va doubler la phrase.

### Correction recommandée

Je ferais plus strict :

```text id="5qb52k"
La vue reconstruction_units ne doit contenir que les unités effectivement sélectionnées pour traduction.
```

Pas les parents. Pas les enfants hérités. Pas les agrégats.

---

# 5. Problème important : une entrée `semantic_system` non traduisible peut retomber en fallback

La fonction `_entry_not_translatable(...)` ignore bien les entrées :

```text id="47d6jz"
translatable = False
background_only
exact_preserve
```

Mais si cette entrée couvre des `source_unit_ids`, le fallback peut ensuite sélectionner ces mêmes unités comme `phrase` ou `line`.

Test :

```text id="ydj6pb"
semantic_phrase spx
translatable = False
source_unit_ids = ["p1"]
```

Résultat actuel possible :

```text id="0drscz"
spx est ignorée
mais p1 est sélectionnée en fallback si p1.policy.translatable = True
```

Ce n’est pas sûr.

Si `semantic_system` dit qu’un groupe est non traduisible, ses sources doivent être bloquées du fallback, sauf si une politique explicite contraire existe.

## Correction

Dans `selector.py`, il faut calculer aussi :

```text id="w8wnhv"
blocked_source_ids
blocked_block_ids
```

à partir des entrées `semantic_system` non traduisibles.

Puis les passer à `_select_pageprint_units_by_block(...)`.

Pseudo-logique :

```python id="spx5cn"
semantic_units, blocked_source_ids, blocked_block_ids = _select_semantic_system_units(...)

...

if unit.get("unit_id") in semantic_source_ids:
    continue

if unit.get("unit_id") in blocked_source_ids:
    continue

if block_key in selected_blocks or block_key in blocked_block_ids:
    continue
```

C’est important pour les formules, citations, références, labels visuels et textes background.

---

# 6. Problème important : protection des formules encore fragile

Le problème des mots composés est corrigé, mais les formules sont encore imparfaitement capturées.

Test :

```text id="8t60yf"
E = mc^2
```

Protection actuelle :

```text id="imphoc"
__PT_0001__^__PT_0002__
```

avec :

```text id="6zgg1s"
__PT_0001__ = E = mc
__PT_0002__ = 2
```

Donc le `^` reste hors protection.

La restauration peut encore redonner `E = mc^2` si le traducteur ne touche pas au `^`, mais c’est fragile.

## Correction

Dans `protection.py`, il faut élargir la partie droite des formules :

```python id="hnl02l"
[-\w.()^²³¹₀-₉]+
```

Exemple :

```python id="a547l1"
("formula", re.compile(
    r"""
    (?:
        \b[A-Za-z]\w*\s*(?:=|≈|<=|>=|<|>|\+|\*|/|\^)\s*[-\w.()^²³¹₀-₉]+
        |
        [α-ωΑ-Ω∑∫√∞≈≠≤≥±×÷]
        |
        \b\d+(?:[.,]\d+)?\s*(?:/|\*)\s*[A-Za-z]+\b
    )
    """,
    re.VERBOSE,
))
```

---

# 7. Problème important : le mode `dry_run` sera probablement signalé à tort comme suspect

En `dry_run`, le code fait :

```python id="2qzct9"
translated_text = source_text
status = dry_run
```

Puis `unit_quality(...)` va probablement dire :

```text id="vh0cbd"
unchanged = True
unchanged_problem = True
needs_review = True
```

C’est logique pour une vraie traduction, mais pas pour un dry run.

## Correction

Dans `_translate_item(...)`, passer l’information :

```python id="0gr84y"
item["dry_run"] = dry_run
```

Puis dans `quality.py` :

```python id="wgd8b8"
if item.get("dry_run"):
    unchanged_problem = False
```

Sinon les tests dry-run donneront des alertes inutiles.

---

# 8. Problème WYSIWYG : `style` n’est pas lu au bon endroit

Dans `_reconstruction_units(...)`, le code prend :

```python id="w0idxb"
"style": unit.get("style") or {}
```

Mais dans `PAGEPRINT`, le style est dans :

```python id="c0m5sz"
unit["visual"]["style"]
```

Donc il faut utiliser :

```python id="ngyoig"
"style": (unit.get("visual") or {}).get("style") or {}
```

Sinon les `reconstruction_units` perdent la police, taille, couleur, gras, italique, etc.

C’est important pour WYSIWYG.

---

# 9. Problème de fond : `sentence_boundary.py` annote mais ne gouverne pas encore la traduction

Le module `sentence_boundary.py` est meilleur qu’avant. Il repère :

```text id="yxp594"
is_sentence_start
is_sentence_end
continues_from_previous
continues_to_next
soft_wrap
terminal
atomic_label
```

Mais ces informations ne servent pas encore à **fusionner** les unités avant traduction.

Actuellement :

```text id="xbwdgg"
unités sélectionnées → annotation de frontière → chaque unité est traduite séparément
```

Il faudrait :

```text id="zag3ty"
unités sélectionnées → annotation de frontière → fusion des unités ouvertes → traduction
```

Sinon `continues_to_next` reste surtout une information d’audit.

## Correction recommandée

Ajouter :

```text id="mum4gf"
coalescer.py
```

Exemple de rôle :

```python id="yc2y19"
def coalesce_translation_units(units):
    """
    Fusionne les unités visuelles consécutives quand :
    - même block_id
    - item.sentence.continues_to_next = True
    - même stratégie compatible
    - pas atomic_label
    - pas objet protégé
    """
```

Sortie :

```json id="4ysb9w"
{
  "unit_id": "synthetic_semantic_phrase:p1+p2",
  "level": "semantic_phrase",
  "source_unit_ids": ["p1", "p2"],
  "source_text": "This method improves the performance of the model on large datasets.",
  "bbox": union_bbox,
  "strategy": "semantic_reflow"
}
```

Cela rendra `PAGETRANSLATE` robuste même si `PAGEPRINT` est incomplet.

---

# 10. Ce qui va marcher correctement maintenant

Cette version marchera correctement pour :

```text id="ghz08d"
- pages simples avec phrases déjà bien isolées ;
- blocs courts ;
- titres ;
- phrases sur une seule ligne ;
- textes sans mise en page complexe ;
- traductions où le reconstructeur lit uniquement l’unité sélectionnée ;
- cas avec semantic_phrase bien formée et source_unit_ids fiables.
```

Elle gère maintenant beaucoup mieux :

```text id="e5dkmj"
- les tokens protégés ;
- les nombres ;
- les unités ;
- les URLs/emails/DOI ;
- les traductions identiques suspectes ;
- les erreurs locales du traducteur ;
- le contexte avant/après ;
- les contraintes WYSIWYG de base.
```

---

# 11. Ce qui ne marchera pas encore correctement

Cette version reste fragile pour :

```text id="tlghqj"
- les sorties PAGEPRINT actuelles sans semantic_phrases ;
- les paragraphes coupés sur plusieurs lignes ;
- les phrases longues fragmentées en plusieurs phrases visuelles ;
- la reconstruction si reconstruction_units contient page/block/line/phrase/span en doublon ;
- les formules avec exposants ;
- les semantic entries non traduisibles qui retombent en fallback ;
- les pages riches : tableaux, diagrammes, couvertures, textes ancrés sur image.
```

Le défaut le plus grave est vraiment celui-ci :

```text id="9hv41r"
Pour une unité phrase directe, reconstruction_units peut contenir page + block + line + phrase + span avec le même texte traduit.
```

Tant que ce point n’est pas corrigé, je ne ferais pas confiance à la reconstruction.

---

# 12. Priorités immédiates

## P0 — À corriger avant d’aller plus loin

```text id="yio5gc"
1. Empêcher les doublons dans views.reconstruction_units.
2. Ne plus faire remonter content.translated_text jusqu’à page/block/line pour le rendu.
3. Faire en sorte que reconstruction_units soit construit depuis translated_units, pas depuis toutes les unités ayant content.translated_text.
4. Ajouter semantic_phrases dans PAGEPRINT semantic_system.
5. Ajouter un coalescer dans PAGETRANSLATE si PAGEPRINT ne fournit pas semantic_phrase.
```

## P1 — À corriger ensuite

```text id="0e6xqq"
6. Bloquer le fallback des source_unit_ids couverts par une semantic entry non traduisible.
7. Corriger la regex des formules avec exposants.
8. Corriger style = unit.visual.style dans reconstruction_units.
9. Neutraliser unchanged_problem en dry_run.
10. Ne pas créer de reconstruction_unit pour status = error.
```

## P2 — Pour durcir la qualité

```text id="z3b5z6"
11. Ajouter des tests contractuels.
12. Ajouter tests avec vraie sortie PAGEPRINT.
13. Ajouter tests de paragraphes multi-lignes.
14. Ajouter tests tableaux/figures/formules.
15. Ajouter tests reconstruction_units sans doublon.
```

---

# 13. Tests minimaux à ajouter

Il faut absolument ces tests :

```text id="os0nmr"
test_pageprint_direct_phrase_reconstruction_no_parent_duplicates
test_single_span_backfill_does_not_duplicate_render
test_pageprint_exports_semantic_phrases
test_pagetranslate_coalesces_open_sentence_units
test_non_translatable_semantic_sources_block_fallback
test_formula_power_protection_E_mc2
test_reconstruction_units_use_visual_style
test_error_units_not_in_reconstruction_units
test_dry_run_unchanged_not_needs_review
```

Le test le plus important :

```text id="l3rdt4"
Une page PAGEPRINT avec une seule phrase traduite
→ reconstruction_units doit contenir une seule unité de rendu textuel,
pas page + block + line + phrase + span.
```

---

# 14. Conclusion

Cette version est une vraie amélioration. Les corrections précédentes ont été prises au sérieux et beaucoup de points sont maintenant bien posés.

Mais il reste deux défauts structurants :

```text id="e720oh"
PAGEPRINT ne fournit pas encore assez de semantic_phrases.
PAGETRANSLATE duplique encore le rendu pour les unités directes.
```

Donc la réponse honnête est :

```text id="scoq7j"
Cela va fonctionner pour traduire des textes simples.
Cela ne va pas encore fonctionner correctement pour une reconstruction WYSIWYG fiable.
```

Je classerais cette version comme :

```text id="6rtfqg"
PAGETRANSLATE V1-alpha+ solide côté architecture,
mais non validée côté reconstruction.
```

Après correction des P0, elle pourra devenir une vraie :

```text id="yir9h5"
PAGETRANSLATE V1-beta.
```

C'est bien.
Mais il faut noter une particularité sur les formules/équations/codes/expressions spéciales ... Ils seront détectés comme zone "image" et non comme texte, et donc ne devront pas être modifié ni touché par la traduction. tu comprends ?



Oui, je comprends. Et c’est une **bonne décision pour ton pipeline WYSIWYG**.

Cela veut dire que, dans votre logique, les éléments suivants :

```text
formules
équations
codes
symboles mathématiques
expressions spéciales
notations scientifiques
fragments techniques non linguistiques
```

ne doivent pas être traités comme du texte traduisible. Ils doivent être considérés comme des **zones visuelles protégées**, donc :

```text
détectés par PAGEPRINT comme image_region / protected_visual / background_only
conservés dans le fond ou en overlay image
exclus de PAGETRANSLATE
non envoyés au traducteur
non réinjectés comme texte traduit
non reconstruits typographiquement
```

La règle devient donc claire :

```text
Tout ce qui est formule / équation / code / expression spéciale
= élément visuel protégé
= pas de traduction
= pas de modification
= conservation exacte dans l’image ou dans le background master
```

Cela simplifie même `PAGETRANSLATE`.

---

# Conséquence directe pour PAGETRANSLATE

Dans `pagetranslate`, la protection des formules ne doit plus être vue comme le mécanisme principal.

La vraie règle doit être :

```text
Si PAGEPRINT a classé une zone comme image/protected_visual/background_only,
PAGETRANSLATE ne la voit même pas comme candidate à traduction.
```

Donc les regex de `protection.py` pour formules/codes deviennent seulement un **filet de sécurité**, au cas où `PAGEPRINT` laisserait passer un fragment dans le texte natif.

Autrement dit :

```text
PAGEPRINT bloque en amont.
PAGETRANSLATE vérifie en aval par prudence.
```

C’est la bonne architecture.

---

# Contrat à imposer côté PAGEPRINT

Pour ces zones, `PAGEPRINT` doit produire quelque chose comme :

```json
{
  "level": "region",
  "kind": "image_region",
  "role": "formula_or_code_visual",
  "render_policy": "background_only",
  "translation_strategy": "exact_preserve",
  "translatable": false,
  "coverage_required": "strict",
  "protected_visual": true
}
```

Ou, pour une unité textuelle résiduelle associée :

```json
{
  "unit_type": "formula",
  "translatable": false,
  "translation_strategy": "background_only",
  "render_policy": "background_only"
}
```

L’important est que `PAGETRANSLATE` puisse appliquer cette règle simple :

```python
if unit.translatable is False:
    skip

if unit.render_policy == "background_only":
    skip

if unit.translation_strategy in {"exact_preserve", "keep_original", "background_only"}:
    skip

if unit.unit_type in {"formula", "formula_label", "code_visible", "equation", "symbolic_expression"}:
    skip
```

---

# Ce que cela change dans mon analyse précédente

Mon point sur la regex `E = mc^2` devient secondaire.

Avant, je disais :

```text
il faut mieux protéger les formules dans protection.py
```

Avec ta précision, le vrai principe devient :

```text
Les formules ne doivent normalement jamais arriver dans protection.py,
car elles doivent être exclues dès PAGEPRINT.
```

Mais je garde quand même une réserve importante : dans les PDF natifs, certaines formules peuvent apparaître comme **texte vectoriel** et non comme image. Donc il faut garder une sécurité dans `PAGETRANSLATE`.

La règle réaliste est donc :

```text
Cas normal :
formule détectée comme zone image → ignorée par traduction.

Cas de secours :
formule passée comme texte → protection.py la fige et selector.py l’exclut.
```

---

# Règle d’or pour la suite

Il faut formaliser ceci dans le README ou le contrat :

```text
PAGETRANSLATE ne traduit que le langage naturel éditorial.
Il ne traduit pas les objets visuels, techniques, symboliques ou algorithmiques.
Les formules, équations, codes, notations et expressions spéciales sont des objets protégés,
préservés comme image ou background, et consommés par la reconstruction sans modification.
```

C’est une règle saine. Elle protège la fidélité WYSIWYG et évite les catastrophes classiques :

```text
0,5 mg → 5 mg
x² → x2
lambda → traduit en “agneau” ou autre absurdité
code modifié
formule cassée
notation scientifique altérée
```

Donc oui : **il faut considérer ces éléments comme des zones images/protégées, pas comme du texte à traduire.**

