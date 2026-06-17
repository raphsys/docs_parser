# Verdict général

L’essai sur 10 pages confirme que le pipeline a progressé, mais **il n’est pas encore prêt pour tester sérieusement le moteur de traduction en conditions réelles**.

Le problème n’est plus seulement la page de table des matières. Sur les 10 pages, on voit des défauts transversaux :

```text id="6tw7i0"
1. PAGEPRINT classe encore trop de texte naturel en protected_visual/background_only.
2. Les rôles structurels sont presque toujours absents dans les unités traduites.
3. semantic_system est vide sur les 10 pages.
4. PAGETRANSLATE compense en fusionnant lui-même, mais il fusionne parfois mal.
5. Les tableaux, index, figures, captions, commandes/code ne sont pas encore traités par des vues spécialisées.
6. L’audit dit “status: ok”, mais ce “ok” signifie surtout JSON valide, pas résultat fonctionnel correct.
```

Donc :

```text id="y51gpq"
Architecture : bonne.
Audit visuel : très utile.
Sélection actuelle : pas encore suffisamment fiable.
Passage au moteur de traduction global : trop tôt.
```

On peut avancer **en parallèle** sur le moteur de traduction, mais il ne faut pas encore utiliser ces 10 pages comme test global du moteur. Sinon, on va confondre erreurs de traduction et erreurs de sélection/extraction.

---

# 1. Synthèse chiffrée des 10 pages

Sur les 10 pages auditées :

```text id="ozozvm"
pages auditées : 10
translation_units sélectionnées : 212
unités coalescées synthétiques : 48
translation_units avec role=None : 212 / 212
semantic_system.semantic_phrases : 0 sur toutes les pages
semantic_system.semantic_groups : 0 sur toutes les pages
needs_review : 70 / 212
pages_with_table_like_signal : []
```

Le chiffre le plus grave est celui-ci :

```text id="ic1469"
role=None : 212 / 212
```

Cela veut dire que `PAGETRANSLATE` traduit sans savoir si l’unité est :

```text id="nqeafg"
titre
paragraphe
caption
table_cell
index_entry
author_name
page_header
footer
diagram_label
code_span
formula
publisher_mark
```

Or c’est précisément ce rôle qui doit piloter la traduction.

Deuxième chiffre grave :

```text id="c9qw7d"
semantic_system vide sur toutes les pages.
```

Donc la promesse méthodologique :

```text id="381pab"
PAGEPRINT comprend la page et fournit les unités sémantiques ;
PAGETRANSLATE traduit ces unités.
```

n’est pas encore tenue. En réalité :

```text id="snwsqb"
PAGEPRINT fournit surtout des unités visuelles ;
PAGETRANSLATE reconstruit lui-même des semantic_phrases avec un coalescer générique.
```

C’est fragile.

---

# 2. Analyse par type de page

## 2.1 Page 58 — diagramme + prose

Points positifs :

```text id="1bjn1u"
- Les bboxes collent globalement au texte.
- Les word/char ne sont pas envoyés en traduction.
- Certains labels de diagramme sont bien détectés.
```

Problèmes :

```text id="sv3dqj"
- Des phrases naturelles sont classées protected_visual.
- Figure 2.1 est envoyée au traducteur.
- Certains labels courts comme Input/Output sont envoyés avec needs_review high.
- Des lignes intermédiaires protégées à tort provoquent des fusions incomplètes.
```

Exemple critique :

```text id="wd4pkt"
We will dive deep into each of these steps.
You will see that building a neural network requires making necessary design decisions...
```

Une partie est classée `protected_visual`, puis le coalescer saute ou fusionne mal les fragments restants.

Cause probable : `policy_compiler._code_like_text()` classe trop vite comme code/protected, notamment à cause de mots comme `if`, `for`, `function`, etc.

---

## 2.2 Page 187 — texte + encadré continué

Points positifs :

```text id="xdx4e5"
- La structure visuelle générale est correcte.
- Les bullets sont mieux visibles.
```

Problèmes :

```text id="nt77c3"
- Une grande partie de texte naturel dans l’encadré est background_only.
- CHAPTER 4 est envoyé au traducteur.
- Des phrases sont coupées en fragments incohérents : "memory), you can determine..."
- Les bullets sont encore dans le texte source au lieu d’être séparés comme markers.
```

Ici, l’encadré `continued` devrait être traité comme :

```text id="rlf273"
continued_label → traduisible ou exact_preserve selon politique
box_body        → texte naturel traduisible
bullet_marker   → exact_preserve
bullet_text     → traduisible
```

Actuellement, la politique mélange tout.

---

## 2.3 Page 206 — projet + liste + code Keras

Points positifs :

```text id="g6dk6o"
- Les items de liste sont visibles.
- Le code en bas est en partie identifié visuellement.
```

Problèmes :

```text id="bkd4g6"
- Des items naturels de liste sont classés protected_visual.
- Les lignes de code Keras ne sont pas représentées comme code_block/code_line propre.
- Certaines phrases sont fusionnées avec césures non résolues : nor- malization, vari- ance.
- Les bullets/dashes ne sont pas encore séparés proprement.
```

Ici il faut produire :

```text id="vy14gr"
project_heading
body_paragraph
ordered_list_item
nested_list_item
code_block
code_line
code_token
diagram_label
```

Pas seulement `phrase`.

---

## 2.4 Page 420 — résumé avec bullets

Points positifs :

```text id="ldq5fu"
- Les bullets sont bien visibles.
- Le flux général de la page est capté.
```

Problèmes :

```text id="j6z6hb"
- Summary est sélectionné deux fois.
- Les bullets sont fusionnés ou coupés par un coalescer générique.
- Plusieurs fragments de phrases naturelles sont protégés à tort.
- Les unités sélectionnées ne portent aucun rôle de list_item.
```

Sur ce type de page, chaque bullet doit devenir :

```json id="m1j36o"
{
  "role": "summary_bullet",
  "marker": "▪",
  "text": "...",
  "translation_strategy": "semantic_reflow",
  "marker_policy": "exact_preserve"
}
```

Le coalescer générique ne doit pas décider seul.

---

## 2.5 Page 11 — About the Authors

Points positifs :

```text id="0c8zqv"
- La page est globalement bien extraite.
- Les biographies sont sélectionnées.
```

Problèmes :

```text id="rkzj69"
- Les noms propres ne sont pas protégés.
- Les rôles author_name / author_bio / affiliation ne sont pas posés.
- Les phrases sont encore trop visuelles, parfois coupées.
- Plusieurs fragments courts comme “He has”, “He was”, “She” deviennent needs_review.
```

La bonne structure serait :

```text id="uwdi28"
page_role = author_bio_page
title = About the Authors
author_entry[]
  author_name
  biography_text
  affiliation
  degrees
```

Les noms :

```text id="e2d0og"
Prof. M. Arif Wani
Dr. Farooq Ahmad Bhat
Dr. Saduf Afzal
Dr. Asif Iqbal Khan
```

doivent être protégés ou annotés comme `person_name`.

---

## 2.6 Page 85 — diagramme d’architecture

Points positifs :

```text id="utou17"
- Le diagramme est repéré.
- Les zones visuelles sont bien visibles.
- Les labels du diagramme sont détectés.
```

Problèmes :

```text id="a6vd8z"
- Les labels courts Input/ReLU/Sigmoid/Softmax sont envoyés au traducteur sans rôle.
- La figure caption est envoyée en bloc complet.
- On ne sait pas si le diagramme doit être traduit ou préservé.
```

Il faut décider une politique claire :

```text id="7tqqi8"
Option A : préserver le diagramme comme image
→ labels non traduits, protected_visual.

Option B : reconstruire le diagramme
→ labels traduisibles, mais avec rôle diagram_label, bbox fixe, contraintes fortes.

Option C : mixte
→ labels techniques verrouillés selon glossaire : ReLU, Softmax, Conv, FC préservés ; Input traduit.
```

Actuellement, la politique est implicite et donc instable.

---

## 2.7 Page 133 — tableau + formules

C’est une page très révélatrice.

Problèmes majeurs :

```text id="b7jyc2"
- Le tableau n’est pas détecté comme table.
- pages_with_table_like_signal est vide.
- Les cellules du tableau sont traitées comme simples phrases.
- Des formules sont partiellement envoyées en traduction.
- Des éléments numériques et mathématiques sont mélangés avec du texte naturel.
```

Exemples sélectionnés :

```text id="kp8xfj"
Total number of images 2346
True positives (TP)
Therefore, Accuracy  88.95%
Precision 
Recall 
```

Il faut distinguer :

```text id="vqa8v6"
table_caption      → traduisible partiellement
table_cell_label   → traduisible
table_cell_number  → exact_preserve
formula_expression → exact_preserve / protected_formula
formula_explanation → traduisible
```

Le système ne peut pas traiter ce type de page avec seulement `phrase`.

---

## 2.8 Page 234 — SQL + note + footer pirate

Points positifs :

```text id="4yts6b"
- Le bloc NOTE est repéré visuellement.
- Les paragraphes principaux sont globalement détectés.
```

Problèmes :

```text id="xz3thm"
- “Estadísticos e-Books & Papers” est envoyé au traducteur.
- Les sorties SQL/table-like ne sont pas structurées.
- Les noms d’entreprise “AGRO Merchants Oakland LLC” sont envoyés au traducteur.
- Les mots SQL COMMIT, UPDATE, START TRANSACTION sont dans le flux sans verrouillage fort.
```

Ici il faut une politique :

```text id="b4h0xg"
publisher_watermark/footer → exclude
company_name               → exact_preserve
sql_keyword                → exact_preserve
paragraph_text             → translate
note_label                 → translate/preserve according style
note_body                  → translate
```

---

## 2.9 Page 418 — tableau de commandes Windows

C’est probablement la page la plus importante après la TOC.

Problèmes majeurs :

```text id="wl35dw"
- Le tableau n’est pas reconnu comme table.
- Les commandes sont envoyées au traducteur : dir, copy, del, findstr.
- Les chemins Windows sont envoyés au traducteur.
- Les cellules sont mélangées : "C:\\Music\\song_favorite.mp3Copy the song.mp3".
- Les descriptions de cellule sont fusionnées ou coupées.
```

Exemples à ne pas envoyer au traducteur comme texte naturel :

```text id="4ov8cn"
dir
copy
del
findstr
copy C:\my-stuff\song.mp3
C:\Music\song_favorite.mp3
findstr "peach" *.txt
```

Ces éléments doivent être `code_token`, `command_name`, `path`, `file_pattern`, `exact_preserve`.

Mais les descriptions doivent être traduites :

```text id="xxgdh2"
Change directory
List directory contents
Copy a file
Delete all files with a .jpg extension...
Search for the text
```

Donc cette page exige une structure `table_cell`.

---

## 2.10 Page 501 — index

Problème principal :

```text id="7h4f2j"
La page est un index, mais elle est classée comme body_text_two_column.
```

Conséquence :

```text id="yxpa6j"
- Des entrées d’index sont fusionnées abusivement.
- Des fonctions techniques sont protégées ou traduites au mauvais niveau.
- Les numéros de page sont mélangés au texte.
- Le footer “Estadísticos e-Books & Papers” est encore envoyé au traducteur.
```

Exemples de fusion incorrecte :

```text id="znanpj"
geometry, 247 displaying version, 243 functions
as delimiter, 26, 43 to redirect output, 311 pivot table.
creating spatial database, 242–243 creating spatial objects, 247 data types, 247
```

Sur un index, il faut des unités comme :

```json id="2zac0u"
{
  "role": "index_entry",
  "head_term": "PostGIS",
  "subentry": "creating spatial database",
  "page_refs": ["242–243"],
  "translatable_text": "creating spatial database",
  "preserve": ["PostGIS", "242–243"]
}
```

Et pour les fonctions SQL :

```text id="1470xq"
ST_AsText()
ST_DWithin()
position()
pg_restore
pg_size_pretty()
```

il faut `exact_preserve`.

---

# 3. Causes racines

## 3.1 `protected_visual` est trop agressif

C’est la cause principale.

Dans `PAGEPRINT`, la logique actuelle a tendance à dire :

```text id="xm53l4"
si ça ressemble à formule/code → protected_visual/background_only
```

Mais elle le fait au mauvais niveau : bloc, ligne, phrase, parfois span.

Exemple critique dans `policy_compiler` :

```python id="zz0bpn"
if re.search(r"\b(def|class|import|return|for|while|if|else|elif|function)\b", text):
    return True
```

C’est trop dangereux.

Un mot comme `if`, `for`, `function` dans une phrase normale ne signifie pas que le texte est du code.

Exemples de faux positifs probables :

```text id="7i18ar"
If there is only time...
The formula is given as...
function of the layer...
for the exercises...
```

Ces textes deviennent `protected_visual`, alors qu’ils doivent être traduits.

### Correction

La détection code doit exiger au moins une preuve forte :

```text id="3shraq"
- police monospace ;
- région code détectée ;
- indentation code ;
- présence de syntaxe dense : (), {}, ;, ==, !=, :=, -> ;
- plusieurs lignes avec structure code ;
- tokens de langage + ponctuation de code ;
- chemin/fichier/commande.
```

Pas seulement un mot-clé isolé.

---

## 3.2 Les régions protégées contaminent les parents

Dans `region_index.py`, une unité est marquée couverte si l’overlap dépasse un seuil. Pour les `protected_visual_region`, le seuil est bas :

```text id="m42yeg"
block/line : 0.35
autres : 0.55
```

Conséquence : une petite région protégée peut contaminer une ligne ou un bloc entier.

C’est ce qui provoque :

```text id="8hd76j"
ligne naturelle → background_only
bloc naturel → background_only
```

### Correction

Il faut distinguer :

```text id="kd79g2"
full_coverage_protected
partial_inline_protection
incidental_overlap
```

Règle recommandée :

```text id="miqdhc"
si protected_region couvre >= 85 % de l’unité :
    l’unité peut devenir background_only

si protected_region couvre 10–85 % :
    ne pas protéger toute l’unité
    créer une translation_protection inline

si protected_region couvre < 10 % :
    ignorer ou signaler incidental_overlap
```

Pour les blocs et lignes, il faut être encore plus strict :

```text id="jnmcwp"
block full protection : overlap >= 0.90
line full protection  : overlap >= 0.85
phrase full protection: overlap >= 0.80
span full protection  : overlap >= 0.75
```

---

## 3.3 Les rôles ne sont pas propagés

Tous les `translation_units` ont :

```text id="1b2g5c"
role: None
```

C’est un problème systémique.

Cela veut dire que les fonctions :

```text id="2de9pp"
page_role detection
page_family detection
layout_type detection
```

existent, mais ne produisent pas encore des rôles exploitables au niveau :

```text id="bhb9hz"
block
line
phrase
span
table_cell
index_entry
figure_caption
diagram_label
```

### Correction

Ajouter une vraie phase :

```text id="ezzkhe"
pageprint/role_resolver.py
```

Elle doit s’exécuter après `unit_factory`, après `region_memberships`, mais avant `policy_compiler`.

Rôles minimaux à produire :

```text id="4wpd9h"
title
section_heading
subsection_heading
body_paragraph
list_item
list_marker
figure_caption
figure_label
diagram_label
table_caption
table_header_cell
table_body_cell
table_numeric_cell
formula_expression
formula_explanation
code_block
code_line
code_token
command_name
path
file_name
url
email
page_header
page_footer
publisher_mark
watermark
author_name
author_bio
index_entry
index_head_term
index_subentry
index_page_reference
toc_entry
toc_page_reference
```

---

## 3.4 `semantic_system` est vide

Sur les 10 pages :

```text id="a5yb20"
semantic_phrases = 0
semantic_groups = 0
```

Donc `PAGETRANSLATE` utilise son `coalescer.py` comme béquille.

Mais le coalescer ne peut pas comprendre :

```text id="t1c04d"
tableaux
index
listes
figures
formules
lignes manquantes
parenthèses techniques
codes
```

### Correction

Déplacer la construction sémantique dans `PAGEPRINT` :

```text id="tq7ht2"
pageprint/semantic_builder.py
```

Il doit produire :

```text id="qjt2ut"
semantic_phrases
semantic_groups
list_items
table_cells
index_entries
figure_caption_parts
code_blocks
```

`PAGETRANSLATE` ne doit coalescer que comme fallback exceptionnel.

---

## 3.5 Les tableaux ne sont pas détectés

`audit_compact.json` dit :

```text id="32166s"
pages_with_table_like_signal: []
```

Alors que les pages 133 et 418 contiennent clairement des tableaux.

C’est un défaut P0.

### Correction

Ajouter dans `PAGEPRINT` :

```text id="jyaecr"
pageprint/table_detector.py
pageprint/table_builder.py
```

Sources à utiliser :

```text id="u3t4ju"
- lignes vectorielles PDF ;
- alignements x/y ;
- fonds alternés gris/blanc ;
- répétition de colonnes ;
- présence de headers ;
- rectangles ou lignes horizontales/verticales ;
- densité de cellules ;
- native PDF words groupés en colonnes.
```

Sortie attendue :

```json id="6zwlqx"
{
  "table_id": "tbl_001",
  "caption": "Table 16-1: Useful Windows Commands",
  "columns": ["Command", "Function", "Example", "Action"],
  "cells": [
    {
      "cell_id": "tbl_001_r2_c1",
      "role": "command_name",
      "text": "copy",
      "translatable": false,
      "preserve_exact_text": true
    },
    {
      "cell_id": "tbl_001_r2_c2",
      "role": "table_cell_text",
      "text": "Copy a file",
      "translatable": true
    }
  ]
}
```

---

## 3.6 L’index n’est pas reconnu

Page 501 est une page d’index, mais le pipeline la traite comme `body_text_two_column`.

### Correction

Ajouter :

```text id="pkvgnr"
pageprint/index_detector.py
pageprint/index_builder.py
```

Signaux :

```text id="f5h6aw"
- beaucoup de lignes courtes ;
- virgule + numéros de page ;
- indentation de sous-entrées ;
- ordre alphabétique ;
- termes techniques suivis de pages ;
- très peu de phrases complètes.
```

Sortie attendue :

```json id="b4abhd"
{
  "page_role": "index",
  "index_entries": [
    {
      "head_term": "PostGIS",
      "page_refs": ["xxviii", "242"],
      "subentries": [
        {
          "text": "creating spatial database",
          "page_refs": ["242–243"]
        }
      ]
    }
  ]
}
```

---

# 4. Corrections sur `PAGEPRINT/`

## 4.1 Ajouter `role_resolver.py`

Emplacement :

```text id="gjn2y1"
pageprint/role_resolver.py
```

Rôle :

```text id="mpdmgj"
prendre units + regions + graph + page_intelligence
et annoter understanding.role / object_type / semantic_kind
```

Pipeline :

```text id="eknf3d"
unit_factory
→ region_index.attach_region_memberships
→ role_resolver.resolve_roles
→ policy_compiler.compile_policies
→ constraint_compiler
→ graph_builder
```

Règle importante : **les rôles doivent être disponibles avant la politique**.

Sinon la politique ne peut pas décider correctement.

---

## 4.2 Modifier `policy_compiler._code_like_text()`

La règle actuelle est trop large.

À remplacer par une logique de score :

```python id="r1z56q"
def _code_like_text(unit, text):
    s = normalize(text)
    style = unit.get("visual", {}).get("style", {})
    role = unit.get("understanding", {}).get("role")
    object_type = unit.get("understanding", {}).get("object_type")

    if role in {"code_block", "code_line", "command_name", "path", "file_name"}:
        return True

    if object_type in {"code", "code_visible", "command", "path"}:
        return True

    monospace = style.get("monospace") or style.get("font_family", "").lower() in {"courier", "consolas"}

    syntax_score = 0
    if re.search(r"[{};]", s): syntax_score += 2
    if re.search(r"(==|!=|:=|=>|->)", s): syntax_score += 2
    if re.search(r"\b(def|class|return|import)\b", s): syntax_score += 1
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\(", s): syntax_score += 2
    if re.search(r"[A-Za-z]:\\|/[\w.-]+/|\*\.\w+", s): syntax_score += 2

    natural_words = len(re.findall(r"[A-Za-z]{3,}", s))
    function_words = len(re.findall(r"\b(the|and|or|to|of|in|for|with|that|this|you|will|can)\b", s, re.I))

    if natural_words >= 6 and function_words >= 2 and not monospace:
        return False

    return monospace and syntax_score >= 1 or syntax_score >= 4
```

Ne jamais faire :

```text id="vvp78z"
if "if" in text → code
```

---

## 4.3 Modifier `policy_compiler._formula_like_text()`

Formule ≠ texte avec parenthèses ou chiffres.

Règle recommandée :

```text id="tdkm21"
formula = vraie structure mathématique
pas simple terme technique
```

Exemples à ne pas classifier formule :

```text id="t6t5st"
(weights)
(3D images)
True positives (TP)
pipe character (|)
Figure 2.1
Table 7.1
```

Exemples à classifier formule :

```text id="govky9"
Accuracy = (TP + TN) / ...
TP / (TP + FP)
Recall = ...
x = y + z
∑, ∫, √, ≤, ≥
```

---

## 4.4 Modifier la politique des régions protégées

Actuellement :

```text id="36jy9v"
protected_visual_region → background_only absolu
```

Il faut introduire trois états :

```text id="ctgbbp"
protected_visual_full
protected_inline_token
protected_region_suspect
```

Exemple :

```json id="t59gzl"
{
  "region_type": "protected_visual_region",
  "coverage_mode": "partial_inline",
  "action": "protect_token_not_parent"
}
```

Puis dans `compile_unit_policy()` :

```python id="cjbdpu"
if protected_overlap >= 0.85:
    background_only
elif protected_overlap >= 0.10:
    add_translation_protection
    keep unit translatable
else:
    ignore
```

---

## 4.5 Ajouter `table_detector.py` et `table_builder.py`

Priorité élevée.

Sans table builder, les pages 133 et 418 resteront impossibles à traduire proprement.

Pour `Table 16-1`, la sortie doit séparer :

```text id="wys3j3"
Command    → exact_preserve / code
Function   → translate
Example    → exact_preserve / code/path
Action     → translate
```

---

## 4.6 Ajouter `index_detector.py`

Priorité élevée pour les livres.

Page 501 doit produire :

```text id="3rk5gr"
page_role = index
layout_type = index_two_column
```

Et interdire le coalescer générique.

---

## 4.7 Ajouter `publisher_mark_detector.py`

Sur les pages SQL, le footer :

```text id="p4bydk"
Estadísticos e-Books & Papers
```

est envoyé au traducteur.

Il doit devenir :

```text id="42lahf"
publisher_mark
translatable = false
translation_strategy = exact_preserve ou exclude
render_policy = fixed_preserve
```

Signaux :

```text id="7nqpin"
- répétition sur plusieurs pages ;
- position bas de page ;
- style couleur/lien ;
- ne fait pas partie du flux de lecture ;
- texte hors marge principale.
```

---

## 4.8 Enrichir `views.translation_units`

Aujourd’hui la vue contient trop peu d’informations.

Elle doit contenir :

```json id="c5irtj"
{
  "unit_id": "...",
  "level": "phrase",
  "text": "...",
  "role": "body_paragraph",
  "object_type": "plain_text",
  "semantic_kind": "prose",
  "bbox": [...],
  "parent_id": "...",
  "source_unit_ids": [...],
  "translation_strategy": "semantic_reflow",
  "render_policy": "anchored_text",
  "protected_tokens": [...],
  "exclusion_reason": null,
  "layout_context": {...}
}
```

Et une vue complémentaire :

```text id="sfekhf"
views.translation_exclusions
```

avec :

```json id="ru6kji"
{
  "unit_id": "...",
  "text": "...",
  "reason": "publisher_mark|code_token|page_reference|formula|protected_visual"
}
```

---

# 5. Corrections sur `PAGETRANSLATE/`

## 5.1 Ne plus dépendre du coalescer générique

Actuellement, 48 unités sur 212 sont des `semantic_phrase` synthétiques créées par `coalescer.py`.

C’est beaucoup trop.

Règle :

```text id="7g6vii"
PAGETRANSLATE ne doit pas inventer massivement la sémantique.
```

Il doit consommer la sémantique produite par `PAGEPRINT`.

Le coalescer doit devenir :

```text id="aluhwu"
fallback limité
```

et non :

```text id="ee8hrz"
moteur principal de segmentation sémantique.
```

---

## 5.2 Coalescer contextuel par type de page

Le coalescer doit être désactivé ou spécialisé selon :

```text id="3sx19i"
toc
index
table
figure
diagram
list
code
author_bio
body_paragraph
```

Règles :

```python id="3uf77z"
if page_role in {"toc", "index"}:
    disable_generic_coalescer()

if page_has_table_region:
    never_coalesce_across_cells()

if role in {"list_item"}:
    coalesce only inside same list item

if role in {"figure_caption"}:
    split label/number from caption, then translate caption

if role in {"code_block", "code_line", "command_name", "path"}:
    do not translate
```

---

## 5.3 Empêcher la fusion à travers des trous

Le coalescer produit des phrases fausses quand une ligne intermédiaire a été protégée à tort.

Il faut refuser la fusion si les unités sources ne sont pas contiguës.

Ajouter :

```text id="bkhzwj"
source_line_index
source_block_id
source_sibling_index
```

Puis :

```python id="t2rstw"
def can_join(prev, curr):
    if curr.block_id != prev.block_id:
        return False
    if curr.line_index != prev.line_index + 1:
        return False
    if exists_skipped_translatable_sibling_between(prev, curr):
        return False
    if prev.role != curr.role:
        return False
    if curr.is_list_start or curr.is_table_cell_start:
        return False
    return True
```

---

## 5.4 Ajouter une logique `caption_splitter`

Pour :

```text id="ycg0oo"
Figure 2.1 Traditional ML algorithms require...
Table 16-1: Useful Windows Commands
Fig. 4.13 Architecture diagram...
```

Il faut produire :

```json id="juvmh9"
{
  "caption_label": "Figure",
  "caption_number": "2.1",
  "caption_text": "Traditional ML algorithms require...",
  "translate": ["caption_text"],
  "preserve": ["caption_label", "caption_number"]
}
```

Selon choix linguistique, `Figure` peut devenir `Figure` en français, mais le numéro reste exact.

---

## 5.5 Ajouter une logique `index_translate_policy`

Pour les index :

```text id="kr7iak"
PostGIS, xxviii, 242
creating spatial database, 242–243
ST_AsText(), 260
```

Politique :

```text id="ur7tiq"
head_term technique connu → preserve
fonction SQL → preserve
page_refs → preserve
subentry naturel → translate
```

Exemple :

```json id="b0l6qp"
{
  "source": "creating spatial database, 242–243",
  "protected": ["242–243"],
  "text_to_translate": "creating spatial database"
}
```

---

## 5.6 Ajouter une logique `table_translate_policy`

Pour les tableaux, ne jamais traduire la ligne brute.

Traduire cellule par cellule :

```text id="hk44ie"
cell role=header_text       → translate
cell role=command_name      → preserve
cell role=code_example      → preserve
cell role=description_text  → translate
cell role=numeric_value     → preserve
```

---

## 5.7 Renforcer `protection.py`

Les placeholders actuels :

```text id="7kar7w"
__PT_0001__
```

sont fragiles.

Utiliser plutôt :

```text id="hvufwh"
⟦PT0001⟧
```

Et restaurer de manière tolérante :

```text id="98t6pl"
⟦ PT0001 ⟧
[[PT0001]]
<nt id="PT0001"/>
```

Ajouter protections pour :

```text id="yhffvb"
person_name
organization_name
acronym
model_name
library_name
command_name
sql_keyword
file_path
function_call
page_reference
equation_number
figure_number
table_number
```

---

## 5.8 La projection doit conserver les rôles

Dans les résultats actuels :

```text id="ck24rb"
reconstruction_units.role = None
```

Il faut que `projection.py` transmette :

```text id="jrloqa"
role
object_type
semantic_kind
page_role
page_family
source_unit_ids
layout_budget
render_contract
```

Sinon le reconstructeur ne pourra pas choisir le bon mode de rendu.

---

# 6. Corrections au-delà de `PAGEPRINT` et `PAGETRANSLATE`

## 6.1 `special_region_detector`

Il doit produire moins de faux positifs.

À ajouter :

```text id="46hzi9"
- confidence
- reason
- class_source
- text_overlap_summary
- protected_mode: full_visual | inline_token | suspicious
```

Un détecteur externe ne doit pas pouvoir imposer directement :

```text id="ywx79f"
background_only
```

sans validation par `PAGEPRINT`.

---

## 6.2 `native_pdf_extractor`

À renforcer pour :

```text id="3kf7v8"
- table lines ;
- native text spans ;
- fonts monospace ;
- figure/table captions ;
- index pages ;
- page headers/footers ;
- repeated publisher marks.
```

Pour les PDFs natifs, il faut exploiter les coordonnées de mots PDF beaucoup plus que l’OCR.

---

## 6.3 `page_policy_matrix`

À enrichir avec des politiques par page type :

```text id="tnmbr2"
toc
index
author_bio
summary
table_page
code_page
diagram_page
formula_page
body_text
publisher_page
```

Chaque page type doit avoir :

```text id="2dzlwz"
allowed_translation_roles
forbidden_translation_roles
coalescing_policy
protected_visual_policy
table_policy
index_policy
```

---

## 6.4 `page_extraction_postprocessors`

À utiliser pour :

```text id="pfpxwr"
- césures : hyperparam- eters → hyperparameters
- mots coupés : con- tains → contains
- colonnes : ne pas fusionner à travers colonne/table
- bullets : séparer marker et texte
- captions : split label/number/text
```

---

## 6.5 Audit fonctionnel

L’audit actuel est utile visuellement, mais il doit devenir plus sévère.

Ajouter dans `audit_compact.json` :

```json id="gfcos7"
{
  "schema_valid": true,
  "functional_valid": false,
  "critical_counts": {
    "role_none_translation_units": 212,
    "semantic_system_empty": 10,
    "natural_text_marked_protected": 158,
    "generic_coalesced_units": 48,
    "table_false_negative_pages": 2,
    "index_false_negative_pages": 1,
    "publisher_mark_sent_to_translation": 3,
    "code_or_command_sent_to_translation": 12
  }
}
```

Et ne plus afficher seulement :

```text id="onrj7r"
status: ok
```

mais :

```text id="p1iq4g"
schema_status: ok
functional_status: ko
```

---

# 7. Tests à ajouter immédiatement

## Tests PAGEPRINT

```text id="3rvkjc"
test_code_keyword_in_prose_not_protected
test_if_for_function_words_do_not_trigger_code
test_formula_text_parentheses_not_protected
test_partial_protected_region_does_not_protect_parent_line
test_table_grid_detected_from_vector_lines
test_command_table_cells_roles
test_index_page_detected
test_publisher_mark_excluded
test_author_names_detected
test_figure_caption_split
test_bullet_marker_split_from_text
test_semantic_system_not_empty_for_body_page
```

## Tests PAGETRANSLATE

```text id="axqfae"
test_no_role_none_translation_units
test_no_generic_coalescing_on_index
test_no_generic_coalescing_on_table
test_no_coalescing_across_skipped_line
test_caption_label_number_preserved
test_sql_command_preserved
test_file_path_preserved
test_index_page_refs_preserved
test_table_descriptions_translated_but_commands_preserved
test_reconstruction_units_keep_roles
```

## Tests audit

```text id="5mesr0"
test_functional_status_ko_when_roles_missing
test_functional_status_ko_when_semantic_system_empty
test_functional_status_ko_when_table_false_negative
test_functional_status_ko_when_natural_text_protected
```

---

# 8. Priorités de développement

## P0 — À corriger avant vrai test moteur

```text id="gz97wc"
1. Réduire drastiquement les faux protected_visual.
2. Corriger _code_like_text et _formula_like_text.
3. Empêcher une région protégée partielle de contaminer bloc/ligne.
4. Ajouter un role_resolver.
5. Ajouter semantic_builder côté PAGEPRINT.
6. Ajouter table_detector/table_builder.
7. Ajouter index_detector/index_builder.
8. Désactiver le coalescer générique sur table/index/toc/list.
9. Séparer captions : label/number/text.
10. Exclure publisher marks.
```

## P1 — Ensuite

```text id="cwwrn1"
11. Ajouter protection robuste des noms propres, acronymes, commandes, chemins.
12. Ajouter politiques spécialisées diagram_label.
13. Ajouter projection enrichie avec rôles.
14. Ajouter audit fonctionnel sévère.
15. Ajouter corpus de tests golden sur ces 10 pages.
```

---

# 9. Peut-on avancer sur le moteur de traduction ?

## Pour validation globale : non

Pas encore.

Raison : les unités envoyées au moteur ne sont pas encore assez propres.

Exemples qui pollueraient le test :

```text id="g9rbdt"
Figure 2.1
CHAPTER 4
Estadísticos e-Books & Papers
dir
copy
del
findstr
C:\Music\song_favorite.mp3Copy the song.mp3
geometry, 247 displaying version, 243 functions
```

Si le moteur traduit mal ces unités, ce ne sera pas forcément sa faute : elles n’auraient jamais dû lui être envoyées telles quelles.

## Pour développement parallèle : oui

On peut développer le moteur sur un corpus propre séparé :

```text id="9ju1fm"
- phrases manuelles propres ;
- paragraphes extraits validés ;
- captions splittées ;
- descriptions de table propres ;
- items de liste propres ;
- termes techniques avec glossaire.
```

Mais il faut le faire comme test moteur isolé, pas comme validation du pipeline complet.

---

# Conclusion

L’audit 10 pages est très utile parce qu’il révèle les défauts structurels récurrents.

Le pipeline est maintenant assez bon pour montrer les erreurs, mais pas encore assez bon pour valider la traduction.

Le prochain jalon ne doit pas être :

```text id="7vi8c8"
tester le moteur de traduction sur tout
```

Le prochain jalon doit être :

```text id="xn9o6n"
rendre PAGEPRINT capable de produire des rôles et unités sémantiques fiables,
puis rendre PAGETRANSLATE consommateur de ces unités, pas inventeur.
```

Critère de passage au moteur :

```text id="vr9rwv"
role_none_translation_units = 0
semantic_system_empty = false
natural_text_marked_protected ≈ 0
table/index pages reconnues
code/command/path/page refs exclus ou protégés
generic_coalesced_units très faible
publisher marks exclus
reconstruction_units avec rôles
```

Sur l’état actuel, on n’y est pas encore. Le système est sur la bonne trajectoire, mais il faut corriger la couche de compréhension avant d’évaluer la qualité linguistique.

