# Analyse `results/recon5f/` — causes fondamentales (implémentation + politique)

> Comparaison `pagereconstruct/` (package modulaire) vs `reconstructor.py` (monolithe legacy, 8645 lignes).
> Données : les 5 pages de `results/recon5f/` (audits + plans + render_ops).
> Date : 2026-06-16.

---

## 1. Résultat observé : 4 pages sur 5 en `ko`, toutes pour la même raison

| Page | status | pub_ready | score | `text_protected_overlap` ko | `text_text_overlap` ko | hard_blocker |
|------|--------|-----------|-------|------------------------------|------------------------|--------------|
| Advances p0002 | **ok** | ✅ true | 1.0 | 0 | 0 | — |
| Advances p0026 | ko | false | 0.6 | ~58 | 5 | `collision_ko` |
| Practical SQL p0046 | ko | false | 0.6 | 47 | 4 | `collision_ko` |
| docintelligence p0065 | ko | false | 0.6 | 50 | 4 | `collision_ko` |
| docintelligence p0414 | ko | false | 0.6 | 47 | 4 | `collision_ko` |

Le blocage est **monocause** : `collision_ko` déclenché par `text_protected_overlap`,
massivement à **ratio = 1.0**. Tout le reste est sain (`unresolved_style = 0`,
`overflow = 0`, `patch_protected_overlap = 0`, `source_text_leak_risk = low`,
`missing_background = false`, `text_presence = 1.0`).

La seule page `ok` (p0002) est une page quasi vide (cleanbg de 7 Ko) : elle passe parce
qu'elle ne contient ni bloc de code, ni page_number/publisher_mark préservé — donc aucune
des deux situations décrites ci-dessous ne se produit.

---

## 2. Le paradoxe central

`visual_qa` produit un score `overlap = 0.0` (catastrophique) sur les pages `ko`, alors que
tous les autres scores visuels valent 1.0 et que **le rendu est correct** : le texte est
placé là où il doit l'être, aucune fuite de texte source, typographie fidèle.

Autrement dit : **la métrique de collision pénalise un recouvrement qui est voulu par
conception, pas un défaut.** Ce n'est pas un bug de placement — c'est un bug de *politique
de mesure*. C'est le point fondamental.

---

## 3. Preuve géométrique (décisive)

### Classe A — contenu préservé rendu PAR-DESSUS sa propre protection

`p0046`, unité `ru_0017` (rôle `publisher_mark`) :

```
texte rendu (TextOp) : [227.2, 768.9, 332.0, 775.9]
région protégée      : [227.2, 768.9, 384.8, 785.0]  (publisher_mark)
→ ratio = 1.0  → severity = ko
```

`p0026`, unité `ru_0001` (rôle `page_number`) :

```
texte rendu : [53.6, 35.8, 62.5, 42.8]
protégé     : [53.6, 35.x, ...]  (page_number, preservation_plan)
→ ratio = 0.95 → ko
```

La même entité (`page_number`, `publisher_mark`, `numeric`, `formula`) est **à la fois** :
- ajoutée à l'index des régions protégées (`preservation_plan` / `exclusion_plan` / `unit_policy`), **et**
- émise comme `TextOp` rendu (la préservation/ré-insertion dessine le contenu exactement à sa place).

Le détecteur voit donc le TextOp légitime tomber à 95–100 % sur « sa » zone protégée → `ko`.

### Classe B — double classification d'une même ligne physique (code vs prose)

`p0046`, bloc de code à `y ∈ [543, 638]`. Pour **chaque ligne physique** du bloc, l'amont
émet DEUX objets :

```
région protégée code_line : [102.0, 543.7, 539.6, 558.7]
+ TextOp body_paragraph    : [102.0, 543.7, 329.2, 550.7]  (ru_0034)
→ ratio = 1.0 → ko
```

`p0026` : **17 régions `code_line` + 1 `code_block`**, et des `body_paragraph` (ex. `ru_0042`,
largeur 7 pt = un seul glyphe/nombre) atterrissent dessus à ratio 1.0.

Les bandes `y` des `code_line` protégés coïncident **exactement** avec les `y` des lignes
`body_paragraph` rendues : ce sont **les mêmes lignes physiques**, vues deux fois — une fois
comme « code » (→ protégée), une fois comme « paragraphe » (→ traduite/rendue).

---

## 4. Causes fondamentales — IMPLÉMENTATION

### C1. Le détecteur de collision n'a aucune notion d'« occupant légitime »
`collision_detector.detect_protected_collisions` (lignes 44-75) compare **toutes** les
boîtes de texte contre **toutes** les régions protégées, purement géométriquement. Aucun
lien `unit_id → région protégée` n'existe pour exempter le TextOp qui *est censé* remplir
cette zone (préservation, overlay exact). Conséquence : préserver un page_number/formule/logo
et le ré-insérer produit **structurellement** une collision `ko`. La préservation est donc
auto-incompatible avec la porte de publication.

### C2. La déduplication de cohérence est par identifiant, la collision est géométrique
`build_protected_region_index` (protected_region_index.py:78-166) applique l'invariant
« une unité traduite ne doit jamais être aussi protégée » via
`u.get("unit_id") in translated_source_ids` et `translated_source_ids & source_unit_ids`.
Mais la duplication réelle se fait sur **des `unit_id` DIFFÉRENTS qui partagent la même
géométrie/contenu** (un `code_line` détecté par le détecteur de régions ≠ le `body_paragraph`
construit par le semantic builder, même s'ils recouvrent la même ligne). Un dédoublonnage
par ID ne peut pas attraper une duplication par géométrie. L'invariant est correct sur le
papier mais inopérant sur le cas réel.

### C3. Multiplicité des régions protégées → inflation des findings
Les régions sont ajoutées depuis plusieurs sources (`special_zone`, `preservation_plan`,
`exclusion_plan`, `unit_policy`, `regions`) **sans fusion**. Exemples observés :
- `p0046` : `publisher_mark` présent 4× (exclusion_plan ×2 + unit_policy ×2), `code_line` en
  paires quasi identiques (`[102,543.7,…]` et `[102,543,…]`).
- `p0026` : 17 `code_line` pour un seul `code_block`.

Une seule unité de texte qui touche ce bloc génère donc jusqu'à 17 findings → les « 47-70 KO »
sont en partie un artefact de comptage, pas 47 zones distinctes.

### C4. Dénominateurs de ratio incohérents
`text_text` utilise `intersection / min(aireA, aireB)` (`_ratio_min`), `text_protected` utilise
`intersection / aire(texte)`. Deux conventions différentes dans le même module → seuils
(`0.02/0.10` vs `0.01/0.10`) non comparables et difficiles à raisonner.

### C5. Seuil `ko` géométrique inatteignable pour le contenu préservé
`ko = 0.10` sur géométrie brute : dès qu'un overlay préservé est à sa place (ratio ~1.0), le
seuil est dépassé de toute façon. Régler le seuil ne sauve rien tant que C1 n'est pas réglé.

### C6. Le `PlacementSolver` (Phase 4) ne peut pas résoudre ces cas
Le contrat prévoit un solveur anti-collision *avant* rendu (« formula/code preserve »). Mais :
- il ne peut pas déplacer un overlay préservé (page_number/publisher_mark) — il **doit** rester là ;
- il ne peut pas déplacer un `body_paragraph` qui *est* le contenu du code.

Le solveur ne peut donc rien faire : la collision est en amont, dans la **classification**, pas
dans le placement.

---

## 5. Causes fondamentales — POLITIQUE du pipeline

### P1. Confusion entre « préserver » et « protéger contre l'écrasement »
Le pipeline traite « garder les pixels/texte d'origine » et « interdire d'écrire du texte
traduit ici » comme la même région protégée. Or l'overlay préservé **écrit** dans cette
région — c'est l'objectif. Tant que ces deux concepts partagent la même liste
`protected_regions` sans provenance, la préservation déclenchera toujours `collision_ko`.

### P2. Une ligne physique reçoit deux identités de rôle (amont)
À l'intérieur d'un bloc de code, chaque ligne est étiquetée à la fois `code_line` (détecteur
de régions / special zone) **et** `body_paragraph` (semantic builder). Les deux entrent dans
des couches différentes (protégée vs traduite) avec des `unit_id` distincts. C'est une
**duplication de classification en amont** que `pagereconstruct` hérite et matérialise au
rendu. La sur-détection formule/protégé corrigée au commit `15c5389` est le même symptôme,
pas encore éteint pour le code.

### P3. La porte de publication a une cible auto-contradictoire
`PLAN_RECONSTRUCT_CONTRAT.md` Phase 7/9 exige simultanément :
- `PreservationOp réellement exécuté` (overlays page_number/formula/logo **réinsérés**), **et**
- `0 text_protected ko` + `overlap ≥ 0.99`.

Avec l'implémentation actuelle (C1), ces deux exigences ne peuvent **jamais** être vraies en
même temps sur une page contenant un overlay préservé ou un bloc de code. La cible est
mathématiquement inatteignable.

### P4. Le score visuel et le détecteur géométrique mesurent la mauvaise chose
`visual_qa.overlap = 1 − max_text_text − max_text_protected` mélange un vrai défaut
(text/text) avec un faux défaut (text/protected-légitime). Un seul overlay préservé écrase le
score à 0 et bloque une page par ailleurs parfaite. La métrique ne reflète pas la qualité
perçue (cf. §2).

---

## 6. Comparaison avec `reconstructor.py` (legacy)

- Le monolithe (8645 l.) **dessinait** le texte sans pipeline de protection/collision : pas de
  `collision_ko`, donc « passait » toujours — au prix de fuites de texte source et de
  recouvrements réels non détectés. Il n'avait simplement pas de garde-fou.
- `pagereconstruct/` a le bon modèle (couches, protections, gates, QA). Le problème n'est
  **pas** l'architecture — elle est nettement supérieure — mais **deux raccords manquants** :
  1. la provenance `occupant légitime ↔ région protégée` (C1/P1) ;
  2. la réconciliation géométrique des classifications dupliquées (C2/P2).
- Conclusion : ne **pas** revenir au legacy. Le legacy ne « marche » qu'en supprimant le
  garde-fou qui, ici, sur-déclenche.

---

## 7. Recommandations (par ordre de levier)

### R1 — (priorité 1, débloque tout) Exempter l'occupant légitime de sa propre protection
Lier chaque `ProtectedRegion` à l'`unit_id` source qui la génère, puis, dans
`detect_protected_collisions`, **ignorer** une collision quand le TextOp testé est l'occupant
attendu de cette région (préservation/overlay exact, ou même `source_unit_id`). C'est le
correctif minimal qui rétablit la cohérence préservation ↔ gate (P1/C1).

### R2 — (priorité 1) Réconcilier les classifications dupliquées par géométrie
Avant de construire l'index : si une ligne est déjà consommée comme `body_paragraph`
(couche traduite), **ne pas** la protéger comme `code_line`, et inversement — décider sur
**recouvrement géométrique**, pas sur identité d'`unit_id` (C2/P2). Une ligne = une seule
couche.

### R3 — (priorité 2) Fusionner/dédupliquer les régions protégées
Fusionner les bboxes quasi identiques et regrouper les `code_line` sous leur `code_block`
(une zone, pas 17). Élimine l'inflation des findings (C3).

### R4 — (priorité 2) Séparer les deux concepts en politique
Deux listes distinctes : `keep_regions` (préserver/écrire l'original — **exempt** de collision)
vs `forbid_text_regions` (interdire le texte traduit — testé). La porte de publication ne teste
que la seconde (P3/P4).

### R5 — (priorité 3) Harmoniser les dénominateurs de ratio (C4) et réévaluer le seuil après R1.

---

## 8. Questions clés à trancher

1. **L'amont** doit-il émettre une ligne de code comme `code_line` OU comme `body_paragraph` —
   jamais les deux ? (Sinon `pagereconstruct` doit arbitrer par géométrie : R2.)
2. Un overlay préservé (page_number, publisher_mark) est-il un *défaut de collision* ou un
   *contenu attendu* ? (La réponse impose R1 + R4.)
3. La cible « 0 text_protected ko » du contrat doit-elle exclure les occupants légitimes ?
   (Sinon elle est inatteignable : P3.)
4. Le score `overlap` doit-il continuer à fusionner text/text (vrai défaut) et text/protected
   (souvent faux défaut) ? (P4.)
5. Pourquoi p0002 passe-t-elle : confirme-t-on que c'est l'**absence** de code/overlay préservé,
   et non un seuil heureux ?

---

## 9. Recommandation finale

**Direction** : garder `pagereconstruct/`, appliquer R1 puis R2. Ce sont deux raccords
chirurgicaux (provenance + arbitrage géométrique), pas une refonte. Ils transforment
`collision_ko` de « toujours déclenché » en « déclenché uniquement sur vrai recouvrement ».

**Signal d'erreur de cap** : si, après R1+R2, des `text_protected_overlap ko` subsistent **avec
des bboxes qui ne coïncident PAS** avec un overlay préservé ni avec une ligne déjà rendue —
alors le problème serait un vrai défaut de placement (ressort du `PlacementSolver`), et la
lecture ci-dessus serait incomplète.

**Prochaine action concrète** : ajouter le lien `unit_id`/`source_unit_id` sur `ProtectedRegion`
et l'exemption d'occupant dans `detect_protected_collisions` (R1), puis rejouer les 5 pages de
`recon5f`. Attendu : p0026/p0046/p0065/p0414 passent de `ko` à `ok`/`review`, sans toucher au
placement ni au rendu.
