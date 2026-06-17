# Plan de correction — `collision_ko` faux positif

> Référence cause : `docs/ANALYSE_RECON5F_CAUSES_FONDAMENTALES.md`.
> Objectif : `collision_ko` ne se déclenche que sur **vrai** recouvrement. Pas de refonte.

## Ordre d'exécution (dépendances)

```
M1.1 ─► M1.2 ─────────────┐
M2.1 ─────────────────────┤
M3.1 ─────────────────────┼─► M5.1 (replay recon5f)
M4.1 (dépend M1.2) ───────┤
M4.2 (dépend M1.2,M4.1) ──┘
```

## Mission 1 — Débloquant : exempter l'occupant légitime (R1)
Cible : préservation/overlay rendu sur sa propre zone ≠ collision.
- **M1.1** `schema.py`/`build_protected_region_index` : champ `owner_unit_id` sur `ProtectedRegion`.
- **M1.2** `collision_detector.detect_protected_collisions` + `visual_qa.assess` : ignorer collision si TextOp = occupant attendu (même owner / op préservation).

## Mission 2 — Débloquant : une ligne = une couche (R2)
- **M2.1** Arbitrage géométrique code/prose dans l'index : ne pas protéger `code_line` déjà consommé comme `body_paragraph` (et inverse). Décision par recouvrement, pas par `unit_id`.

## Mission 3 — Nettoyage findings (R3)
- **M3.1** Fusion bboxes quasi identiques + regrouper `code_line` sous `code_block` (1 zone, pas 17).

## Mission 4 — Politique de mesure (R4/R5)
- **M4.1** Séparer `keep_regions` (écrire l'original, exempt) vs `forbid_text_regions` (testé). Gate ne teste que `forbid`.
- **M4.2** Unifier dénominateurs de ratio ; `visual_qa.overlap` cesse de fusionner le faux défaut ; réévaluer seuils.

## Mission 5 — Validation
- **M5.1** Tests d'invariant + replay recon5f. Attendu : p0026/p0046/p0065/p0414 `ko→ok/review`, p0002 reste `ok`, placement/rendu inchangés.

## Critère de succès / signal d'échec
- ✅ 4 pages débloquées sans toucher au placement.
- ⚠️ Si `text_protected ko` résiduel **avec bbox ne coïncidant pas** avec un overlay/ligne légitime → vrai défaut placement → ressort du `PlacementSolver` (Phase 4), hors périmètre de ce plan.
