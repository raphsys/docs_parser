# 6. Reprendre l’ancienne formule correctement

Tu as parfaitement résumé la vieille logique. Il faut la transformer en pipeline moderne.

## Étape ancienne 1 — trame de fond

Ancienne formule :

```text
Créer et utiliser la trame de fond comme page de base.
```

Version moderne :

```text
BackgroundContract
TextRemovalLedger
BackgroundAudit
BackgroundOp
```

Règle :

```text
Le fond propre est la base.
Pas de fond propre = pas publication-ready.
```

---

## Étape ancienne 2 — positionnement des grands blocs

Ancienne formule :

```text
Positionnement des blocks paragraphes ou grandes zones.
```

Version moderne :

```text
BlockReconstructionContract
FlowRegion
MultiBlockPlacementSolver
BlockLayoutAudit
```

Règle :

```text
On place les grands blocs avant les lignes/mots.
```

C’est précisément ce qu’il faut rétablir. Trop de problèmes viennent du fait que le moteur pense encore parfois en unités fines.

---

## Étape ancienne 3 — positionnement intra-block

Ancienne formule :

```text
Positionnement intelligent des textes traduits et non traduits dans le bloc.
```

Version moderne :

```text
IntraBlockComposer
InlineObjectPlanner
TypographyEngine
LineBreaker
TextFitter
```

Ici, il faut un module clair :

```text
pagereconstruct/composition/intrablock_composer.py
```

Il reçoit :

```text
BlockReconstructionContract
translated text
inline preserved objects
style runs
layout constraints
```

Il sort :

```text
LineLayout[]
InlineObjectPlacement[]
TextRunPlacement[]
```

C’est plus puissant qu’un simple renderer.

---

## Étape ancienne 4 — zones spéciales non traduites

Ancienne formule :

```text
Positionnement des zones spéciales selon la compréhension du bloc/page.
```

Version moderne :

```text
PreservationContract
InlinePreservationContract
ObjectContract
PreservationOps
PreservationAudit
```

Point critique : une formule inline n’est pas juste un crop. Elle appartient à une ligne, dans un bloc, avec un ancrage.

Donc il faut distinguer :

```text
block-level preservation
inline preservation
page-level preservation
background-level preservation
```

---

## Étape ancienne 5 — validation visuelle et amélioration

Ancienne formule :

```text
Validation visuelle par rapport à l’original et aux contrats, amélioration si nécessaire.
```

Version moderne :

```text
PublicationReadyEvaluator
AutoCorrectionLoop
CandidateEngine
VisualImageAudit
```

Ce n’est pas une QA finale seulement. C’est une boucle :

```text
composer produit candidat
audit vérifie
solver corrige
rendu candidat suivant
audit revérifie
```

---

## Étape ancienne 6 — rendu

Ancienne formule :

```text
Rendu.
```

Version moderne :

```text
RenderOps
PDFVectorBackend
RasterDebugBackend
FinalRenderAudit
```

Règle :

```text
Le backend ne décide rien.
Il exécute.
```

---

# 7. Le vrai chaînage cible

Voici la chaîne que je recommande maintenant :

```text
PAGEPRINT
  ↓
PAGETRANSLATE
  ↓
FinalReconstructionContractBuilder
  ↓
PublicationReadyEvaluator.preflight()
  ├── pageprint_audit
  ├── pagetranslate_audit
  ├── contract_audit
  └── background_audit préliminaire
  ↓
LegacyStyleReconstructionPlanner
  ├── background planner
  ├── block planner
  ├── intrablock composer
  ├── preservation planner
  └── typography planner
  ↓
RenderCandidateSet
  ↓
PublicationReadyEvaluator.candidate_audit()
  ↓
AutoCorrectionLoop
  ↓
RenderOps
  ↓
PDF/PNG render
  ↓
PublicationReadyEvaluator.final_visual_audit()
  ↓
PagePublicationReadyReport
  ↓
DocumentPublicationReadyReport
```

---

# 8. Pourquoi le système actuel dévie encore

Actuellement, on a bien certains éléments, mais pas cette hiérarchie.

Le système fait encore trop souvent :

```text
bloc → renderer → TextOp
```

Alors que la bonne formule doit être :

```text
bloc → composition interne → lignes/runs/objets inline → candidats → audit → TextOps
```

Autrement dit : **le renderer arrive trop tôt**.

Il faut ajouter une couche avant renderer :

```text
IntraBlockComposer
```

Le renderer ne doit pas composer. Il doit dessiner une composition déjà validée.

---

# 9. Plan d’implémentation opérationnel

## Phase A — créer l’outil publication-ready

Créer :

```text
pubready/
```

Tâches :

```text
[ ] schema.py avec PagePublicationReadyReport et DocumentPublicationReadyReport
[ ] evaluator.py avec PublicationReadyEvaluator
[ ] page_auditor.py
[ ] document_auditor.py
[ ] gates.py
[ ] scoring.py
[ ] reports.py
[ ] evidence.py
```

Sous-modules :

```text
[ ] stages/pageprint_audit.py
[ ] stages/pagetranslate_audit.py
[ ] stages/contract_audit.py
[ ] stages/background_audit.py
[ ] stages/block_layout_audit.py
[ ] stages/intrablock_audit.py
[ ] stages/preservation_audit.py
[ ] stages/typography_audit.py
[ ] stages/render_ops_audit.py
[ ] stages/visual_image_audit.py
```

---

## Phase B — intégrer l’audit à chaque étape

Dans le pipeline :

```text
après pageprint → audit pageprint
après pagetranslate → audit pagetranslate
après contract build → audit contract
après background clean → audit background
après block planning → audit block layout
après intrablock composition → audit intrablock
après render ops → audit ops
après rendu → audit image final
```

Règle :

```text
Si un audit critique échoue :
    stop publication
    produire report + correction suggestions
```

---

## Phase C — reprendre exactement l’ancienne formule de reconstruction

Créer :

```text
pagereconstruct/composition/
├── block_planner.py
├── intrablock_composer.py
├── inline_object_planner.py
├── line_layout_engine.py
├── block_fit_solver.py
└── legacy_formula_adapter.py
```

Le `legacy_formula_adapter.py` doit reprendre la logique ancienne :

```text
fond
grands blocs
éléments intra-blocs
zones non traduites
validation
rendu
```

Mais avec les données nouvelles.

---

## Phase D — ajouter la boucle correction/amélioration

Créer :

```text
pagereconstruct/autocorrect/
├── correction_loop.py
├── correction_plan.py
├── retry_policy.py
├── block_adjustments.py
├── typography_adjustments.py
├── background_adjustments.py
└── placement_adjustments.py
```

Boucle :

```text
candidate render
audit
if fail:
    generate correction plan
    apply correction
    rerender
max_iter = 3
```

---

# 10. Le point de décision

Il faut arrêter de demander au renderer de sauver la page.

Le renderer doit être le dernier maillon.

La reconstruction publication-ready doit être gagnée avant le rendu :

```text
contrat propre
fond propre
blocs propres
composition intra-bloc propre
objets spéciaux propres
typographie prête
ops validées
```

Ensuite seulement :

```text
rendu
```

---

# 11. Réponse directe à ta question

## Qu’est-ce qui est fait maintenant ?

Il y a déjà :

```text
contrat final
ops
background cleaner
overlay manager
candidate engine
placement solver
visual qa
backend ops-only
```

C’est bien. Mais ce n’est pas encore la vieille formule complète.

## Pourquoi ne pas juste améliorer l’ancienne formule ?

C’est exactement ce qu’il faut faire.

Mais pas en copiant le monolithe. Il faut formaliser l’ancienne formule en modules :

```text
BackgroundPlanner
BlockPlanner
IntraBlockComposer
SpecialZonePreserver
CandidateValidator
PublicationReadyEvaluator
RenderOpsExecutor
```

## Le vrai manque actuel

Il manque trois pièces centrales :

```text
1. PublicationReadyEvaluator transversal.
2. IntraBlockComposer héritier de l’ancien pipeline.
3. AutoCorrectionLoop pilotée par l’audit visuel.
```

Les deux verrous précédents — typographie OCR et placement multi-blocs — doivent être intégrés dans cette formule, pas placés à côté.

---

# 12. Directive finale

La bonne directive à Claude est :

```text
Ne cherche plus à améliorer seulement VisualQA, renderer ou solver.

Reprends la formule de l’ancien pipeline :

1. fond propre ;
2. placement des grands blocs ;
3. composition intra-bloc intelligente ;
4. placement/protection des zones spéciales ;
5. validation publication-ready multi-étapes ;
6. correction automatique ;
7. rendu final.

Implémente un vrai outil pubready/ qui audite chaque étape :
pageprint, pagetranslate, contrat, background, blocs, intra-blocs,
préservation, typographie, render ops, image finale.

Le score publication-ready doit être calculé par page, puis consolidé au niveau document.
Une page ne peut pas être OK si une étape critique échoue.
Un document ne peut pas être OK si une page critique échoue.

Objectif :
ancienne formule robuste + nouvelles données pageprint/pagetranslate + contrat pagereconstruct.
```

C’est cela qui peut réellement amener vers 95 %.

