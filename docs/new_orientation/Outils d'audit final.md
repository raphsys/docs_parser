Oui. Là, tu touches le bon noyau.

Il faut arrêter de penser uniquement en termes de :

```text
VisualQA
validator
publication_ready_score
```

Il faut créer un **vrai outil transversal de calcul publication-ready**, qui audite **tout le pipeline**, page par page, puis document complet.

Et oui : la bonne formule n’est pas de réinventer encore un moteur. La bonne formule est exactement ce que tu viens de dire :

```text
ancienne formule robuste
+
nouvelles données pageprint/pagetranslate
+
contrat final pagereconstruct
+
auditeur publication-ready transversal
```

L’ancien moteur avait déjà les bons concepts : contrat de bloc, opérations de rendu, candidats, placement, fond propre, overlays immuables et text-rescue. C’est confirmé dans le récap : l’ancien moteur avait déjà l’architecture cible avec `BlockReconstructionPlan`, `BlockRenderOp/DrawOp`, `RenderCandidate`, `CandidateScore`, `PlacementResult`, fond propre, overlays immuables et text-rescue. 

Donc la réponse nette est : **il faut revenir à l’ancienne formule, mais la rendre plus stricte, plus contractuelle, plus mesurable et mieux alimentée par `pageprint/` et `pagetranslate/`.**

---

# 1. Ce qui est fait maintenant

Actuellement, d’après le récap, les grosses briques existent déjà :

```text
FinalReconstructionContract
LegacyContractBridge
BlockReconstructionContract
BackgroundContract
PreservationContract
RenderOps
CandidateEngine
PlacementSolver
PDFVectorBackend ops-only
VisualQA
validator durci
background_cleaner
overlay_manager
```

Le score est passé de `0.629` à `0.779`, avec 1 page OK, 6 review, 3 KO, et le leak texte source high est passé de 10/10 à 0/10. 

Donc le système a progressé. Mais ce qui est fait maintenant reste encore trop orienté :

```text
contrat → render ops → rendu → visual_qa
```

Alors que la bonne formule devrait être :

```text
contrat → pré-audit → rendu candidat → audit → correction → rendu final → audit final
```

C’est cela qui manque.

---

# 2. Pourquoi ne pas juste améliorer l’ancienne formule ?

On doit le faire.

Mais il faut comprendre pourquoi l’ancienne formule ne peut pas être reprise brute.

L’ancien pipeline faisait déjà :

```text
1. trame de fond ;
2. placement des grands blocs ;
3. placement intelligent intra-bloc ;
4. zones spéciales non traduites ;
5. validation visuelle ;
6. rendu.
```

C’était la bonne philosophie.

Mais aujourd’hui, on a des données plus riches :

```text
pageprint:
  unités hiérarchisées
  régions
  graphes
  politiques
  contraintes
  styles
  visual_layers
  reconstruction_constraints

pagetranslate:
  unités traduites
  segments protégés
  tokens non traduits
  expansion ratio
  statuts linguistiques
  reconstruction_units

pagereconstruct:
  contrats
  render ops
  preservation ops
  candidate engine
  backend PDF/PNG
```

Donc il faut refaire l’ancienne formule, mais sous forme de **pipeline contractuel mesuré**.

La bonne architecture devient :

```text
PAGEPRINT
  ↓
PAGETRANSLATE
  ↓
FinalReconstructionContract
  ↓
PublicationReadyEvaluator.preflight()
  ↓
LegacyStyle Reconstruction Engine
  ↓
Candidate Render Loop
  ↓
PublicationReadyEvaluator.page_audit()
  ↓
AutoCorrection Loop
  ↓
Final Render
  ↓
PublicationReadyEvaluator.final_audit()
  ↓
DocumentReadyReport
```

---

# 3. Le vrai outil à créer : `pubready/`

Il ne faut pas mettre tout cela dans `visual_qa.py`.

Il faut créer une unité dédiée :

```text
pubready/
```

ou :

```text
quality/publication_ready/
```

Je propose :

```text
pubready/
├── __init__.py
├── schema.py
├── evaluator.py
├── page_auditor.py
├── document_auditor.py
├── scoring.py
├── gates.py
├── reports.py
├── evidence.py
├── stages/
│   ├── pageprint_audit.py
│   ├── pagetranslate_audit.py
│   ├── contract_audit.py
│   ├── background_audit.py
│   ├── block_layout_audit.py
│   ├── intrablock_audit.py
│   ├── preservation_audit.py
│   ├── typography_audit.py
│   ├── render_ops_audit.py
│   ├── visual_image_audit.py
│   └── final_render_audit.py
└── autocorrect/
    ├── suggestions.py
    ├── correction_plan.py
    └── retry_policy.py
```

Ce module ne doit pas reconstruire. Il doit **évaluer, bloquer, expliquer, scorer et proposer des corrections**.

---

# 4. Le modèle de score publication-ready

Il faut un score par page, puis un score global.

## 4.1 Score page

```text
PublicationReadyPageScore:
    page_id
    page_index
    status: ok | review | ko
    score_total: 0.0 - 1.0

    stage_scores:
        pageprint_score
        pagetranslate_score
        contract_score
        background_score
        block_layout_score
        intrablock_score
        preservation_score
        typography_score
        rendering_score
        visual_similarity_score

    hard_blockers:
        missing_text
        source_text_leak
        destroyed_object
        text_overlap
        protected_overlap
        untranslated_required_text
        unresolved_style
        missing_background
        backend_mismatch

    findings
    correction_suggestions
```

## 4.2 Score document global

Le score global ne doit pas être une simple moyenne.

Il faut :

```text
Document score =
    moyenne pondérée des pages
    avec pénalités fortes si :
        une page est KO
        une page contient texte manquant
        une page détruit un objet
        une page a source leak
```

Règle :

```text
Si une seule page a texte traduisible manquant :
    document_status = review ou ko

Si une seule page a source_text_leak critique :
    document_status = review ou ko

Si une seule page détruit formule/image/table :
    document_status = ko
```

Donc :

```text
un document à 95 % ne veut pas dire :
    moyenne = 0.95

Cela veut dire :
    toutes les pages critiques passent les gates
    et la moyenne pondérée >= 0.95
```

---

# 5. Les sous-modules d’évaluation

## 5.1 `pageprint_audit.py`

But : vérifier que `pageprint/` donne assez d’information pour reconstruire.

Il vérifie :

```text
page size
coordinate system
units hierarchy
blocks
phrases
lines
words
regions
visual_layers
style_system
policies
constraints
reconstruction_constraints
protected regions
relations graph
reading order
```

Score :

```text
pageprint_score
```

Hard blockers :

```text
missing page size
missing units
missing visual layers
missing regions
missing reconstruction constraints
unreliable bbox for renderable units
```

Important : ce module ne dit pas “pageprint est bon” en général. Il dit :

```text
pageprint est-il suffisant pour reconstruire CETTE page ?
```

---

## 5.2 `pagetranslate_audit.py`

But : vérifier que la traduction est complète et utilisable.

Il vérifie :

```text
100 % textes traduisibles couverts
tokens protégés respectés
noms propres préservés
code/formules non traduits
segments non vides
expansion ratio acceptable
status linguistique
mapping source_unit_ids → translation_unit_id
```

Hard blockers :

```text
texte source traduisible absent
traduction tronquée
token protégé altéré
code traduit comme prose
formule traduite
```

---

## 5.3 `contract_audit.py`

But : vérifier que `pagereconstruct/` a bien consommé `pageprint/` + `pagetranslate/`.

Il vérifie :

```text
chaque reconstruction_unit a un BlockReconstructionContract
chaque block a layout_contract
chaque block a style_contract
chaque block a renderer_contract
chaque block a preservation_policy
chaque objet hors texte a PreservationContract
chaque zone texte remplacée a TextRemovalContract
chaque layer est ordonné
```

Hard blockers :

```text
translated unit non consommée
preservation unit non consommée
exclusion unit ignorée
bbox manquante
renderer absent
style absent
```

C’est ici que l’on force `pagereconstruct/` à être consommateur fidèle.

---

## 5.4 `background_audit.py`

C’est le module le plus important pour éviter les faux OK.

Il vérifie :

```text
clean background existe
fond propre utilisé comme base
ancien texte source retiré
fond non détruit
couleurs/trames préservées
source image non utilisée comme fond final en publication
```

Il doit produire :

```text
TextRemovalLedger
```

Avec une ligne par zone texte source :

```text
source_text
translated_text
source_bbox
removal_bbox
expected_action
clean_background_verified
residual_ink_score
source_text_leak_score
```

Hard blockers :

```text
texte source encore visible
clean background absent
patch destructeur
fond reconstruit par blanc brutal
```

---

## 5.5 `block_layout_audit.py`

But : vérifier le positionnement des grands blocs.

Il compare :

```text
source block bbox
target layout bbox
actual rendered bbox
reading order
distance inter-blocs
marges
colonnes
figures obstacles
flow regions
```

Score :

```text
block_layout_score
```

Hard blockers :

```text
bloc majeur déplacé hors zone
ordre de lecture cassé
bloc sur figure/formule
bloc sorti page
gros chevauchement
```

---

## 5.6 `intrablock_audit.py`

C’est exactement ce que tu décris dans l’ancienne formule : positionner intelligemment les éléments dans les blocs.

Il vérifie :

```text
texte traduit dans le bloc
non-traduit dans le bloc
inline code
formules inline
puces
numérotation
italiques
gras
retours ligne
line-height
indentation
hanging indent
alignment
```

Il doit comparer :

```text
block content contract
vs
actual rendered lines
```

Hard blockers :

```text
texte hors bloc
élément non traduit perdu
puce perdue
formule inline déplacée
code inline traduit
```

---

## 5.7 `preservation_audit.py`

But : zones spéciales non traduites.

Il vérifie :

```text
formules
équations
code
table grids
logos
watermarks
images
figures
page numbers
publisher marks
axis labels si non traduisibles
diagram labels
```

Il compare :

```text
source crop
clean background crop
final crop
PreservationOp
```

Hard blockers :

```text
formule effacée
image altérée
table grid cassée
logo disparu
numéro de page dupliqué ou perdu
```

---

## 5.8 `typography_audit.py`

But : vérifier la fidélité typographique.

Il ne doit pas seulement lire les métadonnées.

Il doit comparer :

```text
style contract
rendered result
source visual evidence
```

Critères :

```text
font class
font family/fallback
font size em
ink height
line height
bold/italic
color
alignment
indentation
tracking
style runs
density
```

Hard blockers :

```text
font size incohérente
style massivement réparé
texte illisible
titre rendu comme corps
corps rendu comme caption
code non monospace
```

---

## 5.9 `render_ops_audit.py`

But : vérifier que le backend ne triche pas.

Il vérifie :

```text
BackgroundOp existe
PatchOps déclarées
TextOps pour chaque texte traduit
PreservationOps pour chaque objet spécial
PDF et PNG utilisent les mêmes ops
aucun backend ne relit layers directement
aucun source_image_path caché
```

Hard blockers :

```text
backend hidden source background
PDF/PNG divergence
TextOp manquante
PreservationOp manquante
PatchOp destructrice
```

---

## 5.10 `visual_image_audit.py`

But : validation visuelle réelle.

Il compare :

```text
source image
clean background
reconstructed image
overlay/diff
```

Mesures :

```text
image similarity par zones
source text leak
non-text preservation
ink density
overlap raster
visual drift
crop similarity
```

Ce module doit produire des crops de preuve :

```text
failed_crops/
overlay_errors.png
diff_heatmap.png
```

Hard blockers :

```text
visuellement mauvais mais score théorique bon
texte ancien visible
superposition visible
patch visible
objet effacé
```

---


