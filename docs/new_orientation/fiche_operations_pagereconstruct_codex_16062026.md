# FICHE DES OPÉRATIONS — Unité PAGERECONSTRUCT / couverture texte WYSIWYG
**Mise en ordre de la discussion CODEX du 16/06/2026**
Source : `03_unit_pagereconstruct_discussion_CODEX_16062026.md`.

## 1. Synthèse
- **Objet** : Suivre les opérations menées autour de PAGEPRINT → PAGETRANSLATE → PAGERECONSTRUCT, en particulier la garantie que tous les textes extraits par PAGEPRINT se retrouvent dans le rendu final, traduits ou préservés.
- **Constat central** : Les blocks originaux ont fini par être couverts à 100% après correction de pagetranslate/projection.py, mais la couverture de tout le texte jusqu’au rendu visuel final n’est pas encore garantie par un contrôle simple et bloquant.
- **Décision utilisateur** : Ne pas poursuivre une architecture lourde. Mettre en place un petit script direct qui vérifie que tout texte original extrait par PAGEPRINT est présent dans le rendu PAGERECONSTRUCT, traduit ou non.
- **État final de la discussion** : Des corrections et audits ont été engagés, mais la dernière demande remplace la piste lourde SourceTextLifecycleLedger par un script minimal de contrôle de survivance du texte.

## 2. Tableau des opérations
| ID | Phase | Demandé | Constaté / objectif | Fait | État |
|---|---|---|---|---|---|
| OP-01 | Sélection de nouvelles pages | Prendre 10 pages jamais utilisées auparavant et extraire leurs trames de fond. | Besoin d’un échantillon neutre pour juger la reconstruction sans recycler les pages déjà vues. | Deux lots produits. Lot principal utilisé ensuite : p0033, p0077, p0219, p0252, p0264, p0279, p0315, p0324, p0358, p0414. | Fait |
| OP-02 | Trames de fond | Afficher les fonds nettoyés et une version texture amplifiée. | Les fonds doivent être séparés du texte pour juger les bboxes et la recomposition. | Planche contact produite dans results/new10_background_textures_20260616_batch2/contact_sheet_clean_backgrounds_texture.png. | Fait |
| OP-03 | Bboxes des blocks | Sur ces fonds, placer uniquement les bboxes des blocks. | Contrôler l’ancrage des blocks PAGEPRINT sur le fond, sans labels ni couches parasites. | Images individuelles et planche contact générées dans block_bboxes_only/. | Fait |
| OP-04 | Bboxes des lignes après PAGETRANSLATE | Dans les bboxes des blocks, placer les bboxes des lignes/phrases réorganisées après PAGETRANSLATE. | Vérifier la composition intra-block : succession de phrases, retours à la ligne, textes traduits ou préservés. | CODEX a d’abord voulu créer deux fonctions nouvelles, puis a vérifié l’existant et a retiré les ajouts doublons. Usage de compose_block / compose_contract, compile_page_render_plan et intrablock_audit. | Fait, avec correction méthodologique |
| OP-05 | Visuels blocks + lignes | Générer une vue rouge = blocks, bleu = lignes intra-block après PAGETRANSLATE/reconstruction. | Contrôler la présence et le placement des lignes calculées dans les blocks. | Dossier produit : block_and_pagetranslate_line_bboxes/. Les 10 pages sont passées intrablock=ok. | Fait |
| OP-06 | Comparer blocks originaux et éléments projetés | Ne pas utiliser les blocks compilés après PAGETRANSLATE comme référence ; garder les blocks originaux fixes. | Voir si les éléments intra-block après PAGETRANSLATE remplissent les blocks d’origine. | Dossier produit : original_blocks_with_after_pagetranslate_intrablocks/. Rouge = blocks originaux ; bleu = bboxes intra-block après PAGETRANSLATE/composition. | Fait |
| OP-07 | Constat de blocks vides | Identifier ce qui manque dans les blocks originaux après PAGETRANSLATE. | Plusieurs blocks rouges ne sont pas remplis après projection ; il faut savoir si des éléments disparaissent. | Fichier généré : missing_original_blocks_after_pagetranslate.json. Constat : nombreux blocks manquants par page avant correction. | Constaté |
| OP-08 | Correction 100% blocks | Règle imposée : on ne saute rien. Les textes traduisibles sont remplacés par traduction ; les autres restent tels quels. | Chaque block original doit avoir une disposition explicite après PAGETRANSLATE. | Correction dans pagetranslate/projection.py : ajout de _preserve_uncovered_original_units et _preserved_reconstruction_unit ; les unités traduites ont render_contract.mode='translated_text', les autres mode='preserve_original'. | Fait |
| OP-09 | Tests couverture blocks | Ajouter un test qui prouve que les unités originales traduisibles et non traduisibles sont présentes dans reconstruction_units. | Empêcher la régression du bug de disparition des blocks. | Test ajouté : tests/pagetranslate/test_projection_preserves_all_originals.py. Sous-ensemble ciblé : 14 passed. Vérification réelle : missing_blocks=0 sur les 10 pages. | Fait |
| OP-10 | Vues après correction blocks | Régénérer les vues corrigées. | Vérifier visuellement que tous les blocks originaux ont une disposition. | Dossier produit : after_fix_all_original_blocks_views/. Résultat annoncé : missing=0 partout. | Fait |
| OP-11 | Nouveau problème : couverture texte | Constat visuel : dans certains blocks, tout le texte PAGEPRINT ne semble pas réapparaître après PAGERECONSTRUCT. | Le problème n’est plus seulement le block ; c’est la survivance de chaque texte extrait. | Analyse p0279 : les bboxes bleues sont des lignes recomposées, pas les lignes originales. Le pipeline vérifie trop la couverture par unités/bboxes et pas assez la couverture textuelle exacte. | Constaté |
| OP-12 | Audit original_text_coverage | Vérifier que chaque texte PAGEPRINT est soit traduit, préservé, couvert par parent/enfant, ou exclu avec raison valide. | Faire échouer PAGETRANSLATE si un texte original disparaît sans disposition explicite. | Implémentation dans pagetranslate/functional_validator.py : audit_original_text_coverage, TEXT_LEVELS, VALID_EXCLUSION_REASONS. Test ajouté et 5 passed. | Fait partiellement |
| OP-13 | Visuels après audit texte | Refaire les visuels avec l’audit de couverture texte actif. | Vérifier si le problème visuel est réglé. | Dossier lancé : after_text_coverage_audit_views/. L’utilisateur constate que la couverture reste insuffisante visuellement. | Insuffisant |
| OP-14 | Comparaison block par block | Comparer texte original PAGEPRINT et texte entrant dans PAGERECONSTRUCT. | Voir exactement ce qui entre dans PAGERECONSTRUCT block par block. | Premier fichier en dry_run jugé invalide car render == source. Reprise avec vraie traduction CT2 locale. | Fait mais ne répond pas encore à la vraie question |
| OP-15 | Comparaison vraie traduction | Utiliser le translated_text réel de PAGETRANSLATE, sans traduction fabriquée dans le script. | S’assurer que render correspond exactement à reconstruction_unit.translated_text. | Dossier produit : pagereconstruct_input_vs_original_text_by_block_REAL_TRANSLATED/. Un .txt et un .json par page. Exemple p0279 : mélange d’unités translated et preserved signalé. | Fait |
| OP-16 | Recentrage de la vraie question | Demande utilisateur : vérifier si tout texte extrait par PAGEPRINT passe dans PAGETRANSLATE, est reconstruit dans PAGERECONSTRUCT et visible dans le rendu final, traduit ou non. | La vraie preuve attendue est end-to-end : source_unit → décision de traduction/préservation → entrée reconstruction → opération de rendu → visibilité finale. | Constat CODEX : le projet ne le garantit pas encore à 100%. Les validations existantes sont partielles. | Constaté |
| OP-17 | Piste SourceTextLifecycleLedger | Chercher si un ledger complet existe déjà ; sinon l’ajouter. | Suivre chaque texte PAGEPRINT jusqu’à TextOp/PreservationOp et audit visuel. | Des morceaux existent : original_text_coverage, TextRemovalLedger, SourceUnitState, render_ops_audit. Pas de ledger complet. CODEX ajoute une implémentation lourde : source_text_lifecycle_ledger.py, tests, intégration render_ops_audit, schema, plan_compiler. | En cours / à simplifier |
| OP-18 | Décision finale utilisateur | Arrêter la complexification : faire seulement un petit script. | Contrôle demandé : s’assurer que tout le texte original extrait par PAGEPRINT est dans le rendu PAGERECONSTRUCT, traduit ou non. | La discussion s’arrête sur cette instruction. La prochaine action doit être un script simple, pas une architecture lourde. | À faire maintenant |

## 3. Constats techniques
- **C1 — Les trames de fond sont correctement isolées** : Les planches de fonds nettoyés et texture amplifiée sont exploitables pour contrôler les overlays.
- **C2 — Les blocks PAGEPRINT sont visibles sur les fonds** : La couche block_bboxes_only permet de vérifier les cadres originaux.
- **C3 — La composition intra-block existait déjà** : compose_block / compose_contract et intrablock_audit étaient déjà présents ; les doublons créés initialement ont été retirés.
- **C4 — Avant correction, de nombreux blocks originaux disparaissaient après PAGETRANSLATE** : Exemples : p0077 19/20 manquants, p0219 22/23, p0279 15/16, p0414 9/11.
- **C5 — La correction projection.py règle la couverture des blocks** : Après ajout de la préservation des unités non couvertes, missing_blocks=0 sur les 10 pages testées.
- **C6 — La couverture des blocks ne suffit pas** : Un block peut être présent, mais une partie de son texte extrait par PAGEPRINT peut ne pas être prouvée comme rendue visuellement.
- **C7 — Les comparaisons dry_run étaient invalides pour juger la traduction** : Le dry_run garde le texte source ; il ne prouve pas la présence de translated_text réel.
- **C8 — La vraie exigence est end-to-end** : Chaque texte PagePrint doit avoir une chaîne complète : décision → reconstruction_unit → TextOp/PreservationOp → rendu final visible.

## 4. Fichiers de code touchés d’après la discussion
| Fichier | Type | Détail |
|---|---|---|
| pagetranslate/projection.py | Correction principale | Ajout de la préservation des unités originales non couvertes ; ajout du mode render_contract='translated_text' pour les unités traduites. |
| tests/pagetranslate/test_projection_preserves_all_originals.py | Test invariant | Vérifie qu’un block traduisible et un block non traduisible obtiennent tous deux une disposition de reconstruction. |
| pagetranslate/functional_validator.py | Audit texte | Ajout de audit_original_text_coverage pour bloquer les textes originaux sans disposition logique. |
| pagereconstruct/source_text_lifecycle_ledger.py | Ledger lourd | Ajout proposé d’un ledger PagePrint → reconstruction → render ops. À simplifier selon la dernière instruction utilisateur. |
| tests/pubready/test_source_text_lifecycle_audit.py | Tests ledger | Tests pour bloquer une unité PagePrint qui n’a aucun TextOp/PreservationOp. |
| pagereconstruct/render_ops.py | Support source_unit_ids | Ajout de source_unit_ids aux PreservationOp. |
| pagereconstruct/overlay_manager.py | Propagation IDs | Propagation de source_unit_ids vers les PreservationOp. |
| pubready/stages/render_ops_audit.py | Branchement ledger | Consommation du SourceTextLifecycleLedger dans l’audit render_ops. |
| pagereconstruct/schema.py | Plan enrichi | Ajout du champ source_text_lifecycle_ledger au PageRenderPlan. |
| pagereconstruct/plan_compiler.py | Construction ledger | Construction du ledger après génération des render_ops. |
| tests/pagereconstruct/test_render_ops.py | Test adapté | Vérifie la présence du champ source_text_lifecycle_ledger dans le plan, sans imposer qu’il soit non vide sur fixture synthétique. |
| pagereconstruct/composition/intrablock_composer.py | Ajout retiré | Deux fonctions ajoutées puis supprimées après vérification de l’existant. |
| tests/pagereconstruct/test_intrablock_post_pagetranslate.py | Test retiré | Test initial supprimé car il doublonnait la logique existante. |

## 5. Artefacts générés
| Chemin | Objet | Commentaire |
|---|---|---|
| results/new10_background_textures_20260615/ | Premier lot de 10 fonds propres | p0024, p0028, p0038, p0149, p0300, p0334, p0388, p0429, p0445, p0466. |
| results/new10_background_textures_20260616_batch2/ | Second lot de 10 fonds | Lot principal utilisé pour la suite : p0033, p0077, p0219, p0252, p0264, p0279, p0315, p0324, p0358, p0414. |
| block_bboxes_only/ | Blocks seuls | Fonds nettoyés + rectangles des blocks texte. |
| block_and_pagetranslate_line_bboxes/ | Blocks + lignes recomposées | Rouge = blocks post-PAGETRANSLATE ; bleu = lignes intra-block calculées. |
| original_blocks_with_after_pagetranslate_intrablocks/ | Blocks originaux + intrablocks après PAGETRANSLATE | Rouge = blocks originaux ; bleu = bboxes intra-block après composition. |
| missing_original_blocks_after_pagetranslate.json | Audit blocks manquants | Liste des blocks originaux absents après PAGETRANSLATE avant correction. |
| after_fix_all_original_blocks_views/ | Vues après correction blocks | Affiche missing=0 partout sur les 10 pages. |
| after_text_coverage_audit_views/ | Vues après audit couverture texte | Jugé insuffisant par l’utilisateur : la vraie preuve doit aller jusqu’au rendu final. |
| pagereconstruct_input_vs_original_text_by_block/ | Comparaison initiale dry_run | Invalide pour juger la traduction car render == source. |
| pagereconstruct_input_vs_original_text_by_block_REAL_TRANSLATED/ | Comparaison avec vraie traduction | Un TXT et un JSON par page ; render = reconstruction_unit.translated_text fourni par PAGETRANSLATE. |

## 6. Invariants à retenir
- Aucun texte original extrait par PAGEPRINT ne doit disparaître silencieusement.
- Un texte traduisible doit être remplacé par translated_text.
- Un texte non traduisible/protégé doit être rendu tel quel ou préservé comme overlay avec raison explicite.
- Un texte exclu doit porter une raison valide et traçable.
- Une unité texte originale ne doit pas être rendue deux fois sauf cas explicitement autorisé.
- Le contrôle doit être bloquant : une seule unité texte manquante suffit à mettre la page en KO.
- Le script de contrôle ne doit pas fabriquer de traduction ; il consomme strictement les sorties de PAGETRANSLATE.

## 7. Suite opérationnelle recommandée
- **NA-01 — Stopper la piste lourde comme livrable principal** : Le SourceTextLifecycleLedger peut rester comme expérimentation, mais la demande utilisateur est maintenant un script simple et lisible.
- **NA-02 — Créer un script minimal** : Nom recommandé : tools/audit_text_survival.py ou scripts/audit_pageprint_text_survival.py.
- **NA-03 — Entrées du script** : Prendre un input_data PAGEPRINT, la sortie PAGETRANSLATE, le plan/rendu PAGERECONSTRUCT et les render_ops. Ne pas retraduire dans le script.
- **NA-04 — Contrôle attendu** : Pour chaque unité texte PAGEPRINT : source_unit_id, texte, bbox, état traduisible/protégé, translated_text ou preserved_text, reconstruction_unit_id, TextOp/PreservationOp, bbox finale, statut OK/KO.
- **NA-05 — Règle de blocage** : Si un texte PAGEPRINT n’est ni traduit, ni préservé, ni explicitement exclu avec raison valide, ni rendu par TextOp/PreservationOp : KO.
- **NA-06 — Sorties** : CSV + JSON + rapport Markdown. Une ligne par texte original. Résumé par page : total textes, OK, KO, exclus valides, doublons, textes sans render op.
- **NA-07 — Option visuelle** : En V1, ne pas faire OCR image. Vérifier la visibilité par les opérations de rendu. L’OCR/raster audit pourra être une V2.

## 8. État final
La couverture des blocks originaux après PAGETRANSLATE a été corrigée et testée. La couverture complète de tout le texte jusqu’au rendu final n’est pas encore prouvée par un outil simple. La prochaine opération doit être la création d’un script minimal de contrôle de survivance du texte PAGEPRINT dans le rendu PAGERECONSTRUCT, traduit ou préservé.
