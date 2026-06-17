# Analyse critique — Extraction / Segmentation
**Date** : 2026-04-23  
**Périmètre** : extraction du texte depuis les PDFs, structuration en blocs/lignes/phrases/spans, reconstruction en unités sémantiques, cohérence des bboxes, qualité des segmentations y compris LLM.  
**Hors périmètre** : traduction finale, rendu PDF.

---

## SYNTHÈSE

Le pipeline a une architecture solide (hiérarchie claire, extraction hybrid native+OCR, heuristique riche) mais est **dysfonctionnel en pratique** à cause de 4 bugs critiques dans le calcul de bboxes et la segmentation intra-ligne. Le correcteur LLM est **entièrement inopérant** : ses sorties ne sont jamais appliquées, et les 3 fichiers `page_219_semantic_phrases*.txt` sont **byte-for-byte identiques** entre heuristique, Qwen et Phi.

---

## CRITIQUES (bloquants)

### [CRIT-1] `_approximate_text_fragment_bbox()` — bboxes en décalage systématique

**`ocr_server.py`**, fonction `_approximate_text_fragment_bbox` :
```python
fx0 = x0 + (x1 - x0) * (start / total)  # répartition proportionnelle aux caractères
fx1 = x0 + (x1 - x0) * (end / total)
```

Divise la bbox de ligne **proportionnellement au nombre de caractères**, sans tenir compte du kerning, ligatures ou polices variables. Un fragment "The cat." (9 chars sur 18) reçoit exactement la moitié droite de la ligne, même si visuellement il occupe 40 ou 60 %.

**Cause** : les bboxes réelles des spans OCR sont déjà connues mais ignorées. La fonction recalcule au lieu de les utiliser.

**Impact** : touche 100 % des phrases multi-lignes ou intra-ligne. Décalage de 5–15 px en x, propagé à tous les niveaux.

---

### [CRIT-2] Spans synthétiques sans validation — brèches textuelles silencieuses

**`ocr_server.py`**, fallback dans `_semantic_fragment_spans_for_line` :
```python
elif frag_text:
    spans.append({"texte": frag_text, "bbox": frag_bbox, "style": {}})
```

Quand aucun span OCR ne couvre un fragment intra-ligne, un span synthétique est créé avec :
- `style: {}` → rendu sans police/couleur
- `bbox` issue du calcul défectueux de CRIT-1
- Aucune vérification de doublonnage avec les spans voisins

**Impact** : `phrase.text` complet mais `phrase.spans` incomplet ou mal positionné. Certains spans ne sont pas traduits (style vide = skip). Perte silencieuse de traductabilité.

---

### [CRIT-3] Correcteur LLM entièrement inopérant

**`ocr_server.py`**, `_llm_postprocess_blocks()` + **`llm_semantic_corrector.py`** (tout le module) :

La fonction formate les blocs pour le LLM, appelle le modèle, parse la réponse — **mais n'applique jamais les corrections retournées** aux blocs. Le `_llm_semantic_phrases_are_quality_regression()` rejette tout sur la moindre anomalie détectée (logique "all-or-nothing"), ce qui force toujours le fallback vers l'heuristique.

**Preuve concrète** : les fichiers `page_219_semantic_phrases.txt`, `page_219_semantic_phrases_qwen_fast.txt`, `page_219_semantic_phrases_phi_fast.txt` sont **identiques ligne par ligne**. Sur 20 pages (run qwen_fast), 32 blocs marqués "llm_corrected" mais zéro différence de segmentation.

**Impact** : 500+ lignes de code non fonctionnel, coût CPU inutile (~5–10 min/page), aucune amélioration LLM effective.

---

### [CRIT-4] Spans partagés entre phrases adjacentes — duplication de texte

**`ocr_server.py`**, seuil 0.5 dans `_span_x_overlaps` :

Quand `_approximate_text_fragment_bbox()` produit des bboxes décalées (CRIT-1), deux fragments de phrases adjacentes peuvent recevoir des bboxes qui se chevauchent. Le même span OCR passe alors le seuil d'overlap 50 % pour **les deux phrases**, y est inclus deux fois.

**Impact** : texte dupliqué dans le JSON, traduit deux fois, rendu superposé → illisible. Cause visible dans les runs v6.

---

## MAJEURS

| # | Problème | Fichier | Impact |
|---|---|---|---|
| MAJ-1 | `_build_llm_split_fragments()` utilise un fallback proportionnel (même défaut que CRIT-1) pour les splits LLM | `ocr_server.py` | Bboxes fragment LLM incorrectes |
| MAJ-2 | Lignes séparatrices (barres de fraction, `---`) atomisées à tort — formules mathématiques cassées en 3 unités | `ocr_server.py` l.1691–1699 | ~5–10 % des pages sci/tech |
| MAJ-3 | `_split_spans_at_sentence_boundaries()` native PDF trop conservateur — ne reconnaît que la ponctuation, pas les conjonctions + capitalisation | `native_pdf_extractor.py` l.110–135 | Native et OCR divergent sur le nombre de phrases |
| MAJ-4 | Aucune détection explicite de mise en page multi-colonnes — repose uniquement sur `hard_break_before` | `structure_extractor.py` | Texte de colonnes entrecroisé sur ~10–20 % des pages |
| MAJ-5 | Quality check LLM all-or-nothing : 1 anomalie détectée → tout rejeté, même si 8/10 phrases sont meilleures | `ocr_server.py` l.1951–2050 | Bloque toute amélioration LLM même valide |

---

## MINEURS

- **MIN-1** : Résolution de césure basique (`"guaran-teed"` → `"guaranteed"`) sans validation lexicale
- **MIN-2** : `_ATOMIC_BLOCK_ROLES` manque `table_caption`, `algorithm_label`, `code_block`, `equation_label` (~10 lignes à ajouter)
- **MIN-3** : `ocr_confidence_mean` stockée dans chaque `semantic_phrase` mais jamais utilisée pour marquer les phrases à faible confiance

---

## INCOHÉRENCES DE CONCEPTION

**1. Deux heuristiques parallèles, aucune spécification de laquelle prime.**
`native_pdf_extractor.py` et `ocr_server.py` segmentent les phrases intra-ligne avec des logiques différentes. Un même bloc peut avoir 5 phrases natives et 9 phrases OCR. Pas de règle de sélection documentée.

**2. Invariant de cohérence jamais vérifié.**
`phrase.bbox` devrait être l'union des `span.bbox`. Ce n'est pas vérifié nulle part. L'erreur de CRIT-1 peut donc exister en production sans être détectée.

**3. LLM pluggé mais sortie non câblée.**
Architecture "call + ignore" : le LLM est appelé (coût réel) mais ses résultats ne sont pas intégrés. Meilleure approche : `if not SEMANTIC_CORRECTOR_ENABLED: skip`, plutôt que call + reject systématique.

---

## ANALYSE DU CORRECTEUR LLM

### Comparaison page 219 : heuristique vs Qwen vs Phi

Les 3 fichiers `page_219_semantic_phrases*.txt` sont **byte-for-byte identiques** (23 blocs, même découpage phrase par phrase).

### Raison de l'inopérance

1. **`_llm_postprocess_blocks()`** formate et appelle le LLM, parse la réponse, **mais ne met jamais à jour `block["semantic_phrases"]`** avec les résultats.
2. **`_llm_semantic_phrases_are_quality_regression()`** rejette la moindre anomalie → fallback systématique vers l'heuristique.
3. **No-op fallback** ligne ~2201 : `block["semantic_phrases"] = original_phrases` est toujours atteint.

### Métriques concrètes
- 20 pages run qwen_fast : 32 blocs marqués "llm_corrected", 0 différence de segmentation
- Overhead : ~5–10 min CPU par page pour zéro gain

### Risques si corrigé sans garde-fous
- Hallucination : ajout de texte absent du PDF
- Fusion de phrases → perte de contenu
- Réécriture légère → rupture traduction

---

## CORRECTIONS RECOMMANDÉES (priorisées)

| Priorité | Correction | Effort estimé |
|---|---|---|
| **P0** | Remplacer le calcul proportionnel de `_approximate_text_fragment_bbox()` par une union des bboxes des spans OCR qui intersectent le fragment | 1h |
| **P0** | Fixer le partage de spans entre phrases adjacentes : marquer les spans "used" après affectation à une phrase | 1.5h |
| **P0** | Désactiver le correcteur LLM (skip l'appel) OU le câbler vraiment (appliquer les corrections retournées + revoir quality check) | 30 min / 3h |
| **P1** | Ajouter `table_caption`, `algorithm_label`, `code_block` à `_ATOMIC_BLOCK_ROLES` | 30 min |
| **P1** | Implémenter détection multi-colonnes explicite dans `structure_extractor.py` | 1.5h |
| **P1** | Rendre `_split_spans_at_sentence_boundaries()` sensible aux conjonctions + capitalisation | 45 min |
| **P2** | Assertion de cohérence post-construction : `phrase.bbox == union(span.bbox for span in phrase.spans)` | 20 min |
| **P2** | Utiliser `ocr_confidence_mean` pour taguer les phrases `"ocr_quality": "low/medium/high"` | 15 min |

---

## TESTS DE NON-RÉGRESSION À AJOUTER

Fichier cible : `tests/test_extraction_robustness.py`

```python
def test_phrase_bbox_contains_all_span_bboxes():
    """phrase.bbox >= union de tous ses spans"""

def test_no_duplicate_spans_between_adjacent_phrases():
    """aucun span ne peut appartenir à 2 phrases successives"""

def test_intraline_split_produces_non_overlapping_bboxes():
    """bboxes des phrases intra-ligne ne se chevauchent pas"""

def test_native_and_ocr_phrase_texts_are_consistent():
    """même texte total, même si découpé différemment"""

def test_llm_output_is_applied_when_enabled():
    """si LLM activé, au moins 1 différence vs heuristique sur une page test"""
```

---

## ÉTAT HONNÊTE

L'extraction textuelle est fiable (le texte est présent dans le JSON). Les problèmes sont dans le **positionnement** (bboxes intra-ligne systématiquement approximées par calcul proportionnel) et dans le **câblage LLM** (sorties jamais appliquées).

Deux priorités absolues avant toute autre évolution :
1. Fixer CRIT-1 : remplacer `_approximate_text_fragment_bbox()` par union de spans réels
2. Câbler ou désactiver le correcteur LLM
