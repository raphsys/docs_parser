# TRANSITION POUR CLAUDE CLI — Projet `docs_parser` / Pipeline WYSIWYG de traduction documentaire

## 0. Contexte général

Nous travaillons sur un projet nommé `docs_parser`.

L’objectif est de construire un pipeline capable de :

```text
PDF / image / page scannée / page native
→ extraction complète
→ compréhension documentaire
→ traduction
→ reconstruction WYSIWYG fidèle
```

Le vieux fichier monolithique `ocr_server.py` était trop long et non maintenable. Nous sommes en train de le découper en unités indépendantes.

Les unités principales sont :

```text
PAGEPRINT       = tête d’entrée / compilateur documentaire
PAGETRANSLATE   = exécuteur contrôlé du plan de traduction
RECONSTRUCTOR   = reconstruction WYSIWYG après traduction
TRANSLATION_ENGINE = moteur IA branchable pour traduire les unités validées
```

Le principe architectural final est :

```text
PAGEPRINT ne fait pas juste de l’OCR.
PAGEPRINT compile une page en instructions exploitables.

PAGETRANSLATE ne doit pas deviner quoi traduire.
PAGETRANSLATE doit exécuter views.translation_plan produit par PAGEPRINT.

Le moteur de traduction ne doit recevoir que des unités propres,
avec rôles, tokens protégés, contexte, contraintes et render_target.
```

---

# 1. Où nous en sommes

Nous avons progressé par versions successives.

## rev_03

Objectif : supprimer la dépendance principale de `PAGETRANSLATE` à `selector.py` + `coalescer.py`.

Résultat validé :

```text
PAGEPRINT produit views.translation_plan.
PAGETRANSLATE lit views.translation_plan.
selector/coalescer ne sont plus le chemin normal.
fallback autorisé seulement si translation_plan absent.
translation_plan vide = KO, pas de fallback silencieux.
```

Le cœur `PAGEPRINT → translation_plan → PAGETRANSLATE` est validé.

## rev_04

Objectif : renforcer les structures documentaires.

Ajout/renforcement :

```text
body_paragraph_builder.py
table_builder.py
index_builder.py
caption_builder.py
figure_builder.py
publisher_mark_builder.py
toc_builder.py
list_builder.py
code_builder.py
formula_builder.py
functional audit
batch audit
```

Résultat :

```text
PAGEPRINT/PAGETRANSLATE sont maintenant assez mûrs pour commencer les essais moteur.
Pas besoin de nouvelle refonte fondamentale PAGEPRINT/PAGETRANSLATE.
```

## rev_05

Objectif : première mise en place du moteur de traduction.

Ajout :

```text
translation_engines/
mock_engine.py
rule_engine.py
local_model_engine.py
external_model_engine.py
factory.py
tools/run_translation_trial.py
tools/run_document_trial.py
```

Résultat :

```text
Le branchement technique moteur existe.
Les moteurs mock/rule servent aux essais de pipeline.
Mais le vrai moteur IA local n’est pas encore branché proprement.
```

## rev_06

Objectif : bootstrap du runtime IA local.

Ajout :

```text
translation_engines/base.py
translation_engines/request.py
translation_engines/model_registry.py
translation_engines/ct2_engine.py
translation_engines/engine_health.py
translation_engines/placeholder_policy.py
tools/check_translation_engine.py
tools/test_placeholder_roundtrip.py
tools/run_batch_translation_trial.py
```

Les tests ciblés passaient côté refonte :

```text
tests/pageprint
tests/pagetranslate
tests/functional
tests/pipelines
```

Mais le test global du projet peut échouer à cause du legacy :

```text
ModuleNotFoundError: rapidocr_onnxruntime
```

Ce n’est pas bloquant pour `PAGEPRINT/PAGETRANSLATE/translation_engines`.

---

# 2. État actuel exact

Le socle documentaire est considéré comme suffisamment stable.

Il ne faut PAS refaire une grande refonte de :

```text
PAGEPRINT
PAGETRANSLATE
translation_plan
logical_structures
semantic_builder
view_compiler
```

Le prochain travail prioritaire est :

```text
rev_07 = rendre le runtime IA CTranslate2 fiable.
```

La question actuelle n’est plus :

```text
Comment sélectionner les unités à traduire ?
```

Elle est maintenant :

```text
Comment charger et exploiter proprement les modèles IA locaux présents dans ai_models/translation/ ?
```

---

# 3. Dossier IA local attendu

Le dossier local `ai_models/translation/` existe chez l’utilisateur, mais il n’est pas inclus dans l’archive légère.

Il contient approximativement :

```text
ai_models/translation/
├── m2m100_418m_ct2_int8/
├── m2m100_418m_tokenizer/
├── opus_mt_tc_big_en_fr_ct2_int8/
├── opus_mt_tc_big_en_fr_tokenizer/
├── model_inventory.json
├── translation_profiles.json
├── style_tone_profiles.json
└── translation_memory.jsonl
```

Décision modèle :

```text
OPUS-MT EN→FR = moteur principal pour anglais → français.
M2M100 418M = moteur fallback multilingue.
translation_memory.jsonl = mémoire de traduction validée.
translation_profiles.json = profils de traduction.
style_tone_profiles.json = style / ton / registre.
model_inventory.json = registre technique.
```

---

# 4. Décision stratégique actuelle

Ne pas relancer une refonte PAGEPRINT/PAGETRANSLATE.

Travailler maintenant sur :

```text
translation_engines/ct2_engine.py
translation_engines/model_registry.py
translation_engines/translation_memory.py
translation_engines/profile_store.py
pagetranslate/placeholder_policy.py ou translation_engines/placeholder_policy.py
tools/check_translation_engine.py
tools/test_placeholder_roundtrip.py
tools/run_translation_trial.py
tools/run_batch_translation_trial.py
```

Objectif :

```text
Un modèle local CTranslate2 dans ai_models/translation/
doit pouvoir traduire correctement, en batch,
avec tokens protégés, mémoire, glossaire, QA et traces moteur.
```

---

# 5. Problèmes critiques identifiés dans rev_06

## P0.1 — `ct2_engine.py` doit être corrigé

Problème probable actuel : le décodage CTranslate2 est fragile.

CTranslate2 retourne généralement des objets contenant :

```python
result.hypotheses[0]
```

Il faut décoder via le tokenizer.

Implémentation cible :

```python
results = translator.translate_batch(batch_tokens, ...)
output_tokens = results[i].hypotheses[0]
translated = tokenizer.decode(
    tokenizer.convert_tokens_to_ids(output_tokens),
    skip_special_tokens=True
)
```

ou une variante correcte selon le tokenizer.

À créer dans `ct2_engine.py` :

```python
_decode_ct2_result(result, tokenizer) -> str
_encode_source(text, tokenizer, family, source_lang) -> list[str]
_target_prefix(tokenizer, family, target_lang) -> list[str] | None
```

Familles à distinguer :

```text
marian / opus
m2m100
nllb
generic
```

Ne pas utiliser le même traitement pour OPUS et M2M100.

---

## P0.2 — Le batch doit être réel

Problème possible : `translate_batch()` peut recevoir une liste, mais appeler CTranslate2 une fois par item.

Il faut envoyer un vrai batch :

```python
batch_tokens = [...]
results = translator.translate_batch(batch_tokens, ...)
```

Puis décoder toutes les sorties.

Objectif :

```text
une seule requête CT2 par batch,
pas une requête par unité.
```

---

## P0.3 — `model_registry.py` doit sélectionner selon source/target

Le choix modèle doit dépendre de :

```text
source_lang
target_lang
preferred_model
availability
priority
family
```

Pour `en → fr`, ordre cible :

```text
1. opus_mt_tc_big_en_fr
2. m2m100_418m
```

Il faut implémenter :

```python
select_model(source_lang, target_lang, preferred_model=None)
```

Et non prendre seulement le premier modèle disponible.

---

## P0.4 — Les chemins doivent être robustes

Les chemins dans `model_inventory.json` doivent pouvoir être :

```text
absolus
relatifs à la racine projet
relatifs au dossier model_inventory.json
relatifs à TRANSLATION_MODELS_ROOT
```

Le code doit éviter de casser si le script est lancé depuis un autre dossier.

---

## P0.5 — CLI trop limitées

Les outils doivent accepter :

```bash
--inventory
--model
--source-lang
--target-lang
--device
--compute-type
--batch-size
--max-input-tokens
```

À ajouter à :

```text
tools/check_translation_engine.py
tools/test_placeholder_roundtrip.py
tools/run_translation_trial.py
tools/run_batch_translation_trial.py
tools/run_document_trial.py
```

Commande cible :

```bash
python3 tools/check_translation_engine.py \
  --engine ct2 \
  --inventory ai_models/translation/model_inventory.json \
  --model opus_mt_tc_big_en_fr \
  --source-lang en \
  --target-lang fr \
  --device cpu \
  --compute-type int8
```

---

## P0.6 — Les erreurs runtime doivent remonter proprement

Séparer les statuts :

```text
pipeline_status
translation_runtime_status
linguistic_quality_status
publication_readiness_status
```

Si le moteur plante :

```text
pipeline_status peut rester ok
translation_runtime_status doit être ko
linguistic_quality_status doit être unknown ou ko
```

Ne pas mélanger erreur moteur et erreur PAGEPRINT.

---

## P1.1 — Placeholders robustes

Le système protège les tokens avec des placeholders.

Formats possibles :

```text
⟦PT0001⟧
<nt id="PT0001"/>
[[PT0001]]
@@PT0001@@
```

Pour CTranslate2, on doit tester quel format survit le mieux au tokenizer/modèle.

Créer ou renforcer :

```text
tools/test_placeholder_roundtrip.py
translation_engines/placeholder_policy.py
```

La restauration doit être tolérante :

```text
<nt id="PT0001"/>
< nt id = "PT0001" />
<nt id='PT0001' />
[[PT0001]]
@@PT0001@@
PT0001
```

Objectif :

```text
placeholder_corruption_rate = 0 sur les tests de base.
```

---

## P1.2 — Mémoire de traduction

Le fichier existe localement :

```text
ai_models/translation/translation_memory.jsonl
```

Mais il faut créer/brancher proprement :

```text
translation_engines/translation_memory.py
```

Format recommandé :

```json
{"source_lang":"en","target_lang":"fr","source":"Hidden layers","target":"Couches cachées","domain":"deep_learning","validated":true}
```

Ordre correct :

```text
translation_plan item
→ protection
→ terminology
→ translation_memory exact match
→ moteur IA
→ restore placeholders
→ QA
```

Si mémoire validée :

```text
ne pas appeler le modèle.
```

Ajouter dans `engine_trace` :

```json
{
  "memory_hit": true,
  "memory_source": "exact"
}
```

---

## P1.3 — Profils de traduction/style

Les fichiers existent :

```text
translation_profiles.json
style_tone_profiles.json
```

Créer :

```text
translation_engines/profile_store.py
```

Les profils doivent être injectés dans :

```text
context
engine_trace
quality policy
terminology policy
future prompt si moteur LLM
```

Même pour CTranslate2, ces profils servent à :

```text
choisir glossaire
choisir QA
choisir post-traitement
choisir registre de traduction
```

---

## P1.4 — Terminologie typée

Éviter une liste plate de termes verrouillés.

Il faut distinguer :

```text
preserve
preferred_translation
contextual
forbidden_translation
```

Exemple :

```json
{
  "MLP": {"policy": "preserve"},
  "CNN": {"policy": "preserve"},
  "ReLU": {"policy": "preserve"},
  "Softmax": {"policy": "preserve"},
  "precision": {"policy": "preferred_translation", "target": "précision"},
  "recall": {"policy": "preferred_translation", "target": "rappel"},
  "dropout": {"policy": "contextual", "preferred": "dropout"},
  "pooling": {"policy": "contextual", "preferred": "pooling"}
}
```

Ne pas verrouiller automatiquement :

```text
precision
recall
dropout
pooling
```

Ils doivent souvent être traduits ou contextualisés.

---

# 6. Tâches concrètes pour Claude CLI — rev_07

## Bloc A — Corriger CTranslate2 engine

Fichiers :

```text
translation_engines/ct2_engine.py
translation_engines/base.py
translation_engines/request.py
```

Todo :

* [ ] Implémenter vrai encodage source.
* [ ] Implémenter vrai décodage `result.hypotheses[0]`.
* [ ] Séparer `family=marian/opus`, `family=m2m100`, `family=nllb`, `generic`.
* [ ] Corriger `target_prefix`.
* [ ] Faire un vrai batch CT2.
* [ ] Ajouter `input_token_count`.
* [ ] Ajouter `output_token_count`.
* [ ] Ajouter `truncated=True` si troncature.
* [ ] Retourner `raw_output` propre dans `engine_trace`.
* [ ] Remonter les exceptions moteur avec statut runtime clair.

Tests :

```text
tests/translation_engines/test_ct2_engine_decodes_hypotheses.py
tests/translation_engines/test_ct2_engine_batches_requests.py
tests/translation_engines/test_ct2_engine_marian_no_bad_target_prefix.py
tests/translation_engines/test_ct2_engine_m2m100_uses_language_prefix.py
tests/translation_engines/test_ct2_engine_reports_missing_dependencies.py
```

Les tests doivent mocker :

```text
ctranslate2.Translator
transformers.AutoTokenizer
```

Ne pas exiger les vrais modèles pour les tests unitaires.

---

## Bloc B — Corriger model registry

Fichier :

```text
translation_engines/model_registry.py
```

Todo :

* [ ] Résoudre les chemins robustement.
* [ ] Lire `model_inventory.json`.
* [ ] Ajouter `select_model(source_lang, target_lang, preferred_model=None)`.
* [ ] Choisir OPUS pour `en → fr`.
* [ ] Fallback M2M100 si OPUS absent.
* [ ] Valider `model_dir`.
* [ ] Valider `tokenizer_dir`.
* [ ] Valider `source_langs`, `target_langs`, `backend`, `family`.
* [ ] Healthcheck clair.

Tests :

```text
tests/translation_engines/test_model_registry_selects_opus_for_en_fr.py
tests/translation_engines/test_model_registry_fallbacks_to_m2m100.py
tests/translation_engines/test_model_registry_resolves_inventory_relative_paths.py
```

---

## Bloc C — Améliorer les CLI

Fichiers :

```text
tools/check_translation_engine.py
tools/test_placeholder_roundtrip.py
tools/run_translation_trial.py
tools/run_batch_translation_trial.py
tools/run_document_trial.py
```

Todo :

* [ ] Ajouter `--inventory`.
* [ ] Ajouter `--model`.
* [ ] Ajouter `--source-lang`.
* [ ] Ajouter `--target-lang`.
* [ ] Ajouter `--device`.
* [ ] Ajouter `--compute-type`.
* [ ] Ajouter `--batch-size`.
* [ ] Ajouter `--max-input-tokens`.
* [ ] Ajouter `--fail-on runtime|quality|publication|any`.

Commande cible :

```bash
python3 tools/check_translation_engine.py \
  --engine ct2 \
  --inventory ai_models/translation/model_inventory.json \
  --model opus_mt_tc_big_en_fr \
  --source-lang en \
  --target-lang fr \
  --device cpu \
  --compute-type int8
```

---

## Bloc D — Placeholders

Fichiers :

```text
translation_engines/placeholder_policy.py
pagetranslate/protection.py
tools/test_placeholder_roundtrip.py
```

Todo :

* [ ] Restaurer toléramment tous les formats.
* [ ] Détecter placeholder manquant.
* [ ] Détecter placeholder dupliqué.
* [ ] Détecter placeholder non restauré.
* [ ] Détecter corruption de format.
* [ ] Écrire `placeholder_policy.json` si un format est choisi après test.
* [ ] Ajouter métrique `placeholder_corruption_count`.

Tests :

```text
tests/translation_engines/test_placeholder_restore_ascii_xml_variants.py
tests/translation_engines/test_placeholder_policy_roundtrip_contract.py
```

---

## Bloc E — Translation memory

Créer :

```text
translation_engines/translation_memory.py
```

Todo :

* [ ] Charger `translation_memory.jsonl`.
* [ ] Ignorer les entrées non validées si `validated != true`.
* [ ] Lookup exact.
* [ ] Lookup normalisé simple.
* [ ] Respecter `source_lang`, `target_lang`, `domain`.
* [ ] Si hit validé, ne pas appeler le modèle.
* [ ] Ajouter `memory_hit_count`.
* [ ] Ajouter `model_call_count`.

Tests :

```text
tests/translation_engines/test_translation_memory_exact_hit_skips_engine.py
tests/translation_engines/test_translation_memory_ignores_unvalidated.py
```

---

## Bloc F — Profils

Créer :

```text
translation_engines/profile_store.py
```

Todo :

* [ ] Lire `translation_profiles.json`.
* [ ] Lire `style_tone_profiles.json`.
* [ ] Injecter profil dans `context`.
* [ ] Ajouter profil dans `engine_trace`.
* [ ] Ne pas planter si fichiers absents.
* [ ] Utiliser profil pour choisir glossaire ou QA policy.

Tests :

```text
tests/translation_engines/test_profile_store_loads_profiles.py
tests/pagetranslate/test_translation_profile_in_engine_context.py
```

---

## Bloc G — Statuts qualité

Fichiers :

```text
tools/run_translation_trial.py
tools/run_batch_translation_trial.py
pagetranslate/quality.py
```

Todo :

* [ ] Agréger `translation_runtime_status`.
* [ ] Agréger `linguistic_quality_status`.
* [ ] Agréger `publication_readiness_status`.
* [ ] Ne pas confondre runtime KO et pipeline KO.
* [ ] `needs_review_count > 0` doit mettre `linguistic_quality_status = review`.
* [ ] Erreur moteur doit mettre `translation_runtime_status = ko`.
* [ ] Ajouter option `--fail-on`.

Tests :

```text
tests/functional/test_run_translation_trial_runtime_ko_status.py
tests/functional/test_needs_review_sets_quality_review.py
tests/functional/test_pipeline_ok_runtime_ko_possible.py
```

---

# 7. Commandes de validation après rev_07

## Tests sans modèles réels

```bash
python3 -m pytest -q \
  tests/pageprint \
  tests/pagetranslate \
  tests/functional \
  tests/pipelines \
  tests/translation_engines
```

## Healthcheck local avec vrais modèles

```bash
python3 tools/check_translation_engine.py \
  --engine ct2 \
  --inventory ai_models/translation/model_inventory.json \
  --model opus_mt_tc_big_en_fr \
  --source-lang en \
  --target-lang fr \
  --device cpu \
  --compute-type int8
```

## Smoke test texte simple

Créer si absent :

```text
tools/translate_text_smoke.py
```

Commande cible :

```bash
python3 tools/translate_text_smoke.py \
  --engine ct2 \
  --inventory ai_models/translation/model_inventory.json \
  --model opus_mt_tc_big_en_fr \
  --source-lang en \
  --target-lang fr \
  --text "Hidden layers"
```

Résultat attendu approximatif :

```text
Couches cachées
```

## Placeholder roundtrip

```bash
python3 tools/test_placeholder_roundtrip.py \
  --engine ct2 \
  --inventory ai_models/translation/model_inventory.json \
  --model opus_mt_tc_big_en_fr \
  --source-lang en \
  --target-lang fr
```

Attendu :

```text
placeholder_corruption_rate = 0
```

## Trial PAGEPRINT/PAGETRANSLATE + moteur

```bash
python3 tools/run_translation_trial.py input_data.json \
  --engine ct2 \
  --inventory ai_models/translation/model_inventory.json \
  --model opus_mt_tc_big_en_fr \
  --source-lang en \
  --target-lang fr \
  --batch-size 8
```

---

# 8. Critères d’acceptation rev_07

`rev_07` est accepté seulement si :

```text
1. Les tests ciblés existants passent.
2. Les nouveaux tests translation_engines passent sans modèles réels.
3. check_translation_engine détecte correctement les modèles locaux.
4. OPUS est sélectionné par défaut pour en→fr.
5. M2M100 est fallback si OPUS indisponible.
6. ct2_engine traduit réellement une phrase simple.
7. translate_batch fait un vrai batch.
8. placeholders sont restaurés.
9. translation_memory peut éviter un appel modèle.
10. runtime KO remonte dans translation_runtime_status.
11. needs_review remonte dans linguistic_quality_status.
12. les CLI acceptent inventory/model/source/target/device/compute/batch.
```

---

# 9. Ce qu’il ne faut pas faire

Ne pas refondre :

```text
PAGEPRINT
PAGETRANSLATE
translation_plan
semantic_builder
view_compiler
```

Ne pas réintégrer l’ancien gros `translator.py` comme moteur principal obligatoire.

On peut s’en inspirer, mais `translation_engines/ct2_engine.py` doit devenir le nouveau runtime propre.

Ne pas faire d’essai document complet avant :

```text
healthcheck OPUS OK
traduction phrase simple OK
placeholder roundtrip OK
trial page simple OK
```

---

# 10. Résumé final pour Claude

La mission immédiate est :

```text
Transformer rev_06 en rev_07 :
un vrai runtime IA CTranslate2 robuste,
branché sur ai_models/translation/,
capable de charger OPUS EN-FR et M2M100,
de choisir le bon modèle,
de traduire en batch,
de préserver les tokens,
d’utiliser mémoire/profils,
et de produire des statuts/logs fiables.
```

Le reste du pipeline documentaire est suffisamment stable pour l’instant.

Ne pas repartir sur une nouvelle refonte conceptuelle. Continuer par durcissement du runtime IA.

