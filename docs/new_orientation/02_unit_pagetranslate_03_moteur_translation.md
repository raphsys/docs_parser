# Verdict sur `rev_06`

`rev_06` est une **bonne étape d’industrialisation du moteur**, mais je ne le considère pas encore comme prêt pour des essais IA sérieux sur documents complets.

Décision nette :

```text
PAGEPRINT / PAGETRANSLATE : ne pas refondre.
Couche moteur : refonte ciblée encore nécessaire.
rev_06 = bootstrap moteur.
rev_07 = runtime IA réellement fiable.
```

Le projet a maintenant la bonne forme générale :

```text
translation_engines/
tools/check_translation_engine.py
tools/test_placeholder_roundtrip.py
tools/run_translation_trial.py
tools/run_batch_translation_trial.py
ct2_engine.py
model_registry.py
placeholder_policy.py
terminology typée
batch path
engine_trace
quality statuses
```

Mais il reste des problèmes importants dans le vrai chemin CTranslate2, les placeholders, la mémoire de traduction, les profils, les tests IA et les statuts d’erreur runtime.

---

# 1. Vérifications effectuées

J’ai vérifié l’archive `rev_06.zip`.

Compilation ciblée :

```bash
python3 -m py_compile translation_engines/*.py pagetranslate/*.py pageprint/*.py \
  pageprint/detection/*.py pageprint/evidence/*.py pageprint/structure_builders/*.py \
  pipelines/*.py tools/*.py
```

Résultat :

```text
OK
```

Tests ciblés :

```bash
python3 -m pytest -q tests/pageprint tests/pagetranslate tests/functional tests/pipelines
```

Résultat :

```text
72 passed
```

Test global :

```bash
python3 -m pytest -q
```

Résultat :

```text
KO à cause du legacy :
ModuleNotFoundError: rapidocr_onnxruntime
```

Ce n’est pas un blocage pour la couche moteur `rev_06`, mais il faut encore isoler les tests legacy.

Healthcheck CT2 dans mon environnement :

```bash
python3 tools/check_translation_engine.py --engine ct2
```

Résultat :

```json
{
  "status": "missing",
  "engine": "ct2",
  "error": "RuntimeError: No translation model available in registry",
  "registry": {
    "inventory_path": "ai_models/translation/model_inventory.json",
    "model_count": 0,
    "available_model_count": 0
  }
}
```

C’est normal dans l’archive : le dossier `ai_models/translation/` n’est pas fourni. Chez toi, il existe localement. Donc il faudra valider cette partie directement sur ta machine.

---

# 2. Ce qui est bon dans `rev_06`

## 2.1 La structure moteur est enfin présente

On a maintenant :

```text
translation_engines/
├── base.py
├── request.py
├── model_registry.py
├── ct2_engine.py
├── engine_health.py
├── factory.py
├── placeholder_policy.py
├── mock_engine.py
├── rule_engine.py
├── local_model_engine.py
└── external_model_engine.py
```

C’est le bon début.

## 2.2 La factory supporte les moteurs nécessaires

```python
mock
echo
prefix
rule
ct2
local
external
```

Donc le pipeline peut choisir un moteur via :

```bash
TRANSLATION_ENGINE=ct2
```

ou :

```bash
--engine ct2
```

selon l’outil utilisé.

## 2.3 Le batch path existe

`PageTranslationBuilder` sait maintenant utiliser :

```python
translator.translate_batch(...)
```

quand le moteur le supporte.

C’est obligatoire pour les documents réels.

## 2.4 Les statuts qualité sont mieux séparés

On trouve maintenant :

```text
pipeline_status
translation_runtime_status
linguistic_quality_status
publication_readiness_status
```

C’est conceptuellement correct.

## 2.5 La terminologie typée est amorcée

Le système distingue maintenant :

```text
preserve
preferred_translation
contextual
```

C’est mieux que l’ancienne liste plate de termes verrouillés.

## 2.6 Les traces moteur existent

Chaque unité peut maintenant contenir :

```json
{
  "engine_trace": {
    "engine": "...",
    "model_name": "...",
    "model_family": "...",
    "protected_source_text": "...",
    "raw_engine_output": "...",
    "restored_output": "...",
    "post_glossary_output": "...",
    "latency_ms": 0,
    "input_token_count": 0,
    "output_token_count": 0,
    "truncated": false
  }
}
```

C’est indispensable pour analyser les vraies erreurs IA.

---

# 3. Problèmes critiques restants

## P0 — `ct2_engine.py` n’est pas encore fiable pour un vrai modèle CTranslate2

Le fichier existe, mais l’implémentation réelle est encore fragile.

### Problème 1 — décodage CTranslate2 probablement incorrect

Le code fait actuellement :

```python
result = translator.translate_batch(...)
raw_output = " ".join(result[0]) if result and isinstance(result[0], list) else str(result[0])
```

Or CTranslate2 retourne généralement des objets de résultat, avec des hypothèses :

```python
result[0].hypotheses[0]
```

Il faut décoder les tokens générés via le tokenizer, par exemple :

```python
output_tokens = result[0].hypotheses[0]
output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
translated = tokenizer.decode(output_ids, skip_special_tokens=True)
```

ou une variante adaptée au tokenizer exact.

Donc le healthcheck peut charger le modèle, mais cela ne garantit pas encore que la traduction produite sera lisible.

### Problème 2 — `target_prefix=[target_lang]` est probablement faux

Le code utilise :

```python
target_prefix=[target_lang]
```

Ce n’est pas suffisant.

Pour Marian / OPUS, il ne faut généralement pas envoyer simplement `"fr"` comme target prefix.

Pour M2M100, il faut une vraie gestion des codes langues :

```text
source_lang
target_lang
lang token
forced BOS / target prefix adapté au tokenizer
```

Donc `ct2_engine.py` doit avoir une logique par famille :

```text
family = marian / opus
family = m2m100
family = nllb
```

Actuellement, cette distinction n’est pas assez solide.

### Problème 3 — le batch n’est pas vraiment batché

`translate_batch()` reçoit une liste de requêtes, mais boucle ensuite requête par requête :

```python
for req in requests:
    translator.translate_batch([tokens])
```

Donc ce n’est pas du vrai batch.

Il faut regrouper :

```python
batch_tokens = [...]
translator.translate_batch(batch_tokens, ...)
```

puis décoder toutes les sorties.

### Problème 4 — `model_registry.pick()` n’utilise pas source/target dans le chemin réel

Dans `CTranslate2Engine._resolve_entry()`, on voit :

```python
entry = self.registry.pick(engine_name=self.model_name)
```

Mais le choix de modèle devrait dépendre de :

```text
source_lang
target_lang
preferred_model
availability
priority
family
```

Actuellement, si `model_name` est absent, il choisit juste le premier modèle disponible. Pour `en → fr`, on veut :

```text
1. opus_mt_tc_big_en_fr
2. m2m100_418m fallback
```

Il faut donc résoudre le modèle **au moment de la requête**, ou au moins au premier batch avec `source_lang` / `target_lang`.

### Problème 5 — chemins relatifs fragiles

`model_registry.py` utilise :

```python
Path(model_dir)
Path(tokenizer_dir)
```

Si le script est lancé hors racine projet, les chemins peuvent casser.

Il faut résoudre les chemins relativement à :

```text
- la racine du projet
- ou le dossier contenant model_inventory.json
- ou une variable TRANSLATION_MODELS_ROOT
```

---

# 4. Problème P0 — les outils CLI ne donnent pas assez de contrôle

`tools/check_translation_engine.py` accepte seulement :

```bash
--engine
```

Mais pour les vrais essais il faut :

```bash
--inventory
--model
--source-lang
--target-lang
--device
--compute-type
--batch-size
```

Même problème pour :

```text
tools/run_translation_trial.py
tools/run_batch_translation_trial.py
tools/run_document_trial.py
tools/test_placeholder_roundtrip.py
```

Actuellement, on dépend trop des variables d’environnement.

Il faut pouvoir faire :

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

# 5. Problème P0 — les erreurs runtime ne remontent pas assez dans le statut global

Dans `run_translation_trial.py`, le statut global est surtout piloté par les erreurs fonctionnelles et le fallback.

Mais si une page a :

```text
translation_runtime_status = ko
```

il faut que le résultat global ait aussi :

```text
translation_runtime_status = ko
```

et que l’exit code puisse échouer selon le mode demandé.

Aujourd’hui, un échec moteur risque de ne pas être traité assez sévèrement dans le résumé global.

Correction attendue :

```python
if any(page["translation_runtime_status"] == "ko" for page in page_results):
    runtime_status = "ko"
```

Et ensuite :

```text
functional_status peut rester ok
translation_runtime_status doit être ko
linguistic_quality_status peut être unknown ou ko
```

Il faut éviter de mélanger :

```text
pipeline OK
moteur KO
qualité KO
```

---

# 6. Problème P1 — `restore_text()` est tolérant surtout pour le placeholder Unicode

`placeholder_policy.py` choisit :

```text
ct2 → ascii_xml
```

Donc le placeholder devient :

```xml
<nt id="PT0001"/>
```

Mais `restore_text()` n’a une restauration vraiment tolérante que pour le format Unicode :

```python
⟦PT0001⟧
```

Pour `ascii_xml`, il remplace surtout si le modèle renvoie exactement :

```xml
<nt id="PT0001"/>
```

Un vrai modèle peut produire :

```text
< nt id = "PT0001" />
<nt id='PT0001' />
<nt id=PT0001/>
```

La restauration doit être tolérante pour **tous les styles**, pas seulement Unicode.

---

# 7. Problème P1 — la mémoire de traduction n’est pas réellement intégrée

Tu as localement :

```text
translation_memory.jsonl
```

Mais dans le code inspecté, je ne vois pas de module robuste :

```text
translation_engines/translation_memory.py
```

Ni de lookup clairement intégré avant l’appel moteur.

Or l’ordre correct doit être :

```text
translation_plan item
→ protection
→ terminology
→ translation_memory exact match
→ moteur IA
→ restauration
→ QA
```

La mémoire doit éviter d’appeler le modèle si une traduction validée existe déjà.

---

# 8. Problème P1 — les profils ne sont pas réellement exploités

Tu as :

```text
translation_profiles.json
style_tone_profiles.json
```

Mais le runtime moteur ne les exploite pas encore vraiment.

Il faut charger ces fichiers et injecter dans :

```text
context
engine_trace
translation_profile
terminology
style/tone
prompt/config si moteur LLM futur
```

Même avec CTranslate2, ces profils servent à :

```text
choisir modèle
choisir glossaire
choisir stratégie QA
choisir style de post-édition
```

---

# 9. Problème P1 — les tests ne couvrent pas le vrai CT2

Les tests ciblés passent :

```text
72 passed
```

Mais ils testent surtout :

```text
contrats
mock/rule
batch avec faux moteur
placeholder logique
```

Il manque des tests avec faux `ctranslate2.Translator` et faux tokenizer pour vérifier :

```text
- décodage result.hypotheses[0]
- batch réel
- target prefix Marian
- target prefix M2M100
- compteur token
- troncature
- erreur modèle/tokenizer
```

Ces tests n’ont pas besoin des vrais modèles. Ils doivent mocker `ctranslate2` et `transformers`.

---

# 10. Conclusions

## Ce que `rev_06` valide

```text
- architecture moteur présente ;
- factory présente ;
- registry présent ;
- ct2_engine présent ;
- batch path présent ;
- traces moteur présentes ;
- statuts qualité séparés ;
- tests ciblés OK ;
- PAGEPRINT/PAGETRANSLATE toujours stables.
```

## Ce que `rev_06` ne valide pas encore

```text
- vraie traduction CTranslate2 fiable ;
- vrai décodage des sorties CT2 ;
- vrai batch CT2 ;
- gestion correcte Marian / M2M100 ;
- placeholder roundtrip avec vrai modèle ;
- mémoire de traduction ;
- profils de traduction/style ;
- sélection robuste modèle selon source/target ;
- essais documents avec vrai modèle IA.
```

Donc :

```text
rev_06 = bon bootstrap.
rev_06 ≠ runtime IA validé.
```

---

# 11. Décision

On ne touche pas à la refonte documentaire maintenant.

La prochaine étape doit être :

```text
rev_07 = durcissement du runtime IA CTranslate2.
```

Objectif de `rev_07` :

```text
Un modèle réel dans ai_models/translation/ doit traduire une page test
avec :
- décodage correct ;
- placeholders restaurés ;
- batch réel ;
- glossaire typé ;
- mémoire de traduction ;
- logs exploitables ;
- statuts séparés fiables.
```

---

# 12. Directives `rev_07` pour Codex

## Bloc A — Corriger `ct2_engine.py`

### Todo

* [ ] Remplacer le décodage actuel par un décodage réel CTranslate2 :

```python
results = translator.translate_batch(batch_tokens, ...)
output_tokens = results[i].hypotheses[0]
translated = tokenizer.decode(
    tokenizer.convert_tokens_to_ids(output_tokens),
    skip_special_tokens=True
)
```

ou une fonction équivalente adaptée au tokenizer.

* [ ] Implémenter une fonction :

```python
_decode_ct2_result(result, tokenizer) -> str
```

* [ ] Implémenter une fonction :

```python
_encode_source(text, tokenizer, family, source_lang) -> list[str]
```

* [ ] Implémenter une fonction :

```python
_target_prefix(tokenizer, family, target_lang) -> list[str] | None
```

* [ ] Séparer les familles :

```text
marian / opus
m2m100
nllb
generic
```

* [ ] Faire un vrai batch :

```python
batch_tokens = [...]
results = translator.translate_batch(batch_tokens, target_prefix=...)
```

* [ ] Ne plus appeler `translate_batch()` une fois par requête.

* [ ] Ajouter `input_token_count` et `output_token_count`.

* [ ] Ajouter `truncated=True` si troncature.

* [ ] Retourner `raw_output` propre.

---

## Bloc B — Corriger `model_registry.py`

### Todo

* [ ] Ajouter résolution robuste des chemins :

```text
inventory-relative
project-root-relative
absolute
TRANSLATION_MODELS_ROOT
```

* [ ] Ajouter :

```python
select_model(source_lang, target_lang, preferred_model=None)
```

* [ ] Appliquer priorité :

```text
1. preferred_model si compatible et disponible
2. modèle spécialisé source/target
3. modèle fallback multilingue
4. erreur claire
```

* [ ] Pour `en → fr`, choisir par défaut :

```text
opus_mt_tc_big_en_fr
```

* [ ] Si OPUS indisponible, choisir :

```text
m2m100_418m
```

* [ ] `healthcheck()` doit indiquer :

```text
available
compatible
missing_model_dir
missing_tokenizer_dir
unsupported_lang_pair
```

---

## Bloc C — Améliorer les CLI

### Todo

Ajouter à :

```text
tools/check_translation_engine.py
tools/test_placeholder_roundtrip.py
tools/run_translation_trial.py
tools/run_batch_translation_trial.py
tools/run_document_trial.py
```

les options :

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

Exemple obligatoire :

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

## Bloc D — Restauration placeholder robuste

### Todo

* [ ] Rendre `restore_text()` tolérant pour :

```text
unicode_bracket
ascii_xml
plain_ascii
at_token
```

* [ ] Pour chaque protection, déduire l’id `PT0001`, puis restaurer toutes les variantes :

```text
⟦PT0001⟧
<nt id="PT0001"/>
< nt id = "PT0001" />
<nt id='PT0001' />
[[PT0001]]
@@PT0001@@
PT0001
```

* [ ] Ajouter `placeholder_corruption_count` réel :

```text
placeholder manquant
placeholder dupliqué
placeholder non restauré
placeholder déplacé si ordre strict requis
```

---

## Bloc E — Intégrer `translation_memory.jsonl`

### Todo

Créer :

```text
translation_engines/translation_memory.py
```

Fonctions :

```python
load_translation_memory(path) -> TranslationMemory
lookup_exact(source, source_lang, target_lang, domain=None) -> str | None
lookup_normalized(...) -> str | None
```

Format :

```json
{"source_lang":"en","target_lang":"fr","source":"Hidden layers","target":"Couches cachées","domain":"deep_learning","validated":true}
```

Ordre d’exécution :

```text
memory hit validé → pas d’appel modèle
memory miss → moteur IA
```

Ajouter dans `engine_trace` :

```json
{
  "memory_hit": true,
  "memory_source": "exact"
}
```

---

## Bloc F — Charger les profils

### Todo

Créer ou renforcer :

```text
translation_engines/profile_store.py
```

Lire :

```text
ai_models/translation/translation_profiles.json
ai_models/translation/style_tone_profiles.json
```

Injecter dans :

```text
translation_profile
engine_context
engine_trace
quality policy
terminology path
```

---

## Bloc G — Corriger les statuts trial

### Todo

Dans `tools/run_translation_trial.py` :

* [ ] Agréger réellement `translation_runtime_status` depuis les pages.
* [ ] Si une page est `runtime_status=ko`, alors top-level `translation_runtime_status=ko`.
* [ ] Ne pas confondre :

```text
functional_status
translation_runtime_status
linguistic_quality_status
publication_readiness_status
```

* [ ] Ajouter option :

```bash
--fail-on runtime|quality|publication|any
```

Exemple :

```bash
--fail-on runtime
```

échoue si le moteur plante, même si le pipeline est correct.

---

## Bloc H — Tests indispensables

Créer :

```text
tests/translation_engines/
```

Tests à ajouter :

```text
test_model_registry_selects_opus_for_en_fr.py
test_model_registry_fallbacks_to_m2m100.py
test_model_registry_resolves_inventory_relative_paths.py
test_ct2_engine_decodes_hypotheses.py
test_ct2_engine_batches_requests.py
test_ct2_engine_marian_no_bad_target_prefix.py
test_ct2_engine_m2m100_uses_language_prefix.py
test_ct2_engine_reports_missing_dependencies.py
test_placeholder_restore_ascii_xml_variants.py
test_translation_memory_exact_hit_skips_engine.py
test_run_translation_trial_runtime_ko_status.py
```

Les tests CT2 doivent mocker :

```text
ctranslate2.Translator
transformers.AutoTokenizer
```

Pas besoin des modèles réels pour ces tests.

---

# 13. Essais réels à faire après `rev_07`

Une fois les corrections faites :

## Étape 1 — Healthcheck OPUS

```bash
python3 tools/check_translation_engine.py \
  --engine ct2 \
  --inventory ai_models/translation/model_inventory.json \
  --model opus_mt_tc_big_en_fr \
  --source-lang en \
  --target-lang fr
```

Attendu :

```json
{
  "status": "ok",
  "selected_model": "opus_mt_tc_big_en_fr"
}
```

## Étape 2 — Phrase simple

```bash
python3 tools/translate_text_smoke.py \
  --engine ct2 \
  --model opus_mt_tc_big_en_fr \
  --text "Hidden layers"
```

Attendu approximatif :

```text
Couches cachées
```

## Étape 3 — Placeholder

```bash
python3 tools/test_placeholder_roundtrip.py \
  --engine ct2 \
  --model opus_mt_tc_big_en_fr \
  --source-lang en \
  --target-lang fr
```

Attendu :

```text
placeholder_corruption_rate = 0
```

## Étape 4 — Page body

```bash
python3 tools/run_translation_trial.py input_body.json \
  --engine ct2 \
  --model opus_mt_tc_big_en_fr \
  --source-lang en \
  --target-lang fr
```

## Étape 5 — Page TOC

Même chose sur TOC.

## Étape 6 — 10 pages variées

```bash
python3 tools/run_batch_translation_trial.py audit_folder \
  --engine ct2 \
  --model opus_mt_tc_big_en_fr \
  --source-lang en \
  --target-lang fr
```

---

# 14. Conclusion nette

`rev_06` est bonne sur le plan structurel, mais pas encore assez sûre pour lancer des essais IA sérieux sur documents réels complets.

La bonne décision :

```text
Ne pas refaire PAGEPRINT.
Ne pas refaire PAGETRANSLATE.
Corriger maintenant la couche CT2/runtime.
```

Statut :

```text
rev_06 = bootstrap moteur IA.
rev_07 = runtime IA fiable.
```

Après `rev_07`, on pourra réellement dire :

```text
on branche OPUS-MT EN→FR comme moteur principal,
M2M100 comme fallback,
et on lance des essais IA sur documents.
```

Pour l’instant, je recommande seulement :

```text
essais mock/rule : oui
healthcheck local chez toi : oui
traduction IA document réel : attendre correction CT2 rev_07
```

La priorité absolue est :

```text
ct2_engine.py : encoder correctement, traduire réellement en batch, décoder correctement, gérer Marian/M2M100, remonter les erreurs runtime.
```

