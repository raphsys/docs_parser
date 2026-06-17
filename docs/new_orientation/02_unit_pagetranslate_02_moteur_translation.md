# Verdict sur `rev_05`

`rev_05` est une **bonne première mise en place du moteur de traduction**, mais ce n’est pas encore le vrai branchement IA final.

Décision nette :

```text id="ot4i6n"
On ne refait pas PAGEPRINT/PAGETRANSLATE maintenant.
On garde rev_05 comme socle.
La prochaine étape est une refonte ciblée de la couche moteur IA :
rev_06 = Translation Runtime réel branché sur ai_models/translation/.
```

Donc :

```text id="ofhy6d"
PAGEPRINT/PAGETRANSLATE : assez mûrs pour essais.
Couche moteur : encore trop prototype.
Essais documents : oui, mais seulement avec mock/rule pour pipeline, puis IA après runtime propre.
```

---

# 1. Ce que `rev_05` réussit

J’ai inspecté l’archive et exécuté les vérifications cœur :

```bash id="e0ts7y"
python3 -m py_compile translation_engines/*.py tools/*.py pagetranslate/*.py pageprint/*.py pageprint/structure_builders/*.py pipelines/*.py
python3 -m pytest -q tests/pageprint tests/pagetranslate tests/functional tests/pipelines
```

Résultat :

```text id="mfd5wp"
62 passed
compilation OK
```

La livraison contient bien :

```text id="9qc7ge"
translation_engines/
tools/run_translation_trial.py
tools/run_document_trial.py
tests/pagetranslate/test_translation_engines.py
tests/functional/test_translation_trial_runner.py
```

Donc le socle moteur existe.

---

# 2. Analyse de la couche moteur actuelle

## 2.1 Ce qui est bon

`translation_engines/` contient :

```text id="68vjao"
mock_engine.py
rule_engine.py
local_model_engine.py
external_model_engine.py
factory.py
```

La factory fonctionne :

```python id="8kz9ik"
create_translation_engine("mock")
create_translation_engine("rule")
create_translation_engine("local")
create_translation_engine("external")
```

`PAGETRANSLATE` appelle le moteur via `TranslatorBridge`, et le contrat minimal est sain :

```python id="cbczpc"
translate(text, source_lang, target_lang, context) -> str
```

`run_translation_trial.py` fait aussi une bonne chose : il n’appelle le moteur qu’après un audit fonctionnel PAGEPRINT/PAGETRANSLATE.

C’est le bon principe :

```text id="ui3m6v"
préflight PAGEPRINT/PAGETRANSLATE
→ moteur
→ QA
→ logs par unité
```

---

## 2.2 Ce qui est encore insuffisant

Les moteurs IA réels ne sont pas encore branchés.

Actuellement :

```text id="dlbqab"
LocalModelEngine = wrapper vide autour d’un callable déjà injecté
ExternalModelEngine = wrapper vide autour d’un client déjà injecté
```

Ils ne savent pas encore charger :

```text id="hqf8m4"
ai_models/translation/
model_inventory.json
tokenizer
CTranslate2 model
lang codes
device
threads
max length
batching
```

Donc `rev_05` valide :

```text id="q93xll"
le passage de texte vers un moteur
```

mais pas encore :

```text id="rugxka"
le runtime réel d’un modèle IA local.
```

---

# 3. Point très important : `ai_models/translation/`

Tu as précisé que le moteur sera un modèle IA branchable depuis :

```text id="31gb8g"
ai_models/translation/
```

Ce dossier n’est pas dans `rev_05`, donc c’est normal que je ne teste pas le vrai modèle.

Mais le code actuel ne contient pas encore un loader propre pour ce dossier. Il existe de vieux scripts et l’ancien `translator.py` sait déjà beaucoup de choses sur :

```text id="aaz5ld"
CTranslate2
CT2_MODEL_DIR
CT2_TOKENIZER_DIR
model_inventory.json
m2m100
nllb
opus-mt
```

Mais la nouvelle couche `translation_engines/` n’utilise pas encore cette intelligence.

Décision :

```text id="8mpebm"
Ne pas réutiliser directement le gros DocumentTranslator comme cœur final.
Extraire/recoder proprement son runtime CT2 dans translation_engines/ct2_engine.py.
```

`translator.py` peut servir de référence, mais il est trop massif pour être le nouveau moteur propre.

---

# 4. Problème conceptuel majeur : “moteur branché” ≠ “traduction fiable”

Avec `rev_05`, un essai `mock` peut donner :

```text id="9k6twk"
functional_status = ok
```

même si le texte n’est pas vraiment traduit.

Exemple :

```text id="zosogg"
source : This sentence should be translated.
mock   : FR::This sentence should be translated.
```

Techniquement, le pipeline fonctionne. Linguistiquement, ce n’est pas une traduction.

Donc il faut séparer 3 statuts :

```text id="383nsp"
pipeline_status
translation_runtime_status
linguistic_quality_status
```

Actuellement, `functional_status=ok` veut dire :

```text id="vasc50"
le pipeline n’a pas cassé
```

Il ne veut pas dire :

```text id="j1jerq"
la traduction est bonne
```

C’est fondamental.

---

# 5. Problème important : `needs_review` ne bloque pas encore

J’ai testé le moteur `rule` sur une TOC contenant :

```text id="ghwzio"
CONTENTS
Image classification using MLP
Hidden layers
```

Résultat :

```text id="yqjwyd"
functional_status = ok
needs_review_count = 2
```

Pourquoi ?

Parce que :

```text id="rjs2v7"
CONTENTS reste CONTENTS
Image classification using MLP reste Image classification using MLP
Hidden layers devient Couches cachees
```

Le pipeline considère ça fonctionnellement OK parce qu’il n’y a pas d’erreur structurelle. C’est logique.

Mais pour un vrai test moteur, il faut un statut supplémentaire :

```json id="61repi"
{
  "pipeline_status": "ok",
  "translation_quality_status": "ko",
  "reason": "needs_review_count_gt_0"
}
```

Sinon on risque de déclarer réussie une traduction qui n’est qu’un passage technique.

---

# 6. Problème très important : terminologie trop verrouillée

Dans `pagetranslate/terminology.py`, les termes verrouillés par défaut sont :

```text id="zvy1ln"
MLP
CNN
ReLU
Softmax
dropout
pooling
precision
recall
F-score
SQL
API
OCR
```

C’est trop brutal.

Certains termes doivent être préservés :

```text id="4fi35n"
MLP
CNN
ReLU
Softmax
SQL
API
OCR
F-score selon contexte
```

Mais d’autres doivent souvent être traduits ou contextualisés :

```text id="uya8jk"
precision → précision
recall → rappel
pooling → pooling / sous-échantillonnage selon style
dropout → dropout / abandon / couche de dropout selon contexte
```

Actuellement, `precision` et `recall` sont traités comme tokens à protéger. C’est dangereux pour un livre de deep learning.

Il faut passer de :

```text id="rvu9k8"
locked_terms = liste plate
```

à :

```json id="pdwa7x"
{
  "MLP": {
    "policy": "preserve"
  },
  "CNN": {
    "policy": "preserve"
  },
  "precision": {
    "policy": "preferred_translation",
    "target": "précision"
  },
  "recall": {
    "policy": "preferred_translation",
    "target": "rappel"
  },
  "pooling": {
    "policy": "contextual",
    "preferred": "pooling",
    "alternatives": ["sous-échantillonnage"]
  },
  "dropout": {
    "policy": "contextual",
    "preferred": "dropout",
    "alternatives": ["abandon"]
  }
}
```

C’est un chantier prioritaire avant les vrais essais IA.

---

# 7. Problème technique : les placeholders peuvent être abîmés par un vrai modèle IA

Le système utilise actuellement :

```text id="a9qlwn"
⟦PT0001⟧
```

C’est lisible, mais un modèle NMT/LLM peut :

```text id="k4lkl8"
modifier les caractères spéciaux
ajouter des espaces
traduire PT
supprimer une bracket
déplacer le placeholder
dupliquer le placeholder
```

La restauration tolérante est déjà une bonne idée, mais elle ne suffit pas.

Pour un modèle CTranslate2, il faut tester les placeholders avec le vrai tokenizer.

Il faut créer une `PlaceholderPolicy` :

```text id="3q3jug"
unicode_bracket
ascii_xml
plain_ascii
special_token_compatible
```

Exemples à tester :

```text id="fjb1ju"
⟦PT0001⟧
<nt id="PT0001"/>
[[PT0001]]
@@PT0001@@
```

Critère :

```text id="vyua5a"
Le modèle doit restituer le placeholder sans corruption dans >99 % des cas.
```

---

# 8. Problème de performance : pas encore de batch translation

L’interface actuelle traduit unité par unité :

```python id="m9ep3c"
translate(text, source_lang, target_lang, context)
```

Pour un vrai modèle IA local, cela va être lent.

Il faut ajouter une interface optionnelle :

```python id="mlujd5"
translate_batch(items: list[TranslationRequest]) -> list[TranslationResponse]
```

Avec fallback :

```text id="ffhexx"
si moteur supporte batch → batch
sinon → boucle translate()
```

C’est indispensable pour :

```text id="6v5tki"
10 pages
30 pages
document complet
```

Sinon le test complet sera trop lent.

---

# 9. Problème de logs : on ne voit pas assez le vrai dialogue moteur

`run_translation_trial.py` logge :

```text id="4ieq2o"
source_text
protected_tokens
translated_text
status
qa
```

C’est bien, mais insuffisant pour débugger un vrai modèle.

Il faut aussi logguer :

```text id="qg639h"
protected_source_text
raw_engine_output
restored_output
post_glossary_output
latency_ms
engine_name
model_name
model_family
token_count_in
token_count_out
truncation_applied
error
```

Sans ça, on ne saura pas si une erreur vient :

```text id="6x7uzy"
du modèle
du placeholder
du glossaire
de la restauration
du post-traitement
du QA
```

---

# 10. Analyse des conclusions Codex

Les conclusions Codex disent en substance :

```text id="a3yvo2"
Compilation OK
62 passed
Batch audit OK
body/toc trial OK
document trial pages 1/2 OK avec mock
global pytest legacy encore pollué
```

Je suis d’accord avec l’essentiel.

Mais je nuance fortement :

```text id="st4gj8"
Ces conclusions valident le branchement technique.
Elles ne valident pas encore la qualité d’un vrai moteur IA.
```

Les essais `mock` et `rule` sont utiles pour :

```text id="xad2it"
protection
projection
QA plumbing
debug
translation_plan
```

Mais ils ne prouvent pas :

```text id="qixztr"
qualité linguistique
fidélité sémantique
style
terminologie technique
résistance aux placeholders
performance sur document réel
```

Donc la prochaine étape n’est pas une refonte PAGEPRINT. C’est une **industrialisation du runtime IA**.

---

# 11. Décisions

## Décision 1 — Ne pas refaire PAGEPRINT/PAGETRANSLATE

Le cœur est bon.

```text id="093y1a"
Pas de nouvelle refonte fondamentale PAGEPRINT/PAGETRANSLATE.
```

## Décision 2 — Créer `rev_06_translation_runtime`

La prochaine version doit viser :

```text id="fa8gl6"
brancher proprement ai_models/translation/
```

Pas seulement ajouter un moteur mock.

## Décision 3 — Créer un vrai moteur CTranslate2 propre

Créer :

```text id="q033ju"
translation_engines/ct2_engine.py
translation_engines/model_registry.py
translation_engines/engine_health.py
translation_engines/request.py
```

## Décision 4 — Séparer les statuts

Ajouter :

```text id="u6kprk"
pipeline_status
translation_runtime_status
linguistic_quality_status
publication_readiness_status
```

## Décision 5 — Refondre la terminologie

Passer de :

```text id="2p3a5w"
locked_terms plats
```

à :

```text id="thmjmo"
glossaire typé avec politiques.
```

---

# 12. Architecture cible `rev_06`

```text id="t7wd9f"
translation_engines/
├── __init__.py
├── base.py
├── request.py
├── factory.py
├── model_registry.py
├── engine_health.py
├── mock_engine.py
├── rule_engine.py
├── ct2_engine.py
├── local_model_engine.py
└── external_model_engine.py
```

## `base.py`

```python id="7g5vl2"
class TranslationEngine:
    profile: str
    supports_batch: bool = False

    def translate(self, text: str, source_lang: str, target_lang: str, context: dict) -> str:
        raise NotImplementedError

    def translate_batch(self, requests: list[dict]) -> list[dict]:
        return [
            {
                "translated_text": self.translate(
                    req["text"],
                    req.get("source_lang", "auto"),
                    req.get("target_lang", "fr"),
                    req.get("context", {}),
                ),
                "raw_output": None,
                "metadata": {}
            }
            for req in requests
        ]

    def healthcheck(self) -> dict:
        return {"status": "unknown"}
```

## `request.py`

```python id="s7rb79"
{
  "request_id": "...",
  "translation_unit_id": "...",
  "text": "...",
  "protected_text": "...",
  "source_lang": "en",
  "target_lang": "fr",
  "role": "...",
  "object_type": "...",
  "semantic_kind": "...",
  "context": {...},
  "protected_tokens": [...],
  "constraints": {...}
}
```

## `model_registry.py`

Lit :

```text id="ohrgm5"
ai_models/translation/model_inventory.json
```

Format cible :

```json id="n3d2to"
{
  "default_engine": "ct2",
  "models": [
    {
      "name": "opus_mt_tc_big_en_fr",
      "backend": "ctranslate2",
      "family": "marian",
      "source_langs": ["en"],
      "target_langs": ["fr"],
      "model_dir": "ai_models/translation/opus_mt_tc_big_en_fr_ct2_int8",
      "tokenizer_dir": "ai_models/translation/opus_mt_tc_big_en_fr_tokenizer",
      "device": "cpu",
      "compute_type": "int8",
      "max_input_tokens": 512,
      "priority": 10
    },
    {
      "name": "m2m100_418m",
      "backend": "ctranslate2",
      "family": "m2m100",
      "source_langs": ["en", "fr", "auto"],
      "target_langs": ["fr", "en"],
      "model_dir": "ai_models/translation/m2m100_418m_ct2_int8",
      "tokenizer_dir": "ai_models/translation/m2m100_418m_tokenizer",
      "device": "cpu",
      "compute_type": "int8",
      "max_input_tokens": 512,
      "priority": 5
    }
  ]
}
```

---

# 13. Tâches `rev_06` — Todo list complète

## Chantier A — Runtime IA local

* [ ] Créer `translation_engines/base.py`.
* [ ] Créer `translation_engines/request.py`.
* [ ] Créer `translation_engines/model_registry.py`.
* [ ] Créer `translation_engines/ct2_engine.py`.
* [ ] Créer `translation_engines/engine_health.py`.
* [ ] Modifier `translation_engines/factory.py` pour supporter :

```text id="gpu8th"
mock
rule
ct2
local
external
```

* [ ] Ajouter variables :

```bash id="8lsyqz"
TRANSLATION_ENGINE=ct2
TRANSLATION_MODEL_INVENTORY=ai_models/translation/model_inventory.json
TRANSLATION_MODEL_NAME=opus_mt_tc_big_en_fr
TRANSLATION_DEVICE=cpu
TRANSLATION_BATCH_SIZE=8
TRANSLATION_MAX_INPUT_TOKENS=512
```

* [ ] Le moteur doit refuser de démarrer si :

```text id="20f75v"
model_dir absent
tokenizer_dir absent
ctranslate2 absent
transformers absent
langue non supportée
```

* [ ] Ajouter `healthcheck()`.

Critère :

```bash id="ej0kgk"
python3 tools/check_translation_engine.py --engine ct2
```

doit retourner :

```json id="wtfth7"
{
  "status": "ok",
  "engine": "ct2",
  "model_name": "...",
  "model_family": "...",
  "source_langs": [...],
  "target_langs": [...]
}
```

---

## Chantier B — Batch translation

* [ ] Ajouter `translate_batch()` dans le contrat moteur.
* [ ] Modifier `PageTranslationBuilder` pour regrouper les unités.
* [ ] Ajouter option :

```python id="q70pho"
batch_size: int = 8
```

* [ ] `run_translation_trial.py` doit afficher :

```text id="ycrsb0"
batch_count
avg_latency_ms
units_per_second
```

* [ ] Fallback unitaire si moteur sans batch.

Tests :

```text id="1lnz4h"
tests/pagetranslate/test_batch_translation_engine.py
tests/pagetranslate/test_batch_fallback_to_single_translate.py
```

---

## Chantier C — Placeholder policy

* [ ] Créer `pagetranslate/placeholder_policy.py`.
* [ ] Tester plusieurs formats :

```text id="l0t7j1"
⟦PT0001⟧
<nt id="PT0001"/>
[[PT0001]]
@@PT0001@@
```

* [ ] Ajouter outil :

```bash id="s7p918"
python3 tools/test_placeholder_roundtrip.py --engine ct2
```

* [ ] Choisir automatiquement le placeholder le plus robuste pour le modèle.

* [ ] Ajouter métrique :

```text id="dxmq42"
placeholder_corruption_rate
```

Critère :

```text id="to3u0w"
placeholder_corruption_rate = 0 sur corpus de test
```

---

## Chantier D — Terminologie typée

Remplacer `DEFAULT_LOCKED_TERMS` par :

```json id="j7o9ly"
{
  "terms": {
    "MLP": {"policy": "preserve"},
    "CNN": {"policy": "preserve"},
    "ReLU": {"policy": "preserve"},
    "Softmax": {"policy": "preserve"},
    "precision": {"policy": "preferred_translation", "target": "précision"},
    "recall": {"policy": "preferred_translation", "target": "rappel"},
    "dropout": {"policy": "contextual", "preferred": "dropout"},
    "pooling": {"policy": "contextual", "preferred": "pooling"}
  }
}
```

À faire :

* [ ] Créer `ai_models/translation/terminology_en_fr.json`.
* [ ] Modifier `pagetranslate/terminology.py`.
* [ ] Séparer :

```text id="9va1p8"
preserve_terms
preferred_terms
contextual_terms
reserved_terms
```

* [ ] Ne plus protéger automatiquement `precision`, `recall`, `dropout`, `pooling`.
* [ ] Ajouter QA :

```text id="808q5y"
preferred_term_missing
reserved_term_used
preserve_term_modified
```

Tests :

```text id="bmq63z"
tests/pagetranslate/test_typed_terminology.py
tests/pagetranslate/test_precision_recall_not_locked.py
tests/pagetranslate/test_mlp_cnn_preserved.py
```

---

## Chantier E — Statuts qualité séparés

Modifier `run_translation_trial.py`.

Actuellement :

```text id="a2hqp4"
functional_status = ok
```

Il faut ajouter :

```json id="ez0zt1"
{
  "pipeline_status": "ok",
  "translation_runtime_status": "ok",
  "linguistic_quality_status": "ok|review|ko",
  "publication_readiness_status": "ok|review|ko"
}
```

Règles :

```text id="evc2sv"
pipeline_status = erreurs structurelles
translation_runtime_status = moteur chargé / erreurs appels
linguistic_quality_status = needs_review, unchanged, source leak, terminology
publication_readiness_status = overflow, layout, reconstruction readiness
```

Tests :

```text id="9gqcv6"
tests/functional/test_translation_trial_quality_status.py
tests/functional/test_needs_review_makes_quality_review.py
tests/functional/test_functional_ok_quality_ko_possible.py
```

Critère :

```text id="rtzvo5"
needs_review_count > 0 ne doit pas rendre pipeline_status KO,
mais doit rendre linguistic_quality_status = review ou ko.
```

---

## Chantier F — Logs moteur complets

Ajouter dans chaque unité :

```json id="9glm5n"
{
  "engine_trace": {
    "engine": "ct2",
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

Modifier :

```text id="pcl53c"
TranslatorBridge
PageTranslationBuilder
run_translation_trial.py
```

Tests :

```text id="q1xq1m"
tests/pagetranslate/test_engine_trace_present.py
tests/functional/test_trial_logs_raw_engine_output.py
```

---

## Chantier G — Essais IA graduels

Une fois `ct2_engine.py` prêt :

### Niveau 1 — Smoke

```bash id="04utuf"
python3 tools/check_translation_engine.py --engine ct2
```

### Niveau 2 — Phrases propres

```bash id="q6v53a"
python3 tools/run_translation_trial.py tests/golden_documents/body_text/input_data.json --engine ct2 --target-lang fr
```

### Niveau 3 — TOC

```bash id="n748ff"
python3 tools/run_translation_trial.py tests/golden_documents/toc/input_data.json --engine ct2 --target-lang fr
```

### Niveau 4 — 10 pages variées

```bash id="erdvcr"
python3 tools/run_batch_translation_trial.py audit_folder/ --engine ct2 --target-lang fr
```

### Niveau 5 — Document court

```bash id="e4qvi5"
python3 tools/run_document_trial.py tests/doc_pdf/test_docintelligence.pdf --pages 1-5 --engine ct2 --target-lang fr
```

---

# 14. Directives pour Codex

Voici les directives courtes à donner à Codex pour la suite :

```text id="6n1cqv"
rev_05 valide le branchement technique mock/rule.
Ne pas refaire PAGEPRINT/PAGETRANSLATE.
Créer rev_06_translation_runtime.

Objectifs :
1. brancher réellement ai_models/translation/ ;
2. créer CTranslate2Engine ;
3. ajouter model_registry.json ;
4. ajouter healthcheck moteur ;
5. ajouter batch translation ;
6. rendre les placeholders robustes pour modèles IA ;
7. refondre terminology.py en glossaire typé ;
8. séparer pipeline_status, runtime_status et linguistic_quality_status ;
9. ajouter logs moteur complets ;
10. lancer essais IA graduels seulement après preflight OK.

Critère :
un modèle IA local doit pouvoir être chargé depuis ai_models/translation/
sans importer translator.py historique comme cœur obligatoire.
```

---

# 15. Conclusion finale

`rev_05` est une bonne étape.

Elle valide :

```text id="0hkdux"
le contrat moteur
le runner d’essai
le preflight
les moteurs mock/rule
les compteurs QA
la protection/restauration de base
```

Mais elle ne valide pas encore :

```text id="9bb4be"
le vrai moteur IA local
le chargement depuis ai_models/translation/
la robustesse placeholders/tokenizer
la terminologie technique sérieuse
la qualité linguistique
la performance batch
```

Donc la décision est :

```text id="4a7ndd"
On ne fait plus de refonte PAGEPRINT/PAGETRANSLATE.
On fait maintenant rev_06 : vrai runtime IA de traduction.
```

La prochaine version doit répondre à une seule question :

```text id="e7cwrt"
Un modèle IA local dans ai_models/translation/ peut-il traduire proprement,
en batch, avec tokens protégés, glossaire typé, QA exploitable,
et logs suffisants pour séparer erreur moteur / erreur PAGEPRINT / erreur reconstruction ?
```

C’est le prochain jalon.


/home/raphael/Images/Captures d’écran/Capture d’écran du 2026-06-13 09-35-00.png


Oui. Cet aperçu confirme que l’organisation prévue pour `ai_models/translation/` est cohérente avec la direction `rev_06`.

On voit déjà les éléments essentiels :

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

C’est une bonne base. La décision devient maintenant très claire :

```text
OPUS-MT EN→FR = moteur principal spécialisé anglais → français.
M2M100 418M = moteur multilingue de secours / fallback / autres langues.
translation_profiles.json = politique de traduction.
style_tone_profiles.json = style, ton, registre.
translation_memory.jsonl = mémoire de traduction / corrections validées.
model_inventory.json = registre technique des modèles.
```

# 1. Décision immédiate

On ne modifie pas encore `PAGEPRINT` / `PAGETRANSLATE`.

La prochaine étape est :

```text
Implémenter le vrai runtime CTranslate2 qui lit ai_models/translation/model_inventory.json,
charge les modèles CT2,
charge les tokenizers,
choisit le bon modèle,
traduit en batch,
et retourne des traces QA complètes.
```

Donc la prochaine version doit être :

```text
rev_06 = branchement réel des modèles IA locaux
```

---

# 2. Rôle de chaque élément visible

## `opus_mt_tc_big_en_fr_ct2_int8/`

À utiliser comme moteur principal pour :

```text
source_lang = en
target_lang = fr
```

Avantages probables :

```text
rapide
spécialisé anglais-français
léger CPU si int8
meilleur choix par défaut pour ton corpus actuel en anglais technique
```

## `opus_mt_tc_big_en_fr_tokenizer/`

Tokenizer associé. Il doit être chargé avec le modèle OPUS correspondant. Il ne faut pas mélanger le tokenizer OPUS avec M2M100.

## `m2m100_418m_ct2_int8/`

À utiliser comme moteur secondaire :

```text
anglais → français si OPUS échoue
français → anglais
autres langues futures
langue source incertaine
```

M2M100 est plus généraliste, mais souvent plus lourd et parfois moins spécialisé qu’un modèle OPUS bien adapté à une paire de langues précise.

## `m2m100_418m_tokenizer/`

Attention : M2M100 a une gestion spécifique des codes langues. Il faudra probablement gérer :

```text
src_lang
target_lang
forced_bos_token / prefix target selon tokenizer
```

Donc le runtime doit distinguer :

```text
family = marian / opus
family = m2m100
```

## `model_inventory.json`

C’est le fichier central. Il doit devenir la source de vérité pour charger les modèles.

Il doit contenir au minimum :

```json
{
  "default_engine": "ct2",
  "default_model": "opus_mt_tc_big_en_fr",
  "models": [
    {
      "name": "opus_mt_tc_big_en_fr",
      "backend": "ctranslate2",
      "family": "marian",
      "source_langs": ["en"],
      "target_langs": ["fr"],
      "model_dir": "ai_models/translation/opus_mt_tc_big_en_fr_ct2_int8",
      "tokenizer_dir": "ai_models/translation/opus_mt_tc_big_en_fr_tokenizer",
      "device": "cpu",
      "compute_type": "int8",
      "priority": 100,
      "max_input_tokens": 512,
      "batch_size": 8
    },
    {
      "name": "m2m100_418m",
      "backend": "ctranslate2",
      "family": "m2m100",
      "source_langs": ["auto", "en", "fr"],
      "target_langs": ["fr", "en"],
      "model_dir": "ai_models/translation/m2m100_418m_ct2_int8",
      "tokenizer_dir": "ai_models/translation/m2m100_418m_tokenizer",
      "device": "cpu",
      "compute_type": "int8",
      "priority": 50,
      "max_input_tokens": 512,
      "batch_size": 4
    }
  ]
}
```

## `translation_profiles.json`

À utiliser pour choisir :

```text
langue source
langue cible
domaine
niveau de littéralité
traduction technique ou naturelle
préservation acronymes
style de reformulation
```

## `style_tone_profiles.json`

À utiliser après sélection du profil :

```text
professionnel
académique
technique
didactique
neutre
fluide
fidèle
```

## `translation_memory.jsonl`

À utiliser comme mémoire de traduction.

Format recommandé :

```json
{"source_lang":"en","target_lang":"fr","source":"Hidden layers","target":"Couches cachées","domain":"deep_learning","validated":true}
```

Elle doit intervenir :

```text
avant moteur si correspondance exacte
après moteur pour QA terminologique
comme source de corrections humaines validées
```

---

# 3. Architecture à implémenter maintenant

Créer/renforcer :

```text
translation_engines/
├── base.py
├── request.py
├── model_registry.py
├── ct2_engine.py
├── engine_health.py
├── factory.py
├── terminology_store.py
├── translation_memory.py
└── placeholder_policy.py
```

Le flux doit devenir :

```text
translation_plan item
→ protection placeholders
→ terminology locks
→ translation memory exact match
→ CT2 model selection
→ batch translation
→ restore placeholders
→ terminology QA
→ quality QA
→ projection
```

---

# 4. Décision modèle : ordre de sélection

Pour `en → fr`, utiliser :

```text
1. translation_memory exact match
2. opus_mt_tc_big_en_fr
3. m2m100_418m fallback
4. rule/mock seulement en mode test
```

Donc la factory doit pouvoir faire :

```python
engine = create_translation_engine(
    engine_name="ct2",
    model_inventory="ai_models/translation/model_inventory.json",
    preferred_model=None,
)
```

Et le registry choisit automatiquement :

```python
model = registry.select_model(source_lang="en", target_lang="fr")
```

---

# 5. Points techniques critiques à ne pas rater

## 5.1 Ne jamais mélanger modèle et tokenizer

Interdit :

```text
m2m100 model + opus tokenizer
opus model + m2m100 tokenizer
```

Le couple doit venir du même item `model_inventory`.

## 5.2 Gérer les familles différemment

Pour OPUS / Marian :

```text
tokenize source
generate
decode target
```

Pour M2M100 :

```text
définir source_lang
définir target_lang
utiliser la convention du tokenizer M2M100
gérer target prefix / forced language token
```

Donc `ct2_engine.py` doit avoir :

```python
_translate_marian_batch()
_translate_m2m100_batch()
```

ou au minimum une stratégie par `family`.

## 5.3 Tester les placeholders avec les vrais modèles

Avant de traduire des documents, faire :

```text
Hidden layers ⟦PT0001⟧
CNN architecture ⟦PT0002⟧
Figure ⟦PT0003⟧ shows the model
```

et vérifier que les placeholders reviennent intacts.

Si `⟦PT0001⟧` est abîmé, tester :

```text
[[PT0001]]
@@PT0001@@
<nt id="PT0001"/>
```

Puis choisir automatiquement le plus robuste.

## 5.4 Batch obligatoire

Ne pas traduire 1 unité à la fois pour les documents.

Il faut :

```python
translate_batch(requests)
```

avec :

```text
batch_size OPUS = 8 ou 16 CPU
batch_size M2M100 = 2 ou 4 CPU
```

à ajuster selon RAM.

---

# 6. Todo immédiate pour Codex

## Chantier A — Registry modèles

* [ ] Lire `ai_models/translation/model_inventory.json`.
* [ ] Valider que tous les `model_dir` existent.
* [ ] Valider que tous les `tokenizer_dir` existent.
* [ ] Valider `backend`, `family`, `source_langs`, `target_langs`.
* [ ] Ajouter `select_model(source_lang, target_lang, preferred_model=None)`.
* [ ] Ajouter fallback si modèle principal indisponible.

## Chantier B — CTranslate2 engine

* [ ] Créer `translation_engines/ct2_engine.py`.
* [ ] Charger `ctranslate2.Translator`.
* [ ] Charger tokenizer depuis `tokenizer_dir`.
* [ ] Supporter `family=marian`.
* [ ] Supporter `family=m2m100`.
* [ ] Ajouter `translate()`.
* [ ] Ajouter `translate_batch()`.
* [ ] Ajouter `healthcheck()`.
* [ ] Ajouter erreurs explicites si dépendance absente.

## Chantier C — Translation memory

* [ ] Créer `translation_engines/translation_memory.py`.
* [ ] Lire `translation_memory.jsonl`.
* [ ] Faire lookup exact.
* [ ] Faire lookup normalisé simple.
* [ ] Retourner `memory_hit=true/false`.
* [ ] Si mémoire validée, ne pas appeler le modèle.

## Chantier D — Profils

* [ ] Lire `translation_profiles.json`.
* [ ] Lire `style_tone_profiles.json`.
* [ ] Injecter le profil dans `context`.
* [ ] Ajouter au log moteur :

```json
{
  "translation_profile": "...",
  "style_tone_profile": "..."
}
```

## Chantier E — Terminologie typée

* [ ] Ne pas tout verrouiller.
* [ ] Distinguer :

```text
preserve
preferred_translation
contextual
forbidden_translation
```

* [ ] Garder `MLP`, `CNN`, `ReLU`, `Softmax`.
* [ ] Traduire ou recommander :

```text
precision → précision
recall → rappel
hidden layers → couches cachées
output layer → couche de sortie
input layer → couche d’entrée
```

## Chantier F — Placeholder roundtrip

* [ ] Créer `tools/test_placeholder_roundtrip.py`.
* [ ] Tester OPUS.
* [ ] Tester M2M100.
* [ ] Choisir le meilleur placeholder.
* [ ] Écrire le résultat dans :

```text
ai_models/translation/placeholder_policy.json
```

## Chantier G — Trial réel

* [ ] Créer ou finaliser :

```text
tools/run_translation_trial.py
tools/run_batch_translation_trial.py
```

* [ ] Ajouter option :

```bash
--engine ct2
--model opus_mt_tc_big_en_fr
--target-lang fr
--batch-size 8
```

* [ ] Sortie obligatoire :

```json
{
  "pipeline_status": "ok",
  "translation_runtime_status": "ok",
  "linguistic_quality_status": "review",
  "engine": "ct2",
  "model_name": "opus_mt_tc_big_en_fr",
  "memory_hit_count": 0,
  "model_call_count": 0,
  "protected_token_mismatch_count": 0,
  "number_mismatch_count": 0,
  "needs_review_count": 0
}
```

---

# 7. Tests à créer

```text
tests/translation_engines/test_model_registry.py
tests/translation_engines/test_ct2_engine_healthcheck.py
tests/translation_engines/test_translation_memory.py
tests/translation_engines/test_engine_model_selection.py
tests/translation_engines/test_placeholder_policy.py
tests/translation_engines/test_batch_translation_contract.py
tests/pagetranslate/test_ct2_engine_bridge.py
tests/functional/test_real_engine_trial_contract.py
```

Tests sans modèle réel :

```text
mock registry
fake ct2 translator
fake tokenizer
```

Tests avec modèle réel :

```text
marqués @pytest.mark.ai
désactivés par défaut
```

---

# 8. Commandes de validation

## Validation sans IA réelle

```bash
python3 -m pytest -q tests/pageprint tests/pagetranslate tests/functional tests/pipelines tests/translation_engines
```

## Healthcheck réel

```bash
python3 tools/check_translation_engine.py \
  --engine ct2 \
  --inventory ai_models/translation/model_inventory.json \
  --model opus_mt_tc_big_en_fr
```

## Test placeholder

```bash
python3 tools/test_placeholder_roundtrip.py \
  --engine ct2 \
  --model opus_mt_tc_big_en_fr \
  --target-lang fr
```

## Test traduction simple

```bash
python3 tools/run_translation_trial.py \
  input_data.json \
  --engine ct2 \
  --model opus_mt_tc_big_en_fr \
  --source-lang en \
  --target-lang fr
```

---

# 9. Conclusion

L’arborescence montrée est bonne. Elle correspond exactement à ce qu’il faut pour passer à `rev_06`.

Décision :

```text
On branche maintenant les vrais modèles.
```

Mais on le fait proprement :

```text
model_inventory.json
→ model_registry
→ ct2_engine
→ tokenizer adapté
→ translation_memory
→ terminology typée
→ placeholder roundtrip
→ batch translation
→ trial audité
```

La règle finale :

```text
OPUS EN-FR est le moteur principal.
M2M100 est le fallback multilingue.
La mémoire et le glossaire passent avant le modèle.
Aucun essai document complet avant healthcheck + placeholder roundtrip + trial page simple.
```

