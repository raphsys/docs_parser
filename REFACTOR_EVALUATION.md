# REFACTOR_EVALUATION rev_05

## Verdict

`rev_05` valide le socle moteur de traduction branché sur la racine du projet.
Le chemin normal reste strict :

```text
PAGEPRINT logical_structures
-> semantic_builder.translation_segments
-> view_compiler.views.translation_plan
-> PAGETRANSLATE translation_plan_reader
-> translation_engines/*
-> projection.reconstruction_units
```

Le comportement attendu est maintenant celui-ci :

```text
translation_plan present et non vide -> PAGETRANSLATE selection_mode=translation_plan
translation_plan present mais vide -> KO, aucun fallback silencieux
translation_plan absent -> fallback legacy seulement si explicitement autorise
```

## Ce qui a été ajouté

- `translation_engines/`
- `ai_models/translation/terminology_en_fr.json`
- `tools/run_translation_trial.py`
- `tools/run_document_trial.py`
- `tools/run_batch_translation_trial.py`
- `tools/check_translation_engine.py`
- `tools/test_placeholder_roundtrip.py`
- `translation_engines/model_registry.py`
- `translation_engines/ct2_engine.py`
- `translation_engines/placeholder_policy.py`
- `translation_engines/engine_health.py`
- `translation_engines/request.py`
- `translation_engines/base.py`
- `pagetranslate/terminology.py` enrichi avec glossaire typé
- `pagetranslate/protection.py` avec politiques de placeholders
- `pagetranslate/translator_bridge.py` avec batch translation
- `pagetranslate/builder.py` avec runtime, quality et batch path
- `pagetranslate/quality.py` avec corruption de placeholders
- `pagetranslate/context_builder.py` avec `terms_path`
- `tests/pagetranslate/*` pour moteurs, batch, glossaire, placeholders
- `tests/functional/*` pour trial et quality status
- `tests/golden_documents/*`

## Ce qui a été branché

- `mock`, `echo`, `prefix`, `rule`, `local`, `external`, `ct2` sont disponibles via la factory.
- `ct2` lit `ai_models/translation/model_inventory.json`.
- `ct2` est validé par healthcheck avec modèle et tokenizer réels présents dans `ai_models/translation/`.
- `TranslatorBridge` privilégie la nouvelle couche moteur avant le fallback legacy.
- `run_translation_trial.py` n’appelle le moteur qu’après audit PAGEPRINT/PAGETRANSLATE.
- `run_translation_trial.py` logge :
  - `protected_source_text`
  - `raw_engine_output`
  - `restored_output`
  - `post_glossary_output`
  - `latency_ms`
  - `input_token_count`
  - `output_token_count`
  - `truncated`
- `quality.py` expose :
  - `protected_token_mismatch_count`
  - `number_mismatch_count`
  - `terminology_warning_count`
  - `placeholder_corruption_count`
  - `needs_review_count`

## Glossaire

Le glossaire a été rendu typé :

- `preserve`
- `preferred_translation`
- `contextual`

Exemples couverts :

- `MLP`, `CNN`, `ReLU`, `Softmax`, `SQL`, `API`, `OCR` en préservation
- `precision -> précision`
- `recall -> rappel`
- `dropout`, `pooling`, `F-score` en politique contextuelle

## Placeholder policy

La politique de placeholders gère plusieurs styles :

- `unicode_bracket`
- `ascii_xml`
- `plain_ascii`
- `at_token`

Le round-trip a été validé sur le moteur `echo`.

## Validation technique

Commandes exécutées :

```bash
python3 -m py_compile translation_engines/*.py pagetranslate/*.py tools/*.py
.docs-parser/bin/python -m pytest -q tests/pageprint tests/pagetranslate tests/functional tests/pipelines
.docs-parser/bin/python tools/check_translation_engine.py --engine ct2
.docs-parser/bin/python tools/test_placeholder_roundtrip.py --engine echo
```

Résultats :

```text
Compilation OK
72 passed in 0.37s
CT2 healthcheck OK
placeholder roundtrip OK
```

## Essais documentaires

Les essais contrôlés sur PDF réel passent sur la chaîne de base :

- `selection_mode=translation_plan`
- `fallback_selector_used=false`
- `functional_status=ok`

Le moteur `mock` sert au smoke test du pipeline.
Le moteur `rule` et les essais documentaires montrent le bon passage de la structure vers la traduction.
Le moteur `ct2` est disponible et détecté par healthcheck, prêt pour essais réels sur corpus.

## Limites restantes

- Le pytest global historique reste pollué par des suites legacy et le dossier `revisions/`.
- Les modèles `local` et `external` restent des adaptateurs, à brancher sur un fournisseur concret.
- Le corpus golden réel doit encore être densifié pour tables complexes, index denses et figures réelles.
- Les essais IA de production doivent rester contrôlés page par page et lot par lot.

## Conclusion

`rev_05` est la bonne base pour brancher un runtime IA réel et faire des essais sur documents.
Le code actif est bien dans la racine du projet, pas dans `revisions/`.
