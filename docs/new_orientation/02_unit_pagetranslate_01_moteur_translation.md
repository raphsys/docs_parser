# Verdict

Oui, **on est maintenant prêt à brancher des moteurs de traduction et à faire des essais sur documents**, mais en **mode expérimental contrôlé**, pas encore en mode production aveugle.

On ne doit plus faire une grosse refonte de fond avant cela. Le socle `rev_04` est suffisant pour commencer les essais réels.

Décision :

```text
rev_04 = socle fonctionnel validé pour essais moteur.
Pas de nouvelle refonte fondamentale maintenant.
Prochaine étape = branchement moteur + audits sur documents réels.
```

---

# 1. Ce que j’ai vérifié dans `rev_04`

## 1.1 Compilation cœur : OK

J’ai compilé :

```bash
python3 -m py_compile pageprint/*.py \
  pageprint/detection/*.py \
  pageprint/evidence/*.py \
  pageprint/structure_builders/*.py \
  pagetranslate/*.py \
  pipelines/*.py \
  tools/*.py
```

Résultat :

```text
OK
```

## 1.2 Tests ciblés refonte : OK

J’ai exécuté :

```bash
python3 -m pytest -q tests/pageprint tests/pagetranslate tests/functional tests/pipelines
```

Résultat :

```text
55 passed
```

C’est important : les tests ciblés `rev_04` passent.

## 1.3 Le chemin `translation_plan` est maintenant solide

Le comportement est conforme :

```text
translation_plan présent et non vide
→ PAGETRANSLATE utilise translation_plan

translation_plan présent mais vide
→ KO, pas de fallback silencieux

translation_plan absent
→ fallback seulement si allow_fallback=True
```

Donc le problème principal des versions précédentes est réglé.

## 1.4 Les builders documentaires sont présents

`rev_04` contient maintenant :

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
```

Et `semantic_builder.py` les consomme réellement.

Donc la chaîne suivante existe :

```text
logical_structures
→ semantic_builder.translation_segments
→ view_compiler.translation_plan
→ PAGETRANSLATE
```

C’est le bon modèle.

---

# 2. Limite importante

Le test global :

```bash
python3 -m pytest -q
```

échoue encore dans mon environnement à cause de dépendances legacy :

```text
ModuleNotFoundError: rapidocr_onnxruntime
```

Cela concerne des tests anciens qui importent `ocr_server.py` et des composants historiques.

Ce n’est pas un blocage pour tester `PAGEPRINT/PAGETRANSLATE rev_04`, mais cela veut dire :

```text
le cœur rev_04 est testable ;
le projet complet legacy n’est pas encore proprement isolé.
```

À faire plus tard :

```text
séparer tests core / tests legacy / tests modèles AI / tests intégration lourde.
```

---

# 3. Décision claire

## On ne fait pas une nouvelle refonte de fond maintenant

Ne pas repartir sur :

```text
nouvelle architecture
nouveau contrat
nouveau découpage
nouvelle philosophie
```

La philosophie est maintenant assez saine :

```text
PAGEPRINT compile la page.
PAGETRANSLATE exécute le translation_plan.
selector/coalescer restent fallback.
```

## On passe aux essais de traduction

Mais avec un protocole strict :

```text
1. Auditer PAGEPRINT avant traduction.
2. Traduire seulement les pages functional_status=ok.
3. Brancher le moteur via TranslatorBridge.
4. Mesurer la qualité linguistique et structurelle.
5. Ne corriger PAGEPRINT que si l’audit montre une mauvaise unité envoyée.
```

---

# 4. Plan immédiat

# Phase 1 — Geler `rev_04`

Créer une base stable :

```text
rev_04 = baseline
```

À faire :

* [ ] Taguer ou copier `rev_04` comme version de référence.
* [ ] Ne plus modifier le contrat `translation_plan`.
* [ ] Ne plus remettre `selector/coalescer` en chemin principal.
* [ ] Garder les 55 tests ciblés comme garde-fous obligatoires.

Critère :

```text
tests/pageprint + tests/pagetranslate + tests/functional + tests/pipelines = OK
```

---

# Phase 2 — Brancher un moteur de traduction simple

Le code est prêt pour recevoir un moteur avec cette interface :

```python
class TranslationEngine:
    def translate(self, text: str, source_lang: str, target_lang: str, context: dict) -> str:
        ...
```

À faire :

* [ ] Créer un dossier :

```text
translation_engines/
```

* [ ] Ajouter :

```text
translation_engines/mock_engine.py
translation_engines/rule_engine.py
translation_engines/local_model_engine.py
translation_engines/external_model_engine.py
```

* [ ] Commencer par un moteur très simple :

```python
class PrefixEngine:
    def translate(self, text, source_lang, target_lang, context):
        return f"FR::{text}"
```

Ce test ne sert pas à juger la traduction. Il sert à vérifier :

```text
protection
restauration
projection
reconstruction_units
QA
audit
```

* [ ] Ensuite brancher un vrai moteur local ou externe.

---

# Phase 3 — Essais sur documents réels, mais petits

Ne pas commencer par 480 pages.

Ordre recommandé :

```text
1 page
3 pages
10 pages variées
30 pages
document complet
```

Pour chaque essai :

```text
PDF source
→ PAGEPRINT
→ audit fonctionnel
→ PAGETRANSLATE dry_run
→ PAGETRANSLATE vrai moteur
→ QA traduction
→ reconstruction si disponible
→ audit visuel
```

---

# 5. Protocole obligatoire avant traduction

Pour chaque page, exécuter :

```bash
python3 tools/run_functional_audit.py input_data.json --run-pagetranslate --dry-run
```

Ou pour un dossier :

```bash
python3 tools/run_batch_functional_audit.py audit_folder/ --run-pagetranslate --dry-run
```

La page est autorisée pour traduction seulement si :

```text
functional_status = ok
pages_using_fallback = 0
role_none_translation_items = 0
word_char_translation_items = 0
reconstruction_units_missing_roles = 0
caption_raw_block_translation = 0
table_pages_without_tables = 0
index_pages_without_index_entries = 0
publisher_mark_sent_to_translation = 0
```

Sinon :

```text
ne pas accuser le moteur ;
corriger PAGEPRINT.
```

---

# 6. Ce qu’on teste maintenant avec le moteur

On ne teste pas seulement :

```text
est-ce que la phrase est traduite ?
```

On teste :

```text
est-ce que le moteur reçoit les bonnes unités ?
est-ce que les tokens protégés restent protégés ?
est-ce que les nombres restent identiques ?
est-ce que les acronymes restent corrects ?
est-ce que la projection garde les rôles ?
est-ce que la reconstruction peut exploiter les sorties ?
```

Métriques à suivre :

```text
protected_token_mismatch_count
number_mismatch_count
terminology_warning_count
needs_review_count
unchanged_suspect_count
empty_translation_count
overflow_risk_count
fallback_selector_usage
```

---

# 7. Ce qu’il ne faut pas faire

Ne pas lancer directement :

```text
PDF complet → vrai moteur → reconstruction complète
```

sans audit intermédiaire.

Ne pas corriger une mauvaise traduction si la mauvaise unité venait de PAGEPRINT.

Exemple :

```text
source_text = "copy C:\Music\file.mp3 Delete all files"
```

Ce n’est pas un problème moteur. C’est un problème `translation_plan`.

Ne pas créer de rustines :

```python
if text == "(weights)":
    ...
```

Toujours corriger par :

```text
role
structure
policy
preservation_mode
test
```

---

# 8. Travaux à faire en parallèle pendant les essais

## Chantier A — Moteurs

* [ ] Ajouter `translation_engines/`.
* [ ] Ajouter moteur mock.
* [ ] Ajouter moteur local.
* [ ] Ajouter moteur externe.
* [ ] Ajouter configuration :

```text
TRANSLATION_ENGINE=mock|local|external
```

* [ ] Ajouter logs par appel moteur :

```json
{
  "unit_id": "...",
  "role": "...",
  "source_text": "...",
  "protected_tokens": [...],
  "translated_text": "...",
  "qa": {...}
}
```

## Chantier B — Corpus d’essai

Créer :

```text
tests/golden_documents/
```

Avec au minimum :

```text
toc.pdf
body_text.pdf
table_commands.pdf
index_page.pdf
caption_figure.pdf
diagram_labels.pdf
cover_image.pdf
mixed_page.pdf
```

Pour chaque document, garder :

```text
input_data.json
translation_plan.json
expected_translation_units.json
audit_expected.json
```

## Chantier C — Audit moteur

Ajouter :

```text
tools/run_translation_trial.py
```

Fonction :

```bash
python3 tools/run_translation_trial.py input_data.json \
  --engine mock \
  --target-lang fr \
  --output trial_result.json
```

Sortie attendue :

```json
{
  "pageprint_functional_status": "ok",
  "pagetranslate_functional_status": "ok",
  "engine": "mock",
  "unit_count": 0,
  "qa": {
    "protected_token_mismatch_count": 0,
    "number_mismatch_count": 0,
    "needs_review_count": 0
  }
}
```

## Chantier D — Nettoyage tests legacy

Pas prioritaire, mais à prévoir :

* [ ] Marquer les tests legacy avec `pytest.mark.legacy`.
* [ ] Marquer les tests dépendant modèles AI avec `pytest.mark.ai`.
* [ ] Marquer les tests nécessitant PDF lourds avec `pytest.mark.integration`.
* [ ] Configurer :

```bash
pytest -q tests/pageprint tests/pagetranslate tests/functional tests/pipelines
```

comme suite core.

---

# 9. Décision : moteur maintenant ou encore refonte ?

Réponse nette :

```text
On branche les moteurs maintenant.
```

Mais :

```text
On ne fait pas encore des essais de production.
On fait des essais contrôlés, audités, page par page, puis par lots.
```

Donc :

```text
Pas de nouvelle refonte fondamentale avant les essais.
Seulement des corrections ciblées issues des audits réels.
```

---

# 10. Plan concret de la prochaine étape

## Étape 1

Créer `translation_engines/mock_engine.py`.

## Étape 2

Créer `tools/run_translation_trial.py`.

## Étape 3

Faire un essai sur une page body simple.

## Étape 4

Faire un essai sur une page TOC.

## Étape 5

Faire un essai sur les 10 pages variées déjà utilisées.

## Étape 6

Analyser séparément :

```text
erreurs PAGEPRINT
erreurs PAGETRANSLATE
erreurs moteur
erreurs reconstruction
```

## Étape 7

Seulement ensuite brancher un vrai moteur local/externe.

---

# Conclusion

`rev_04` franchit le seuil nécessaire.

On peut maintenant passer à :

```text
branchement moteur de traduction
+ essais contrôlés sur documents
+ audits qualité
```

On ne doit plus retarder avec une nouvelle grande refonte.

La règle pour la suite :

```text
si l’unité envoyée au moteur est mauvaise → corriger PAGEPRINT ;
si l’unité est bonne mais la traduction est mauvaise → corriger moteur/terminologie ;
si la traduction est bonne mais le rendu est mauvais → corriger projection/reconstruction.
```

Donc oui : **on avance vers les moteurs de traduction maintenant**.

