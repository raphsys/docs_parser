# Rule Studio — console de gouvernance des règles

Outil interne pour **rendre visibles, gouvernables, testables et modifiables**
les règles dispersées dans le pipeline vSense / WYSIWYG
(`PAGEPRINT → PAGETRANSLATE → PAGERECONSTRUCT → QA`).

Ce n'est pas un visualiseur décoratif : il scanne réellement le dépôt, extrait
les règles, les classe par unité de pipeline, les audite, les simule et propose
des modifications tracées (diff + backup Git), sans jamais écrire en silence.

## Installation

```bash
pip install -r tools/rule_studio/requirements.txt
```

## Lancement de l'interface

```bash
streamlit run tools/rule_studio/app.py
```

Puis dans l'UI :

1. saisir la racine du projet (pré-remplie) ;
2. **Scanner** ;
3. filtrer / cocher / éditer les règles dans le tableau ;
4. créer une **nouvelle règle** gérée ;
5. **simuler** une règle sur une fixture ou un contexte JSON ;
6. générer un **diff** et l'**appliquer** (backup Git + confirmation) ;
7. exporter CSV / JSON / Markdown ;
8. générer le **rapport d'audit** `RULE_AUDIT_REPORT.md`.

## Usage en ligne de commande

```bash
python -m tools.rule_studio.cli scan
python -m tools.rule_studio.cli audit --scan --out RULE_AUDIT_REPORT.md
python -m tools.rule_studio.cli export --scan --format csv --out rules.csv
```

## Architecture

```
tools/rule_studio/
  app.py              # interface Streamlit
  cli.py              # interface ligne de commande
  core/
    models.py         # RuleRecord + vocabulaires contrôlés
    scanner.py        # parcours du dépôt + routage extracteurs
    classifier.py     # classement par unité de pipeline (pondéré)
    rule_registry.py  # règles gérées (managed_rules.yaml)
    simulator.py      # moteur DSL sûr (when/then), sans eval/exec
    usage_analyzer.py # usage statique + compteur dynamique
    patcher.py        # diff + application sécurisée
    git_guard.py      # garde-fous Git (backup, refus d'écrasement)
    test_runner.py    # wrapper pytest
    exporters.py      # CSV / JSON / Markdown + rapport d'audit
    studio.py         # façade (scan/store/export)
  extractors/
    python_ast_extractor.py    # if/elif/match/assert/raise/enum/dataclass/const
    config_extractor.py        # yaml/json/toml
    markdown_extractor.py      # phrases normatives (doit/jamais/si…alors)
    comment_rule_extractor.py  # blocs # RULE-... / # END-RULE
    schema_extractor.py        # schémas (réutilise l'AST)
  storage/
    rule_store.py     # SQLite + fusion préservant les éditions humaines
    migrations.py
  data/
    managed_rules.yaml         # règles gérées explicitement (seed)
    rules.sqlite               # généré au scan
  fixtures/           # contextes de simulation
  tests/              # tests unitaires
```

## Catégories de règles distinguées

L'outil sépare nettement, pour éviter le bruit :

| Catégorie | `source_type` | éditable | gérée |
|---|---|---|---|
| détectée auto (code) | `python_ast` / `comment_block` | oui | si déclaré |
| détectée auto (config) | `config` | oui | non |
| documentée | `documentation` | non | non |
| gérée explicitement | `managed` / `manual` | oui | oui |

Chaque règle porte une **confiance** (0–1), un **statut**, un **usage**
(`used_static` / `not_referenced` / …) et une **validation**.

## Sécurité

- Simulation : moteur DSL déclaratif, **jamais** `eval`/`exec`.
- Patch : `git_guard` vérifie le dépôt, refuse d'écraser un fichier non suivi,
  crée une branche de backup, et n'applique **rien** sans confirmation explicite.
- V1 : l'« application » se limite à documenter la règle (bloc commentaire) dans
  le fichier cible. L'injection de logique reste manuelle/relue.

## Promotion d'une règle

Une règle candidate détectée peut être **promue en règle gérée** (bouton dédié) :
elle rejoint alors `managed_rules.yaml`, devient simulable et versionnée.

## Agents IA (gouvernance code ↔ langage naturel)

Rule Studio fait les deux sens : **code → explication naturelle** et **décision naturelle → patch contrôlé**.
L'IA *explique et propose* ; les tests et audits *prouvent*. Aucun agent n'applique de patch.

- `agents/model_client.py` — interface branchable `RuleStudioModelClient` + `DummyModelClient` (déterministe, sans modèle externe ; adaptateurs OpenAI/Claude/local optionnels).
- `agents/rule_interpreter_agent.py` — remplit les champs naturels (titre, règle, objectif, compréhension, impact, risque, décision conseillée).
- `agents/rule_coding_agent.py` — décision naturelle → patch **proposé** + fichiers/tests/risque (jamais appliqué).
- `agents/rule_validation_agent.py` — cohérence/patch/tests (validation humaine/auto/tests/simulation/audit).
- `agents/agent_runner.py` — orchestration + trace JSON dans `data/agent_runs/`.

### Cycle de vie d'une règle

`detected → interpreted → needs_human_review → edited_by_human → candidate → ready_to_generate → patch_generated → tests_failed | tests_passed → ready_to_apply → applied | rejected | deprecated`

(`active`, `valid`, `applied`, `interpreted` sont des états **distincts**.)

### Workflows UI

- **🤖 Interpréter avec IA** — remplit les champs naturels des règles sélectionnées (ou non interprétées).
- **💾 Enregistrer les éditions** — persiste, marque `edited_by_human`, et déclenche interprétation / patch proposé / validation selon les cases `implement` / `validate`. **Jamais d'application silencieuse.**
- **➕ Nouvelle règle** — crée une règle `candidate` (`source_type=manual`, non implémentée).
- **🩹 Diff & patch** — patch proposé par l'agent + diff + tests + application **confirmée** (git_guard/patcher/test_runner).
- **🔬 Audits pipeline** — disponibilité/statut des vrais audits (`functional_validator`, `source_text_lifecycle_ledger`, `render_ops_audit`, `visual_image_audit`).

### Règles fondamentales (gouvernance)

`NO-DROP-001`, `PT-COVERAGE-001`, `PR-COVERAGE-001`, `RENDER-COVERAGE-001`, `RULE-NL-001`, `RULE-CODING-001` (voir `data/managed_rules.yaml`).

### Tests

```bash
PYTHONPATH=. pytest tools/rule_studio/tests
```
