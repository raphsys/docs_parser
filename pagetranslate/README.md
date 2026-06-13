# PAGETRANSLATE

Deuxième unité du pipeline WYSIWYG.

Entrée :

```text
PAGEPRINT INPUT_DATA
```

Sortie :

```text
pagetranslate.output.v1
+ translated_input_data
```

## Rôle

`pagetranslate` encapsule le traducteur historique dans une unité autonome qui
consomme PagePrint :

1. lire `INPUT_DATA`, `views.translation_units`, `semantic_system` et `units` ;
2. sélectionner `semantic_phrase > semantic_group > phrase > line > block` ;
3. faire le fallback par bloc, jamais par page entière ;
4. exclure les tokens fins `word/char` et les objets non traduisibles ;
5. déterminer les caractéristiques de phrase :
   début, fin, continuation, ponctuation terminale, type de frontière ;
6. protéger les tokens intouchables avant l'appel traducteur ;
7. appeler `DocumentTranslator.translate_text(...)` ou un traducteur injectable ;
8. contrôler qualité, expansion, tokens protégés, nombres et unités ;
9. réinjecter `content.translated_text` dans une copie de l'INPUT_DATA.

## Modules

- `selector.py` : sélection sémantique et fallback par bloc.
- `sentence_boundary.py` : début/fin de phrase, abréviations, multi-ligne,
  hard break / soft wrap.
- `protection.py` : placeholderisation puis restauration des URLs, DOI, emails,
  nombres, unités, formules, chemins et références.
- `context_builder.py` : contexte avant/après, page, domaine, style, ton et
  contraintes WYSIWYG.
- `translator_bridge.py` : appel propre à `DocumentTranslator`, retry si sortie
  vide ou identique.
- `quality.py` : contrôles qualité et `needs_review`.
- `projection.py` : réinjection dans `INPUT_DATA` traduit et vue compatible
  reconstruction.
- `schema.py` : constantes et DTOs du contrat de traduction.

## Méthodologie

La V1 reprend la logique existante d'`ocr_server.py` / `translator.py` :

- unités textuelles sémantiques quand elles existent ;
- contexte de page et de document ;
- style, ton, domaine, terminologie ;
- stratégies `semantic_reflow`, `layout_constrained`, `exact_preserve` ;
- protection des éléments non traduisibles ;
- contrôle de longueur, qualité de sortie et risque overflow WYSIWYG.

Elle ajoute le contrat propre attendu après PagePrint :

- PagePrint reste la source de vérité ;
- la traduction est une passe séparée ;
- `word/char` restent auxiliaires, pas des unités de traduction ;
- le résultat est auditable par unité.
