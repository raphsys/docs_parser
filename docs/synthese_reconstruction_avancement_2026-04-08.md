# Synthese Reconstruction - 2026-04-08

## Contexte

L'objectif du moteur de reconstruction est le suivant :

- conserver les `blocks` fixes a leur place ;
- reconstruire bloc par bloc ;
- reconstruire les unites extraites dans l'ordre spatial et logique ;
- rendre une `phrase` comme une phrase traduite complete ;
- rendre un `span/expression` hors phrase comme un span/expression traduit ;
- rendre un nombre/chiffre comme le meme nombre/chiffre ;
- conserver strictement les attributs typographiques :
  - police
  - taille
  - couleur
  - gras
  - italique
  - souligne
  - casse
- respecter la continuite ou le retour a la ligne ;
- assurer en priorite la presence complete des extractions dans le rendu final.

Le point central fixe par la spec est :

- pas de logique specifique par type de page ;
- pas de logique TOC, formulaire, page technique, etc. ;
- une logique unique, universelle, pilotee par les extractions et leur geometrie relative.

## Ce qui a ete fait

### 1. Travail sur l'extraction

Nous avons enrichi l'extraction pour disposer d'un niveau de description beaucoup plus riche :

- `blocks`
- `lines`
- `semantic_phrases`
- `semantic_spans`
- `semantic_runs`
- `semantic_groups`
- `editorial_semantics`
- `editorial_relations`
- `expression_semantics`
- `expression_relations`
- `relative_geometry`
- `layout_attributes`
- `style_attributes`
- `text_attributes`

Nous avons aussi ajoute un frontend d'inspection permettant de visualiser :

- les bboxes
- les unites extraites
- les attributs
- les relations
- les correspondances phrase / ligne / bloc / span

Cela a permis de verifier un point essentiel :

- la qualite finale de reconstruction depend enormement de la qualite et du type des unites extraites ;
- certaines unites extraites servent a l'analyse et ne doivent pas etre utilisees telles quelles comme unites de rendu final.

### 2. Travail sur la spec de reconstruction

Deux documents ont ete rediges dans `scripts/` :

- [algorithme_reconstruction_hierarchique.md](/home/raphael/Mes_Projets/docs_parser/scripts/algorithme_reconstruction_hierarchique.md)
- [plan_integration_spec_dans_reconstructor.md](/home/raphael/Mes_Projets/docs_parser/scripts/plan_integration_spec_dans_reconstructor.md)

Ils definissent :

- le contrat de reconstruction ;
- l'architecture cible ;
- les dataclasses ;
- le pipeline bloc par bloc ;
- la separation entre extraction, planification, layout et dessin ;
- le principe d'un moteur hiérarchique.

### 3. Refonte du reconstructor

Le vieux moteur a ete sauvegarde en :

- [reconstructor.py.bak](/home/raphael/Mes_Projets/docs_parser/reconstructor.py.bak)

Le nouveau moteur a ete reintroduit dans :

- [reconstructor.py](/home/raphael/Mes_Projets/docs_parser/reconstructor.py)

Avec notamment :

- `BlockReconstructionPlan`
- `LineTemplate`
- `PlacableUnit`
- `GraphEdge`
- `BlockRenderOp`
- `EditorialBlockRenderer`
- `HeadingBlockRenderer`
- `CaptionBlockRenderer`
- `AnnotationBlockRenderer`
- `CodeBlockRenderer`
- `TableBlockRenderer`

### 4. Correctifs deja obtenus

Plusieurs causes importantes ont deja ete identifiees et corrigees.

#### 4.1 Fonds blancs locaux

Probleme observe :

- le moteur posait correctement un `bg_master` nettoye ;
- puis repeignait encore localement un rectangle blanc sur le bloc ;
- puis dessinait le texte dessus.

Correction :

- si la page a deja un fond propre (`background_path` / `bg_master`), le moteur ne force plus un `whiteout` local systematique.

Effet :

- disparition ou forte reduction des aplats blancs derriere les paragraphes.

#### 4.2 Duplication de contenu dans les blocs

Probleme observe notamment sur la page `148` :

- presque tout le texte etait present ;
- mais il etait rejoue plusieurs fois ;
- le paragraphe "begayait" ;
- le meme contenu ressortait a plusieurs granularites.

Cause :

- des `semantic_phrases` glissantes et chevauchantes etaient traitees comme unites de rendu final ;
- leur texte etait parfois reconstruit a partir des lignes traduites ;
- plusieurs fenetres de phrases recouvrant les memes lignes etaient toutes dessinees.

Correction :

- detection des `semantic_phrases` chevauchantes ;
- dans ce cas, remplacement par des `translated_line` disjointes pour le rendu final.

Effet :

- forte baisse des duplications ;
- forte baisse des overlaps sur `148`.

#### 4.3 Presence des extractions

Probleme observe :

- certaines pages n'affichaient presque aucun texte ;
- seuls quelques petits titres ou labels passaient ;
- les fonds etaient visibles mais pas les traductions.

Cause :

- le moteur rejetait trop de blocs apres validation ;
- certaines unites n'etaient pas correctement remontees comme candidates au rendu.

Corrections successives :

- meilleure remontee des traductions depuis les lignes ;
- meilleure prise en compte des unites externes ;
- correction du fallback de presence ;
- desactivation de certaines heuristiques trop agressives.

Effet :

- forte progression du `rendered_text_coverage_score` sur plusieurs pages.

## Ce qui a ete observe sur les pages de test

### Page 148

Cas directeur utile pour les duplications et le fond blanc.

Etat apres corrections importantes :

- `rendered_text_coverage_score`: `0.95`
- `word_overlaps`: `126 -> 4`
- `visual_similarity_score`: `0.7308 -> 0.8019`

Conclusion :

- le contenu est presque complet ;
- la duplication structurelle a ete largement corrigee ;
- le moteur a bien progresse sur ce type de bloc editorial.

### Page 8

Cas directeur utile pour les blocs denses et les rangées complexes.

Etat actuel :

- `rendered_text_coverage_score`: `0.9524`
- `rendered_covered_units`: `79 / 84`
- `word_overlaps`: `51`
- `visual_similarity_score`: `0.7619`

Conclusion :

- la presence des extractions est deja tres bonne ;
- le probleme principal restant est la composition locale des unites dans certains blocs denses ;
- ce n'est plus un probleme de "texte absent", mais un probleme de layout final.

## Ce qui ne marche pas encore, de facon generale

Le vrai probleme general n'est plus :

- l'OCR
- la traduction
- la conservation basique des styles

Le vrai probleme general est plus profond :

- le moteur ne transforme pas encore toujours les extractions en une representation finale canonique par bloc avant de dessiner.

Autrement dit, il manque encore une etape solide entre :

- extraction / enrichissement

et

- dessin final sur la page.

## Diagnostic general de l'algorithme actuel

### 1. Le moteur ne resout pas completement le layout avant dessin

Il fait encore trop souvent :

- prendre des unites
- les dessiner
- puis constater qu'il y a overlap ou composition incorrecte

alors qu'il devrait faire :

- construire les unites finales d'un bloc
- construire les rangées finales
- resoudre les conflits locaux
- verifier la couverture
- dessiner ensuite seulement

### 2. Les unites extraites ne sont pas encore converties partout en unites de rendu finales

Aujourd'hui remontent encore parfois jusqu'au dessin :

- des phrases d'analyse
- des lignes OCR eclatees
- des segments externes page-level
- des fragments utiles a la comprehension mais pas au rendu final

Le moteur devrait d'abord fabriquer, pour chaque bloc, une representation canonique :

- quelles unites sont finales
- quel niveau on garde
- quel ordre on garde
- quelles unites sont fusionnees
- quelles unites sont exclues du dessin

### 3. L'appartenance bloc -> rangée -> unite reste fragile

Pour certaines pages denses, il faut encore mieux resoudre :

- a quel bloc une unite appartient
- a quelle rangée locale elle appartient
- si elle est continue avec la precedente
- si elle doit etre seule sur sa ligne

Quand ce maillon est imparfait, on obtient :

- collisions
- compactage excesif
- ordre visuel imparfait

### 4. Il manque encore un vrai solveur de conflits local

Quand plusieurs unites veulent la meme zone, le moteur devrait pouvoir choisir robustement entre :

- maintien inline
- retour a la ligne
- repli sur plusieurs lignes
- extension sur la rangée suivante
- compactage typographique borne
- maintien strict des nombres et labels proteges

Aujourd'hui, cela existe partiellement, mais pas encore comme couche generale et stable.

### 5. Le moteur reste encore trop "dessin oriente"

Il manque encore un `BlockReconstructionPlan` entierement resolu, avec :

- unites finales du bloc
- rangées finales
- ordre final
- regles de rupture finales
- slots relatifs finaux
- verification de couverture avant dessin

En l'etat, nous avons un bon debut de planification, mais pas encore une normalisation finale suffisamment forte.

## Cause generale du non-100%

La raison generale pour laquelle nous n'avons pas encore 100% est donc :

- nous savons de mieux en mieux extraire ;
- nous savons de mieux en mieux traduire ;
- nous savons deja souvent conserver le style ;
- mais nous ne construisons pas encore de facon parfaitement fiable la structure finale de rendu du bloc avant le dessin.

En une phrase :

- le moteur reconstruit encore trop directement depuis des unites extraites,
- alors qu'il devrait d'abord resoudre completement la structure finale de rendu du bloc.

## Ce vers quoi on va

La direction de travail est maintenant claire.

### Etape cible manquante

Il faut introduire une vraie couche bloc-locale finale, entre extraction et dessin :

1. normalisation des unites
2. choix du niveau final de rendu pour le bloc
3. construction des rangées finales
4. resolution des conflits locaux
5. verification de couverture complete
6. dessin final

### Objectif exact

Pour chaque bloc :

- partir des extractions ;
- construire une representation finale unique et non ambigue ;
- garantir que toutes les extractions attendues du bloc sont presentes ;
- garantir que leur ordre, leur style et leur relation spatiale restent coherents ;
- dessiner ensuite seulement.

### Priorite de travail

Priorite absolue :

- presence complete des extractions traduites

mais

- dans le cadre strict de la logique de reconstruction universelle definie plus haut.

Cela veut dire :

- pas de logique par type de page ;
- pas de logique TOC speciale ;
- pas de heuristique metier par categorie de page ;
- une logique unique pilotee par :
  - les blocs
  - les phrases
  - les spans/expressions
  - les nombres
  - la geometrie relative
  - les attributs typographiques
  - la continuite / rupture

## Ce que Claude Code devra comprendre rapidement

S'il doit reprendre le sujet, les points les plus importants sont :

1. Le sujet n'est plus l'extraction brute seule.
2. Le sujet n'est plus la traduction seule.
3. Le sujet n'est plus la police seule.
4. Le vrai verrou est la transformation des extractions en structure finale de rendu par bloc.
5. Le moteur doit etre universel, pas specifique aux types de page.
6. Le bloc reste fixe.
7. Les phrases se rendent comme phrases.
8. Les spans/expressions hors phrase se rendent comme spans/expressions.
9. Les nombres se rendent comme les memes nombres.
10. La couverture complete du rendu est la priorite, mais sans casser la logique de reconstruction.

## Fichiers principaux a lire

- [reconstructor.py](/home/raphael/Mes_Projets/docs_parser/reconstructor.py)
- [reconstructor.py.bak](/home/raphael/Mes_Projets/docs_parser/reconstructor.py.bak)
- [ocr_server.py](/home/raphael/Mes_Projets/docs_parser/ocr_server.py)
- [translator.py](/home/raphael/Mes_Projets/docs_parser/translator.py)
- [tests/test_translation_enrichment.py](/home/raphael/Mes_Projets/docs_parser/tests/test_translation_enrichment.py)
- [algorithme_reconstruction_hierarchique.md](/home/raphael/Mes_Projets/docs_parser/scripts/algorithme_reconstruction_hierarchique.md)
- [plan_integration_spec_dans_reconstructor.md](/home/raphael/Mes_Projets/docs_parser/scripts/plan_integration_spec_dans_reconstructor.md)

## Resultats utiles recents

### Page 148

- [comparaison](/home/raphael/Mes_Projets/docs_parser/results/test_docintelligence-148_rewrite_direct_units8_20260408/page_001_side_by_side.png)
- [rendu](/home/raphael/Mes_Projets/docs_parser/results/test_docintelligence-148_rewrite_direct_units8_20260408/page_001_translated.png)
- [PDF](/home/raphael/Mes_Projets/docs_parser/results/test_docintelligence-148_rewrite_direct_units8_20260408/reconstructed/fr_translated_reconstructed.pdf)
- [QA](/home/raphael/Mes_Projets/docs_parser/results/test_docintelligence-148_rewrite_direct_units8_20260408/reconstructed/publication_qa.json)

### Page 8

- [comparaison](/home/raphael/Mes_Projets/docs_parser/results/test_docintelligence-8_rewrite_direct_units12_20260408/page_001_side_by_side.png)
- [rendu](/home/raphael/Mes_Projets/docs_parser/results/test_docintelligence-8_rewrite_direct_units12_20260408/page_001_translated.png)
- [PDF](/home/raphael/Mes_Projets/docs_parser/results/test_docintelligence-8_rewrite_direct_units12_20260408/reconstructed/fr_translated_reconstructed.pdf)
- [QA](/home/raphael/Mes_Projets/docs_parser/results/test_docintelligence-8_rewrite_direct_units12_20260408/reconstructed/publication_qa.json)

## Resume final

Nous avons deja fait beaucoup :

- meilleure extraction
- meilleure instrumentation
- meilleure formalisation
- nouveau reconstructor
- nette progression sur la presence du contenu
- reduction forte de certains defects majeurs

Mais le passage final vers le 100% demande encore une couche manquante :

- une normalisation finale des unites de rendu par bloc,
- suivie d'une vraie resolution de layout locale avant dessin.

Tant que cette couche n'est pas totalement stabilisee, le moteur peut :

- bien couvrir le contenu
- bien conserver certains styles
- bien tenir la macro-geometrie

mais il n'atteint pas encore la fidelite totale de reconstruction.
