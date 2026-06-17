# Audit WYSIWYG random20 - newpages - 2026-06-01

Source de l'audit: `results/reconstruction_validation_random20_newpages_20260601_seed20260605`.

Perimetre: 20 pages nouvelles, selectionnees sans recouvrement avec `results/reconstruction_validation_random20_20260601_afterfix4`.

Objectif: analyse page -> blocs -> phrases/expressions -> lignes, puis comparaison avec l'original et la page reconstruite, avec remontee des causes dans le pipeline, le code et les branchements de fonctions.

## Synthese globale

- Texte present: `98.39%`
- Texte dans bloc d'origine: `91.23%`
- Ordre de lecture: `99.83%`
- Geometrie moyenne: `0.779`
- Styles visibles: `66.79%`
- Render findings: `22`
- Glyphes: `2`
- Cellules tableau en echec: `0`
- Tokens proteges reinserees: `90.00%`
- Non traduits reinserees: `98.57%`
- Surimpressions source: `2`
- Mots potentiellement colles: `12`

Lecture technique:

- Le pipeline tient globalement le texte, mais la conservation des styles, des fonts et des geometries reste fragile sur des pages structurees.
- Les echecs ne sont plus massivement des pertes de texte, ce sont surtout des erreurs de contrat de rendu, de classification de bloc, de redistribution de lignes et de preservation visuelle.
- Plusieurs pages restent "texte OK / style KO" ou "texte OK / geometrie KO". Ce sont les bons signaux pour un plan de corrections universel: il faut corriger les mecanismes, pas les cas.

## Analyse page par page

| Page | Symptome dominant | Lecture bloc/ligne | Cause pipeline probable | Priorite |
|---|---|---|---|---|
| `doc3_p00` | couverture fragmentee | lettres de titre splittees en blocs `title` separes, IoU tres bas (`0.39`), 1 render finding | classification/assemblage des titres de couverture, probablement trop litteral et trop morcelle | P1 |
| `doc1_p03` | propre | dedications et lignes courtes rendues correctement, aucun finding | cas de reference stable | baseline |
| `doc1_p33` | texte OK, style faible | titres visuels et petites etiquettes, style `75%`, 2 render findings | font floor trop agressif sur labels/titres courts | P1 |
| `doc1_p08` | TOC fragile | lignes de sommaire, joints de mots et retours ligne fragiles, 5 render findings, `line_breaks_ok=95%` | TOC / leaders / labels courts insuffisamment contraints | P1 |
| `doc1_p12` | geometrie cassee | texte present mais `origin=0`, `IoU=0`, `style=0`; aucun bloc exploitable | echec de re-mappage bloc/page ou de back-projection geometrie | P0 |
| `doc1_p10` | TOC/figure-label mix | titres de sommaire, quelques overlays, 3 render findings | routage renderer hybride et preservation des petits labels | P1 |
| `doc2_p08` | propre | TOC stable, leaders et pages corrects | cas de reference stable | baseline |
| `doc2_p26` | propre | figure caption et body propres, styles stables | cas de reference stable | baseline |
| `doc2_p00` | style faible | page auteur/signature, 5 render findings, style header `50%` | exact-preserve des noms/profils et font floor insuffisants | P1 |
| `doc2_p30` | glyphes et blocs manquants | bloc math/ligne de formule, `glyph=1`, `disappeared=n_5`, 3 render findings | nettoyage de glyphes de controle, classification math/figure, preservation de symbole | P0 |
| `doc2_p15` | texte OK, style moyen | corps stable mais header/figure caption a `66.67%` | style floor trop bas sur sections/headers | P2 |
| `doc3_p21` | texte OK, style moyen | bloc body stable, bloc inline exotique en `equation_inline` | style floor et classification d'inline exotique | P2 |
| `doc3_p17` | stable | texte/origine/styles globalement bons | baseline utile | baseline |
| `doc3_p29` | overlay source | body complet mais `source_overlay=3`, style `0%` sur un bloc body | nettoyage fond/reconstruit incomplet ou faux positif d'overlay apres reflow long | P0 |
| `doc3_p23` | texte OK, style moyen | body stable, inline exotique style `50%` | style floor et normalisation des inline tokens | P2 |
| `doc1_p02` | overlay/glyphes | front matter / legal, 2 overlay warnings, 1 glyph warning | exact-preserve des mentions legales / names / urls insuffisant | P1 |
| `doc1_p37` | style tres faible | paragraphe long + listes a puces, style `35.71%`, 2 findings | redistribution de lignes et markers de liste trop agressifs | P0 |
| `doc2_p09` | propre | TOC/chapitre stable | baseline utile | baseline |
| `doc2_p13` | texte OK, style moyen | corps stable, footer faible | style floor sur body/footer | P2 |
| `doc1_p32` | figure caption fragile | caption + header, style `33.33%`, 1 render finding | rendu de caption/figure en slot trop petit ou pas assez ancre | P1 |

## Observations detaillees

### 1. Pages propres

`doc1_p03`, `doc2_p08`, `doc2_p26`, `doc2_p09`, `doc3_p17` sont de bons baselines. Elles montrent que le pipeline sait encore produire une page WYSIWYG quand:

- le bloc est bien classe
- la traduction reste proche de la source en volume
- le renderer choisi est coherent avec le contrat

Ces pages doivent rester dans le jeu de regression.

### 2. Pages "texte OK / style KO"

`doc1_p33`, `doc2_p15`, `doc3_p21`, `doc3_p23`, `doc2_p13`, `doc1_p32`, `doc3_p29` montrent un pattern commun:

- le texte est present
- la geometrie est acceptable ou bonne
- le style, la famille ou la taille tombe trop bas

Ce n'est pas un probleme de traduction. C'est un probleme de conservation de contrat typographique.

### 3. Pages a geometry/origin cassees

`doc1_p12` est le signal le plus net:

- texte present a 100%
- mais `origin=0`, `IoU=0`, `style=0`

Cela indique une rupture de correspondance entre la structure reconstruite et la validation geometrique. Le pipeline a bien emise du texte, mais il n'a pas conserve de correspondance exploitable avec le bloc source.

### 4. Pages a forte erreur de redistribution

`doc2_p30` et `doc1_p37` montrent le risque majeur des longues lignes preservees:

- le texte principal existe
- mais des lignes sont deplacees, des glyphes parasites apparaissent, des blocs peuvent disparaitre
- la logique de redistribution et de rebalancing devient plus agressive que la page ne le permet

### 5. Pages a overlay source

`doc3_p29` et `doc1_p02` montrent le probleme inverse:

- le texte est present
- mais l'ancienne source reste visible ou est encore detectee

Ce sont des cas de re-render / cleanup insuffisants, ou de faux positifs de detection si les lignes exact-preserve sont des noms/mentions legales.

### 6. Pages TOC / labels courts

`doc1_p08`, `doc1_p10`, `doc3_p00` et `doc2_p00` confirment que les pages avec:

- leaders
- numerotation
- labels courts
- titres de couverture

restent fragiles si le renderer ne garde pas un contrat de slots strict ou si la ligne est traduite en paragraphe trop large.

## Causes universelles reperees

| Probleme | Cause racine | Fonctions en cause | Type de cause | Correction universelle a implementer | Effet attendu |
|---|---|---|---|---|---|
| Expansion de lignes preservees trop forte | la redistribution de paragraphes peut reconstituer des lignes trop longues ou dupliquees | [translator.py](</home/raphael/Mes_Projets/docs_parser/translator.py:3805>), [translator.py](</home/raphael/Mes_Projets/docs_parser/translator.py:3962>), [translator.py](</home/raphael/Mes_Projets/docs_parser/translator.py:6389>) | traduction + rebalancing | ajouter une reparation finale des lignes `preserved` avec garde d'expansion et retours au source pour les lignes purement nominatives si besoin | plus de lignes interminables, moins de texte qui fuit hors bloc |
| Derniere ligne de queue signataire ou localisation mal alignee | les fins de bloc structurales se font absorber par le paragraphe courant | [translator.py](</home/raphael/Mes_Projets/docs_parser/translator.py:3805>), [translator.py](</home/raphael/Mes_Projets/docs_parser/translator.py:6428>) | logique de branchement | detecter les queues signataires / localisations / noms propres et les conserver ligne par ligne, avec ancrage final | signatures lisibles, pas de permutation de noms |
| Routage renderer trop permissif | un bloc `line_preserve` ou hybride peut tomber dans un renderer qui n'est pas assez contraint | [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:6960>), [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:7150>), [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:4561>) | branchement / contrat de rendu | renforcer la selection de renderer et preferer linewise ou slotwise seulement si le contrat est vraiment satisfaisant | moins de pages rendues "correctes mais mal placees" |
| Style/famille/font size qui s'effondre | le renderer mesure la hauteur du slot mais laisse tomber le font floor de la source | [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:6991>), [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:7053>), [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:836>) | typographie | garder la taille source comme reference, borner la reduction, reporter le fallback de police et les flags | meilleure preservation visuelle, moins de style KO |
| Geometry/origin perdues alors que le texte est la | la page est valide textuellement mais plus associee au bon bloc source | [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:6804>), [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:6825>) | geometrique | ajouter une re-association de bloc et une relecture de bbox quand `origin=0` ou `IoU=0` mais texte present | restaurer les blocs sans casser le texte |
| Overlay source residuel | le fond est nettoye ou re-rendu mais la source reste detectee | [ocr_server.py](</home/raphael/Mes_Projets/docs_parser/ocr_server.py:4689>), [ocr_server.py](</home/raphael/Mes_Projets/docs_parser/ocr_server.py:4802>), [coverage_validator.py](</home/raphael/Mes_Projets/docs_parser/coverage_validator.py>) | nettoyage / validation | ajouter un second passage de cleanup + re-render si la source reste visible, avec heuristique qui distingue les noms exact-preserve des vrais overlays | overlay count a 0 sans sacrifier les noms propres |
| Glyphes de controle / symboles parasites | certains tokens OCR ou math sont encore propagés comme caracteres invisibles ou glyphes bruts | [ocr_server.py](</home/raphael/Mes_Projets/docs_parser/ocr_server.py:4725>), [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:818>) | normalisation texte | normaliser les glyphes de controle en amont et traiter les symboles mathematiques via contrat exact-preserve | moins de glyph failures et moins de blocs disparus |
| TOC / labels courts / leaders | les labels courts et leaders sont soit traduits trop librement, soit remis dans des lignes trop longues | [translator.py](</home/raphael/Mes_Projets/docs_parser/translator.py:2740>), [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:7149>) | typographie structurale | imposer le mode slotwise exact pour TOC/labels, conserver numerotation et leaders, et limiter les expansions | TOC lisibles, moins de render findings |
| Figures/captions et listes a puces | les puces, sous-titres et captions sont dupliques ou perds dans le reflow | [translator.py](</home/raphael/Mes_Projets/docs_parser/translator.py:2740>), [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:7112>) | layout hybride | ajouter des contrats explicites pour caption/list/figure-label et privilegier l'ancrage avant le reflow | captions stables et listes sans duplication |

## Plan de corrections universel

| Priorite | Correction generale | A quoi elle s'applique | Fonction(s) a toucher | Effet attendu |
|---:|---|---|---|---|
| P0 | Reparation finale des lignes preservees | toutes les pages avec longs paragraphes, signatures, postes, TOC hybrides | [translator.py](</home/raphael/Mes_Projets/docs_parser/translator.py:3805>), [translator.py](</home/raphael/Mes_Projets/docs_parser/translator.py:6389>) | plus de lignes gonflees ou de noms permutes |
| P0 | Reconciliation bloc/geometrie quand le texte est la mais le bloc est faux | pages type `doc1_p12` | [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:6804>), [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:6825>) | `origin` et `IoU` remontent sans casser le texte |
| P0 | Cleanup source overlay avec second passage de rendu | pages front matter, legal, chapter bodies, figures | [ocr_server.py](</home/raphael/Mes_Projets/docs_parser/ocr_server.py:4689>), [ocr_server.py](</home/raphael/Mes_Projets/docs_parser/ocr_server.py:4802>) | overlays a zero sur les cas reels |
| P1 | Preservation typographique stricte | pages a style faible mais texte correct | [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:6991>), [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:7053>), [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:836>) | meilleurs ratios de style, tailles plus proches de la source |
| P1 | Contractualisation forte des TOC, labels courts et leaders | sommaires, titres de chapitre, pages de couverture | [translator.py](</home/raphael/Mes_Projets/docs_parser/translator.py:2740>), [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:7149>) | moins de pertes sur `toc_front`, plus de lignes exactes |
| P1 | Normalisation des glyphes et caracteres de controle | equa, formules, symboles, OCR parasite | [ocr_server.py](</home/raphael/Mes_Projets/docs_parser/ocr_server.py:4725>), [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:818>) | moins de glyph failures et de blocs disparus |
| P2 | Router les captions/listes/figures vers un contrat adapte | pages avec figures, bullets, captions | [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:7112>), [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:6960>) | moins de duplication de puces et de captions |
| P2 | Garder les pages tres stables comme regression tests fixes | `doc1_p03`, `doc2_p08`, `doc2_p26`, `doc2_p09`, `doc3_p17` | validation et CI | verifier qu'une correction universelle ne casse pas le bon chemin |

## Ordre de mise en oeuvre propose

1. Fermer la geometrie perdue sur les pages a `origin=0` / `IoU=0`.
2. Stabiliser la reparation finale des lignes preservees et des queues signataires.
3. Renforcer les contrats TOC / labels courts / captions.
4. Remonter le font floor et les flags typographiques.
5. Ajouter le second passage de cleanup pour les overlays source.
6. Generaliser la normalisation des glyphes de controle et des symboles math.

## Fonctions a surveiller en priorite

- [translator.py](</home/raphael/Mes_Projets/docs_parser/translator.py:2740>) pour les fallback TOC / short labels.
- [translator.py](</home/raphael/Mes_Projets/docs_parser/translator.py:3805>) pour les paragraphes preserve et la redistribution ligne par ligne.
- [translator.py](</home/raphael/Mes_Projets/docs_parser/translator.py:6389>) pour la reparation finale des lignes preservees.
- [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:6960>) et [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:7150>) pour le routage des renderers structurels.
- [reconstructor.py](</home/raphael/Mes_Projets/docs_parser/reconstructor.py:6991>) pour la construction des slots, hauteurs, alignements et tailles.
- [ocr_server.py](</home/raphael/Mes_Projets/docs_parser/ocr_server.py:4689>) pour l'extraction, le nettoyage et le fond nettoye.

Conclusion:

- Le pipeline sait maintenant sauver le texte sur presque toutes les pages du batch.
- Le point dur n'est plus la traduction brute mais la preservation universelle du contrat visuel et geometrique.
- Les corrections suivantes doivent etre formulees comme des mecanismes generaux, avec fallback et validation, pas comme des rustines de page.
