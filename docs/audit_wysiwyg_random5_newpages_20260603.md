# Audit WYSIWYG - random5_newpages_20260603

- Seed: `20260603`
- Pages: `5`
- Documents: `doc1: 4`, `doc2: 1`
- Batch: `results/reconstruction_validation_random5_newpages_20260603`

## Conclusion globale

Les corrections récentes ont bien porté sur ce lot:

- le texte est conservé sur 4 pages sur 5 à `100%`;
- les overlays source sont à `0`;
- les glyphes parasites sont à `0`;
- les retours à la ligne tiennent bien;
- l’ordre de lecture reste haut;
- la géométrie est globalement stable, sauf sur `doc2_p11`.

Le lot montre toutefois deux fragilités persistantes:

1. les pages d’abréviations / glossaires restent le point dur structurel;
2. les zones avec légendes, URLs, petites lignes et blocs très compacts restent sensibles au reflow et au collage de mots.

## Verdict par page

| Page | Publication-ready | Verdict synthétique | Impact des corrections |
|---|---|---|---|
| `doc1_p01` | Oui, avec réserve sur le style | Page de couverture propre, texte placé correctement, aucune collision visible. Les titres secondaires et signatures restent lisibles. Le style reste le point faible, pas la géométrie. | Corrections positives. Pas de régression visible. |
| `doc1_p18` | Non | Le contenu principal est là, mais une ligne de fin est tronquée et le bloc texte a perdu de la précision géométrique. Le contrat de tokens protégés est incomplet sur cette page. | Corrections insuffisantes sur ce cas de texte long avec URL et fragments protégés. |
| `doc1_p34` | Partiellement | Page globalement lisible, bonne conservation du contenu et des blocs. La légende de figure reste fragile et déclenche un défaut de chevauchement local. | Corrections utiles sur le texte général, mais pas assez sur caption / figure. |
| `doc1_p35` | Oui, avec réserve sur le style | Page solide. Les paragraphes sont bien reconstruits, la hiérarchie tient, la figure est propre. Quelques écarts de style et de ratio de police restent visibles. | Corrections globalement positives. |
| `doc2_p11` | Non | Page d’abréviations encore fragile: texte présent, mais la géométrie est mauvaise, plusieurs entrées sont condensées, et certains termes techniques sont mal normalisés. | C’est le cas limite du batch. Les corrections n’ont pas encore suffisamment couvert le mode glossaire dense. |

## Analyse détaillée par page

### `doc1_p01`

#### Lecture visuelle

- La couverture reste propre et publication-compatible à distance.
- Le titre principal garde sa hiérarchie.
- Le texte français est lisible et ne chevauche pas les autres zones.
- Les noms propres et les signatures gardent leur position visuelle.

#### Blocs

| Bloc | Rôle | Observation visuelle | Écart principal |
|---|---|---|---|
| `n_0` | body | Titre principal traduit correctement, bloc plus compact que la source | Style plus faible que l’original |
| `n_1` | title | Nom d’auteur lisible, bonne conservation | Faible dérive de taille |
| `n_2` | title | Marque éditeur bien rendue | Stable |
| `n_3` | title | Logo/éditeur un peu déplacé visuellement, mais cohérent | Légère perte de finesse |

#### Phrase / expression

- `Deep Learning for Vision Systems` -> `Apprendre profondément pour Systèmes de vision`
  - traduction sémantiquement faible, mais visuellement stable;
  - le problème ici n’est pas la collision, c’est la qualité de l’expression cible.
- `Mohamed Elgendy`, `Manning`, `Shelter Island`
  - conservés correctement comme entités nominales.

#### Verdict

La page est rendue correctement sur le plan visuel, mais la qualité de traduction du titre reste discutable. Pas de régression de rendu.

### `doc1_p18`

#### Lecture visuelle

- C’est la page la plus fragile du lot côté texte long.
- Le bloc principal a l’air correct globalement, mais un fragment final est tronqué.
- La zone avec URL et phrases longues montre une densité trop forte.
- La page n’est pas publication-ready à cause de cette perte locale.

#### Blocs

| Bloc | Rôle | Observation visuelle | Écart principal |
|---|---|---|---|
| `n_0` | header | En-tête propre | Stable |
| `n_1` | body | Texte long, mais un fragment est tronqué en fin de bloc | Présence partielle, tokens protégés incomplets |

#### Phrase / expression

- URL `https://livebook.manning.com/...`
  - visuellement trop proche du texte courant;
  - la zone a perdu un peu de respiration;
  - c’est typiquement un cas où le pipeline doit préserver la forme de l’adresse et la séparation des lignes.
- Dernière partie de paragraphe
  - la ligne finale semble coupée ou incomplète;
  - le rendu montre une perte de fin de phrase.

#### Verdict

Le texte principal est là, mais la reconstruction n’est pas assez robuste sur une page riche en URL et fragments protégés. Ce n’est pas une régression globale, c’est un trou ponctuel de contrat.

### `doc1_p34`

#### Lecture visuelle

- Page globalement correcte et lisible.
- Les grands blocs de texte tiennent bien.
- La figure et sa légende restent le point faible.
- Le rendu est proche du WYSIWYG mais pas assez strict sur la caption.

#### Blocs

| Bloc | Rôle | Observation visuelle | Écart principal |
|---|---|---|---|
| `n_0` | header | Correct, mais style légèrement affaibli | Style |
| `n_1` | body | Paragraphe stable | Bon |
| `n_4` | body | Texte bien placé dans la zone image/caption | Bon |
| `n_7` | title | Sous-titre lisible | Bon |
| `n_10` | figure_caption | Défaut de chevauchement local, caption trop serrée | Rendement figure-aware insuffisant |
| `n_2` | body | Bloc long bien conservé | Bon |
| `n_3` | body | Bon texte, style un peu adouci | Style |

#### Phrase / expression

- `Figure 1.9 Generative adversarial networks (GANs)...`
  - traduction correcte au fond;
  - la légende est la seule vraie anomalie visuelle;
  - un recouvrement local est signalé et visible.

#### Verdict

Bonne page de validation générale. Le pipeline tient le texte et la hiérarchie, mais la caption de figure doit être traitée comme une zone semi-figée.

### `doc1_p35`

#### Lecture visuelle

- Page très correcte au niveau publication.
- Hiérarchie, paragraphes et figure tiennent ensemble.
- Le style reste un peu trop lissé sur quelques blocs.
- Aucune collision grave visible.

#### Blocs

| Bloc | Rôle | Observation visuelle | Écart principal |
|---|---|---|---|
| `n_0` | header | En-tête bien placé | Bon |
| `n_1` | section_heading | Titre bien rendu | Style un peu léger |
| `n_5` | section_heading | Titre propre | Bon |
| `n_8` | body | Légende et phrase d’intro un peu serrées | Style |
| `n_7` | body | Très bon bloc, stable | Bon |
| `n_2` | body | Reflow propre | Bon |
| `n_3` | body | Bonne conservation du contenu | Style légèrement affaibli |
| `n_6` | body | Bon bloc, lisible | Style |

#### Phrase / expression

- `Face recognition (FR)` / `Image recommendation system`
  - traduction sémantique acceptable;
  - les termes techniques restent lisibles;
  - pas de perte de structure.
- Les zones avec `GANs`, `Edmond de Belamy`, et la figure sont bien tenues.

#### Verdict

Cette page confirme que les corrections n’ont pas cassé le rendu général. Les écarts restants sont surtout typographiques.

### `doc2_p11`

#### Lecture visuelle

- C’est le cas limite du batch.
- La page n’est pas publication-ready.
- Le texte est présent, mais les colonnes d’abréviations sont trop compressées.
- Plusieurs expressions sont mal traduites ou trop littéralisées.
- La géométrie est très faible malgré l’absence d’overlay et de glyphes cassés.

#### Blocs

| Bloc | Rôle | Observation visuelle | Écart principal |
|---|---|---|---|
| `n_0` | header | Titre propre | Bon |
| `n_1_abbr_*` | body | Liste d’abréviations très dense, condensée, avec mots collés | Géométrie et normalisation |
| `n_2` | footer | Numéro de page stable | Bon |

#### Points de rupture

- `Convolutional deep belief networks`
  - rendu français trop littéral ou mal normalisé;
  - plusieurs entrées perdent leur forme.
- `CNN-AFC`, `ConvNet`, `DenseNet`, `ImageNet`
  - collision de tokens et répétitions visuelles;
  - les ensembles d’abréviations ne sont pas assez protégés.
- `Gabor wavelet transformer`, `Kullback-Leibler`, `IIIT-D`, `ILSVRC`
  - certains éléments restent correctement conservés;
  - d’autres sont compressés ou collés.

#### Verdict

La correction actuelle améliore le cadre général, mais elle ne suffit pas encore pour un glossaire dense à deux colonnes. Le mode `glossary_pair` doit être renforcé dans la traduction et dans le reflow.

## Synthèse des effets des corrections

| Famille | Effet observé | Conclusion |
|---|---|---|
| Texte principal | Très bon sur 4 pages sur 5 | Corrigé de manière utile |
| Surimpressions source | 0 | Amélioration nette |
| Glyphes | 0 | Stable |
| Ordre de lecture | Très bon | Stable |
| Styles | Encore faibles sur les pages de couverture / denses | Pas réglé |
| URL / fragments protégés | Fragile sur `doc1_p18` | À renforcer |
| Captions de figure | Fragile sur `doc1_p34` | À traiter comme zone spéciale |
| Glossaire / abréviations | Casse encore sur `doc2_p11` | Point dur restant |

## Conclusion opérationnelle

Les corrections ont porté:

- elles n’ont pas introduit de surimpression,
- elles n’ont pas cassé l’ordre de lecture,
- elles ont stabilisé la conservation du texte sur la plupart des pages,
- elles ont amélioré la robustesse globale du rendu.

Elles ont aussi révélé le reste du chantier:

- le style n’est pas encore suffisamment fidèle;
- les captions et figures doivent être traitées comme des zones à contrat spécial;
- les pages de glossaires / abréviations restent le vrai test de résistance du pipeline.
