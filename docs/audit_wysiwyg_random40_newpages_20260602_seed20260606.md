# Audit visuel détaillé random40 nouvelles pages

Batch: `results/reconstruction_validation_random40_newpages_20260602_seed20260606`

Pages analysées: **42** (`41` applicables, `1` N/A)

## Synthèse

| Indicateur | Moyenne |
|---|---:|
| Texte présent | 97.91% |
| Texte dans l'origine | 96.17% |
| Ordre de lecture | 99.89% |
| IoU géométrique | 0.809 |
| Styles | 47.43% |
| Findings de rendu | 35 |
| Verdicts de rendu échoués | 22 |
| Overlays source | 2 |
| Glyphes | 6 |
| Cellules table | 0 |
| Joined words | 54 |

Répartition du lot: `7` publication-ready, `25` lisibles mais fragiles, `9` non publication-ready, `1` non applicable.

## Vue par page

| Page | Doc | État | Texte / origine / IoU / styles / rendus | Lecture visuelle | Blocs à surveiller |
|---|---|---|---|---|---|
| `doc1_p04` | doc1 | **N/A** | 100.0% / 0.0% / 0.000 / 0.0% / RF 0 | Non applicable<br>Structure metrics non applicables; page atypique sans blocs exploitables. | Aucun |
| `doc1_p05` | doc1 | **Non** | 90.9% / 81.8% / 0.592 / 54.5% / RF 3 | Non publication-ready<br>Sommaire trop comprimé: collisions visibles sur les entrées de niveau 1 et les sous-entrées. | n_4:style; glyph; disappear; presence 0.00; iou 0.00; text_missing|text_overlap<br>n_7:disappear; presence 0.00; iou 0.01; text_missing|text_overlap<br>n_5:style; iou 0.80; text_missing|text_overlap<br>+6 autres |
| `doc1_p09` | doc1 | **Non** | 95.0% / 88.9% / 0.850 / 66.7% / RF 14 | Non publication-ready<br>Sommaire globalement bon; la zone dense 6.5/6.6 reste trop serrée et dégrade la finesse locale. | n_13:style; presence 0.86; text_missing|text_overlap<br>n_11:presence 0.73; iou 0.75; text_missing|text_overlap<br>n_15:style; iou 0.42<br>+4 autres |
| `doc1_p13` | doc1 | **Partiel** | 100.0% / 100.0% / 0.874 / 66.7% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_2:style<br>n_1:iou 0.77 |
| `doc1_p14` | doc1 | **Partiel** | 100.0% / 100.0% / 0.938 / 50.0% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_1:style<br>n_4:style |
| `doc1_p21` | doc1 | **Partiel** | 100.0% / 100.0% / 0.649 / 0.0% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_0:style; iou 0.42<br>n_1:style; iou 0.68<br>n_2:style |
| `doc1_p23` | doc1 | **Partiel** | 100.0% / 100.0% / 0.835 / 40.0% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_1:style<br>n_2:style<br>n_4:style<br>+1 autres |
| `doc1_p25` | doc1 | **Non** | 100.0% / 100.0% / 0.848 / 53.8% / RF 1 | Non publication-ready<br>Page de vision humaine globalement lisible; certains labels/figures et une overlay source demeurent fragiles. | n_13:style; text_missing|text_overlap<br>n_5:style; overlay<br>n_1:style<br>+6 autres |
| `doc1_p28` | doc1 | **Partiel** | 100.0% / 100.0% / 0.892 / 16.7% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_0:style<br>n_1:style<br>n_3:style<br>+2 autres |
| `doc1_p30` | doc1 | **Partiel** | 100.0% / 100.0% / 0.924 / 33.3% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_0:style<br>n_4:style<br>n_5:style<br>+3 autres |
| `doc1_p31` | doc1 | **Non** | 86.1% / 77.8% / 0.877 / 66.7% / RF 2 | Non publication-ready<br>Zone de figure/caption en bas de page encore chevauchée; le texte principal reste lisible. | n_3:style; presence 0.73; text_missing|text_overlap<br>n_8:presence 0.33; text_missing|text_overlap<br>n_1:style<br>+3 autres |
| `doc1_p38` | doc1 | **Non** | 90.7% / 86.7% / 0.805 / 26.7% / RF 6 | Non publication-ready<br>Bloc figure+équations trop dense: labels et expressions mathématiques se chevauchent localement. | n_7:glyph; presence 0.43; iou 0.70; text_missing|text_overlap<br>n_17:style; iou 0.62; text_missing|text_overlap<br>n_12:style; text_missing<br>+9 autres |
| `doc2_p01` | doc2 | **Partiel** | 100.0% / 100.0% / 0.762 / 25.0% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_0:style; iou 0.63<br>n_3:style; iou 0.64<br>n_2:style |
| `doc2_p02` | doc2 | **Partiel** | 100.0% / 100.0% / 0.992 / 0.0% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_0:style |
| `doc2_p10` | doc2 | **Partiel** | 100.0% / 100.0% / 0.816 / 33.3% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_0:style<br>n_1:style<br>n_5:iou 0.52 |
| `doc2_p12` | doc2 | **Non** | 70.2% / 69.6% / 0.279 / 67.4% / RF 0 | Non publication-ready<br>Glossaire d’abréviations cassé par expansion et répétition: beaucoup de collisions et de lignes perdues. | n_0_abbr_11:style; disappear; presence 0.00; iou 0.20<br>n_0_abbr_29:style; disappear; presence 0.00; iou 0.31<br>n_0_abbr_31:style; disappear; presence 0.00; iou 0.32<br>+35 autres |
| `doc2_p14` | doc2 | **Partiel** | 100.0% / 100.0% / 0.894 / 16.7% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_1:style<br>n_2:style<br>n_3:style<br>+2 autres |
| `doc2_p16` | doc2 | **Non** | 92.6% / 66.7% / 0.628 / 50.0% / RF 1 | Non publication-ready<br>Paragraphe central trop serré, une partie du bloc est mal ré-encadrée. | 2:presence 0.50; iou 0.34; text_missing|text_overlap<br>block_4:disappear; presence 0.00; iou 0.00; render<br>1:style; iou 0.58<br>+2 autres |
| `doc2_p17` | doc2 | **Partiel** | 100.0% / 100.0% / 0.771 / 42.9% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | 2:style; iou 0.72<br>block_4:style; iou 0.63; render<br>n_1:style; iou 0.65<br>+2 autres |
| `doc2_p19` | doc2 | **Partiel** | 100.0% / 100.0% / 0.950 / 66.7% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_1:style |
| `doc2_p20` | doc2 | **Partiel** | 100.0% / 100.0% / 0.940 / 50.0% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_1:style<br>n_2:style |
| `doc2_p21` | doc2 | **Partiel** | 100.0% / 100.0% / 0.936 / 50.0% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_1:style<br>n_3:style |
| `doc2_p28` | doc2 | **Oui** | 100.0% / 100.0% / 0.502 / 94.1% / RF 0 | Stable, publication-ready<br>Rendu stable et propre à l’œil; pas de collision visible ni d’overlay source. | n_9:style<br>1:iou 0.54<br>10:iou 0.47<br>+10 autres |
| `doc2_p29` | doc2 | **Non** | 88.7% / 71.4% / 0.592 / 47.6% / RF 8 | Non publication-ready<br>Page dense avec plusieurs collisions locales, glyphes parasites et faible cohérence géométrique. | n_10:glyph; presence 0.50; iou 0.34; text_missing<br>n_15:glyph; presence 0.50; iou 0.37; text_missing<br>1:style; iou 0.56; text_missing<br>+13 autres |
| `doc2_p31` | doc2 | **Partiel** | 100.0% / 100.0% / 0.938 / 33.3% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_1:style<br>n_10:style<br>n_5:style<br>+1 autres |
| `doc2_p32` | doc2 | **Partiel** | 100.0% / 100.0% / 0.802 / 40.0% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_1:style<br>n_2:style<br>n_3:style<br>+1 autres |
| `doc2_p39` | doc2 | **Oui** | 100.0% / 100.0% / 0.919 / 80.0% / RF 0 | Stable, publication-ready<br>Rendu stable et propre à l’œil; pas de collision visible ni d’overlay source. | n_2:style |
| `doc3_p02` | doc3 | **Non** | 100.0% / 100.0% / 0.870 / 66.7% / RF 0 | Non publication-ready<br>Non publication-ready: collisions, expansion locale, manque de texte ou reflow insuffisant. | n_1:overlay<br>n_8:style; iou 0.66<br>n_5:style |
| `doc3_p05` | doc3 | **Partiel** | 100.0% / 100.0% / 0.826 / 0.0% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_0:style; iou 0.77<br>n_2:style; iou 0.72<br>n_1:style |
| `doc3_p06` | doc3 | **Partiel** | 100.0% / 100.0% / 0.845 / 31.8% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_16:style; iou 0.79<br>n_18:style; iou 0.75<br>n_21:style; iou 0.66<br>+14 autres |
| `doc3_p07` | doc3 | **Partiel** | 100.0% / 100.0% / 0.708 / 50.0% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_3:style; iou 0.64<br>n_1:style<br>n_2:iou 0.43 |
| `doc3_p08` | doc3 | **Partiel** | 100.0% / 100.0% / 0.818 / 62.5% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_0:style; iou 0.67<br>n_11:style; iou 0.66<br>n_1:style<br>+2 autres |
| `doc3_p19` | doc3 | **Oui** | 100.0% / 100.0% / 0.788 / 75.0% / RF 0 | Stable, publication-ready<br>Rendu stable et propre à l’œil; pas de collision visible ni d’overlay source. | n_3:style; iou 0.64<br>n_2:iou 0.72 |
| `doc3_p22` | doc3 | **Partiel** | 100.0% / 100.0% / 0.858 / 0.0% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_5:style; iou 0.72<br>n_0:style |
| `doc3_p24` | doc3 | **Oui** | 100.0% / 100.0% / 0.841 / 75.0% / RF 0 | Stable, publication-ready<br>Rendu stable et propre à l’œil; pas de collision visible ni d’overlay source. | n_6:style; iou 0.66<br>n_0:iou 0.72 |
| `doc3_p26` | doc3 | **Partiel** | 100.0% / 100.0% / 0.865 / 40.0% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_4:style; iou 0.74<br>n_7:style; iou 0.66<br>n_0:style |
| `doc3_p31` | doc3 | **Partiel** | 100.0% / 100.0% / 0.776 / 33.3% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_6:style; iou 0.66<br>n_0:style<br>n_2:style<br>+3 autres |
| `doc3_p32` | doc3 | **Partiel** | 100.0% / 100.0% / 0.827 / 62.5% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_8:style; iou 0.66<br>n_5:style<br>n_6:style<br>+2 autres |
| `doc3_p33` | doc3 | **Oui** | 100.0% / 100.0% / 0.830 / 75.0% / RF 0 | Stable, publication-ready<br>Rendu stable et propre à l’œil; pas de collision visible ni d’overlay source. | n_10:style; iou 0.72 |
| `doc3_p34` | doc3 | **Oui** | 100.0% / 100.0% / 0.858 / 75.0% / RF 0 | Stable, publication-ready<br>Rendu stable et propre à l’œil; pas de collision visible ni d’overlay source. | n_9:style; iou 0.66 |
| `doc3_p36` | doc3 | **Oui** | 100.0% / 100.0% / 0.868 / 85.7% / RF 0 | Stable, publication-ready<br>Rendu stable et propre à l’œil; pas de collision visible ni d’overlay source. | n_8:style; iou 0.64<br>n_6:iou 0.76 |
| `doc3_p38` | doc3 | **Partiel** | 100.0% / 100.0% / 0.803 / 40.0% / RF 0 | Lisible mais fragile<br>Lisible mais pas strictement WYSIWYG: style, géométrie ou adaptation locale encore fragiles. | n_10:style; iou 0.66<br>n_7:style<br>n_8:style<br>+1 autres |

## Blocs problématiques

Les lignes ci-dessous regroupent les blocs qui cassent le plus clairement le WYSIWYG, ou qui montrent une dérive visuelle utile pour des corrections universelles.

| Page | Bloc | Rôle | Source originale (extrait) | Rendu FR (extrait) | Problème visuel | Cause probable |
|---|---|---|---|---|---|---|
| `doc1_p05` | `n_4` | title | Welcome to computer vision / 3 | 1.1. Vision informatique 4 | style; glyph; disappear; presence 0.00; iou 0.00; text_missing|text_overlap | reflow trop agressif dans une zone dense; candidate rendu rejeté ou texte non recollé; taille/police/flags mal conservés; normalisation de glyphes insuffisante; bloc perdu au rendu; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc1_p05` | `n_7` | section_heading | 1.2 / Applications of computer vision | informatique | disappear; presence 0.00; iou 0.01; text_missing|text_overlap | reflow trop agressif dans une zone dense; candidate rendu rejeté ou texte non recollé; bloc perdu au rendu; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc1_p38` | `n_7` | body | First you see a wheel feature; could this be a car, a motorcycle, or a dog? / a | de déterminer ce qui est dans l'image: ? que a Ce ce n'est soit une pas un moto chien, qu'une car voiture. les chiens... | glyph; presence 0.43; iou 0.70; text_missing|text_overlap | reflow trop agressif dans une zone dense; candidate rendu rejeté ou texte non recollé; normalisation de glyphes insuffisante; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc1_p05` | `n_5` | section_heading | 1.1 / Computer vision | 1.1. Vision informatique 4 | style; iou 0.80; text_missing|text_overlap | reflow trop agressif dans une zone dense; candidate rendu rejeté ou texte non recollé; taille/police/flags mal conservés; réassociation géométrique faible |
| `doc1_p09` | `n_13` | body | MNIST / 263 | MNIST ImageNet 265 263 264 264 ■ ■ ■ Google MS COCO Ouvrir ■ MNISTE des images Fashion ■ ■ CIFAR 266 267 C'est un kag... | style; presence 0.86; text_missing|text_overlap | reflow trop agressif dans une zone dense; candidate rendu rejeté ou texte non recollé; taille/police/flags mal conservés; lignes non replacées complètement dans la zone |
| `doc1_p31` | `n_3` | body | Increasing numbers of image classification tasks are being solved with / NOTE | Tout les cours comme que le vous diagnostic voulez. D'autres de cancer exemples et les exemples de classification de ... | style; presence 0.73; text_missing|text_overlap | reflow trop agressif dans une zone dense; candidate rendu rejeté ou texte non recollé; taille/police/flags mal conservés; lignes non replacées complètement dans la zone |
| `doc1_p38` | `n_17` | figure_caption | Figure 1.14 / Using the machine learning model to predict the probability of the motorcycle object from the | Graphique 1.14 Utilisation du modèle d'apprentissage automatique pour prédire la probabilité de l'objet moto de la cl... | style; iou 0.62; text_missing|text_overlap | reflow trop agressif dans une zone dense; candidate rendu rejeté ou texte non recollé; taille/police/flags mal conservés; réassociation géométrique faible |
| `doc1_p09` | `n_11` | body | Scenario 1: Target dataset is small and similar to the source / dataset | 260 ensemble de données ■ Scénario 2 : L'ensemble de données cible est important 261 à l'ensemble de données source ■... | presence 0.73; iou 0.75; text_missing|text_overlap | reflow trop agressif dans une zone dense; candidate rendu rejeté ou texte non recollé; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc1_p25` | `n_13` | figure_caption | Figure 1.1 / The human vision system uses the eye and brain to sense and interpret an image. | Graphique 1.1 Le système de vision humaine utilise l'œil et le cerveau pour percevoir et interpréter une image. | style; text_missing|text_overlap | reflow trop agressif dans une zone dense; candidate rendu rejeté ou texte non recollé; taille/police/flags mal conservés |
| `doc2_p16` | `2` | title | Deep / Learning | C'est Apprendre | presence 0.50; iou 0.34; text_missing|text_overlap | reflow trop agressif dans une zone dense; candidate rendu rejeté ou texte non recollé; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc1_p31` | `n_8` | figure_caption | Figure 1.5 / Vision systems are now able to learn patterns in X-ray images to identify tumors in earlier | Graphique 1.5 Les les stades systèmes de de développement. vision par ordinateur sont maintenant capables d'apprendre... | presence 0.33; text_missing|text_overlap | reflow trop agressif dans une zone dense; candidate rendu rejeté ou texte non recollé; lignes non replacées complètement dans la zone |
| `doc2_p29` | `n_10` | body |  / F ( i , j )  ( A ∗ K )( i , j )  | F(i, j) =(A * K)(i, j) =ZA(m, F (i, j) A K)(i, j) ( m | glyph; presence 0.50; iou 0.34; text_missing | candidate rendu rejeté ou texte non recollé; normalisation de glyphes insuffisante; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc2_p29` | `n_15` | body |  / F ( i , j )  ( A ∗ K )( i , j )  | F(i, j) =(A * K)(i, j) =ZZA(i F (i, j) A K)(i, j) ( m | glyph; presence 0.50; iou 0.37; text_missing | candidate rendu rejeté ou texte non recollé; normalisation de glyphes insuffisante; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_11` | body | Mean squared error | Réseau Réseau de de voyance voyance profonde profonde Erreur Erreur carrière carrière moyenne moyenne Institut nation... | style; disappear; presence 0.00; iou 0.20 | taille/police/flags mal conservés; bloc perdu au rendu; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_29` | body | Robust restricted Boltzmann machines | Racine moyenne carré composantes Machines Machines à à commander commander reposer reposer à à boulezmann boulezmann ... | style; disappear; presence 0.00; iou 0.31 | taille/police/flags mal conservés; bloc perdu au rendu; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_31` | body | Recurrent temporal restricted Boltzmann machines | Machines Machines à à commander commander reposer reposer à à boulezmann boulezmann robustes robustes composantes Mac... | style; disappear; presence 0.00; iou 0.32 | taille/police/flags mal conservés; bloc perdu au rendu; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_9` | body | Multiresolution deep belief network | Perceptron multicouche Réseau Réseau de de voyance voyance profonde profonde multirésolution multirésolution Erreur E... | style; disappear; presence 0.00; iou 0.29 | taille/police/flags mal conservés; bloc perdu au rendu; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc2_p29` | `1` | equation_inline | F(i, j) =(A * K)(i, j) =ZA(m, n)K(i -m, j -n) | F(i, j) =(A * K)(i, j) =ZA(m, n)K(i -m, j -n) F (i, j) A K)(i, j) A(m,n )K (i −m, −n) j ( n m | style; iou 0.56; text_missing | candidate rendu rejeté ou texte non recollé; taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p29` | `2` | equation_inline | F(i, j) =(A * K)(i, j) =ZZA(i -m, j -n)K(m, n) | F(i, j) =(A * K)(i, j) =ZZA(i -m, j -n)K(m, n) F (i, j) A K)(i, j) A(i −m, −n)K (m, n) j ( n m | style; iou 0.58; text_missing | candidate rendu rejeté ou texte non recollé; taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p29` | `n_17` | title |  | =ZZA(i A(i | style; disappear; presence 0.00; iou 0.06; render | taille/police/flags mal conservés; bloc perdu au rendu; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc1_p38` | `n_12` | equation_inline | P (motorcycle) = 0.85 | P (motorcycle) = 0.85 | style; text_missing | candidate rendu rejeté ou texte non recollé; taille/police/flags mal conservés |
| `doc1_p38` | `n_15` | equation_inline | P (car) = 0.14 | P (car) = 0.14 | style; text_missing | candidate rendu rejeté ou texte non recollé; taille/police/flags mal conservés |
| `doc1_p38` | `n_16` | equation_inline | P (dog) = 0.01 | P (dog) = 0.01 | style; text_missing | candidate rendu rejeté ou texte non recollé; taille/police/flags mal conservés |
| `doc2_p29` | `n_4` | equation_inline |  / ( f ∗ g )( n )  | f( (f * g) (n) (f * g) (n) m | style; glyph; presence 0.50; iou 0.35 | taille/police/flags mal conservés; normalisation de glyphes insuffisante; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc2_p29` | `n_5` | title | f ( m ) g ( n − m ) / m | f( m)g(n−m) g) (n) m | style; text_missing | candidate rendu rejeté ou texte non recollé; taille/police/flags mal conservés |
| `doc2_p29` | `n_7` | equation_inline |  / ( f ∗ g )( n )  | f( (f * g) (n) (f * g) (n) m | style; glyph; presence 0.50; iou 0.33 | taille/police/flags mal conservés; normalisation de glyphes insuffisante; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc2_p29` | `n_8` | title | f ( n − m ) g ( m ) / m | f( n−m)g(m) g) (n) m | style; text_missing | candidate rendu rejeté ou texte non recollé; taille/police/flags mal conservés |
| `doc2_p12` | `n_0_abbr_15` | body | NIST Special Database 4 | Institut national de normalisation carrée NIST NIST Base Base de de données données spécialisées spécialisées 4 4 ens... | disappear; presence 0.00; iou 0.21 | bloc perdu au rendu; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_19` | body | Principal component analysis | ensemble de données visuelles carrée Analyse Analyse des des principales principales composants composants Fonction d... | disappear; presence 0.00; iou 0.25 | bloc perdu au rendu; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_23` | body | Restricted Boltzmann machine | Fonction de base radiale spéciale 4 La La machine machine Boltzmann Boltzmann repose repose spéciale 4 Unité linéaire | disappear; presence 0.00; iou 0.28 | bloc perdu au rendu; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc2_p16` | `block_4` | title | Artificial Intelligence |  | disappear; presence 0.00; iou 0.00; render | bloc perdu au rendu; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc2_p29` | `n_12` | title |  | =ZA(m, A(m,n | disappear; presence 0.00; iou 0.05; render | bloc perdu au rendu; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc2_p29` | `n_13` | body | A ( m , n ) K ( i − m , j − n ) / (2.1) | =ZA(m, n)K(i -m, j -n) A(m,n )K (i −m, −n) (2.1) j n | iou 0.78; text_missing | candidate rendu rejeté ou texte non recollé; réassociation géométrique faible |
| `doc2_p29` | `n_18` | body | A ( i − m , j − n ) K ( m , n ) / (2.2) | =ZZA(i -m, j -n)K(m, n) A(i −m, −n)K (m, n) (2.2) j n | iou 0.76; text_missing | candidate rendu rejeté ou texte non recollé; réassociation géométrique faible |
| `doc1_p25` | `n_5` | body | H UMAN VISION SYSTEMS / At the highest level, vision systems are pretty much the same for humans, animals, | H SYSTÈMES DE VISION UMAN Au plus haut niveau, les systèmes de vision par ordinateur sont à peu près les mêmes pour l... | style; overlay | taille/police/flags mal conservés; résidu source non nettoyé |
| `doc1_p38` | `n_9` | body | The next feature is rear mudguards —again, there is a higher probability that / c | voiture. La prochaine caractéristique est les garde-boue arrière — encore une fois, il y a une probabilité que c c es... | style; presence 0.67; iou 0.56 | taille/police/flags mal conservés; lignes non replacées complètement dans la zone; réassociation géométrique faible |
| `doc1_p05` | `n_6` | body | What is visual perception? / 5 | 5 5 ■ ■ Systèmes de vision Qu'est­ce que la perception visuelle? 7 8 Bienvenue à la vision ■ ■ Appareils d'interpréta... | style; iou 0.55 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc1_p09` | `n_15` | title | Object detection with R-CNN, SSD, and YOLO / 283 | 283 Détection d'objets avec R-CNN, SSD et YOLO 7.1. 285 | style; iou 0.42 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc1_p21` | `n_0` | title | Part 1 | Première partie | style; iou 0.42 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc1_p21` | `n_1` | body | Deep learning foundation | Fondement de l'apprentissage approfondi | style; iou 0.68 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc1_p38` | `n_11` | header | 3. Feature extraction | 3. Extraction des éléments | style; iou 0.71 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc1_p38` | `n_14` | title | Features vector | Caractéristiques vectoriel | style; iou 0.68 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p01` | `n_0` | header | Studies in Big Data | Études sur les données massives | style; iou 0.63 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p01` | `n_3` | body | Janusz Kacprzyk, Polish Academy of Sciences, Warsaw, Poland | Janusz Kacprzyk, Académie polonaise des sciences, Varsovie, Pologne | style; iou 0.64 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_13` | body | National Institute of Standards and Technology | Erreur Erreur carrière carrière moyenne moyenne Institut national de normalisation et de technologie carrée | style; iou 0.30 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_17` | body | Olivetti Research Ltd face dataset | carrée NIST NIST Base Base de de données données spécialisées spécialisées 4 4 ensemble de données visuelles carrée A... | style; iou 0.27 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_3` | body | Modular deep belief networks | Normalisation des réponses locales Réseaux modulaires de croyances profondes | style; iou 0.45 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_33` | body | Stochastic gradient descent | composantes Machines Machines à à boulonzmann boulonzmann limitées limitées temporellement temporellement Descente de... | style; iou 0.19 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_5` | body | Missed detection rate | Réseaux modulaires de croyances Taux de détection manquant Perceptron multicouche | style; iou 0.30 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_7` | body | Multilayer perceptron | Taux de détection manquant Perceptron multicouche Réseau Réseau de de voyance voyance profonde profonde | style; iou 0.31 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p16` | `1` | title | Machine Learning | Apprentissage automatique | style; iou 0.58 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p17` | `2` | title | Input Image | image d'entrée | style; iou 0.72 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p17` | `block_4` | header | Output layer | couche de sortie | style; iou 0.63; render | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p17` | `n_1` | figure_caption | Fig. 1.3 A deep learning network for digit classification | Figure 1.3 Un réseau d'apprentissage approfondi pour la classification des chiffres | style; iou 0.65 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p29` | `n_11` | title | m | =ZA(m, m | style; iou 0.04 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc2_p29` | `n_16` | title | m | =ZZA(i m | style; iou 0.03 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p02` | `n_1` | body | All rights reserved. No part of this work may be reproduced or transmitted in any form or by any / means, electronic ... | Tous droits réservés. Aucune partie de cette oeuvre ne peut être reproduite ou transmise sous quelque forme que ce so... | overlay | résidu source non nettoyé |
| `doc3_p02` | `n_8` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.66 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p05` | `n_0` | title | About the Technical Reviewer | À propos de l'examinateur technique | style; iou 0.77 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p05` | `n_2` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.72 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p06` | `n_16` | title | Chapter 13: Mining Text to Find Meaningful Data | Chapitre 13: Texte de l'extraction pour trouver des données significatives | style; iou 0.79 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p06` | `n_18` | body | Chapter 15: Saving Time with Views, Functions, and Triggers | Chapitre 15: Sauver le temps avec les vues, les fonctions et les déclencheurs | style; iou 0.75 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p06` | `n_21` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.66 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p06` | `n_4` | title | Chapter 1: Creating Your First Database and Table | Chapitre 1: Création de votre première base de données et de votre première table | style; iou 0.74 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p06` | `n_9` | title | Chapter 6: Joining Tables in a Relational Database | Chapitre 6: Rejoindre des tableaux dans une base de données relationnelles | style; iou 0.76 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p07` | `n_3` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.64 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p08` | `n_0` | title | CONTENTS IN DETAIL | SOMMAIRE EN DÉTAILLANT | style; iou 0.67 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p08` | `n_11` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.66 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p19` | `n_3` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.64 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p22` | `n_5` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.72 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p24` | `n_6` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.66 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p26` | `n_4` | title | Why Use SQL? | Pourquoi utiliserSQL? | style; iou 0.74 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p26` | `n_7` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.66 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p31` | `n_6` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.66 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p32` | `n_8` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.66 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p33` | `n_10` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.72 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p34` | `n_9` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.66 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p36` | `n_8` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.64 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc3_p38` | `n_10` | equation_inline | Estadísticos e-Books & Papers | Estadísticos e-Books & Papers | style; iou 0.66 | taille/police/flags mal conservés; réassociation géométrique faible |
| `doc1_p05` | `n_3` | body | P ART 1 / D EEP LEARNING FOUNDATION .............................1 | D EEP LEARNING FOUNDATION.............................1 P ART 1 1 1.1. Vision informatique 4 5 5 ■ ■ Systèmes de visi... | style | taille/police/flags mal conservés |
| `doc1_p05` | `n_8` | body | Image classification / 10 | 10 12 classification d'image ■ ■ Détection et localisation des objets 12 13 Génération de l'art (transfert de style) ... | style | taille/police/flags mal conservés |
| `doc1_p09` | `n_17` | body | Region proposals / 286 | 286 287 ■ Prévisions du réseau Propositions régionales Suppression non maximale (NMS) 288 ■ Détecteur d'objets 289 le... | style | taille/police/flags mal conservés |
| `doc1_p09` | `n_5` | title | Transfer learning / 240 | 240 Transfert de l apprentissage | style | taille/police/flags mal conservés |
| `doc1_p09` | `n_7` | body | How do neural networks learn features? / 252 | 252 ■ Transférabilité des Comment les réseaux neuronaux apprennent-ils les fonctionnalités? 254 caractéristiques extr... | style | taille/police/flags mal conservés |
| `doc1_p09` | `n_9` | body | Using a pretrained network as a classifier / 254 | 254 Utilisation d'un réseau pré-formé comme classificateur ■ Utilisation d'un pré-entraînement 256 258 ■ Fin de régla... | style | taille/police/flags mal conservés |
| `doc1_p13` | `n_2` | body | Two years ago, I decided to write a book to teach deep learning for computer vision / from an intuitive perspective. ... | Il y a deux ans, j'ai décidé d'écrire un livre pour enseigner l'apprentissage profond pour la vision informatique dan... | style | taille/police/flags mal conservés |
| `doc1_p14` | `n_1` | body | As a beginner, I searched but couldn’t find anything to meet these needs. So now / I’ve written it. My goal has been ... | Comme débutant, j'ai cherché mais je n'ai pas trouvé quoi que ce soit pour répondre à ces besoins. Alors maintenant, ... | style | taille/police/flags mal conservés |
| `doc1_p14` | `n_4` | body | At the time of writing, I believe this is the only deep learning for vision systems / resource that is taught this wa... | Au moment de l'écriture, je crois que c'est le seul apprentissage profond pour les ressources des systèmes de vision ... | style | taille/police/flags mal conservés |
| `doc1_p21` | `n_2` | body | C omputer vision is a technological area that’s been advancing rapidly / thanks to the tremendous advances in artific... | La vision de l'ompeur est un domaine technologique qui progresse rapidement grâce aux énormes progrès de l'intelligen... | style | taille/police/flags mal conservés |
| `doc1_p23` | `n_1` | title | Welcome to / computer vision | Bienvenue à vision de l'ordinateur | style | taille/police/flags mal conservés |
| `doc1_p23` | `n_2` | body | Hello! I’m very excited that you are here. You are making a great decision—to / grasp deep learning (DL) and computer... | Bonjour! Je suis très excitée que vous soyez ici. Vous prenez une grande décision... • la compréhension de l'apprenti... | style | taille/police/flags mal conservés |
| `doc1_p23` | `n_4` | body |  Components of the vision system /  Applications of computer vision | Éléments du système de vision Applications de la vision informatique Comprendre le pipeline de vision informatique Pr... | style | taille/police/flags mal conservés |
| `doc1_p25` | `n_1` | header | 1.1.1 / What is visual perception? | 1.1.1. Qu'est-ce que la perception visuelle? | style | taille/police/flags mal conservés |
| `doc1_p25` | `n_3` | section_heading | 1.1.2 / Vision systems | 1.1.2. Systèmes de vision | style | taille/police/flags mal conservés |
| `doc1_p25` | `n_7` | title | Human vision system | Système de vision humaine | style | taille/police/flags mal conservés |
| `doc1_p25` | `n_9` | title | Brain (interpreting device / responsible for understanding | Cerveau (appareil d'interprétation) responsable de la compréhension le contenu de l image) | style | taille/police/flags mal conservés |
| `doc1_p28` | `n_0` | header | 8 / C HAPTER 1 | 8 C HAPTER 1 Bienvenue à la vision informatique | style | taille/police/flags mal conservés |
| `doc1_p28` | `n_1` | section_heading | 1.1.4 / Interpreting devices | 1.1.4 Appareils d'interprétation | style | taille/police/flags mal conservés |
| `doc1_p28` | `n_3` | body | (continued) / eyes that consist of multiple lenses (as many as 30,000 lenses in a single compound | (suite) yeux qui se composent de plusieurs lentilles (jusqu'à 30 000 lentilles dans un seul composé oeil). Les yeux c... | style | taille/police/flags mal conservés |
| `doc1_p28` | `n_4` | title | Compound eyes are low resolution but sensitive to motion. | Les yeux composés sont de basse résolution mais sensibles aux mouvements. | style | taille/police/flags mal conservés |
| `doc1_p28` | `n_5` | title | Compound eyes / How bees see a flower | Yeux composés Comment les abeilles voient une fleur | style | taille/police/flags mal conservés |
| `doc1_p30` | `n_0` | header | 10 / C HAPTER 1 | 10 C HAPTER 1 Bienvenue à la vision informatique | style | taille/police/flags mal conservés |
| `doc1_p30` | `n_4` | body | Recent AI and DL advances have allowed machines to surpass human visual ability in / many image classification and ob... | Les récentes avancées de l'IA et de DL ont permis aux machines de dépasser les capacités visuelles humaines dans de n... | style | taille/police/flags mal conservés |
| `doc1_p30` | `n_5` | section_heading | 1.2 / Applications of computer vision | 1.2. Applications de la vision informatique | style | taille/police/flags mal conservés |
| `doc1_p30` | `n_6` | body | Computers began to be able to recognize human faces in images decades ago, but now / AI systems are rivaling the abil... | Les ordinateurs ont commencé à reconnaître les visages humains dans les images il y a des décennies, mais maintenant ... | style | taille/police/flags mal conservés |
| `doc1_p30` | `n_8` | body | Image classification is the task of assigning to an image a label from a predefined set of / categories. A convolutio... | La classification des images est la tâche d'attribuer à une image une étiquette à partir d'un ensemble prédéfini de c... | style | taille/police/flags mal conservés |
| `doc1_p30` | `n_9` | body |  Lung cancer diagnosis —Lung cancer is a growing problem. The main reason lung / cancer is very dangerous is that wh... | -Le diagnostic du cancer du poumon — Le cancer du poumon est un problème croissant. La principale raison du cancer du... | style | taille/police/flags mal conservés |
| `doc1_p31` | `n_1` | body | late stages. When diagnosing lung cancer, doctors typically use their eyes to / examine CT scan images, looking for s... | les étapes tardives. Lors du diagnostic du cancer du poumon, les médecins utilisent généralement leurs yeux pour exam... | style | taille/police/flags mal conservés |
| `doc1_p31` | `n_2` | body |  Traffic sign recognition —Traditionally, standard CV methods were employed to / detect and classify traffic signs, ... | Reconnaissance de la signalisation routière — Traditionnellement, les méthodes standard CV ont été utilisées pour de ... | style | taille/police/flags mal conservés |
| `doc1_p38` | `n_0` | header | 18 / C HAPTER 1 | 18 C HAPTER 1 Bienvenue à la vision informatique | style | taille/police/flags mal conservés |
| `doc1_p38` | `n_1` | body | An image classifier is an algorithm that takes in an image as input / DEFINITIONS | DÉFINITIONS Un classificateur d'image est un algorithme qui prend une image comme entrée et sort une étiquette ou -cl... | style | taille/police/flags mal conservés |
| `doc1_p38` | `n_3` | body | A computer receives visual input from an imaging device like a camera. This / 1 | 1 Un ordinateur reçoit une entrée visuelle d'un appareil d'imagerie comme une caméra. Cette entrée est généralement c... | style | taille/police/flags mal conservés |
| `doc1_p38` | `n_5` | body | We extract features. Features are what help us define objects, and they are usu- / 3 | comparer et les analyser plus avant. Nous extrayons des éléments. Les fonctionnalités sont ce qui nous aide à définir... | style | taille/police/flags mal conservés |
| `doc2_p01` | `n_2` | title | Series editor | Éditeur de série | style | taille/police/flags mal conservés |
| `doc2_p02` | `n_0` | body | The series “ Studies in Big Data ” (SBD) publishes new developments and advances / in the various areas of Big Data —... | La série -Studies in Big Data (SBD) publie les nouveaux développements et les avancées dans les différents domaines d... | style | taille/police/flags mal conservés |
| `doc2_p10` | `n_0` | header | About the Authors | À propos des auteurs | style | taille/police/flags mal conservés |
| `doc2_p10` | `n_1` | body | Prof. M. Arif Wani completed his M.Tech. in Computer Technology at the Indian / Institute of Technology, Delhi and hi... | Professeur M. Arif Wani a complété son M.Tech. en informatique à l'Institut indien de technologie, Delhi et son docto... | style | taille/police/flags mal conservés |
| `doc2_p12` | `n_0_abbr_1` | body | Local response normalization | Normalisation des réponses locales | style | taille/police/flags mal conservés |
| `doc2_p14` | `n_1` | body | learning. Reinforcement learning has been successful in applications as diverse as / autonomous helicopter flight, ro... | l'apprentissage. L'apprentissage du renforcement a été couronné de succès dans des applications aussi diverses que le... | style | taille/police/flags mal conservés |
| `doc2_p14` | `n_2` | section_heading | 1.2 / Shallow Learning | 1.2. Apprentissage peu profond | style | taille/police/flags mal conservés |
| `doc2_p14` | `n_3` | body | Shallow architectures are well understood and perform good on many common / machine learning problems, and they are s... | Les architectures peu profondes sont bien comprises et fonctionnent bien sur de nombreux problèmes d'apprentissage au... | style | taille/police/flags mal conservés |
| `doc2_p14` | `n_4` | section_heading | 1.3 / Deep Learning | 1.3. Enseignement approfondi | style | taille/police/flags mal conservés |
| `doc2_p14` | `n_5` | body | Deep learning is a new area of machine learning which has gained popularity in recent / past. Deep learning refers to... | L'apprentissage profond est un nouveau domaine de l'apprentissage automatique qui a gagné en popularité dans le passé... | style | taille/police/flags mal conservés |
| `doc2_p16` | `n_1` | body | ple mappings. The word “deep” refers to learning successive layers of increasingly / meaningful representations of in... | Les cartes sont nombreuses. Le mot -deep-- est une référence à l'apprentissage de couches successives de représentati... | style | taille/police/flags mal conservés |
| `doc2_p16` | `n_2` | figure_caption | Fig. 1.2 Relationship between AI, machine learning, and deep learning | Figure 1.2 Relation entre l IA, l apprentissage automatique et l apprentissage profond | style | taille/police/flags mal conservés |
| `doc2_p17` | `n_3` | body | The term artificial neural networks has a reference to neuroscience but deep learn- / ing networks are not models of ... | Le terme réseaux neuronaux artificiels a une référence aux neurosciences mais l'apprentissage profond- Les réseaux ne... | style | taille/police/flags mal conservés |
| `doc2_p19` | `n_1` | body | Machine learning practitioners have spent a huge time to extract informative / features from the data. At the time of... | Les praticiens de l'apprentissage automatique ont passé un temps d'enseignement à extraire des informations caractéri... | style | taille/police/flags mal conservés |
| `doc2_p20` | `n_1` | body | The exceptional performance of deep models can be mainly attributed to their flex- / ibility in representing a rich s... | La performance exceptionnelle des modèles profonds peut être principalement attribuée à leur flexibilité en représent... | style | taille/police/flags mal conservés |
| `doc2_p20` | `n_2` | section_heading | 1.5 / How Deep Learning Works | 1,5 Comment fonctionne l'apprentissage profond | style | taille/police/flags mal conservés |
| `doc2_p21` | `n_1` | body | the difference between the predicted output obtained from the network and the true / target value for a specific exam... | la différence entre la sortie prévue obtenue du réseau et la valeur cible réelle pour un exemple spécifique. Cela don... | style | taille/police/flags mal conservés |
| `doc2_p21` | `n_3` | body | Deep learning networks have brought their own set of problems and challenges which / outweighed the benefits of deep ... | Les réseaux d'apprentissage approfondi ont apporté leur propre ensemble de problèmes et de défis qui ont l'avantage d... | style | taille/police/flags mal conservés |
| `doc2_p28` | `n_9` | body | tions. Xception has 36 convolutional layers organized into 14 modules, all having / linear residual connections aroun... | Il est important de bien comprendre les conséquences de la crise. Xception a 36 couches convolutionnelles organisées ... | style | taille/police/flags mal conservés |
| `doc2_p29` | `n_14` | body | The convolution operation is commutative in nature, so we can write Eq. 2.1 as | L'opération de convolution est de nature commutative, donc nous pouvons écrire Eq. 2.1 comme | style | taille/police/flags mal conservés |
| `doc2_p29` | `n_9` | body | Convolution operation is one of the important operations used in digital signal / processing and is used in many area... | L'opération de convolution est une des opérations importantes utilisées dans le traitement du signal numérique et est... | style | taille/police/flags mal conservés |
| `doc2_p31` | `n_1` | body | (i) Local Receptive Field / In a traditional neural network, each neuron or hidden unit is connected to every | i) Champ de recuit local Dans un réseau neuronal traditionnel, chaque neurone ou unité cachée est connecté à chaque n... | style | taille/police/flags mal conservés |
| `doc2_p31` | `n_10` | body | Convolution layer is the core building block of a convolutional neural network which / uses convolution operation (re... | La couche de convolution est le noyau de construction d'un réseau neuronal convolutionnel qui utilise l'opération de ... | style | taille/police/flags mal conservés |
| `doc2_p31` | `n_5` | body | • Convolutional layer, / • Activation function layer (ReLU), | • couches convolutionnelles, • • Couche de fonction d'activation (ReLU), • Couche de mise en commun, • couches entièr... | style | taille/police/flags mal conservés |
| `doc2_p31` | `n_6` | body | These layers are stacked up to make a full ConvNet architecture. Convolutional / and activation function layers are u... | Ces couches sont empilées pour faire une architecture ConvNet complète. Les couches de fonctions convolutionnelles et... | style | taille/police/flags mal conservés |
| `doc2_p32` | `n_1` | figure_caption | Fig. 2.5 Example of convolution operation | Figure 2.5 Exemple de fonctionnement de convolution | style | taille/police/flags mal conservés |
| `doc2_p32` | `n_2` | body | the filter with the input image, adding a bias term, and then applying an activation / function. The input area on wh... | le filtre avec l'image d'entrée, ajoutant un terme de biais, puis appliquant une fonction d'activation. La zone d'ent... | style | taille/police/flags mal conservés |
| `doc2_p32` | `n_3` | body | Filters/Kernels / The weights in each convolutional layer specify the convolution filters and there may | Filtres/noyau Les poids dans chaque couche convolutionnelle spécifient les filtres convolutionnels et il peut y avoir... | style | taille/police/flags mal conservés |
| `doc2_p39` | `n_2` | body | ConvNets have evolved over the years and have achieved very good performance / on various visual tasks like classific... | ConvNets a évolué au fil des ans et a obtenu de très bonnes performances sur diverses tâches visuelles comme la class... | style | taille/police/flags mal conservés |
| `doc3_p02` | `n_5` | title | Library of Congress Cataloging-in-Publication Data | Bibliothèque du Congrès Données de catalogage en publication | style | taille/police/flags mal conservés |
| `doc3_p05` | `n_1` | body | Josh Berkus is a “hacker emeritus” for the PostgreSQL Project, where he / served on the Core Team for 13 years. He wa... | Josh Berkus est un émérite de hacker pour le projet PostgreSQL, où il a a été membre de l'équipe centrale pendant 13 ... | style | taille/police/flags mal conservés |
| `doc3_p06` | `n_1` | title | Foreword by Sarah Frostenson | Avant-propos de Sarah Frostenson | style | taille/police/flags mal conservés |
| `doc3_p06` | `n_10` | title | Chapter 7: Table Design That Works for You | Chapitre 7: Conception de table qui fonctionne pour vous | style | taille/police/flags mal conservés |
| `doc3_p06` | `n_12` | title | Chapter 9: Inspecting and Modifying Data | Chapitre 9: Inspection et modification des données | style | taille/police/flags mal conservés |
| `doc3_p06` | `n_13` | title | Chapter 10: Statistical Functions in SQL | Chapitre 10: Fonctions statistiques dans SQL | style | taille/police/flags mal conservés |
| `doc3_p06` | `n_14` | title | Chapter 11: Working with Dates and Times | Chapitre 11: Travailler avec les dates et les heures | style | taille/police/flags mal conservés |
| `doc3_p06` | `n_19` | body | Chapter 16: Using PostgreSQL from the Command Line | Chapitre 16: Utilisation de PostgreSQL à partir de la ligne de commande | style | taille/police/flags mal conservés |
| `doc3_p06` | `n_20` | title | Chapter 17: Maintaining Your Database | Chapitre 17: La tenue à jour de votre base de données | style | taille/police/flags mal conservés |
| `doc3_p06` | `n_6` | title | Chapter 3: Understanding Data Types | Chapitre 3: Comprendre les types de données | style | taille/police/flags mal conservés |
| `doc3_p06` | `n_7` | title | Chapter 4: Importing and Exporting Data | Chapitre 4: Importation et exportation de données | style | taille/police/flags mal conservés |
| `doc3_p06` | `n_8` | title | Chapter 5: Basic Math and Stats with SQL | Chapitre 5: Maths et statistiques de base avec SQL | style | taille/police/flags mal conservés |
| `doc3_p07` | `n_1` | title | Appendix: Additional PostgreSQL Resources | Annexe: Ressources supplémentaires de PostgreSQL | style | taille/police/flags mal conservés |
| `doc3_p08` | `n_1` | title | FOREWORD by Sarah Frostenson | AVANT-PROPOS de Sarah Frostenson | style | taille/police/flags mal conservés |
| `doc3_p22` | `n_0` | body | this book). By adding the PostGIS extension to the database, you can / create spatial data that you can then export a... | ce livre). En ajoutant l'extension PostGIS à la base de données, vous pouvez créer des données spatiales que vous pou... | style | taille/police/flags mal conservés |
| `doc3_p26` | `n_0` | title | What Is SQL? | Ce qui estSQL? | style | taille/police/flags mal conservés |
| `doc3_p31` | `n_0` | body | Also in the .sql files, you’ll see lines that begin with two hyphens ( -- ) / and a space. These are comments that pr... | Aussi dans les fichiers.sql, vous verrez des lignes qui commencent par deux tirets (--) et un espace. Ce sont des com... | style | taille/police/flags mal conservés |
| `doc3_p31` | `n_2` | body | After downloading data, Windows users might need to provide permission / for the database to read files. To do so, ri... | Après le téléchargement des données, les utilisateurs de Windows pourraient avoir nécessité de fournir une autorisati... | style | taille/police/flags mal conservés |
| `doc3_p31` | `n_4` | body | In this book, I’ll teach you SQL using the open source PostgreSQL / database system. PostgreSQL, or simply Postgres, ... | Dans ce livre, je vais vous enseigner SQL en utilisant le logiciel open source PostgreSQLZ système de base de données... | style | taille/police/flags mal conservés |
| `doc3_p32` | `n_5` | body | Always install the latest available version of PostgreSQL for your operating / system to ensure that it’s up to date ... | Installez toujours la dernière version disponible de PostgreSQL pour votre fonctionnement système pour s'assurer qu'i... | style | taille/police/flags mal conservés |
| `doc3_p32` | `n_6` | title | Windows Installation | Installation de Windows | style | taille/police/flags mal conservés |
| `doc3_p38` | `n_7` | body | If pgAdmin doesn’t show a default under Servers, you’ll need to add it. / Right-click Servers, and choose the Create ... | Si pgAdmin ne montre pas un défaut sous Serveurs, vous devrez l'ajouter. Cliquez-droit sur Serveurs, et choisissez l'... | style | taille/police/flags mal conservés |
| `doc3_p38` | `n_8` | body | This collection of objects defines every feature of your database server. / There’s a lot here, but for now we’ll foc... | Cette collection d'objets définit chaque fonction de votre serveur de base de données. Il y a beaucoup ici, mais pour... | style | taille/police/flags mal conservés |
| `doc1_p05` | `n_0` | footer | v | v | iou 0.35 | réassociation géométrique faible |
| `doc1_p05` | `n_1` | title | contents | Sommaire | iou 0.70 | réassociation géométrique faible |
| `doc1_p05` | `n_10` | body | Image as functions / 19 | 19 19 21 ■ ■ Comment les ordinateurs voient les images Image en tant que fonctions 21 Images couleur | iou 0.56 | réassociation géométrique faible |
| `doc1_p13` | `n_1` | title | preface | Préface | iou 0.77 | réassociation géométrique faible |
| `doc1_p23` | `n_0` | footer | 3 | 3 | iou 0.44 | réassociation géométrique faible |
| `doc1_p25` | `n_0` | header | Computer vision / 5 | Vision informatique 5 | iou 0.59 | réassociation géométrique faible |
| `doc1_p25` | `n_10` | title | Dogs / grass | Chiens Gazon | iou 0.76 | réassociation géométrique faible |
| `doc1_p25` | `n_6` | title | POOL | POUVOIR | iou 0.56 | réassociation géométrique faible |
| `doc1_p31` | `n_5` | title | Tumor | Tumeur | iou 0.75 | réassociation géométrique faible |
| `doc1_p31` | `n_7` | title | Tumor | Tumeur | iou 0.79 | réassociation géométrique faible |
| `doc2_p10` | `n_5` | footer | xi | xi | iou 0.52 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_0` | body | LRN | LRN M-DBNs | iou 0.23 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_10` | body | MSE | MrDBN MSE NIST | iou 0.18 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_12` | body | NIST | MSE NIST NIST­DB4 | iou 0.14 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_14` | body | NIST-DB4 | NIST NIST­DB4 ORL | iou 0.27 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_16` | body | ORL | NIST­DB4 ORL PCA | iou 0.12 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_18` | body | PCA | ORL PCA RBF | iou 0.26 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_2` | body | M-DBNs | LRN M-DBNs MDR | iou 0.29 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_20` | body | RBF | PCA RBF RBM | iou 0.24 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_21` | body | Radial basis function | Analyse Analyse des des principales principales composants composants Fonction de base radiale spéciale La La machine... | iou 0.22 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_22` | body | RBM | RBF RBM ReLU | iou 0.26 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_24` | body | Recti fi ed Linear Unit | La La machine machine Boltzmann Boltzmann repose repose spéciale Unité linéaire Research Ltd Racine moyenne carré | iou 0.23 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_25` | body | ReLU | RBM ReLU RMS | iou 0.28 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_26` | body | RMS | ReLU RMS RoBMs | iou 0.19 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_27` | body | Root mean square | Unité linéaire Research Ltd Racine moyenne carré composantes | iou 0.19 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_28` | body | RoBMs | RMS RoBMs RTRBMs | iou 0.23 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_30` | body | RTRBMs | RoBMs RTRBMs SGD | iou 0.28 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_32` | body | SGD | RTRBMs SGD SVM | iou 0.15 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_34` | body | SVM | SGD SVM TRBM | iou 0.22 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_35` | body | Support vector machine | Descente de gradient stochastique restreinte Machine vectorielle de soutien Machine Boltzmann à température restreinte | iou 0.19 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_36` | body | TRBM | SVM TRBM | iou 0.41 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_37` | body | Temperature-based restricted Boltzmann machine | restreinte Machine vectorielle de soutien Machine Boltzmann à température limitée restreinte | iou 0.42 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_4` | body | MDR | M-DBNs MDR MLP | iou 0.17 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_6` | body | MLP | MDR MLP MrDBN | iou 0.17 | réassociation géométrique faible |
| `doc2_p12` | `n_0_abbr_8` | body | MrDBN | MLP MrDBN MSE | iou 0.28 | réassociation géométrique faible |
| `doc2_p17` | `1` | title | Layer1 Layer 2 layer 3 | Calque1 Couche 2 couche 3 | iou 0.66 | réassociation géométrique faible |
| `doc2_p28` | `1` | title | 30 28 | 30 28 | iou 0.54 | réassociation géométrique faible |
| `doc2_p28` | `10` | section_heading | 3.6 | 3.6 | iou 0.47 | réassociation géométrique faible |
| `doc2_p28` | `11` | title | 2010 2011 2012 2013 2014 Human 2015 | 2010 2011 2012 2013 2014 Humain 2015 | iou 0.45 | réassociation géométrique faible |
| `doc2_p28` | `2` | title | 25 | 25 | iou 0.31 | réassociation géométrique faible |
| `doc2_p28` | `3` | title | 25 | 25 | iou 0.31 | réassociation géométrique faible |
| `doc2_p28` | `4` | title | 20 | 20 | iou 0.31 | réassociation géométrique faible |
| `doc2_p28` | `5` | title | 15 | 15 | iou 0.31 | réassociation géométrique faible |
| `doc2_p28` | `6` | title | 15 | 15 | iou 0.29 | réassociation géométrique faible |
| `doc2_p28` | `7` | section_heading | 11.2 | 11.2 | iou 0.33 | réassociation géométrique faible |
| `doc2_p28` | `8` | title | 10 | 10 | iou 0.31 | réassociation géométrique faible |
| `doc2_p28` | `9` | section_heading | 6.7 | 6.7 | iou 0.29 | réassociation géométrique faible |
| `doc2_p28` | `block_3` | equation_inline | ILSRVTop-5Erroron ImageNet | ILSRVTop-5Erroron ImageNet | iou 0.00; render | réassociation géométrique faible |
| `doc2_p32` | `block_1` | title | Kernel | noyau | iou 0.28; render | réassociation géométrique faible |
| `doc3_p06` | `n_0` | title | BRIEF CONTENTS | SOMMAIRE | iou 0.58 | réassociation géométrique faible |
| `doc3_p06` | `n_2` | title | Acknowledgments | Remerciements | iou 0.70 | réassociation géométrique faible |
| `doc3_p07` | `n_2` | title | Index | Sommaire | iou 0.43 | réassociation géométrique faible |
| `doc3_p08` | `n_2` | title | ACKNOWLEDGMENTS | REMERCIEMENTS | iou 0.66 | réassociation géométrique faible |
| `doc3_p08` | `n_3` | title | INTRODUCTION | INTRODUCTION | iou 0.78 | réassociation géométrique faible |
| `doc3_p19` | `n_2` | title | INDEX | INDICE | iou 0.72 | réassociation géométrique faible |
| `doc3_p24` | `n_0` | title | ACKNOWLEDGMENTS | REMERCIEMENTS | iou 0.72 | réassociation géométrique faible |
| `doc3_p31` | `n_1` | title | NOTE | REMARQUE | iou 0.39 | réassociation géométrique faible |
| `doc3_p31` | `n_3` | title | Using PostgreSQL | UtilisationPostgreSQL | iou 0.69 | réassociation géométrique faible |
| `doc3_p32` | `n_1` | title | Installing PostgreSQL | InstallationPostgreSQL | iou 0.74 | réassociation géométrique faible |
| `doc3_p32` | `n_4` | title | NOTE | REMARQUE | iou 0.39 | réassociation géométrique faible |
| `doc3_p36` | `n_6` | title | Working with pgAdmin | Travail avecpgAdmin | iou 0.76 | réassociation géométrique faible |
| `doc3_p38` | `n_6` | title | NOTE | REMARQUE | iou 0.39 | réassociation géométrique faible |

## Lecture technique

- Le pipeline tient bien le contenu textuel: le taux de texte présent et le maintien en origine restent élevés sur le lot.
- Le point faible structurel reste la conservation fine des styles, surtout sur les pages denses, les glossaires, les TOC et les figures annotées.
- Les cas `Non` sont dominés par deux familles: expansion/reflow trop agressifs dans des zones compactes, et dérives typographiques sur les labels courts, acronymes et noms propres.
- Les pages `Partiel` sont généralement lisibles, mais pas strictement WYSIWYG: la géométrie, le style ou l'adaptation locale restent insuffisants.
- Les pages `Oui` sont visuellement propres; certaines ont une IoU faible malgré l'absence de collision visible, ce qui suggère un layout sparse ou une empreinte géométrique peu dense plutôt qu'une vraie dégradation de rendu.

## Fichier source

- `results/reconstruction_validation_random40_newpages_20260602_seed20260606`
- `compare.jpg` et `page.png` ont servi de référence visuelle, les JSON de validation pour les métriques et les verdicts bloc par bloc.
