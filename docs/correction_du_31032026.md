• Liste des défauts observés sur les rendus v6:

  - Texte traduit encore partiellement mélangé avec de l’anglais ou du texte
    non traduit.
  - Surimpression réelle de lignes/spans de texte dans le PDF final.
  - Mots ou fragments encore hors page.
  - Unités de texte attendues non rendues.
  - Couverture de rendu insuffisante sur une partie du contenu.
  - Collisions texte/image encore nombreuses.
  - Overlaps entre mots/lignes de texte.
  - Taille de police souvent non respectée par rapport à l’original.
  - Quelques fallbacks de police subsistent.
  - Une incohérence de couleur subsiste encore, même si elle est rare.
  - Alignement parfois modifié par rapport à l’original.
  - Fallbacks d’alignement déclenchés parce que la ligne est plus large que son
    slot.
  - Espacement vertical et horizontal encore instable.
  - Hiérarchie visuelle parfois mal restituée.
  - Fidélité des tableaux insuffisante.
  - Certains blocs restent trop comprimés.
  - Certains blocs restent trop “inchangés” géométriquement alors que la
    traduction nécessitait une recomposition.
  - Certaines contraintes de préservation ne sont pas respectées correctement.
  - Quelques reflows restent inchangés ou inadéquats.
  - Similarité visuelle globale encore trop basse pour validation.
  - Le rendu n’est pas encore publication-ready sur les trois lots.
  
  
   Point structurant avant le tableau: page_case_v2 et layout_descriptor_v3
  existent, mais le rendu actif dépend encore surtout du page_case legacy, du
  descriptor legacy et des heuristiques du renderer. Donc, sur plusieurs
  défauts, classifier et descriptor ne sont pas “faux” seulement par ce qu’ils
  produisent, mais aussi par le fait que leur production n’est pas assez
  exécutable ou pas assez consommée.

  | Défaut | Exemple(page, document, texte) | Extractor | classifier |
  descriptor | reconstructor |
  |---|---|---|---|---|---|
  | Fuite de langue source / texte partiellement non traduit | p.2,
  test_docintelligence, Adds 2 couches cachées with 512 nodes | A: fournir des
  unités propres prose vs terme technique.<br>R: l’extraction texte est
  globalement correcte; elle n’est pas la cause primaire. | A: signaler
  fortement “translation sensitive / technical heading / mixed technical
  prose”.<br>R: page_case_classifier.py reste surtout global (layout_type,
  page_family, fallback_policy), et v2 n’est pas moteur du rendu. | A: propager
  des unités exécutables “terme à préserver” vs “prose à traduire”.<br>R: le
  descriptor ne construit pas de graphe sémantique exploitable pour empêcher
  les sorties mixtes. | A: refuser un rendu mixte ou au moins le signaler comme
  invalide.<br>R: le renderer imprime simplement translated_text tel qu’il
  arrive; il n’a pas de garde “mixed-language output”. |
  | Surimpression texte/texte | p.35, Advances, sigmoïde superposé à est que
  les entrées négatives... | A: remonter une granularité ligne/phrase
  suffisante pour composer localement sans collision.<br>R: sur les pages
  denses, les blocs restent parfois trop gros; la géométrie utile est
  insuffisante. | A: identifier les pages narratives denses à haut risque de
  collision locale.<br>R: le classifier legacy reste trop coarse; il ne pilote
  pas un mode “dense paragraph composer” spécifique. | A: fournir des
  contraintes exécutables de stacking/baseline/no-overlap.<br>R: same_band,
  same_row, continues_paragraph restent des résumés, pas des contraintes fortes
  de placement. | A: garantir que deux bboxes rendues de texte ne
  s’intersectent jamais.<br>R: le composeur utilise surtout une hauteur
  nominale de ligne et peut produire des intersections réelles entre spans
  traduits. |
  | Fragments hors page | p.30, Practical SQL, fragment de fin de ligne type
  tabl | A: fournir bbox/slot/colonne corrects.<br>R: sur le natif PDF, c’est
  généralement correct; le défaut n’est pas surtout ici. | A: signaler les cas
  “column overflow risk” ou “locked cell overflow risk”.<br>R: le classifier
  legacy ne porte pas ce risque comme contrainte active; fallback_policy reste
  trop grossier. | A: fournir des bornes dures de colonne/cellule consommables
  par le renderer.<br>R: les primary_flow_regions et groupes relationnels
  restent trop synthétiques. | A: couper/recomposer avant le bord droit, jamais
  après.<br>R: le renderer réduit tardivement, autorise encore des lignes trop
  larges, puis laisse des fragments sortir du cadre. |
  | Unités attendues non rendues / texte tronqué | p.476, Practical SQL, entrée
  courte de type index using GROUP BY clause, 120–123 | A: extraire de petites
  unités stables plutôt que de gros blocs fragiles.<br>R: certaines pages
  restent agrégées en gros blocs, donc une erreur locale fait perdre beaucoup
  de texte. | A: choisir un mode de composition qui favorise la complétude des
  unités courtes.<br>R: safe_mixed et les modes stricts peuvent encore
  sacrifier des queues de texte. | A: transformer les unités importantes en
  render units explicites.<br>R: trop d’unités restent implicites dans des
  blocs, surtout hors TOC/abbreviations. | A: toujours faire sortir tout le
  texte attendu, quitte à reflow localement.<br>R: une partie du texte part en
  overflow non rendu quand les frames/slots sont saturés. |
  | Collisions texte/image | p.7, test_docintelligence, paragraphe voisin d’un
  visuel/annotation | A: détecter précisément images, annotations, non-text
  zones et leurs marges de sécurité.<br>R: la détection est utile mais souvent
  trop grossière pour le placement fin. | A: distinguer nettement pages
  annotated/figure-heavy des pages textuelles simples.<br>R: le classifieur
  sait le faire grossièrement, mais ce signal ne protège pas réellement le
  rendu. | A: fournir des keep_out_zones réellement exécutables.<br>R: les
  zones existent mais restent des guides larges, pas des contraintes de
  placement dures. | A: ne jamais écrire du texte dans une zone visuelle
  interdite.<br>R: le renderer flow/anchor compose encore dans des zones qui
  mordent sur figures, labels ou overlays. |
  | Dérive de taille de police | p.30, Practical SQL, corps attendu 15.0 pt,
  rendu plus petit | A: remonter la taille native exacte.<br>R:
  native_pdf_extractor.py remonte bien style["size"] dans la majorité des cas.
  | A: marquer davantage de cas en “preserve extracted typography”.<br>R: le
  classifier ne pilote pas directement cette préservation; il ne fait que
  fournir un contexte coarse. | A: produire une contrainte “font-size hard” par
  unité quand la fidélité typographique est prioritaire.<br>R: le descriptor
  encode des classes typo, pas une contrainte exécutable ferme. | A: garder la
  taille extraite, sauf impossibilité absolue.<br>R: les boucles de shrink dans
  le renderer réduisent encore souvent la taille quand line_wider_than_slot. |
  | Perte de hiérarchie typographique intra-ligne ou intra-bloc | p.30,
  Practical SQL, mélange attendu Bold/Roman/Mono, rendu aplati | A: remonter
  les spans avec police/gras/italique/mono exacts.<br>R: l’extraction native
  sait le faire, mais l’agrégation bloc/ligne n’en garde pas toujours toute la
  force opérationnelle. | A: distinguer les blocs où la variation de style est
  structurante.<br>R: le classifier ne sait pas produire un signal de priorité
  sur cette variation fine. | A: représenter des unités de rendu au niveau
  span, pas seulement bloc/ligne.<br>R: le descriptor reste majoritairement
  bloc-centré. | A: préserver bold/italic/mono segment par segment.<br>R: le
  renderer homogénéise encore certains blocs/lignes, malgré les progrès sur
  quelques cas mixtes. |
  | Fallback de police | p.3, test_docintelligence, ligne de corps rendue avec
  police de repli | A: fournir embedded_font_path ou un nom de police
  résoluble.<br>R: sur le natif c’est globalement bon, mais pas parfait sur
  tous les cas; en OCR c’est plus faible. | A: sans objet direct; au plus un
  signal de risque typographique.<br>R: le classifier n’aide pas ici. | A: sans
  objet direct; au plus une contrainte “preserve family”.<br>R: le descriptor
  ne porte pas cette contrainte de façon exécutable. | A: toujours résoudre la
  famille exacte ou une substitution typographiquement proche contrôlée.<br>R:
  quelques chemins tombent encore en fallback de police. |
  | Incohérence de couleur | p.3, test_docintelligence, label/texte accent dont
  la teinte diverge | A: extraire la couleur exacte par span.<br>R: sur le
  natif PDF, c’est globalement bien extrait. | A: sans rôle réel.<br>R: non
  impliqué. | A: sans rôle réel, sauf si la couleur participe à la hiérarchie
  visuelle.<br>R: non exécutable. | A: réutiliser la couleur extraite sur tous
  les chemins de rendu.<br>R: il reste au moins un chemin où la couleur finale
  diffère de l’attendu. |
  | Désalignement / fallback d’alignement | p.26, Practical SQL, ligne qui
  devrait garder son alignement source | A: remonter correctement l’origine
  géométrique de la ligne.<br>R: c’est généralement disponible via bbox/line
  origin. | A: choisir un mode compatible avec l’alignement source réel.<br>R:
  le classifier reste trop global pour gérer ces cas ligne par ligne. | A:
  fournir des alignment_guides vraiment contraignants.<br>R: ils restent
  surtout descriptifs. | A: respecter l’alignement source tant que
  possible.<br>R: _resolve_applied_alignment() bascule quand
  line_wider_than_slot; c’est fréquent dans les audits. |
  | Espacement vertical et horizontal instable | p.26, Practical SQL, front
  matter très tassé; p.23, Advances, espacement irrégulier | A: remonter une
  métrique de baseline/interligne exploitable.<br>R: l’extraction donne des
  bboxes, mais pas une vraie grille typographique robuste. | A: signaler “dense
  page rhythm / front matter rhythm / table rhythm”.<br>R: pas de sortie
  réellement consommée pour ça. | A: produire baseline_clusters, row_clusters,
  rythmes verticaux exécutables.<br>R: le descriptor sait décrire, pas encore
  gouverner fermement. | A: composer selon les hauteurs glyphiques et la grille
  source.<br>R: le renderer applique encore des facteurs fixes (1.22, gaps
  heuristiques, baselines heuristiques). |
  | Hiérarchie visuelle mal restituée | p.26, Practical SQL, PRACTICAL SQL. /
  bloc de copyright / autres éléments | A: fournir rôles et styles assez précis
  pour séparer titre, sous-titre, corps, métadonnées.<br>R: extraction OK sur
  le style, mais pas toujours suffisante sur les rôles structurels. | A:
  reconnaître correctement front matter, chapter opening, title page,
  etc.<br>R: le legacy reste parfois faux ou trop prudent (unknown,
  safe_mixed). | A: construire une hiérarchie de sections/dépendances stable et
  exécutable.<br>R: heads_content, section_sibling restent heuristiques et
  faibles. | A: rendre chaque niveau avec son propre conteneur et sa propre
  discipline typographique.<br>R: le renderer peut aplatir la hiérarchie en un
  flux ou un ensemble de blocs trop semblables. |
  | Fidélité des tableaux / grilles insuffisante | p.241, test_docintelligence,
  cellule/ligne de tableau rendue comme texte ordinaire | A: extraire
  explicitement lignes/colonnes/cellules et relations de grille.<br>R: selon
  les pages, la structure table reste partielle ou fusionnée avec du body. | A:
  classer plus finement que table_dominant ou mixed_blocks.<br>R: la décision
  reste grossière. | A: fournir un graphe exécutable row/column/cell.<br>R: le
  descriptor legacy ne va pas assez loin; le v3 n’est pas encore moteur. | A:
  avoir un moteur de rendu tableau dédié.<br>R: le renderer traite encore
  beaucoup de cellules comme des blocs textuels contraints. |
  | Blocs surcompressés | p.26, Practical SQL, bloc éditorial tassé pour
  “rentrer” | A: remonter une hauteur de bloc et des sous-unités
  suffisantes.<br>R: quand le bloc est trop compact dès l’entrée, la correction
  en aval devient difficile. | A: choisir entre reflow, lock, expansion
  verticale.<br>R: le classifier ne porte pas ce choix à un niveau assez fin. |
  A: marquer les blocs qui peuvent s’étendre verticalement vs ceux qui doivent
  rester verrouillés.<br>R: le descriptor ne le transforme pas en contrainte
  dure. | A: éviter le tassement en préférant recomposition ou expansion
  locale.<br>R: le renderer compresse encore certains blocs par shrink ou
  réduction d’interlignage implicite. |
  | Préservation trop rigide / reflow insuffisant | p.143, Advances, paragraphe
  dense qui reste trop proche de la géométrie source et se dégrade | A:
  extraire source lines/spans assez fines pour autoriser une recomposition
  contrôlée.<br>R: la granularité reste parfois trop grossière. | A: signaler
  clairement “translation expansion expected”.<br>R: le classifier legacy garde
  souvent un biais “safe_mixed / exact-ish” trop prudent. | A: fournir une
  recette de transformation locale, pas seulement des groupes
  relationnels.<br>R: reconstruction_plan reste trop résumé. | A: reflow local
  intelligent quand la traduction allonge la ligne.<br>R: le renderer hésite
  encore entre préservation rigide et reflow trop tardif; d’où
  layout_unchanged, preserve_mismatch, unchanged_reflow. |

  Verdict froid:

  - Extractor: plutôt solide sur le texte natif et la typo native, mais encore
    trop grossier sur la granularité structurelle de certaines pages.
  - classifier: trop global, trop tôt dans la chaîne, pas assez orienté
    contraintes de rendu.
  - descriptor: riche en labels, trop pauvre en contraintes exécutables.
  - reconstructor: reste le principal lieu où les défauts visibles se
    matérialisent.

  Si tu veux, l’étape suivante logique est que je transforme ce tableau en plan
  d’attaque: Défaut -> racine prioritaire -> correction à coder -> test de non-
  régression.





