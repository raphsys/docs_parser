• 1. Le document traduit est-il vraiment “ready publication” ?
     Non. Sur ce cas de référence, je ne le considérerais pas publication-ready.
     Les signaux objectifs vont dans ce sens:

  - la comparaison visuelle globale est correcte mais pas suffisante: overall =
    0.7935, avec alignment_consistency = 0.7416 dans visual_compare.py
  - le PDF reconstruit a encore 5 chevauchements de mots et 3 collisions texte/
    image selon evaluate_layout_quality.py
  - le rendu textuel final contient des formulations manifestement mauvaises ou
    non éditoriales, par exemple Les produits de l'industrie répété, ou Il s '
    agit d ' un projet de loi sur les droits de l ' homme. dans le PDF traduit
    reconstructed_test_docintelligence-8_alignment_check_FR.pdf

  2. Est-ce que tous les textes de l’original figurent traduits dans le document
     traduit ?
     Non, pas complètement. Le PDF final montre une perte de contenu textuel
     exploitable.

  - l’original extrait 101 lignes, dont 46 lignes significatives
  - le PDF traduit n’en extrait plus que 48, dont 24 significatives
  - plusieurs entrées visibles dans l’original disparaissent ou sont réduites à
    des numéros dans le traduit. Exemples visibles dans les extraits:
      - original: Image classification using MLP, Putting it all together, CNN
        architecture, The big picture
      - traduit extrait: beaucoup de lignes deviennent juste 94, 96, 97, 3.2.,
        3.3., etc.
  - certains textes restent aussi non traduits, par exemple CONTENTS et
    Convolutional neural networks

  Donc la réponse est non: tout le texte utile de l’original ne se retrouve pas
  proprement traduit dans le document final.

  3. La présentation du document traduit est-elle optimale par rapport à
     l’original ?
     Non. Elle est meilleure qu’avant sur l’alignement, mais elle n’est pas
     optimale.

  - point positif: l’audit d’alignement est maintenant propre,
    alignment_mismatch = 0 dans reconstructed_test_docintelligence-
    8_alignment_check_FR_style_audit.json:1
  - points encore faibles:
      - font_fallback_count = 4
      - consistance visuelle moyenne sur l’alignement global de page seulement
        0.7416
      - présence de collisions texte/image
      - densité de texte et hiérarchie réduites par rapport à l’original
      - contenu éditorial parfois dégradé, ce qui casse la qualité perçue même
        si la mise en page “tient”

  Conclusion courte

  - Ready publication : non
  - Tous les textes traduits présents : non
  - Présentation optimale par rapport à l’original : non

  Le verrou principal n’est plus seulement l’alignement. Maintenant, ce qui
  bloque clairement la qualité finale, c’est la couverture réelle du contenu
  traduit et la qualité éditoriale des traductions reconstruites.


› En analysant les données et codes (structures des documents, extraction des
  éléments de structures du document original, méthode d'extraction des
  éléments de mise en forme, traduction, et reconstruction du document
  traduit), que proposes tu pour les corrections afin d'avoir des éléments OUI
  partout ?
  Sois créatif et ingénieurs de code et ingénieurs de modèles IA.
  Réfléchis profondément, recherche des solutions sur le web, dans les bases de
  connaissances, dans les modèles IA, etc ...
  Ne te limites pas à tes connaissances, n'inventes rien, adapte les solutions
  trouvées à la structure de notre projet. Proposes des solutions innovantes,
  même si elles sont compliquées, même si elle font appels à des outils autres
  que celles que nous avons déjà.

  Go !


• Je vais élargir le diagnostic avec des sources primaires sur l’état de l’art
  pertinent pour votre pipeline: parsing PDF natif, OCR/layout, traduction
  contrainte par structure, et reconstruction fidèle. Ensuite je te propose des
  corrections adaptées à votre code, classées par impact et complexité.

• Searched PyMuPDF official documentation text extraction dict spans fonts
  page.get_text dict

• Searched site:docling-project.github.io Docling official PDF pipeline

• Searched site:paddleocr.ai PP-StructureV3 document recovery

• Searched CTranslate2 official documentation constrained decoding glossary

• Searched site:ai.meta.com No Language Left Behind paper official

• Searched site:learn.microsoft.com document translation preserve formatting
  official

─ Worked for 1m 01s ────────────────────────────────────────────────────────────

• Verdict

  Pour obtenir des OUI partout, je ne crois pas qu’un simple tuning local du
  reconstructeur suffise. Le projet a déjà franchi une étape importante sur
  l’alignement, mais l’analyse du cas test_docintelligence-8.pdf montre que le
  vrai verrou est maintenant architectural:

  - la structure source n’est pas assez fidèle au niveau utile pour une
    reconstruction publication-ready
  - la traduction n’est pas assez contrainte par la structure et la géométrie
  - la reconstruction ne vérifie pas assez la couverture et la qualité
    éditoriale avant d’écrire le PDF final

  Donc ma proposition est une refonte en pipeline à garde-fous, pas juste des
  correctifs ponctuels.

  ———

  Ce qu’il faut changer

  ### 1. Passer d’un pipeline “OCR -> traduction -> reconstruction” à un
  pipeline “parse natif prioritaire -> traduction contrainte -> validation ->
  reconstruction”

  Aujourd’hui, le backend de ocr_server.py et native_pdf_extractor.py mélange
  extraction native, OCR, heuristiques et traduction dans une chaîne unique.

  Je recommande de formaliser 4 couches:

  1. Source Fidelity Layer
  2. Translation Planning Layer
  3. Layout-Constrained Translation Layer
  4. Publication QA Gate

  Sans cette séparation, le système traduit parfois le mauvais niveau d’unité,
  perd des blocs, ou réinjecte un texte correct dans une mauvaise boîte.

  ### 2. Introduire un format intermédiaire canonique plus riche

  Votre structure actuelle est déjà bonne, mais pas encore suffisante pour
  garantir la fidélité publication-ready.

  Je créerais un schéma intermédiaire explicite, par exemple layout.v3, avec
  pour chaque unité:

  - unit_id
  - page_id
  - source_kind: native_span, ocr_span, merged_line, ocr_block, figure_text,
    page_number, running_header, toc_entry, equation, symbol, list_marker
  - text_source
  - text_normalized
  - bbox_px et bbox_pt
  - font_raw, font_family_guess, font_metrics_profile
  - alignment, indent, tab_stops, reading_order_rank
  - translatable: true/false
  - translation_strategy
  - coverage_required: strict/optional
  - render_mode: exact/reflow/overlay/background_only

  En pratique, c’est cette couche qui évitera les pertes de contenu.

  ———

  Corrections ciblées par module

  ### 1. native_pdf_extractor.py: passer au niveau caractère et span réel

  Aujourd’hui, vous utilisez surtout page.get_text("dict"). Les docs PyMuPDF
  indiquent que extractRAWDICT() / RAWJSON donnent un niveau plus fin, jusqu’au
  caractère, avec ordre et structure plus détaillés.
  Source: PyMuPDF official docs: TextPage RAWDICT
  (https://pymupdf.qubitpi.org/en/latest/textpage.html), Appendix on DICT/RAWDI
  CT (https://pymupdf.readthedocs.io/en/latest/app1.html)

  Ce que je propose

  - remplacer ou compléter get_text("dict") par get_text("rawdict", sort=False)
    sur les PDF natifs
  - construire des spans “métriques” par regroupement intelligent de chars, pas
    uniquement par spans PDF fournis
  - stocker:
      - largeur réelle par char
      - baseline estimée
      - gaps inter-mots
      - tabulations implicites
      - sauts de ligne d’origine
  - détecter explicitement:
      - running headers/footers
      - numéros de page
      - TOC entries
      - leaders de points
      - list markers
      - références inline
      - éléments non traduisibles

  Pourquoi

  - aujourd’hui, quand un bloc significatif disparaît ou devient un numéro isolé
    dans le PDF traduit, c’est souvent que l’unité source n’était pas bonne
  - le niveau caractère permet de reconstruire ce qui est réellement affiché,
    pas ce qu’une heuristique croit lire

  ### 2. structure_extractor.py: transformer l’extraction en “parser de
  document”, pas juste “extracteur de blocs”

  C’est ici qu’il faut mettre l’intelligence de segmentation documentaire.

  Ce que je propose

  - ajouter une vraie phase de document role tagging
  - modèles/règles spécifiques pour:
      - toc_entry(title, page_number, leader, level)
      - header(title, page_number)
      - section_heading(number, title, page_ref?)
      - list_item(marker, body, page_number?)
      - caption
      - equation_inline
      - diagram_label
  - utiliser le XML de layout déjà généré comme vérité partielle, mais le
    compléter avec une version plus riche

  Innovant mais pertinent

  - brancher un parseur externe en mode validation, pas forcément en
    remplacement:
      - Docling pour lecture d’ordre, structure, tables, formules, export
        lossless JSON
        Source: Docling GitHub (https://github.com/docling-project/docling)
      - PP-StructureV3 pour layout complexe, reading order, formula/table
        recovery, Markdown recovery
        Source: PP-StructureV3 official

  (https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-Stru
ctureV3/PP-StructureV3.html)

  Architecture recommandée

  - garder votre parser maison comme moteur principal
  - lancer Docling ou PP-StructureV3 comme second opinion parser
  - comparer:
      - nombre d’unités
      - reading order
      - titres détectés
      - lignes TOC
  - si divergence forte, marquer la page en high-risk et basculer sur une
    stratégie de traduction/reconstruction plus prudente

  Ça, c’est une vraie approche ingénierie de production.

  ———

  ### 3. translator.py: arrêter la traduction “au texte”, passer à la traduction
  “au plan de page”

  C’est ici qu’il y a aujourd’hui le plus gros manque.

  Le traducteur devrait recevoir non seulement du texte, mais aussi:

  - le type d’unité
  - la boîte disponible
  - la politique de reflow
  - les unités voisines
  - la criticité de couverture

  Je recommande 3 modes de traduction

  #### Mode A: exact-preserve

  Pour:

  - numéros
  - symboles
  - listes
  - équations
  - références
  - headers spéciaux
  - acronymes

  Traduction interdite ou ultra-contrainte.

  #### Mode B: layout-constrained

  Pour:

  - TOC
  - headings
  - captions
  - labels
  - blocs courts

  Le traducteur doit produire une sortie qui respecte:

  - longueur cible max
  - termes obligatoires
  - interdits lexicaux
  - style du bloc

  #### Mode C: semantic-reflow

  Pour:

  - paragraphes longs

  Là, on autorise plus de liberté, mais avec validation postérieure.

  Techniquement
  Votre usage CT2 est bon pour l’inférence rapide.
  Source: CTranslate2 official (https://github.com/OpenNMT/CTranslate2)

  Mais pour obtenir de la qualité publication-ready, je recommande une
  architecture à deux étages:

  1. draft translation avec votre modèle rapide CT2
  2. constrained post-edit avec un modèle plus intelligent

  Ce second étage peut être:

  - local VLM/LLM si dispo
  - ou service externe sur pages haut risque seulement

  Exemple de prompt structuré pour post-édition:

  - unité: section_heading
  - source: A closer look at feature extraction
  - cible provisoire
  - largeur source
  - langue cible
  - style attendu: court, éditorial, titre de sommaire
  - contrainte: max 38 caractères, pas d’anglais résiduel

  Important
  Il faut arrêter de traduire des blocs sans objectif de forme.
  Le traducteur doit produire une traduction compatible avec la boîte.

  ———

  ### 4. text_composer.py + reconstructor.py: transformer le moteur de rendu en
  solveur de contraintes

  Là, il faut assumer que reconstruire fidèlement est un problème
  d’optimisation.

  Aujourd’hui, le moteur compose. Il doit devenir un layout solver.

  Pour chaque unité traduisible, résoudre

  - couverture du texte
  - respect bbox
  - alignement
  - police
  - taille
  - interlignage
  - collisions
  - proximité visuelle à l’original

  Fonction objectif
  Minimiser:

  - overflow
  - underfill extrême
  - collision texte/image
  - décalage de baseline
  - écart de largeur par ligne
  - nombre de lignes différent de la source
  - mismatch de rôle typographique

  Stratégie

  - garder les lignes source quand elles sont fiables
  - si traduction trop longue:
      1. police métriquement proche
      2. micro-ajustement tracking
      3. variation limitée de taille
      4. reflow strict
      5. pagination contrôlée
  - si aucune solution propre:
      - ne pas publier
      - marquer publication_gate_failed

  Autrement dit: il faut accepter qu’un PDF puisse être “refusé” par le pipeline
  final.

  ### 5. font_resolver.py: passer à une résolution métrique multi-candidats

  Le vrai besoin n’est pas “une police qui ressemble”, mais “une police qui
  remplit la même largeur”.

  Je recommande

  - index de polices locales avec métriques:
      - largeur alphabet de référence
      - hauteur x
      - ascender/descender
      - largeur moyenne mots courts
  - résolution par score:
      - similarité style
      - similarité largeur
      - similarité serif/sans
      - similarité bold/italic

  Ensuite, dans reconstructor.py, choisir la police par minimisation de l’erreur
  de largeur, pas juste par famille.

  ### 6. ocr_server.py: introduire une vraie QA finale avant livraison

  Le pipeline doit sortir un statut qualité, pas seulement un PDF.

  Je créerais une phase publication_qa() qui calcule:

  - content_coverage_score
  - translation_quality_score
  - layout_fidelity_score
  - collision_score
  - english_leak_score
  - font_risk_score
  - publication_ready: yes/no

  Règles minimales

  - si contenu source significatif absent dans le PDF final: no
  - si collisions > seuil: no
  - si lignes anglaises résiduelles dans un PDF FR: no
  - si trop de blocs haute importance sans traduction valide: no

  Le PDF n’est livré au client que si publication_ready == yes, sinon:

  - livrer un draft
  - plus un rapport d’erreurs exploitable

  ———

  Solutions innovantes que je recommande vraiment

  ### A. Double parsing compétitif

  Exécuter en parallèle:

  - votre parseur maison
  - Docling
  - PP-StructureV3

  Puis fusionner ou voter.

  Pourquoi c’est intéressant:

  - Docling est fort pour représentation structurée et PDF understanding
  - PP-StructureV3 annonce amélioration layout, table, formula, reading order,
    Markdown recovery
    Source: PP-StructureV3 official

  (https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-Stru
ctureV3/PP-StructureV3.html)
  - votre pipeline est déjà bon pour la reconstruction PDF finale

  Donc:

  - parse externe
  - reconstruct interne

  C’est probablement le meilleur compromis.

  ### B. “Document translation oracle” externe comme juge, pas forcément comme
  moteur

  Pour les pages critiques, utiliser un service de document translation qui
  préserve mise en forme comme référence de comparaison:

  - Google Cloud Document Translation: préservation du format/layout,
    glossaries, PDF natif vs scanné
    Source: Google Cloud Document Translation
    (https://cloud.google.com/translate/docs/advanced/translate-documents)
  - Microsoft Document Translation: préservation format/structure
    Source: Microsoft Learn OneDrive document translation
    (https://learn.microsoft.com/en-us/sharepoint/onedrive-document-translation)

  Je ne dis pas de remplacer votre pipeline par Google/Microsoft.
  Je dis de les utiliser comme:

  - oracle qualité
  - baseline
  - fallback enterprise pour pages impossibles

  C’est une pratique de vraie équipe produit.

  ### C. “Coverage validator” par alignement source-cible

  Créer un module qui vérifie que chaque unité significative source a un
  correspondant dans le document final.

  Méthode:

  - construire des embeddings ou signatures lexicales par unité
  - faire correspondre source unit -> translated unit -> rendered unit
  - détecter:
      - supprimé
      - compressé excessivement
      - fusionné
      - laissé en anglais
      - transformé en artefact

  Sans cette brique, vous ne pourrez pas garantir “tous les textes figurent
  traduits”.

  ### D. VLM juge de page

  Utiliser un VLM local ou cloud pour noter:

  - “ce PDF traduit a-t-il la même structure que l’original ?”
  - “le sommaire traduit conserve-t-il tous les items ?”
  - “voit-on du texte non traduit ?”
  - “y a-t-il des collisions ?”

  Ce n’est pas le moteur principal, c’est un judge.
  Vu que vous avez déjà des modèles GGUF/Qwen en local, c’est cohérent avec le
  repo.

  ### E. Apprentissage ciblé sur vos cas réels

  Créer un dataset de supervision à partir de vos propres PDF:

  - page source
  - structure gold
  - traduction gold
  - reconstruction gold
  - verdict publication-ready

  Ensuite:

  - fine-tune ou distiller un classifieur de risque de page
  - fine-tune un reranker de traduction pour TOC/headings/labels
  - fine-tune une résolution de police orientée métrique

  C’est plus long, mais c’est le chemin vers des OUI robustes.

  ———

  Plan d’implémentation réaliste

  ### Phase 1. Fiabiliser la vérité source

  - native_pdf_extractor.py: RAWDICT + char-level spans
  - structure_extractor.py: tags documentaires forts
  - layout.v3 comme schéma canonique

  ### Phase 2. Traduction contrainte

  - étendre translator.py avec translation_strategy par type d’unité
  - ajouter un second étage post_edit_constrained
  - intégrer glossaires hiérarchiques par rôle de bloc, pas seulement domaine

  ### Phase 3. Solveur de reconstruction

  - font_resolver.py: scoring métrique
  - text_composer.py: solveur avec coût
  - reconstructor.py: publication gate + diagnostic détaillé

  ### Phase 4. Vérification automatique

  - coverage_validator.py
  - publication_qa.py
  - benchmark sur tests/doc_pdf/ avec seuils stricts

  ### Phase 5. Hybridation externe

  - brancher Docling / PP-StructureV3
  - éventuellement benchmark Google/Microsoft Document Translation comme oracle
    externe

  ———

  Réponse franche

  Si tu veux des OUI partout, il faut viser un système qui:

  - représente mieux le document source
  - traduit en fonction de la boîte et du rôle
  - refuse les sorties non publiables
  - utilise des parseurs concurrents et des juges automatiques

  La vraie correction n’est pas “mieux reconstruire”.
  La vraie correction est: faire du document engineering avec validation qualité
  explicite.

  Sources

  - PyMuPDF RAWDICT / text extraction:
    https://pymupdf.qubitpi.org/en/latest/textpage.html
  - PyMuPDF Appendix on DICT/RAWDICT:
    https://pymupdf.readthedocs.io/en/latest/app1.html
  - PP-StructureV3 official:

  https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/algorithm/PP-Struc
tureV3/PP-StructureV3.html
  - Docling official: https://github.com/docling-project/docling
  - CTranslate2 official: https://github.com/OpenNMT/CTranslate2
  - Google Cloud Document Translation:
    https://cloud.google.com/translate/docs/advanced/translate-documents
  - Microsoft document translation:
    https://learn.microsoft.com/en-us/sharepoint/onedrive-document-translation

  Si tu veux, l’étape suivante utile est que je te transforme ça en plan
  d’architecture concret du repo, fichier par fichier, avec ce qu’on garde, ce
  qu’on remplace, et l’ordre exact d’implémentation.

