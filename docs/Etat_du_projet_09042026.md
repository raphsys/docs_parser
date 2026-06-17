Analyse globale du projet docs_parser

  Architecture générale

  Le pipeline est : PDF/image → OCR (ocr_server.py) → Extraction structurelle
  (structure_extractor.py) → Enrichissement layout (layout_descriptor_v3.py +
  nouveaux modules) → Traduction (translator.py) → Reconstruction
  (reconstructor.py) → PDF final

  ---
  Ce qui marche bien

  1. Extraction et enrichissement (Portes 1-3)

  - P2 Appartenance bloc = 1.0 sur les 10 pages auditées — le rattachement
  élément→bloc est fiable et stable. Ne pas y toucher.
  - P4 Traduction = 1.0 partout — le pipeline de traduction (CT2/M2M100) est
  fonctionnel, les protections (code, nombres, références) sont opérationnelles.
  - ocr_server.py (3832 lignes) est très complet : merge de blocs natifs/OCR,
  détection d'overlays immutables, sémantique inline, groupes sémantiques,
  contrats de traduction — tout cela fonctionne.
  - Le pipeline de relations (relative_geometry.py, element_relations.py,
  positioning_policy.py, element_rulesets.py) est documenté et
  architecturalement solide — les 5 modules s'enchaînent proprement.

  2. Reconstructor — ossature nouvelle

  - Les dataclasses (BlockReconstructionPlan, LineTemplate, PlacableUnit,
  GraphEdge) sont en place et cohérentes avec la spec.
  - Le dispatcher _block_supported_by_hierarchical_engine() fonctionne et route
  vers le nouveau moteur.
  - EditorialBlockRenderer est réel, pas un stub — il gère bbox-anchored,
  relative-slots, wrapping, scaling.
  - Le fallback de présence (2ème passe à la fin de reconstruct()) est une
  sécurité filet utile.

  3. Scores observés

  - Page 148 : rendered_text_coverage = 0.95, word_overlaps 126→4,
  visual_similarity 0.73→0.80
  - Page 8 : rendered_text_coverage = 0.9524, 79/84 units couvertes

  ---
  Ce qui ne marche pas encore

  Problème #1 — P1 Extraction = 0.0 sur pages académiques denses

  Fichier : ocr_server.py

  La segmentation OCR produit des spans "gloutons" sur les blocs denses : un
  paragraphe entier en une seule unité, des micro-symboles (indices, exposants,
  variables) perdus. Le score P1 tombe à 0 sur Advances in Deep Learning.pdf
  p.101. Tant que P1 est à 0, P3 et P5 n'ont pas de sens à optimiser.

  Problème #2 — P5 Reconstruction = 0.43 sur pages SQL/code annoté

  Fichier : reconstructor.py

  Les marqueurs ➊➋➌➍➎ sont absorbés dans le flux editorial au lieu d'être
  traités comme unités protégées. Les longues lignes SQL cassent au mauvais
  endroit. Le solveur de placement n'a pas de mode "bloc technique avec
  relations label→valeur→continuation".

  Problème #3 — P6 Fond = 0.11 sur pages techniques mixtes

  Fichier : ocr_server.py → logique bg_master

  Le fond reconstruit garde des résidus ou efface trop. Le double-whiteout
  (bg_master propre + effacement local redondant) réintroduit des artefacts. La
  correction du whiteout redondant n'est que partielle.

  Problème #4 — CodeBlockRenderer est un quasi-stub

  class CodeBlockRenderer(BaseBlockRenderer):
      def render(self, page, plan):
          overlay_ops = self._overlay_ops_for_matching_immutable_overlays(plan)
          if overlay_ops:
              return overlay_ops
          return []  # ← si pas d'overlay : rendu vide
  Si le code n'a pas d'overlay pré-capturé, il disparaît complètement du rendu.

  Problème #5 — TableBlockRenderer : reflow paragraphe, pas cellule par cellule

  Le renderer de tableau concatène les lignes et réduit la fonte en boucle
  jusqu'à ce que ça rentre. C'est exactement ce que la spec interdit (pas de
  reflow tableau comme paragraphe). Les cellules complexes sont aplaties.

  Problème #6 — Couplage fort au legacy via _legacy_call

  BaseBlockRenderer._resolve_style, TableBlockRenderer.render,
  _wrap_text_for_bbox, EditorialBlockRenderer._measure_text — tous appellent
  _legacy_call("_resolve_style_font", ...) et
  _legacy_call("_measure_text_width", ...). Le nouveau moteur est
  architecturalement dépendant du vieux pour les fonctions de bas niveau
  critiques. Cela fragilise toute la chaîne si le .bak diverge.

  Problème #7 — Couche de normalisation finale manquante

  La synthèse du 08/04 le dit clairement : il manque une étape entre extraction
  et dessin — la normalisation finale des unités de rendu par bloc (choix du
  niveau, déduplication, résolution des conflits). _canonicalize_block_units()
  existe (l.1287) mais reste incomplet selon les logs de déduplication.

  Problème #8 — pytest non installé dans .docs-parser

  Les tests ne peuvent pas tourner dans l'environnement projet. Cela casse la
  boucle de vérification continue.

  ---
  Propositions priorisées

  Axe A — Extraction (impact immédiat sur P1 et P6)

  Priorité 1 — ocr_server.py : segmentation fine sur blocs denses
  - Détecter et séparer : texte éditorial continu / symboles isolés /
  indices-exposants / variables courtes
  - Limiter la taille max d'un span "glouton" (seuil de caractères ou de mots)
  - Fiabiliser les bboxes des micro-unités (formules inline, indice i, αβγ)

  Priorité 2 — Fond maître : éliminer le double-effacement
  - Dans _insert_page_background ou _build_block_geometry_context : si bg_master
   propre détecté, ne pas déclencher de whiteout local sur ce bloc
  - Tracer explicitement background_strategy = "preserve" quand le fond est déjà
   net

  Axe B — Reconstruction (impact sur P5)

  Priorité 3 — CodeBlockRenderer : fallback texte si pas d'overlay
  # Quand overlay_ops est vide, rendre le texte source ligne par ligne
  # avec style monospace exact, sans traduction, sans reflow

  Priorité 4 — Marqueurs annotés protégés (➊➋➌)
  - Dans _normalize_placable_units ou _phrase_units : détecter les symboles
  annotés Unicode (➊-➓, ①-⑩, etc.)
  - Les marquer protected_inline=True, immutable=True,
  group_class="annotation_marker"
  - Le EditorialBlockRenderer doit les placer comme unités atomiques
  non-wrappables

  Priorité 5 — TableBlockRenderer : cellule par cellule
  - Remplacer la boucle "réduire la fonte" par un rendu par cellule réelle
  - Utiliser block.cells si disponible, sinon lines comme proxy de cellules
  - Respecter l'alignement local par cellule

  Axe C — Infrastructure

  Priorité 6 — Extraire _resolve_style_font et _measure_text_width hors du
  legacy
  - Ces deux fonctions sont le seul vrai couplage bloquant avec .bak
  - Les déplacer dans reconstructor.py (ou un helper dédié) pour que le nouveau
  moteur soit autonome

  Priorité 7 — Installer pytest dans .docs-parser
  .docs-parser/bin/pip install pytest
  Les 16 fichiers de tests sont écrits, ils ne demandent qu'à tourner.

  ---
  Ordre recommandé

  ┌──────────┬─────────────────────┬───────────────────────┬────────────────┐
  │ Priorité │       Action        │        Fichier        │ Impact attendu │
  ├──────────┼─────────────────────┼───────────────────────┼────────────────┤
  │ 1        │ Segmentation fine   │ ocr_server.py         │ P1 ↑ pages     │
  │          │ blocs denses        │                       │ académiques    │
  ├──────────┼─────────────────────┼───────────────────────┼────────────────┤
  │          │ Éliminer            │ ocr_server.py +       │                │
  │ 2        │ double-effacement   │ reconstructor.py      │ P6 ↑           │
  │          │ fond                │                       │                │
  ├──────────┼─────────────────────┼───────────────────────┼────────────────┤
  │ 3        │ CodeBlockRenderer   │ reconstructor.py      │ Code non perdu │
  │          │ fallback texte      │                       │                │
  ├──────────┼─────────────────────┼───────────────────────┼────────────────┤
  │ 4        │ Marqueurs annotés   │ reconstructor.py      │ P5 ↑           │
  │          │ protégés            │                       │ SQL/annoté     │
  ├──────────┼─────────────────────┼───────────────────────┼────────────────┤
  │ 5        │ TableBlockRenderer  │ reconstructor.py      │ Tables fidèles │
  │          │ par cellule         │                       │                │
  ├──────────┼─────────────────────┼───────────────────────┼────────────────┤
  │          │ Extraire fonctions  │                       │ Déscouplage    │
  │ 6        │ de mesure hors      │ reconstructor.py      │ architectural  │
  │          │ legacy              │                       │                │
  ├──────────┼─────────────────────┼───────────────────────┼────────────────┤
  │ 7        │ Installer pytest    │ .docs-parser/bin/pip  │ CI locale      │
  │          │                     │                       │ fonctionnelle  │
  └──────────┴─────────────────────┴───────────────────────┴────────────────┘

  Le verrou le plus prioritaire reste P1 = 0 sur les pages académiques — tant   
  qu'il n'est pas corrigé, toute optimisation du reconstructor travaille sur un
  payload corrompu.  
