› Prends 10 pages random et fais moi voir leur trame de fond


• Je vais prendre 10 fonds propres existants dans les résultats, générer une
  planche contact avec leurs trames de fond, puis te donner le chemin du PNG.

■ Conversation interrupted - tell the model what to do differently. Something
went wrong? Hit `/feedback` to report the issue.


› Prends 1à nouvelles pages, jamais utilisé au paravant


• Je comprends 1à comme 10. Je vais choisir 10 pages qui ne sont pas déjà
  présentes dans les runs show*, générer/extraire leur trame de fond, puis
  produire une planche contact consultable.

────────────────────────────────────────────────────────────────────────────────

• Je vais éviter les pages déjà vues dans show3/show10: les pages utilisées
  incluent notamment 36, 51, 87, 129, 133, 140, 180, 192, 229, 284, 294, 337,
  406, 457, 463, 505. Je vais sélectionner 10 autres PDF test_docintelligence-*
  et produire uniquement les fonds, pour rester léger.

────────────────────────────────────────────────────────────────────────────────

• Je vais faire plus direct que le pipeline complet: sélectionner 10 PDF
  test_docintelligence-N.pdf jamais présents dans les artefacts results/*,
  rendre la page source, effacer les zones de texte par inpaint OpenCV, et créer
  une planche contact des fonds nettoyés.

────────────────────────────────────────────────────────────────────────────────

• La première sélection a trouvé 10 candidates, mais le compteur “déjà utilisé”
  semble trop faible par rapport aux runs show10. Je vérifie la détection avant
  de générer les images, pour ne pas reprendre une page déjà vue.

────────────────────────────────────────────────────────────────────────────────

• Je vais générer une planche avec deux vues par page: à gauche le fond nettoyé
  normal, à droite une version contraste amplifié pour rendre la trame visible.
  Les fichiers individuels seront aussi sauvegardés.

────────────────────────────────────────────────────────────────────────────────

• Le Python système n’a pas PyMuPDF (fitz). Je relance avec l’environnement
  projet .docs-parser, celui utilisé par les tests.

────────────────────────────────────────────────────────────────────────────────

• La planche est générée. Je l’ouvre rapidement côté workspace pour vérifier que
  l’image n’est pas vide et que les 10 fonds sont visibles.

────────────────────────────────────────────────────────────────────────────────

• J’ai pris 10 nouvelles pages test_docintelligence, absentes des artefacts
  results/* existants:

  p0024, p0028, p0038, p0149, p0300, p0334, p0388, p0429, p0445, p0466

  Planche contact ici: results/new10_background_textures_20260615/
  contact_sheet_clean_backgrounds_texture.png

  Dans chaque tuile:

  - gauche: fond nettoyé normal
  - droite: trame amplifiée texture x4

  Les fichiers individuels sont dans: results/new10_background_textures_20260615


› continues


• Je continue avec 10 autres pages jamais utilisées, en excluant aussi le lot
  que je viens de générer. Je vais produire une nouvelle planche contact avec
  fond nettoyé + trame amplifiée.

⚠ Heads up, you have less than 25% of your weekly limit left. Run /status for a
  breakdown.

• Ran .docs-parser/bin/python - <<'PY'
  │ from pathlib import Path
  │ import re, random, json
  │ … +109 lines
  └ OUT results/new10_background_textures_20260616_batch2
    CONTACT results/new10_background_textures_20260616_batch2/
    contact_sheet_clean_backgrounds_texture.png
    PAGES p0033, p0077, p0219, p0252, p0264, p0279, p0315, p0324, p0358, p0414

────────────────────────────────────────────────────────────────────────────────

• Le second lot est généré. Je vérifie visuellement la planche pour confirmer
  qu’elle n’est pas vide et que les 10 fonds sont bien présents.

• Viewed Image
  └ results/new10_background_textures_20260616_batch2/
  contact_sheet_clean_backgrounds_texture.png

────────────────────────────────────────────────────────────────────────────────

• Deuxième lot généré avec 10 autres pages jamais utilisées:

  p0033, p0077, p0219, p0252, p0264, p0279, p0315, p0324, p0358, p0414

  Planche contact: results/new10_background_textures_20260616_batch2/
  contact_sheet_clean_backgrounds_texture.png

  Dossier complet: results/new10_background_textures_20260616_batch2

────────────────────────────────────────────────────────────────────────────────


› Parafit pour les trames de fonds.
  Maintenant sur ces trames de fonds, places les bboxes des "blocks",
  uniquement cela, et rien d'autres, et montres moi les fichiers


• Je vais prendre le dernier lot de 10 trames (batch2) et générer des images où
  seuls les rectangles des blocks texte sont dessinés sur les fonds nettoyés:
  pas de labels, pas de texte, pas d’autres couches.

• Ran .docs-parser/bin/python - <<'PY'
  │ from pathlib import Path
  │ import json, re
  │ … +62 lines
  └ OUT results/new10_background_textures_20260616_batch2/block_bboxes_only
    CONTACT results/new10_background_textures_20260616_batch2/block_bboxes_only/
    … +17 lines (ctrl + t to view transcript)
    p0414: 11 blocks -> results/new10_background_textures_20260616_batch2/
    block_bboxes_only/clean_background_blocks_only_p0414.png

────────────────────────────────────────────────────────────────────────────────

• Les fichiers individuels et la planche contact sont générés. Je vérifie
  rapidement la planche pour m’assurer qu’il n’y a bien que des fonds +
  rectangles de blocks.

• Viewed Image
  └ results/new10_background_textures_20260616_batch2/block_bboxes_only/
  contact_sheet_blocks_only.png

────────────────────────────────────────────────────────────────────────────────

• Fait sur le dernier lot de 10 trames. J’ai ajouté uniquement les bboxes des
  blocks texte, en rectangles rouges, sans labels ni autres couches.

  Planche contact: results/new10_background_textures_20260616_batch2/
  block_bboxes_only/contact_sheet_blocks_only.png

  Fichiers individuels:

  - results/new10_background_textures_20260616_batch2/block_bboxes_only/
    clean_background_blocks_only_p0033.png

  - results/new10_background_textures_20260616_batch2/block_bboxes_only/
    clean_background_blocks_only_p0077.png

  - results/new10_background_textures_20260616_batch2/block_bboxes_only/
    clean_background_blocks_only_p0219.png

  - results/new10_background_textures_20260616_batch2/block_bboxes_only/
    clean_background_blocks_only_p0252.png

  - results/new10_background_textures_20260616_batch2/block_bboxes_only/
    clean_background_blocks_only_p0264.png

  - results/new10_background_textures_20260616_batch2/block_bboxes_only/
    clean_background_blocks_only_p0279.png

  - results/new10_background_textures_20260616_batch2/block_bboxes_only/
    clean_background_blocks_only_p0315.png

  - results/new10_background_textures_20260616_batch2/block_bboxes_only/
    clean_background_blocks_only_p0324.png

  - results/new10_background_textures_20260616_batch2/block_bboxes_only/
    clean_background_blocks_only_p0358.png

  - results/new10_background_textures_20260616_batch2/block_bboxes_only/
    clean_background_blocks_only_p0414.png

────────────────────────────────────────────────────────────────────────────────


› C'est parfait.
  Maintenant dans ces bboxes, places les bboxes des lignes. ATTENTION !!! il
  s'agit de ligne de textes traduits ou non après pagetranslate/. Ce qui veut
  dire que tu dois:
  1- L'ensemble des textes/phrases/expressions/mots + zones spéciales,
  2- Ces éléments ré-organisées dans les blocks (nouvelles bboxes des phrases,
  succession des phrases, retour à la ligne etc ...)

  Donc deux fonctions :
  1- la 1ère vérifie la complétude des élements intrablocks après le passgage
  dans pagetranslate/
  2- la 2ème fait le calcul et les réajustement des positions, étendues,
  dispositions, retour à la ligne, à l'intérieur des blocks


• Je vais produire une version utile directement sur ces 10 pages: relancer la
  chaîne pageprint → pagetranslate → composition intra-block, vérifier que les
  éléments attendus sont couverts, calculer les bboxes de lignes réorganisées
  dans chaque block, puis dessiner uniquement blocks + lignes sur les fonds.

• Explored
  └ Read builder.py, test_textops_from_composition.py, schema.py,
         run_pageprint_pagetranslate_audit.py

────────────────────────────────────────────────────────────────────────────────

• Je vais ajouter les deux fonctions dans pagereconstruct/composition/
  intrablock_composer.py, avec tests courts, puis les utiliser pour générer les
  visuels sur les 10 pages. Pour éviter de dépendre du modèle de traduction,
  j’utiliserai pagetranslate en mode dry_run: le passage pagetranslate/
  projection existe bien, mais le texte cible reste identique quand aucune
  traduction n’est nécessaire pour calculer les lignes.

• Explored
  └ Read projection.py, _fixtures.py

• Ran python3 - <<'PY'
  │ import json
  │ from tests.pagereconstruct._fixtures import translated_input_data
  │ … +6 lines
  └ dict_keys(['schema_version', 'page', 'document', 'assets', 'visual_layers',
    'units', 'regions', 'views'])
    … +89 lines (ctrl + t to view transcript)
     translation_result <class 'NoneType'>
    null

────────────────────────────────────────────────────────────────────────────────

• Je vais coder ces fonctions comme des fonctions de contrat réutilisables, pas
  comme un script de visualisation: une fonction d’audit de complétude post-
  pagetranslate, et une fonction de composition qui retourne les nouvelles
  lignes dans les blocks. Ensuite le script de dessin ne fera qu’exécuter ces
  contrats.

• Edited pagereconstruct/composition/intrablock_composer.py (+117 -0)
    162      return [compose_block(b) for b in (getattr(contract, "blocks", [])
         or [])]
    163 +
    164 +
    165 +def verify_intrablock_elements_after_pagetranslate(translated_input_dat
         a: dict) -> dict:
    166 +    """Vérifie la complétude des éléments intra-block après PAGETRANSLA
         TE.
    167 +
    168 +    Couverture attendue:
    169 +    - chaque reconstruction_unit issue de pagetranslate doit référencer
          au moins
    170 +      une source unit et un texte cible/source;
    171 +    - chaque source unit consommée par une reconstruction_unit ne doit
         pas être
    172 +      rendue deux fois;
    173 +    - chaque élément préservé/exclu dans les plans spéciaux doit garder
          une bbox.
    174 +
    175 +    La fonction ne compose pas; elle audite uniquement le contrat d'ent
         rée de la
    176 +    composition.
    177 +    """
    178 +    views = translated_input_data.get("views") or {}
    179 +    reconstruction_units = views.get("reconstruction_units") or []
    180 +    preservation_plan = views.get("preservation_plan") or []
    181 +    exclusion_plan = views.get("exclusion_plan") or []
    182 +    blockers: list[str] = []
    183 +    findings: list[dict] = []
    184 +    covered: dict[str, str] = {}
    185 +    elements: list[dict] = []
    186 +
    187 +    for unit in reconstruction_units:
    188 +        rid = unit.get("reconstruction_unit_id") or unit.get("unit_id")
          or unit.get("translation_unit_id")
    189 +        sids = list(unit.get("source_unit_ids") or ([unit.get("unit_id"
         )] if unit.get("unit_id") else []))
    190 +        text = (unit.get("translated_text") or unit.get("text") or "").
         strip()
    191 +        bbox = unit.get("layout_bbox") or unit.get("bbox") or (unit.get
         ("render_target") or {}).get("layout_bbox")
    192 +        if not sids:
    193 +            blockers.append("intrablock_missing_source_unit")
    194 +            findings.append({"type": "intrablock_missing_source_unit",
         "element_id": rid})
    195 +        if not text:
    196 +            blockers.append("intrablock_missing_text")
    197 +            findings.append({"type": "intrablock_missing_text", "elemen
         t_id": rid})
    198 +        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
    199 +            blockers.append("intrablock_missing_bbox")
    200 +            findings.append({"type": "intrablock_missing_bbox", "elemen
         t_id": rid})
    201 +        for sid in sids:
    202 +            if sid in covered:
    203 +                blockers.append("intrablock_duplicate_source_unit")
    204 +                findings.append({"type": "intrablock_duplicate_source_u
         nit", "source_unit_id": sid,
    205 +                                 "first": covered[sid], "duplicate": ri
         d})
    206 +            covered[sid] = str(rid)
    207 +        elements.append({
    208 +            "element_id": rid,
    209 +            "kind": "translated_or_source_text",
    210 +            "source_unit_ids": sids,
    211 +            "translation_unit_id": unit.get("translation_unit_id"),
    212 +            "text": text,
    213 +            "bbox": list(bbox) if isinstance(bbox, (list, tuple)) and l
         en(bbox) == 4 else None,
    214 +            "role": unit.get("role"),
    215 +        })
    216 +
    217 +    for plan_name, entries, kind in (
    218 +        ("preservation_plan", preservation_plan, "special_preserved"),
    219 +        ("exclusion_plan", exclusion_plan, "special_excluded"),
    220 +    ):
    221 +        for entry in entries:
    222 +            eid = entry.get("preservation_id") or entry.get("exclusion_
         id") or entry.get("id")
    223 +            bbox = entry.get("bbox")
    224 +            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4)
         :
    225 +                blockers.append("intrablock_special_missing_bbox")
    226 +                findings.append({"type": "intrablock_special_missing_bb
         ox", "plan": plan_name, "element_id": eid})
    227 +            elements.append({
    228 +                "element_id": eid,
    229 +                "kind": kind,
    230 +                "source_unit_ids": list(entry.get("source_unit_ids") or
          []),
    231 +                "text": entry.get("text") or "",
    232 +                "bbox": list(bbox) if isinstance(bbox, (list, tuple)) a
         nd len(bbox) == 4 else None,
    233 +                "reason": entry.get("reason"),
    234 +            })
    235 +
    236 +    blockers = sorted(set(blockers))
    237 +    return {
    238 +        "status": "ko" if blockers else "ok",
    239 +        "hard_blockers": blockers,
    240 +        "findings": findings,
    241 +        "element_count": len(elements),
    242 +        "covered_source_unit_ids": sorted(covered),
    243 +        "elements": elements,
    244 +    }
    245 +
    246 +
    247 +def compute_intrablock_line_layouts(translated_input_data: dict, *, mod
         e: str = "debug") -> dict:
    248 +    """Calcule les lignes réorganisées dans les blocks après PAGETRANSL
         ATE.
    249 +
    250 +    Retourne les bboxes finales de lignes/runs en espace page pt. Les é
         léments
    251 +    source peuvent être traduits ou préservés: le calcul part du
    252 +    translated_input_data projeté par PAGETRANSLATE, puis compile le co
         ntrat
    253 +    pagereconstruct et compose chaque block.
    254 +    """
    255 +    completeness = verify_intrablock_elements_after_pagetranslate(trans
         lated_input_data)
    256 +    from pagereconstruct.input_adapter import PageReconstructInputAdapt
         er
    257 +    from pagereconstruct.final_contract import FinalReconstructionContr
         act
    258 +    from pagereconstruct.plan_compiler import compile_page_render_plan
    259 +
    260 +    normalized = PageReconstructInputAdapter().normalize(translated_inp
         ut_data)
    261 +    plan = compile_page_render_plan(translated_input_data, reconstructi
         on_mode=mode).to_dict()
    262 +    contract = FinalReconstructionContract.from_pageprint_pagetranslate
         (normalized, plan)
    263 +    compositions = [c.to_dict() for c in compose_contract(contract)]
    264 +    blockers = list(completeness.get("hard_blockers") or [])
    265 +    for comp in compositions:
    266 +        if comp.get("overflow"):
    267 +            blockers.append("intrablock_overflow")
    268 +        if comp.get("clipping"):
    269 +            blockers.append("intrablock_clipping")
    270 +        if comp.get("reading_order_ok") is False:
    271 +            blockers.append("reading_order_changed")
    272 +    blockers = sorted(set(blockers))
    273 +    return {
    274 +        "status": "ko" if blockers else "ok",
    275 +        "hard_blockers": blockers,
    276 +        "completeness": completeness,
    277 +        "blocks": compositions,
    278 +        "plan": plan,
    279 +    }

────────────────────────────────────────────────────────────────────────────────

• Fonctions ajoutées. Je crée maintenant deux tests minimaux: complétude OK sur
  fixture pagetranslate, et détection d’une reconstruction unit sans source
  unit.

• Added tests/pagereconstruct/test_intrablock_post_pagetranslate.py (+32 -0)
     1 +from pagereconstruct.composition.intrablock_composer import (
     2 +    compute_intrablock_line_layouts,
     3 +    verify_intrablock_elements_after_pagetranslate,
     4 +)
     5 +from tests.pagereconstruct._fixtures import translated_input_data
     6 +
     7 +
     8 +def test_verify_intrablock_elements_after_pagetranslate_complete():
     9 +    result = verify_intrablock_elements_after_pagetranslate(translated_i
        nput_data())
    10 +
    11 +    assert result["status"] == "ok"
    12 +    assert result["element_count"] >= 2
    13 +    assert "blk1" in result["covered_source_unit_ids"]
    14 +
    15 +
    16 +def test_verify_intrablock_elements_detects_missing_source_unit():
    17 +    data = translated_input_data()
    18 +    data["views"]["reconstruction_units"][0]["source_unit_ids"] = []
    19 +    data["views"]["reconstruction_units"][0]["unit_id"] = None
    20 +
    21 +    result = verify_intrablock_elements_after_pagetranslate(data)
    22 +
    23 +    assert result["status"] == "ko"
    24 +    assert "intrablock_missing_source_unit" in result["hard_blockers"]
    25 +
    26 +
    27 +def test_compute_intrablock_line_layouts_returns_line_bboxes():
    28 +    result = compute_intrablock_line_layouts(translated_input_data())
    29 +
    30 +    assert result["blocks"]
    31 +    assert result["blocks"][0]["lines"]
    32 +    assert result["blocks"][0]["lines"][0]["bbox"]

────────────────────────────────────────────────────────────────────────────────

⚠ Heads up, you have less than 10% of your weekly limit left. Run /status for a
  breakdown.

■ Conversation interrupted - tell the model what to do differently. Something
went wrong? Hit `/feedback` to report the issue.


› Est ce que ces codes n'existent pas encore ? fais d'abord ces vérifications
  avant d'en créer de nouveaux. Si ils existent, alors fais en usage, et
  améliore les ...
  PROTEGE MES TOKENS


• Tu as raison. Je vais d’abord vérifier l’existant précisément, puis je
  garderai seulement ce qui complète un trou réel. Si mes ajouts doublonnent une
  logique déjà présente, je les réduirai ou les raccorderai à l’existant au lieu
  de créer une voie parallèle.

• Ran git diff -- pagereconstruct/composition/intrablock_composer.py tests/
  │ pagereconstruct/test_intrablock_post_pagetranslate.py
  └ (no output)

• Explored
  └ Search intrablock|compose_block|compose_contract|LineLayout|
  TextRunPlacement|
           reconstruction_units|translation_unit_id|source_unit_ids|line_boxes|
           layout_bbox|reading_order in pagereconstruct

────────────────────────────────────────────────────────────────────────────────

• Vérification faite: la fonction 2 existe déjà (compose_block /
  compose_contract, avec measure_block). La fonction 1 existe partiellement via
  pagetranslate.functional_validator + pubready.stages.intrablock_audit; je ne
  vais donc pas créer un chemin parallèle. Je retire mes ajouts précédents et
  j’utilise l’existant pour générer les fichiers.

• Edited 2 files (+0 -149)
  └ pagereconstruct/composition/intrablock_composer.py (+0 -117)
    162      return [compose_block(b) for b in (getattr(contract, "blocks", [])
         or [])]
    163 -
    164 -
    165 -def verify_intrablock_elements_after_pagetranslate(translated_input_dat
         a: dict) -> dict:
    166 -    """Vérifie la complétude des éléments intra-block après PAGETRANSLA
         TE.
    167 -
    168 -    Couverture attendue:
    169 -    - chaque reconstruction_unit issue de pagetranslate doit référencer
          au moins
    170 -      une source unit et un texte cible/source;
    171 -    - chaque source unit consommée par une reconstruction_unit ne doit
         pas être
    172 -      rendue deux fois;
    173 -    - chaque élément préservé/exclu dans les plans spéciaux doit garder
          une bbox.
    174 -
    175 -    La fonction ne compose pas; elle audite uniquement le contrat d'ent
         rée de la
    176 -    composition.
    177 -    """
    178 -    views = translated_input_data.get("views") or {}
    179 -    reconstruction_units = views.get("reconstruction_units") or []
    180 -    preservation_plan = views.get("preservation_plan") or []
    181 -    exclusion_plan = views.get("exclusion_plan") or []
    182 -    blockers: list[str] = []
    183 -    findings: list[dict] = []
    184 -    covered: dict[str, str] = {}
    185 -    elements: list[dict] = []
    186 -
    187 -    for unit in reconstruction_units:
    188 -        rid = unit.get("reconstruction_unit_id") or unit.get("unit_id")
          or unit.get("translation_unit_id")
    189 -        sids = list(unit.get("source_unit_ids") or ([unit.get("unit_id"
         )] if unit.get("unit_id") else []))
    190 -        text = (unit.get("translated_text") or unit.get("text") or "").
         strip()
    191 -        bbox = unit.get("layout_bbox") or unit.get("bbox") or (unit.get
         ("render_target") or {}).get("layout_bbox")
    192 -        if not sids:
    193 -            blockers.append("intrablock_missing_source_unit")
    194 -            findings.append({"type": "intrablock_missing_source_unit",
         "element_id": rid})
    195 -        if not text:
    196 -            blockers.append("intrablock_missing_text")
    197 -            findings.append({"type": "intrablock_missing_text", "elemen
         t_id": rid})
    198 -        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
    199 -            blockers.append("intrablock_missing_bbox")
    200 -            findings.append({"type": "intrablock_missing_bbox", "elemen
         t_id": rid})
    201 -        for sid in sids:
    202 -            if sid in covered:
    203 -                blockers.append("intrablock_duplicate_source_unit")
    204 -                findings.append({"type": "intrablock_duplicate_source_u
         nit", "source_unit_id": sid,
    205 -                                 "first": covered[sid], "duplicate": ri
         d})
    206 -            covered[sid] = str(rid)
    207 -        elements.append({
    208 -            "element_id": rid,
    209 -            "kind": "translated_or_source_text",
    210 -            "source_unit_ids": sids,
    211 -            "translation_unit_id": unit.get("translation_unit_id"),
    212 -            "text": text,
    213 -            "bbox": list(bbox) if isinstance(bbox, (list, tuple)) and l
         en(bbox) == 4 else None,
    214 -            "role": unit.get("role"),
    215 -        })
    216 -
    217 -    for plan_name, entries, kind in (
    218 -        ("preservation_plan", preservation_plan, "special_preserved"),
    219 -        ("exclusion_plan", exclusion_plan, "special_excluded"),
    220 -    ):
    221 -        for entry in entries:
    222 -            eid = entry.get("preservation_id") or entry.get("exclusion_
         id") or entry.get("id")
    223 -            bbox = entry.get("bbox")
    224 -            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4)
         :
    225 -                blockers.append("intrablock_special_missing_bbox")
    226 -                findings.append({"type": "intrablock_special_missing_bb
         ox", "plan": plan_name, "element_id": eid})
    227 -            elements.append({
    228 -                "element_id": eid,
    229 -                "kind": kind,
    230 -                "source_unit_ids": list(entry.get("source_unit_ids") or
          []),
    231 -                "text": entry.get("text") or "",
    232 -                "bbox": list(bbox) if isinstance(bbox, (list, tuple)) a
         nd len(bbox) == 4 else None,
    233 -                "reason": entry.get("reason"),
    234 -            })
    235 -
    236 -    blockers = sorted(set(blockers))
    237 -    return {
    238 -        "status": "ko" if blockers else "ok",
    239 -        "hard_blockers": blockers,
    240 -        "findings": findings,
    241 -        "element_count": len(elements),
    242 -        "covered_source_unit_ids": sorted(covered),
    243 -        "elements": elements,
    244 -    }
    245 -
    246 -
    247 -def compute_intrablock_line_layouts(translated_input_data: dict, *, mod
         e: str = "debug") -> dict:
    248 -    """Calcule les lignes réorganisées dans les blocks après PAGETRANSL
         ATE.
    249 -
    250 -    Retourne les bboxes finales de lignes/runs en espace page pt. Les é
         léments
    251 -    source peuvent être traduits ou préservés: le calcul part du
    252 -    translated_input_data projeté par PAGETRANSLATE, puis compile le co
         ntrat
    253 -    pagereconstruct et compose chaque block.
    254 -    """
    255 -    completeness = verify_intrablock_elements_after_pagetranslate(trans
         lated_input_data)
    256 -    from pagereconstruct.input_adapter import PageReconstructInputAdapt
         er
    257 -    from pagereconstruct.final_contract import FinalReconstructionContr
         act
    258 -    from pagereconstruct.plan_compiler import compile_page_render_plan
    259 -
    260 -    normalized = PageReconstructInputAdapter().normalize(translated_inp
         ut_data)
    261 -    plan = compile_page_render_plan(translated_input_data, reconstructi
         on_mode=mode).to_dict()
    262 -    contract = FinalReconstructionContract.from_pageprint_pagetranslate
         (normalized, plan)
    263 -    compositions = [c.to_dict() for c in compose_contract(contract)]
    264 -    blockers = list(completeness.get("hard_blockers") or [])
    265 -    for comp in compositions:
    266 -        if comp.get("overflow"):
    267 -            blockers.append("intrablock_overflow")
    268 -        if comp.get("clipping"):
    269 -            blockers.append("intrablock_clipping")
    270 -        if comp.get("reading_order_ok") is False:
    271 -            blockers.append("reading_order_changed")
    272 -    blockers = sorted(set(blockers))
    273 -    return {
    274 -        "status": "ko" if blockers else "ok",
    275 -        "hard_blockers": blockers,
    276 -        "completeness": completeness,
    277 -        "blocks": compositions,
    278 -        "plan": plan,
    279 -    }

  └ tests/pagereconstruct/test_intrablock_post_pagetranslate.py (+0 -32)
     1 -from pagereconstruct.composition.intrablock_composer import (
     2 -    compute_intrablock_line_layouts,
     3 -    verify_intrablock_elements_after_pagetranslate,
     4 -)
     5 -from tests.pagereconstruct._fixtures import translated_input_data
     6 -
     7 -
     8 -def test_verify_intrablock_elements_after_pagetranslate_complete():
     9 -    result = verify_intrablock_elements_after_pagetranslate(translated_i
        nput_data())
    10 -
    11 -    assert result["status"] == "ok"
    12 -    assert result["element_count"] >= 2
    13 -    assert "blk1" in result["covered_source_unit_ids"]
    14 -
    15 -
    16 -def test_verify_intrablock_elements_detects_missing_source_unit():
    17 -    data = translated_input_data()
    18 -    data["views"]["reconstruction_units"][0]["source_unit_ids"] = []
    19 -    data["views"]["reconstruction_units"][0]["unit_id"] = None
    20 -
    21 -    result = verify_intrablock_elements_after_pagetranslate(data)
    22 -
    23 -    assert result["status"] == "ko"
    24 -    assert "intrablock_missing_source_unit" in result["hard_blockers"]
    25 -
    26 -
    27 -def test_compute_intrablock_line_layouts_returns_line_bboxes():
    28 -    result = compute_intrablock_line_layouts(translated_input_data())
    29 -
    30 -    assert result["blocks"]
    31 -    assert result["blocks"][0]["lines"]
    32 -    assert result["blocks"][0]["lines"][0]["bbox"]

• Explored
  └ Read __init__.py
    Search def build_page_translation in builder.py
    Read builder.py

────────────────────────────────────────────────────────────────────────────────

• Je vais maintenant générer les overlays en utilisant le chemin existant:
  PipelineOrchestrator pour PAGEPRINT, build_page_translation(..., dry_run=True)
  pour PAGETRANSLATE/projection, compile_page_render_plan puis
  plan.intrablock_compositions pour les lignes calculées.

────────────────────────────────────────────────────────────────────────────────

• Le script est en cours; il relance l’extraction PAGEPRINT sur les 10 PDFs,
  donc c’est plus long que le simple dessin direct. Je récupère la sortie dès
  qu’il termine.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json, re
from PIL import Image, ImageDraw
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
from pagereconstruct import compile_page_render_plan
from pubready.stages import intrablock_audit
batch = Path('results/new10_background_textures_20260616_batch2')
summary = json.loads((batch / 'summary.json').read_text())
out = batch / 'block_and_pagetranslate_line_bboxes'
render_dir = out / '_render'
out.mkdir(parents=True, exist_ok=True)
render_dir.mkdir(parents=True, exist_ok=True)
orchestrator = PipelineOrchestrator(
    enable_ocr=False,
    enable_understanding=True,
    enable_postprocessors=True,
    enable_special_regions=True,
    save_render_dir=str(render_dir),
)
items=[]
for item in summary:
    page_num = int(item['page'])
    pdf = Path(item['pdf'])
    clean_path = Path(item['clean_background'])
    tag = f'test_docintelligence_p{page_num:04d}'
    doc = orchestrator.run(str(pdf), pages='1', language={'source_lang': 'en',
'target_lang': 'fr'})
    ok_pages = [p for p in (doc.get('pages') or []) if p.get('status') == 'ok']
    if not ok_pages:
        items.append({'page': page_num, 'error': 'pageprint_failed'})
        continue
    input_data = ok_pages[0]['input_data']
    trial = build_page_translation(input_data, target_lang='fr',
source_lang='en', dry_run=True, allow_fallback=True)
    tid = trial['translated_input_data']
    plan = compile_page_render_plan(tid).to_dict()
    stage = intrablock_audit.audit_page(plan, {'units': input_data.get('units')
or []})
    img = Image.open(clean_path).convert('RGB')
    draw = ImageDraw.Draw(img, 'RGBA')
    page = plan.get('page') or {}
    wpt = float(page.get('width_pt') or ((input_data.get('page') or
{}).get('geometry') or {}).get('width') or 1)
    hpt = float(page.get('height_pt') or ((input_data.get('page') or
{}).get('geometry') or {}).get('height') or 1)
    sx = img.width / wpt
    sy = img.height / hpt
    block_count = 0
    line_count = 0
    # Blocks from existing plan translated_text layer: post-pagetranslate
reconstruction blocks.
    for block in (plan.get('layers') or {}).get('translated_text') or []:
        bbox = block.get('layout_bbox') or block.get('bbox')
        if isinstance(bbox, list) and len(bbox) == 4:
            xy = [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]
            draw.rectangle(xy, outline=(230, 0, 0, 255), width=4)
            block_count += 1
    # Lines from existing IntraBlockComposition: calculated/reflowed inside
blocks.
    for comp in plan.get('intrablock_compositions') or []:
        for line in comp.get('lines') or []:
            bbox = line.get('bbox')
            if isinstance(bbox, list) and len(bbox) == 4:
                xy = [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]
                draw.rectangle(xy, outline=(0, 90, 255, 255), width=2)
                line_count += 1
    out_path = out /
f'clean_background_blocks_lines_after_pagetranslate_p{page_num:04d}.png'
    img.save(out_path)
    (out / f'pagetranslate_compiled_plan_{tag}.json').write_text(json.dumps({
        'pagetranslate_statuses': {k: trial.get(k) for k in
('pipeline_status','translation_runtime_status','linguistic_quality_status','pub
lication_readiness_status')},
        'translation_unit_count': len(trial.get('translation_units') or []),
        'intrablock_audit': stage.to_dict(),
        'layers_translated_text': (plan.get('layers') or
{}).get('translated_text') or [],
        'intrablock_compositions': plan.get('intrablock_compositions') or [],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    items.append({
        'page': page_num,
        'file': str(out_path),
        'block_count': block_count,
        'line_count': line_count,
        'translation_unit_count': len(trial.get('translation_units') or []),
        'intrablock_status': stage.status,
        'intrablock_score': stage.score,
        'hard_blockers': list(stage.hard_blockers),
    })
# Contact sheet.
tile_w, tile_h = 340, 250
sheet = Image.new('RGB', (5*tile_w, 2*tile_h), 'white')
for idx, it in enumerate(items):
    if not it.get('file'):
        continue
    im = Image.open(it['file']).convert('RGB')
    im.thumbnail((tile_w, tile_h), Image.LANCZOS)
    x = (idx % 5) * tile_w + (tile_w - im.width)//2
    y = (idx // 5) * tile_h + (tile_h - im.height)//2
    sheet.paste(im, (x, y))
contact = out / 'contact_sheet_blocks_lines_after_pagetranslate.png'
sheet.save(contact)
(out / 'summary.json').write_text(json.dumps(items, ensure_ascii=False,
indent=2), encoding='utf-8')
print('OUT', out)
print('CONTACT', contact)
for it in items:
    if it.get('error'):
        print(f"p{it['page']:04d}: ERROR {it['error']}")
    else:
        print(f"p{it['page']:04d}: blocks={it['block_count']}
lines={it['line_count']} tu={it['translation_unit_count']}
intrablock={it['intrablock_status']} -> {it['file']}")
PY

────────────────────────────────────────────────────────────────────────────────

• Toujours en cours, probablement dans le passage pageprint/pagetranslate. Je
  continue à attendre, pas de nouveau code lancé en parallèle.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json, re
from PIL import Image, ImageDraw
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
from pagereconstruct import compile_page_render_plan
from pubready.stages import intrablock_audit
batch = Path('results/new10_background_textures_20260616_batch2')
summary = json.loads((batch / 'summary.json').read_text())
out = batch / 'block_and_pagetranslate_line_bboxes'
render_dir = out / '_render'
out.mkdir(parents=True, exist_ok=True)
render_dir.mkdir(parents=True, exist_ok=True)
orchestrator = PipelineOrchestrator(
    enable_ocr=False,
    enable_understanding=True,
    enable_postprocessors=True,
    enable_special_regions=True,
    save_render_dir=str(render_dir),
)
items=[]
for item in summary:
    page_num = int(item['page'])
    pdf = Path(item['pdf'])
    clean_path = Path(item['clean_background'])
    tag = f'test_docintelligence_p{page_num:04d}'
    doc = orchestrator.run(str(pdf), pages='1', language={'source_lang': 'en',
'target_lang': 'fr'})
    ok_pages = [p for p in (doc.get('pages') or []) if p.get('status') == 'ok']
    if not ok_pages:
        items.append({'page': page_num, 'error': 'pageprint_failed'})
        continue
    input_data = ok_pages[0]['input_data']
    trial = build_page_translation(input_data, target_lang='fr',
source_lang='en', dry_run=True, allow_fallback=True)
    tid = trial['translated_input_data']
    plan = compile_page_render_plan(tid).to_dict()
    stage = intrablock_audit.audit_page(plan, {'units': input_data.get('units')
or []})
    img = Image.open(clean_path).convert('RGB')
    draw = ImageDraw.Draw(img, 'RGBA')
    page = plan.get('page') or {}
    wpt = float(page.get('width_pt') or ((input_data.get('page') or
{}).get('geometry') or {}).get('width') or 1)
    hpt = float(page.get('height_pt') or ((input_data.get('page') or
{}).get('geometry') or {}).get('height') or 1)
    sx = img.width / wpt
    sy = img.height / hpt
    block_count = 0
    line_count = 0
    # Blocks from existing plan translated_text layer: post-pagetranslate
reconstruction blocks.
    for block in (plan.get('layers') or {}).get('translated_text') or []:
        bbox = block.get('layout_bbox') or block.get('bbox')
        if isinstance(bbox, list) and len(bbox) == 4:
            xy = [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]
            draw.rectangle(xy, outline=(230, 0, 0, 255), width=4)
            block_count += 1
    # Lines from existing IntraBlockComposition: calculated/reflowed inside
blocks.
    for comp in plan.get('intrablock_compositions') or []:
        for line in comp.get('lines') or []:
            bbox = line.get('bbox')
            if isinstance(bbox, list) and len(bbox) == 4:
                xy = [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]
                draw.rectangle(xy, outline=(0, 90, 255, 255), width=2)
                line_count += 1
    out_path = out /
f'clean_background_blocks_lines_after_pagetranslate_p{page_num:04d}.png'
    img.save(out_path)
    (out / f'pagetranslate_compiled_plan_{tag}.json').write_text(json.dumps({
        'pagetranslate_statuses': {k: trial.get(k) for k in
('pipeline_status','translation_runtime_status','linguistic_quality_status','pub
lication_readiness_status')},
        'translation_unit_count': len(trial.get('translation_units') or []),
        'intrablock_audit': stage.to_dict(),
        'layers_translated_text': (plan.get('layers') or
{}).get('translated_text') or [],
        'intrablock_compositions': plan.get('intrablock_compositions') or [],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    items.append({
        'page': page_num,
        'file': str(out_path),
        'block_count': block_count,
        'line_count': line_count,
        'translation_unit_count': len(trial.get('translation_units') or []),
        'intrablock_status': stage.status,
        'intrablock_score': stage.score,
        'hard_blockers': list(stage.hard_blockers),
    })
# Contact sheet.
tile_w, tile_h = 340, 250
sheet = Image.new('RGB', (5*tile_w, 2*tile_h), 'white')
for idx, it in enumerate(items):
    if not it.get('file'):
        continue
    im = Image.open(it['file']).convert('RGB')
    im.thumbnail((tile_w, tile_h), Image.LANCZOS)
    x = (idx % 5) * tile_w + (tile_w - im.width)//2
    y = (idx // 5) * tile_h + (tile_h - im.height)//2
    sheet.paste(im, (x, y))
contact = out / 'contact_sheet_blocks_lines_after_pagetranslate.png'
sheet.save(contact)
(out / 'summary.json').write_text(json.dumps(items, ensure_ascii=False,
indent=2), encoding='utf-8')
print('OUT', out)
print('CONTACT', contact)
for it in items:
    if it.get('error'):
        print(f"p{it['page']:04d}: ERROR {it['error']}")
    else:
        print(f"p{it['page']:04d}: blocks={it['block_count']}
lines={it['line_count']} tu={it['translation_unit_count']}
intrablock={it['intrablock_status']} -> {it['file']}")
PY

────────────────────────────────────────────────────────────────────────────────

• Le job ne sort pas encore de logs intermédiaires. Il peut traiter les 10 pages
  séquentiellement; je le laisse finir pour éviter des artefacts incomplets.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json, re
from PIL import Image, ImageDraw
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
from pagereconstruct import compile_page_render_plan
from pubready.stages import intrablock_audit
batch = Path('results/new10_background_textures_20260616_batch2')
summary = json.loads((batch / 'summary.json').read_text())
out = batch / 'block_and_pagetranslate_line_bboxes'
render_dir = out / '_render'
out.mkdir(parents=True, exist_ok=True)
render_dir.mkdir(parents=True, exist_ok=True)
orchestrator = PipelineOrchestrator(
    enable_ocr=False,
    enable_understanding=True,
    enable_postprocessors=True,
    enable_special_regions=True,
    save_render_dir=str(render_dir),
)
items=[]
for item in summary:
    page_num = int(item['page'])
    pdf = Path(item['pdf'])
    clean_path = Path(item['clean_background'])
    tag = f'test_docintelligence_p{page_num:04d}'
    doc = orchestrator.run(str(pdf), pages='1', language={'source_lang': 'en',
'target_lang': 'fr'})
    ok_pages = [p for p in (doc.get('pages') or []) if p.get('status') == 'ok']
    if not ok_pages:
        items.append({'page': page_num, 'error': 'pageprint_failed'})
        continue
    input_data = ok_pages[0]['input_data']
    trial = build_page_translation(input_data, target_lang='fr',
source_lang='en', dry_run=True, allow_fallback=True)
    tid = trial['translated_input_data']
    plan = compile_page_render_plan(tid).to_dict()
    stage = intrablock_audit.audit_page(plan, {'units': input_data.get('units')
or []})
    img = Image.open(clean_path).convert('RGB')
    draw = ImageDraw.Draw(img, 'RGBA')
    page = plan.get('page') or {}
    wpt = float(page.get('width_pt') or ((input_data.get('page') or
{}).get('geometry') or {}).get('width') or 1)
    hpt = float(page.get('height_pt') or ((input_data.get('page') or
{}).get('geometry') or {}).get('height') or 1)
    sx = img.width / wpt
    sy = img.height / hpt
    block_count = 0
    line_count = 0
    # Blocks from existing plan translated_text layer: post-pagetranslate
reconstruction blocks.
    for block in (plan.get('layers') or {}).get('translated_text') or []:
        bbox = block.get('layout_bbox') or block.get('bbox')
        if isinstance(bbox, list) and len(bbox) == 4:
            xy = [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]
            draw.rectangle(xy, outline=(230, 0, 0, 255), width=4)
            block_count += 1
    # Lines from existing IntraBlockComposition: calculated/reflowed inside
blocks.
    for comp in plan.get('intrablock_compositions') or []:
        for line in comp.get('lines') or []:
            bbox = line.get('bbox')
            if isinstance(bbox, list) and len(bbox) == 4:
                xy = [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]
                draw.rectangle(xy, outline=(0, 90, 255, 255), width=2)
                line_count += 1
    out_path = out /
f'clean_background_blocks_lines_after_pagetranslate_p{page_num:04d}.png'
    img.save(out_path)
    (out / f'pagetranslate_compiled_plan_{tag}.json').write_text(json.dumps({
        'pagetranslate_statuses': {k: trial.get(k) for k in
('pipeline_status','translation_runtime_status','linguistic_quality_status','pub
lication_readiness_status')},
        'translation_unit_count': len(trial.get('translation_units') or []),
        'intrablock_audit': stage.to_dict(),
        'layers_translated_text': (plan.get('layers') or
{}).get('translated_text') or [],
        'intrablock_compositions': plan.get('intrablock_compositions') or [],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    items.append({
        'page': page_num,
        'file': str(out_path),
        'block_count': block_count,
        'line_count': line_count,
        'translation_unit_count': len(trial.get('translation_units') or []),
        'intrablock_status': stage.status,
        'intrablock_score': stage.score,
        'hard_blockers': list(stage.hard_blockers),
    })
# Contact sheet.
tile_w, tile_h = 340, 250
sheet = Image.new('RGB', (5*tile_w, 2*tile_h), 'white')
for idx, it in enumerate(items):
    if not it.get('file'):
        continue
    im = Image.open(it['file']).convert('RGB')
    im.thumbnail((tile_w, tile_h), Image.LANCZOS)
    x = (idx % 5) * tile_w + (tile_w - im.width)//2
    y = (idx // 5) * tile_h + (tile_h - im.height)//2
    sheet.paste(im, (x, y))
contact = out / 'contact_sheet_blocks_lines_after_pagetranslate.png'
sheet.save(contact)
(out / 'summary.json').write_text(json.dumps(items, ensure_ascii=False,
indent=2), encoding='utf-8')
print('OUT', out)
print('CONTACT', contact)
for it in items:
    if it.get('error'):
        print(f"p{it['page']:04d}: ERROR {it['error']}")
    else:
        print(f"p{it['page']:04d}: blocks={it['block_count']}
lines={it['line_count']} tu={it['translation_unit_count']}
intrablock={it['intrablock_status']} -> {it['file']}")
PY

────────────────────────────────────────────────────────────────────────────────

• Ça prend plus que prévu. Je ne relance rien; j’attends encore une tranche,
  puis si nécessaire je basculerai sur les artefacts déjà produits dans _render
  pour ne pas perdre de temps.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json, re
from PIL import Image, ImageDraw
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
from pagereconstruct import compile_page_render_plan
from pubready.stages import intrablock_audit
batch = Path('results/new10_background_textures_20260616_batch2')
summary = json.loads((batch / 'summary.json').read_text())
out = batch / 'block_and_pagetranslate_line_bboxes'
render_dir = out / '_render'
out.mkdir(parents=True, exist_ok=True)
render_dir.mkdir(parents=True, exist_ok=True)
orchestrator = PipelineOrchestrator(
    enable_ocr=False,
    enable_understanding=True,
    enable_postprocessors=True,
    enable_special_regions=True,
    save_render_dir=str(render_dir),
)
items=[]
for item in summary:
    page_num = int(item['page'])
    pdf = Path(item['pdf'])
    clean_path = Path(item['clean_background'])
    tag = f'test_docintelligence_p{page_num:04d}'
    doc = orchestrator.run(str(pdf), pages='1', language={'source_lang': 'en',
'target_lang': 'fr'})
    ok_pages = [p for p in (doc.get('pages') or []) if p.get('status') == 'ok']
    if not ok_pages:
        items.append({'page': page_num, 'error': 'pageprint_failed'})
        continue
    input_data = ok_pages[0]['input_data']
    trial = build_page_translation(input_data, target_lang='fr',
source_lang='en', dry_run=True, allow_fallback=True)
    tid = trial['translated_input_data']
    plan = compile_page_render_plan(tid).to_dict()
    stage = intrablock_audit.audit_page(plan, {'units': input_data.get('units')
or []})
    img = Image.open(clean_path).convert('RGB')
    draw = ImageDraw.Draw(img, 'RGBA')
    page = plan.get('page') or {}
    wpt = float(page.get('width_pt') or ((input_data.get('page') or
{}).get('geometry') or {}).get('width') or 1)
    hpt = float(page.get('height_pt') or ((input_data.get('page') or
{}).get('geometry') or {}).get('height') or 1)
    sx = img.width / wpt
    sy = img.height / hpt
    block_count = 0
    line_count = 0
    # Blocks from existing plan translated_text layer: post-pagetranslate
reconstruction blocks.
    for block in (plan.get('layers') or {}).get('translated_text') or []:
        bbox = block.get('layout_bbox') or block.get('bbox')
        if isinstance(bbox, list) and len(bbox) == 4:
            xy = [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]
            draw.rectangle(xy, outline=(230, 0, 0, 255), width=4)
            block_count += 1
    # Lines from existing IntraBlockComposition: calculated/reflowed inside
blocks.
    for comp in plan.get('intrablock_compositions') or []:
        for line in comp.get('lines') or []:
            bbox = line.get('bbox')
            if isinstance(bbox, list) and len(bbox) == 4:
                xy = [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]
                draw.rectangle(xy, outline=(0, 90, 255, 255), width=2)
                line_count += 1
    out_path = out /
f'clean_background_blocks_lines_after_pagetranslate_p{page_num:04d}.png'
    img.save(out_path)
    (out / f'pagetranslate_compiled_plan_{tag}.json').write_text(json.dumps({
        'pagetranslate_statuses': {k: trial.get(k) for k in
('pipeline_status','translation_runtime_status','linguistic_quality_status','pub
lication_readiness_status')},
        'translation_unit_count': len(trial.get('translation_units') or []),
        'intrablock_audit': stage.to_dict(),
        'layers_translated_text': (plan.get('layers') or
{}).get('translated_text') or [],
        'intrablock_compositions': plan.get('intrablock_compositions') or [],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    items.append({
        'page': page_num,
        'file': str(out_path),
        'block_count': block_count,
        'line_count': line_count,
        'translation_unit_count': len(trial.get('translation_units') or []),
        'intrablock_status': stage.status,
        'intrablock_score': stage.score,
        'hard_blockers': list(stage.hard_blockers),
    })
# Contact sheet.
tile_w, tile_h = 340, 250
sheet = Image.new('RGB', (5*tile_w, 2*tile_h), 'white')
for idx, it in enumerate(items):
    if not it.get('file'):
        continue
    im = Image.open(it['file']).convert('RGB')
    im.thumbnail((tile_w, tile_h), Image.LANCZOS)
    x = (idx % 5) * tile_w + (tile_w - im.width)//2
    y = (idx // 5) * tile_h + (tile_h - im.height)//2
    sheet.paste(im, (x, y))
contact = out / 'contact_sheet_blocks_lines_after_pagetranslate.png'
sheet.save(contact)
(out / 'summary.json').write_text(json.dumps(items, ensure_ascii=False,
indent=2), encoding='utf-8')
print('OUT', out)
print('CONTACT', contact)
for it in items:
    if it.get('error'):
        print(f"p{it['page']:04d}: ERROR {it['error']}")
    else:
        print(f"p{it['page']:04d}: blocks={it['block_count']}
lines={it['line_count']} tu={it['translation_unit_count']}
intrablock={it['intrablock_status']} -> {it['file']}")
PY

────────────────────────────────────────────────────────────────────────────────

• Le run complet est trop lent pour cette demande visuelle. Je l’arrête et je
  vais utiliser la sortie PAGEPRINT/PDF déjà disponible pour calculer les mêmes
  lignes via les fonctions existantes (compose_block/measure_block) sans
  repasser toute l’orchestration.

• Ran .docs-parser/bin/python - <<'PY'
  │ from pathlib import Path
  │ import json, re
  │ … +126 lines
  └ /home/raphael/Mes_Projets/docs_parser/.docs-parser/lib/python3.11/
    site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3
    … +34 lines (ctrl + t to view transcript)
    block_and_pagetranslate_line_bboxes/
    clean_background_blocks_lines_after_pagetranslate_p0414.png

────────────────────────────────────────────────────────────────────────────────

• Le job a finalement terminé proprement. Je vérifie la planche contact:
  rectangles rouges = blocks post-pagetranslate, rectangles bleus = lignes
  calculées par composition intra-block.

• Viewed Image
  └ results/new10_background_textures_20260616_batch2/
  block_and_pagetranslate_line_bboxes/
  contact_sheet_blocks_lines_after_pagetranslate.png

────────────────────────────────────────────────────────────────────────────────

• J’ai vérifié l’existant avant de continuer: la composition intra-block
  existait déjà via compose_block / compose_contract, donc je l’ai utilisée au
  lieu de créer une nouvelle voie.

  Fichiers générés sur les 10 pages du dernier lot:

  Planche contact: results/new10_background_textures_20260616_batch2/
  block_and_pagetranslate_line_bboxes/
  contact_sheet_blocks_lines_after_pagetranslate.png

  Dossier complet: results/new10_background_textures_20260616_batch2/
  block_and_pagetranslate_line_bboxes

  Code couleur:

  - rouge: bboxes des blocks après pagetranslate
  - bleu: bboxes des lignes recalculées dans les blocks par composition intra-
    block

  Résumé: les 10 pages sont passées en intrablock=ok; détails dans results/
  new10_background_textures_20260616_batch2/block_and_pagetranslate_line_bboxes/
  summary.json.

─ Worked for 5m 22s ────────────────────────────────────────────────────────────


› Pour ces mêmes page, fais le bboxes de sblocks, et à côté fais le remplissage
  de ces bboxes on va comparer


• Je vais d’abord vérifier si sblock existe comme contrat/champ dans le code ou
  les artefacts. Ensuite je générerai, sur les mêmes 10 pages, une comparaison
  côte à côte: bboxes seules vs bboxes remplies.

• Explored
  └ List block_and_pagetranslate_line_bboxes
    Read summary.json
    Search \bsblocks?\b|sub[_-]?blocks?|semantic_blocks?|sblock in .

■ Conversation interrupted - tell the model what to do differently. Something
went wrong? Hit `/feedback` to report the issue.


› Pour ces mêmes page, fais le bboxes des blocks, et à côté fais le remplissage
  de ces bboxes on va comparer


• Je vais utiliser les mêmes 10 pages et les blocks déjà compilés après
  pagetranslate. Je génère pour chaque page une image côte à côte: gauche =
  bboxes seules, droite = mêmes bboxes remplies.

■ Conversation interrupted - tell the model what to do differently. Something
went wrong? Hit `/feedback` to report the issue.


› Non. pas ça! pas les blocks compilé après page translate, les blocks
  originaux. c'est les bboxes intra-blocks (phrases, textes, expressions, mots,
  zones spéciales non traduisibles ...) qui sont celle après pagetranslate/
  Je veux voir si il y a une différence entre avant et après


• Compris. Je garde les blocks originaux comme cadre fixe. Je vais comparer
  dedans: à gauche les éléments intra-block avant pagetranslate, à droite les
  éléments intra-block après pagetranslate/composition, sans remplacer les
  blocks par ceux compilés.

• Ran python3 - <<'PY'
  │ import json
  │ from pathlib import Path
  │ … +10 lines
  └ dict_keys(['pagetranslate_statuses', 'translation_unit_count',
    'intrablock_audit', 'layers_translated_text', 'intrablock_compositions'])
    … +60 lines (ctrl + t to view transcript)
        }
      ]

• Explored
  └ List block_and_pagetranslate_line_bboxes

────────────────────────────────────────────────────────────────────────────────

• Je vais produire une comparaison claire: chaque image aura deux panneaux.
  Gauche = avant pagetranslate dans les blocks originaux; droite = après
  pagetranslate/composition, toujours dans les mêmes blocks originaux. Les
  blocks originaux restent les cadres rouges.

■ Conversation interrupted - tell the model what to do differently. Something
went wrong? Hit `/feedback` to report the issue.


› Non. Ne compares pas. rends moi juste les pages.


• Je génère uniquement les pages finales, sans comparaison côte à côte: fond
  propre + blocks originaux en rouge + bboxes intra-block après pagetranslate en
  bleu.

• Ran .docs-parser/bin/python - <<'PY'
  │ from pathlib import Path
  │ import json, re
  │ … +78 lines
  └ OUT results/new10_background_textures_20260616_batch2/
    original_blocks_with_after_pagetranslate_intrablocks
    … +35 lines (ctrl + t to view transcript)
    original_blocks_with_after_pagetranslate_intrablocks/
    page_original_blocks_after_pagetranslate_intrablocks_p0414.png

────────────────────────────────────────────────────────────────────────────────

• Les pages sont générées sans comparaison. Je vérifie rapidement la planche
  contact avant de te donner les fichiers.

• Viewed Image
  └ results/new10_background_textures_20260616_batch2/
  original_blocks_with_after_pagetranslate_intrablocks/
  contact_sheet_pages_original_blocks_after_pagetranslate_intrablocks.png

────────────────────────────────────────────────────────────────────────────────

• Fait. Pas de comparaison côte à côte, uniquement les pages.

  Planche contact: results/new10_background_textures_20260616_batch2/
  original_blocks_with_after_pagetranslate_intrablocks/
  contact_sheet_pages_original_blocks_after_pagetranslate_intrablocks.png

  Dossier avec les 10 pages: results/new10_background_textures_20260616_batch2/
  original_blocks_with_after_pagetranslate_intrablocks

  Code couleur:

  - rouge: blocks originaux
  - bleu: bboxes intra-block après pagetranslate / composition

────────────────────────────────────────────────────────────────────────────────


› On constate qu'il y a de blocks qui ne sont pas rempls, pratiquement
  plusieurs. Il s'agit de quoi ?


• Je vais vérifier objectivement les blocks rouges non couverts: pour chaque
  page, je compare les bboxes originales avec les compositions bleues et
  j’extrais le texte/role probable des blocks vides.

• Ran .docs-parser/bin/python - <<'PY'
  │ from pathlib import Path
  │ import json, re
  │ … +41 lines
  └
    p0077: empty 14/20
    … +51 lines (ctrl + t to view transcript)
    by an artist engineer, can be used to create beauti- | ful art on par with
    that pr'

■ Conversation interrupted - tell the model what to do differently. Something
went wrong? Hit `/feedback` to report the issue.


› Non. Regardes dans les blocks originaux(avant pagetranslate/), et regardes au
  niveau de ces positions manquantes, les éléments qui ne sont pas présents
  après pagetranslate/. C'est tout.
  instruction : Protèges mes tokens, pas trop de verbiage, vas à l'essentiel


• Je vais comparer uniquement les blocks originaux avec les source_unit_ids
  conservés après pagetranslate, puis sortir une liste courte des éléments
  originaux absents.

• Ran .docs-parser/bin/python - <<'PY'
  │ from pathlib import Path
  │ import json, re
  │ … +47 lines
  └ p0033: missing 3/6
      p001_block_002 [57.1, 315.8, 245.9, 326.3] '1.2.2 | Object detection and
    … +74 lines (ctrl + t to view transcript)
    JSON results/new10_background_textures_20260616_batch2/
    missing_original_blocks_after_pagetranslate.json

────────────────────────────────────────────────────────────────────────────────

• Voici les éléments des blocks originaux qui ne sont pas présents après
  pagetranslate/.

  Fichier complet: results/new10_background_textures_20260616_batch2/
  missing_original_blocks_after_pagetranslate.json

  Constat rapide:

  - p0033: 3/6 blocks manquants
  - p0077: 19/20
  - p0219: 22/23
  - p0252: 11/13
  - p0264: 4/6
  - p0279: 15/16
  - p0315: 10/11
  - p0324: 6/7
  - p0358: 3/4
  - p0414: 9/11

  Exemples typiques manquants:

  - paragraphes de corps complets
  - titres/sections
  - captions de figures
  - formules/expressions
  - labels de schémas/axes
  - notes/listes à puces

  Donc oui: après pagetranslate, beaucoup de blocks originaux ne sont pas
  projetés comme unités intra-block présentes.

────────────────────────────────────────────────────────────────────────────────


› Donc oui: après pagetranslate, beaucoup de blocks originaux ne sont pas
    projetés comme unités intra-block présentes.

  EST CE NORMAL ? POURQUOI ? QUELLE CODE (FONCTION) GERE ÇA? COMMENT CORRIGER ?
  COMMENT S'ASSURER QUE CE SERA TOUJOURS LE CAS ?


• Je vais vérifier les fonctions exactes dans pagetranslate/pagereconstruct
  avant de répondre, parce que le point clé est de distinguer “block original
  absent” de “block consommé par une unité sémantique/phrase projetée
  autrement”.

• Explored
  └ Read input_adapter.py, plan_compiler.py, intrablock_audit.py, selector.py,
         projection.py

■ Conversation interrupted - tell the model what to do differently. Something
went wrong? Hit `/feedback` to report the issue.


› Non. réponds simplement à ma question. Ne divague pas !


• Non, ce n’est pas normal si ces blocks contiennent du texte traduisible.

  C’est normal seulement pour:

  - formules,
  - numéros de page,
  - références,
  - URLs/DOI,
  - zones protégées,
  - objets visuels,
  - textes explicitement background_only / exact_preserve.

  Pourquoi ça arrive
  Le pipeline pagetranslate ne garantit pas aujourd’hui que chaque block
  original soit représenté après traduction. Il sélectionne des unités selon une
  priorité:

  semantic_phrase > semantic_group > phrase > line > block

  Donc certains blocks originaux sont:

  - remplacés par des phrases/lignes enfants,
  - ignorés car classés non traduisibles,
  - exclus par heuristique,
  - consommés par une unité sémantique,
  - ou perdus si la projection ne rattache pas correctement l’unité au block
    parent.

  Code qui gère ça
  Principalement:

  - sélection avant traduction: pagetranslate/selector.py
      - select_translation_units
      - _select_semantic_system_units
      - _select_pageprint_units_by_block
      - _is_excluded_unit

  - projection après traduction: pagetranslate/projection.py
      - project_translations
      - _reconstruction_units
      - _semantic_reconstruction_unit
      - _direct_reconstruction_unit

  - consommation finale côté reconstruction: pagereconstruct/plan_compiler.py
      - boucle sur translated_units
      - consumed_source_unit_ids
      - duplicate_render_skipped
      - unit_inside_special_zone

  Comment corriger
  Ajouter un audit contractuel juste après pagetranslate:

  Pour chaque block original:

  - s’il est traduisible, il doit être couvert par au moins une unité projetée;
  - si non traduisible, il doit avoir une raison explicite: preserve_exact,
    background_only, formula, page_number, etc.;

  - si couvert par phrase/line/semantic_phrase, le block doit porter une
    relation claire: covered_by_translation_unit_id.

  Puis corriger projection.py pour toujours remonter la couverture au block
  parent.

  Comment s’assurer que ça reste vrai
  Créer un test invariant:

  original_translatable_block_must_be_covered_after_pagetranslate

  Il doit échouer si un block original traduisible n’est ni:

  - traduit,
  - couvert par ses enfants,
  - préservé avec raison,
  - exclu avec raison.

  Donc: le bug est dans le contrat de couverture pagetranslate ->
  pagereconstruct, pas dans l’affichage des bboxes.


› On doit avoir 100% des éléments origanux doivent se retrouver dans le rendu
  après reconstruction, que ces élements soit traduits ou non !!!! 100% et rien
  ne doit manquer. Ce qui sont tradyit seront remplacé par leur traduction. les
  autres tels quels...


• Je vais implémenter ça comme invariant de contrat, pas comme ajustement
  visuel: tout élément original doit avoir un destin explicite après
  pagetranslate/reconstruction (translated, preserved, excluded_with_reason, ou
  covered_by_parent/child). Je commence par ajouter le test qui échoue sur les
  cas actuels, puis je corrige pagetranslate/projection.

• Explored
  └ Read test_projection_bboxes.py
    Search coverage|covered_by|source_units_consumed|skip_individual_render|
           reconstruction_units|preservation_plan|exclusion_plan in
           pagetranslate

────────────────────────────────────────────────────────────────────────────────

• Je vais le traiter comme invariant dur: chaque unité originale doit avoir un
  disposition explicite après pagetranslate, pas seulement les unités traduites.
  Je touche le validateur/projection, pas le rendu visuel.

• Explored
  └ Read functional_validator.py, view_compiler.py

────────────────────────────────────────────────────────────────────────────────

• Je vais ajouter un contrôle 100% disposition dans
  pagetranslate.functional_validator: il ne change pas la traduction, mais il
  fait échouer la sortie si une unité originale textuelle n’est ni traduite/
  couverte, ni préservée, ni exclue avec raison.

• Explored
  └ Read view_compiler.py

────────────────────────────────────────────────────────────────────────────────

• Le trou principal est clair: dans selector.py, dès qu’une unité sémantique
  touche un block, le block entier peut être sauté. Je corrige ça: on ne saute
  un block que si ses feuilles textuelles sont réellement toutes couvertes.

■ Conversation interrupted - tell the model what to do differently. Something
went wrong? Hit `/feedback` to report the issue.


› On ne saute RIEN ! On garde tout ! tout ! les traduit seront remplacés par
  leur traduction, les auttes restent tel quel!


• D’accord. Règle: aucune unité originale ne disparaît. Je vais corriger
  pagetranslate/projection.py pour ajouter une disposition explicite pour toutes
  les unités originales: traduite si sélectionnée, sinon préservée telle quelle.

• Explored
  └ Read projection.py

────────────────────────────────────────────────────────────────────────────────

• J’ajoute d’abord le test minimal: un block traduisible et un block formule/
  non-traduisible doivent tous deux exister dans reconstruction_units, l’un
  traduit, l’autre préservé.

• Added tests/pagetranslate/test_projection_preserves_all_originals.py (+46 -0)
     1 +from pagetranslate import build_page_translation
     2 +
     3 +
     4 +def test_all_original_units_get_reconstruction_disposition():
     5 +    input_data = {
     6 +        "schema_version": "pageprint.input.v1",
     7 +        "input_id": "all-originals",
     8 +        "document": {"language": {"source_lang": "en", "target_lang": "f
        r"}},
     9 +        "page": {},
    10 +        "translation_context": {},
    11 +        "units": [
    12 +            {
    13 +                "unit_id": "b1",
    14 +                "level": "block",
    15 +                "content": {"text": "This paragraph must be translated."
        },
    16 +                "geometry": {"bbox": [0, 0, 200, 40], "reading_order_ind
        ex": 1},
    17 +                "understanding": {"role": "body_paragraph", "object_type
        ": "natural_text"},
    18 +                "policy": {"translatable": True, "translation_strategy":
         "layout_constrained"},
    19 +                "visual": {"style": {}},
    20 +                "children_ids": [],
    21 +            },
    22 +            {
    23 +                "unit_id": "f1",
    24 +                "level": "block",
    25 +                "content": {"text": "z = beta0 + beta1 age"},
    26 +                "geometry": {"bbox": [0, 60, 200, 80], "reading_order_in
        dex": 2},
    27 +                "understanding": {"role": "formula_expression", "object_
        type": "formula"},
    28 +                "policy": {"translatable": False, "translation_strategy"
        : "exact_preserve"},
    29 +                "visual": {"style": {}},
    30 +                "children_ids": [],
    31 +            },
    32 +        ],
    33 +        "views": {},
    34 +    }
    35 +
    36 +    result = build_page_translation(input_data, dry_run=True)
    37 +    reconstruction_units = result["translated_input_data"]["views"]["rec
        onstruction_units"]
    38 +    by_source = {
    39 +        sid: unit
    40 +        for unit in reconstruction_units
    41 +        for sid in unit.get("source_unit_ids") or []
    42 +    }
    43 +
    44 +    assert set(by_source) == {"b1", "f1"}
    45 +    assert by_source["b1"]["render_contract"]["mode"] == "translated_tex
        t"
    46 +    assert by_source["f1"]["render_contract"]["mode"] == "preserve_origi
        nal"

────────────────────────────────────────────────────────────────────────────────

• Je lance ce test seul pour confirmer l’échec avant correction.

• Ran .docs-parser/bin/pytest tests/pagetranslate/
  │ test_projection_preserves_all_originals.py
  └ ============================= test session starts
    ==============================
    … +64 lines (ctrl + t to view transcript)
    ============================== 1 failed in 0.05s
    ===============================

────────────────────────────────────────────────────────────────────────────────

• Le test prouve le bug: f1 disparaît. Je corrige la projection pour ajouter
  automatiquement des unités preserve_original pour toute unité originale non
  couverte.

• Edited pagetranslate/projection.py (+73 -0)
    225          output.append(_direct_reconstruction_unit(unit, item, unit_map)
         )
    226 +    output.extend(_preserve_uncovered_original_units(translated_input,
         output, unit_map))
    227      return output
        ⋮
    229
    230 +def _preserve_uncovered_original_units(translated_input: dict, reconstr
         uction_units: list[dict], unit_map: dict[str, dict]) -> list[dict]:
    231 +    covered = {
    232 +        sid
    233 +        for item in reconstruction_units
    234 +        for sid in item.get("source_unit_ids") or []
    235 +    }
    236 +    out = []
    237 +    for unit in sorted(unit_map.values(), key=lambda u: (u.get("geometr
         y") or {}).get("reading_order_index") or 0):
    238 +        uid = unit.get("unit_id")
    239 +        if uid in covered:
    240 +            continue
    241 +        if unit.get("level") not in {"block", "line", "phrase", "span"}
         :
    242 +            continue
    243 +        text = normalize_spaces((unit.get("content") or {}).get("text")
         )
    244 +        bbox = (unit.get("geometry") or {}).get("bbox")
    245 +        if not text or not (isinstance(bbox, (list, tuple)) and len(bbo
         x) == 4):
    246 +            continue
    247 +        out.append(_preserved_reconstruction_unit(unit, unit_map))
    248 +    return out
    249 +
    250 +
    251 +def _preserved_reconstruction_unit(unit: dict, unit_map: dict[str, dict
         ]) -> dict:
    252 +    understanding = unit.get("understanding") or {}
    253 +    policy = unit.get("policy") or {}
    254 +    bbox = (unit.get("geometry") or {}).get("bbox")
    255 +    uid = unit.get("unit_id")
    256 +    text = (unit.get("content") or {}).get("text")
    257 +    reason = (
    258 +        policy.get("non_translatable_reason")
    259 +        or policy.get("preservation_reason")
    260 +        or policy.get("translation_strategy")
    261 +        or understanding.get("role")
    262 +        or "not_selected_for_translation"
    263 +    )
    264 +    return {
    265 +        "unit_id": uid,
    266 +        "translation_unit_id": None,
    267 +        "logical_unit_id": uid,
    268 +        "level": unit.get("level"),
    269 +        "render_level": unit.get("level"),
    270 +        "role": understanding.get("role"),
    271 +        "object_type": understanding.get("object_type") or policy.get("
         unit_type"),
    272 +        "semantic_kind": understanding.get("semantic_kind"),
    273 +        "page_role": understanding.get("page_role"),
    274 +        "preservation_mode": "preserve_original",
    275 +        "text": text,
    276 +        "translated_text": text,
    277 +        "bbox": bbox,
    278 +        "layout_bbox": bbox,
    279 +        "patch_bbox": None,
    280 +        "coverage_bbox": bbox,
    281 +        "anchor_bbox": bbox,
    282 +        "source_unit_ids": [uid],
    283 +        "consume_source_units": False,
    284 +        "source_units_consumed": False,
    285 +        "preferred_over_children": False,
    286 +        "skip_original_units": False,
    287 +        "render_as": unit.get("level"),
    288 +        "overflow_policy": "preserve_original",
    289 +        "line_break_policy": "source_layout",
    290 +        "layout_budget": _layout_budget(bbox),
    291 +        "style": _dominant_style(unit, unit_map),
    292 +        "render_target": {"bbox": bbox, "layout_bbox": bbox, "coverage_
         bbox": bbox, "style_source_unit_id": uid},
    293 +        "render_contract": {
    294 +            "mode": "preserve_original",
    295 +            "preservation_mode": "preserve_original",
    296 +            "reason": reason,
    297 +        },
    298 +        "translation": {"status": "preserved", "reason": reason},
    299 +    }
    300 +
    301 +
    302  def _direct_reconstruction_unit(unit: dict, item: dict, unit_map: dict[
         str, dict]) -> dict:

────────────────────────────────────────────────────────────────────────────────

• Je relance le test d’invariant, puis les tests pagetranslate proches.

• Ran .docs-parser/bin/pytest tests/pagetranslate/
  │ test_projection_preserves_all_originals.py tests/pagetranslate/
  │ test_projection_bboxes.py tests/pagetranslate/
  │ … +2 lines
  └ ============================= test session starts
    ==============================
    … +65 lines (ctrl + t to view transcript)
    ========================= 1 failed, 13 passed in 0.19s
    =========================

────────────────────────────────────────────────────────────────────────────────

• La couverture est maintenant OK, mais le contrat traduit ne marque pas
  explicitement mode=translated_text. J’ajoute ce marqueur pour que le statut
  soit non ambigu.

• Edited pagetranslate/projection.py (+5 -1)
    328          "render_target": render_target,
    329 -        "render_contract": unit.get("render_contract") or {},
    329 +        "render_contract": {
    330 +            **(unit.get("render_contract") or {}),
    331 +            "mode": "translated_text",
    332 +            "strategy": item.get("strategy"),
    333 +        },
    334          "translation": item,

────────────────────────────────────────────────────────────────────────────────

• Je relance le sous-ensemble.

• Ran .docs-parser/bin/pytest tests/pagetranslate/
  │ test_projection_preserves_all_originals.py tests/pagetranslate/
  │ test_projection_bboxes.py tests/pagetranslate/
  │ … +2 lines
  └ ============================= test session starts
    ==============================
    … +14 lines (ctrl + t to view transcript)
    ============================== 14 passed in 0.18s
    ==============================

────────────────────────────────────────────────────────────────────────────────

• Je vérifie maintenant rapidement sur les mêmes 10 pages que les blocks
  originaux ont bien une disposition après pagetranslate avec la correction.

────────────────────────────────────────────────────────────────────────────────

• La vérification réelle relance l’extraction des 10 pages, donc elle prend un
  peu. Je récupère seulement le résumé à la fin.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
batch=Path('results/new10_background_textures_20260616_batch2')
summary=json.loads((batch/'summary.json').read_text())
orch=PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(batch/'_verify_render'))
for item in summary[:10]:
    n=int(item['page']); pdf=Path(item['pdf'])
    doc=orch.run(str(pdf), pages='1',
language={'source_lang':'en','target_lang':'fr'})
    ok=[p for p in doc.get('pages') or [] if p.get('status')=='ok'][0]
    input_data=ok['input_data']
    res=build_page_translation(input_data, dry_run=True)
    rus=res['translated_input_data']['views']['reconstruction_units']
    covered={sid for ru in rus for sid in ru.get('source_unit_ids') or []}
    blocks=[u for u in input_data.get('units') or [] if u.get('level')=='block'
and ((u.get('content') or {}).get('text') or '').strip()]
    missing=[u.get('unit_id') for u in blocks if u.get('unit_id') not in
covered]
    print(f'p{n:04d}: missing_blocks={len(missing)}/{len(blocks)}
reconstruction_units={len(rus)}')
PY

────────────────────────────────────────────────────────────────────────────────

• Toujours en cours. Je laisse finir.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
batch=Path('results/new10_background_textures_20260616_batch2')
summary=json.loads((batch/'summary.json').read_text())
orch=PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(batch/'_verify_render'))
for item in summary[:10]:
    n=int(item['page']); pdf=Path(item['pdf'])
    doc=orch.run(str(pdf), pages='1',
language={'source_lang':'en','target_lang':'fr'})
    ok=[p for p in doc.get('pages') or [] if p.get('status')=='ok'][0]
    input_data=ok['input_data']
    res=build_page_translation(input_data, dry_run=True)
    rus=res['translated_input_data']['views']['reconstruction_units']
    covered={sid for ru in rus for sid in ru.get('source_unit_ids') or []}
    blocks=[u for u in input_data.get('units') or [] if u.get('level')=='block'
and ((u.get('content') or {}).get('text') or '').strip()]
    missing=[u.get('unit_id') for u in blocks if u.get('unit_id') not in
covered]
    print(f'p{n:04d}: missing_blocks={len(missing)}/{len(blocks)}
reconstruction_units={len(rus)}')
PY

────────────────────────────────────────────────────────────────────────────────

• Encore en cours; pas de nouvelle action.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
batch=Path('results/new10_background_textures_20260616_batch2')
summary=json.loads((batch/'summary.json').read_text())
orch=PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(batch/'_verify_render'))
for item in summary[:10]:
    n=int(item['page']); pdf=Path(item['pdf'])
    doc=orch.run(str(pdf), pages='1',
language={'source_lang':'en','target_lang':'fr'})
    ok=[p for p in doc.get('pages') or [] if p.get('status')=='ok'][0]
    input_data=ok['input_data']
    res=build_page_translation(input_data, dry_run=True)
    rus=res['translated_input_data']['views']['reconstruction_units']
    covered={sid for ru in rus for sid in ru.get('source_unit_ids') or []}
    blocks=[u for u in input_data.get('units') or [] if u.get('level')=='block'
and ((u.get('content') or {}).get('text') or '').strip()]
    missing=[u.get('unit_id') for u in blocks if u.get('unit_id') not in
covered]
    print(f'p{n:04d}: missing_blocks={len(missing)}/{len(blocks)}
reconstruction_units={len(rus)}')
PY

────────────────────────────────────────────────────────────────────────────────

• Le run d’extraction complet est lent, mais il progresse sans sortie
  intermédiaire. Je vais attendre une dernière tranche; les tests unitaires de
  l’invariant sont déjà verts.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
batch=Path('results/new10_background_textures_20260616_batch2')
summary=json.loads((batch/'summary.json').read_text())
orch=PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(batch/'_verify_render'))
for item in summary[:10]:
    n=int(item['page']); pdf=Path(item['pdf'])
    doc=orch.run(str(pdf), pages='1',
language={'source_lang':'en','target_lang':'fr'})
    ok=[p for p in doc.get('pages') or [] if p.get('status')=='ok'][0]
    input_data=ok['input_data']
    res=build_page_translation(input_data, dry_run=True)
    rus=res['translated_input_data']['views']['reconstruction_units']
    covered={sid for ru in rus for sid in ru.get('source_unit_ids') or []}
    blocks=[u for u in input_data.get('units') or [] if u.get('level')=='block'
and ((u.get('content') or {}).get('text') or '').strip()]
    missing=[u.get('unit_id') for u in blocks if u.get('unit_id') not in
covered]
    print(f'p{n:04d}: missing_blocks={len(missing)}/{len(blocks)}
reconstruction_units={len(rus)}')
PY

────────────────────────────────────────────────────────────────────────────────

• Ça dépasse le temps utile pour une vérification visuelle. Je garde le résultat
  de test automatisé et je ne vais pas multiplier les runs lents.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
batch=Path('results/new10_background_textures_20260616_batch2')
summary=json.loads((batch/'summary.json').read_text())
orch=PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(batch/'_verify_render'))
for item in summary[:10]:
    n=int(item['page']); pdf=Path(item['pdf'])
    doc=orch.run(str(pdf), pages='1',
language={'source_lang':'en','target_lang':'fr'})
    ok=[p for p in doc.get('pages') or [] if p.get('status')=='ok'][0]
    input_data=ok['input_data']
    res=build_page_translation(input_data, dry_run=True)
    rus=res['translated_input_data']['views']['reconstruction_units']
    covered={sid for ru in rus for sid in ru.get('source_unit_ids') or []}
    blocks=[u for u in input_data.get('units') or [] if u.get('level')=='block'
and ((u.get('content') or {}).get('text') or '').strip()]
    missing=[u.get('unit_id') for u in blocks if u.get('unit_id') not in
covered]
    print(f'p{n:04d}: missing_blocks={len(missing)}/{len(blocks)}
reconstruction_units={len(rus)}')
PY

• Ran .docs-parser/bin/python - <<'PY'
  │ from pathlib import Path
  │ import json
  │ … +25 lines
  └ /home/raphael/Mes_Projets/docs_parser/.docs-parser/lib/python3.11/
    site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3
    … +11 lines (ctrl + t to view transcript)
    p0358: missing_blocks=0/4 reconstruction_units=28
    p0414: missing_blocks=0/11 reconstruction_units=141

────────────────────────────────────────────────────────────────────────────────

• Je tente d’arrêter ce run de vérification long; il n’est pas nécessaire pour
  la correction elle-même.

⚠ Heads up, you have less than 5% of your weekly limit left. Run /status for a
  breakdown.

────────────────────────────────────────────────────────────────────────────────

• Oui. Règle appliquée: on ne saute rien.

  Correction faite dans pagetranslate/projection.py:

  - les unités traduites restent mode="translated_text";
  - toute unité originale non traduite/non sélectionnée est ajoutée en
    mode="preserve_original";

  - donc chaque élément original a une disposition explicite après
    pagetranslate.

  Test ajouté:
  tests/pagetranslate/test_projection_preserves_all_originals.py

  Vérification:

  - 14 passed sur les tests ciblés.
  - Sur les 10 pages réelles:
      - p0033: missing blocks 0/6
      - p0077: 0/20
      - p0219: 0/23
      - p0252: 0/13
      - p0264: 0/6
      - p0279: 0/16
      - p0315: 0/11
      - p0324: 0/14
      - p0358: 0/4
      - p0414: 0/11

  Donc maintenant: 100% des blocks originaux sont présents après pagetranslate,
  traduits ou préservés.

─ Worked for 5m 17s ────────────────────────────────────────────────────────────


› Et les vues des fichiers ?


• Je vais régénérer les vues sur les mêmes 10 pages avec la correction
  appliquée, pour que les fichiers montrent bien 0 block manquant.

• La régénération relance l’extraction des 10 pages; j’attends la sortie finale
  et je te donne les liens.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json
from PIL import Image, ImageDraw
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
from pagereconstruct import compile_page_render_plan
batch=Path('results/new10_background_textures_20260616_batch2')
summary=json.loads((batch/'summary.json').read_text())
out=batch/'after_fix_all_original_blocks_views'
render_dir=out/'_render'
out.mkdir(parents=True, exist_ok=True); render_dir.mkdir(exist_ok=True)
orch=PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(render_dir))
items=[]
for item in summary:
    n=int(item['page']); pdf=Path(item['pdf']);
clean=Path(item['clean_background'])
    doc=orch.run(str(pdf), pages='1',
language={'source_lang':'en','target_lang':'fr'})
    ok=[p for p in doc.get('pages') or [] if p.get('status')=='ok'][0]
    input_data=ok['input_data']
    res=build_page_translation(input_data, dry_run=True)
    tid=res['translated_input_data']
    plan=compile_page_render_plan(tid).to_dict()
    img=Image.open(clean).convert('RGB')
    draw=ImageDraw.Draw(img,'RGBA')
    geom=(input_data.get('page') or {}).get('geometry') or {}
    sx=img.width/float(geom.get('width') or plan.get('page',{}).get('width_pt')
or 1)
    sy=img.height/float(geom.get('height') or plan.get('page',
{}).get('height_pt') or 1)
    original_blocks=[u for u in input_data.get('units') or [] if
u.get('level')=='block' and ((u.get('content') or {}).get('text') or
'').strip()]
    covered={sid for ru in (tid.get('views') or {}).get('reconstruction_units')
or [] for sid in ru.get('source_unit_ids') or []}
    missing=[u.get('unit_id') for u in original_blocks if u.get('unit_id') not
in covered]
    # red: original blocks
    for u in original_blocks:
        b=(u.get('geometry') or {}).get('bbox')
        if isinstance(b,list) and len(b)==4:
            draw.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],
outline=(230,0,0,255), width=4)
    # blue: after pagetranslate/composition intrablock lines
    line_count=0
    for comp in plan.get('intrablock_compositions') or []:
        for line in comp.get('lines') or []:
            b=line.get('bbox')
            if isinstance(b,list) and len(b)==4:
                draw.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],
outline=(0,90,255,255), width=2)
                line_count+=1
    out_path=out/f'page_after_fix_all_original_blocks_p{n:04d}.png'
    img.save(out_path)
    (out/f'projection_after_fix_p{n:04d}.json').write_text(json.dumps({
        'page': n,
        'original_block_count': len(original_blocks),
        'missing_original_blocks': missing,
        'reconstruction_unit_count': len((tid.get('views') or
{}).get('reconstruction_units') or []),
        'intrablock_line_count': line_count,
        'translation_result': tid.get('translation_result'),
        'reconstruction_units': (tid.get('views') or
{}).get('reconstruction_units') or [],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
items.append({'page':n,'file':str(out_path),'original_block_count':len(original_
blocks),'missing_count':len(missing),'line_count':line_count})
# contact sheet
tile_w,tile_h=340,250
sheet=Image.new('RGB',(5*tile_w,2*tile_h),'white')
for idx,it in enumerate(items):
    im=Image.open(it['file']).convert('RGB'); im.thumbnail((tile_w,tile_h),
Image.LANCZOS)
    x=(idx%5)*tile_w+(tile_w-im.width)//2; y=(idx//5)*tile_h+(tile_h-
im.height)//2
    sheet.paste(im,(x,y))
contact=out/'contact_sheet_after_fix_all_original_blocks.png'
sheet.save(contact)
(out/'summary.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),en
coding='utf-8')
print('OUT',out)
print('CONTACT',contact)
for it in items:
    print(f"p{it['page']:04d}: missing={it['missing_count']}/
{it['original_block_count']} lines={it['line_count']} -> {it['file']}")
PY

• Toujours en cours; pas de nouveau lancement.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json
from PIL import Image, ImageDraw
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
from pagereconstruct import compile_page_render_plan
batch=Path('results/new10_background_textures_20260616_batch2')
summary=json.loads((batch/'summary.json').read_text())
out=batch/'after_fix_all_original_blocks_views'
render_dir=out/'_render'
out.mkdir(parents=True, exist_ok=True); render_dir.mkdir(exist_ok=True)
orch=PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(render_dir))
items=[]
for item in summary:
    n=int(item['page']); pdf=Path(item['pdf']);
clean=Path(item['clean_background'])
    doc=orch.run(str(pdf), pages='1',
language={'source_lang':'en','target_lang':'fr'})
    ok=[p for p in doc.get('pages') or [] if p.get('status')=='ok'][0]
    input_data=ok['input_data']
    res=build_page_translation(input_data, dry_run=True)
    tid=res['translated_input_data']
    plan=compile_page_render_plan(tid).to_dict()
    img=Image.open(clean).convert('RGB')
    draw=ImageDraw.Draw(img,'RGBA')
    geom=(input_data.get('page') or {}).get('geometry') or {}
    sx=img.width/float(geom.get('width') or plan.get('page',{}).get('width_pt')
or 1)
    sy=img.height/float(geom.get('height') or plan.get('page',
{}).get('height_pt') or 1)
    original_blocks=[u for u in input_data.get('units') or [] if
u.get('level')=='block' and ((u.get('content') or {}).get('text') or
'').strip()]
    covered={sid for ru in (tid.get('views') or {}).get('reconstruction_units')
or [] for sid in ru.get('source_unit_ids') or []}
    missing=[u.get('unit_id') for u in original_blocks if u.get('unit_id') not
in covered]
    # red: original blocks
    for u in original_blocks:
        b=(u.get('geometry') or {}).get('bbox')
        if isinstance(b,list) and len(b)==4:
            draw.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],
outline=(230,0,0,255), width=4)
    # blue: after pagetranslate/composition intrablock lines
    line_count=0
    for comp in plan.get('intrablock_compositions') or []:
        for line in comp.get('lines') or []:
            b=line.get('bbox')
            if isinstance(b,list) and len(b)==4:
                draw.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],
outline=(0,90,255,255), width=2)
                line_count+=1
    out_path=out/f'page_after_fix_all_original_blocks_p{n:04d}.png'
    img.save(out_path)
    (out/f'projection_after_fix_p{n:04d}.json').write_text(json.dumps({
        'page': n,
        'original_block_count': len(original_blocks),
        'missing_original_blocks': missing,
        'reconstruction_unit_count': len((tid.get('views') or
{}).get('reconstruction_units') or []),
        'intrablock_line_count': line_count,
        'translation_result': tid.get('translation_result'),
        'reconstruction_units': (tid.get('views') or
{}).get('reconstruction_units') or [],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
items.append({'page':n,'file':str(out_path),'original_block_count':len(original_
blocks),'missing_count':len(missing),'line_count':line_count})
# contact sheet
tile_w,tile_h=340,250
sheet=Image.new('RGB',(5*tile_w,2*tile_h),'white')
for idx,it in enumerate(items):
    im=Image.open(it['file']).convert('RGB'); im.thumbnail((tile_w,tile_h),
Image.LANCZOS)
    x=(idx%5)*tile_w+(tile_w-im.width)//2; y=(idx//5)*tile_h+(tile_h-
im.height)//2
    sheet.paste(im,(x,y))
contact=out/'contact_sheet_after_fix_all_original_blocks.png'
sheet.save(contact)
(out/'summary.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),en
coding='utf-8')
print('OUT',out)
print('CONTACT',contact)
for it in items:
    print(f"p{it['page']:04d}: missing={it['missing_count']}/
{it['original_block_count']} lines={it['line_count']} -> {it['file']}")
PY

• Le traitement continue. Je récupère dès que terminé.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json
from PIL import Image, ImageDraw
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
from pagereconstruct import compile_page_render_plan
batch=Path('results/new10_background_textures_20260616_batch2')
summary=json.loads((batch/'summary.json').read_text())
out=batch/'after_fix_all_original_blocks_views'
render_dir=out/'_render'
out.mkdir(parents=True, exist_ok=True); render_dir.mkdir(exist_ok=True)
orch=PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(render_dir))
items=[]
for item in summary:
    n=int(item['page']); pdf=Path(item['pdf']);
clean=Path(item['clean_background'])
    doc=orch.run(str(pdf), pages='1',
language={'source_lang':'en','target_lang':'fr'})
    ok=[p for p in doc.get('pages') or [] if p.get('status')=='ok'][0]
    input_data=ok['input_data']
    res=build_page_translation(input_data, dry_run=True)
    tid=res['translated_input_data']
    plan=compile_page_render_plan(tid).to_dict()
    img=Image.open(clean).convert('RGB')
    draw=ImageDraw.Draw(img,'RGBA')
    geom=(input_data.get('page') or {}).get('geometry') or {}
    sx=img.width/float(geom.get('width') or plan.get('page',{}).get('width_pt')
or 1)
    sy=img.height/float(geom.get('height') or plan.get('page',
{}).get('height_pt') or 1)
    original_blocks=[u for u in input_data.get('units') or [] if
u.get('level')=='block' and ((u.get('content') or {}).get('text') or
'').strip()]
    covered={sid for ru in (tid.get('views') or {}).get('reconstruction_units')
or [] for sid in ru.get('source_unit_ids') or []}
    missing=[u.get('unit_id') for u in original_blocks if u.get('unit_id') not
in covered]
    # red: original blocks
    for u in original_blocks:
        b=(u.get('geometry') or {}).get('bbox')
        if isinstance(b,list) and len(b)==4:
            draw.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],
outline=(230,0,0,255), width=4)
    # blue: after pagetranslate/composition intrablock lines
    line_count=0
    for comp in plan.get('intrablock_compositions') or []:
        for line in comp.get('lines') or []:
            b=line.get('bbox')
            if isinstance(b,list) and len(b)==4:
                draw.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],
outline=(0,90,255,255), width=2)
                line_count+=1
    out_path=out/f'page_after_fix_all_original_blocks_p{n:04d}.png'
    img.save(out_path)
    (out/f'projection_after_fix_p{n:04d}.json').write_text(json.dumps({
        'page': n,
        'original_block_count': len(original_blocks),
        'missing_original_blocks': missing,
        'reconstruction_unit_count': len((tid.get('views') or
{}).get('reconstruction_units') or []),
        'intrablock_line_count': line_count,
        'translation_result': tid.get('translation_result'),
        'reconstruction_units': (tid.get('views') or
{}).get('reconstruction_units') or [],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
items.append({'page':n,'file':str(out_path),'original_block_count':len(original_
blocks),'missing_count':len(missing),'line_count':line_count})
# contact sheet
tile_w,tile_h=340,250
sheet=Image.new('RGB',(5*tile_w,2*tile_h),'white')
for idx,it in enumerate(items):
    im=Image.open(it['file']).convert('RGB'); im.thumbnail((tile_w,tile_h),
Image.LANCZOS)
    x=(idx%5)*tile_w+(tile_w-im.width)//2; y=(idx//5)*tile_h+(tile_h-
im.height)//2
    sheet.paste(im,(x,y))
contact=out/'contact_sheet_after_fix_all_original_blocks.png'
sheet.save(contact)
(out/'summary.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),en
coding='utf-8')
print('OUT',out)
print('CONTACT',contact)
for it in items:
    print(f"p{it['page']:04d}: missing={it['missing_count']}/
{it['original_block_count']} lines={it['line_count']} -> {it['file']}")
PY

• Encore en cours. Je laisse finir, car il doit écrire les vues corrigées.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json
from PIL import Image, ImageDraw
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
from pagereconstruct import compile_page_render_plan
batch=Path('results/new10_background_textures_20260616_batch2')
summary=json.loads((batch/'summary.json').read_text())
out=batch/'after_fix_all_original_blocks_views'
render_dir=out/'_render'
out.mkdir(parents=True, exist_ok=True); render_dir.mkdir(exist_ok=True)
orch=PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(render_dir))
items=[]
for item in summary:
    n=int(item['page']); pdf=Path(item['pdf']);
clean=Path(item['clean_background'])
    doc=orch.run(str(pdf), pages='1',
language={'source_lang':'en','target_lang':'fr'})
    ok=[p for p in doc.get('pages') or [] if p.get('status')=='ok'][0]
    input_data=ok['input_data']
    res=build_page_translation(input_data, dry_run=True)
    tid=res['translated_input_data']
    plan=compile_page_render_plan(tid).to_dict()
    img=Image.open(clean).convert('RGB')
    draw=ImageDraw.Draw(img,'RGBA')
    geom=(input_data.get('page') or {}).get('geometry') or {}
    sx=img.width/float(geom.get('width') or plan.get('page',{}).get('width_pt')
or 1)
    sy=img.height/float(geom.get('height') or plan.get('page',
{}).get('height_pt') or 1)
    original_blocks=[u for u in input_data.get('units') or [] if
u.get('level')=='block' and ((u.get('content') or {}).get('text') or
'').strip()]
    covered={sid for ru in (tid.get('views') or {}).get('reconstruction_units')
or [] for sid in ru.get('source_unit_ids') or []}
    missing=[u.get('unit_id') for u in original_blocks if u.get('unit_id') not
in covered]
    # red: original blocks
    for u in original_blocks:
        b=(u.get('geometry') or {}).get('bbox')
        if isinstance(b,list) and len(b)==4:
            draw.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],
outline=(230,0,0,255), width=4)
    # blue: after pagetranslate/composition intrablock lines
    line_count=0
    for comp in plan.get('intrablock_compositions') or []:
        for line in comp.get('lines') or []:
            b=line.get('bbox')
            if isinstance(b,list) and len(b)==4:
                draw.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],
outline=(0,90,255,255), width=2)
                line_count+=1
    out_path=out/f'page_after_fix_all_original_blocks_p{n:04d}.png'
    img.save(out_path)
    (out/f'projection_after_fix_p{n:04d}.json').write_text(json.dumps({
        'page': n,
        'original_block_count': len(original_blocks),
        'missing_original_blocks': missing,
        'reconstruction_unit_count': len((tid.get('views') or
{}).get('reconstruction_units') or []),
        'intrablock_line_count': line_count,
        'translation_result': tid.get('translation_result'),
        'reconstruction_units': (tid.get('views') or
{}).get('reconstruction_units') or [],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
items.append({'page':n,'file':str(out_path),'original_block_count':len(original_
blocks),'missing_count':len(missing),'line_count':line_count})
# contact sheet
tile_w,tile_h=340,250
sheet=Image.new('RGB',(5*tile_w,2*tile_h),'white')
for idx,it in enumerate(items):
    im=Image.open(it['file']).convert('RGB'); im.thumbnail((tile_w,tile_h),
Image.LANCZOS)
    x=(idx%5)*tile_w+(tile_w-im.width)//2; y=(idx//5)*tile_h+(tile_h-
im.height)//2
    sheet.paste(im,(x,y))
contact=out/'contact_sheet_after_fix_all_original_blocks.png'
sheet.save(contact)
(out/'summary.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),en
coding='utf-8')
print('OUT',out)
print('CONTACT',contact)
for it in items:
    print(f"p{it['page']:04d}: missing={it['missing_count']}/
{it['original_block_count']} lines={it['line_count']} -> {it['file']}")
PY

• C’est long mais le processus n’a pas échoué. J’attends la fin pour éviter de
  te donner des fichiers partiels.

• Ran .docs-parser/bin/python - <<'PY'
  │ from pathlib import Path
  │ import json
  │ … +92 lines
  └ /home/raphael/Mes_Projets/docs_parser/.docs-parser/lib/python3.11/
    site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3
    … +34 lines (ctrl + t to view transcript)
    after_fix_all_original_blocks_views/
    page_after_fix_all_original_blocks_p0414.png

────────────────────────────────────────────────────────────────────────────────

• Les vues corrigées sont prêtes. Je vérifie la planche contact rapidement.

• Viewed Image
  └ results/new10_background_textures_20260616_batch2/
  after_fix_all_original_blocks_views/
  contact_sheet_after_fix_all_original_blocks.png

────────────────────────────────────────────────────────────────────────────────

• Voici les vues régénérées après correction.

  Planche contact: results/new10_background_textures_20260616_batch2/
  after_fix_all_original_blocks_views/
  contact_sheet_after_fix_all_original_blocks.png

  Dossier complet: results/new10_background_textures_20260616_batch2/
  after_fix_all_original_blocks_views

  Résultat sur les 10 pages: missing=0 partout.
  Rouge = blocks originaux. Bleu = lignes intra-block après pagetranslate/
  reconstruction.

› C'est parfait.
  Dans les blocks, on voit que TOUT CE QUI EST TEXTE, extrait dans pageprint/
  N'EST PAS RETROUVE À 100% DANS LES BLOCKS APRES PAGERECONSTRUCT/. Certains
  textes disparaissent ou quoi ? NE DOIS T-ON PAS AVOIR 100% DE TEXTES
  RETROUVÉS? C'est comme si certains textes disparaissent dans pagetranslate/
  et dans pagereconstruct.
  Pour cela regarde seulement original : @results/
  new10_background_textures_20260616_batch2/
  after_fix_all_original_blocks_views/_render/src_test_docintelligence-
  279_p001.png et son rendu bboxes regarde le blocks il y manque des lignes de
  textes) @results/new10_background_textures_20260616_batch2/
  after_fix_all_original_blocks_views/
  page_after_fix_all_original_blocks_p0279.png

  POURQUOI? QUEL EST LA CAUSE ?


• Je vais vérifier uniquement p0279: source originale, vue bboxes, puis le JSON
  projeté après correction pour identifier quelles unités texte de pageprint ne
  deviennent pas des lignes dans pagereconstruct.

• Viewed Image
  └ results/new10_background_textures_20260616_batch2/
  after_fix_all_original_blocks_views/_render/src_test_docintelligence-
  279_p001.png

• Ran ls -la results/new10_background_textures_20260616_batch2/
  │ after_fix_all_original_blocks_views | sed -n '1,80p' && python3 - <<'PY'
  │ import json
  │ … +18 lines
  └ total 6628
    drwxrwxr-x 3 raphael raphael   4096 Jun 16 07:53 .
    … +57 lines (ctrl + t to view transcript)
    source domain and fine-tune them to fit the target domain.' [93.0, 209.0,
    412.0, 219.0]

• Viewed Image
  └ results/new10_background_textures_20260616_batch2/
  after_fix_all_original_blocks_views/
  page_after_fix_all_original_blocks_p0279.png

────────────────────────────────────────────────────────────────────────────────

• Non, on ne doit pas perdre le texte. Mais ce que tu vois sur p0279 vient
  surtout d’un autre problème:

  Les bboxes bleues ne sont pas les bboxes originales ligne/par/phrase.
  Ce sont les lignes recomposées par pagereconstruct après sélection/projection.

  Donc il y a 3 phénomènes:

  1. pagetranslate regroupe
     Code: pagetranslate/selector.py
      - select_translation_units
      - _select_semantic_system_units
      - _select_pageprint_units_by_block

     Il peut prendre un paragraphe entier ou une phrase sémantique au lieu de
     garder chaque ligne originale.

  2. pagereconstruct déduplique parent/enfants
     Code: pagereconstruct/plan_compiler.py

     Partie responsable:
      - rendered_ids
      - _descendants(...)
      - duplicate_render_skipped

     Si un block parent est rendu, ses lignes/enfants ne sont pas rendus
     séparément, pour éviter le double rendu.

  3. pagereconstruct reflow le texte
     Code: pagereconstruct/composition/intrablock_composer.py
      - compose_block
      - compose_contract

     Donc les lignes finales peuvent ne pas correspondre aux lignes originales.
     Le texte peut être là, mais avec des bboxes différentes.

  Sur p0279, après correction:

  - blocks originaux: 16
  - blocks manquants: 0
  - reconstruction units: 161
  - lignes recomposées: 56

  Donc les blocks ne disparaissent plus.
  Mais tous les éléments texte originaux ne sont pas forcément visibles comme
  bboxes individuelles, car ils sont parfois couverts par une unité parent
  recomposée.

  La vraie cause:
  le pipeline vérifie trop la couverture par unités/bboxes, pas assez la
  couverture textuelle exacte.

  Ce qu’il faut ajouter maintenant:

  - un audit TextContentCoverageAudit
  - pour chaque texte extrait par pageprint, vérifier qu’il est soit:
      - présent dans une traduction,
      - présent tel quel dans une préservation,
      - présent dans une unité parent rendue,
      - explicitement exclu avec raison valide.

  Et si un texte original n’est dans aucun de ces cas: KO.

─ Worked for 1m 36s ────────────────────────────────────────────────────────────


› - pour chaque texte extrait par pageprint, vérifier qu’il est soit:
        - présent dans une traduction,
        - présent tel quel dans une préservation,
        - présent dans une unité parent rendue,
        - explicitement exclu avec raison valide.
  C'est ça qu'il faut....


• Je vais l’ajouter comme audit de contenu dans le validateur existant
  pagetranslate.functional_validator, avec test invariant. Pas de nouveau
  pipeline parallèle.

• Explored
  └ Read functional_validator.py, test_projection_preserves_all_originals.py

────────────────────────────────────────────────────────────────────────────────

• J’ajoute l’audit original_text_coverage directement dans
  validate_functional_translation: il échoue si un texte PagePrint n’a aucune
  disposition explicite.

• Edited pagetranslate/functional_validator.py (+89 -0)
      6  PRESERVE_ROLES = {"command_name", "path", "file_name", "url", "email",
         "toc_page_reference", "index_page_reference"}
      7 +TEXT_LEVELS = {"block", "line", "phrase", "span", "word"}
      8 +VALID_EXCLUSION_REASONS = {
      9 +    "artifact", "publisher_mark", "watermark", "page_number", "formula"
         , "code",
     10 +    "protected_visual_region", "background_only", "exclude_as_artifact"
         ,
     11 +}
     12
        ⋮
     28          "translation_plan_input_count": int(debug.get("translation_plan
         _input_count") or 0),
     29 +        "original_text_unit_count": 0,
     30 +        "original_text_missing_disposition": 0,
     31      }
        ⋮
     58              errors.append(f"reconstruction_unit_render_target_missing:{
         item.get('unit_id')}")
     59 +    coverage = audit_original_text_coverage(translated_input)
     60 +    metrics["original_text_unit_count"] = coverage["original_text_unit_
         count"]
     61 +    metrics["original_text_missing_disposition"] = coverage["missing_co
         unt"]
     62 +    for item in coverage["missing"]:
     63 +        errors.append(f"original_text_missing_disposition:{item['unit_i
         d']}")
     64      return {
        ⋮
     69          "metrics": metrics,
     70 +        "original_text_coverage": coverage,
     71      }
     72 +
     73 +
     74 +def audit_original_text_coverage(translated_input: dict) -> dict:
     75 +    """Every original PagePrint text unit must survive with a dispositi
         on.
     76 +
     77 +    A text unit is covered when it is rendered translated, preserved as
          original,
     78 +    covered by a rendered parent, covered by rendered children, or expl
         icitly
     79 +    excluded with a valid reason.
     80 +    """
     81 +    units = [
     82 +        u for u in translated_input.get("units") or []
     83 +        if isinstance(u, dict)
     84 +        and u.get("unit_id")
     85 +        and u.get("level") in TEXT_LEVELS
     86 +        and _text(u)
     87 +    ]
     88 +    by_id = {u["unit_id"]: u for u in units}
     89 +    children: dict[str, list[str]] = {}
     90 +    for u in translated_input.get("units") or []:
     91 +        pid = u.get("parent_id")
     92 +        if pid:
     93 +            children.setdefault(pid, []).append(u.get("unit_id"))
     94 +
     95 +    views = translated_input.get("views") or {}
     96 +    rendered = {
     97 +        sid
     98 +        for ru in views.get("reconstruction_units") or []
     99 +        for sid in ru.get("source_unit_ids") or []
    100 +    }
    101 +    excluded = {}
    102 +    for ex in views.get("exclusion_plan") or []:
    103 +        reason = str(ex.get("reason") or "")
    104 +        for sid in ex.get("source_unit_ids") or []:
    105 +            excluded[sid] = reason
    106 +
    107 +    def ancestors(uid: str):
    108 +        parent = (translated_input.get("_unit_parent_index") or {}).get
         (uid)
    109 +        if parent is None:
    110 +            parent = next((u.get("parent_id") for u in translated_input
         .get("units") or [] if u.get("unit_id") == uid), None)
    111 +        while parent:
    112 +            yield parent
    113 +            parent = next((u.get("parent_id") for u in translated_input
         .get("units") or [] if u.get("unit_id") == parent), None)
    114 +
    115 +    def covered_by_children(uid: str) -> bool:
    116 +        kids = [k for k in children.get(uid, []) if k in by_id]
    117 +        return bool(kids) and all(is_covered(k) for k in kids)
    118 +
    119 +    def is_covered(uid: str) -> bool:
    120 +        if uid in rendered:
    121 +            return True
    122 +        if any(a in rendered for a in ancestors(uid)):
    123 +            return True
    124 +        if uid in excluded and excluded[uid] in VALID_EXCLUSION_REASONS
         :
    125 +            return True
    126 +        return covered_by_children(uid)
    127 +
    128 +    missing = []
    129 +    for u in units:
    130 +        if not is_covered(u["unit_id"]):
    131 +            missing.append({
    132 +                "unit_id": u["unit_id"],
    133 +                "level": u.get("level"),
    134 +                "text": _text(u)[:160],
    135 +                "bbox": (u.get("geometry") or {}).get("bbox"),
    136 +            })
    137 +
    138 +    return {
    139 +        "status": "ok" if not missing else "ko",
    140 +        "original_text_unit_count": len(units),
    141 +        "missing_count": len(missing),
    142 +        "missing": missing,
    143 +    }
    144 +
    145 +
    146 +def _text(unit: dict) -> str:
    147 +    return str((unit.get("content") or {}).get("text") or "").strip()

────────────────────────────────────────────────────────────────────────────────

• J’ajoute un test court qui casse volontairement la couverture et vérifie que
  l’audit le détecte.

• Edited tests/pagetranslate/test_projection_preserves_all_originals.py (+30 -0)
     1  from pagetranslate import build_page_translation
     2 +from pagetranslate.functional_validator import validate_functional_trans
        lation
     3
       ⋮
    47      assert by_source["f1"]["render_contract"]["mode"] == "preserve_origi
        nal"
    48 +    assert result["functional_validation"]["metrics"]["original_text_mis
        sing_disposition"] == 0
    49 +
    50 +
    51 +def test_original_text_coverage_audit_blocks_missing_text_unit():
    52 +    result = build_page_translation({
    53 +        "schema_version": "pageprint.input.v1",
    54 +        "input_id": "missing-original",
    55 +        "document": {"language": {"source_lang": "en", "target_lang": "f
        r"}},
    56 +        "page": {},
    57 +        "units": [{
    58 +            "unit_id": "b1",
    59 +            "level": "block",
    60 +            "content": {"text": "This text must not disappear."},
    61 +            "geometry": {"bbox": [0, 0, 200, 40], "reading_order_index":
         1},
    62 +            "understanding": {"role": "body_paragraph", "object_type": "
        natural_text"},
    63 +            "policy": {"translatable": True, "translation_strategy": "la
        yout_constrained"},
    64 +            "visual": {"style": {}},
    65 +            "children_ids": [],
    66 +        }],
    67 +        "views": {},
    68 +    }, dry_run=True)
    69 +    broken = dict(result)
    70 +    broken["translated_input_data"] = dict(result["translated_input_data
        "])
    71 +    broken["translated_input_data"]["views"] = {"reconstruction_units":
        []}
    72 +
    73 +    validation = validate_functional_translation(broken)
    74 +
    75 +    assert validation["functional_status"] == "ko"
    76 +    assert validation["metrics"]["original_text_missing_disposition"] ==
         1

────────────────────────────────────────────────────────────────────────────────

• Je lance les tests ciblés.

• Ran .docs-parser/bin/pytest tests/pagetranslate/
  │ test_projection_preserves_all_originals.py tests/pagetranslate/
  │ test_projection_bboxes.py tests/pagetranslate/
  │ … +1 lines
  └ ============================= test session starts
    ==============================
    … +12 lines (ctrl + t to view transcript)
    ============================== 5 passed in 0.04s
    ===============================

────────────────────────────────────────────────────────────────────────────────

• C’est ajouté.

  Fonction concernée:
  pagetranslate/functional_validator.py

  Nouvel audit:
  audit_original_text_coverage

  Il vérifie chaque texte PagePrint original:

  - rendu traduit,
  - ou préservé tel quel,
  - ou couvert par un parent rendu,
  - ou couvert par ses enfants rendus,
  - ou exclu avec raison valide.

  Sinon: functional_status = ko.

  Tests:
  tests/pagetranslate/test_projection_preserves_all_originals.py

  Validation:
  5 passed

  Donc maintenant, si un texte PagePrint disparaît sans disposition explicite,
  pagetranslate échoue.

─ Worked for 1m 39s ────────────────────────────────────────────────────────────


On refait les visuels je vais voir


• Je régénère les visuels sur les mêmes 10 pages avec l’audit de couverture
  texte actif, puis je te donne la planche et les fichiers.

• Context compacted

• Je reprends le rendu des mêmes pages avec l’audit de couverture texte activé.
  Je vais sortir un contact sheet et les PNG page par page, uniquement pour que
  tu voies les bboxes originales et le remplissage intra-block projeté.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json
from PIL import Image, ImageDraw
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
from pagereconstruct import compile_page_render_plan
batch=Path('results/new10_background_textures_20260616_batch2')
summary=json.loads((batch/'summary.json').read_text())
out=batch/'after_text_coverage_audit_views'
render_dir=out/'_render'
out.mkdir(parents=True, exist_ok=True); render_dir.mkdir(exist_ok=True)
orch=PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(render_dir))
items=[]
for item in summary:
    n=int(item['page']); pdf=Path(item['pdf']);
clean=Path(item['clean_background'])
    doc=orch.run(str(pdf), pages='1',
language={'source_lang':'en','target_lang':'fr'})
    ok=[p for p in doc.get('pages') or [] if p.get('status')=='ok'][0]
    input_data=ok['input_data']
    result=build_page_translation(input_data, dry_run=True)
    tid=result['translated_input_data']
    plan=compile_page_render_plan(tid).to_dict()
    img=Image.open(clean).convert('RGB')
    draw=ImageDraw.Draw(img,'RGBA')
    geom=(input_data.get('page') or {}).get('geometry') or {}
    sx=img.width/float(geom.get('width') or plan.get('page',{}).get('width_pt')
or 1)
    sy=img.height/float(geom.get('height') or plan.get('page',
{}).get('height_pt') or 1)
    original_blocks=[u for u in input_data.get('units') or [] if
u.get('level')=='block' and ((u.get('content') or {}).get('text') or
'').strip()]
    for u in original_blocks:
        b=(u.get('geometry') or {}).get('bbox')
        if isinstance(b,list) and len(b)==4:
            draw.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],
outline=(230,0,0,255), width=4)
    line_count=0
    for comp in plan.get('intrablock_compositions') or []:
        for line in comp.get('lines') or []:
            b=line.get('bbox')
            if isinstance(b,list) and len(b)==4:
                draw.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],
outline=(0,90,255,255), width=2)
                line_count+=1
    cov=(result.get('functional_validation') or
{}).get('original_text_coverage') or {}
    out_path=out/f'page_text_coverage_audit_p{n:04d}.png'
    img.save(out_path)
    (out/f'coverage_projection_p{n:04d}.json').write_text(json.dumps({
        'page': n,
        'functional_validation': result.get('functional_validation'),
        'translation_result': tid.get('translation_result'),
        'reconstruction_units': (tid.get('views') or
{}).get('reconstruction_units') or [],
        'intrablock_compositions': plan.get('intrablock_compositions') or [],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    items.append({
        'page': n,
        'file': str(out_path),
        'original_blocks': len(original_blocks),
        'line_count': line_count,
        'coverage_status': cov.get('status'),
        'missing_text_count': cov.get('missing_count'),
        'functional_status': (result.get('functional_validation') or
{}).get('functional_status'),
    })
# contact sheet
tile_w,tile_h=340,250
sheet=Image.new('RGB',(5*tile_w,2*tile_h),'white')
for idx,it in enumerate(items):
    im=Image.open(it['file']).convert('RGB'); im.thumbnail((tile_w,tile_h),
Image.LANCZOS)
    x=(idx%5)*tile_w+(tile_w-im.width)//2; y=(idx//5)*tile_h+(tile_h-
im.height)//2
    sheet.paste(im,(x,y))
contact=out/'contact_sheet_text_coverage_audit.png'
sheet.save(contact)
(out/'summary.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),en
coding='utf-8')
print('OUT',out)
print('CONTACT',contact)
for it in items:
    print(f"p{it['page']:04d}: coverage={it['coverage_status']}
missing_text={it['missing_text_count']} blocks={it['original_blocks']}
lines={it['line_count']} -> {it['file']}")
PY

• Le job tourne encore. Il relance le pipeline page par page, donc je laisse
  finir au lieu de dupliquer les calculs.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json
from PIL import Image, ImageDraw
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate import build_page_translation
from pagereconstruct import compile_page_render_plan
batch=Path('results/new10_background_textures_20260616_batch2')
summary=json.loads((batch/'summary.json').read_text())
out=batch/'after_text_coverage_audit_views'
render_dir=out/'_render'
out.mkdir(parents=True, exist_ok=True); render_dir.mkdir(exist_ok=True)
orch=PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(render_dir))
items=[]
for item in summary:
    n=int(item['page']); pdf=Path(item['pdf']);
clean=Path(item['clean_background'])
    doc=orch.run(str(pdf), pages='1',
language={'source_lang':'en','target_lang':'fr'})
    ok=[p for p in doc.get('pages') or [] if p.get('status')=='ok'][0]
    input_data=ok['input_data']
    result=build_page_translation(input_data, dry_run=True)
    tid=result['translated_input_data']
    plan=compile_page_render_plan(tid).to_dict()
    img=Image.open(clean).convert('RGB')
    draw=ImageDraw.Draw(img,'RGBA')
    geom=(input_data.get('page') or {}).get('geometry') or {}
    sx=img.width/float(geom.get('width') or plan.get('page',{}).get('width_pt')
or 1)
    sy=img.height/float(geom.get('height') or plan.get('page',
{}).get('height_pt') or 1)
    original_blocks=[u for u in input_data.get('units') or [] if
u.get('level')=='block' and ((u.get('content') or {}).get('text') or
'').strip()]
    for u in original_blocks:
        b=(u.get('geometry') or {}).get('bbox')
        if isinstance(b,list) and len(b)==4:
            draw.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],
outline=(230,0,0,255), width=4)
    line_count=0
    for comp in plan.get('intrablock_compositions') or []:
        for line in comp.get('lines') or []:
            b=line.get('bbox')
            if isinstance(b,list) and len(b)==4:
                draw.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],
outline=(0,90,255,255), width=2)
                line_count+=1
    cov=(result.get('functional_validation') or
{}).get('original_text_coverage') or {}
    out_path=out/f'page_text_coverage_audit_p{n:04d}.png'
    img.save(out_path)
    (out/f'coverage_projection_p{n:04d}.json').write_text(json.dumps({
        'page': n,
        'functional_validation': result.get('functional_validation'),
        'translation_result': tid.get('translation_result'),
        'reconstruction_units': (tid.get('views') or
{}).get('reconstruction_units') or [],
        'intrablock_compositions': plan.get('intrablock_compositions') or [],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    items.append({
        'page': n,
        'file': str(out_path),
        'original_blocks': len(original_blocks),
        'line_count': line_count,
        'coverage_status': cov.get('status'),
        'missing_text_count': cov.get('missing_count'),
        'functional_status': (result.get('functional_validation') or
{}).get('functional_status'),
    })
# contact sheet
tile_w,tile_h=340,250
sheet=Image.new('RGB',(5*tile_w,2*tile_h),'white')
for idx,it in enumerate(items):
    im=Image.open(it['file']).convert('RGB'); im.thumbnail((tile_w,tile_h),
Image.LANCZOS)
    x=(idx%5)*tile_w+(tile_w-im.width)//2; y=(idx//5)*tile_h+(tile_h-
im.height)//2
    sheet.paste(im,(x,y))
contact=out/'contact_sheet_text_coverage_audit.png'
sheet.save(contact)
(out/'summary.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),en
coding='utf-8')
print('OUT',out)
print('CONTACT',contact)
for it in items:
    print(f"p{it['page']:04d}: coverage={it['coverage_status']}
missing_text={it['missing_text_count']} blocks={it['original_blocks']}
lines={it['line_count']} -> {it['file']}")
PY

• Toujours en cours, sans erreur affichée. J’attends la fin pour vérifier
  directement l’image récapitulative.

■ Conversation interrupted - tell the model what to do differently. Something
went wrong? Hit `/feedback` to report the issue.


› Pareil !!!!! la couverture texte est insuffisante.
  Prends le texte fournit entrant dans pagereconstruct/ et prend le texte
  original, block par block, on va comparer. Juste ça et rien d'autre ! et mets
  moi ça en fichier pour que je comprae aussi


• Je vais produire un fichier de comparaison texte par block : original
  PagePrint vs texte entrant dans PageReconstruct. Pas d’analyse longue, juste
  les artefacts comparables.

• Explored
  └ List new10_background_textures_20260616_batch2

• Ran ps -p 77241 -o pid=,cmd=
  └ (no output)

• Explored
  └ Search def build_page_translation|def compile_page_render_plan|
           reconstruction_units|intrablock_compositions in pagetranslate

────────────────────────────────────────────────────────────────────────────────

• Le précédent rendu est terminé. Je vais maintenant générer un dossier texte,
  pas de visuel : un fichier par page avec les blocks originaux et, en face, les
  unités texte qui entrent dans pagereconstruct/.

• Explored
  └ Read projection.py

• Ran head -80 results/new10_background_textures_20260616_batch2/summary.json
  └ [
      {
    … +80 lines (ctrl + t to view transcript)
        "words_masked": 522,
        "previously_used": false

• Explored
  └ List after_text_coverage_audit_views
    Read builder.py

────────────────────────────────────────────────────────────────────────────────

• Je vais grouper par block original. Pour chaque block : texte PagePrint
  original + lignes/enfants originaux + views.reconstruction_units envoyées à
  PageReconstruct qui couvrent ce block.

• Explored
  └ Read coverage_projection_p0279.json

────────────────────────────────────────────────────────────────────────────────

• Je vais utiliser les JSON déjà produits si possibles pour éviter de relancer
  l’extraction. S’ils ne contiennent pas assez, je relance seulement la
  construction texte, sans images.

• Explored
  └ Search "translated_input_data"|"views"|"reconstruction_units" in
           coverage_projection_p0279.json

────────────────────────────────────────────────────────────────────────────────

• Les JSON existants contiennent déjà les reconstruction_units; je complète avec
  les blocks originaux en relançant PagePrint léger pour grouper proprement.

• Explored
  └ Read coverage_projection_p0279.json
    Search PipelineOrchestrator|run_page|extract in scripts
    Read coverage_projection_p0279.json

────────────────────────────────────────────────────────────────────────────────

• Je vais générer deux formats : TXT lisible pour toi et JSON structuré pour
  audit/code. Même contenu, par page et par block.

• Explored
  └ Search class PipelineOrchestrator|PipelineOrchestrator in .

────────────────────────────────────────────────────────────────────────────────

• Je vérifie l’appel exact de l’orchestrateur puis je lance la génération dans
  un nouveau dossier dédié.

• Explored
  └ Read orchestrator.py, run_pageprint_pagetranslate_audit.py

────────────────────────────────────────────────────────────────────────────────

• Je lance la génération maintenant. Sortie prévue : un dossier avec
  block_text_compare_pXXXX.txt et .json pour chaque page, plus un index.

• Ran .docs-parser/bin/python - <<'PY'
  │ from pathlib import Path
  │ import json
  │ … +183 lines
  └ Traceback (most recent call last):
      File "<stdin>", line 55, in <module>
    IndexError: list index out of range

────────────────────────────────────────────────────────────────────────────────

• Le script a appelé la mauvaise indexation de page (1 au lieu de 0). Je relance
  avec l’index interne correct.

────────────────────────────────────────────────────────────────────────────────

• La génération est lancée. Elle écrit les fichiers au fur et à mesure dans
  results/new10_background_textures_20260616_batch2/
  pagereconstruct_input_vs_original_text_by_block/.

■ Conversation interrupted - tell the model what to do differently. Something
went wrong? Hit `/feedback` to report the issue.


› Les phrases rendu devraient être traduites ou bien ?
  Protèges mes tokens


• Oui.

  Dans le rendu final :

  - tout texte traduisible doit être remplacé par sa traduction ;
  - tout texte non traduisible/protégé doit rester tel quel ;
  - rien ne doit disparaître ;
  - rien ne doit être rendu deux fois ;
  - le texte source original ne doit pas rester visible sous la traduction.

  Donc les phrases rendues après pagereconstruct/ doivent normalement venir de
  pagetranslate/ :

  source phrase -> translated phrase -> intrablock layout -> TextOp -> rendu

  Si une phrase rendue reste en anglais alors qu’elle est traduisible, ce n’est
  acceptable que dans un cas explicite :

  status = preserved / not_translatable / protected
  reason = valide

  Sinon c’est un bug : soit pagetranslate ne l’a pas traduite, soit
  pagereconstruct utilise encore le texte source au lieu du translated_text.


› continues


• Je continue uniquement sur le fichier de comparaison texte block par block. Je
  récupère la fin du job puis je te donne les chemins.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json
import re
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate.builder import build_page_translation
BASE = Path('results/new10_background_textures_20260616_batch2')
OUT = BASE / 'pagereconstruct_input_vs_original_text_by_block'
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((BASE / 'summary.json').read_text(encoding='utf-8'))
def norm(s):
    return re.sub(r'\s+', ' ', str(s or '')).strip()
def bbox(u):
    g = u.get('geometry') or {}
    return g.get('bbox') or u.get('bbox') or []
def text_of(u):
    c = u.get('content') or {}
    return norm(c.get('text') or u.get('text') or c.get('value'))
def ro(u):
    return (u.get('geometry') or {}).get('reading_order_index') or
u.get('reading_order_index') or 0
def overlap(a,b):
    if not a or not b or len(a)<4 or len(b)<4:
        return 0.0
    ax0,ay0,ax1,ay1 = map(float,a[:4]); bx0,by0,bx1,by1 = map(float,b[:4])
    ix0,iy0=max(ax0,bx0),max(ay0,by0); ix1,iy1=min(ax1,bx1),min(ay1,by1)
    if ix1<=ix0 or iy1<=iy0: return 0.0
    inter=(ix1-ix0)*(iy1-iy0)
    area=max(1.0,(ax1-ax0)*(ay1-ay0))
    return inter/area
def descendants(unit_id, children_by_parent):
    out=[]; stack=list(children_by_parent.get(unit_id, []))
    seen=set()
    while stack:
        cid=stack.pop(0)
        if cid in seen: continue
        seen.add(cid); out.append(cid); stack.extend(children_by_parent.get(cid,
[]))
    return out
def ru_text(ru):
    return norm(ru.get('translated_text') or ru.get('text') or
ru.get('source_text'))
orch = PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(OUT / '_render'))
index=[]
for item in summary:
    page_no=int(item['page'])
    pdf=item['pdf']
    tag=f'p{page_no:04d}'
    run=orch.run(pdf, pages=[0])
    page=run['pages'][0]
    input_data=page.get('input_data') or page.get('pageprint') or
page.get('pageprint_input') or page
    translated=build_page_translation(input_data, dry_run=True)
    translated_input=translated['translated_input_data']
    units=translated_input.get('units') or []
    unit_by_id={u.get('unit_id'):u for u in units if u.get('unit_id')}
    children_by_parent={}
    for u in units:
        pid=u.get('parent_id') or u.get('parent_unit_id')
        if pid:
            children_by_parent.setdefault(pid,[]).append(u.get('unit_id'))
        for cid in u.get('children_ids') or []:
            children_by_parent.setdefault(u.get('unit_id'),[]).append(cid)
    for k,v in list(children_by_parent.items()):
        children_by_parent[k]=list(dict.fromkeys([x for x in v if x]))
    blocks=[u for u in units if u.get('level')=='block' and text_of(u)]
    blocks.sort(key=ro)
    reconstruction_units=(translated_input.get('views') or
{}).get('reconstruction_units') or []
    page_rows=[]
    lines=[]
    lines.append(f'PAGE {tag} | {pdf}')
    lines.append('='*100)
    lines.append('ORIGINAL = texte PagePrint dans le block original')
    lines.append('PAGERECONSTRUCT_INPUT = views.reconstruction_units donnees a
pagereconstruct/')
    lines.append('')
    for i, b in enumerate(blocks, 1):
        bid=b.get('unit_id')
        desc=set(descendants(bid, children_by_parent))
        src_ids={bid,*desc}
        bb=bbox(b)
        original_children=[]
        for cid in desc:
            cu=unit_by_id.get(cid) or {}
            if cu.get('level') in {'line','phrase','span'} and text_of(cu):
original_children.append({'unit_id':cid,'level':cu.get('level'),'reading_order':
ro(cu),'bbox':bbox(cu),'text':text_of(cu)})
        original_children.sort(key=lambda x:(x['reading_order'], x['bbox'][1] if
x['bbox'] else 0))
        rus=[]
        for ru in reconstruction_units:
            ids=set(ru.get('source_unit_ids') or [])
            reason=None
            if ids & src_ids:
                reason='source_unit_ids'
            elif overlap(ru.get('bbox') or ru.get('layout_bbox') or [], bb) >
0.15:
                reason='bbox_overlap'
            if reason:
                rb=ru.get('bbox') or ru.get('layout_bbox') or []
                rus.append({
                    'unit_id':ru.get('unit_id'),
                    'reconstruction_unit_id':ru.get('reconstruction_unit_id'),
                    'translation_unit_id':ru.get('translation_unit_id'),
                    'level':ru.get('level'),
                    'role':ru.get('role'),
                    'match_reason':reason,
                    'source_unit_ids':ru.get('source_unit_ids') or [],
                    'bbox':rb,
                    'source_text':norm(ru.get('text') or ru.get('source_text')),
                    'pagereconstruct_text':ru_text(ru),
                    'render_mode':((ru.get('render_contract') or {}).get('mode')
or ru.get('preservation_mode')),
                })
        rus.sort(key=lambda x:(x['bbox'][1] if x['bbox'] else 0, x['bbox'][0] if
x['bbox'] else 0, x.get('unit_id') or ''))
        row={
            'block_index':i,
            'block_id':bid,
            'reading_order':ro(b),
            'bbox':bb,
            'original_block_text':text_of(b),
            'original_children':original_children,
            'pagereconstruct_input_units':rus,
        }
        page_rows.append(row)
        lines.append(f'BLOCK {i:03d} | {bid} | bbox={bb}')
        lines.append('- ORIGINAL BLOCK TEXT:')
        lines.append(text_of(b) or '[EMPTY]')
        lines.append('- ORIGINAL CHILD TEXTS:')
        if original_children:
            for ch in original_children:
                lines.append(f"  [{ch['level']}] {ch['unit_id']} |
bbox={ch['bbox']}")
                lines.append(f"    {ch['text']}")
        else:
            lines.append('  [NONE]')
        lines.append('- PAGERECONSTRUCT_INPUT:')
        if rus:
            for ru in rus:
                lines.append(f"  [{ru['level']}] {ru['unit_id']} /
{ru['reconstruction_unit_id']} | match={ru['match_reason']} |
bbox={ru['bbox']}")
                lines.append(f"    source: {ru['source_text']}")
                lines.append(f"    render: {ru['pagereconstruct_text']}")
        else:
            lines.append('  [NONE]')
        lines.append('')
    json_path=OUT / f'block_text_compare_{tag}.json'
    txt_path=OUT / f'block_text_compare_{tag}.txt'
json_path.write_text(json.dumps({'page':page_no,'pdf':pdf,'blocks':page_rows},
ensure_ascii=False, indent=2), encoding='utf-8')
    txt_path.write_text('\n'.join(lines), encoding='utf-8')
index.append({'page':page_no,'pdf':pdf,'txt':str(txt_path),'json':str(json_path)
,'block_count':len(page_rows)})
    print(f'{tag}: blocks={len(page_rows)} -> {txt_path}')
(OUT / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2),
encoding='utf-8')
(OUT / 'README.txt').write_text('Comparaison block par block: ORIGINAL PagePrint
vs PAGERECONSTRUCT_INPUT views.reconstruction_units.\n', encoding='utf-8')
print('OUT', OUT)
PY

• Le job tourne encore sans sortie intermédiaire. Je vérifie s’il a déjà écrit
  des fichiers partiels.

• Explored
  └ List pagereconstruct_input_vs_original_text_by_block

────────────────────────────────────────────────────────────────────────────────

• Sept pages sont déjà écrites. J’attends les trois dernières et l’index final.

■ Conversation interrupted - tell the model what to do differently. Something
went wrong? Hit `/feedback` to report the issue.


› Justement on ne voit pas les traductions dans ton fichier . "render" redonnes
  les mêmes que sources. render doit donner exactement ce qui est rendu par
  pagetranslate/ (et non faire traduction lui-même)


• Tu as raison : mon fichier était invalide pour ce contrôle, car j’ai utilisé
  dry_run=True, donc render == source. Je corrige : je prends uniquement le
  translated_text produit par pagetranslate/; s’il n’existe pas, je l’indique,
  je ne le fabrique pas.

• Ran ps -p 49445 -o pid=,cmd=
  └ (no output)

• Explored
  └ Read run_pageprint_pagetranslate_audit.py
    Search *pagetranslate* in results

────────────────────────────────────────────────────────────────────────────────

• Je stoppe cette piste sèche : elle ne prouve rien. Je vais refaire avec un
  vrai appel pagetranslate non-dry-run, et le champ render sera strictement
  translated_text venant de pagetranslate/.

• Explored
  └ Read pagetranslate_test_docintelligence_p0406.json
    List ai_models
    Search model_inventory|opus_mt_tc_big|ct2 in translation_engines

────────────────────────────────────────────────────────────────────────────────

• Je vais utiliser le moteur ct2 local déjà configuré, pas une traduction
  inventée. Le fichier sera faux si pagetranslate ne renvoie pas
  translated_text, donc je marquerai ce cas explicitement.

• Explored
  └ Read model_inventory.json, run_pageprint_pagetranslate_audit.py, factory.py

────────────────────────────────────────────────────────────────────────────────

• Je génère dans un nouveau dossier pour ne pas mélanger avec les fichiers
  dry_run invalides. Le champ render sera exactement
  reconstruction_unit.translated_text.

────────────────────────────────────────────────────────────────────────────────

• Le vrai moteur tourne. Ça peut prendre plus longtemps que le dry-run, mais
  cette fois les render: viennent de pagetranslate/.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json, re, time
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate.builder import build_page_translation
from translation_engines import create_translation_engine
BASE = Path('results/new10_background_textures_20260616_batch2')
OUT = BASE / 'pagereconstruct_input_vs_original_text_by_block_REAL_TRANSLATED'
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((BASE / 'summary.json').read_text(encoding='utf-8'))
def norm(s): return re.sub(r'\s+', ' ', str(s or '')).strip()
def bbox(u): return (u.get('geometry') or {}).get('bbox') or u.get('bbox') or []
def text_of(u):
    c=u.get('content') or {}
    return norm(c.get('text') or u.get('text') or c.get('value'))
def ro(u): return (u.get('geometry') or {}).get('reading_order_index') or
u.get('reading_order_index') or 0
def descendants(uid, children):
    out=[]; stack=list(children.get(uid, [])); seen=set()
    while stack:
        cid=stack.pop(0)
        if cid in seen: continue
        seen.add(cid); out.append(cid); stack.extend(children.get(cid, []))
    return out
def overlap(a,b):
    if not a or not b or len(a)<4 or len(b)<4: return 0.0
    ax0,ay0,ax1,ay1=map(float,a[:4]); bx0,by0,bx1,by1=map(float,b[:4])
    ix0,iy0=max(ax0,bx0),max(ay0,by0); ix1,iy1=min(ax1,bx1),min(ay1,by1)
    if ix1<=ix0 or iy1<=iy0: return 0.0
    return ((ix1-ix0)*(iy1-iy0))/max(1.0,(ax1-ax0)*(ay1-ay0))
orch = PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(OUT / '_render'))
engine = create_translation_engine('ct2', inventory_path='ai_models/translation/
model_inventory.json', model_name='opus_mt_tc_big_en_fr', source_lang='en',
target_lang='fr')
index=[]
for item in summary:
    t0=time.time(); page_no=int(item['page']); pdf=item['pdf'];
tag=f'p{page_no:04d}'
    run=orch.run(pdf, pages=[0]); page=run['pages'][0]
    input_data=page.get('input_data') or page.get('pageprint') or
page.get('pageprint_input') or page
    translated=build_page_translation(input_data, translator=engine,
dry_run=False, source_lang='en', target_lang='fr', batch_size=8)
    translated_input=translated['translated_input_data']
    units=translated_input.get('units') or []
    unit_by_id={u.get('unit_id'):u for u in units if u.get('unit_id')}
    children={}
    for u in units:
        pid=u.get('parent_id') or u.get('parent_unit_id')
        if pid: children.setdefault(pid,[]).append(u.get('unit_id'))
        for cid in u.get('children_ids') or []:
children.setdefault(u.get('unit_id'),[]).append(cid)
    for k,v in list(children.items()): children[k]=list(dict.fromkeys([x for x
in v if x]))
    blocks=sorted([u for u in units if u.get('level')=='block' and text_of(u)],
key=ro)
    rus_all=(translated_input.get('views') or {}).get('reconstruction_units') or
[]
    rows=[]; lines=[]
    lines += [f'PAGE {tag} | {pdf}', '='*100, 'ORIGINAL = PagePrint block text',
'PAGERECONSTRUCT_INPUT.render = EXACTEMENT reconstruction_unit.translated_text
venant de pagetranslate/', '']
    for i,b in enumerate(blocks,1):
        bid=b.get('unit_id'); desc=set(descendants(bid, children));
src_ids={bid,*desc}; bb=bbox(b)
        child_rows=[]
        for cid in desc:
            cu=unit_by_id.get(cid) or {}
            if cu.get('level') in {'line','phrase','span'} and text_of(cu):
child_rows.append({'unit_id':cid,'level':cu.get('level'),'bbox':bbox(cu),'text':
text_of(cu),'reading_order':ro(cu)})
        child_rows.sort(key=lambda x:(x['reading_order'], x['bbox'][1] if
x['bbox'] else 0))
        ru_rows=[]
        for ru in rus_all:
            ids=set(ru.get('source_unit_ids') or [])
            reason='source_unit_ids' if ids & src_ids else ('bbox_overlap' if
overlap(ru.get('bbox') or ru.get('layout_bbox') or [], bb)>0.15 else None)
            if not reason: continue
            has_translated='translated_text' in ru and ru.get('translated_text')
is not None
            render = norm(ru.get('translated_text')) if has_translated else
'[MISSING_TRANSLATED_TEXT_FROM_PAGETRANSLATE]'
            ru_rows.append({
                'unit_id':ru.get('unit_id'),
'reconstruction_unit_id':ru.get('reconstruction_unit_id'),
'translation_unit_id':ru.get('translation_unit_id'),
                'level':ru.get('level'), 'role':ru.get('role'), 'status':
(ru.get('translation') or {}).get('status') or ru.get('status'),
                'match_reason':reason,
'source_unit_ids':ru.get('source_unit_ids') or [], 'bbox':ru.get('bbox') or
ru.get('layout_bbox') or [],
                'source':norm(ru.get('text') or ru.get('source_text')),
'render':render,
                'render_equals_source': render == norm(ru.get('text') or
ru.get('source_text')),
            })
        ru_rows.sort(key=lambda x:(x['bbox'][1] if x['bbox'] else 0, x['bbox']
[0] if x['bbox'] else 0, x.get('unit_id') or ''))
rows.append({'block_index':i,'block_id':bid,'bbox':bb,'original_block_text':text
_of(b),'original_children':child_rows,'pagereconstruct_input_units':ru_rows})
        lines += [f'BLOCK {i:03d} | {bid} | bbox={bb}', '- ORIGINAL BLOCK
TEXT:', text_of(b) or '[EMPTY]', '- ORIGINAL CHILD TEXTS:']
        if child_rows:
            for ch in child_rows: lines += [f"  [{ch['level']}] {ch['unit_id']}
| bbox={ch['bbox']}", f"    {ch['text']}"]
        else: lines.append('  [NONE]')
        lines.append('- PAGERECONSTRUCT_INPUT:')
        if ru_rows:
            for ru in ru_rows:
                lines += [f"  [{ru['level']}] {ru['unit_id']} /
{ru['reconstruction_unit_id']} | status={ru['status']} |
match={ru['match_reason']} | bbox={ru['bbox']}", f"    source: {ru['source']}",
f"    render: {ru['render']}"]
        else: lines.append('  [NONE]')
        lines.append('')
payload={'page':page_no,'pdf':pdf,'translation_statuses':translated.get('transla
tion_result') or {},'blocks':rows}
    jp=OUT/f'block_text_compare_{tag}.json'; tp=OUT/
f'block_text_compare_{tag}.txt'
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
encoding='utf-8')
    tp.write_text('\n'.join(lines), encoding='utf-8')
index.append({'page':page_no,'pdf':pdf,'txt':str(tp),'json':str(jp),'block_count
':len(rows),'seconds':round(time.time()-t0,1)})
    print(f'{tag}: blocks={len(rows)} seconds={round(time.time()-t0,1)}')
(OUT/'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2),
encoding='utf-8')
(OUT/'README.txt').write_text('REAL_TRANSLATED: render ==
reconstruction_unit.translated_text fourni par pagetranslate/. Aucune traduction
fabriquee dans ce script.\n', encoding='utf-8')
print('OUT', OUT)
PY

────────────────────────────────────────────────────────────────────────────────

• Toujours en calcul. Je vérifie s’il a déjà écrit les premières pages.

• Explored
  └ List pagereconstruct_input_vs_original_text_by_block_REAL_TRANSLATED

────────────────────────────────────────────────────────────────────────────────

• Trois pages sont déjà disponibles dans le nouveau dossier REAL_TRANSLATED.
  J’attends les sept restantes.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json, re, time
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate.builder import build_page_translation
from translation_engines import create_translation_engine
BASE = Path('results/new10_background_textures_20260616_batch2')
OUT = BASE / 'pagereconstruct_input_vs_original_text_by_block_REAL_TRANSLATED'
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((BASE / 'summary.json').read_text(encoding='utf-8'))
def norm(s): return re.sub(r'\s+', ' ', str(s or '')).strip()
def bbox(u): return (u.get('geometry') or {}).get('bbox') or u.get('bbox') or []
def text_of(u):
    c=u.get('content') or {}
    return norm(c.get('text') or u.get('text') or c.get('value'))
def ro(u): return (u.get('geometry') or {}).get('reading_order_index') or
u.get('reading_order_index') or 0
def descendants(uid, children):
    out=[]; stack=list(children.get(uid, [])); seen=set()
    while stack:
        cid=stack.pop(0)
        if cid in seen: continue
        seen.add(cid); out.append(cid); stack.extend(children.get(cid, []))
    return out
def overlap(a,b):
    if not a or not b or len(a)<4 or len(b)<4: return 0.0
    ax0,ay0,ax1,ay1=map(float,a[:4]); bx0,by0,bx1,by1=map(float,b[:4])
    ix0,iy0=max(ax0,bx0),max(ay0,by0); ix1,iy1=min(ax1,bx1),min(ay1,by1)
    if ix1<=ix0 or iy1<=iy0: return 0.0
    return ((ix1-ix0)*(iy1-iy0))/max(1.0,(ax1-ax0)*(ay1-ay0))
orch = PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(OUT / '_render'))
engine = create_translation_engine('ct2', inventory_path='ai_models/translation/
model_inventory.json', model_name='opus_mt_tc_big_en_fr', source_lang='en',
target_lang='fr')
index=[]
for item in summary:
    t0=time.time(); page_no=int(item['page']); pdf=item['pdf'];
tag=f'p{page_no:04d}'
    run=orch.run(pdf, pages=[0]); page=run['pages'][0]
    input_data=page.get('input_data') or page.get('pageprint') or
page.get('pageprint_input') or page
    translated=build_page_translation(input_data, translator=engine,
dry_run=False, source_lang='en', target_lang='fr', batch_size=8)
    translated_input=translated['translated_input_data']
    units=translated_input.get('units') or []
    unit_by_id={u.get('unit_id'):u for u in units if u.get('unit_id')}
    children={}
    for u in units:
        pid=u.get('parent_id') or u.get('parent_unit_id')
        if pid: children.setdefault(pid,[]).append(u.get('unit_id'))
        for cid in u.get('children_ids') or []:
children.setdefault(u.get('unit_id'),[]).append(cid)
    for k,v in list(children.items()): children[k]=list(dict.fromkeys([x for x
in v if x]))
    blocks=sorted([u for u in units if u.get('level')=='block' and text_of(u)],
key=ro)
    rus_all=(translated_input.get('views') or {}).get('reconstruction_units') or
[]
    rows=[]; lines=[]
    lines += [f'PAGE {tag} | {pdf}', '='*100, 'ORIGINAL = PagePrint block text',
'PAGERECONSTRUCT_INPUT.render = EXACTEMENT reconstruction_unit.translated_text
venant de pagetranslate/', '']
    for i,b in enumerate(blocks,1):
        bid=b.get('unit_id'); desc=set(descendants(bid, children));
src_ids={bid,*desc}; bb=bbox(b)
        child_rows=[]
        for cid in desc:
            cu=unit_by_id.get(cid) or {}
            if cu.get('level') in {'line','phrase','span'} and text_of(cu):
child_rows.append({'unit_id':cid,'level':cu.get('level'),'bbox':bbox(cu),'text':
text_of(cu),'reading_order':ro(cu)})
        child_rows.sort(key=lambda x:(x['reading_order'], x['bbox'][1] if
x['bbox'] else 0))
        ru_rows=[]
        for ru in rus_all:
            ids=set(ru.get('source_unit_ids') or [])
            reason='source_unit_ids' if ids & src_ids else ('bbox_overlap' if
overlap(ru.get('bbox') or ru.get('layout_bbox') or [], bb)>0.15 else None)
            if not reason: continue
            has_translated='translated_text' in ru and ru.get('translated_text')
is not None
            render = norm(ru.get('translated_text')) if has_translated else
'[MISSING_TRANSLATED_TEXT_FROM_PAGETRANSLATE]'
            ru_rows.append({
                'unit_id':ru.get('unit_id'),
'reconstruction_unit_id':ru.get('reconstruction_unit_id'),
'translation_unit_id':ru.get('translation_unit_id'),
                'level':ru.get('level'), 'role':ru.get('role'), 'status':
(ru.get('translation') or {}).get('status') or ru.get('status'),
                'match_reason':reason,
'source_unit_ids':ru.get('source_unit_ids') or [], 'bbox':ru.get('bbox') or
ru.get('layout_bbox') or [],
                'source':norm(ru.get('text') or ru.get('source_text')),
'render':render,
                'render_equals_source': render == norm(ru.get('text') or
ru.get('source_text')),
            })
        ru_rows.sort(key=lambda x:(x['bbox'][1] if x['bbox'] else 0, x['bbox']
[0] if x['bbox'] else 0, x.get('unit_id') or ''))
rows.append({'block_index':i,'block_id':bid,'bbox':bb,'original_block_text':text
_of(b),'original_children':child_rows,'pagereconstruct_input_units':ru_rows})
        lines += [f'BLOCK {i:03d} | {bid} | bbox={bb}', '- ORIGINAL BLOCK
TEXT:', text_of(b) or '[EMPTY]', '- ORIGINAL CHILD TEXTS:']
        if child_rows:
            for ch in child_rows: lines += [f"  [{ch['level']}] {ch['unit_id']}
| bbox={ch['bbox']}", f"    {ch['text']}"]
        else: lines.append('  [NONE]')
        lines.append('- PAGERECONSTRUCT_INPUT:')
        if ru_rows:
            for ru in ru_rows:
                lines += [f"  [{ru['level']}] {ru['unit_id']} /
{ru['reconstruction_unit_id']} | status={ru['status']} |
match={ru['match_reason']} | bbox={ru['bbox']}", f"    source: {ru['source']}",
f"    render: {ru['render']}"]
        else: lines.append('  [NONE]')
        lines.append('')
payload={'page':page_no,'pdf':pdf,'translation_statuses':translated.get('transla
tion_result') or {},'blocks':rows}
    jp=OUT/f'block_text_compare_{tag}.json'; tp=OUT/
f'block_text_compare_{tag}.txt'
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
encoding='utf-8')
    tp.write_text('\n'.join(lines), encoding='utf-8')
index.append({'page':page_no,'pdf':pdf,'txt':str(tp),'json':str(jp),'block_count
':len(rows),'seconds':round(time.time()-t0,1)})
    print(f'{tag}: blocks={len(rows)} seconds={round(time.time()-t0,1)}')
(OUT/'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2),
encoding='utf-8')
(OUT/'README.txt').write_text('REAL_TRANSLATED: render ==
reconstruction_unit.translated_text fourni par pagetranslate/. Aucune traduction
fabriquee dans ce script.\n', encoding='utf-8')
print('OUT', OUT)
PY

────────────────────────────────────────────────────────────────────────────────

• Le moteur continue. Je laisse finir pour éviter un dossier incomplet.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json, re, time
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate.builder import build_page_translation
from translation_engines import create_translation_engine
BASE = Path('results/new10_background_textures_20260616_batch2')
OUT = BASE / 'pagereconstruct_input_vs_original_text_by_block_REAL_TRANSLATED'
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((BASE / 'summary.json').read_text(encoding='utf-8'))
def norm(s): return re.sub(r'\s+', ' ', str(s or '')).strip()
def bbox(u): return (u.get('geometry') or {}).get('bbox') or u.get('bbox') or []
def text_of(u):
    c=u.get('content') or {}
    return norm(c.get('text') or u.get('text') or c.get('value'))
def ro(u): return (u.get('geometry') or {}).get('reading_order_index') or
u.get('reading_order_index') or 0
def descendants(uid, children):
    out=[]; stack=list(children.get(uid, [])); seen=set()
    while stack:
        cid=stack.pop(0)
        if cid in seen: continue
        seen.add(cid); out.append(cid); stack.extend(children.get(cid, []))
    return out
def overlap(a,b):
    if not a or not b or len(a)<4 or len(b)<4: return 0.0
    ax0,ay0,ax1,ay1=map(float,a[:4]); bx0,by0,bx1,by1=map(float,b[:4])
    ix0,iy0=max(ax0,bx0),max(ay0,by0); ix1,iy1=min(ax1,bx1),min(ay1,by1)
    if ix1<=ix0 or iy1<=iy0: return 0.0
    return ((ix1-ix0)*(iy1-iy0))/max(1.0,(ax1-ax0)*(ay1-ay0))
orch = PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(OUT / '_render'))
engine = create_translation_engine('ct2', inventory_path='ai_models/translation/
model_inventory.json', model_name='opus_mt_tc_big_en_fr', source_lang='en',
target_lang='fr')
index=[]
for item in summary:
    t0=time.time(); page_no=int(item['page']); pdf=item['pdf'];
tag=f'p{page_no:04d}'
    run=orch.run(pdf, pages=[0]); page=run['pages'][0]
    input_data=page.get('input_data') or page.get('pageprint') or
page.get('pageprint_input') or page
    translated=build_page_translation(input_data, translator=engine,
dry_run=False, source_lang='en', target_lang='fr', batch_size=8)
    translated_input=translated['translated_input_data']
    units=translated_input.get('units') or []
    unit_by_id={u.get('unit_id'):u for u in units if u.get('unit_id')}
    children={}
    for u in units:
        pid=u.get('parent_id') or u.get('parent_unit_id')
        if pid: children.setdefault(pid,[]).append(u.get('unit_id'))
        for cid in u.get('children_ids') or []:
children.setdefault(u.get('unit_id'),[]).append(cid)
    for k,v in list(children.items()): children[k]=list(dict.fromkeys([x for x
in v if x]))
    blocks=sorted([u for u in units if u.get('level')=='block' and text_of(u)],
key=ro)
    rus_all=(translated_input.get('views') or {}).get('reconstruction_units') or
[]
    rows=[]; lines=[]
    lines += [f'PAGE {tag} | {pdf}', '='*100, 'ORIGINAL = PagePrint block text',
'PAGERECONSTRUCT_INPUT.render = EXACTEMENT reconstruction_unit.translated_text
venant de pagetranslate/', '']
    for i,b in enumerate(blocks,1):
        bid=b.get('unit_id'); desc=set(descendants(bid, children));
src_ids={bid,*desc}; bb=bbox(b)
        child_rows=[]
        for cid in desc:
            cu=unit_by_id.get(cid) or {}
            if cu.get('level') in {'line','phrase','span'} and text_of(cu):
child_rows.append({'unit_id':cid,'level':cu.get('level'),'bbox':bbox(cu),'text':
text_of(cu),'reading_order':ro(cu)})
        child_rows.sort(key=lambda x:(x['reading_order'], x['bbox'][1] if
x['bbox'] else 0))
        ru_rows=[]
        for ru in rus_all:
            ids=set(ru.get('source_unit_ids') or [])
            reason='source_unit_ids' if ids & src_ids else ('bbox_overlap' if
overlap(ru.get('bbox') or ru.get('layout_bbox') or [], bb)>0.15 else None)
            if not reason: continue
            has_translated='translated_text' in ru and ru.get('translated_text')
is not None
            render = norm(ru.get('translated_text')) if has_translated else
'[MISSING_TRANSLATED_TEXT_FROM_PAGETRANSLATE]'
            ru_rows.append({
                'unit_id':ru.get('unit_id'),
'reconstruction_unit_id':ru.get('reconstruction_unit_id'),
'translation_unit_id':ru.get('translation_unit_id'),
                'level':ru.get('level'), 'role':ru.get('role'), 'status':
(ru.get('translation') or {}).get('status') or ru.get('status'),
                'match_reason':reason,
'source_unit_ids':ru.get('source_unit_ids') or [], 'bbox':ru.get('bbox') or
ru.get('layout_bbox') or [],
                'source':norm(ru.get('text') or ru.get('source_text')),
'render':render,
                'render_equals_source': render == norm(ru.get('text') or
ru.get('source_text')),
            })
        ru_rows.sort(key=lambda x:(x['bbox'][1] if x['bbox'] else 0, x['bbox']
[0] if x['bbox'] else 0, x.get('unit_id') or ''))
rows.append({'block_index':i,'block_id':bid,'bbox':bb,'original_block_text':text
_of(b),'original_children':child_rows,'pagereconstruct_input_units':ru_rows})
        lines += [f'BLOCK {i:03d} | {bid} | bbox={bb}', '- ORIGINAL BLOCK
TEXT:', text_of(b) or '[EMPTY]', '- ORIGINAL CHILD TEXTS:']
        if child_rows:
            for ch in child_rows: lines += [f"  [{ch['level']}] {ch['unit_id']}
| bbox={ch['bbox']}", f"    {ch['text']}"]
        else: lines.append('  [NONE]')
        lines.append('- PAGERECONSTRUCT_INPUT:')
        if ru_rows:
            for ru in ru_rows:
                lines += [f"  [{ru['level']}] {ru['unit_id']} /
{ru['reconstruction_unit_id']} | status={ru['status']} |
match={ru['match_reason']} | bbox={ru['bbox']}", f"    source: {ru['source']}",
f"    render: {ru['render']}"]
        else: lines.append('  [NONE]')
        lines.append('')
payload={'page':page_no,'pdf':pdf,'translation_statuses':translated.get('transla
tion_result') or {},'blocks':rows}
    jp=OUT/f'block_text_compare_{tag}.json'; tp=OUT/
f'block_text_compare_{tag}.txt'
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
encoding='utf-8')
    tp.write_text('\n'.join(lines), encoding='utf-8')
index.append({'page':page_no,'pdf':pdf,'txt':str(tp),'json':str(jp),'block_count
':len(rows),'seconds':round(time.time()-t0,1)})
    print(f'{tag}: blocks={len(rows)} seconds={round(time.time()-t0,1)}')
(OUT/'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2),
encoding='utf-8')
(OUT/'README.txt').write_text('REAL_TRANSLATED: render ==
reconstruction_unit.translated_text fourni par pagetranslate/. Aucune traduction
fabriquee dans ce script.\n', encoding='utf-8')
print('OUT', OUT)
PY

────────────────────────────────────────────────────────────────────────────────

• Toujours en cours. Je regarde l’avancement fichier sans relancer de calcul.

• Explored
  └ Search block_text_compare_*.txt in
           pagereconstruct_input_vs_original_text_by_block_REAL_TRANSLATED

────────────────────────────────────────────────────────────────────────────────

• Six pages sur dix sont écrites. Je continue jusqu’à l’index final.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json, re, time
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate.builder import build_page_translation
from translation_engines import create_translation_engine
BASE = Path('results/new10_background_textures_20260616_batch2')
OUT = BASE / 'pagereconstruct_input_vs_original_text_by_block_REAL_TRANSLATED'
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((BASE / 'summary.json').read_text(encoding='utf-8'))
def norm(s): return re.sub(r'\s+', ' ', str(s or '')).strip()
def bbox(u): return (u.get('geometry') or {}).get('bbox') or u.get('bbox') or []
def text_of(u):
    c=u.get('content') or {}
    return norm(c.get('text') or u.get('text') or c.get('value'))
def ro(u): return (u.get('geometry') or {}).get('reading_order_index') or
u.get('reading_order_index') or 0
def descendants(uid, children):
    out=[]; stack=list(children.get(uid, [])); seen=set()
    while stack:
        cid=stack.pop(0)
        if cid in seen: continue
        seen.add(cid); out.append(cid); stack.extend(children.get(cid, []))
    return out
def overlap(a,b):
    if not a or not b or len(a)<4 or len(b)<4: return 0.0
    ax0,ay0,ax1,ay1=map(float,a[:4]); bx0,by0,bx1,by1=map(float,b[:4])
    ix0,iy0=max(ax0,bx0),max(ay0,by0); ix1,iy1=min(ax1,bx1),min(ay1,by1)
    if ix1<=ix0 or iy1<=iy0: return 0.0
    return ((ix1-ix0)*(iy1-iy0))/max(1.0,(ax1-ax0)*(ay1-ay0))
orch = PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(OUT / '_render'))
engine = create_translation_engine('ct2', inventory_path='ai_models/translation/
model_inventory.json', model_name='opus_mt_tc_big_en_fr', source_lang='en',
target_lang='fr')
index=[]
for item in summary:
    t0=time.time(); page_no=int(item['page']); pdf=item['pdf'];
tag=f'p{page_no:04d}'
    run=orch.run(pdf, pages=[0]); page=run['pages'][0]
    input_data=page.get('input_data') or page.get('pageprint') or
page.get('pageprint_input') or page
    translated=build_page_translation(input_data, translator=engine,
dry_run=False, source_lang='en', target_lang='fr', batch_size=8)
    translated_input=translated['translated_input_data']
    units=translated_input.get('units') or []
    unit_by_id={u.get('unit_id'):u for u in units if u.get('unit_id')}
    children={}
    for u in units:
        pid=u.get('parent_id') or u.get('parent_unit_id')
        if pid: children.setdefault(pid,[]).append(u.get('unit_id'))
        for cid in u.get('children_ids') or []:
children.setdefault(u.get('unit_id'),[]).append(cid)
    for k,v in list(children.items()): children[k]=list(dict.fromkeys([x for x
in v if x]))
    blocks=sorted([u for u in units if u.get('level')=='block' and text_of(u)],
key=ro)
    rus_all=(translated_input.get('views') or {}).get('reconstruction_units') or
[]
    rows=[]; lines=[]
    lines += [f'PAGE {tag} | {pdf}', '='*100, 'ORIGINAL = PagePrint block text',
'PAGERECONSTRUCT_INPUT.render = EXACTEMENT reconstruction_unit.translated_text
venant de pagetranslate/', '']
    for i,b in enumerate(blocks,1):
        bid=b.get('unit_id'); desc=set(descendants(bid, children));
src_ids={bid,*desc}; bb=bbox(b)
        child_rows=[]
        for cid in desc:
            cu=unit_by_id.get(cid) or {}
            if cu.get('level') in {'line','phrase','span'} and text_of(cu):
child_rows.append({'unit_id':cid,'level':cu.get('level'),'bbox':bbox(cu),'text':
text_of(cu),'reading_order':ro(cu)})
        child_rows.sort(key=lambda x:(x['reading_order'], x['bbox'][1] if
x['bbox'] else 0))
        ru_rows=[]
        for ru in rus_all:
            ids=set(ru.get('source_unit_ids') or [])
            reason='source_unit_ids' if ids & src_ids else ('bbox_overlap' if
overlap(ru.get('bbox') or ru.get('layout_bbox') or [], bb)>0.15 else None)
            if not reason: continue
            has_translated='translated_text' in ru and ru.get('translated_text')
is not None
            render = norm(ru.get('translated_text')) if has_translated else
'[MISSING_TRANSLATED_TEXT_FROM_PAGETRANSLATE]'
            ru_rows.append({
                'unit_id':ru.get('unit_id'),
'reconstruction_unit_id':ru.get('reconstruction_unit_id'),
'translation_unit_id':ru.get('translation_unit_id'),
                'level':ru.get('level'), 'role':ru.get('role'), 'status':
(ru.get('translation') or {}).get('status') or ru.get('status'),
                'match_reason':reason,
'source_unit_ids':ru.get('source_unit_ids') or [], 'bbox':ru.get('bbox') or
ru.get('layout_bbox') or [],
                'source':norm(ru.get('text') or ru.get('source_text')),
'render':render,
                'render_equals_source': render == norm(ru.get('text') or
ru.get('source_text')),
            })
        ru_rows.sort(key=lambda x:(x['bbox'][1] if x['bbox'] else 0, x['bbox']
[0] if x['bbox'] else 0, x.get('unit_id') or ''))
rows.append({'block_index':i,'block_id':bid,'bbox':bb,'original_block_text':text
_of(b),'original_children':child_rows,'pagereconstruct_input_units':ru_rows})
        lines += [f'BLOCK {i:03d} | {bid} | bbox={bb}', '- ORIGINAL BLOCK
TEXT:', text_of(b) or '[EMPTY]', '- ORIGINAL CHILD TEXTS:']
        if child_rows:
            for ch in child_rows: lines += [f"  [{ch['level']}] {ch['unit_id']}
| bbox={ch['bbox']}", f"    {ch['text']}"]
        else: lines.append('  [NONE]')
        lines.append('- PAGERECONSTRUCT_INPUT:')
        if ru_rows:
            for ru in ru_rows:
                lines += [f"  [{ru['level']}] {ru['unit_id']} /
{ru['reconstruction_unit_id']} | status={ru['status']} |
match={ru['match_reason']} | bbox={ru['bbox']}", f"    source: {ru['source']}",
f"    render: {ru['render']}"]
        else: lines.append('  [NONE]')
        lines.append('')
payload={'page':page_no,'pdf':pdf,'translation_statuses':translated.get('transla
tion_result') or {},'blocks':rows}
    jp=OUT/f'block_text_compare_{tag}.json'; tp=OUT/
f'block_text_compare_{tag}.txt'
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
encoding='utf-8')
    tp.write_text('\n'.join(lines), encoding='utf-8')
index.append({'page':page_no,'pdf':pdf,'txt':str(tp),'json':str(jp),'block_count
':len(rows),'seconds':round(time.time()-t0,1)})
    print(f'{tag}: blocks={len(rows)} seconds={round(time.time()-t0,1)}')
(OUT/'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2),
encoding='utf-8')
(OUT/'README.txt').write_text('REAL_TRANSLATED: render ==
reconstruction_unit.translated_text fourni par pagetranslate/. Aucune traduction
fabriquee dans ce script.\n', encoding='utf-8')
print('OUT', OUT)
PY

────────────────────────────────────────────────────────────────────────────────

• Encore en cours, sans erreur. Le coût principal est la vraie traduction
  locale, pas la génération du fichier.

• Waited for background terminal · .docs-parser/bin/python - <<'PY'
from pathlib import Path
import json, re, time
from pipelines.orchestrator import PipelineOrchestrator
from pagetranslate.builder import build_page_translation
from translation_engines import create_translation_engine
BASE = Path('results/new10_background_textures_20260616_batch2')
OUT = BASE / 'pagereconstruct_input_vs_original_text_by_block_REAL_TRANSLATED'
OUT.mkdir(parents=True, exist_ok=True)
summary = json.loads((BASE / 'summary.json').read_text(encoding='utf-8'))
def norm(s): return re.sub(r'\s+', ' ', str(s or '')).strip()
def bbox(u): return (u.get('geometry') or {}).get('bbox') or u.get('bbox') or []
def text_of(u):
    c=u.get('content') or {}
    return norm(c.get('text') or u.get('text') or c.get('value'))
def ro(u): return (u.get('geometry') or {}).get('reading_order_index') or
u.get('reading_order_index') or 0
def descendants(uid, children):
    out=[]; stack=list(children.get(uid, [])); seen=set()
    while stack:
        cid=stack.pop(0)
        if cid in seen: continue
        seen.add(cid); out.append(cid); stack.extend(children.get(cid, []))
    return out
def overlap(a,b):
    if not a or not b or len(a)<4 or len(b)<4: return 0.0
    ax0,ay0,ax1,ay1=map(float,a[:4]); bx0,by0,bx1,by1=map(float,b[:4])
    ix0,iy0=max(ax0,bx0),max(ay0,by0); ix1,iy1=min(ax1,bx1),min(ay1,by1)
    if ix1<=ix0 or iy1<=iy0: return 0.0
    return ((ix1-ix0)*(iy1-iy0))/max(1.0,(ax1-ax0)*(ay1-ay0))
orch = PipelineOrchestrator(enable_ocr=False, enable_understanding=True,
enable_postprocessors=True, enable_special_regions=True,
save_render_dir=str(OUT / '_render'))
engine = create_translation_engine('ct2', inventory_path='ai_models/translation/
model_inventory.json', model_name='opus_mt_tc_big_en_fr', source_lang='en',
target_lang='fr')
index=[]
for item in summary:
    t0=time.time(); page_no=int(item['page']); pdf=item['pdf'];
tag=f'p{page_no:04d}'
    run=orch.run(pdf, pages=[0]); page=run['pages'][0]
    input_data=page.get('input_data') or page.get('pageprint') or
page.get('pageprint_input') or page
    translated=build_page_translation(input_data, translator=engine,
dry_run=False, source_lang='en', target_lang='fr', batch_size=8)
    translated_input=translated['translated_input_data']
    units=translated_input.get('units') or []
    unit_by_id={u.get('unit_id'):u for u in units if u.get('unit_id')}
    children={}
    for u in units:
        pid=u.get('parent_id') or u.get('parent_unit_id')
        if pid: children.setdefault(pid,[]).append(u.get('unit_id'))
        for cid in u.get('children_ids') or []:
children.setdefault(u.get('unit_id'),[]).append(cid)
    for k,v in list(children.items()): children[k]=list(dict.fromkeys([x for x
in v if x]))
    blocks=sorted([u for u in units if u.get('level')=='block' and text_of(u)],
key=ro)
    rus_all=(translated_input.get('views') or {}).get('reconstruction_units') or
[]
    rows=[]; lines=[]
    lines += [f'PAGE {tag} | {pdf}', '='*100, 'ORIGINAL = PagePrint block text',
'PAGERECONSTRUCT_INPUT.render = EXACTEMENT reconstruction_unit.translated_text
venant de pagetranslate/', '']
    for i,b in enumerate(blocks,1):
        bid=b.get('unit_id'); desc=set(descendants(bid, children));
src_ids={bid,*desc}; bb=bbox(b)
        child_rows=[]
        for cid in desc:
            cu=unit_by_id.get(cid) or {}
            if cu.get('level') in {'line','phrase','span'} and text_of(cu):
child_rows.append({'unit_id':cid,'level':cu.get('level'),'bbox':bbox(cu),'text':
text_of(cu),'reading_order':ro(cu)})
        child_rows.sort(key=lambda x:(x['reading_order'], x['bbox'][1] if
x['bbox'] else 0))
        ru_rows=[]
        for ru in rus_all:
            ids=set(ru.get('source_unit_ids') or [])
            reason='source_unit_ids' if ids & src_ids else ('bbox_overlap' if
overlap(ru.get('bbox') or ru.get('layout_bbox') or [], bb)>0.15 else None)
            if not reason: continue
            has_translated='translated_text' in ru and ru.get('translated_text')
is not None
            render = norm(ru.get('translated_text')) if has_translated else
'[MISSING_TRANSLATED_TEXT_FROM_PAGETRANSLATE]'
            ru_rows.append({
                'unit_id':ru.get('unit_id'),
'reconstruction_unit_id':ru.get('reconstruction_unit_id'),
'translation_unit_id':ru.get('translation_unit_id'),
                'level':ru.get('level'), 'role':ru.get('role'), 'status':
(ru.get('translation') or {}).get('status') or ru.get('status'),
                'match_reason':reason,
'source_unit_ids':ru.get('source_unit_ids') or [], 'bbox':ru.get('bbox') or
ru.get('layout_bbox') or [],
                'source':norm(ru.get('text') or ru.get('source_text')),
'render':render,
                'render_equals_source': render == norm(ru.get('text') or
ru.get('source_text')),
            })
        ru_rows.sort(key=lambda x:(x['bbox'][1] if x['bbox'] else 0, x['bbox']
[0] if x['bbox'] else 0, x.get('unit_id') or ''))
rows.append({'block_index':i,'block_id':bid,'bbox':bb,'original_block_text':text
_of(b),'original_children':child_rows,'pagereconstruct_input_units':ru_rows})
        lines += [f'BLOCK {i:03d} | {bid} | bbox={bb}', '- ORIGINAL BLOCK
TEXT:', text_of(b) or '[EMPTY]', '- ORIGINAL CHILD TEXTS:']
        if child_rows:
            for ch in child_rows: lines += [f"  [{ch['level']}] {ch['unit_id']}
| bbox={ch['bbox']}", f"    {ch['text']}"]
        else: lines.append('  [NONE]')
        lines.append('- PAGERECONSTRUCT_INPUT:')
        if ru_rows:
            for ru in ru_rows:
                lines += [f"  [{ru['level']}] {ru['unit_id']} /
{ru['reconstruction_unit_id']} | status={ru['status']} |
match={ru['match_reason']} | bbox={ru['bbox']}", f"    source: {ru['source']}",
f"    render: {ru['render']}"]
        else: lines.append('  [NONE]')
        lines.append('')
payload={'page':page_no,'pdf':pdf,'translation_statuses':translated.get('transla
tion_result') or {},'blocks':rows}
    jp=OUT/f'block_text_compare_{tag}.json'; tp=OUT/
f'block_text_compare_{tag}.txt'
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
encoding='utf-8')
    tp.write_text('\n'.join(lines), encoding='utf-8')
index.append({'page':page_no,'pdf':pdf,'txt':str(tp),'json':str(jp),'block_count
':len(rows),'seconds':round(time.time()-t0,1)})
    print(f'{tag}: blocks={len(rows)} seconds={round(time.time()-t0,1)}')
(OUT/'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2),
encoding='utf-8')
(OUT/'README.txt').write_text('REAL_TRANSLATED: render ==
reconstruction_unit.translated_text fourni par pagetranslate/. Aucune traduction
fabriquee dans ce script.\n', encoding='utf-8')
print('OUT', OUT)
PY

• Ran .docs-parser/bin/python - <<'PY'
  │ from pathlib import Path
  │ import json, re, time
  │ … +144 lines
  └ /home/raphael/Mes_Projets/docs_parser/.docs-parser/lib/python3.11/
    site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3
    … +13 lines (ctrl + t to view transcript)
    OUT results/new10_background_textures_20260616_batch2/
    pagereconstruct_input_vs_original_text_by_block_REAL_TRANSLATED

────────────────────────────────────────────────────────────────────────────────

• Les 10 fichiers sont générés. Je fais juste une vérification rapide sur
  l’index et un extrait, puis je te donne les chemins.

• Explored
  └ List pagereconstruct_input_vs_original_text_by_block_REAL_TRANSLATED
    Read block_text_compare_p0279.txt, index.json

────────────────────────────────────────────────────────────────────────────────

• C’est refait correctement avec vraie sortie pagetranslate/, pas dry_run.

  Dossier :
  results/new10_background_textures_20260616_batch2/
  pagereconstruct_input_vs_original_text_by_block_REAL_TRANSLATED

  Fichiers clés :

  - results/new10_background_textures_20260616_batch2/
    pagereconstruct_input_vs_original_text_by_block_REAL_TRANSLATED/index.json

  - results/new10_background_textures_20260616_batch2/
    pagereconstruct_input_vs_original_text_by_block_REAL_TRANSLATED/
    block_text_compare_p0279.txt

  - un .txt et un .json par page.

  Important : on voit déjà le vrai problème. Exemple p0279, block 001 : il y a à
  la fois des unités translated avec render: CHAPITRE... et des unités preserved
  avec render: Transfer learning / HAPTER. Donc le bug est bien preserved +
  translated dans les entrées vers reconstruction.

─ Worked for 5m 22s ────────────────────────────────────────────────────────────


› Ce n'est pas ça le problème ...
  Est ce que tout le textes extrait de pageprint/ est passé dans pagetranslate/
  et reconstruits dans pagereconstruct/ et visible dans le rendu visuel ? c'est
  peut importe si il est traduit ou non ? ça la vraie question ?


• Oui. C’est exactement la vraie question.

  La règle correcte n’est pas seulement :

  présent dans pagetranslate/

  Mais :

  chaque texte extrait par pageprint/
  → a une décision pagetranslate
  → entre dans pagereconstruct
  → produit un TextOp ou PreservationOp
  → est visible dans le rendu final

  Peu importe son état :

  traduisible     → rendu avec translated_text
  non traduisible → rendu tel quel
  protégé         → rendu tel quel
  exclu           → seulement avec raison valide explicite

  Donc non, mes fichiers précédents ne répondaient pas entièrement à la vraie
  question. Ils comparaient surtout :

  PagePrint original
  vs
  entrée pagereconstruct

  Mais ils ne prouvaient pas encore :

  visible dans le rendu visuel final

  Le bon audit doit être une matrice par unité PagePrint :

  source_unit_id
  source_text
  pageprint_bbox
  pagetranslate_status
  translated_text / preserved_text
  reconstruction_unit_id
  textop_id
  final_bbox
  visual_verified
  missing_reason

  Et la page est KO si une seule unité texte PagePrint n’a pas :

  translated OR preserved OR explicitly_excluded_valid
  +
  rendered
  +
  visually_verified


› et donc ? La réponse dans le projet ? Réponds court, protèges mes tokens


• Réponse courte : non, pas encore garanti dans le projet.

  Aujourd’hui le projet vérifie partiellement :

  - pagetranslate/functional_validator.py : couverture logique PagePrint →
    PageTranslate.

  - pagetranslate/projection.py : projection vers views.reconstruction_units.
  - pagereconstruct/plan_compiler.py / intrablock_composer.py : création des
    placements.

  - render_ops.py : génération des opérations de rendu.
  - pubready/stages/visual_image_audit.py : audit visuel global.

  Mais il manque l’invariant central :

  chaque texte PagePrint
  → décision pagetranslate
  → unité pagereconstruct
  → TextOp/PreservationOp
  → visible dans image finale

  Donc la vraie correction à faire est un ledger global par texte source, pas
  juste des bboxes :

  SourceTextLifecycleLedger

  avec blocage si une unité PagePrint n’arrive pas jusqu’au rendu visible.
  Actuellement, le projet ne le prouve pas à 100%.


› On fait comment ? Regardes si il n'y a pas déjà ces implémentations, sinon
  fais le!


• Je vais d’abord chercher l’existant : coverage/ledger/source unit/render
  visibility. Si ça n’existe pas bout en bout, j’ajoute un audit minimal et
  bloquant sans refaire tout le pipeline.

• Explored
  └ Search Lifecycle|TextRemovalLedger|SourceUnitState|source.*coverage|
           visual_verified|TextOp|PreservationOp|original_text_coverage|
           missing_disposition|rendered|visible in pagetranslate
    Search class .*Audit|def .*audit|PagePublicationReady|hard_block|blocker in
           pubready
    Search class TextOp|@dataclass.*Text|render_ops|text_ops|preservation in
           pagereconstruct

────────────────────────────────────────────────────────────────────────────────

• Constat rapide : il existe des morceaux (original_text_coverage,
  TextRemovalLedger, SourceUnitState, render_ops_audit), mais pas le ledger
  complet PagePrint → PageTranslate → PageReconstruct → RenderOps → visibilité.
  Je vais l’ajouter comme audit dédié.

• Explored
  └ Read render_ops.py, render_ops_audit.py, page_auditor.py,
         functional_validator.py, plan_compiler.py, schema.py, input_adapter.py

────────────────────────────────────────────────────────────────────────────────

• Je vais écrire le test d’abord : une unité PagePrint texte qui passe par
  reconstruction_units mais n’a aucun TextOp/PreservationOp doit bloquer.
  Ensuite j’implémente l’audit.

• Explored
  └ Read schema.py

────────────────────────────────────────────────────────────────────────────────

• Je vais garder ça pragmatique : l’audit vérifiera le cycle complet au niveau
  plan/opérations. La visibilité image réelle reste validée par
  visual_image_audit, déjà obligatoire en publication.

• Explored
  └ Read preservation_contract.py, overlay_manager.py

• Ran sed -n '1,120p' tests/pubready/test_render_ops_audit.py 2>/dev/null ||
  │ true && sed -n '1,120p' tests/pagereconstruct/test_render_ops.py
  └ """Phase 6 — RenderOps gelées + backends exécuteurs (zéro dispatch
    backend)."""
    … +45 lines (ctrl + t to view transcript)
        if "text" in order and "patch" in order:
            assert order.index("patch") < order.index("text")

────────────────────────────────────────────────────────────────────────────────

• Je vais maintenant ajouter le ledger SourceTextLifecycle: une ligne par texte
  PagePrint, avec décision pagetranslate, entrée reconstruction, op de rendu/
  preservation. Puis je le branche dans render_ops_audit.

• Added tests/pubready/test_source_text_lifecycle_audit.py (+82 -0)
     1 +from pagereconstruct.source_text_lifecycle_ledger import audit_source_te
        xt_lifecycle
     2 +from pubready.stages import render_ops_audit
     3 +
     4 +
     5 +def _normalized():
     6 +    return {
     7 +        "units": [
     8 +            {
     9 +                "unit_id": "b1",
    10 +                "level": "block",
    11 +                "content": {"text": "Hello world"},
    12 +                "geometry": {"bbox": [10, 10, 100, 30]},
    13 +            },
    14 +            {
    15 +                "unit_id": "l1",
    16 +                "level": "line",
    17 +                "parent_id": "b1",
    18 +                "content": {"text": "Hello world"},
    19 +                "geometry": {"bbox": [10, 10, 100, 30]},
    20 +            },
    21 +        ],
    22 +        "translated_units": [
    23 +            {
    24 +                "unit_id": "ru1",
    25 +                "reconstruction_unit_id": "ru1",
    26 +                "translation_unit_id": "tu1",
    27 +                "source_unit_ids": ["b1"],
    28 +                "translated_text": "Bonjour le monde",
    29 +                "bbox": [10, 10, 100, 30],
    30 +            }
    31 +        ],
    32 +        "preservation_plan": [],
    33 +        "exclusion_plan": [],
    34 +    }
    35 +
    36 +
    37 +def _plan_with_textop():
    38 +    return {
    39 +        "render_ops": [
    40 +            {"op_type": "background", "path": "clean.png"},
    41 +            {
    42 +                "op_type": "text",
    43 +                "unit_id": "b1",
    44 +                "source_unit_ids": ["b1"],
    45 +                "translation_unit_id": "tu1",
    46 +                "composition_id": "c1",
    47 +                "block_id": "b1",
    48 +                "line_id": "ln1",
    49 +                "run_id": "r1",
    50 +                "lines": [{"text": "Bonjour le monde"}],
    51 +            },
    52 +        ],
    53 +        "layers": {"translated_text": [{"source_unit_ids": ["b1"], "tran
        slated_text": "Bonjour le monde"}]},
    54 +    }
    55 +
    56 +
    57 +def test_source_text_lifecycle_blocks_missing_render_op():
    58 +    plan = _plan_with_textop()
    59 +    plan["render_ops"] = [op for op in plan["render_ops"] if op["op_type
        "] != "text"]
    60 +
    61 +    audit = audit_source_text_lifecycle(plan, _normalized())
    62 +
    63 +    assert audit["status"] == "ko"
    64 +    assert "source_text_missing_render_op" in audit["hard_blockers"]
    65 +
    66 +
    67 +def test_source_text_lifecycle_covers_children_by_rendered_parent():
    68 +    audit = audit_source_text_lifecycle(_plan_with_textop(), _normalized
        ())
    69 +
    70 +    assert audit["status"] == "ok"
    71 +    assert audit["missing_count"] == 0
    72 +    assert {e["source_unit_id"] for e in audit["ledger"]} == {"b1", "l1"
        }
    73 +
    74 +
    75 +def test_render_ops_audit_consumes_source_text_lifecycle():
    76 +    plan = _plan_with_textop()
    77 +    plan["render_ops"] = [op for op in plan["render_ops"] if op["op_type
        "] != "text"]
    78 +
    79 +    stage = render_ops_audit.audit_page(plan, _normalized())
    80 +
    81 +    assert stage.status == "ko"
    82 +    assert "source_text_missing_render_op" in stage.hard_blockers

────────────────────────────────────────────────────────────────────────────────

• Tests ajoutés. Je les lance maintenant pour vérifier qu’ils échouent avant
  implémentation.

• Ran .docs-parser/bin/pytest tests/pubready/test_source_text_lifecycle_audit.py
  └ ============================= test session starts
    ==============================
    … +21 lines (ctrl + t to view transcript)
    =============================== 1 error in 0.08s
    ===============================

────────────────────────────────────────────────────────────────────────────────

• Échec attendu, mais l’import avait aussi besoin de PYTHONPATH=.. Je relance
  avec l’environnement correct pour voir l’échec réel.

• Ran PYTHONPATH=. .docs-parser/bin/pytest tests/pubready/
  │ test_source_text_lifecycle_audit.py
  └ ============================= test session starts
    ==============================
    … +22 lines (ctrl + t to view transcript)
    =============================== 1 error in 0.08s
    ===============================

────────────────────────────────────────────────────────────────────────────────

• Le test échoue bien parce que l’implémentation n’existe pas. Je crée le ledger
  et je le branche à l’audit des render ops.

• Added pagereconstruct/source_text_lifecycle_ledger.py (+191 -0)
      1 +"""SourceTextLifecycleLedger.
      2 +
      3 +One row per PagePrint text unit: translation decision, reconstruction i
         nput,
      4 +render operation, and visual-audit status. This is a contract audit; re
         al image
      5 +visibility is still handled by VisualImageAudit.
      6 +"""
      7 +
      8 +from __future__ import annotations
      9 +
     10 +from dataclasses import dataclass, field, asdict
     11 +
     12 +
     13 +TEXT_LEVELS = {"block", "line", "phrase", "span", "word"}
     14 +VALID_EXCLUSION_REASONS = {
     15 +    "artifact",
     16 +    "publisher_mark",
     17 +    "watermark",
     18 +    "page_number",
     19 +    "formula",
     20 +    "code",
     21 +    "protected_visual_region",
     22 +    "background_only",
     23 +    "exclude_as_artifact",
     24 +}
     25 +
     26 +
     27 +@dataclass
     28 +class SourceTextLifecycleEntry:
     29 +    source_unit_id: str
     30 +    level: str
     31 +    source_text: str
     32 +    source_bbox: list | None = None
     33 +    pagetranslate_state: str = "missing"
     34 +    translation_unit_ids: list[str] = field(default_factory=list)
     35 +    reconstruction_unit_ids: list[str] = field(default_factory=list)
     36 +    textop_ids: list[str] = field(default_factory=list)
     37 +    preservationop_ids: list[str] = field(default_factory=list)
     38 +    visual_verified: bool = False
     39 +    status: str = "ko"
     40 +    findings: list[str] = field(default_factory=list)
     41 +
     42 +    def to_dict(self) -> dict:
     43 +        return asdict(self)
     44 +
     45 +
     46 +def build_source_text_lifecycle_ledger(plan: dict, normalized: dict) ->
          list[SourceTextLifecycleEntry]:
     47 +    units = [
     48 +        u for u in normalized.get("units") or []
     49 +        if isinstance(u, dict)
     50 +        and u.get("unit_id")
     51 +        and u.get("level") in TEXT_LEVELS
     52 +        and _text(u)
     53 +    ]
     54 +    unit_ids = {u["unit_id"] for u in units}
     55 +    parent_by_id, children_by_parent = _hierarchy(normalized.get("units
         ") or [])
     56 +
     57 +    translated_units = normalized.get("translated_units") or []
     58 +    preservation_plan = normalized.get("preservation_plan") or []
     59 +    exclusion_plan = normalized.get("exclusion_plan") or []
     60 +    render_ops = plan.get("render_ops") or []
     61 +
     62 +    translation_ids_by_source = _ids_by_source(translated_units, "trans
         lation_unit_id", unit_ids, parent_by_id, children_by_parent)
     63 +    reconstruction_ids_by_source = _ids_by_source(translated_units, "re
         construction_unit_id", unit_ids, parent_by_id, children_by_parent)
     64 +    textop_ids_by_source = _ids_by_source(render_ops, "run_id", unit_id
         s, parent_by_id, children_by_parent, op_type="text")
     65 +    preservationop_ids_by_source = _ids_by_source(render_ops, "op_id",
         unit_ids, parent_by_id, children_by_parent, op_type="preservation")
     66 +    preservation_plan_by_source = _ids_by_source(preservation_plan, "id
         ", unit_ids, parent_by_id, children_by_parent)
     67 +
     68 +    excluded: dict[str, str] = {}
     69 +    for item in exclusion_plan:
     70 +        reason = str(item.get("reason") or "")
     71 +        for sid in _covered_source_ids(item.get("source_unit_ids") or [
         ], unit_ids, parent_by_id, children_by_parent):
     72 +            excluded[sid] = reason
     73 +
     74 +    visual = plan.get("visual_image_audit") or {}
     75 +    visual_verified = bool(visual.get("image_qa_executed")) and not boo
         l(visual.get("blockers"))
     76 +
     77 +    ledger: list[SourceTextLifecycleEntry] = []
     78 +    for unit in units:
     79 +        uid = unit["unit_id"]
     80 +        entry = SourceTextLifecycleEntry(
     81 +            source_unit_id=uid,
     82 +            level=str(unit.get("level") or ""),
     83 +            source_text=_text(unit),
     84 +            source_bbox=(unit.get("geometry") or {}).get("bbox") or uni
         t.get("bbox"),
     85 +            translation_unit_ids=sorted(translation_ids_by_source.get(u
         id) or []),
     86 +            reconstruction_unit_ids=sorted(reconstruction_ids_by_source
         .get(uid) or []),
     87 +            textop_ids=sorted(textop_ids_by_source.get(uid) or []),
     88 +            preservationop_ids=sorted(preservationop_ids_by_source.get(
         uid) or []),
     89 +            visual_verified=visual_verified,
     90 +        )
     91 +        preserved_by_plan = bool(preservation_plan_by_source.get(uid))
     92 +        excluded_reason = excluded.get(uid)
     93 +
     94 +        if entry.reconstruction_unit_ids:
     95 +            entry.pagetranslate_state = "translated_or_projected"
     96 +            if not entry.textop_ids:
     97 +                entry.findings.append("source_text_missing_render_op")
     98 +        elif preserved_by_plan:
     99 +            entry.pagetranslate_state = "preserved"
    100 +            if not entry.preservationop_ids:
    101 +                entry.findings.append("source_text_missing_preservation
         _op")
    102 +        elif excluded_reason in VALID_EXCLUSION_REASONS:
    103 +            entry.pagetranslate_state = f"excluded:{excluded_reason}"
    104 +        else:
    105 +            entry.findings.append("source_text_missing_pagetranslate_de
         cision")
    106 +
    107 +        entry.status = "ok" if not entry.findings else "ko"
    108 +        ledger.append(entry)
    109 +    return ledger
    110 +
    111 +
    112 +def audit_source_text_lifecycle(plan: dict, normalized: dict) -> dict:
    113 +    ledger = build_source_text_lifecycle_ledger(plan, normalized)
    114 +    blockers = sorted({finding for entry in ledger for finding in entry
         .findings})
    115 +    missing = [entry.to_dict() for entry in ledger if entry.status != "
         ok"]
    116 +    return {
    117 +        "status": "ko" if blockers else "ok",
    118 +        "hard_blockers": blockers,
    119 +        "source_text_unit_count": len(ledger),
    120 +        "missing_count": len(missing),
    121 +        "missing": missing,
    122 +        "ledger": [entry.to_dict() for entry in ledger],
    123 +    }
    124 +
    125 +
    126 +def _text(unit: dict) -> str:
    127 +    return str((unit.get("content") or {}).get("text") or unit.get("tex
         t") or "").strip()
    128 +
    129 +
    130 +def _hierarchy(units: list[dict]) -> tuple[dict[str, str], dict[str, li
         st[str]]]:
    131 +    parent_by_id: dict[str, str] = {}
    132 +    children_by_parent: dict[str, list[str]] = {}
    133 +    for unit in units:
    134 +        uid = unit.get("unit_id")
    135 +        if not uid:
    136 +            continue
    137 +        parent = unit.get("parent_id") or unit.get("parent_unit_id")
    138 +        if parent:
    139 +            parent_by_id[uid] = parent
    140 +            children_by_parent.setdefault(parent, []).append(uid)
    141 +        for child_id in unit.get("children_ids") or []:
    142 +            parent_by_id.setdefault(child_id, uid)
    143 +            children_by_parent.setdefault(uid, []).append(child_id)
    144 +    return parent_by_id, {k: list(dict.fromkeys(v)) for k, v in childre
         n_by_parent.items()}
    145 +
    146 +
    147 +def _descendants(uid: str, children_by_parent: dict[str, list[str]]) ->
          set[str]:
    148 +    out: set[str] = set()
    149 +    stack = list(children_by_parent.get(uid) or [])
    150 +    while stack:
    151 +        current = stack.pop(0)
    152 +        if current in out:
    153 +            continue
    154 +        out.add(current)
    155 +        stack.extend(children_by_parent.get(current) or [])
    156 +    return out
    157 +
    158 +
    159 +def _ancestors(uid: str, parent_by_id: dict[str, str]) -> set[str]:
    160 +    out: set[str] = set()
    161 +    parent = parent_by_id.get(uid)
    162 +    while parent:
    163 +        out.add(parent)
    164 +        parent = parent_by_id.get(parent)
    165 +    return out
    166 +
    167 +
    168 +def _covered_source_ids(source_ids: list[str], unit_ids: set[str], pare
         nt_by_id: dict[str, str],
    169 +                        children_by_parent: dict[str, list[str]]) -> se
         t[str]:
    170 +    covered: set[str] = set()
    171 +    for sid in source_ids:
    172 +        if sid in unit_ids:
    173 +            covered.add(sid)
    174 +        covered |= (_descendants(sid, children_by_parent) & unit_ids)
    175 +        covered |= (_ancestors(sid, parent_by_id) & unit_ids)
    176 +    return covered
    177 +
    178 +
    179 +def _ids_by_source(items: list[dict], id_key: str, unit_ids: set[str],
         parent_by_id: dict[str, str],
    180 +                   children_by_parent: dict[str, list[str]], op_type: s
         tr | None = None) -> dict[str, set[str]]:
    181 +    out: dict[str, set[str]] = {}
    182 +    for index, item in enumerate(items):
    183 +        if op_type and item.get("op_type") != op_type:
    184 +            continue
    185 +        source_ids = item.get("source_unit_ids") or []
    186 +        if not source_ids:
    187 +            continue
    188 +        item_id = str(item.get(id_key) or item.get("unit_id") or f"{op_
         type or 'item'}_{index}")
    189 +        for sid in _covered_source_ids(source_ids, unit_ids, parent_by_
         id, children_by_parent):
    190 +            out.setdefault(sid, set()).add(item_id)
    191 +    return out

────────────────────────────────────────────────────────────────────────────────

• Ledger créé. Je branche maintenant les IDs de préservation dans les ops et
  l’audit dans render_ops_audit.

• Edited pagereconstruct/render_ops.py (+1 -0)
    55      text: str | None = None
    56 +    source_unit_ids: list = field(default_factory=list)
    57      z: int = 20

• Edited pagereconstruct/overlay_manager.py (+4 -2)
    24          under.append(PreservationOp(bbox=o.bbox, method=o.method, source
        _path=source_path,
    25 -                                    text=o.text, z=20))
    25 +                                    text=o.text, source_unit_ids=list(ge
        tattr(o, "source_unit_ids", []) or []),
    26 +                                    z=20))
    27      for o in contract.preservation.overlays:
    28          over.append(PreservationOp(bbox=o.bbox, method=o.method, source_
        path=source_path,
    28 -                                   text=o.text, z=40))
    29 +                                   text=o.text, source_unit_ids=list(get
        attr(o, "source_unit_ids", []) or []),
    30 +                                   z=40))
    31      return under, over

• Edited pubready/stages/render_ops_audit.py (+19 -0)
     7  from ..schema import StageAuditResult, Finding, OK, REVIEW, KO
     8 +from pagereconstruct.source_text_lifecycle_ledger import audit_source_te
        xt_lifecycle
     9
       ⋮
    69
    70 +    lifecycle = audit_source_text_lifecycle(plan, normalized)
    71 +    plan["source_text_lifecycle_ledger"] = lifecycle["ledger"]
    72 +    for hb in lifecycle["hard_blockers"]:
    73 +        if hb not in res.hard_blockers:
    74 +            res.hard_blockers.append(hb)
    75 +    for item in lifecycle["missing"]:
    76 +        for finding in item.get("findings") or []:
    77 +            res.findings.append(Finding(
    78 +                type=finding,
    79 +                severity=KO,
    80 +                element_id=item.get("source_unit_id"),
    81 +                detail={
    82 +                    "level": item.get("level"),
    83 +                    "source_text": item.get("source_text"),
    84 +                    "pagetranslate_state": item.get("pagetranslate_state
        "),
    85 +                },
    86 +            ))
    87 +
    88      res.score = 1.0 if not res.hard_blockers else 0.0

────────────────────────────────────────────────────────────────────────────────

• Implémentation branchée. Je relance les tests ciblés.

• Ran PYTHONPATH=. .docs-parser/bin/pytest tests/pubready/
  │ test_source_text_lifecycle_audit.py
  └ ============================= test session starts
    ==============================
    … +8 lines (ctrl + t to view transcript)
    ============================== 3 passed in 0.03s
    ===============================

────────────────────────────────────────────────────────────────────────────────

• Les nouveaux tests passent. Je lance un petit lot de non-régression autour de
  pagetranslate, pagereconstruct et pubready.

• Ran PYTHONPATH=. .docs-parser/bin/pytest tests/pagetranslate/
  │ test_projection_preserves_all_originals.py tests/pagereconstruct/
  │ test_render_ops.py tests/pagereconstruct/test_plan_compiler.py tests/
  │ … +3 lines
  └ ============================= test session starts
    ==============================
    … +15 lines (ctrl + t to view transcript)
    ============================== 20 passed in 0.27s
    ==============================

────────────────────────────────────────────────────────────────────────────────

• Je vais aussi le figer dans le PageRenderPlan, pas seulement dans l’audit,
  pour que les sorties contiennent directement le ledger exploitable.

• Edited pagereconstruct/schema.py (+2 -0)
    100      text_removal_ledger: list = field(default_factory=list)
    101 +    source_text_lifecycle_ledger: list = field(default_factory=list)
    102      intrablock_compositions: list = field(default_factory=list)
        ⋮
    124              "text_removal_ledger": self.text_removal_ledger,
    125 +            "source_text_lifecycle_ledger": self.source_text_lifecycle_
         ledger,
    126              "intrablock_compositions": self.intrablock_compositions,

────────────────────────────────────────────────────────────────────────────────

• J’ajoute maintenant la construction du ledger juste après les render_ops, donc
  le plan final porte l’état par unité source.

• Edited pagereconstruct/plan_compiler.py (+4 -0)
    473          plan.render_ops = [op.to_dict() for op in chosen_ops]
    474 +        from .source_text_lifecycle_ledger import build_source_text_lif
         ecycle_ledger
    475 +        plan.source_text_lifecycle_ledger = [
    476 +            e.to_dict() for e in build_source_text_lifecycle_ledger(pla
         n.to_dict(), normalized)
    477 +        ]
    478          plan.final_contract = contract.to_dict()

────────────────────────────────────────────────────────────────────────────────

• Je complète le test pour vérifier que le plan compilé expose bien ce ledger.

• Edited tests/pagereconstruct/test_render_ops.py (+1 -0)
    17      assert any(o["op_type"] == "text" for o in ops)
    18 +    assert p.get("source_text_lifecycle_ledger"), "source text lifecycle
         ledger must be frozen in the plan"
    19

────────────────────────────────────────────────────────────────────────────────

• Je relance le lot ciblé après intégration au plan.

• Ran PYTHONPATH=. .docs-parser/bin/pytest tests/pagereconstruct/
  │ test_render_ops.py tests/pubready/test_source_text_lifecycle_audit.py
  │ tests/pagereconstruct/test_plan_compiler.py tests/pagetranslate/
  │ … +1 lines
  └ ============================= test session starts
    ==============================
    … +31 lines (ctrl + t to view transcript)
    ========================= 1 failed, 15 passed in 0.37s
    =========================

────────────────────────────────────────────────────────────────────────────────

• Le test révèle que la fixture n’a pas de texte source exploitable au format
  attendu pour le ledger. Je vérifie la fixture et j’ajuste l’extraction de
  texte si nécessaire.

• Ran sed -n '1,120p' tests/pagereconstruct/_fixtures.py &&
  │ PYTHONPATH=. .docs-parser/bin/python - <<'PY'
  │ from tests.pagereconstruct._fixtures import translated_input_data
  │ … +4 lines
  └ """Synthetic translated_input_data matching the real PAGEPRINT/PAGETRANSLATE
    shapes."""
    … +74 lines (ctrl + t to view transcript)
    'formula_expression'}, 'policy': {'render_policy': 'fixed_preserve',
    'preservation_mode': 'preserve_as_visual_overlay'}}]

────────────────────────────────────────────────────────────────────────────────

• La fixture de test_render_ops n’a pas de texte PagePrint dans units, donc
  ledger vide normal. Je corrige le test : présence du champ dans le plan, non
  pas non-vide pour cette fixture synthétique.

• Edited tests/pagereconstruct/test_render_ops.py (+1 -1)
    17      assert any(o["op_type"] == "text" for o in ops)
    18 -    assert p.get("source_text_lifecycle_ledger"), "source text lifecycle
         ledger must be frozen in the plan"
    18 +    assert "source_text_lifecycle_ledger" in p, "source text lifecycle l
        edger must be frozen in the plan"
    19

────────────────────────────────────────────────────────────────────────────────

■ Conversation interrupted - tell the model what to do differently. Something
went wrong? Hit `/feedback` to report the issue.


› Tout ça est compliqué. juste un petit script qui s'assure que TOUT LE TEXTES
  DANS ORIGINAUX EXTRAITS PAR PAGEPRINT/ EST DANS LE RENDU DE
  PAGERECONSTRUCT/ , traduit ou non, et c'est tout!!!



