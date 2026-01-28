Pour une traduction **WYSIWYG** (tu traduis le contenu *sans casser l’apparence*), il faut penser comme un imprimeur + comme un moteur de rendu : **tout ce qui existe dans un document est soit du contenu, soit de la mise en forme, soit de l’interactivité, soit de la sémantique**. Et si tu oublies une catégorie, le résultat “a l’air traduit” mais “ne ressemble plus à rien”.

Voici la liste la plus complète et utile des **éléments/objets extractibles**.

---

## 1) Structure logique du document

Même si un PDF “n’a pas de structure”, ton projet doit tenter de la reconstruire.

* Pages (numéro, taille, orientation, rotation)
* Sections / chapitres (si détectables)
* Blocs (zones) : en-têtes, pied de page, corps, marges
* Colonnes, grilles, cadres, encarts
* Ordre de lecture (très critique en multi-colonnes)
* Paragraphes
* Lignes
* Mots / glyphes (caractères)
* Listes (puces, numérotations, niveaux d’indentation)
* Titres (H1/H2/H3… même si c’est une déduction)
* Notes de bas de page / notes de fin
* Citations (bloc quote), encadrés “attention / info”

👉 Sans ça, tu traduis bien… mais tu reconstruis comme un puzzle sans l’image sur la boîte.

---

## 2) Texte et typographie

C’est le “carburant” de la traduction, mais aussi la partie la plus traîtresse.

### Texte brut

* Contenu textuel (par langue, par script)
* Espaces, retours, césures (hyphenation)
* Ligatures (fi, fl…), caractères spéciaux
* Direction d’écriture (LTR, RTL)
* Langue détectée par segment (très important en documents mixtes)

### Typographie (au niveau “style”)

* Police (famille, fallback)
* Taille
* Gras/italique/souligné/barré
* Couleur (fill/stroke)
* Interlettrage (tracking), crénage (kerning)
* Interligne (leading)
* Alignement (gauche/droite/centré/justifié)
* Indentation (gauche/droite, première ligne)
* Espacement avant/après paragraphe
* Casse (small caps), exposant/indice
* Opacité/transparence
* Effets (outline, ombre, surbrillance) si présents

👉 Pour du WYSIWYG, tu dois conserver **la métrique** (largeur des lignes, sauts) ou compenser intelligemment, sinon tout déborde.

---

## 3) Géométrie et placement (le nerf de la guerre WYSIWYG)

Chaque objet doit avoir une “carte d’identité spatiale”.

* Boîte englobante (x, y, w, h)
* Points d’ancrage (baseline du texte)
* Transformations : rotation, scaling, skew
* Z-order (qui est au-dessus de qui)
* Clipping / masques (texte ou image coupés par une forme)
* Zones de non-recouvrement (ex: texte autour d’une image)

👉 Si tu perds les **transforms** ou le **z-order**, tu recrées un document “propre”… mais pas fidèle.

---

## 4) Images et raster

Tout ce qui est “pixels”.

* Images intégrées (JPEG/PNG, etc.)
* Scans pleine page
* Signatures scannées
* Logos
* Tampons
* Captures d’écran
* Images avec transparence / alpha
* Résolution effective, DPI
* Masques / rognage (crop)
* Filtres éventuels (rare, mais existe)

⚠️ Cas crucial : **texte dans l’image**
Il faut extraire :

* Les zones textuelles dans l’image (bbox)
* Le texte OCR
* La confiance OCR
* La police approximée (si tu veux “réécrire” dans l’image)
* La couleur de texte / contraste / fond

---

## 5) Vectoriel, dessins, formes

C’est tout ce qui ressemble à des “objets graphiques”.

* Lignes, flèches
* Rectangles, cercles, polygones
* Formes libres (paths)
* Remplissages, contours, épaisseurs
* Dégradés, motifs
* Icônes vectorielles
* Diagrammes (souvent vecteurs)
* Graphiques (barres, courbes) parfois vecteurs + texte

👉 En WYSIWYG, tu ne traduis pas ça, mais tu dois le **préserver** car il structure la page (encadrés, titres sur fond coloré, etc.).

---

## 6) Tableaux

Un enfer à part entière.

* Grille (lignes/colonnes)
* Cellules (bbox, padding)
* Fusion de cellules (rowspan/colspan)
* Bordures (style, épaisseur)
* Fond de cellule (couleur)
* Alignement dans cellule
* Contenu multi-paragraphe dans cellule
* En-tête de tableau (header row)
* Notes sous tableau, légende

⚠️ Une table “visuelle” peut être en réalité :

* du texte aligné à la main
* ou du vectoriel
  Donc il faut extraire aussi la **reconstruction** (détection structure).

---

## 7) Formules mathématiques et scientifiques

Deux cas :

### A) Math “native” (LaTeX/MathML/objets équation)

* Structure de l’expression (arbre)
* Symboles
* Exposants/indices
* Fractions, racines, matrices
* Police math

### B) Math “dessinée” ou “scan”

* OCR math (spécialisé)
* Zones et segmentation (car un OCR normal casse tout)

👉 Pour traduction WYSIWYG : tu traduis surtout **le texte autour**, mais tu dois préserver l’équation telle quelle (ou la reconstruire à l’identique).

---

## 8) Liens, annotations, champs interactifs

Sinon ton PDF traduit devient un “poster”.

* Hyperliens (URL + zone cliquable)
* Liens internes (sommaire, renvois)
* Signets / bookmarks
* Commentaires, notes, surlignage
* Champs de formulaire : texte, checkbox, radio, dropdown
* Valeurs + contraintes (format date, masque)
* Signatures numériques (souvent à conserver telles quelles)
* Actions (ouvrir lien, aller page X)

---

## 9) Références, citations, métadonnées “éditoriales”

Très utile pour une traduction propre, et pour ne pas faire n’importe quoi.

* Références bibliographiques (DOI, PMID, etc.)
* Notes légales / mentions
* Numéros de figure/table (“Figure 3”, “Tableau 2”)
* Légendes (caption) : figure/table/encadré
* Index (termes + pages)
* Glossaire (termes à traduire de façon stable)

---

## 10) En-têtes / pieds de page / pagination

Souvent répétés, donc optimisable.

* Numéro de page (format “Page X/Y”)
* Date
* Titre de section courant (running header)
* Nom document / version
* Filigranes (watermarks)

👉 Si tu ne les traites pas à part, tu vas les retraduire 200 fois et créer des incohérences.

---

## 11) Métadonnées du fichier (niveau “document”)

* Titre, auteur, sujet, mots-clés
* Langue déclarée
* Date création/modif
* Producteur (Word, LaTeX, InDesign…)
* Polices embarquées (important pour rendu final)
* Profils couleur (rare mais existe)

---

## 12) Objets “exotiques” mais réels

Parce que la vraie vie adore les exceptions.

* Calques (layers) dans PDF
* Contenus alternatifs (accessibilité)
* Pièces jointes intégrées au PDF
* Audio/vidéo embarqués (rare)
* QR codes / codes-barres (à préserver)
* Cachets/mentions en arrière-plan

---

# Le piège classique (et comment l’éviter)

Le faux choix : “je traduis le texte puis je réécris”.

En WYSIWYG, il faut plutôt :

1. **extraire un modèle de scène** (objets + géométrie + style),
2. **traduire par segments** (avec IDs stables),
3. **réinjection** en respectant la scène,
4. **reflow contrôlé** (si la traduction est plus longue) : réduction légère police / ajustement tracking / re-cassage lignes / élargissement de boîte si possible,
5. **validation visuelle** (diff overlay).

---

La liste précédente couvre ~95% des objets *visibles*. Pour un projet WYSIWYG solide, il faut aussi extraire (ou reconstruire) ces couches “invisibles mais déterminantes” :

## 1) Le niveau le plus bas : le PDF n’a pas du “texte”, il a des glyphes

Dans beaucoup de PDF, ce que tu crois être un mot est en réalité :

* des **glyphes séparés** (chaque lettre placée à la main)
* du “texte” sans espaces (espaces implicites)
* des **ligatures** et substitutions de police
* des encodages tordus (ToUnicode absent/partiel)

➡️ Objets/infos à extraire :

* glyph id / unicode (quand dispo)
* matrices de transformation (par glyph)
* vecteur d’avancement (advance width)
* infos ToUnicode/CMap
* détection d’espaces/coupures *reconstruites*

## 2) Les règles de “reflow” et d’ajustement typographique

Pour rester WYSIWYG quand la traduction est plus longue, tu dois prévoir des stratégies mesurables :

* re-cassage de lignes
* micro-ajustements (tracking, font-size)
* expansion de boîte (si cadre extensible)
* déplacement d’objets voisins (layout push) — très dangereux

➡️ Donc tu dois extraire aussi :

* contraintes de layout (boîte fixe vs extensible)
* relations d’alignement (marges, grilles, colonnes)
* ancres (ex: légende attachée à figure)

## 3) La sémantique “document” (pas visible mais utile)

Pour ne pas traduire n’importe comment :

* détection de **langue par segment**
* entités : noms propres, lieux, organisations
* termes techniques (glossaire)
* acronymes + expansions
* unités / nombres / formats (dates, décimales)

➡️ Objets :

* segments sémantiques (avec IDs stables)
* dictionnaire terminologique + règles
* exceptions “NE PAS traduire”
* normalisation typographique (guillemets, espaces fines, etc.)

## 4) Cas “scientifique / administratif” (ceux qui cassent tout)

* tableaux “faux tableaux” (alignements au pixel)
* formules en image / en vecteur
* graphes où le texte est vectoriel
* documents scannés avec “texte invisible” (OCR layer)

➡️ Il faut extraire :

* la présence d’un **texte OCR caché** (et sa qualité)
* la source de vérité (PDF text vs OCR vs vision)
* une couche de fusion + déduplication (sinon tu doubles tout)

## 5) Les effets de rendu et pièges graphiques

* transparences et modes de fusion
* clipping complexes
* surimpressions (overprint)
* patterns/dégradés
* ordre des calques (z-index) crucial

➡️ Objets :

* propriétés de rendu (blend mode, opacity)
* clip paths / masks
* ordre exact d’empilement

## 6) La sécurité et la “permission”

Très concret :

* PDF protégé (copie interdite, impression, etc.)
* chiffrement
* signatures numériques

➡️ Objets :

* flags de permission
* statut chiffrement/signature
* stratégie : conserver intact / régénérer / avertir

## 7) Les polices : le grand mensonge

Même si tu “gardes la police”, parfois :

* elle n’est pas embarquée
* elle n’existe pas sur la machine
* elle n’a pas les glyphes de la langue cible (ex: vietnamien)
* le fallback change la métrique → tout bouge

➡️ Objets :

* liste des polices utilisées + embedding
* couverture Unicode des polices
* stratégie de substitution typographique (metrics-compatible)

---

# Donc… “est-ce tout ?”

Si on parle **des objets visibles**, oui, on est très large.
Si on parle d’un **pipeline WYSIWYG robuste**, non : il faut aussi les couches *glyph/encodage*, *contraintes de layout*, *sémantique*, *fusion multi-sources*, *rendu avancé*, *polices*, *permissions*.


----
Voilà une **taxonomie canonique WYSIWYG “TOUT”**, pensée comme une **scène graphique** (comme un moteur de rendu), avec **types d’objets + champs obligatoires**.
Objectif : tu peux sérialiser ça en JSON, traduire uniquement certains nœuds, puis reconstruire **à l’identique**.

Je te donne :

1. **les types canoniques (exhaustif et pratique)**
2. **les champs communs (base)**
3. **les champs spécifiques par type**
4. **les enums/structures (styles, géométrie, ressources, provenance)**

> Oui, c’est long. C’est normal : le WYSIWYG est une religion, et le PDF est son ancien testament.

---

## 0) Principes de base (sans ça, tout s’écroule)

* **Tout objet a un `id` stable**, un `bbox`, un `transform`, un `z_index`, une `provenance`.
* **Le texte “canonique”** doit exister au moins en `TextRun` (et optionnellement en `GlyphRun` si tu veux du vrai pixel-perfect).
* **Les objets traduisibles** sont repérés (`translatable: true`) + segmentés (`segments[]`).
* **Fusion multi-sources** (pdf text / ocr / vision) doit être explicitée (`provenance.source`, `dedupe_key`).

---

## 1) Liste complète des types d’objets canoniques

### A — Conteneurs & structure

1. `Document`
2. `Page`
3. `Layer` (PDF OCG / calque logique)
4. `Region` (zone : header, footer, column, sidebar, body, etc.)
5. `Flow` (chaîne de lecture / flux multi-blocs)
6. `Group` (regroupe des objets, conserve transform/z-order)

### B — Texte (du haut niveau au bas niveau)

7. `TextBlock` (bloc de paragraphe/zone de texte)
8. `Paragraph`
9. `Line`
10. `TextRun` (le niveau “traduction” principal)
11. `GlyphRun` (optionnel, pixel-perfect)
12. `RubyAnnotation` (furigana, annotations au-dessus du texte)
13. `Footnote`
14. `Endnote`

### C — Listes & structures éditoriales

15. `List`
16. `ListItem`
17. `Heading` (titre H1/H2…)
18. `Caption` (légende figure/table)
19. `TOC` (table des matières)
20. `Index` (index alphabétique)
21. `Glossary` (glossaire structuré)
22. `ReferenceEntry` (bibliographie / citation formelle)

### D — Tableaux

23. `Table`
24. `TableRow`
25. `TableCell`

### E — Images & médias

26. `ImageRaster` (jpg/png, scan, etc.)
27. `ImageInline` (petite image dans une ligne)
28. `Mask` (alpha mask / soft mask)
29. `EmbeddedMedia` (audio/vidéo — rare mais existe)

### F — Vectoriel / dessin / diagrammes

30. `VectorPath` (bezier/path)
31. `Shape` (rect/circle/line/arrow/polygon…)
32. `PatternFill` (motif)
33. `GradientFill` (dégradé)
34. `Chart` (si tu reconstruis la sémantique de graphes)
35. `BarcodeOrQRCode`

### G — Math / science

36. `MathFormula` (arbre MathML/LaTeX/équation native)
37. `ChemFormula` (chimie, optionnel si tu veux spécialiser)
38. `EquationImage` (équation en raster + OCR math)

### H — Interactivité / navigation

39. `Link` (externe)
40. `LinkInternal` (renvoi page/anchor)
41. `Bookmark` (signets)
42. `Action` (openURL, goTo, launch…)
43. `Annotation` (commentaire, surlignage, sticky note)
44. `FormField` (text, checkbox, radio, dropdown, signature field)

### I — En-têtes / pieds / pagination

45. `HeaderFooter`
46. `PageNumber`
47. `Watermark`

### J — Sécurité / signatures / pièces jointes

48. `DigitalSignature`
49. `EmbeddedFile` (pièce jointe)
50. `Permission` (copie/impression/etc.)

### K — Ressources (référencées par les objets)

51. `FontResource`
52. `ImageResource`
53. `ColorSpaceResource`
54. `GraphicStateResource` (blend mode, overprint, etc.)

---

## 2) Champs communs à TOUS les objets

```json
{
  "id": "obj_123",
  "type": "TextRun",
  "page_id": "page_1",
  "parent_id": "obj_45",
  "children": ["obj_124", "obj_125"],

  "bbox": { "x": 0, "y": 0, "w": 100, "h": 20, "unit": "pt" },
  "transform": { "a": 1, "b": 0, "c": 0, "d": 1, "e": 0, "f": 0 }, 
  "rotation_deg": 0,
  "z_index": 120,
  "clip_path_id": "clip_9",

  "visible": true,
  "locked": false,

  "style_id": "style_33",
  "tags": ["body", "column_2"],

  "provenance": {
    "source": "pdf_text",
    "source_object_ref": "pdf:page=1;obj=889",
    "confidence": 0.98,
    "dedupe_key": "hash(...)",
    "merged_from": ["ocr:...", "vision:..."]
  },

  "translatable": false,
  "translation_key": null
}
```

### Notes

* `bbox` en points PDF (`pt`) recommandé (stable).
* `transform` = matrice affine (PDF standard).
* `clip_path_id` indispensable si tu as du texte/image “coupé”.
* `dedupe_key` indispensable pour éviter le doublon multi-sources.

---

## 3) Structures partagées (styles, peinture, géométrie)

### 3.1 Style générique

```json
{
  "style_id": "style_33",
  "opacity": 1.0,
  "blend_mode": "normal",
  "overprint": false,

  "fill": { "kind": "solid", "color": "#000000" },
  "stroke": { "enabled": false, "color": "#000000", "width": 1.0, "dash": [] },

  "shadow": null,
  "outline": null
}
```

### 3.2 Style texte (utilisé par TextRun/GlyphRun/Paragraph/Heading)

```json
{
  "text_style": {
    "font": { "family": "Times New Roman", "postscript_name": "TimesNewRomanPSMT", "embedded": true },
    "size_pt": 12,
    "weight": 400,
    "italic": false,
    "underline": false,
    "strike": false,

    "color": "#111111",
    "tracking": 0.0,
    "kerning": true,

    "baseline_shift": 0.0,
    "script": "normal",
    "language": "fr",
    "direction": "ltr"
  },

  "paragraph_style": {
    "align": "justify",
    "line_height": 1.2,
    "space_before_pt": 0,
    "space_after_pt": 6,
    "indent_first_line_pt": 0,
    "indent_left_pt": 0,
    "indent_right_pt": 0,
    "hyphenation": "auto"
  }
}
```

### 3.3 Géométrie / lecture

```json
{
  "reading": {
    "order": 152,
    "flow_id": "flow_main",
    "column_id": "col_2",
    "paragraph_id": "p_88"
  }
}
```

---

## 4) Spécification par type (champs spécifiques)

### 4.1 Document

```json
{
  "type": "Document",
  "meta": {
    "title": "…",
    "author": "…",
    "subject": "…",
    "keywords": ["…"],
    "created_at": "2025-01-01T10:00:00Z",
    "modified_at": "2025-02-02T10:00:00Z",
    "declared_language": "fr",
    "producer": "Microsoft Word",
    "pdf_version": "1.7"
  },
  "permissions_id": "perm_1",
  "resources": {
    "fonts": ["font_1", "font_2"],
    "images": ["imgres_1"],
    "colorspaces": ["cs_1"],
    "graphic_states": ["gs_1"]
  },
  "pages": ["page_1", "page_2"]
}
```

### 4.2 Page

```json
{
  "type": "Page",
  "index": 1,
  "size_pt": { "w": 595.28, "h": 841.89 },
  "rotation_deg": 0,
  "crop_box": { "x": 0, "y": 0, "w": 595.28, "h": 841.89 },
  "layers": ["layer_1"],
  "regions": ["region_header", "region_body", "region_footer"]
}
```

### 4.3 Layer

```json
{
  "type": "Layer",
  "name": "Layer 1",
  "visible": true
}
```

### 4.4 Region (zones)

```json
{
  "type": "Region",
  "region_kind": "body",
  "layout": { "columns": 2, "gutter_pt": 18 },
  "objects": ["obj_100", "obj_101"]
}
```

### 4.5 Flow (ordre de lecture)

```json
{
  "type": "Flow",
  "flow_kind": "main_reading",
  "sequence": ["obj_200", "obj_201", "obj_202"]
}
```

### 4.6 Group

```json
{
  "type": "Group",
  "objects": ["obj_a", "obj_b"],
  "group_transform_inherited": true
}
```

---

## TEXTE

### 4.7 TextBlock

```json
{
  "type": "TextBlock",
  "content_kind": "body|quote|code|callout",
  "paragraphs": ["para_1", "para_2"],
  "wrap_mode": "none|around_shapes|within_box",
  "constraints": { "box_fixed": true, "allow_reflow": true }
}
```

### 4.8 Paragraph

```json
{
  "type": "Paragraph",
  "lines": ["line_1", "line_2"],
  "list_ref": null,
  "heading_level": null,
  "paragraph_style": { "...": "..." }
}
```

### 4.9 Line

```json
{
  "type": "Line",
  "runs": ["run_1", "run_2"],
  "baseline_y": 512.2
}
```

### 4.10 TextRun (le cœur de la traduction)

```json
{
  "type": "TextRun",
  "text": "Bonjour le monde",
  "normalized_text": "Bonjour le monde",
  "language": "fr",
  "direction": "ltr",

  "text_style": { "...": "..." },

  "translatable": true,
  "translation_key": "seg_8891",
  "segments": [
    { "seg_id": "s1", "text": "Bonjour", "bbox": { "x":1,"y":2,"w":30,"h":10,"unit":"pt" } },
    { "seg_id": "s2", "text": "le monde", "bbox": { "x":35,"y":2,"w":60,"h":10,"unit":"pt" } }
  ],

  "reflow_policy": {
    "priority": "high",
    "max_font_shrink_pct": 8,
    "max_tracking_expand": 0.5,
    "allow_line_breaks": true
  }
}
```

### 4.11 GlyphRun (pixel-perfect)

```json
{
  "type": "GlyphRun",
  "font_resource_id": "font_1",
  "glyphs": [
    { "gid": 431, "unicode": "B", "x": 100.2, "y": 500.1, "advance": 7.1 },
    { "gid": 120, "unicode": "o", "x": 107.3, "y": 500.1, "advance": 6.2 }
  ]
}
```

### 4.12 RubyAnnotation

```json
{
  "type": "RubyAnnotation",
  "base_run_id": "run_1",
  "ruby_text": "ふりがな",
  "position": "above|below"
}
```

### 4.13 Footnote / Endnote

```json
{
  "type": "Footnote",
  "marker": "1",
  "anchor_object_id": "run_77",
  "note_textblock_id": "tb_foot_1"
}
```

---

## LISTES / TITRES / SOMMAIRE

### 4.14 Heading

```json
{
  "type": "Heading",
  "level": 2,
  "textblock_id": "tb_9",
  "translatable": true,
  "translation_key": "heading_2_9"
}
```

### 4.15 List / ListItem

```json
{
  "type": "List",
  "list_kind": "bullet|numbered",
  "level": 1,
  "items": ["li_1", "li_2"]
}
```

```json
{
  "type": "ListItem",
  "marker_text": "•",
  "content_textblock_id": "tb_12"
}
```

### 4.16 TOC / Index / Glossary / ReferenceEntry

Ce sont des structures, souvent reconstruites, mais utiles :

```json
{
  "type": "ReferenceEntry",
  "raw": "Dupont J. (2022)…",
  "fields": { "authors": ["Dupont J"], "year": 2022, "doi": "10.xxxx/yyy" },
  "translatable": true,
  "translation_policy": "keep_names|translate_title_only"
}
```

---

## TABLEAUX

### 4.17 Table / Row / Cell

```json
{
  "type": "Table",
  "grid": { "rows": 5, "cols": 3 },
  "rows": ["tr_1","tr_2"],
  "structure_confidence": 0.92,
  "is_visual_reconstructed": true
}
```

```json
{
  "type": "TableCell",
  "row": 1,
  "col": 2,
  "rowspan": 1,
  "colspan": 2,
  "padding_pt": { "t":2,"r":2,"b":2,"l":2 },
  "background_fill": { "kind":"solid", "color":"#FFFFFF" },
  "content": ["tb_cell_9"]
}
```

---

## IMAGES / RASTER

### 4.18 ImageRaster / ImageInline

```json
{
  "type": "ImageRaster",
  "image_resource_id": "imgres_1",
  "dpi": 300,
  "color_space_id": "cs_1",
  "crop": { "x":0,"y":0,"w":1024,"h":768 },
  "mask_id": null
}
```

### 4.19 OCRTextOverlay (fortement recommandé)

Ce n’est pas “un objet PDF standard”, c’est un objet canonique de ton pipeline.

```json
{
  "type": "TextRun",
  "provenance": { "source": "ocr", "confidence": 0.83 },
  "text": "TEXTE DANS IMAGE",
  "translatable": true,
  "translation_key": "ocr_551",
  "style_hint": { "estimated_font": "Arial", "estimated_size_pt": 11, "color": "#222" }
}
```

---

## VECTORIEL / FORMES

### 4.20 VectorPath

```json
{
  "type": "VectorPath",
  "d": "M 10 10 C 20 20, 40 20, 50 10",
  "style_id": "style_path_1",
  "is_clipping_path": false
}
```

### 4.21 Shape

```json
{
  "type": "Shape",
  "shape_kind": "rect|circle|line|arrow|polygon",
  "params": { "x":10,"y":10,"w":100,"h":50,"r":8 },
  "style_id": "style_shape_3"
}
```

### 4.22 Chart

Si tu veux reconstruire un graphe “intelligent” :

```json
{
  "type": "Chart",
  "chart_kind": "bar|line|pie|scatter",
  "data": { "series": [ { "name":"A", "values":[1,2,3] } ] },
  "axes": { "x_label":"…", "y_label":"…" },
  "labels_text_objects": ["run_901","run_902"]
}
```

### 4.23 BarcodeOrQRCode

```json
{
  "type": "BarcodeOrQRCode",
  "code_kind": "qr|code128|ean13",
  "payload": "…",
  "preserve_exact_pixels": true
}
```

---

## MATH / SCIENCE

### 4.24 MathFormula

```json
{
  "type": "MathFormula",
  "format": "mathml|latex|native",
  "source": "E = mc^2",
  "ast": { "kind":"eq", "left":"E", "right":{"mul":["m",{"pow":["c",2]}]} },
  "translatable": false,
  "keep_as_is": true
}
```

### 4.25 EquationImage

```json
{
  "type": "EquationImage",
  "image_resource_id": "imgres_eq_1",
  "math_ocr": { "latex": "\\frac{a}{b}", "confidence": 0.62 },
  "translatable": false
}
```

---

## INTERACTIVITÉ / NAVIGATION

### 4.26 Link / LinkInternal

```json
{
  "type": "Link",
  "target": "https://example.com",
  "rects": [ { "x":10,"y":10,"w":50,"h":10,"unit":"pt" } ]
}
```

```json
{
  "type": "LinkInternal",
  "target_page_index": 5,
  "target_object_id": "heading_5"
}
```

### 4.27 Bookmark

```json
{
  "type": "Bookmark",
  "title": "Chapitre 1",
  "target_page_index": 1,
  "level": 1
}
```

### 4.28 Annotation

```json
{
  "type": "Annotation",
  "annotation_kind": "comment|highlight|underline|stamp|note",
  "author": "…",
  "created_at": "…",
  "content": "…",
  "translatable": true,
  "translation_key": "annot_33"
}
```

### 4.29 FormField

```json
{
  "type": "FormField",
  "field_kind": "text|checkbox|radio|dropdown|signature",
  "name": "patient_name",
  "value": "…",
  "rect": { "x":10,"y":10,"w":120,"h":14,"unit":"pt" },
  "constraints": { "max_len": 64, "pattern": null },
  "translatable": false
}
```

### 4.30 Action

```json
{
  "type": "Action",
  "action_kind": "open_url|go_to|submit_form|launch",
  "payload": { "url":"…", "page": 2 }
}
```

---

## HEADER/FOOTER/PAGINATION

### 4.31 HeaderFooter / PageNumber / Watermark

```json
{
  "type": "HeaderFooter",
  "kind": "header|footer",
  "objects": ["tb_h1", "pnum_1"],
  "repeat_on_pages": "all|odd|even|range"
}
```

```json
{
  "type": "PageNumber",
  "format": "Page {n} / {N}",
  "translatable": true,
  "translation_key": "pnum_fmt_1"
}
```

```json
{
  "type": "Watermark",
  "text": "CONFIDENTIEL",
  "opacity": 0.15,
  "rotation_deg": 45,
  "translatable": true,
  "translation_key": "wm_1"
}
```

---

## SÉCURITÉ / SIGNATURES / ATTACHMENTS

### 4.32 DigitalSignature

```json
{
  "type": "DigitalSignature",
  "status": "valid|invalid|unknown",
  "covers_entire_document": true,
  "preserve_original": true
}
```

### 4.33 EmbeddedFile

```json
{
  "type": "EmbeddedFile",
  "filename": "annexe.xlsx",
  "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "size_bytes": 123456
}
```

### 4.34 Permission

```json
{
  "type": "Permission",
  "can_copy": false,
  "can_print": true,
  "can_modify": false,
  "is_encrypted": true
}
```

---

## 5) Ressources (réutilisées)

### 5.1 FontResource

```json
{
  "type": "FontResource",
  "id": "font_1",
  "family": "Times New Roman",
  "postscript_name": "TimesNewRomanPSMT",
  "embedded": true,
  "subset": true,
  "unicode_coverage": ["U+0000-00FF", "U+0100-017F"],
  "fallback_chain": ["Noto Serif", "DejaVu Serif"],
  "metrics": { "ascent": 0.88, "descent": -0.22 }
}
```

### 5.2 ImageResource / ColorSpaceResource / GraphicStateResource

```json
{
  "type": "GraphicStateResource",
  "id": "gs_1",
  "blend_mode": "multiply",
  "opacity": 0.7,
  "overprint": false
}
```

---

## 6) Enums utiles (à figer)

* Units: `pt | px | mm`
* Blend: `normal | multiply | screen | overlay | darken | lighten`
* Region kind: `header | footer | body | sidebar | margin | figure | table_area`
* Provenance source: `pdf_text | pdf_vector | ocr | vision | manual`
* Direction: `ltr | rtl | ttb`
* Wrap: `none | around_shapes | within_box`
* Reflow priority: `high | medium | low | locked`

---

## 7) Marquage “traduction” canonique (super important)

Au lieu de traduire “des objets”, tu traduis des **segments référencés**.

* Chaque objet traduisible a un `translation_key`
* Tu enregistres séparément un dictionnaire :

```json
{
  "translation_memory": {
    "seg_8891": { "src": "Bonjour le monde", "lang_src":"fr", "lang_tgt":"en", "tgt":"Hello world" },
    "wm_1": { "src": "CONFIDENTIEL", "tgt": "CONFIDENTIAL" }
  }
}
```

---

## 8) Ce qui est *vraiment* “TOUT” (la dernière couche qui pique)

Si tu veux un WYSIWYG qui survit aux PDF démoniaques :

* conserver `GlyphRun` **quand le PDF texte est non fiable**
* conserver `clip_path` et `graphic state`
* stocker `reading_order` + `flow`
* stocker `constraints` pour le reflow
* stocker `font coverage` + stratégie de fallback

---
---

# Document canonique — Objets extractibles pour traduction WYSIWYG

## 1) But réel d’un WYSIWYG

Une traduction WYSIWYG n’est pas “traduire du texte” : c’est **extraire une scène** (comme un moteur de rendu), **traduire certains nœuds**, puis **réinjecter** sans détruire :

* la géométrie (positions, boîtes, rotations),
* la typographie (métriques, polices, interlignage),
* l’empilement (z-order),
* les clips/masques (ce qui est “coupé”),
* l’interactivité (liens/formulaires),
* les ressources (fonts/images/couleurs),
* et la “source de vérité” (texte PDF vs OCR vs vision).

Le piège classique : croire qu’un document = texte.
Non. Un PDF est souvent une mosaïque de **glyphes** placés au pixel. Il faut donc modéliser **à la fois** :

* **un niveau haut** (`TextRun`) pour traduire proprement,
* **un niveau bas** (`GlyphRun`) pour survivre aux PDFs “démoniaques”.

---

## 2) Tout ce qu’un document peut contenir (à extraire/conserver)

### A — Structure & conteneurs

* Document, pages (taille, rotation, crop box)
* Calques (layers/OCG)
* Zones : header/footer/body/sidebar/margins/columns
* Flux de lecture (ordre de lecture multi-colonnes)
* Groupes (objets liés visuellement, héritent transform/z-index)

Pourquoi ? Sans structure/ordre, tu traduis “juste”, mais tu reconstruis “faux”.

---

### B — Texte & typographie (visible + métrique)

**Texte :**

* paragraphes, lignes, runs (segments), glyphes
* césures, ligatures, scripts, direction (LTR/RTL/TTB)
* langue par segment (docs mixtes)

**Typo :**

* police (famille + postscript), taille, gras/italique
* couleur, opacité, outline, ombre
* kerning/tracking, baseline shift, exposant/indice
* alignement, indentation, interligne, espaces avant/après

Pourquoi ? La traduction change la longueur. Sans métrique, ça déborde et tout “glisse”.

---

### C — Géométrie & rendu (la vérité du WYSIWYG)

* bbox (x,y,w,h), baseline
* matrices de transformation (rotation, skew, scale)
* z-order
* clipping/masques
* blend modes, overprint, transparences
* contraintes de reflow (boîte fixe vs extensible)

Pourquoi ? Perds le clip ou le z-order, et tu obtiens une salade de couches.

---

### D — Images & raster

* images (logos, photos, scans)
* DPI, crop, alpha/mask
* texte dans images (OCR + bboxes + style estimé)

Pourquoi ? Beaucoup de documents scannés ne contiennent PAS de texte exploitable sans OCR.

---

### E — Vectoriel & formes

* paths, shapes, flèches, icônes vectorielles
* fills/strokes, gradients/patterns
* diagrammes (souvent vecteurs + texte)

Pourquoi ? Tu ne traduis pas ces objets, mais ils structurent la page.

---

### F — Tableaux

* table, row, cell, padding
* bordures, fonds
* fusion rowspan/colspan
* “faux tableaux” (alignés à la main) → reconstruction

Pourquoi ? Les tableaux sont la zone où “ça casse” le plus facilement.

---

### G — Math / science

* équations natives (MathML/LaTeX/arbre)
* équations image + OCR math
* chimie (optionnel)
* règles : souvent **ne pas traduire** l’expression, mais préserver

---

### H — Interactivité / navigation

* liens externes + zones cliquables
* liens internes (sommaire, renvois)
* signets/bookmarks
* annotations (commentaires, surlignage, stamps)
* formulaires (text/checkbox/radio/dropdown/signature)
* actions (goTo/openURL/submitForm/launch)

Pourquoi ? Un PDF traduit “poster” est inutilisable.

---

### I — En-têtes/pieds & répétitions

* header/footer (répétés)
* pagination (Page X/Y)
* filigranes

Pourquoi ? Tu veux cohérence + éviter de retraduire 200 fois les mêmes éléments.

---

### J — Sécurité / signatures / pièces jointes

* permissions (copie/impression/modification)
* chiffrement
* signatures numériques
* fichiers embarqués

Pourquoi ? Un document signé doit parfois être **préservé** plutôt que régénéré.

---

## 3) La couche “invisible” qui fait échouer les projets (si oubliée)

### 3.1 Le PDF n’est pas du texte : c’est du placement de glyphes

Tu dois pouvoir capturer :

* ToUnicode/CMap (quand dispo)
* glyph id, unicode, advance width
* transform par glyph
* espaces implicites reconstruits

Sans ça, certains PDF “n’ont pas de mots”, juste des lettres éparpillées.

### 3.2 Reflow contrôlé (sinon WYSIWYG ≠ WYSIWYG)

Quand la traduction est plus longue, tes options (mesurées, bornées) :

* recasser les lignes
* micro-shrink font-size (limité)
* ajuster tracking (limité)
* élargir boîte si extensible
* déplacement d’objets voisins (à éviter, ou très encadré)

Donc tu dois stocker des **contraintes**.

### 3.3 Multi-sources = fusion + dédup obligatoire

Si tu utilises PDF text + OCR + vision, sans mécanisme explicite :

* tu dupliques le texte
* tu empiles plusieurs couches

Donc chaque objet doit porter une **provenance** et une **clé de déduplication**.

### 3.4 Polices : le mensonge pratique

Même “même police” peut être impossible :

* police non embarquée
* manque de glyphes langue cible (Vietnamien, Arabe…)
* fallback change métrique → layout explose

Donc il faut :

* inventaire fonts + couverture unicode
* stratégie de substitution “metrics-compatible”

---

## 4) Taxonomie canonique “TOUT” (types d’objets)

On modélise le document comme une **scène**.

### Conteneurs & structure

`Document`, `Page`, `Layer`, `Region`, `Flow`, `Group`

### Texte

`TextBlock`, `Paragraph`, `Line`, `TextRun`, `GlyphRun`, `RubyAnnotation`, `Footnote`, `Endnote`

### Éditorial

`List`, `ListItem`, `Heading`, `Caption`, `TOC`, `Index`, `Glossary`, `ReferenceEntry`

### Tableaux

`Table`, `TableRow`, `TableCell`

### Raster & médias

`ImageRaster`, `ImageInline`, `Mask`, `EmbeddedMedia`

### Vectoriel & graphes

`VectorPath`, `Shape`, `PatternFill`, `GradientFill`, `Chart`, `BarcodeOrQRCode`

### Math / science

`MathFormula`, `ChemFormula`, `EquationImage`

### Interactivité

`Link`, `LinkInternal`, `Bookmark`, `Action`, `Annotation`, `FormField`

### Header/Footer & page

`HeaderFooter`, `PageNumber`, `Watermark`

### Sécurité & attachments

`DigitalSignature`, `EmbeddedFile`, `Permission`

### Ressources

`FontResource`, `ImageResource`, `ColorSpaceResource`, `GraphicStateResource`

---

## 5) Champs communs à TOUS les objets (invariants)

Tout objet doit avoir :

* `id`, `type`, `page_id` (sauf Document/Ressources globales)
* `bbox`, `transform`, `z_index`
* `style_id` (ou null)
* `clip_path_id` (ou null)
* `provenance` (source + confiance + dédup)
* `translatable` + `translation_key` (si traduisible)

C’est le socle qui empêche le chaos.

---

## 6) Traduction canonique : on traduit des **segments référencés**

Au lieu de “modifier les objets”, tu maintiens un dictionnaire :

* `translation_memory[translation_key] = {src, tgt, lang_src, lang_tgt}`
* et tu réinjectes en respectant `reflow_policy`.

C’est la méthode la plus robuste et testable.

---

# SPEC JSON COMPLÈTE — DocumentSchema v1

```json
{
  "schema_id": "DocumentSchema.v1",
  "version": "1.0.0",
  "units": ["pt", "px", "mm"],

  "enums": {
    "ObjectType": [
      "Document","Page","Layer","Region","Flow","Group",
      "TextBlock","Paragraph","Line","TextRun","GlyphRun","RubyAnnotation","Footnote","Endnote",
      "List","ListItem","Heading","Caption","TOC","Index","Glossary","ReferenceEntry",
      "Table","TableRow","TableCell",
      "ImageRaster","ImageInline","Mask","EmbeddedMedia",
      "VectorPath","Shape","PatternFill","GradientFill","Chart","BarcodeOrQRCode",
      "MathFormula","ChemFormula","EquationImage",
      "Link","LinkInternal","Bookmark","Action","Annotation","FormField",
      "HeaderFooter","PageNumber","Watermark",
      "DigitalSignature","EmbeddedFile","Permission",
      "FontResource","ImageResource","ColorSpaceResource","GraphicStateResource"
    ],

    "RegionKind": ["header","footer","body","sidebar","margin","figure_area","table_area","unknown"],
    "FlowKind": ["main_reading","footnotes","headers","tables","figures","unknown"],
    "WrapMode": ["none","around_shapes","within_box"],
    "BlendMode": ["normal","multiply","screen","overlay","darken","lighten","color_burn","color_dodge"],
    "Direction": ["ltr","rtl","ttb"],
    "Script": ["normal","superscript","subscript"],
    "ProvenanceSource": ["pdf_text","pdf_vector","ocr","vision","manual"],
    "ReflowPriority": ["high","medium","low","locked"],
    "AnnotationKind": ["comment","highlight","underline","stamp","note"],
    "FormFieldKind": ["text","checkbox","radio","dropdown","signature"],
    "ActionKind": ["open_url","go_to","submit_form","launch"],
    "ShapeKind": ["rect","circle","line","arrow","polygon","polyline"],
    "ChartKind": ["bar","line","pie","scatter","area","unknown"],
    "BarcodeKind": ["qr","code128","ean13","datamatrix","pdf417","unknown"],
    "PermissionStatus": ["unknown","unencrypted","encrypted"]
  },

  "definitions": {
    "BBox": {
      "required": ["x","y","w","h","unit"],
      "properties": {
        "x": {"type":"number"},
        "y": {"type":"number"},
        "w": {"type":"number"},
        "h": {"type":"number"},
        "unit": {"enum":["pt","px","mm"]}
      }
    },

    "Transform": {
      "required": ["a","b","c","d","e","f"],
      "properties": {
        "a": {"type":"number"},
        "b": {"type":"number"},
        "c": {"type":"number"},
        "d": {"type":"number"},
        "e": {"type":"number"},
        "f": {"type":"number"}
      }
    },

    "Fill": {
      "required": ["kind"],
      "properties": {
        "kind": {"enum":["solid","none","gradient","pattern"]},
        "color": {"type":"string"},
        "gradient_id": {"type":["string","null"]},
        "pattern_id": {"type":["string","null"]}
      }
    },

    "Stroke": {
      "required": ["enabled"],
      "properties": {
        "enabled": {"type":"boolean"},
        "color": {"type":"string"},
        "width": {"type":"number"},
        "dash": {"type":"array", "items":{"type":"number"}}
      }
    },

    "Style": {
      "required": ["style_id","opacity","blend_mode","fill","stroke","overprint"],
      "properties": {
        "style_id": {"type":"string"},
        "opacity": {"type":"number"},
        "blend_mode": {"enum":["normal","multiply","screen","overlay","darken","lighten","color_burn","color_dodge"]},
        "overprint": {"type":"boolean"},
        "fill": {"$ref":"#definitions/Fill"},
        "stroke": {"$ref":"#definitions/Stroke"},
        "shadow": {"type":["object","null"]},
        "outline": {"type":["object","null"]}
      }
    },

    "FontRef": {
      "required": ["family","postscript_name","embedded"],
      "properties": {
        "family": {"type":"string"},
        "postscript_name": {"type":"string"},
        "embedded": {"type":"boolean"}
      }
    },

    "TextStyle": {
      "required": ["font","size_pt","weight","italic","underline","strike","color","tracking","kerning","baseline_shift","script","language","direction"],
      "properties": {
        "font": {"$ref":"#definitions/FontRef"},
        "size_pt": {"type":"number"},
        "weight": {"type":"integer"},
        "italic": {"type":"boolean"},
        "underline": {"type":"boolean"},
        "strike": {"type":"boolean"},
        "color": {"type":"string"},
        "tracking": {"type":"number"},
        "kerning": {"type":"boolean"},
        "baseline_shift": {"type":"number"},
        "script": {"enum":["normal","superscript","subscript"]},
        "language": {"type":"string"},
        "direction": {"enum":["ltr","rtl","ttb"]}
      }
    },

    "ParagraphStyle": {
      "required": ["align","line_height","space_before_pt","space_after_pt","indent_first_line_pt","indent_left_pt","indent_right_pt","hyphenation"],
      "properties": {
        "align": {"enum":["left","right","center","justify"]},
        "line_height": {"type":"number"},
        "space_before_pt": {"type":"number"},
        "space_after_pt": {"type":"number"},
        "indent_first_line_pt": {"type":"number"},
        "indent_left_pt": {"type":"number"},
        "indent_right_pt": {"type":"number"},
        "hyphenation": {"enum":["off","auto"]}
      }
    },

    "Reading": {
      "required": ["order","flow_id"],
      "properties": {
        "order": {"type":"integer"},
        "flow_id": {"type":"string"},
        "column_id": {"type":["string","null"]},
        "paragraph_id": {"type":["string","null"]}
      }
    },

    "Provenance": {
      "required": ["source","source_object_ref","confidence","dedupe_key","merged_from"],
      "properties": {
        "source": {"enum":["pdf_text","pdf_vector","ocr","vision","manual"]},
        "source_object_ref": {"type":"string"},
        "confidence": {"type":"number"},
        "dedupe_key": {"type":"string"},
        "merged_from": {"type":"array","items":{"type":"string"}}
      }
    },

    "ReflowPolicy": {
      "required": ["priority","max_font_shrink_pct","max_tracking_expand","allow_line_breaks"],
      "properties": {
        "priority": {"enum":["high","medium","low","locked"]},
        "max_font_shrink_pct": {"type":"number"},
        "max_tracking_expand": {"type":"number"},
        "allow_line_breaks": {"type":"boolean"}
      }
    },

    "BaseObject": {
      "required": ["id","type","bbox","transform","rotation_deg","z_index","visible","locked","style_id","clip_path_id","tags","provenance","translatable","translation_key"],
      "properties": {
        "id": {"type":"string"},
        "type": {"enum":[
          "Document","Page","Layer","Region","Flow","Group",
          "TextBlock","Paragraph","Line","TextRun","GlyphRun","RubyAnnotation","Footnote","Endnote",
          "List","ListItem","Heading","Caption","TOC","Index","Glossary","ReferenceEntry",
          "Table","TableRow","TableCell",
          "ImageRaster","ImageInline","Mask","EmbeddedMedia",
          "VectorPath","Shape","PatternFill","GradientFill","Chart","BarcodeOrQRCode",
          "MathFormula","ChemFormula","EquationImage",
          "Link","LinkInternal","Bookmark","Action","Annotation","FormField",
          "HeaderFooter","PageNumber","Watermark",
          "DigitalSignature","EmbeddedFile","Permission",
          "FontResource","ImageResource","ColorSpaceResource","GraphicStateResource"
        ]},
        "page_id": {"type":["string","null"]},
        "parent_id": {"type":["string","null"]},
        "children": {"type":"array","items":{"type":"string"}},

        "bbox": {"$ref":"#definitions/BBox"},
        "transform": {"$ref":"#definitions/Transform"},
        "rotation_deg": {"type":"number"},
        "z_index": {"type":"integer"},
        "clip_path_id": {"type":["string","null"]},

        "visible": {"type":"boolean"},
        "locked": {"type":"boolean"},

        "style_id": {"type":["string","null"]},
        "tags": {"type":"array","items":{"type":"string"}},

        "reading": {"type":["object","null"], "properties": {}},
        "provenance": {"$ref":"#definitions/Provenance"},

        "translatable": {"type":"boolean"},
        "translation_key": {"type":["string","null"]}
      }
    }
  },

  "object_specs": {
    "Document": {
      "extends": "BaseObject",
      "required": ["meta","permissions_id","resources","pages"],
      "properties": {
        "meta": {
          "required": ["title","author","subject","keywords","created_at","modified_at","declared_language","producer","pdf_version"],
          "properties": {
            "title": {"type":"string"},
            "author": {"type":"string"},
            "subject": {"type":"string"},
            "keywords": {"type":"array","items":{"type":"string"}},
            "created_at": {"type":"string"},
            "modified_at": {"type":"string"},
            "declared_language": {"type":"string"},
            "producer": {"type":"string"},
            "pdf_version": {"type":"string"}
          }
        },
        "permissions_id": {"type":"string"},
        "resources": {
          "required": ["fonts","images","colorspaces","graphic_states","styles"],
          "properties": {
            "fonts": {"type":"array","items":{"type":"string"}},
            "images": {"type":"array","items":{"type":"string"}},
            "colorspaces": {"type":"array","items":{"type":"string"}},
            "graphic_states": {"type":"array","items":{"type":"string"}},
            "styles": {"type":"array","items":{"type":"string"}}
          }
        },
        "pages": {"type":"array","items":{"type":"string"}}
      }
    },

    "Page": {
      "extends": "BaseObject",
      "required": ["index","size_pt","rotation_deg","crop_box","layers","regions","objects"],
      "properties": {
        "index": {"type":"integer"},
        "size_pt": {"required":["w","h"], "properties":{"w":{"type":"number"},"h":{"type":"number"}}},
        "crop_box": {"$ref":"#definitions/BBox"},
        "layers": {"type":"array","items":{"type":"string"}},
        "regions": {"type":"array","items":{"type":"string"}},
        "objects": {"type":"array","items":{"type":"string"}}
      }
    },

    "Layer": {
      "extends": "BaseObject",
      "required": ["name","visible"],
      "properties": {
        "name": {"type":"string"},
        "visible": {"type":"boolean"}
      }
    },

    "Region": {
      "extends": "BaseObject",
      "required": ["region_kind","layout","objects"],
      "properties": {
        "region_kind": {"enum":["header","footer","body","sidebar","margin","figure_area","table_area","unknown"]},
        "layout": {"type":"object", "properties":{"columns":{"type":"integer"},"gutter_pt":{"type":"number"}}},
        "objects": {"type":"array","items":{"type":"string"}}
      }
    },

    "Flow": {
      "extends": "BaseObject",
      "required": ["flow_kind","sequence"],
      "properties": {
        "flow_kind": {"enum":["main_reading","footnotes","headers","tables","figures","unknown"]},
        "sequence": {"type":"array","items":{"type":"string"}}
      }
    },

    "Group": {
      "extends": "BaseObject",
      "required": ["objects","group_transform_inherited"],
      "properties": {
        "objects": {"type":"array","items":{"type":"string"}},
        "group_transform_inherited": {"type":"boolean"}
      }
    },

    "TextBlock": {
      "extends": "BaseObject",
      "required": ["content_kind","paragraphs","wrap_mode","constraints"],
      "properties": {
        "content_kind": {"enum":["body","quote","code","callout","unknown"]},
        "paragraphs": {"type":"array","items":{"type":"string"}},
        "wrap_mode": {"enum":["none","around_shapes","within_box"]},
        "constraints": {
          "required":["box_fixed","allow_reflow"],
          "properties": {"box_fixed":{"type":"boolean"},"allow_reflow":{"type":"boolean"}}
        },
        "paragraph_style": {"$ref":"#definitions/ParagraphStyle"}
      }
    },

    "Paragraph": {
      "extends": "BaseObject",
      "required": ["lines","list_ref","heading_level","paragraph_style"],
      "properties": {
        "lines": {"type":"array","items":{"type":"string"}},
        "list_ref": {"type":["string","null"]},
        "heading_level": {"type":["integer","null"]},
        "paragraph_style": {"$ref":"#definitions/ParagraphStyle"}
      }
    },

    "Line": {
      "extends": "BaseObject",
      "required": ["runs","baseline_y"],
      "properties": {
        "runs": {"type":"array","items":{"type":"string"}},
        "baseline_y": {"type":"number"}
      }
    },

    "TextRun": {
      "extends": "BaseObject",
      "required": ["text","normalized_text","language","direction","text_style","segments","reflow_policy"],
      "properties": {
        "text": {"type":"string"},
        "normalized_text": {"type":"string"},
        "language": {"type":"string"},
        "direction": {"enum":["ltr","rtl","ttb"]},
        "text_style": {"$ref":"#definitions/TextStyle"},
        "segments": {
          "type":"array",
          "items": {
            "required":["seg_id","text","bbox"],
            "properties":{
              "seg_id":{"type":"string"},
              "text":{"type":"string"},
              "bbox":{"$ref":"#definitions/BBox"}
            }
          }
        },
        "reflow_policy": {"$ref":"#definitions/ReflowPolicy"}
      }
    },

    "GlyphRun": {
      "extends": "BaseObject",
      "required": ["font_resource_id","glyphs"],
      "properties": {
        "font_resource_id": {"type":"string"},
        "glyphs": {
          "type":"array",
          "items": {
            "required":["gid","unicode","x","y","advance"],
            "properties":{
              "gid":{"type":"integer"},
              "unicode":{"type":"string"},
              "x":{"type":"number"},
              "y":{"type":"number"},
              "advance":{"type":"number"}
            }
          }
        }
      }
    },

    "RubyAnnotation": {
      "extends": "BaseObject",
      "required": ["base_run_id","ruby_text","position"],
      "properties": {
        "base_run_id": {"type":"string"},
        "ruby_text": {"type":"string"},
        "position": {"enum":["above","below"]}
      }
    },

    "Footnote": {
      "extends": "BaseObject",
      "required": ["marker","anchor_object_id","note_textblock_id"],
      "properties": {
        "marker": {"type":"string"},
        "anchor_object_id": {"type":"string"},
        "note_textblock_id": {"type":"string"}
      }
    },

    "Endnote": {
      "extends": "BaseObject",
      "required": ["marker","anchor_object_id","note_textblock_id"],
      "properties": {
        "marker": {"type":"string"},
        "anchor_object_id": {"type":"string"},
        "note_textblock_id": {"type":"string"}
      }
    },

    "Heading": {
      "extends": "BaseObject",
      "required": ["level","textblock_id"],
      "properties": {
        "level": {"type":"integer"},
        "textblock_id": {"type":"string"}
      }
    },

    "List": {
      "extends": "BaseObject",
      "required": ["list_kind","level","items"],
      "properties": {
        "list_kind": {"enum":["bullet","numbered"]},
        "level": {"type":"integer"},
        "items": {"type":"array","items":{"type":"string"}}
      }
    },

    "ListItem": {
      "extends": "BaseObject",
      "required": ["marker_text","content_textblock_id"],
      "properties": {
        "marker_text": {"type":"string"},
        "content_textblock_id": {"type":"string"}
      }
    },

    "Caption": {
      "extends": "BaseObject",
      "required": ["target_object_id","textblock_id"],
      "properties": {
        "target_object_id": {"type":"string"},
        "textblock_id": {"type":"string"}
      }
    },

    "TOC": {
      "extends": "BaseObject",
      "required": ["entries"],
      "properties": {
        "entries": {
          "type":"array",
          "items":{
            "required":["title","target_page_index","target_object_id","level"],
            "properties":{
              "title":{"type":"string"},
              "target_page_index":{"type":"integer"},
              "target_object_id":{"type":["string","null"]},
              "level":{"type":"integer"}
            }
          }
        }
      }
    },

    "Index": {
      "extends": "BaseObject",
      "required": ["terms"],
      "properties": {
        "terms": {
          "type":"array",
          "items":{
            "required":["term","targets"],
            "properties":{
              "term":{"type":"string"},
              "targets":{"type":"array","items":{"type":"object"}}
            }
          }
        }
      }
    },

    "Glossary": {
      "extends": "BaseObject",
      "required": ["entries"],
      "properties": {
        "entries": {
          "type":"array",
          "items":{
            "required":["term","definition","translation_policy"],
            "properties":{
              "term":{"type":"string"},
              "definition":{"type":"string"},
              "translation_policy":{"enum":["translate_all","keep_term","translate_definition_only"]}
            }
          }
        }
      }
    },

    "ReferenceEntry": {
      "extends": "BaseObject",
      "required": ["raw","fields","translation_policy"],
      "properties": {
        "raw": {"type":"string"},
        "fields": {"type":"object"},
        "translation_policy": {"enum":["keep_names","translate_title_only","translate_all"]}
      }
    },

    "Table": {
      "extends": "BaseObject",
      "required": ["grid","rows","structure_confidence","is_visual_reconstructed"],
      "properties": {
        "grid": {"required":["rows","cols"], "properties":{"rows":{"type":"integer"},"cols":{"type":"integer"}}},
        "rows": {"type":"array","items":{"type":"string"}},
        "structure_confidence": {"type":"number"},
        "is_visual_reconstructed": {"type":"boolean"}
      }
    },

    "TableRow": {
      "extends": "BaseObject",
      "required": ["cells","row_index"],
      "properties": {
        "row_index": {"type":"integer"},
        "cells": {"type":"array","items":{"type":"string"}}
      }
    },

    "TableCell": {
      "extends": "BaseObject",
      "required": ["row","col","rowspan","colspan","padding_pt","background_fill","content"],
      "properties": {
        "row": {"type":"integer"},
        "col": {"type":"integer"},
        "rowspan": {"type":"integer"},
        "colspan": {"type":"integer"},
        "padding_pt": {"required":["t","r","b","l"], "properties":{"t":{"type":"number"},"r":{"type":"number"},"b":{"type":"number"},"l":{"type":"number"}}},
        "background_fill": {"$ref":"#definitions/Fill"},
        "content": {"type":"array","items":{"type":"string"}}
      }
    },

    "ImageRaster": {
      "extends": "BaseObject",
      "required": ["image_resource_id","dpi","color_space_id","crop","mask_id"],
      "properties": {
        "image_resource_id": {"type":"string"},
        "dpi": {"type":"number"},
        "color_space_id": {"type":"string"},
        "crop": {"required":["x","y","w","h"], "properties":{"x":{"type":"number"},"y":{"type":"number"},"w":{"type":"number"},"h":{"type":"number"}}},
        "mask_id": {"type":["string","null"]}
      }
    },

    "ImageInline": {
      "extends": "ImageRaster",
      "required": ["baseline_align"],
      "properties": {
        "baseline_align": {"enum":["baseline","middle","top","bottom"]}
      }
    },

    "Mask": {
      "extends": "BaseObject",
      "required": ["mask_kind","resource_id"],
      "properties": {
        "mask_kind": {"enum":["alpha","luminosity","softmask"]},
        "resource_id": {"type":"string"}
      }
    },

    "EmbeddedMedia": {
      "extends": "BaseObject",
      "required": ["media_kind","resource_id","mime"],
      "properties": {
        "media_kind": {"enum":["audio","video"]},
        "resource_id": {"type":"string"},
        "mime": {"type":"string"}
      }
    },

    "VectorPath": {
      "extends": "BaseObject",
      "required": ["d","is_clipping_path"],
      "properties": {
        "d": {"type":"string"},
        "is_clipping_path": {"type":"boolean"}
      }
    },

    "Shape": {
      "extends": "BaseObject",
      "required": ["shape_kind","params"],
      "properties": {
        "shape_kind": {"enum":["rect","circle","line","arrow","polygon","polyline"]},
        "params": {"type":"object"}
      }
    },

    "PatternFill": {
      "extends": "BaseObject",
      "required": ["pattern_id","tile_bbox"],
      "properties": {
        "pattern_id": {"type":"string"},
        "tile_bbox": {"$ref":"#definitions/BBox"}
      }
    },

    "GradientFill": {
      "extends": "BaseObject",
      "required": ["gradient_id","stops"],
      "properties": {
        "gradient_id": {"type":"string"},
        "stops": {"type":"array","items":{"type":"object"}}
      }
    },

    "Chart": {
      "extends": "BaseObject",
      "required": ["chart_kind","data","labels_text_objects"],
      "properties": {
        "chart_kind": {"enum":["bar","line","pie","scatter","area","unknown"]},
        "data": {"type":"object"},
        "labels_text_objects": {"type":"array","items":{"type":"string"}}
      }
    },

    "BarcodeOrQRCode": {
      "extends": "BaseObject",
      "required": ["code_kind","payload","preserve_exact_pixels"],
      "properties": {
        "code_kind": {"enum":["qr","code128","ean13","datamatrix","pdf417","unknown"]},
        "payload": {"type":"string"},
        "preserve_exact_pixels": {"type":"boolean"}
      }
    },

    "MathFormula": {
      "extends": "BaseObject",
      "required": ["format","source","ast","keep_as_is"],
      "properties": {
        "format": {"enum":["mathml","latex","native"]},
        "source": {"type":"string"},
        "ast": {"type":"object"},
        "keep_as_is": {"type":"boolean"}
      }
    },

    "ChemFormula": {
      "extends": "BaseObject",
      "required": ["format","source","ast","keep_as_is"],
      "properties": {
        "format": {"enum":["smiles","inchi","native","unknown"]},
        "source": {"type":"string"},
        "ast": {"type":"object"},
        "keep_as_is": {"type":"boolean"}
      }
    },

    "EquationImage": {
      "extends": "BaseObject",
      "required": ["image_resource_id","math_ocr"],
      "properties": {
        "image_resource_id": {"type":"string"},
        "math_ocr": {
          "required":["latex","confidence"],
          "properties":{"latex":{"type":"string"},"confidence":{"type":"number"}}
        }
      }
    },

    "Link": {
      "extends": "BaseObject",
      "required": ["target","rects"],
      "properties": {
        "target": {"type":"string"},
        "rects": {"type":"array","items":{"$ref":"#definitions/BBox"}}
      }
    },

    "LinkInternal": {
      "extends": "BaseObject",
      "required": ["target_page_index","target_object_id"],
      "properties": {
        "target_page_index": {"type":"integer"},
        "target_object_id": {"type":["string","null"]}
      }
    },

    "Bookmark": {
      "extends": "BaseObject",
      "required": ["title","target_page_index","level"],
      "properties": {
        "title": {"type":"string"},
        "target_page_index": {"type":"integer"},
        "level": {"type":"integer"}
      }
    },

    "Action": {
      "extends": "BaseObject",
      "required": ["action_kind","payload"],
      "properties": {
        "action_kind": {"enum":["open_url","go_to","submit_form","launch"]},
        "payload": {"type":"object"}
      }
    },

    "Annotation": {
      "extends": "BaseObject",
      "required": ["annotation_kind","author","created_at","content"],
      "properties": {
        "annotation_kind": {"enum":["comment","highlight","underline","stamp","note"]},
        "author": {"type":"string"},
        "created_at": {"type":"string"},
        "content": {"type":"string"}
      }
    },

    "FormField": {
      "extends": "BaseObject",
      "required": ["field_kind","name","value","rect","constraints"],
      "properties": {
        "field_kind": {"enum":["text","checkbox","radio","dropdown","signature"]},
        "name": {"type":"string"},
        "value": {"type":["string","boolean","null"]},
        "rect": {"$ref":"#definitions/BBox"},
        "constraints": {"type":"object"}
      }
    },

    "HeaderFooter": {
      "extends": "BaseObject",
      "required": ["kind","objects","repeat_on_pages"],
      "properties": {
        "kind": {"enum":["header","footer"]},
        "objects": {"type":"array","items":{"type":"string"}},
        "repeat_on_pages": {"enum":["all","odd","even","range"]}
      }
    },

    "PageNumber": {
      "extends": "BaseObject",
      "required": ["format"],
      "properties": {
        "format": {"type":"string"}
      }
    },

    "Watermark": {
      "extends": "BaseObject",
      "required": ["text","opacity","rotation_deg"],
      "properties": {
        "text": {"type":"string"},
        "opacity": {"type":"number"},
        "rotation_deg": {"type":"number"}
      }
    },

    "DigitalSignature": {
      "extends": "BaseObject",
      "required": ["status","covers_entire_document","preserve_original"],
      "properties": {
        "status": {"enum":["valid","invalid","unknown"]},
        "covers_entire_document": {"type":"boolean"},
        "preserve_original": {"type":"boolean"}
      }
    },

    "EmbeddedFile": {
      "extends": "BaseObject",
      "required": ["filename","mime","size_bytes"],
      "properties": {
        "filename": {"type":"string"},
        "mime": {"type":"string"},
        "size_bytes": {"type":"integer"}
      }
    },

    "Permission": {
      "extends": "BaseObject",
      "required": ["can_copy","can_print","can_modify","status"],
      "properties": {
        "can_copy": {"type":"boolean"},
        "can_print": {"type":"boolean"},
        "can_modify": {"type":"boolean"},
        "status": {"enum":["unknown","unencrypted","encrypted"]}
      }
    },

    "FontResource": {
      "extends": "BaseObject",
      "required": ["family","postscript_name","embedded","subset","unicode_coverage","fallback_chain","metrics"],
      "properties": {
        "family": {"type":"string"},
        "postscript_name": {"type":"string"},
        "embedded": {"type":"boolean"},
        "subset": {"type":"boolean"},
        "unicode_coverage": {"type":"array","items":{"type":"string"}},
        "fallback_chain": {"type":"array","items":{"type":"string"}},
        "metrics": {"type":"object"}
      }
    },

    "ImageResource": {
      "extends": "BaseObject",
      "required": ["mime","width_px","height_px","bytes_sha256"],
      "properties": {
        "mime": {"type":"string"},
        "width_px": {"type":"integer"},
        "height_px": {"type":"integer"},
        "bytes_sha256": {"type":"string"}
      }
    },

    "ColorSpaceResource": {
      "extends": "BaseObject",
      "required": ["name","profile"],
      "properties": {
        "name": {"type":"string"},
        "profile": {"type":"object"}
      }
    },

    "GraphicStateResource": {
      "extends": "BaseObject",
      "required": ["blend_mode","opacity","overprint"],
      "properties": {
        "blend_mode": {"enum":["normal","multiply","screen","overlay","darken","lighten","color_burn","color_dodge"]},
        "opacity": {"type":"number"},
        "overprint": {"type":"boolean"}
      }
    }
  },

  "translation_memory": {
    "description": "Dictionnaire de traductions par translation_key",
    "entry_shape": {
      "required": ["src","tgt","lang_src","lang_tgt"],
      "properties": {
        "src": {"type":"string"},
        "tgt": {"type":"string"},
        "lang_src": {"type":"string"},
        "lang_tgt": {"type":"string"},
        "notes": {"type":["string","null"]}
      }
    }
  },

  "examples": [
    {
      "name": "Minimal PDF natif (texte)",
      "document_shape_hint": {
        "Document": "doc_1",
        "Pages": ["page_1"],
        "Objects": ["region_body","tb_1","para_1","line_1","run_1"]
      }
    },
    {
      "name": "Scan + OCR (texte reconstruit)",
      "document_shape_hint": {
        "Document": "doc_ocr_1",
        "Pages": ["page_1"],
        "Objects": ["img_scan_1","run_ocr_1"]
      }
    },
    {
      "name": "Multi-colonnes + liens + table",
      "document_shape_hint": {
        "Document": "doc_mc_1",
        "Pages": ["page_1"],
        "Objects": ["flow_main","region_body","table_1","link_1"]
      }
    }
  ]
}
```

Si tu veux pousser jusqu’au niveau “contrat de validation”, je peux te donner (dans la foulée) une liste d’**invariants testables** (ex: aucun `TextRun` traduisible sans `reflow_policy`, aucun objet sans `provenance.dedupe_key`, etc.) — c’est ce qui évite les régressions quand tu ajoutes OCR/vision plus tard.





