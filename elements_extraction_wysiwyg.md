Pour reconstruire **fidèlement** un document en WYSIWYG (notamment dans le cadre du projet VERSE), voici la liste **exhaustive** des éléments à extraire, classés par catégorie. Chaque élément est essentiel pour garantir l'équivalence visuelle et structurelle parfaite.

---

### **1. MÉTADONNÉES DOCUMENT GLOBALES**
- Format source exact (PDF v1.x, DOCX OOXML vXX, PPTX, etc.)
- Dimensions standard des pages (A4, Letter, etc.) ou custom
- Nombre total de pages
- Orientation globale (portrait/landscape) par page
- Encodage du fichier, profil ICC intégré
- Dictionnaire de polices embarquées vs références système
- Version de génération du document (logiciel créateur)
- Histoire du document (versions, modifications, métadonnées XMP)
- Permissions et restrictions (DRM, mot de passe)
- Nombre de couches dans un PDF (annotations, formulaires, etc.)

---

### **2. MÉTADONNÉES PAGE PAR PAGE**
- Dimensions exactes (width, height) en unités (points, mm, etc.)
- Résolution effective (DPI réel, non théorique)
- Rotation appliquée (0°, 90°, 180°, 270°)
- Zone de média (media box), zone de crop (crop box), zone de contenu (trim/bleed boxes)
- Couleur de fond (background color) ou image de fond
- Transparence de la page (alpha channel)
- Index de la page dans l'ordre de lecture vs ordre physique
- Paramètres de colonnes (gutters, séparateurs)

---

### **3. GÉOMÉTRIE & MISE EN PAGE ABSOLUE**
- **Bounding boxes précis** pour chaque objet (x0, y0, x1, y1) en coordonnées absolues
- **Baseline** exacte de chaque ligne de texte (y, pente, courbure si texte curviligne)
- **Contours** des blocs (polygones non rectangulaires, trapezes)
- **Colonne detection** : nombre de colonnes, largeur de chaque colonne, gutter entre colonnes
- **Zones de séparation** : blancs stratégiques, espaces verticaux/horizontaux
- **Alignement** : left, right, center, justified (avec type de justification : inter-mot, inter-caractère, inter-cadratin)
- **Indentation** : first-line indent, hanging indent, left/right indent en valeur exacte
- **Spacing** : 
  - Space before/after paragraphe (exact value)
  - Interligne (line spacing exact : single, 1.5, double, at least, exactly, multiple)
  - Leading (espace entre baseline)
  - Kerning pairs individuels entre caractères
  - Tracking (letter-spacing global)
- **Margins** : top, bottom, left, right, header, footer, gutters
- **Pagination** : points de coupure, veuves et orphelines, keep with next, keep lines together

---

### **4. TYPOGRAPHIE (NIVEAU CARACTÈRE/GLYPH)**
- **Font family** exacte (nom PostScript, nom complet système)
- **Font subset** utilisé (quel glyphs inclus)
- **Font size** en points (valeur réelle, non arrondie)
- **Font weight** (100 à 900, thin, light, normal, bold, black)
- **Font style** : normal, italic, oblique, backslant
- **Font variant** : small-caps, petite-caps, titling-caps
- **Font stretch** : condensed, expanded, valeur % exacte
- **Rendering mode** : mode de hinting, anti-aliasing, subpixel rendering
- **Glyph id** dans la font (pour substitution ligature)
- **Ligatures** : fi, fl, ffi, ffl, etc. (glyph unique vs caractères séparés)
- **Kerning** : paires spécifiques (A-V, T-o, etc.) et valeurs
- **Position** : normal, superscript, subscript, ordinals
- **Underline** : style (simple, double, traitillé), épaisseur, offset vertical, couleur
- **Strikethrough** : style, épaisseur, offset
- **Outline** : contour, épaisseur, couleur du contour
- **Fill** : couleur de remplissage du glyph, dégradé, pattern
- **Shadow** : offset x/y, blur, couleur, mode de fusion
- **Effects** : emboss, engrave, glow, reflection
- **OpenType features** : ligatures, contextual alternates, stylistic sets, number forms (lining, oldstyle), etc.
- **Font embedding** : type (subset, complet), permissions

---

### **5. COLORIMÉTIRE & ESTHÉTIQUE**
- **Couleur texte** : espace colorimétrique exact (RGB, CMYK, Lab, ICC-based), valeurs, profondeur (8-bit, 16-bit)
- **Couleur de fond** des blocs (y compris transparence alpha)
- **Dégradés** : type (linéaire, radial), angles, stops, couleurs, espaces colorimétriques
- **Patterns** : motifs de remplissage, images de fond répétées
- **Bordures** : style (solid, dashed, dotted, double), épaisseur, couleur, espacement, arrondi (border-radius)
- **Shading** : patterns de trame (hachures, points), opacité
- **Material/Texture** : effets 3D, relief, verre, métal
- **Highlights** : zones en surbrillance (jaune fluo, etc.)

---

### **6. CONTENU TEXTUEL & SÉMANTIQUE**
- **String unicode exacte** (avec contrôle de bidirectionnalité)
- **Direction du texte** : LTR, RTL, TTB (top-to-bottom)
- **Script** : latin, cyrillique, arabe, han, etc.
- **Langue** : tag BCP-47 (fr-FR, en-US, etc.)
- **Numéro de bloc** dans la hiérarchie
- **Numéro de ligne** dans le bloc
- **Numéro de token/mot** dans la ligne
- **Rôles sémantiques** :
  - Titre (niveau H1 à H6, numérotation)
  - Paragraphe normal
  - Citation (block quotes)
  - Listes (ordered/unordered, niveau d'imbrication, style de puce, valeur numérique)
  - Légende (figure, table)
  - Note de bas de page / note de fin
  - Référence croisée (hyperlien interne, signet)
  - Champ de formulaire
  - Commentaire/revision
  - En-tête/pied de page (header/footer)
  - Numéro de page
  - Filigrane (watermark)
  - Mention légale
- **Annotations** : commentaires, highlights, révisions (insertion, deletion)
- **Hyperliens** : URL, action, destination, bounding box cliquable
- **Signets** : nom, destination, hiérarchie
- **Metadata du contenu** : sujet, mots-clés, catégorie

---

### **7. STRUCTURE DE TABLEAUX**
- **Bornes géométriques** exactes du tableau
- **Nombre de lignes et colonnes** (avant/après spans)
- **Cellules fusionnées** : row span, column span
- **Contenu de chaque cellule** (texte formaté complet avec tout le style ci-dessus)
- **Bordures de cellules** : chaque côté individuel (top, bottom, left, right) avec style, épaisseur, couleur
- **Fond de cellule** : couleur, dégradé
- **Alignement vertical** dans la cellule (top, middle, bottom, justified)
- **Padding** de cellule (top, bottom, left, right)
- **Lignes de séparation** : ruling lines (épaisseur, couleur, style)
- **En-têtes** : identification des lignes/colonnes d'en-tête, répétition en haut de page
- **Direction texte** dans la cellule
- **Formule de cellule** (si tableau calculé, avec références)
- **Type de données** : texte, nombre, date, monnaie
- **Format numérique** : décimales, séparateurs, symbole monétaire

---

### **8. OBJETS IMAGES & GRAPHIQUES**
- **Format** : JPEG, PNG, TIFF, BMP, SVG, WMF, EMF
- **Résolution native** (DPI)
- **Dimensions** pixels (width, height)
- **Bounding box** sur la page
- **Méthode d'intégration** : embarquée (inline), flottante (wrap), en arrière-plan
- **Transparence** : canal alpha, mask
- **Crop** : zone visible si image rognée
- **Rotation/flip** : angle, miroir horizontal/vertical
- **Effets images** : ombre portée, réflexion, bordure artistique, recoloration, brightness/contrast
- **Alternative text** (alt text, accessibility)
- **Cluster d'images** : si image découpée en morceaux (effet sprite)
- **Vector graphics** : chemins SVG, couleurs de remplissage, gradients, strokes
- **SmartArt / Diagram** : structure de données sous-jacente, layout automatique

---

### **9. OBJETS SPÉCIAUX & ÉMBARQUÉS**
- **Formules mathématiques** : structure MathML, LaTeX, OMML, avec style (italique variables, gras vecteurs)
- **Graphiques/charts** : type (barre, ligne, camembert), données sources, titres, légendes, axes, grilles
- **Objets OLE** : type, données binaires, icône de présentation
- **Vidéos/audio** : codecs, durée, thumbnail, contrôles
- **Boutons & formulaires** : champs texte, cases à cocher, listes déroulantes, actions JavaScript
- **Signatures numériques** : position, certificat, état de validation
- **Annotations** : ink (dessin main levée), tampons (stamps), highlights, underline, strikeout
- **Cadres (Frames)** : zones de texte positionnées, liées entre elles (text flow)
- **Table des matières** : structure hiérarchique, numéros de page, hyperliens
- **Index/glossaire** : entrées, références croisées

---

### **10. FLUX & ORDRE DE LECTURE**
- **Ordre de lecture** séquentiel (liste de tokens dans l'ordre)
- **Groupes de blocs** : colonnes, sections, zones de contenu principal vs annexes
- **Relations** : 
  - "caption-of" (légende liée à image/tableau)
  - "footnote-of" (note liée à référence)
  - "reference-to" (citation vers bibliographie)
  - "flow continuation" (suite de colonne, suite de cadre)
  - "header/footer-for" (lien avec section)
- **Baselines multiples** : texte vertical, texte sur courbe, texte roté
- **Wrap/flow** : zones d'exclusion autour d'images, contours de wrap personnalisés
- **Hyphenation** : points de coupure, algorithme utilisé, exceptions manuelles

---

### **11. PARAMÈTRES DE RENDU & IMPRESSION**
- **Imposition** : recto/verso, livret, pages en miroir
- **Échelle d'impression** : fit to page, shrink oversized, actual size
- **Qualité de sortie** : screen, printer, high fidelity
- **Profil couleur** : destination (sRGB, Coated FOGRA39, etc.)
- **Overprint** : réglages de surimpression pour l'impression professionnelle
- **Trapping** : paramètres d'accroche
- **Output intent** (PDF/X, PDF/A)

---

### **12. MÉTRIQUES DE QUALITÉ & CONFIANCE**
- **Source d'extraction** (pdf_native, ocr_primary, ocr_secondary, vision)
- **Score de confiance** par token, ligne, bloc (0-1)
- **Incertitudes** : tokens ambigus, reconnaissance douteuse
- **Overlaps détectés** : zones où sources se chevauchent
- **Métriques OCR** : CER (Character Error Rate), WER (Word Error Rate)
- **Anomalies typographiques** : caractères non reconnus, glyphs manquants
- **Rapport de conformité** : éléments non supportés, perte d'information

---

### **13. INFORMATIONS TECHNIQUES POUR RECONSTRUCTION**
- **Font metrics** : ascent, descent, cap height, x-height, line gap (units per em)
- **Kerning tables** disponibles pour chaque font
- **Ligature sets** supportés
- **Font substitution map** : quelle font utiliser si manquante
- **Width estimation** : fonction de calcul de largeur de texte pour chaque font
- **Fallback strategy** : chaîne de fonts de secours
- **Rendering engine** : spécifique (Harfbuzz, CoreText, DirectWrite)
- **Text encoding** : Unicode normalization form (NFC, NFD)
- **Glyph positioning** : mode (monospace, proportional, kerning, ligatures)

---

### **14. ARCHITECTURE & HÉRITAGE DE SYLE**
- **Style hierarchique** : style de paragraphe > style de caractère > override direct
- **Style sheets** : définitions nommées, basées sur autre style
- **Conditionnal formatting** : règles (si valeur > X alors gras)
- **Inherited properties** : pour chaque token, trace de qui définit la propriété
- **Style overrides explicites** : différences par rapport au style parent

---

### **15. ACCESSIBILITÉ & SÉMANTIQUE AVANCÉE**
- **ARIA labels** / structure de tags PDF/UA
- **Langue par bloc** (xml:lang)
- **Alt text** complet pour images, formules
- **Structure de titre** hiérarchique (H1-H6 correctement imbriqués)
- **Tableaux de données** vs tableaux de mise en forme
- **Listes imbriquées** : niveau, type, style
- **Liens descriptifs** : texte du lien vs URL
- **Contraste couleur** : ratio calculé

---

### **16. ÉLÉMENTS DE SÉCURITÉ & PROTECTION**
- **Permissions** : copier, imprimer, modifier, extraire contenu
- **Watermarking** : texte invisible, tracking dots
- **Signatures** : visual appearance, certificat
- **Contenu masqué** : texte sous image, calque invisible
- **Javascript** : actions associées à des objets

---

### **17. DONNÉES BINAIRES & METADATA TECHNIQUES**
- **Checksum** du fichier source
- **Embedded files** : pièces jointes, formats originaux
- **Thumbnails** : images de prévisualisation
- **Preview stream** (PowerPoint)
- **Compression** : algorithmes appliqués (JPEG, ZIP, LZW, etc.)
- **Object streams** : dans PDF, délimitation des objets

---

### **18. DÉTAILS DE RASTERISATION & RENDU SUBPIXEL**
- **Font hinting** : instructions TrueType pour ajuster les contours aux pixels (grid-fitting)
- **Subpixel positioning** : position des glyphs à des fractions de pixel (0.25px, 0.5px)
- **Anti-aliasing** : type (grayscale, subpixel RGB/BGR, ClearType, FreeType)
- **Stem snapping** : alignement des traits verticaux/horizontaux sur la grille pixel
- **Gamma correction** : courbe de correction appliquée au rendu texte
- **Rasterization engine version** (differences entre Adobe, Foxit, macOS, Windows)
- **Hinting zones** : zones d'influence des instructions (blue zones, family blues)
- **Blue scale adjustment** : compensation de la taille pour petites fonts
- **Delta hints** : micro-ajustements pixel par pixel à certaines tailles

---

### **19. COMPORTEMENTS CONDITIONNELS & CONTENU DYNAMIQUE**
- **Fields & formulas** : champs calculés (date, numéro de page, section)
- **Conditional text** : blocs affichés selon règles (si...alors)
- **Data links** : liens vers sources de données externes (Excel, XML)
- **Bibliographic fields** : citations qui se mettent à jour automatiquement
- **Cross-references with update** : références qui changent si cible change
- **Macro triggers** : événements qui modifient le contenu (ouverture, impression)
- **Layers visibility** : calques PDF visibles/imprimables selon contexte
- **Optional content groups** (OCG) : contenu conditionnel par usage (screen/print)

---

### **20. MÉTADONNÉES DE PROCESSUS & PROVENANCE**
- **Creation/Modification/Print dates** exactes avec fuseau horaire
- **Software version chain** : liste des logiciels ayant touché le fichier
- **User metadata** : nom créateur, entreprise, machine
- **Revision history** : chronologie des sauvegardes, ID de version
- **Undo stack** : actions annulables encore en mémoire (dans formats natifs)
- **Template source** : fichier modèle d'origine
- **Digital fingerprint** : empreinte créée par l'outil (ex: Adobe UUID)
- **Color separationIntent** : pour impression professionnelle (trapping, overprint)

---

### **21. STRUCTURE DE DOCUMENT & NAVIGATION**
- **Outline hierarchy** : bookmarks avec niveaux, destinations, actions
- **Named destinations** : positions nommées pour navigation
- **Page labels** : numérotation personnalisée (i, ii, 1, A-1)
- **Article threads** : flux de lecture séquentiels sur plusieurs pages (PDF)
- **Destination syntax** : [page /FitH 842] vs [page /XYZ 100 500 1.5]
- **Link actions** : GoTo, Launch, URI, JavaScript, Named action
- **File attachments** : emplacement, icône, nom
- **Collection dictionaries** : pour portfolios PDF
- **Structure tree** (PDF/UA) : structure parent-enfant (P, H1, Span, etc.)

---

### **22. TRANSPARENCE & EFFETS DE COMPOSITION**
- **Blend modes** : Multiply, Screen, Overlay, Darken, etc. (formules Porter-Duff)
- **Soft masks** : transparence variable selon canal alpha
- **Transparency groups** : isolation, knockout
- **Page group** : transparence au niveau page
- **Backdrop color** : couleur de fond pour calcul de transparence
- **Opacity** : valeurs à 4 décimales (0.7534)
- **Spot color density** : pour impressions séparées
- **Halftone screens** : fréquence, angle, forme des points

---

### **23. ANNOTATIONS & INTERACTIONS AVANCÉES**
- **Ink annotations** : polygones de dessin, pression stylet, épaisseur variable
- **Rich media** : vidéos (codec, bitrate), audio (sample rate), contrôles
- **3D annotations** : modèles U3D/PRC, vues, animations, scripts
- **Sound objects** : format (MP3, WAV), activation
- **Stamp annotations** : apparence personnalisée, bitmap, vector
- **File attachment annotations** : icône, emplacement, nom
- **Redaction** : zones censurées avec motif de remplissage
- **Caret annotations** : position d'insertion suggérée
- **Popup configurations** : position, ouvert/fermé par défaut

---

### **24. ACCESSIBILITÉ PROFONDE (PDF/UA, Section 508)**
- **Tag structure complete tree** : Document, Part, Sect, Div, P, H, Span, etc.
- **Alternate text for every line art** (pas seulement images bitmap)
- **ActualText** pour remplacer visuel par texte (ex: logo comme texte)
- **Expansion dictionary** : abréviations à épeler
- **Language tags** au niveau ligne/token, pas seulement bloc
- **List numbering** : auto vs manual vs none (pour lecteurs d'écran)
- **Table headers scope** : row, column, both (id, headers)
- **Artifacts** : marquage des éléments décoratifs (header lines, background shapes)
- **Bounding boxes des tags** vsbounding boxes visuels (peuvent diverger)

---

### **25. SÉCURITÉ & CERTIFICATION**
- **Digital certificates** : chaine complète, émetteur, validity, algorithm
- **Signature appearance** : layer n0, n2, raison, lieu, date
- **Lock dictionary** : quels champs sont verrouillés après signature
- **Usage rights** : Reader Extensions (remplissage formulaire, sauvegarde)
- **Encryption dictionary** : filter, V, R, length, crypt filter
- **DocMDP/UR transforms** : modifications autorisées
- **Hidden traps** : graphics state qui empêche copie (même si permission "copier" est true)

---

### **26. SUBTILITÉS DE FORMATS SPÉCIFIQUES**

**PDF spécifiques:**
- **Object streams compression** (PDF 1.5+)
- **Cross-reference stream** vs table
- **Linearized PDF** (web optimization) avec hint tables
- **Incrementally updated PDF** : quelles parties sont modifiées
- **XMP metadata stream** (format XML complet)
- **Embedded file streams** avec mime type
- **Collection schema** pour portfolios
- **Output intent ICC profile** (PDF/A, PDF/X)

**DOCX spécifiques:**
- **Office Open XML parts** : relations, content types, overrides
- **Theme component** : couleurs thème, fonts thème, effects thème
- **Styles gallery** : style visible dans l'UI vs style caché
- **Table styles** : conditional formatting (first row, last column, etc.)
- **Content Controls** : plain text, rich text, dropdown, date picker
- **Custom XML parts** : data islands
- **SmartArt data** : XML de structure avec layout algorithm
- **MathOMML** : equations Office MathML
- **Track changes** ( revisions ) : qui, quand, quoi, comment
- **Comments** : threads, @mentions
- **Document protection** : forms, tracked changes, read-only
- **Compatibility mode** : version d'Office simulée

**PPTX spécifiques:**
- **Slide masters** : layouts, placeholders, inheritance
- **Slide layouts** : type de contenu autorisé
- **Placeholder types** : title, body, picture, chart, table, etc.
- **Slide XML timing** : animations, triggers, durées
- **Slide transitions** : type, direction, duration, sound
- **Embedded charts** : data behind chart (Excel), types de séries
- **Slide notes** : texte speaker notes
- **Handout master** : pour impression
- **Presentation properties** : loop mode, kiosk mode, pen color

---

### **27. MÉTRIQUES & DIAGNOSTICS DE RECONSTRUCTION**
- **Token provenance chain** : natif → fallback → OCR → manual correction
- **Confidence decay** : quand une opération réduit la confiance
- **Reconstruction error per pixel** : MAE (Mean Absolute Error) sur rendu
- **Layout drift** : distance moyenne entre bbox original et reconstruit
- **Font substitution distance** : métrique de similarité visuelle (SSIM between glyph images)
- **Reading order inversion count** : combien de swaps par rapport à l'original
- **Text overflow instances** : où et combien de caractères débordent
- **Font metrics mismatch** : quand ascent/descent diffèrent de l'original
- **Color difference** : Delta E 2000 entre couleurs originales et rendues
- **Transparency flattening artifacts** : zones où transparence multi-couches est aplatie

---

### **28. ÉLÉMENTS "ZÉRO-LARGEUR" & INVISIBLES**
- **Zero-width spaces** (U+200B) pour césure contrôle
- **Zero-width joiners** (U+200D) pour ligatures complexes
- **Left-to-right mark** / **Right-to-left mark** (U+200E, U+200F)
- **Object replacement character** (U+FFFC) pour objets inline
- **End marks** : fin de ligne, fin de paragraphe (pas juste newline)
- **Section breaks** : continu, next page, odd, even
- **Column breaks** : forçage manuel
- **Page breaks** : avant/après paragraphe
- **Invisible graphics** : lines with 0 width, white on white
- **Hidden text** : attribut "hidden" (invisible mais dans le flux)
- **Placeholder text** : "Click here to enter text"

---

### **29. DÉTAILS DE GRILLE & GUIDES**
- **Guides visibles** : position, orientation, couleur
- **Grid settings** : spacing, start point, snap behavior
- **Baseline grid** : increment, offset, couleur guide
- **Snap-to-zone** : zones magnétiques pour objets
- **Smart guides** : distances relatives entre objets

---

### **30. GESTION DES EXCEPTIONS MANUELLES**
- **Manual line breaks** (Shift+Enter) vs paragraph break
- **Manual page breaks** : avec style appliqué
- **Manual hyphenation** : césure avec trait d'union dur
- **No-break spaces** : empêcher césure mot
- **Keep with next** : forcé manuellement
- **Widow/orphan control** : exceptions manuelles
- **Character spacing override** : tracking manuel sur plage
- **Kerning pairs manual** : activation/désactivation spécifique
- **Ligature manual** : forcer ligature ou l'empêcher

---

### **31. SUBTILITÉS DE MISE EN FORME CONDITIONNELLE**
- **Highlight colors** : jaune, vert, rose (pas de fond de paragraphe)
- **Text borders** : bordure autour de texte (pas de paragraphe)
- **Text effects** : gradient fill sur texte, texture fill, pattern fill
- **Shadow options** : distance, blur, angle, couleur, transparence
- **Reflection** : transparence, taille, offset
- **Glow** : rayon, transparence, couleur
- **3D format** : bevel, depth, contour, surface material
- **3D rotation** : X, Y, Z

---

### **32. GESTION DES CONFLITS & AMBIGUÏTÉS**
- **Overlapping text frames** : z-index, bring to front/back
- **Clipped content** : texte qui dépasse de son cadre (overflow hidden)
- **Rotated text frames** : angle avec rewrap dynamique
- **Skewed text** : transformation affine (shear)
- **Perspective warp** : texte en perspective (rare mais existe)
- **Text on path** : suivre un cercle, courbe Bézier
- **WordArt** : transformations non-linéaires

---

### **33. FLUX DE TRAVAIL & COLLABORATION**
- **Comments threads** : réponses, résolution, auteurs
- **Track changes authors** : attributions par utilisateur
- **Versioning info** : major/minor version, baseline
- **Co-authoring locks** : zones verrouillées par d'autres utilisateurs
- **Document template link** : lien vers modèle .dotx
- **Quick parts** : morceaux de contenu réutilisables
- **Building blocks** : galeries de composants

---

### **34. DONNÉES DE DIAGNOSTIC & DEBUG**
- **Render time per object** : logs de performance
- **Font loading errors** : quelle font manquante et où
- **Glyph fallback trace** : quel glyph a remplacé un manquant
- **OCR debug zones** : zones où OCR a été utilisé vs natif
- **Confidence heatmap** : visualisation spatial des confiances
- **Reconstruction diff layers** : couches où original ≠ reconstruit

---

Voici la liste **généralisée et exhaustive** des éléments à extraire pour une reconstruction WYSIWYG fidèle, à partir des subtilités atomiques :

---

### **35. TYPOGRAPHIE ATOMIQUE & GLYPH-LEVEL**
- **Optical sizing** : variations de dessin selon taille (caption, text, display)
- **Font hinting bytecode** : instructions TrueType/grid-fitting spécifiques par taille de pixel
- **Blue zones & family blues** : zones de référence pour alignement des arrondis sur grille
- **Stem weights** : épaisseur exacte traits verticaux/horizontaux (valeurs décimales px)
- **Character variants** (`cv01`-`cv99`) : glyphes alternatifs par caractère
- **Stylistic alternates** (`salt`) : formes stylistiques exceptionnelles
- **Swash** : ornements initiaux/finaux (Q avec queue decorative)
- **Contextual alternates** (`calt`) : glyphe adaptatif selon voisins (ex: f avant i)
- **Discretionary ligatures** : ligatures optionnelles (st, ct, sp)
- **Number forms** : tabular vs proportional, lining vs oldstyle
- **Fraction forms** : numérateurs/dénominateurs complets vs simples
- **Superscript/Superior** vs **Subscript/Inferior** : échelles et positions normalisées
- **Small caps synthesis** : vraies petites capitales (glyphs dédiés) vs simulées (scaled down)
- **All caps simulation** : attribut de transformation vs refonte glyphe
- **Word spacing adjustment** : espacement inter-mots variable (ex: justification large)
- **Individual glyph rotation** : caractère pivoté isolément (ex: date verticale en marge)
- **Skew/shear** : oblique artificiel vs vrai italic
- **Glyph embossing/engraving** : simulation relief (ombre portée interne)
- **Ink traps** : zones techniques de compensation dans fontes haute résolution
- **OpenType feature tags** : `liga`, `dlig`, `smcp`, `onum`, `pnum`, `tnum`, `frac`, `zero`, etc.

---

### **36. ESPACES & SÉPARATEURS INVISIBLES MAIS CRITIQUES**
- **Non-breaking space** (U+00A0) vs **narrow no-break space** (U+202F)
- **Figure space** (U+2007) : largeur chiffre
- **Punctuation space** (U+2008) : largeur point
- **Thin space** (U+2009) et **hair space** (U+200A)
- **Zero-width space** (U+200B) : contrôle césure
- **Zero-width non-joiner/joiner** (U+200C/U+200D) : contrôle ligatures complexes
- **Directional marks** : LRM (U+200E), RLM (U+200F), LRE, RLE, LRO, RLO (U+202A-U+202E)
- **Bidi isolates** : LRI, RLI, FSI, PDI (U+2066-U+2069)
- **Soft hyphen** (U+00AD) : césure conditionnelle
- **Non-breaking hyphen** (U+2011) et **word joiner** (U+2060)
- **Object replacement character** (U+FFFC) : placeholder objet inline
- **Line separator** (U+2028) vs **paragraph separator** (U+2029)
- **Manual line break** (Shift+Enter) vs **paragraph break** (Enter)
- **Column break** et **page break** forcés manuellement
- **Section break types** : next page, odd page, even page, continuous (avec propriétés)
- **Frame break** : saut entre zones de texte flottantes
- **Style separator** : char caché qui sépare styles dans même paragraphe

---

### **37. PONCTUATION & SYMBOLES PERSONNALISÉS**
- **Typographic quotes** : guillemets français « », allemands „“, anglais “ ”, japonais 「 」
- **Hanging punctuation** : ponctuation qui dépasse marge de bloc
- **Optical margin alignment** : dépassement caractères asymétriques (A, W, " , ")
- **Prime symbols** : `′` pour pied/pouces vs apostrophe `'`
- **Mathematical symbols** : `−` (minus sign) vs `-` (hyphen), multiplication `×` vs `x`
- **Dashes** : figure dash (‑), en dash (–), em dash (—), horizontal bar (―)
- **Ellipsis** : `…` (single glyph) vs `...` (three dots)
- **Symbols vs letter variants** : `№` vs `No`, `™` vs `TM`
- **Section/paragraph marks** : §, ¶, †, ‡ avec styles
- **Reference marks** : asterism (⁂), manicule (☞)
- **Footnote markers** : style numérique, symbolique (*, †, ‡), position supérieure ou normale
- **Cross-reference marks** : "see page X" avec champs dynamiques
- **Commercial AT** : `@` dans email vs mention sociale
- **Copyright/Trademark** : ©, ®, ™, ℠, avec espacements ajustés
- **Currency symbols** : position avant/après montant, espacement

---

### **38. STRUCTURES DE LISTES & ÉNUMÉRATIONS**
- **List style type** : disc, circle, square, decimal, lower-roman, upper-alpha, custom glyph
- **Custom bullet character** : glyph ou image personnalisée (logo, icône)
- **Bullet font** : fonte dédiée pour puces (Wingdings, Symbol, custom)
- **List level** : indentation, style par niveau (1.1, 1.1.1, a), i))
- **List alignment** : puce alignée gauche/droite/justifiée
- **Start value** : numéro de départ (ex: commencer à 5)
- **Restart numbering** : renumérotation forcée après changement de section
- **Multilevel lists** : mapping niveaux à styles hiérarchiques (legal, outline)
- **List continuation** : reprise liste interrompue après bloc inséré
- **List-specific spacing** : espace puce-texte, entre items, avant/après liste
- **Hanging indent** : position première ligne vs suite
- **Numbering alignment** : aligné à droite, gauche, ou sur ponctuation
- **Text wrapping** : texte qui continue sous puce (second line align)

---

### **39. TABLEAUX - SUBTILITÉS SUPÉRIEURES**
- **Table style** : style nommé avec bandes conditionnelles (first row, last column)
- **Conditional formatting** : appliquer style selon valeur (ex: >100 = rouge)
- **Cell margins** : padding individuel top/bottom/left/right
- **Cell text direction** : 0°, 90°, 270°, orientation verticale
- **Diagonal cell borders** : ligne diagonale (single, double, crossing)
- **Fit text to cell** : shrink font size, shrink on overflow
- **Text wrapping** dans cellule (clip vs overflow)
- **Recalculate on open** : liens Excel qui se rafraîchissent
- **Nested tables** : profondeur illimitée (table dans cellule)
- **Floating tables** : positionnés comme images flottantes (wrap text)
- **Text-to-table conversion** : historique de conversion tab text → table
- **Allow row break across pages** : on/off par ligne
- **Repeat header row** : marquage sémantique
- **Table caption** : légende intégrée ou distincte
- **Summary & alt text** : description accessible
- **Column width rules** : fixed width, auto-fit, distribute evenly
- **Row height rules** : at least, exactly, auto
- **Table alignment** : left/right/center, indentation
- **Table-to-text boundaries** : zones de texte qui deviennent tableau

---

### **40. IMAGES & OBJETS GRAPHIQUES - SUBTILITÉS TECHNIQUES**
- **EXIF data** : camera, date, GPS, focal length (même si embarqué)
- **XMP metadata** : keywords, rights, creator, description
- **Color profile** : sRGB, AdobeRGB, ProPhoto, avec intents (perceptual, relative colormetric)
- **Compression parameters** : JPEG quality factor (1-100), chroma subsampling (4:4:4, 4:2:0)
- **Alpha channel** : pré-multiplied vs straight alpha
- **Separate mask** : image masque de transparence externe
- **Color key transparency** : transparence par couleur référence
- **Resolution override** : DPI spécifié dans document vs DPI natif
- **Logical vs physical size** : pixels natifs vs dimension affichée
- **Interpolation method** : bicubic, bilinear, nearest neighbor, Lanczos
- **Image recoloring** : noir & blanc, sépia, washout, duotone, tint
- **Picture styles** : frame, ombre portée, lueur, relief, biseau, réflexion
- **Crop to shape** : rognage selon forme vectorielle (étoile, cercle, flèche)
- **Artistic effects** : filtre peinture, mouvement, céramique (convolution)
- **Background removal** : masque automatique basé sur détection sujet
- **SmartArt & Diagrams** : data XML structure, layout algorithm
- **3D models** : U3D, PRC, vues, animations, scripts
- **Linked images** : path externe relatif/absolu, état broken
- **Compression per instance** : même image réutilisée avec settings différents
- **Image tiling** : grande image découpée en tiles (mémoire)

---

### **41. CONTENU CACHÉ & MÉTADATA TECHNIQUE OBSESSIVE**
- **White text on white background** : invisible mais sélectionnable/copiable
- **Text behind image** : layer OCR derrière image scannée (PDF sandwich)
- **Invisible layers** : calques désactivés dans UI mais présents
- **Optional content** : visible en écran mais pas à l'impression (ou inverse)
- **Private app data** : clés constructeur (Adobe Private, Halfbaked)
- **PieceInfo** : fragments d'édition InDesign (objets non finis)
- **Job Ticket** : instructions JDF d'impression (preset PDF)
- **Watermark text** : transparent, roté, derrière contenu (anti-contrefaçon)
- **Steganographic patterns** : points jaunes imprimante, LSB images
- **Copy detection pattern** : motif moiré pour vérification authenticité
- **Digital watermark** : imperceptible (DWT dans images)
- **Orphaned objects** : objets non référencés dans page tree (déchets)
- **Unused resources** : fonts/images chargées mais jamais affichées
- **Compression artifacts** : artefacts JPEG persistence après multiples saves
- **Previous version thumbnails** : miniatures versions antérieures
- **XML comments** : `<!-- -->` dans OOXML
- **Hidden sheets/charts** : objets marqués "very hidden" (Excel)
- **PUA characters** : glyphes custom mappés Private Use Area (U+E000-U+F8FF)

---

### **42. COMPORTEMENTS DYNAMIQUES & CONDITIONNELS**
- **Fields & formulas** : `{ DATE }`, `{ PAGE }`, calculs automatiques
- **Conditional text** : blocs affichés selon expression logique (si...alors)
- **Data links** : liens vers sources externes (Excel, XML, JSON, base de données)
- **Bibliographic fields** : citations qui se mettent à jour dynamiquement
- **Cross-references** : "voir page X" qui change avec cible
- **Macro triggers** : événements (open, close, print) qui modifient contenu
- **Layers visibility** : calques visibles selon contexte (zoom, media)
- **Optional content groups** (OCG) : usage View/Print/Export
- **Actions JS** : actions JavaScript sur objets (calcul field, alert)
- **Form field logic** : mandatory, readonly, calculate order
- **Dropdown values** : list items avec display vs export value
- **Content controls** : text riche, dropdown, date picker, checkbox

---

### **43. SÉCURITÉ, CERTIFICATION & CONFORMITÉ**
- **Digital certificates** : chaîne complète (root, intermediate), validity, algorithm (RSA-2048)
- **Signature appearance** : layer n0, n2, raison, lieu, date, contact
- **Lock dictionary** : champs verrouillés après signature
- **Usage rights** : Reader Extensions (remplissage, sauvegarde, commentaire)
- **DocMDP/UR transforms** : modifications autorisées (none, form fill, annotation)
- **Encryption dictionary** : filter (Standard, AESV3), V/R, length, crypt filter
- **Password protection** : user/owner password, permissions (copy, print, edit)
- **Hidden traps** : graphics state qui empêche extraction malgré permissions
- **Redaction** : zones censurées avec motif de remplissage (pattern overlay)
- **Certification signature** : MDP (modification detection & prevention)
- **Timestamp authority** : TSA RFC 3161, estampillage temps
- **LTV (Long Term Validation)** : informations OCSP/CRL pour validation future
- **DRM policies** : expiration date, max opens, printing count

---

### **44. ACCESSIBILITÉ PROFONDE (PDF/UA, WCAG)**
- **Complete tag tree** : Document, Part, Sect, Div, P, H1-H6, Span, Art, Sect
- **Alternate text line art** : description chemins vectoriels (logo, diagram)
- **ActualText** : remplacement texte pour glyphe décoratif (ex: logo → "CompanyName")
- **Expansion dictionary** : abréviations à épeler (abbrev)
- **Language tags per token** : `lang="fr"` au niveau span, pas juste bloc
- **List numbering semantics** : auto vs manual vs none (pour lecteur écran)
- **Table headers scope** : id/headers, scope=row/col/both (pour navigation cellule)
- **Artifacts** : marquage éléments décoratifs (lignes header, background shapes)
- **Tag bounding boxes** vs visual bounding boxes (peuvent diverger)
- **Reading order tree** : ordre exact de lecture défini par structure, pas par géométrie
- **Title/Description** : pour tout objet non texte (images, formulaires)
- **Bookmarks tagged** : structure outline liée aux titres (non juste positional)

---

### **45. SÉMANTIQUE & COMPRÉHENSION DU CONTENU**
- **Named entities** : noms personnes, organisations, lieux (NER)
- **Acronyms & abbreviations** : détection, expansion (ISBN → International Standard Book Number)
- **Measurements** : valeurs + unités (20 Baldwin Road, 15 percent, PO Box 761)
- **Contact info** : téléphones, emails, URLs avec format normalisé
- **Dates & times** : ©2020, 2020-01-20, formats localisés
- **Currency** : $19.99, €15, position symbole, espacement
- **References** : citations bibliographiques (APA, IEEE), numéros de brevets
- **Legal identifiers** : ISBN, ISSN, DOI, numéros de lois
- **Document structure semantics** : frontmatter, body, backmatter, colophon
- **Semantic roles** : author, publisher, edition, imprint, distributor
- **Citation intent** : citation, reference, quotation, mention
- **Document genre** : book, report, contract, manual

---

### **46. INTERNALS DE FORMATS & DATA STREAMS**
- **Embedded font files** : TTF, OTF complets ou subsets (avec extraction possible)
- **CIDFonts composites** : pour CJK (Identity-H, Identity-V encodings)
- **CMap files** : mapping code-to-glyph pour fonts composites
- **ColorSpace dictionaries** : /DeviceGray, /DeviceRGB, /CalRGB, /Lab, /ICCBased avec paramètres
- **Shading dictionaries** : fonctions de gradient (axial, radial, freeform)
- **Pattern dictionaries** : type 1 (tiling) avec spacing, type 2 (shading)
- **Halftone dictionaries** : fréquence (lpi), angle (degrés), spot function
- **ExtGState** : état graphics (transparence, blend, font, overprint)
- **XObject Forms** : objets réutilisables (logo, entête) avec matrice
- **Optional content properties** : usage (View, Print, Export, Zoom), visibility expression
- **Object streams** : compression objets multiples (PDF 1.5+)
- **Cross-reference streams** : vs tables traditionnelles
- **Linearized PDF** : hint tables pour web optimization
- **Incremental updates** : deltas contenant anciennes révisions

---

### **47. MÉTRIQUES DE QUALITÉ & DIAGNOSTIC DE RECONSTRUCTION**
- **Token provenance chain** : natif → fallback → OCR → correction manuelle
- **Confidence decay log** : chaque opération qui réduit confiance
- **Per-pixel reconstruction error** : MAE, MSE, PSNR entre original/reconstruit
- **Layout drift** : distance moyenne entre bbox original et reconstruit (en pt)
- **Font substitution delta** : différence largeur texte (en % ou px)
- **Reading order inversion count** : nombre de swaps illogiques
- **Text overflow instances** : localisation et sévérité débordement
- **Color Delta E 2000 map** : heatmap de différence couleur perceptuelle
- **OCR confusion matrix** : quels caractères confondus (0-O, l-1)
- **Overlap density distribution** : histogramme IoU tokens
- **Style inheritance depth** : niveaux de style parent
- **SSIM per block** : similarity structurelle zone par zone
- **Glyph dropout detection** : glyphes manquants non substitués (tofu)
- **Anti-aliasing fingerprint** : pattern gris identifiant moteur rendu original

---

### **48. FONCTIONNALITÉS PRODUITS SPÉCIFIQUES**
- **Smart tags** : reconnaissance type données (adresse, date, financial)
- **Content controls** : dropdown, date picker, plain text, rich text, checkbox
- **Form field options** : list items, export values, default, calc order
- **Quick parts** : morceaux réutilisables (building blocks)
- **Document properties** : Company, Category, Keywords, Hyperlink base
- **Custom XML data islands** : data persistance dans OOXML
- **Themes** : couleurs thème, fonts thème, effects thème (Office)
- **Slide masters** : layouts, placeholders, inheritance (PowerPoint)
- **Content controls databinding** : liaison XML
- **Document automation** : champs { MERGEFIELD }, { IF }, { DOCPROPERTY }

---

### **49. MARQUES & PROPRIÉTÉ INTELLECTUELLE**
- **Copyright notice** : symbole exact ©, wording, date, holder
- **Trademark attribution** : ™, ®, ℠, avec footnotes explicatives
- **Patent references** : numéros complets avec juridiction
- **License notices** : texte complet MIT, GPL, CC BY-SA
- **Third-party attribution** : logos contributors avec taille contrainte
- **Publisher imprint** : logo avec couleurs Pantone/CMJN spécifiques
- **Edition statement** : "First edition, first printing" avec date impression
- **All rights reserved** wording variant (international)
- **Legal jurisdiction** : clauses choix de loi (NY, USA; England & Wales)
- **Export control** : EAR, ITAR notices
- **Accessibility compliance** : WCAG, Section 508, PDF/UA statements
- **Environmental statement** : recycled paper, FSC certification

---

### **50. HISTOIRE, MÉTADONNÉES & CONTENU FANTÔME**
- **Revision history** : chronologie saves, ID version, authors
- **Undo stack** : actions annulables (dans formats natifs)
- **Deleted content persistence** : dans incremental update streams
- **Orphaned objects** : objets non référencés (déchets de création)
- **Previous version thumbnails** : previews versions antérieures
- **Template link** : lien fichier modèle .dotx, .potx (relatif/absolu)
- **Embedded preflight** : profils de conformité embarqués
- **Private editor metadata** : PieceInfo (InDesign), Halfbaked (Acrobat)
- **Hidden until used** : styles/objets non affichés tant qu'utilisés
- **Document skeleton** : structure vide avant remplissage
- **Conversion trail** : artefacts Word→PDF, InDesign→PDF (faux bolds multiples)

---

### **51. PARAMÈTRES D'AFFICHAGE, RENDU & VIEWER**
- **Screen vs Print media** : disposition différente selon media
- **Night mode adaptation** : inversion couleurs détectée
- **High contrast accessibility** : remplacement couleurs forcé
- **Zoom level persistence** : niveau zoom sauvegardé dans doc
- **View settings** : single page, continuous, facing pages, show cover page
- **Initial view** : page ouverture, magnification, layout, UI elements hidden
- **Page transitions** : dissolve, wipe, fly (mode présentation)
- **Auto-advance timing** : temps par page en mode slide show
- **Default printer** : nom printer, settings embarqués
- **Print scaling** : fit to page, shrink oversized, tile large pages

---

### **52. SUBTILITÉS MANUELLES & OVERRIDES IMPOSSIBLES À INFERER**
- **Manually adjusted hyphens** : césure humaine qui viole algorithme
- **Optical margin kerning** : "A" ou "W" qui dépasse marge de bloc
- **Individual glyph nudging** : déplacement glyphe de quelques milliemes de point
- **Fake bold/italic simulation** : outline vs font true bold/italic
- **Simulated superscript/subscript** : font-size réduit + baseline shift manuel
- **Text converted to outlines** : vectorisé (perdu caractères)
- **Rasterized text** : texte rendu en image (OCR nécessaire)
- **Manual keep with next** : forcé localement violant style global
- **Widow/orphan manual exceptions** : violations règles globales
- **Character spacing override** : tracking manuel sur plage spécifique
- **Kerning pairs manual** : activation/désactivation glyph par glyph

---

### **53. MISE EN FORME CONDITIONNELLE & SMART**
- **IF fields logic** : `{ IF { MERGEFIELD X } = "val" "show" "hide" }`
- **Conditional text blocks** : affichage selon expression variable
- **Style separators** : char caché qui sépare styles dans même paragraphe
- **Hidden until used** : styles/objets non listés tant qu'appliqués
- **Linked styles** : style paragraphe qui agit comme style caractère si sélection partielle
- **Contextual spacing** : espace différent selon style voisin
- **Mirror indents** : indent gauche devient droit en recto verso
- **Gutter positioning** : marge binding qui change côté selon parité page
- **Footnote layout options** : collector, placement sous texte vs fin section
- **Endnote story** : séquence séparée du corps principal
- **SmartArt logic** : layout algorithm fonction nombre items, level depth

---

### **54. EXPORT, CONFORMITÉ & INTEROPÉRABILITÉ**
- **PDF/A compliance** : a, b, u, niveau (A, B, U), avec justification
- **PDF/X output intent** : profil ICC sortie impression (Coated FOGRA39, GRACoL)
- **PDF/UA identifier** : claim conformité WCAG 2.0/2.1, niveau (A, AA, AAA)
- **Tagged PDF structure** : tree complet avec rôles personnalisés (P, Div, Sect)
- **XMP metadata** : Dublin Core, PRISM, IPTC, EXIF, camera raw
- **Schema mapping** : propriétés personnalisées mapping à standards
- **Embedded preflight profile** : Enfocus PitStop, Adobe preflight
- **Layer metadata** : usage (View, Export, Print, Zoom), visibility expression
- **PDF version** : 1.4, 1.5, 1.7, 2.0 (différence features)
- **Linearization** : PDF optimisé web avec hint tables
- **Object streams** : compression objets multiples (PDF 1.5+)
- **Cross-reference stream** : format moderne vs table classique

---

### **55. FORENSICS & FINGERPRINTING TECHNIQUE**
- **Creation tool fingerprint** : pattern unique InDesign, Word, LaTeX, Quark
- **Conversion artifacts** : artefacts Word→PDF (faux bolds multiples, faux italic)
- **Generator timestamp skew** : horodatage batch processing
- **Font subset naming pattern** : `Arial+BOLD+12345` (indicateur subset)
- **Object identifier sequence** : séquence qui indique librairie de génération
- **Compression library signature** : pattern deflate qui identifie zlib vs 7zip
- **Byte alignment** : padding qui indique architecture 32-bit vs 64-bit
- **Memory layout** : ordre objets qui révèle processus création
- **Rasterization engine fingerprint** : pattern anti-aliasing spécifique Adobe vs Foxit vs macOS
- **Incremental update trails** : croissance fichier qui révèle éditions
- **Unused objects** : patterns types (Font, ColorSpace) non référencés

---

### **56. ÉMERGENTS - IA & DEEPFAKE**
- **AI-generated content marker** : watermark invisible indiquant provenance IA (SynthID)
- **Model fingerprint** : caractéristiques spécifiques GPT, Claude, Gemini
- **Deepfake detection features** : patterns artefacts GAN (spectre fréquences)
- **Synthetic text signature** : distribution n-grams anormale, perplexité
- **AI style transfer** : texte imitant manuscrit avec variations aléatoires
- **Generated image metadata** : Stable Diffusion prompt, Midjourney parameters, DALL-E seed
- **Hallucination markers** : passages IA avec références fabriquées
- **Plagiarism detection** : fingerprint texte pour comparaison bases
- **Adversarial examples** : modifications invisibles perturbant OCR/classification

---

### **57. DOCUMENTS PATRIMOINE & ANCIENS**
- **Manuscript bleed-through** : encre traversant papier (scans anciens)
- **Foxing** : taches brunes oxydation papier
- **Yellowing** : couleur fond changée par âge
- **Ink corrosion** : dégradation papier autour encre métallique
- **Binding marks** : zones usées reliure (gauche/droite, haut/bas)
- **Marginalia** : annotations manuscrites marges, couleur encre différente
- **Stamps** : tampons "Bibliothèque", "Confidential" avec encre réelle
- **Seals** : empreintes cire relief 3D
- **Aging simulation** : documents modernes imitant anciens (faux patine)
- **Degraded fonts** : glyphes incomplets, érosion physiques
- **Historical orthography** : s long (ſ), ligatures historiques (ct, st)
- **Period-specific glyphs** : formes anciennes caractères (ex: danse)

---

### **58. SÉCURITÉ PHYSIQUE & ÉLÉMENTS ADVANCED**
- **Polarized ink** : visible sous polarisation lumière
- **UV ink** : texte invisible UV (sécurité documents)
- **IR ink** : encre visible infrarouge (IR-sensitive)
- **Thermochromic ink** : couleur change avec température
- **Holographic elements** : images 3D angle-dépendantes
- **Microtext** : texte < 0.5pt anti-contrefaçon
- **Guilloché patterns** : motifs complexes billets banque
- **Void pantographs** : "VOID" apparaît en photocopie
- **Magnetic ink** : MICR (codes bancaires)
- **Optical variable ink** : couleur change selon angle
- **Watermark paper** : filigrane physique papier (transmit light)

---

### **59. SUBTILITÉS DE CONVERSION & NORMALISATION**
- **PDF version conversion** : 1.4 → 2.0 feature loss (transparence flattening)
- **Color space conversion** : RGB → CMYK → RGB lossy (out-of-gamut)
- **Font substitution on open** : Helvetica → Arial (metrics differences)
- **Transparency flattening artifacts** : objets fondus multi-modes blend
- **Layer merging** : OCG flatten perdant sélectivité
- **Tagged PDF generation** : heuristique création structure depuis layout
- **OCR overlay alignment** : shift sub-pixel entre image/texte
- **Font embedding subsetting** : glyphes dropped non utilisés
- **Resolution resampling** : downsample 300dpi → 150dpi artifacts
- **Linearization corruption** : hint tables breakées
- **Metadata scrubbing** : suppression XMP/EXIF (privacy)

---

### **60. MÉTA-DONNÉES DE JOB & WORKFLOW**
- **Job ID** : identifiant unique traitement
- **Queue time** : temps attente avant processing
- **Processing time** : durée chaque étape pipeline
- **Quality gate scores** : scores seuil (SSIM > 0.95, CER < 0.01)
- **Human review flags** : zones signalées pour relecture humaine
- **Reconstruction attempt logs** : logs détaillés par objet
- **Fallback actions** : quand plan A échoue, plan B appliqué
- **Audit trail** : qui, quand, quelle action sur doc
- **Versioning decisions** : pourquoi version N créée
- **Cost metrics** : tokens AI, compute time, storage

---

**Synthèse :** Cette liste représente **l'ensemble des dimensions** qu'un système WYSIWYG de haute fidélité doit mesurer, tracer et reproduire. Chaque élément a :
- **Une valeur** (ex: font-size=9.5pt)
- **Une provenance** (native, OCR, vision, heuristique)
- **Une confiance** (0.0-1.0)
- **Une dépendance** (hérité, override, calculé)
- **Une criticité** (bloquant vs tolerant)

La reconstruction fidèle est un **problème d'optimisation multi-contraintes** où la solution doit minimiser la **distance perçue** par l'humain tout en respectant les **contraintes matérielles** (pagination, références croisées, signatures).

Cette liste est **cliniquement exhaustive**. Pour chaque élément, la reconstruction WYSIWYG exige non seulement la valeur, mais sa **provenance**, sa **géométrie exacte** et son **contexte d'application** (style hérité, override, condition). La moindre omission provoque un décalage perceptible.

**NB :** Chaque élément doit être associé à sa **provenance exacte** (source et confiance), sa **géométrie précise**, et ses **dépendances** avec d'autres éléments. La reconstruction WYSIWYG imposera de refaire "sembler" ces éléments identiques visuellement, même si la font originale n'existe pas sur le système de rendu.
