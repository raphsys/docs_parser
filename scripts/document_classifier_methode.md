Il faut distinguer **deux problèmes différents** :

1. **classifier le type de document/page**
   Exemple : facture, CV, article scientifique, journal, formulaire, lettre, slide, rapport, page de livre.

2. **classifier le type de mise en page / présentation**
   Exemple : 1 colonne, 2 colonnes, page très dense, page avec tableau dominant, page avec image dominante, page type formulaire, page type sommaire, page type couverture.

Les deux ne sont pas équivalents. Deux documents de nature différente peuvent partager la même structure visuelle, et inversement.

## 1. La bonne manière de penser la classification

Le plus robuste est de faire une **classification multi-niveaux**.

### Niveau A — catégorie documentaire

Ce que représente la page :

* lettre
* facture
* reçu
* formulaire
* contrat
* article scientifique
* page de livre
* journal / magazine
* CV
* diapositive
* rapport administratif
* page web imprimée
* publicité / affiche
* page mixte inconnue

### Niveau B — structure de page

Comment la page est organisée :

* pleine page texte
* une colonne
* deux colonnes
* trois colonnes ou plus
* grille / blocs
* formulaire à champs
* tableau dominant
* image dominante
* page mixte texte + figures
* sommaire / table des matières
* couverture / page de titre
* annexe / références
* page avec encadrés latéraux

### Niveau C — densité et style visuel

Descripteurs plus fins :

* dense / aérée
* marges étroites / larges
* alignement justifié / à gauche
* hiérarchie typographique faible / forte
* beaucoup de titres / peu de titres
* beaucoup de listes / peu de listes
* présence de numérotation
* présence d’en-tête / pied de page
* présence de tableaux, images, figures, légendes, notes de bas de page

En pratique, un bon système renvoie quelque chose comme :

> `document_type=article_scientifique`
> `layout_type=deux_colonnes`
> `density=élevée`
> `main_regions=[title, abstract, section-header, text, figure, caption, footer]`

C’est bien plus exploitable qu’un seul label.

---

## 2. Ce qu’il faut extraire pour classifier correctement

Il ne faut pas se baser seulement sur le texte OCR. Il faut combiner **texte + géométrie + typographie + objets de page**.

### A. Caractéristiques géométriques

* largeur / hauteur de page
* ratio de page
* nombre de colonnes
* positions des blocs
* taille moyenne des blocs
* distribution des blocs sur la page
* taux d’occupation de la surface
* marges haut/bas/gauche/droite
* régularité d’alignement
* présence de zones vides structurées

### B. Caractéristiques textuelles

* longueur moyenne des lignes
* longueur moyenne des paragraphes
* fréquence des chiffres
* fréquence des dates
* fréquence des montants
* fréquence des mots-clés (“invoice”, “abstract”, “references”, “chapter”, “table of contents”, etc.)
* proportion de texte en majuscules
* présence de listes numérotées ou à puces
* motifs de formulaire : `Nom:`, `Date:`, `Signature:`

### C. Caractéristiques typographiques

* nombre de tailles de police distinctes
* taille dominante
* contraste titre / corps
* gras / italique / souligné
* centrage du titre
* capitalisation des titres
* espacement entre lignes et entre blocs

### D. Caractéristiques de layout sémantique

Après détection de régions :

* nombre de titres
* nombre de blocs texte
* nombre d’images
* nombre de tableaux
* nombre de légendes
* nombre de en-têtes / pieds de page
* présence de notes de bas de page
* présence d’une grande zone “table”

C’est précisément l’idée des pipelines modernes de **document layout analysis** : détecter des régions comme *title, text, table, picture, caption, footnote, header, footer*, puis utiliser cette structure pour mieux comprendre la page. Des jeux de données comme **DocLayNet** ont été créés justement parce que les anciens datasets étaient trop centrés sur les articles scientifiques et généralisaient mal aux mises en page variées. DocLayNet contient 80 863 pages annotées et 11 classes de layout, ce qui en fait une base utile pour une approche généraliste. ([GitHub][1])

---

## 3. Les trois approches possibles

## Approche 1 — règles heuristiques

La plus simple pour démarrer.

Exemples :

* si `nb_colonnes >= 2` et présence de `abstract` ou `references` → article scientifique probable
* si beaucoup de cellules alignées + chiffres + montants → facture ou tableau financier
* si texte court + grosse image + peu de blocs → affiche / slide / couverture
* si beaucoup de paires `label: valeur` → formulaire
* si lignes de points + numéros de page → sommaire

### Avantages

* rapide
* explicable
* utile pour bootstraper un dataset

### Limites

* fragile
* casse vite sur des documents réels
* gère mal les cas hybrides

Cette approche est bonne pour faire une **v1** ou produire des pseudo-labels.

---

## Approche 2 — features structurées + classifieur classique

Ici, tu extrais des variables de layout, puis tu entraînes :

* Random Forest
* XGBoost
* SVM
* LightGBM

Exemple de vecteur de features :

* `num_text_blocks`
* `num_tables`
* `num_images`
* `num_columns`
* `avg_font_size`
* `font_size_std`
* `text_coverage_ratio`
* `header_present`
* `footer_present`
* `digit_ratio`
* `avg_line_width`
* `toc_pattern_score`

### Avantages

* très bon compromis
* interprétable
* fonctionne bien si tes classes sont bien définies

### Limites

* dépend fortement de la qualité de l’extraction
* moins robuste que les modèles multimodaux sur layouts complexes

C’est souvent la meilleure voie si tu veux un système industriel contrôlable.

---

## Approche 3 — modèle profond multimodal

La plus puissante.

Tu utilises :

* image de page
* texte OCR
* boîtes englobantes
* parfois objets de layout détectés

Les familles de modèles pertinentes incluent **LayoutLMv3**, conçu comme modèle multimodal généraliste pour le Document AI et pouvant être affiné pour la **classification de documents** et pour la **layout analysis**. ([Hugging Face][2])

Autres briques utiles :

* détecteur de layout en amont : DocLayout-YOLO, RT-DETR/DFINE, etc.
* encodeur document multimodal en aval
* clustering ou classification supervisée sur embeddings

Des travaux récents montrent aussi que les **embeddings multimodaux** peuvent mieux distinguer des **templates** proches au sein d’une même catégorie documentaire que des approches purement textuelles. ([arXiv][3])

### Avantages

* plus robuste
* capture le visuel global
* gère mieux les cas réels

### Limites

* plus lourd
* demande des données annotées
* moins transparent si tu dois expliquer chaque décision

---

## 4. Pipeline recommandé en pratique

Je te conseille un pipeline en **deux étages**, pas un modèle monolithique.

### Étape 1 — analyse structurelle

Détecter les régions :

* titre
* texte
* tableau
* image
* légende
* header
* footer
* list-item
* section-header
* footnote
* formule

C’est exactement le type de taxonomie utilisé dans DocLayNet. ([Hugging Face][4])

### Étape 2 — construction d’un “profil de page”

À partir des régions détectées, calculer :

* nombre de colonnes
* densité
* surface couverte par texte
* surface couverte par tableaux
* hiérarchie de titres
* ratio texte / image / table
* présence de structures répétitives

### Étape 3 — classification

Faire trois sorties séparées :

* `document_type`
* `layout_type`
* `style_profile`

### Étape 4 — éventuellement clustering

Si tu n’as pas encore les classes finales, tu peux :

* générer un embedding par page
* faire clustering
* nommer les clusters ensuite

C’est souvent meilleur que d’inventer dès le début 50 classes mal définies.

---

## 5. Taxonomie concrète que je te recommande

Pour éviter le flou, voici une taxonomie utile.

### Types de documents

* book_page
* scientific_paper
* report
* invoice
* receipt
* form
* letter
* cv_resume
* newspaper_magazine
* slide
* contract
* manual
* advertisement_poster
* web_print
* mixed_unknown

### Types de layout

* single_column
* double_column
* multi_column
* title_page
* toc_page
* text_dense
* image_dominant
* table_dominant
* form_layout
* mixed_blocks
* annotated_page
* reference_page

### Types de style

* minimalist
* dense_professional
* academic
* editorial
* administrative
* marketing_visual
* tabular_structured
* scanned_noisy

Il vaut mieux avoir **plusieurs axes** qu’un seul label confus.

---

## 6. Comment détecter automatiquement certaines structures clés

### Détection du nombre de colonnes

Méthodes possibles :

* projection verticale des pixels/boîtes texte
* clustering des centres x des blocs
* analyse des gaps verticaux continus

### Détection d’une page sommaire

Signaux :

* lignes courtes répétées
* points de suite
* numéro de page aligné à droite
* beaucoup d’entrées sur une structure répétitive

### Détection d’un formulaire

Signaux :

* couples champ/valeur
* zones alignées
* lignes ou cases
* libellés courts suivis d’espace vide

### Détection d’une facture

Signaux :

* zones d’identité
* date
* numéro
* montants
* tableau de lignes
* total / taxes / sous-total

### Détection d’un article scientifique

Signaux :

* titre centré en haut
* auteurs
* abstract
* sections numérotées
* 2 colonnes
* figures + captions
* références en fin

---

## 7. Ce que beaucoup de pipelines font mal

Le point critique : beaucoup de systèmes confondent :

* **classe documentaire** et **layout visuel**
* **page** et **document entier**
* **OCR** et **structure**
* **bloc détecté** et **rôle sémantique**

Exemple : une page de rapport avec un grand tableau peut être classée “table” alors qu’elle appartient à un document “rapport administratif”. Il faut donc souvent avoir :

* une classification **par page**
* puis une agrégation **au niveau document**

Exemple :

* pages 1–2 : title_page / executive_summary
* pages 3–20 : report / single_column
* pages 21–25 : annex / table_dominant

La vérité utile est souvent au niveau **séquence de pages**, pas seulement au niveau page isolée.

---

## 8. Stratégie réaliste si tu veux implémenter cela

### Option pragmatique

1. Détecter le layout avec un modèle entraîné sur DocLayNet ou équivalent.
2. Extraire des features géométriques + textuelles + typographiques.
3. Entraîner un classifieur XGBoost sur tes propres classes métier.
4. Ajouter des règles métier au-dessus pour corriger certains cas.

C’est souvent meilleur qu’un gros modèle end-to-end aveugle.

### Option avancée

1. OCR + layout detection
2. Encodage multimodal type LayoutLMv3
3. Fine-tuning multi-tâches :

   * document type
   * layout type
   * structural tags

### Option exploratoire

1. Embeddings multimodaux de pages
2. UMAP / HDBSCAN / clustering hiérarchique
3. Inspection manuelle des clusters
4. Définition de la taxonomie finale
5. Fine-tuning supervisé

---

## 9. Schéma cible pour ton cas

Vu tes projets de reconstruction WYSIWYG, il ne faut pas seulement classifier la page, mais produire une **fiche structurelle exploitable** :

```json
{
  "document_type": "scientific_paper",
  "layout_type": "double_column",
  "style_profile": "academic_dense",
  "page_role": "body_page",
  "regions": [
    {"type": "title", "bbox": [x1,y1,x2,y2]},
    {"type": "text", "bbox": [x1,y1,x2,y2]},
    {"type": "figure", "bbox": [x1,y1,x2,y2]},
    {"type": "caption", "bbox": [x1,y1,x2,y2]}
  ],
  "features": {
    "num_columns": 2,
    "text_coverage": 0.71,
    "table_coverage": 0.00,
    "image_coverage": 0.12,
    "header_present": false,
    "footer_present": true,
    "font_size_levels": 4
  }
}
```

C’est cela qui sera utile ensuite pour :

* extraction structurée
* traduction
* reconstruction fidèle
* routage vers des pipelines spécialisés

---

## 10. Réponse directe à ta question

Pour classifier des documents/pages selon la présentation, la mise en forme et la mise en page, il faut :

1. **définir une taxonomie claire en plusieurs axes**
   type documentaire, type de layout, style visuel

2. **extraire la structure de page**
   blocs, colonnes, tableaux, images, titres, headers, footers

3. **combiner plusieurs signaux**
   texte OCR, géométrie, typographie, régions détectées

4. **entraîner un classifieur sur ces signaux**
   soit classique sur features, soit multimodal

5. **agréger si besoin au niveau du document entier**
   car une page isolée ne suffit souvent pas

Le point le plus important : **ne pas réduire le problème à une simple classification d’image**. Un document est un objet structuré, pas une photo ordinaire. Les approches modernes de Document AI, notamment autour de LayoutLMv3 et des datasets de layout variés comme DocLayNet, vont précisément dans ce sens. ([Hugging Face][2])

-----


Voici une version **prête à coder**, structurée pour un pipeline réel.

Le point de départ est simple : ne fais pas une seule classe “type de page”. Fais une **classification multi-axes**. C’est plus juste, plus stable, et beaucoup plus utile ensuite pour l’extraction, le routage et la reconstruction. Les jeux de données et outils modernes de Document AI vont dans ce sens : DocLayNet fournit 80 863 pages annotées avec 11 classes de layout, LayoutLMv3 est conçu pour des tâches Document AI multimodales, et PP-StructureV3 vise explicitement l’analyse de layout, les tableaux, les formules et l’ordre de lecture. ([GitHub][1])

## 1. Taxonomie recommandée

### Axe A — `document_type`

C’est la nature documentaire probable de la page ou du document :

* `book_page`
* `scientific_paper`
* `report`
* `administrative_letter`
* `invoice`
* `receipt`
* `form`
* `contract`
* `cv_resume`
* `newspaper_magazine`
* `slide`
* `manual_guide`
* `advertisement_poster`
* `web_print`
* `mixed_unknown`

### Axe B — `layout_type`

C’est la structure visuelle dominante :

* `single_column`
* `double_column`
* `multi_column`
* `title_page`
* `toc_page`
* `form_layout`
* `table_dominant`
* `image_dominant`
* `mixed_blocks`
* `dense_text`
* `reference_page`
* `annotated_page`

### Axe C — `page_role`

Le rôle de la page dans le document :

* `cover`
* `front_matter`
* `title`
* `summary`
* `body`
* `appendix`
* `references`
* `toc`
* `index`
* `back_matter`
* `unknown`

### Axe D — `style_profile`

Descripteur de style global :

* `academic_dense`
* `administrative_clean`
* `editorial_visual`
* `tabular_structured`
* `marketing_visual`
* `scanned_noisy`
* `minimalist`
* `mixed_irregular`

Cette séparation évite une erreur fréquente : une page peut être `document_type=report`, `layout_type=table_dominant`, `page_role=appendix`. Un seul label écraserait cette réalité.

---

## 2. Sortie JSON cible

Je te conseille une structure comme celle-ci :

```json
{
  "page_id": "doc_001_p003",
  "document_type": "scientific_paper",
  "layout_type": "double_column",
  "page_role": "body",
  "style_profile": "academic_dense",
  "confidence": {
    "document_type": 0.93,
    "layout_type": 0.97,
    "page_role": 0.88,
    "style_profile": 0.81
  },
  "regions": [
    {
      "type": "title",
      "bbox": [120, 80, 980, 180],
      "score": 0.98
    },
    {
      "type": "text",
      "bbox": [90, 220, 530, 1280],
      "score": 0.96
    },
    {
      "type": "text",
      "bbox": [570, 220, 1010, 1280],
      "score": 0.95
    },
    {
      "type": "figure",
      "bbox": [620, 850, 980, 1080],
      "score": 0.91
    },
    {
      "type": "caption",
      "bbox": [620, 1085, 980, 1150],
      "score": 0.89
    }
  ],
  "features": {
    "page_width": 1107,
    "page_height": 1388,
    "aspect_ratio": 0.7976,
    "num_columns": 2,
    "num_text_blocks": 14,
    "num_title_blocks": 1,
    "num_table_blocks": 0,
    "num_figure_blocks": 1,
    "num_caption_blocks": 1,
    "text_coverage_ratio": 0.71,
    "table_coverage_ratio": 0.00,
    "figure_coverage_ratio": 0.11,
    "header_present": false,
    "footer_present": true,
    "avg_line_length_px": 356.4,
    "font_size_levels": 4,
    "alignment_mode": "justified",
    "ocr_digit_ratio": 0.06,
    "toc_pattern_score": 0.01,
    "form_pattern_score": 0.02
  }
}
```

Pour une reconstruction WYSIWYG ou un routage vers des pipelines spécialisés, c’est ce format qu’il faut viser, pas juste un label final.

---

## 3. Les features à calculer

Il faut combiner **géométrie + texte + typographie + structure détectée**.

### A. Features géométriques

* `page_width`, `page_height`, `aspect_ratio`
* `num_blocks_total`
* `block_area_mean`, `block_area_std`
* `text_coverage_ratio`
* `image_coverage_ratio`
* `table_coverage_ratio`
* `top_margin`, `bottom_margin`, `left_margin`, `right_margin`
* `whitespace_ratio`
* `num_columns`
* `column_balance_score`
* `x_alignment_entropy`
* `y_spacing_mean`, `y_spacing_std`

### B. Features textuelles OCR

* `num_words`
* `num_lines`
* `avg_words_per_line`
* `avg_chars_per_line`
* `digit_ratio`
* `uppercase_ratio`
* `punctuation_ratio`
* `currency_symbol_count`
* `date_pattern_count`
* `email_count`
* `url_count`
* `toc_pattern_score`
* `form_pattern_score`
* `scientific_pattern_score`
* `invoice_pattern_score`

### C. Features typographiques

* `font_size_mean`
* `font_size_std`
* `font_size_levels`
* `bold_ratio`
* `italic_ratio`
* `centered_title_present`
* `heading_body_ratio`
* `line_spacing_mean`
* `paragraph_spacing_mean`

### D. Features sémantiques de régions

Après layout detection :

* `num_title_blocks`
* `num_text_blocks`
* `num_table_blocks`
* `num_picture_blocks`
* `num_caption_blocks`
* `num_header_blocks`
* `num_footer_blocks`
* `num_formula_blocks`
* `num_list_blocks`
* `reading_order_complexity`

Les pipelines modernes d’analyse documentaire mettent justement l’accent sur la détection de régions telles que texte, titres, tableaux, figures, légendes, en-têtes et pieds de page, parce que ces régions structurent beaucoup mieux le document que l’OCR seul. ([PaddlePaddle][2])

---

## 4. Scores heuristiques utiles

Avant même un modèle, calcule quelques scores métiers.

### `toc_pattern_score`

Augmente si :

* nombreuses lignes courtes
* points de suite fréquents
* numéros alignés à droite
* structure répétitive “titre … page”

### `form_pattern_score`

Augmente si :

* présence de `:` fréquents
* nombreux couples label/valeur
* alignements horizontaux répétitifs
* cases, lignes, champs

### `invoice_pattern_score`

Augmente si :

* montants
* dates
* références de facture
* tableau de lignes
* sous-total / total / taxe / remise

### `scientific_pattern_score`

Augmente si :

* `abstract`, `keywords`, `references`
* sections numérotées
* doubles colonnes
* figures + captions
* citations bibliographiques

Ces scores sont utiles soit comme features, soit comme garde-fous.

---

## 5. Pipeline recommandé

Je te recommande ce pipeline en 6 étapes.

### Étape 1 — normalisation d’entrée

Pour chaque page :

* rasteriser proprement le PDF
* corriger orientation si besoin
* deskew
* normaliser DPI
* garder aussi la version PDF native si disponible

Des pipelines documentaires récents incluent justement des modules de prétraitement comme classification d’orientation et redressement. ([PaddlePaddle][3])

### Étape 2 — layout detection

Détecter les régions :

* `title`
* `text`
* `table`
* `picture`
* `caption`
* `header`
* `footer`
* `formula`
* `list_item`
* `section_header`
* `footnote`

Pour cela, tu peux démarrer avec un pipeline du type PP-StructureV3 ou tout autre détecteur de layout crédible, car il cible explicitement la détection de layout et la restauration d’ordre de lecture. ([PaddlePaddle][2])

### Étape 3 — OCR / extraction native

* texte natif PDF si disponible
* sinon OCR
* puis fusion avec les boîtes

docTR est utile ici comme brique OCR généraliste, mais ce n’est pas à lui seul un moteur complet de compréhension de mise en page. ([GitHub][4])

### Étape 4 — extraction de features

Construire un vecteur tabulaire par page.

### Étape 5 — classification multi-tête

Trois possibilités :

1. **baseline heuristique**
2. **XGBoost / LightGBM sur features**
3. **modèle multimodal type LayoutLMv3 pour raffiner**

LayoutLMv3 est bien adapté quand tu veux fusionner image, texte et positions, et il est présenté comme un modèle généraliste pour le Document AI, applicable à la classification documentaire et à l’analyse de layout. ([Hugging Face][5])

### Étape 6 — agrégation au niveau document

Ne t’arrête pas à la page.

Exemple :

* page 1 = `title_page`
* page 2 = `toc_page`
* pages 3–40 = `report/body`
* pages 41–52 = `appendix/table_dominant`

Le document final peut alors être :

* `document_type=report`
* structure interne = séquence de rôles de pages

C’est beaucoup plus réaliste.

---

## 6. Architecture logicielle

Une structure Python propre :

```text
doc_classifier/
├── configs/
│   ├── labels.yaml
│   ├── features.yaml
│   └── train.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── annotations/
├── src/
│   ├── io/
│   │   ├── pdf_loader.py
│   │   ├── image_loader.py
│   │   └── page_renderer.py
│   ├── preprocess/
│   │   ├── deskew.py
│   │   ├── normalize.py
│   │   └── orientation.py
│   ├── layout/
│   │   ├── detector.py
│   │   ├── reading_order.py
│   │   └── region_schema.py
│   ├── ocr/
│   │   ├── native_pdf.py
│   │   ├── doctr_runner.py
│   │   └── fusion.py
│   ├── features/
│   │   ├── geometry.py
│   │   ├── text.py
│   │   ├── typography.py
│   │   ├── structure.py
│   │   └── builder.py
│   ├── heuristics/
│   │   ├── toc_score.py
│   │   ├── form_score.py
│   │   ├── invoice_score.py
│   │   └── scientific_score.py
│   ├── models/
│   │   ├── xgb_classifier.py
│   │   ├── layoutlmv3_classifier.py
│   │   └── ensemble.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── train_xgb.py
│   │   ├── train_layoutlmv3.py
│   │   └── evaluate.py
│   ├── inference/
│   │   ├── predict_page.py
│   │   ├── predict_document.py
│   │   └── explain.py
│   └── schemas/
│       ├── page_profile.py
│       └── document_profile.py
└── notebooks/
```

---

## 7. Dataset d’annotation que je te conseille

Il faut annoter **par page** avec ce format minimal :

```json
{
  "page_id": "sample_001_p01",
  "document_type": "invoice",
  "layout_type": "table_dominant",
  "page_role": "body",
  "style_profile": "tabular_structured"
}
```

Et idéalement ajouter :

```json
{
  "regions": [
    {"type": "header", "bbox": [..]},
    {"type": "table", "bbox": [..]},
    {"type": "footer", "bbox": [..]}
  ]
}
```

### Volume minimal réaliste

Pour une première version exploitable :

* 200 à 500 pages annotées par grande famille si tes classes sont bien définies
* au moins 50 à 100 pages pour les classes rares
* sinon tu vas surapprendre du bruit

### Règle importante

Il faut annoter **ce que tu veux décider**, pas ce qui “ressemble vaguement”.

Exemple :

* une facture scannée floue reste `invoice`
* un relevé financier tabulaire n’est pas automatiquement `invoice`

La taxonomie doit être juridiquement et fonctionnellement stable.

---

## 8. Méthode d’entraînement conseillée

### V1 robuste et simple

Je te conseille de commencer par :

* layout detector
* OCR / texte natif
* extraction de features
* XGBoost multi-sorties ou 4 classifieurs séparés

Pourquoi ?
Parce que c’est interprétable, moins coûteux et souvent très compétitif sur des données métier structurées.

### V2

Ajouter ensuite un raffinement multimodal :

* LayoutLMv3 comme classifieur de page
* fusion avec les features tabulaires
* petit ensemble final

### Stratégie d’ensemble

Par exemple :

* `XGBoost_features`
* `LayoutLMv3_page_embedding`
* `heuristic_scores`

Puis méta-classifieur léger.

---

## 9. Critères d’évaluation

Ne mesure pas seulement l’accuracy globale.

Il faut au minimum :

* macro F1
* weighted F1
* confusion matrix
* F1 par classe
* top-2 accuracy
* calibration des probabilités

Et surtout :

* évaluation **par source documentaire**
* évaluation **par template**
* évaluation **hors distribution**

Sinon ton modèle va juste mémoriser quelques gabarits.

---

## 10. Cas difficiles à traiter explicitement

Tu dois prévoir une classe ou un mécanisme pour :

* pages hybrides
* scans très dégradés
* pages quasi vides
* annexes atypiques
* pages manuscrites
* pages uniquement graphiques
* documents composites

Il faut aussi prévoir un seuil de rejet :

```json
{
  "document_type": "mixed_unknown",
  "reason": "low_confidence / conflicting_signals"
}
```

C’est préférable à une fausse certitude.

---

## 11. Règles simples de post-traitement

Quelques règles métier valent la peine :

* si plusieurs pages consécutives ont le même `document_type`, lisser la séquence
* si page 1 = `title_page` et pages suivantes = `report/body`, alors document global probablement `report`
* si une seule page est `toc_page`, ne pas en déduire que tout le document est un sommaire
* si `layout_type=double_column` + `scientific_pattern_score` élevé, augmenter la probabilité `scientific_paper`
* si `form_pattern_score` élevé mais présence d’un grand tableau de lignes et montants, privilégier `invoice` sur `form`

Ce genre de logique améliore nettement la stabilité.

---

## 12. Pseudo-code du pipeline

```python
def classify_page(page_image, pdf_native=None):
    page = preprocess_page(page_image)

    layout_regions = detect_layout(page)
    ocr_tokens = extract_text(page, pdf_native=pdf_native)

    features = build_features(
        page=page,
        layout_regions=layout_regions,
        ocr_tokens=ocr_tokens,
    )

    heuristic_scores = compute_heuristic_scores(
        ocr_tokens=ocr_tokens,
        layout_regions=layout_regions,
        features=features,
    )

    pred_structured = xgb_model.predict_proba(features, heuristic_scores)
    pred_multimodal = layoutlmv3_model.predict(page, ocr_tokens, layout_regions)

    final_pred = ensemble_predict(pred_structured, pred_multimodal)

    return {
        "document_type": final_pred["document_type"],
        "layout_type": final_pred["layout_type"],
        "page_role": final_pred["page_role"],
        "style_profile": final_pred["style_profile"],
        "regions": layout_regions,
        "features": features,
        "heuristics": heuristic_scores,
        "confidence": final_pred["confidence"],
    }
```

---

## 13. Ce que je te recommande concrètement pour démarrer

Pour un système sérieux mais réaliste :

### Stack v1

* rendu PDF → image
* détection de layout
* OCR / texte natif
* features structurées
* XGBoost
* post-règles

### Stack v2

* embeddings / classification avec LayoutLMv3
* ensemble final

### Pourquoi cette voie

Parce qu’elle est :

* plus contrôlable
* plus explicable
* plus compatible avec tes besoins de reconstruction
* meilleure pour faire du debug que du end-to-end opaque

---

## 14. Verdict direct

La bonne méthode n’est pas “classifier l’image de page comme une photo ImageNet”.
La bonne méthode est :

1. **détecter la structure**
2. **extraire des features multi-niveaux**
3. **classer selon plusieurs axes**
4. **agréger au niveau document**
5. **prévoir l’incertitude et les cas mixtes**

C’est la seule approche sérieuse si ton but final est l’exploitation documentaire réelle.

