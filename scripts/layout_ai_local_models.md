# Layout AI Local Models

Pour activer `PP-StructureV3` en mode 100% local sur CPU dans ce projet, déposer les modèles dans:

`ai_models/ppstructurev3/`

Le code utilise maintenant ce chemin par défaut.

Tu peux aussi surcharger avec une variable d’environnement si nécessaire:

```bash
export LAYOUT_AI_ENABLE=1
export LAYOUT_AI_BACKEND=ppstructurev3
export LAYOUT_AI_MODELS_ROOT=/chemin/vers/layout_ai_models
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```

## Arborescence minimale recommandée

Le module [layout_ai_enricher.py](/home/raphael/Mes_Projets/docs_parser/layout_ai_enricher.py) cherche automatiquement ces sous-dossiers:

```text
ai_models/ppstructurev3/
  PP-DocLayout_plus-L/
  PP-Chart2Table/
  PP-DocBlockLayout/
  PP-OCRv5_server_det/
  PP-LCNet_x1_0_textline_ori/
  PP-OCRv5_server_rec/
```

## Rôle des modèles minimaux

- `PP-DocLayout_plus-L`
  - layout global de page
- `PP-DocBlockLayout`
  - détection de blocs/régions
- `PP-Chart2Table`
  - composant requis par le pipeline `PP-StructureV3` dans la pratique
- `PP-OCRv5_server_det`
  - détection de lignes/texte pour les contenus régionaux
- `PP-LCNet_x1_0_textline_ori`
  - orientation de lignes
- `PP-OCRv5_server_rec`
  - reconnaissance texte

## Pourquoi ce sous-ensemble

Le projet n’active pas encore dans l’enrichisseur IA:
- reconnaissance complète des tableaux
- reconnaissance de formules
- chart-to-table
- seal recognition

Donc ce sous-ensemble suffit pour une première couche de compréhension structurelle:
- régions
- layout
- texte régional

## Modèles complets optionnels

Si tu veux plus tard activer le pipeline complet `PP-StructureV3`, il faudra aussi prévoir:

```text
PP-Chart2Table/
PP-LCNet_x1_0_table_cls/
SLANeXt_wired/
SLANet_plus/
RT-DETR-L_wired_table_cell_det/
RT-DETR-L_wireless_table_cell_det/
PP-FormulaNet_plus-L/
PP-OCRv4_server_seal_det/
PP-LCNet_x1_0_doc_ori/
UVDoc/
```

## Vérification attendue

Quand les modèles minimaux sont présents localement, ce test doit passer:

```python
from layout_ai_enricher import get_layout_ai_enricher
enricher = get_layout_ai_enricher()
print(enricher.status())
```

Le statut attendu:
- `enabled = True`
- `backend = "ppstructurev3"`
- `models_root` renseigné
- `ready = False` avant premier chargement

Puis au premier `process_page(...)`, l’enrichisseur doit pouvoir s’instancier sans tentative de téléchargement.
