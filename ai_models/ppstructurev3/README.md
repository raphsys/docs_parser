Model root réservé à `LayoutAIEnricher`.

Les sous-répertoires attendus sont:

- `PP-DocLayout_plus-L`
- `PP-Chart2Table`
- `PP-DocBlockLayout`
- `PP-OCRv5_server_det`
- `PP-LCNet_x1_0_textline_ori`
- `PP-OCRv5_server_rec`

Pour le profil avancé `PP-StructureV3`, ajouter aussi:

- `PP-LCNet_x1_0_doc_ori`
- `UVDoc`
- `PP-LCNet_x1_0_table_cls`
- `SLANeXt_wired`
- `SLANet_plus`
- `RT-DETR-L_wired_table_cell_det`
- `RT-DETR-L_wireless_table_cell_det`
- `PP-FormulaNet_plus-L`
- `PP-OCRv4_server_seal_det`

Téléchargement recommandé:

```bash
source .docs-parser/bin/activate
python scripts/download_layout_ai_models.py --profile advanced
```

Quand le bundle minimal est présent, `LayoutAIEnricher` s’active automatiquement. Quand le bundle complet est présent, il bascule automatiquement sur le profil `advanced`.
