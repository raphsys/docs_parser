# Frontend Expo

Interface React Native unique pour piloter l'API `ocr_server.py` sur:
- web
- Android
- iOS

## Démarrage

```bash
cd /home/raphael/Mes_Projets/docs_parser
source .docs-parser/bin/activate
python ocr_server.py
```

Dans un second terminal:

```bash
cd /home/raphael/Mes_Projets/docs_parser/frontend-expo
npm install
npm run web
```

Ou mobile:

```bash
npm run android
npm run ios
```

## Backend attendu

L'UI appelle:
- `POST /ocr`
- `POST /reconstruct`

Par défaut:
- web: `http://127.0.0.1:8001`
- Android émulateur: `http://10.0.2.2:8001`

Sur appareil physique, il faut renseigner l'IP LAN de la machine qui exécute `ocr_server.py`.

## CORS web

Le backend active CORS pour les origines locales `localhost` et `127.0.0.1`
sur les ports de dev les plus courants, y compris quand Expo bascule
automatiquement vers `8082`, `8083`, etc.

Pour autoriser d'autres origines:

```bash
export DOCS_PARSER_CORS_ORIGINS="http://192.168.1.20:19006,http://localhost:19006"
python /home/raphael/Mes_Projets/docs_parser/ocr_server.py
```

## Fonctions exposées

- sélection d'un PDF ou d'une image
- options `force_ai`, `font_ai_audit`, `text_removal_mode`
- options `target_lang`, `style`, `tone`
- toggles `debug_compare` et `export_html`
- exécution `OCR seulement`
- exécution `Reconstruction` à partir du dernier OCR
- exécution `Pipeline complet`
- affichage des payloads `coverage_report`, `publication_qa`, `visual_compare`
- ouverture du PDF et du HTML générés
