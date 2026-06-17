# docs_parser Demo Studio — Flutter Desktop

Application locale non-web pour piloter les unités du pipeline :

- `pageprint`
- `pagetranslate`
- `pagereconstruct`
- `view_background`
- `audit_translation_selection`
- `audit_text_survival`
- `full`

## Lancement

Depuis la racine `docs_parser/` :

```bash
bash cmd_bash/run_demo_studio.sh
```

La première exécution lance :

```bash
flutter create --platforms=linux .
flutter pub get
flutter run -d linux
```

## Sorties

Les essais sont écrits dans :

```text
results/demo_studio_YYYYMMDD_HHMMSS/
```

Fichiers utiles :

```text
source_*.png
pageprint_bboxes_*.png
pageprint_*.json
pagetranslate_*.json
translated_input_data_*.json
cleanbg_*.png
background_compare_*.jpg
pagereconstruct_overlay_*.png
reconstructed_*.png
reconstructed_*.pdf
audit_*.json
audit_text_survival_*.json
audit_translation_selection_*.json
summary.json
report.md
contact_sheet.jpg
```

## Pré-requis Debian

```bash
sudo apt install clang cmake ninja-build pkg-config libgtk-3-dev python3-tk
```

Puis installer Flutter Desktop Linux si nécessaire.
