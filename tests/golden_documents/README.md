# Golden Documents

This folder is the controlled corpus area for translation trials.

The current committed fixtures are lightweight JSON contracts used by tests.
Real PDF fixtures can be added here later with one folder per document:

```text
toc/
body_text/
table_commands/
index_page/
caption_figure/
diagram_labels/
cover_image/
mixed_page/
```

Each real fixture should keep:

```text
source.pdf
input_data.json
translation_plan.json
expected_translation_units.json
audit_expected.json
```
