import tempfile
import unittest
from pathlib import Path

from html_exporter import HtmlStyleExporter


class HtmlExporterTests(unittest.TestCase):
    def test_layout_v2_toc_uses_translated_labels_in_html(self):
        exporter = HtmlStyleExporter()
        page = {
            "schema_version": "layout.v2",
            "page_role": "toc",
            "layout": {"margins": {"left": 40, "right": 40, "top": 40, "bottom": 40}},
            "blocks": [
                {
                    "role": "body",
                    "translated_text": "",
                    "lines": [{"phrases": [{"texte": "CONTENTS"}]}],
                }
            ],
            "toc": {
                "toc_rows": [
                    {"role": "toc_title", "label": "CONTENTS", "translated_label": "Sommaire", "page": "vii"},
                    {
                        "role": "chapter_title",
                        "label": "Convolutional neural networks",
                        "translated_label": "Reseaux de neurones convolutionnels",
                        "page": "92",
                    },
                ]
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "toc.html"
            exporter.export([page], str(output_path))
            content = output_path.read_text(encoding="utf-8")

        self.assertIn("Sommaire", content)
        self.assertIn("Reseaux de neurones convolutionnels", content)
        self.assertNotIn(">CONTENTS<", content)
        self.assertNotIn(">Convolutional neural networks<", content)


if __name__ == "__main__":
    unittest.main()
