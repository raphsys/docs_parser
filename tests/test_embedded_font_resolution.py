import unittest
from pathlib import Path

import fitz

from font_resolver import FontResolver
from native_pdf_extractor import NativePDFExtractor


class EmbeddedFontResolutionTests(unittest.TestCase):
    def setUp(self):
        self.extractor = NativePDFExtractor()
        self.resolver = FontResolver()
        self.pdf_path = Path(__file__).resolve().parent / "doc_pdf" / "test_docintelligence.pdf"

    def _find_new_baskerville_style(self, page_index=8):
        with fitz.open(self.pdf_path) as doc:
            native = self.extractor.extract_page(doc[page_index], sx=1.0, sy=1.0)

        for block in native.get("blocks", []):
            for line in block.get("lines", []):
                for phrase in line.get("phrases", []):
                    for span in phrase.get("spans", []):
                        style = span.get("style") or {}
                        if "newbaskerville" in str(style.get("font") or "").lower():
                            return style
        self.fail("No NewBaskerville span found on the reference TOC page")

    def test_native_extractor_attaches_embedded_font_path(self):
        style = self._find_new_baskerville_style()

        embedded_font_path = Path(style.get("embedded_font_path") or "")
        self.assertTrue(embedded_font_path.is_file(), embedded_font_path)
        self.assertIn(embedded_font_path.suffix.lower(), {".cff", ".ttf", ".otf"})

    def test_font_resolver_prefers_embedded_font_file(self):
        style = self._find_new_baskerville_style()

        resolved = self.resolver.resolve(style)

        self.assertEqual(resolved.get("fontfile"), style.get("embedded_font_path"))
        self.assertIsNone(resolved.get("builtin"))


if __name__ == "__main__":
    unittest.main()
