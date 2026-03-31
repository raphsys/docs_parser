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
        self.advances_pdf_path = Path(__file__).resolve().parent / "doc_pdf" / "Advances in Deep Learning.pdf"

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

    def _find_style_matching(self, page_index, predicate):
        with fitz.open(self.pdf_path) as doc:
            native = self.extractor.extract_page(doc[page_index], sx=1.0, sy=1.0)

        for block in native.get("blocks", []):
            for line in block.get("lines", []):
                for phrase in line.get("phrases", []):
                    for span in phrase.get("spans", []):
                        style = span.get("style") or {}
                        if predicate(style):
                            return style
        self.fail("Matching style not found")

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

    def test_native_extractor_resolves_truncated_italic_variant_to_embedded_font(self):
        style = self._find_style_matching(
            461,
            lambda style: "newbaskerville" in str(style.get("font") or "").lower() and "itali" in str(style.get("font") or "").lower(),
        )

        embedded_font_path = Path(style.get("embedded_font_path") or "")
        self.assertTrue(embedded_font_path.is_file(), embedded_font_path)
        self.assertIn("italic", str(style.get("font_key_normalized") or "") + embedded_font_path.stem)

    def test_native_block_style_is_promoted_from_spans(self):
        with fitz.open(self.pdf_path) as doc:
            native = self.extractor.extract_page(doc[461], sx=1.0, sy=1.0)

        styled_blocks = [block for block in native.get("blocks", []) if isinstance(block.get("style"), dict) and block.get("style")]
        self.assertTrue(styled_blocks)
        self.assertTrue(any(block.get("resolved_style") for block in styled_blocks))

    def test_font_resolver_can_recover_from_missing_embedded_path_using_embedded_cache(self):
        style = self._find_style_matching(
            461,
            lambda style: "newbaskerville" in str(style.get("font") or "").lower() and "itali" in str(style.get("font") or "").lower(),
        )
        degraded_style = dict(style)
        degraded_style["embedded_font_path"] = None

        resolved = self.resolver.resolve(degraded_style)

        self.assertTrue(Path(resolved.get("fontfile") or "").is_file(), resolved)
        self.assertIsNone(resolved.get("builtin"))

    def test_native_extractor_resolves_subset_obfuscated_font_to_embedded_font(self):
        with fitz.open(self.advances_pdf_path) as doc:
            native = self.extractor.extract_page(doc[11], sx=1.0, sy=1.0)

        target_style = None
        for block in native.get("blocks", []):
            for line in block.get("lines", []):
                for phrase in line.get("phrases", []):
                    for span in phrase.get("spans", []):
                        style = span.get("style") or {}
                        if str(style.get("font") or "").startswith("DrdjnpKbqxwpPmnpjpAdvTT5"):
                            target_style = style
                            break
                    if target_style:
                        break
                if target_style:
                    break
            if target_style:
                break

        self.assertIsNotNone(target_style)
        embedded_font_path = Path(target_style.get("embedded_font_path") or "")
        self.assertTrue(embedded_font_path.is_file(), embedded_font_path)
        self.assertEqual(embedded_font_path.suffix.lower(), ".cff")

    def test_font_resolver_resolves_subset_obfuscated_font_without_fallback(self):
        with fitz.open(self.advances_pdf_path) as doc:
            self.extractor.extract_page(doc[11], sx=1.0, sy=1.0)
        style = {
            "font": "DrdjnpKbqxwpPmnpjpAdvTT5",
            "font_key_normalized": "drdjnpkbqxwppmnpjpadvtt5",
            "embedded_font_path": None,
            "flags": {"bold": False, "italic": False, "serif": False, "monospace": False},
        }
        resolved = self.resolver.resolve(style)
        self.assertTrue(Path(resolved.get("fontfile") or "").is_file(), resolved)
        self.assertIsNone(resolved.get("builtin"))


if __name__ == "__main__":
    unittest.main()
