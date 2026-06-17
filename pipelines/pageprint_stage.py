"""PagePrintStage — unité 5 du pipeline cible : la tête PAGEPRINT.

    5. PAGEPRINT
       - normalise
       - résout conflits
       - fabrique units
       - indexe regions
       - construit graph
       - compile policies
       - compile constraints
       - évalue quality
       - produit INPUT_DATA
"""

from __future__ import annotations

from pageprint import PagePrintBuilder


class PagePrintStage:
    def __init__(self, *, default_dpi: float = 150):
        self.builder = PagePrintBuilder(default_dpi=default_dpi)

    def build(self, *, page_structure: dict, source_context: dict,
              extraction_result: dict, assets: dict | None = None,
              page_index: int = 0, page_image=None, pdf_page=None) -> dict:
        # Le handle PyMuPDF ne doit pas fuiter dans INPUT_DATA.
        clean_source_context = {
            k: v for k, v in (source_context or {}).items() if k != "document"
        }
        return self.builder.build(
            page_structure=page_structure,
            source_context=clean_source_context,
            extraction_result=extraction_result,
            assets=assets or {},
            page_image=page_image,
            pdf_page=pdf_page,
            page_index=page_index,
        )
