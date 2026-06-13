"""PAGETRANSLATE — PagePrint-driven translation unit.

Deuxième tête du pipeline WYSIWYG. Consomme INPUT_DATA produit par
PAGEPRINT, sélectionne les unités textuelles traduisibles, ajoute le
contexte linguistique/documentaire, traduit, puis réinjecte les textes
traduits dans une copie de l'INPUT_DATA.
"""

from .builder import PageTranslationBuilder, build_page_translation
from .schema import PAGETRANSLATE_SCHEMA_VERSION

__all__ = [
    "PAGETRANSLATE_SCHEMA_VERSION",
    "PageTranslationBuilder",
    "build_page_translation",
]
