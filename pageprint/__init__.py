"""PAGEPRINT — Canonical Page Intelligence Input.

Première tête du pipeline WYSIWYG. Transforme une page source en une
représentation canonique, normalisée, enrichie, vérifiable et consommable
par tous les modules aval : traduction, reconstruction, mise en forme,
optimisation de layout, QA et export.

    PDF / IMAGE / DOCX / PPTX
            ↓
         PAGEPRINT
            ↓
      INPUT_DATA canonique
            ↓
    TRADUCTION / RECONSTRUCTION / MISE_EN_FORME / QA
"""

from .builder import PagePrintBuilder, build_pageprint_input_data
from .schema import (
    CANONICAL_UNIT,
    INPUT_DATA_V1_REQUIRED,
    PAGEPRINT_SCHEMA_VERSION,
    UNIT_LEVELS,
    UNIT_REQUIRED,
    empty_input_data,
    empty_unit,
)
from .serializers import load_input_data, save_input_data, to_json
from .validators import validate_input_data

__all__ = [
    "PagePrintBuilder",
    "build_pageprint_input_data",
    "PAGEPRINT_SCHEMA_VERSION",
    "CANONICAL_UNIT",
    "INPUT_DATA_V1_REQUIRED",
    "UNIT_REQUIRED",
    "UNIT_LEVELS",
    "empty_input_data",
    "empty_unit",
    "validate_input_data",
    "to_json",
    "save_input_data",
    "load_input_data",
]
