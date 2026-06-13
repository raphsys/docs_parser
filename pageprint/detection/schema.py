"""Schema constants for PagePrint PAGE_REGION_DETECT."""

PAGE_REGION_DETECT_SCHEMA_VERSION = "pageprint.region_detect.v1"

PROTECTED_SPECIAL_CLASSES = {
    "formula",
    "formula_region",
    "equation",
    "math_expression",
    "chemical_formula",
    "symbolic_expression",
    "code",
    "code_region",
    "code_block",
    "inline_code",
    "algorithm_block",
    "special_notation",
    "table_formula_cell",
    "diagram_label_non_linguistic",
    "logo",
    "signature",
    "stamp",
    "barcode",
    "qr_code",
    "protected_visual",
    "protected_visual_region",
}
