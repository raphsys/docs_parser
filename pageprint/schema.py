"""PAGEPRINT Schema — contrat canonique de INPUT_DATA (pageprint.input.v1).

PAGEPRINT est la première tête du pipeline WYSIWYG. Son rôle est de
transformer une page source en une représentation canonique, normalisée,
enrichie, vérifiable et consommable par tous les modules aval :
traduction, reconstruction, mise en forme, optimisation de layout, QA et export.

PAGEPRINT = le générateur de INPUT_DATA
INPUT_DATA = la sortie canonique de PAGEPRINT

Ce module définit le contrat. Le contrat est plus important que l'algorithme.
"""

from __future__ import annotations

PAGEPRINT_SCHEMA_VERSION = "pageprint.input.v1"

CANONICAL_UNIT = "pt"
POINTS_PER_INCH = 72.0

# Seuils de signifiance des objets visuels (images/dessins).
# En dessous, l'objet est un trait décoratif (filet de bordure, règle) :
# il reste préservé par la couche background et dans compatibility.legacy,
# mais ne devient ni unité canonique ni région.
MIN_VISUAL_DIM_PT = 2.0
MIN_VISUAL_AREA_PT2 = 9.0

# Clés de premier niveau obligatoires de la V1 (contrat figé).
INPUT_DATA_V1_REQUIRED = {
    "schema_version",
    "document",
    "page",
    "assets",
    "units",
    "regions",
    "page_intelligence",
    "style_system",
    "relations",
    "policies",
    "translation_context",
    "reconstruction_constraints",
    "quality",
    "provenance",
}

REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version",
    "document",
    "page",
    "assets",
    "units",
    "regions",
    "relations",
    "page_intelligence",
    "policies",
    "constraints",
    "quality",
    "provenance",
}

# Contrat obligatoire de chaque unité canonique.
UNIT_REQUIRED = {
    "unit_id",
    "level",
    "parent_id",
    "children_ids",
    "content",
    "geometry",
    "visual",
    "extraction",
    "understanding",
    "policy",
    "relations",
    "constraints",
    "provenance",
}

# Niveaux d'unités canoniques.
UNIT_LEVELS = {
    "page",
    "region",
    "block",
    "line",
    "phrase",
    "span",
    "word",
    "char",
    "image",
    "drawing",
    "table",
    "cell",
    "formula",
    "code",
    "overlay",
}

# Relations essentielles du graphe documentaire.
RELATION_TYPES = {
    "contains",
    "belongs_to",
    "flows_to",
    "continues",
    "same_paragraph",
    "same_list",
    "same_table",
    "same_row",
    "same_column",
    "caption_of",
    "label_of",
    "legend_of",
    "axis_of",
    "annotates",
    "overlaps",
    "near",
    "aligned_with",
    "style_similar_to",
}

# Types de régions reconnus par le Region Index.
REGION_TYPES = {
    "body_region",
    "header_region",
    "footer_region",
    "table_region",
    "table_cell",
    "figure_region",
    "chart_region",
    "chart_tick",
    "formula_region",
    "code_region",
    "caption_region",
    "annotation_region",
    "toc_region",
    "background_region",
    "non_text_zone",
    "image_region",
    "drawing_region",
}

# V1 non obligatoire mais prévue (innovations documentées) :
OPTIONAL_V1_KEYS = {
    "semantic_pressure",
    "translation_forecast",
    "visual_anchor_map",
    "audit_overlay",
    "replay",
}


def empty_input_data() -> dict:
    """Squelette du schéma final figé (V1).

    Source de vérité = INPUT_DATA. Les vues legacy (compatibility)
    sont des exports temporaires dérivés, pas la source de vérité.
    """
    return {
        "schema_version": PAGEPRINT_SCHEMA_VERSION,
        "input_id": None,

        "document": {},
        "page": {},

        "assets": {},
        "visual_layers": {},

        "units": [],
        "regions": [],
        "graph": {"nodes": [], "edges": []},
        "relations": {},

        "page_intelligence": {},
        "style_system": {},
        "semantic_system": {},

        "policies": {},
        "constraints": {},
        "reconstruction_constraints": {},

        "translation_context": {},
        "reconstruction_context": {},

        "views": {
            "hierarchical": {},
            "flat_units": [],
            "translation_units": [],
            "render_units": [],
            "debug_units": [],
        },

        "quality": {},
        "risks": {},
        "provenance": {},
        "debug": {},
        "compatibility": {},
    }


def empty_unit(unit_id: str, level: str, parent_id: str | None = None) -> dict:
    """Unité canonique minimale respectant UNIT_REQUIRED."""
    return {
        "unit_id": unit_id,
        "level": level,
        "parent_id": parent_id,
        "children_ids": [],
        "content": {
            "text": None,
            "raw_text": None,
            "normalized_text": None,
            "translated_text": None,
            "language": None,
        },
        "geometry": {
            "bbox": None,
            "bbox_unit": CANONICAL_UNIT,
            "bbox_origin": "top_left",
            "bbox_px": None,
            "bbox_px_dpi": None,
            "baseline": None,
            "rotation": 0,
            "reading_order_index": None,
            "render_order_index": None,
        },
        "visual": {
            "style": {},
            "style_class": None,
            "style_confidence": None,
        },
        "extraction": {
            "source": None,
            "source_kind": None,
            "confidence": None,
            "ocr_confidence_mean": None,
            "native_confidence": None,
            "dedupe_status": "kept",
        },
        "understanding": {
            "role": None,
            "object_type": None,
            "object_class": None,
            "page_family": None,
            "layout_type": None,
            "document_type": None,
            "region_memberships": [],
            "structure_hints": {},
            "semantic_kind": None,
        },
        "policy": {
            "translatable": None,
            "translation_strategy": None,
            "render_policy": None,
            "coverage_required": None,
            "preserve_exact_text": False,
            "preserve_visual": False,
        },
        "relations": {
            "previous_unit_id": None,
            "next_unit_id": None,
            "parent_region_id": None,
            "flow_to_next": {},
            "flow_from_previous": {},
        },
        "constraints": {},
        "provenance": {
            "created_by": None,
            "postprocessed_by": [],
            "decision_trace": [],
        },
        "lifecycle": {
            "created_at_stage": None,
            "current_stage": "pageprint_final",
            "status": "active",
            "merged_from": [],
            "split_from": None,
            "superseded_by": None,
            "discard_reason": None,
        },
        "confidence": {
            "text": None,
            "bbox": None,
            "style": None,
            "role": None,
            "reading_order": None,
            "region_membership": None,
            "translation_policy": None,
            "overall": None,
        },
        "evidence": {},
    }
