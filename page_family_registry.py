PAGE_FAMILY_REGISTRY = {
    "toc": {
        "group": "toc",
        "translation_style": "professionnel",
        "translation_tone": "neutre",
        "description": "Sommaire et pages assimilées TOC.",
    },
    "body_text": {
        "group": "body_text",
        "translation_style": "professionnel",
        "translation_tone": "neutre",
        "description": "Texte courant générique.",
    },
    "body_text_two_column": {
        "group": "body_text",
        "translation_style": "professionnel",
        "translation_tone": "analytique",
        "description": "Texte courant en deux colonnes.",
    },
    "body_text_two_column_sectioned": {
        "group": "body_text",
        "translation_style": "professionnel",
        "translation_tone": "analytique",
        "description": "Texte en deux colonnes avec intertitres.",
    },
    "body_text_two_column_equations": {
        "group": "body_text",
        "translation_style": "technique",
        "translation_tone": "analytique",
        "description": "Texte en deux colonnes avec équations/expressions techniques.",
    },
    "body_text_single_column_sparse": {
        "group": "body_text",
        "translation_style": "professionnel",
        "translation_tone": "neutre",
        "description": "Page texte légère, simple colonne, structure peu dense.",
    },
    "body_with_figure": {
        "group": "body_with_figure",
        "translation_style": "technique",
        "translation_tone": "didactique",
        "description": "Page texte + figure + légende.",
    },
    "body_with_diagram": {
        "group": "body_with_diagram",
        "translation_style": "technique",
        "translation_tone": "didactique",
        "description": "Page texte + diagramme/labels.",
    },
    "illustrated_label_page": {
        "group": "body_with_diagram",
        "translation_style": "technique",
        "translation_tone": "didactique",
        "description": "Page centrée sur un schéma illustré avec labels courts distribués autour du visuel.",
    },
    "chart_label_page": {
        "group": "body_with_figure",
        "translation_style": "technique",
        "translation_tone": "didactique",
        "description": "Page de graphique ou histogramme avec labels de catégories et annotations courtes.",
    },
    "table_page": {
        "group": "table_page",
        "translation_style": "technique",
        "translation_tone": "neutre",
        "description": "Page principalement tabulaire.",
    },
    "table_diagram_example": {
        "group": "table_page",
        "translation_style": "technique",
        "translation_tone": "didactique",
        "description": "Page hybride tableau/diagramme avec labels courts, formules d'exemple et fragments de code visibles.",
    },
    "mixed_page": {
        "group": "mixed_page",
        "translation_style": "professionnel",
        "translation_tone": "analytique",
        "description": "Page mixte texte + éléments non textuels variés.",
    },
    "mixed_dense_illustrated": {
        "group": "mixed_page",
        "translation_style": "technique",
        "translation_tone": "didactique",
        "description": "Page mixte dense avec nombreuses zones illustrées et labels courts.",
    },
    "mixed_formula_annotation_page": {
        "group": "mixed_page",
        "translation_style": "technique",
        "translation_tone": "didactique",
        "description": "Page mixte dense avec formules, annotations courtes et texte explicatif.",
    },
    "narrative_reference_page": {
        "group": "body_text",
        "translation_style": "professionnel",
        "translation_tone": "neutre",
        "description": "Page narrative contenant URL, liens ou références éditoriales visibles.",
    },
    "citation_heavy_body_page": {
        "group": "body_text",
        "translation_style": "technique",
        "translation_tone": "analytique",
        "description": "Page de corps avec citations bibliographiques ou sources longues intégrées au texte.",
    },
    "unknown": {
        "group": "unknown",
        "translation_style": "professionnel",
        "translation_tone": "analytique",
        "description": "Famille inconnue, fallback prudent.",
    },
}


def get_family_config(page_family):
    family = str(page_family or "unknown").strip().lower()
    return PAGE_FAMILY_REGISTRY.get(family, PAGE_FAMILY_REGISTRY["unknown"])


def get_family_group(page_family):
    return str(get_family_config(page_family).get("group") or "unknown")
