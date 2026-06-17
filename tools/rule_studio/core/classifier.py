"""Classification d'une règle dans une unité de pipeline.

Stratégie : pondération de signaux issus du chemin de fichier, du nom de
symbole et du code/texte. L'unité au plus haut score gagne ; les suivantes au
dessus d'un seuil deviennent unités secondaires. La classification reste
heuristique et alimente `confidence`.
"""

from __future__ import annotations

import re

from .models import PIPELINE_UNITS

# Signaux par unité : (regex, poids). Appliqués sur le chemin et le texte.
# L'ordre n'a pas d'importance ; on somme les poids.
_UNIT_SIGNALS: dict[str, list[tuple[str, float]]] = {
    "INPUT_DATA": [(r"pageprint", 1.0), (r"input_?data", 1.2), (r"normalizer", 0.8)],
    "PAGE_CLASSIFICATION": [
        (r"page_?case_?classifier", 1.5),
        (r"page_?classif", 1.5),
        (r"page_?family", 1.0),
        (r"page_?profile", 0.8),
        (r"\bcover\b", 0.6),
        (r"page_role", 0.6),
    ],
    "PAGE_DETECTION": [(r"page_?detect", 1.2), (r"detect_page", 1.0)],
    "DOCUMENT_COMPREHENSION": [
        (r"document_?comprehension", 1.5),
        (r"context_classifier", 1.0),
        (r"page_understanding", 1.0),
        (r"document_context", 0.8),
    ],
    "SEGMENTATION": [
        (r"segment", 1.2),
        (r"sentence_boundary", 1.2),
        (r"coalesc", 0.8),
        (r"logical_block", 0.8),
    ],
    "REGION_DETECTION": [
        (r"region", 1.0),
        (r"special_region_detector", 1.5),
        (r"region_index", 1.2),
    ],
    "UNIT_FACTORY": [(r"unit_?factory", 1.6)],
    "STYLE_EXTRACTION": [
        (r"style_profiler", 1.4),
        (r"style_resolver", 1.2),
        (r"style_contract", 1.0),
        (r"\bstyle\b", 0.5),
    ],
    "SEMANTIC_ROLE_RESOLUTION": [
        (r"role_?resolver", 1.6),
        (r"semantic_builder", 1.2),
        (r"\brole\b", 0.5),
        (r"semantic", 0.6),
    ],
    "POLICY_COMPILATION": [
        (r"policy_?compiler", 1.6),
        (r"page_policy_matrix", 1.2),
        (r"positioning_policy", 1.0),
        (r"\bpolicy\b", 0.6),
        (r"render_policy", 0.8),
    ],
    "CONSTRAINT_COMPILATION": [
        (r"constraint_?compiler", 1.6),
        (r"\bconstraint", 0.7),
    ],
    "TRANSLATION": [
        (r"pagetranslate", 1.4),
        (r"translat", 0.8),
        (r"terminology", 0.8),
        (r"translation_memory", 1.0),
    ],
    "PROTECTED_CONTENT": [
        (r"protect", 1.0),
        (r"protected_region", 1.4),
        (r"technical_protection", 1.4),
        (r"non.?translat", 0.8),
        (r"\bformula\b", 0.6),
    ],
    "RECONSTRUCTION": [
        (r"pagereconstruct", 1.4),
        (r"reconstruct", 1.0),
        (r"plan_compiler", 1.0),
        (r"patch_planner", 1.0),
        (r"render_ops", 0.8),
    ],
    "LAYOUT_SOLVER": [
        (r"layout_?solver", 1.6),
        (r"placement_solver", 1.4),
        (r"multiblock_layout", 1.4),
        (r"layout_box_resolver", 1.0),
        (r"\blayout\b", 0.5),
        (r"overflow", 0.6),
        (r"\bbbox\b", 0.5),
        (r"\banchor", 0.6),
    ],
    "TYPOGRAPHY": [
        (r"font_resolver", 1.2),
        (r"typograph", 1.4),
        (r"ocr_typography", 1.4),
        (r"text_measure", 1.0),
        (r"\bfont\b", 0.5),
    ],
    "QA": [
        (r"publication_qa", 1.4),
        (r"quality_assessor", 1.4),
        (r"quality_contract", 1.2),
        (r"visual_qa", 1.4),
        (r"coverage_validator", 1.2),
        (r"functional_validator", 1.2),
        (r"\bvalidator", 0.7),
        (r"\bvalidate", 0.5),
        (r"quality", 0.5),
    ],
    "EXPORT": [
        (r"html_exporter", 1.4),
        (r"serializ", 0.8),
        (r"exporter", 0.8),
        (r"pdf_vector", 0.8),
    ],
    "LEGACY_COMPATIBILITY": [
        (r"legacy", 1.4),
        (r"_v2\b", 0.6),
        (r"_v3\b", 0.6),
        (r"compat", 0.8),
    ],
    "DEBUG": [(r"raster_debug", 1.2), (r"\bdebug\b", 0.6), (r"audit", 0.5)],
    "GLOBAL": [(r"runtime_config", 1.0), (r"\bschema\b", 0.4)],
}

# Pré-compilation
_COMPILED: dict[str, list[tuple[re.Pattern[str], float]]] = {
    unit: [(re.compile(pat, re.IGNORECASE), w) for pat, w in sigs]
    for unit, sigs in _UNIT_SIGNALS.items()
}

_SECONDARY_THRESHOLD = 0.6


def classify(
    file_path: str,
    symbol_name: str | None = None,
    text: str | None = None,
) -> tuple[str, list[str], float]:
    """Renvoie (unité_principale, unités_secondaires, score_confiance).

    Le chemin et le symbole pèsent davantage que le corps de code (plus bruité).
    """
    scores: dict[str, float] = {u: 0.0 for u in PIPELINE_UNITS}

    path_blob = file_path.lower()
    sym_blob = (symbol_name or "").lower()
    body_blob = (text or "")[:4000].lower()  # plafonné pour la perf

    for unit, sigs in _COMPILED.items():
        for rx, weight in sigs:
            if rx.search(path_blob):
                scores[unit] += weight * 1.0
            if sym_blob and rx.search(sym_blob):
                scores[unit] += weight * 0.8
            if body_blob and rx.search(body_blob):
                scores[unit] += weight * 0.4

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_unit, best_score = ranked[0]

    if best_score <= 0:
        return "UNKNOWN", [], 0.0

    secondary = [u for u, s in ranked[1:] if s >= _SECONDARY_THRESHOLD][:3]

    # confiance bornée [0,1] : ~0.5 pour un signal franc, sature vers 1.
    confidence = min(1.0, best_score / 2.5)
    return best_unit, secondary, round(confidence, 3)
