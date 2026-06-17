"""Entrée unique de l'évaluateur publication-ready.

Normalise l'input traduit (via PageReconstructInputAdapter) puis lance
evaluate_page sur le plan rendu. Additif : ne touche pas le chemin de rendu.
"""

from __future__ import annotations

from .page_auditor import evaluate_page
from .document_auditor import evaluate_document


def evaluate_reconstruction(translated_input_data: dict, plan: dict, *,
                            page_id: str = "", page_index: int = 0,
                            mode: str = "publication",
                            source_image_path: str | None = None,
                            reconstructed_image_path: str | None = None,
                            out_dir: str | None = None,
                            normalized: dict | None = None):
    """tid -> normalized -> evaluate_page. Si `normalized` fourni, on le réutilise."""
    if normalized is None:
        from pagereconstruct.input_adapter import PageReconstructInputAdapter
        normalized = PageReconstructInputAdapter().normalize(translated_input_data)
    return evaluate_page(
        plan, normalized, page_id=page_id, page_index=page_index, mode=mode,
        source_image_path=source_image_path,
        reconstructed_image_path=reconstructed_image_path, out_dir=out_dir,
    )


__all__ = ["evaluate_reconstruction", "evaluate_document"]
