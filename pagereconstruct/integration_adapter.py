"""Integration adapter for the two external 95%-unlock modules.

Call this after FinalReconstructionContract creation and before build_render_ops.
The adapter is intentionally optional and fail-safe.
"""
from __future__ import annotations

from typing import Any, Dict

from .ocr_typography_engine import enhance_contract_typography, apply_typography_patches_in_place
from .multiblock_layout_solver import solve_multiblock_layout, apply_layout_patches_in_place


def enhance_contract_for_publication(contract: Any, *, pageprint_data: dict | None = None,
                                     page_image_path: str | None = None,
                                     enable_typography: bool = True,
                                     enable_multiblock: bool = True,
                                     mutate: bool = True) -> tuple[Any, Dict[str, Any]]:
    report: Dict[str, Any] = {"typography": None, "multiblock": None, "findings": []}
    target = contract

    if enable_typography:
        try:
            typ = enhance_contract_typography(target, pageprint_data=pageprint_data, page_image_path=page_image_path)
            report["typography"] = typ.to_dict()
            if mutate:
                target = apply_typography_patches_in_place(target, typ)
        except Exception as exc:  # fail-safe: never break existing reconstruction
            report["findings"].append({"type": "external_typography_engine_failed", "error": repr(exc), "severity": "review"})

    if enable_multiblock:
        try:
            mb = solve_multiblock_layout(target, enabled=True)
            report["multiblock"] = mb.to_dict()
            if mutate and mb.status != "ko":
                target = apply_layout_patches_in_place(target, mb)
        except Exception as exc:
            report["findings"].append({"type": "external_multiblock_solver_failed", "error": repr(exc), "severity": "review"})

    if hasattr(target, "findings"):
        target.findings.extend(report["findings"])
    return target, report
