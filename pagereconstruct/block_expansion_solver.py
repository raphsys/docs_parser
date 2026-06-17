"""Block expansion + neighbour reflow solver, v2.

This file keeps the public API used by plan_compiler:
    solve_block_expansion(contract, enabled=True)

Internally it now delegates to flow_geometry_optimizer, which handles:
    - atomic line blocks created by text-survival patches;
    - width expansion when safe;
    - vertical block growth;
    - cascade of adjacent blocks;
    - protected-object avoidance;
    - AI/model-aware policy hints from ai_models/.
"""

from __future__ import annotations

from typing import Any

from .flow_geometry_optimizer import (
    solve_flow_geometry,
    apply_flow_geometry_patches_in_place,
    _style_for_measure,
    _needed_height,
)



def _page_bottom_limit(contract: Any, margin: float = 0.0) -> float | None:
    # Return the hard page bottom used to cap expansion patches.
    for obj_name in ("page_info", "page"):
        page = getattr(contract, obj_name, None)
        if page is None:
            continue
        ps = getattr(page, "page_size", None)
        if isinstance(ps, (list, tuple)) and len(ps) == 2 and ps[1]:
            return max(1.0, float(ps[1]) - margin)
        for attr in ("height_pt", "height"):
            v = getattr(page, attr, None)
            if v:
                return max(1.0, float(v) - margin)
    return None


def _cap_result_patches_to_page_bottom(contract: Any, result):
    bottom = _page_bottom_limit(contract, margin=0.0)
    if bottom is None:
        return result
    for patch in (getattr(result, "patches_by_block_id", {}) or {}).values():
        nb = getattr(patch, "new_bbox", None)
        if not (isinstance(nb, (list, tuple)) and len(nb) == 4):
            continue
        x0, y0, x1, y1 = [float(x) for x in nb]
        if y1 <= bottom and y1 > y0:
            continue
        old = getattr(patch, "old_bbox", None)
        old_h = 0.0
        if isinstance(old, (list, tuple)) and len(old) == 4:
            old_h = max(1.0, float(old[3]) - float(old[1]))
        if y0 >= bottom:
            y0 = max(0.0, bottom - max(1.0, old_h))
        max_h = max(1.0, bottom - y0)
        new_h = min(max(1.0, old_h or max_h), max_h)
        capped = (x0, y0, x1, min(bottom, y0 + new_h))
        if capped[3] <= capped[1]:
            capped = (x0, max(0.0, bottom - 1.0), x1, bottom)
        try:
            patch.new_bbox = capped
            patch.findings.append({
                "type": "block_expansion_patch_capped_to_page_bottom",
                "severity": "review",
                "old_new_bbox": list(nb),
                "capped_new_bbox": list(capped),
            })
        except Exception:
            pass
    return result


def solve_block_expansion(contract: Any, *, enabled: bool = True, page_margin_pt: float = 6.0):
    # page_margin_pt is kept for backward compatibility. The v2 solver computes
    # its own safe page bounds from the FinalReconstructionContract.
    result = solve_flow_geometry(contract, enabled=enabled)
    return _cap_result_patches_to_page_bottom(contract, result)


# Backward-compatible alias for callers that expect an apply function here.
apply_block_expansion_patches_in_place = apply_flow_geometry_patches_in_place
