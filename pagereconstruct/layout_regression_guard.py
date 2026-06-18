"""Regression guard for text-safe layout.

This guard exists because geometry/layout optimizers may be exploratory.
They must never break the locked guarantees:
    - every visible source line has an output owner;
    - translated blocks keep sane bboxes;
    - layout boxes must never become inverted or jump absurdly across the page.

The guard is deliberately conservative. If a layout bbox is invalid, outside the
page, or shifted too far compared with the original anchor/source bbox, it is
restored to a sane source/coverage bbox. This preserves text presence and keeps
the renderer from drawing blocks at the wrong end of the page.
"""

from __future__ import annotations

from typing import Any, Iterable


def _bbox(v):
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            b = [float(x) for x in v]
            if all(x == x for x in b):  # no NaN
                return b
        except Exception:
            return None
    return None


def _valid_bbox(b) -> bool:
    b = _bbox(b)
    return bool(b and b[2] > b[0] and b[3] > b[1])


def _height(b) -> float:
    b = _bbox(b) or [0, 0, 0, 0]
    return max(0.0, b[3] - b[1])


def _width(b) -> float:
    b = _bbox(b) or [0, 0, 0, 0]
    return max(0.0, b[2] - b[0])


def _center_y(b) -> float:
    b = _bbox(b) or [0, 0, 0, 0]
    return (b[1] + b[3]) / 2.0


def _center_x(b) -> float:
    b = _bbox(b) or [0, 0, 0, 0]
    return (b[0] + b[2]) / 2.0


def _page_size(contract: Any) -> tuple[float, float]:
    pi = getattr(contract, "page_info", None)
    ps = getattr(pi, "page_size", None)
    if isinstance(ps, (list, tuple)) and len(ps) == 2 and ps[0] and ps[1]:
        return float(ps[0]), float(ps[1])
    return 595.0, 842.0


def _layout(block: Any):
    return getattr(block, "layout", None)


def _block_id(block: Any) -> str:
    return str(getattr(block, "block_id", "") or "")


def _text(block: Any) -> str:
    return " ".join(str(getattr(block, "translated_text", "") or getattr(block, "source_text", "") or "").split())


def _role(block: Any) -> str:
    return str(getattr(block, "role", "") or "")


def _locked_role(role: str) -> bool:
    return role in {
        "page_number", "page_reference", "formula", "formula_expression", "equation",
        "code", "code_line", "code_block", "table_body_cell", "table_header_cell",
        "table_numeric_cell", "diagram_label", "caption_label", "caption_number",
        "publisher_mark", "watermark", "exclude_as_artifact",
    }


def _best_reference_bbox(layout: Any):
    for attr in ("source_bbox", "coverage_bbox", "anchor_bbox", "patch_bbox", "layout_bbox"):
        b = _bbox(getattr(layout, attr, None))
        if _valid_bbox(b):
            return b
    return None


def _current_bbox(layout: Any):
    for attr in ("layout_bbox", "safe_bbox", "overflow_bbox", "coverage_bbox", "source_bbox"):
        b = _bbox(getattr(layout, attr, None))
        if b:
            return b
    return None


def _set_layout_bbox(layout: Any, b: list[float]) -> None:
    nb = [float(x) for x in b]
    if hasattr(layout, "layout_bbox"):
        setattr(layout, "layout_bbox", list(nb))
    if hasattr(layout, "safe_bbox"):
        setattr(layout, "safe_bbox", list(nb))
    if hasattr(layout, "overflow_bbox"):
        setattr(layout, "overflow_bbox", list(nb))


def _is_dangerous_shift(cur: list[float], ref: list[float], *, role: str, page_h: float) -> tuple[bool, str]:
    if not _valid_bbox(cur):
        return True, "invalid_bbox"
    if cur[1] < -2 or cur[3] > page_h + 2:
        return True, "outside_page"
    if _height(cur) <= 0.25 or _width(cur) <= 0.25:
        return True, "near_zero_bbox"

    # Locked/small blocks must not be globally repositioned.
    dy = abs(_center_y(cur) - _center_y(ref))
    dx = abs(_center_x(cur) - _center_x(ref))
    ref_h = max(1.0, _height(ref))
    if _locked_role(role) and (dy > 2.0 or dx > 8.0):
        return True, "locked_role_shift"

    # Normal flow blocks can move a little, but not jump from top to bottom.
    if dy > max(64.0, ref_h * 5.0):
        return True, "excessive_vertical_shift"

    # An expansion may increase height, but inverted bottom-clamped boxes are not allowed.
    if cur[3] <= cur[1]:
        return True, "inverted_bbox"

    return False, ""


def sanitize_contract_layouts_in_place(contract: Any, *, findings: list | None = None, render_policy: dict | None = None) -> dict:
    """Restore impossible bboxes produced by an unsafe layout optimizer."""
    findings = findings if findings is not None else []
    render_policy = render_policy if render_policy is not None else {}
    _, page_h = _page_size(contract)

    fixed = 0
    checked = 0
    for block in getattr(contract, "blocks", []) or []:
        if not _text(block):
            continue
        layout = _layout(block)
        if layout is None:
            continue
        ref = _best_reference_bbox(layout)
        cur = _current_bbox(layout)
        if not ref or not cur:
            continue
        checked += 1
        bad, reason = _is_dangerous_shift(cur, ref, role=_role(block), page_h=page_h)
        if not bad and getattr(layout, "allow_local_shift", False) is not True:
            dy = abs(_center_y(cur) - _center_y(ref))
            dx = abs(_center_x(cur) - _center_x(ref))
            if dy > 2.0 or dx > 2.0:
                bad, reason = True, "source_anchor_shift_without_permission"
        if not bad:
            continue
        _set_layout_bbox(layout, ref)
        fixed += 1
        findings.append({
            "type": "layout_regression_guard_restored_bbox",
            "severity": "review",
            "block_id": _block_id(block),
            "role": _role(block),
            "reason": reason,
            "bad_bbox": list(cur),
            "restored_bbox": list(ref),
        })

    render_policy["layout_regression_guard_checked"] = checked
    render_policy["layout_regression_guard_fixed"] = fixed
    return {"checked": checked, "fixed": fixed}
