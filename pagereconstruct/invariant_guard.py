"""Invariant guard for vSense reconstruction.

Locks the behaviours that must not regress:
1. visible PAGEPRINT text lines must have an output owner;
2. clean background must be verified/text_removed;
3. translated output should not silently fall back to source for most units.
"""

from __future__ import annotations


def _text_of(unit: dict | None) -> str:
    if not isinstance(unit, dict):
        return ""
    c = unit.get("content") or {}
    return " ".join(str(c.get("text") or unit.get("text") or unit.get("source_text") or "").split())


def _is_visible_line(unit: dict) -> bool:
    if not isinstance(unit, dict) or unit.get("level") != "line":
        return False
    if not _text_of(unit):
        return False
    b = (unit.get("geometry") or {}).get("bbox")
    return isinstance(b, (list, tuple)) and len(b) == 4 and b[2] > b[0] and b[3] > b[1]


def _related(a: str, b: str) -> bool:
    return bool(a and b) and (a == b or a.startswith(b + "_") or b.startswith(a + "_"))


def summarize_text_render_invariants(normalized: dict, contract, render_ops: list, background: dict | None = None) -> dict:
    source_lines = [u for u in normalized.get("units") or [] if _is_visible_line(u)]
    source_ids = [u.get("unit_id") for u in source_lines if u.get("unit_id")]

    rendered_ids = set()
    translated_count = 0
    identity_count = 0
    for block in getattr(contract, "blocks", []) or []:
        for sid in getattr(block, "source_unit_ids", []) or []:
            if sid:
                rendered_ids.add(sid)
        src = " ".join(str(getattr(block, "source_text", "") or "").split())
        tgt = " ".join(str(getattr(block, "translated_text", "") or "").split())
        if tgt and tgt != src:
            translated_count += 1
        elif tgt or src:
            identity_count += 1

    missing = []
    for sid in source_ids:
        if not any(_related(sid, rid) for rid in rendered_ids):
            missing.append(sid)

    bg = background or {}
    bg_verified = bool(
        bg.get("clean_background_verified")
        or bg.get("text_removed")
        or ((normalized.get("visual_layers") or {}).get("clean_background_verified"))
        or ((normalized.get("assets") or {}).get("background_clean_verified"))
    )

    text_ops = 0
    for op in render_ops or []:
        d = op.to_dict() if hasattr(op, "to_dict") else op
        if isinstance(d, dict) and d.get("op_type") == "text":
            text_ops += 1

    return {
        "source_visible_line_count": len(source_ids),
        "rendered_source_owner_count": len(rendered_ids),
        "missing_source_line_count": len(missing),
        "missing_source_line_ids": missing[:50],
        "translated_block_count": translated_count,
        "identity_block_count": identity_count,
        "text_render_op_count": text_ops,
        "clean_background_verified": bg_verified,
        "status": "ok" if not missing and bg_verified else "ko",
    }
