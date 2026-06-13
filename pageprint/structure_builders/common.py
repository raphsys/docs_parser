"""Shared helpers for PAGEPRINT structure builders."""

from __future__ import annotations


TEXT_LEVELS = {"block", "line", "phrase", "span", "cell"}


def text_of(unit: dict) -> str:
    return str((unit.get("content") or {}).get("text") or "").strip()


def role_of(unit: dict) -> str:
    return str((unit.get("understanding") or {}).get("role") or "unknown")


def bbox_of(unit: dict):
    return (unit.get("geometry") or {}).get("bbox")


def reading_order(unit: dict) -> float:
    return float((unit.get("geometry") or {}).get("reading_order_index") or 0)


def eligible_text_units(units: list[dict]) -> list[dict]:
    return sorted(
        [u for u in units if isinstance(u, dict) and u.get("level") in TEXT_LEVELS and text_of(u)],
        key=reading_order,
    )


def bbox_union(bboxes: list[object]) -> list[float] | None:
    clean = [b for b in bboxes if isinstance(b, (list, tuple)) and len(b) == 4]
    if not clean:
        return None
    return [
        min(float(b[0]) for b in clean),
        min(float(b[1]) for b in clean),
        max(float(b[2]) for b in clean),
        max(float(b[3]) for b in clean),
    ]
