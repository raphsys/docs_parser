from __future__ import annotations

from .common import bbox_of


def build_figures(
    units: list[dict],
    *,
    captions: list[dict] | None = None,
    page_intelligence: dict | None = None,
) -> list[dict]:
    figures = []
    visual_units = [
        unit for unit in units
        if isinstance(unit, dict)
        and unit.get("level") in {"image", "drawing", "region"}
        and _looks_like_figure(unit)
    ]
    for idx, unit in enumerate(visual_units, start=1):
        figures.append({
            "logical_unit_id": f"figure_{idx:04d}",
            "figure_id": f"figure_{idx:04d}",
            "type": "figure",
            "source_unit_ids": [unit.get("unit_id")],
            "bbox": bbox_of(unit),
            "caption_ids": _nearby_caption_ids(unit, captions or []),
            "diagram_labels": _diagram_labels(unit),
            "translation_policy": "translate_caption_and_natural_labels_only",
        })
    return figures


def _looks_like_figure(unit: dict) -> bool:
    understanding = unit.get("understanding") or {}
    role = str(understanding.get("role") or "").lower()
    object_type = str(understanding.get("object_type") or "").lower()
    region_type = str(unit.get("region_type") or unit.get("type") or "").lower()
    return any(token in f"{role} {object_type} {region_type}" for token in ("figure", "diagram", "image"))


def _nearby_caption_ids(unit: dict, captions: list[dict]) -> list[str]:
    bbox = bbox_of(unit)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return []
    ids = []
    for caption in captions:
        cb = caption.get("bbox")
        if not isinstance(cb, (list, tuple)) or len(cb) != 4:
            continue
        vertical_gap = min(abs(float(cb[1]) - float(bbox[3])), abs(float(bbox[1]) - float(cb[3])))
        horizontal_overlap = min(float(bbox[2]), float(cb[2])) - max(float(bbox[0]), float(cb[0]))
        if vertical_gap <= 80 and horizontal_overlap > 0:
            ids.append(caption.get("caption_id") or caption.get("logical_unit_id"))
    return [cid for cid in ids if cid]


def _diagram_labels(unit: dict) -> list[dict]:
    labels = []
    for label in (unit.get("diagram_labels") or []):
        if not isinstance(label, dict):
            continue
        text = str(label.get("text") or "").strip()
        if not text:
            continue
        labels.append({
            "text": text,
            "bbox": label.get("bbox"),
            "label_kind": "technical" if text in {"ReLU", "Softmax", "Conv", "FC"} else "natural_label",
            "translation_mode": "preserve_text_exactly" if text in {"ReLU", "Softmax", "Conv", "FC"} else "translate",
        })
    return labels
