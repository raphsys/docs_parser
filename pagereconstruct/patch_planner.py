"""Plan explicit patch (cleanup) zones for translated text (directive §14).

Each translated text unit gets a patch zone over the source-text area. Patches
are refused/flagged where they would erase a hard protected region (formula,
image, logo…). The backend executes only declared patches — it never invents
whiteouts.
"""

from __future__ import annotations

from .schema import PatchZone


def _overlap_ratio(a, b) -> float:
    if not (a and b and len(a) == 4 and len(b) == 4):
        return 0.0
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area




def _expand_patch_bbox(bbox, x_pad: float = 2.2, y_pad: float = 1.1):
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return bbox
    return [float(bbox[0]) - x_pad, float(bbox[1]) - y_pad,
            float(bbox[2]) + x_pad, float(bbox[3]) + y_pad]

def _inter(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def _subtract(rect, hole):
    """Rectangle subtraction (guillotine): rect minus hole -> up to 4 sub-rects.
    Removes the protected area from the patch so a patch never erases it."""
    rx0, ry0, rx1, ry1 = rect
    hx0, hy0, hx1, hy1 = hole
    if _inter(rect, hole) <= 0:
        return [rect]
    out = []
    if hy0 > ry0:                       # top band
        out.append([rx0, ry0, rx1, hy0])
    if hy1 < ry1:                       # bottom band
        out.append([rx0, hy1, rx1, ry1])
    my0, my1 = max(ry0, hy0), min(ry1, hy1)
    if hx0 > rx0:                       # left band
        out.append([rx0, my0, hx0, my1])
    if hx1 < rx1:                       # right band
        out.append([hx1, my0, rx1, my1])
    return [r for r in out if r[2] - r[0] > 1.0 and r[3] - r[1] > 1.0]


def _expand_bbox(bbox, *, x_pad: float = 1.8, y_pad: float = 0.9) -> list[float]:
    x0, y0, x1, y1 = [float(x) for x in bbox]
    return [x0 - x_pad, y0 - y_pad, x1 + x_pad, y1 + y_pad]



def plan_patches(translated_units, protected_index) -> tuple[list, list]:
    """Return (patches, findings). A patch that overlaps a hard protected region
    is CLIPPED around it (split into protected-free sub-rects), never painted
    over it (directive Phase 7: pas de patch destructeur)."""
    patches, findings = [], []
    prot_boxes = [list(r.bbox) for r in getattr(protected_index, "regions", [])
                  if getattr(r, "hard", True) and isinstance(r.bbox, (list, tuple)) and len(r.bbox) == 4]
    for t in translated_units:
        bbox = t.patch_bbox or t.coverage_bbox or t.bbox
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        overlap = protected_index.overlap_ratio(bbox) if protected_index else 0.0
        bg = (t.style or {}).get("background_color")
        method = "sampled_color_patch"
        # Text bboxes from PDF extraction are often too tight around glyph ink;
        # pad slightly so descenders, antialiasing halos and OCR/PDF baseline
        # drift do not leave ghosts in the background.
        rects = [_expand_bbox(bbox)]
        if overlap > 0.01 and prot_boxes:
            for hole in prot_boxes:
                nxt = []
                for r in rects:
                    nxt.extend(_subtract(r, hole))
                rects = nxt
            findings.append({"type": "patch_split_around_protected", "unit_id": t.id,
                             "overlap_ratio": round(overlap, 3), "pieces": len(rects), "severity": "info"})
        for r in rects:
            patches.append(PatchZone(
                op_type="patch_text_zone", unit_id=t.id, bbox=r,
                method=method, background_color=bg, protected_overlap_ratio=0.0,
                padding=[1.8, 0.9, 1.8, 0.9],
            ))
    return patches, findings
