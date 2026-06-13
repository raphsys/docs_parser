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


def plan_patches(translated_units, protected_index) -> tuple[list, list]:
    """Return (patches, findings)."""
    patches, findings = [], []
    for t in translated_units:
        bbox = t.patch_bbox or t.coverage_bbox or t.bbox
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        overlap = protected_index.overlap_ratio(bbox) if protected_index else 0.0
        bg = (t.style or {}).get("background_color")
        method = "sampled_color_patch" if bg else "sampled_whiteout"
        if overlap > 0.05:
            findings.append({"type": "patch_protected_overlap", "unit_id": t.id,
                             "overlap_ratio": round(overlap, 3), "severity": "review"})
        patches.append(PatchZone(
            op_type="patch_text_zone", unit_id=t.id, bbox=[float(x) for x in bbox],
            method=method, background_color=bg, protected_overlap_ratio=round(overlap, 4),
        ))
    return patches, findings
