"""Detect residual source text under the reconstruction (directive PR-Lot 5).

A patch zone that is (nearly) identical between the source page and the
reconstructed page means the old text was NOT removed -> leak. Operates on two
PIL images and the patch bboxes in pixel space.
"""

from __future__ import annotations


def _grayscale_crop(img, box):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.width, x1), min(img.height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return img.convert("L").crop((x0, y0, x1, y1))


def _mean_abs_diff(a, b) -> float:
    da, db = list(a.getdata()), list(b.getdata())
    n = min(len(da), len(db))
    if not n:
        return 255.0
    return sum(abs(da[i] - db[i]) for i in range(n)) / n


def detect(source_img, reconstructed_img, patches_px, *, min_change: float = 12.0) -> dict:
    """Return {leak_count, findings}. A patch with mean change < min_change leaks."""
    findings = []
    for b in patches_px or []:
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        sc = _grayscale_crop(source_img, b)
        rc = _grayscale_crop(reconstructed_img, b)
        if sc is None or rc is None:
            continue
        change = _mean_abs_diff(sc, rc)
        if change < min_change:
            findings.append({"type": "source_text_leak_detected", "bbox": list(b),
                             "mean_change": round(change, 2), "severity": "review"})
    return {"leak_count": len(findings), "findings": findings}
