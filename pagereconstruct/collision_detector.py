"""Detect overlaps on the ACTUAL rendered geometry (directive PR-Lot 4).

Inputs are RenderResults (with actual_text_bbox / actual_line_boxes). Detects
text/text and text/protected collisions and assigns a status from thresholds:
  text/text  > 0.02 -> review, > 0.10 -> ko
  text/hard-protected > 0.01 -> review/ko
"""

from __future__ import annotations


def _inter(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def _area(b) -> float:
    return max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))


def _ratio_min(a, b) -> float:
    return _inter(a, b) / min(_area(a), _area(b))


def detect_text_collisions(results, *, review=0.02, ko=0.10) -> dict:
    boxes = [(r.unit_id, r.actual_text_bbox) for r in results
             if getattr(r, "actual_text_bbox", None) and len(r.actual_text_bbox) == 4]
    findings, max_r = [], 0.0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            r = _ratio_min(boxes[i][1], boxes[j][1])
            max_r = max(max_r, r)
            if r > review:
                findings.append({"type": "text_text_overlap", "a": boxes[i][0], "b": boxes[j][0],
                                 "ratio": round(r, 3), "severity": "ko" if r > ko else "review"})
    status = "ko" if max_r > ko else ("review" if max_r > review else "ok")
    return {"status": status, "max_overlap": round(max_r, 3), "collisions": findings}


def detect_protected_collisions(results, protected_px, *, review=0.01, ko=0.10) -> dict:
    findings, max_r = [], 0.0
    for r in results:
        tb = getattr(r, "actual_text_bbox", None)
        if not (tb and len(tb) == 4):
            continue
        for pr in protected_px or []:
            ratio = _inter(tb, pr) / _area(tb)
            if ratio > review:
                max_r = max(max_r, ratio)
                findings.append({"type": "text_protected_overlap", "unit_id": r.unit_id,
                                 "ratio": round(ratio, 3), "severity": "ko" if ratio > ko else "review"})
    status = "ko" if max_r > ko else ("review" if max_r > review else "ok")
    return {"status": status, "max_overlap": round(max_r, 3), "collisions": findings}


def detect(results, protected_px=None) -> dict:
    tt = detect_text_collisions(results)
    tp = detect_protected_collisions(results, protected_px or [])
    order = {"ok": 0, "review": 1, "ko": 2}
    status = max(tt["status"], tp["status"], key=lambda s: order[s])
    return {"status": status, "text_text": tt, "text_protected": tp,
            "findings": tt["collisions"] + tp["collisions"]}
