"""Visual QA: score a reconstruction against the 6 publication criteria
(directive PR-Lot 8/9). Works on the plan + the real measured geometry.

Scores (0..1): text_presence, non_text_presence, overlap, position, typography.
publication_ready = weighted, with hard gates.
"""

from __future__ import annotations

from .collision_detector import detect as detect_collisions
from .quality import assess as assess_quality
from .renderer_dispatcher import dispatch

_W = {"text_presence": 0.30, "non_text_presence": 0.15, "overlap": 0.20,
      "position": 0.15, "typography": 0.20}


def _scale(page: dict):
    wpt, hpt = page.get("width_pt"), page.get("height_pt")
    rw, rh = page.get("render_width_px"), page.get("render_height_px")
    if wpt and hpt and rw and rh:
        return rw / wpt, rh / hpt
    return 1.0, 1.0


def measure_page(plan: dict):
    page = plan.get("page") or {}
    sx, sy = _scale(page)
    rw = page.get("render_width_px")
    results = []
    for t in (plan.get("layers") or {}).get("translated_text") or []:
        results.append(dispatch(t.get("renderer"), t.get("role")).measure(t, sx, sy, page_w_px=rw))
    return results, sx, sy


def assess(plan: dict) -> dict:
    q = assess_quality(plan)
    rp = plan.get("render_policy") or {}
    results, sx, sy = measure_page(plan)
    protected_px = [[r["bbox"][0] * sx, r["bbox"][1] * sy, r["bbox"][2] * sx, r["bbox"][3] * sy]
                    for r in plan.get("protected_regions") or [] if r.get("bbox")]
    coll = detect_collisions(results, protected_px)

    n = max(1, q["text_units"])
    coverage = rp.get("translation_coverage_ratio")
    text_presence = 1.0 if coverage is None else min(1.0, float(coverage))
    if rp.get("publication_blocked"):
        text_presence = min(text_presence, 0.5)
    non_text_presence = max(0.0, 1.0 - q["patch_protected_overlap"] / n)
    overlap = max(0.0, 1.0 - coll["text_text"]["max_overlap"] - coll["text_protected"]["max_overlap"])
    position = max(0.0, 1.0 - q["layout_repaired"] / n)
    typography = max(0.0, 1.0 - (q["unresolved_style"] + q["font_size_repaired"] + q["unknown_renderer"]) / n)

    scores = {"text_presence": round(text_presence, 3), "non_text_presence": round(non_text_presence, 3),
              "overlap": round(overlap, 3), "position": round(position, 3), "typography": round(typography, 3)}
    ready = sum(_W[k] * scores[k] for k in _W)
    # hard gates
    if text_presence < 1.0:
        ready = min(ready, 0.80)
    if coll["status"] == "ko":
        ready = min(ready, 0.60)
    if typography < 0.95:
        ready = min(ready, 0.90)
    if q["source_text_leak_risk"] == "high":
        ready = min(ready, 0.70)

    findings = list(coll["findings"])
    return {
        "scores": scores,
        "publication_ready_score": round(ready, 3),
        "collision_status": coll["status"],
        "quality": q,
        "findings": findings,
    }
