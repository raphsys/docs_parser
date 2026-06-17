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


class _OpResult:
    """Résultat de mesure léger bâti depuis un TextOp (géométrie finale placée)."""
    def __init__(self, unit_id, renderer, boxes_px):
        self.unit_id = unit_id
        self.renderer = renderer
        self.actual_line_boxes = boxes_px
        self.actual_text_bbox = ([min(b[0] for b in boxes_px), min(b[1] for b in boxes_px),
                                  max(b[2] for b in boxes_px), max(b[3] for b in boxes_px)]
                                 if boxes_px else None)
        self.findings = []


def measure_page(plan: dict):
    page = plan.get("page") or {}
    sx, sy = _scale(page)
    rw = page.get("render_width_px")
    # Contract path: QA mesure les RenderOps réellement placés (post-solver),
    # pas le plan brut — sinon les corrections de placement ne comptent pas.
    ops = plan.get("render_ops") or []
    if ops:
        results = []
        for op in ops:
            if op.get("op_type") != "text":
                continue
            boxes = [[ln["x"] * sx, ln["y_top"] * sy, ln["x1"] * sx, ln["y_bottom"] * sy]
                     for ln in op.get("lines") or [] if "x1" in ln]
            if boxes:
                results.append(_OpResult(op.get("unit_id"), op.get("role") or "text", boxes))
        return results, sx, sy
    results = []
    for t in (plan.get("layers") or {}).get("translated_text") or []:
        results.append(dispatch(t.get("renderer"), t.get("role")).measure(t, sx, sy, page_w_px=rw))
    return results, sx, sy


def _image_leak_score(plan, sx, sy, source_image_path, reconstructed_image_path):
    """Image-réelle (optionnel): un patch dont les pixels n'ont PAS changé entre
    source et reconstruit = ancien texte encore visible = fuite."""
    try:
        from PIL import Image
        from .source_text_leak_detector import detect as leak_detect
        src = Image.open(source_image_path).convert("RGB")
        rec = Image.open(reconstructed_image_path).convert("RGB")
        patches_px = [[p["bbox"][0] * sx, p["bbox"][1] * sy, p["bbox"][2] * sx, p["bbox"][3] * sy]
                      for p in (plan.get("layers") or {}).get("patches") or [] if p.get("bbox")]
        if not patches_px:
            return 1.0, []
        res = leak_detect(src, rec, patches_px)
        score = max(0.0, 1.0 - res["leak_count"] / max(1, len(patches_px)))
        return score, res["findings"]
    except Exception as exc:  # pragma: no cover
        return 1.0, [{"type": "image_leak_check_failed", "message": str(exc)}]


def assess(plan: dict, *, source_image_path: str | None = None,
           reconstructed_image_path: str | None = None) -> dict:
    q = assess_quality(plan)
    rp = plan.get("render_policy") or {}
    results, sx, sy = measure_page(plan)
    protected_px = [{"bbox": [r["bbox"][0] * sx, r["bbox"][1] * sy, r["bbox"][2] * sx, r["bbox"][3] * sy],
                     "reason": r.get("reason")}
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
    # Typography failures = NO style at all / unknown renderer. A font size
    # inferred from consistent line geometry is NOT a failure (the only upstream
    # size is a glyph metric ~4pt); it is flagged separately as unreliable.
    typo_failures = q["unresolved_style"] + q["unknown_renderer"]
    typography = max(0.0, 1.0 - typo_failures / n)
    font_repair_ratio = q["font_size_repaired"] / n
    # Avec l'em résolu depuis l'image sur la majorité des unités, une page reste
    # fiable tant que < la moitié des tailles sont seulement réparées par géométrie.
    style_unreliable = font_repair_ratio > 0.50
    if style_unreliable:
        # Inferred sizes can never grade as publication-ok typography.
        typography = min(typography, 0.90)

    # source_text_leak_score : image-réelle si fournie, sinon dérivé du risque.
    leak_findings = []
    if source_image_path and reconstructed_image_path:
        source_text_leak, leak_findings = _image_leak_score(plan, sx, sy, source_image_path, reconstructed_image_path)
    else:
        source_text_leak = {"low": 1.0, "medium": 0.9, "high": 0.5}.get(q["source_text_leak_risk"], 0.5)

    scores = {"text_presence": round(text_presence, 3), "non_text_presence": round(non_text_presence, 3),
              "overlap": round(overlap, 3), "position": round(position, 3),
              "typography": round(typography, 3), "source_text_leak": round(source_text_leak, 3)}
    ready = sum(_W[k] * scores[k] for k in _W)
    # hard gates
    if text_presence < 1.0:
        ready = min(ready, 0.80)
    if coll["status"] == "ko":
        ready = min(ready, 0.60)
    if typography < 0.95:
        ready = min(ready, 0.90)
    if style_unreliable:
        ready = min(ready, 0.80)   # page_style_unreliable -> review, jamais ok
    if q["source_text_leak_risk"] == "high" or source_text_leak < 0.98:
        ready = min(ready, 0.70)

    # Blockers durs (directive Phase 9) : publication-ready impossible si l'un est vrai.
    hard_blockers = []
    if q["source_text_leak_risk"] == "high":
        hard_blockers.append("source_text_leak_high")
    if q["text_units"] and q["styled_units"] == 0:
        hard_blockers.append("no_styled_text")
    if q["patch_protected_overlap"]:
        hard_blockers.append("patch_protected_overlap")
    if coll["status"] == "ko":
        hard_blockers.append("collision_ko")

    findings = list(coll["findings"]) + list(leak_findings)
    if style_unreliable:
        findings.append({"type": "page_style_unreliable",
                         "font_repair_ratio": round(font_repair_ratio, 2), "severity": "review"})
    return {
        "scores": scores,
        "publication_ready_score": round(ready, 3),
        "collision_status": coll["status"],
        "hard_blockers": hard_blockers,
        "quality": q,
        "findings": findings,
    }
