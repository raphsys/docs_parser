"""Reconstruction validator: findings govern the status (directive Lot 10).

ok      no critical/review finding
review  style repaired, font substitution, layout repaired, overflow, leak risk…
ko      text expected but absent, missing background with text, mostly-protected
        patch overlap
"""

from __future__ import annotations

from .quality import assess


def validate(plan: dict) -> dict:
    q = assess(plan)
    rp = plan.get("render_policy") or {}
    if rp.get("publication_blocked"):
        return {"status": "ko", "quality": q,
                "findings": [{"type": "publication_blocked", "severity": "ko"}],
                "publication_ready": False, "publication_ready_score": 0.0, "visual_scores": {}}
    findings: list = []
    status = "ok"

    has_text = q["text_units"] > 0

    # --- ko conditions
    if has_text and q["missing_background"]:
        findings.append({"type": "missing_background_with_text", "severity": "ko"})
        status = "ko"
    if has_text and q["styled_units"] == 0:
        findings.append({"type": "no_styled_text", "severity": "ko"})
        status = "ko"

    # --- review conditions
    # font_size_repaired n'est plus un déclencheur direct de review : la fidélité
    # typo est jugée par le score typography + le gate page_style_unreliable
    # (sinon une seule taille inférée bloquerait une page par ailleurs fiable).
    review_metrics = ("unresolved_style", "layout_repaired",
                      "overflow", "patch_protected_overlap", "unknown_renderer")
    if status != "ko":
        if any(q[m] for m in review_metrics) or q["source_text_leak_risk"] == "high":
            for m in review_metrics:
                if q[m]:
                    findings.append({"type": m, "count": q[m], "severity": "review"})
            if q["source_text_leak_risk"] == "high":
                findings.append({"type": "source_text_leak_risk_high", "severity": "review"})
            status = "review"

    # Visual QA: scores + strict publication_ready gate.
    from .visual_qa import assess as visual_assess
    vqa = visual_assess(plan)
    scores = vqa["scores"]
    if vqa["collision_status"] == "ko":
        status = "ko"
    findings.extend(vqa["findings"])
    publication_ready = (
        status == "ok"
        and not vqa.get("hard_blockers")
        and scores["text_presence"] >= 1.0
        and scores["non_text_presence"] >= 0.99
        and scores["overlap"] >= 0.99
        and scores["typography"] >= 0.95
        and scores["position"] >= 0.95
        and scores.get("source_text_leak", 0.0) >= 0.98
        and q["source_text_leak_risk"] != "high"
        and vqa["publication_ready_score"] >= 0.95
    )
    return {"status": status, "quality": q, "findings": findings,
            "visual_scores": scores, "publication_ready_score": vqa["publication_ready_score"],
            "hard_blockers": vqa.get("hard_blockers", []),
            "publication_ready": publication_ready}
