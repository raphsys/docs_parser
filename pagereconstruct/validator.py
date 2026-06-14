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
                "findings": [{"type": "publication_blocked", "severity": "ko"}]}
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
    review_metrics = ("unresolved_style", "font_size_repaired", "layout_repaired",
                      "overflow", "patch_protected_overlap", "unknown_renderer")
    if status != "ko":
        if any(q[m] for m in review_metrics) or q["source_text_leak_risk"] == "high":
            for m in review_metrics:
                if q[m]:
                    findings.append({"type": m, "count": q[m], "severity": "review"})
            if q["source_text_leak_risk"] == "high":
                findings.append({"type": "source_text_leak_risk_high", "severity": "review"})
            status = "review"

    return {"status": status, "quality": q, "findings": findings}
