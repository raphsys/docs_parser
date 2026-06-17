"""Audit FOND PROPRE : trame vraiment nettoyée + registre de suppression complet
+ patches non destructeurs. S'appuie sur TextRemovalLedger + la QA visuelle.
"""

from __future__ import annotations

from ..schema import StageAuditResult, Finding, OK, REVIEW, KO
from ..gates import STAGE_THRESHOLDS


def audit_text_removal_stage(plan: dict) -> StageAuditResult:
    from pagereconstruct.text_removal_ledger import audit_text_removal_ledger, build_ledger
    res = StageAuditResult(stage_name="text_removal")
    layers = plan.get("layers") or {}
    required = []
    for t in layers.get("translated_text") or []:
        required.extend(t.get("source_unit_ids") or [])
    for p in (layers.get("preserved_underlays") or []) + (layers.get("preserved_overlays") or []):
        required.extend(p.get("source_unit_ids") or [])
    ledger = plan.get("text_removal_ledger")
    if ledger is None:
        bg = (layers.get("background") or [{}])[0].get("mode")
        ledger = [e.to_dict() for e in build_ledger(plan, background_mode=bg)]
    audit = audit_text_removal_ledger(required, ledger)
    res.hard_blockers.extend(audit["hard_blockers"])
    for f in audit["findings"]:
        res.findings.append(Finding(type=f["type"], severity=KO, detail=f))
    res.score = 1.0 if not res.hard_blockers else 0.0
    res.status = KO if res.hard_blockers else OK
    return res


def audit_page(plan: dict, normalized: dict, *, mode: str = "publication") -> StageAuditResult:
    from pagereconstruct.text_removal_ledger import build_ledger
    res = StageAuditResult(stage_name="background")
    layers = plan.get("layers") or {}
    bg = (layers.get("background") or [{}])[0] or {}
    mode_bg = bg.get("mode")
    leak = bg.get("source_text_leak_risk")

    # 1. fond
    if mode_bg == "clean_background":
        res.score = 1.0
        if bg.get("clean_background_verified") is not True and mode == "publication":
            res.hard_blockers.append("clean_background_not_verified")
            res.score = 0.0
    elif mode_bg == "source_background":
        res.score = 0.5
        if mode == "publication":
            res.hard_blockers.append("source_background_forbidden")
        res.findings.append(Finding(type="source_background_used", severity=REVIEW))
    else:
        res.score = 0.0
        res.hard_blockers.append("missing_clean_background")
    if leak == "high":
        res.hard_blockers.append("source_text_leak")
        res.score = min(res.score, 0.5)

    # 2. registre : toutes les unités traduites couvertes + patches non destructeurs
    ledger = build_ledger(plan, background_mode=mode_bg)
    res.evidence.append(type("E", (), {"to_dict": lambda s: {"text_removal_entries": len(ledger)}})())
    tr_stage = audit_text_removal_stage(plan)
    for hb in tr_stage.hard_blockers:
        if hb not in res.hard_blockers:
            res.hard_blockers.append(hb)
    res.score = min(res.score, tr_stage.score)
    # patch sur zone protégée = destructeur
    prot = [r.get("bbox") for r in plan.get("protected_regions") or [] if r.get("bbox")]
    for p in layers.get("patches") or []:
        b = p.get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        for pr in prot:
            if _ov(b, pr) > 0.05:
                if "patch_protected_overlap" not in res.hard_blockers:
                    res.hard_blockers.append("patch_protected_overlap")
                res.findings.append(Finding(type="destructive_patch", severity=KO,
                                            detail={"bbox": [round(x) for x in b]}))
                break

    res.status = KO if res.hard_blockers else (OK if res.score >= STAGE_THRESHOLDS["background"] else REVIEW)
    return res


def _ov(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ix / area
