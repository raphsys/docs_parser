"""Audit du FinalReconstructionContract : pagereconstruct consomme-t-il
fidèlement (chaque bloc a layout/style/renderer ; layer_order ; préservation) ?"""

from __future__ import annotations

from ..schema import StageAuditResult, Finding, OK, REVIEW, KO


def audit_source_unit_states(final_contract: dict) -> dict:
    blockers: list[str] = []
    findings: list[dict] = []
    for state in final_contract.get("source_unit_states") or []:
        sid = state.get("source_unit_id")
        st = state.get("state")
        textops = state.get("textop_ids") or []
        preservations = state.get("preservationop_ids") or []
        sf = set(state.get("findings") or [])
        if (st == "translated_and_rendered" and preservations) or "source_unit_both_preserved_and_translated" in sf:
            blockers.append("source_unit_both_preserved_and_translated")
            findings.append({"type": "source_unit_both_preserved_and_translated", "source_unit_id": sid})
        if st == "translated_and_rendered" and state.get("still_visible_in_background"):
            blockers.append("source_unit_still_visible_in_background")
        if st == "preserved_exact" and state.get("translated_text_changed"):
            blockers.append("preserved_source_unit_translated")
        if st == "removed_from_background" and not textops and not state.get("ignored_reason"):
            blockers.append("removed_source_unit_missing_translation")
    blockers = sorted(set(blockers))
    return {"status": "ko" if blockers else "ok", "hard_blockers": blockers, "findings": findings}


def audit_page(plan: dict, normalized: dict) -> StageAuditResult:
    res = StageAuditResult(stage_name="contract")
    fc = plan.get("final_contract") or {}
    if not fc:
        res.score = 0.0
        res.hard_blockers.append("missing_block_contract")
        res.status = KO
        res.findings.append(Finding(type="final_contract_absent", severity=KO))
        return res

    blocks = fc.get("blocks") or []
    n = max(1, len(blocks))
    miss_layout = miss_style = miss_renderer = 0
    for b in blocks:
        if not (b.get("layout") or {}).get("layout_bbox"):
            miss_layout += 1
        if not (b.get("style") or {}):
            miss_style += 1
        rn = (b.get("render") or {}).get("renderer_name")
        if not rn or rn == "anchored_label_review":
            miss_renderer += 1

    if not fc.get("layer_order"):
        res.hard_blockers.append("layer_order_missing")
    state_audit = audit_source_unit_states(fc)
    for hb in state_audit["hard_blockers"]:
        if hb not in res.hard_blockers:
            res.hard_blockers.append(hb)
    for f in state_audit["findings"]:
        res.findings.append(Finding(type=f["type"], severity=KO, detail=f))
    if miss_layout:
        res.hard_blockers.append("missing_layout_contract")
        res.findings.append(Finding(type="missing_layout_contract", severity=KO, detail={"count": miss_layout}))
    if miss_style:
        res.findings.append(Finding(type="missing_style_contract", severity=REVIEW, detail={"count": miss_style}))
    if miss_renderer:
        res.findings.append(Finding(type="unresolved_renderer", severity=REVIEW, detail={"count": miss_renderer}))

    # objets non-texte critiques → preservation présente (souple : présence du contrat).
    score = 1.0 - 0.5 * (miss_layout / n) - 0.25 * (miss_style / n) - 0.25 * (miss_renderer / n)
    res.score = round(max(0.0, score), 3)
    res.status = KO if res.hard_blockers else (OK if res.score >= 0.98 else REVIEW)
    return res
