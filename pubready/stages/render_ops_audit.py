"""Audit RenderOps : on ne rend que ce qui est validé. BackgroundOp présent,
TextOps couvrant les unités traduites, PatchOps non destructrices, parité PDF/PNG
(par construction: les deux backends exécutent le MÊME plan.render_ops)."""

from __future__ import annotations

from ..schema import StageAuditResult, Finding, OK, REVIEW, KO
from pagereconstruct.source_text_lifecycle_ledger import audit_source_text_lifecycle


def _ov_a(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / max(1e-6, (a[2]-a[0])*(a[3]-a[1]))


def audit_page(plan: dict, normalized: dict) -> StageAuditResult:
    res = StageAuditResult(stage_name="render_ops")
    ops = plan.get("render_ops") or []
    if not ops:
        res.score = 0.0
        res.hard_blockers.append("final_render_missing")
        res.status = KO
        return res
    kinds = {}
    for o in ops:
        kinds[o.get("op_type")] = kinds.get(o.get("op_type"), 0) + 1
    if not kinds.get("background"):
        res.hard_blockers.append("missing_background_op")
        res.findings.append(Finding(type="missing_background_op", severity=KO))

    # TextOps vs unités traduites attendues (texte non vide, non préservé)
    layers = plan.get("layers") or {}
    expect = [t for t in layers.get("translated_text") or []
              if (t.get("translated_text") or "").strip()
              and (t.get("role") or "") not in {"formula_expression", "code_line", "code_block"}]
    text_ops = [o for o in ops if o.get("op_type") == "text"]
    if expect and not text_ops:
        res.hard_blockers.append("missing_textop")
        res.findings.append(Finding(type="missing_textop", severity=KO))
    for o in text_ops:
        if not o.get("composition_id"):
            res.hard_blockers.append("textop_missing_composition_id")
            res.findings.append(Finding(type="textop_missing_composition_id", severity=KO,
                                        element_id=o.get("unit_id")))
        if o.get("source") == "raw_reconstruction_unit":
            res.hard_blockers.append("textop_from_raw_reconstruction_unit")
            res.findings.append(Finding(type="textop_from_raw_reconstruction_unit", severity=KO,
                                        element_id=o.get("unit_id")))
        for key in ("block_id", "line_id", "run_id", "source_unit_ids", "translation_unit_id"):
            if key == "translation_unit_id" and o.get("role") in {"page_number"}:
                continue
            if not o.get(key):
                res.hard_blockers.append(f"textop_missing_{key}")
                res.findings.append(Finding(type=f"textop_missing_{key}", severity=KO,
                                            element_id=o.get("unit_id")))

    # PatchOps non destructrices (aucune sur zone protégée dure)
    prot = [r.get("bbox") for r in plan.get("protected_regions") or [] if r.get("bbox")]
    for o in ops:
        if o.get("op_type") != "patch":
            continue
        b = o.get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        if any(_ov_a(b, p) > 0.05 for p in prot):
            res.hard_blockers.append("patch_protected_overlap")
            res.findings.append(Finding(type="destructive_patch_op", severity=KO))
            break

    lifecycle = audit_source_text_lifecycle(plan, normalized)
    plan["source_text_lifecycle_ledger"] = lifecycle["ledger"]
    for hb in lifecycle["hard_blockers"]:
        if hb not in res.hard_blockers:
            res.hard_blockers.append(hb)
    for item in lifecycle["missing"]:
        for finding in item.get("findings") or []:
            res.findings.append(Finding(
                type=finding,
                severity=KO,
                element_id=item.get("source_unit_id"),
                detail={
                    "level": item.get("level"),
                    "source_text": item.get("source_text"),
                    "pagetranslate_state": item.get("pagetranslate_state"),
                },
            ))

    res.score = 1.0 if not res.hard_blockers else 0.0
    res.status = KO if res.hard_blockers else OK
    return res
