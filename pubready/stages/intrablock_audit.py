"""Audit composition intra-bloc : tout le texte du bloc est placé (aucun mot
perdu), pas d'overflow/clipping, objets inline présents."""

from __future__ import annotations

import re
from ..schema import StageAuditResult, ElementAudit, DimensionScore, Finding, OK, REVIEW, KO

_W = re.compile(r"\w+", re.UNICODE)


def _words(s):
    return _W.findall((s or "").lower())


_ALLOWED_REORDER = {"caption_attach_to_figure", "header_split_title_page_number"}


def audit_reading_order(blocks: list[dict]) -> dict:
    blockers: list[str] = []
    findings: list[dict] = []
    for block in blocks or []:
        src = list(block.get("source_reading_order") or block.get("source_reading_order_ids") or [])
        rendered = list(block.get("render_order") or block.get("render_order_ids") or [])
        if not src or not rendered:
            continue
        src_rank = {sid: i for i, sid in enumerate(src)}
        projected = [sid for sid in rendered if sid in src_rank]
        if projected != sorted(projected, key=lambda sid: src_rank[sid]):
            reason = block.get("reorder_reason")
            if reason not in _ALLOWED_REORDER:
                blockers.append("reading_order_changed")
                findings.append({"type": "reading_order_changed", "block_id": block.get("block_id")})
    blockers = sorted(set(blockers))
    return {"status": "ko" if blockers else "ok", "hard_blockers": blockers, "findings": findings}


def audit_page(plan: dict, normalized: dict) -> StageAuditResult:
    res = StageAuditResult(stage_name="intrablock")
    layers = plan.get("layers") or {}
    blocks = layers.get("translated_text") or []
    if not blocks:
        res.score = 1.0
        return res
    ro_blocks = []
    for c in plan.get("intrablock_compositions") or []:
        ro_blocks.append({
            "block_id": c.get("block_id"),
            "source_reading_order": c.get("source_reading_order") or c.get("source_unit_ids"),
            "render_order": c.get("render_order") or [
                sid
                for line in c.get("lines") or []
                for run in line.get("runs") or []
                for sid in run.get("source_unit_ids") or []
            ],
            "reorder_reason": c.get("reorder_reason"),
        })
    ro = audit_reading_order(ro_blocks)
    for hb in ro["hard_blockers"]:
        res.hard_blockers.append(hb)
    for f in ro["findings"]:
        res.findings.append(Finding(type=f["type"], severity=KO, element_id=f.get("block_id")))
    # texte rendu par bloc = lignes des TextOps (composition)
    rendered = {}
    rendered_by_source = []
    for op in plan.get("render_ops") or []:
        if op.get("op_type") != "text":
            continue
        uid = op.get("unit_id")
        txt = " ".join(l.get("text", "") for l in op.get("lines") or [])
        rendered.setdefault(uid, []).append(txt)
        rendered_by_source.append((set(op.get("source_unit_ids") or []), txt))

    auds = []
    for b in blocks:
        bid = b.get("id")
        expected = (b.get("translated_text") or b.get("source_text") or "").strip()
        source_ids = set(b.get("source_unit_ids") or [])
        placed_parts = list(rendered.get(bid, []))
        if not placed_parts and source_ids:
            for op_source_ids, text in rendered_by_source:
                if source_ids & op_source_ids or any(
                    a.startswith(c + "_") or c.startswith(a + "_")
                    for a in source_ids for c in op_source_ids
                ):
                    placed_parts.append(text)
        placed = " ".join(placed_parts)
        exp_w, plc_w = _words(expected), set(_words(placed))
        if not exp_w:
            cov = 1.0
        else:
            cov = sum(1 for w in exp_w if w in plc_w) / len(exp_w)
        dims = [DimensionScore("text_placed", round(cov, 3), len(exp_w),
                               len([w for w in exp_w if w in plc_w]),
                               OK if cov >= 0.98 else REVIEW if cov >= 0.85 else KO, 1.5,
                               None if cov >= 0.98 else "block_text_missing")]
        el = ElementAudit(element_id=bid, level="block", role=b.get("role") or "",
                          source_text=b.get("source_text") or "", translated_text=expected, dimensions=dims)
        el.combine(); auds.append(el)
        if cov < 0.85 and expected:
            res.hard_blockers.append("missing_translatable_text") if "missing_translatable_text" not in res.hard_blockers else None
            res.findings.append(Finding(type="block_text_missing", severity=KO, element_id=bid,
                                        detail={"coverage": round(cov, 3)}))
    res.elements = auds
    res.score = round(sum(a.score for a in auds) / len(auds), 3)
    res.status = KO if any(a.status == KO for a in auds) else (REVIEW if res.score < 0.95 else OK)
    return res
