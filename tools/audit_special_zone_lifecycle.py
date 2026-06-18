#!/usr/bin/env python3
"""Audit special-zone lifecycle across PAGEPRINT -> PAGERECONSTRUCT.

Usage:
  python tools/audit_special_zone_lifecycle.py results/demo_studio_YYYYMMDD_HHMMSS

It answers:
  1. ONNX/YOLO detector status.
  2. Are formula/code/protected zones hard-protected?
  3. Are zones present in PAGEPRINT?
  4. Did they survive into PAGERECONSTRUCT protected_regions/final_contract?
  5. Are they emitted as PreservationOps in render_ops?
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SPECIAL_RE = re.compile(r"formula|equation|math|code|protected_visual", re.I)
FORMULA_TEXT_RE = re.compile(r"[∑∫√∞≈≠≤≥±×÷∂∆λµπσΔαβγ=<>*/^]|\b[A-Za-z]\s*\(.*\)")


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def tag_from(path: Path, prefix: str) -> str:
    stem = path.stem
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def special_regions(pageprint: dict) -> list[dict]:
    regs = []
    for r in pageprint.get("regions") or []:
        if SPECIAL_RE.search(str(r.get("region_type") or r.get("object_type") or r.get("reason") or "")):
            regs.append(r)
    return regs


def detector_status(pageprint: dict) -> dict:
    dbg = ((pageprint.get("debug") or {}).get("page_region_detect") or {})
    det = dbg.get("detectors") or {}
    return {
        "onnx_yolo": det.get("onnx_yolo") or {},
        "pdf_glyph_formula": det.get("pdf_glyph_formula") or {},
        "hybrid": det.get("hybrid_special_region_detector") or {},
        "warnings": dbg.get("warnings") or [],
    }


def protected_regions(plan: dict) -> list[dict]:
    return [r for r in plan.get("protected_regions") or [] if SPECIAL_RE.search(str(r.get("reason") or r.get("source") or ""))]


def preserved_special(plan: dict) -> list[dict]:
    layers = plan.get("layers") or {}
    out = []
    for key in ("preserved_underlays", "preserved_overlays"):
        for p in layers.get(key) or []:
            if p.get("source") == "special_zone" or SPECIAL_RE.search(str(p.get("reason") or "")):
                out.append({**p, "layer": key})
    return out


def render_preservation_ops(plan: dict) -> list[dict]:
    return [op for op in plan.get("render_ops") or [] if op.get("op_type") == "preservation" and SPECIAL_RE.search(str(op.get("text") or op.get("source") or op.get("reason") or op.get("source_unit_ids") or "") + str(op.get("bbox") or ""))]


def formula_like_translated(plan: dict) -> list[dict]:
    layers = plan.get("layers") or {}
    out = []
    for t in layers.get("translated_text") or []:
        txt = " ".join(str(t.get(k) or "") for k in ("source_text", "translated_text", "role", "object_type"))
        if FORMULA_TEXT_RE.search(txt):
            out.append(t)
    return out


def main(root: str) -> int:
    rootp = Path(root)
    if not rootp.exists():
        print(f"missing path: {root}", file=sys.stderr)
        return 2
    pageprints = {tag_from(p, "pageprint_full_"): p for p in rootp.glob("pageprint_full_*.json")}
    if not pageprints:
        pageprints = {tag_from(p, "pageprint_"): p for p in rootp.glob("pageprint_*.json")}
    plans = {tag_from(p, "pagereconstruct_plan_"): p for p in rootp.glob("pagereconstruct_plan_*.json")}
    rows = []
    for tag, ppath in sorted(pageprints.items()):
        pp = load(ppath)
        plan = load(plans.get(tag, Path("__missing__"))) if tag in plans else {}
        regs = special_regions(pp)
        prots = protected_regions(plan)
        pres = preserved_special(plan)
        rops = render_preservation_ops(plan)
        ftxt = formula_like_translated(plan)
        det = detector_status(pp)
        rows.append({
            "tag": tag,
            "q1_yolo": det["onnx_yolo"],
            "q2_protected_special_count": len(prots),
            "q3_pageprint_special_count": len(regs),
            "q4_preserved_special_count": len(pres),
            "q5_render_preservation_ops_count": len(rops),
            "formula_like_translated_count": len(ftxt),
            "status": "ok" if regs and prots and pres else "review" if regs else "no_special_detected",
        })
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
