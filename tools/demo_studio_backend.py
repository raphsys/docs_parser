#!/usr/bin/env python3
"""Backend local pour Demo Studio Flutter.

Ce script est volontairement non-web. Il exécute les unités du pipeline
PAGEPRINT / PAGETRANSLATE / PAGERECONSTRUCT sur les pages choisies, écrit les
artefacts dans results/, puis émet des évènements JSON sur stdout pour l'UI.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

from PIL import Image, ImageDraw

from pagetranslate import build_page_translation
from pagereconstruct import compile_page_render_plan, validate
from tools.run_pageprint_pagetranslate_audit import (
    make_engine,
    make_orchestrator,
    render_bboxes,
)
from tools.run_pipeline_full_demo import render_plan_overlay

STAGES = {
    "full",
    "pageprint",
    "pagetranslate",
    "pagereconstruct",
    "view_background",
    "audit_translation_selection",
    "audit_text_survival",
}


def emit(event: str, **payload) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def parse_pages(spec: str, max_pages: int | None = None) -> list[int]:
    """Parse `1,4,7-9` into sorted unique 1-based page numbers."""
    out: set[int] = set()
    for part in re.split(r"[,;\s]+", str(spec or "")):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            if end < start:
                start, end = end, start
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    pages = sorted(p for p in out if p > 0 and (max_pages is None or p <= max_pages))
    if not pages:
        raise ValueError("aucune page valide sélectionnée")
    return pages


def pdf_info(pdf: Path) -> dict:
    if fitz is None:
        raise RuntimeError("PyMuPDF/fitz indisponible")
    doc = fitz.open(str(pdf))
    try:
        return {"path": str(pdf), "name": pdf.name, "page_count": doc.page_count}
    finally:
        doc.close()


def inspect_pdf_main(pdf: Path) -> int:
    print(json.dumps(pdf_info(pdf), ensure_ascii=False))
    return 0


def _scale(geom: dict) -> tuple[float, float]:
    sx = geom.get("scale_x_px_per_pt")
    sy = geom.get("scale_y_px_per_pt")
    if sx and sy:
        return float(sx), float(sy)
    w, h = geom.get("width"), geom.get("height")
    rw, rh = geom.get("render_width_px"), geom.get("render_height_px")
    if w and h and rw and rh:
        return float(rw) / float(w), float(rh) / float(h)
    return 1.0, 1.0


def _slim_pageprint(d: dict) -> dict:
    units = []
    for u in d.get("units") or []:
        if u.get("level") not in {"block", "line", "phrase", "region", "table", "cell"}:
            continue
        units.append({
            "unit_id": u.get("unit_id"),
            "level": u.get("level"),
            "parent_id": u.get("parent_id"),
            "bbox": (u.get("geometry") or {}).get("bbox"),
            "role": (u.get("understanding") or {}).get("role"),
            "object_type": (u.get("understanding") or {}).get("object_type"),
            "translatable": (u.get("policy") or {}).get("translatable"),
            "text": (u.get("content") or {}).get("text"),
        })
    views = d.get("views") or {}
    return {
        "schema_version": d.get("schema_version"),
        "page": d.get("page"),
        "page_intelligence": d.get("page_intelligence"),
        "logical_structures_counts": {
            k: len(v) for k, v in (d.get("logical_structures") or {}).items() if isinstance(v, list)
        },
        "views_counts": {k: (len(v) if isinstance(v, list) else "…") for k, v in views.items()},
        "units_block_line_phrase": units,
    }


def _copy_source(input_data: dict, out: Path, tag: str) -> Path | None:
    src = (input_data.get("assets") or {}).get("source_image_path")
    if src and Path(src).is_file():
        dst = out / f"source_{tag}.png"
        Image.open(src).convert("RGB").save(dst)
        return dst
    return None


def _render_background_preview(input_data: dict, out: Path, tag: str) -> Path | None:
    try:
        from pipelines.background_cover import build_deterministic_text_cover_background
        dst = out / f"background_preview_{tag}.png"
        cp = build_deterministic_text_cover_background(input_data, out_path=str(dst))
        if cp:
            return Path(cp)
    except Exception:
        pass

    src = (input_data.get("assets") or {}).get("source_image_path")
    if not src or not Path(src).is_file():
        return None
    img = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(img)
    geom = (input_data.get("page") or {}).get("geometry") or {}
    sx, sy = _scale(geom)
    for unit in input_data.get("units") or []:
        if unit.get("level") not in {"line", "phrase", "word"}:
            continue
        if not str((unit.get("content") or {}).get("text") or "").strip():
            continue
        bbox = (unit.get("geometry") or {}).get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        x0, y0, x1, y1 = bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy
        draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], fill=(255, 255, 255))
    dst = out / f"background_preview_{tag}.png"
    img.save(dst)
    return dst


def _regen_clean_background(tid: dict, out: Path, tag: str) -> str | None:
    try:
        from pipelines.background_cleaner import build_clean_background
        cp = build_clean_background(tid, out_path=str(out / f"cleanbg_{tag}.png"))
        if cp:
            tid.setdefault("visual_layers", {})["clean_background_path"] = cp
            tid.setdefault("visual_layers", {})["clean_background_verified"] = True
            tid.setdefault("visual_layers", {})["text_removed"] = True
            tid.setdefault("assets", {})["background_clean_path"] = cp
            tid.setdefault("assets", {})["background_clean_verified"] = True
            tid.setdefault("assets", {})["text_removed"] = True
            return cp
    except Exception as exc:
        emit("warning", tag=tag, message=f"clean background impossible: {exc}")
    return None


def _write_background_compare(out: Path, tag: str) -> None:
    files = [out / f"source_{tag}.png", out / f"cleanbg_{tag}.png", out / f"background_preview_{tag}.png"]
    imgs = []
    for f in files:
        if f.is_file():
            im = Image.open(f).convert("RGB")
            im.thumbnail((360, 480))
        else:
            im = Image.new("RGB", (360, 480), "white")
            ImageDraw.Draw(im).text((10, 10), f"absent\n{f.name}", fill="black")
        imgs.append(im)
    w = sum(i.width for i in imgs) + 80
    h = max(i.height for i in imgs) + 50
    sheet = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)
    x = 20
    for label, im in zip(["source", "cleanbg", "preview"], imgs):
        draw.text((x, 8), label, fill="black")
        sheet.paste(im, (x, 30))
        x += im.width + 20
    sheet.save(out / f"background_compare_{tag}.jpg", quality=92)


def _write_contact_sheet(out: Path) -> Path | None:
    tags = sorted({p.stem.replace("source_", "") for p in out.glob("source_*.png")})
    if not tags:
        return None
    labels = ["source", "cleanbg", "bboxes", "overlay", "reconstructed"]
    def path_for(label: str, tag: str) -> Path:
        if label == "bboxes":
            return out / f"pageprint_bboxes_{tag}.png"
        if label == "overlay":
            return out / f"pagereconstruct_overlay_{tag}.png"
        return out / f"{label}_{tag}.png"
    rows = []
    row_h = 0
    max_w = 0
    for tag in tags:
        ims = []
        for label in labels:
            p = path_for(label, tag)
            if p.is_file():
                im = Image.open(p).convert("RGB")
                im.thumbnail((260, 360))
            else:
                im = Image.new("RGB", (260, 360), "white")
                ImageDraw.Draw(im).text((8, 8), f"absent\n{p.name}", fill="black")
            ims.append(im)
        rows.append((tag, ims))
        max_w = max(max_w, sum(i.width for i in ims) + 30 * (len(ims) + 1))
        row_h = max(row_h, max(i.height for i in ims) + 65)
    sheet = Image.new("RGB", (max_w, row_h * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for tag, ims in rows:
        x = 18
        draw.text((x, y + 5), tag, fill="black")
        for label, im in zip(labels, ims):
            draw.text((x, y + 24), label, fill="black")
            sheet.paste(im, (x, y + 42))
            x += im.width + 28
        y += row_h
    dst = out / "contact_sheet.jpg"
    sheet.save(dst, quality=92)
    return dst


def _write_report(out: Path, summaries: list[dict]) -> None:
    lines = ["# Demo Studio — rapport", "", f"Dossier: `{out}`", "", "| page | stage | translated | protected | preserved | audit | pubready |", "|---|---|---:|---:|---:|---|---|"]
    for s in summaries:
        pr = s.get("pubready") or {}
        lines.append(
            f"| `{s.get('tag')}` | `{s.get('stage')}` | {s.get('translated_text_count', '')} | "
            f"{s.get('protected_region_count', '')} | {s.get('preserved_count', '')} | "
            f"{s.get('status', '')} | {pr.get('score', '')} {pr.get('status', '')} |"
        )
    lines += ["", "## Lecture", "", "- `source_*.png`: rendu original de la page", "- `pageprint_bboxes_*.png`: bboxes PAGEPRINT", "- `cleanbg_*.png`: fond nettoyé", "- `pagereconstruct_overlay_*.png`: plan de reconstruction", "- `reconstructed_*.png`: rendu final", "- `contact_sheet.jpg`: synthèse visuelle"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_page(orchestrator, engine, pdf: Path, page: int, stage: str, out: Path, args) -> dict:
    tag = f"{pdf.stem[:24]}_p{page:04d}"
    emit("page_start", page=page, tag=tag)
    doc = orchestrator.run(str(pdf), pages=str(page), language={"source_lang": args.source_lang, "target_lang": args.target_lang})
    ok = [p for p in (doc.get("pages") or []) if p.get("status") == "ok"]
    if not ok:
        return {"tag": tag, "stage": stage, "error": "extraction_failed"}
    input_data = ok[0]["input_data"]

    (out / f"pageprint_{tag}.json").write_text(json.dumps(_slim_pageprint(input_data), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"pageprint_full_{tag}.json").write_text(json.dumps(input_data, ensure_ascii=False, indent=2), encoding="utf-8")
    src_png = _copy_source(input_data, out, tag)
    if src_png:
        render_bboxes(input_data, src_png, out / f"pageprint_bboxes_{tag}.png")
    _render_background_preview(input_data, out, tag)

    if stage == "audit_translation_selection":
        try:
            from tools.audit_translation_selection import audit_page, write_markdown
            audit = audit_page(input_data)
            (out / f"audit_translation_selection_{tag}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            write_markdown(tag, audit, out / f"audit_translation_selection_{tag}.md")
        except Exception as exc:
            emit("warning", tag=tag, message=f"audit_translation_selection KO: {exc}")

    if stage == "pageprint" or stage == "audit_translation_selection":
        summary = {"tag": tag, "stage": stage, "translated_text_count": 0, "protected_region_count": len(input_data.get("regions") or []), "preserved_count": 0, "status": "pageprint_done"}
        emit("page_done", **summary)
        return summary

    result = build_page_translation(input_data, translator=engine, target_lang=args.target_lang, source_lang=args.source_lang, allow_fallback=True)
    tid = result["translated_input_data"]
    (out / f"translated_input_data_{tag}.json").write_text(json.dumps(tid, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"pagetranslate_{tag}.json").write_text(json.dumps({
        "statuses": {k: result.get(k) for k in ("pipeline_status", "translation_runtime_status", "linguistic_quality_status", "publication_readiness_status")},
        "units": [{
            "translation_unit_id": u.get("translation_unit_id"),
            "unit_id": u.get("unit_id"),
            "source_unit_ids": u.get("source_unit_ids"),
            "bbox": u.get("bbox") or (u.get("render_target") or {}).get("bbox"),
            "role": u.get("role"),
            "status": u.get("status"),
            "source_text": u.get("source_text"),
            "translated_text": u.get("translated_text"),
        } for u in result.get("translation_units") or []],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    if stage == "pagetranslate":
        summary = {"tag": tag, "stage": stage, "translated_text_count": len(result.get("translation_units") or []), "protected_region_count": len(input_data.get("regions") or []), "preserved_count": 0, "status": "pagetranslate_done"}
        emit("page_done", **summary)
        return summary

    _regen_clean_background(tid, out, tag)
    _write_background_compare(out, tag)

    if stage == "view_background":
        summary = {"tag": tag, "stage": stage, "translated_text_count": len(result.get("translation_units") or []), "protected_region_count": len(input_data.get("regions") or []), "preserved_count": 0, "status": "background_done"}
        emit("page_done", **summary)
        return summary

    plan = compile_page_render_plan(tid)
    plan_dict = plan.to_dict()
    (out / f"pagereconstruct_plan_{tag}.json").write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    if src_png:
        render_plan_overlay(tid, plan, src_png, out / f"pagereconstruct_overlay_{tag}.png")
        from pagereconstruct.render_backend import reconstruct_to_png
        reconstruct_to_png(plan_dict, str(src_png), str(out / f"reconstructed_{tag}.png"))
    try:
        from pagereconstruct.backends import pdf_vector
        if pdf_vector.is_available():
            pdf_vector.render(plan_dict, str(out / f"reconstructed_{tag}.pdf"))
    except Exception as exc:
        emit("warning", tag=tag, message=f"PDF vector KO: {exc}")
    audit = validate(plan_dict)
    (out / f"audit_{tag}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    if stage == "audit_text_survival":
        try:
            from tools.audit_text_survival import audit_text_survival, write_reports
            rep = audit_text_survival(tid, page_id=tag, plan=plan_dict)
            write_reports(rep, out)
            (out / f"audit_text_survival_{tag}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            emit("warning", tag=tag, message=f"audit_text_survival KO: {exc}")

    pubready_summary = None
    if stage == "full":
        try:
            from pubready import evaluate_reconstruction
            from pubready.reports import write_page_report
            rec_png = out / f"reconstructed_{tag}.png"
            rep = evaluate_reconstruction(
                tid,
                plan_dict,
                page_id=tag,
                mode=args.pubready_mode,
                source_image_path=str(src_png) if src_png and src_png.is_file() else None,
                reconstructed_image_path=str(rec_png) if rec_png.is_file() else None,
                out_dir=str(out / f"pubready_{tag}"),
            )
            write_page_report(rep, str(out))
            pubready_summary = {"score": rep.publication_ready_score, "status": rep.status, "publication_ready": rep.publication_ready, "hard_blockers": list(rep.hard_blockers)}
        except Exception as exc:
            pubready_summary = {"error": str(exc)}

    ps = plan.summary()
    preserved = (ps.get("preserved_overlay_count") or 0) + (ps.get("preserved_underlay_count") or 0)
    summary = {"tag": tag, "stage": stage, "translated_text_count": ps.get("translated_text_count"), "protected_region_count": ps.get("protected_region_count"), "preserved_count": preserved, "finding_count": ps.get("finding_count"), "status": audit.get("status"), "quality": audit.get("quality"), "pubready": pubready_summary}
    emit("page_done", **summary)
    return summary


def run_main(args) -> int:
    out = Path(args.out) if args.out else ROOT / "results" / f"demo_studio_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    emit("run_start", out=str(out), stage=args.stage)

    info = pdf_info(Path(args.pdf))
    pages = parse_pages(args.pages, info["page_count"])
    if args.random_count:
        rng = random.Random(args.seed)
        pages = sorted(rng.sample(range(1, info["page_count"] + 1), min(args.random_count, info["page_count"])))
    emit("selection", pdf=info, pages=pages)

    orchestrator = make_orchestrator(str(out / "_render"), enable_ocr=args.ocr)
    engine = make_engine(args.engine, model=args.model, source_lang=args.source_lang, target_lang=args.target_lang)
    summaries = []
    for i, page in enumerate(pages, 1):
        emit("progress", current=i, total=len(pages), message=f"Page {page}")
        try:
            summaries.append(process_page(orchestrator, engine, Path(args.pdf), page, args.stage, out, args))
        except Exception as exc:
            err = {"tag": f"{Path(args.pdf).stem[:24]}_p{page:04d}", "stage": args.stage, "error": str(exc)}
            summaries.append(err)
            emit("page_error", **err)
    (out / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(out, summaries)
    contact = _write_contact_sheet(out)
    emit("run_done", out=str(out), contact_sheet=str(contact) if contact else None, report=str(out / "report.md"), summary=str(out / "summary.json"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Backend Demo Studio docs_parser")
    ap.add_argument("--inspect-pdf", default=None)
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--pages", default="1")
    ap.add_argument("--random-count", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--stage", default="full", choices=sorted(STAGES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--engine", default="ct2")
    ap.add_argument("--model", default="opus_mt_tc_big_en_fr")
    ap.add_argument("--source-lang", default="en")
    ap.add_argument("--target-lang", default="fr")
    ap.add_argument("--pubready-mode", default="review", choices=["debug", "review", "publication"])
    ap.add_argument("--ocr", action="store_true")
    args = ap.parse_args(argv)
    if args.inspect_pdf:
        return inspect_pdf_main(Path(args.inspect_pdf))
    if not args.pdf:
        ap.error("--pdf est obligatoire")
    if args.stage not in STAGES:
        ap.error(f"stage inconnu: {args.stage}")
    return run_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
