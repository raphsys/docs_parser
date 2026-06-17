#!/usr/bin/env python3
"""Local non-web demo runner for docs_parser.

This script is the command-line backend used by tools/local_demo_app.py.
It can run individual pipeline levels on explicit PDF pages and writes all
artefacts into results/<run_name>/.

Stages:
  pageprint                    PDF -> PAGEPRINT INPUT_DATA + bbox visuals
  pagetranslate                PAGEPRINT -> PAGETRANSLATE + translated_input_data
  pagereconstruct              PAGEPRINT -> PAGETRANSLATE -> PAGERECONSTRUCT plan + PNG/PDF
  view_background              PAGEPRINT/PAGETRANSLATE -> clean background inspection
  audit_translation_selection  PAGEPRINT + translation-selection audit
  audit_text_survival          Full enough to audit text survival
  full                         All of the above useful artefacts
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - handled at runtime
    fitz = None

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - handled at runtime
    Image = None
    ImageDraw = None


@dataclass
class PageRun:
    pdf: Path
    page: int
    tag: str
    started: float
    input_data: dict[str, Any] | None = None
    translation_result: dict[str, Any] | None = None
    tid: dict[str, Any] | None = None
    plan: Any | None = None
    plan_dict: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None


def slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "")).strip("_")
    return (s or "doc")[:limit]


def parse_pages(spec: str) -> list[int]:
    pages: set[int] = set()
    for part in re.split(r"[,;\s]+", str(spec or "").strip()):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
            if start > end:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return sorted(p for p in pages if p > 0)


def page_count(pdf: Path) -> int:
    if fitz is None:
        return 0
    with fitz.open(pdf) as doc:
        return int(doc.page_count)


def validate_pages(pdf: Path, pages: Iterable[int]) -> list[int]:
    n = page_count(pdf)
    if n <= 0:
        return list(pages)
    return [p for p in pages if 1 <= p <= n]


def page_tag(pdf: Path, page: int) -> str:
    return f"{slug(pdf.stem, 32)}_p{page:04d}"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_if_exists(src: str | os.PathLike[str] | None, dst: Path) -> bool:
    if not src:
        return False
    p = Path(src)
    if not p.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    return True


def make_contact_sheet(out_dir: Path) -> Path | None:
    if Image is None or ImageDraw is None:
        return None
    tags = sorted({p.stem.replace("source_", "") for p in out_dir.glob("source_*.png")})
    if not tags:
        return None

    columns = [
        ("source", lambda tag: out_dir / f"source_{tag}.png"),
        ("cleanbg", lambda tag: out_dir / f"cleanbg_{tag}.png"),
        ("bboxes", lambda tag: out_dir / f"pageprint_bboxes_{tag}.png"),
        ("overlay", lambda tag: out_dir / f"pagereconstruct_overlay_{tag}.png"),
        ("reconstructed", lambda tag: out_dir / f"reconstructed_{tag}.png"),
    ]
    thumb_w, thumb_h = 260, 360
    pad_x, pad_y = 18, 44
    header_h = 42
    row_h = thumb_h + header_h + 12
    width = len(columns) * (thumb_w + pad_x) + pad_x
    height = len(tags) * row_h + pad_y
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    y = pad_y // 2
    for tag in tags:
        draw.text((pad_x, y), tag, fill="black")
        x = pad_x
        for label, fn in columns:
            p = fn(tag)
            draw.text((x, y + 18), label, fill="black")
            if p.is_file():
                im = Image.open(p).convert("RGB")
                im.thumbnail((thumb_w, thumb_h))
                canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
                canvas.paste(im, ((thumb_w - im.width) // 2, 0))
            else:
                canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
                d = ImageDraw.Draw(canvas)
                d.rectangle([0, 0, thumb_w - 1, thumb_h - 1], outline=(210, 210, 210))
                d.text((8, 8), "absent", fill=(90, 90, 90))
            sheet.paste(canvas, (x, y + header_h))
            x += thumb_w + pad_x
        y += row_h

    out = out_dir / "contact_sheet.jpg"
    sheet.save(out, quality=92)
    return out


def render_background_compare(out_dir: Path, tag: str) -> Path | None:
    if Image is None or ImageDraw is None:
        return None
    src = out_dir / f"source_{tag}.png"
    bg = out_dir / f"cleanbg_{tag}.png"
    if not (src.is_file() and bg.is_file()):
        return None
    images = []
    for label, path in [("source", src), ("clean_background", bg)]:
        im = Image.open(path).convert("RGB")
        im.thumbnail((520, 720))
        canvas = Image.new("RGB", (520, im.height + 28), "white")
        d = ImageDraw.Draw(canvas)
        d.text((5, 6), label, fill="black")
        canvas.paste(im, ((520 - im.width) // 2, 28))
        images.append(canvas)
    h = max(i.height for i in images)
    sheet = Image.new("RGB", (images[0].width + images[1].width + 20, h), "white")
    sheet.paste(images[0], (0, 0))
    sheet.paste(images[1], (images[0].width + 20, 0))
    out = out_dir / f"background_compare_{tag}.jpg"
    sheet.save(out, quality=92)
    return out


def extract_page(run: PageRun, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    from tools.run_pageprint_pagetranslate_audit import make_orchestrator, render_bboxes, render_background

    orchestrator = make_orchestrator(str(out_dir / "_render"), enable_ocr=args.enable_ocr)
    doc = orchestrator.run(
        str(run.pdf),
        pages=str(run.page),
        language={"source_lang": args.source_lang, "target_lang": args.target_lang},
    )
    ok_pages = [p for p in (doc.get("pages") or []) if p.get("status") == "ok"]
    if not ok_pages:
        raise RuntimeError(f"PAGEPRINT extraction failed for {run.pdf.name} p{run.page}: {doc}")

    run.input_data = ok_pages[0]["input_data"]
    write_json(out_dir / f"pageprint_{run.tag}.json", run.input_data)

    img_path = (run.input_data.get("assets") or {}).get("source_image_path")
    copy_if_exists(img_path, out_dir / f"source_{run.tag}.png")
    try:
        render_bboxes(run.input_data, Path(img_path) if img_path else None, out_dir / f"pageprint_bboxes_{run.tag}.png")
        render_background(run.input_data, Path(img_path) if img_path else None, out_dir / f"background_{run.tag}.png")
    except Exception as exc:
        print(f"{run.tag}: pageprint visualisation warning: {exc}", flush=True)

    return run.input_data


def translate_page(run: PageRun, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if run.input_data is None:
        extract_page(run, out_dir, args)

    from pagetranslate import build_page_translation
    from tools.run_pageprint_pagetranslate_audit import make_engine

    cache_file = Path(args.tid_cache) / f"translated_input_data_{run.tag}.json"
    if args.reuse_tid and cache_file.is_file():
        tid = json.loads(cache_file.read_text(encoding="utf-8"))
        run.tid = tid
        run.translation_result = {"translated_input_data": tid, "translation_units": []}
    else:
        engine = make_engine(args.engine, model=args.model, source_lang=args.source_lang, target_lang=args.target_lang)
        result = build_page_translation(
            run.input_data,
            translator=engine,
            target_lang=args.target_lang,
            source_lang=args.source_lang,
            allow_fallback=True,
        )
        run.translation_result = result
        run.tid = result.get("translated_input_data") or run.input_data
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(cache_file, run.tid)

    units = (run.translation_result or {}).get("translation_units") or []
    write_json(out_dir / f"pagetranslate_{run.tag}.json", {
        "tag": run.tag,
        "source_pdf": str(run.pdf),
        "page": run.page,
        "statuses": {k: (run.translation_result or {}).get(k) for k in (
            "pipeline_status", "translation_runtime_status", "linguistic_quality_status", "publication_readiness_status"
        )},
        "unit_count": len(units),
        "units": units,
    })
    write_json(out_dir / f"translated_input_data_{run.tag}.json", run.tid)
    return run.tid


def build_background(run: PageRun, out_dir: Path, args: argparse.Namespace) -> None:
    if run.tid is None:
        translate_page(run, out_dir, args)
    try:
        from pipelines.background_cleaner import build_clean_background
        cp = build_clean_background(run.tid, out_path=str(out_dir / f"cleanbg_{run.tag}.png"))
        if cp:
            run.tid.setdefault("visual_layers", {})["clean_background_path"] = cp
            run.tid.setdefault("assets", {})["background_clean_path"] = cp
            write_json(out_dir / f"translated_input_data_{run.tag}.json", run.tid)
            render_background_compare(out_dir, run.tag)
    except Exception as exc:
        print(f"{run.tag}: background cleaning warning: {exc}", flush=True)


def reconstruct_page(run: PageRun, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if run.tid is None:
        translate_page(run, out_dir, args)
    build_background(run, out_dir, args)

    from pagereconstruct import compile_page_render_plan, validate
    from tools.run_pipeline_full_demo import render_plan_overlay

    run.plan = compile_page_render_plan(run.tid, reconstruction_mode=args.reconstruction_mode)
    run.plan_dict = run.plan.to_dict()
    write_json(out_dir / f"pagereconstruct_plan_{run.tag}.json", run.plan_dict)

    img_path = (run.input_data or run.tid or {}).get("assets", {}).get("source_image_path")
    if not img_path:
        img_path = out_dir / f"source_{run.tag}.png"
    if img_path and Path(img_path).is_file():
        try:
            copy_if_exists(img_path, out_dir / f"source_{run.tag}.png")
            render_plan_overlay(run.input_data or run.tid, run.plan, img_path, out_dir / f"pagereconstruct_overlay_{run.tag}.png")
        except Exception as exc:
            print(f"{run.tag}: reconstruct overlay warning: {exc}", flush=True)

    try:
        from pagereconstruct.render_backend import reconstruct_to_png
        source_png = out_dir / f"source_{run.tag}.png"
        if source_png.is_file():
            reconstruct_to_png(run.plan_dict, str(source_png), str(out_dir / f"reconstructed_{run.tag}.png"))
    except Exception as exc:
        print(f"{run.tag}: raster reconstruction warning: {exc}", flush=True)

    try:
        from pagereconstruct.backends import pdf_vector
        if pdf_vector.is_available():
            pdf_vector.render(run.plan_dict, str(out_dir / f"reconstructed_{run.tag}.pdf"))
    except Exception as exc:
        print(f"{run.tag}: PDF vector reconstruction warning: {exc}", flush=True)

    audit = validate(run.plan_dict)
    write_json(out_dir / f"audit_{run.tag}.json", audit)
    return run.plan_dict


def audit_translation_selection(run: PageRun, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if run.input_data is None:
        extract_page(run, out_dir, args)
    try:
        from tools.audit_translation_selection import audit_page
        result = audit_page(run.input_data)
    except Exception as exc:
        result = {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
    write_json(out_dir / f"audit_translation_selection_{run.tag}.json", result)
    return result


def audit_text_survival(run: PageRun, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if run.plan_dict is None:
        reconstruct_page(run, out_dir, args)
    try:
        from tools.audit_text_survival import audit_text_survival as audit_fn
        result = audit_fn(run.tid or {}, page_id=run.tag, mode=args.reconstruction_mode, plan=run.plan_dict)
    except Exception as exc:
        result = {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
    write_json(out_dir / f"audit_text_survival_{run.tag}.json", result)
    return result


def evaluate_pubready(run: PageRun, out_dir: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    if run.plan_dict is None:
        return None
    try:
        from pubready import evaluate_reconstruction
        from pubready.reports import write_page_report
        src_png = out_dir / f"source_{run.tag}.png"
        rec_png = out_dir / f"reconstructed_{run.tag}.png"
        has_imgs = src_png.is_file() and rec_png.is_file()
        rep = evaluate_reconstruction(
            run.tid or {},
            run.plan_dict,
            page_id=run.tag,
            mode=args.pubready_mode,
            source_image_path=str(src_png) if has_imgs else None,
            reconstructed_image_path=str(rec_png) if has_imgs else None,
            out_dir=str(out_dir / f"pubready_{run.tag}"),
        )
        write_page_report(rep, str(out_dir))
        return {
            "score": rep.publication_ready_score,
            "status": rep.status,
            "publication_ready": rep.publication_ready,
            "hard_blockers": list(rep.hard_blockers),
        }
    except Exception as exc:
        return {"error": str(exc)}


def run_one(pdf: Path, page: int, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    run = PageRun(pdf=pdf, page=page, tag=page_tag(pdf, page), started=time.perf_counter())
    print(f"\n=== {run.tag} | stage={args.stage} ===", flush=True)
    result: dict[str, Any] = {"tag": run.tag, "pdf": str(pdf), "page": page, "stage": args.stage}

    try:
        if args.stage == "pageprint":
            extract_page(run, out_dir, args)
        elif args.stage == "pagetranslate":
            translate_page(run, out_dir, args)
        elif args.stage == "view_background":
            translate_page(run, out_dir, args)
            build_background(run, out_dir, args)
        elif args.stage == "pagereconstruct":
            reconstruct_page(run, out_dir, args)
        elif args.stage == "audit_translation_selection":
            audit_translation_selection(run, out_dir, args)
        elif args.stage == "audit_text_survival":
            audit_text_survival(run, out_dir, args)
        elif args.stage == "full":
            reconstruct_page(run, out_dir, args)
            audit_translation_selection(run, out_dir, args)
            audit_text_survival(run, out_dir, args)
            result["pubready"] = evaluate_pubready(run, out_dir, args)
        else:
            raise ValueError(f"unknown stage: {args.stage}")

        if run.translation_result is not None:
            units = run.translation_result.get("translation_units") or []
            result["translation_units"] = len(units)
            result["translated_non_empty"] = sum(1 for u in units if (u.get("translated_text") or "").strip())
        if run.plan is not None:
            try:
                result.update(run.plan.summary())
            except Exception:
                pass
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        write_json(out_dir / f"error_{run.tag}.json", result)
        print(result["traceback"], flush=True)

    result["duration_s"] = round(time.perf_counter() - run.started, 2)
    print(f"{run.tag}: {result.get('status')} duration={result['duration_s']}s", flush=True)
    return result


def write_report(out_dir: Path, summary: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = []
    lines.append("# Rapport démo locale docs_parser")
    lines.append("")
    lines.append(f"Date: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"Stage: `{args.stage}`")
    lines.append(f"PDF: `{args.pdf}`")
    lines.append(f"Pages: `{args.pages}`")
    lines.append(f"Dossier: `{out_dir}`")
    lines.append("")
    lines.append("## Résumé")
    lines.append("")
    lines.append("| tag | status | translated | protected | preserved | findings | duration_s |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in summary:
        preserved = (row.get("preserved_overlay_count") or 0) + (row.get("preserved_underlay_count") or 0)
        lines.append(
            f"| `{row.get('tag')}` | {row.get('status')} | {row.get('translated_text_count', row.get('translated_non_empty', ''))} | "
            f"{row.get('protected_region_count', '')} | {preserved} | {row.get('finding_count', '')} | {row.get('duration_s', '')} |"
        )
    lines.append("")
    lines.append("## Artefacts produits")
    lines.append("")
    lines.append("- `source_*.png` : image source")
    lines.append("- `pageprint_*.json` : sortie PAGEPRINT complète")
    lines.append("- `pageprint_bboxes_*.png` : bboxes PAGEPRINT")
    lines.append("- `pagetranslate_*.json` : unités traduites")
    lines.append("- `translated_input_data_*.json` : entrée traduite consommée par PAGERECONSTRUCT")
    lines.append("- `cleanbg_*.png` et `background_compare_*.jpg` : fond nettoyé")
    lines.append("- `pagereconstruct_plan_*.json` : plan de reconstruction")
    lines.append("- `pagereconstruct_overlay_*.png` : overlay du plan")
    lines.append("- `reconstructed_*.png/.pdf` : rendu final")
    lines.append("- `audit_*.json`, `audit_text_survival_*.json`, `audit_translation_selection_*.json` : audits")
    lines.append("- `contact_sheet.jpg` : vue comparative globale")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="PDF source")
    parser.add_argument("--pages", required=True, help="Pages 1-based: '1,3-5'")
    parser.add_argument("--stage", default="full", choices=[
        "pageprint", "pagetranslate", "pagereconstruct", "view_background",
        "audit_translation_selection", "audit_text_survival", "full",
    ])
    parser.add_argument("--out", default=None, help="Output directory. Default: results/local_demo_<timestamp>")
    parser.add_argument("--engine", default="ct2")
    parser.add_argument("--model", default="opus_mt_tc_big_en_fr")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="fr")
    parser.add_argument("--pubready-mode", default="review", choices=["debug", "review", "publication"])
    parser.add_argument("--reconstruction-mode", default="debug")
    parser.add_argument("--tid-cache", default="results/_tid_cache")
    parser.add_argument("--reuse-tid", action="store_true")
    parser.add_argument("--enable-ocr", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    pdf = Path(args.pdf).expanduser().resolve()
    if not pdf.is_file():
        print(f"ERREUR: PDF introuvable: {pdf}", file=sys.stderr)
        return 2

    pages = validate_pages(pdf, parse_pages(args.pages))
    if not pages:
        print("ERREUR: aucune page valide sélectionnée", file=sys.stderr)
        return 2

    out_dir = Path(args.out or (ROOT / "results" / f"local_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / "run_config.json", vars(args) | {"pdf_resolved": str(pdf), "pages_resolved": pages})
    print(f"Output: {out_dir}", flush=True)
    print(f"PDF: {pdf}", flush=True)
    print(f"Pages: {pages}", flush=True)

    summary: list[dict[str, Any]] = []
    for page in pages:
        row = run_one(pdf, page, out_dir, args)
        summary.append(row)
        write_json(out_dir / "summary.json", summary)
        if args.fail_fast and row.get("status") != "ok":
            break

    contact = make_contact_sheet(out_dir)
    if contact:
        print(f"contact_sheet: {contact}", flush=True)
    write_report(out_dir, summary, args)
    print(f"report: {out_dir / 'report.md'}", flush=True)
    print(f"summary: {out_dir / 'summary.json'}", flush=True)
    return 0 if all(r.get("status") == "ok" for r in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
