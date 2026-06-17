#!/usr/bin/env python3
"""Run the full pipeline (PAGEPRINT -> PAGETRANSLATE -> PAGERECONSTRUCT plan)
on one or more pages and dump the artefacts for visual analysis.

Per page, in --out:
  source_<tag>.png              rendered source page
  pageprint_<tag>.json          PAGEPRINT input_data
  pagetranslate_<tag>.json      translation units (source<->translation, status)
  pagereconstruct_plan_<tag>.json  PageRenderPlan (Pass 1 — no PDF yet)
  pagereconstruct_overlay_<tag>.png  plan visualisation on the page

NOTE: PAGERECONSTRUCT is at Pass 1 (plan only). No PDF is rendered; the overlay
shows what the plan would render where.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from pagereconstruct import compile_page_render_plan
from pagetranslate import build_page_translation
from tools.run_pageprint_pagetranslate_audit import collect_pages, make_engine, make_orchestrator


def _scale(geom):
    return float(geom.get("scale_x_px_per_pt") or 1.0), float(geom.get("scale_y_px_per_pt") or 1.0)


def render_plan_overlay(input_data: dict, plan, image_path, out_path: Path) -> bool:
    if not image_path or not Path(image_path).is_file():
        return False
    sx, sy = _scale((input_data.get("page") or {}).get("geometry") or {})
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    def rect(b, outline, width=2, fill=None):
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            return
        xy = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
        if fill:
            draw.rectangle(xy, fill=fill)
        draw.rectangle(xy, outline=outline, width=width)

    for r in plan.protected_regions:            # red = protected (formula/image/code…)
        rect(r.bbox, (220, 20, 60, 255), 1, (220, 20, 60, 30))
    for p in plan.preserved_overlays:           # blue = preserved-as-text overlays
        rect(p.bbox, (30, 90, 220, 255), 2)
    for t in plan.translated_text:              # green = translated text to render
        rect(t.bbox, (26, 127, 55, 255), 2, (26, 127, 55, 35))
    img.save(out_path)
    return True


def _slim_pageprint(d: dict) -> dict:
    keep = {"block", "line", "phrase", "region", "table", "cell"}
    units = [{"unit_id": u.get("unit_id"), "level": u.get("level"), "parent_id": u.get("parent_id"),
              "bbox": (u.get("geometry") or {}).get("bbox"),
              "role": (u.get("understanding") or {}).get("role"),
              "object_type": (u.get("understanding") or {}).get("object_type"),
              "translatable": (u.get("policy") or {}).get("translatable"),
              "text": (u.get("content") or {}).get("text")}
             for u in d.get("units") or [] if u.get("level") in keep]
    v = d.get("views") or {}
    return {
        "schema_version": d.get("schema_version"), "page": d.get("page"),
        "page_intelligence": d.get("page_intelligence"),
        "semantic_system_counts": {k: (len(val) if isinstance(val, list) else val) for k, val in (d.get("semantic_system") or {}).items()},
        "logical_structures_counts": {k: len(val) for k, val in (d.get("logical_structures") or {}).items() if isinstance(val, list)},
        "views_counts": {k: (len(val) if isinstance(val, list) else "…") for k, val in v.items()},
        "units_block_line_phrase": units,
    }


def _regen_clean_background(tid: dict, out: Path, tag: str) -> None:
    """(Re)génère le fond propre avec le cleaner corrigé et le branche dans le tid
    (clean_background prioritaire au rendu, pas la source)."""
    try:
        from pipelines.background_cleaner import build_clean_background
        cp = build_clean_background(tid, out_path=str(out / f"cleanbg_{tag}.png"))
        if cp:
            tid.setdefault("visual_layers", {})["clean_background_path"] = cp
            tid.setdefault("assets", {})["background_clean_path"] = cp
    except Exception as exc:
        print(f"{tag}: clean_bg regen failed: {exc}", flush=True)


def process(orchestrator, engine, pdf: Path, page: int, out: Path, source_lang: str, target_lang: str,
            pubready_mode: str = "review", tid_cache: Path | None = None, reuse_tid: bool = False) -> dict:
    tag = f"{pdf.stem[:24]}_p{page:04d}"
    cache_file = (tid_cache / f"tid_{tag}.json") if tid_cache else None

    if reuse_tid and cache_file and cache_file.is_file():
        # tid GELÉ : pas de re-traduction (langue/texte/styles stables entre essais).
        tid = json.loads(cache_file.read_text(encoding="utf-8"))
        input_data = tid
        result = {}
        print(f"{tag}: tid réutilisé (gelé)", flush=True)
    else:
        doc = orchestrator.run(str(pdf), pages=str(page), language={"source_lang": source_lang, "target_lang": target_lang})
        ok = [p for p in (doc.get("pages") or []) if p.get("status") == "ok"]
        if not ok:
            print(f"{tag}: extraction KO", flush=True)
            return {"tag": tag, "error": "extraction_failed"}
        input_data = ok[0]["input_data"]
        result = build_page_translation(input_data, translator=engine, target_lang=target_lang, source_lang=source_lang, allow_fallback=True)
        tid = result["translated_input_data"]
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(tid, ensure_ascii=False), encoding="utf-8")

    # Fond propre corrigé, branché dans le tid AVANT compilation.
    _regen_clean_background(tid, out, tag)
    plan = compile_page_render_plan(tid)

    (out / f"pageprint_{tag}.json").write_text(json.dumps(_slim_pageprint(input_data), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"pagetranslate_{tag}.json").write_text(json.dumps({
        "statuses": {k: result.get(k) for k in ("pipeline_status", "translation_runtime_status", "linguistic_quality_status", "publication_readiness_status")},
        "units": [{"role": u.get("role"), "status": u.get("status"), "needs_review": (u.get("quality") or {}).get("needs_review"),
                   "source_text": u.get("source_text"), "translated_text": u.get("translated_text")}
                  for u in result.get("translation_units") or []],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"pagereconstruct_plan_{tag}.json").write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    plan_dict = plan.to_dict()
    img_path = (input_data.get("assets") or {}).get("source_image_path")
    if img_path and Path(img_path).is_file():
        Image.open(img_path).convert("RGB").save(out / f"source_{tag}.png")
        render_plan_overlay(input_data, plan, img_path, out / f"pagereconstruct_overlay_{tag}.png")
        from pagereconstruct.render_backend import reconstruct_to_png
        reconstruct_to_png(plan_dict, str(out / f"source_{tag}.png"), str(out / f"reconstructed_{tag}.png"))
    # Vector PDF output + validator audit.
    from pagereconstruct import validate
    from pagereconstruct.backends import pdf_vector
    if pdf_vector.is_available():
        pdf_vector.render(plan_dict, str(out / f"reconstructed_{tag}.pdf"))
    audit = validate(plan_dict)
    (out / f"audit_{tag}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # Évaluateur publication-ready (additif, ne touche pas le rendu).
    src_png = out / f"source_{tag}.png"
    rec_png = out / f"reconstructed_{tag}.png"
    pubready_summary = None
    try:
        from pubready import evaluate_reconstruction
        from pubready.reports import write_page_report
        has_imgs = src_png.is_file() and rec_png.is_file()
        rep = evaluate_reconstruction(
            tid, plan_dict, page_id=tag, mode=pubready_mode,
            source_image_path=str(src_png) if has_imgs else None,
            reconstructed_image_path=str(rec_png) if has_imgs else None,
            out_dir=str(out / f"pubready_{tag}"),
        )
        write_page_report(rep, str(out))
        pubready_summary = {"score": rep.publication_ready_score, "status": rep.status,
                            "publication_ready": rep.publication_ready,
                            "hard_blockers": list(rep.hard_blockers)}
    except Exception as exc:  # additif : ne bloque jamais le demo
        pubready_summary = {"error": str(exc)}

    summary = plan.summary()
    summary["tag"] = tag
    summary["status"] = audit["status"]
    summary["quality"] = audit["quality"]
    summary["pubready"] = pubready_summary
    pr = pubready_summary or {}
    print(f"{tag}: translated={summary['translated_text_count']} protected={summary['protected_region_count']} "
          f"preserved={summary['preserved_overlay_count']+summary['preserved_underlay_count']} findings={summary['finding_count']} "
          f"pubready={pr.get('score')}({pr.get('status')})", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=None, help="Specific PDF (else random from --pdf-dir)")
    parser.add_argument("--page", type=int, default=None, help="Specific page (1-based) when --pdf is given")
    parser.add_argument("--pdf-dir", default="tests/doc_pdf")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--min-pages", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--engine", default="ct2")
    parser.add_argument("--model", default="opus_mt_tc_big_en_fr")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="fr")
    parser.add_argument("--pubready-mode", default="review", choices=["debug", "review", "publication"])
    parser.add_argument("--tid-cache", default="results/_tid_cache", help="dossier de gel des translated_input_data")
    parser.add_argument("--reuse-tid", action="store_true", help="réutiliser le tid gelé (pas de re-traduction)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    orchestrator = make_orchestrator(str(out / "_render"), enable_ocr=False)
    engine = make_engine(args.engine, model=args.model, source_lang=args.source_lang, target_lang=args.target_lang)

    if args.pdf and args.page:
        picks = [(Path(args.pdf), args.page)]
    else:
        picks = collect_pages(Path(args.pdf_dir), args.count, args.seed, args.min_pages)

    summaries = []
    for pdf, page in picks:
        summaries.append(process(orchestrator, engine, Path(pdf), page, out, args.source_lang, args.target_lang,
                                  pubready_mode=args.pubready_mode,
                                  tid_cache=Path(args.tid_cache), reuse_tid=args.reuse_tid))
    (out / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Output:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
