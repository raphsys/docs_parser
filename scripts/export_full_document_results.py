#!/usr/bin/env python3
import argparse
import asyncio
import copy
import fcntl
import json
import os
import shutil
import sys
from pathlib import Path

import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ocr_server
from scripts.metadata_explorer_builder import build_metadata_explorer


def _ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _json_default(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _norm_bbox(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None


def _render_bbox_overlay_image(img, blocks, ai_regions=None):
    img_draw = img.copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img_draw)
    for block in blocks or []:
        bbox = block.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            draw.rectangle(bbox, outline="blue", width=3)
        for line in block.get("lines", []) or []:
            lb = line.get("bbox")
            if isinstance(lb, (list, tuple)) and len(lb) == 4:
                draw.rectangle(lb, outline="green", width=1)
            for phrase in line.get("phrases", []) or []:
                pb = phrase.get("bbox")
                if isinstance(pb, (list, tuple)) and len(pb) == 4:
                    draw.rectangle(pb, outline="red", width=1)
    for region in ai_regions or []:
        bbox = region.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            draw.rectangle(bbox, outline="orange", width=2)
    return img_draw
    try:
        return [float(v) for v in bbox]
    except Exception:
        return None


def _process_pdf(pdf_path: Path, page_limit=None, page_indices=None):
    doc = fitz.open(pdf_path)
    pages = []
    try:
        allowed = set(page_indices or [])
        for idx, page in enumerate(doc):
            if allowed and idx not in allowed:
                continue
            if page_limit is not None and len(pages) >= page_limit:
                break
            pix = page.get_pixmap(dpi=150, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_data = ocr_server.process_page(img, idx, pdf_path.name, pdf_page=page)
            pages.append(page_data)
            if (len(pages) % 25) == 0:
                print(f"[export] extracted {len(pages)} selected pages", flush=True)
        return pages
    finally:
        doc.close()


def _write_classifier_outputs(pages, out_dir: Path):
    page_entries = []
    family_counts = {}
    layout_counts = {}
    doc_type_counts = {}
    style_counts = {}
    role_counts = {}

    for page in pages:
        entry = {
            "page": page.get("page"),
            "page_role": page.get("page_role"),
            "document_type": page.get("document_type"),
            "layout_type": page.get("layout_type"),
            "style_profile": page.get("style_profile"),
            "page_family": page.get("page_family"),
            "page_family_group": page.get("page_family_group"),
            "page_case": page.get("page_case"),
        }
        page_entries.append(entry)
        for counter, key in (
            (family_counts, entry["page_family"] or "unknown"),
            (layout_counts, entry["layout_type"] or "unknown"),
            (doc_type_counts, entry["document_type"] or "unknown"),
            (style_counts, entry["style_profile"] or "unknown"),
            (role_counts, entry["page_role"] or "unknown"),
        ):
            counter[key] = counter.get(key, 0) + 1

    summary = {
        "document": str(args.pdf),
        "page_count": len(page_entries),
        "counts": {
            "page_family": family_counts,
            "layout_type": layout_counts,
            "document_type": doc_type_counts,
            "style_profile": style_counts,
            "page_role": role_counts,
        },
        "pages": page_entries,
    }
    (out_dir / "document_classifier_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for entry in page_entries:
        (out_dir / f"page_{int(entry['page']):04d}.json").write_text(
            json.dumps(entry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _write_descriptor_outputs(pages, out_dir: Path):
    summary = {
        "document": str(args.pdf),
        "page_count": len(pages),
        "pages": [],
    }
    for page in pages:
        structure = page.get("structure") or {}
        descriptor = (
            structure.get("layout_descriptor")
            or (structure.get("layout") or {}).get("layout_descriptor")
            or {}
        )
        page_num = int(page.get("page") or 0)
        (out_dir / f"page_{page_num:04d}_descriptor.json").write_text(
            json.dumps(descriptor, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["pages"].append(
            {
                "page": page_num,
                "descriptor_version": descriptor.get("descriptor_version"),
                "region_count": len(descriptor.get("regions") or []),
                "element_count": len(descriptor.get("elements") or []),
                "group_count": len(descriptor.get("groups") or []),
                "relation_count": len(descriptor.get("relations") or []),
                "constraint_count": len(descriptor.get("constraints") or []),
            }
        )
    (out_dir / "document_descriptor_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_page_payloads(pages, out_path: Path):
    normalized_pages = []
    for page_idx, page in enumerate(pages or []):
        if not isinstance(page, dict):
            normalized_pages.append(page)
            continue
        structure = page.get("structure")
        if isinstance(structure, dict):
            normalized = copy.deepcopy(structure)
            if normalized.get("page") in {None, ""}:
                normalized["page"] = page.get("page") if page.get("page") not in {None, ""} else (page_idx + 1)
            if normalized.get("page_index") in {None, ""}:
                normalized["page_index"] = page_idx
            normalized_pages.append(normalized)
            continue
        normalized = copy.deepcopy(page)
        if isinstance(normalized, dict):
            if normalized.get("page") in {None, ""}:
                normalized["page"] = page_idx + 1
            if normalized.get("page_index") in {None, ""}:
                normalized["page_index"] = page_idx
        normalized_pages.append(normalized)
    out_path.write_text(
        json.dumps({"pages": normalized_pages}, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _copy_source_page_images(pages, out_dir: Path):
    for idx, page in enumerate(pages or []):
        src = str((page or {}).get("source_image_path") or "").strip()
        if not src:
            continue
        src_path = Path(src)
        if not src_path.exists():
            continue
        page_num = int((page or {}).get("page") or (idx + 1))
        shutil.copy2(src_path, out_dir / f"page_{page_num:03d}_original.png")


def _render_reconstructed_page_images(pdf_path: Path, out_dir: Path):
    if not pdf_path.exists():
        return
    doc = fitz.open(pdf_path)
    try:
        for idx in range(len(doc)):
            pix = doc[idx].get_pixmap(dpi=150, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img.save(out_dir / f"page_{idx + 1:03d}_translated.png")
    finally:
        doc.close()


def _write_side_by_side_images(out_dir: Path):
    originals = sorted(out_dir.glob("page_*_original.png"))
    for original in originals:
        translated = out_dir / original.name.replace("_original.png", "_translated.png")
        if not translated.exists():
            continue
        src_img = Image.open(original).convert("RGB")
        rec_img = Image.open(translated).convert("RGB")
        w = max(src_img.width, rec_img.width)
        h = max(src_img.height, rec_img.height)
        src_canvas = Image.new("RGB", (w, h), "white")
        rec_canvas = Image.new("RGB", (w, h), "white")
        src_canvas.paste(src_img, (0, 0))
        rec_canvas.paste(rec_img, (0, 0))
        side = Image.new("RGB", (w * 2, h), "white")
        side.paste(src_canvas, (0, 0))
        side.paste(rec_canvas, (w, 0))
        side.save(out_dir / original.name.replace("_original.png", "_side_by_side.png"))


def _write_extraction_overlay_pdf(pdf_path: Path, pages, out_pdf: Path):
    src = fitz.open(pdf_path)
    out = fitz.open()
    try:
        for idx in range(min(len(src), len(pages))):
            page_payload = pages[idx]
            src_page_idx = int(page_payload.get("page") or (idx + 1)) - 1
            if src_page_idx < 0 or src_page_idx >= len(src):
                continue
            src_page = src[src_page_idx]
            pix = src_page.get_pixmap(dpi=150, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            structure = (page_payload.get("structure") or {})
            overlay_img = _render_bbox_overlay_image(
                img,
                structure.get("blocks") or [],
                (structure.get("ai_layout_regions") or []),
            )
            overlay_tmp = out_pdf.parent / f"_overlay_{src_page_idx + 1:04d}.png"
            overlay_img.save(overlay_tmp)
            overlay_img.save(out_pdf.parent / f"vis_page_{src_page_idx + 1:04d}.jpg")
            dst_page = out.new_page(width=src_page.rect.width, height=src_page.rect.height)
            dst_page.insert_image(dst_page.rect, filename=str(overlay_tmp), overlay=True, keep_proportion=False)
            dst_page.insert_text(
                (18, 18),
                f"page {src_page_idx + 1} | blue=block green=line red=phrase orange=ai_region",
                fontsize=8,
                fontname="helv",
                color=(0, 0, 0),
            )
        out.save(out_pdf)
    finally:
        for tmp in out_pdf.parent.glob("_overlay_*.png"):
            try:
                tmp.unlink()
            except Exception:
                pass
        out.close()
        src.close()


def _clear_previous_reconstruct_artifacts():
    results_dir = ROOT / "ocr_results"
    for pattern in (
        "reconstructed_output.pdf",
        "reconstructed_output_style_audit.json",
        "reconstructed_output_layout_debug_p*.jpg",
    ):
        for path in results_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


class _ReconstructLock:
    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None


async def _write_reconstructed_outputs(pages, out_dir: Path):
    lock_path = ROOT / "ocr_results" / "reconstructed_output.lock"
    with _ReconstructLock(lock_path):
        _clear_previous_reconstruct_artifacts()
        response = await ocr_server.reconstruct_document(
            {"pages": pages},
            target_lang=args.target_lang,
            debug_compare=True,
            include_debug_pages=True,
        )
        payload = json.loads(response.body.decode("utf-8"))
        src_pdf = ROOT / "ocr_results" / "reconstructed_output.pdf"
        dst_pdf = out_dir / f"{args.target_lang}_translated_reconstructed.pdf"
        if src_pdf.exists():
            shutil.copy2(src_pdf, dst_pdf)
        src_audit = ROOT / "ocr_results" / f"{src_pdf.stem}_style_audit.json"
        if src_audit.exists():
            shutil.copy2(src_audit, out_dir / f"{dst_pdf.stem}_style_audit.json")
        for debug_img in (ROOT / "ocr_results").glob(f"{src_pdf.stem}_layout_debug_p*.jpg"):
            shutil.copy2(debug_img, out_dir / debug_img.name.replace(src_pdf.stem, dst_pdf.stem, 1))

    for name in ("coverage_report", "publication_qa", "visual_compare"):
        if payload.get(name) is not None:
            (out_dir / f"{name}.json").write_text(
                json.dumps(payload.get(name), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return payload


async def main():
    base_name = Path(args.pdf).stem
    suffix = args.output_suffix or "full_results"
    export_root = ROOT / "results" / f"{base_name}_{suffix}"
    extraction_dir = export_root / "extraction_bboxes"
    classifier_dir = export_root / "classifier"
    descriptor_dir = export_root / "descriptors"
    payloads_dir = export_root / "payloads"
    reconstructed_dir = export_root / "reconstructed"

    _ensure_clean_dir(export_root)
    extraction_dir.mkdir(parents=True, exist_ok=True)
    classifier_dir.mkdir(parents=True, exist_ok=True)
    descriptor_dir.mkdir(parents=True, exist_ok=True)
    payloads_dir.mkdir(parents=True, exist_ok=True)
    reconstructed_dir.mkdir(parents=True, exist_ok=True)

    print("[export] starting extraction", flush=True)
    selected_indices = None
    if args.page_indices:
        selected_indices = sorted(
            {
                max(0, int(v.strip()))
                for v in args.page_indices.split(",")
                if str(v).strip()
            }
        )
    pages = _process_pdf(ROOT / args.pdf, page_limit=args.page_limit, page_indices=selected_indices)
    print(f"[export] extraction done: {len(pages)} pages", flush=True)

    print("[export] writing extraction bbox pdf", flush=True)
    _write_extraction_overlay_pdf(ROOT / args.pdf, pages, extraction_dir / "extraction_bboxes.pdf")
    print("[export] writing classifier outputs", flush=True)
    _write_classifier_outputs(pages, classifier_dir)
    print("[export] writing descriptor outputs", flush=True)
    _write_descriptor_outputs(pages, descriptor_dir)
    print("[export] reconstructing translated pdf", flush=True)
    payload = await _write_reconstructed_outputs(pages, reconstructed_dir)
    print("[export] writing extracted payloads", flush=True)
    source_pages = payload.get("source_pages")
    if isinstance(source_pages, list):
        _write_page_payloads(source_pages, payloads_dir / "source_pages.json")
        _copy_source_page_images(source_pages, export_root)
    else:
        extracted_pages = copy.deepcopy(pages)
        _write_page_payloads(extracted_pages, payloads_dir / "source_pages.json")
        _copy_source_page_images(extracted_pages, export_root)
    print("[export] writing translated payloads", flush=True)
    translated_pages = payload.get("translated_pages")
    if isinstance(translated_pages, list):
        _write_page_payloads(translated_pages, payloads_dir / "translated_pages.json")
    else:
        _write_page_payloads(pages, payloads_dir / "translated_pages.json")
    _render_reconstructed_page_images(reconstructed_dir / f"{args.target_lang}_translated_reconstructed.pdf", export_root)
    _write_side_by_side_images(export_root)
    build_metadata_explorer(export_root)
    print("[export] reconstruction done", flush=True)

    manifest = {
        "document": args.pdf,
        "page_count": len(pages),
        "page_numbers": [int(p.get("page") or 0) for p in pages],
        "folders": {
            "extraction_bboxes": str(extraction_dir.relative_to(ROOT)),
            "classifier": str(classifier_dir.relative_to(ROOT)),
            "descriptors": str(descriptor_dir.relative_to(ROOT)),
            "payloads": str(payloads_dir.relative_to(ROOT)),
            "reconstructed": str(reconstructed_dir.relative_to(ROOT)),
        },
    }
    (export_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export full-document extraction/classification/descriptors/reconstruction results.")
    parser.add_argument("--pdf", required=True, help="PDF path relative to repository root")
    parser.add_argument("--target-lang", default="fr", help="Target language code")
    parser.add_argument("--page-limit", type=int, default=None, help="Optional max number of pages to process")
    parser.add_argument("--page-indices", default="", help="Optional comma-separated 0-based page indices to process")
    parser.add_argument("--output-suffix", default="", help="Suffix for results folder name")
    args = parser.parse_args()
    asyncio.run(main())
