import os
import shutil
import re
import uuid
import glob
import subprocess
import uvicorn
import fitz
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from rapidocr_onnxruntime import RapidOCR
from structure_extractor import DocumentParser
from reconstructor import DocumentReconstructor
from remove_text_generic import inpaint_opencv
from layout_optimizer import LayoutOptimizer
from font_ai_matcher import FontAIMatcher
from text_removal_strategy import TextRemovalStrategy
from native_pdf_extractor import NativePDFExtractor

# --- Configuration ---
MODEL_TEXT_PATH = './ai_models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf'
UPLOAD_DIR, CONV_DIR, RESULTS_DIR = 'uploads', 'converted_pages', 'ocr_results'
TARGET_DPI = 150 
FONT_AI_ENABLED = os.getenv("FONT_AI_ENABLED", "1") == "1"
FONT_AI_SCORE_THRESHOLD = float(os.getenv("FONT_AI_SCORE_THRESHOLD", "0.45"))
FONT_AI_AUDIT_DEFAULT = os.getenv("FONT_AI_AUDIT", "0") == "1"
OFFICE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".odt", ".odp"}

app = FastAPI(title="IA Document OCR - Stable Precision")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

for d in [UPLOAD_DIR, CONV_DIR, RESULTS_DIR]:
    if not os.path.exists(d): os.makedirs(d)

engine_ocr = RapidOCR()
parser = DocumentParser()
layout_optimizer = LayoutOptimizer()
font_ai_matcher = FontAIMatcher() if FONT_AI_ENABLED else None
text_removal_strategy = TextRemovalStrategy()
native_pdf_extractor = NativePDFExtractor()
_translator_instance = None


def get_translator():
    global _translator_instance
    if _translator_instance is not None:
        return _translator_instance
    try:
        from translator import DocumentTranslator
    except Exception as e:
        raise RuntimeError(f"Impossible de charger le module de traduction: {e}") from e
    _translator_instance = DocumentTranslator(MODEL_TEXT_PATH)
    return _translator_instance


def _find_office_binary():
    for cand in ("soffice", "libreoffice"):
        path = shutil.which(cand)
        if path:
            return path
    return None


def _convert_office_to_pdf(input_path):
    office_bin = _find_office_binary()
    if not office_bin:
        raise RuntimeError("Conversion Office indisponible: binaire 'soffice/libreoffice' introuvable")

    convert_dir = os.path.join(CONV_DIR, uuid.uuid4().hex)
    os.makedirs(convert_dir, exist_ok=True)
    profile_dir = os.path.join("/tmp", f"lo_profile_{uuid.uuid4().hex}")
    os.makedirs(profile_dir, exist_ok=True)
    cmd = [
        office_bin,
        "--headless",
        f"-env:UserInstallation=file://{profile_dir}",
        "--convert-to",
        "pdf:writer_pdf_Export",
        "--outdir",
        convert_dir,
        input_path,
    ]
    env = dict(os.environ)
    env.setdefault("HOME", "/tmp")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)

    base = os.path.splitext(os.path.basename(input_path))[0]
    expected = os.path.join(convert_dir, f"{base}.pdf")
    if os.path.exists(expected):
        return expected
    # LibreOffice peut normaliser légèrement le nom de sortie.
    candidates = sorted(glob.glob(os.path.join(convert_dir, "*.pdf")))
    if candidates:
        return candidates[0]

    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    details = stderr or stdout or "aucun détail"
    if proc.returncode != 0:
        raise RuntimeError(f"Echec conversion Office->PDF: aucun PDF généré ({details})")
    raise RuntimeError(f"Echec conversion Office->PDF: aucun PDF généré ({details})")


def _norm_text(s):
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _token_set(s):
    s = _norm_text(s)
    return {t for t in re.split(r"[^a-z0-9]+", s) if t}

def _text_sim(a, b):
    ta = _token_set(a)
    tb = _token_set(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / max(1, union)


def _block_text(block):
    parts = []
    for line in block.get("lines", []):
        for phrase in line.get("phrases", []):
            t = (phrase.get("texte") or "").strip()
            if t:
                parts.append(t)
    return _norm_text(" ".join(parts))


def _bbox_overlap_ratio(b1, b2):
    r1 = fitz.Rect(b1)
    r2 = fitz.Rect(b2)
    inter = (r1 & r2).get_area()
    if inter <= 0:
        return 0.0
    return inter / max(1e-9, min(r1.get_area(), r2.get_area()))


def _dedupe_final_blocks(native_blocks, ocr_blocks):
    if not native_blocks or not ocr_blocks:
        return native_blocks + ocr_blocks
    native_texts = [_block_text(nb) for nb in native_blocks]
    kept_ocr = []
    for ob in ocr_blocks:
        ob_text = _block_text(ob)
        ob_bbox = ob.get("bbox", [0, 0, 0, 0])
        drop = False
        for i, nb in enumerate(native_blocks):
            ov = _bbox_overlap_ratio(ob_bbox, nb.get("bbox", [0, 0, 0, 0]))
            if ov < 0.35:
                continue
            nt = native_texts[i]
            # Same area + identical/contained content => OCR duplicate of native.
            if ob_text and nt and (ob_text == nt or ob_text in nt or nt in ob_text):
                drop = True
                break
            if ob_text and nt and _text_sim(ob_text, nt) >= 0.72:
                drop = True
                break
            if ov >= 0.70:
                drop = True
                break
        if not drop:
            kept_ocr.append(ob)
    return native_blocks + kept_ocr


def _infer_alignment(bbox, content_bbox, page_w):
    x0, _, x1, _ = bbox
    c0, _, c1, _ = content_bbox
    content_w = max(1.0, c1 - c0)
    block_w = max(1.0, x1 - x0)
    left_gap = max(0.0, x0 - c0)
    right_gap = max(0.0, c1 - x1)
    near_tol = max(8.0, content_w * 0.03)
    center_tol = max(10.0, content_w * 0.06)

    if block_w >= content_w * 0.88 and left_gap <= near_tol and right_gap <= near_tol:
        return "justify", left_gap
    if abs(left_gap - right_gap) <= center_tol and block_w < content_w * 0.88:
        return "center", left_gap
    if right_gap <= near_tol and left_gap > near_tol:
        return "right", left_gap
    return "left", left_gap


def _annotate_layout(blocks, img_w, img_h):
    valid = []
    for b in blocks:
        bb = b.get("bbox", [0, 0, 0, 0])
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in bb]
        if x1 <= x0 or y1 <= y0:
            continue
        valid.append([x0, y0, x1, y1])

    if valid:
        content_bbox = [
            int(min(v[0] for v in valid)),
            int(min(v[1] for v in valid)),
            int(max(v[2] for v in valid)),
            int(max(v[3] for v in valid)),
        ]
    else:
        content_bbox = [0, 0, int(img_w), int(img_h)]

    margins = {
        "left": int(max(0, content_bbox[0])),
        "right": int(max(0, img_w - content_bbox[2])),
        "top": int(max(0, content_bbox[1])),
        "bottom": int(max(0, img_h - content_bbox[3])),
    }
    top_band_h = max(24, int(img_h * 0.10))
    bottom_band_h = max(24, int(img_h * 0.10))
    header_band = [0, min(int(img_h), top_band_h)]
    footer_band = [max(0, int(img_h) - bottom_band_h), int(img_h)]

    for block in blocks:
        bb = block.get("bbox", [0, 0, 0, 0])
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in bb]
        cy = (y0 + y1) / 2.0
        bw = max(1.0, x1 - x0)
        text = _block_text(block)
        is_short = len(text) <= 140

        role = "body"
        text_l = (text or "").lower()
        has_section_pattern = bool(re.match(r"^\s*(\d+(\.\d+)+)\b", text_l))
        is_figure_caption = bool(re.match(r"^\s*(figure|fig\.?)\s*\d+", text_l))
        is_short_title = is_short and (len(text.split()) <= 12)
        if y1 <= header_band[1] and is_short:
            role = "header"
        elif y0 >= footer_band[0] and is_short:
            role = "footer"
        elif y1 <= header_band[1] and bw < (content_bbox[2] - content_bbox[0]) * 0.8:
            role = "header"
        elif y0 >= footer_band[0] and bw < (content_bbox[2] - content_bbox[0]) * 0.8:
            role = "footer"
        elif cy <= header_band[1] and is_short:
            role = "header"
        elif cy >= footer_band[0] and is_short:
            role = "footer"
        elif is_figure_caption:
            role = "figure_caption"
        elif has_section_pattern:
            role = "section_heading"
        elif is_short_title and bw < (content_bbox[2] - content_bbox[0]) * 0.8:
            role = "title"

        align, indent_px = _infer_alignment([x0, y0, x1, y1], content_bbox, img_w)
        block["role"] = role
        block["alignment"] = align
        block["indent_px"] = float(max(0.0, indent_px))

        for line in block.get("lines", []):
            lb = line.get("bbox", bb)
            if isinstance(lb, (list, tuple)) and len(lb) == 4:
                l_align, l_indent = _infer_alignment([float(v) for v in lb], content_bbox, img_w)
            else:
                l_align, l_indent = align, indent_px
            line["alignment"] = l_align
            line["indent_px"] = float(max(0.0, l_indent))
            line["role"] = role
            for phrase in line.get("phrases", []):
                pb = phrase.get("bbox", lb)
                if isinstance(pb, (list, tuple)) and len(pb) == 4:
                    p_align, p_indent = _infer_alignment([float(v) for v in pb], content_bbox, img_w)
                else:
                    p_align, p_indent = l_align, l_indent
                phrase["alignment"] = p_align
                phrase["indent_px"] = float(max(0.0, p_indent))
                phrase["role"] = role

    return {
        "margins": margins,
        "content_bbox": content_bbox,
        "header_band": header_band,
        "footer_band": footer_band,
    }


def apply_ai_font_matching(ocr_blocks, pil_img, enable_audit=False):
    summary = {
        "enabled": bool(font_ai_matcher),
        "ready": bool(font_ai_matcher and font_ai_matcher.is_ready()),
        "threshold": FONT_AI_SCORE_THRESHOLD,
        "total_spans": 0,
        "attempted": 0,
        "matched": 0,
        "promoted": 0,
        "reasons": {},
    }

    def add_reason(reason):
        summary["reasons"][reason] = summary["reasons"].get(reason, 0) + 1

    if not font_ai_matcher:
        add_reason("font_ai_disabled")
        return summary
    if not font_ai_matcher.is_ready():
        add_reason("matcher_not_ready")
        return summary

    img_w, img_h = pil_img.size
    for block in ocr_blocks:
        for line in block.get("lines", []):
            for phrase in line.get("phrases", []):
                for span in phrase.get("spans", []):
                    summary["total_spans"] += 1
                    txt = (span.get("texte") or "").strip()
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    style = span.setdefault("style", {})
                    audit = None
                    if enable_audit:
                        audit = {
                            "font_original": style.get("font"),
                            "font_ai": None,
                            "score": None,
                            "selected_font": style.get("font"),
                            "applied": False,
                            "reason": "unknown",
                        }
                    if len(txt) < 2:
                        add_reason("text_too_short")
                        if audit is not None:
                            audit["reason"] = "text_too_short"
                            style["font_ai_audit"] = audit
                        continue

                    x0, y0, x1, y1 = [int(v) for v in bbox]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(img_w, x1), min(img_h, y1)
                    if x1 <= x0 or y1 <= y0:
                        add_reason("invalid_bbox")
                        if audit is not None:
                            audit["reason"] = "invalid_bbox"
                            style["font_ai_audit"] = audit
                        continue
                    if (x1 - x0) < 12 or (y1 - y0) < 10:
                        add_reason("bbox_too_small")
                        if audit is not None:
                            audit["reason"] = "bbox_too_small"
                            style["font_ai_audit"] = audit
                        continue

                    crop = pil_img.crop((x0, y0, x1, y1))
                    summary["attempted"] += 1
                    match = font_ai_matcher.match_crop(crop)
                    if not match:
                        add_reason("no_match")
                        if audit is not None:
                            audit["reason"] = "no_match"
                            style["font_ai_audit"] = audit
                        continue

                    summary["matched"] += 1
                    style["font_ai"] = match.font_name
                    style["font_ai_score"] = round(match.score, 4)
                    style["font_ai_path"] = match.font_path

                    # Promote AI-predicted font if confidence is acceptable.
                    if match.score >= FONT_AI_SCORE_THRESHOLD:
                        style["font"] = match.font_name
                        summary["promoted"] += 1
                        add_reason("threshold_passed")
                        flags = style.setdefault("flags", {})
                        for k, v in match.flags.items():
                            if k not in flags:
                                flags[k] = v
                        if audit is not None:
                            audit["font_ai"] = match.font_name
                            audit["score"] = round(match.score, 4)
                            audit["selected_font"] = style.get("font")
                            audit["applied"] = True
                            audit["reason"] = "threshold_passed"
                            style["font_ai_audit"] = audit
                    else:
                        add_reason("threshold_not_met")
                        if audit is not None:
                            audit["font_ai"] = match.font_name
                            audit["score"] = round(match.score, 4)
                            audit["selected_font"] = style.get("font")
                            audit["applied"] = False
                            audit["reason"] = "threshold_not_met"
                            style["font_ai_audit"] = audit

    return summary

def process_page(img, idx, filename, pdf_page=None, translate_to=None, force_ai=False, font_ai_audit=False, text_removal_mode="default"):
    sx = img.width / pdf_page.rect.width if pdf_page else 1.0
    sy = img.height / pdf_page.rect.height if pdf_page else 1.0
    
    # 1. Extraction du Texte Natif
    native_blocks = []
    non_text_zones = []
    native_images = []
    native_drawings = []
    if pdf_page and not force_ai:
        native = native_pdf_extractor.extract_page(pdf_page, sx=sx, sy=sy)
        native_blocks = native.get("blocks", [])
        non_text_zones = native.get("non_text_zones", [])
        native_images = native.get("images", [])
        native_drawings = native.get("drawings", [])

    # 2. OCR pour le reste
    result, _ = engine_ocr(np.array(img))
    raw_ocr = []
    if result:
        for res in result:
            b, txt, s = res
            bbox = [int(min([p[0] for p in b])), int(min([p[1] for p in b])), int(max([p[0] for p in b])), int(max([p[1] for p in b]))]
            # Filtre : ne pas ajouter si déjà couvert par du texte natif
            r_fitz = fitz.Rect(bbox)
            r_area = r_fitz.get_area()
            if r_area <= 0:
                continue
            if not any((r_fitz & fitz.Rect(nb["bbox"])).get_area() / r_area > 0.5 for nb in native_blocks):
                raw_ocr.append({"label": txt, "bbox": bbox, "score": float(s)})
    
    ocr_structure = parser.parse(raw_ocr, img) if raw_ocr else []
    font_ai_summary = {
        "enabled": bool(font_ai_matcher),
        "ready": bool(font_ai_matcher and font_ai_matcher.is_ready()),
        "threshold": FONT_AI_SCORE_THRESHOLD,
        "total_spans": 0,
        "attempted": 0,
        "matched": 0,
        "promoted": 0,
        "reasons": {},
    }
    if ocr_structure:
        font_ai_summary = apply_ai_font_matching(ocr_structure, img, enable_audit=font_ai_audit)
    final_blocks = _dedupe_final_blocks(native_blocks, ocr_structure)

    layout_meta = _annotate_layout(final_blocks, img.width, img.height)

    # 3. GÉNERATION DU FOND MAÎTRE NETTOYÉ (Workflow Inpainting IA)
    bg_master_path = ""
    mask_master_path = ""
    text_removal_debug = {}
    try:
        text_regions = []
        for b in final_blocks:
            bb = b.get("bbox", [0, 0, 0, 0])
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                text_regions.append([int(v) for v in bb])
        clean_bgr, mask, text_removal_debug = text_removal_strategy.remove(img, text_regions, mode=text_removal_mode)
        bg_name = f"bg_master_{filename}_{idx}.png"
        bg_master_path = os.path.join(RESULTS_DIR, bg_name)
        cv2.imwrite(bg_master_path, clean_bgr)
        mask_name = f"mask_master_{filename}_{idx}.png"
        mask_master_path = os.path.join(RESULTS_DIR, mask_name)
        cv2.imwrite(mask_master_path, mask)
    except Exception as e:
        print(f"Erreur génération fond maître chirurgical : {e}")

    # 4. VISUALISATION (Bboxes colorées pour l aperçu)
    vis_fn = f"vis_{filename}_{idx}.jpg"
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    
    for block in final_blocks:
        draw.rectangle(block["bbox"], outline="blue", width=3)
        for line in block["lines"]:
             draw.rectangle(line["bbox"], outline="green", width=1)
             for p in line["phrases"]: 
                draw.rectangle(p["bbox"], outline="red", width=1)
    img_draw.save(os.path.join(RESULTS_DIR, vis_fn))

    # 5. CONSTRUCTION DU CONTENU DÉTAILLÉ (Pour l affichage Flutter)
    display_text = f"DOC: {img.width}x{img.height} | DPI: {TARGET_DPI}\n"
    display_text += f"FONTS: {len(native_blocks)} blocs natifs | OCR: {len(ocr_structure)} blocs IA\n"
    
    for block in final_blocks:
        source_tag = "NATIVE" if block.get("source") == "native" else "OCR"
        display_text += f"\n[{source_tag} BLOC {block.get('id')} - bbox={block['bbox']}]\n"
        for line in block["lines"]:
            for p in line["phrases"]:
                for span in p.get("spans", []):
                    s, b = span.get("style", {}), span.get("bbox", [0, 0, 0, 0])
                    font_name = s.get("font", "Unknown")
                    font_size = float(s.get("size", 12.0))
                    color = s.get("color", "#000000")
                    # On affiche le nom de la font et la couleur
                    display_text += f"  - [{font_name} {font_size:.1f}pt {color}] {span.get('texte', '')}\n"
                    if font_ai_audit and "font_ai_audit" in s:
                        audit = s["font_ai_audit"]
                        display_text += (
                            "    font_ai_audit: "
                            f"candidate={audit.get('font_ai')} score={audit.get('score')} "
                            f"selected={audit.get('selected_font')} reason={audit.get('reason')}\n"
                        )

    return {
        "page": idx + 1, 
        "content": display_text,
        "structure": {
            "blocks": final_blocks, 
            "background_path": bg_master_path,
            "mask_master_path": mask_master_path,
            "text_removal_debug": text_removal_debug,
            "non_text_zones": non_text_zones,
            "images": native_images,
            "drawings": native_drawings,
            "layout": layout_meta,
            "font_ai_summary": font_ai_summary,
            "layout_version": "v3_layout_roles_alignment_margins",
            "dimensions": {"width": img.width, "height": img.height}
        },
        "visual_url": f"/results/{vis_fn}"
    }

def json_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
    if isinstance(obj, (np.bool_, bool)): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {str(k): json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [json_serializable(i) for i in obj]
    return obj

@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...), force_ai: bool = False, font_ai_audit: bool = FONT_AI_AUDIT_DEFAULT, text_removal_mode: str = "default"):
    try:
        base_name = os.path.basename(file.filename or "upload.bin")
        safe_name = f"{uuid.uuid4().hex}_{base_name}"
        save_path = os.path.join(UPLOAD_DIR, safe_name)
        with open(save_path, "wb") as b: shutil.copyfileobj(file.file, b)
        pages_results = []
        ext = os.path.splitext(base_name.lower())[1]
        if ext in OFFICE_EXTENSIONS:
            converted_pdf = _convert_office_to_pdf(save_path)
            doc = fitz.open(converted_pdf)
            for i in range(len(doc)):
                pix = doc[i].get_pixmap(dpi=TARGET_DPI)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                pages_results.append(
                    process_page(
                        img,
                        i,
                        base_name,
                        pdf_page=doc[i],
                        force_ai=force_ai,
                        font_ai_audit=font_ai_audit,
                        text_removal_mode=text_removal_mode,
                    )
                )
            doc.close()
        elif ext == '.pdf':
            doc = fitz.open(save_path)
            for i in range(len(doc)):
                pix = doc[i].get_pixmap(dpi=TARGET_DPI)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                # On passe l'objet page PDF original pour l'extraction multimédia
                pages_results.append(
                    process_page(
                        img,
                        i,
                        base_name,
                        pdf_page=doc[i],
                        force_ai=force_ai,
                        font_ai_audit=font_ai_audit,
                        text_removal_mode=text_removal_mode,
                    )
                )
            doc.close()
        else:
            img = Image.open(save_path).convert("RGB")
            pages_results.append(process_page(img, 0, base_name, force_ai=force_ai, font_ai_audit=font_ai_audit, text_removal_mode=text_removal_mode))
        
        # Nettoyage récursif des types numpy avant envoi
        cleaned_results = json_serializable(pages_results)
        return JSONResponse(content={"status": "success", "results": cleaned_results})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/reconstruct")
async def reconstruct_document(data: dict, target_lang: str = None):
    try:
        pages = data.get("pages", [])
        if target_lang:
            print(f"  [Pipeline] Traduction vers {target_lang} activée...")
            translator = get_translator()
            for idx, page in enumerate(pages):
                # 1. Traduction par bloc
                page = translator.translate_page(page, target_lang=target_lang)
                # 2. Réajustement géométrique des blocs pour limiter les collisions
                page = layout_optimizer.adjust_layout(page)
                pages[idx] = page

        recon = DocumentReconstructor()
        output_path = os.path.join(RESULTS_DIR, "reconstructed_output.pdf")
        recon.reconstruct({"pages": pages}, output_path)
        return JSONResponse(content={"status": "success", "pdf_url": f"/results/reconstructed_output.pdf"})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8001)
