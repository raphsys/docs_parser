import os
import shutil
import re
import uuid
import glob
import subprocess
import json
import copy
import logging
import xml.etree.ElementTree as ET
import uvicorn
import fitz
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from rapidocr_onnxruntime import RapidOCR
from structure_extractor import DocumentParser, LayoutV2Builder
from reconstructor import DocumentReconstructor
from remove_text_generic import inpaint_opencv
from layout_optimizer import LayoutOptimizer
from text_removal_strategy import TextRemovalStrategy
from native_pdf_extractor import NativePDFExtractor
from style_profiler import build_page_style_profile
from visual_compare import compare_reconstruction
from html_exporter import HtmlStyleExporter
from coverage_validator import analyze_document_coverage, analyze_rendered_text_coverage
from publication_qa import publication_qa
from page_policy_matrix import PagePolicyMatrix
from page_extraction_postprocessors import apply_page_extraction_postprocessors
from layout_ai_enricher import get_layout_ai_enricher

# --- Configuration ---
UPLOAD_DIR, CONV_DIR, RESULTS_DIR = 'uploads', 'converted_pages', 'ocr_results'
TARGET_DPI = 150 
FONT_AI_AUDIT_DEFAULT = False
EXTRACTION_AI_ENABLED = os.getenv("DOCS_PARSER_ENABLE_EXTRACTION_AI", "0") == "1"
LAYOUT_OPTIMIZER_ON_TRANSLATION = os.getenv("LAYOUT_OPTIMIZER_ON_TRANSLATION", "0") == "1"
OFFICE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".odt", ".odp"}

app = FastAPI(title="IA Document OCR - Stable Precision")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


def _parse_cors_origins():
    raw = os.getenv("DOCS_PARSER_CORS_ORIGINS", "").strip()
    if not raw:
        return [
            "http://127.0.0.1:19006",
            "http://localhost:19006",
            "http://127.0.0.1:8081",
            "http://localhost:8081",
            "http://127.0.0.1:8001",
            "http://localhost:8001",
        ]
    origins = []
    for part in raw.split(","):
        value = part.strip()
        if value:
            origins.append(value)
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    # Expo web may move between 19006/8081/8082/... when ports are occupied.
    # Accept localhost loopback dev origins on arbitrary ports by default.
    allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1)(?::\d+)?$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

for d in [UPLOAD_DIR, CONV_DIR, RESULTS_DIR]:
    if not os.path.exists(d): os.makedirs(d)

engine_ocr = RapidOCR()
parser = DocumentParser()
layout_v2_builder = LayoutV2Builder()
layout_optimizer = LayoutOptimizer()
text_removal_strategy = TextRemovalStrategy()
native_pdf_extractor = NativePDFExtractor()
html_exporter = HtmlStyleExporter()
page_policy_matrix = PagePolicyMatrix()
layout_ai_enricher = get_layout_ai_enricher()
_translator_instance = None


def get_translator():
    global _translator_instance
    if _translator_instance is not None:
        return _translator_instance
    try:
        from translator import DocumentTranslator
    except Exception as e:
        raise RuntimeError(f"Impossible de charger le module de traduction: {e}") from e
    _translator_instance = DocumentTranslator()
    return _translator_instance


def _safe_runtime_stem(filename: str) -> str:
    name = os.path.basename(str(filename or "")).strip()
    if not name:
        return "document"
    name = re.sub(r"[^\w.\-]+", "_", name)
    return name[:180] or "document"


def _disabled_layout_ai_info():
    return {
        "enabled": False,
        "ready": False,
        "applied": False,
        "bypassed": True,
        "reason": "extraction_ai_bypassed",
        "regions_added": 0,
        "feature_flags": {},
        "prediction_summary": {},
        "load_error": None,
        "predict_error": None,
        "inference_rescaled": False,
    }


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
        "pdf",
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


def _is_equation_like_text(text):
    s = _norm_text(text or "")
    if not s:
        return False
    words = re.findall(r"[a-zà-ÿ][a-zà-ÿ0-9'\-]*", s, flags=re.IGNORECASE)
    # Pure lexical parenthetical notes like "(weights)" are editorial text,
    # not formula overlays that should stay baked into the background.
    if re.fullmatch(r"\([a-zà-ÿ][a-zà-ÿ0-9'\-\s]{0,32}\)", s, flags=re.IGNORECASE):
        return False
    # Natural-language fragments with several lexical words are not equations,
    # even if they contain hyphens or punctuation.
    if len(words) >= 4 and not re.search(r"[=<>±×÷∑∫∞≈≠≤≥√∆∂µλΩα-ωΑ-Ω]", s):
        return False
    # Typical compact inline math/fragments.
    if len(s) <= 36:
        if re.search(r"(d[a-z]\s*/\s*d[a-z]|[a-z]\s*/\s*[a-z])", s):
            return True
        if re.search(r"[\=\+\-\*/\^\(\)\[\]<>±×÷∑∫∞≈≠≤≥√∆∂µλΩα-ωΑ-Ω]", s):
            return True
    # Ratio of symbolic chars indicates formula-like fragment.
    sym = len(re.findall(r"[^a-z0-9\s]", s))
    if len(s) <= 64 and sym >= max(3, int(0.18 * len(s))):
        return True
    return False


def _is_reference_like_text(text):
    s = _norm_text(text or "")
    if not s:
        return False
    # (2), (i), [12], [3-5], superscript-like footnote markers.
    if re.fullmatch(r"\((\d+|[ivxlcdm]+|[a-z])\)", s, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"\[\d+([,\-\s]*\d+)*\]", s):
        return True
    if re.fullmatch(r"\d{1,2}", s):
        return False
    return False


def _contains_greek_or_symbol(text):
    s = text or ""
    return bool(re.search(r"[α-ωΑ-ΩµλΩ∑∫∞≈≠≤≥√∆∂±×÷]", s))


def _is_immutable_inline_text(text):
    s = _norm_text(text or "")
    if not s:
        return False
    lexical_words = re.findall(r"[a-zà-ÿ][a-zà-ÿ0-9'\-]*", s, flags=re.IGNORECASE)
    if re.fullmatch(r"[a-z]+-score", s, flags=re.IGNORECASE):
        return False
    # Preserve visual list markers / numbering exactly from source.
    if re.fullmatch(r"[•▪◦·\-\*]", s):
        return True
    # Plain page numbers / ordinal labels must be re-rendered as normal units,
    # not frozen into the clean background.
    if re.fullmatch(r"\d{1,3}[.)]?", s):
        return False
    if _is_equation_like_text(s):
        return True
    if _contains_greek_or_symbol(s):
        return True
    if _is_reference_like_text(s):
        return True
    if len(lexical_words) >= 4:
        return False
    # chemistry-like formulas
    if re.fullmatch(r"(?:[A-Z][a-z]?\d*){2,}", s):
        return True
    # compact abbreviations/acronyms should remain unchanged
    if re.fullmatch(r"[A-Z]{2,8}", s):
        return True
    return False


def _span_is_monospace(span):
    if not isinstance(span, dict):
        return False
    style = span.get("style") or {}
    flags = style.get("flags") or {}
    font_name = str(style.get("font") or "").strip().lower()
    return bool(flags.get("monospace")) or "courier" in font_name


def _looks_like_programming_code_text(text):
    s = _normalize_spaces(text)
    if not s:
        return False
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", s):
        return True
    if "=" in s and re.search(r"[A-Za-z_][A-Za-z0-9_]*", s):
        return True
    if re.search(r"\b(?:name|padding|stride|strides|filters?_[A-Za-z0-9_]+)\s*=\s*", s):
        return True
    if s.count(",") >= 2 and ("_" in s or "(" in s or ")" in s):
        return True
    return False


def _phrase_is_immutable_programming_code(block, line, phrase):
    if not isinstance(phrase, dict):
        return False
    text = _normalize_spaces(
        phrase.get("text")
        or phrase.get("texte")
        or ""
    )
    if not text:
        return False
    unit_types = {
        str((node or {}).get("unit_type") or "").strip().lower()
        for node in (block, line, phrase)
        if isinstance(node, dict)
    }
    if "code_visible" in unit_types:
        return True
    spans = phrase.get("spans") or []
    if any(_span_is_monospace(span) for span in spans) and _looks_like_programming_code_text(text):
        return True
    return False


def _extract_immutable_overlays(blocks, pil_img, filename, page_idx):
    overlays = []
    if pil_img is None:
        return overlays
    img_w, img_h = pil_img.size
    seen = set()
    n = 0

    def _save_overlay(bb, txt, reason):
        nonlocal n
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            return None
        x0, y0, x1, y1 = [int(float(v)) for v in bb]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(img_w, x1), min(img_h, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        sig = (x0, y0, x1, y1, txt, reason)
        if sig in seen:
            return None
        seen.add(sig)
        n += 1
        out_name = f"immutable_{filename}_{page_idx}_{n}.png"
        out_path = os.path.join(RESULTS_DIR, out_name)
        try:
            crop = pil_img.crop((x0, y0, x1, y1))
            crop.save(out_path)
        except Exception:
            return None
        return {
            "bbox": [x0, y0, x1, y1],
            "path": out_path,
            "url": f"/results/{out_name}",
            "text": txt,
            "reason": reason,
        }

    for block in blocks:
        for line in block.get("lines", []):
            for phrase in line.get("phrases", []):
                phrase_text = (phrase.get("text") or phrase.get("texte") or "").strip()
                phrase_bbox = phrase.get("bbox") or line.get("bbox") or block.get("bbox")
                if _phrase_is_immutable_programming_code(block, line, phrase):
                    overlay = _save_overlay(phrase_bbox, phrase_text, "immutable_code")
                    if overlay:
                        overlays.append(overlay)
                        for sp in phrase.get("spans", []) or []:
                            if isinstance(sp, dict):
                                sp["skip_render"] = True
                        phrase["render_mode"] = "background_only"
                    continue
                spans = phrase.get("spans", [])
                immutable_spans = []
                for sp in spans:
                    txt = (sp.get("texte") or "").strip()
                    bb = sp.get("bbox") or phrase.get("bbox")
                    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                        continue
                    if not _is_immutable_inline_text(txt):
                        continue
                    overlay = _save_overlay(bb, txt, "immutable_inline")
                    if not overlay:
                        continue
                    overlays.append(overlay)
                    sp["skip_render"] = True
                    immutable_spans.append(sp)
                if spans and len(immutable_spans) == len(spans):
                    phrase["render_mode"] = "background_only"
    return overlays


def _is_meaningful_diagram_text_label(text):
    s = _norm_text(text or "")
    if len(s) < 4 or len(s) > 48:
        return False
    if not re.search(r"[a-z]", s):
        return False
    if re.fullmatch(r"[a-z]", s):
        return False
    if re.fullmatch(r"[0-9\.\- ]+", s):
        return False
    # keep short axis/semantic labels (e.g., "Goal weight", "Error")
    return True

def _rect_from_bbox(b):
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return fitz.Rect(0, 0, 0, 0)
    return fitz.Rect(float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def _bbox_from_rect(r):
    return [int(round(r.x0)), int(round(r.y0)), int(round(r.x1)), int(round(r.y1))]


def _block_avg_font_size(block):
    sizes = []
    for line in block.get("lines", []):
        for phrase in line.get("phrases", []):
            for span in phrase.get("spans", []):
                s = span.get("style", {}).get("size")
                if isinstance(s, (int, float)) and s > 0:
                    sizes.append(float(s))
    return float(np.median(sizes)) if sizes else 10.0


def _block_style_signature(block):
    for line in block.get("lines", []):
        for phrase in line.get("phrases", []):
            for span in phrase.get("spans", []):
                st = span.get("style", {}) or {}
                flags = st.get("flags", {}) if isinstance(st.get("flags", {}), dict) else {}
                return (
                    st.get("font", ""),
                    round(float(st.get("size", 0.0) or 0.0), 1),
                    st.get("color", "#000000"),
                    bool(flags.get("bold")),
                    bool(flags.get("italic")),
                )
    return ("", 0.0, "#000000", False, False)


def _style_is_compatible(sig_a, sig_b):
    if not sig_a or not sig_b:
        return False
    same_font = sig_a[0] == sig_b[0]
    same_color = sig_a[2] == sig_b[2]
    size_close = abs(float(sig_a[1]) - float(sig_b[1])) <= 1.2
    weight_close = sig_a[3] == sig_b[3]
    italic_close = sig_a[4] == sig_b[4]
    return same_font and same_color and size_close and weight_close and italic_close


def _horizontal_overlap_ratio(r1, r2):
    inter = max(0.0, min(r1.x1, r2.x1) - max(r1.x0, r2.x0))
    den = max(1.0, min(r1.width, r2.width))
    return inter / den


def _line_sort_key(line):
    b = line.get("bbox", [0, 0, 0, 0])
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return (0.0, 0.0)
    return (float(b[1]), float(b[0]))


def _merge_two_blocks(base, extra):
    rb = _rect_from_bbox(base.get("bbox", [0, 0, 0, 0]))
    re = _rect_from_bbox(extra.get("bbox", [0, 0, 0, 0]))
    runion = rb | re
    base["bbox"] = _bbox_from_rect(runion)
    lines = list(base.get("lines", [])) + list(extra.get("lines", []))
    lines.sort(key=_line_sort_key)
    base["lines"] = lines
    if "role" not in base and extra.get("role"):
        base["role"] = extra.get("role")
    return base


def _merge_native_blocks(blocks):
    ordered = sorted(blocks, key=lambda b: (_rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).y0, _rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).x0))
    merged = []
    for blk in ordered:
        if not merged:
            merged.append(blk)
            continue
        cur = merged[-1]
        if cur.get("source") != "native" or blk.get("source") != "native":
            merged.append(blk)
            continue
        cur_role = cur.get("role", "body")
        blk_role = blk.get("role", "body")
        if cur_role != blk_role or cur_role not in {"body", "title", "section_heading", "figure_caption"}:
            merged.append(blk)
            continue
        rc = _rect_from_bbox(cur.get("bbox", [0, 0, 0, 0]))
        rb = _rect_from_bbox(blk.get("bbox", [0, 0, 0, 0]))
        gap = rb.y0 - rc.y1
        if gap < -2.0 or gap > max(22.0, 0.75 * max(rc.height, rb.height)):
            merged.append(blk)
            continue
        if _horizontal_overlap_ratio(rc, rb) < 0.55:
            merged.append(blk)
            continue
        if not _style_is_compatible(_block_style_signature(cur), _block_style_signature(blk)):
            merged.append(blk)
            continue
        merged[-1] = _merge_two_blocks(cur, blk)
    return merged


def _group_diagram_labels(blocks, img_w, img_h):
    candidates = []
    keep = []
    for i, b in enumerate(blocks):
        text = _block_text(b)
        r = _rect_from_bbox(b.get("bbox", [0, 0, 0, 0]))
        area = r.get_area()
        avg_size = _block_avg_font_size(b)
        is_native = b.get("source") == "native"
        short_text = len(text) <= 12
        tiny_box = area <= 2600
        low_region = r.y0 >= (img_h * 0.55)
        if is_native and low_region and short_text and tiny_box and avg_size <= 8.2:
            candidates.append((i, b, r))
        else:
            keep.append(b)

    if len(candidates) < 3:
        return blocks

    parent = list(range(len(candidates)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pad = max(14.0, img_w * 0.01)
    for i in range(len(candidates)):
        _, _, ri = candidates[i]
        ei = fitz.Rect(ri.x0 - pad, ri.y0 - pad, ri.x1 + pad, ri.y1 + pad)
        for j in range(i + 1, len(candidates)):
            _, _, rj = candidates[j]
            ej = fitz.Rect(rj.x0 - pad, rj.y0 - pad, rj.x1 + pad, rj.y1 + pad)
            if (ei & ej).get_area() > 0:
                union(i, j)

    groups = {}
    for idx, item in enumerate(candidates):
        root = find(idx)
        groups.setdefault(root, []).append(item)

    for g_items in groups.values():
        if len(g_items) < 3:
            for _, b, _ in g_items:
                keep.append(b)
            continue
        g_blocks = [b for _, b, _ in g_items]
        g_blocks.sort(key=lambda x: (_rect_from_bbox(x.get("bbox", [0, 0, 0, 0])).y0, _rect_from_bbox(x.get("bbox", [0, 0, 0, 0])).x0))
        base = dict(g_blocks[0])
        base["id"] = f"diag_{base.get('id', '0')}"
        base["role"] = "diagram_label"
        for extra in g_blocks[1:]:
            base = _merge_two_blocks(base, extra)
        keep.append(base)

    keep.sort(key=lambda b: (_rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).y0, _rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).x0))
    return keep


def _expand_diagram_groups(blocks, img_w, img_h):
    diag_idx = [i for i, b in enumerate(blocks) if (b.get("role") == "diagram_label")]
    if not diag_idx:
        return blocks

    used = set()
    out = [dict(b) for b in blocks]
    # Absorb small nearby native labels/ticks into nearest diagram group.
    for i, blk in enumerate(out):
        if i in diag_idx:
            continue
        if blk.get("source") != "native":
            continue
        r = _rect_from_bbox(blk.get("bbox", [0, 0, 0, 0]))
        if r.get_area() <= 0:
            continue
        txt = _block_text(blk)
        avg_size = _block_avg_font_size(blk)
        low_region = r.y0 >= (img_h * 0.52)
        if not low_region:
            continue
        # Conservative: only small textual artifacts likely from charts.
        if len(txt) > 24:
            continue
        if avg_size > 9.6:
            continue
        if r.get_area() > 26000:
            continue
        role = blk.get("role", "body")
        if role not in {"body", "title", "section_heading", "figure_caption"}:
            continue
        if _is_meaningful_diagram_text_label(txt):
            blk["role"] = "diagram_text_label"
            continue

        nearest = None
        best_dist = 1e18
        for di in diag_idx:
            dr = _rect_from_bbox(out[di].get("bbox", [0, 0, 0, 0]))
            # expanded proximity window
            pad_x = max(24.0, img_w * 0.06)
            pad_y = max(20.0, img_h * 0.05)
            exp = fitz.Rect(dr.x0 - pad_x, dr.y0 - pad_y, dr.x1 + pad_x, dr.y1 + pad_y)
            if (exp & r).get_area() <= 0:
                continue
            cx, cy = (r.x0 + r.x1) * 0.5, (r.y0 + r.y1) * 0.5
            dcx, dcy = (dr.x0 + dr.x1) * 0.5, (dr.y0 + dr.y1) * 0.5
            d2 = (cx - dcx) ** 2 + (cy - dcy) ** 2
            if d2 < best_dist:
                best_dist = d2
                nearest = di
        if nearest is None:
            continue
        out[nearest] = _merge_two_blocks(out[nearest], blk)
        out[nearest]["role"] = "diagram_label"
        used.add(i)

    filtered = [b for i, b in enumerate(out) if i not in used]
    filtered.sort(key=lambda b: (_rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).y0, _rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).x0))
    return filtered


def _merge_overlapping_diagram_labels(blocks):
    out = [dict(b) for b in blocks]
    changed = True
    while changed:
        changed = False
        merged = []
        used = set()
        for i, bi in enumerate(out):
            if i in used:
                continue
            if bi.get("role") != "diagram_label":
                merged.append(bi)
                used.add(i)
                continue
            ri = _rect_from_bbox(bi.get("bbox", [0, 0, 0, 0]))
            acc = dict(bi)
            used.add(i)
            for j in range(i + 1, len(out)):
                if j in used:
                    continue
                bj = out[j]
                if bj.get("role") != "diagram_label":
                    continue
                rj = _rect_from_bbox(bj.get("bbox", [0, 0, 0, 0]))
                inter = (ri & rj).get_area()
                if inter <= 0:
                    continue
                ratio = inter / max(1.0, min(ri.get_area(), rj.get_area()))
                if ratio >= 0.22:
                    acc = _merge_two_blocks(acc, bj)
                    ri = _rect_from_bbox(acc.get("bbox", [0, 0, 0, 0]))
                    used.add(j)
                    changed = True
            merged.append(acc)
        out = sorted(merged, key=lambda b: (_rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).y0, _rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).x0))
    return out


def _attach_inline_equation_blocks(blocks):
    out = [dict(b) for b in blocks]
    remove_idx = set()
    for i, b in enumerate(out):
        if b.get("source") != "native":
            continue
        if b.get("role") not in {"body", "title", "section_heading"}:
            continue
        rb = _rect_from_bbox(b.get("bbox", [0, 0, 0, 0]))
        area_b = rb.get_area()
        if area_b <= 0:
            continue
        text_b = _block_text(b)
        if len(text_b) < 80:
            continue
        for j, c in enumerate(out):
            if i == j or j in remove_idx:
                continue
            if c.get("source") != "native":
                continue
            if c.get("role") not in {"body", "title", "section_heading"}:
                continue
            rc = _rect_from_bbox(c.get("bbox", [0, 0, 0, 0]))
            area_c = rc.get_area()
            if area_c <= 0:
                continue
            if area_c / max(1.0, area_b) > 0.10:
                continue
            tc = _block_text(c)
            if len(tc) > 28:
                continue
            # Keep standalone equation/math fragments as dedicated blocks.
            if _is_equation_like_text(tc):
                continue
            # Candidate is small and sits inside/near paragraph area.
            overlap = (rb & rc).get_area() / max(1.0, area_c)
            near_inside = overlap > 0.45 or (
                rc.x0 >= rb.x0 - 16 and rc.x1 <= rb.x1 + 16 and rc.y0 >= rb.y0 - 12 and rc.y1 <= rb.y1 + 20
            )
            if not near_inside:
                continue
            out[i] = _merge_two_blocks(out[i], c)
            remove_idx.add(j)

    filtered = [b for idx, b in enumerate(out) if idx not in remove_idx]
    filtered.sort(key=lambda b: (_rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).y0, _rect_from_bbox(b.get("bbox", [0, 0, 0, 0])).x0))
    return filtered


def _postprocess_blocks(blocks, img_w, img_h):
    if not blocks:
        return blocks
    merged = _merge_native_blocks(blocks)
    merged = _attach_inline_equation_blocks(merged)
    grouped = _group_diagram_labels(merged, img_w=img_w, img_h=img_h)
    grouped = _expand_diagram_groups(grouped, img_w=img_w, img_h=img_h)
    grouped = _merge_overlapping_diagram_labels(grouped)
    for b in grouped:
        if b.get("role") == "diagram_label":
            b["render_mode"] = "background_only"
    _enrich_layout_markers(grouped)
    return grouped


def _annotate_translation_contracts(blocks, page_context=None):
    page_context = page_context or {}
    page_role = page_context.get("page_role") or "body"
    page_family = page_context.get("page_family") or "body_text"
    page_family_group = page_context.get("page_family_group") or page_family
    document_type = page_context.get("document_type") or "mixed_unknown"
    layout_type = page_context.get("layout_type") or "mixed_blocks"
    style_profile = page_context.get("style_profile") or "mixed_irregular"
    page_case = page_context.get("page_case") or {}
    fallback_policy = page_case.get("fallback_policy") or ""

    def classify_text(text, role, source_kind):
        return page_policy_matrix.classify_unit_policy(
            text=text,
            role=role,
            source_kind=source_kind,
            page_role=page_role,
            page_family=page_family,
            page_family_group=page_family_group,
            document_type=document_type,
            layout_type=layout_type,
            style_profile=style_profile,
            fallback_policy=fallback_policy,
        )

    def inherit_reference_policy(parent_unit_type, parent_policy, child_policy):
        parent_kind = _normalize_spaces(parent_unit_type).lower()
        child_kind = _normalize_spaces((child_policy or {}).get("unit_type") or "").lower()
        if parent_kind not in {"citation", "reference_link"}:
            return child_policy
        if child_kind in {"citation", "reference_link", "code_visible", "formula", "formula_label"}:
            return child_policy
        inherited = dict(child_policy or {})
        inherited["unit_type"] = parent_kind
        inherited["translatable"] = bool(parent_policy.get("translatable"))
        inherited["translation_strategy"] = parent_policy.get("translation_strategy")
        inherited["coverage_required"] = parent_policy.get("coverage_required")
        inherited["render_policy"] = parent_policy.get("render_policy")
        return inherited

    def apply_explicit_render_mode(unit, policy):
        render_mode = _normalize_spaces(unit.get("render_mode") or unit.get("render_policy") or "").lower()
        if render_mode != "background_only":
            return dict(policy or {})
        forced = dict(policy or {})
        forced["translatable"] = False
        forced["translation_strategy"] = "background_only"
        forced["coverage_required"] = "strict"
        forced["render_policy"] = "background_only"
        return forced

    def aggregate_child_policy(parent_policy, children):
        result = dict(parent_policy or {})
        child_items = [child for child in (children or []) if isinstance(child, dict)]
        if not child_items:
            return result

        child_render_modes = {
            _normalize_spaces(child.get("render_policy") or child.get("render_mode") or "").lower()
            for child in child_items
        }
        if child_render_modes == {"background_only"}:
            result["translatable"] = False
            result["translation_strategy"] = "background_only"
            result["coverage_required"] = "strict"
            result["render_policy"] = "background_only"
            return result

        any_translatable = any(bool(child.get("translatable")) for child in child_items)
        any_locked_child = any(
            _normalize_spaces(child.get("translation_strategy") or "").lower() in {"exact_preserve", "keep_original", "background_only"}
            or not bool(child.get("translatable"))
            for child in child_items
        )
        if any_translatable and any_locked_child:
            result["translatable"] = True
            result["translation_strategy"] = "layout_constrained"
            if _normalize_spaces(result.get("render_policy") or "").lower() == "background_only":
                result["render_policy"] = "anchored_text"
        return result

    def _span_flags(span):
        style = span.get("style") if isinstance(span.get("style"), dict) else {}
        flags = style.get("flags") if isinstance(style.get("flags"), dict) else {}
        return {name: bool(flags.get(name)) for name in ["bold", "italic", "underline", "highlight", "uppercase", "monospace", "serif"]}

    def _inline_reference_like(text):
        s = _normalize_spaces(text)
        if not s:
            return False
        return bool(re.search(r"(https?://|www\.|[\w\.-]+@[\w\.-]+\.\w+|doi:\s*|arxiv:|^\[\d+\]$|^\(\d+\)$)", s, flags=re.IGNORECASE))

    def _inline_technical_like(text):
        s = _normalize_spaces(text)
        if not s:
            return False
        if re.search(r"\b[A-Z]{2,}[A-Z0-9\-]*\b", s):
            return True
        if re.search(r"\b[A-Za-z]+Net\b|\b[A-Za-z]+GAN\b|\bYOLOv?\d*\b|\bResNet\d*\b|\bAlexNet\b|\bVGG\d*\b", s):
            return True
        if re.search(r"\b[a-z]+[A-Z][A-Za-z0-9]*\b", s):
            return True
        return False

    def _infer_expression_inline_class(span):
        text = _normalize_spaces(span.get("text") or span.get("texte") or "")
        unit_type = _normalize_spaces(span.get("unit_type") or "").lower()
        flags = _span_flags(span)
        if unit_type == "code_visible" or _looks_like_programming_code_text(text):
            return "code"
        if unit_type in {"reference_link", "citation"} or _inline_reference_like(text):
            return "reference"
        if unit_type in {"formula", "formula_label"}:
            return "formula"
        # Micro-symboles mathématiques : lettre grecque isolée, opérateur math, variable courte
        # avec exposant/indice — traités comme formule pour ne pas se perdre dans le flux
        if text and len(text) <= 6:
            if re.fullmatch(r"[α-ωΑ-Ωα-ωµ][\u2080-\u2089\u00B2\u00B3\u00B9]?", text):
                return "formula"
            if re.fullmatch(r"[∑∫∞≈≠≤≥√∆∂±×÷⊕⊗∈∉∩∪⊂⊃∀∃∇]", text):
                return "formula"
            if re.search(r"[α-ωΑ-Ωα-ωµ∑∫∞≈≠≤≥√∆∂±×÷]", text) and len(re.findall(r"[A-Za-z0-9]", text)) <= 3:
                return "formula"
        if flags.get("monospace") or _inline_technical_like(text):
            return "technical_inline"
        if unit_type in {"diagram_label", "chart_label", "short_label"}:
            return "label"
        return "plain_text"

    def _expression_emphasis_level(span):
        flags = _span_flags(span)
        if flags.get("highlight") or flags.get("bold"):
            return "strong"
        if flags.get("italic") or flags.get("underline") or flags.get("uppercase"):
            return "moderate"
        return "neutral"

    def _style_signature(span):
        style = span.get("style") if isinstance(span.get("style"), dict) else {}
        flags = _span_flags(span)
        size = style.get("size")
        try:
            size = round(float(size), 1) if size not in {None, ""} else None
        except Exception:
            size = None
        active_flags = tuple(sorted(name for name, enabled in flags.items() if enabled))
        return (
            str(style.get("font") or ""),
            size,
            str(style.get("color") or ""),
            active_flags,
        )

    def _neighbor_relation(current, neighbor):
        if not isinstance(neighbor, dict):
            return {"exists": False}
        cur_sem = current.get("expression_semantics") if isinstance(current.get("expression_semantics"), dict) else {}
        nxt_sem = neighbor.get("expression_semantics") if isinstance(neighbor.get("expression_semantics"), dict) else {}
        same_inline = cur_sem.get("inline_class") == nxt_sem.get("inline_class")
        same_style = _style_signature(current) == _style_signature(neighbor)
        relation = "continuation"
        if same_inline and not same_style:
            relation = "emphasis_shift"
        elif not same_inline:
            relation = "semantic_shift"
        return {
            "exists": True,
            "neighbor_id": neighbor.get("unit_id"),
            "neighbor_text": _normalize_spaces(neighbor.get("text") or neighbor.get("texte") or ""),
            "relation": relation,
            "continuation": bool(same_inline),
            "same_inline_class": bool(same_inline),
            "same_style": bool(same_style),
        }

    def _editorial_flow_class(unit):
        role_local = _normalize_spaces(unit.get("role") or "").lower()
        render_policy_local = _normalize_spaces(unit.get("render_policy") or "").lower()
        unit_type_local = _normalize_spaces(unit.get("unit_type") or "").lower()
        if render_policy_local == "background_only":
            return "protected_visual"
        if role_local in {"section_heading", "title", "header", "footer"}:
            return "heading_like"
        if role_local in {"diagram_label", "diagram_text_label", "chart_label"}:
            return "anchored_annotation"
        if role_local == "figure_caption":
            return "caption"
        if unit_type_local in {"reference_link", "citation"}:
            return "reference_run"
        return "editorial_body"

    def _editorial_semantics(unit):
        flow_class = _editorial_flow_class(unit)
        render_policy_local = _normalize_spaces(unit.get("render_policy") or "").lower()
        return {
            "flow_class": flow_class,
            "reflowable": bool(render_policy_local not in {"background_only", "exact_preserve"}),
            "anchored_annotation": bool(flow_class == "anchored_annotation"),
            "heading_like": bool(flow_class == "heading_like"),
            "protected_visual": bool(flow_class == "protected_visual"),
            "caption_like": bool(flow_class == "caption"),
        }

    def _structural_context(unit_level, unit, parent=None, block=None, line=None):
        ctx = {
            "level": unit_level,
            "unit_id": unit.get("unit_id"),
            "parent_unit_id": parent.get("unit_id") if isinstance(parent, dict) else None,
            "block_unit_id": block.get("unit_id") if isinstance(block, dict) else None,
            "line_unit_id": line.get("unit_id") if isinstance(line, dict) else None,
        }
        if unit_level == "block":
            ctx["child_line_count"] = len(unit.get("lines", []) or [])
            ctx["child_semantic_phrase_count"] = len(unit.get("semantic_phrases", []) or [])
        elif unit_level == "line":
            ctx["child_phrase_count"] = len(unit.get("phrases", []) or [])
            ctx["line_index"] = unit.get("line_index")
        elif unit_level == "semantic_phrase":
            ctx["child_span_count"] = len(unit.get("spans", []) or [])
            ctx["start_line_index"] = unit.get("start_line_index")
            ctx["end_line_index"] = unit.get("end_line_index")
            ctx["line_indices"] = list(unit.get("line_indices", []) or [])
        elif unit_level == "phrase":
            ctx["child_span_count"] = len(unit.get("spans", []) or [])
            ctx["line_index"] = unit.get("line_index")
        elif unit_level == "span":
            ctx["line_index"] = line.get("line_index") if isinstance(line, dict) else None
        return ctx

    def _block_gap(current, previous):
        try:
            cb = current.get("bbox") or [0, 0, 0, 0]
            pb = previous.get("bbox") or [0, 0, 0, 0]
            return max(0.0, float(cb[1]) - float(pb[3]))
        except Exception:
            return 0.0

    def _x_anchor_delta(current, previous):
        try:
            cx = float((current.get("bbox") or [0, 0, 0, 0])[0])
            px = float((previous.get("bbox") or [0, 0, 0, 0])[0])
            return abs(cx - px)
        except Exception:
            return 0.0

    def _editorial_relation(current, previous, hard_break=False):
        if not isinstance(previous, dict):
            return {"exists": False}
        current_sem = current.get("editorial_semantics") if isinstance(current.get("editorial_semantics"), dict) else _editorial_semantics(current)
        previous_sem = previous.get("editorial_semantics") if isinstance(previous.get("editorial_semantics"), dict) else _editorial_semantics(previous)
        relation = "separate"
        continuation = False
        if previous_sem.get("heading_like") and current_sem.get("flow_class") == "editorial_body":
            relation = "heading_to_body"
            continuation = True
        elif current_sem.get("flow_class") == "editorial_body" and previous_sem.get("flow_class") == "editorial_body":
            if hard_break:
                relation = "paragraph_break"
            else:
                relation = "paragraph_continuation"
                continuation = True
        elif current_sem.get("caption_like") and previous_sem.get("anchored_annotation"):
            relation = "annotation_to_caption"
        elif current_sem.get("anchored_annotation") and previous_sem.get("anchored_annotation"):
            relation = "annotation_cluster"
            continuation = True
        elif current_sem.get("caption_like") and previous_sem.get("caption_like"):
            relation = "caption_continuation"
            continuation = True
        return {
            "exists": True,
            "neighbor_id": previous.get("unit_id"),
            "neighbor_role": previous.get("role"),
            "relation": relation,
            "continuation": bool(continuation),
            "gap_px": round(_block_gap(current, previous), 2),
            "x_anchor_delta_px": round(_x_anchor_delta(current, previous), 2),
            "hard_break": bool(hard_break),
        }

    for b_idx, block in enumerate(blocks or []):
        role = block.get("role", "body")
        source = str(block.get("source") or "").strip().lower()
        source_kind_hint = str(block.get("source_kind") or "").strip().lower()
        if not source:
            if source_kind_hint.startswith("native_"):
                source = "native"
            elif source_kind_hint.startswith("ocr_"):
                source = "ocr"
            else:
                source = "ocr"
        source_kind = block.get("source_kind") or ("native_block" if source == "native" else "ocr_block")
        text = block.get("text") or _block_text(block)
        policy = classify_text(text, role, source_kind)
        policy = apply_explicit_render_mode(block, policy)
        block["unit_id"] = block.get("id") or f"blk_{b_idx}"
        block["source"] = source
        block["source_kind"] = source_kind
        block["text"] = text
        block["text_normalized"] = _normalize_spaces(text)
        block["translatable"] = bool(policy["translatable"])
        block["unit_type"] = policy.get("unit_type") or ""
        block["translation_strategy"] = policy["translation_strategy"]
        block["coverage_required"] = policy["coverage_required"]
        block["render_policy"] = policy["render_policy"]
        block["editorial_semantics"] = _editorial_semantics(block)
        block["structural_context"] = _structural_context("block", block, parent=None, block=block)
        for sp_idx, semantic_phrase in enumerate(block.get("semantic_phrases", []) or []):
            sp_txt = _normalize_spaces(
                semantic_phrase.get("text")
                or semantic_phrase.get("texte")
                or _phrase_render_text(semantic_phrase)
            )
            sp_kind = semantic_phrase.get("source_kind") or ("native_semantic_phrase" if source == "native" else "ocr_semantic_phrase")
            sp_policy = classify_text(sp_txt, role, sp_kind)
            sp_policy = inherit_reference_policy(block.get("unit_type"), policy, sp_policy)
            sp_policy = apply_explicit_render_mode(semantic_phrase, sp_policy)
            semantic_phrase["unit_id"] = semantic_phrase.get("sentence_id") or f"{block['unit_id']}:semantic_phrase:{sp_idx}"
            semantic_phrase["source"] = str(semantic_phrase.get("source") or source).strip().lower() or source
            semantic_phrase["source_kind"] = sp_kind
            semantic_phrase["text"] = sp_txt
            semantic_phrase["texte"] = sp_txt
            semantic_phrase["text_normalized"] = _normalize_spaces(semantic_phrase["text"])
            semantic_phrase["translatable"] = bool(sp_policy["translatable"])
            semantic_phrase["unit_type"] = sp_policy.get("unit_type") or ""
            semantic_phrase["translation_strategy"] = sp_policy["translation_strategy"]
            semantic_phrase["coverage_required"] = sp_policy["coverage_required"]
            semantic_phrase["render_policy"] = sp_policy["render_policy"]
            semantic_phrase["editorial_semantics"] = _editorial_semantics(semantic_phrase)
            semantic_phrase["structural_context"] = _structural_context("semantic_phrase", semantic_phrase, parent=block, block=block)
        for l_idx, line in enumerate(block.get("lines", []) or []):
            l_txt = line.get("line_text") or _line_phrase_text(line)
            l_kind = line.get("source_kind") or ("native_line" if source == "native" else "ocr_line")
            l_policy = classify_text(l_txt, role, l_kind)
            l_policy = inherit_reference_policy(block.get("unit_type"), policy, l_policy)
            l_policy = apply_explicit_render_mode(line, l_policy)
            line["unit_id"] = f"{block['unit_id']}:line:{l_idx}"
            line["source"] = str(line.get("source") or source).strip().lower() or source
            line["source_kind"] = l_kind
            line["text"] = l_txt
            line["text_normalized"] = _normalize_spaces(l_txt)
            line["translatable"] = bool(l_policy["translatable"])
            line["unit_type"] = l_policy.get("unit_type") or ""
            line["translation_strategy"] = l_policy["translation_strategy"]
            line["coverage_required"] = l_policy["coverage_required"]
            line["render_policy"] = l_policy["render_policy"]
            line["editorial_semantics"] = _editorial_semantics(line)
            line["structural_context"] = _structural_context("line", line, parent=block, block=block, line=line)
            for p_idx, phrase in enumerate(line.get("phrases", []) or []):
                p_render_txt = _phrase_render_text(phrase)
                p_txt = _phrase_source_text(phrase) or p_render_txt
                p_kind = phrase.get("source_kind") or ("native_phrase" if source == "native" else "ocr_phrase")
                p_policy = classify_text(p_txt, role, p_kind)
                p_policy = inherit_reference_policy(line.get("unit_type") or block.get("unit_type"), l_policy, p_policy)
                p_policy = apply_explicit_render_mode(phrase, p_policy)
                phrase["unit_id"] = f"{line['unit_id']}:phrase:{p_idx}"
                phrase["source"] = str(phrase.get("source") or line.get("source") or source).strip().lower() or source
                phrase["source_kind"] = p_kind
                phrase["text"] = p_txt
                phrase["texte"] = p_txt
                phrase["render_text"] = p_render_txt
                phrase["text_normalized"] = _normalize_spaces(p_txt)
                phrase["translatable"] = bool(p_policy["translatable"])
                phrase["unit_type"] = p_policy.get("unit_type") or ""
                phrase["translation_strategy"] = p_policy["translation_strategy"]
                phrase["coverage_required"] = p_policy["coverage_required"]
                phrase["render_policy"] = p_policy["render_policy"]
                phrase["editorial_semantics"] = _editorial_semantics(phrase)
                phrase["structural_context"] = _structural_context("phrase", phrase, parent=line, block=block, line=line)
                for s_idx, span in enumerate(phrase.get("spans", []) or []):
                    s_txt = _normalize_spaces(span.get("texte", ""))
                    s_kind = span.get("source_kind") or ("native_span" if source == "native" else "ocr_span")
                    s_policy = classify_text(s_txt, role, s_kind)
                    s_policy = inherit_reference_policy(phrase.get("unit_type") or line.get("unit_type") or block.get("unit_type"), p_policy, s_policy)
                    s_policy = apply_explicit_render_mode(span, s_policy)
                    span["unit_id"] = f"{phrase['unit_id']}:span:{s_idx}"
                    span["source"] = str(span.get("source") or phrase.get("source") or line.get("source") or source).strip().lower() or source
                    span["source_kind"] = s_kind
                    span["text"] = s_txt
                    span["text_normalized"] = s_txt
                    span["translatable"] = bool(s_policy["translatable"])
                    span["unit_type"] = s_policy.get("unit_type") or ""
                    span["translation_strategy"] = s_policy["translation_strategy"]
                    span["coverage_required"] = s_policy["coverage_required"]
                    span["render_policy"] = s_policy["render_policy"]
                    inline_class = _infer_expression_inline_class(span)
                    emphasis_flags = _span_flags(span)
                    protected_inline = (
                        (not bool(span.get("translatable")))
                        or inline_class in {"code", "reference", "formula"}
                        or str(span.get("translation_strategy") or "").strip().lower() in {"exact_preserve", "keep_original", "background_only"}
                    )
                    span["expression_semantics"] = {
                        "inline_class": inline_class,
                        "protected_inline": bool(protected_inline),
                        "immutable_inline": bool(protected_inline and inline_class in {"code", "reference", "formula"}),
                        "technical_inline": bool(inline_class == "technical_inline"),
                        "emphasis_level": _expression_emphasis_level(span),
                        "emphasis_flags": emphasis_flags,
                    }
                    span["structural_context"] = _structural_context("span", span, parent=phrase, block=block, line=line)
                phrase_spans = phrase.get("spans", []) or []
                for s_idx, span in enumerate(phrase_spans):
                    previous_span = phrase_spans[s_idx - 1] if s_idx > 0 else None
                    next_span = phrase_spans[s_idx + 1] if s_idx + 1 < len(phrase_spans) else None
                    span["expression_relations"] = {
                        "with_previous": _neighbor_relation(span, previous_span),
                        "with_next": _neighbor_relation(span, next_span),
                    }

            line_policy = aggregate_child_policy(l_policy, line.get("phrases", []) or [])
            line["translatable"] = bool(line_policy["translatable"])
            line["translation_strategy"] = line_policy["translation_strategy"]
            line["coverage_required"] = line_policy["coverage_required"]
            line["render_policy"] = line_policy["render_policy"]
            line["editorial_semantics"] = _editorial_semantics(line)
            line["structural_context"] = _structural_context("line", line, parent=block, block=block, line=line)

        semantic_phrase_list = block.get("semantic_phrases", []) or []
        for sp_idx, semantic_phrase in enumerate(semantic_phrase_list):
            previous_phrase = semantic_phrase_list[sp_idx - 1] if sp_idx > 0 else None
            next_phrase = semantic_phrase_list[sp_idx + 1] if sp_idx + 1 < len(semantic_phrase_list) else None
            hard_break_before = bool(sp_idx == 0 or (semantic_phrase.get("start_line_index", 0) != (previous_phrase or {}).get("end_line_index", -1) + 1))
            next_hard_break = bool((next_phrase or {}).get("start_line_index", 0) != semantic_phrase.get("end_line_index", 0) + 1) if isinstance(next_phrase, dict) else False
            semantic_phrase["editorial_relations"] = {
                "with_previous": _editorial_relation(semantic_phrase, previous_phrase, hard_break=hard_break_before),
                "with_next": _editorial_relation(next_phrase, semantic_phrase, hard_break=next_hard_break) if isinstance(next_phrase, dict) else {"exists": False},
            }

        line_list = block.get("lines", []) or []
        for l_idx, line in enumerate(line_list):
            previous_line = line_list[l_idx - 1] if l_idx > 0 else None
            next_line = line_list[l_idx + 1] if l_idx + 1 < len(line_list) else None
            hard_break_before = bool(line.get("hard_break_before", False))
            next_hard_break = bool(next_line.get("hard_break_before", False)) if isinstance(next_line, dict) else False
            line["editorial_relations"] = {
                "with_previous": _editorial_relation(line, previous_line, hard_break=hard_break_before),
                "with_next": _editorial_relation(next_line, line, hard_break=next_hard_break) if isinstance(next_line, dict) else {"exists": False},
            }

    for b_idx, block in enumerate(blocks or []):
        previous_block = blocks[b_idx - 1] if b_idx > 0 else None
        next_block = blocks[b_idx + 1] if b_idx + 1 < len(blocks) else None
        block["editorial_relations"] = {
            "with_previous": _editorial_relation(block, previous_block, hard_break=False),
            "with_next": _editorial_relation(next_block, block, hard_break=False) if isinstance(next_block, dict) else {"exists": False},
        }


def _detect_leading_marker(text):
    s = (text or "").lstrip()
    if not s:
        return ""
    m = re.match(r"^([•▪◦·\-\*])\s+", s)
    if m:
        return m.group(1)
    m = re.match(r"^((?:\d+|[A-Za-z])[.)])\s+", s)
    if m:
        return m.group(1)
    return ""


def _line_phrase_text(line):
    parts = []
    for p in line.get("phrases", []) or []:
        t = _phrase_source_text(p)
        if t:
            parts.append(t)
    return _normalize_spaces(" ".join(parts))


def _infer_source_layout_mode(block):
    lines = [line for line in (block.get("lines") or []) if isinstance(line, dict)]
    role = str(block.get("role") or "").strip().lower()
    if not lines:
        return {
            "mode": "empty",
            "line_flow": "none",
            "render_contract": "none",
            "preserve_line_breaks": False,
            "preserve_paragraph_breaks": False,
            "can_reflow_within_paragraph": False,
            "line_count": 0,
            "paragraph_count": 0,
            "line_breaks": [],
        }

    line_count = len(lines)
    texts = [_normalize_spaces(line.get("line_text") or line.get("text") or _line_phrase_text(line)) for line in lines]
    markers = [str(line.get("leading_marker") or "").strip() for line in lines]
    marker_count = sum(1 for marker in markers if marker)
    hard_breaks = [bool(line.get("hard_break_before", False)) for line in lines]
    paragraph_count = max(1, sum(1 for value in hard_breaks if value))
    terminal_count = sum(1 for text in texts if re.search(r"[.!?]\s*$", text))
    colon_count = sum(1 for text in texts if text.endswith(":"))
    short_line_count = sum(1 for text in texts if len(re.findall(r"[A-Za-zÀ-ÿ0-9]+", text)) <= 4)
    nonempty_count = sum(1 for text in texts if text)
    all_short = bool(nonempty_count and short_line_count >= max(1, int(nonempty_count * 0.75)))

    fixed_roles = {
        "diagram_label",
        "diagram_text_label",
        "chart_label",
        "figure_label",
        "page_header",
        "page_footer",
        "header",
        "footer",
        "equation_inline",
    }
    caption_roles = {"figure_caption", "table_caption"}
    line_breaks = []
    for idx, line in enumerate(lines):
        reason = "soft_wrap"
        if idx == line_count - 1:
            reason = "block_end"
        elif marker_count:
            reason = "list_item"
        elif idx + 1 < line_count and bool(lines[idx + 1].get("hard_break_before", False)):
            reason = "paragraph_break"
        elif role in fixed_roles or all_short:
            reason = "fixed_line"
        line_breaks.append(
            {
                "line_index": int(line.get("line_index", idx) or idx),
                "after": reason,
                "hard": reason in {"paragraph_break", "list_item", "fixed_line", "block_end"},
            }
        )

    if role in fixed_roles:
        mode = "fixed_labels"
        line_flow = "fixed_lines"
        render_contract = "fixed_slots"
        preserve_line_breaks = True
        can_reflow = False
    elif role in caption_roles:
        mode = "caption"
        line_flow = "paragraph_reflow" if paragraph_count <= 1 else "preserve_paragraphs"
        render_contract = "reflow_block"
        preserve_line_breaks = False
        can_reflow = True
    elif marker_count >= 1:
        mode = "list"
        line_flow = "preserve_line_breaks"
        render_contract = "preserve_breaks"
        preserve_line_breaks = True
        can_reflow = True
    elif line_count == 1:
        mode = "single_line"
        line_flow = "single_line"
        render_contract = "single_line_or_shrink"
        preserve_line_breaks = True
        can_reflow = False
    elif all_short and terminal_count <= 1:
        mode = "labels"
        line_flow = "fixed_lines"
        render_contract = "fixed_slots"
        preserve_line_breaks = True
        can_reflow = False
    elif paragraph_count > 1:
        mode = "paragraphs_with_hard_breaks"
        line_flow = "preserve_paragraphs"
        render_contract = "paragraph_reflow"
        preserve_line_breaks = False
        can_reflow = True
    elif colon_count and line_count <= 3 and terminal_count == 0:
        mode = "lead_in"
        line_flow = "preserve_line_breaks"
        render_contract = "preserve_breaks"
        preserve_line_breaks = True
        can_reflow = True
    else:
        mode = "continuous_paragraph"
        line_flow = "inline_reflow"
        render_contract = "reflow_block"
        preserve_line_breaks = False
        can_reflow = True

    return {
        "mode": mode,
        "line_flow": line_flow,
        "render_contract": render_contract,
        "preserve_line_breaks": bool(preserve_line_breaks),
        "preserve_paragraph_breaks": bool(paragraph_count > 1 or line_flow in {"preserve_paragraphs", "preserve_line_breaks", "fixed_lines"}),
        "can_reflow_within_paragraph": bool(can_reflow),
        "line_count": int(line_count),
        "paragraph_count": int(paragraph_count),
        "leading_marker_count": int(marker_count),
        "terminal_line_count": int(terminal_count),
        "short_line_count": int(short_line_count),
        "line_breaks": line_breaks,
    }


def _apply_llm_layout_mode(block: dict, layout_mode: dict | None) -> None:
    if not isinstance(block, dict) or not isinstance(layout_mode, dict):
        return
    line_flow = str(layout_mode.get("line_flow") or "").strip().lower()
    if line_flow not in {"inline_reflow", "preserve_line_breaks", "preserve_paragraphs", "fixed_lines"}:
        return
    current = dict(block.get("source_layout_mode") or _infer_source_layout_mode(block))
    mode_by_flow = {
        "inline_reflow": "continuous_paragraph",
        "preserve_line_breaks": "explicit_lines",
        "preserve_paragraphs": "paragraphs_with_hard_breaks",
        "fixed_lines": "fixed_labels",
    }
    contract_by_flow = {
        "inline_reflow": "reflow_block",
        "preserve_line_breaks": "preserve_breaks",
        "preserve_paragraphs": "paragraph_reflow",
        "fixed_lines": "fixed_slots",
    }
    breaks = []
    raw_breaks = layout_mode.get("breaks") or []
    valid_breaks = {"soft_wrap", "new_line", "paragraph_break", "list_item", "fixed_line", "block_end"}
    for item in raw_breaks:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("i", -1))
        except (TypeError, ValueError):
            continue
        after = str(item.get("after") or "").strip().lower()
        if idx < 0 or after not in valid_breaks:
            continue
        breaks.append({"line_index": idx, "after": after, "hard": after != "soft_wrap"})
    if not breaks:
        breaks = list(current.get("line_breaks") or [])
    current.update(
        {
            "mode": mode_by_flow[line_flow],
            "line_flow": line_flow,
            "render_contract": contract_by_flow[line_flow],
            "preserve_line_breaks": line_flow in {"preserve_line_breaks", "fixed_lines"},
            "preserve_paragraph_breaks": line_flow in {"preserve_line_breaks", "preserve_paragraphs", "fixed_lines"},
            "can_reflow_within_paragraph": line_flow != "fixed_lines",
            "line_breaks": breaks,
            "llm_refined": True,
        }
    )
    paragraph_breaks = sum(1 for item in breaks if str((item or {}).get("after") or "") == "paragraph_break")
    current["paragraph_count"] = max(1, paragraph_breaks + 1)
    block["source_layout_mode"] = current


def _enrich_layout_markers(blocks):
    # Extraction fidelity metadata used by reconstruction:
    # bullets/dashes, per-line breaks, indentation and paragraph transitions.
    for block in blocks or []:
        lines = block.get("lines", []) or []
        if not lines:
            continue
        lines.sort(key=lambda ln: (float((ln.get("bbox") or [0, 0, 0, 0])[1]), float((ln.get("bbox") or [0, 0, 0, 0])[0])))
        block_line_texts = []
        paragraph_break_before = []
        prev_bbox = None
        prev_indent = None
        avg_h = float(block.get("avg_line_height", 0.0) or 0.0)
        for i, line in enumerate(lines):
            bb = line.get("bbox") or [0, 0, 0, 0]
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                bb = [0, 0, 0, 0]
            x0, y0, x1, y1 = [float(v) for v in bb]
            line_h = max(1.0, y1 - y0)
            if avg_h <= 0.0:
                avg_h = line_h
            text = _line_phrase_text(line)
            marker = _detect_leading_marker(text)
            indent_px = float(x0 - float((block.get("bbox") or [x0, 0, 0, 0])[0]))
            hard_break = False
            if prev_bbox is not None:
                py1 = float(prev_bbox[3])
                v_gap = max(0.0, y0 - py1)
                indent_delta = abs(indent_px - float(prev_indent or 0.0))
                if v_gap > max(2.5, avg_h * 0.42) or indent_delta > max(8.0, avg_h * 0.45):
                    hard_break = True
            line["line_index"] = i
            line["line_text"] = text
            line["leading_marker"] = marker
            line["indent_px"] = indent_px
            line["line_break_after"] = True
            line["hard_break_before"] = bool(i == 0 or hard_break)
            block_line_texts.append(text)
            paragraph_break_before.append(bool(i == 0 or hard_break))
            prev_bbox = bb
            prev_indent = indent_px
            for phrase in line.get("phrases", []) or []:
                phrase["line_index"] = i
                phrase["leading_marker"] = marker
                phrase["indent_px"] = indent_px
                phrase["line_break_after"] = True
                phrase["hard_break_before"] = bool(i == 0 or hard_break)

        block["line_texts"] = block_line_texts
        block["render_text_with_breaks"] = "\n".join([t for t in block_line_texts if t]).strip()
        block["paragraph_break_before"] = paragraph_break_before
        block["source_layout_mode"] = _infer_source_layout_mode(block)


def _semantic_fragment_spans_for_line(line, fragment_bbox):
    """Retourne les spans OCR de la ligne qui correspondent au fragment donné.

    Quand le fragment ne couvre qu'une portion de la ligne (frontière de phrase
    en milieu de ligne), seuls les spans dont la majorité (>50 % de la largeur)
    tombe dans la plage x du fragment sont retenus.  Cela évite que le même
    span full-ligne soit dupliqué entre la fin d'une phrase et le début de la
    suivante — cause principale du bug "texte sur texte".
    """
    spans = []
    if not isinstance(fragment_bbox, (list, tuple)) or len(fragment_bbox) != 4:
        return spans
    frag_rect = fitz.Rect(fragment_bbox)
    frag_x0 = float(fragment_bbox[0])
    frag_x1 = float(fragment_bbox[2])
    frag_width = max(1.0, frag_x1 - frag_x0)

    # Déterminer si le fragment est sub-ligne (couvre < 85 % de la largeur de la ligne)
    line_bb = line.get("bbox")
    if isinstance(line_bb, (list, tuple)) and len(line_bb) == 4:
        line_width = max(1.0, float(line_bb[2]) - float(line_bb[0]))
    else:
        line_width = frag_width
    is_sub_line = frag_width < line_width * 0.85

    def _span_x_overlaps(sp_bb):
        """Retourne True si ≥50 % de la largeur du span est dans la plage x du fragment."""
        if not isinstance(sp_bb, (list, tuple)) or len(sp_bb) != 4:
            return True  # pas de bbox fiable → on garde
        sp_x0 = float(sp_bb[0])
        sp_x1 = float(sp_bb[2])
        sp_width = max(1.0, sp_x1 - sp_x0)
        inter = max(0.0, min(frag_x1, sp_x1) - max(frag_x0, sp_x0))
        return (inter / sp_width) >= 0.5

    for phrase in line.get("phrases", []) or []:
        pbb = phrase.get("bbox")
        if not isinstance(pbb, (list, tuple)) or len(pbb) != 4:
            continue
        p_rect = fitz.Rect(pbb)
        overlap = (frag_rect & p_rect).get_area()
        if overlap <= 0 and not frag_rect.intersects(p_rect):
            continue
        phrase_spans = phrase.get("spans", []) or []
        if phrase_spans:
            for sp in phrase_spans:
                if is_sub_line and not _span_x_overlaps(sp.get("bbox")):
                    continue  # exclure les spans qui tombent hors de la plage x du fragment
                spans.append(copy.deepcopy(sp))
        else:
            if not is_sub_line or _span_x_overlaps(pbb):
                spans.append(
                    {
                        "texte": _normalize_spaces(phrase.get("texte", "")),
                        "bbox": phrase.get("bbox"),
                        "style": copy.deepcopy(phrase.get("style", {})) if isinstance(phrase.get("style"), dict) else {},
                    }
                )
    return spans


def _approximate_line_words_from_text(line):
    text = _normalize_spaces(line.get("line_text") or line.get("text") or "")
    bb = line.get("bbox")
    if not text or not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return []
    x0, y0, x1, y1 = [float(v) for v in bb]
    tokens = text.split()
    if not tokens:
        return []
    total_chars = max(1, sum(len(tok) for tok in tokens) + max(0, len(tokens) - 1))
    cursor = x0
    words = []
    for idx, token in enumerate(tokens):
        token_chars = len(token)
        token_w = max(4.0, (x1 - x0) * (token_chars / total_chars))
        words.append(
            {
                "label": token,
                "bbox": [cursor, y0, min(x1, cursor + token_w), y1],
                "score": float(line.get("ocr_confidence_mean", 0.0) or 0.0),
            }
        )
        cursor = min(x1, cursor + token_w + max(2.0, (x1 - x0) * (1.0 / total_chars)))
    return words


def _split_line_text_into_sentence_chunks(text):
    s = _normalize_spaces(text or "")
    if not s:
        return []
    chunks = []
    start = 0
    _SENT_END_RE = re.compile(r'[.!?\u2026]+[\u201d\u2019"\')\]}\u00bb]*')
    for match in _SENT_END_RE.finditer(s):
        end = match.end()
        chunk_text = s[start:end].strip()
        if not chunk_text:
            continue
        last_token = chunk_text.split()[-1] if chunk_text.split() else ""
        next_part = s[end:]
        if _is_abbreviation_token(last_token):
            continue
        if next_part and not re.match(r'^\s+[A-Z\u201c\u2018"\'(\[]', next_part) and next_part.strip():
            continue
        chunks.append({"text": chunk_text, "start": start, "end": end, "ends_sentence": True})
        start = end
        while start < len(s) and s[start].isspace():
            start += 1
    if start < len(s):
        tail = s[start:].strip()
        if tail:
            tail_start = s.find(tail, start)
            chunks.append({"text": tail, "start": max(start, tail_start), "end": max(start, tail_start) + len(tail), "ends_sentence": False})

    # Découpage doux pour les lignes académiques denses sans ponctuation terminale :
    # si une seule chunk reste et dépasse 40 mots, on tente de couper à une frontière naturelle
    # (point-virgule, deux-points suivi de majuscule, virgule devant conjonction).
    # Cela évite les spans gloutons sur les paragraphes académiques continus.
    if len(chunks) == 1 and not chunks[0].get("ends_sentence"):
        chunk_text = chunks[0]["text"]
        words = chunk_text.split()
        if len(words) > 20:
            # Cherche un point de coupe naturel autour du milieu du texte
            mid = len(chunk_text) // 2
            best_pos = None
            # Priorité : "; " ou ": [A-Z]" près du milieu
            for m in re.finditer(r";[ ]|:[ ][A-Z]|,[ ](and|or|but|while|whereas|however|although|yet)\b", chunk_text, flags=re.IGNORECASE):
                pos = m.start() + 1  # coupe après le signe de ponctuation
                if best_pos is None or abs(pos - mid) < abs(best_pos - mid):
                    best_pos = pos
            if best_pos and best_pos > 10 and best_pos < len(chunk_text) - 10:
                part1 = chunk_text[:best_pos].strip()
                part2 = chunk_text[best_pos:].strip()
                if part1 and part2:
                    orig_start = chunks[0]["start"]
                    chunks = [
                        {"text": part1, "start": orig_start, "end": orig_start + best_pos, "ends_sentence": False},
                        {"text": part2, "start": orig_start + best_pos, "end": orig_start + len(chunk_text), "ends_sentence": False},
                    ]
    return chunks


def _span_based_fragment_bbox(line, content, start, end, x0, y0, x1, y1):
    """Estime la bbox d'un fragment [start, end] en s'appuyant sur les bboxes des spans OCR réels.

    Retourne None si la correspondance texte→spans échoue (fallback proportionnel attendu).
    """
    span_data = []
    for phrase in (line.get("phrases") or []):
        for span in (phrase.get("spans") or []):
            text = _normalize_spaces(span.get("texte") or span.get("text") or "")
            bb = _clean_bbox(span.get("bbox"))
            if text and bb:
                span_data.append((text, float(bb[0]), float(bb[2])))
    if not span_data:
        return None
    # Aligner les spans sur le texte de la ligne pour obtenir des positions caractère
    cursor = 0
    positions = []
    for text, sx0, sx1 in span_data:
        pos = content.find(text, cursor)
        if pos == -1:
            return None  # span non localisable → abandon, fallback proportionnel
        positions.append((pos, pos + len(text), sx0, sx1))
        cursor = pos + len(text)
    # Interpolation proportionnelle au sein de chaque span qui intersecte [start, end].
    # On N'utilise PAS l'union des bboxes entières : un span large (ex. pleine ligne)
    # recevrait la même bbox pour deux fragments adjacents → is_sub_line=False → CRIT-4.
    # En interpolant t0/t1 dans la plage de caractères du span, chaque fragment reçoit
    # une tranche x proportionnelle à sa position dans le span, garantissant des bboxes
    # non-chevauchantes entre fragments consécutifs issus de la même ligne.
    frag_x0 = None
    frag_x1 = None
    for cstart, cend, sx0, sx1 in positions:
        if cend > start and cstart < end:
            span_len = max(1, cend - cstart)
            overlap_s = max(start, cstart)
            overlap_e = min(end, cend)
            t0 = (overlap_s - cstart) / span_len
            t1 = (overlap_e - cstart) / span_len
            ix0 = sx0 + (sx1 - sx0) * t0
            ix1 = sx0 + (sx1 - sx0) * t1
            if frag_x0 is None or ix0 < frag_x0:
                frag_x0 = ix0
            if frag_x1 is None or ix1 > frag_x1:
                frag_x1 = ix1
    if frag_x0 is None:
        return None
    frag_x0 = max(x0, frag_x0)
    frag_x1 = min(x1, max(frag_x0 + 4.0, frag_x1))
    return [int(round(frag_x0)), int(round(y0)), int(round(frag_x1)), int(round(y1))]


def _approximate_text_fragment_bbox(line_bbox, full_text, start, end, line=None):
    bb = _clean_bbox(line_bbox)
    if not bb:
        return [0, 0, 0, 0]
    x0, y0, x1, y1 = bb
    content = _normalize_spaces(full_text or "")
    total = max(1, len(content))
    start = max(0, min(int(start or 0), total - 1))
    end = max(start + 1, min(int(end or total), total))
    if line is not None:
        span_bb = _span_based_fragment_bbox(line, content, start, end, x0, y0, x1, y1)
        if span_bb is not None:
            return span_bb
    fx0 = x0 + (x1 - x0) * (start / total)
    fx1 = x0 + (x1 - x0) * (end / total)
    return [int(round(fx0)), int(round(y0)), int(round(max(fx0 + 4.0, fx1))), int(round(y1))]


def _make_semantic_phrase(block, sentence_index, sentence_fragments, end_reason=""):
    fragments_in = [frag for frag in (sentence_fragments or []) if _normalize_spaces(frag.get("text") or "")]
    if not fragments_in:
        return None
    text = ""
    for frag in fragments_in:
        text = _append_fragment(text, frag.get("text") or "")
    text = _normalize_spaces(text)
    if not text:
        return None
    bbox = [
        int(min(float(frag["bbox"][0]) for frag in fragments_in)),
        int(min(float(frag["bbox"][1]) for frag in fragments_in)),
        int(max(float(frag["bbox"][2]) for frag in fragments_in)),
        int(max(float(frag["bbox"][3]) for frag in fragments_in)),
    ]
    fragments = []
    spans = []
    line_indices = []
    confidence_values = []
    for frag_index, frag in enumerate(fragments_in):
        frag_text = _normalize_spaces(frag.get("text") or "")
        frag_bbox = list(frag.get("bbox") or [0, 0, 0, 0])
        line_obj = frag.get("line") or {}
        line_index = int(frag.get("line_index", 0) or 0)
        line_indices.append(line_index)
        if line_obj.get("ocr_confidence_mean") is not None:
            confidence_values.append(float(line_obj.get("ocr_confidence_mean", 0.0) or 0.0))
        fragment_spans = _semantic_fragment_spans_for_line(line_obj, frag_bbox)
        if fragment_spans:
            spans.extend(fragment_spans)
        elif frag_text:
            # Fallback : aucun span OCR ne couvre ce fragment.
            # On hérite le style du premier span disponible dans la ligne pour
            # que la reconstruction puisse appliquer la bonne police/couleur.
            best_style: dict = {}
            for _ph in (line_obj.get("phrases") or []):
                for _sp in (_ph.get("spans") or []):
                    if _sp.get("style"):
                        best_style = copy.deepcopy(_sp["style"])
                        break
                if best_style:
                    break
            spans.append({"texte": frag_text, "bbox": frag_bbox, "style": best_style})
        fragments.append(
            {
                "fragment_index": frag_index,
                "line_index": line_index,
                "text": frag_text,
                "bbox": frag_bbox,
                "word_count": int(_text_statistics(frag_text).get("word_count", 0) or 0),
                "source_line_text": line_obj.get("line_text", ""),
            }
        )
    # Dédupliquer les spans identiques accumulés depuis plusieurs fragments
    # (arrive quand deux bboxes de fragments adjacents se chevauchent légèrement).
    seen_span_keys: set = set()
    deduped_spans = []
    for sp in spans:
        key = (tuple(sp.get("bbox") or []), sp.get("texte") or sp.get("text") or "")
        if key not in seen_span_keys:
            seen_span_keys.add(key)
            deduped_spans.append(sp)
    spans = deduped_spans
    if not spans:
        spans = [{"texte": text, "bbox": bbox, "style": {}}]
    line_indices = sorted(set(line_indices))
    return {
        "texte": text,
        "text": text,
        "bbox": bbox,
        "spans": spans,
        "fragments": fragments,
        "source": str(block.get("source") or "ocr"),
        "source_kind": "ocr_semantic_phrase",
        "sentence_index": int(sentence_index),
        "start_line_index": int(line_indices[0]) if line_indices else 0,
        "end_line_index": int(line_indices[-1]) if line_indices else 0,
        "line_indices": line_indices,
        "multi_line": bool(len(line_indices) > 1),
        "fragment_count": len(fragments),
        "ocr_word_count": int(_text_statistics(text).get("word_count", 0) or 0),
        "ocr_confidence_mean": float(np.mean(confidence_values)) if confidence_values else 0.0,
        "ocr_confidence_min": float(np.min(confidence_values)) if confidence_values else 0.0,
        "sentence_end_reason": end_reason or "eof",
    }


_MAX_SEMANTIC_PHRASE_WORDS = 50  # plafond anti-span-glouton pour les blocs denses

# Rôles de blocs dont chaque ligne est une unité sémantique atomique indivisible.
# Pour ces blocs, on ne fait pas de reconstruction cross-line : 1 ligne = 1 semantic_phrase.
_ATOMIC_BLOCK_ROLES = frozenset({
    "diagram_label",
    "diagram_text_label",
    "chart_label",
    "figure_caption",
    "page_header",
    "page_footer",
})

# Mots fonctionnels de prose : déterminants, conjonctions, auxiliaires, pronoms.
# Leur présence signale du texte continu, pas des labels/tableaux.
_PROSE_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "from",
    "and", "or", "but", "that", "this", "which", "when", "if", "as", "while",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "not", "it", "its", "we", "they", "their", "our", "you", "your", "he",
    "she", "his", "her", "also", "then", "however", "therefore", "thus",
    "so", "because", "since", "although", "though", "both", "either",
})


def _block_is_implicitly_atomic(lines: list) -> tuple:
    """Detecte si un bloc doit etre traite ligne par ligne (mode atomique implicite).

    Analyse les lignes pour identifier colonnes de tableau, labels de diagramme,
    valeurs numeriques — sans dependre d'un role preassigne par le layout AI.

    Score positif = label/tableau  |  penalite si prose forte.
    Seuil d'activation : 5.0

    Retourne (is_atomic: bool, reason: str).
    """
    if len(lines) < 2:
        return False, ""

    texts = [
        _normalize_spaces(ln.get("line_text") or ln.get("text") or "")
        for ln in lines
    ]
    texts = [t for t in texts if t]
    n = len(texts)
    if n < 2:
        return False, ""

    tokens_per_line = [t.split() for t in texts]
    word_counts = [len(toks) for toks in tokens_per_line]
    avg_words = sum(word_counts) / n
    max_words = max(word_counts)

    short_line_ratio = sum(1 for w in word_counts if w <= 5) / n
    terminal_punct_ratio = sum(1 for t in texts if t[-1] in ".!?;") / n

    all_tokens_lower = [w.lower() for toks in tokens_per_line for w in toks]
    total_words = max(1, len(all_tokens_lower))
    prose_count = sum(1 for w in all_tokens_lower if w in _PROSE_FUNCTION_WORDS)
    prose_ratio = prose_count / total_words

    unique_tokens = set(all_tokens_lower)
    repetition_ratio = 1.0 - (len(unique_tokens) / max(1, total_words))

    bboxes = [ln.get("bbox") for ln in lines if isinstance(ln.get("bbox"), (list, tuple))]
    is_column = False
    if bboxes:
        bw = max(b[2] for b in bboxes) - min(b[0] for b in bboxes)
        bh = max(b[3] for b in bboxes) - min(b[1] for b in bboxes)
        is_column = bw > 0 and (bh / bw) >= 1.8

    math_line_count = sum(1 for t in texts if _MATH_CHARS_RE.search(t))
    math_dominant = math_line_count / n >= 0.6

    score = 0.0
    reasons: list = []

    if avg_words <= 3 and max_words <= 5:
        score += 3.0
        reasons.append("very_short(avg={:.1f})".format(avg_words))
    elif short_line_ratio >= 0.85:
        score += 2.0
        reasons.append("short_lines({:.0%})".format(short_line_ratio))
    elif short_line_ratio >= 0.70:
        score += 1.0

    if terminal_punct_ratio == 0.0:
        score += 2.0
        reasons.append("no_terminal_punct")
    elif terminal_punct_ratio < 0.15:
        score += 0.5

    if prose_ratio < 0.05:
        score += 2.0
        reasons.append("no_prose_words")
    elif prose_ratio < 0.12:
        score += 1.0

    if repetition_ratio >= 0.5:
        score += 2.0
        reasons.append("high_repetition({:.0%})".format(repetition_ratio))
    elif repetition_ratio >= 0.3:
        score += 1.0

    if is_column:
        score += 1.0
        reasons.append("column_shape")

    if math_dominant:
        score += 1.0
        reasons.append("math_content")

    if n >= 4:
        score += 0.5

    if prose_ratio >= 0.20:
        score -= 2.5
        reasons.append("PROSE_PENALTY({:.0%})".format(prose_ratio))
    elif prose_ratio >= 0.10:
        score -= 1.0

    is_atomic = score >= 5.0
    reason = ", ".join(reasons) if is_atomic else ""
    return is_atomic, reason

# Ligne séparatrice : uniquement composée de tirets, underscores, signes égal, etc.
# Typique des barres de fraction, lignes horizontales OCR, séparateurs visuels.
_SEPARATOR_LINE_RE = re.compile(r"^[-_=~\u2014\u2015\u2500-\u257F]{2,}$")


def _line_is_separator(text: str) -> bool:
    """Retourne True si la ligne est une barre séparatrice pure (fraction, HR…)."""
    t = _normalize_spaces(text or "").strip()
    return bool(t) and bool(_SEPARATOR_LINE_RE.match(t))


_MATH_CHARS_RE = re.compile(r"[0-9+\-*/=<>^×÷±∑∫∂∞√π()[\]{}]")


def _separator_is_math_fraction(prev_text: str, sep_text: str, next_text: str) -> bool:
    """Retourne True si la ligne séparatrice est une barre de fraction mathématique.

    Critères : séparateur entre deux lignes courtes dont au moins une contient
    des symboles mathématiques. On ne fusionne que si les lignes adjacentes
    sont suffisamment courtes pour être un numérateur/dénominateur.
    """
    if not next_text:
        return False
    if len(prev_text.split()) > 15 or len(next_text.split()) > 15:
        return False
    return bool(_MATH_CHARS_RE.search(prev_text) or _MATH_CHARS_RE.search(next_text))


def _line_is_all_caps_heading(text: str) -> bool:
    """Retourne True si la ligne ressemble à un titre ALL-CAPS (ex. 'THE DIRECTION (GRADIENT)').

    Critère : au moins 4 caractères alpha, dont ≥ 80 % sont des majuscules.
    Exclut les lignes qui se terminent par une ponctuation terminale (déjà gérées ailleurs).
    """
    t = _normalize_spaces(text or "").strip()
    if not t or len(t) < 4:
        return False
    alpha = [c for c in t if c.isalpha()]
    if len(alpha) < 3:
        return False
    upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
    return upper_ratio >= 0.80


def _build_atomic_semantic_phrases(block, lines):
    """Construit une semantic_phrase par ligne pour les blocs atomiques (diagram_label, etc.)."""
    semantic_phrases = []
    for i, line in enumerate(lines):
        line_text = _normalize_spaces(line.get("line_text") or line.get("text") or "")
        if not line_text:
            continue
        line_bbox = line.get("bbox") or [0, 0, 0, 0]
        frag = {
            "line_index": int(line.get("line_index", i) or i),
            "line": line,
            "text": line_text,
            "bbox": list(line_bbox),
        }
        phrase = _make_semantic_phrase(block, i, [frag], end_reason="atomic_line")
        if phrase:
            semantic_phrases.append(phrase)
    return semantic_phrases


def _build_semantic_phrases_for_block(block):
    existing_semantic_phrases = copy.deepcopy(block.get("semantic_phrases") or [])
    lines = sorted(
        [ln for ln in (block.get("lines") or []) if isinstance(ln, dict)],
        key=lambda ln: (int(ln.get("line_index", 0) or 0), float((ln.get("bbox") or [0, 0, 0, 0])[1])),
    )

    # Blocs atomiques : 1 ligne = 1 semantic_phrase, sans reconstruction cross-line.
    # Deux chemins pour détecter ce cas :
    #   1. Rôle explicite assigné par le layout AI (diagram_label, figure_caption…)
    #   2. Détection implicite basée sur le contenu (colonnes de tableau, labels courts,
    #      texte répétitif sans prose) — _block_is_implicitly_atomic().
    block_role = str(block.get("role") or "").lower()
    implicit_atomic, implicit_reason = _block_is_implicitly_atomic(lines)
    if block_role in _ATOMIC_BLOCK_ROLES or implicit_atomic:
        semantic_phrases = _build_atomic_semantic_phrases(block, lines)
        for sp in semantic_phrases:
            sentence_id = f"{block.get('id') or block.get('unit_id') or 'block'}:semantic_phrase:{sp.get('sentence_index', 0)}"
            sp["sentence_id"] = sentence_id
            for frag in sp.get("fragments", []) or []:
                frag["sentence_id"] = sentence_id
        block["semantic_phrases"] = semantic_phrases
        block["semantic_phrase_count"] = len(semantic_phrases)
        if implicit_atomic and block_role not in _ATOMIC_BLOCK_ROLES:
            block["implicit_atomic"] = True
            block["implicit_atomic_reason"] = implicit_reason
        return

    semantic_phrases = []
    current_fragments = []
    sentence_index = 0
    current_text = ""

    def flush(end_reason=""):
        nonlocal current_fragments, sentence_index, current_text
        phrase = _make_semantic_phrase(block, sentence_index, current_fragments, end_reason=end_reason)
        current_fragments = []
        current_text = ""
        if phrase:
            semantic_phrases.append(phrase)
            sentence_index += 1

    def _current_word_count():
        return len(re.findall(r"\S+", current_text))

    for line_pos, line in enumerate(lines):
        line_text = _normalize_spaces(line.get("line_text") or line.get("text") or "")
        previous_line = lines[line_pos - 1] if line_pos > 0 else None

        # Fix P3 — ligne séparatrice (barre de fraction "------", HR visuel…) :
        # Si le contexte est mathématique (lignes courtes avec symboles), on ne
        # coupe pas la phrase — numérateur et dénominateur restent groupés.
        # Sinon on flush ce qui précède et on ignore la ligne séparatrice.
        if line_text and _line_is_separator(line_text):
            prev_text = _normalize_spaces(previous_line.get("line_text") or previous_line.get("text") or "") if previous_line else ""
            next_line_obj = lines[line_pos + 1] if line_pos + 1 < len(lines) else None
            next_text = _normalize_spaces(next_line_obj.get("line_text") or next_line_obj.get("text") or "") if next_line_obj else ""
            if _separator_is_math_fraction(prev_text, line_text, next_text):
                continue  # barre de fraction : ne pas couper, ne pas ajouter aux fragments
            if current_fragments:
                flush(end_reason="separator_line")
            continue

        if current_fragments and bool(line.get("hard_break_before", False)) and _semantic_phrase_should_break_on_hard_boundary(current_text, line, previous_line):
            flush(end_reason="hard_break_before")
        # Plafond anti-glouton : si la phrase en cours dépasse MAX mots, on flush
        # au prochain saut de ligne même sans ponctuation terminale.
        # Cela casse les paragraphes académiques longs en unités exploitables.
        elif current_fragments and _current_word_count() >= _MAX_SEMANTIC_PHRASE_WORDS:
            flush(end_reason="word_count_cap")
        if not line_text:
            continue
        line_chunks = _split_line_text_into_sentence_chunks(line_text)
        if not line_chunks:
            line_chunks = [{"text": line_text, "start": 0, "end": len(line_text), "ends_sentence": False}]
        for chunk in line_chunks:
            current_text = _append_fragment(current_text, chunk.get("text") or "")
            current_fragments.append(
                {
                    "line_index": int(line.get("line_index", line_pos) or line_pos),
                    "line": line,
                    "text": chunk.get("text") or "",
                    "bbox": _approximate_text_fragment_bbox(line.get("bbox"), line_text, chunk.get("start", 0), chunk.get("end", len(line_text)), line=line),
                }
            )
            if bool(chunk.get("ends_sentence")):
                flush(end_reason="terminal_punctuation")

        # Fix P1 — titre ALL-CAPS (ex. "THE DIRECTION (GRADIENT)") :
        # après avoir accumulé la ligne, on la flush immédiatement pour éviter
        # qu'elle soit fusionnée avec la prose qui suit.
        # On ne flush que si la ligne entière (sans ses chunks fractionnés par ponctuation)
        # forme encore un fragment en attente (i.e. n'a pas déjà été flushé par terminal_punctuation).
        if current_fragments and _line_is_all_caps_heading(line_text):
            flush(end_reason="heading_line")

    if current_fragments:
        flush(end_reason="eof")

    if not semantic_phrases and existing_semantic_phrases:
        semantic_phrases = existing_semantic_phrases

    for semantic_phrase in semantic_phrases:
        sentence_id = f"{block.get('id') or block.get('unit_id') or 'block'}:semantic_phrase:{semantic_phrase.get('sentence_index', 0)}"
        semantic_phrase["sentence_id"] = sentence_id
        for frag in semantic_phrase.get("fragments", []) or []:
            frag["sentence_id"] = sentence_id
    block["semantic_phrases"] = semantic_phrases
    block["semantic_phrase_count"] = len(semantic_phrases)


def _build_semantic_phrases_for_blocks(blocks):
    for block in blocks or []:
        _build_semantic_phrases_for_block(block)


# ---------------------------------------------------------------------------
# Intégration LLM post-processeur (correcteur de segmentation ambiguë)
# ---------------------------------------------------------------------------

def _build_llm_split_fragments(line: dict, split_texts: list[str]) -> list[dict]:
    """Construit des fragments intra-ligne à partir de segments textuels renvoyés par le LLM."""
    line_text = _normalize_spaces(line.get("line_text") or line.get("text") or "")
    normalized_splits = [_normalize_spaces(text) for text in (split_texts or []) if _normalize_spaces(text)]
    if not line_text or len(normalized_splits) < 2:
        return []

    fragments = []
    cursor = 0
    exact_match = True
    for split_text in normalized_splits:
        pos = line_text.find(split_text, cursor)
        if pos == -1:
            exact_match = False
            break
        end = pos + len(split_text)
        fragments.append(
            {
                "line_index": int(line.get("line_index", 0) or 0),
                "line": line,
                "text": split_text,
                "bbox": _approximate_text_fragment_bbox(line.get("bbox"), line_text, pos, end, line=line),
            }
        )
        cursor = end

    if exact_match and len(fragments) >= 2:
        return fragments

    total_chars = sum(max(1, len(text)) for text in normalized_splits)
    if total_chars <= 0:
        return []

    proportional = []
    consumed = 0
    current_start = 0
    for idx, split_text in enumerate(normalized_splits):
        if idx == len(normalized_splits) - 1:
            current_end = len(line_text)
        else:
            consumed += max(1, len(split_text))
            current_end = max(current_start + 1, min(len(line_text), round(len(line_text) * (consumed / total_chars))))
        proportional.append(
            {
                "line_index": int(line.get("line_index", 0) or 0),
                "line": line,
                "text": split_text,
                "bbox": _approximate_text_fragment_bbox(line.get("bbox"), line_text, current_start, current_end, line=line),
            }
        )
        current_start = current_end
    return proportional


def _semantic_phrase_quality_issues(semantic_phrases: list[dict]) -> list[dict]:
    """Detect obvious regressions introduced by semantic re-segmentation.

    The LLM is allowed to move boundaries, but it must not duplicate the seam
    between two consecutive semantic phrases. Those artifacts are worse than a
    conservative heuristic split because they leak repeated text into
    translation and rendering.
    """
    issues: list[dict] = []
    texts = [_normalize_spaces((phrase or {}).get("text") or (phrase or {}).get("texte") or "") for phrase in (semantic_phrases or [])]

    def tokenise(text: str) -> list[str]:
        return [tok.lower() for tok in re.findall(r"[A-Za-zÀ-ÿ0-9]+", text or "")]

    for idx, text in enumerate(texts):
        if not text:
            continue
        if re.search(r"\b[A-Za-zÀ-ÿ]+-\s+[A-Za-zÀ-ÿ]+\b", text):
            issues.append({"type": "hyphen_fragment", "index": idx, "text": text})
        # Typical bad split: "sentence. This" or "sentence. It has a 53-layer"
        # kept at the end of the previous phrase while the next phrase starts
        # with the same tail.
        sentence_tail = re.search(
            r"[.!?]\s+([A-Z][A-Za-zÀ-ÿ0-9'\-]*(?:\s+[A-Za-zÀ-ÿ0-9'\-]+){0,5})$",
            text,
        )
        if sentence_tail and not re.search(r"[.!?][\"')\]]*$", text):
            issues.append({"type": "dangling_sentence_tail", "index": idx, "tail": sentence_tail.group(1), "text": text})

    for idx in range(1, len(texts)):
        prev_text = texts[idx - 1]
        next_text = texts[idx]
        if not prev_text or not next_text:
            continue
        prev_tokens = tokenise(prev_text)
        next_tokens = tokenise(next_text)
        if not prev_tokens or not next_tokens:
            continue
        max_k = min(12, len(prev_tokens), len(next_tokens))
        for k in range(max_k, 2, -1):
            if prev_tokens[-k:] == next_tokens[:k]:
                issues.append({"type": "edge_token_overlap", "index": idx, "tokens": prev_tokens[-k:]})
                break
        else:
            # Single-token overlap is common for this failure mode ("This",
            # "YOLOv3"). Ignore tiny tokens to avoid false positives on labels.
            if prev_tokens[-1] == next_tokens[0] and len(prev_tokens[-1]) >= 4:
                issues.append({"type": "edge_token_overlap", "index": idx, "tokens": [prev_tokens[-1]]})

    return issues


def _llm_semantic_phrases_are_quality_regression(before: list[dict], after: list[dict]) -> bool:
    if not before or not after:
        return False
    before_issues = _semantic_phrase_quality_issues(before)
    after_issues = _semantic_phrase_quality_issues(after)
    if len(after_issues) <= len(before_issues):
        return False
    # Tolérance d'un seul problème supplémentaire quand le texte est intégralement
    # préservé (couverture ≥ 95 % des tokens). Évite de rejeter des corrections
    # LLM globalement bonnes à cause d'un artefact mineur de jonction.
    if len(after_issues) == len(before_issues) + 1:
        before_words = set(re.findall(r"[A-Za-zÀ-ɏ0-9]+", " ".join(
            _normalize_spaces(p.get("text") or "") for p in before
        ).lower()))
        after_words = set(re.findall(r"[A-Za-zÀ-ɏ0-9]+", " ".join(
            _normalize_spaces(p.get("text") or "") for p in after
        ).lower()))
        coverage = len(before_words & after_words) / max(1, len(before_words))
        if coverage >= 0.95:
            return False
    return True


def _apply_llm_corrections(
    block: dict,
    corrections: list,
    sentence_boundaries: list | None = None,
    line_boundaries: list[int] | None = None,
) -> None:
    """Reconstruit les semantic_phrases d'un bloc selon les corrections LLM.

    Actions supportées (cf. llm_semantic_corrector.py) :
    - "heading"  → la ligne devient une phrase autonome (titre)
    - "formula"  → groupe de lignes fusionné, translatable=False
    - "skip"     → ligne exclue du contenu sémantique
    - "atomic"   → chaque ligne du groupe = 1 phrase indépendante
    - "keep"     → conserver la décision heuristique (pas de rebuild pour ces lignes)

    sentence_boundaries permet en plus de découper une ligne en plusieurs
    semantic_phrases quand le LLM a identifié plusieurs unités sémantiques
    sur la même ligne.

    line_boundaries permet de forcer une frontière de phrase APRÈS certaines
    lignes, même sans ponctuation terminale fiable.

    Si toutes les corrections sont "keep" (ou la liste est vide), le bloc n'est pas modifié.
    """
    sentence_boundaries = sentence_boundaries or []
    line_boundaries = line_boundaries or []
    if not corrections and not sentence_boundaries and not line_boundaries:
        return

    # Si tout est "keep", rien à faire
    if corrections and all(c.get("a") == "keep" for c in corrections) and not sentence_boundaries and not line_boundaries:
        return

    # Construire un mapping line_index → action
    line_action: dict[int, str] = {}
    for corr in corrections:
        action = str(corr.get("a") or "keep")
        for li in (corr.get("li") or []):
            line_action[int(li)] = action

    line_splits: dict[int, list[str]] = {}
    for split in sentence_boundaries:
        try:
            line_index = int(split.get("i", -1))
        except (ValueError, TypeError, AttributeError):
            continue
        split_texts = split.get("s") if isinstance(split, dict) else None
        if line_index < 0 or not isinstance(split_texts, list):
            continue
        normalized_splits = [_normalize_spaces(text) for text in split_texts if _normalize_spaces(text)]
        if len(normalized_splits) >= 2:
            line_splits[line_index] = normalized_splits

    forced_line_breaks = {int(idx) for idx in line_boundaries if isinstance(idx, int) or str(idx).isdigit()}

    lines_sorted = sorted(
        [ln for ln in (block.get("lines") or []) if isinstance(ln, dict)],
        key=lambda ln: (int(ln.get("line_index", 0) or 0), float((ln.get("bbox") or [0, 0, 0, 0])[1])),
    )

    new_sps: list[dict] = []
    sentence_index = 0
    buffer_frags: list[dict] = []
    buffer_is_formula = False

    def flush_buffer(reason: str, is_formula: bool = False) -> None:
        nonlocal sentence_index, buffer_frags, buffer_is_formula
        if not buffer_frags:
            return
        phrase = _make_semantic_phrase(block, sentence_index, buffer_frags, end_reason=reason)
        if phrase:
            if is_formula:
                phrase["translatable"] = False
                phrase["formula"] = True
            new_sps.append(phrase)
            sentence_index += 1
        buffer_frags = []
        buffer_is_formula = False

    for ln in lines_sorted:
        li = int(ln.get("line_index", 0) or 0)
        line_text = _normalize_spaces(ln.get("line_text") or ln.get("text") or "")
        action = line_action.get(li, "keep")

        if action == "skip":
            # Flush le buffer courant, ignorer cette ligne
            flush_buffer("separator_line", buffer_is_formula)
            continue

        if action == "heading":
            # Flush le buffer courant, puis créer une phrase autonome pour ce titre
            flush_buffer("pre_heading", buffer_is_formula)
            if line_text:
                heading_frag = {
                    "line_index": li,
                    "line": ln,
                    "text": line_text,
                    "bbox": list(ln.get("bbox") or [0, 0, 0, 0]),
                }
                phrase = _make_semantic_phrase(block, sentence_index, [heading_frag], end_reason="heading_line")
                if phrase:
                    new_sps.append(phrase)
                    sentence_index += 1
            continue

        if action == "atomic":
            # Flush le buffer courant, puis 1 phrase par ligne dans ce groupe
            flush_buffer("pre_atomic", buffer_is_formula)
            if line_text:
                frag = {
                    "line_index": li,
                    "line": ln,
                    "text": line_text,
                    "bbox": list(ln.get("bbox") or [0, 0, 0, 0]),
                }
                phrase = _make_semantic_phrase(block, sentence_index, [frag], end_reason="atomic_line")
                if phrase:
                    new_sps.append(phrase)
                    sentence_index += 1
            continue

        if action == "formula":
            # Si on change de mode, flush le buffer non-formula
            if buffer_frags and not buffer_is_formula:
                flush_buffer("pre_formula", False)
            if line_text:
                buffer_frags.append({
                    "line_index": li,
                    "line": ln,
                    "text": line_text,
                    "bbox": list(ln.get("bbox") or [0, 0, 0, 0]),
                })
                buffer_is_formula = True
            continue

        # "keep" ou action inconnue → accumulate normalement
        if buffer_is_formula and buffer_frags:
            # On sortait d'un groupe formula, flush avant de continuer en mode normal
            flush_buffer("formula", True)
        if li in line_splits:
            split_frags = _build_llm_split_fragments(ln, line_splits[li])
            if split_frags:
                current_split_index = 0
                if buffer_frags:
                    buffer_frags.append(split_frags[0])
                    flush_buffer("llm_sentence_boundary", False)
                    current_split_index = 1
                elif split_frags:
                    buffer_frags.append(split_frags[0])
                    current_split_index = 1
                for frag in split_frags[current_split_index:]:
                    flush_buffer("llm_sentence_boundary", False)
                    buffer_frags.append(frag)
                continue
        if line_text:
            buffer_frags.append({
                "line_index": li,
                "line": ln,
                "text": line_text,
                "bbox": list(ln.get("bbox") or [0, 0, 0, 0]),
            })
        if li in forced_line_breaks:
            flush_buffer("llm_line_boundary", False)

    flush_buffer("eof", buffer_is_formula)

    if new_sps:
        # Réattribuer les sentence_id
        for sp in new_sps:
            sid = f"{block.get('id') or block.get('unit_id') or 'block'}:semantic_phrase:{sp.get('sentence_index', 0)}"
            sp["sentence_id"] = sid
            for frag in sp.get("fragments", []) or []:
                frag["sentence_id"] = sid
        block["semantic_phrases"] = new_sps
        block["semantic_phrase_count"] = len(new_sps)
        block["llm_corrected"] = True


def _apply_hyphen_joins(block: dict, hyphen_joins: list) -> None:
    """Applique les résolutions de césures de mots proposées par le LLM.

    Pour chaque entrée {"i": line_index, "w": "full_word"}, retrouve le mot
    fragment tronqué en fin de ligne (se terminant par "-") et le remplace
    par le mot complet dans line_text. Supprime aussi le fragment de début
    de la ligne suivante.
    """
    if not hyphen_joins:
        return
    # Mapping line_index → full word
    hj_map: dict[int, str] = {}
    for hj in hyphen_joins:
        try:
            hj_map[int(hj["i"])] = str(hj.get("w") or "").strip()
        except (KeyError, ValueError, TypeError):
            continue
    if not hj_map:
        return

    lines_sorted = sorted(
        [ln for ln in (block.get("lines") or []) if isinstance(ln, dict)],
        key=lambda ln: int(ln.get("line_index", 0) or 0),
    )

    for idx, ln in enumerate(lines_sorted):
        li = int(ln.get("line_index", 0) or 0)
        if li not in hj_map:
            continue
        full_word = hj_map[li]
        line_text = _normalize_spaces(ln.get("line_text") or ln.get("text") or "")
        # Chercher le fragment tronqué en fin de ligne (dernier "mot-")
        m = re.search(r'(\S+)-\s*$', line_text)
        if not m:
            continue
        fragment = m.group(1)  # partie avant le tiret
        # Remplacer le fragment- par le mot complet dans cette ligne
        ln["line_text"] = re.sub(re.escape(fragment) + r'-\s*$', full_word, line_text)
        # Supprimer la suite du mot en début de la ligne suivante
        if idx + 1 < len(lines_sorted):
            next_ln = lines_sorted[idx + 1]
            next_text = _normalize_spaces(next_ln.get("line_text") or next_ln.get("text") or "")
            # La suite est le reste du mot (full_word sans fragment)
            suffix = full_word[len(fragment):] if full_word.lower().startswith(fragment.lower()) else ""
            if suffix and next_text.lower().startswith(suffix.lower()):
                next_ln["line_text"] = _normalize_spaces(next_text[len(suffix):])


def _llm_postprocess_blocks(blocks: list) -> None:
    """Post-traitement LLM des blocs à segmentation ambiguë.

    Appelé après _build_semantic_phrases_for_blocks(). Ne fait rien si le modèle
    n'est pas disponible ou si aucun bloc n'atteint le seuil d'ambiguïté.
    """
    try:
        import llm_semantic_corrector as _corrector
    except ImportError:
        return

    threshold = _corrector.AMBIGUITY_THRESHOLD
    scored_ambiguous = [
        (_corrector.score_block_ambiguity(b), b)
        for b in (blocks or [])
    ]
    ambiguous = [
        (score, block)
        for score, block in scored_ambiguous
        if score >= threshold
    ]
    if not ambiguous:
        logging.getLogger(__name__).debug(
            "LLM corrector : 0/%d blocs atteignent le seuil %.2f (limite chars=%s mots=%s)",
            len(blocks or []),
            threshold,
            getattr(_corrector, "_MAX_BLOCK_CHARS", "?"),
            getattr(_corrector, "_MAX_BLOCK_WORDS", "?"),
        )
        return

    max_blocks = max(0, int(getattr(_corrector, "_MAX_BLOCKS_PER_PAGE", 0) or 0))
    if max_blocks:
        ambiguous = sorted(ambiguous, key=lambda item: item[0], reverse=True)[:max_blocks]

    pipe = _corrector.load_pipeline_if_needed()
    if pipe is None:
        return

    strong_model_id = _corrector._configured_strong_model_id()
    strong_pipe = None
    base_model_id = str(pipe.get("model_id") or "")
    use_strong_model = bool(strong_model_id) and str(strong_model_id) != base_model_id

    logging.getLogger(__name__).info(
        "LLM corrector : %d/%d blocs ambigus", len(ambiguous), len(blocks or [])
    )
    for score, block in ambiguous:
        result = _corrector.get_corrections(block, pipe)
        if use_strong_model and _corrector.block_needs_strong_retry(block, score, result=result):
            if strong_pipe is None:
                strong_pipe = _corrector.load_pipeline_if_needed(model_id=strong_model_id)
            if strong_pipe is not None:
                strong_result = _corrector.get_corrections(block, strong_pipe)
                if any(strong_result.get(key) for key in ("c", "sb", "lb", "hj")):
                    result = strong_result
        # Compatibilité : get_corrections renvoie dict {"c":[...], "sb":[...], "hj":[...]} ou list legacy
        if isinstance(result, dict):
            corrections = result.get("c") or []
            sentence_boundaries = result.get("sb") or []
            line_boundaries = result.get("lb") or []
            layout_mode = result.get("lm") if isinstance(result.get("lm"), dict) else None
            hyphen_joins = result.get("hj") or []
        else:
            corrections = result or []
            sentence_boundaries = []
            line_boundaries = []
            layout_mode = None
            hyphen_joins = []
        if not any([corrections, sentence_boundaries, line_boundaries, layout_mode, hyphen_joins]):
            logging.getLogger(__name__).debug(
                "LLM corrector : bloc '%s' → corrections vides (modèle n'a rien proposé)",
                block.get("id") or block.get("unit_id") or "?",
            )
        if layout_mode:
            _apply_llm_layout_mode(block, layout_mode)
        if corrections or sentence_boundaries or line_boundaries:
            heuristic_phrases = copy.deepcopy(block.get("semantic_phrases") or [])
            _apply_llm_corrections(
                block,
                corrections,
                sentence_boundaries=sentence_boundaries,
                line_boundaries=line_boundaries,
            )
            corrected_phrases = block.get("semantic_phrases") or []
            if _llm_semantic_phrases_are_quality_regression(heuristic_phrases, corrected_phrases):
                block["semantic_phrases"] = heuristic_phrases
                block["semantic_phrase_count"] = len(heuristic_phrases)
                block.pop("llm_corrected", None)
                block["llm_correction_rejected"] = "semantic_phrase_quality_regression"
        if hyphen_joins:
            _apply_hyphen_joins(block, hyphen_joins)


def _p1_agent_postprocess_blocks(blocks: list) -> None:
    """Post-traitement des blocs via P1ExtractionAgent (pipeline_agents).

    Alternative provider-agnostique à _llm_postprocess_blocks(). Activé par
    PIPELINE_AGENT_P1_ENABLE=1. Utilise le même jeu de fonctions d'application
    (_apply_llm_corrections, _apply_llm_layout_mode, _apply_hyphen_joins).
    """
    log = logging.getLogger(__name__)
    try:
        from pipeline_agents import get_agent
        from pipeline_agents.p1_extraction import P1ExtractionAgent
    except ImportError:
        log.debug("P1ExtractionAgent indisponible — pipeline_agents non installé")
        return

    threshold = float(os.environ.get("PIPELINE_AGENT_P1_THRESHOLD", "0.25"))
    max_blocks = max(0, int(os.environ.get("PIPELINE_AGENT_P1_MAX_BLOCKS", "5")))

    try:
        agent = get_agent("p1_extraction")
    except Exception as exc:
        log.debug("P1ExtractionAgent: impossible de charger l'agent: %s", exc)
        return

    if not agent.is_available():
        log.debug("P1ExtractionAgent: modèle indisponible, skip")
        return

    scored = [
        (P1ExtractionAgent.score_block(b), b)
        for b in (blocks or [])
    ]
    ambiguous = [
        (score, block) for score, block in scored
        if score >= threshold
    ]
    if not ambiguous:
        log.debug(
            "P1ExtractionAgent: 0/%d blocs atteignent le seuil %.2f",
            len(blocks or []), threshold,
        )
        return
    if max_blocks:
        ambiguous = sorted(ambiguous, key=lambda item: item[0], reverse=True)[:max_blocks]

    log.info("P1ExtractionAgent: %d/%d blocs ambigus", len(ambiguous), len(blocks or []))
    for _score, block in ambiguous:
        lines = block.get("lines") or []
        input_data = {
            "role": str(block.get("role") or "body"),
            "lines": [
                {
                    "i": int(ln.get("line_index", idx) or idx),
                    "t": str(ln.get("line_text") or ln.get("text") or "").strip()[:80],
                    "bb": list(ln.get("bbox") or [0, 0, 0, 0]),
                }
                for idx, ln in enumerate(lines[:12])
                if isinstance(ln, dict)
            ],
        }
        hs = [
            {
                "li": [int(x) for x in (sp.get("line_indices") or [])[:4]],
                "t": str(sp.get("text") or sp.get("texte") or "")[:90],
                "r": str(sp.get("sentence_end_reason") or ""),
            }
            for sp in (block.get("semantic_phrases") or [])[:8]
            if isinstance(sp, dict) and str(sp.get("text") or sp.get("texte") or "").strip()
        ]
        if hs:
            input_data["hs"] = hs

        result = agent.run(input_data)
        if not result:
            continue

        corrections = result.get("c") or []
        sentence_boundaries = result.get("sb") or []
        line_boundaries = result.get("lb") or []
        layout_mode = result.get("lm") if isinstance(result.get("lm"), dict) else None
        hyphen_joins = result.get("hj") or []

        if not any([corrections, sentence_boundaries, line_boundaries, layout_mode, hyphen_joins]):
            log.debug(
                "P1ExtractionAgent: bloc '%s' → corrections vides",
                block.get("id") or block.get("unit_id") or "?",
            )
            continue

        if layout_mode:
            _apply_llm_layout_mode(block, layout_mode)
        if corrections or sentence_boundaries or line_boundaries:
            heuristic_phrases = copy.deepcopy(block.get("semantic_phrases") or [])
            _apply_llm_corrections(
                block,
                corrections,
                sentence_boundaries=sentence_boundaries,
                line_boundaries=line_boundaries,
            )
            corrected_phrases = block.get("semantic_phrases") or []
            if _llm_semantic_phrases_are_quality_regression(heuristic_phrases, corrected_phrases):
                block["semantic_phrases"] = heuristic_phrases
                block["semantic_phrase_count"] = len(heuristic_phrases)
                block.pop("llm_corrected", None)
                block["llm_correction_rejected"] = "semantic_phrase_quality_regression"
        if hyphen_joins:
            _apply_hyphen_joins(block, hyphen_joins)


def _postprocess_blocks_semantic(blocks: list) -> None:
    """Dispatche vers P1ExtractionAgent ou llm_semantic_corrector selon l'environnement.

    Utiliser PIPELINE_AGENT_P1_ENABLE=1 pour activer l'agent open-source.
    Par défaut, utilise llm_semantic_corrector (comportement existant).
    """
    if os.environ.get("PIPELINE_AGENT_P1_ENABLE") == "1":
        _p1_agent_postprocess_blocks(blocks)
    else:
        _llm_postprocess_blocks(blocks)


def _p6_audit_background(
    blocks: list,
    inpaint_regions: list,
    text_removal_debug: dict,
    page_id: str,
    img_width: int = 0,
    img_height: int = 0,
) -> dict:
    """Audit qualité du background master via P6BackgroundAgent.

    Activé par PIPELINE_AGENT_P6_ENABLE=1. Construit l'input depuis les
    métadonnées d'inpainting disponibles et retourne le résultat d'audit
    (quality, artifacts, reprocess, ok).

    Retourne {} si l'agent est désactivé ou indisponible.
    """
    log = logging.getLogger(__name__)
    if os.environ.get("PIPELINE_AGENT_P6_ENABLE") != "1":
        return {}
    try:
        from pipeline_agents import get_agent
        agent = get_agent("p6_background")
    except Exception as exc:
        log.debug("P6BackgroundAgent: chargement échoué: %s", exc)
        return {}
    if not agent.is_available():
        log.debug("P6BackgroundAgent: modèle indisponible, skip")
        return {}

    total_blocks = len(blocks or [])
    n_inpainted = len(inpaint_regions or [])
    coverage = round(n_inpainted / max(1, total_blocks), 3)
    # Proxy de confiance : ratio pixels masqués / surface totale de la page
    mask_nonzero = int((text_removal_debug or {}).get("mask_nonzero") or 0)
    page_area = max(1, int(img_width) * int(img_height))
    avg_conf = round(1.0 - min(0.99, mask_nonzero / page_area), 3)

    input_data = {
        "page_id": str(page_id),
        "blocks_removed": n_inpainted,
        "inpaint_regions": [r for r in (inpaint_regions or []) if isinstance(r, (list, tuple)) and len(r) >= 4],
        "coverage_ratio": coverage,
        "avg_confidence": avg_conf,
    }
    try:
        result = agent.run(input_data)
    except Exception as exc:
        log.debug("P6BackgroundAgent: run() échoué: %s", exc)
        return {}
    if result:
        if not result.get("ok"):
            log.warning(
                "P6BackgroundAgent: fond suspect (qualité=%.2f, %d artéfacts, page=%s)",
                result.get("quality") or 0.0,
                len(result.get("artifacts") or []),
                page_id,
            )
    return result or {}


def _semantic_span_style_signature(span):
    style = span.get("style") if isinstance(span.get("style"), dict) else {}
    flags = style.get("flags") if isinstance(style.get("flags"), dict) else {}
    try:
        size = round(float(style.get("size")), 1) if style.get("size") not in {None, ""} else None
    except Exception:
        size = None
    return (
        str(style.get("font") or ""),
        size,
        str(style.get("color") or ""),
        tuple(sorted(name for name, enabled in flags.items() if enabled)),
    )


def _semantic_span_ends_sentence(text):
    s = _normalize_spaces(text or "")
    if not s:
        return False
    return bool(re.search(r'[.!?\u2026]+[\u201d\u2019"\')\]}\u00bb]*$', s)) and not _is_abbreviation_token(s.split()[-1] if s.split() else s)


def _build_semantic_spans_for_block(block):
    existing_semantic_spans = copy.deepcopy(block.get("semantic_spans") or [])
    flat_spans = []
    for line in sorted(block.get("lines", []) or [], key=lambda ln: int(ln.get("line_index", 0) or 0)):
        line_index = int(line.get("line_index", 0) or 0)
        line_phrases = line.get("phrases", []) or []
        for phrase_index, phrase in enumerate(line_phrases):
            phrase_spans = phrase.get("spans", []) or []
            if not phrase_spans:
                text = _normalize_spaces(phrase.get("texte") or phrase.get("text") or "")
                if text and isinstance(phrase.get("bbox"), (list, tuple)) and len(phrase.get("bbox")) == 4:
                    phrase_spans = [{
                        "texte": text,
                        "text": text,
                        "bbox": phrase.get("bbox"),
                        "style": {},
                        "source_kind": "synthetic_phrase_span",
                        "phrase_unit_id": phrase.get("unit_id"),
                    }]
            for span_index, span in enumerate(phrase_spans):
                text = _normalize_spaces(span.get("texte") or span.get("text") or "")
                bbox = span.get("bbox")
                if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    continue
                flat_spans.append(
                    {
                        "line_index": line_index,
                        "line": line,
                        "phrase_index": phrase_index,
                        "phrase": phrase,
                        "span_index": span_index,
                        "span": span,
                        "text": text,
                        "bbox": list(bbox),
                        "is_first_in_line": phrase_index == 0 and span_index == 0,
                        "is_last_in_line": False,
                    }
                )
        line_entries = [entry for entry in flat_spans if entry["line_index"] == line_index]
        if line_entries:
            line_entries[-1]["is_last_in_line"] = True

    semantic_spans = []
    current_group = []

    def flush():
        nonlocal current_group
        if not current_group:
            return
        text = ""
        bbox = None
        line_indices = []
        fragments = []
        for frag_index, entry in enumerate(current_group):
            text = _append_fragment(text, entry["text"])
            eb = entry["bbox"]
            bbox = eb if bbox is None else [
                min(float(bbox[0]), float(eb[0])),
                min(float(bbox[1]), float(eb[1])),
                max(float(bbox[2]), float(eb[2])),
                max(float(bbox[3]), float(eb[3])),
            ]
            line_indices.append(entry["line_index"])
            fragments.append(
                {
                    "fragment_index": frag_index,
                    "line_index": entry["line_index"],
                    "text": entry["text"],
                    "bbox": eb,
                    "source_span_unit_id": entry["span"].get("unit_id"),
                    "source_phrase_unit_id": entry["phrase"].get("unit_id"),
                }
            )
        first_span = current_group[0]["span"]
        semantic_spans.append(
            {
                "unit_id": f"{block.get('unit_id') or block.get('id') or 'block'}:semantic_span:{len(semantic_spans)}",
                "texte": _normalize_spaces(text),
                "text": _normalize_spaces(text),
                "bbox": [int(round(v)) for v in (bbox or [0, 0, 0, 0])],
                "line_indices": sorted(set(int(v) for v in line_indices)),
                "start_line_index": min(line_indices),
                "end_line_index": max(line_indices),
                "multi_line": len(set(line_indices)) > 1,
                "fragment_count": len(fragments),
                "fragments": fragments,
                "source": first_span.get("source") or block.get("source") or "ocr",
                "source_kind": "semantic_span",
                "style": copy.deepcopy(first_span.get("style", {})) if isinstance(first_span.get("style"), dict) else {},
                "translatable": bool(first_span.get("translatable", True)),
                "translation_strategy": first_span.get("translation_strategy"),
                "coverage_required": first_span.get("coverage_required"),
                "render_policy": first_span.get("render_policy"),
                "unit_type": first_span.get("unit_type") or "",
                "expression_semantics": copy.deepcopy(first_span.get("expression_semantics", {})),
            }
        )
        current_group = []

    def should_merge(previous_entry, current_entry):
        if previous_entry is None:
            return False
        previous_span = previous_entry["span"]
        current_span = current_entry["span"]
        if current_entry["line_index"] == previous_entry["line_index"]:
            return False
        if current_entry["line_index"] != previous_entry["line_index"] + 1:
            return False
        if not previous_entry["is_last_in_line"] or not current_entry["is_first_in_line"]:
            return False
        if _semantic_span_style_signature(previous_span) != _semantic_span_style_signature(current_span):
            return False
        prev_class = ((previous_span.get("expression_semantics") or {}).get("inline_class") or "")
        cur_class = ((current_span.get("expression_semantics") or {}).get("inline_class") or "")
        if prev_class != cur_class:
            return False
        if prev_class in {"reference", "formula"}:
            return False
        next_line_relation = (((previous_entry["line"].get("editorial_relations") or {}).get("with_next")) or {})
        if next_line_relation.get("exists") and not next_line_relation.get("continuation"):
            return False
        if bool(current_entry["line"].get("hard_break_before", False)) and not next_line_relation.get("continuation"):
            return False
        if _semantic_span_ends_sentence(previous_entry["text"]):
            return False
        return True

    previous_entry = None
    for entry in flat_spans:
        if current_group and not should_merge(previous_entry, entry):
            flush()
        current_group.append(entry)
        previous_entry = entry
    flush()

    if not semantic_spans and existing_semantic_spans:
        semantic_spans = existing_semantic_spans

    for index, semantic_span in enumerate(semantic_spans):
        previous_span = semantic_spans[index - 1] if index > 0 else None
        next_span = semantic_spans[index + 1] if index + 1 < len(semantic_spans) else None
        semantic_span["structural_context"] = {
            "level": "semantic_span",
            "unit_id": semantic_span.get("unit_id"),
            "parent_unit_id": block.get("unit_id"),
            "block_unit_id": block.get("unit_id"),
            "line_unit_id": None,
            "child_fragment_count": len(semantic_span.get("fragments", []) or []),
            "line_indices": list(semantic_span.get("line_indices", []) or []),
        }
        semantic_span["expression_relations"] = {
            "with_previous": {
                "exists": bool(previous_span),
                "neighbor_id": previous_span.get("unit_id") if previous_span else None,
                "neighbor_text": _normalize_spaces(previous_span.get("text") or "") if previous_span else "",
                "relation": "continuation" if previous_span and previous_span.get("expression_semantics", {}).get("inline_class") == semantic_span.get("expression_semantics", {}).get("inline_class") else "semantic_shift",
                "continuation": bool(previous_span and previous_span.get("expression_semantics", {}).get("inline_class") == semantic_span.get("expression_semantics", {}).get("inline_class")),
                "same_inline_class": bool(previous_span and previous_span.get("expression_semantics", {}).get("inline_class") == semantic_span.get("expression_semantics", {}).get("inline_class")),
                "same_style": bool(previous_span and _semantic_span_style_signature(previous_span) == _semantic_span_style_signature(semantic_span)),
            } if previous_span else {"exists": False},
            "with_next": {
                "exists": bool(next_span),
                "neighbor_id": next_span.get("unit_id") if next_span else None,
                "neighbor_text": _normalize_spaces(next_span.get("text") or "") if next_span else "",
                "relation": "continuation" if next_span and next_span.get("expression_semantics", {}).get("inline_class") == semantic_span.get("expression_semantics", {}).get("inline_class") else "semantic_shift",
                "continuation": bool(next_span and next_span.get("expression_semantics", {}).get("inline_class") == semantic_span.get("expression_semantics", {}).get("inline_class")),
                "same_inline_class": bool(next_span and next_span.get("expression_semantics", {}).get("inline_class") == semantic_span.get("expression_semantics", {}).get("inline_class")),
                "same_style": bool(next_span and _semantic_span_style_signature(next_span) == _semantic_span_style_signature(semantic_span)),
            } if next_span else {"exists": False},
        }
    block["semantic_spans"] = semantic_spans
    block["semantic_span_count"] = len(semantic_spans)


def _build_semantic_spans_for_blocks(blocks):
    for block in blocks or []:
        _build_semantic_spans_for_block(block)


def _semantic_run_text(unit):
    return _normalize_spaces(unit.get("text") or unit.get("texte") or "")


def _semantic_run_inline_class(unit):
    sem = unit.get("expression_semantics") if isinstance(unit.get("expression_semantics"), dict) else {}
    return _normalize_spaces(sem.get("inline_class") or "")


def _semantic_run_style_signature(unit):
    style = unit.get("style") if isinstance(unit.get("style"), dict) else {}
    flags = style.get("flags") if isinstance(style.get("flags"), dict) else {}
    try:
        size = round(float(style.get("size")), 1) if style.get("size") not in {None, ""} else None
    except Exception:
        size = None
    return (
        str(style.get("font") or ""),
        size,
        str(style.get("color") or ""),
        tuple(sorted(name for name, enabled in flags.items() if enabled)),
    )


def _semantic_run_line_indices(unit):
    if isinstance(unit.get("line_indices"), list) and unit.get("line_indices"):
        return [int(v) for v in unit.get("line_indices", []) if isinstance(v, (int, float))]
    structural = unit.get("structural_context") if isinstance(unit.get("structural_context"), dict) else {}
    if isinstance(structural.get("line_indices"), list) and structural.get("line_indices"):
        return [int(v) for v in structural.get("line_indices", []) if isinstance(v, (int, float))]
    line_index = structural.get("line_index", unit.get("line_index"))
    if isinstance(line_index, (int, float)):
        return [int(line_index)]
    return []


def _build_semantic_runs_from_units(units, parent_unit_id):
    candidates = [unit for unit in (units or []) if _semantic_run_text(unit)]
    if not candidates:
        return []
    runs = []
    current_group = []

    def merge_allowed(previous_unit, current_unit):
        if not previous_unit:
            return False
        prev_class = _semantic_run_inline_class(previous_unit)
        cur_class = _semantic_run_inline_class(current_unit)
        prev_text = _semantic_run_text(previous_unit)
        cur_text = _semantic_run_text(current_unit)
        if not prev_text or not cur_text:
            return False
        protected = {"code", "reference", "formula"}
        if prev_class in protected or cur_class in protected:
            return prev_class == cur_class
        if _semantic_span_ends_sentence(prev_text):
            return False
        if prev_class == cur_class:
            return True
        broad = {"plain_text", "technical_inline", "label"}
        return prev_class in broad and cur_class in broad

    def flush():
        nonlocal current_group
        if not current_group:
            return
        text = ""
        bbox = None
        line_indices = []
        style_signatures = []
        run_inline_classes = []
        protected_inline = False
        fragments = []
        for fragment_index, unit in enumerate(current_group):
            text = _append_fragment(text, _semantic_run_text(unit))
            ub = unit.get("bbox") or [0, 0, 0, 0]
            bbox = ub if bbox is None else [
                min(float(bbox[0]), float(ub[0])),
                min(float(bbox[1]), float(ub[1])),
                max(float(bbox[2]), float(ub[2])),
                max(float(bbox[3]), float(ub[3])),
            ]
            line_indices.extend(_semantic_run_line_indices(unit))
            style_signatures.append(_semantic_run_style_signature(unit))
            inline_class = _semantic_run_inline_class(unit)
            if inline_class:
                run_inline_classes.append(inline_class)
            sem = unit.get("expression_semantics") if isinstance(unit.get("expression_semantics"), dict) else {}
            protected_inline = protected_inline or bool(sem.get("protected_inline"))
            fragments.append(
                {
                    "fragment_index": fragment_index,
                    "source_unit_id": unit.get("unit_id"),
                    "text": _semantic_run_text(unit),
                    "bbox": unit.get("bbox"),
                    "line_indices": _semantic_run_line_indices(unit),
                }
            )
        dominant_class = run_inline_classes[0] if run_inline_classes else "plain_text"
        if len(set(run_inline_classes)) > 1 and all(cls in {"plain_text", "technical_inline", "label"} for cls in run_inline_classes):
            dominant_class = "technical_inline" if "technical_inline" in run_inline_classes else "plain_text"
        run = {
            "unit_id": f"{parent_unit_id}:semantic_run:{len(runs)}",
            "text": _normalize_spaces(text),
            "texte": _normalize_spaces(text),
            "bbox": [int(round(v)) for v in (bbox or [0, 0, 0, 0])],
            "line_indices": sorted(set(int(v) for v in line_indices)),
            "multi_line": len(set(line_indices)) > 1,
            "fragment_count": len(fragments),
            "fragments": fragments,
            "mixed_style": len(set(style_signatures)) > 1,
            "expression_semantics": {
                "inline_class": dominant_class,
                "protected_inline": bool(protected_inline),
                "mixed_style": len(set(style_signatures)) > 1,
            },
            "source_kind": "semantic_run",
        }
        for unit in current_group:
            unit["parent_semantic_run_id"] = run["unit_id"]
        runs.append(run)
        current_group = []

    previous_unit = None
    for unit in candidates:
        if current_group and not merge_allowed(previous_unit, unit):
            flush()
        current_group.append(unit)
        previous_unit = unit
    flush()
    return runs


def _build_semantic_runs_for_block(block):
    block_runs = []
    for semantic_phrase in block.get("semantic_phrases", []) or []:
        phrase_units = [copy.deepcopy(sp) for sp in (semantic_phrase.get("spans", []) or []) if _semantic_run_text(sp)]
        if not phrase_units:
            phrase_units = [{
                "unit_id": f"{semantic_phrase.get('unit_id')}:synthetic-run-span:0",
                "text": semantic_phrase.get("text") or semantic_phrase.get("texte") or "",
                "texte": semantic_phrase.get("text") or semantic_phrase.get("texte") or "",
                "bbox": semantic_phrase.get("bbox"),
                "line_indices": list(semantic_phrase.get("line_indices", []) or []),
                "expression_semantics": {"inline_class": "plain_text", "protected_inline": False},
                "style": {},
            }]
        semantic_runs = _build_semantic_runs_from_units(phrase_units, semantic_phrase.get("unit_id") or block.get("unit_id") or block.get("id") or "phrase")
        semantic_phrase["semantic_runs"] = semantic_runs
        semantic_phrase["semantic_run_count"] = len(semantic_runs)
        block_runs.extend(copy.deepcopy(semantic_runs))
    block["semantic_runs"] = block_runs
    block["semantic_run_count"] = len(block_runs)


def _build_semantic_runs_for_blocks(blocks):
    for block in blocks or []:
        _build_semantic_runs_for_block(block)


def _semantic_group_text(unit):
    return _normalize_spaces(unit.get("text") or unit.get("texte") or "")


def _semantic_group_inline_class(unit):
    sem = unit.get("expression_semantics") if isinstance(unit.get("expression_semantics"), dict) else {}
    return _normalize_spaces(sem.get("inline_class") or "")


def _semantic_group_line_indices(unit):
    if isinstance(unit.get("line_indices"), list) and unit.get("line_indices"):
        return [int(v) for v in unit.get("line_indices", []) if isinstance(v, (int, float))]
    return []


def _classify_semantic_group(units):
    texts = [_semantic_group_text(unit) for unit in (units or []) if _semantic_group_text(unit)]
    classes = [_semantic_group_inline_class(unit) for unit in (units or []) if _semantic_group_inline_class(unit)]
    joined = _normalize_spaces(" ".join(texts))
    if texts and texts[0].endswith(":") and len(texts) >= 2:
        return "label_value"
    if "technical_inline" in classes:
        return "technical_group"
    if set(classes) <= {"code"} and classes:
        return "code_group"
    if set(classes) <= {"reference"} and classes:
        return "reference_group"
    if set(classes) <= {"formula"} and classes:
        return "formula_group"
    if re.search(r"\b(v|ver\.?|version)\s*\d+(\.\d+)*\b", joined, flags=re.IGNORECASE):
        return "name_version"
    return "editorial_group"


def _build_semantic_groups_from_runs(runs, parent_unit_id):
    candidates = [run for run in (runs or []) if _semantic_group_text(run)]
    if not candidates:
        return []
    groups = []
    current_group = []

    def merge_allowed(previous_run, current_run):
        if not previous_run:
            return False
        prev_text = _semantic_group_text(previous_run)
        cur_text = _semantic_group_text(current_run)
        if not prev_text or not cur_text:
            return False
        prev_class = _semantic_group_inline_class(previous_run)
        cur_class = _semantic_group_inline_class(current_run)
        prev_lines = _semantic_group_line_indices(previous_run)
        cur_lines = _semantic_group_line_indices(current_run)
        if prev_lines and cur_lines and min(cur_lines) - max(prev_lines) > 1:
            return False
        if _semantic_span_ends_sentence(prev_text):
            return False
        if prev_text.endswith(":"):
            return True
        protected = {"code", "reference", "formula"}
        if prev_class in protected or cur_class in protected:
            return prev_class == cur_class
        broad = {"plain_text", "technical_inline", "label"}
        if prev_class in broad and cur_class in broad:
            return True
        return prev_class == cur_class

    def flush():
        nonlocal current_group
        if not current_group:
            return
        text = ""
        bbox = None
        line_indices = []
        fragments = []
        for fragment_index, run in enumerate(current_group):
            text = _append_fragment(text, _semantic_group_text(run))
            rb = run.get("bbox") or [0, 0, 0, 0]
            bbox = rb if bbox is None else [
                min(float(bbox[0]), float(rb[0])),
                min(float(bbox[1]), float(rb[1])),
                max(float(bbox[2]), float(rb[2])),
                max(float(bbox[3]), float(rb[3])),
            ]
            line_indices.extend(_semantic_group_line_indices(run))
            fragments.append(
                {
                    "fragment_index": fragment_index,
                    "source_run_id": run.get("unit_id"),
                    "text": _semantic_group_text(run),
                    "bbox": run.get("bbox"),
                    "line_indices": _semantic_group_line_indices(run),
                }
            )
        group = {
            "unit_id": f"{parent_unit_id}:semantic_group:{len(groups)}",
            "text": _normalize_spaces(text),
            "texte": _normalize_spaces(text),
            "bbox": [int(round(v)) for v in (bbox or [0, 0, 0, 0])],
            "line_indices": sorted(set(int(v) for v in line_indices)),
            "multi_line": len(set(line_indices)) > 1,
            "fragment_count": len(fragments),
            "fragments": fragments,
            "group_class": _classify_semantic_group(current_group),
            "source_kind": "semantic_group",
        }
        for run in current_group:
            run["parent_semantic_group_id"] = group["unit_id"]
        groups.append(group)
        current_group = []

    previous_run = None
    for run in candidates:
        if current_group and not merge_allowed(previous_run, run):
            flush()
        current_group.append(run)
        previous_run = run
    flush()
    return groups


def _build_semantic_groups_for_block(block):
    block_groups = []
    for semantic_phrase in block.get("semantic_phrases", []) or []:
        phrase_runs = [copy.deepcopy(run) for run in (semantic_phrase.get("semantic_runs", []) or []) if _semantic_group_text(run)]
        semantic_groups = _build_semantic_groups_from_runs(
            phrase_runs,
            semantic_phrase.get("unit_id") or block.get("unit_id") or block.get("id") or "phrase",
        )
        semantic_phrase["semantic_groups"] = semantic_groups
        semantic_phrase["semantic_group_count"] = len(semantic_groups)
        for group in semantic_groups:
            group["structural_context"] = {
                "level": "semantic_group",
                "unit_id": group.get("unit_id"),
                "parent_unit_id": semantic_phrase.get("unit_id"),
                "block_unit_id": block.get("unit_id"),
                "line_unit_id": None,
                "child_fragment_count": len(group.get("fragments", []) or []),
                "line_indices": list(group.get("line_indices", []) or []),
            }
        block_groups.extend(copy.deepcopy(semantic_groups))
    block["semantic_groups"] = block_groups
    block["semantic_group_count"] = len(block_groups)


def _build_semantic_groups_for_blocks(blocks):
    for block in blocks or []:
        _build_semantic_groups_for_block(block)


def _bbox_overlap_ratio(b1, b2):
    r1 = fitz.Rect(b1)
    r2 = fitz.Rect(b2)
    inter = (r1 & r2).get_area()
    if inter <= 0:
        return 0.0
    return inter / max(1e-9, min(r1.get_area(), r2.get_area()))


def _erase_uncovered_pdf_words(clean_bgr, pdf_page, already_blanked_regions, sx, sy):
    """Efface dans clean_bgr les mots PDF qui n'ont pas été couverts par l'extraction.

    Cela permet de nettoyer les en-têtes courants, pieds de page, grands titres
    décoratifs ou tout texte qui était dans le PDF mais pas dans final_blocks.
    On applique un whiteout (255, 255, 255) avec une légère marge.
    """
    import numpy as _np
    try:
        words = pdf_page.get_text("words")  # [(x0, y0, x1, y1, text, block_no, line_no, word_no), ...]
    except Exception:
        return clean_bgr

    if not words:
        return clean_bgr

    # Construire un masque des régions déjà effacées (en coordonnées pixels)
    h, w = clean_bgr.shape[:2]
    already_covered = _np.zeros((h, w), dtype=_np.uint8)
    for r in already_blanked_regions or []:
        if not isinstance(r, (list, tuple)) or len(r) != 4:
            continue
        x0, y0, x1, y1 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 > x0 and y1 > y0:
            already_covered[y0:y1, x0:x1] = 1

    result = clean_bgr.copy()
    margin = 2  # pixels de marge autour de chaque mot
    for word in words:
        wx0, wy0, wx1, wy1 = word[0], word[1], word[2], word[3]
        # Convertir coordonnées points → pixels
        px0 = int(wx0 * sx) - margin
        py0 = int(wy0 * sy) - margin
        px1 = int(wx1 * sx) + margin
        py1 = int(wy1 * sy) + margin
        px0, py0 = max(0, px0), max(0, py0)
        px1, py1 = min(w, px1), min(h, py1)
        if px1 <= px0 or py1 <= py0:
            continue
        # Vérifier si cette région est déjà couverte par l'extraction
        region_already = already_covered[py0:py1, px0:px1]
        coverage = float(region_already.sum()) / max(1, region_already.size)
        if coverage >= 0.25:
            continue  # Déjà nettoyé par le passage principal
        # Appliquer le whiteout
        result[py0:py1, px0:px1] = 255

    return result


def _collect_text_regions_for_inpainting(blocks, non_text_zones, immutable_overlays=None):
    regions = []
    protected = []
    for z in non_text_zones or []:
        if isinstance(z, (list, tuple)) and len(z) == 4:
            zr = fitz.Rect([float(v) for v in z])
            if zr.get_area() > 0:
                protected.append(zr)
    for ov in immutable_overlays or []:
        bb = ov.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            rr = fitz.Rect([float(v) for v in bb])
            if rr.get_area() > 0:
                protected.append(rr)
    # Preserve chart/figure internals (labels, ticks, symbols) as image.
    for b in blocks or []:
        if (b.get("role") or "").lower() == "diagram_label":
            bb = b.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                rr = fitz.Rect([float(v) for v in bb])
                if rr.get_area() > 0:
                    protected.append(rr)

    def is_protected(bb):
        rr = fitz.Rect([float(v) for v in bb])
        area = rr.get_area()
        if area <= 0:
            return True
        for pr in protected:
            ov = (rr & pr).get_area() / area
            # Keep figures/graphics intact in background master.
            if ov >= 0.25:
                return True
        return False

    for b in blocks:
        if (b.get("role") or "").lower() == "diagram_label":
            continue
        for line in b.get("lines", []):
            for phrase in line.get("phrases", []):
                pb = phrase.get("bbox") or line.get("bbox") or b.get("bbox")
                if not isinstance(pb, (list, tuple)) or len(pb) != 4:
                    continue
                if phrase.get("render_mode") == "background_only":
                    continue
                bb = [int(float(v)) for v in pb]
                if bb[2] <= bb[0] or bb[3] <= bb[1]:
                    continue
                if is_protected(bb):
                    continue
                regions.append(bb)
    if regions:
        return regions

    # Fallback only if no phrase-level boxes available.
    for b in blocks:
        bb = b.get("bbox", [0, 0, 0, 0])
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        bb = [int(float(v)) for v in bb]
        if bb[2] <= bb[0] or bb[3] <= bb[1]:
            continue
        if is_protected(bb):
            continue
        regions.append(bb)
    return regions


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


def _bbox_width(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    try:
        return max(0.0, float(bbox[2]) - float(bbox[0]))
    except Exception:
        return 0.0


def _bbox_height(bbox):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return 0.0
    try:
        return max(0.0, float(bbox[3]) - float(bbox[1]))
    except Exception:
        return 0.0


def _bbox_x_overlap_ratio(a, b):
    if not (isinstance(a, (list, tuple)) and len(a) == 4 and isinstance(b, (list, tuple)) and len(b) == 4):
        return 0.0
    try:
        ix0 = max(float(a[0]), float(b[0]))
        ix1 = min(float(a[2]), float(b[2]))
        if ix1 <= ix0:
            return 0.0
        overlap = ix1 - ix0
        base = max(1.0, min(float(a[2]) - float(a[0]), float(b[2]) - float(b[0])))
        return overlap / base
    except Exception:
        return 0.0


def _recompute_ocr_block_geometry(block):
    if not isinstance(block, dict):
        return block
    lines = [ln for ln in (block.get("lines") or []) if isinstance(ln, dict)]
    if not lines:
        return block
    bboxes = [ln.get("bbox") for ln in lines if isinstance(ln.get("bbox"), (list, tuple)) and len(ln.get("bbox")) == 4]
    if bboxes:
        block["bbox"] = [
            int(min(float(bb[0]) for bb in bboxes)),
            int(min(float(bb[1]) for bb in bboxes)),
            int(max(float(bb[2]) for bb in bboxes)),
            int(max(float(bb[3]) for bb in bboxes)),
        ]
    block["text"] = " ".join(
        str((line or {}).get("text") or (line or {}).get("line_text") or "").strip()
        for line in lines
        if str((line or {}).get("text") or (line or {}).get("line_text") or "").strip()
    ).strip()
    heights = [_bbox_height(line.get("bbox")) for line in lines if _bbox_height(line.get("bbox")) > 0.0]
    peaks = [
        float(line.get("peak_y"))
        for line in lines
        if isinstance(line.get("peak_y"), (int, float))
    ]
    if heights:
        block["avg_line_height"] = float(np.median(heights))
    if len(lines) >= 2:
        gaps = []
        peak_diffs = []
        for idx in range(1, len(lines)):
            prev_bbox = lines[idx - 1].get("bbox")
            cur_bbox = lines[idx].get("bbox")
            if isinstance(prev_bbox, (list, tuple)) and len(prev_bbox) == 4 and isinstance(cur_bbox, (list, tuple)) and len(cur_bbox) == 4:
                gaps.append(float(cur_bbox[1]) - float(prev_bbox[3]))
            if idx - 1 < len(peaks) and idx < len(peaks):
                peak_diffs.append(peaks[idx] - peaks[idx - 1])
        block["line_spacing"] = float(np.median(gaps)) if gaps else 0.0
        block["peak_to_peak_spacing"] = float(np.median(peak_diffs)) if peak_diffs else 0.0
    else:
        block["line_spacing"] = 0.0
        block["peak_to_peak_spacing"] = 0.0
    confs = [
        float(line.get("ocr_confidence_mean", 0.0) or 0.0)
        for line in lines
        if float(line.get("ocr_confidence_mean", 0.0) or 0.0) > 0.0
    ]
    if confs:
        block["ocr_confidence_mean"] = float(np.mean(confs))
        block["ocr_confidence_min"] = float(np.min(confs))
    block["ocr_line_count"] = len(lines)
    block["ocr_word_count"] = int(sum(int((line or {}).get("ocr_word_count", 0) or 0) for line in lines))
    return block


def _prune_weak_ocr_lines(ocr_blocks):
    cleaned_blocks = []
    for block in ocr_blocks or []:
        if not isinstance(block, dict):
            continue
        lines = [copy.deepcopy(ln) for ln in (block.get("lines") or []) if isinstance(ln, dict)]
        if len(lines) < 2:
            cleaned_blocks.append(block)
            continue
        best_conf = max(float((ln or {}).get("ocr_confidence_mean", 0.0) or 0.0) for ln in lines)
        kept_lines = []
        for idx, line in enumerate(lines):
            text = _normalize_spaces(line.get("text") or line.get("line_text") or "")
            conf = float(line.get("ocr_confidence_mean", 0.0) or 0.0)
            token_count = len(re.findall(r"[A-Za-zÀ-ÿ0-9]+", text))
            char_count = len(re.sub(r"\s+", "", text))
            line_bbox = line.get("bbox")
            prev_bbox = lines[idx - 1].get("bbox") if idx > 0 else None
            dangling_short_tail = bool(
                idx > 0
                and token_count <= 1
                and char_count <= 6
                and conf < max(0.78, best_conf - 0.10)
                and _bbox_width(line_bbox) <= max(54.0, _bbox_height(line_bbox) * 3.6)
                and _bbox_x_overlap_ratio(prev_bbox, line_bbox) >= 0.30
            )
            if dangling_short_tail:
                continue
            kept_lines.append(line)
        if not kept_lines:
            continue
        new_block = copy.deepcopy(block)
        new_block["lines"] = kept_lines
        cleaned_blocks.append(_recompute_ocr_block_geometry(new_block))
    return cleaned_blocks


def _infer_alignment(bbox, content_bbox, page_w):
    return _infer_alignment_with_context(bbox, content_bbox, page_w=page_w)


def _clean_bbox(bb):
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bb]
    except Exception:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _text_case_profile(text):
    s = _normalize_spaces(text or "")
    alpha = [ch for ch in s if ch.isalpha()]
    if not alpha:
        return "empty"
    if s.isupper() and len(alpha) >= 2:
        return "uppercase"
    if s.islower() and len(alpha) >= 2:
        return "lowercase"
    title_words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", s)
    if title_words and sum(1 for w in title_words if w[:1].isupper()) >= max(1, int(len(title_words) * 0.75)):
        return "title_like"
    return "mixed"


def _text_statistics(text):
    s = _normalize_spaces(text or "")
    chars = len(s)
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", s)
    digits = re.findall(r"\d", s)
    punct = re.findall(r"[^\w\s]", s, flags=re.UNICODE)
    alpha = [ch for ch in s if ch.isalpha()]
    uppercase = [ch for ch in alpha if ch.isupper()]
    return {
        "char_count": chars,
        "word_count": len(words),
        "digit_count": len(digits),
        "punctuation_count": len(punct),
        "uppercase_ratio": round(len(uppercase) / max(1, len(alpha)), 4),
        "digit_ratio": round(len(digits) / max(1, chars), 4),
        "punctuation_ratio": round(len(punct) / max(1, chars), 4),
        "case_profile": _text_case_profile(s),
    }


def _bbox_relative_attributes(bbox, container_bbox):
    bb = _clean_bbox(bbox)
    cb = _clean_bbox(container_bbox)
    if not bb or not cb:
        return {}
    x0, y0, x1, y1 = bb
    c0, t0, c1, t1 = cb
    cw = max(1.0, c1 - c0)
    ch = max(1.0, t1 - t0)
    left_gap = max(0.0, x0 - c0)
    right_gap = max(0.0, c1 - x1)
    top_gap = max(0.0, y0 - t0)
    bottom_gap = max(0.0, t1 - y1)
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    center_x = ((x0 + x1) / 2.0 - c0) / cw
    center_y = ((y0 + y1) / 2.0 - t0) / ch
    fill_ratio_x = bw / cw
    fill_ratio_y = bh / ch
    near_x = max(8.0, cw * 0.04)
    near_y = max(6.0, ch * 0.08)
    center_tol_x = max(10.0, cw * 0.08)
    center_tol_y = max(8.0, ch * 0.12)
    if fill_ratio_x >= 0.9 and left_gap <= near_x and right_gap <= near_x:
        horizontal_anchor = "stretch"
    elif abs(left_gap - right_gap) <= center_tol_x:
        horizontal_anchor = "center"
    elif left_gap <= near_x:
        horizontal_anchor = "left"
    elif right_gap <= near_x:
        horizontal_anchor = "right"
    else:
        horizontal_anchor = "free"
    if fill_ratio_y >= 0.9 and top_gap <= near_y and bottom_gap <= near_y:
        vertical_anchor = "stretch"
    elif abs(top_gap - bottom_gap) <= center_tol_y:
        vertical_anchor = "middle"
    elif top_gap <= near_y:
        vertical_anchor = "top"
    elif bottom_gap <= near_y:
        vertical_anchor = "bottom"
    else:
        vertical_anchor = "free"
    return {
        "container_bbox": [round(v, 2) for v in cb],
        "left_gap_px": round(left_gap, 2),
        "right_gap_px": round(right_gap, 2),
        "top_gap_px": round(top_gap, 2),
        "bottom_gap_px": round(bottom_gap, 2),
        "width_px": round(bw, 2),
        "height_px": round(bh, 2),
        "width_ratio": round(fill_ratio_x, 4),
        "height_ratio": round(fill_ratio_y, 4),
        "center_x_ratio": round(center_x, 4),
        "center_y_ratio": round(center_y, 4),
        "horizontal_anchor": horizontal_anchor,
        "vertical_anchor": vertical_anchor,
    }


def _infer_alignment_with_context(bbox, content_bbox, page_w=None, text="", role="body"):
    bb = _clean_bbox(bbox)
    cb = _clean_bbox(content_bbox)
    if not bb or not cb:
        return "left", 0.0
    x0, _, x1, _ = bb
    c0, _, c1, _ = cb
    content_w = max(1.0, c1 - c0)
    block_w = max(1.0, x1 - x0)
    left_gap = max(0.0, x0 - c0)
    right_gap = max(0.0, c1 - x1)
    near_tol = max(8.0, content_w * 0.03)
    center_tol = max(10.0, content_w * 0.06)
    stats = _text_statistics(text)
    word_count = int(stats.get("word_count") or 0)
    char_count = int(stats.get("char_count") or 0)
    long_prose = bool(role == "body" and (word_count >= 8 or char_count >= 48))
    short_label = bool(word_count <= 6 and char_count <= 42)

    if block_w >= content_w * 0.92 and left_gap <= near_tol and right_gap <= near_tol:
        return ("justify" if long_prose else "center"), left_gap
    if abs(left_gap - right_gap) <= center_tol and block_w < content_w * 0.86:
        return "center", left_gap
    if short_label and right_gap <= near_tol and left_gap > near_tol:
        return "right", left_gap
    return "left", left_gap


def _collect_style_entries_from_phrase(phrase):
    styles = []
    for span in (phrase or {}).get("spans", []) or []:
        style = span.get("style") if isinstance(span, dict) else None
        if isinstance(style, dict) and style:
            styles.append(style)
    style = (phrase or {}).get("style")
    if isinstance(style, dict) and style:
        styles.append(style)
    return styles


def _aggregate_style_characteristics(style_entries, text=""):
    entries = [st for st in (style_entries or []) if isinstance(st, dict)]
    sizes = []
    fonts = []
    colors = []
    flag_true_counts = {"bold": 0, "italic": 0, "underline": 0, "highlight": 0, "uppercase": 0, "monospace": 0, "serif": 0}
    for st in entries:
        if st.get("font"):
            fonts.append(str(st.get("font")))
        if st.get("color"):
            colors.append(str(st.get("color")))
        try:
            if st.get("size") not in {None, ""}:
                sizes.append(float(st.get("size")))
        except Exception:
            pass
        flags = st.get("flags") if isinstance(st.get("flags"), dict) else {}
        for key in flag_true_counts:
            if bool(flags.get(key)):
                flag_true_counts[key] += 1
    primary_font = max(fonts, key=fonts.count) if fonts else ""
    primary_color = max(colors, key=colors.count) if colors else ""
    size_median = float(np.median(sizes)) if sizes else 0.0
    size_min = min(sizes) if sizes else 0.0
    size_max = max(sizes) if sizes else 0.0
    total_entries = max(1, len(entries))
    text_stats = _text_statistics(text)
    if not any(flag_true_counts.values()) and text_stats.get("case_profile") == "uppercase":
        flag_true_counts["uppercase"] = total_entries
    return {
        "font_family_primary": primary_font,
        "font_size_pt_median": round(size_median, 2),
        "font_size_pt_min": round(size_min, 2),
        "font_size_pt_max": round(size_max, 2),
        "color_primary": primary_color,
        "flags_any": {k: bool(v) for k, v in flag_true_counts.items()},
        "flags_ratio": {k: round(v / total_entries, 4) for k, v in flag_true_counts.items()},
    }


def _infer_block_alignment_from_lines(lines, container_bbox, fallback_alignment="left", role="body"):
    valid = []
    for line in lines or []:
        bb = _clean_bbox((line or {}).get("bbox"))
        if not bb:
            continue
        txt = _normalize_spaces((line or {}).get("line_text") or (line or {}).get("text") or "")
        if not txt:
            continue
        valid.append((bb, txt))
    if len(valid) < 2:
        return fallback_alignment
    cb = _clean_bbox(container_bbox)
    if not cb:
        return fallback_alignment
    cw = max(1.0, cb[2] - cb[0])
    lefts = [bb[0] for bb, _ in valid]
    rights = [bb[2] for bb, _ in valid]
    widths = [max(1.0, bb[2] - bb[0]) for bb, _ in valid]
    left_spread = max(lefts) - min(lefts)
    right_spread = max(rights) - min(rights)
    tol = max(10.0, cw * 0.05)
    width_ratio_median = float(np.median(widths)) / cw
    if left_spread <= tol and right_spread > tol * 1.2:
        return "left"
    if right_spread <= tol and left_spread > tol * 1.2:
        return "right"
    if left_spread <= tol and right_spread <= tol and width_ratio_median >= 0.88 and role == "body":
        return "justify"
    centers = [((bb[0] + bb[2]) / 2.0 - cb[0]) / cw for bb, _ in valid]
    center_spread = max(centers) - min(centers)
    if center_spread <= 0.08 and width_ratio_median < 0.75:
        return "center"
    return fallback_alignment


def _attach_textual_characteristics(final_blocks, content_bbox):
    for block in final_blocks or []:
        block_lines = block.get("lines", []) or []
        semantic_phrases = block.get("semantic_phrases", []) or []
        semantic_spans = block.get("semantic_spans", []) or []
        semantic_runs = block.get("semantic_runs", []) or []
        semantic_groups = block.get("semantic_groups", []) or []
        line_texts = [_normalize_spaces(line.get("line_text") or line.get("text") or "") for line in block_lines]
        block_text = _normalize_spaces(" ".join(t for t in line_texts if t))
        block_style_entries = []
        for line in block_lines:
            for phrase in line.get("phrases", []) or []:
                block_style_entries.extend(_collect_style_entries_from_phrase(phrase))
        block["text_attributes"] = _text_statistics(block_text)
        block["style_attributes"] = _aggregate_style_characteristics(block_style_entries, block_text)
        block["layout_attributes"] = _bbox_relative_attributes(block.get("bbox"), content_bbox)
        block["layout_attributes"]["horizontal_alignment"] = str(block.get("alignment") or "left")
        block["layout_attributes"]["indent_px"] = float(block.get("indent_px", 0.0) or 0.0)
        block["layout_attributes"]["line_count"] = len(block_lines)
        block["layout_attributes"]["phrase_count"] = sum(len((line.get("phrases") or [])) for line in block_lines)
        block["layout_attributes"]["source_layout_mode"] = dict(block.get("source_layout_mode") or {})
        for line in block_lines:
            line_text = _normalize_spaces(line.get("line_text") or line.get("text") or "")
            line_style_entries = []
            for phrase in line.get("phrases", []) or []:
                line_style_entries.extend(_collect_style_entries_from_phrase(phrase))
            line["text_attributes"] = _text_statistics(line_text)
            line["style_attributes"] = _aggregate_style_characteristics(line_style_entries, line_text)
            line["layout_attributes"] = _bbox_relative_attributes(line.get("bbox"), block.get("bbox") or content_bbox)
            line["layout_attributes"]["horizontal_alignment"] = str(line.get("alignment") or block.get("alignment") or "left")
            line["layout_attributes"]["indent_px"] = float(line.get("indent_px", 0.0) or 0.0)
            line["layout_attributes"]["line_index"] = int(line.get("line_index") or 0)
            for phrase in line.get("phrases", []) or []:
                phrase_text = _normalize_spaces(_phrase_render_text(phrase) or phrase.get("texte") or "")
                phrase_style_entries = _collect_style_entries_from_phrase(phrase)
                phrase["text_attributes"] = _text_statistics(phrase_text)
                phrase["style_attributes"] = _aggregate_style_characteristics(phrase_style_entries, phrase_text)
                phrase["layout_attributes"] = _bbox_relative_attributes(phrase.get("bbox"), line.get("bbox") or block.get("bbox") or content_bbox)
                phrase["layout_attributes"]["horizontal_alignment"] = str(phrase.get("alignment") or line.get("alignment") or "left")
                phrase["layout_attributes"]["indent_px"] = float(phrase.get("indent_px", 0.0) or 0.0)
                for span in phrase.get("spans", []) or []:
                    span_text = _normalize_spaces(span.get("texte") or span.get("text") or "")
                    span_style = span.get("style") if isinstance(span.get("style"), dict) else {}
                    span["text_attributes"] = _text_statistics(span_text)
                    span["style_attributes"] = _aggregate_style_characteristics([span_style] if span_style else [], span_text)
                    span["layout_attributes"] = _bbox_relative_attributes(span.get("bbox"), phrase.get("bbox") or line.get("bbox") or block.get("bbox") or content_bbox)
                    span["layout_attributes"]["horizontal_alignment"] = str(
                        span.get("alignment") or phrase.get("alignment") or line.get("alignment") or "left"
                    )
                    span["layout_attributes"]["indent_px"] = float(
                        span.get("indent_px", phrase.get("indent_px", line.get("indent_px", 0.0))) or 0.0
                    )
                    span["layout_attributes"]["line_index"] = int(line.get("line_index") or 0)
        for phrase in semantic_phrases:
            phrase_text = _normalize_spaces(_phrase_render_text(phrase) or phrase.get("texte") or "")
            phrase_style_entries = _collect_style_entries_from_phrase(phrase)
            phrase["text_attributes"] = _text_statistics(phrase_text)
            phrase["style_attributes"] = _aggregate_style_characteristics(phrase_style_entries, phrase_text)
            phrase["layout_attributes"] = _bbox_relative_attributes(phrase.get("bbox"), block.get("bbox") or content_bbox)
            phrase["layout_attributes"]["horizontal_alignment"] = str(phrase.get("alignment") or block.get("alignment") or "left")
            phrase["layout_attributes"]["indent_px"] = float(phrase.get("indent_px", 0.0) or 0.0)
        for semantic_span in semantic_spans:
            span_text = _normalize_spaces(semantic_span.get("texte") or semantic_span.get("text") or "")
            span_style = semantic_span.get("style") if isinstance(semantic_span.get("style"), dict) else {}
            semantic_span["text_attributes"] = _text_statistics(span_text)
            semantic_span["style_attributes"] = _aggregate_style_characteristics([span_style] if span_style else [], span_text)
            semantic_span["layout_attributes"] = _bbox_relative_attributes(semantic_span.get("bbox"), block.get("bbox") or content_bbox)
            semantic_span["layout_attributes"]["horizontal_alignment"] = str(
                semantic_span.get("alignment") or block.get("alignment") or "left"
            )
            semantic_span["layout_attributes"]["indent_px"] = 0.0
        for semantic_run in semantic_runs:
            run_text = _normalize_spaces(semantic_run.get("texte") or semantic_run.get("text") or "")
            semantic_run["text_attributes"] = _text_statistics(run_text)
            semantic_run["style_attributes"] = _aggregate_style_characteristics([], run_text)
            semantic_run["layout_attributes"] = _bbox_relative_attributes(semantic_run.get("bbox"), block.get("bbox") or content_bbox)
            semantic_run["layout_attributes"]["horizontal_alignment"] = str(block.get("alignment") or "left")
            semantic_run["layout_attributes"]["indent_px"] = 0.0
        for semantic_group in semantic_groups:
            group_text = _normalize_spaces(semantic_group.get("texte") or semantic_group.get("text") or "")
            semantic_group["text_attributes"] = _text_statistics(group_text)
            semantic_group["style_attributes"] = _aggregate_style_characteristics([], group_text)
            semantic_group["layout_attributes"] = _bbox_relative_attributes(semantic_group.get("bbox"), block.get("bbox") or content_bbox)
            semantic_group["layout_attributes"]["horizontal_alignment"] = str(block.get("alignment") or "left")
            semantic_group["layout_attributes"]["indent_px"] = 0.0


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

        existing_role = (block.get("role") or "").lower()
        if existing_role in {"diagram_label", "figure_label"}:
            role = "diagram_label"
        elif existing_role == "diagram_text_label":
            role = "diagram_text_label"
        else:
            role = "body"
        text_l = (text or "").lower()
        is_equation_like = _is_equation_like_text(text_l)
        has_section_pattern = bool(re.match(r"^\s*(\d+(\.\d+)+)\b", text_l))
        is_figure_caption = bool(re.match(r"^\s*(figure|fig\.?)\s*\d+", text_l))
        is_short_title = is_short and (len(text.split()) <= 12)
        if role == "diagram_text_label":
            pass
        elif is_equation_like:
            role = "equation_inline"
        elif role == "diagram_label":
            pass
        elif y1 <= header_band[1] and is_short:
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

        line_alignment_candidates = []

        for line in block.get("lines", []):
            lb = line.get("bbox", bb)
            if isinstance(lb, (list, tuple)) and len(lb) == 4:
                line_text = line.get("line_text") or line.get("text") or _line_phrase_text(line)
                l_align, l_indent = _infer_alignment_with_context([float(v) for v in lb], content_bbox, page_w=img_w, text=line_text, role=role)
            else:
                l_align, l_indent = "left", max(0.0, x0 - content_bbox[0])
            line["alignment"] = l_align
            line["indent_px"] = float(max(0.0, l_indent))
            line["role"] = role
            line_alignment_candidates.append(line)
            for phrase in line.get("phrases", []):
                pb = phrase.get("bbox", lb)
                if isinstance(pb, (list, tuple)) and len(pb) == 4:
                    phrase_text = _phrase_render_text(phrase) or phrase.get("texte") or ""
                    phrase_container = lb if isinstance(lb, (list, tuple)) and len(lb) == 4 else content_bbox
                    p_align, p_indent = _infer_alignment_with_context([float(v) for v in pb], phrase_container, page_w=img_w, text=phrase_text, role=role)
                else:
                    p_align, p_indent = l_align, l_indent
                phrase["alignment"] = p_align
                phrase["indent_px"] = float(max(0.0, p_indent))
                phrase["role"] = role

        align, indent_px = _infer_alignment_with_context([x0, y0, x1, y1], content_bbox, page_w=img_w, text=text, role=role)
        block["role"] = role
        block["alignment"] = _infer_block_alignment_from_lines(line_alignment_candidates, content_bbox, fallback_alignment=align, role=role)
        block["indent_px"] = float(max(0.0, indent_px))
        for phrase in block.get("semantic_phrases", []) or []:
            pb = phrase.get("bbox", [x0, y0, x1, y1])
            if isinstance(pb, (list, tuple)) and len(pb) == 4:
                phrase_text = _phrase_render_text(phrase) or phrase.get("texte") or ""
                p_align, p_indent = _infer_alignment_with_context([float(v) for v in pb], block.get("bbox") or content_bbox, page_w=img_w, text=phrase_text, role=role)
            else:
                p_align, p_indent = block["alignment"], block["indent_px"]
            phrase["alignment"] = p_align
            phrase["indent_px"] = float(max(0.0, p_indent))
            phrase["role"] = role

    _attach_textual_characteristics(blocks, content_bbox)

    return {
        "margins": margins,
        "content_bbox": content_bbox,
        "header_band": header_band,
        "footer_band": footer_band,
    }


def apply_ai_font_matching(ocr_blocks, pil_img, enable_audit=False):
    return {
        "enabled": False,
        "ready": False,
        "threshold": None,
        "total_spans": 0,
        "attempted": 0,
        "matched": 0,
        "promoted": 0,
        "reasons": {"font_ai_removed": 1},
    }


def _normalize_spaces(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def _phrase_render_text(phrase):
    spans = phrase.get("spans", [])
    if spans:
        kept = []
        for sp in spans:
            if sp.get("skip_render"):
                continue
            t = _normalize_spaces(sp.get("texte", ""))
            if t:
                kept.append(t)
        if kept:
            return _normalize_spaces(" ".join(kept))
    return _normalize_spaces(phrase.get("texte", ""))


def _phrase_source_text(phrase):
    if not isinstance(phrase, dict):
        return ""
    spans = phrase.get("spans", [])
    if spans:
        kept = []
        for sp in spans:
            if not isinstance(sp, dict):
                continue
            t = _normalize_spaces(sp.get("texte") or sp.get("text") or "")
            if t:
                kept.append(t)
        if kept:
            return _normalize_spaces(" ".join(kept))
    return _normalize_spaces(phrase.get("text") or phrase.get("texte") or "")


def _semantic_phrase_text(phrase):
    if not isinstance(phrase, dict):
        return ""
    text = _normalize_spaces(phrase.get("text") or phrase.get("texte") or "")
    if text:
        return text
    return _phrase_render_text(phrase)


def _semantic_phrase_inline_classes(phrase):
    classes = set()
    if not isinstance(phrase, dict):
        return classes
    expr = phrase.get("expression_semantics") if isinstance(phrase.get("expression_semantics"), dict) else {}
    inline_class = _normalize_spaces(expr.get("inline_class") or "")
    if inline_class:
        classes.add(inline_class)
    for span in phrase.get("spans", []) or []:
        if not isinstance(span, dict):
            continue
        sem = span.get("expression_semantics") if isinstance(span.get("expression_semantics"), dict) else {}
        inline_class = _normalize_spaces(sem.get("inline_class") or "")
        if inline_class:
            classes.add(inline_class)
    for run in phrase.get("semantic_runs", []) or []:
        if not isinstance(run, dict):
            continue
        sem = run.get("expression_semantics") if isinstance(run.get("expression_semantics"), dict) else {}
        inline_class = _normalize_spaces(sem.get("inline_class") or "")
        if inline_class:
            classes.add(inline_class)
    return classes


def _semantic_phrase_lexical_word_count(text):
    return len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", text or ""))


def _classify_semantic_phrase_kind(block, phrase):
    text = _semantic_phrase_text(phrase)
    if not text:
        return "empty"

    role = _normalize_spaces((phrase or {}).get("role") or (block or {}).get("role") or "")
    reason = _normalize_spaces((phrase or {}).get("sentence_end_reason") or "")
    lower_text = text.lower()
    if role in {"header", "footer"}:
        return "header_footer"
    if role in {"figure_caption", "table_caption"}:
        return "caption"
    if role in {"diagram_label", "diagram_text_label", "figure_label"}:
        return "structural"
    if role == "equation_inline" or bool((phrase or {}).get("formula")):
        return "formula"

    inline_classes = _semantic_phrase_inline_classes(phrase)
    if "formula" in inline_classes:
        return "formula"
    if inline_classes & {"code", "reference"}:
        return "structural"
    if inline_classes & {"label"}:
        return "structural"

    lexical_words = _semantic_phrase_lexical_word_count(text)
    bullet_markers = re.findall(r"[•▪◦·]", text)
    step_markers = re.findall(r"(?:^|\s)\d+\.\s+[A-Za-z]", text)
    has_math_symbol = _contains_greek_or_symbol(text) or bool(re.search(r"[=<>±×÷∑∫∞≈≠≤≥√∆∂]", text))
    symbolic_chars = len(re.findall(r"[^A-Za-zÀ-ÿ0-9\s]", text))
    case_profile = _text_case_profile(text)
    has_terminal_punctuation = bool(re.search(r"[.!?]\s*$", text))
    short_tokens = re.findall(r"\b[A-Za-z]{1,2}\b", text)
    compact_tokens = re.findall(r"\b[A-Za-z0-9]{1,2}\b", text)
    short_structural_leads = {
        "this chapter covers",
    }

    if re.fullmatch(r"\d{1,4}", text):
        return "structural"
    if re.fullmatch(r"\d+(?:\.\d+){1,4}", text):
        return "structural"
    if re.match(r"^\d+(?:\.\d+){0,4}\s+[A-ZÀ-Ý]", text) and lexical_words <= 12:
        return "structural"

    if bullet_markers:
        if len(bullet_markers) >= 2 or lexical_words <= 12:
            return "structural"
    if len(step_markers) >= 2:
        return "structural"
    if re.fullmatch(r"(?:\d+\s+){2,}\d+", text):
        return "structural"
    if re.fullmatch(r"(?:[A-Z]\s+){3,}[A-Z]?", text):
        return "structural"
    if text.endswith(":") and lexical_words <= 12:
        return "structural"
    if lower_text in short_structural_leads:
        return "structural"
    if (
        reason == "eof"
        and lexical_words <= 16
        and not has_terminal_punctuation
        and re.search(r"\bis given (?:by|as)\b|\bare given (?:by|as)\b|\bis defined as\b", lower_text)
    ):
        return "structural"
    if (
        reason == "eof"
        and lexical_words <= 8
        and not has_terminal_punctuation
        and (
            case_profile == "title_like"
            or re.match(r"^[A-Z][A-Za-z0-9'\-]*(?:\s+(?:and|of|for|to|in|on|with|without|based|using|neural|style|deep|feature|input|output|layer|layers|maps?|transfer|descent|optimization|techniques?|variants?))[A-Za-z0-9'\-\s]*$", text)
        )
        and not re.match(
            r"^(?:this|that|these|those|we|you|they|he|she|it|in|on|at|for|from|during|since|because|if|when|while|after|before|first,?)\b",
            lower_text,
        )
    ):
        return "structural"

    if _is_equation_like_text(text):
        return "formula"
    if lexical_words <= 6 and "(" in text and ")" in text:
        if re.search(r"\b[ds]\s+[A-Za-z]\s*\(", text):
            return "formula"
        if re.search(r"\b[A-Za-z]\s+[A-Za-z]\s*\(\s*[A-Za-z]", text):
            return "formula"
    if lexical_words <= 14 and "(" in text and ")" in text and len(compact_tokens) >= 6:
        return "formula"
    if lexical_words <= 6 and len(short_tokens) >= max(2, lexical_words - 1) and any(tok.lower() in {"d", "e", "i", "j", "k", "n", "s", "x", "y", "z"} for tok in short_tokens):
        return "formula"
    if has_math_symbol and lexical_words <= 5:
        return "formula"
    if lexical_words <= 3 and symbolic_chars >= 3:
        return "formula"
    if lexical_words <= 8 and any(ord(ch) < 32 for ch in text):
        return "formula"

    if lexical_words <= 8 and case_profile == "upper":
        return "structural"

    return "prose"


def _normalize_word_label(word):
    if not isinstance(word, dict):
        return ""
    return _normalize_spaces(word.get("label") or word.get("text") or "")


def _starts_with_sentence_case(token):
    s = re.sub(r'^["\'\u201c\u201d\u2018\u2019(\[{]+', "", _normalize_spaces(token or ""))
    return bool(s[:1].isupper())


def _is_abbreviation_token(token):
    s = _normalize_spaces(token or "")
    if not s:
        return False
    core = re.sub(r'["\'\u201c\u201d\u2018\u2019)\]}]+$', "", s)
    lowered = core.lower()
    # Titles and discourse markers that cannot end a sentence on their own.
    # "al." is intentionally absent: "et al." at line-end IS a sentence
    # terminator when followed by a capitalised next token.
    if lowered in {
        # Titles
        "mr.", "mrs.", "ms.", "dr.", "prof.",
        # Common latin/discourse markers
        "etc.", "e.g.", "i.e.", "cf.", "vs.",
        # Document/publication references (cannot stand alone at a sentence end)
        "fig.", "eq.", "no.", "vol.", "p.", "pp.",
        "ch.", "sec.", "app.", "ref.", "tab.",
        "alg.", "def.", "thm.", "lem.", "prop.", "cor.", "rem.",
        # Statistical/measurement abbreviations
        "approx.", "est.", "repr.", "avg.", "max.", "min.", "std.", "var.",
    }:
        return True
    if re.fullmatch(r"(?:[A-Za-z]\.){2,}", core):
        return True
    return False


def _token_ends_sentence(token, next_token="", next_line_hard_break=False):
    s = _normalize_spaces(token or "")
    if not s:
        return False
    if not re.search(r'[.!?\u2026]+[\u201d\u2019"\')\]}\u00bb]*$', s):
        return False
    # C3 fix: check only the last word, not the full multi-word string.
    # Passing the full string never matched multi-word inputs like "see Fig."
    last_word = s.split()[-1] if s.split() else s
    if _is_abbreviation_token(last_word):
        return False
    nxt = _normalize_spaces(next_token or "")
    if nxt and _starts_with_sentence_case(nxt):
        return True
    if nxt and not _starts_with_sentence_case(nxt) and not next_line_hard_break:
        return False
    return True


def _looks_like_sentence(text):
    s = _normalize_spaces(text)
    if not s:
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", s)
    symbol_ratio = 0.0
    if s:
        symbol_ratio = len(re.findall(r"[^\w\sÀ-ÿ]", s)) / max(1, len(s))
    if symbol_ratio > 0.22:
        return False
    if re.search(r"[.!?]\s*$", s) and len(words) >= 2:
        return True
    # Some OCR segments can miss final punctuation.
    return len(words) >= 6


def _split_residual_chunks(text):
    # Keep separators out; residual chunks are lexical/numeric/symbol groups.
    return [c for c in re.split(r"[,:;|/()\[\]{}]+", text or "") if _normalize_spaces(c)]


def _append_fragment(base, frag):
    b = _normalize_spaces(base)
    f = _normalize_spaces(frag)
    if not b:
        return f
    if not f:
        return b
    if b.endswith("-"):
        return _normalize_spaces(f"{b[:-1]}{f}")
    if re.match(r"^[\.,;:!?%\)\]\}]", f):
        return f"{b}{f}"
    return _normalize_spaces(f"{b} {f}")


def _semantic_phrase_should_break_on_hard_boundary(current_text, current_line, previous_line=None):
    text = _normalize_spaces(current_text)
    if not text:
        return False
    if _token_ends_sentence(text):
        return True
    if isinstance(current_line, dict):
        if current_line.get("leading_marker") or current_line.get("paragraph_break_before"):
            return True
    current_indent = 0.0
    previous_indent = 0.0
    if isinstance(current_line, dict):
        current_indent = float(
            current_line.get("indent_px", (current_line.get("layout_attributes") or {}).get("indent_px", 0.0)) or 0.0
        )
    if isinstance(previous_line, dict):
        previous_indent = float(
            previous_line.get("indent_px", (previous_line.get("layout_attributes") or {}).get("indent_px", 0.0)) or 0.0
        )
    if current_indent - previous_indent > 24.0:
        return True
    return False


def _flush_pending_sentence(pending, residual_pool, sentence_min_words=6):
    t = _normalize_spaces(pending)
    if not t:
        return ""
    wc = len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", t))
    if wc >= sentence_min_words:
        return t
    residual_pool.append(t)
    return ""


def _strip_leading_bullets(text):
    s = text or ""
    m = re.match(r"^\s*([•▪◦·\-\*]+)\s*", s)
    if not m:
        return _normalize_spaces(s), ""
    bullet = m.group(1).strip()
    rest = s[m.end():]
    return _normalize_spaces(rest), bullet


def _normalize_spaced_caps(text):
    s = text or ""
    # "C HAPTER" -> "CHAPTER", "I MAGE N ET" -> "IMAGENET"
    return re.sub(r"\b(?:[A-Z]\s+){1,}[A-Z]\b", lambda m: m.group(0).replace(" ", ""), s)


def _is_hard_protected_text(text):
    s = _normalize_spaces(text)
    if not s:
        return True
    if re.search(r"(https?://|www\.|[\w\.-]+@[\w\.-]+\.\w+|doi:\s*|arxiv:)", s, flags=re.IGNORECASE):
        return True
    if re.search(r"[=<>±×÷∑∫∞≈≠≤≥√∆∂µλΩα-ωΑ-Ω]", s):
        return True
    return False


def _split_for_translation(text, max_chars=260):
    s = _normalize_spaces(text)
    if not s:
        return []
    parts = re.split(r"(?<=[\.\!\?\:\;])\s+", s)
    chunks = []
    for p in parts:
        p = _normalize_spaces(p)
        if not p:
            continue
        if len(p) <= max_chars:
            chunks.append(p)
            continue
        # Fallback split for very long segments.
        sub = re.split(r"(?<=,)\s+", p)
        cur = ""
        for t in sub:
            t = _normalize_spaces(t)
            if not t:
                continue
            cand = t if not cur else f"{cur} {t}"
            if len(cand) <= max_chars:
                cur = cand
            else:
                if cur:
                    chunks.append(cur)
                cur = t
        if cur:
            chunks.append(cur)
    return chunks


def _translate_direct_ct2_chunks(tr, text, target_lang):
    src = _normalize_spaces(text)
    if not src:
        return src
    chunks = _split_for_translation(src)
    if not chunks:
        return src
    out_chunks = []
    for ch in chunks:
        if _is_hard_protected_text(ch):
            out_chunks.append(ch)
            continue
        try:
            t = tr._ct2_translate(ch, target_lang=target_lang)
            t = _normalize_spaces(t)
            out_chunks.append(t if t else ch)
        except Exception:
            out_chunks.append(ch)
    return _normalize_spaces(" ".join(out_chunks))


def _target_lang_code(tr, target_lang):
    try:
        return (tr._normalize_lang_code(target_lang) or "").strip().lower()
    except Exception:
        return str(target_lang or "").strip().lower()


def _translation_leak_score(tr, text, target_lang):
    s = _normalize_spaces(text)
    if not s:
        return 1e9
    tgt = _target_lang_code(tr, target_lang)
    if tgt in {"", "en", "english"}:
        return 0.0
    try:
        en = float(tr._language_marker_counts(s, "en"))
        tg = float(tr._language_marker_counts(s, tgt))
        words = max(1.0, len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", s)))
        # Lower is better: penalize english markers and reward target markers.
        return (en * 1.4 - tg * 0.9) / words
    except Exception:
        # Fallback heuristic.
        en_common = len(re.findall(r"\b(the|and|with|for|from|this|that|are|you|your|will)\b", s, flags=re.IGNORECASE))
        return en_common / max(1.0, len(s.split()))


def _classify_char_token(tok):
    s = _normalize_spaces(tok)
    if not s:
        return None
    if re.fullmatch(r"\d+", s):
        return "chiffre" if len(s) == 1 else "nombre"
    if len(s) == 1 and re.fullmatch(r"[A-Za-zÀ-ÿ]", s):
        return "lettre"
    if re.fullmatch(r"[A-Za-zÀ-ÿ]+", s):
        return "mot_alpha"
    return "symbole"


def _build_hierarchical_extraction(blocks):
    phrases = []
    groupes = []
    mots = []
    residuels = []
    chiffres = []
    nombres = []
    lettres = []
    symboles = []
    residual_texts = []
    pending_sentence = ""
    line_stream = []

    semantic_phrase_stream = []
    for block in blocks or []:
        for phrase in block.get("semantic_phrases", []) or []:
            if phrase.get("render_mode") == "background_only":
                continue
            txt = _normalize_spaces(_phrase_render_text(phrase) or phrase.get("texte") or "")
            if txt:
                semantic_phrase_stream.append(txt)
    if semantic_phrase_stream:
        phrases = semantic_phrase_stream

    # Reconstitute sentence stream from visual lines (handles multi-line sentences better).
    allowed_roles = {"body", "figure_caption"}
    ignored_roles = {"diagram_label", "diagram_text_label", "equation_inline", "header", "footer"}
    if not phrases:
        for block in blocks or []:
            role = (block.get("role") or "").strip().lower()
            if role in ignored_roles:
                continue
            if role and role not in allowed_roles:
                continue
            for line in block.get("lines", []):
                if line.get("render_mode") == "background_only":
                    continue
                line_txt = _normalize_spaces(line.get("line_text", ""))
                if not line_txt:
                    # Build line from phrases when line_text is unavailable.
                    parts = []
                    for phrase in line.get("phrases", []):
                        if phrase.get("render_mode") == "background_only":
                            continue
                        ptxt = _phrase_render_text(phrase)
                        if ptxt:
                            parts.append(ptxt)
                    line_txt = _normalize_spaces(" ".join(parts))
                if line_txt:
                    # Skip isolated markers/noise in sentence stream.
                    if re.fullmatch(r"(?:\.\.\.|[•▪◦·\-\*]|\d{1,2}[.)]?)", line_txt):
                        continue
                    line_stream.append(line_txt)

        for ltxt in line_stream:
            # Join soft hyphenation across lines: "fea-" + "tures" => "features"
            if pending_sentence.endswith("-") and re.match(r"^[A-Za-zÀ-ÿ]", ltxt):
                pending_sentence = _normalize_spaces(f"{pending_sentence[:-1]}{ltxt}")
            else:
                pending_sentence = _append_fragment(pending_sentence, ltxt)
            if re.search(r"[.!?…]\s*$", pending_sentence):
                final_sentence = _normalize_spaces(pending_sentence)
                if final_sentence:
                    phrases.append(final_sentence)
                pending_sentence = ""

        if pending_sentence:
            flushed = _flush_pending_sentence(pending_sentence, residual_texts, sentence_min_words=4)
            if flushed:
                phrases.append(flushed)

    for rtxt in residual_texts:
        for chunk in _split_residual_chunks(rtxt):
            c = _normalize_spaces(chunk)
            if not c:
                continue
            words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", c)
            if len(words) >= 2:
                groupes.append(c)
                continue
            if len(words) == 1 and _normalize_spaces(words[0]) == c:
                mots.append(c)
                continue

            # Residual characters/tokens not covered by phrases/groups/words.
            for atom in re.findall(r"\d+|[A-Za-zÀ-ÿ]+|[^\s]", c):
                kind = _classify_char_token(atom)
                if kind in {"chiffre", "nombre", "lettre", "symbole"}:
                    entry = {"type": kind, "value": atom}
                    residuels.append(entry)
                    if kind == "chiffre":
                        chiffres.append(atom)
                    elif kind == "nombre":
                        nombres.append(atom)
                    elif kind == "lettre":
                        lettres.append(atom)
                    else:
                        symboles.append(atom)

    return {
        "phrases": phrases,
        "groupes_mots": groupes,
        "mots": mots,
        "residuels": residuels,
        "chiffres": chiffres,
        "nombres": nombres,
        "lettres": lettres,
        "symboles": symboles,
        "counts": {
            "phrases": len(phrases),
            "groupes_mots": len(groupes),
            "mots": len(mots),
            "residuels": len(residuels),
            "chiffres": len(chiffres),
            "nombres": len(nombres),
            "lettres": len(lettres),
            "symboles": len(symboles),
        },
    }


def _build_fidelity_layout_export(blocks):
    out_blocks = []
    for block in blocks or []:
        b = {
            "id": block.get("id"),
            "source": block.get("source"),
            "role": block.get("role"),
            "alignment": block.get("alignment"),
            "indent_px": float(block.get("indent_px", 0.0) or 0.0),
            "bbox": block.get("bbox"),
            "line_texts": list(block.get("line_texts", []) or []),
            "render_text_with_breaks": block.get("render_text_with_breaks", ""),
            "source_layout_mode": dict(block.get("source_layout_mode") or {}),
            "semantic_phrases": [],
            "semantic_spans": [],
            "lines": [],
        }
        for phrase in block.get("semantic_phrases", []) or []:
            b["semantic_phrases"].append(
                {
                    "sentence_id": phrase.get("sentence_id") or phrase.get("unit_id"),
                    "text": _phrase_render_text(phrase),
                    "raw_text": phrase.get("texte", ""),
                    "bbox": phrase.get("bbox"),
                    "start_line_index": phrase.get("start_line_index"),
                    "end_line_index": phrase.get("end_line_index"),
                    "line_indices": list(phrase.get("line_indices", []) or []),
                    "multi_line": bool(phrase.get("multi_line", False)),
                    "fragment_count": int(phrase.get("fragment_count", 0) or 0),
                    "fragments": list(phrase.get("fragments", []) or []),
                }
            )
        for semantic_span in block.get("semantic_spans", []) or []:
            b["semantic_spans"].append(
                {
                    "unit_id": semantic_span.get("unit_id"),
                    "text": semantic_span.get("text") or semantic_span.get("texte") or "",
                    "bbox": semantic_span.get("bbox"),
                    "line_indices": list(semantic_span.get("line_indices", []) or []),
                    "multi_line": bool(semantic_span.get("multi_line", False)),
                    "fragment_count": int(semantic_span.get("fragment_count", 0) or 0),
                    "fragments": list(semantic_span.get("fragments", []) or []),
                    "expression_semantics": semantic_span.get("expression_semantics", {}),
                    "expression_relations": semantic_span.get("expression_relations", {}),
                    "structural_context": semantic_span.get("structural_context", {}),
                }
            )
        for line in block.get("lines", []) or []:
            marker = line.get("leading_marker") or ""
            marker_norm = "•" if marker in {"", "▪", "◦", "·", "*"} else marker
            l = {
                "line_index": line.get("line_index"),
                "bbox": line.get("bbox"),
                "line_text": line.get("line_text", ""),
                "leading_marker": marker,
                "leading_marker_norm": marker_norm,
                "leading_marker_code": (" ".join(f"U+{ord(ch):04X}" for ch in str(marker)) if marker else ""),
                "indent_px": float(line.get("indent_px", 0.0) or 0.0),
                "hard_break_before": bool(line.get("hard_break_before", False)),
                "line_break_after": bool(line.get("line_break_after", True)),
                "phrases": [],
            }
            for phrase in line.get("phrases", []) or []:
                p = {
                    "bbox": phrase.get("bbox"),
                    "line_index": phrase.get("line_index"),
                    "leading_marker": phrase.get("leading_marker", ""),
                    "indent_px": float(phrase.get("indent_px", 0.0) or 0.0),
                    "hard_break_before": bool(phrase.get("hard_break_before", False)),
                    "line_break_after": bool(phrase.get("line_break_after", True)),
                    "text": _phrase_render_text(phrase),
                    "raw_text": phrase.get("texte", ""),
                    "translated_text": phrase.get("translated_text", ""),
                    "spans": [],
                }
                for span in phrase.get("spans", []) or []:
                    st = span.get("style", {}) if isinstance(span.get("style"), dict) else {}
                    p["spans"].append(
                        {
                            "unit_id": span.get("unit_id"),
                            "text": span.get("texte", ""),
                            "bbox": span.get("bbox"),
                            "source_kind": span.get("source_kind"),
                            "translatable": bool(span.get("translatable", False)),
                            "translation_strategy": span.get("translation_strategy"),
                            "translated_text": span.get("translated_text", ""),
                            "skip_render": bool(span.get("skip_render", False)),
                            "text_attributes": span.get("text_attributes", {}),
                            "style_attributes": span.get("style_attributes", {}),
                            "layout_attributes": span.get("layout_attributes", {}),
                            "style": {
                                "font": st.get("font"),
                                "size": st.get("size"),
                                "color": st.get("color"),
                                "highlight_color": st.get("highlight_color"),
                                "flags": st.get("flags", {}),
                            },
                        }
                    )
                l["phrases"].append(p)
            b["lines"].append(l)
        out_blocks.append(b)
    return {"blocks": out_blocks, "count_blocks": len(out_blocks)}


def _fmt_bbox(bb):
    if not isinstance(bb, (list, tuple)) or len(bb) != 4:
        return ""
    try:
        return ",".join(str(int(round(float(v)))) for v in bb)
    except Exception:
        return ""


def _write_layout_xml(blocks, filename, page_idx, img_w, img_h):
    root = ET.Element(
        "document_layout",
        {
            "source_file": str(filename or ""),
            "page": str(int(page_idx) + 1),
            "width": str(int(img_w)),
            "height": str(int(img_h)),
            "dpi": str(int(TARGET_DPI)),
            "version": "v1_fidelity_layout_xml",
        },
    )
    blocks_el = ET.SubElement(root, "blocks")
    for block in blocks or []:
        b_el = ET.SubElement(
            blocks_el,
            "block",
            {
                "id": str(block.get("id", "")),
                "source": str(block.get("source", "")),
                "role": str(block.get("role", "")),
                "alignment": str(block.get("alignment", "")),
                "indent_px": str(float(block.get("indent_px", 0.0) or 0.0)),
                "bbox": _fmt_bbox(block.get("bbox")),
            },
        )
        semantic_el = ET.SubElement(b_el, "semantic_phrases")
        for phrase in block.get("semantic_phrases", []) or []:
            sp_el = ET.SubElement(
                semantic_el,
                "semantic_phrase",
                {
                    "id": str(phrase.get("sentence_id") or phrase.get("unit_id") or ""),
                    "bbox": _fmt_bbox(phrase.get("bbox")),
                    "start_line_index": str(phrase.get("start_line_index", "")),
                    "end_line_index": str(phrase.get("end_line_index", "")),
                    "multi_line": "1" if bool(phrase.get("multi_line", False)) else "0",
                },
            )
            sp_el.text = _phrase_render_text(phrase)
        for line in block.get("lines", []) or []:
            marker = line.get("leading_marker") or ""
            l_el = ET.SubElement(
                b_el,
                "line",
                {
                    "index": str(line.get("line_index", "")),
                    "bbox": _fmt_bbox(line.get("bbox")),
                    "marker": marker,
                    "marker_norm": ("•" if marker in {"", "▪", "◦", "·", "*"} else marker),
                    "indent_px": str(float(line.get("indent_px", 0.0) or 0.0)),
                    "hard_break_before": "1" if bool(line.get("hard_break_before", False)) else "0",
                    "line_break_after": "1" if bool(line.get("line_break_after", True)) else "0",
                },
            )
            l_txt = ET.SubElement(l_el, "text")
            l_txt.text = line.get("line_text") or _line_phrase_text(line)
            for phrase in line.get("phrases", []) or []:
                p_el = ET.SubElement(
                    l_el,
                    "phrase",
                    {
                        "bbox": _fmt_bbox(phrase.get("bbox")),
                        "text": _phrase_render_text(phrase),
                        "raw_text": str(phrase.get("texte", "")),
                    },
                )
                for span in phrase.get("spans", []) or []:
                    st = span.get("style", {}) if isinstance(span.get("style"), dict) else {}
                    ET.SubElement(
                        p_el,
                        "span",
                        {
                            "text": str(span.get("texte", "")),
                            "bbox": _fmt_bbox(span.get("bbox")),
                            "skip_render": "1" if bool(span.get("skip_render", False)) else "0",
                            "font": str(st.get("font", "")),
                            "size": str(st.get("size", "")),
                            "color": str(st.get("color", "")),
                            "flags": json.dumps(st.get("flags", {}), ensure_ascii=False),
                        },
                    )
    out_name = f"layout_{_safe_runtime_stem(filename)}_{page_idx}.xml"
    out_path = os.path.join(RESULTS_DIR, out_name)
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def process_page(img, idx, filename, pdf_page=None, translate_to=None, force_ai=False, font_ai_audit=False, text_removal_mode="default", include_debug_layers=False):
    safe_filename = _safe_runtime_stem(filename)
    source_fn = f"src_{uuid.uuid4().hex}_{idx + 1}.png"
    source_path = os.path.join(RESULTS_DIR, source_fn)
    try:
        img.save(source_path)
    except Exception:
        source_path = ""

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
    if ocr_structure:
        ocr_structure = _prune_weak_ocr_lines(ocr_structure)
    font_ai_summary = {
        "enabled": False,
        "ready": False,
        "threshold": None,
        "total_spans": 0,
        "attempted": 0,
        "matched": 0,
        "promoted": 0,
        "reasons": {"extraction_ai_bypassed": 1} if not EXTRACTION_AI_ENABLED else {"font_ai_removed": 1},
    }
    if EXTRACTION_AI_ENABLED and ocr_structure:
        font_ai_summary = apply_ai_font_matching(ocr_structure, img, enable_audit=font_ai_audit)
    final_blocks = _dedupe_final_blocks(native_blocks, ocr_structure)
    merged_blocks_pre_postprocess = copy.deepcopy(final_blocks)
    final_blocks = _postprocess_blocks(final_blocks, img.width, img.height)
    _enrich_layout_markers(final_blocks)
    _build_semantic_phrases_for_blocks(final_blocks)
    _postprocess_blocks_semantic(final_blocks)
    _annotate_translation_contracts(final_blocks)
    _build_semantic_spans_for_blocks(final_blocks)
    _build_semantic_runs_for_blocks(final_blocks)
    _build_semantic_groups_for_blocks(final_blocks)
    merged_blocks_postprocess = copy.deepcopy(final_blocks)
    immutable_overlays = _extract_immutable_overlays(final_blocks, img, filename, idx)

    layout_meta = _annotate_layout(final_blocks, img.width, img.height)
    visual_style_profile = {}
    try:
        final_blocks, visual_style_profile = build_page_style_profile(
            final_blocks,
            layout_meta=layout_meta,
            page_width=img.width,
            page_height=img.height,
        )
    except Exception as e:
        print(f"Erreur style profiler : {e}")
    # Ensure layout markers and semantic phrases remain present after style profiling transforms.
    _enrich_layout_markers(final_blocks)
    _build_semantic_phrases_for_blocks(final_blocks)
    _postprocess_blocks_semantic(final_blocks)
    _annotate_translation_contracts(final_blocks)
    _build_semantic_spans_for_blocks(final_blocks)
    _build_semantic_runs_for_blocks(final_blocks)
    _build_semantic_groups_for_blocks(final_blocks)

    # 3. GÉNERATION DU FOND MAÎTRE NETTOYÉ (Workflow Inpainting IA)
    bg_master_path = ""
    mask_master_path = ""
    text_removal_debug = {}
    try:
        text_regions = _collect_text_regions_for_inpainting(final_blocks, non_text_zones, immutable_overlays=immutable_overlays)
        clean_bgr, mask, text_removal_debug = text_removal_strategy.remove(img, text_regions, mode=text_removal_mode)
        # Pass complémentaire : effacer les mots PDF non couverts par l'extraction
        # (en-têtes, pieds de page, grands titres décoratifs, etc.)
        if pdf_page is not None:
            clean_bgr = _erase_uncovered_pdf_words(clean_bgr, pdf_page, text_regions, sx, sy)
        bg_name = f"bg_master_{safe_filename}_{idx}.png"
        bg_master_path = os.path.join(RESULTS_DIR, bg_name)
        cv2.imwrite(bg_master_path, clean_bgr)
        mask_name = f"mask_master_{safe_filename}_{idx}.png"
        mask_master_path = os.path.join(RESULTS_DIR, mask_name)
        cv2.imwrite(mask_master_path, mask)
    except Exception as e:
        print(f"Erreur génération fond maître chirurgical : {e}")

    p6_bg_audit = _p6_audit_background(
        final_blocks,
        text_regions if "text_regions" in dir() else [],
        text_removal_debug,
        page_id=f"{filename}_{idx}",
        img_width=img.width,
        img_height=img.height,
    )

    # 4. VISUALISATION (Bboxes colorées pour l aperçu)
    vis_fn = f"vis_{safe_filename}_{idx}.jpg"
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
    display_text += f"FONTS: {len(native_blocks)} blocs natifs | OCR: {len(ocr_structure)} blocs OCR\n"
    
    for block in final_blocks:
        source_tag = "NATIVE" if block.get("source") == "native" else "OCR"
        display_text += (
            f"\n[{source_tag} BLOC {block.get('id')} - role={block.get('role')} "
            f"align={block.get('alignment')} indent_px={block.get('indent_px', 0)} "
            f"bbox={block['bbox']}]\n"
        )
        for line in block["lines"]:
            line_txt = _line_phrase_text(line)
            marker = (line.get("leading_marker") or "")
            if marker:
                marker_code = " ".join(f"U+{ord(ch):04X}" for ch in str(marker))
            else:
                marker_code = ""
            marker_norm = "•" if marker in {"", "▪", "◦", "·", "*"} else marker
            indent_px = float(line.get("indent_px", 0.0) or 0.0)
            hard_break = bool(line.get("hard_break_before", False))
            line_break = bool(line.get("line_break_after", True))
            display_text += (
                "  [LINE "
                f"idx={line.get('line_index', '?')} marker={repr(marker)} marker_norm={repr(marker_norm)} marker_code={marker_code} "
                f"indent_px={indent_px:.1f} hard_break_before={int(hard_break)} "
                f"line_break_after={int(line_break)} bbox={line.get('bbox')}]\n"
            )
            display_text += f"    text: {line_txt}\n"
            for p in line.get("phrases", []):
                ptxt = _phrase_render_text(p)
                display_text += (
                    "    [PHRASE "
                    f"line={p.get('line_index', '?')} marker={repr(p.get('leading_marker', ''))} "
                    f"indent_px={float(p.get('indent_px', 0.0) or 0.0):.1f} "
                    f"hard_break_before={int(bool(p.get('hard_break_before', False)))} "
                    f"line_break_after={int(bool(p.get('line_break_after', True)))} "
                    f"bbox={p.get('bbox')}]\n"
                )
                display_text += f"      text: {ptxt}\n"
                for span in p.get("spans", []):
                    s, b = span.get("style", {}), span.get("bbox", [0, 0, 0, 0])
                    font_name = s.get("font", "Unknown")
                    font_size = float(s.get("size", 12.0))
                    color = s.get("color", "#000000")
                    flags = s.get("flags", {}) if isinstance(s.get("flags"), dict) else {}
                    flag_txt = ",".join([k for k, v in flags.items() if v]) or "none"
                    display_text += (
                        "      - "
                        f"[{font_name} {font_size:.1f}pt {color} flags={flag_txt} bbox={b}] "
                        f"{span.get('texte', '')}\n"
                    )
                    if font_ai_audit and "font_ai_audit" in s:
                        audit = s["font_ai_audit"]
                        display_text += (
                            "        font_ai_audit: "
                            f"candidate={audit.get('font_ai')} score={audit.get('score')} "
                            f"selected={audit.get('selected_font')} reason={audit.get('reason')}\n"
                        )
    hierarchical_extraction = _build_hierarchical_extraction(final_blocks)
    fidelity_layout = _build_fidelity_layout_export(final_blocks)
    layout_xml_path = ""
    try:
        layout_xml_path = _write_layout_xml(final_blocks, filename, idx, img.width, img.height)
    except Exception as e:
        print(f"Erreur écriture layout XML: {e}")

    if hierarchical_extraction.get("phrases"):
        display_text += "\n[PHRASES_RECONSTITUEES]\n"
        for i, s in enumerate(hierarchical_extraction.get("phrases", []), start=1):
            display_text += f"  {i}. {s}\n"

    page_structure = {
        "blocks": final_blocks,
        "background_path": bg_master_path,
        "source_image_path": source_path,
        "source_image_url": f"/results/{source_fn}" if source_path else "",
        "mask_master_path": mask_master_path,
        "immutable_overlays": immutable_overlays,
        "text_removal_debug": text_removal_debug,
        "p6_bg_audit": p6_bg_audit,
        "non_text_zones": non_text_zones,
        "images": native_images,
        "drawings": native_drawings,
        "layout": layout_meta,
        "layout_xml_path": layout_xml_path,
        "visual_style_profile": visual_style_profile,
        "style_profile": visual_style_profile,
        "font_ai_summary": font_ai_summary,
        "layout_version": "v4_layout_roles_alignment_style_profile",
        "dimensions": {"width": img.width, "height": img.height, "dpi": TARGET_DPI, "unit": "px"},
    }
    if include_debug_layers:
        page_structure["debug_extraction_layers"] = {
            "raw_ocr_words": copy.deepcopy(raw_ocr),
            "ocr_blocks_parsed": copy.deepcopy(ocr_structure),
            "merged_blocks_pre_postprocess": merged_blocks_pre_postprocess,
            "merged_blocks_postprocess": merged_blocks_postprocess,
        }
    try:
        # Canonical structure enrichment for stable translation/reconstruction.
        page_structure = layout_v2_builder.build(page_structure)
        if EXTRACTION_AI_ENABLED:
            page_structure, layout_ai_info = layout_ai_enricher.enrich(page_structure, img)
            if layout_ai_info.get("applied"):
                page_structure = layout_v2_builder.build(page_structure)
        else:
            layout_ai_info = _disabled_layout_ai_info()
        page_structure, postprocess_info = apply_page_extraction_postprocessors(page_structure)
        if postprocess_info.get("changed") or postprocess_info.get("applied") or page_structure.get("native_structure"):
            page_structure = layout_v2_builder.build(page_structure)
        page_structure["layout_ai"] = layout_ai_info
        page_structure["extraction_postprocess"] = postprocess_info
        _enrich_layout_markers(page_structure.get("blocks", []))
        _build_semantic_phrases_for_blocks(page_structure.get("blocks", []))
        _postprocess_blocks_semantic(page_structure.get("blocks", []))
        _annotate_translation_contracts(page_structure.get("blocks", []), page_context=page_structure)
        _build_semantic_spans_for_blocks(page_structure.get("blocks", []))
        _build_semantic_runs_for_blocks(page_structure.get("blocks", []))
        _build_semantic_groups_for_blocks(page_structure.get("blocks", []))
        hierarchical_extraction = _build_hierarchical_extraction(page_structure.get("blocks", []) or [])
        fidelity_layout = _build_fidelity_layout_export(page_structure.get("blocks", []) or [])
    except Exception as e:
        print(f"Erreur LayoutV2Builder: {e}")

    return {
        "page": idx + 1, 
        "content": display_text,
        "hierarchical_extraction": hierarchical_extraction,
        "fidelity_layout": fidelity_layout,
        "page_role": page_structure.get("page_role"),
        "document_type": page_structure.get("document_type"),
        "layout_type": page_structure.get("layout_type"),
        "style_profile": page_structure.get("style_profile"),
        "page_family": page_structure.get("page_family"),
        "page_family_group": page_structure.get("page_family_group"),
        "page_case": page_structure.get("page_case"),
        "structure": page_structure,
        "visual_url": f"/results/{vis_fn}"
    }


def _translate_unit_text(tr, text, target_lang, context, unit_kind="phrase"):
    src0 = _normalize_spaces(text)
    src_norm = _normalize_spaced_caps(src0)
    src, bullet = _strip_leading_bullets(src_norm)
    if not src:
        return src0

    word_count = len(re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'\-]*", src))
    force_translate = unit_kind == "phrase" and word_count >= 5 and not _is_hard_protected_text(src)
    if (not force_translate) and tr._is_protected_segment(src, block_role="body"):
        return src0

    domain = tr._detect_domain(context or src)
    subdomain = tr._detect_subdomain(context or src, domain=domain)
    out = tr._translate_text_hierarchical(
        src,
        target_lang=target_lang,
        block_context=context,
        block_role="body",
        domain=domain,
        subdomain=subdomain,
    )
    out = tr._restore_protected_tokens(src, out)
    out = tr._normalize_translation(out, target_lang=target_lang, original=src, context_text=context)
    out = tr._apply_domain_glossary(out, source_text=src, target_lang=target_lang, domain=domain, subdomain=subdomain)
    out = _normalize_spaces(out)

    # Retry once when translation falls back to source unexpectedly.
    if target_lang.lower() != "english" and (out.lower() == src.lower()) and not _is_hard_protected_text(src):
        out2 = tr._translate_text_hierarchical(
            src,
            target_lang=target_lang,
            block_context="",
            block_role="body",
            domain=domain,
            subdomain=subdomain,
        )
        out2 = tr._restore_protected_tokens(src, out2)
        out2 = tr._normalize_translation(out2, target_lang=target_lang, original=src, context_text="")
        out2 = tr._apply_domain_glossary(out2, source_text=src, target_lang=target_lang, domain=domain, subdomain=subdomain)
        out2 = _normalize_spaces(out2)
        if out2 and out2.lower() != src.lower():
            out = out2
        else:
            out3 = _translate_direct_ct2_chunks(tr, src, target_lang=target_lang)
            if out3 and out3.lower() != src.lower():
                out = out3

    # Language coherence pass: reduce EN leak in target language outputs.
    if target_lang.lower() != "english" and unit_kind in {"phrase", "group"}:
        leak_now = _translation_leak_score(tr, out, target_lang)
        leak_src = _translation_leak_score(tr, src, target_lang)
        if leak_now >= (leak_src - 0.01):
            out_alt = _translate_direct_ct2_chunks(tr, src, target_lang=target_lang)
            if out_alt:
                leak_alt = _translation_leak_score(tr, out_alt, target_lang)
                if leak_alt + 0.015 < leak_now:
                    out = out_alt

    if bullet:
        out = f"{bullet} {out}".strip()
    return _normalize_spaces(out)

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
        
        for page_result in pages_results:
            if not isinstance(page_result, dict):
                continue
            structure = page_result.get("structure")
            if not isinstance(structure, dict):
                continue
            blocks = structure.get("blocks", []) or []
            _enrich_layout_markers(blocks)
            _build_semantic_phrases_for_blocks(blocks)
            _postprocess_blocks_semantic(blocks)
            _annotate_translation_contracts(blocks, page_context=structure)
            _build_semantic_spans_for_blocks(blocks)
            _build_semantic_runs_for_blocks(blocks)
            _build_semantic_groups_for_blocks(blocks)
            page_result["hierarchical_extraction"] = _build_hierarchical_extraction(blocks)
            page_result["fidelity_layout"] = _build_fidelity_layout_export(blocks)

        # Nettoyage récursif des types numpy avant envoi
        cleaned_results = json_serializable(pages_results)
        return JSONResponse(content={"status": "success", "results": cleaned_results})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/translate-units")
async def translate_units(data: dict):
    try:
        pages = data.get("pages", []) if isinstance(data, dict) else []
        target_lang = (data.get("target_lang") if isinstance(data, dict) else None) or "French"
        tr = get_translator()

        out_pages = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            page_num = page.get("page")
            phrases = page.get("phrases", []) or []
            groupes = page.get("groupes_mots", []) or []
            mots = page.get("mots", []) or []
            context = _normalize_spaces(" ".join([str(x) for x in phrases[:12]]))

            t_phrases = [_translate_unit_text(tr, str(x), target_lang, context, unit_kind="phrase") for x in phrases]
            t_groupes = [_translate_unit_text(tr, str(x), target_lang, context, unit_kind="group") for x in groupes]
            t_mots = [_translate_unit_text(tr, str(x), target_lang, context, unit_kind="word") for x in mots]

            out_pages.append(
                {
                    "page": page_num,
                    "phrases": t_phrases,
                    "groupes_mots": t_groupes,
                    "mots": t_mots,
                }
            )

        return JSONResponse(content={"status": "success", "target_lang": target_lang, "pages": out_pages})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/translate")
async def translate_structure_endpoint(data: dict, target_lang: str = "French", style: str = None, tone: str = None):
    try:
        structure = data
        if isinstance(structure, dict) and "structure" in structure:
            structure = structure["structure"]

        translator = get_translator()
        if (
            isinstance(structure, dict)
            and structure.get("schema_version") == "layout.v2"
            and structure.get("page_role") == "toc"
        ):
            translated = translator.translate_layout_v2(structure, target_lang=target_lang)
            return JSONResponse(content={"status": "success", "structure": json_serializable(translated)})

        if isinstance(structure, dict) and isinstance(structure.get("pages"), list):
            pages = []
            for page in structure.get("pages", []):
                if (
                    isinstance(page, dict)
                    and page.get("schema_version") == "layout.v2"
                    and page.get("page_role") == "toc"
                ):
                    pages.append(translator.translate_layout_v2(page, target_lang=target_lang))
                else:
                    pages.append(translator.translate_page(page, target_lang=target_lang, style=style, tone=tone) if isinstance(page, dict) else page)
            return JSONResponse(content={"status": "success", "structure": json_serializable({"pages": pages})})

        if isinstance(structure, dict):
            translated = translator.translate_page(structure, target_lang=target_lang, style=style, tone=tone)
            return JSONResponse(content={"status": "success", "structure": json_serializable(translated)})

        return JSONResponse(content={"error": "Invalid payload: expected dict structure"}, status_code=400)
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/reconstruct")
async def reconstruct_document(data: dict, target_lang: str = None, debug_compare: bool = False, export_html: bool = False, style: str = None, tone: str = None, include_debug_pages: bool = False):
    try:
        if isinstance(data, dict) and "structure" in data:
            data = data["structure"]

        raw_pages = []
        if isinstance(data, dict) and isinstance(data.get("pages"), list):
            raw_pages = data.get("pages", [])
        elif isinstance(data, list):
            raw_pages = data
        elif isinstance(data, dict):
            raw_pages = [data]

        pages = []
        for page in raw_pages:
            if isinstance(page, dict):
                structure = page.get("structure")
                if isinstance(structure, dict):
                    pages.append(structure)
                    continue
            pages.append(page)
        source_pages_for_qa = copy.deepcopy(pages)
        if target_lang:
            print(f"  [Pipeline] Traduction vers {target_lang} activée...")
            translator = get_translator()
            for idx, page in enumerate(pages):
                if (
                    isinstance(page, dict)
                    and page.get("schema_version") == "layout.v2"
                    and page.get("page_role") == "toc"
                ):
                    page = translator.translate_layout_v2(page, target_lang=target_lang)
                else:
                    # 1. Traduction par bloc
                    page = translator.translate_page(page, target_lang=target_lang, style=style, tone=tone)
                # 2. Réajustement géométrique optionnel (désactivé par défaut pour préserver les positions source).
                if LAYOUT_OPTIMIZER_ON_TRANSLATION:
                    page = layout_optimizer.adjust_layout(page)
                pages[idx] = page

        recon = DocumentReconstructor()
        output_path = os.path.join(RESULTS_DIR, "reconstructed_output.pdf")
        recon.reconstruct({"pages": pages}, output_path)

        response = {"status": "success", "pdf_url": f"/results/reconstructed_output.pdf"}
        original_paths = [((p.get("source_image_path") if isinstance(p, dict) else "") or "") for p in source_pages_for_qa]
        if target_lang:
            coverage_report = analyze_document_coverage(
                source_pages_for_qa,
                pages,
                target_lang=(target_lang or "fr"),
            )
            rendered_text_report = analyze_rendered_text_coverage(
                source_pages_for_qa,
                pages,
                output_path,
                target_lang=(target_lang or "fr"),
            )
            coverage_report["rendered_text_report"] = rendered_text_report
            response["coverage_report"] = coverage_report
            response["publication_qa"] = publication_qa(
                source_pages_for_qa,
                pages,
                output_path,
                coverage_report=coverage_report,
                target_lang=(target_lang or "fr"),
                original_image_paths=original_paths,
            )
        if include_debug_pages:
            response["source_pages"] = json_serializable(source_pages_for_qa)
            response["translated_pages"] = json_serializable(pages)
        if debug_compare:
            if any(original_paths):
                response["visual_compare"] = compare_reconstruction(original_paths, output_path, dpi=TARGET_DPI)
            else:
                response["visual_compare"] = {"error": "source_image_path_absent_in_pages"}
        if export_html:
            html_path = os.path.join(RESULTS_DIR, "reconstructed_output.html")
            html_exporter.export(pages, html_path)
            response["html_url"] = "/results/reconstructed_output.html"
        return JSONResponse(content=response)
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/debug/visual-compare")
async def debug_visual_compare(data: dict):
    try:
        pages = data.get("pages", [])
        pdf_url = data.get("pdf_url", "/results/reconstructed_output.pdf")
        pdf_path = data.get("pdf_path")
        if not pdf_path:
            if isinstance(pdf_url, str) and pdf_url.startswith("/results/"):
                pdf_path = os.path.join(RESULTS_DIR, pdf_url.split("/results/", 1)[1])
            else:
                pdf_path = os.path.join(RESULTS_DIR, "reconstructed_output.pdf")

        original_paths = []
        for p in pages:
            if not isinstance(p, dict):
                original_paths.append("")
                continue
            sp = p.get("source_image_path")
            if sp:
                original_paths.append(sp)
                continue
            su = p.get("source_image_url", "")
            if isinstance(su, str) and su.startswith("/results/"):
                original_paths.append(os.path.join(RESULTS_DIR, su.split("/results/", 1)[1]))
            else:
                original_paths.append("")

        metrics = compare_reconstruction(original_paths, pdf_path, dpi=TARGET_DPI)
        return JSONResponse(content={"status": "success", "metrics": metrics})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/export_html")
async def export_html_document(data: dict):
    try:
        pages = data.get("pages", [])
        out_name = data.get("output_name", "reconstructed_output.html")
        out_name = os.path.basename(out_name)
        if not out_name.lower().endswith(".html"):
            out_name += ".html"
        out_path = os.path.join(RESULTS_DIR, out_name)
        html_exporter.export(pages, out_path)
        return JSONResponse(content={"status": "success", "html_url": f"/results/{out_name}"})
    except Exception as e:
        import traceback; print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/healthz")
async def healthcheck():
    return JSONResponse(
        content={
            "status": "ok",
            "service": "docs-parser",
            "layout_ai_available": bool(layout_ai_enricher) and EXTRACTION_AI_ENABLED,
            "extraction_ai_enabled": EXTRACTION_AI_ENABLED,
            "results_url": "/results/",
        }
    )

if __name__ == "__main__": uvicorn.run(app, host="0.0.0.0", port=8001)
