import os
from statistics import median

import cv2
import fitz
import numpy as np
from PIL import Image


def _safe_score(x):
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0


def _text_mask(gray):
    # Otsu inverse threshold: text-ish pixels become white(255).
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th


def _component_heights(mask):
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    hs = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 6 or w < 2 or h < 2:
            continue
        hs.append(float(h))
    return hs


def _height_signature(mask):
    hs = _component_heights(mask)
    if not hs:
        return [0.0, 0.0, 0.0]
    hs_sorted = sorted(hs)
    n = len(hs_sorted)
    p50 = hs_sorted[int(0.50 * (n - 1))]
    p80 = hs_sorted[int(0.80 * (n - 1))]
    p95 = hs_sorted[int(0.95 * (n - 1))]
    return [p50, p80, p95]


def _gap_signature(mask):
    # row occupancy => white-space gaps between text bands.
    row_sum = np.sum(mask > 0, axis=1)
    rows = row_sum > max(1, int(mask.shape[1] * 0.002))
    gaps = []
    run = 0
    in_gap = False
    for v in rows:
        if not v:
            run += 1
            in_gap = True
        elif in_gap:
            if run > 0:
                gaps.append(run)
            run = 0
            in_gap = False
    if run > 0:
        gaps.append(run)
    if not gaps:
        return [0.0, 0.0]
    gaps = sorted(gaps)
    return [float(median(gaps)), float(np.percentile(gaps, 90))]


def _alignment_signature(mask):
    # Left-edge profile of rows containing text.
    h, w = mask.shape
    row_sum = np.sum(mask > 0, axis=1)
    rows = np.where(row_sum > max(1, int(w * 0.002)))[0]
    if rows.size == 0:
        return [0.0, 0.0]
    lefts = []
    for y in rows:
        xs = np.where(mask[y] > 0)[0]
        if xs.size:
            lefts.append(float(xs.min()))
    if not lefts:
        return [0.0, 0.0]
    return [float(median(lefts)), float(np.std(lefts))]


def _table_lines_score(gray_ref, gray_rec):
    def line_density(gray):
        inv = 255 - gray
        bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
        dens = (np.mean(h_lines > 0) + np.mean(v_lines > 0)) / 2.0
        return float(dens)

    d_ref = line_density(gray_ref)
    d_rec = line_density(gray_rec)
    # If no table-like line pattern is present in both, keep metric neutral.
    if d_ref < 0.003 and d_rec < 0.003:
        return 0.5
    return _safe_score(1.0 - abs(d_ref - d_rec) / max(d_ref, d_rec, 1e-6))


def _color_distance_score(rgb_ref, rgb_rec):
    # Compare median dark/text pixels color.
    def text_color(rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = gray < min(170, int(np.percentile(gray, 45)))
        if np.mean(mask) < 0.005:
            mask = gray < 140
        if np.mean(mask) < 0.003:
            return np.array([30.0, 30.0, 30.0], dtype=np.float32)
        vals = rgb[mask]
        return np.median(vals, axis=0).astype(np.float32)

    c1 = text_color(rgb_ref)
    c2 = text_color(rgb_rec)
    dist = float(np.linalg.norm(c1 - c2))
    # max distance in RGB approx 441.
    return _safe_score(1.0 - dist / 220.0)


def _vec_similarity(v1, v2):
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)
    rel = np.abs(a - b) / denom
    return _safe_score(1.0 - float(np.mean(rel)))


def compare_page_images(original_rgb, reconstructed_rgb):
    if original_rgb.shape != reconstructed_rgb.shape:
        reconstructed_rgb = cv2.resize(reconstructed_rgb, (original_rgb.shape[1], original_rgb.shape[0]), interpolation=cv2.INTER_AREA)

    gray_o = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    gray_r = cv2.cvtColor(reconstructed_rgb, cv2.COLOR_RGB2GRAY)
    m_o = _text_mask(gray_o)
    m_r = _text_mask(gray_r)

    hierarchy_consistency = _vec_similarity(_height_signature(m_o), _height_signature(m_r))
    spacing_consistency = _vec_similarity(_gap_signature(m_o), _gap_signature(m_r))
    alignment_consistency = _vec_similarity(_alignment_signature(m_o), _alignment_signature(m_r))
    color_distance = _color_distance_score(original_rgb, reconstructed_rgb)
    table_fidelity = _table_lines_score(gray_o, gray_r)
    overall = _safe_score(
        0.26 * hierarchy_consistency
        + 0.22 * spacing_consistency
        + 0.20 * alignment_consistency
        + 0.14 * color_distance
        + 0.18 * table_fidelity
    )
    return {
        "hierarchy_consistency": round(hierarchy_consistency, 4),
        "spacing_consistency": round(spacing_consistency, 4),
        "alignment_consistency": round(alignment_consistency, 4),
        "color_distance": round(color_distance, 4),
        "table_fidelity": round(table_fidelity, 4),
        "overall": round(overall, 4),
    }


def _render_pdf_pages(pdf_path, dpi=150):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=dpi)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        pages.append(rgb)
    doc.close()
    return pages


def compare_reconstruction(original_image_paths, reconstructed_pdf_path, dpi=150):
    if not os.path.exists(reconstructed_pdf_path):
        raise FileNotFoundError(f"PDF reconstruit introuvable: {reconstructed_pdf_path}")
    rec_pages = _render_pdf_pages(reconstructed_pdf_path, dpi=dpi)
    orig_pages = []
    for p in original_image_paths:
        if not p or not os.path.exists(p):
            orig_pages.append(None)
            continue
        arr = np.array(Image.open(p).convert("RGB"))
        orig_pages.append(arr)

    n = min(len(orig_pages), len(rec_pages))
    per_page = []
    for i in range(n):
        if orig_pages[i] is None:
            per_page.append({"page": i + 1, "error": "source_image_missing"})
            continue
        metrics = compare_page_images(orig_pages[i], rec_pages[i])
        metrics["page"] = i + 1
        per_page.append(metrics)

    valid = [m for m in per_page if "overall" in m]
    if not valid:
        aggregate = {"overall": 0.0}
    else:
        keys = ["hierarchy_consistency", "spacing_consistency", "alignment_consistency", "color_distance", "table_fidelity", "overall"]
        aggregate = {k: round(float(np.mean([m[k] for m in valid])), 4) for k in keys}

    return {
        "pages_compared": len(valid),
        "pages_total": len(per_page),
        "aggregate": aggregate,
        "per_page": per_page,
    }
