#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import hashlib
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import cv2
import requests
from PIL import Image


# -----------------------------
# IO helpers
# -----------------------------

def read_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def read_gray(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))

def write_rgb(path: str, rgb: np.ndarray) -> None:
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)

def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# -----------------------------
# Mask utilities
# -----------------------------

def make_mask_from_explicit(mask_img: np.ndarray, thresh: int = 8) -> np.ndarray:
    """
    Any mask image -> binary mask {0,255}. White-ish => inpaint.
    """
    if mask_img.ndim == 3:
        mask_img = cv2.cvtColor(rgb_to_bgr(mask_img), cv2.COLOR_BGR2GRAY)
    mask = (mask_img > thresh).astype(np.uint8) * 255
    return mask

def make_mask_from_holes(original_rgb: np.ndarray, holes_rgb: np.ndarray, diff_thresh: int = 12) -> np.ndarray:
    """
    Mask = absdiff(original, holes) threshold.
    """
    if original_rgb.shape != holes_rgb.shape:
        holes_rgb = cv2.resize(holes_rgb, (original_rgb.shape[1], original_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    diff = cv2.absdiff(rgb_to_bgr(original_rgb), rgb_to_bgr(holes_rgb))
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mask = (gray > diff_thresh).astype(np.uint8) * 255
    return mask

def refine_mask(mask: np.ndarray, dilate_px: int = 10, feather_px: int = 4) -> np.ndarray:
    """
    Dilation to cover anti-aliased edges, then feather to avoid seams.
    Output is still uint8 {0,255}.
    """
    m = mask.copy().astype(np.uint8)

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, kernel, iterations=1)

    if feather_px > 0:
        k = 2 * feather_px + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
        m = (m > 16).astype(np.uint8) * 255

    return m


# -----------------------------
# OpenCV inpaint
# -----------------------------

def inpaint_opencv(original_rgb: np.ndarray, mask_u8: np.ndarray, method: str = "telea", radius: int = 3) -> np.ndarray:
    bgr = rgb_to_bgr(original_rgb)
    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    out = cv2.inpaint(bgr, mask_u8, float(radius), flag)
    return bgr_to_rgb(out)


# -----------------------------
# Optional: DBNet auto-mask (OpenCV DNN)
# -----------------------------
@dataclass(frozen=True)
class DBNetModelSpec:
    name: str
    gdrive_id: str
    sha1: str
    input_size: Tuple[int, int]

DBNET_MODELS = {
    # OpenCV official IDs / SHA1 from tutorial: https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html
    "ic15_r50": DBNetModelSpec("DB_IC15_resnet50.onnx", "17_ABp79PlFt9yPCxSaarVc_DKTmrSGGf",
                              "bef233c28947ef6ec8c663d20a2b326302421fa3", (1280, 736)),
    "ic15_r18": DBNetModelSpec("DB_IC15_resnet18.onnx", "1vY_KsDZZZb_svd5RT6pjyI8BS1nPbBSX",
                              "19543ce09b2efd35f49705c235cc46d0e22df30b", (1280, 736)),
}

def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def download_gdrive(file_id: str, out_path: str) -> None:
    url = "https://drive.google.com/uc?export=download&id=" + file_id
    sess = requests.Session()
    r = sess.get(url, stream=True)
    r.raise_for_status()

    confirm = None
    if "text/html" in (r.headers.get("Content-Type", "").lower()):
        m = re.search(r"confirm=([0-9A-Za-z_]+)", r.text)
        if m:
            confirm = m.group(1)

    if confirm:
        url2 = f"https://drive.google.com/uc?export=download&confirm={confirm}&id={file_id}"
        r = sess.get(url2, stream=True)
        r.raise_for_status()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    os.replace(tmp, out_path)

def ensure_dbnet(model_dir: str, key: str) -> str:
    spec = DBNET_MODELS[key]
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, spec.name)
    if os.path.exists(path) and sha1_file(path) == spec.sha1:
        return path
    if os.path.exists(path):
        os.remove(path)
    print(f"[model] downloading {spec.name} ...")
    download_gdrive(spec.gdrive_id, path)
    if sha1_file(path) != spec.sha1:
        raise RuntimeError("DBNet model SHA1 mismatch after download.")
    return path

def dbnet_mask_opencv(original_rgb: np.ndarray, model_path: str, det_input: int = 512,
                      bin_thresh: float = 0.6, poly_thresh: float = 0.75, min_conf: float = 0.65) -> np.ndarray:
    # Build DBNet detector (high-level API)
    if hasattr(cv2.dnn, "TextDetectionModel_DB"):
        model = cv2.dnn.TextDetectionModel_DB(model_path)
    else:
        model = cv2.dnn_TextDetectionModel_DB(model_path)

    model.setBinaryThreshold(float(bin_thresh))
    model.setPolygonThreshold(float(poly_thresh))
    model.setUnclipRatio(2.0)
    model.setMaxCandidates(200)

    # normalization from OpenCV tutorial
    model.setInputParams(1.0/255.0, (det_input, det_input), (122.67891434, 116.66876762, 104.00698793))

    bgr = rgb_to_bgr(original_rgb)
    out = model.detect(bgr)

    if isinstance(out, tuple) and len(out) == 2:
        dets, confs = out
    else:
        dets, confs = out, None

    mask = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.uint8)
    if dets is None:
        return mask

    polys: List[np.ndarray] = []
    for i, det in enumerate(dets):
        if confs is not None and float(confs[i]) < float(min_conf):
            continue
        pts = np.array(det, dtype=np.float32).reshape(-1, 2)
        polys.append(np.round(pts).astype(np.int32))

    if polys:
        cv2.fillPoly(mask, polys, 255)
    return mask


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Inpaint with explicit mask (preferred), or holes diff-mask, or DBNet auto-mask.")
    ap.add_argument("original", help="Original image path")
    ap.add_argument("-o", "--output", required=True, help="Output image path")

    ap.add_argument("--mask", default=None, help="Explicit mask image path (white=inpaint). Highest priority.")
    ap.add_argument("--holes", default=None, help="Optional: holes image path (same content with removed text).")

    ap.add_argument("--method", default="telea", choices=["telea", "ns"])
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--dilate", type=int, default=10)
    ap.add_argument("--feather", type=int, default=4)
    ap.add_argument("--diff-thresh", type=int, default=12)

    # DBNet fallback if neither mask nor holes provided
    ap.add_argument("--dbnet", action="store_true", help="Use DBNet auto-mask if no --mask/--holes.")
    ap.add_argument("--model-dir", default="./models")
    ap.add_argument("--dbnet-model", default="ic15_r50", choices=list(DBNET_MODELS.keys()))
    ap.add_argument("--det-input", type=int, default=512)
    ap.add_argument("--bin-thresh", type=float, default=0.6)
    ap.add_argument("--poly-thresh", type=float, default=0.75)
    ap.add_argument("--min-conf", type=float, default=0.65)

    ap.add_argument("--save-mask", default=None, help="Optional: save final mask used for inpainting.")
    args = ap.parse_args()

    original_rgb = read_rgb(args.original)

    # Choose mask source
    if args.mask:
        m = read_gray(args.mask)
        mask = make_mask_from_explicit(m, thresh=8)
    elif args.holes:
        holes_rgb = read_rgb(args.holes)
        mask = make_mask_from_holes(original_rgb, holes_rgb, diff_thresh=args.diff_thresh)
    else:
        if not args.dbnet:
            raise SystemExit("No --mask or --holes provided. Add --dbnet to auto-detect text regions, or provide --mask.")
        model_path = ensure_dbnet(args.model_dir, args.dbnet_model)
        mask = dbnet_mask_opencv(
            original_rgb,
            model_path=model_path,
            det_input=args.det_input,
            bin_thresh=args.bin_thresh,
            poly_thresh=args.poly_thresh,
            min_conf=args.min_conf,
        )

    # Refine mask
    mask = refine_mask(mask, dilate_px=args.dilate, feather_px=args.feather)

    if args.save_mask:
        Image.fromarray(mask).save(args.save_mask)

    # Inpaint
    out_rgb = inpaint_opencv(original_rgb, mask, method=args.method, radius=args.radius)
    write_rgb(args.output, out_rgb)
    print(f"[ok] wrote: {args.output}")
    if args.save_mask:
        print(f"[ok] wrote mask: {args.save_mask}")


if __name__ == "__main__":
    main()

