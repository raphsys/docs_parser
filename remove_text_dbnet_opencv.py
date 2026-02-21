#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import requests
from PIL import Image


# -----------------------------
# OpenCV official DBNet models (Google Drive IDs + SHA1)
# Source: OpenCV text spotting tutorial (DB models) :contentReference[oaicite:2]{index=2}
# -----------------------------

@dataclass(frozen=True)
class DBNetModelSpec:
    name: str
    gdrive_id: str
    sha1: str
    # default input size (w, h)
    input_size: Tuple[int, int]


DBNET_MODELS = {
    # Trained on ICDAR2015 (mostly English scene text) recommended: 736x1280 :contentReference[oaicite:3]{index=3}
    "ic15_r50": DBNetModelSpec(
        name="DB_IC15_resnet50.onnx",
        gdrive_id="17_ABp79PlFt9yPCxSaarVc_DKTmrSGGf",
        sha1="bef233c28947ef6ec8c663d20a2b326302421fa3",
        input_size=(1280, 736),
    ),
    "ic15_r18": DBNetModelSpec(
        name="DB_IC15_resnet18.onnx",
        gdrive_id="1vY_KsDZZZb_svd5RT6pjyI8BS1nPbBSX",
        sha1="19543ce09b2efd35f49705c235cc46d0e22df30b",
        input_size=(1280, 736),
    ),
    # TD500 detects English + Chinese; recommended: 736x736 :contentReference[oaicite:4]{index=4}
    "td500_r50": DBNetModelSpec(
        name="DB_TD500_resnet50.onnx",
        gdrive_id="19YWhArrNccaoSza0CfkXlA8im4-lAGsR",
        sha1="1b4dd21a6baa5e3523156776970895bd3db6960a",
        input_size=(736, 736),
    ),
    "td500_r18": DBNetModelSpec(
        name="DB_TD500_resnet18.onnx",
        gdrive_id="1sZszH3pEt8hliyBlTmB-iulxHP1dCQWV",
        sha1="8a3700bdc13e00336a815fc7afff5dcc1ce08546",
        input_size=(736, 736),
    ),
}


# -----------------------------
# Utilities
# -----------------------------

def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_gdrive(file_id: str, out_path: str, chunk_size: int = 1024 * 1024) -> None:
    """
    Minimal Google Drive downloader (no gdown).
    """
    url = "https://drive.google.com/uc?export=download&id=" + file_id
    session = requests.Session()

    r = session.get(url, stream=True)
    r.raise_for_status()

    # If Google asks for confirmation for large files, a confirm token is embedded in HTML.
    confirm = None
    content_type = r.headers.get("Content-Type", "")
    if "text/html" in content_type.lower():
        text = r.text
        m = re.search(r"confirm=([0-9A-Za-z_]+)", text)
        if m:
            confirm = m.group(1)

    if confirm:
        url2 = f"https://drive.google.com/uc?export=download&confirm={confirm}&id={file_id}"
        r = session.get(url2, stream=True)
        r.raise_for_status()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".part"

    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

    os.replace(tmp, out_path)


def ensure_model(model_dir: str, spec: DBNetModelSpec) -> str:
    path = os.path.join(model_dir, spec.name)
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(path):
        if sha1_file(path) == spec.sha1:
            return path
        # bad file -> redownload
        os.remove(path)

    print(f"[model] downloading {spec.name} ...")
    download_gdrive(spec.gdrive_id, path)
    got = sha1_file(path)
    if got != spec.sha1:
        raise RuntimeError(f"SHA1 mismatch for {spec.name}: got {got}, expected {spec.sha1}")
    return path


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def read_image_any(path: str) -> np.ndarray:
    # robust reading (supports weird PNGs)
    pil_img = Image.open(path)
    return pil_to_bgr(pil_img)


def write_image_any(path: str, bgr: np.ndarray) -> None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path)


# -----------------------------
# Text detection (DBNet via OpenCV high-level API)
# -----------------------------

def build_dbnet_detector(
    model_path: str,
    input_size: Tuple[int, int],
    bin_thresh: float = 0.55,
    poly_thresh: float = 0.65,
    max_candidates: int = 200,
    unclip_ratio: float = 2.0,
):
    """
    Creates OpenCV TextDetectionModel_DB with parameters tuned for fewer false positives.

    OpenCV recommended normalization for DB:
      scale = 1/255, mean=(122.6789,116.6688,104.0070) :contentReference[oaicite:5]{index=5}
    Post-processing params shown in OpenCV tutorial: binThresh, polyThresh, maxCandidates, unclipRatio :contentReference[oaicite:6]{index=6}
    """
    if not hasattr(cv2.dnn, "TextDetectionModel_DB") and not hasattr(cv2, "dnn_TextDetectionModel_DB"):
        raise RuntimeError(
            "OpenCV build missing TextDetectionModel_DB. Install opencv-contrib-python.\n"
            "Example: pip install -U opencv-contrib-python"
        )

    # Different bindings exist depending on build.
    # Prefer cv2.dnn.TextDetectionModel_DB if present; else fallback.
    if hasattr(cv2.dnn, "TextDetectionModel_DB"):
        model = cv2.dnn.TextDetectionModel_DB(model_path)
    else:
        model = cv2.dnn_TextDetectionModel_DB(model_path)

    # Postprocess thresholds (DB)
    model.setBinaryThreshold(float(bin_thresh))
    model.setPolygonThreshold(float(poly_thresh))
    model.setMaxCandidates(int(max_candidates))
    model.setUnclipRatio(float(unclip_ratio))

    # Normalization + input size
    # scale=1/255, mean shown in OpenCV tutorial :contentReference[oaicite:7]{index=7}
    scale = 1.0 / 255.0
    mean = (122.67891434, 116.66876762, 104.00698793)
    model.setInputParams(scale, input_size, mean)

    return model


def detect_text_polygons(
    bgr: np.ndarray,
    detector,
    min_conf: float = 0.55,
) -> List[np.ndarray]:
    """
    Returns list of polygons (Nx2 int32) for each detected text region.
    OpenCV detect() signature (Python): detect(frame) -> detections, confidences :contentReference[oaicite:8]{index=8}
    """
    out = detector.detect(bgr)
    # detect may return (detections, confidences) or only detections depending on build.
    if isinstance(out, tuple) and len(out) == 2:
        detections, confidences = out
    else:
        detections, confidences = out, None

    polys: List[np.ndarray] = []
    if detections is None:
        return polys

    # detections is expected as a list of quadrangles (4 points) in OpenCV DB pipeline.
    for i, det in enumerate(detections):
        if confidences is not None:
            c = float(confidences[i])
            if c < float(min_conf):
                continue

        pts = np.array(det, dtype=np.float32).reshape(-1, 2)
        # convert to int
        pts = np.round(pts).astype(np.int32)
        polys.append(pts)

    return polys


# -----------------------------
# Mask building: combine "holes" + detected text
# -----------------------------

def mask_from_holes(
    original_bgr: np.ndarray,
    holes_bgr: np.ndarray,
    diff_thresh: int = 8,
    morph_close: int = 3,
) -> np.ndarray:
    """
    If you provide an 'image with holes', this builds a mask by abs-diff.
    Works even if holes image is identical -> mask becomes empty.
    """
    if original_bgr.shape != holes_bgr.shape:
        holes_bgr = cv2.resize(holes_bgr, (original_bgr.shape[1], original_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

    diff = cv2.absdiff(original_bgr, holes_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mask = (gray > diff_thresh).astype(np.uint8) * 255

    if morph_close and morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    return mask


def mask_from_polygons(
    shape_hw: Tuple[int, int],
    polys: List[np.ndarray],
    dilate: int = 10,
    feather: int = 4,
) -> np.ndarray:
    """
    Builds a binary mask from polygons, with optional dilation and feather.
    """
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    if polys:
        cv2.fillPoly(mask, polys, 255)

    if dilate and dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))
        mask = cv2.dilate(mask, k, iterations=1)

    if feather and feather > 0:
        # feather by blur (keeps 0..255) then re-threshold soft edges during inpaint
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=float(feather), sigmaY=float(feather))
        # keep it uint8
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    # Ensure binary-ish mask for inpaint (OpenCV expects 0/255; feather OK but we hard threshold lightly)
    mask = (mask > 16).astype(np.uint8) * 255
    return mask


# -----------------------------
# Main function: remove text (DBNet -> mask -> inpaint)
# -----------------------------

def remove_text_generic_dbnet_cpu(
    original_bgr: np.ndarray,
    holes_bgr: Optional[np.ndarray] = None,
    *,
    model_dir: str = "./models",
    model_key: str = "ic15_r50",
    # CPU budget / target (you asked 512/1024). This controls the detector input size (not output).
    det_input: int = 512,
    # detection thresholds to reduce false positives
    bin_thresh: float = 0.55,
    poly_thresh: float = 0.65,
    min_conf: float = 0.55,
    unclip_ratio: float = 2.0,
    max_candidates: int = 200,
    # mask shaping
    dilate: int = 10,
    feather: int = 4,
    # inpainting
    method: str = "telea",
    radius: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (inpainted_bgr, final_mask).

    - If holes_bgr is given, its diff-mask is ORed with DBNet text mask.
    - det_input=512 is the requested CPU target. For best accuracy you can use 736/1280 as recommended by OpenCV. :contentReference[oaicite:9]{index=9}
    """
    if model_key not in DBNET_MODELS:
        raise ValueError(f"Unknown model_key={model_key}. Choices: {list(DBNET_MODELS.keys())}")

    spec = DBNET_MODELS[model_key]
    model_path = ensure_model(model_dir, spec)

    # Build detector with your CPU target size.
    # OpenCV tutorial recommends 736x736 or 736x1280 depending on model. :contentReference[oaicite:10]{index=10}
    # Here we allow forcing a smaller square input for CPU.
    input_size = (int(det_input), int(det_input))

    detector = build_dbnet_detector(
        model_path=model_path,
        input_size=input_size,
        bin_thresh=bin_thresh,
        poly_thresh=poly_thresh,
        max_candidates=max_candidates,
        unclip_ratio=unclip_ratio,
    )

    polys = detect_text_polygons(original_bgr, detector, min_conf=min_conf)
    text_mask = mask_from_polygons(
        (original_bgr.shape[0], original_bgr.shape[1]),
        polys,
        dilate=dilate,
        feather=feather,
    )

    final_mask = text_mask.copy()

    if holes_bgr is not None:
        holes_mask = mask_from_holes(original_bgr, holes_bgr, diff_thresh=8, morph_close=3)
        final_mask = cv2.bitwise_or(final_mask, holes_mask)

    # Inpaint
    flags = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    out = cv2.inpaint(original_bgr, final_mask, float(radius), flags)

    return out, final_mask


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Remove text on images using DBNet (OpenCV DNN) + inpainting (CPU, no Paddle).")
    ap.add_argument("original", help="Path to original image")
    ap.add_argument("-H", "--holes", default=None, help="Optional: image with holes (will be merged into mask)")
    ap.add_argument("-o", "--output", required=True, help="Output image path")
    ap.add_argument("--mask-out", default=None, help="Optional: save final mask path (debug)")

    ap.add_argument("--model-dir", default="./models")
    ap.add_argument("--model", default="ic15_r50", choices=list(DBNET_MODELS.keys()))
    ap.add_argument("--det-input", type=int, default=512, help="Detector input size (square). 512 for CPU. Try 736 for better accuracy.")

    ap.add_argument("--bin-thresh", type=float, default=0.55, help="Higher -> fewer positives")
    ap.add_argument("--poly-thresh", type=float, default=0.65, help="Higher -> fewer positives")
    ap.add_argument("--min-conf", type=float, default=0.55, help="Filter low confidence detections")
    ap.add_argument("--unclip", type=float, default=2.0)
    ap.add_argument("--max-candidates", type=int, default=200)

    ap.add_argument("--dilate", type=int, default=10)
    ap.add_argument("--feather", type=int, default=4)

    ap.add_argument("--method", default="telea", choices=["telea", "ns"])
    ap.add_argument("--radius", type=int, default=3)

    args = ap.parse_args()

    original = read_image_any(args.original)
    holes = read_image_any(args.holes) if args.holes else None

    out, mask = remove_text_generic_dbnet_cpu(
        original,
        holes,
        model_dir=args.model_dir,
        model_key=args.model,
        det_input=args.det_input,
        bin_thresh=args.bin_thresh,
        poly_thresh=args.poly_thresh,
        min_conf=args.min_conf,
        unclip_ratio=args.unclip,
        max_candidates=args.max_candidates,
        dilate=args.dilate,
        feather=args.feather,
        method=args.method,
        radius=args.radius,
    )

    write_image_any(args.output, out)
    if args.mask_out:
        Image.fromarray(mask).save(args.mask_out)

    print(f"[ok] wrote: {args.output}")
    if args.mask_out:
        print(f"[ok] wrote mask: {args.mask_out}")


if __name__ == "__main__":
    main()

