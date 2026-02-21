#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import urllib.request
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image


EAST_URL = "https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/master/frozen_east_text_detection.pb"


def download_if_missing(path: str, url: str = EAST_URL) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    print(f"[DL] Downloading EAST model -> {path}")
    urllib.request.urlretrieve(url, path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError("Model download failed or produced empty file.")


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def decode_east(scores, geometry, score_thresh: float) -> Tuple[List[Tuple[int,int,int,int]], List[float]]:
    """
    Decode EAST output into bounding boxes.
    Returns:
      rects: list of (x1,y1,x2,y2)
      confidences: list of scores
    """
    (num_rows, num_cols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]

        for x in range(num_cols):
            s = scores_data[x]
            if s < score_thresh:
                continue

            # EAST uses 4x downsampling
            offset_x = x * 4.0
            offset_y = y * 4.0

            angle = angles[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x0[x] + x2[x]
            w = x1[x] + x3[x]

            end_x = int(offset_x + (cos * x1[x]) + (sin * x2[x]))
            end_y = int(offset_y - (sin * x1[x]) + (cos * x2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(float(s))

    return rects, confidences


def build_text_mask_east(
    bgr: np.ndarray,
    net,
    input_size: int = 640,
    score_thresh: float = 0.5,
    nms_thresh: float = 0.4,
    box_dilate_px: int = 8,
    feather_blur_px: int = 4,
) -> np.ndarray:
    """
    Detect text boxes with EAST and return a mask uint8 (255=inpaint).
    """
    H, W = bgr.shape[:2]

    # resize while keeping ratio to multiple of 32
    newW = input_size
    newH = int(round(H * (newW / W)))
    newH = max(32, (newH // 32) * 32)
    newW = max(32, (newW // 32) * 32)

    rW = W / float(newW)
    rH = H / float(newH)

    resized = cv2.resize(bgr, (newW, newH))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), swapRB=False, crop=False)

    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    rects, confidences = decode_east(scores, geometry, score_thresh)
    if len(rects) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    # NMS
    boxes = np.array(rects)
    conf = np.array(confidences)
    idxs = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=conf.tolist(),
        score_threshold=score_thresh,
        nms_threshold=nms_thresh
    )

    mask = np.zeros((H, W), dtype=np.uint8)
    if len(idxs) == 0:
        return mask

    for i in idxs.flatten():
        (x1, y1, x2, y2) = boxes[i]
        # scale back
        x1 = int(max(0, x1 * rW))
        y1 = int(max(0, y1 * rH))
        x2 = int(min(W - 1, x2 * rW))
        y2 = int(min(H - 1, y2 * rH))
        # fill box
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

    # dilate to cover edges / antialias text
    if box_dilate_px > 0:
        k = 2 * box_dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    # feather
    if feather_blur_px > 0:
        k = 2 * feather_blur_px + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    return mask


def inpaint_opencv(bgr: np.ndarray, mask: np.ndarray, method: str = "telea", radius: int = 3) -> np.ndarray:
    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(bgr, mask, float(radius), flag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input image path")
    ap.add_argument("-o", "--output", required=True, help="Output image path")
    ap.add_argument("--east", default="models/frozen_east_text_detection.pb", help="Path to EAST .pb model")
    ap.add_argument("--input_size", type=int, default=640, help="EAST input size (multiple of 32). 512/640/768")
    ap.add_argument("--score", type=float, default=0.5, help="Score threshold")
    ap.add_argument("--nms", type=float, default=0.4, help="NMS threshold")
    ap.add_argument("--dilate", type=int, default=10, help="Mask dilation (px)")
    ap.add_argument("--feather", type=int, default=4, help="Mask feather blur (px)")
    ap.add_argument("--method", default="telea", choices=["telea", "ns"], help="Inpaint method")
    ap.add_argument("--radius", type=int, default=3, help="Inpaint radius")
    ap.add_argument("--save_mask", default=None, help="Optional: save mask path")
    args = ap.parse_args()

    download_if_missing(args.east)

    net = cv2.dnn.readNet(args.east)

    img = Image.open(args.input)
    bgr = pil_to_bgr(img)

    mask = build_text_mask_east(
        bgr,
        net,
        input_size=args.input_size,
        score_thresh=args.score,
        nms_thresh=args.nms,
        box_dilate_px=args.dilate,
        feather_blur_px=args.feather,
    )

    if args.save_mask:
        os.makedirs(os.path.dirname(args.save_mask) or ".", exist_ok=True)
        cv2.imwrite(args.save_mask, mask)

    out = inpaint_opencv(bgr, mask, method=args.method, radius=args.radius)
    bgr_to_pil(out).save(args.output)
    print(f"[OK] Saved: {args.output}")


if __name__ == "__main__":
    main()
