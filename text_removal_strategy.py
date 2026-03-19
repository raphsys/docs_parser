import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

from remove_text_generic import inpaint_opencv


class TextRemovalStrategy:
    """
    Unified text removal interface.
    mode:
      - "default": fast CPU inpaint (Telea) with bbox mask
    """

    def __init__(self):
        pass

    def remove(self, page_image: Image.Image, regions: List[List[int]], mode: str = "default") -> Tuple[np.ndarray, np.ndarray, Dict]:
        bgr = cv2.cvtColor(np.array(page_image.convert("RGB")), cv2.COLOR_RGB2BGR)
        mask = self._build_bbox_mask(bgr.shape[:2], regions)
        debug = {"mode": mode, "regions": len(regions), "mask_nonzero": int(np.count_nonzero(mask))}

        if np.count_nonzero(mask) == 0:
            return bgr, mask, debug

        out = inpaint_opencv(bgr, mask, method="telea", radius=6.0)
        debug["engine"] = "telea"

        return out, mask, debug

    def _build_bbox_mask(self, shape_hw: Tuple[int, int], regions: List[List[int]]) -> np.ndarray:
        h, w = shape_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        for r in regions:
            if not isinstance(r, (list, tuple)) or len(r) != 4:
                continue
            x0, y0, x1, y1 = [int(v) for v in r]
            x0 = max(0, min(w - 1, x0 - 2))
            y0 = max(0, min(h - 1, y0 - 2))
            x1 = max(0, min(w, x1 + 2))
            y1 = max(0, min(h, y1 + 2))
            if x1 <= x0 or y1 <= y0:
                continue
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        return mask
