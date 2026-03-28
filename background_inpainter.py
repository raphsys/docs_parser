import os
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency at runtime
    cv2 = None


class BackgroundInpainter:
    DEFAULT_MODELS_ROOT = os.path.join(os.path.dirname(__file__), "ai_models", "inpainting")
    MODEL_CANDIDATES = (
        "lama/lama_fp32.onnx",
        "lama/model.onnx",
        "big-lama/model.onnx",
        "lama.onnx",
    )

    def __init__(self, enabled=None, backend=None, models_root=None):
        self.enabled = (
            (os.getenv("BACKGROUND_INPAINT_ENABLE", "1") == "1")
            if enabled is None
            else bool(enabled)
        )
        self.backend = str(os.getenv("BACKGROUND_INPAINT_BACKEND", "auto") if backend is None else backend).strip().lower()
        self.models_root = os.path.realpath(
            str(os.getenv("BACKGROUND_INPAINT_MODELS_ROOT", self.DEFAULT_MODELS_ROOT) if models_root is None else models_root)
        )
        self.inpaint_radius = max(1, int(os.getenv("BACKGROUND_INPAINT_RADIUS", "3")))
        self.lama_max_mask_ratio = max(0.01, min(0.95, float(os.getenv("BACKGROUND_INPAINT_LAMA_MAX_MASK_RATIO", "0.12"))))
        self.lama_max_mask_rects = max(1, int(os.getenv("BACKGROUND_INPAINT_LAMA_MAX_MASK_RECTS", "4")))
        self.lama_min_crop_edge = max(32, int(os.getenv("BACKGROUND_INPAINT_LAMA_MIN_CROP_EDGE", "96")))
        self._model_candidate_paths = self._collect_model_candidates()
        self._resolved_model_path = ""
        self._session = None
        self._ready_backend = None

    def status(self):
        backend = self._resolve_backend_name()
        return {
            "enabled": bool(self.enabled),
            "backend": backend,
            "models_root": self.models_root,
            "model_path": self._resolved_model_path,
            "ready": bool(backend),
        }

    def save_inpaint_overlay(self, source_image_path, crop_bbox, mask_rects, out_dir, kind="background_inpaint"):
        if not self.enabled:
            return None
        if not source_image_path or not os.path.exists(source_image_path):
            return None
        if not isinstance(crop_bbox, (list, tuple)) or len(crop_bbox) != 4:
            return None
        mask_rects = [r for r in (mask_rects or []) if isinstance(r, (list, tuple)) and len(r) == 4]
        if not mask_rects:
            return None
        try:
            with Image.open(source_image_path).convert("RGB") as im:
                crop_x0 = max(0, int(round(float(crop_bbox[0]))))
                crop_y0 = max(0, int(round(float(crop_bbox[1]))))
                crop_x1 = min(im.width, int(round(float(crop_bbox[2]))))
                crop_y1 = min(im.height, int(round(float(crop_bbox[3]))))
                if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
                    return None
                crop = im.crop((crop_x0, crop_y0, crop_x1, crop_y1))
                mask = Image.new("L", crop.size, 0)
                draw = ImageDraw.Draw(mask)
                for rect in mask_rects:
                    x0 = max(0, int(round(float(rect[0]) - crop_x0)))
                    y0 = max(0, int(round(float(rect[1]) - crop_y0)))
                    x1 = min(crop.width, int(round(float(rect[2]) - crop_x0)))
                    y1 = min(crop.height, int(round(float(rect[3]) - crop_y0)))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    draw.rectangle([x0, y0, x1, y1], fill=255)
                if not np.any(np.array(mask, dtype=np.uint8) > 0):
                    return None
                restored = self._inpaint_crop(crop, mask, mask_rects=mask_rects)
                if restored is None:
                    return None
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{kind}_{uuid.uuid4().hex[:12]}.png")
                restored.save(out_path)
                return {
                    "path": out_path,
                    "bbox": [crop_x0, crop_y0, crop_x1, crop_y1],
                    "kind": kind,
                    "backend": self._ready_backend or self._resolve_backend_name(),
                }
        except Exception:
            return None

    def _collect_model_candidates(self):
        candidates = []
        for rel_path in self.MODEL_CANDIDATES:
            candidate = os.path.join(self.models_root, rel_path)
            if os.path.isfile(candidate):
                candidates.append(os.path.realpath(candidate))
        return candidates

    def _lama_backend_requested(self):
        return self.enabled and self.backend in {"auto", "lama", "lama_onnx"}

    def _onnxruntime_available(self):
        try:
            import onnxruntime  # noqa: F401

            return True
        except Exception:
            return False

    def _resolve_backend_name(self):
        if not self.enabled:
            return ""
        if self._lama_backend_requested() and self._onnxruntime_available():
            if self._session is not None or self._ensure_session() is not None:
                return "lama_onnx"
        if self.backend in {"auto", "opencv", "cv2"} and cv2 is not None:
            return "opencv"
        return ""

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        if not self._lama_backend_requested() or not self._onnxruntime_available():
            return None
        try:
            import onnxruntime as ort
        except Exception:
            return None
        for candidate in self._model_candidate_paths:
            try:
                self._session = ort.InferenceSession(
                    candidate,
                    providers=["CPUExecutionProvider"],
                )
                self._resolved_model_path = candidate
                return self._session
            except Exception:
                continue
        self._session = None
        self._resolved_model_path = ""
        return None

    def _session_target_hw(self, session):
        if session is None:
            return None
        for inp in session.get_inputs():
            shape = tuple(inp.shape or ())
            if len(shape) != 4:
                continue
            h = shape[2]
            w = shape[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                return int(h), int(w)
        return None

    def _mask_fill_ratio(self, mask):
        mask_np = np.array(mask.convert("L"), dtype=np.uint8)
        if mask_np.size == 0:
            return 0.0
        return float(np.count_nonzero(mask_np)) / float(mask_np.size)

    def _select_backend(self, crop, mask, mask_rects=None):
        mask_rects = list(mask_rects or [])
        if self.backend in {"opencv", "cv2"}:
            return "opencv" if cv2 is not None else ""
        if self.backend in {"lama", "lama_onnx"}:
            return "lama_onnx" if self._ensure_session() is not None else ""
        if self.backend != "auto":
            return ""
        if cv2 is None:
            return "lama_onnx" if self._ensure_session() is not None else ""
        if self._ensure_session() is None:
            return "opencv"
        if min(crop.size) < self.lama_min_crop_edge:
            return "opencv"
        if len(mask_rects) > self.lama_max_mask_rects:
            return "opencv"
        if self._mask_fill_ratio(mask) > self.lama_max_mask_ratio:
            return "opencv"
        return "lama_onnx"

    def _inpaint_crop(self, crop, mask, mask_rects=None):
        backend = self._select_backend(crop, mask, mask_rects=mask_rects)
        self._ready_backend = backend
        if backend == "lama_onnx":
            restored = self._inpaint_crop_lama(crop, mask)
            if restored is not None:
                return restored
            if cv2 is not None:
                self._ready_backend = "opencv"
                return self._inpaint_crop_opencv(crop, mask)
            return crop
        if backend == "opencv":
            return self._inpaint_crop_opencv(crop, mask)
        return crop

    def _inpaint_crop_opencv(self, crop, mask):
        if cv2 is None:
            return crop
        img_rgb = np.array(crop.convert("RGB"), dtype=np.uint8)
        mask_np = np.array(mask.convert("L"), dtype=np.uint8)
        if img_rgb.size == 0 or mask_np.size == 0:
            return crop
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        restored_bgr = cv2.inpaint(img_bgr, mask_np, self.inpaint_radius, cv2.INPAINT_TELEA)
        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(restored_rgb)

    def _inpaint_crop_lama(self, crop, mask):
        session = self._ensure_session()
        if session is None:
            return None
        try:
            img_pil = crop.convert("RGB")
            mask_pil = mask.convert("L")
            orig_w, orig_h = img_pil.size
            target_hw = self._session_target_hw(session)
            if target_hw:
                target_h, target_w = target_hw
                if (orig_h, orig_w) != (target_h, target_w):
                    img_pil = img_pil.resize((target_w, target_h), Image.Resampling.BILINEAR)
                    mask_pil = mask_pil.resize((target_w, target_h), Image.Resampling.NEAREST)
            img_rgb = np.array(img_pil, dtype=np.float32) / 255.0
            mask_np = (np.array(mask_pil, dtype=np.float32) > 0).astype(np.float32)
            h, w = img_rgb.shape[:2]
            if not target_hw:
                pad_h = (8 - (h % 8)) % 8
                pad_w = (8 - (w % 8)) % 8
                if pad_h or pad_w:
                    img_rgb = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                    mask_np = np.pad(mask_np, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
            image_tensor = np.transpose(img_rgb, (2, 0, 1))[None, ...].astype(np.float32)
            mask_tensor = mask_np[None, None, ...].astype(np.float32)
            ort_inputs = {}
            for inp in session.get_inputs():
                name = str(inp.name or "")
                shape = tuple(inp.shape or ())
                if "mask" in name.lower() or (len(shape) == 4 and shape[1] == 1):
                    ort_inputs[name] = mask_tensor
                else:
                    ort_inputs[name] = image_tensor
            output = session.run(None, ort_inputs)[0]
            if output.ndim == 4 and output.shape[1] in {1, 3}:
                output = np.transpose(output[0], (1, 2, 0))
            elif output.ndim == 4:
                output = output[0]
            output = np.clip(output, 0.0, 1.0)
            if output.shape[2] == 1:
                output = np.repeat(output, 3, axis=2)
            output = (output[:h, :w] * 255.0).astype(np.uint8)
            out_img = Image.fromarray(output, mode="RGB")
            if out_img.size != (orig_w, orig_h):
                out_img = out_img.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
            return out_img
        except Exception:
            return None


_BACKGROUND_INPAINTER = None


def get_background_inpainter():
    global _BACKGROUND_INPAINTER
    if _BACKGROUND_INPAINTER is None:
        _BACKGROUND_INPAINTER = BackgroundInpainter()
    return _BACKGROUND_INPAINTER
