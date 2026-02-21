import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image


DEFAULT_MODEL_PATH = os.getenv("FONT_AI_MODEL_PATH", "./ai_models/fonts/teacher/model.onnx")
DEFAULT_CONFIG_PATH = os.getenv("FONT_AI_CONFIG_PATH", "./ai_models/fonts/teacher/model_config.yaml")
DEFAULT_MAPPING_PATH = os.getenv("FONT_AI_MAPPING_PATH", "./ai_models/fonts/teacher/fonts_mapping.yaml")
DEFAULT_FONT_DIRS = (
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
)


@dataclass
class FontMatch:
    font_name: str
    font_path: str
    score: float
    flags: Dict[str, bool]


class FontAIMatcher:
    """
    Storia font classifier based matcher.
    - No local training required.
    - No embedding index required.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        config_path: str = DEFAULT_CONFIG_PATH,
        mapping_path: str = DEFAULT_MAPPING_PATH,
        font_dirs: Tuple[str, ...] = DEFAULT_FONT_DIRS,
        index_path: str = "",
        input_size: int = 320,
        top_k: int = 1,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.mapping_path = mapping_path
        self.font_dirs = font_dirs
        self.index_path = index_path
        self.input_size = input_size
        self.top_k = top_k

        self._session = None
        self._input_name = None
        self._output_name = None
        self._input_shape = None
        self._ready = False

        self._class_names: List[str] = []
        self._mapping: Dict[str, str] = {}
        self._local_font_catalog: Dict[str, str] = {}

        # Compatibility with existing smoke logs.
        self._index_names: List[str] = []
        self._index_paths: List[str] = []

        self._try_initialize()

    def is_ready(self) -> bool:
        return self._ready

    def has_model(self) -> bool:
        return self._session is not None

    def rebuild_index(self, max_fonts: int = 0) -> None:
        # Kept for compatibility with old scripts.
        self._build_local_font_catalog()

    def _try_initialize(self) -> None:
        if not os.path.exists(self.model_path):
            print(f"[FontAI] Storia model not found: {self.model_path}")
            return

        try:
            self._session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            self._input_shape = self._session.get_inputs()[0].shape
        except Exception as exc:
            print(f"[FontAI] Failed to load Storia ONNX model: {exc}")
            return

        self._class_names = self._load_class_names(self.config_path)
        self._mapping = self._load_mapping(self.mapping_path)
        if not self._class_names and self._mapping:
            self._class_names = list(self._mapping.keys())

        if not self._class_names:
            print(f"[FontAI] No class names loaded from {self.config_path}")
            return

        self._build_local_font_catalog()
        self._index_names = list(self._class_names)
        self._index_paths = [self._resolve_local_font_path(x) for x in self._class_names]

        self._ready = True
        print(
            f"[FontAI] Storia ready | classes={len(self._class_names)} "
            f"| local_fonts={len(self._local_font_catalog)}"
        )

    def _normalize_key(self, s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    def _load_class_names(self, path: str) -> List[str]:
        if not path or not os.path.exists(path):
            return []
        names: List[str] = []
        in_block = False
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.rstrip("\n")
                stripped = line.strip()
                if stripped == "classnames:":
                    in_block = True
                    continue
                if not in_block:
                    continue
                if stripped.startswith("- "):
                    names.append(stripped[2:].strip())
                    continue
                if stripped and not stripped.startswith("#"):
                    break
        return names

    def _load_mapping(self, path: str) -> Dict[str, str]:
        if not path or not os.path.exists(path):
            return {}
        out: Dict[str, str] = {}
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip().strip("\"'")
        return out

    def _build_local_font_catalog(self) -> None:
        catalog: Dict[str, str] = {}
        for root in self.font_dirs:
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    low = fn.lower()
                    if not (low.endswith(".ttf") or low.endswith(".otf")):
                        continue
                    full = os.path.join(dirpath, fn)
                    stem = os.path.splitext(fn)[0]
                    n1 = self._normalize_key(fn)
                    n2 = self._normalize_key(stem)
                    if n1 and n1 not in catalog:
                        catalog[n1] = full
                    if n2 and n2 not in catalog:
                        catalog[n2] = full
        self._local_font_catalog = catalog

    def _resolve_local_font_path(self, class_name: str) -> str:
        candidates = []
        mapped = self._mapping.get(class_name)
        if mapped:
            candidates.append(mapped)
            candidates.append(os.path.splitext(mapped)[0])
        candidates.append(class_name)

        for c in candidates:
            key = self._normalize_key(c)
            if key in self._local_font_catalog:
                return self._local_font_catalog[key]

        # Fallback: partial token match.
        class_key = self._normalize_key(class_name)
        for key, path in self._local_font_catalog.items():
            if class_key and (class_key in key or key in class_key):
                return path
        return ""

    def _infer_flags_from_name(self, font_name: str) -> Dict[str, bool]:
        n = font_name.lower()
        return {
            "bold": any(t in n for t in ("bold", "black", "heavy", "semibold", "demi")),
            "italic": any(t in n for t in ("italic", "oblique")),
            "serif": any(t in n for t in ("serif", "times", "garamond", "georgia", "roman")),
            "monospace": any(t in n for t in ("mono", "courier", "consolas", "menlo", "code")),
        }

    def _target_size(self) -> Tuple[int, int, str]:
        h = self.input_size
        w = self.input_size
        layout = "nchw"
        shape = self._input_shape
        if len(shape) == 4:
            if isinstance(shape[2], (int, np.integer)) and isinstance(shape[3], (int, np.integer)):
                h = int(shape[2])
                w = int(shape[3])
                layout = "nchw"
            elif isinstance(shape[1], (int, np.integer)) and isinstance(shape[2], (int, np.integer)):
                h = int(shape[1])
                w = int(shape[2])
                layout = "nhwc"
        return h, w, layout

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        gray = img.convert("L")
        arr = np.array(gray)
        ys, xs = np.where(arr < 245)
        if len(xs) > 0 and len(ys) > 0:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            gray = gray.crop((x0, y0, x1 + 1, y1 + 1))

        h, w, layout = self._target_size()
        iw, ih = gray.size
        scale = min(w / max(iw, 1), h / max(ih, 1))
        nw = max(1, int(round(iw * scale)))
        nh = max(1, int(round(ih * scale)))
        rs = gray.resize((nw, nh), Image.Resampling.BILINEAR)

        canvas = Image.new("L", (w, h), color=255)
        ox = (w - nw) // 2
        oy = (h - nh) // 2
        canvas.paste(rs, (ox, oy))

        x = np.array(canvas).astype(np.float32) / 255.0
        x3 = np.stack([x, x, x], axis=0)
        if layout == "nchw":
            return x3.reshape(1, 3, h, w).astype(np.float32)
        x3_hw = np.transpose(x3, (1, 2, 0))
        return x3_hw.reshape(1, h, w, 3).astype(np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        ez = np.exp(z)
        return ez / np.clip(np.sum(ez), 1e-9, None)

    def match_crop(self, crop: Image.Image) -> Optional[FontMatch]:
        if not self._ready:
            return None
        try:
            x = self._preprocess(crop)
            out = self._session.run([self._output_name], {self._input_name: x})[0]
            logits = np.array(out, dtype=np.float32).reshape(-1)
            probs = self._softmax(logits)
            idx = int(np.argmax(probs))
            score = float(probs[idx])
        except Exception:
            return None

        name = self._class_names[idx] if idx < len(self._class_names) else f"class_{idx}"
        path = self._resolve_local_font_path(name)
        flags = self._infer_flags_from_name(name)
        return FontMatch(font_name=name, font_path=path, score=score, flags=flags)
