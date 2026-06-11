from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


_DEFAULT_MODEL_ALIASES = {
    "smolvlm-500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "smolvlm-256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "smoldocling": "ds4sd/SmolDocling-256M-preview",
    "florence2-base": "microsoft/Florence-2-base",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
}


def _extract_json(text: str) -> dict:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


class VisualDocumentLayoutModel:
    """Small local VLM wrapper for publication-layout visual critique.

    The wrapper is deliberately optional. If no local VLM is available, callers
    receive an empty result and fall back to deterministic QA.
    """

    def __init__(self, model_id: str | None = None, device: str | None = None) -> None:
        self.model_id = self._resolve_model_id(model_id or os.environ.get("VISUAL_LAYOUT_MODEL_ID") or "smolvlm-500m")
        self.device = device or os.environ.get("VISUAL_LAYOUT_DEVICE", "cpu")
        self.max_new_tokens = max(64, min(700, int(os.environ.get("VISUAL_LAYOUT_MAX_NEW_TOKENS", "220"))))
        self.max_image_side = max(384, min(1600, int(os.environ.get("VISUAL_LAYOUT_MAX_IMAGE_SIDE", "900"))))
        self._loaded = False
        self._available = False
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    @staticmethod
    def _resolve_model_id(model_id: str) -> str:
        raw = str(model_id or "").strip()
        if not raw:
            return raw
        alias = _DEFAULT_MODEL_ALIASES.get(raw.lower(), raw)
        base = Path(__file__).resolve().parent / "ai_models" / "visual_layout" / raw
        if base.exists():
            return str(base)
        return alias

    def is_available(self) -> bool:
        if not self._loaded:
            self.load()
        return self._available

    def load(self) -> bool:
        if self._loaded:
            return self._available
        self._loaded = True
        try:
            import torch
            from transformers import AutoProcessor

            self._torch = torch
            self._processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=True, trust_remote_code=True)
            self._model = self._load_model_class()
            if self._model is None:
                self._available = False
                return False
            self._model.eval()
            self._available = True
            return True
        except Exception:
            self._available = False
            return False

    def _load_model_class(self) -> Any:
        from transformers import AutoModelForCausalLM

        model_classes = []
        try:
            from transformers import AutoModelForImageTextToText

            model_classes.append(AutoModelForImageTextToText)
        except Exception:
            pass
        try:
            from transformers import AutoModelForVision2Seq

            model_classes.append(AutoModelForVision2Seq)
        except Exception:
            pass
        model_classes.append(AutoModelForCausalLM)

        last_exc = None
        for cls in model_classes:
            try:
                return cls.from_pretrained(
                    self.model_id,
                    local_files_only=True,
                    trust_remote_code=True,
                    torch_dtype=getattr(self._torch, "float32", None),
                ).to(self.device)
            except Exception as exc:
                last_exc = exc
        if last_exc:
            return None
        return None

    def _make_panel(self, original_image_path: str | None, rendered_image: Image.Image) -> Image.Image:
        rendered = rendered_image.convert("RGB")
        if original_image_path and os.path.exists(str(original_image_path)):
            original = Image.open(str(original_image_path)).convert("RGB")
        else:
            original = Image.new("RGB", rendered.size, "white")
        original.thumbnail((self.max_image_side, self.max_image_side), Image.Resampling.LANCZOS)
        rendered.thumbnail((self.max_image_side, self.max_image_side), Image.Resampling.LANCZOS)
        w = original.width + rendered.width
        h = max(original.height, rendered.height) + 28
        panel = Image.new("RGB", (w, h), "white")
        panel.paste(original, (0, 28))
        panel.paste(rendered, (original.width, 28))
        draw = ImageDraw.Draw(panel)
        draw.text((6, 6), "ORIGINAL", fill=(0, 0, 0))
        draw.text((original.width + 6, 6), "RENDERED", fill=(0, 0, 0))
        return panel

    def analyze(self, *, original_image_path: str | None, rendered_image: Image.Image, layout_payload: dict) -> dict:
        if not self.is_available():
            return {"available": False, "faults": [], "placements": [], "warnings": ["visual_layout_model_unavailable"]}
        panel = self._make_panel(original_image_path, rendered_image)
        prompt = (
            "Compare ORIGINAL and RENDERED PDF page images. Return JSON only with: "
            '{"faults":[{"type":"overlap|missing_text|glued_words|formula_damaged|english_leak|bad_order","bbox":[x0,y0,x1,y1],"block_id":"...","severity":"blocking","reason":"..."}],'
            '"placements":[{"block_id":"...","bbox":[x0,y0,x1,y1],"reason":"..."}],"warnings":[]}. '
            "Use PDF point coordinates from the supplied layout JSON. Fix only publication-readiness errors: no overlaps, no missing text, preserve formulas/images, preserve reading order. "
            f"Layout JSON: {json.dumps(layout_payload, ensure_ascii=False, separators=(',', ':'))[:5000]}"
        )
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": panel},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            if hasattr(self._processor, "apply_chat_template"):
                text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self._processor(text=text, images=panel, return_tensors="pt").to(self.device)
            else:
                inputs = self._processor(images=panel, text=prompt, return_tensors="pt").to(self.device)
            with self._torch.inference_mode():
                generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
            if input_ids is not None and getattr(generated, "shape", None) is not None and generated.shape[-1] > input_ids.shape[-1]:
                generated = generated[:, input_ids.shape[-1]:]
            decoded = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
            parsed = _extract_json(decoded)
            faults = parsed.get("faults") if isinstance(parsed.get("faults"), list) else []
            placements = parsed.get("placements") if isinstance(parsed.get("placements"), list) else []
            warnings = parsed.get("warnings") if isinstance(parsed.get("warnings"), list) else []
            return {
                "available": True,
                "faults": faults[:30],
                "placements": placements[:30],
                "warnings": [str(w)[:160] for w in warnings[:10]],
                "raw_preview": decoded[:600],
            }
        except Exception as exc:
            return {"available": False, "faults": [], "placements": [], "warnings": [f"visual_layout_model_error: {exc}"]}
