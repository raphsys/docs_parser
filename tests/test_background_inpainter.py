import tempfile
import unittest
from pathlib import Path
import sys
import types

import fitz
from PIL import Image, ImageDraw

from background_inpainter import BackgroundInpainter
from reconstructor import DocumentReconstructor


class _FakeInpainter:
    def __init__(self, overlay_path):
        self.enabled = True
        self.overlay_path = overlay_path
        self.calls = []

    def save_inpaint_overlay(self, source_image_path, crop_bbox, mask_rects, out_dir, kind="background_inpaint"):
        self.calls.append(
            {
                "source_image_path": source_image_path,
                "crop_bbox": crop_bbox,
                "mask_rects": mask_rects,
                "out_dir": out_dir,
                "kind": kind,
            }
        )
        return {
            "path": self.overlay_path,
            "bbox": crop_bbox,
            "kind": kind,
            "backend": "fake",
        }


class BackgroundInpainterTests(unittest.TestCase):
    def test_opencv_backend_restores_masked_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = Path(tmp) / "src.png"
            img = Image.new("RGB", (80, 60), "#f4f0d0")
            draw = ImageDraw.Draw(img)
            draw.rectangle([20, 18, 56, 36], fill="#101010")
            img.save(src_path)
            inpainter = BackgroundInpainter(enabled=True, backend="opencv", models_root=tmp)
            ov = inpainter.save_inpaint_overlay(
                source_image_path=str(src_path),
                crop_bbox=[0, 0, 80, 60],
                mask_rects=[[20, 18, 56, 36]],
                out_dir=tmp,
                kind="unit_test",
            )
            self.assertIsNotNone(ov)
            self.assertTrue(Path(ov["path"]).exists())
            restored = Image.open(ov["path"]).convert("RGB")
            center_px = restored.getpixel((38, 27))
            self.assertNotEqual(center_px, (16, 16, 16))

    def test_prepare_visual_group_background_uses_group_text_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = Path(tmp) / "src.png"
            overlay_path = Path(tmp) / "overlay.png"
            Image.new("RGB", (200, 160), "#dde7f2").save(src_path)
            Image.new("RGB", (200, 160), "#dde7f2").save(overlay_path)

            recon = DocumentReconstructor()
            recon.background_inpainter = _FakeInpainter(str(overlay_path))
            recon._restored_background_rects = {}
            recon._prepared_visual_groups = {}

            page_data = {
                "source_image_path": str(src_path),
                "background_path": str(Path(tmp) / "bg.png"),
            }
            item_a = {
                "role": "header",
                "bbox": fitz.Rect(20, 10, 90, 24),
                "descriptor_group_render_mode": "annotation_group",
                "descriptor_group_ids": {"annotation_group_id": "g1"},
                "page_data": page_data,
            }
            item_b = {
                "role": "title",
                "bbox": fitz.Rect(36, 32, 100, 46),
                "descriptor_group_render_mode": "annotation_group",
                "descriptor_group_ids": {"annotation_group_id": "g1"},
                "page_data": page_data,
            }
            doc = fitz.open()
            page = doc.new_page(width=120, height=100)
            ok = recon._prepare_visual_group_background(
                page,
                item_a,
                fitz.Rect(0, 0, 180, 120),
                group_items=[item_a, item_b],
            )
            self.assertTrue(ok)
            self.assertEqual(len(recon.background_inpainter.calls), 1)
            call = recon.background_inpainter.calls[0]
            self.assertEqual(len(call["mask_rects"]), 2)
            self.assertEqual(call["kind"], "annotation_group_group_bg_restore")
            doc.close()

    def test_lama_backend_skips_invalid_first_model_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_root = Path(tmp)
            lama_dir = models_root / "lama"
            lama_dir.mkdir(parents=True, exist_ok=True)
            (lama_dir / "lama_fp32.onnx").write_bytes(b"broken")
            (lama_dir / "model.onnx").write_bytes(b"valid")

            calls = []

            class _FakeSession:
                def get_inputs(self):
                    return []

            class _FakeORTModule:
                @staticmethod
                def InferenceSession(path, providers=None):
                    calls.append((Path(path).name, tuple(providers or ())))
                    if str(path).endswith("lama_fp32.onnx"):
                        raise RuntimeError("invalid model")
                    return _FakeSession()

            previous_ort = sys.modules.get("onnxruntime")
            sys.modules["onnxruntime"] = types.SimpleNamespace(
                InferenceSession=_FakeORTModule.InferenceSession
            )
            try:
                inpainter = BackgroundInpainter(enabled=True, backend="lama_onnx", models_root=tmp)
                status = inpainter.status()
            finally:
                if previous_ort is None:
                    sys.modules.pop("onnxruntime", None)
                else:
                    sys.modules["onnxruntime"] = previous_ort

            self.assertEqual(status["backend"], "lama_onnx")
            self.assertTrue(status["model_path"].endswith("lama/model.onnx"))
            self.assertEqual(
                calls,
                [
                    ("lama_fp32.onnx", ("CPUExecutionProvider",)),
                    ("model.onnx", ("CPUExecutionProvider",)),
                ],
            )

    def test_auto_backend_prefers_opencv_for_dense_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_path = Path(tmp) / "src.png"
            img = Image.new("RGB", (220, 140), "#d9d2b6")
            draw = ImageDraw.Draw(img)
            draw.rectangle([20, 18, 90, 44], fill="#101010")
            draw.rectangle([100, 18, 170, 44], fill="#101010")
            draw.rectangle([20, 52, 90, 78], fill="#101010")
            draw.rectangle([100, 52, 170, 78], fill="#101010")
            draw.rectangle([20, 86, 90, 112], fill="#101010")
            img.save(src_path)

            inpainter = BackgroundInpainter(enabled=True, backend="auto", models_root=tmp)
            inpainter._ensure_session = lambda: object()
            calls = []
            inpainter._inpaint_crop_lama = lambda crop, mask: calls.append("lama") or crop
            inpainter._inpaint_crop_opencv = lambda crop, mask: calls.append("opencv") or crop

            ov = inpainter.save_inpaint_overlay(
                source_image_path=str(src_path),
                crop_bbox=[0, 0, 220, 140],
                mask_rects=[
                    [20, 18, 90, 44],
                    [100, 18, 170, 44],
                    [20, 52, 90, 78],
                    [100, 52, 170, 78],
                    [20, 86, 90, 112],
                ],
                out_dir=tmp,
                kind="unit_test_dense_mask",
            )
            self.assertIsNotNone(ov)
            self.assertEqual(calls, ["opencv"])
            self.assertEqual(inpainter._ready_backend, "opencv")


if __name__ == "__main__":
    unittest.main()
