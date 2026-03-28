import tempfile
import unittest
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

from publication_qa import PIXEL_TO_POINT, _evaluate_visual_annotation_fidelity, publication_qa


def _pt_rect_to_px(rect):
    return [v / PIXEL_TO_POINT for v in (rect.x0, rect.y0, rect.x1, rect.y1)]


class PublicationQATests(unittest.TestCase):
    def _build_pdf_with_image_and_text(self):
        tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(tmp.name)
        source_img_path = tmp_path / "source.png"
        pdf_path = tmp_path / "sample.pdf"

        canvas = Image.new("RGB", (417, 417), "white")
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([83, 83, 333, 333], fill="#c9d2dc")
        canvas.save(source_img_path)

        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        page.insert_image(fitz.Rect(40, 40, 160, 160), filename=str(source_img_path))
        page.insert_text((60, 82), "Label", fontsize=12)
        doc.save(pdf_path)
        doc.close()
        return tmp, source_img_path, pdf_path

    def test_visual_annotation_collision_is_not_blocking(self):
        tmp, _, pdf_path = self._build_pdf_with_image_and_text()
        try:
            text_rect = fitz.Rect(57, 70, 92, 86)
            translated_pages = [
                {
                    "page": 0,
                    "blocks": [
                        {
                            "role": "title",
                            "bbox": _pt_rect_to_px(text_rect),
                            "descriptor_band_role": "annotation_band",
                            "descriptor_group_render_mode": "annotation_group",
                            "descriptor_structural_role": "diagram_label",
                            "translated_text": "Étiquette",
                            "lines": [],
                        }
                    ],
                }
            ]
            qa = publication_qa(
                source_pages=[],
                translated_pages=translated_pages,
                pdf_path=str(pdf_path),
                coverage_report={
                    "summary": {"coverage_score": 1.0, "missing_units": 0, "warning_units": 0},
                    "rendered_text_report": {"summary": {"rendered_coverage_score": 1.0, "rendered_missing_units": 0, "rendered_warning_units": 0}},
                },
                target_lang="fr",
                original_image_paths=[],
            )
            self.assertTrue(qa["publication_ready"])
            self.assertEqual(qa["layout_metrics"]["text_img_collisions"], 0)
            self.assertGreaterEqual(qa["layout_metrics"]["ignored_visual_annotation_text_img_collisions"], 1)
        finally:
            tmp.cleanup()

    def test_regular_text_collision_remains_blocking(self):
        tmp, _, pdf_path = self._build_pdf_with_image_and_text()
        try:
            text_rect = fitz.Rect(57, 70, 92, 86)
            translated_pages = [
                {
                    "page": 0,
                    "blocks": [
                        {
                            "role": "body",
                            "bbox": _pt_rect_to_px(text_rect),
                            "translated_text": "Étiquette",
                            "lines": [],
                        }
                    ],
                }
            ]
            qa = publication_qa(
                source_pages=[],
                translated_pages=translated_pages,
                pdf_path=str(pdf_path),
                coverage_report={
                    "summary": {"coverage_score": 1.0, "missing_units": 0, "warning_units": 0},
                    "rendered_text_report": {"summary": {"rendered_coverage_score": 1.0, "rendered_missing_units": 0, "rendered_warning_units": 0}},
                },
                target_lang="fr",
                original_image_paths=[],
            )
            self.assertFalse(qa["publication_ready"])
            self.assertIn("text_image_collision_detected", qa["blocking_reasons"])
            self.assertGreaterEqual(qa["layout_metrics"]["text_img_collisions"], 1)
        finally:
            tmp.cleanup()

    def test_visual_annotation_fidelity_metric_is_reported(self):
        tmp, source_img_path, pdf_path = self._build_pdf_with_image_and_text()
        try:
            text_rect = fitz.Rect(57, 70, 92, 86)
            translated_pages = [
                {
                    "page": 0,
                    "blocks": [
                        {
                            "role": "diagram_text_label",
                            "bbox": _pt_rect_to_px(text_rect),
                            "descriptor_band_role": "annotation_band",
                            "descriptor_group_render_mode": "annotation_group",
                            "descriptor_structural_role": "diagram_label",
                            "translated_text": "Étiquette",
                            "lines": [],
                        }
                    ],
                }
            ]
            metrics = _evaluate_visual_annotation_fidelity(
                [str(source_img_path)],
                translated_pages,
                str(pdf_path),
                dpi=150,
            )
            self.assertEqual(metrics["region_count"], 1)
            self.assertEqual(metrics["pages_evaluated"], 1)
            self.assertIsNotNone(metrics["background_similarity_score"])
            self.assertGreater(metrics["background_similarity_score"], 0.5)
        finally:
            tmp.cleanup()

    def test_locked_equation_overlay_collision_is_not_blocking(self):
        tmp, _, pdf_path = self._build_pdf_with_image_and_text()
        try:
            text_rect = fitz.Rect(57, 70, 92, 86)
            translated_pages = [
                {
                    "page": 0,
                    "blocks": [
                        {
                            "role": "equation_inline",
                            "bbox": _pt_rect_to_px(text_rect),
                            "render_mode": "background_only",
                            "translated_text": "dE / dw",
                            "lines": [],
                        }
                    ],
                }
            ]
            qa = publication_qa(
                source_pages=[],
                translated_pages=translated_pages,
                pdf_path=str(pdf_path),
                coverage_report={
                    "summary": {"coverage_score": 1.0, "missing_units": 0, "warning_units": 0},
                    "rendered_text_report": {"summary": {"rendered_coverage_score": 1.0, "rendered_missing_units": 0, "rendered_warning_units": 0}},
                },
                target_lang="fr",
                original_image_paths=[],
            )
            self.assertTrue(qa["publication_ready"])
            self.assertEqual(qa["layout_metrics"]["text_img_collisions"], 0)
            self.assertGreaterEqual(qa["layout_metrics"]["ignored_locked_equation_text_img_collisions"], 1)
        finally:
            tmp.cleanup()

    def test_locked_equation_image_does_not_block_wider_text_line_bbox(self):
        tmp, _, pdf_path = self._build_pdf_with_image_and_text()
        try:
            equation_rect = fitz.Rect(57, 70, 92, 86)
            wider_line_rect = fitz.Rect(45, 65, 145, 90)
            translated_pages = [
                {
                    "page": 0,
                    "blocks": [
                        {
                            "role": "equation_inline",
                            "bbox": _pt_rect_to_px(equation_rect),
                            "render_mode": "background_only",
                            "translated_text": "dE / dw",
                            "lines": [],
                        }
                    ],
                }
            ]
            qa = publication_qa(
                source_pages=[],
                translated_pages=translated_pages,
                pdf_path=str(pdf_path),
                coverage_report={
                    "summary": {"coverage_score": 1.0, "missing_units": 0, "warning_units": 0},
                    "rendered_text_report": {"summary": {"rendered_coverage_score": 1.0, "rendered_missing_units": 0, "rendered_warning_units": 0}},
                },
                target_lang="fr",
                original_image_paths=[],
            )
            self.assertEqual(qa["layout_metrics"]["text_img_collisions"], 0)
            self.assertGreaterEqual(qa["layout_metrics"]["ignored_locked_equation_text_img_collisions"], 1)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
