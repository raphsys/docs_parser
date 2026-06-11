"""PageRenderer — unité 2 du pipeline cible.

Responsabilités :
    - rend la page en image ;
    - garde DPI, scale (px par pt), dimensions points et pixels.

Le rendu pixels sert uniquement à l'OCR, aux masques et au debug.
Le modèle documentaire reste en points.
"""

from __future__ import annotations

DEFAULT_TARGET_DPI = 150


class PageRenderer:
    def __init__(self, *, target_dpi: int = DEFAULT_TARGET_DPI):
        self.target_dpi = target_dpi

    def render_pdf_page(self, pdf_page) -> dict:
        from PIL import Image

        pix = pdf_page.get_pixmap(dpi=self.target_dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        width_pt = float(pdf_page.rect.width)
        height_pt = float(pdf_page.rect.height)
        sx = img.width / max(1e-9, width_pt)
        sy = img.height / max(1e-9, height_pt)
        return {
            "image": img,
            "dimensions": {
                "width": img.width,
                "height": img.height,
                "width_pt": round(width_pt, 3),
                "height_pt": round(height_pt, 3),
                "render_dpi": self.target_dpi,
                "scale_x_px_per_pt": round(sx, 6),
                "scale_y_px_per_pt": round(sy, 6),
            },
            "sx": sx,
            "sy": sy,
        }

    def load_image(self, image_path: str) -> dict:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        scale = self.target_dpi / 72.0
        return {
            "image": img,
            "dimensions": {
                "width": img.width,
                "height": img.height,
                "render_dpi": self.target_dpi,
                "scale_x_px_per_pt": round(scale, 6),
                "scale_y_px_per_pt": round(scale, 6),
            },
            "sx": scale,
            "sy": scale,
        }
