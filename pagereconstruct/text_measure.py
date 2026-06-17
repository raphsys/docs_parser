"""text_measure — mesure de texte canonique (extraite de reconstructor.py /
renderers.base). Pure: pas de dessin. Reprend `_measure_text(_width)` legacy.

Une seule porte de mesure pour tout pagereconstruct : largeur de ligne, wrap,
ajustement de taille à la boîte. Les renderers et le PlacementSolver l'utilisent.
"""

from __future__ import annotations

from .renderers.base import layout_text as _layout_text, font_path as _font_path

# API stable
font_path = _font_path


def measure_block(text: str, bbox_pt, style: dict, *, align: str = "left",
                  min_ratio: float = 0.86, allow_lines=None) -> dict:
    """Mesure `text` dans `bbox_pt` (espace pt). Renvoie font/size/lines/line_boxes
    /overflow + text_bbox englobant. Aucune décision de rôle, aucun dessin."""
    lay = _layout_text(text, [float(x) for x in bbox_pt], dict(style or {}),
                       align=align, min_ratio=min_ratio, allow_lines=allow_lines)
    boxes = lay.get("line_boxes") or []
    lay["text_bbox"] = ([min(b[0] for b in boxes), min(b[1] for b in boxes),
                         max(b[2] for b in boxes), max(b[3] for b in boxes)] if boxes else None)
    return lay


def text_width(text: str, font_size_pt: float, style: dict | None = None) -> float:
    """Largeur d'une chaîne à une taille donnée (pt). Legacy _measure_text_width."""
    from PIL import ImageDraw, ImageFont, Image
    fp = _font_path(style or {})
    font = ImageFont.truetype(fp, max(1, int(round(font_size_pt))))
    return ImageDraw.Draw(Image.new("RGB", (4, 4))).textlength(str(text or ""), font=font)
