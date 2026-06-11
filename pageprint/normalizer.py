"""PAGEPRINT Normalizer — normalise unités, points, styles.

Règle d'architecture :
    Modèle documentaire interne  → points PDF
    Rendu image / OCR / masque   → pixels
    Conversion                   → seulement aux frontières

Contrat imposé :
    bbox = points, bbox_unit = "pt", bbox_px = optionnel pour debug/rendu.

Responsabilité : convertir, nettoyer, borner, typer et annoter toutes les unités.
"""

from __future__ import annotations

from .schema import CANONICAL_UNIT, POINTS_PER_INCH

DEFAULT_RENDER_DPI = 150


def scale_from_dimensions(dimensions: dict | None, *, default_dpi: float = DEFAULT_RENDER_DPI):
    """Déduit (sx, sy) = pixels par point depuis les dimensions de page.

    Ordre de résolution :
    1. scale_x_px_per_pt / scale_y_px_per_pt explicites ;
    2. width_pt/height_pt + width/height pixels (legacy) ;
    3. unit == "pt" + render_width_px/render_height_px ;
    4. render_dpi (ou default_dpi) → dpi / 72.
    """
    dims = dimensions or {}
    sx = dims.get("scale_x_px_per_pt")
    sy = dims.get("scale_y_px_per_pt")
    if sx and sy:
        return float(sx), float(sy)

    width = dims.get("width")
    height = dims.get("height")
    width_pt = dims.get("width_pt")
    height_pt = dims.get("height_pt")
    unit = dims.get("unit")

    if unit == CANONICAL_UNIT:
        # width/height déjà en points : l'échelle relie le rendu pixels.
        render_w = dims.get("render_width_px")
        render_h = dims.get("render_height_px")
        if width and height and render_w and render_h:
            return float(render_w) / float(width), float(render_h) / float(height)
    elif width_pt and height_pt and width and height:
        # Legacy : width/height en pixels, width_pt/height_pt en points.
        return float(width) / float(width_pt), float(height) / float(height_pt)

    dpi = float(dims.get("render_dpi") or dims.get("dpi") or default_dpi)
    scale = dpi / POINTS_PER_INCH
    return scale, scale


def bbox_px_to_pt(bbox, sx: float, sy: float):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        return [
            round(x0 / max(1e-9, sx), 3),
            round(y0 / max(1e-9, sy), 3),
            round(x1 / max(1e-9, sx), 3),
            round(y1 / max(1e-9, sy), 3),
        ]
    except Exception:
        return None


def bbox_pt_to_px(bbox, sx: float, sy: float):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        return [
            round(x0 * sx, 3),
            round(y0 * sy, 3),
            round(x1 * sx, 3),
            round(y1 * sy, 3),
        ]
    except Exception:
        return None


def annotate_bbox_units(node: dict, *, bbox_pt=None, bbox_px=None,
                        sx: float = 1.0, sy: float = 1.0,
                        dpi: float = DEFAULT_RENDER_DPI, source: str | None = None) -> dict:
    """Annote un nœud avec bbox canonique en points + bbox_px de debug."""
    if not isinstance(node, dict):
        return node

    if bbox_pt is None and bbox_px is not None:
        bbox_pt = bbox_px_to_pt(bbox_px, sx, sy)
    if bbox_px is None and bbox_pt is not None:
        bbox_px = bbox_pt_to_px(bbox_pt, sx, sy)

    if bbox_pt is not None:
        node["bbox"] = bbox_pt
        node["bbox_unit"] = CANONICAL_UNIT
        node["bbox_origin"] = "top_left"
    if bbox_px is not None:
        node["bbox_px"] = bbox_px
        node["bbox_px_dpi"] = dpi
    if source:
        node["bbox_source"] = source
    return node


def normalize_bbox_to_pt(bbox, unit: str | None, sx: float, sy: float):
    """Retourne (bbox_pt, bbox_px) quel que soit le référentiel d'origine."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None, None
    if unit in {CANONICAL_UNIT, "pdf_pt"}:
        bbox_pt = [round(float(v), 3) for v in bbox]
        return bbox_pt, bbox_pt_to_px(bbox_pt, sx, sy)
    bbox_px = [round(float(v), 3) for v in bbox]
    return bbox_px_to_pt(bbox_px, sx, sy), bbox_px


def normalize_color(color) -> str | None:
    """Normalise une couleur vers le format hex #rrggbb."""
    if color is None:
        return None
    if isinstance(color, str):
        value = color.strip()
        if value.startswith("#"):
            if len(value) == 7:
                return value.lower()
            if len(value) == 4:  # #rgb
                return ("#" + "".join(ch * 2 for ch in value[1:])).lower()
        return value.lower() or None
    if isinstance(color, int):
        return "#{:06x}".format(color & 0xFFFFFF)
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            vals = [float(v) for v in color[:3]]
            if all(0.0 <= v <= 1.0 for v in vals):
                vals = [v * 255.0 for v in vals]
            return "#{:02x}{:02x}{:02x}".format(
                *(max(0, min(255, int(round(v)))) for v in vals)
            )
        except Exception:
            return None
    return None


def clamp_confidence(value) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return None


def normalize_style(style: dict | None, *, sx: float = 1.0,
                    source: str | None = None) -> dict:
    """Normalise un style vers le contrat canonique.

    Règle : font_size_pt = font_size_px / sx si le style vient d'une image OCR.
    Pour un PDF natif, conserver directement la taille PDF en points.
    """
    style = style or {}
    out = {
        "font_family": style.get("font_family") or style.get("font") or None,
        "font_size_pt": None,
        "font_size_px": None,
        "font_size_unit": CANONICAL_UNIT,
        "color": normalize_color(style.get("color")),
        "fill_color": normalize_color(style.get("fill_color")) or normalize_color(style.get("color")),
        "background_color": normalize_color(style.get("background_color")),
        "flags": {
            "bold": bool(style.get("bold") or (style.get("flags") or {}).get("bold")),
            "italic": bool(style.get("italic") or (style.get("flags") or {}).get("italic")),
            "underline": bool(style.get("underline") or (style.get("flags") or {}).get("underline")),
            "serif": bool(style.get("serif") or (style.get("flags") or {}).get("serif")),
            "monospace": bool(style.get("monospace") or (style.get("flags") or {}).get("monospace")),
            "uppercase": bool(style.get("uppercase") or (style.get("flags") or {}).get("uppercase")),
        },
    }

    size_pt = style.get("font_size_pt") or style.get("size_pt")
    size_px = style.get("font_size_px") or style.get("size_px")
    raw_size = style.get("size")

    if size_pt is None and raw_size is not None:
        if source == "native_pdf" or style.get("size_unit") in {CANONICAL_UNIT, "pdf_pt"}:
            size_pt = raw_size
        else:
            size_px = size_px if size_px is not None else raw_size

    try:
        if size_pt is not None:
            out["font_size_pt"] = round(float(size_pt), 2)
            if size_px is None and sx:
                size_px = float(size_pt) * sx
        if size_px is not None:
            out["font_size_px"] = round(float(size_px), 2)
            if out["font_size_pt"] is None and sx:
                out["font_size_pt"] = round(float(size_px) / max(1e-9, sx), 2)
    except Exception:
        pass

    return out


def normalize_structure_bboxes_to_pt(node, sx: float, sy: float,
                                     dpi: float = DEFAULT_RENDER_DPI):
    """Traverse récursivement une structure et convertit toutes les bbox en points.

    Les bbox encore en pixels sont converties ; les bbox déjà en points sont
    annotées et reçoivent leur projection bbox_px (debug/rendu).
    """
    if isinstance(node, dict):
        bbox = node.get("bbox")
        unit = node.get("bbox_unit") or node.get("bbox_source_unit")
        if bbox is not None:
            if unit not in {CANONICAL_UNIT, "pdf_pt"}:
                bbox_px = list(bbox)
                annotate_bbox_units(node, bbox_px=bbox_px, sx=sx, sy=sy, dpi=dpi)
            else:
                node["bbox_unit"] = CANONICAL_UNIT
                node["bbox_origin"] = "top_left"
                node.setdefault("bbox_px", bbox_pt_to_px(bbox, sx, sy))
                node.setdefault("bbox_px_dpi", dpi)
        for value in node.values():
            normalize_structure_bboxes_to_pt(value, sx, sy, dpi)
    elif isinstance(node, list):
        for item in node:
            normalize_structure_bboxes_to_pt(item, sx, sy, dpi)
    return node


def normalize_page_geometry(dimensions: dict | None, *,
                            default_dpi: float = DEFAULT_RENDER_DPI) -> dict:
    """Produit la géométrie de page canonique (points en premier, pixels secondaires)."""
    dims = dimensions or {}
    sx, sy = scale_from_dimensions(dims, default_dpi=default_dpi)
    dpi = float(dims.get("render_dpi") or dims.get("dpi") or default_dpi)

    unit = dims.get("unit")
    width = dims.get("width")
    height = dims.get("height")
    width_pt = dims.get("width_pt")
    height_pt = dims.get("height_pt")

    if unit == CANONICAL_UNIT and width and height:
        page_w_pt, page_h_pt = float(width), float(height)
        render_w = dims.get("render_width_px") or round(page_w_pt * sx)
        render_h = dims.get("render_height_px") or round(page_h_pt * sy)
    elif width_pt and height_pt:
        page_w_pt, page_h_pt = float(width_pt), float(height_pt)
        render_w = width or round(page_w_pt * sx)
        render_h = height or round(page_h_pt * sy)
    elif width and height:
        # Legacy : dimensions en pixels.
        page_w_pt = float(width) / max(1e-9, sx)
        page_h_pt = float(height) / max(1e-9, sy)
        render_w, render_h = width, height
    else:
        page_w_pt = page_h_pt = 0.0
        render_w = render_h = 0

    orientation = "landscape" if page_w_pt > page_h_pt else "portrait"

    return {
        "width": round(page_w_pt, 3),
        "height": round(page_h_pt, 3),
        "unit": CANONICAL_UNIT,
        "origin": "top_left",
        "render_width_px": render_w,
        "render_height_px": render_h,
        "render_dpi": dpi,
        "scale_x_px_per_pt": round(sx, 6),
        "scale_y_px_per_pt": round(sy, 6),
        "orientation": orientation,
    }
