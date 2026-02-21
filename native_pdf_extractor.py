import re


def _normalize_font_key(font_name: str) -> str:
    raw = (font_name or "").split("+", 1)[-1].strip()
    return re.sub(r"[^a-z0-9]+", "", raw.lower())


def _color_int_to_hex(value) -> str:
    try:
        if isinstance(value, (tuple, list)) and len(value) >= 3:
            r = max(0, min(255, int(round(float(value[0]) * 255 if value[0] <= 1.0 else value[0]))))
            g = max(0, min(255, int(round(float(value[1]) * 255 if value[1] <= 1.0 else value[1]))))
            b = max(0, min(255, int(round(float(value[2]) * 255 if value[2] <= 1.0 else value[2]))))
            return f"#{r:02x}{g:02x}{b:02x}"
        return "#{:06x}".format(int(value) & 0xFFFFFF)
    except Exception:
        return "#000000"


class NativePDFExtractor:
    """
    Extract native (vector) text structure from a PDF page.
    Coordinates are returned in raster pixel space using sx/sy.
    """

    def extract_page(self, pdf_page, sx: float = 1.0, sy: float = 1.0):
        dict_text = pdf_page.get_text("dict")
        native_blocks = []
        non_text_zones = []
        images = []
        drawings = []

        for b in dict_text.get("blocks", []):
            block_bbox = b.get("bbox")
            if not block_bbox:
                continue
            bb_px = [
                int(block_bbox[0] * sx),
                int(block_bbox[1] * sy),
                int(block_bbox[2] * sx),
                int(block_bbox[3] * sy),
            ]

            # Image block from native PDF.
            if b.get("type") == 1:
                non_text_zones.append(bb_px)
                images.append({"bbox": bb_px, "source": "native_pdf_image"})
                continue

            lines = []
            for l in b.get("lines", []):
                phrases = []
                current_spans = []
                for s in l.get("spans", []):
                    txt = s.get("text", "")
                    if not txt:
                        continue
                    sb = s.get("bbox", block_bbox)
                    bbox_px = [
                        int(sb[0] * sx),
                        int(sb[1] * sy),
                        int(sb[2] * sx),
                        int(sb[3] * sy),
                    ]
                    font_name = s.get("font", "")
                    style = {
                        "font": font_name,
                        "font_name_raw": font_name,
                        "font_key_normalized": _normalize_font_key(font_name),
                        "size": float(s.get("size", 12.0)),  # native point size
                        "color": _color_int_to_hex(s.get("color", 0)),
                        "flags": {
                            "bold": "bold" in font_name.lower(),
                            "italic": "italic" in font_name.lower() or "oblique" in font_name.lower(),
                            "serif": any(x in font_name.lower() for x in ("times", "serif", "tiro", "roman")),
                            "uppercase": txt.isupper(),
                        },
                    }
                    current_spans.append(
                        {
                            "texte": txt,
                            "bbox": bbox_px,
                            "bbox_pt": [float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3])],
                            "style": style,
                        }
                    )

                if not current_spans:
                    continue
                phrase_text = "".join(sp["texte"] for sp in current_spans)
                line_bbox = [
                    min(sp["bbox"][0] for sp in current_spans),
                    min(sp["bbox"][1] for sp in current_spans),
                    max(sp["bbox"][2] for sp in current_spans),
                    max(sp["bbox"][3] for sp in current_spans),
                ]
                phrases.append({"texte": phrase_text, "bbox": line_bbox, "spans": current_spans})
                lines.append({"bbox": line_bbox, "phrases": phrases})

            if lines:
                b_b = [
                    min(l["bbox"][0] for l in lines),
                    min(l["bbox"][1] for l in lines),
                    max(l["bbox"][2] for l in lines),
                    max(l["bbox"][3] for l in lines),
                ]
                native_blocks.append(
                    {
                        "id": f"n_{len(native_blocks)}",
                        "bbox": b_b,
                        "lines": lines,
                        "source": "native",
                    }
                )

        # Optional vector drawings as no-text zones (advanced fidelity hooks).
        try:
            for d in pdf_page.get_drawings():
                r = d.get("rect")
                if not r:
                    continue
                rb = [int(r.x0 * sx), int(r.y0 * sy), int(r.x1 * sx), int(r.y1 * sy)]
                if rb[2] > rb[0] and rb[3] > rb[1]:
                    drawings.append({"bbox": rb, "source": "native_pdf_drawing"})
        except Exception:
            drawings = []

        return {
            "blocks": native_blocks,
            "non_text_zones": non_text_zones,
            "images": images,
            "drawings": drawings,
        }

