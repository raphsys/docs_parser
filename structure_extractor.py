import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageStat, ImageDraw
import io
import math
from collections import Counter
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

class VisualAttributeExtractor:
    """Analyse visuelle de précision industrielle"""
    
    def analyze(self, pil_image, bbox):
        try:
            w, h = pil_image.size
            bbox = [max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])]
            crop = pil_image.crop(bbox)
        except:
            return self._default_style()

        if crop.width < 2 or crop.height < 2:
            return self._default_style()

        color_hex = self._get_precise_color(crop)
        
        gray = crop.convert("L")
        arr = np.array(gray)
        threshold = np.mean(arr) * 0.85
        bw = (arr < threshold).astype(np.uint8)
        
        density = np.mean(bw)
        is_bold = bool(density > 0.26)
        
        # Heuristique Serif
        edge_zone = max(1, bw.shape[0]//10)
        is_serif = bool(np.mean(bw[:edge_zone, :] > 0) > 0.3)

        return {
            "font": "Times-New-Roman" if is_serif else "Arial/Helvetica",
            "size": bbox[3] - bbox[1],
            "color": color_hex,
            "flags": {
                "bold": is_bold,
                "italic": False,
                "serif": is_serif,
                "mono": False
            }
        }

    def _get_precise_color(self, img):
        img = img.convert("RGB").quantize(colors=4).convert("RGB")
        counts = Counter(list(img.getdata())).most_common(4)
        if not counts: return "#000000"
        sorted_by_light = sorted(counts, key=lambda x: sum(x[0]), reverse=True)
        text_color = sorted_by_light[1][0] if len(sorted_by_light) > 1 else sorted_by_light[0][0]
        return '#{:02x}{:02x}{:02x}'.format(*text_color)

    def _default_style(self):
        return {"font": "Arial", "size": 0, "color": "#000000", "flags": {}}

class DocumentParser:
    def __init__(self):
        self.visual_extractor = VisualAttributeExtractor()

    def parse_page(self, page, page_index, force_ai=False, ai_results=None, scale_x=1.0, scale_y=1.0):
        if force_ai and ai_results:
             return self._parse_visual_page(page, page_index, ai_results)
        else:
             return self._parse_native_page(page, page_index, scale_x, scale_y)

    def _parse_native_page(self, page, index, scale_x, scale_y):
        def scale_bbox(bbox):
            return [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]

        raw_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
        font_stats = self._analyze_font_stats(raw_dict)
        
        structured_page = {
            "page_number": index + 1,
            "dimensions": {"width": page.rect.width * scale_x, "height": page.rect.height * scale_y, "unit": "px"},
            "layout": {
                "rotation": page.rotation,
                "margins": {"top": 40.0*scale_y, "bottom": 40.0*scale_y, "left": 40.0*scale_x, "right": 40.0*scale_x},
                "reading_order": "logical" 
            },
            "source": "native",
            "blocks": [],
            "tables": [],
            "links": []
        }

        tabs = page.find_tables()
        tables_rects = [fitz.Rect(t.bbox) for t in tabs]
        for tab in tabs:
            data = tab.extract()
            structured_page["tables"].append({
                "bbox": scale_bbox(tab.bbox),
                "grid": {"rows": len(data), "cols": len(data[0]) if data else 0},
                "markdown": self._to_markdown(data)
            })

        for b_idx, block in enumerate(raw_dict["blocks"]):
            b_rect = fitz.Rect(block["bbox"])
            in_table = any(b_rect.intersects(tr) for tr in tables_rects)
            
            parsed_block = {
                "type": "text" if block["type"] == 0 else "image",
                "bbox": scale_bbox(block["bbox"]),
                "role": self._detect_role(block, font_stats, page.rect),
                "in_table": in_table,
                "source": "native",
                "lines": []
            }

            if block["type"] == 0:
                for line in block["lines"]:
                    line_text = "".join([s["text"] for s in line["spans"]]).strip()
                    if not line_text: continue
                    parsed_line = {
                        "bbox": scale_bbox(line["bbox"]),
                        "spans": [],
                        "alignment": self._detect_alignment(line, block["bbox"]),
                        "lang": self._detect_lang(line_text),
                        "is_math": any(s in line_text for s in ['∑', '∫', 'π', '×', '√', 'ŷ', '='])
                    }
                    for span in line["spans"]:
                        parsed_line["spans"].append({
                            "text": span["text"],
                            "style": {
                                "font": span["font"],
                                "size": round(span["size"] * scale_y, 1),
                                "color": "#{:06x}".format(span["color"] & 0xFFFFFF),
                                "flags": self._parse_flags(span["flags"])
                            },
                            "bbox": scale_bbox(span["bbox"]),
                            "origin": [span["origin"][0] * scale_x, span["origin"][1] * scale_y]
                        })
                    parsed_block["lines"].append(parsed_line)
            structured_page["blocks"].append(parsed_block)
        return structured_page

    def _to_markdown(self, data):
        if not data: return ""
        lines = ["| " + " | ".join([str(c).replace("\n", " ") for c in r]) + " |" for r in data]
        if lines: lines.insert(1, "| " + " | ".join(["---"]*len(data[0])) + " |")
        return "\n".join(lines)

    def _detect_lang(self, text):
        if len(text) < 10: return "unknown"
        try: return detect(text)
        except: return "unknown"

    def _analyze_font_stats(self, raw_dict):
        sizes = [round(s["size"], 1) for b in raw_dict["blocks"] if b["type"] == 0 for l in b["lines"] for s in l["spans"]]
        return {"body": Counter(sizes).most_common(1)[0][0] if sizes else 12}

    def _detect_role(self, block, stats, page_rect):
        b = block["bbox"]
        if b[1] < page_rect.height * 0.08: return "header"
        if b[3] > page_rect.height * 0.92: return "footer"
        max_s = max([s["size"] for l in block["lines"] for s in l["spans"]] if block["type"]==0 and block["lines"] else [0])
        if max_s > stats["body"] * 1.5: return "h1"
        return "paragraph"

    def _detect_alignment(self, line, b_bbox):
        l_c = (line["bbox"][0] + line["bbox"][2]) / 2
        b_c = (b_bbox[0] + b_bbox[2]) / 2
        if abs(l_c - b_c) < 15: return "center"
        if abs(line["bbox"][2] - b_bbox[2]) < 15: return "right"
        return "left"

    def _parse_flags(self, flags):
        return {"italic": bool(flags & 2**1), "serif": bool(flags & 2**2), "mono": bool(flags & 2**3), "bold": bool(flags & 2**4)}

    def _parse_visual_page(self, page, index, ai_results):
        pix = page.get_pixmap(dpi=120)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        structured_page = {
            "page_number": index + 1,
            "dimensions": {"width": img.width, "height": img.height, "unit": "px"},
            "layout": {"rotation": 0, "margins": {"top": 40.0, "bottom": 40.0, "left": 40.0, "right": 40.0}, "reading_order": "visual"},
            "source": "ai_vision",
            "blocks": []
        }
        for i, item in enumerate(ai_results):
            bbox = item.get("bbox")
            text = item.get("label", "").strip()
            style = self.visual_extractor.analyze(img, bbox)
            structured_page["blocks"].append({
                "id": f"p{index}_ai_{i}",
                "type": "text", "bbox": bbox, "role": "h1" if style['size'] > 30 else "paragraph",
                "source": "ai",
                "lines": [{
                    "bbox": bbox, "lang": self._detect_lang(text), "alignment": "left",
                    "spans": [{"text": text, "style": style, "bbox": bbox}]
                }]
            })
        return structured_page