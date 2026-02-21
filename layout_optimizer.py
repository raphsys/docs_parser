import fitz

class LayoutOptimizer:
    def __init__(self):
        self.pixel_to_point = 72.0 / 150.0

    def adjust_layout(self, structure):
        if not structure.get("blocks"): return structure
        blocks = sorted(structure["blocks"], key=lambda b: b["bbox"][1])
        zones = [z for z in structure.get("non_text_zones", []) if isinstance(z, (list, tuple)) and len(z) == 4]
        dims = structure.get("dimensions") or {}
        page_w = float(dims.get("width", 0) or 0)
        page_h = float(dims.get("height", 0) or 0)
        
        for i, block in enumerate(blocks):
            max_w_px = 0
            for line in block["lines"]:
                for phrase in line["phrases"]:
                    if not phrase.get("spans"): continue
                    style = phrase["spans"][0]["style"]
                    font = "hebo" if style["flags"].get("bold") else "helv"
                    fs = style["size"] * self.pixel_to_point

                    ptxt = phrase.get("translated_text") or phrase.get("texte") or ""
                    w_pts = fitz.get_text_length(ptxt, fontname=font, fontsize=fs)
                    w_px = w_pts / self.pixel_to_point
                    if w_px > max_w_px: max_w_px = w_px
            
            old_w = block["bbox"][2] - block["bbox"][0]
            if max_w_px > old_w:
                block["bbox"][2] = block["bbox"][0] + int(max_w_px)
            self._clamp_block_to_page(block, page_w, page_h)

            # Keep text away from image/non-text zones.
            for _ in range(10):
                if not self._intersects_any(block["bbox"], zones):
                    break
                before = list(block["bbox"])
                shift = self._next_zone_push(block["bbox"], zones)
                if shift <= 0:
                    break
                self._shift_block(block, shift)
                self._clamp_block_to_page(block, page_w, page_h)
                if before == block["bbox"]:
                    break

            if i < len(blocks) - 1:
                next_b = blocks[i+1]
                margin = 15
                if block["bbox"][3] + margin > next_b["bbox"][1]:
                    shift = (block["bbox"][3] + margin) - next_b["bbox"][1]
                    self._shift_block(next_b, shift)
                    self._clamp_block_to_page(next_b, page_w, page_h)
        return structure

    def _shift_block(self, block, shift):
        block["bbox"][1] += int(shift)
        block["bbox"][3] += int(shift)
        for l in block["lines"]:
            l["bbox"][1] += int(shift); l["bbox"][3] += int(shift)
            for p in l["phrases"]:
                p["bbox"][1] += int(shift); p["bbox"][3] += int(shift)
                for s in p["spans"]:
                    s["bbox"][1] += int(shift); s["bbox"][3] += int(shift)

    def _intersects_any(self, bbox, zones):
        r1 = fitz.Rect(bbox)
        for z in zones:
            if (r1 & fitz.Rect(z)).get_area() > 0:
                return True
        return False

    def _next_zone_push(self, bbox, zones):
        r1 = fitz.Rect(bbox)
        pushes = []
        for z in zones:
            r2 = fitz.Rect(z)
            if (r1 & r2).get_area() <= 0:
                continue
            pushes.append(max(1, int(r2.y1 - r1.y0 + 8)))
        return min(pushes) if pushes else 0

    def _clamp_block_to_page(self, block, page_w, page_h):
        if page_w <= 0 or page_h <= 0:
            return
        x0, y0, x1, y1 = [int(v) for v in block["bbox"]]
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        if bw >= page_w:
            x0, x1 = 0, int(page_w)
        else:
            if x0 < 0:
                x1 -= x0
                x0 = 0
            if x1 > page_w:
                d = x1 - int(page_w)
                x0 -= d
                x1 -= d
        if bh >= page_h:
            y0, y1 = 0, int(page_h)
        else:
            if y0 < 0:
                y1 -= y0
                y0 = 0
            if y1 > page_h:
                d = y1 - int(page_h)
                y0 -= d
                y1 -= d
        dx = x0 - block["bbox"][0]
        dy = y0 - block["bbox"][1]
        block["bbox"] = [x0, y0, x1, y1]
        if dx != 0 or dy != 0:
            for l in block["lines"]:
                l["bbox"][0] += int(dx); l["bbox"][2] += int(dx)
                l["bbox"][1] += int(dy); l["bbox"][3] += int(dy)
                for p in l["phrases"]:
                    p["bbox"][0] += int(dx); p["bbox"][2] += int(dx)
                    p["bbox"][1] += int(dy); p["bbox"][3] += int(dy)
                    for s in p["spans"]:
                        s["bbox"][0] += int(dx); s["bbox"][2] += int(dx)
                        s["bbox"][1] += int(dy); s["bbox"][3] += int(dy)
