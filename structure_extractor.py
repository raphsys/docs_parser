import fitz
import numpy as np
import cv2
import re
import statistics
from PIL import Image, ImageOps

class VisualAttributeExtractor:
    def analyze(self, pil_image, bbox, text=""):
        try:
            w, h = pil_image.size
            bbox = [max(0, int(bbox[0])), max(0, int(bbox[1])), min(w, int(bbox[2])), min(h, int(bbox[3]))]
            crop = pil_image.crop(bbox)
        except: return [{"texte": text, "style": self._default_style(), "bbox": bbox}]

        # 1. Préparation et nettoyage drastique
        gray = crop.convert("L")
        arr = np.array(gray)
        
        # Détection de contours CANNY (bords nets uniquement)
        # C'est souvent plus précis que le thresholding pour la géométrie
        edges = cv2.Canny(arr, 50, 150)
        
        # Dilatation pour connecter les lettres d'un même mot
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.dilate(edges, kernel, iterations=1)
        
        # Trouver les contours externes
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
             return [{"texte": text, "style": self._default_style(), "bbox": bbox}]

        # Filtrer les contours minuscules (bruit pur)
        valid_rects = []
        img_h, img_w = binary.shape
        total_area = img_h * img_w
        
        for cnt in contours:
            x, y, w_c, h_c = cv2.boundingRect(cnt)
            # On garde si c'est assez grand OU si c'est un point (mais pas un pixel isolé)
            if w_c * h_c > total_area * 0.001 or (w_c > 1 and h_c > 1): 
                valid_rects.append((x, y, w_c, h_c))
        
        if not valid_rects:
             return [{"texte": text, "style": self._default_style(), "bbox": bbox}]

        # Stratégie "Ligne Médiane" : Le texte principal croise forcément le centre de masse vertical
        # On projette tout sur l'axe Y pour trouver où est le texte
        y_coverage = np.zeros(img_h)
        for (x, y, w_c, h_c) in valid_rects:
            y_coverage[y:y+h_c] += w_c # On pondère par la largeur

        max_val = np.max(y_coverage)
        if max_val == 0: return [{"texte": text, "style": self._default_style(), "bbox": bbox}]

        # On définit la zone active comme étant là où il y a de la densité significative
        active_y_indices = np.where(y_coverage > max_val * 0.05)[0] # 5% du max (plus robuste pour les descendantes)
        
        if len(active_y_indices) == 0:
             return [{"texte": text, "style": self._default_style(), "bbox": bbox}]

        # Limites Y "sûres" du texte principal
        safe_ymin, safe_ymax = active_y_indices[0], active_y_indices[-1]

        # On ne garde que les contours qui intersectent cette bande sûre
        final_rects = []
        for (x, y, w_c, h_c) in valid_rects:
            # Intersection verticale
            iy1 = max(y, safe_ymin)
            iy2 = min(y+h_c, safe_ymax)
            if iy2 > iy1: # Il y a intersection
                final_rects.append((x, y, w_c, h_c))

        if not final_rects:
            ymin, ymax = safe_ymin, safe_ymax
            xmin, xmax = 0, img_w # Fallback X
        else:
            # BBox englobante de tous les contours validés
            min_x = min([r[0] for r in final_rects])
            min_y = min([r[1] for r in final_rects])
            max_x = max([r[0] + r[2] for r in final_rects])
            max_y = max([r[1] + r[3] for r in final_rects])
            
            xmin, ymin, xmax, ymax = min_x, min_y, max_x, max_y

        # Calcul de la bbox finale absolue
        real_tight_bbox = [bbox[0] + xmin, bbox[1] + ymin, bbox[0] + xmax, bbox[1] + ymax]
        
        # Reconstruction du masque propre pour l'analyse de style
        clean_mask = np.zeros_like(binary)
        clean_mask[ymin:ymax, xmin:xmax] = binary[ymin:ymax, xmin:xmax]
        is_ink = clean_mask > 0

        # --- OPTIMISATION VERTICALE "PEAK-AND-EXPAND" ---
        # Si le crop est grand, il se peut que le contouring ait tout gardé à cause d'un bruit connecté
        # On refait une passe d'analyse de profil strict sur la zone "nettoyée"
        
        row_density = np.sum(is_ink, axis=1)
        max_dens = np.max(row_density)
        peak_y = (ymin + ymax) // 2 # Fallback
        
        if max_dens > 0:
            # On trouve le pic de densité (le coeur du texte)
            peak_y = np.argmax(row_density)
            
            # Seuil d'arrêt ajusté : 5% de la densité max (plus inclusif pour les ascendantes/pendantes)
            stop_thresh = max_dens * 0.05
            
            # Expansion vers le HAUT
            new_ymin = peak_y
            while new_ymin > 0 and row_density[new_ymin] > stop_thresh:
                new_ymin -= 1
            new_ymin = max(ymin, new_ymin)

            # Expansion vers le BAS
            new_ymax = peak_y
            while new_ymax < len(row_density) - 1 and row_density[new_ymax] > stop_thresh:
                new_ymax += 1
            new_ymax = min(ymax, new_ymax)
            
            # Mise à jour des coordonnées
            if new_ymax > new_ymin + 2: # Sécurité anti-écrasement
                ymin = int(new_ymin)
                ymax = int(new_ymax)
                real_tight_bbox = [bbox[0] + xmin, bbox[1] + ymin, bbox[0] + xmax, bbox[1] + ymax]
                
                # Mise à jour du masque is_ink pour le calcul de style suivant
                is_ink[:] = False
                is_ink[ymin:ymax, xmin:xmax] = clean_mask[ymin:ymax, xmin:xmax]

        # --- Analyse du Style sur la zone propre ---
        full_style = self._extract_segment_style(crop, is_ink, [xmin, ymin, xmax-1, ymax-1], text)
        
        # Détection Majuscule basée sur le texte OCR
        full_style["flags"]["uppercase"] = text.isupper() if len(text) > 1 else False
        
        # Détection Surlignage (Highlight) plus robuste
        bg_color = self._get_background_color(crop, is_ink)
        is_highlighted = False
        try:
            c = bg_color.lstrip('#')
            rgb_bg = tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
            # On compare à la couleur de l'encre (si l'encre est proche du fond, ce n'est pas un highlight)
            ink_c = full_style["color"].lstrip('#')
            rgb_ink = tuple(int(ink_c[i:i+2], 16) for i in (0, 2, 4))
            
            # Seuil plus strict : Fond sombre ET forte différence de saturation/couleur avec l'encre
            if sum(rgb_bg) / 3 < 190 and np.linalg.norm(np.array(rgb_bg) - np.array(rgb_ink)) > 110:
                is_highlighted = True
        except: pass
        
        full_style["flags"]["highlight"] = is_highlighted
        full_style["highlight_color"] = bg_color if is_highlighted else "#ffffff"
        full_style["flags"]["underline"] = self._check_underline(is_ink, ymax)

        # Calcul du peak_y absolu
        abs_peak_y = bbox[1] + peak_y

        return [{"texte": text, "style": full_style, "bbox": real_tight_bbox, "peak_y": abs_peak_y}]

    def _extract_segment_style(self, crop, is_ink, local_bbox, text=""):
        xmin, ymin, xmax, ymax = local_bbox
        mask = is_ink[ymin:ymax+1, xmin:xmax+1]
        if mask.size == 0: return self._default_style()

        img_arr = np.array(crop.convert("RGB"))[ymin:ymax+1, xmin:xmax+1]
        
        color = self._get_pure_ink_color(img_arr, mask)
        height = ymax - ymin
        
        # Calcul de la graisse (Bold) 
        dist_map = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        max_stroke = np.max(dist_map) if dist_map.size > 0 else 0
        avg_stroke_width = max_stroke * 2
        
        # Seuil adaptatif : pour les titres (height > 60), on est très tolérant
        if height > 60:
            threshold = 0.04 # Les titres ont des traits fins par rapport à leur taille immense
        else:
            threshold = 0.20 - (min(max(height, 20), 60) - 20) * (0.05 / 40)
            
        is_bold = bool(avg_stroke_width > (height * threshold))

        is_serif = self._check_serif(mask)
        
        return {
            "font": "Times-New-Roman" if is_serif else "Arial",
            "size": int(height),
            "stroke_width": float(avg_stroke_width),
            "color": color,
            "flags": {
                "bold": is_bold,
                "italic": bool(self._check_italic(mask)),
                "serif": is_serif,
                "uppercase": False 
            }
        }

    def _get_segments(self, col_ink):
        segments = []
        in_seg = False
        start = 0
        for x in range(len(col_ink)):
            if col_ink[x] and not in_seg:
                start = x
                in_seg = True
            elif not col_ink[x] and in_seg:
                segments.append((start, x))
                in_seg = False
        if in_seg: segments.append((start, len(col_ink)))
        return segments

    def _get_pure_ink_color(self, img_arr, mask):
        pixels = img_arr[mask]
        if len(pixels) == 0: return "#000000"
        return '#{:02x}{:02x}{:02x}'.format(*np.median(pixels, axis=0).astype(int))

    def _get_background_color(self, crop, is_ink):
        bg_pixels = np.array(crop.convert("RGB"))[~is_ink]
        if len(bg_pixels) == 0: return "#ffffff"
        return '#{:02x}{:02x}{:02x}'.format(*np.median(bg_pixels, axis=0).astype(int))

    def _check_underline(self, is_ink, ymax):
        # Cherche une ligne horizontale continue en bas
        bottom_line = is_ink[max(0, ymax-3):ymax+1, :]
        if bottom_line.shape[1] == 0: return False
        return np.mean(np.any(bottom_line, axis=0)) > 0.7

    def _check_italic(self, is_ink):
        h, w = is_ink.shape
        if h < 10 or w < 5: return False
        # Analyse de l'inclinaison par moments ou par centres de masse
        top_half = is_ink[:h//2, :]
        bot_half = is_ink[h//2:, :]
        if not np.any(top_half) or not np.any(bot_half): return False
        
        m_top = np.mean(np.where(top_half)[1])
        m_bot = np.mean(np.where(bot_half)[1])
        return m_top > m_bot + (h * 0.05)

    def _check_serif(self, mask):
        h, w = mask.shape
        if h < 12 or w < 6: return False
        
        # Les empattements sont souvent horizontaux aux extrémités verticales
        top_zone = mask[:int(h*0.15), :]
        bot_zone = mask[int(h*0.85):, :]
        
        # Si la largeur aux extrémités est significativement plus grande que le milieu
        mid_width = np.sum(mask[h//2, :])
        top_width = np.mean(np.sum(top_zone, axis=1))
        bot_width = np.mean(np.sum(bot_zone, axis=1))
        
        return bool(top_width > mid_width * 1.3 or bot_width > mid_width * 1.3)

    def _default_style(self):
        return {"font": "Arial", "size": 12, "color": "#000000", "flags": {"bold":False, "italic":False, "serif":False, "underline":False, "highlight":False}}

class DocumentParser:
    def __init__(self):
        self.visual_extractor = VisualAttributeExtractor()

    def parse(self, ocr_results, image):
        # 1. Nettoyage et déduplication
        clean_results = self._deduplicate_ocr(ocr_results)
        
        # 2. Groupement en lignes (basé sur OCR raw)
        raw_lines = self._group_into_lines(clean_results)
        
        # 3. Raffinement visuel de chaque ligne
        refined_lines_info = []
        for line_words in raw_lines:
            line_bbox = self._merge_bboxes([w["bbox"] for w in line_words])
            text_content = " ".join([w["label"] for w in line_words])
            
            # On analyse la ligne globale
            analysis = self.visual_extractor.analyze(image, line_bbox, text_content)
            res = analysis[0]
            
            refined_lines_info.append({
                "raw_words": line_words,
                "bbox": res["bbox"],
                "peak_y": res.get("peak_y", (res["bbox"][1] + res["bbox"][3]) / 2),
                "style": res["style"],
                "text": text_content
            })
            
        # 4. Groupement en blocs basé sur les bboxes raffinées
        blocks = self._group_into_blocks(refined_lines_info)
        
        # 5. Enrichissement final (reconstruction de la structure)
        final_structure = []
        for blk in blocks:
            enriched_lines = []
            for line_info in blk["lines_info"]:
                # On ré-extrait les phrases pour le style local si nécessaire
                # Mais ici on peut déjà utiliser line_info
                phrases = self._extract_phrases(line_info["raw_words"], image)
                enriched_lines.append({
                    "bbox": line_info["bbox"],
                    "peak_y": line_info["peak_y"],
                    "phrases": phrases
                })
            
            final_structure.append({
                "id": blk["id"],
                "bbox": blk["bbox"],
                "lines": enriched_lines,
                "line_spacing": blk["line_spacing"],
                "avg_line_height": blk["avg_line_height"],
                "peak_to_peak_spacing": blk.get("peak_to_peak_spacing", 0)
            })
            
        return final_structure

    def _deduplicate_ocr(self, results):
        if not results: return []
        # Tri par aire décroissante (les grands rectangles d'abord)
        results.sort(key=lambda x: (x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]), reverse=True)
        keep = []
        for candidate in results:
            is_dup = False
            cb = candidate["bbox"]
            for accepted in keep:
                ab = accepted["bbox"]
                # Intersection
                ix0, iy0 = max(cb[0], ab[0]), max(cb[1], ab[1])
                ix1, iy1 = min(cb[2], ab[2]), min(cb[3], ab[3])
                if ix1 > ix0 and iy1 > iy0:
                    inter_area = (ix1-ix0) * (iy1-iy0)
                    cand_area = (cb[2]-cb[0]) * (cb[3]-cb[1])
                    if cand_area <= 0:
                        is_dup = True
                        break
                    # Si plus de 85% du candidat est couvert par un existant
                    if inter_area / cand_area > 0.85:
                        # Et que le texte est contenu (souvent le cas pour les sous-parties)
                        if candidate["label"].lower() in accepted["label"].lower():
                            is_dup = True; break
            if not is_dup: keep.append(candidate)
        return keep

    def _group_into_lines(self, results):
        if not results: return []
        # Tri vertical
        results.sort(key=lambda x: (x["bbox"][1] + x["bbox"][3]) / 2)
        lines = []
        curr_line = [results[0]]
        
        for i in range(1, len(results)):
            prev, curr = curr_line[-1], results[i]
            c_prev = (prev["bbox"][1] + prev["bbox"][3]) / 2
            c_curr = (curr["bbox"][1] + curr["bbox"][3]) / 2
            h_ref = min(prev["bbox"][3]-prev["bbox"][1], curr["bbox"][3]-curr["bbox"][1])
            
            # Seuil de tolérance vertical pour être sur la même ligne (40% de la hauteur)
            if abs(c_curr - c_prev) < h_ref * 0.4:
                curr_line.append(curr)
            else:
                lines.append(sorted(curr_line, key=lambda x: x["bbox"][0]))
                curr_line = [results[i]]
        lines.append(sorted(curr_line, key=lambda x: x["bbox"][0]))
        return lines

    def _group_into_blocks(self, lines_info):
        if not lines_info: return []
        blocks = []
        
        # Initialisation du premier bloc
        curr_block_lines = [lines_info[0]]
        
        for i in range(1, len(lines_info)):
            prev_line = lines_info[i-1]
            curr_line = lines_info[i]
            
            # Hauteur moyenne et gaps
            prev_h = prev_line["bbox"][3] - prev_line["bbox"][1]
            curr_h = curr_line["bbox"][3] - curr_line["bbox"][1]
            avg_h = (prev_h + curr_h) / 2
            
            # Espacement entre centres de lignes (plus stable)
            prev_c = (prev_line["bbox"][1] + prev_line["bbox"][3]) / 2
            curr_c = (curr_line["bbox"][1] + curr_line["bbox"][3]) / 2
            c_gap = curr_c - prev_c
            
            v_gap = curr_line["bbox"][1] - prev_line["bbox"][3]
            h_diff_left = abs(curr_line["bbox"][0] - prev_line["bbox"][0])
            
            # CRITÈRES DE RUPTURE DE BLOC (Paragraphe)
            is_break = False
            # 1. Espacement vertical significatif (plus tolérant pour les titres)
            if v_gap > avg_h * 2.0: is_break = True
            # 2. Saut de ligne disproportionné par rapport à la hauteur
            elif c_gap > avg_h * 3.5: is_break = True
            # 3. Changement d'indentation notable
            elif h_diff_left > avg_h * 2.5: is_break = True
            # 4. Changement de style (hauteur de ligne)
            elif abs(curr_h - prev_h) > avg_h * 0.5: is_break = True
            
            if is_break:
                blocks.append(self._finalize_block(len(blocks), curr_block_lines))
                curr_block_lines = [curr_line]
            else:
                curr_block_lines.append(curr_line)
                
        blocks.append(self._finalize_block(len(blocks), curr_block_lines))
        return blocks

    def _finalize_block(self, bid, lines_info):
        # Bbox globale du bloc
        block_bbox = self._merge_bboxes([l["bbox"] for l in lines_info])
        
        # Calcul des interlignes
        # On utilise le gap entre bboxes serrées ET la distance entre pics pour plus de fidélité
        gaps = []
        peak_diffs = []
        heights = []
        
        for i in range(len(lines_info)):
            h = lines_info[i]["bbox"][3] - lines_info[i]["bbox"][1]
            heights.append(h)
            if i > 0:
                gap = lines_info[i]["bbox"][1] - lines_info[i-1]["bbox"][3]
                gaps.append(gap)
                p_diff = lines_info[i]["peak_y"] - lines_info[i-1]["peak_y"]
                peak_diffs.append(p_diff)
        
        # L'interligne "fidèle" est souvent mieux représenté par la médiane des gaps
        # pour ignorer les lignes avec des descendantes/ascendantes extrêmes
        avg_spacing = np.median(gaps) if gaps else 0
        avg_height = np.median(heights) if heights else 0
        
        return {
            "id": bid,
            "lines_info": lines_info,
            "bbox": block_bbox,
            "line_spacing": float(avg_spacing),
            "avg_line_height": float(avg_height),
            "peak_to_peak_spacing": float(np.median(peak_diffs)) if peak_diffs else 0
        }

    def _extract_phrases(self, line_words, img):
        # Groupement horizontal simple
        if not line_words: return []
        
        # Seuil d'espace entre mots
        avg_h = sum([w["bbox"][3] - w["bbox"][1] for w in line_words]) / len(line_words)
        gap_threshold = avg_h * 1.5
        
        phrases = []
        curr_p = [line_words[0]]
        
        for j in range(1, len(line_words)):
            dist = line_words[j]["bbox"][0] - line_words[j-1]["bbox"][2]
            if dist > gap_threshold:
                phrases.append(self._assemble_phrase(curr_p, img))
                curr_p = [line_words[j]]
            else:
                curr_p.append(line_words[j])
        phrases.append(self._assemble_phrase(curr_p, img))
        return phrases

    def _assemble_phrase(self, words, img):
        text = " ".join([w["label"] for w in words])
        bbox = self._merge_bboxes([w["bbox"] for w in words])
        
        # Analyse visuelle fine pour extraire le style
        styled_spans = self.visual_extractor.analyze(img, bbox, text)
        
        return {
            "texte": text,
            "bbox": styled_spans[0]["bbox"] if styled_spans else bbox,
            "spans": styled_spans
        }

    def _merge_bboxes(self, bboxes):
        if not bboxes: return [0,0,0,0]
        x0 = min([b[0] for b in bboxes])
        y0 = min([b[1] for b in bboxes])
        x1 = max([b[2] for b in bboxes])
        y1 = max([b[3] for b in bboxes])
        return [int(x0), int(y0), int(x1), int(y1)]


# =============================================================================
# Layout V2 Builder (canonical structure for translator/reconstructor)
# =============================================================================
class LayoutV2Builder:
    """
    Build a canonical page layout representation (layout.v2) suitable for:
    - stable translation that does not destroy structure (TOC, lists)
    - stable reconstruction with appropriate render modes (tabbed layout for TOC)
    """

    SCHEMA_VERSION = "layout.v2"

    def __init__(self):
        pass

    def build(self, page_data):
        """
        Input: page_data as produced by the current pipeline (blocks/lines/phrases),
               typically coming from native extraction or OCR.
        Output: same page_data augmented with:
          - schema_version
          - page_role
          - layout (margins, columns, tab_stops)
          - toc (toc_rows) when detected
          - indent_level injected per line
        """
        if not isinstance(page_data, dict):
            return page_data

        page_data.setdefault("schema_version", self.SCHEMA_VERSION)

        dims = page_data.get("dimensions") or {}
        page_w = float(dims.get("width", 0.0) or 0.0)
        page_h = float(dims.get("height", 0.0) or 0.0)

        all_lines = self._iter_lines(page_data)
        margins = self._infer_margins(all_lines, page_w, page_h)
        columns = self._infer_columns(all_lines, margins, page_w)
        page_data.setdefault("layout", {})
        page_data["layout"]["margins"] = margins
        page_data["layout"]["columns"] = columns

        page_role = self._detect_page_role(page_data, all_lines)
        page_data["page_role"] = page_role
        page_family = self._detect_page_family(page_data, all_lines, page_role=page_role)
        page_data["page_family"] = page_family
        page_data["layout"]["page_family"] = page_family

        self._inject_indent_levels(page_data, columns, margins)

        if page_role == "toc":
            toc_rows, tab_stops = self._build_toc_rows(page_data, columns, margins)
            page_data["toc"] = {
                "toc_rows": toc_rows,
                "tab_stops": tab_stops,
            }
        return page_data

    def _iter_lines(self, page_data):
        lines = []
        for b in (page_data.get("blocks") or []):
            for ln in (b.get("lines") or []):
                bb = ln.get("bbox")
                if not bb or len(bb) != 4:
                    continue
                try:
                    x0, y0, x1, y1 = map(float, bb)
                except Exception:
                    continue
                lines.append({"bbox": [x0, y0, x1, y1], "block": b, "line": ln})
        return lines

    def _infer_margins(self, lines, page_w, page_h):
        if not lines or page_w <= 0 or page_h <= 0:
            return {"left": 0.0, "right": page_w, "top": 0.0, "bottom": page_h}
        xs0 = sorted([l["bbox"][0] for l in lines])
        xs1 = sorted([l["bbox"][2] for l in lines])
        ys0 = sorted([l["bbox"][1] for l in lines])
        ys1 = sorted([l["bbox"][3] for l in lines])

        def pct(arr, p):
            if not arr:
                return 0.0
            k = int(round((len(arr) - 1) * p))
            return float(arr[max(0, min(len(arr) - 1, k))])

        left = pct(xs0, 0.03)
        right = pct(xs1, 0.97)
        top = pct(ys0, 0.03)
        bottom = pct(ys1, 0.97)
        left = max(0.0, min(left, page_w))
        right = max(left + 1.0, min(right, page_w))
        top = max(0.0, min(top, page_h))
        bottom = max(top + 1.0, min(bottom, page_h))
        return {"left": left, "right": right, "top": top, "bottom": bottom}

    def _infer_columns(self, lines, margins, page_w):
        if not lines:
            return [{"id": 0, "x0": margins["left"], "x1": margins["right"]}]
        x0s = sorted([l["bbox"][0] for l in lines])
        if len(x0s) < 10:
            return [{"id": 0, "x0": margins["left"], "x1": margins["right"]}]

        diffs = [x0s[i + 1] - x0s[i] for i in range(len(x0s) - 1)]
        if not diffs:
            return [{"id": 0, "x0": margins["left"], "x1": margins["right"]}]

        max_gap = max(diffs)
        gap_i = diffs.index(max_gap)
        if max_gap > max(40.0, (margins["right"] - margins["left"]) * 0.12):
            split_x = (x0s[gap_i] + x0s[gap_i + 1]) / 2.0
            split_x = max(margins["left"] + 50.0, min(split_x, margins["right"] - 50.0))
            return [
                {"id": 0, "x0": margins["left"], "x1": split_x},
                {"id": 1, "x0": split_x, "x1": margins["right"]},
            ]
        return [{"id": 0, "x0": margins["left"], "x1": margins["right"]}]

    def _detect_page_role(self, page_data, lines):
        blocks = page_data.get("blocks") or []
        dims = page_data.get("dimensions") or {}
        page_h = float(dims.get("height", 1.0) or 1.0)

        top_title = False
        toc_like = 0
        total = 0
        page_marker_lines = 0
        label_lines = 0
        paired_rows = 0
        flat_lines = []

        for b in blocks:
            bb = b.get("bbox") or [0, 0, 0, 0]
            try:
                by0 = float(bb[1])
            except Exception:
                by0 = 1e9
            if by0 < page_h * 0.18:
                t = (b.get("text") or b.get("translated_text") or "").strip()
                if re.search(r"\b(CONTENTS|SOMMAIRE|TABLE\s+OF\s+CONTENTS)\b", t, flags=re.I):
                    top_title = True
            for ln in (b.get("lines") or []):
                total += 1
                lt = (ln.get("line_text") or ln.get("translated_text") or "").strip()
                if not lt:
                    parts = []
                    for ph in (ln.get("phrases") or []):
                        tx = (ph.get("text") or ph.get("translated_text") or ph.get("texte") or "").strip()
                        if tx:
                            parts.append(tx)
                    lt = " ".join(parts).strip()
                if not lt:
                    continue
                flat_lines.append((lt, ln))
                if re.search(r"^\s*\d+(?:\.\d+)*\s+.+\s+\d{1,3}\s*$", lt):
                    toc_like += 1
                elif re.search(r"^[A-Za-z].+\s+\d{1,3}\s*$", lt) and len(lt) <= 100:
                    toc_like += 1
                elif re.search(r"^[ivxlcdm]+\s*$", lt, flags=re.I):
                    toc_like += 1
                if re.fullmatch(r"\d{1,3}|[ivxlcdm]+", lt, flags=re.I):
                    page_marker_lines += 1
                elif len(lt) <= 120:
                    label_lines += 1

        for idx in range(len(flat_lines) - 1):
            cur, cur_ln = flat_lines[idx]
            nxt, nxt_ln = flat_lines[idx + 1]
            if not re.fullmatch(r"\d{1,3}|[ivxlcdm]+", nxt, flags=re.I):
                continue
            if re.fullmatch(r"\d{1,3}|[ivxlcdm]+", cur, flags=re.I):
                continue
            if len(cur) > 140:
                continue
            try:
                cy0 = float((cur_ln.get("bbox") or [0, 0, 0, 0])[1])
                ny0 = float((nxt_ln.get("bbox") or [0, 0, 0, 0])[1])
            except Exception:
                cy0, ny0 = 0.0, 0.0
            if abs(ny0 - cy0) <= 120.0:
                paired_rows += 1

        if top_title and toc_like >= 6:
            return "toc"
        if top_title and paired_rows >= 8 and page_marker_lines >= 8 and label_lines >= 8:
            return "toc"
        if toc_like >= 10 and total >= 12:
            return "toc"
        return "body"

    def _detect_page_family(self, page_data, lines, page_role="body"):
        role = str(page_role or "body").strip().lower()
        if role == "toc":
            return "toc"

        blocks = page_data.get("blocks") or []
        images = page_data.get("images") or []
        drawings = page_data.get("drawings") or []
        non_text_zones = page_data.get("non_text_zones") or []

        figure_captions = 0
        body_blocks = 0
        short_native_labels = 0
        diagram_roles = 0
        tableish_lines = 0

        for block in blocks:
            block_role = str(block.get("role") or "body").strip().lower()
            text = (block.get("translated_text") or block.get("text") or "").strip()
            source = str(block.get("source") or "ocr").strip().lower()
            line_count = len(block.get("lines") or [])

            if block_role == "figure_caption":
                figure_captions += 1
            if block_role == "body":
                body_blocks += 1
            if block_role in {"diagram_label", "diagram_text_label", "axis_label", "legend_label"}:
                diagram_roles += 1
            if (
                block_role == "title"
                and source == "native"
                and text
                and len(text) <= 120
                and line_count <= 3
            ):
                short_native_labels += 1

            for line in block.get("lines") or []:
                line_text = (line.get("translated_text") or line.get("line_text") or "").strip()
                if not line_text:
                    continue
                if (
                    "|" in line_text
                    or "\t" in line_text
                    or len(re.findall(r"\b\d+(?:[.,]\d+)?\b", line_text)) >= 3
                ):
                    tableish_lines += 1

        visual_non_text = len(images) + len(drawings) + len(non_text_zones)

        if tableish_lines >= 6:
            return "table_page"
        if figure_captions >= 1 and (visual_non_text >= 1 or short_native_labels >= 2):
            return "body_with_figure"
        if visual_non_text >= 1 and (diagram_roles >= 1 or short_native_labels >= 3):
            return "body_with_diagram"
        if visual_non_text >= 1 and body_blocks >= 1:
            return "mixed_page"
        return "body_text"

    def _inject_indent_levels(self, page_data, columns, margins):
        col0_left = columns[0]["x0"] if columns else margins.get("left", 0.0)
        indents = []
        for b in (page_data.get("blocks") or []):
            for ln in (b.get("lines") or []):
                bb = ln.get("bbox") or None
                if not bb:
                    continue
                try:
                    x0 = float(bb[0])
                except Exception:
                    continue
                indents.append(max(0.0, x0 - col0_left))
        if not indents:
            return

        rounded = [int(round(v / 10.0) * 10) for v in indents]
        uniq = sorted(set(rounded))
        if len(uniq) > 6:
            qs = statistics.quantiles(rounded, n=4, method="inclusive")
            cuts = [float(qs[i]) for i in range(3)]

            def level(v):
                if v <= cuts[0]:
                    return 0
                if v <= cuts[1]:
                    return 1
                if v <= cuts[2]:
                    return 2
                return 3
        else:
            mapping = {u: i for i, u in enumerate(uniq)}

            def level(v):
                return int(mapping.get(int(round(v / 10.0) * 10), 0))

        for b in (page_data.get("blocks") or []):
            for ln in (b.get("lines") or []):
                bb = ln.get("bbox") or None
                if not bb:
                    continue
                try:
                    x0 = float(bb[0])
                except Exception:
                    continue
                indent_px = max(0.0, x0 - col0_left)
                ln["indent_px"] = float(indent_px)
                ln["indent_level"] = int(level(indent_px))

    def _build_toc_rows(self, page_data, columns, margins):
        blocks = page_data.get("blocks") or []
        col_left = columns[0]["x0"] if columns else margins["left"]
        col_right = columns[-1]["x1"] if columns else margins["right"]

        def pick_line_style(line, block):
            for ph in (line.get("phrases") or []):
                st = ph.get("style") or {}
                if st:
                    return dict(st)
            st = block.get("resolved_style") or block.get("style") or {}
            return dict(st) if isinstance(st, dict) else {}

        def style_score(style):
            if not isinstance(style, dict):
                return -1.0
            try:
                size = float(style.get("size") or style.get("font_size_pt") or 0.0)
            except Exception:
                size = 0.0
            flags = style.get("flags") or {}
            bonus = 0.0
            if flags.get("bold"):
                bonus += 0.6
            if flags.get("italic"):
                bonus += 0.3
            return size + bonus

        def infer_toc_role(label_text, marker, indent_level, style, page_value):
            text = (label_text or "").strip()
            lower = text.lower()
            flags = (style or {}).get("flags") or {}
            try:
                font_size = float((style or {}).get("size") or (style or {}).get("font_size_pt") or 0.0)
            except Exception:
                font_size = 0.0
            if lower in {"contents", "sommaire", "table of contents"}:
                return "toc_title"
            if re.fullmatch(r"\d{1,3}|[ivxlcdm]+", (page_value or "").strip(), flags=re.I) and not text:
                return "page_marker"
            if indent_level <= 0 and font_size >= 12.0:
                return "chapter_title"
            if indent_level <= 0 and flags.get("bold"):
                return "section_heading"
            if marker:
                return "subentry_marker"
            if indent_level >= 1:
                return "subentry"
            return "section_heading"

        flat_lines = []
        for b in blocks:
            for ln in (b.get("lines") or []):
                bb = ln.get("bbox") or None
                if not bb:
                    continue
                text = (ln.get("line_text") or ln.get("translated_text") or "").strip()
                if not text:
                    parts = []
                    for ph in (ln.get("phrases") or []):
                        tx = (ph.get("text") or ph.get("translated_text") or ph.get("texte") or "").strip()
                        if tx:
                            parts.append(tx)
                    text = " ".join(parts).strip()
                if not text:
                    continue
                try:
                    x0, y0, x1, y1 = map(float, bb)
                except Exception:
                    continue
                style = pick_line_style(ln, b)
                flat_lines.append(
                    {
                        "text": text,
                        "bbox": [x0, y0, x1, y1],
                        "y": y0,
                        "indent_level": int(ln.get("indent_level", 0) or 0),
                        "indent_px": float(ln.get("indent_px", 0.0) or 0.0),
                        "style": style,
                    }
                )

        flat_lines.sort(key=lambda r: (float(r.get("y", 0.0)), float((r.get("bbox") or [0, 0, 0, 0])[0])))

        merged = []
        page_num_right_x_candidates = []
        pending = []

        def flush_pending(page_value="", page_bbox=None):
            nonlocal pending
            if not pending:
                return
            label_text = " ".join((p.get("text") or "").strip() for p in pending if (p.get("text") or "").strip()).strip()
            marker, label_text = self._extract_leading_marker(label_text)
            best_style = {}
            if pending:
                best_style = max((p.get("style") or {} for p in pending), key=style_score, default={}) or {}
            role = infer_toc_role(
                label_text=label_text,
                marker=marker,
                indent_level=min(int(p.get("indent_level", 0) or 0) for p in pending),
                style=best_style,
                page_value=page_value,
            )
            merged.append(
                {
                    "y": float(pending[0].get("y", 0.0)),
                    "indent_level": min(int(p.get("indent_level", 0) or 0) for p in pending),
                    "indent_px": float(pending[0].get("indent_px", 0.0) or 0.0),
                    "marker": marker,
                    "label": label_text.strip(),
                    "page": (page_value or "").strip(),
                    "role": role,
                    "style": best_style,
                }
            )
            if page_bbox:
                page_num_right_x_candidates.append(float(page_bbox[2]))
            pending = []

        for entry in flat_lines:
            text = (entry.get("text") or "").strip()
            if re.fullmatch(r"\d{1,3}|[ivxlcdm]+", text, flags=re.I):
                flush_pending(text, page_bbox=entry.get("bbox"))
                continue
            pending.append(entry)

        flush_pending("", page_bbox=None)

        if page_num_right_x_candidates:
            page_num_right_x = float(statistics.median(page_num_right_x_candidates))
        else:
            page_num_right_x = col_right

        current_chapter = ""
        for idx, row in enumerate(merged):
            label = str(row.get("label") or "").strip()
            role = str(row.get("role") or "").strip().lower()
            m = re.match(r"^(\d+)\.\d+\b", label)
            if m:
                current_chapter = m.group(1)
            if role == "chapter_title":
                if not current_chapter:
                    for nxt in merged[idx + 1:]:
                        nm = re.match(r"^(\d+)\.\d+\b", str(nxt.get("label") or "").strip())
                        if nm:
                            current_chapter = nm.group(1)
                            break
                if current_chapter:
                    row["chapter_number"] = current_chapter

        tab_stops = {
            "page_num_right_x": page_num_right_x,
            "column_left_x": col_left,
            "column_right_x": col_right,
        }
        return merged, tab_stops

    def _join_tokens(self, tokens, skip_text=None):
        toks = sorted(tokens, key=lambda z: z["bbox"][0])
        parts = []
        for t in toks:
            tx = t.get("text", "").strip()
            if not tx:
                continue
            if skip_text and tx == skip_text:
                continue
            parts.append(tx)
        return " ".join(parts).strip()

    def _choose_style(self, tokens):
        for t in tokens:
            st = t.get("style") or {}
            if st:
                return st
        return {}

    def _extract_leading_marker(self, label):
        s = (label or "").strip()
        m = re.match(r"^([•■·\-\u2022])\s+(.*)$", s)
        if m:
            return m.group(1), m.group(2)
        return "", s
