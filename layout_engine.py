import fitz
import numpy as np

class LayoutEngine:
    def __init__(self):
        self.pixel_to_point = 72.0 / 150.0

    def optimize_layout(self, structure):
        """
        Recalcule les bboxes pour que les traductions ne se chevauchent pas.
        """
        print("  [Layout] Optimisation de la mise en page traduite...")
        
        # On traite les blocs de haut en bas
        blocks = structure["blocks"]
        blocks.sort(key=lambda b: b["bbox"][1]) # Tri vertical

        for i, block in enumerate(blocks):
            # 1. Calculer la nouvelle largeur nécessaire
            max_w_needed = 0
            for line in block["lines"]:
                for phrase in line["phrases"]:
                    # On estime la largeur avec PyMuPDF
                    # Note: On utilise Helvetica comme base de mesure
                    style = phrase["spans"][0]["style"]
                    fontname = "hebo" if style["flags"]["bold"] else "helv"
                    fs = style["size"] * self.pixel_to_point
                    
                    text_w_pts = fitz.get_text_length(phrase["texte"], fontname=fontname, fontsize=fs)
                    text_w_px = text_w_pts / self.pixel_to_point
                    
                    if text_w_px > max_w_needed:
                        max_w_needed = text_w_px
            
            # 2. Mise à jour de la bbox du bloc
            old_bbox = block["bbox"]
            new_width = max(old_bbox[2] - old_bbox[0], max_w_needed)
            block["bbox"] = [old_bbox[0], old_bbox[1], old_bbox[0] + int(new_width), old_bbox[3]]
            
            # Mise à jour des bboxes des lignes et phrases internes
            for line in block["lines"]:
                line["bbox"] = [block["bbox"][0], line["bbox"][1], block["bbox"][2], line["bbox"][3]]
                for phrase in line["phrases"]:
                    phrase["bbox"] = [line["bbox"][0], phrase["bbox"][1], line["bbox"][2], phrase["bbox"][3]]
                    for span in phrase.get("spans", []):
                        span["bbox"] = phrase["bbox"]

            # 3. ÉVITER LE CHEVAUCHEMENT VERTICAL
            # Si ce bloc agrandi touche le bloc suivant, on pousse le suivant vers le bas
            if i < len(blocks) - 1:
                next_block = blocks[i+1]
                margin = 10 # 10 pixels de sécurité
                if block["bbox"][3] + margin > next_block["bbox"][1]:
                    shift = (block["bbox"][3] + margin) - next_block["bbox"][1]
                    self._shift_block_vertically(next_block, shift)

        return structure

    def _shift_block_vertically(self, block, shift):
        block["bbox"][1] += int(shift)
        block["bbox"][3] += int(shift)
        for line in block["lines"]:
            line["bbox"][1] += int(shift)
            line["bbox"][3] += int(shift)
            for phrase in line["phrases"]:
                phrase["bbox"][1] += int(shift)
                phrase["bbox"][3] += int(shift)
                for span in phrase.get("spans", []):
                    span["bbox"][1] += int(shift)
                    span["bbox"][3] += int(shift)
