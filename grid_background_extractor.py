import os
import fitz
import numpy as np
import cv2
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

class GridBackgroundExtractor:
    def __init__(self, dpi=150, tile_size_px=4):
        self.dpi = dpi
        self.tile_size = tile_size_px
        self.engine_ocr = RapidOCR()

    def get_thick_text_mask(self, img, text_rects):
        """
        Crée un masque binaire de l encre des textes, dilaté pour couvrir l épaisseur.
        """
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        scale_px = self.dpi / 72

        for tr in text_rects:
            # Coordonnées en pixels
            x0, y0 = int(tr.x0 * scale_px), int(tr.y0 * scale_px)
            x1, y1 = int(tr.x1 * scale_px), int(tr.y1 * scale_px)
            
            # Sécurité bornes
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 <= x0 or y1 <= y0: continue

            # Extraction de la zone de texte
            roi = img[y0:y1, x0:x1]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Détection de l encre (Seuillage adaptatif)
            # On détecte tout ce qui est plus sombre que le fond local
            binary_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 15, 4)
            
            # PRISE EN COMPTE DE L ÉPAISSEUR : Dilatation
            # On dilate de 2 pixels pour être sûr de manger tout le trait et l anti-aliasing
            kernel = np.ones((3,3), np.uint8)
            thick_roi = cv2.dilate(binary_roi, kernel, iterations=1)
            
            mask[y0:y1, x0:x1] = cv2.bitwise_or(mask[y0:y1, x0:x1], thick_roi)
            
        return mask

    def process_pdf(self, input_path, output_path):
        print(f"--- Extraction Ultra-Fine avec Épaisseur : {os.path.basename(input_path)} ---")
        doc = fitz.open(input_path)
        res_doc = fitz.open()

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=self.dpi)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 1. Collecte des bboxes de texte
            text_rects = []
            for b in page.get_text("dict")["blocks"]:
                if "lines" in b: text_rects.append(fitz.Rect(b["bbox"]))
            
            ocr_res, _ = self.engine_ocr(img)
            if ocr_res:
                scale_to_pts = 72 / self.dpi
                for res in ocr_res:
                    c = res[0]
                    text_rects.append(fitz.Rect(min(p[0] for p in c)*scale_to_pts, min(p[1] for p in c)*scale_to_pts,
                                                max(p[0] for p in c)*scale_to_pts, max(p[1] for p in c)*scale_to_pts))

            # 2. Génération du masque d épaisseur
            full_mask = self.get_thick_text_mask(img, text_rects)

            # 3. Application de la grille sur le masque
            h, w = img.shape[:2]
            clean_img = img.copy()

            for y in range(0, h, self.tile_size):
                for x in range(0, w, self.tile_size):
                    y1, x1 = min(y + self.tile_size, h), min(x + self.tile_size, w)
                    
                    # Si le carré de la grille touche au moins un pixel du masque d épaisseur
                    if np.any(full_mask[y:y1, x:x1] > 0):
                        # On blanchit le carré
                        clean_img[y:y1, x:x1] = [255, 255, 255]
            
            # 4. Export
            new_page = res_doc.new_page(width=page.rect.width, height=page.rect.height)
            _, buf = cv2.imencode(".png", clean_img)
            new_page.insert_image(page.rect, stream=buf.tobytes())

        res_doc.save(output_path)
        res_doc.close()
        doc.close()
        print(f"SUCCESS : Fond Ultra-Fin généré dans {output_path}")

if __name__ == "__main__":
    extractor = GridBackgroundExtractor(tile_size_px=4)
    extractor.process_pdf("uploads/test_docintelligence-1.pdf", "ocr_results/GRID_BACKGROUND.pdf")
