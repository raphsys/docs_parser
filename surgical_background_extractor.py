import os
import fitz
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from rapidocr_onnxruntime import RapidOCR

class SurgicalBackgroundExtractor:
    def __init__(self, dpi=150):
        self.dpi = dpi
        self.engine_ocr = RapidOCR()

    def remove_text_surgically(self, img, text_rects):
        """
        Supprime le texte lettre par lettre avec une précision au pixel.
        """
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        scale_px = self.dpi / 72

        for tr in text_rects:
            # 1. Position globale (bbox)
            x0, y0 = int(tr.x0 * scale_px), int(tr.y0 * scale_px)
            x1, y1 = int(tr.x1 * scale_px), int(tr.y1 * scale_px)
            
            x0, y0, x1, y1 = max(0, x0-2), max(0, y0-2), min(w, x1+2), min(h, y1+2)
            if x1 <= x0 or y1 <= y0: continue

            # 2. Reconnaissance de chaque lettre (Segmentation)
            roi = img[y0:y1, x0:x1]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Seuillage adaptatif pour isoler l encre (les lettres)
            # On cherche les formes sombres sur fond clair
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 15, 5)
            
            # 3. Analyse de l épaisseur et des pixels de la lettre
            # On utilise les composantes connexes pour isoler chaque glyphe
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            char_mask = np.zeros_like(binary)
            for i in range(1, num_labels): # On ignore le fond (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                # Filtrer le bruit (trop petit) ou les blocs trop gros (fond)
                if area > 2: 
                    # Ce label appartient à une lettre
                    char_mask[labels == i] = 255
            
            # 4. Masquage de couleur (Blanc par défaut pour le fond)
            # On dilate très légèrement (1px) pour couvrir l anti-aliasing et les empattements fins
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            final_char_mask = cv2.dilate(char_mask, kernel, iterations=1)
            
            # Application du masque chirurgical sur l image principale
            img[y0:y1, x0:x1][final_char_mask > 0] = [255, 255, 255]
            
        return img

    def process_pdf(self, input_path, output_path):
        print(f"--- Extraction Chirurgicale (Lettre par Lettre) : {os.path.basename(input_path)} ---")
        doc = fitz.open(input_path)
        res_doc = fitz.open()

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=self.dpi)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
            
            # 1. Collecte des positions globales
            text_rects = []
            for b in page.get_text("dict")["blocks"]:
                if "lines" in b: text_rects.append(fitz.Rect(b["bbox"]))
            
            ocr_res, _ = self.engine_ocr(img)
            if ocr_res:
                scale_pts = 72 / self.dpi
                for res in ocr_res:
                    c = res[0]
                    text_rects.append(fitz.Rect(min(p[0] for p in c)*scale_pts, min(p[1] for p in c)*scale_pts,
                                                max(p[0] for p in c)*scale_pts, max(p[1] for p in c)*scale_pts))

            # 2. Suppression chirurgicale
            clean_img = self.remove_text_surgically(img, text_rects)
            
            # 3. Génération du PDF
            new_page = res_doc.new_page(width=page.rect.width, height=page.rect.height)
            _, buf = cv2.imencode(".png", clean_img)
            new_page.insert_image(page.rect, stream=buf.tobytes())

        res_doc.save(output_path)
        res_doc.close()
        doc.close()
        print(f"SUCCESS : Fond extrait avec précision glyphe dans {output_path}")

if __name__ == "__main__":
    extractor = SurgicalBackgroundExtractor(dpi=200) # Augmentation DPI pour plus de finesse
    extractor.process_pdf("uploads/test_docintelligence-1.pdf", "ocr_results/ONLY_BACKGROUND.pdf")
