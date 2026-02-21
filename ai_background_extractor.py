import fitz
import numpy as np
import cv2
from rapidocr_onnxruntime import RapidOCR
from sd_inpainting_engine import SmartInpainter

class AIBackgroundExtractor:
    def __init__(self, dpi=150):
        self.dpi = dpi
        self.engine_ocr = RapidOCR()
        self.inpainter = SmartInpainter()

    def get_text_zones(self, page, img_bgr):
        """Identifie toutes les bboxes de texte (Natif + OCR)."""
        zones = []
        # A. Natif
        for b in page.get_text("dict")["blocks"]:
            if "lines" in b: zones.append(fitz.Rect(b["bbox"]))
        
        # B. OCR (avec conversion coord)
        ocr_res, _ = self.engine_ocr(img_bgr)
        if ocr_res:
            scale = 72 / self.dpi
            for res in ocr_res:
                c = res[0]
                zones.append(fitz.Rect(min(p[0] for p in c)*scale, min(p[1] for p in c)*scale,
                                       max(p[0] for p in c)*scale, max(p[1] for p in c)*scale))
        return zones

    def create_transparent_holes(self, img_bgr, zones):
        """Crée une image RGBA où les zones de texte sont transparentes (Alpha=0)."""
        # Conversion BGR -> RGBA
        img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
        scale_px = self.dpi / 72
        
        for z in zones:
            x0, y0 = int(z.x0 * scale_px), int(z.y0 * scale_px)
            x1, y1 = int(z.x1 * scale_px), int(z.y1 * scale_px)
            # Sécurité bornes
            h, w = img_rgba.shape[:2]
            x0, y0, x1, y1 = max(0, x0-1), max(0, y0-1), min(w, x1+1), min(h, y1+1)
            # Mise à transparence (Alpha = 0)
            img_rgba[y0:y1, x0:x1, 3] = 0
            
        return img_rgba

    def process_document(self, pdf_path, output_path):
        doc = fitz.open(pdf_path)
        res_doc = fitz.open()

        for i in range(len(doc)):
            print(f"Extraction Fond Page {i+1}...")
            page = doc[i]
            
            # 1. Image de base
            pix = page.get_pixmap(dpi=self.dpi)
            img_bgr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
            
            # 2. Identification des zones
            zones = self.get_text_zones(page, img_bgr)
            
            # 3. Création du fond avec transparence (ÉTAPE DEMANDÉE)
            img_with_holes = self.create_transparent_holes(img_bgr, zones)
            
            # 4. Inpainting sur les zones transparentes
            print("  > Reconstruction du fond par IA (Inpainting)...")
            alpha = img_with_holes[:, :, 3]
            mask = (alpha == 0).astype(np.uint8) * 255
            final_bg_bgr = self.inpainter.inpaint_document_image(img_bgr, mask_gray=mask)
            
            # 5. Création PDF
            new_page = res_doc.new_page(width=page.rect.width, height=page.rect.height)
            _, buf = cv2.imencode(".png", final_bg_bgr)
            new_page.insert_image(page.rect, stream=buf.tobytes())

        res_doc.save(output_path)
        res_doc.close()
        doc.close()
        print(f"SUCCESS : Fond reconstitué sauvé dans {output_path}")

if __name__ == "__main__":
    cleaner = AIBackgroundExtractor()
    cleaner.process_document("uploads/test_docintelligence-1.pdf", "ocr_results/AI_RECONSTRUCTED_BACKGROUND.pdf")
