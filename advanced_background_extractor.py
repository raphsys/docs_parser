import os
import fitz
import cv2
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from sd_inpainting_engine import SmartInpainter

class AdvancedBackgroundExtractor:
    def __init__(self, dpi=150):
        self.dpi = dpi
        self.engine_ocr = RapidOCR()
        self.painter = SmartInpainter()

    def get_text_zones_px(self, page, img_bgr):
        """Récupère les bboxes en pixels."""
        zones = []
        scale = self.dpi / 72
        
        # Natif
        for b in page.get_text("dict")["blocks"]:
            if "lines" in b:
                z = [int(c * scale) for c in b["bbox"]]
                zones.append(z)
        
        # OCR
        ocr_res, _ = self.engine_ocr(img_bgr)
        if ocr_res:
            for res in ocr_res:
                c = res[0]
                z = [int(min(p[0] for p in c)), int(min(p[1] for p in c)),
                     int(max(p[0] for p in c)), int(max(p[1] for p in c))]
                zones.append(z)
        return zones

    def create_mask(self, shape, zones):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        for z in zones:
            # On dessine les bboxes sur le masque avec une petite dilatation
            cv2.rectangle(mask, (z[0]-2, z[1]-2), (z[2]+2, z[3]+2), 255, -1)
        return mask

    def process(self, input_pdf, output_pdf):
        doc = fitz.open(input_pdf)
        res_doc = fitz.open()

        for i in range(len(doc)):
            print(f"Traitement Page {i+1}...")
            page = doc[i]
            pix = page.get_pixmap(dpi=self.dpi)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 1. Détection
            zones = self.get_text_zones_px(page, img)
            mask = self.create_mask(img.shape, zones)
            
            # 2. IA Stable Diffusion Inpainting
            print(f"  > Initialisation du rendu Génératif (Stable Diffusion)...")
            clean_bg = self.painter.inpaint_document_image(img, mask, zones)
            
            # 3. PDF
            new_page = res_doc.new_page(width=page.rect.width, height=page.rect.height)
            _, buf = cv2.imencode(".png", clean_bg)
            new_page.insert_image(page.rect, stream=buf.tobytes())

        res_doc.save(output_pdf)
        print(f"SUCCESS : Fond généré par Stable Diffusion dans {output_pdf}")

if __name__ == "__main__":
    extractor = AdvancedBackgroundExtractor()
    extractor.process("uploads/test_docintelligence-1.pdf", "ocr_results/SD_BACKGROUND.pdf")
