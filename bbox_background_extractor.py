import os
import fitz
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

def extract_background_with_bboxes(input_path, output_path, dpi=150):
    """
    Efface le texte en appliquant des masques blancs sur les bboxes originales.
    Utilise le PDF source directement pour garantir ZERO décalage.
    """
    print(f"--- Extraction du fond par masquage BBox : {os.path.basename(input_path)} ---")
    
    # 1. Ouvrir le document original
    doc = fitz.open(input_path)
    engine_ocr = RapidOCR()
    
    for page in doc:
        # A. Détection du texte NATIF
        # On récupère les bboxes de chaque bloc de texte
        native_blocks = page.get_text("dict")["blocks"]
        for b in native_blocks:
            if "lines" in b:
                # On applique un masque blanc sur la zone
                page.draw_rect(b["bbox"], color=None, fill=(1, 1, 1), overlay=True)
        
        # B. Détection du texte OCR (incrusté dans les images)
        # On doit faire un rendu image pour l'OCR
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_result, _ = engine_ocr(np.array(img))
        
        if ocr_result:
            # Facteur de conversion Pixels -> Points PDF
            scale = 72 / dpi
            for res in ocr_result:
                coords = res[0]
                # Coordonnées OCR (pixels) vers points PDF
                x0 = min(p[0] for p in coords) * scale
                y0 = min(p[1] for p in coords) * scale
                x1 = max(p[0] for p in coords) * scale
                y1 = max(p[1] for p in coords) * scale
                
                # Rectangle de masquage
                rect = fitz.Rect(x0, y0, x1, y1)
                page.draw_rect(rect, color=None, fill=(1, 1, 1), overlay=True)

    # 2. Sauvegarder le résultat
    # On utilise garbage=4 pour nettoyer les objets texte devenus invisibles
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    print(f"SUCCESS : Fond généré avec masques BBox dans {output_path}")

if __name__ == "__main__":
    if not os.path.exists('ocr_results'): os.makedirs('ocr_results')
    extract_background_with_bboxes("uploads/test_docintelligence-1.pdf", "ocr_results/ONLY_BACKGROUND.pdf")
