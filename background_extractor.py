import os
import fitz
import numpy as np
import cv2
from PIL import Image, ImageDraw
from io import BytesIO
from rapidocr_onnxruntime import RapidOCR

def remove_ink_from_image(image_bytes, text_rects, img_rect_fitz):
    """
    Analyse l'image pour identifier les pixels de texte et les blanchir
    en suivant la forme des lettres.
    """
    # 1. Charger l'image avec OpenCV
    nparray = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
    if img is None: return image_bytes
    
    h, w = img.shape[:2]
    
    for tr in text_rects:
        # Calcul de l'intersection entre l'image et le rectangle de texte
        intersect = img_rect_fitz & tr
        if intersect.get_area() > 0:
            # Coordonnées locales
            lx0 = int((intersect.x0 - img_rect_fitz.x0) * (w / img_rect_fitz.width))
            ly0 = int((intersect.y0 - img_rect_fitz.y0) * (h / img_rect_fitz.height))
            lx1 = int((intersect.x1 - img_rect_fitz.x0) * (w / img_rect_fitz.width))
            ly1 = int((intersect.y1 - img_rect_fitz.y0) * (h / img_rect_fitz.height))
            
            # Sécurité des bornes
            lx0, ly0 = max(0, lx0), max(0, ly0)
            lx1, ly1 = min(w, lx1), min(h, ly1)
            
            if lx1 <= lx0 or ly1 <= ly0: continue
            
            # Extraction de la zone de texte
            roi = img[ly0:ly1, lx0:lx1]
            
            # 2. Détection de la silhouette (Thresholding)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # On utilise un seuillage adaptatif pour détecter les lettres quel que soit le fond
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # 3. Nettoyage du masque (suppression du bruit)
            kernel = np.ones((2,2), np.uint8)
            mask = cv2.dilate(thresh, kernel, iterations=1)
            
            # 4. Application du masque : on met à blanc UNIQUEMENT les pixels des lettres
            roi[mask > 0] = [255, 255, 255]
            img[ly0:ly1, lx0:lx1] = roi

    # Ré-encodage de l'image
    _, buffer = cv2.imencode(".png", img)
    return buffer.tobytes()

def extract_clean_background(pdf_path, output_path, dpi=150):
    print(f"--- Extraction du fond (Masquage par Lettre) : {os.path.basename(pdf_path)} ---")
    
    doc = fitz.open(pdf_path)
    res_doc = fitz.open()
    engine_ocr = RapidOCR()
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        
        # 1. Détection OCR pour localiser les lettres dans les images
        pix = page.get_pixmap(dpi=dpi)
        img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_result, _ = engine_ocr(np.array(img_pil))
        
        text_rects = []
        # Ajouter rectangles natifs
        for b in page.get_text("dict")["blocks"]:
            if "lines" in b: text_rects.append(fitz.Rect(b["bbox"]))
        # Ajouter rectangles OCR
        if ocr_result:
            scale = 72 / dpi
            for res in ocr_result:
                c = res[0]
                text_rects.append(fitz.Rect(min(p[0] for p in c)*scale + page.rect.x0, 
                                            min(p[1] for p in c)*scale + page.rect.y0,
                                            max(p[0] for p in c)*scale + page.rect.x0, 
                                            max(p[1] for p in c)*scale + page.rect.y0))

        # 2. Création de la page
        new_page = res_doc.new_page(width=page.rect.width, height=page.rect.height)
        
        # 3. Traitement des images avec masquage par lettre
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            for r in page.get_image_rects(xref):
                # On "nettoie" l'image pixel par pixel là où il y a du texte
                cleaned_bytes = remove_ink_from_image(image_bytes, text_rects, r)
                new_page.insert_image(r, stream=cleaned_bytes)
                
        # 4. Copie des tracés vectoriels (non texte)
        for d in page.get_drawings():
            # On ignore les tracés qui ressemblent à du texte (petites formes denses)
            if d["rect"].width > 2 and d["rect"].height > 2:
                new_page.draw_rect(d["rect"], color=d.get("color"), fill=d.get("fill"), width=d.get("width", 1))

    res_doc.save(output_path)
    res_doc.close()
    doc.close()
    print(f"SUCCESS : Fond avec lettres masquées généré dans {output_path}")

if __name__ == "__main__":
    extract_clean_background("uploads/test_docintelligence-1.pdf", "ocr_results/ONLY_BACKGROUND.pdf")
