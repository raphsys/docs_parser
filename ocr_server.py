import os
import shutil
import torch
import uvicorn
import subprocess
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForCausalLM

# --- Optimisation CPU Haute Performance ---
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(os.cpu_count() or 4)
torch.set_num_interop_threads(1)
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

# --- Configuration ---
MODEL_PATH = './ai_models/florence2-base'
UPLOAD_DIR = 'uploads'
CONV_DIR = 'converted_pages'
RESULTS_DIR = 'ocr_results'

app = FastAPI(title="IA Document OCR (Hybrid Native/IA)")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# Dossiers de travail
for d in [UPLOAD_DIR, CONV_DIR, RESULTS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# Chargement du modèle
print(f"Démarrage du chargement expert (Florence-2 Base) depuis {MODEL_PATH}")
try:
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager"
    )
    print(">>> [SUCCÈS] Le modèle expert (Base) est chargé. <<<")
except Exception as e:
    print(f">>> [ERREUR FATALE] Échec du chargement : {e} <<<")
    model = None

from concurrent.futures import ThreadPoolExecutor

import re

def draw_bboxes(pil_img, content, output_path):
    """Dessine les bboxes sur l'image avec un parseur plus robuste (Regex)"""
    draw = ImageDraw.Draw(pil_img)
    lines = content.split('\n')
    drawn_count = 0
    
    # Regex pour capturer (x0, y0, x1, y1) de manière flexible
    bbox_pattern = re.compile(r'bbox=\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)')
    
    for line in lines:
        match = bbox_pattern.search(line)
        if match:
            try:
                x0, y0, x1, y1 = map(int, match.groups())
                # S'assurer que les coordonnées sont dans le bon ordre pour PIL
                left, top = min(x0, x1), min(y0, y1)
                right, bottom = max(x0, x1), max(y0, y1)
                
                # Vérifier que la box a une surface
                if right > left and bottom > top:
                    draw.rectangle([left, top, right, bottom], outline="red", width=3)
                    drawn_count += 1
            except Exception as e:
                print(f"Erreur dessin bbox sur ligne : {line} -> {e}")
    
    pil_img.save(output_path)
    print(f"Image sauvegardée : {output_path} ({drawn_count} bboxes dessinées)")

def process_single_page(args):
    """Extracteur Universel d'Objets : Texte, Tables, Dessins, Liens, Images, Formules"""
    i, page_data, force_ai, filename = args
    try:
        render_dpi = 120
        pix = page_data.get_pixmap(dpi=render_dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        scale_x = pix.width / page_data.rect.width
        scale_y = pix.height / page_data.rect.height
        all_elements = []

        # 1. LIENS (Natif)
        links = page_data.get_links()
        for link in links:
            b = link["from"]
            x0, y0, x1, y1 = int(b.x0*scale_x), int(b.y0*scale_y), int(b.x1*scale_x), int(b.y1*scale_y)
            uri = link.get("uri") or link.get("page")
            all_elements.append(f'[LINK] "{uri}" bbox=({x0},{y0},{x1},{y1}) conf=100.0')

        # 2. TABLEAUX (Multi-stratégies)
        for strategy in ["lines", "lines_strict", "text"]:
            try:
                tabs = page_data.find_tables(strategy=strategy)
                if tabs.tables:
                    for tab in tabs:
                        b = tab.bbox
                        x0, y0, x1, y1 = int(b[0]*scale_x), int(b[1]*scale_y), int(b[2]*scale_x), int(b[3]*scale_y)
                        table_data = tab.extract()
                        rows = [" | ".join([str(c).replace("\n", " ") if c else "" for c in r]) for r in table_data]
                        content = " [ROW] ".join(rows)
                        all_elements.append(f'[TABLE] "{content}" bbox=({x0},{y0},{x1},{y1}) conf=100.0')
                    break # On s'arrête dès qu'une stratégie trouve des tables
            except: continue

        # 3. DESSINS VECTORIELS & GRAPHISMES
        drawings = page_data.get_drawings()
        if drawings:
            # On regroupe les dessins par proximité pour identifier des graphiques
            for draw in drawings[:10]: # Limiter pour ne pas saturer
                b = draw["rect"]
                if b.width > 5 and b.height > 5:
                    x0, y0, x1, y1 = int(b.x0*scale_x), int(b.y0*scale_y), int(b.x1*scale_x), int(b.y1*scale_y)
                    all_elements.append(f'[GRAPHIC/DRAWING] "Vector_{draw["type"]}" bbox=({x0},{y0},{x1},{y1}) conf=100.0')

        # 4. IMAGES
        for img_info in page_data.get_images(full=True):
            for img_rect in page_data.get_image_rects(img_info[0]):
                x0, y0, x1, y1 = int(img_rect.x0*scale_x), int(img_rect.y0*scale_y), int(img_rect.x1*scale_x), int(img_rect.y1*scale_y)
                all_elements.append(f'[IMAGE] "ID_{img_info[0]}" bbox=({x0},{y0},{x1},{y1}) conf=100.0')

        # 5. TEXTE (Dict avec gestion intelligente des espaces)
        page_dict = page_data.get_text("dict")
        for block in page_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    # Joindre les spans avec des espaces si nécessaire
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    line_text = line_text.strip()
                    
                    if line_text:
                        b = line["bbox"]
                        x0, y0, x1, y1 = int(b[0]*scale_x), int(b[1]*scale_y), int(b[2]*scale_x), int(b[3]*scale_y)
                        all_elements.append(f'[TEXT] "{line_text}" bbox=({x0},{y0},{x1},{y1}) conf=100.0')

        # 6. IA EXPERT (Pour boucher les trous : Formules, Textes complexes)
        # On utilise l'IA systématiquement si force_ai est activé
        if force_ai or len(all_elements) < 10:
            ai_content = run_pro_ocr(img)
            final_content = "\n".join(all_elements) + "\n" + ai_content
        else:
            final_content = "\n".join(all_elements)

        # Générer l'image annotée
        vis_filename = f"vis_{filename}_p{i+1}.jpg"
        vis_path = os.path.join(RESULTS_DIR, vis_filename)
        draw_bboxes(img.copy(), final_content, vis_path)
        
        return {
            "page": i + 1, 
            "content": final_content, 
            "method": "universal_expert",
            "visual_url": f"/results/{vis_filename}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"page": i + 1, "content": f"Erreur : {e}", "method": "error"}

def process_pdf(file_path, force_ai=False):
    """Traitement parallèle des pages du PDF"""
    results = []
    filename = os.path.basename(file_path)
    try:
        doc = fitz.open(file_path)
        pages_to_process = [(i, doc[i], force_ai, filename) for i in range(len(doc))]
        
        # On utilise un pool de threads
        num_workers = 2 
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_single_page, pages_to_process))
            
        doc.close()
        # Trier par numéro de page
        results.sort(key=lambda x: x["page"])
    except Exception as e:
        print(f"Erreur PDF : {e}")
    return results

def run_pro_ocr(pil_image):
    if model is None: return "Modèle non chargé."
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    w, h = pil_image.size
    max_dim = 1024 # Résolution augmentée pour les petits objets
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        proc_img = pil_image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    else:
        proc_img = pil_image

    task_prompt = '<OCR_WITH_REGION>'
    try:
        inputs = processor(text=task_prompt, images=proc_img, return_tensors="pt")
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1536, # Très haut pour tout capturer
                num_beams=1,
                do_sample=False,
                use_cache=False 
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(w, h))
        
        result_data = parsed_answer.get(task_prompt, {})
        if not result_data: return "Aucun élément détecté."

        lines = []
        if 'quad_boxes' in result_data:
            for box, label in zip(result_data.get('quad_boxes', []), result_data.get('labels', [])):
                clean_label = label.replace("</s>", "").replace("<s>", "").strip()
                if not clean_label: continue
                xs, ys = box[0::2], box[1::2]
                lines.append(f'[AI_TEXT] "{clean_label}" bbox=({int(min(xs))},{int(min(ys))},{int(max(xs))},{int(max(ys))}) conf=98.0')
        
        return "\n".join(lines)
    except Exception as e:
        return f"Erreur IA : {e}"

@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...), force_ai: bool = False):
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    ext = os.path.splitext(file.filename)[1].lower()
    print(f"\n[REQUEST] {file.filename} ({ext}) | Force AI: {force_ai}")

    try:
        if ext == '.pdf':
            results = process_pdf(save_path, force_ai=force_ai)
        elif ext in ['.docx', '.doc', '.pptx', '.ppt']:
            # Conversion Office -> PDF puis traitement hybride
            subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', CONV_DIR, save_path], check=True)
            pdf_path = os.path.join(CONV_DIR, os.path.splitext(file.filename)[0] + ".pdf")
            results = process_pdf(pdf_path, force_ai=force_ai)
        else:
            # Images : IA pure
            img = Image.open(save_path).convert("RGB")
            content = run_pro_ocr(img)
            
            # Générer l'image annotée pour la visualisation
            filename = os.path.basename(save_path)
            vis_filename = f"vis_{filename}.jpg"
            vis_path = os.path.join(RESULTS_DIR, vis_filename)
            draw_bboxes(img.copy(), content, vis_path)
            
            results = [{
                "page": 1, 
                "content": content, 
                "method": "ai",
                "visual_url": f"/results/{vis_filename}"
            }]
        
        return JSONResponse(content={"results": results, "status": "success"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)