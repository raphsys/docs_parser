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
import re
from concurrent.futures import ThreadPoolExecutor
from structure_extractor import DocumentParser
from reconstructor import DocumentReconstructor

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

app = FastAPI(title="IA Document WYSIWYG")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

for d in [UPLOAD_DIR, CONV_DIR, RESULTS_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# Chargement du modèle
try:
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    print(">>> [SUCCÈS] Modèle IA chargé. <<<")
except Exception as e:
    print(f"Erreur chargement modèle: {e}")
    model = None

def draw_bboxes(pil_img, content, output_path):
    """Dessine les bboxes avec un code couleur par type d'objet"""
    draw = ImageDraw.Draw(pil_img)
    lines = content.split('\n')
    
    # Couleurs par type
    colors = {
        '[TABLE]': 'green',
        '[TEXT]': 'blue',
        '[AI_TEXT]': 'red',
        '[LINK]': 'cyan',
        '[IMAGE]': 'orange',
        '[GRAPHIC': 'orange', # Capture [GRAPHIC/DRAWING]
    }
    
    bbox_pattern = re.compile(r'bbox=\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)')
    
    for line in lines:
        match = bbox_pattern.search(line)
        if match:
            try:
                # Déterminer la couleur selon le tag au début de la ligne
                color = 'red' # Par défaut
                for tag, col in colors.items():
                    if tag in line:
                        color = col
                        break
                
                x0, y0, x1, y1 = map(int, match.groups())
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            except: continue
    
    pil_img.save(output_path)

def reconstruct_legacy_content(structure):
    lines = []
    # Infos Page
    dim = structure.get("dimensions", {})
    lay = structure.get("layout", {})
    m = lay.get("margins", {"top":0, "bottom":0, "left":0, "right":0})
    lines.append(f'[PAGE_INFO] num={structure.get("page_number")} size={dim.get("width")}x{dim.get("height")} rot={lay.get("rotation")} margins={m}')

    # Liens & Tableaux
    for link in structure.get("links", []):
        b = [int(c) for c in link["bbox"]]
        lines.append(f'[LINK] "{link.get("uri")}" bbox=({b[0]},{b[1]},{b[2]},{b[3]})')
    for table in structure.get("tables", []):
        b = [int(c) for c in table["bbox"]]
        lines.append(f'[TABLE]\n{table.get("markdown")}\nbbox=({b[0]},{b[1]},{b[2]},{b[3]})')

    # Blocs Texte
    for block in structure.get("blocks", []):
        if block.get("in_table") or block["type"] != "text": continue
        source = block.get("source", "native")
        tag = "AI_TEXT" if source == "ai" else "TEXT"
        role = block.get("role", "paragraph")
        
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            txt = "".join([s["text"] for s in spans]).strip()
            if not txt: continue
            lb = [int(c) for c in line["bbox"]]
            st = spans[0].get("style", {})
            style_str = f" role={role} lang={line.get('alignment')} style={{font:{st.get('font')}, size:{st.get('size',0):.1f}, color:{st.get('color')}}}"
            lines.append(f'[{tag}] "{txt}" bbox=({lb[0]},{lb[1]},{lb[2]},{lb[3]}){style_str}')
    return "\n".join(lines)

def run_pro_ocr(pil_image):
    if not model: return []
    w, h = pil_image.size
    max_dim = 1024 # Résolution optimisée pour Florence-2 Base
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
        parsed = processor.post_process_generation(generated_text, task=task_prompt, image_size=(w, h))
        
        res = parsed.get(task_prompt, {})
        results = []
        boxes = res.get('quad_boxes') or res.get('bboxes')
        labels = res.get('labels')
        if boxes and labels:
            for box, label in zip(boxes, labels):
                clean_label = label.replace("</s>", "").replace("<s>", "").strip()
                if not clean_label: continue
                
                if len(box) == 8: 
                    xs, ys = box[0::2], box[1::2]
                    bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                else: 
                    bbox = [int(c) for c in box]
                
                results.append({"label": clean_label, "bbox": bbox})
        return results
    except Exception as e:
        print(f"Erreur IA : {e}")
        return []

def process_single_page(args):
    i, page_data, force_ai, filename = args
    parser = DocumentParser()
    try:
        pix = page_data.get_pixmap(dpi=120)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        scale_x, scale_y = pix.width / page_data.rect.width, pix.height / page_data.rect.height

        # 1. Natif
        structure = parser.parse_page(page_data, i, scale_x=scale_x, scale_y=scale_y)
        
        # 2. IA et Fusion
        native_text_len = sum(len(s["text"]) for b in structure["blocks"] if b["type"]=="text" for l in b["lines"] for s in l["spans"])
        if force_ai or native_text_len < 500:
            ai_data = run_pro_ocr(img)
            from structure_extractor import VisualAttributeExtractor
            ve = VisualAttributeExtractor()
            native_rects = [fitz.Rect(b["bbox"]) for b in structure["blocks"] if b["type"] == "text"]
            
            for item in ai_data:
                ai_rect = fitz.Rect(item["bbox"])
                # Fusion : on n'ajoute que si pas de recouvrement majeur avec du texte natif existant
                if not any((ai_rect & nr).get_area() / ai_rect.get_area() > 0.4 for nr in native_rects):
                    style = ve.analyze(img, item["bbox"])
                    structure["blocks"].append({"type": "text", "bbox": item["bbox"], "source": "ai", "role": "title" if style['size']>30 else "paragraph",
                                   "lines": [{"bbox": item["bbox"], "lang": "unknown", "alignment": "left", "spans": [{"text": item["label"], "style": style, "bbox": item["bbox"]}]}]})

        final_content = reconstruct_legacy_content(structure)
        vis_fn = f"vis_{filename}_p{i+1}.jpg"
        draw_bboxes(img.copy(), final_content, os.path.join(RESULTS_DIR, vis_fn))
        return {"page": i+1, "content": final_content, "structure": structure, "visual_url": f"/results/{vis_fn}", "status": "success"}
    except Exception as e:
        return {"page": i+1, "content": f"Erreur: {e}", "status": "error"}

@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...), force_ai: bool = False):
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    ext = os.path.splitext(file.filename)[1].lower()
    print(f"\n[REQUEST] {file.filename} ({ext}) | Force AI: {force_ai}")

    try:
        # Traitement des fichiers Office
        if ext in ['.docx', '.doc', '.pptx', '.ppt']:
            # Conversion Office -> PDF
            subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', CONV_DIR, save_path], check=True)
            pdf_filename = os.path.splitext(file.filename)[0] + ".pdf"
            save_path = os.path.join(CONV_DIR, pdf_filename)
            ext = '.pdf'
            
        if ext == '.pdf':
            doc = fitz.open(save_path)
            with ThreadPoolExecutor(max_workers=2) as ex:
                results = list(ex.map(process_single_page, [(i, doc[i], force_ai, file.filename) for i in range(len(doc))]))
            return JSONResponse(content={"results": results, "status": "success"})
        else:
            img = Image.open(save_path).convert("RGB")
            ai_data = run_pro_ocr(img)
            # Construction structure image simple pour compatibilité
            from structure_extractor import VisualAttributeExtractor
            ve = VisualAttributeExtractor()
            blocks = []
            for item in ai_data:
                style = ve.analyze(img, item["bbox"])
                blocks.append({"type": "text", "bbox": item["bbox"], "source": "ai", "role": "paragraph",
                               "lines": [{"bbox": item["bbox"], "lang": "unknown", "alignment": "left", "spans": [{"text": item["label"], "style": style, "bbox": item["bbox"]}]}]})
            structure = {"page_number": 1, "dimensions": {"width": img.width, "height": img.height}, "layout": {"rotation": 0, "margins": {"top":0,"bottom":0,"left":0,"right":0}}, "source": "image_ai", "blocks": blocks}
            content = reconstruct_legacy_content(structure)
            vis_fn = f"vis_{file.filename}.jpg"
            draw_bboxes(img.copy(), content, os.path.join(RESULTS_DIR, vis_fn))
            return JSONResponse(content={"results": [{"page": 1, "content": content, "structure": structure, "visual_url": f"/results/{vis_fn}"}], "status": "success"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/reconstruct")
async def reconstruct_document(data: dict):
    try:
        recon = DocumentReconstructor()
        output_filename = "reconstructed_output.pdf"
        output_path = os.path.join(RESULTS_DIR, output_filename)
        recon.reconstruct(data, output_path)
        return JSONResponse(content={"status": "success", "pdf_url": f"/results/{output_filename}"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)