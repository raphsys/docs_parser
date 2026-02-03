import os
import shutil
import torch
import uvicorn
import subprocess
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageEnhance
from transformers import AutoProcessor, AutoModelForCausalLM
import re
from structure_extractor import DocumentParser
from reconstructor import DocumentReconstructor

# --- Optimisation CPU Haute Performance ---
# On utilise tous les coeurs disponibles
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
torch.set_num_threads(os.cpu_count() or 4)

# --- Configuration ---
MODEL_PATH = './ai_models/florence2-base'
UPLOAD_DIR, CONV_DIR, RESULTS_DIR = 'uploads', 'converted_pages', 'ocr_results'
TARGET_DPI = 200  # Résolution de rendu pour une précision maximale

app = FastAPI(title="IA Document WYSIWYG")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

for d in [UPLOAD_DIR, CONV_DIR, RESULTS_DIR]:
    if not os.path.exists(d): os.makedirs(d)

# Chargement du modèle avec Optimisation Int8
try:
    print(f">>> Chargement du modèle depuis {MODEL_PATH}...")
    import transformers.modeling_utils
    transformers.modeling_utils.PreTrainedModel._supports_sdpa = False
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager"
    )
    # ACCÉLÉRATION CPU : Divise le temps par 2
    model = torch.quantization.quantize_dynamic(_model, {torch.nn.Linear}, dtype=torch.qint8).eval()
    print(">>> [SUCCÈS] Modèle IA chargé et optimisé (Int8). <<<")
except Exception as e:
    print(f"Erreur chargement modèle: {e}")
    model = None

def draw_bboxes(pil_img, content, output_path):
    draw = ImageDraw.Draw(pil_img)
    colors = {'[TABLE]': 'green', '[TEXT]': 'blue', '[AI_TEXT]': 'red', '[LINK]': 'cyan'}
    bbox_pattern = re.compile(r'bbox=\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)')
    for line in content.split('\n'):
        match = bbox_pattern.search(line)
        if match:
            try:
                color = 'red'
                for tag, col in colors.items():
                    if tag in line: color = col; break
                x0, y0, x1, y1 = map(int, match.groups())
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            except: continue
    pil_img.save(output_path)

def reconstruct_legacy_content(structure):
    lines = []
    dim, lay = structure.get("dimensions", {}), structure.get("layout", {})
    m = lay.get("margins", {"top":0, "bottom":0, "left":0, "right":0})
    lines.append(f'[PAGE_INFO] num={structure.get("page_number")} size={dim.get("width")}x{dim.get("height")} rot={lay.get("rotation")} margins={m}')
    for link in structure.get("links", []):
        b = [int(c) for c in link["bbox"]]; lines.append(f'[LINK] "{link.get("uri")}" bbox=({b[0]},{b[1]},{b[2]},{b[3]})')
    for table in structure.get("tables", []):
        b = [int(c) for c in table["bbox"]]; lines.append(f'[TABLE]\n{table.get("markdown")}\nbbox=({b[0]},{b[1]},{b[2]},{b[3]})')
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
    
    # 0. Amélioration de l'image (Précision Stabilisée)
    pil_image = ImageEnhance.Sharpness(pil_image).enhance(1.5)
    pil_image = ImageEnhance.Contrast(pil_image).enhance(1.1)
    
    w_orig, h_orig = pil_image.size
    image_batch, offsets, scales = [], [], []
    
    # 1. Global (2048px : Ultra-Précision)
    max_dim_g = 2048
    s_g = max_dim_g / max(w_orig, h_orig) if max(w_orig, h_orig) > max_dim_g else 1.0
    img_g = pil_image.resize((int(w_orig*s_g), int(h_orig*s_g)), Image.Resampling.BILINEAR) if s_g < 1.0 else pil_image
    image_batch.append(img_g); offsets.append((0, 0)); scales.append(s_g)
    
    # 2. Tuiles 2x2 (1024px avec recouvrement)
    if max(w_orig, h_orig) > 1200:
        max_dim_t = 1024
        hw, hh, ov = w_orig // 2, h_orig // 2, 100
        coords = [(0,0,hw+ov,hh+ov), (hw-ov,0,w_orig,hh+ov), (0,hh-ov,hw+ov,h_orig), (hw-ov,hh-ov,w_orig,h_orig)]
        for x1, y1, x2, y2 in coords:
            tile = pil_image.crop((x1, y1, x2, y2))
            s_t = max_dim_t / max(tile.size) if max(tile.size) > max_dim_t else 1.0
            if s_t < 1.0: tile = tile.resize((int(tile.width*s_t), int(tile.height*s_t)), Image.Resampling.BILINEAR)
            image_batch.append(tile); offsets.append((x1, y1)); scales.append(s_t)

    results = []
    try:
        task = '<OCR_WITH_REGION>'
        inputs = processor(text=[task]*len(image_batch), images=image_batch, return_tensors="pt", padding=True)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=1024, num_beams=1, do_sample=False, use_cache=False)
        texts = processor.batch_decode(ids, skip_special_tokens=False)
        
        for idx, gen_text in enumerate(texts):
            parsed = processor.post_process_generation(gen_text, task=task, image_size=image_batch[idx].size)
            raw = parsed.get(task, {})
            off_x, off_y = offsets[idx]
            s = scales[idx]
            boxes = raw.get('quad_boxes') or raw.get('bboxes')
            labels = raw.get('labels')
            if boxes and labels:
                for box, label in zip(boxes, labels):
                    txt = label.replace("</s>","").replace("<s>","").strip()
                    if not txt: continue
                    b = [int(min(box[0::2])), int(min(box[1::2])), int(max(box[0::2])), int(max(box[1::2]))] if len(box)==8 else [int(c) for c in box]
                    real_bbox = [int(b[0]/s)+off_x, int(b[1]/s)+off_y, int(b[2]/s)+off_x, int(b[3]/s)+off_y]
                    results.append({"label": txt, "bbox": real_bbox})
    except Exception as e: print(f"Erreur IA : {e}")
    
    # NMS Avancé : Supprime les doublons et les sous-chaînes parasites (ex: 'ems' vs 'systems')
    final = []
    # On trie par longueur décroissante pour garder les mots complets
    results.sort(key=lambda x: len(x['label']), reverse=True)
    for cand in results:
        is_dup = False
        cb = cand['bbox']
        c_area = float((cb[2]-cb[0])*(cb[3]-cb[1]))
        if c_area <= 0: continue
        for ex in final:
            eb = ex['bbox']
            inter = max(0, min(eb[2],cb[2])-max(eb[0],cb[0])) * max(0, min(eb[3],cb[3])-max(eb[1],cb[1]))
            # Si recouvrement majeur (>60%)
            if inter / c_area > 0.6:
                # Si c'est une sous-chaîne d'un texte déjà retenu (plus long)
                if cand['label'].lower() in ex['label'].lower():
                    is_dup = True; break
        if not is_dup: final.append(cand)
    return final

def process_single_page(args):
    i, page_data, force_ai, filename = args
    parser = DocumentParser()
    try:
        print(f"--- Page {i+1} ---")
        pix = page_data.get_pixmap(dpi=TARGET_DPI)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        sx, sy = pix.width / page_data.rect.width, pix.height / page_data.rect.height
        structure = parser.parse_page(page_data, i, scale_x=sx, scale_y=sy)
        
        # On lance l'IA si forcé ou si peu de texte natif
        native_len = sum(len(s["text"]) for b in structure["blocks"] if b["type"]=="text" for l in b["lines"] for s in l["spans"])
        if force_ai or native_len < 600:
            print(f"  > Inférence IA...")
            ai_data = run_pro_ocr(img)
            from structure_extractor import VisualAttributeExtractor
            ve = VisualAttributeExtractor(); blocks = []
            for item in ai_data:
                ai_rect = fitz.Rect(item["bbox"])
                if ai_rect.get_area() <= 0: continue
                # Fusion : on n'ajoute que si pas de recouvrement majeur avec du texte natif
                if not any((ai_rect & r).get_area() / ai_rect.get_area() > 0.5 for r in [fitz.Rect(b["bbox"]) for b in structure["blocks"] if b["type"] == "text"]):
                    style = ve.analyze(img, item["bbox"])
                    structure["blocks"].append({"type": "text", "bbox": item["bbox"], "source": "ai", "role": "paragraph",
                                   "lines": [{"bbox": item["bbox"], "lang": "unknown", "alignment": "left", "spans": [{"text": item["label"], "style": style, "bbox": item["bbox"]}]}]})
        
        content = reconstruct_legacy_content(structure)
        vis_path = os.path.join(RESULTS_DIR, f"vis_{filename}_p{i+1}.jpg")
        draw_bboxes(img.copy(), content, vis_path)
        return {"page": i+1, "content": content, "structure": structure, "visual_url": f"/results/vis_{filename}_p{i+1}.jpg", "status": "success"}
    except Exception as e: 
        print(f"Erreur Page {i+1}: {e}")
        return {"page": i+1, "content": f"Erreur: {e}", "status": "error"}

@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...), force_ai: bool = False):
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    ext = os.path.splitext(file.filename)[1].lower()
    print(f"\n[REQUEST] {file.filename} | Force AI: {force_ai}")
    try:
        if ext in ['.docx', '.doc', '.pptx', '.ppt']:
            subprocess.run(['libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', CONV_DIR, save_path], check=True)
            save_path = os.path.join(CONV_DIR, os.path.splitext(file.filename)[0] + ".pdf"); ext = '.pdf'
        if ext == '.pdf':
            doc = fitz.open(save_path)
            # SÉQUENTIEL pour éviter Timeout
            results = [process_single_page((i, doc[i], force_ai, file.filename)) for i in range(len(doc))]
            return JSONResponse(content={"results": results, "status": "success"})
        else:
            img = Image.open(save_path).convert("RGB")
            ai_data = run_pro_ocr(img)
            from structure_extractor import VisualAttributeExtractor
            ve = VisualAttributeExtractor(); blocks = []
            for item in ai_data:
                style = ve.analyze(img, item["bbox"])
                blocks.append({"type": "text", "bbox": item["bbox"], "source": "ai", "role": "paragraph",
                               "lines": [{"bbox": item["bbox"], "lang": "unknown", "alignment": "left", "spans": [{"text": item["label"], "style": style, "bbox": item["bbox"]}]}]})
            structure = {"page_number": 1, "dimensions": {"width": img.width, "height": img.height}, "layout": {"rotation": 0, "margins": {"top":0,"bottom":0,"left":0,"right":0}}, "source": "image_ai", "blocks": blocks}
            content = reconstruct_legacy_content(structure)
            vis_fn = f"vis_{file.filename}.jpg"
            vis_path = os.path.join(RESULTS_DIR, vis_fn)
            draw_bboxes(img.copy(), content, vis_path)
            return JSONResponse(content={"results": [{"page": 1, "content": content, "structure": structure, "visual_url": f"/results/{vis_fn}"}], "status": "success"})
    except Exception as e: return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/reconstruct")
async def reconstruct_document(data: dict):
    try:
        recon = DocumentReconstructor(); output_path = os.path.join(RESULTS_DIR, "reconstructed_output.pdf")
        recon.reconstruct(data, output_path)
        return JSONResponse(content={"status": "success", "pdf_url": f"/results/reconstructed_output.pdf"})
    except Exception as e: return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
