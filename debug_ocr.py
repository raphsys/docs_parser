import fitz
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import os

# Config
MODEL_PATH = './ai_models/florence2-base'
PDF_PATH = 'uploads/test_docintelligence-1.pdf'

def debug_ocr():
    target_path = PDF_PATH
    if not os.path.exists(target_path):
        print(f"Fichier non trouvé: {target_path}")
        files = [f for f in os.listdir('uploads') if f.endswith('.pdf')]
        if files:
            target_path = os.path.join('uploads', files[0])
            print(f"Utilisation de {target_path} à la place.")
        else:
            return

    print(">>> Chargement Modèle...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Int8 optimization
    # model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    print(f">>> Traitement de {target_path}...")
    doc = fitz.open(target_path)
    page = doc[0] # Page 1
    
    # 1. Rendu Image
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    print(f"Image Size: {img.size}")

    # 2. Inférence IA (Code simplifié de ocr_server)
    task_prompt = '<OCR_WITH_REGION>'
    
    # Resize comme le serveur
    max_dim = 1024
    w, h = img.size
    scale = max_dim / max(w, h)
    img_resized = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    
    inputs = processor(text=task_prompt, images=img_resized, return_tensors="pt")
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=1,
        use_cache=True
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=img.size)
    
    print("\n--- RÉSULTATS BRUTS IA (Ce que Florence voit) ---")
    data = parsed_answer['<OCR_WITH_REGION>']
    for bbox, label in zip(data['bboxes'], data['labels']):
        print(f"AI Detected: '{label}' at {bbox}")

    print("\n--- TEXTE NATIF (Ce que le PDF contient) ---")
    native_blocks = page.get_text("dict")["blocks"]
    for b in native_blocks:
        if b["type"] == 0: # Text
            for l in b["lines"]:
                txt = "".join([s["text"] for s in l["spans"]])
                print(f"Native: '{txt.strip()}' at {b['bbox']}")

if __name__ == "__main__":
    debug_ocr()