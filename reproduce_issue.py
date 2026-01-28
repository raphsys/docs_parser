import os
import torch
import fitz  # PyMuPDF
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_PATH = './ai_models/florence2-base'
PDF_PATH = 'uploads/test_docintelligence-1.pdf'

def run():
    print(f"Loading model from {MODEL_PATH}...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager"
    )
    print("Model loaded.")

    print(f"Processing {PDF_PATH}...")
    doc = fitz.open(PDF_PATH)
    page = doc[0] # Page 1
    
    # Simulate the server logic
    pix = page.get_pixmap(dpi=150)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    print(f"Image extracted. Size: {img.size}, Mode: {img.mode}")

    run_pro_ocr(model, processor, img)

def run_pro_ocr(model, processor, pil_image):
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    
    w, h = pil_image.size
    max_dim = 1024 
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        proc_img = pil_image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    else:
        proc_img = pil_image

    task_prompt = '<OCR_WITH_REGION>'
    print("Running inference...")
    try:
        inputs = processor(text=task_prompt, images=proc_img, return_tensors="pt")
        print("Processor inputs created.")
        
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=1,
                do_sample=False,
                use_cache=False
            )
        print("Generation complete.")
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        print(f"Generated text: {generated_text[:100]}...")
        
        parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(w, h))
        print("Post-processing complete.")
        print("Result keys:", parsed_answer.keys())
        
    except Exception as e:
        print(f"CAUGHT EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run()
