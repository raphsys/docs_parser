import os
import sys
import torch
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from janus.models import MultiModalityCausalLM, VLChatProcessor

# Configuration du modèle
# Nous utilisons Janus-Pro-1B qui est plus léger et très performant pour la vision
MODEL_NAME = 'deepseek-ai/Janus-Pro-1B'
OUTPUT_DIR = 'ocr_results'

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def create_dummy_image(filename="test_sample.jpg"):
    """Crée une image simple avec du texte pour le test."""
    print(f"Création d'une image de test : {filename}")
    img = Image.new('RGB', (800, 600), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    
    text = (
        "Bonjour, ceci est un test de DeepSeek Janus-Pro.\n"
        "Nous testons la détection de mots, phrases et lettres.\n"
        "L'OCR est une technologie clé.\n"
        "Code: 12345-ABCD"
    )
    
    d.text((50, 50), text, fill=(0, 0, 0), font=font)
    img.save(filename)
    return filename

def load_model():
    """Charge le modèle et le processor."""
    print(f"Chargement du modèle {MODEL_NAME}...")
    try:
        # Utilisation des classes spécifiques de Janus
        processor = VLChatProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer = processor.tokenizer

        model = MultiModalityCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        # Force la conversion en float32 pour être sûr
        model = model.to(torch.float32)
        print(f"Modèle chargé. Dtype: {model.dtype}, Device: {model.device}")
        
        return model, processor
        
        return model, processor
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

def run_inference_on_image(model, processor, pil_image, context_text=""):
    """Fonction helper pour exécuter l'inférence sur une image PIL."""
    
    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>\nTranscribe the text in this image exactly as it appears.",
            "images": [pil_image], # Note: le processor attend souvent une liste d'images dans le contenu ou à part
        },
        {"role": "Assistant", "content": ""},
    ]

    # Préparation des entrées
    # L'API de VLChatProcessor peut varier. Essayons l'approche standard suggérée par DeepSeek Janus
    prepare_inputs = processor(
        conversations=conversation,
        images=[pil_image],
        force_batchify=True
    ).to(model.device, dtype=model.dtype)

    # Gestion des embeddings d'images
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    with torch.no_grad():
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

    # Décodage
    answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    # Nettoyage
    # L'output contient souvent tout le contexte. On essaie d'extraire la réponse.
    # Mais ici on décode juste la génération.
    # Si inputs_embeds est utilisé, model.generate retourne souvent juste la suite ou tout ?
    # Avec language_model.generate, c'est souvent juste la suite si on ne passe pas input_ids.
    # Mais attention, model.language_model est un LlamaForCausalLM standard.
    
    return answer

def process_pdf(pdf_path, model, processor):
    """Traite un fichier PDF page par page."""
    print(f"Traitement du PDF : {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num, page in enumerate(doc):
        print(f" - Page {page_num + 1}/{len(doc)}...")
        
        # Convertir la page PDF en image (pixmap)
        pix = page.get_pixmap(dpi=150) # 150 DPI est souvent suffisant pour l'OCR
        
        # Convertir en Image PIL
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Exécuter l'OCR
        page_text = run_inference_on_image(model, processor, img)
        
        header = f"\n--- Page {page_num + 1} ---\n"
        full_text += header + page_text + "\n"
        print(f"   > Texte extrait ({len(page_text)} caractères)")

    return full_text

def run_ocr(file_path):
    """Point d'entrée principal pour l'OCR (Image ou PDF)."""
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} n'existe pas.")
        return

    ensure_output_dir()
    model, processor = load_model()

    file_ext = os.path.splitext(file_path)[1].lower()
    
    result_text = ""
    
    if file_ext == '.pdf':
        result_text = process_pdf(file_path, model, processor)
    else:
        # Supposons que c'est une image
        print(f"Traitement de l'image : {file_path}")
        try:
            img = Image.open(file_path).convert("RGB")
            result_text = run_inference_on_image(model, processor, img)
        except Exception as e:
            print(f"Erreur lors de l'ouverture de l'image : {e}")
            return

    print("\n--- Résultat Final de l'OCR ---")
    print(result_text)
    print("-------------------------------")
    
    # Sauvegarde
    output_filename = f"result_{os.path.basename(file_path)}.txt"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, "w") as f:
        f.write(result_text)
    print(f"Résultat sauvegardé dans : {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "test_sample.jpg"
        if not os.path.exists(input_file):
            create_dummy_image(input_file)
            
    run_ocr(input_file)
