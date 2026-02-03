from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import os

model_id = "Qwen/Qwen2-VL-2B-Instruct"
save_directory = "./ai_models/qwen2vl"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

print(f"Téléchargement de {model_id} vers {save_directory}...")
print("Note : Ce modèle pèse environ 4.5 Go. Cela peut prendre du temps.")

# Téléchargement du processeur
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
processor.save_pretrained(save_directory)

# Téléchargement du modèle
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    torch_dtype="auto", 
    device_map="cpu"
)
model.save_pretrained(save_directory)

print(f"\n[SUCCÈS] Modèle sauvegardé dans {save_directory}")

