from transformers import AutoModelForCausalLM, AutoProcessor
import os

model_id = "microsoft/Florence-2-large"
save_directory = "./ai_models/florence2"

print(f"Downloading {model_id} to {save_directory}...")
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, attn_implementation="eager")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
print("Download complete.")
