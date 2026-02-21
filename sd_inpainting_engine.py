import cv2
from typing import Union
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageStat
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

def _load_image(img: Union[str, Image.Image]) -> Image.Image:
    if isinstance(img, Image.Image): return img
    return Image.open(img)

def _fit_into_square_with_padding(img: Image.Image, size: int = 512, pad_mode: str = "edge"):
    img = img.convert("RGBA") if img.mode in ("RGBA", "LA") else img.convert("RGB")
    w, h = img.size
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img_rs = img.resize((new_w, new_h), resample=Image.LANCZOS)
    
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    pad_right = size - new_w - pad_left
    pad_bottom = size - new_h - pad_top

    # --- LOGIQUE DE PADDING PAR REPLICATION (Fidelité de texture) ---
    base = np.array(img_rs)
    out = np.zeros((size, size, base.shape[2]), dtype=base.dtype)
    y0, x0 = pad_top, pad_left
    out[y0:y0+new_h, x0:x0+new_w] = base
    
    # Replicate edges to guide the diffusion process
    if new_h > 0 and new_w > 0:
        out[:y0, x0:x0+new_w] = base[0:1, :, :]
        out[y0+new_h:, x0:x0+new_w] = base[-1:, :, :]
        out[:, :x0] = out[:, x0:x0+1]
        out[:, x0+new_w:] = out[:, x0+new_w-1:x0+new_w]
        
    img_sq = Image.fromarray(out)
    return img_sq, (pad_left, pad_top, pad_right, pad_bottom), (new_w, new_h)

def _unpad_and_resize_back(img_sq, pad, target_size, resized_wh):
    pl, pt, pr, pb = pad
    cropped = img_sq.crop((pl, pt, pl + resized_wh[0], pt + resized_wh[1]))
    return cropped.resize(target_size, resample=Image.LANCZOS)

def _mask_from_alpha(img_rgba, threshold=10):
    a = np.array(img_rgba.split()[-1])
    mask = (a <= threshold).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")

def _refine_mask(mask, blur_px=8, dilate_px=3):
    m = mask.convert("L")
    if dilate_px > 0: m = m.filter(ImageFilter.MaxFilter(size=1 + 2 * dilate_px))
    if blur_px > 0: m = m.filter(ImageFilter.GaussianBlur(radius=blur_px / 2))
    return m

class SmartInpainter:
    def __init__(self):
        self.pipe = None

    def lazy_load(self):
        if self.pipe is None:
            model_id = "stable-diffusion-v1-5/stable-diffusion-inpainting"
            print(f"Chargement du Moteur de Diffusion de Texture sur CPU...")
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id, torch_dtype=torch.float32, safety_checker=None
            )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.to("cpu")
            self.pipe.enable_attention_slicing()

    def get_texture_prompt(self, img_pil):
        # Analyse statistique pour guider la diffusion
        stat = ImageStat.Stat(img_pil.convert("RGB"))
        mean_color = [int(c) for c in stat.mean]
        # On evite le blanc pur ou le noir pur dans le prompt pour forcer la texture
        return f"seamless organic {mean_color} paper texture, high resolution background, matching grain, no text, consistent lighting"

    def inpaint_document_image(self, img_bgr, mask_gray=None, text_zones=None):
        # 1. Préparation des entrées
        if img_bgr is None or not isinstance(img_bgr, np.ndarray):
            raise ValueError("img_bgr must be a valid numpy array (BGR image).")
        if img_bgr.ndim != 3 or img_bgr.shape[2] not in (3, 4):
            raise ValueError("img_bgr must have shape HxWx3 or HxWx4.")

        if img_bgr.shape[2] == 4:
            orig_bgr = img_bgr[:, :, :3]
        else:
            orig_bgr = img_bgr.copy()

        orig_pil = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))

        if mask_gray is None:
            if img_bgr.shape[2] == 4:
                alpha = img_bgr[:, :, 3]
                mask_gray = (alpha <= 10).astype(np.uint8) * 255
            else:
                return orig_bgr

        if mask_gray.ndim == 3:
            mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)
        mask_gray = (mask_gray > 0).astype(np.uint8) * 255
        if np.count_nonzero(mask_gray) == 0:
            return orig_bgr

        mask_pil = Image.fromarray(mask_gray, mode="L")
        self.lazy_load()
        
        # 2. Mise au carre avec replication des bords (Contexte de texture)
        orig_sq, pad, r_wh = _fit_into_square_with_padding(orig_pil, size=512)
        mask_rs = mask_pil.resize(r_wh, resample=Image.NEAREST)
        mask_sq = Image.new("L", (512, 512), 0)
        mask_sq.paste(mask_rs, (pad[0], pad[1]))
        
        # 3. Masque de precision et Prompt Statistique
        mask = _refine_mask(mask_sq)
        prompt = self.get_texture_prompt(orig_pil)
        
        print(f"  [Diffusion] Texture guidee : {prompt}")

        # 4. DIFFUSION REELLE (Pas de pre-remplissage Telea, on laisse l IA diffuser depuis les bords)
        # On utilise une force (strength) elevee pour que l IA reconstruise vraiment la matiere
        result_sq = self.pipe(
            prompt=prompt,
            negative_prompt="text, letters, words, white blocks, gray patches, blurry, artifacts, symbols",
            image=orig_sq.convert("RGB"),
            mask_image=mask,
            num_inference_steps=35,
            guidance_scale=9.0
        ).images[0]

        # 5. Restauration des dimensions
        final_pil = _unpad_and_resize_back(result_sq, pad, orig_pil.size, r_wh)
        
        # 6. FUSION CHIRURGICALE (On ne garde QUE les pixels diffuses dans les trous)
        final_bgr = cv2.cvtColor(np.array(final_pil), cv2.COLOR_RGB2BGR)
        mask_final = mask_gray > 0
        
        res_img = orig_bgr.copy()
        res_img[mask_final] = final_bgr[mask_final]
        
        return res_img
