import os
import re
import unicodedata
from llama_cpp import Llama

class DocumentTranslator:
    def __init__(self, model_path='./ai_models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf'):
        print(f"Chargement du traducteur hiérarchique...")
        self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, verbose=False)

    def translate_page(self, structure, target_lang="French"):
        blacklist = ["MANNING", "M A N N I N G", "O REILLY", "PACKT", "PEARSON"]
        tech_dict = {"Deep Learning": "Apprentissage profond", "Vision Systems": "Systèmes de vision"}

        for block in structure.get("blocks", []):
            block_role = block.get("role", "body")
            block_context = []
            phrases_to_translate = []
            for line in block.get("lines", []):
                for phrase in line.get("phrases", []):
                    src_text = self._normalize_spaces(phrase.get("texte", ""))
                    if src_text:
                        block_context.append(src_text)
                    phrases_to_translate.append(phrase)

            block_ctx_txt = " ".join(block_context)[:300]
            for phrase in phrases_to_translate:
                orig_phrase_text = self._normalize_spaces(phrase.get("texte", ""))
                if len(orig_phrase_text) < 2:
                    phrase["translated_text"] = orig_phrase_text
                    continue

                if orig_phrase_text in tech_dict:
                    translated_phrase = tech_dict[orig_phrase_text]
                elif orig_phrase_text.upper() in blacklist:
                    translated_phrase = orig_phrase_text
                else:
                    prompt = (
                        f"<|im_start|>system\n"
                        f"You are a technical translator. Translate to {target_lang}. "
                        f"Context: {block_ctx_txt} "
                        f"Output constraints: return translation text only, no instructions, no metadata.\n"
                        f"<|im_end|>\n"
                        f"<|im_start|>user\n{orig_phrase_text}<|im_end|>\n"
                        f"<|im_start|>assistant\n"
                    )
                    output = self.llm(prompt, max_tokens=128, stop=["<|im_end|>"], echo=False)
                    raw_translation = output["choices"][0]["text"].strip()
                    translated_phrase = self._sanitize_translation(raw_translation, orig_phrase_text)

                translated_phrase = self._normalize_translation(translated_phrase, target_lang=target_lang, original=orig_phrase_text)
                phrase["texte_original"] = orig_phrase_text
                phrase["translated_text"] = translated_phrase
                # Backward compatibility: main field now carries translated phrase for downstream renderers.
                phrase["texte"] = translated_phrase
                for span in phrase.get("spans", []):
                    span["texte_original"] = span.get("texte", "")
                    # Keep span text untouched to preserve OCR/native source record.
                    self._normalize_span_style(span, role=block_role)

            for line in block.get("lines", []):
                translated_line = self._normalize_spaces(" ".join(
                    (p.get("translated_text") or p.get("texte") or "").strip()
                    for p in line.get("phrases", [])
                ))
                line["translated_text"] = self._dedupe_sentence_runs(translated_line)

            block_translated = self._normalize_spaces(" ".join(
                (ln.get("translated_text") or "").strip()
                for ln in block.get("lines", [])
            ))
            block["translated_text"] = self._dedupe_sentence_runs(block_translated)

        self._post_dedupe_translated_blocks(structure)
        return structure

    def _sanitize_translation(self, text, original):
        t = (text or "").strip()
        if not t:
            return original

        # Remove common instruction leakage / template markers.
        leak_patterns = [
            r"<\|im_start\|>.*",
            r"<\|im_end\|>.*",
            r"^\s*(system|assistant|user)\s*[:\-].*$",
            r"^\s*(rule|r[èe]gle)\s*[:\-].*$",
            r"^\s*(output constraints|instruction)\s*[:\-].*$",
            r"\b(propose|provide)\s+only\b.*",
        ]
        for pat in leak_patterns:
            t = re.sub(pat, "", t, flags=re.IGNORECASE | re.MULTILINE).strip()

        # Keep only first clean paragraph.
        t = re.split(r"\n{2,}", t)[0].strip()
        t = re.sub(r"\s+", " ", t).strip()

        # Reject obviously corrupted outputs.
        if len(t) < 2:
            return original
        if len(t) > max(400, len(original) * 6):
            return original
        if re.search(r"(rule|r[èe]gle|assistant|system|im_start|im_end)", t, flags=re.IGNORECASE):
            return original

        return t

    def _normalize_spaces(self, text):
        return re.sub(r"\s+", " ", (text or "")).strip()

    def _dedupe_sentence_runs(self, text):
        s = self._normalize_spaces(text)
        parts = [p.strip() for p in re.split(r"(?<=[\.\!\?;:])\s+", s) if p.strip()]
        if not parts:
            return s
        out = []
        for p in parts:
            key = re.sub(r"\W+", "", p).lower()
            if out and re.sub(r"\W+", "", out[-1]).lower() == key:
                continue
            out.append(p)
        return " ".join(out)

    def _normalize_translation(self, text, target_lang="French", original=""):
        s = unicodedata.normalize("NFC", text or "")
        s = self._normalize_spaces(s)
        fixes = {
            "c-ur": "coeur",
            "c-urs": "coeurs",
            "n-ud": "noeud",
            "n-uds": "noeuds",
            "d-": "d'",
            "l-": "l'",
        }
        for k, v in fixes.items():
            s = s.replace(k, v)
        s = re.sub(r"\s+([,;:\.\!\?])", r"\1", s)
        s = self._dedupe_sentence_runs(s)
        if target_lang.lower().startswith("french"):
            en_markers = len(re.findall(r"\b(the|and|of|for|with|from|to|in|is|are)\b", s, flags=re.IGNORECASE))
            fr_markers = len(re.findall(r"\b(le|la|les|de|des|du|et|est|sont|pour|avec|dans)\b", s, flags=re.IGNORECASE))
            if en_markers > fr_markers * 2 and original:
                # Avoid orphan English sentence when target is French.
                return original
        return s

    def _post_dedupe_translated_blocks(self, structure):
        blocks = structure.get("blocks", [])
        kept = []
        for b in blocks:
            txt = self._normalize_spaces(b.get("translated_text") or "")
            bb = b.get("bbox", [0, 0, 0, 0])
            if len(bb) != 4:
                kept.append(b)
                continue
            bx0, by0, bx1, by1 = [float(v) for v in bb]
            area = max(1.0, (bx1 - bx0) * (by1 - by0))
            is_dup = False
            for kb in kept:
                ktxt = self._normalize_spaces(kb.get("translated_text") or "")
                kbb = kb.get("bbox", [0, 0, 0, 0])
                if len(kbb) != 4:
                    continue
                kx0, ky0, kx1, ky1 = [float(v) for v in kbb]
                ix0, iy0 = max(bx0, kx0), max(by0, ky0)
                ix1, iy1 = min(bx1, kx1), min(by1, ky1)
                inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                if inter <= 0:
                    continue
                karea = max(1.0, (kx1 - kx0) * (ky1 - ky0))
                ov = inter / max(1.0, min(area, karea))
                if ov < 0.55:
                    continue
                if txt and ktxt and (txt == ktxt or txt in ktxt or ktxt in txt):
                    is_dup = True
                    break
            if not is_dup:
                kept.append(b)
        structure["blocks"] = kept

    def _normalize_span_style(self, span, role="body"):
        st = span.get("style")
        if not isinstance(st, dict):
            return
        c = (st.get("color") or "#000000").lstrip("#")
        if len(c) != 6:
            return
        try:
            r = int(c[0:2], 16) / 255.0
            g = int(c[2:4], 16) / 255.0
            b = int(c[4:6], 16) / 255.0
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if role == "body" and lum > 0.82:
                st["color"] = "#101010"
        except Exception:
            return
