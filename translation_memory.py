import json
import os
from typing import Dict, Optional


class TranslationMemory:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.getenv(
            "TRANSLATION_MEMORY_PATH",
            "ai_models/translation/translation_memory.jsonl",
        )
        self._memory: Dict[str, str] = {}
        self._load()

    def _key(self, source_text: str, source_lang: str, target_lang: str, block_role: str, strategy: str, style: str = "", tone: str = "") -> str:
        return "||".join([
            (source_lang or "").lower(),
            (target_lang or "").lower(),
            (block_role or "").lower(),
            (strategy or "").lower(),
            (style or "").lower(),
            (tone or "").lower(),
            (source_text or "").strip(),
        ])

    def _load(self):
        path = self.path
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    key = self._key(
                        row.get("source_text", ""),
                        row.get("source_lang", ""),
                        row.get("target_lang", ""),
                        row.get("block_role", ""),
                        row.get("strategy", ""),
                        row.get("style", ""),
                        row.get("tone", ""),
                    )
                    self._memory[key] = row.get("translated_text", "")
        except Exception:
            self._memory = {}

    def lookup(self, source_text: str, source_lang: str, target_lang: str, block_role: str, strategy: str, style: str = "", tone: str = "") -> Optional[str]:
        return self._memory.get(self._key(source_text, source_lang, target_lang, block_role, strategy, style, tone))

    def store(self, source_text: str, translated_text: str, source_lang: str, target_lang: str, block_role: str, strategy: str, style: str = "", tone: str = ""):
        if not source_text or not translated_text:
            return
        key = self._key(source_text, source_lang, target_lang, block_role, strategy, style, tone)
        if self._memory.get(key) == translated_text:
            return
        self._memory[key] = translated_text
        path = self.path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        row = {
            "source_text": source_text,
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "block_role": block_role,
            "strategy": strategy,
            "style": style,
            "tone": tone,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
