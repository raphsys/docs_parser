"""Validated translation memory for PAGETRANSLATE/translation_engines.

A memory hit lets the runtime skip the model entirely. The store tolerates two
record shapes:

    {"source": "...", "target": "...", "source_lang": "en", "target_lang": "fr",
     "domain": "deep_learning", "validated": true}

and the historical export shape:

    {"source_text": "...", "translated_text": "...", "source_lang": "en",
     "target_lang": "fr", "block_role": "body"}

An entry is treated as validated unless ``validated`` is explicitly ``false``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _lang(value) -> str:
    return str(value or "").lower().split("-")[0]


class TranslationMemory:
    def __init__(self):
        self._exact: dict[tuple[str, str, str], str] = {}
        self._normalized: dict[tuple[str, str, str], str] = {}
        self.entry_count = 0
        self.skipped_unvalidated = 0

    def add(self, entry: dict) -> bool:
        if not isinstance(entry, dict):
            return False
        if entry.get("validated") is False:
            self.skipped_unvalidated += 1
            return False
        source = entry.get("source")
        if source is None:
            source = entry.get("source_text")
        target = entry.get("target")
        if target is None:
            target = entry.get("translated_text")
        if not source or not target:
            return False
        source_lang = _lang(entry.get("source_lang"))
        target_lang = _lang(entry.get("target_lang") or "fr")
        domain = str(entry.get("domain") or entry.get("block_role") or "").lower()
        self._exact[(str(source), source_lang, target_lang)] = str(target)
        self._normalized[(_normalize(source), source_lang, target_lang)] = str(target)
        # Domain-scoped variants keep an extra keyed lookup for stricter callers.
        if domain:
            self._exact[(str(source), source_lang, target_lang, domain)] = str(target)
            self._normalized[(_normalize(source), source_lang, target_lang, domain)] = str(target)
        self.entry_count += 1
        return True

    def lookup_exact(self, source: str, source_lang: str, target_lang: str, domain: str | None = None) -> str | None:
        sl = _lang(source_lang)
        tl = _lang(target_lang)
        if domain:
            hit = self._exact.get((str(source), sl, tl, str(domain).lower()))
            if hit is not None:
                return hit
        return self._exact.get((str(source), sl, tl))

    def lookup_normalized(self, source: str, source_lang: str, target_lang: str, domain: str | None = None) -> str | None:
        sl = _lang(source_lang)
        tl = _lang(target_lang)
        key = _normalize(source)
        if domain:
            hit = self._normalized.get((key, sl, tl, str(domain).lower()))
            if hit is not None:
                return hit
        return self._normalized.get((key, sl, tl))

    def lookup(self, source: str, source_lang: str, target_lang: str, domain: str | None = None) -> dict | None:
        """Return {'target', 'memory_source'} on a hit, else None."""
        hit = self.lookup_exact(source, source_lang, target_lang, domain)
        if hit is not None:
            return {"target": hit, "memory_source": "exact"}
        hit = self.lookup_normalized(source, source_lang, target_lang, domain)
        if hit is not None:
            return {"target": hit, "memory_source": "normalized"}
        return None

    def __len__(self) -> int:
        return self.entry_count


def load_translation_memory(path: str | os.PathLike | None = None) -> TranslationMemory:
    """Load a JSONL translation memory. Returns an empty memory when absent."""
    memory = TranslationMemory()
    candidate = path or os.getenv("TRANSLATION_MEMORY_PATH")
    if not candidate:
        return memory
    file_path = Path(candidate)
    if not file_path.is_file():
        return memory
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                memory.add(entry)
    except Exception:
        return memory
    return memory
