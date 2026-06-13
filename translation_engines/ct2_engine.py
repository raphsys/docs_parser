from __future__ import annotations

import os

from .base import TranslationEngine
from .model_registry import TranslationModelRegistry
from .placeholder_policy import choose_placeholder_style


# NLLB uses BCP-47-ish script codes; map the common pairs we ship.
_NLLB_LANG_CODES = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "pt": "por_Latn",
    "it": "ita_Latn",
}


def _normalize_family(family: str) -> str:
    fam = str(family or "").strip().lower()
    if fam in {"marian", "opus", "opus-mt", "opus_mt", "enfr"}:
        return "marian"
    if fam in {"m2m100", "m2m_100", "m2m"}:
        return "m2m100"
    if fam in {"nllb", "nllb200", "nllb_200"}:
        return "nllb"
    return "generic"


def _nllb_code(lang: str) -> str:
    base = str(lang or "").lower().split("-")[0]
    return _NLLB_LANG_CODES.get(base, base)


def _encode_source(text: str, tokenizer, family: str, source_lang: str) -> list[str]:
    """Encode source text into CTranslate2 sub-word tokens for the given family."""
    if family == "m2m100":
        try:
            tokenizer.src_lang = str(source_lang or "en").lower().split("-")[0]
        except Exception:
            pass
    elif family == "nllb":
        try:
            tokenizer.src_lang = _nllb_code(source_lang)
        except Exception:
            pass
    ids = tokenizer.encode(text)
    return tokenizer.convert_ids_to_tokens(ids)


def _target_prefix(tokenizer, family: str, target_lang: str) -> list[str] | None:
    """Return the forced target prefix tokens, or None when not required.

    Marian/OPUS bilingual models do not take a language target prefix; sending a
    bare ``"fr"`` token corrupts the decoding. M2M100/NLLB require a language
    token at the start of the target sequence.
    """
    if family == "m2m100":
        base = str(target_lang or "fr").lower().split("-")[0]
        mapping = getattr(tokenizer, "lang_code_to_token", None) or {}
        token = mapping.get(base) or f"__{base}__"
        return [token]
    if family == "nllb":
        return [_nllb_code(target_lang)]
    return None


def _decode_ct2_result(result, tokenizer, family: str) -> tuple[str, list[str]]:
    """Decode a CTranslate2 translation result into clean target text."""
    hypotheses = getattr(result, "hypotheses", None)
    if hypotheses is None and isinstance(result, dict):
        hypotheses = result.get("hypotheses")
    if hypotheses is None and isinstance(result, (list, tuple)):
        hypotheses = result
    output_tokens = list(hypotheses[0]) if hypotheses else []
    decode_tokens = output_tokens
    # M2M100/NLLB prepend the forced target language token; drop it before decode.
    if family in {"m2m100", "nllb"} and decode_tokens:
        first = str(decode_tokens[0])
        if first.startswith("__") or first.endswith("_Latn") or first.endswith("_Cyrl"):
            decode_tokens = decode_tokens[1:]
    ids = tokenizer.convert_tokens_to_ids(decode_tokens)
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return str(text).strip(), output_tokens


class CTranslate2Engine(TranslationEngine):
    profile = "ct2"
    supports_batch = True

    def __init__(
        self,
        *,
        model_name: str | None = None,
        inventory_path: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        batch_size: int | None = None,
        max_input_tokens: int | None = None,
        source_lang: str | None = None,
        target_lang: str | None = None,
    ):
        self.registry = TranslationModelRegistry(inventory_path=inventory_path)
        self.model_name = model_name or os.getenv("TRANSLATION_MODEL_NAME")
        self.device = device or os.getenv("TRANSLATION_DEVICE") or "cpu"
        self.compute_type = compute_type or os.getenv("TRANSLATION_COMPUTE_TYPE") or "int8"
        self.batch_size = int(batch_size or os.getenv("TRANSLATION_BATCH_SIZE") or 8)
        self.max_input_tokens = int(max_input_tokens or os.getenv("TRANSLATION_MAX_INPUT_TOKENS") or 512)
        self.source_lang = source_lang or os.getenv("TRANSLATION_SOURCE_LANG")
        self.target_lang = target_lang or os.getenv("TRANSLATION_TARGET_LANG") or "fr"
        self.placeholder_style = choose_placeholder_style(engine_name="ct2")
        self._model_entry = None
        self._translator = None
        self._tokenizer = None
        self._load_attempted = False

    # ------------------------------------------------------------------ model
    @property
    def model_family(self) -> str | None:
        return self._model_entry.family if self._model_entry else None

    def _resolve_entry(self, source_lang: str | None = None, target_lang: str | None = None):
        if self._model_entry is not None:
            return self._model_entry
        entry = self.registry.select_model(
            source_lang=source_lang or self.source_lang,
            target_lang=target_lang or self.target_lang,
            preferred_model=self.model_name,
        )
        if entry is None:
            raise RuntimeError("No translation model available in registry")
        self._model_entry = entry
        return entry

    def _ensure_loaded(self, source_lang: str | None = None, target_lang: str | None = None):
        if self._load_attempted:
            if self._translator is None or self._tokenizer is None:
                raise RuntimeError("CTranslate2 engine is unavailable")
            return self._translator, self._tokenizer
        self._load_attempted = True
        entry = self._resolve_entry(source_lang, target_lang)
        if not entry.model_path.is_dir():
            raise RuntimeError(f"Model directory missing: {entry.model_dir}")
        if not entry.tokenizer_path.is_dir():
            raise RuntimeError(f"Tokenizer directory missing: {entry.tokenizer_dir}")
        try:
            import ctranslate2
            from transformers import AutoTokenizer
        except Exception as exc:
            raise RuntimeError(f"CTranslate2/Transformers unavailable: {exc}") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(entry.tokenizer_dir)
        self._translator = ctranslate2.Translator(
            entry.model_dir,
            device=self.device,
            compute_type=self.compute_type,
            inter_threads=int(os.getenv("CT2_INTER_THREADS") or 1),
            intra_threads=int(os.getenv("CT2_INTRA_THREADS") or 4),
        )
        return self._translator, self._tokenizer

    # ------------------------------------------------------------------ translate
    def translate(self, text: str, source_lang: str, target_lang: str, context: dict) -> str:
        return self.translate_batch([
            {
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "context": context,
            }
        ])[0]["translated_text"]

    def translate_batch(self, requests: list[dict]) -> list[dict]:
        if not requests:
            return []
        first = requests[0]
        translator, tokenizer = self._ensure_loaded(
            source_lang=first.get("source_lang"),
            target_lang=first.get("target_lang"),
        )
        entry = self._resolve_entry()
        family = _normalize_family(entry.family)

        encoded: list[list[str]] = []
        prefixes: list[list[str] | None] = []
        truncations: list[bool] = []
        input_counts: list[int] = []
        for req in requests:
            text = str(req.get("text") or "")
            source_lang = str(req.get("source_lang") or self.source_lang or "auto")
            target_lang = str(req.get("target_lang") or self.target_lang or "fr")
            tokens = _encode_source(text, tokenizer, family, source_lang)
            truncated = False
            if len(tokens) > self.max_input_tokens:
                tokens = tokens[: self.max_input_tokens]
                truncated = True
            encoded.append(tokens)
            prefixes.append(_target_prefix(tokenizer, family, target_lang))
            truncations.append(truncated)
            input_counts.append(len(tokens))

        # One real batched call to CTranslate2 (not one call per request).
        kwargs = {"max_batch_size": self.batch_size}
        if any(prefix is not None for prefix in prefixes):
            kwargs["target_prefix"] = [prefix or [] for prefix in prefixes]
        results = translator.translate_batch(encoded, **kwargs)

        outputs = []
        for index, req in enumerate(requests):
            result = results[index]
            translated, output_tokens = _decode_ct2_result(result, tokenizer, family)
            outputs.append({
                "translated_text": translated,
                "raw_output": translated,
                "metadata": {
                    "engine": self.profile,
                    "model_name": entry.name,
                    "model_family": entry.family,
                    "source_lang": str(req.get("source_lang") or self.source_lang or "auto"),
                    "target_lang": str(req.get("target_lang") or self.target_lang or "fr"),
                    "device": self.device,
                    "compute_type": self.compute_type,
                    "placeholder_style": self.placeholder_style,
                    "input_token_count": input_counts[index],
                    "output_token_count": len(output_tokens),
                    "truncated": truncations[index],
                    "context": req.get("context") or {},
                },
            })
        return outputs

    # ------------------------------------------------------------------ health
    def healthcheck(self) -> dict:
        registry_health = self.registry.healthcheck(self.source_lang, self.target_lang)
        try:
            entry = self._resolve_entry()
        except Exception as exc:
            return {
                "status": "missing",
                "engine": self.profile,
                "error": f"{type(exc).__name__}: {exc}",
                "registry": registry_health,
            }
        loaded = False
        error = None
        try:
            self._ensure_loaded()
            loaded = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {
            "status": "ok" if loaded else "missing",
            "engine": self.profile,
            "selected_model": entry.name,
            "model_name": entry.name,
            "model_family": entry.family,
            "model_dir": str(entry.model_path),
            "tokenizer_dir": str(entry.tokenizer_path),
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "device": self.device,
            "compute_type": self.compute_type,
            "batch_size": self.batch_size,
            "max_input_tokens": self.max_input_tokens,
            "placeholder_style": self.placeholder_style,
            "error": error,
            "registry": registry_health,
        }
