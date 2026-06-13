from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelEntry:
    name: str
    backend: str
    family: str
    model_dir: str
    tokenizer_dir: str
    source_langs: list[str]
    target_langs: list[str]
    device: str = "cpu"
    compute_type: str = "int8"
    max_input_tokens: int = 512
    priority: int = 0

    @property
    def model_path(self) -> Path:
        return Path(self.model_dir)

    @property
    def tokenizer_path(self) -> Path:
        return Path(self.tokenizer_dir)

    @property
    def available(self) -> bool:
        return self.model_path.is_dir() and self.tokenizer_path.is_dir()

    @property
    def is_multilingual(self) -> bool:
        return not self.source_langs or not self.target_langs

    def supports_pair(self, source_lang: str | None, target_lang: str | None) -> bool:
        src = str(source_lang or "").lower().split("-")[0]
        tgt = str(target_lang or "").lower().split("-")[0]
        if src and self.source_langs and src not in {lang.lower().split("-")[0] for lang in self.source_langs if lang}:
            return False
        if tgt and self.target_langs and tgt not in {lang.lower().split("-")[0] for lang in self.target_langs if lang}:
            return False
        return True


# Langs implicitly covered by a bucket when the inventory does not list them.
_BUCKET_LANGS = {
    "enfr": (["en"], ["fr"]),
}


class TranslationModelRegistry:
    def __init__(self, inventory_path: str | None = None):
        self.inventory_path = Path(
            inventory_path
            or os.getenv("TRANSLATION_MODEL_INVENTORY")
            or os.getenv("TRANSLATION_MODEL_INVENTORY_PATH")
            or "ai_models/translation/model_inventory.json"
        )
        self.models_root = os.getenv("TRANSLATION_MODELS_ROOT")
        self.payload = self._load()

    # ------------------------------------------------------------------ paths
    def _resolve_path(self, raw: str) -> str:
        raw = str(raw or "").strip()
        if not raw:
            return raw
        candidate = Path(raw)
        if candidate.is_absolute():
            return str(candidate)
        bases = []
        if self.inventory_path:
            bases.append(self.inventory_path.resolve().parent)
        if self.models_root:
            bases.append(Path(self.models_root))
        bases.append(PROJECT_ROOT)
        bases.append(Path.cwd())
        for base in bases:
            resolved = base / raw
            if resolved.exists():
                return str(resolved)
        # Default to the project-root interpretation so reporting stays stable.
        return str(PROJECT_ROOT / raw)

    # ------------------------------------------------------------------ load
    def _load(self) -> dict:
        default = {"default_engine": "ct2", "models": []}
        if not self.inventory_path.is_file():
            return default
        try:
            payload = json.loads(self.inventory_path.read_text(encoding="utf-8"))
        except Exception:
            return default
        if not isinstance(payload, dict):
            return default
        if "models" in payload and isinstance(payload.get("models"), list):
            return {
                "default_engine": str(payload.get("default_engine") or "ct2"),
                "models": [item for item in payload.get("models") or [] if isinstance(item, dict)],
            }
        flattened: list[dict] = []
        # Higher priority for the specialised en->fr bucket, then primary, then fallback.
        bucket_priority = {"enfr": 30, "primary": 20, "fallback": 10}
        for bucket in ("enfr", "primary", "fallback"):
            for item in payload.get(bucket) or []:
                if not isinstance(item, dict):
                    continue
                src = item.get("source_langs") or []
                tgt = item.get("target_langs") or []
                if not src or not tgt:
                    implied = _BUCKET_LANGS.get(bucket)
                    if implied:
                        src = src or list(implied[0])
                        tgt = tgt or list(implied[1])
                flattened.append(
                    {
                        "name": item.get("name"),
                        "backend": item.get("backend") or "ctranslate2",
                        "family": item.get("family") or bucket,
                        "model_dir": item.get("model_dir"),
                        "tokenizer_dir": item.get("tokenizer_dir"),
                        "source_langs": src,
                        "target_langs": tgt,
                        "device": item.get("device") or "cpu",
                        "compute_type": item.get("compute_type") or "int8",
                        "max_input_tokens": item.get("max_input_tokens") or 512,
                        "priority": item.get("priority") if item.get("priority") is not None else bucket_priority.get(bucket, 0),
                    }
                )
        return {"default_engine": "ct2", "models": flattened}

    def list_models(self) -> list[ModelEntry]:
        models = []
        seen = set()
        for item in self.payload.get("models") or []:
            model_dir = str(item.get("model_dir") or "").strip()
            tokenizer_dir = str(item.get("tokenizer_dir") or "").strip()
            if not model_dir or not tokenizer_dir:
                continue
            resolved_model = self._resolve_path(model_dir)
            resolved_tokenizer = self._resolve_path(tokenizer_dir)
            name = str(item.get("name") or Path(resolved_model).name)
            key = (name, resolved_model)
            if key in seen:
                continue
            seen.add(key)
            models.append(
                ModelEntry(
                    name=name,
                    backend=str(item.get("backend") or "ctranslate2"),
                    family=str(item.get("family") or "auto"),
                    model_dir=resolved_model,
                    tokenizer_dir=resolved_tokenizer,
                    source_langs=[str(lang) for lang in (item.get("source_langs") or [])],
                    target_langs=[str(lang) for lang in (item.get("target_langs") or [])],
                    device=str(item.get("device") or "cpu"),
                    compute_type=str(item.get("compute_type") or "int8"),
                    max_input_tokens=int(item.get("max_input_tokens") or 512),
                    priority=int(item.get("priority") or 0),
                )
            )
        return sorted(models, key=lambda model: (-model.priority, model.name))

    # ------------------------------------------------------------------ selection
    def select_model(
        self,
        source_lang: str | None = None,
        target_lang: str | None = None,
        preferred_model: str | None = None,
    ) -> ModelEntry | None:
        """Pick the best model for a language pair.

        Priority:
            1. preferred_model when compatible and available
            2. specialised model for the source/target pair
            3. multilingual fallback model
            4. None
        """
        models = self.list_models()
        # 1. explicit preference.
        if preferred_model:
            for model in models:
                if model.name == preferred_model and model.available and model.supports_pair(source_lang, target_lang):
                    return model
            # A named but unavailable preference still wins so callers can surface
            # the precise availability problem via healthcheck.
            for model in models:
                if model.name == preferred_model and model.supports_pair(source_lang, target_lang):
                    return model
        compatible = [model for model in models if model.supports_pair(source_lang, target_lang)]
        available = [model for model in compatible if model.available]
        pool = available or compatible
        # 2. specialised (declares an explicit source/target pair).
        specialised = [model for model in pool if not model.is_multilingual]
        if specialised:
            return sorted(specialised, key=lambda model: (-model.priority, model.name))[0]
        # 3. multilingual fallback.
        multilingual = [model for model in pool if model.is_multilingual]
        if multilingual:
            return sorted(multilingual, key=lambda model: (-model.priority, model.name))[0]
        return None

    def pick(self, *, source_lang: str | None = None, target_lang: str | None = None, engine_name: str | None = None) -> ModelEntry | None:
        return self.select_model(source_lang=source_lang, target_lang=target_lang, preferred_model=engine_name)

    # ------------------------------------------------------------------ health
    def model_status(self, model: ModelEntry, source_lang: str | None = None, target_lang: str | None = None) -> str:
        if not model.model_path.is_dir():
            return "missing_model_dir"
        if not model.tokenizer_path.is_dir():
            return "missing_tokenizer_dir"
        if (source_lang or target_lang) and not model.supports_pair(source_lang, target_lang):
            return "unsupported_lang_pair"
        return "available"

    def healthcheck(self, source_lang: str | None = None, target_lang: str | None = None) -> dict:
        models = self.list_models()
        existing = [model for model in models if model.available]
        selected = self.select_model(source_lang=source_lang, target_lang=target_lang)
        return {
            "status": "ok" if existing else "missing",
            "inventory_path": str(self.inventory_path),
            "model_count": len(models),
            "available_model_count": len(existing),
            "default_engine": self.payload.get("default_engine") or "ct2",
            "selected_model": selected.name if selected else None,
            "models": [
                {
                    "name": model.name,
                    "family": model.family,
                    "backend": model.backend,
                    "model_dir": model.model_dir,
                    "tokenizer_dir": model.tokenizer_dir,
                    "available": model.available,
                    "status": self.model_status(model, source_lang, target_lang),
                    "compatible": model.supports_pair(source_lang, target_lang),
                    "source_langs": model.source_langs,
                    "target_langs": model.target_langs,
                }
                for model in models
            ],
        }
