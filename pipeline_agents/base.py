"""
Base classes pour les agents open-source du pipeline docs_parser.

Chaque agent couvre une étape du pipeline (P1-P6) et expose une interface
identique, indépendante du backend LLM sous-jacent (transformers, ct2, gguf…).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ModelRuntime — backend LLM provider-agnostique
# ---------------------------------------------------------------------------

_MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CT2_COMPUTE_TYPE = os.environ.get("PIPELINE_AGENT_CT2_COMPUTE", "int8")
_CPU_THREADS = max(1, int(os.environ.get("PIPELINE_AGENT_CPU_THREADS", "2")))
_USE_CACHE = os.environ.get("PIPELINE_AGENT_USE_CACHE", "0") == "1"

# Répertoire racine des modèles locaux
_LOCAL_MODELS_DIR = os.path.join(_MODULE_DIR, "ai_models")

# Alias de modèles — clé normalisée → répertoire absolu ou identifiant HF
_MODEL_ALIASES: dict[str, str] = {
    "phi": os.path.join(_LOCAL_MODELS_DIR, "phi35_mini_instruct"),
    "phi35": os.path.join(_LOCAL_MODELS_DIR, "phi35_mini_instruct"),
    "phi35-mini": os.path.join(_LOCAL_MODELS_DIR, "phi35_mini_instruct"),
    "phi-3.5-mini": os.path.join(_LOCAL_MODELS_DIR, "phi35_mini_instruct"),
    "phi-3.5-mini-instruct": os.path.join(_LOCAL_MODELS_DIR, "phi35_mini_instruct"),
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "qwen-small": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
}

# Répertoires CT2 dérivés du répertoire semantic_corrector
_CT2_DIRS: dict[str, str] = {
    os.path.join(_LOCAL_MODELS_DIR, "phi35_mini_instruct"): os.path.join(
        _LOCAL_MODELS_DIR, "semantic_corrector", "phi35_mini_ct2_int8"
    ),
}


def _resolve_alias(model_id: str) -> str:
    raw = str(model_id or "").strip()
    # Essayer d'abord les alias HF communs depuis le cache local
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    normalized = raw.lower().replace("/", "--")
    hf_snapshot_pattern = os.path.join(hf_cache, f"models--{normalized}", "snapshots")
    if os.path.isdir(hf_snapshot_pattern):
        try:
            snapshots = sorted(os.listdir(hf_snapshot_pattern))
            if snapshots:
                return os.path.join(hf_snapshot_pattern, snapshots[-1])
        except OSError:
            pass
    return _MODEL_ALIASES.get(raw.lower(), raw)


def _ct2_dir_for(resolved_id: str) -> str:
    return _CT2_DIRS.get(resolved_id, "")


def _ct2_available(ct2_dir: str) -> bool:
    return bool(ct2_dir) and os.path.isfile(os.path.join(ct2_dir, "model.bin"))


def _weights_available(model_dir: str) -> bool:
    if not model_dir or not os.path.isdir(model_dir):
        return False
    return any(
        f.endswith(".safetensors") or f.endswith(".bin")
        for f in os.listdir(model_dir)
    )


def _detect_family(resolved_id: str) -> str:
    low = resolved_id.lower()
    if "phi" in low:
        return "phi"
    if "qwen" in low:
        return "qwen"
    if "llama" in low or "mistral" in low:
        return "llama"
    if os.path.isdir(resolved_id):
        cfg_path = os.path.join(resolved_id, "config.json")
        try:
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
            mt = str(cfg.get("model_type") or "").lower()
            if "phi" in mt:
                return "phi"
            if "qwen" in mt:
                return "qwen"
        except Exception:
            pass
    return "generic"


def _resolve_backend(resolved_id: str, requested: str = "auto") -> tuple[str, str]:
    """Retourne (backend, ct2_dir). backend ∈ {"transformers", "ct2"}."""
    requested = str(requested or "auto").lower()
    ct2_dir = _ct2_dir_for(resolved_id)
    has_ct2 = _ct2_available(ct2_dir)

    if requested == "ct2":
        return ("ct2", ct2_dir)
    if requested == "transformers":
        return ("transformers", "")
    # auto
    if has_ct2:
        return ("ct2", ct2_dir)
    return ("transformers", "")


class ModelSpec:
    """Description d'un modèle à charger."""

    def __init__(
        self,
        model_id: str,
        backend: str = "auto",
    ) -> None:
        self.requested = str(model_id or "").strip()
        self.resolved_id = _resolve_alias(self.requested)
        self.family = _detect_family(self.resolved_id)
        self.backend, self.ct2_dir = _resolve_backend(self.resolved_id, backend)
        self.trust_remote_code = self.family == "phi" or os.path.isfile(
            os.path.join(self.resolved_id, "modeling_phi3.py")
        )

    def cache_key(self) -> str:
        return "|".join([self.resolved_id, self.backend, self.ct2_dir])

    def is_locally_available(self) -> bool:
        if self.backend == "ct2":
            return _ct2_available(self.ct2_dir)
        return _weights_available(self.resolved_id)

    def __repr__(self) -> str:
        return (
            f"ModelSpec({self.resolved_id!r}, family={self.family}, "
            f"backend={self.backend})"
        )


class ModelRuntime:
    """
    Wrapper provider-agnostique autour d'un modèle LLM.

    Supporte actuellement :
    - ``transformers`` — chargement HuggingFace standard
    - ``ct2`` — CTranslate2 int8 quantisé (plus rapide, moins de mémoire)

    Le backend est sélectionné automatiquement selon ce qui est disponible.
    Utilisez ``PIPELINE_AGENT_BACKEND=ct2|transformers|auto`` pour forcer.
    """

    _GLOBAL_CACHE: dict[str, "ModelRuntime"] = {}
    _FAILURES: set[str] = set()

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._pipe: dict[str, Any] | None = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Factory

    @classmethod
    def for_model(cls, model_id: str, backend: str = "auto") -> "ModelRuntime":
        spec = ModelSpec(model_id, backend=backend)
        key = spec.cache_key()
        if key in cls._GLOBAL_CACHE:
            return cls._GLOBAL_CACHE[key]
        runtime = cls(spec)
        cls._GLOBAL_CACHE[key] = runtime
        return runtime

    # ------------------------------------------------------------------
    # Chargement

    def load(self) -> bool:
        """Charge le modèle en mémoire. Retourne True si réussi."""
        if self._loaded:
            return self._pipe is not None
        self._loaded = True
        key = self.spec.cache_key()
        if key in self._FAILURES:
            return False
        try:
            self._pipe = self._do_load()
        except Exception as exc:
            logger.warning("ModelRuntime: chargement échoué (%s): %s", self.spec, exc)
            self._FAILURES.add(key)
            return False
        if self._pipe is None:
            self._FAILURES.add(key)
            return False
        logger.info("ModelRuntime: prêt — %s", self.spec)
        return True

    def _do_load(self) -> dict[str, Any] | None:
        if self.spec.backend == "ct2":
            return self._load_ct2()
        return self._load_transformers()

    def _load_ct2(self) -> dict[str, Any] | None:
        import ctranslate2
        from transformers import AutoTokenizer

        tokenizer_src = self.spec.ct2_dir if _weights_available(self.spec.ct2_dir) else self.spec.resolved_id
        tok = AutoTokenizer.from_pretrained(
            tokenizer_src, local_files_only=True, trust_remote_code=False
        )
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        gen = ctranslate2.Generator(
            self.spec.ct2_dir, device="cpu", compute_type=_CT2_COMPUTE_TYPE
        )
        return {
            "backend": "ct2",
            "tokenizer": tok,
            "generator": gen,
            "family": self.spec.family,
        }

    def _load_transformers(self) -> dict[str, Any] | None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            torch.set_num_threads(_CPU_THREADS)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        offline = os.environ.get("HF_HUB_OFFLINE") == "1"
        dtype = torch.bfloat16 if self.spec.family in {"qwen", "phi"} else torch.float32
        tok = AutoTokenizer.from_pretrained(
            self.spec.resolved_id,
            local_files_only=offline,
            trust_remote_code=bool(self.spec.trust_remote_code),
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.spec.resolved_id,
            local_files_only=offline,
            low_cpu_mem_usage=True,
            dtype=dtype,
            trust_remote_code=bool(self.spec.trust_remote_code),
        )
        model.eval()
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        return {
            "backend": "transformers",
            "tokenizer": tok,
            "model": model,
            "torch": torch,
            "family": self.spec.family,
        }

    # ------------------------------------------------------------------
    # Inférence

    def is_available(self) -> bool:
        if not self._loaded:
            self.load()
        return self._pipe is not None

    def generate(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        """Génère du texte à partir d'une liste de messages chat."""
        if not self.is_available():
            return ""
        pipe = self._pipe
        assert pipe is not None
        backend = pipe["backend"]
        tok = pipe["tokenizer"]

        # Rendu du prompt
        max_in = int(os.environ.get("PIPELINE_AGENT_MAX_INPUT_TOKENS", "1024"))
        if hasattr(tok, "apply_chat_template"):
            if backend == "ct2":
                input_ids = tok.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    truncation=True,
                    max_length=max_in,
                )
                if input_ids and isinstance(input_ids[0], list):
                    input_ids = input_ids[0]
                tokens = tok.convert_ids_to_tokens(input_ids)
            else:
                tokens = tok.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_in,
                )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
            if backend == "ct2":
                ids = tok(prompt, truncation=True, max_length=max_in)["input_ids"]
                tokens = tok.convert_ids_to_tokens(ids)
            else:
                tokens = tok(prompt, return_tensors="pt", truncation=True, max_length=max_in)["input_ids"]

        if backend == "ct2":
            gen = pipe["generator"]
            eos = getattr(tok, "eos_token", None)
            kwargs: dict[str, Any] = {
                "beam_size": 1,
                "sampling_topk": 1,
                "max_length": max_new_tokens,
                "include_prompt_in_result": False,
            }
            if eos:
                kwargs["end_token"] = eos
            out = gen.generate_batch([tokens], **kwargs)
            seqs_ids = getattr(out[0], "sequences_ids", None)
            if seqs_ids:
                return tok.decode(seqs_ids[0], skip_special_tokens=True).strip()
            seqs = getattr(out[0], "sequences", [[]])[0]
            return tok.decode(tok.convert_tokens_to_ids(seqs), skip_special_tokens=True).strip()

        # transformers
        model = pipe["model"]
        torch = pipe["torch"]
        input_ids = tokens if not isinstance(tokens, dict) else tokens["input_ids"]
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=_USE_CACHE,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        prompt_len = int(input_ids.shape[-1])
        return tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Utilitaire JSON
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | list | None:
    """Extrait le premier objet/tableau JSON depuis une réponse brute."""
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start: i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# ---------------------------------------------------------------------------
# PipelineAgent — classe de base abstraite
# ---------------------------------------------------------------------------

class PipelineAgent(ABC):
    """
    Agent IA pour une étape du pipeline docs_parser.

    Chaque sous-classe implémente ``run(input_data)`` qui reçoit un dict
    représentant les données d'entrée de l'étape et retourne un dict résultat.

    Si le modèle est indisponible, ``run`` retourne ``{}`` sans lever
    d'exception, permettant la dégradation gracieuse vers les heuristiques.
    """

    #: Identifiant de l'étape (ex: "p1_extraction")
    stage: str = ""
    #: Nombre max de tokens générés par défaut
    default_max_new_tokens: int = 256
    #: Version du prompt (pour le cache)
    prompt_version: str = "v1"

    def __init__(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime
        self._cache: dict[str, dict | list] = {}

    # ------------------------------------------------------------------
    # Interface publique

    def is_available(self) -> bool:
        return self.runtime.is_available()

    def run(self, input_data: dict, *, use_cache: bool = True) -> dict:
        """
        Lance l'agent sur ``input_data``.

        Retourne un dict résultat (potentiellement vide si modèle absent ou
        réponse invalide). Ne lève jamais d'exception.
        """
        if not self.is_available():
            return {}
        try:
            cache_key = self._cache_key(input_data) if use_cache else None
            if cache_key and cache_key in self._cache:
                return self._cache[cache_key]  # type: ignore[return-value]

            messages = self.build_messages(input_data)
            raw = self.runtime.generate(messages, max_new_tokens=self.default_max_new_tokens)
            result = self.parse_response(raw, input_data)

            if cache_key:
                self._cache[cache_key] = result
            return result
        except Exception as exc:
            logger.debug("[%s] run() échec: %s", self.stage, exc)
            return {}

    # ------------------------------------------------------------------
    # À implémenter dans les sous-classes

    @abstractmethod
    def build_messages(self, input_data: dict) -> list[dict]:
        """Construit la liste de messages chat pour le LLM."""

    @abstractmethod
    def parse_response(self, raw: str, input_data: dict) -> dict:
        """Parse la réponse brute du LLM et retourne un dict normalisé."""

    # ------------------------------------------------------------------
    # Utilitaires

    def _cache_key(self, input_data: dict) -> str:
        payload = json.dumps(
            {"stage": self.stage, "v": self.prompt_version, "data": input_data},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.md5(payload.encode()).hexdigest()

    def _extract_json(self, text: str) -> dict | list | None:
        return _extract_json(text)

    @staticmethod
    def _normalize_text(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()
