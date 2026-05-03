"""
LLM léger post-processeur pour la correction des segmentations sémantiques ambiguës.

Ce module assure la COMPRÉHENSION DU TEXTE pour améliorer l'extraction :
- Détection de titres implicites (pas seulement ALL-CAPS)
- Regroupement de légendes multi-lignes (figure/table captions)
- Identification de formules/code non-traductibles
- Résolution de césures de mots (lignes finissant par "-")
- Marquage des labels de diagramme courts à sauter ou atomiser
- Découpage intra-ligne en unités sémantiques quand une ligne contient
  plusieurs phrases/segments distincts

Le module est optionnel : si transformers / le modèle ne sont pas disponibles,
le pipeline dégrade silencieusement sur les seules heuristiques.

Sur CPU avec Qwen2.5-3B ou Phi-3.5-mini, il faut rester conservateur :
- nombre de blocs traités par page limité
- cache de génération désactivé pour limiter le pic mémoire
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

_QWEN3B_SNAPSHOT = (
    "/home/raphael/.cache/huggingface/hub"
    "/models--Qwen--Qwen2.5-3B-Instruct/snapshots"
    "/aa8e72537993ba99e69dfaafa59ed015b17504d1"
)
_QWEN05B_SNAPSHOT = (
    "/home/raphael/.cache/huggingface/hub"
    "/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
    "/7ae557604adf67be50417f59c2c2f167def9a775"
)
_PHI35_LOCAL_MODEL_DIR = os.path.join(_MODULE_DIR, "ai_models", "phi35_mini_instruct")
_QWEN05B_CT2_INT8_DIR = os.path.join(_MODULE_DIR, "ai_models", "semantic_corrector", "qwen2_5_0_5b_ct2_int8")
_PHI35_CT2_INT8_DIR = os.path.join(_MODULE_DIR, "ai_models", "semantic_corrector", "phi35_mini_ct2_int8")

_MODEL_ALIASES = {
    "qwen": _QWEN3B_SNAPSHOT,
    "qwen-fast": _QWEN05B_SNAPSHOT,
    "qwen2.5-3b": _QWEN3B_SNAPSHOT,
    "qwen2.5-3b-instruct": _QWEN3B_SNAPSHOT,
    "qwen2.5-3b-local": _QWEN3B_SNAPSHOT,
    "qwen-small": _QWEN05B_SNAPSHOT,
    "qwen2.5-0.5b": _QWEN05B_SNAPSHOT,
    "qwen2.5-0.5b-instruct": _QWEN05B_SNAPSHOT,
    "qwen2.5-0.5b-local": _QWEN05B_SNAPSHOT,
    "phi": _PHI35_LOCAL_MODEL_DIR,
    "phi-fast": _PHI35_LOCAL_MODEL_DIR,
    "phi35": _PHI35_LOCAL_MODEL_DIR,
    "phi35-mini": _PHI35_LOCAL_MODEL_DIR,
    "phi35-mini-local": _PHI35_LOCAL_MODEL_DIR,
    "phi-3.5-mini": _PHI35_LOCAL_MODEL_DIR,
    "phi-3.5-mini-local": _PHI35_LOCAL_MODEL_DIR,
    "phi-3.5-mini-instruct": _PHI35_LOCAL_MODEL_DIR,
}

MODEL_ID: str = os.environ.get(
    "SEMANTIC_CORRECTOR_MODEL",
    _QWEN3B_SNAPSHOT,
)
# Fallback léger si le 3B n'est pas encore téléchargé
CPU_FALLBACK_MODEL_ID: str = os.environ.get(
    "SEMANTIC_CORRECTOR_CPU_FALLBACK_MODEL",
    _QWEN05B_SNAPSHOT,
)
STRONG_MODEL_ID: str = os.environ.get(
    "SEMANTIC_CORRECTOR_STRONG_MODEL",
    _QWEN3B_SNAPSHOT,
)
# Seuil minimal d'ambiguïté pour déclencher l'appel LLM (0=jamais, 1=toujours)
AMBIGUITY_THRESHOLD: float = float(os.environ.get("SEMANTIC_CORRECTOR_THRESHOLD", "0.25"))
STRONG_AMBIGUITY_THRESHOLD: float = float(os.environ.get("SEMANTIC_CORRECTOR_STRONG_THRESHOLD", "0.45"))

# Nombre max de tokens générés par le LLM
_MAX_NEW_TOKENS: int = int(os.environ.get("SEMANTIC_CORRECTOR_MAX_NEW_TOKENS", "180"))
# Nombre max de lignes envoyées au LLM par bloc
_MAX_LINES: int = int(os.environ.get("SEMANTIC_CORRECTOR_MAX_LINES", "12"))
_MAX_BLOCKS_PER_PAGE: int = int(os.environ.get("SEMANTIC_CORRECTOR_MAX_BLOCKS_PER_PAGE", "5"))
_CPU_THREADS: int = max(1, int(os.environ.get("SEMANTIC_CORRECTOR_CPU_THREADS", "2")))
_USE_CACHE: bool = os.environ.get("SEMANTIC_CORRECTOR_USE_CACHE", "0") == "1"
_MAX_FEW_SHOT_EXAMPLES: int = max(0, int(os.environ.get("SEMANTIC_CORRECTOR_MAX_FEW_SHOT_EXAMPLES", "4")))
_MAX_BLOCK_CHARS: int = int(os.environ.get("SEMANTIC_CORRECTOR_MAX_BLOCK_CHARS", "1000"))
_MAX_BLOCK_WORDS: int = int(os.environ.get("SEMANTIC_CORRECTOR_MAX_BLOCK_WORDS", "160"))
_MAX_CANDIDATE_VOTES_PER_BLOCK: int = max(0, int(os.environ.get("SEMANTIC_CORRECTOR_MAX_CANDIDATE_VOTES", "5")))

_CT2_MODEL_DIRS = {
    _QWEN05B_SNAPSHOT: _QWEN05B_CT2_INT8_DIR,
    _PHI35_LOCAL_MODEL_DIR: _PHI35_CT2_INT8_DIR,
}

# ---------------------------------------------------------------------------
# Prompt système — compréhension du texte élargie
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a text-structure expert. Return JSON only.
Output schema: {"c":[...],"bc":[...],"sb":[...],"lb":[...],"lm":null|{...},"hj":[...]}
- c = corrections: [{"li":[line indices],"a":"heading|formula|skip|atomic|keep"}]
- bc = selected boundary candidates by id, chosen from the provided candidate list
- sb = intra-line semantic boundaries: [{"i":line_index,"s":["segment 1","segment 2", ...]}]
- lb = semantic breaks after full lines: [line_index, ...]
- lm = layout mode override: {"line_flow":"inline_reflow|preserve_line_breaks|preserve_paragraphs|fixed_lines","breaks":[{"i":line_index,"after":"soft_wrap|new_line|paragraph_break|list_item|fixed_line|block_end"}]}
- hj = hyphen joins: [{"i":line_index,"w":"full_word"}] when a line ends with a hyphenated word fragment

Correction rules:
- heading: a section title, chapter title, or bold label that is NOT part of a paragraph (even mixed-case, even if on same block as prose)
- formula: math notation, code snippet, technical symbol sequence (not translatable as prose)
- skip: page number, running header/footer, separator line, isolated axis label (single letter like E or W next to a figure)
- atomic: figure caption, table caption, diagram label spanning multiple lines — keep ALL caption lines together as one unit
- keep: normal prose, no change needed
- bc: prefer choosing candidate ids from the provided candidate list whenever possible
- sb: use this when ONE line contains multiple semantic units and they should become separate phrases, even without terminal punctuation; also use it when a line starts with the tail of the previous sentence and then begins a new sentence later on the same line
- lb: use this when the semantic sentence boundary is BETWEEN two lines even if punctuation is missing or unreliable
- lm: use inline_reflow for visual soft-wrapped prose; preserve_line_breaks for lists, steps, labels, headings-with-body; preserve_paragraphs when blank/gap/indent indicates a paragraph break; fixed_lines for tables/diagram labels/math
- hj: when line text ends with a word fragment ending in "-", provide the joined full word

If the first line of a block looks like a heading title followed by body prose on the NEXT line, mark the first line as heading.
If several consecutive lines form a figure/table caption (starting with "Figure", "Table", "Fig.", etc. or continuing a caption), mark them all as atomic.
If a line contains only 1-2 isolated characters that are axis/diagram labels, mark as skip.
You may receive hs = heuristic semantic phrases. Treat them as provisional guesses only. If hs contains truncation, duplicated prefix/suffix carry-over, or misses a semantic boundary, override it with sb or lb.
If boundary candidates are provided, prefer returning bc with selected ids instead of inventing new boundaries.
If unsure return {"c":[],"bc":[],"sb":[],"lb":[],"lm":null,"hj":[]}."""

_BOUNDARY_VOTE_SYSTEM_PROMPT = """\
You decide whether a proposed semantic boundary should be kept.
Answer with YES or NO only.
Keep the boundary when it separates two semantic text units, even if punctuation is missing or unreliable.
Reject the boundary when both sides clearly belong to the same semantic sentence."""

# Few-shot examples couvrant les cas réels du pipeline
_FEW_SHOT_EXAMPLES = [
    # 1: titre de section collé au paragraphe suivant → heading + keep
    (
        '{"role":"body","lines":[{"i":0,"t":"Too-high vs. too-low learning rate Setting the learning","bb":[0,0,100,10]},{"i":1,"t":"rate high or low is a trade-off between speed and accuracy.","bb":[0,12,100,22]}]}',
        '{"c":[{"li":[0],"a":"heading"},{"li":[1],"a":"keep"}],"bc":[],"sb":[],"lb":[],"lm":{"line_flow":"preserve_line_breaks","breaks":[{"i":0,"after":"new_line"},{"i":1,"after":"block_end"}]},"hj":[]}',
    ),
    # 2: légende figure multi-lignes → atomic
    (
        '{"role":"body","lines":[{"i":0,"t":"Figure 4.17","bb":[0,0,30,10]},{"i":1,"t":"A learning rate larger than the","bb":[0,12,100,22]},{"i":2,"t":"ideal lr value: the optimizer overshoots.","bb":[0,24,90,34]}]}',
        '{"c":[{"li":[0,1,2],"a":"atomic"}],"bc":[],"sb":[],"lb":[],"lm":{"line_flow":"preserve_line_breaks","breaks":[{"i":0,"after":"new_line"},{"i":1,"after":"new_line"},{"i":2,"after":"block_end"}]},"hj":[]}',
    ),
    # 3: labels d'axe isolés (E et W) → skip
    (
        '{"role":"body","lines":[{"i":0,"t":"E","bb":[0,0,10,10]},{"i":1,"t":"Figure 4.18","bb":[0,15,40,25]},{"i":2,"t":"W","bb":[0,30,10,40]}]}',
        '{"c":[{"li":[0],"a":"skip"},{"li":[1,2],"a":"atomic"}],"bc":[],"sb":[],"lb":[],"hj":[]}',
    ),
    # 4: code Python → formula
    (
        '{"role":"body","lines":[{"i":0,"t":"x = Conv2D(64, (3,3), padding=\'same\')(x)","bb":[0,0,100,10]},{"i":1,"t":"x = BatchNormalization()(x)","bb":[0,12,100,22]}]}',
        '{"c":[{"li":[0,1],"a":"formula"}],"bc":[],"sb":[],"lb":[],"hj":[]}',
    ),
    # 5: prose normale qui s'enroule sur deux lignes → pas de correction
    (
        '{"role":"body","lines":[{"i":0,"t":"Concatenates together the","bb":[0,0,60,10]},{"i":1,"t":"depth of the three filters.","bb":[0,12,60,22]}]}',
        '{"c":[],"bc":[],"sb":[],"lb":[],"lm":{"line_flow":"inline_reflow","breaks":[{"i":0,"after":"soft_wrap"},{"i":1,"after":"block_end"}]},"hj":[]}',
    ),
    # 6: césure de mot en fin de ligne → hj
    (
        '{"role":"body","lines":[{"i":0,"t":"The algorithm is guaran-","bb":[0,0,100,10]},{"i":1,"t":"teed to converge eventually.","bb":[0,12,100,22]}]}',
        '{"c":[],"bc":[],"sb":[],"lb":[],"hj":[{"i":0,"w":"guaranteed"}]}',
    ),
    # 7: ALL-CAPS heading avant prose
    (
        '{"role":"body","lines":[{"i":0,"t":"THE GRADIENT","bb":[0,0,40,8]},{"i":1,"t":"Suppose you are standing on the hill.","bb":[0,10,100,18]}]}',
        '{"c":[{"li":[0],"a":"heading"},{"li":[1],"a":"keep"}],"bc":[],"sb":[],"lb":[],"hj":[]}',
    ),
    # 8: fraction mathématique
    (
        '{"role":"body","lines":[{"i":0,"t":"dE","bb":[40,0,60,30]},{"i":1,"t":"---","bb":[35,35,65,45]},{"i":2,"t":"dw","bb":[40,50,60,80]}]}',
        '{"c":[{"li":[0,2],"a":"formula"},{"li":[1],"a":"skip"}],"bc":[],"sb":[],"lb":[],"hj":[]}',
    ),
    # 9: plusieurs unités sémantiques sur une seule ligne
    (
        '{"role":"body","lines":[{"i":0,"t":"Pooling reduces dimensions the next stage applies normalization","bb":[0,0,100,12]}],"bc":[{"id":"sb:0:0","k":"sb","i":0,"s":["Pooling reduces dimensions","the next stage applies normalization"]}]}',
        '{"c":[],"bc":["sb:0:0"],"sb":[],"lb":[],"hj":[]}',
    ),
    # 10: une ligne commence par la fin de la phrase précédente puis ouvre
    # une nouvelle phrase complète plus loin sur la même ligne
    (
        '{"role":"body","lines":[{"i":0,"t":"Depth is referred to as the color channel, where depth is 1 for grayscale images","bb":[0,0,100,10]},{"i":1,"t":"and 3 for color images. In the later layers, the images still have depth","bb":[0,12,100,22]}],"bc":[{"id":"sb:1:0","k":"sb","i":1,"s":["and 3 for color images.","In the later layers, the images still have depth"]}]}',
        '{"c":[],"bc":["sb:1:0"],"sb":[],"lb":[],"hj":[]}',
    ),
    # 11: frontière sémantique entre deux lignes sans ponctuation terminale fiable
    (
        '{"role":"body","lines":[{"i":0,"t":"The network learns hierarchical representations","bb":[0,0,100,10]},{"i":1,"t":"Each later stage extracts more abstract features","bb":[0,12,100,22]}],"bc":[{"id":"lb:0","k":"lb","after":0,"prev":"The network learns hierarchical representations","next":"Each later stage extracts more abstract features"}]}',
        '{"c":[],"bc":["lb:0"],"sb":[],"lb":[],"lm":{"line_flow":"preserve_line_breaks","breaks":[{"i":0,"after":"new_line"},{"i":1,"after":"block_end"}]},"hj":[]}',
    ),
]

# ---------------------------------------------------------------------------
# Chargement du modèle (singleton lazy)
# ---------------------------------------------------------------------------

_pipe: Any = None
_load_attempted: bool = False
_effective_model_id: str | None = None
_effective_model_family: str | None = None
_MAX_INPUT_TOKENS: int = int(os.environ.get("SEMANTIC_CORRECTOR_MAX_INPUT_TOKENS", "512"))
_PIPELINE_CACHE: dict[str, Any | None] = {}
_PIPELINE_FAILURES: set[str] = set()


def _configured_model_id() -> str:
    return str(os.environ.get("SEMANTIC_CORRECTOR_MODEL", MODEL_ID) or MODEL_ID)


def _configured_cpu_fallback_model_id() -> str:
    return str(os.environ.get("SEMANTIC_CORRECTOR_CPU_FALLBACK_MODEL", CPU_FALLBACK_MODEL_ID) or CPU_FALLBACK_MODEL_ID)


def _configured_strong_model_id() -> str:
    return str(os.environ.get("SEMANTIC_CORRECTOR_STRONG_MODEL", STRONG_MODEL_ID) or STRONG_MODEL_ID)


def _configured_max_new_tokens() -> int:
    try:
        return max(1, int(os.environ.get("SEMANTIC_CORRECTOR_MAX_NEW_TOKENS", str(_MAX_NEW_TOKENS)) or _MAX_NEW_TOKENS))
    except Exception:
        return _MAX_NEW_TOKENS


def _configured_max_input_tokens() -> int:
    try:
        return max(64, int(os.environ.get("SEMANTIC_CORRECTOR_MAX_INPUT_TOKENS", str(_MAX_INPUT_TOKENS)) or _MAX_INPUT_TOKENS))
    except Exception:
        return _MAX_INPUT_TOKENS


def _resolve_model_alias(model_id: str) -> str:
    raw = str(model_id or "").strip()
    if not raw:
        return raw
    return _MODEL_ALIASES.get(raw.lower(), raw)


def _read_local_model_config(model_id: str) -> dict[str, Any]:
    resolved_id = _resolve_model_alias(model_id)
    if not resolved_id or not os.path.isdir(resolved_id):
        return {}
    cfg_path = os.path.join(resolved_id, "config.json")
    if not os.path.isfile(cfg_path):
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolve_requested_backend() -> str:
    backend = str(os.environ.get("SEMANTIC_CORRECTOR_BACKEND", "auto") or "auto").strip().lower()
    if backend not in {"auto", "transformers", "ct2"}:
        return "auto"
    return backend


def _ct2_model_is_available(model_dir: str) -> bool:
    return bool(model_dir) and os.path.isdir(model_dir) and os.path.isfile(os.path.join(model_dir, "model.bin"))


def _resolve_ct2_model_dir(model_id: str) -> str:
    resolved_id = _resolve_model_alias(model_id)
    return _CT2_MODEL_DIRS.get(resolved_id, "")


def _model_is_available(model_id: str) -> bool:
    """Vérifie si le répertoire du modèle local contient au moins un fichier de poids."""
    resolved_id = _resolve_model_alias(model_id)
    if not resolved_id or not os.path.isdir(resolved_id):
        return False
    for fname in os.listdir(resolved_id):
        if fname.endswith(".safetensors") or fname.endswith(".bin"):
            return True
    return False


def _detect_model_family(model_id: str) -> str:
    resolved_id = _resolve_model_alias(model_id)
    lowered = str(resolved_id or "").strip().lower()
    if "qwen" in lowered:
        return "qwen"
    if "phi" in lowered:
        return "phi"
    cfg = _read_local_model_config(resolved_id)
    model_type = str(cfg.get("model_type") or "").strip().lower()
    if model_type.startswith("phi"):
        return "phi"
    if model_type.startswith("qwen"):
        return "qwen"
    architectures = [str(v).lower() for v in (cfg.get("architectures") or []) if isinstance(v, str)]
    if any("phi" in arch for arch in architectures):
        return "phi"
    if any("qwen" in arch for arch in architectures):
        return "qwen"
    return "generic"


def _resolve_model_spec(model_id: str) -> dict[str, Any]:
    resolved_id = _resolve_model_alias(model_id)
    family = _detect_model_family(resolved_id)
    cfg = _read_local_model_config(resolved_id)
    requested_backend = _resolve_requested_backend()
    ct2_model_dir = _resolve_ct2_model_dir(resolved_id)
    has_ct2 = _ct2_model_is_available(ct2_model_dir)
    backend = "transformers"
    if requested_backend == "ct2":
        backend = "ct2" if has_ct2 else "transformers"
    elif requested_backend == "auto" and has_ct2:
        backend = "ct2"
    trust_remote_code = bool(
        family == "phi"
        or isinstance(cfg.get("auto_map"), dict)
        or os.path.isfile(os.path.join(resolved_id, "modeling_phi3.py"))
    )
    return {
        "requested": str(model_id or "").strip(),
        "model_id": resolved_id,
        "family": family,
        "backend": backend,
        "ct2_model_dir": ct2_model_dir if has_ct2 else "",
        "tokenizer_source": ct2_model_dir if backend == "ct2" and has_ct2 else resolved_id,
        "compute_type": "int8" if backend == "ct2" and has_ct2 else "",
        "trust_remote_code": trust_remote_code,
    }


def _resolve_model_spec_for_runtime() -> dict[str, Any]:
    requested = str(_configured_model_id() or "").strip()
    fallback = str(_configured_cpu_fallback_model_id() or "").strip()
    # Utiliser le modèle principal s'il est disponible localement
    if _model_is_available(requested):
        return _resolve_model_spec(requested)
    # Sinon, tenter le fallback local
    if fallback and _model_is_available(fallback):
        spec = _resolve_model_spec(fallback)
        logger.info("LLM corrector: modèle principal absent, fallback → %s", spec["model_id"])
        return spec
    # En dernier recours, retourner le modèle principal (téléchargement HF)
    return _resolve_model_spec(requested)


def _resolve_torch_dtype(torch_module: Any, model_spec: dict[str, Any]):
    value = str((model_spec or {}).get("model_id") or "").strip().lower()
    family = str((model_spec or {}).get("family") or "").strip().lower()
    # bfloat16 pour les modèles ≥ 1B (meilleure précision que float16, moins de VRAM que float32)
    if family in {"qwen", "phi"} or "qwen2.5-3b" in value or "qwen2.5-1.5b" in value or "qwen2.5-0.5b" in value or "qwen2" in value:
        return torch_module.bfloat16
    return torch_module.float32


def _pipeline_cache_key(model_spec: dict[str, Any]) -> str:
    return "|".join(
        [
            str(model_spec.get("model_id") or ""),
            str(model_spec.get("backend") or ""),
            str(model_spec.get("ct2_model_dir") or ""),
        ]
    )


def _build_pipeline_from_model_spec(model_spec: dict[str, Any]) -> Any | None:
    try:
        logger.info(
            "LLM corrector: chargement de %s (family=%s, backend=%s)…",
            model_spec["model_id"],
            model_spec["family"],
            model_spec.get("backend"),
        )
        if model_spec.get("backend") == "ct2":
            import ctranslate2
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_spec["tokenizer_source"],
                local_files_only=True,
                trust_remote_code=False,
            )
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            generator = ctranslate2.Generator(
                model_spec["ct2_model_dir"],
                device="cpu",
                compute_type=str(model_spec.get("compute_type") or "int8"),
            )
            _pipe = {
                "backend": "ct2",
                "tokenizer": tokenizer,
                "generator": generator,
                "model_id": model_spec["model_id"],
                "model_family": model_spec["family"],
                "ct2_model_dir": model_spec["ct2_model_dir"],
            }
            return _pipe
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            try:
                torch.set_num_threads(_CPU_THREADS)
                torch.set_num_interop_threads(1)
            except Exception:
                pass

            resolved_dtype = _resolve_torch_dtype(torch, model_spec)
            tokenizer = AutoTokenizer.from_pretrained(
                model_spec["model_id"],
                local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
                trust_remote_code=bool(model_spec.get("trust_remote_code")),
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_spec["model_id"],
                local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
                low_cpu_mem_usage=True,
                dtype=resolved_dtype,
                trust_remote_code=bool(model_spec.get("trust_remote_code")),
            )
            model.eval()
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            _pipe = {
                "backend": "transformers",
                "tokenizer": tokenizer,
                "model": model,
                "torch": torch,
                "model_id": model_spec["model_id"],
                "model_family": model_spec["family"],
            }
            return _pipe
    except Exception as exc:
        logger.warning("LLM corrector non disponible (%s). Pipeline heuristique seul.", exc)
        return None


def _load_pipeline(model_id: str | None = None) -> Any | None:
    """Charge tokenizer+model en mémoire (CPU, une seule fois par modèle/backend)."""
    global _pipe, _load_attempted, _effective_model_id, _effective_model_family
    model_spec = _resolve_model_spec_for_runtime() if model_id is None else _resolve_model_spec(model_id)
    cache_key = _pipeline_cache_key(model_spec)
    if cache_key in _PIPELINE_CACHE:
        cached_pipe = _PIPELINE_CACHE[cache_key]
        if model_id is None:
            _pipe = cached_pipe
            _load_attempted = True
            _effective_model_id = str(model_spec.get("model_id") or "")
            _effective_model_family = str(model_spec.get("family") or "")
        return cached_pipe
    if cache_key in _PIPELINE_FAILURES:
        return None

    pipe = _build_pipeline_from_model_spec(model_spec)
    if pipe is None:
        _PIPELINE_FAILURES.add(cache_key)
        if model_id is None:
            _load_attempted = True
        return None

    _PIPELINE_CACHE[cache_key] = pipe
    logger.info("LLM corrector: modèle prêt (%s).", model_spec["model_id"])
    if model_id is None:
        _pipe = pipe
        _load_attempted = True
        _effective_model_id = str(model_spec.get("model_id") or "")
        _effective_model_family = str(model_spec.get("family") or "")
    return pipe


# ---------------------------------------------------------------------------
# Formatage du bloc pour le prompt
# ---------------------------------------------------------------------------

def _normalise_bbox(
    bb: list | None, bx0: float, by0: float, bw: float, bh: float
) -> list[int]:
    """Normalise une bbox pixel → [0,100] dans le référentiel du bloc."""
    if not bb or len(bb) < 4:
        return [0, 0, 0, 0]
    return [
        round((bb[0] - bx0) / bw * 100),
        round((bb[1] - by0) / bh * 100),
        round((bb[2] - bx0) / bw * 100),
        round((bb[3] - by0) / bh * 100),
    ]


def _truncate_text(text: Any, limit: int) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 1)].rstrip() + "…"


def _line_starts_like_new_sentence(text: Any) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return bool(re.match(r'^[A-Z"\(\[\u201c\u2018]', normalized))


def _line_ends_with_terminal_punctuation(text: Any) -> bool:
    return bool(re.search(r'[.!?]["\')\]]*$', _normalize_text(text)))


def _split_line_text_for_boundary_candidates(text: Any) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    segments = []
    start = 0
    for match in re.finditer(r'[.!?\u2026]+[\u201d\u2019"\')\]}\u00bb]*', normalized):
        end = match.end()
        segment = normalized[start:end].strip()
        next_part = normalized[end:]
        if not segment:
            continue
        if next_part and not re.match(r'^\s+[A-Z\u201c\u2018"\'(\[]', next_part):
            continue
        segments.append(segment)
        start = end
        while start < len(normalized) and normalized[start].isspace():
            start += 1
    if start < len(normalized):
        tail = normalized[start:].strip()
        if tail:
            segments.append(tail)
    return segments if len(segments) >= 2 else []


def _boundary_candidate_dict(candidate_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    candidate = {"id": candidate_id}
    candidate.update(payload)
    return candidate


def _build_boundary_candidates(block: dict) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    lines = [ln for ln in (block.get("lines") or [])[:_MAX_LINES] if isinstance(ln, dict)]
    candidates: list[dict[str, Any]] = []
    candidate_map: dict[str, dict[str, Any]] = {}

    def add_candidate(candidate_id: str, payload: dict[str, Any]) -> None:
        if candidate_id in candidate_map:
            return
        candidate = _boundary_candidate_dict(candidate_id, payload)
        candidate_map[candidate_id] = candidate
        candidates.append(candidate)

    for line in lines:
        li = int(line.get("line_index", 0) or 0)
        line_text = line.get("line_text") or line.get("text") or ""
        split_segments = _split_line_text_for_boundary_candidates(line_text)
        if len(split_segments) >= 2:
            add_candidate(
                f"sb:{li}:0",
                {
                    "k": "sb",
                    "i": li,
                    "s": [_truncate_text(segment, 90) for segment in split_segments[:3]],
                },
            )
            continue

        phrase_candidates = []
        for phrase in (line.get("phrases") or []):
            if not isinstance(phrase, dict):
                continue
            phrase_text = _normalize_text(phrase.get("texte") or phrase.get("text") or "")
            if phrase_text:
                phrase_candidates.append(phrase_text)
        if len(phrase_candidates) >= 2:
            add_candidate(
                f"sb:{li}:1",
                {
                    "k": "sb",
                    "i": li,
                    "s": [_truncate_text(segment, 90) for segment in phrase_candidates[:3]],
                },
            )

    for prev_line, next_line in zip(lines, lines[1:]):
        prev_text = _normalize_text(prev_line.get("line_text") or prev_line.get("text") or "")
        next_text = _normalize_text(next_line.get("line_text") or next_line.get("text") or "")
        if not prev_text or not next_text:
            continue
        prev_index = int(prev_line.get("line_index", 0) or 0)
        # Si la ligne suivante contient déjà un carry-over suivi d'une nouvelle
        # phrase, la frontière utile est intra-ligne dans cette ligne suivante,
        # pas entre les deux lignes.
        if _line_has_carryover_then_new_sentence(next_text):
            continue
        if not (
            _line_ends_with_terminal_punctuation(prev_text)
            or _line_starts_like_new_sentence(next_text)
            or _shared_edge_word_count(prev_text, next_text) >= 3
        ):
            continue
        add_candidate(
            f"lb:{prev_index}",
            {
                "k": "lb",
                "after": prev_index,
                "prev": _truncate_text(prev_text, 70),
                "next": _truncate_text(next_text, 70),
            },
        )

    return candidates, candidate_map


def _build_block_prompt_payload(block: dict) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Sérialise un bloc en payload prompt + candidats de frontières."""
    lines = block.get("lines") or []
    role = str(block.get("role") or "body")
    bb_ref = block.get("bbox") or [0, 0, 1000, 1000]
    bx0, by0 = float(bb_ref[0]), float(bb_ref[1])
    bw = max(1.0, float(bb_ref[2]) - bx0)
    bh = max(1.0, float(bb_ref[3]) - by0)

    formatted = []
    for idx, ln in enumerate(lines[:_MAX_LINES]):
        li = int(ln.get("line_index", idx) or idx)
        text = (ln.get("line_text") or ln.get("text") or "").strip()[:80]
        bbox = _normalise_bbox(ln.get("bbox"), bx0, by0, bw, bh)
        entry = {"i": li, "t": text, "bb": bbox}
        phrase_candidates = []
        for phrase in (ln.get("phrases") or []):
            if not isinstance(phrase, dict):
                continue
            phrase_text = str(phrase.get("texte") or phrase.get("text") or "").strip()
            if phrase_text:
                phrase_candidates.append(phrase_text[:60])
        if len(phrase_candidates) >= 2:
            entry["ph"] = phrase_candidates[:4]
        formatted.append(entry)

    heuristic_phrases = []
    for phrase in (block.get("semantic_phrases") or [])[:8]:
        if not isinstance(phrase, dict):
            continue
        phrase_text = _normalize_text(phrase.get("text") or phrase.get("texte") or "")
        if not phrase_text:
            continue
        heuristic_phrases.append(
            {
                "li": [int(x) for x in (phrase.get("line_indices") or [])[:4]],
                "t": phrase_text[:90],
                "r": str(phrase.get("sentence_end_reason") or ""),
            }
        )

    payload = {"role": role, "lines": formatted}
    source_layout_mode = block.get("source_layout_mode")
    if isinstance(source_layout_mode, dict) and source_layout_mode:
        payload["lm0"] = {
            "mode": str(source_layout_mode.get("mode") or ""),
            "line_flow": str(source_layout_mode.get("line_flow") or ""),
            "render_contract": str(source_layout_mode.get("render_contract") or ""),
        }
    if heuristic_phrases:
        payload["hs"] = heuristic_phrases
    candidates, candidate_map = _build_boundary_candidates(block)
    if candidates:
        payload["bc"] = candidates
    return payload, candidate_map


def format_block_for_prompt(block: dict) -> str:
    """Sérialise un bloc en JSON compact adapté au prompt."""
    payload, _ = _build_block_prompt_payload(block)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Construction du prompt et appel LLM
# ---------------------------------------------------------------------------

def _build_messages(block_json: str) -> list[dict]:
    """Construit la conversation few-shot pour le chat template."""
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for user_ex, assistant_ex in _FEW_SHOT_EXAMPLES[:_MAX_FEW_SHOT_EXAMPLES]:
        messages.append({"role": "user", "content": user_ex})
        messages.append({"role": "assistant", "content": assistant_ex})
    messages.append({"role": "user", "content": block_json})
    return messages


def _build_boundary_vote_messages(block: dict, candidate: dict[str, Any]) -> list[dict]:
    lines = []
    for line in (block.get("lines") or [])[:_MAX_LINES]:
        if not isinstance(line, dict):
            continue
        li = int(line.get("line_index", 0) or 0)
        text = _truncate_text(line.get("line_text") or line.get("text") or "", 120)
        if text:
            lines.append(f"{li}: {text}")

    candidate_kind = str(candidate.get("k") or "")
    if candidate_kind == "sb":
        candidate_desc = (
            f'Candidate {candidate.get("id")}: split line {candidate.get("i")} into '
            f'{json.dumps(candidate.get("s") or [], ensure_ascii=False)}'
        )
    else:
        candidate_desc = (
            f'Candidate {candidate.get("id")}: break after line {candidate.get("after")}. '
            f'prev={candidate.get("prev","")} | next={candidate.get("next","")}'
        )

    user_content = "\n".join(
        [
            "Block lines:",
            *lines,
            "",
            candidate_desc,
            "",
            "Answer YES or NO only.",
        ]
    )
    return [
        {"role": "system", "content": _BOUNDARY_VOTE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _render_prompt(runtime: dict[str, Any], messages: list[dict]) -> Any:
    tokenizer = runtime["tokenizer"]
    if runtime.get("backend") == "ct2":
        max_input_tokens = _configured_max_input_tokens()
        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=max_input_tokens,
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
            input_ids = tokenizer(
                prompt,
                truncation=True,
                max_length=max_input_tokens,
            )["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return tokenizer.convert_ids_to_tokens(input_ids)
    if hasattr(tokenizer, "apply_chat_template"):
        max_input_tokens = _configured_max_input_tokens()
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    max_input_tokens = _configured_max_input_tokens()
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    return encoded["input_ids"]


def _extract_outermost_json(text: str) -> str | None:
    """Extrait le premier objet JSON complet en comptant les accolades imbriquées."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_response(raw: str) -> dict:
    """Extrait et valide les corrections et hyphen-joins depuis la réponse brute du LLM.

    Retourne un dict {"c": [...], "bc": [...], "sb": [...], "lb": [...], "lm": {...}|None, "hj": [...]}.
    """
    json_str = _extract_outermost_json(raw)
    if not json_str:
        return {"c": [], "bc": [], "sb": [], "lb": [], "lm": None, "hj": []}
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        m = re.search(r'"c"\s*:\s*(\[.*?\])', raw, re.DOTALL)
        if not m:
            return {"c": [], "bc": [], "sb": [], "lb": [], "lm": None, "hj": []}
        try:
            parsed = {"c": json.loads(m.group(1)), "bc": [], "sb": [], "lb": [], "lm": None, "hj": []}
        except json.JSONDecodeError:
            return {"c": [], "bc": [], "sb": [], "lb": [], "lm": None, "hj": []}

    # --- Corrections ---
    corrections_raw = parsed.get("c") or []
    if not isinstance(corrections_raw, list):
        corrections_raw = []

    valid_corrections: list[dict] = []
    valid_actions = {"heading", "formula", "skip", "atomic", "keep"}
    for corr in corrections_raw:
        if not isinstance(corr, dict):
            continue
        action = str(corr.get("a") or "keep")
        if action not in valid_actions:
            continue
        li = corr.get("li")
        if not isinstance(li, list) or not li:
            continue
        try:
            valid_corrections.append({"li": [int(x) for x in li], "a": action})
        except (ValueError, TypeError):
            continue

    # Rejeter si le LLM marque TOUTES les lignes comme "skip" (réponse fausse)
    if valid_corrections and all(c["a"] == "skip" for c in valid_corrections):
        valid_corrections = []

    # --- Boundary candidate ids ---
    bc_raw = parsed.get("bc") or []
    if not isinstance(bc_raw, list):
        bc_raw = []

    valid_bc: list[str] = []
    for candidate_id in bc_raw:
        candidate_text = str(candidate_id or "").strip()
        if candidate_text:
            valid_bc.append(candidate_text)
    valid_bc = list(dict.fromkeys(valid_bc))

    # --- Intra-line sentence boundaries ---
    sb_raw = parsed.get("sb") or []
    if not isinstance(sb_raw, list):
        sb_raw = []

    valid_sb: list[dict] = []
    for split in sb_raw:
        if not isinstance(split, dict):
            continue
        try:
            idx = int(split.get("i", -1))
        except (ValueError, TypeError):
            continue
        segments = split.get("s")
        if idx < 0 or not isinstance(segments, list):
            continue
        normalized_segments = [str(seg or "").strip() for seg in segments]
        normalized_segments = [seg for seg in normalized_segments if seg]
        if len(normalized_segments) < 2:
            continue
        valid_sb.append({"i": idx, "s": normalized_segments})

    # --- Inter-line semantic boundaries ---
    lb_raw = parsed.get("lb") or []
    if not isinstance(lb_raw, list):
        lb_raw = []

    valid_lb: list[int] = []
    for line_index in lb_raw:
        try:
            idx = int(line_index)
        except (ValueError, TypeError):
            continue
        if idx < 0:
            continue
        valid_lb.append(idx)
    valid_lb = sorted(set(valid_lb))

    # --- Layout mode override ---
    lm_raw = parsed.get("lm")
    valid_lm = None
    if isinstance(lm_raw, dict):
        line_flow = str(lm_raw.get("line_flow") or "").strip().lower()
        valid_line_flows = {"inline_reflow", "preserve_line_breaks", "preserve_paragraphs", "fixed_lines"}
        if line_flow in valid_line_flows:
            breaks = []
            for item in lm_raw.get("breaks") or []:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("i", -1))
                except (TypeError, ValueError):
                    continue
                after = str(item.get("after") or "").strip().lower()
                if idx < 0 or after not in {"soft_wrap", "new_line", "paragraph_break", "list_item", "fixed_line", "block_end"}:
                    continue
                breaks.append({"i": idx, "after": after})
            valid_lm = {"line_flow": line_flow, "breaks": breaks}

    # --- Hyphen joins ---
    hj_raw = parsed.get("hj") or []
    if not isinstance(hj_raw, list):
        hj_raw = []

    valid_hj: list[dict] = []
    for hj in hj_raw:
        if not isinstance(hj, dict):
            continue
        try:
            idx = int(hj.get("i", -1))
        except (ValueError, TypeError):
            continue
        if idx < 0:
            continue
        word = str(hj.get("w") or "").strip()
        if word:
            valid_hj.append({"i": idx, "w": word})

    return {"c": valid_corrections, "bc": valid_bc, "sb": valid_sb, "lb": valid_lb, "lm": valid_lm, "hj": valid_hj}


# ---------------------------------------------------------------------------
# Scoring d'ambiguïté
# ---------------------------------------------------------------------------

def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _tokenize_words(text: Any) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", normalized.lower())


def _shared_edge_word_count(left: Any, right: Any, min_words: int = 3, max_words: int = 8) -> int:
    left_words = _tokenize_words(left)
    right_words = _tokenize_words(right)
    max_overlap = min(max_words, len(left_words), len(right_words))
    for overlap in range(max_overlap, min_words - 1, -1):
        if left_words[-overlap:] == right_words[:overlap]:
            return overlap
    return 0


def _line_has_carryover_then_new_sentence(text: Any) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if not re.match(
        r"^(and|or|but|so|yet|nor|for|because|however|therefore|thus|then|also|instead|rather|meanwhile|whereas)\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        return False
    return bool(re.search(r'[.!?]["\')\]]*\s+[A-Z]', normalized))


def _block_has_overlap_artifact(block: dict) -> bool:
    semantic_phrases = block.get("semantic_phrases") or []
    for previous_phrase, current_phrase in zip(semantic_phrases, semantic_phrases[1:]):
        if _shared_edge_word_count(previous_phrase.get("text"), current_phrase.get("text")) >= 3:
            return True

    for line in block.get("lines") or []:
        line_text = line.get("line_text") or line.get("text") or ""
        if _line_has_carryover_then_new_sentence(line_text):
            return True
    return False


def block_needs_strong_retry(block: dict, score: float, result: dict | None = None) -> bool:
    if score < STRONG_AMBIGUITY_THRESHOLD:
        return False
    if result:
        if any(result.get(key) for key in ("c", "bc", "sb", "lb", "hj")):
            return False
    if _block_has_overlap_artifact(block):
        return True

    for phrase in (block.get("semantic_phrases") or []):
        if phrase.get("sentence_end_reason") != "eof":
            continue
        if len(_tokenize_words(phrase.get("text"))) < 4:
            continue
        if not re.search(r'[.!?]["\')\]]*$', _normalize_text(phrase.get("text"))):
            return True
    return False

def score_block_ambiguity(block: dict) -> float:
    """Retourne un score [0,1] indiquant le besoin de correction LLM.

    Déclenché pour :
    - Phrases multi-lignes terminées par "eof" sur des lignes de longueur normale
    - Blocs contenant des lignes très courtes (potentiels labels ou titres isolés)
    - Lignes dont le texte ressemble à un titre collé à du corps de texte
    - Lignes se terminant par un tiret (césure de mot potentielle)
    """
    sps = block.get("semantic_phrases") or []
    lines = block.get("lines") or []
    if not lines or not sps:
        return 0.0

    overlap_artifact = _block_has_overlap_artifact(block)

    full_text = " ".join(
        (ln.get("line_text") or ln.get("text") or "").strip()
        for ln in lines
        if (ln.get("line_text") or ln.get("text") or "").strip()
    ).strip()
    if _MAX_BLOCK_CHARS > 0 and len(full_text) > _MAX_BLOCK_CHARS and not overlap_artifact:
        return 0.0
    if _MAX_BLOCK_WORDS > 0 and len(full_text.split()) > _MAX_BLOCK_WORDS and not overlap_artifact:
        return 0.0

    # Raisons produites par des heuristiques explicites → bloc déjà résolu
    _RESOLVED_REASONS = {
        "heading_line", "atomic_line", "separator_line",
        "terminal_punctuation", "hard_break_before", "word_count_cap",
        "llm_corrected", "llm_split", "llm_sentence_boundary", "pre_heading", "pre_atomic",
        "pre_formula", "formula",
    }
    all_reasons = {sp.get("sentence_end_reason") for sp in sps}
    if all_reasons <= _RESOLVED_REASONS and not overlap_artifact:
        return 0.0

    line_texts = [
        (ln.get("line_text") or ln.get("text") or "").strip()
        for ln in lines
    ]

    # Signal 1 : phrases multi-lignes sans frontière détectée
    eof_multiline = sum(
        1 for sp in sps
        if sp.get("sentence_end_reason") == "eof" and sp.get("multi_line")
    )
    dangling_eof = sum(
        1
        for sp in sps
        if sp.get("sentence_end_reason") == "eof"
        and len(_tokenize_words(sp.get("text"))) >= 4
        and not re.search(r'[.!?]["\')\]]*$', _normalize_text(sp.get("text")))
    )

    # Signal 2 : lignes très courtes (potentiels labels d'axe ou titres isolés)
    very_short_lines = sum(1 for t in line_texts if 0 < len(t) <= 3)

    # Signal 3 : ligne se terminant par un tiret (césure)
    hyphen_lines = sum(1 for t in line_texts if t.endswith("-"))

    # Signal 4 : premier mot de la ligne ressemble à un titre de section
    # (ex: "Figure 4.17", "Too-high vs. too-low learning rate Setting...")
    caption_start = any(
        re.match(r'^(Figure|Fig\.|Table|Tab\.|Listing)\s+\d', t, re.IGNORECASE)
        for t in line_texts
    )

    # Signal 5 : la ligne possède déjà plusieurs sous-phrases OCR/natives,
    # mais la segmentation sémantique n'a pas encore été clarifiée.
    inline_phrase_candidates = sum(
        1
        for ln in lines
        if len([p for p in (ln.get("phrases") or []) if isinstance(p, dict) and str(p.get("texte") or p.get("text") or "").strip()]) >= 2
    )
    overlap_candidates = 1 if overlap_artifact else 0

    score = 0.0
    if eof_multiline:
        score += 0.35 * eof_multiline
    if dangling_eof:
        score += 0.30 * dangling_eof
    if very_short_lines:
        score += 0.30 * very_short_lines
    if hyphen_lines:
        score += 0.20 * hyphen_lines
    if caption_start:
        score += 0.25
    if inline_phrase_candidates:
        score += 0.20 * inline_phrase_candidates
    if overlap_candidates:
        score += 0.45

    return min(1.0, score)


# ---------------------------------------------------------------------------
# Cache en mémoire
# ---------------------------------------------------------------------------

_cache: dict[str, dict] = {}
_PROMPT_CACHE_VERSION = "semantic-corrector-v3"


def _resolve_selected_boundary_candidates(
    parsed: dict[str, Any],
    candidate_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidate_map = candidate_map or {}
    resolved = {
        "c": list(parsed.get("c") or []),
        "bc": list(parsed.get("bc") or []),
        "sb": list(parsed.get("sb") or []),
        "lb": list(parsed.get("lb") or []),
        "lm": parsed.get("lm") if isinstance(parsed.get("lm"), dict) else None,
        "hj": list(parsed.get("hj") or []),
    }

    for candidate_id in resolved["bc"]:
        candidate = candidate_map.get(candidate_id)
        if not candidate:
            continue
        kind = str(candidate.get("k") or "")
        if kind == "sb":
            resolved["sb"].append(
                {
                    "i": int(candidate.get("i", -1)),
                    "s": [str(segment or "").strip() for segment in (candidate.get("s") or []) if str(segment or "").strip()],
                }
            )
        elif kind == "lb":
            try:
                resolved["lb"].append(int(candidate.get("after", -1)))
            except (TypeError, ValueError):
                continue

    deduped_sb: list[dict[str, Any]] = []
    seen_sb: set[tuple[int, tuple[str, ...]]] = set()
    for split in resolved["sb"]:
        try:
            idx = int(split.get("i", -1))
        except (ValueError, TypeError, AttributeError):
            continue
        segments = [str(segment or "").strip() for segment in (split.get("s") or []) if str(segment or "").strip()]
        key = (idx, tuple(segments))
        if idx < 0 or len(segments) < 2 or key in seen_sb:
            continue
        seen_sb.add(key)
        deduped_sb.append({"i": idx, "s": segments})
    resolved["sb"] = deduped_sb

    resolved["lb"] = sorted({int(idx) for idx in resolved["lb"] if isinstance(idx, int) or str(idx).isdigit()})
    return resolved


def _generate_raw_response(pipe: Any, messages: list[dict], max_new_tokens: int) -> str:
    backend = str(pipe.get("backend") or "transformers")
    tokenizer = pipe["tokenizer"]
    rendered = _render_prompt(pipe, messages)
    if backend == "ct2":
        generator = pipe["generator"]
        generation_kwargs = {
            "beam_size": 1,
            "sampling_topk": 1,
            "sampling_topp": 1.0,
            "sampling_temperature": 1.0,
            "max_length": max_new_tokens,
            "include_prompt_in_result": False,
        }
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token:
            generation_kwargs["end_token"] = eos_token
        output = generator.generate_batch([rendered], **generation_kwargs)
        generated_ids = getattr(output[0], "sequences_ids", None)
        if generated_ids:
            return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        generated_tokens = getattr(output[0], "sequences", [[]])[0]
        return tokenizer.decode(tokenizer.convert_tokens_to_ids(generated_tokens), skip_special_tokens=True).strip()

    model = pipe["model"]
    torch = pipe["torch"]
    if isinstance(rendered, dict):
        input_ids = rendered["input_ids"]
        attention_mask = rendered.get("attention_mask")
    else:
        input_ids = rendered
        attention_mask = None

    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": _USE_CACHE,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if attention_mask is not None:
        generation_kwargs["attention_mask"] = attention_mask
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id

    with torch.inference_mode():
        output = model.generate(**generation_kwargs)

    prompt_len = int(input_ids.shape[-1])
    generated_ids = output[0][prompt_len:]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not raw:
        raw = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return raw


def _parse_yes_no_answer(raw: str) -> bool | None:
    normalized = str(raw or "").strip().upper()
    if re.match(r"^(YES|Y)\b", normalized):
        return True
    if re.match(r"^(NO|N)\b", normalized):
        return False
    return None


def _score_boundary_candidate(block: dict, candidate: dict[str, Any]) -> float:
    kind = str(candidate.get("k") or "")
    score = 0.0
    if kind == "sb":
        segments = [str(segment or "").strip() for segment in (candidate.get("s") or []) if str(segment or "").strip()]
        if len(segments) >= 2:
            if _line_ends_with_terminal_punctuation(segments[0]):
                score += 0.4
            if _line_starts_like_new_sentence(segments[1]):
                score += 0.3
            if _line_has_carryover_then_new_sentence(" ".join(segments[:2])):
                score += 0.5
    elif kind == "lb":
        prev_text = str(candidate.get("prev") or "")
        next_text = str(candidate.get("next") or "")
        if _shared_edge_word_count(prev_text, next_text) >= 3:
            score += 0.7
        if _line_starts_like_new_sentence(next_text):
            score += 0.3
        if _line_ends_with_terminal_punctuation(prev_text):
            score += 0.2
    if _block_has_overlap_artifact(block):
        score += 0.2
    return score


def _auto_select_boundary_candidates(block: dict, candidate_map: dict[str, dict[str, Any]]) -> list[str]:
    selected_ids: list[str] = []
    for candidate in candidate_map.values():
        candidate_id = str(candidate.get("id") or "")
        kind = str(candidate.get("k") or "")
        if not candidate_id:
            continue

        if kind == "sb":
            segments = [str(segment or "").strip() for segment in (candidate.get("s") or []) if str(segment or "").strip()]
            if len(segments) < 2:
                continue
            if _line_ends_with_terminal_punctuation(segments[0]) and _line_starts_like_new_sentence(segments[1]):
                selected_ids.append(candidate_id)
                continue
            if _line_has_carryover_then_new_sentence(" ".join(segments[:2])):
                selected_ids.append(candidate_id)
                continue

        if kind == "lb":
            prev_text = str(candidate.get("prev") or "")
            next_text = str(candidate.get("next") or "")
            if _line_ends_with_terminal_punctuation(prev_text) and _line_starts_like_new_sentence(next_text):
                selected_ids.append(candidate_id)
                continue
            if _shared_edge_word_count(prev_text, next_text) >= 3 and _line_has_carryover_then_new_sentence(next_text):
                selected_ids.append(candidate_id)
                continue

    return list(dict.fromkeys(selected_ids))


def _select_boundary_candidates_via_votes(
    block: dict,
    pipe: Any,
    candidate_map: dict[str, dict[str, Any]],
    max_candidates: int | None = None,
) -> list[str]:
    if not candidate_map:
        return []
    vote_budget = _MAX_CANDIDATE_VOTES_PER_BLOCK if max_candidates is None else max_candidates
    if vote_budget <= 0:
        return []

    ranked_candidates = sorted(
        candidate_map.values(),
        key=lambda candidate: _score_boundary_candidate(block, candidate),
        reverse=True,
    )[:vote_budget]

    selected_ids: list[str] = []
    for candidate in ranked_candidates:
        messages = _build_boundary_vote_messages(block, candidate)
        try:
            raw = _generate_raw_response(pipe, messages, max_new_tokens=4)
        except Exception as exc:
            logger.debug("Boundary vote failed: %s", exc)
            continue
        keep = _parse_yes_no_answer(raw)
        if keep:
            selected_ids.append(str(candidate.get("id") or ""))
    return [candidate_id for candidate_id in selected_ids if candidate_id]


def get_corrections(block: dict, pipe: Any) -> dict:
    """Appelle le LLM et retourne les corrections + hyphen joins (avec cache).

    Retourne un dict {"c": [...], "sb": [...], "lb": [...], "lm": {...}|None, "hj": [...]}.
    """
    payload, candidate_map = _build_block_prompt_payload(block)
    block_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    backend = str(pipe.get("backend") or "transformers")
    model_id = str(pipe.get("model_id") or "")
    key_payload = json.dumps(
        {
            "v": _PROMPT_CACHE_VERSION,
            "model_id": model_id,
            "backend": backend,
            "block": block_json,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    key = hashlib.md5(key_payload.encode("utf-8")).hexdigest()
    if key in _cache:
        return _cache[key]

    messages = _build_messages(block_json)
    try:
        max_new_tokens = _configured_max_new_tokens()
        raw = _generate_raw_response(pipe, messages, max_new_tokens=max_new_tokens)
    except Exception as exc:
        logger.debug("Appel LLM échoué : %s", exc)
        result: dict = {"c": [], "bc": [], "sb": [], "lb": [], "lm": None, "hj": []}
        _cache[key] = result
        return result

    result = _resolve_selected_boundary_candidates(_parse_response(raw), candidate_map=candidate_map)
    if not any(result.get(key) for key in ("c", "bc", "sb", "lb", "lm", "hj")) and candidate_map:
        auto_selected_ids = _auto_select_boundary_candidates(block, candidate_map)
        voted_ids: list[str] = []
        if len(auto_selected_ids) < len(candidate_map):
            remaining_candidates = {
                candidate_id: candidate
                for candidate_id, candidate in candidate_map.items()
                if candidate_id not in auto_selected_ids
            }
            voted_ids = _select_boundary_candidates_via_votes(block, pipe, remaining_candidates)
        selected_ids = list(dict.fromkeys(auto_selected_ids + voted_ids))
        if selected_ids:
            result = _resolve_selected_boundary_candidates(
                {"c": [], "bc": selected_ids, "sb": [], "lb": [], "lm": None, "hj": []},
                candidate_map=candidate_map,
            )
    _cache[key] = result
    logger.debug(
        "LLM corrector [%s|%s|%s…] → %d corrections, %d bc, %d sb, %d lb, lm=%s, %d hj",
        model_id,
        backend,
        block_json[:60],
        len(result["c"]),
        len(result["bc"]),
        len(result["sb"]),
        len(result["lb"]),
        bool(result.get("lm")),
        len(result["hj"]),
    )
    return result


# ---------------------------------------------------------------------------
# Point d'entrée public
# ---------------------------------------------------------------------------

def load_pipeline_if_needed(model_id: str | None = None) -> Any | None:
    """Charge le pipeline si nécessaire. Retourne None si indisponible."""
    return _load_pipeline(model_id=model_id)


def get_runtime_model_info(model_id: str | None = None) -> dict[str, str]:
    """Expose le backend sélectionné pour les tests et le debug."""
    spec = _resolve_model_spec_for_runtime() if model_id is None else _resolve_model_spec(model_id)
    return {
        "requested": str(spec.get("requested") or ""),
        "model_id": str(spec.get("model_id") or ""),
        "family": str(spec.get("family") or ""),
        "backend": str(spec.get("backend") or ""),
        "ct2_model_dir": str(spec.get("ct2_model_dir") or ""),
    }
