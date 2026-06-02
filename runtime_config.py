from __future__ import annotations

import os
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
AI_MODELS_ROOT = PIPELINE_ROOT / "ai_models"
TRANSLATION_ROOT = AI_MODELS_ROOT / "translation"


def _set_default_env(key: str, value: str) -> None:
    if os.getenv(key) in {None, ""}:
        os.environ[key] = value


def configure_agentless_environment() -> dict[str, str]:
    config = {
        "DOCS_PARSER_PIPELINE_VARIANT": "agentless",
        "PIPELINE_AGENT_P1_ENABLE": "0",
        "PIPELINE_AGENT_P3_ENABLE": "0",
        "PIPELINE_AGENT_P4_ENABLE": "0",
        "PIPELINE_AGENT_P6_ENABLE": "0",
        "DOCS_PARSER_ENABLE_SEMANTIC_LLM": "0",
        "DOCS_PARSER_ENABLE_EXTRACTION_AI": "0",
        "ELEMENT_RELATIONS_AI_ENABLE": "0",
        "LAYOUT_AI_ENABLE": "0",
        "BACKGROUND_INPAINT_MODELS_ROOT": str(AI_MODELS_ROOT / "inpainting"),
        "LAYOUT_AI_MODELS_ROOT": str(AI_MODELS_ROOT / "ppstructurev3"),
        "ELEMENT_RELATIONS_AI_MODEL_DIR": str(AI_MODELS_ROOT / "element_relations_nli"),
        "TRANSLATION_PROFILES_PATH": str(TRANSLATION_ROOT / "translation_profiles.json"),
        "TRANSLATION_STYLE_TONE_PROFILES_PATH": str(TRANSLATION_ROOT / "style_tone_profiles.json"),
        "TRANSLATION_MODEL_INVENTORY_PATH": str(TRANSLATION_ROOT / "model_inventory.json"),
        "TRANSLATOR_GLOSSARY_DIR": str(TRANSLATION_ROOT / "glossaries"),
        "TRANSLATOR_TERMINOLOGY_TABLE": str(TRANSLATION_ROOT / "glossaries" / "terminology_master.csv"),
        "TRANSLATION_MEMORY_PATH": str(TRANSLATION_ROOT / "translation_memory.jsonl"),
        "CT2_MODEL_DIR": str(TRANSLATION_ROOT / "m2m100_418m_ct2_int8"),
        "CT2_TOKENIZER_DIR": str(TRANSLATION_ROOT / "m2m100_418m_tokenizer"),
        "CT2_FALLBACK_MODEL_DIR": str(TRANSLATION_ROOT / "m2m100_418m_ct2_int8"),
        "CT2_FALLBACK_TOKENIZER_DIR": str(TRANSLATION_ROOT / "m2m100_418m_tokenizer"),
        "CT2_ENFR_MODEL_DIR": str(TRANSLATION_ROOT / "opus_mt_tc_big_en_fr_ct2_int8"),
        "CT2_ENFR_TOKENIZER_DIR": str(TRANSLATION_ROOT / "opus_mt_tc_big_en_fr_tokenizer"),
    }
    for key, value in config.items():
        _set_default_env(key, value)
    return config
