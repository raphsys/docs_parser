"""
pipeline_agents — agents open-source par étape du pipeline docs_parser.

Chaque agent (P1-P6) encapsule un LLM local (Phi-3.5 Mini, Qwen2.5, GGUF…)
et expose une interface identique, provider-agnostique.

Utilisation rapide :
    from pipeline_agents import get_agent

    agent = get_agent("p5_render")
    result = agent.run({"role": "body", "line_count": 4, ...})
    # result = {"strategy": "prose_reflow", "confidence": 0.9, ...}

Variables d'environnement :
    PIPELINE_AGENT_MODEL          modèle par défaut (alias ou chemin)
    PIPELINE_AGENT_BACKEND        "auto" | "ct2" | "transformers"
    PIPELINE_AGENT_P1_MODEL       modèle spécifique pour P1
    PIPELINE_AGENT_P5_MODEL       modèle spécifique pour P5
    ...etc.
    PIPELINE_AGENT_CPU_THREADS    nombre de threads CPU (défaut : 2)
    PIPELINE_AGENT_MAX_INPUT_TOKENS  longueur max du prompt en tokens (défaut : 512)
"""

from .base import ModelRuntime, ModelSpec, PipelineAgent
from .registry import AgentRegistry, get_agent, list_agents
from .p1_extraction import P1ExtractionAgent
from .p2_structure import P2StructureAgent
from .p3_layout import P3LayoutAgent
from .p4_translation import P4TranslationAgent
from .p5_render import P5RenderAgent
from .p6_background import P6BackgroundAgent

__all__ = [
    "ModelRuntime",
    "ModelSpec",
    "PipelineAgent",
    "AgentRegistry",
    "get_agent",
    "list_agents",
    "P1ExtractionAgent",
    "P2StructureAgent",
    "P3LayoutAgent",
    "P4TranslationAgent",
    "P5RenderAgent",
    "P6BackgroundAgent",
]
