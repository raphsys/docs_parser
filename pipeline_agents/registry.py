"""
Registre des agents du pipeline — factory provider-agnostique.

Usage :
    from pipeline_agents import get_agent, AgentRegistry

    agent = get_agent("p5_render")          # modèle par défaut (env ou phi35)
    agent = get_agent("p5_render", model="qwen-small")
    agent = get_agent("p1_extraction", model="/path/to/local/model")

    result = agent.run(input_data)
"""

from __future__ import annotations

import os
from typing import Type

from .base import ModelRuntime, ModelSpec, PipelineAgent


# ---------------------------------------------------------------------------
# Modèle par défaut
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = os.environ.get(
    "PIPELINE_AGENT_MODEL",
    "phi35",  # phi35-mini-instruct local si disponible, sinon download HF
)
_DEFAULT_BACKEND = os.environ.get("PIPELINE_AGENT_BACKEND", "auto")

# Modèles spécialisés par étape (surchargeable via PIPELINE_AGENT_P5_MODEL, etc.)
_STAGE_MODELS: dict[str, str] = {
    "p1_extraction": os.environ.get("PIPELINE_AGENT_P1_MODEL", ""),
    "p2_structure": os.environ.get("PIPELINE_AGENT_P2_MODEL", ""),
    "p3_layout": os.environ.get("PIPELINE_AGENT_P3_MODEL", ""),
    "p4_translation": os.environ.get("PIPELINE_AGENT_P4_MODEL", ""),
    "p5_render": os.environ.get("PIPELINE_AGENT_P5_MODEL", ""),
    "p6_background": os.environ.get("PIPELINE_AGENT_P6_MODEL", ""),
}


# ---------------------------------------------------------------------------
# Registre
# ---------------------------------------------------------------------------

class AgentRegistry:
    """
    Registre singleton des classes d'agents.

    Permet l'enregistrement de classes personnalisées et la factory ``get``.
    """

    _classes: dict[str, Type[PipelineAgent]] = {}
    _instances: dict[str, PipelineAgent] = {}

    @classmethod
    def register(cls, stage: str, agent_class: Type[PipelineAgent]) -> None:
        cls._classes[stage] = agent_class

    @classmethod
    def get(
        cls,
        stage: str,
        model: str | None = None,
        backend: str | None = None,
    ) -> PipelineAgent:
        """
        Retourne l'agent pour l'étape ``stage``.

        L'instance est mise en cache par (stage, model, backend) pour réutiliser
        le runtime LLM déjà chargé.
        """
        agent_class = cls._classes.get(stage)
        if agent_class is None:
            raise ValueError(
                f"Agent '{stage}' non enregistré. "
                f"Agents disponibles : {sorted(cls._classes)}"
            )
        resolved_model = model or _STAGE_MODELS.get(stage) or _DEFAULT_MODEL
        resolved_backend = backend or _DEFAULT_BACKEND
        cache_key = f"{stage}|{resolved_model}|{resolved_backend}"
        if cache_key not in cls._instances:
            runtime = ModelRuntime.for_model(resolved_model, backend=resolved_backend)
            cls._instances[cache_key] = agent_class(runtime)
        return cls._instances[cache_key]

    @classmethod
    def available_stages(cls) -> list[str]:
        return sorted(cls._classes)

    @classmethod
    def clear_instances(cls) -> None:
        """Libère les instances mises en cache (utile pour les tests)."""
        cls._instances.clear()


# ---------------------------------------------------------------------------
# Enregistrement automatique des agents built-in
# ---------------------------------------------------------------------------

def _register_builtins() -> None:
    from .p1_extraction import P1ExtractionAgent
    from .p2_structure import P2StructureAgent
    from .p3_layout import P3LayoutAgent
    from .p4_translation import P4TranslationAgent
    from .p5_render import P5RenderAgent
    from .p6_background import P6BackgroundAgent

    AgentRegistry.register("p1_extraction", P1ExtractionAgent)
    AgentRegistry.register("p2_structure", P2StructureAgent)
    AgentRegistry.register("p3_layout", P3LayoutAgent)
    AgentRegistry.register("p4_translation", P4TranslationAgent)
    AgentRegistry.register("p5_render", P5RenderAgent)
    AgentRegistry.register("p6_background", P6BackgroundAgent)


_register_builtins()


# ---------------------------------------------------------------------------
# Helpers publics
# ---------------------------------------------------------------------------

def get_agent(
    stage: str,
    model: str | None = None,
    backend: str | None = None,
) -> PipelineAgent:
    """Retourne l'agent pour l'étape ``stage`` (factory principale)."""
    return AgentRegistry.get(stage, model=model, backend=backend)


def list_agents() -> list[str]:
    """Liste les étapes disponibles."""
    return AgentRegistry.available_stages()
