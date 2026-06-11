"""Pipelines — orchestrateur du pipeline WYSIWYG (clone objectif d'ocr_server.py).

Architecture cible :

    API /ocr
      ↓
    PipelineOrchestrator
      ↓
    SourceLoader
      ↓
    PageRenderer
      ↓
    RawExtractors
      ↓
    PageUnderstanding
      ↓
    PAGEPRINT Builder
      ↓
    INPUT_DATA

Le serveur devient mince : il ne contient plus l'intelligence.
ocr_server.py historique n'est pas modifié — ce package est le nouvel
orchestrateur, construit unité par unité, dont la première tête est PAGEPRINT.
"""

from .orchestrator import PipelineOrchestrator

__all__ = ["PipelineOrchestrator"]
