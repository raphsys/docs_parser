"""PAGEPRINT Provenance — mémoire de décision et pipeline rejouable.

Chaque décision importante doit être traçable : pourquoi ce bloc est une
légende, pourquoi cette phrase n'est pas traduite, pourquoi cette zone est
une formule, pourquoi ces fragments ont été fusionnés.

Inclut aussi la couche replay : chaque INPUT_DATA doit permettre de rejouer
une partie du pipeline (hashes source/rendu/config, versions de modules).
Sans provenance, on ne peut pas corriger les erreurs finement.
"""

from __future__ import annotations

import hashlib
import json
import os

from .schema import PAGEPRINT_SCHEMA_VERSION


def file_hash(path: str | None) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def config_hash(config: dict | None) -> str | None:
    if not config:
        return None
    try:
        payload = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except Exception:
        return None


def collect_decision_traces(units: list[dict],
                            extra_traces: list[dict] | None = None) -> list[dict]:
    """Agrège les decision_trace des unités + traces additionnelles."""
    traces: list[dict] = list(extra_traces or [])
    for unit in units:
        for trace in (unit.get("provenance") or {}).get("decision_trace") or []:
            traces.append(trace)
    return traces


def build_replay(*, source_path: str | None = None,
                 render_path: str | None = None,
                 config: dict | None = None,
                 environment_flags: dict | None = None,
                 module_versions: dict | None = None) -> dict:
    return {
        "source_file_hash": file_hash(source_path),
        "page_render_hash": file_hash(render_path),
        "config_hash": config_hash(config),
        "module_versions": module_versions or {},
        "random_seed": None,
        "environment_flags": environment_flags or {},
    }


def build_provenance(*, page_structure: dict | None = None,
                     units: list[dict] | None = None,
                     created_by: str = "pageprint.builder",
                     modules: dict | None = None,
                     extra_traces: list[dict] | None = None,
                     replay: dict | None = None) -> dict:
    structure = page_structure or {}
    detected_modules = {
        "pageprint": "enabled",
        "layout_v2_builder": "enabled" if structure.get("schema_version") == "layout.v2"
        or structure.get("layout") else "unknown",
        "page_case_classifier_v2": "enabled" if structure.get("page_case_v2") else "unknown",
        "special_region_detector": "enabled" if structure.get("special_regions")
        or (structure.get("layout") or {}).get("special_regions") else "unknown",
        "element_relations": "enabled" if structure.get("element_relations") else "unknown",
        "page_policy_matrix": "enabled" if any(
            isinstance(b, dict) and b.get("translation_contract")
            for b in structure.get("blocks") or []
        ) else "unknown",
    }
    if modules:
        detected_modules.update(modules)

    provenance = {
        "created_by": created_by,
        "pipeline_version": PAGEPRINT_SCHEMA_VERSION,
        "modules": detected_modules,
        "decision_trace": collect_decision_traces(units or [], extra_traces),
    }
    if replay:
        provenance["replay"] = replay
    return provenance
