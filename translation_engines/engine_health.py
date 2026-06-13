from __future__ import annotations


def build_engine_health(engine, *, registry_health: dict | None = None) -> dict:
    health = {}
    try:
        health = engine.healthcheck()
    except Exception as exc:
        health = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if registry_health is not None:
        health["registry"] = registry_health
    return health
