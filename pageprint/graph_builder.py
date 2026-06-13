"""PAGEPRINT Graph Builder — construit le graphe documentaire.

Une page n'est pas seulement une liste de blocs. C'est un graphe :
    page contient region, region contient block, block contient line,
    line contient phrase, phrase contient span, caption décrit image,
    label annote schéma, cell appartient à table, phrase continue phrase
    précédente.

Graphes produits : reading graph, containment graph, layout graph,
semantic graph, visual attachment graph — fusionnés en nodes + edges.
"""

from __future__ import annotations


# Le moteur expose le graphe complet, mais word/char ne sont pas la vue de
# compréhension documentaire principale. Ils sont des tokens auxiliaires
# d'audit, d'ancrage et d'alignement.
GRAPH_LEVELS = {
    "page", "region", "block", "line", "phrase", "span", "word", "char",
    "image", "drawing", "table", "cell", "formula", "code", "overlay",
}
AUXILIARY_TOKEN_LEVELS = {"word", "char"}


def build_graph(units: list[dict], regions: list[dict],
                page_structure: dict | None = None, *,
                page_node_id: str = "page") -> dict:
    """Construit {"nodes": [], "edges": []} au niveau documentaire complet.

    Nœuds : régions indexées + toutes les unités canoniques.
    Arêtes : containment hiérarchique, appartenance aux régions, flux de
    lecture par niveau, matérialisation région→unité et relations
    sémantiques amont.
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    known_ids: set[str] = set()
    legacy_to_unit: dict[str, str] = {}
    region_to_materialized_unit: dict[str, str] = {}

    for region in regions:
        region_id = region["region_id"]
        nodes.append({"id": region_id, "type": region["region_type"], "kind": "region_index"})
        known_ids.add(region_id)
        edges.append({
            "source": page_node_id,
            "target": region_id,
            "relation": "contains",
        })

    graph_units = [u for u in units if u["level"] in GRAPH_LEVELS]

    for unit in units:
        legacy_id = unit.get("legacy_id")
        if legacy_id is not None:
            legacy_to_unit[str(legacy_id)] = unit["unit_id"]
        parent_region_id = (unit.get("relations") or {}).get("parent_region_id")
        if unit.get("level") in {"region", "table", "cell", "formula", "code"} and parent_region_id:
            region_to_materialized_unit[str(parent_region_id)] = unit["unit_id"]

    for unit in graph_units:
        nodes.append({
            "id": unit["unit_id"],
            "type": unit["level"],
            "kind": "canonical_unit",
            "view": "auxiliary_token" if unit["level"] in AUXILIARY_TOKEN_LEVELS else "document",
            "role": (unit.get("understanding") or {}).get("role"),
            "object_type": (unit.get("understanding") or {}).get("object_type"),
        })
        known_ids.add(unit["unit_id"])

    if page_node_id not in known_ids:
        nodes.insert(0, {"id": page_node_id, "type": "page", "kind": "canonical_unit"})
        known_ids.add(page_node_id)

    for region_id, unit_id in region_to_materialized_unit.items():
        if region_id in known_ids and unit_id in known_ids:
            edges.append({
                "source": region_id,
                "target": unit_id,
                "relation": "materializes",
            })

    for unit in graph_units:
        if unit["unit_id"] == page_node_id:
            continue
        # Containment hiérarchique complet.
        parent_id = unit.get("parent_id")
        edges.append({
            "source": parent_id if parent_id in known_ids else page_node_id,
            "target": unit["unit_id"],
            "relation": "contains",
        })

        # Appartenance aux régions.
        for membership in (unit.get("understanding") or {}).get("region_memberships") or []:
            region_id = membership.get("region_id")
            if region_id in known_ids:
                edges.append({
                    "source": unit["unit_id"],
                    "target": region_id,
                    "relation": "belongs_to",
                    "overlap_ratio": membership.get("overlap_ratio"),
                })

    # Flux de lecture par niveau.
    for level in GRAPH_LEVELS - {"page"}:
        ordered = sorted(
            (u for u in graph_units
             if u["level"] == level
             and (u.get("geometry") or {}).get("reading_order_index") is not None),
            key=lambda u: u["geometry"]["reading_order_index"],
        )
        for prev, nxt in zip(ordered, ordered[1:]):
            edges.append({
                "source": prev["unit_id"],
                "target": nxt["unit_id"],
                "relation": "flows_to",
                "level": level,
            })

    # Semantic graph : reprendre les relations element_relations si présentes.
    payload = (page_structure or {}).get("element_relations") or {}
    flat_relations = payload.get("flat_relations") or payload.get("pair_relations") or []
    for relation in flat_relations:
        if not isinstance(relation, dict):
            continue
        source_id = relation.get("source_id") or relation.get("from_id")
        target_id = relation.get("target_id") or relation.get("to_id")
        source_unit = legacy_to_unit.get(str(source_id), source_id)
        target_unit = legacy_to_unit.get(str(target_id), target_id)
        if not source_unit or not target_unit:
            continue
        edges.append({
            "source": source_unit,
            "target": target_unit,
            "relation": relation.get("relation")
            or relation.get("logical_relation")
            or "continues",
            "logical_relation": relation.get("logical_relation"),
            "confidence": relation.get("confidence"),
            "source_module": "element_relations",
        })

    return {"nodes": nodes, "edges": edges}


def build_relations(units: list[dict], graph: dict,
                    page_structure: dict | None = None) -> dict:
    """Construit la couche relations (ordres de lecture + arêtes sémantiques)."""
    ordered = sorted(
        (u for u in units
         if (u.get("geometry") or {}).get("reading_order_index") is not None),
        key=lambda u: u["geometry"]["reading_order_index"],
    )
    reading_order = [u["unit_id"] for u in ordered if u["level"] == "block"]
    reading_order_by_level: dict[str, list[str]] = {}
    primary_reading_order_by_level: dict[str, list[str]] = {}
    for unit in ordered:
        reading_order_by_level.setdefault(unit["level"], []).append(unit["unit_id"])
        if unit["level"] not in AUXILIARY_TOKEN_LEVELS:
            primary_reading_order_by_level.setdefault(unit["level"], []).append(unit["unit_id"])
    semantic_edges = [
        e for e in graph.get("edges", [])
        if e.get("relation") not in {"contains", "belongs_to", "flows_to", "materializes"}
        or e.get("source_module") == "element_relations"
    ]
    return {
        "schema_version": "relations.v1",
        "reading_order": reading_order,
        "reading_order_by_level": reading_order_by_level,
        "primary_reading_order_by_level": primary_reading_order_by_level,
        "auxiliary_token_levels": sorted(AUXILIARY_TOKEN_LEVELS),
        "edges": semantic_edges,
    }
