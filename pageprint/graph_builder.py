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


# Niveaux structurels représentés dans le graphe. Les niveaux fins
# (line/phrase/span) n'y figurent pas : leur containment est déjà porté
# par parent_id/children_ids, leur appartenance aux régions par
# region_memberships et leur flux par previous/next_unit_id — le graphe
# ne duplique pas ce qui est dérivable des unités.
GRAPH_LEVELS = {"region", "block", "table", "image", "drawing", "overlay"}


def build_graph(units: list[dict], regions: list[dict],
                page_structure: dict | None = None, *,
                page_node_id: str = "page") -> dict:
    """Construit {"nodes": [], "edges": []} au niveau structurel.

    Nœuds : page, régions et unités de niveau GRAPH_LEVELS.
    Arêtes : containment/appartenance/flux au niveau bloc + toutes les
    arêtes sémantiques (element_relations), qui elles ne sont pas
    dérivables des unités.
    """
    nodes = [{"id": page_node_id, "type": "page"}]
    edges: list[dict] = []
    known_ids = {page_node_id}
    legacy_to_unit: dict[str, str] = {}

    for region in regions:
        nodes.append({"id": region["region_id"], "type": region["region_type"]})
        known_ids.add(region["region_id"])
        edges.append({
            "source": page_node_id,
            "target": region["region_id"],
            "relation": "contains",
        })

    structural_units = [u for u in units if u["level"] in GRAPH_LEVELS]

    for unit in units:
        legacy_id = unit.get("legacy_id")
        if legacy_id is not None:
            legacy_to_unit[str(legacy_id)] = unit["unit_id"]

    for unit in structural_units:
        nodes.append({"id": unit["unit_id"], "type": unit["level"]})
        known_ids.add(unit["unit_id"])

    for unit in structural_units:
        # Containment au niveau structurel.
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

    # Flux de lecture bloc à bloc.
    ordered_blocks = sorted(
        (u for u in structural_units
         if u["level"] == "block"
         and (u.get("geometry") or {}).get("reading_order_index") is not None),
        key=lambda u: u["geometry"]["reading_order_index"],
    )
    for prev, nxt in zip(ordered_blocks, ordered_blocks[1:]):
        edges.append({
            "source": prev["unit_id"],
            "target": nxt["unit_id"],
            "relation": "flows_to",
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
    """Construit la couche relations (ordre de lecture + arêtes sémantiques)."""
    ordered = sorted(
        (u for u in units
         if (u.get("geometry") or {}).get("reading_order_index") is not None),
        key=lambda u: u["geometry"]["reading_order_index"],
    )
    reading_order = [u["unit_id"] for u in ordered if u["level"] == "block"]
    semantic_edges = [
        e for e in graph.get("edges", [])
        if e.get("relation") not in {"contains", "belongs_to", "flows_to"}
        or e.get("source_module") == "element_relations"
    ]
    return {
        "schema_version": "relations.v1",
        "reading_order": reading_order,
        "edges": semantic_edges,
    }
