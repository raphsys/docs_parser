"""PAGEPRINT Builder — construit INPUT_DATA, l'empreinte canonique d'une page.

PAGEPRINT produit l'empreinte canonique d'une page, sous forme d'INPUT_DATA,
afin que les modules de traduction, reconstruction, mise en forme et QA
travaillent sur une représentation unique, normalisée, auditable et riche.

Frontière nette :
    Entrée  : page source + ressources + extraction brute
    Sortie  : INPUT_DATA canonique

PAGEPRINT ne traduit pas, ne réécrit pas, n'optimise pas la mise en page
traduite, ne produit pas le PDF final, ne gère pas les endpoints HTTP,
ne fait pas d'export utilisateur.

Usage :
    input_data = PagePrintBuilder().build(
        source_context=source_context,
        extraction_result=extraction_result,
        page_structure=page_structure,
        assets=assets,
    )
"""

from __future__ import annotations

import uuid

from .constraint_compiler import compile_page_constraints
from .detection.builder import PageRegionDetectBuilder
from .evidence import collect_claims, resolve_all
from .functional_validator import validate_functional
from .graph_builder import build_graph, build_relations
from .normalizer import (
    DEFAULT_RENDER_DPI,
    normalize_page_geometry,
    scale_from_dimensions,
)
from .policy_compiler import compile_policies
from .provenance import build_provenance, build_replay
from .quality_assessor import assess
from .region_index import attach_region_memberships, build_regions
from .role_resolver import infer_page_role, resolve_roles
from .schema import PAGEPRINT_SCHEMA_VERSION, empty_input_data
from .semantic_builder import build_semantic_system
from .preservation_compiler import compile_preservation
from .structure_builders import build_logical_structures
from .unit_factory import build_region_units, build_units
from .validators import validate_input_data
from .view_compiler import compile_views


AUXILIARY_TOKEN_LEVELS = {"word", "char"}
PRIMARY_DOCUMENT_LEVELS = {
    "page", "region", "block", "line", "phrase", "span",
    "image", "drawing", "table", "cell", "formula", "code",
    "protected_visual", "overlay",
}


class PagePrintBuilder:
    """Assemble les couches de INPUT_DATA depuis la structure de page legacy.

    page_structure reste une structure de travail interne ; INPUT_DATA est
    le contrat canonique stabilisé pour tout le pipeline. Les vues legacy
    (compatibility) sont des exports temporaires dérivés, pas la source
    de vérité.
    """

    def __init__(self, *, default_dpi: float = DEFAULT_RENDER_DPI):
        self.default_dpi = default_dpi

    def build(self, *, page_structure: dict,
              source_context: dict | None = None,
              extraction_result: dict | None = None,
              assets: dict | None = None,
              page_image=None,
              pdf_page=None,
              page_index: int = 0,
              validate: bool = True) -> dict:
        page_structure = page_structure or {}
        source_context = source_context or {}
        extraction_result = extraction_result or {}
        assets = assets or {}

        input_data = empty_input_data()
        input_data["input_id"] = f"pageprint_{uuid.uuid4().hex[:12]}"

        # --- page (géométrie canonique en points) ---
        dimensions = page_structure.get("dimensions") or {}
        geometry = normalize_page_geometry(dimensions, default_dpi=self.default_dpi)
        sx, sy = scale_from_dimensions(dimensions, default_dpi=self.default_dpi)
        dpi = geometry["render_dpi"]

        page_structure, region_detect_result = PageRegionDetectBuilder().build(
            page_structure=page_structure,
            page_image=page_image,
            pdf_page=pdf_page,
            sx=sx,
            sy=sy,
            run_detector=False,
        )
        dimensions = page_structure.get("dimensions") or dimensions

        input_data["page"] = {
            "page_index": page_index,
            "page_number": page_index + 1,
            "page_role": page_structure.get("page_role") or "unknown",
            "geometry": geometry,
            "rotation": page_structure.get("rotation") or 0,
            "orientation": geometry.pop("orientation"),
            "format_probable": page_structure.get("format_probable") or "custom",
        }

        # --- document ---
        language = (source_context.get("language") or {})
        input_data["document"] = {
            "document_id": source_context.get("document_id"),
            "source_path": source_context.get("source_path"),
            "file_name": source_context.get("file_name"),
            "file_type": source_context.get("file_type"),
            "page_count": source_context.get("page_count"),
            "detected_document_type": page_structure.get("document_type")
            or "mixed_unknown",
            "language": {
                "source_lang": language.get("source_lang"),
                "target_lang": language.get("target_lang"),
                "detected_languages": language.get("detected_languages") or [],
            },
        }

        # --- assets / visual_layers ---
        input_data["assets"] = {
            "source_image_path": assets.get("source_image_path"),
            "source_image_url": assets.get("source_image_url"),
            "background_path": assets.get("background_path")
            or page_structure.get("background_path"),
            "mask_master_path": assets.get("mask_master_path"),
            "visual_debug_path": assets.get("visual_debug_path"),
            "background": assets.get("background") or {},
            "immutable_overlays": assets.get("immutable_overlays")
            or page_structure.get("immutable_overlays") or [],
        }
        input_data["visual_layers"] = {
            "background": {
                "path": input_data["assets"]["background_path"],
                "type": "cleaned_background",
                "text_removed": bool(input_data["assets"]["background_path"]),
            },
            "source_render": {
                "path": input_data["assets"]["source_image_path"],
                "dpi": dpi,
            },
            "masks": assets.get("masks") or {
                "text_mask": input_data["assets"]["mask_master_path"],
            },
            "overlays": [
                {
                    "overlay_id": overlay.get("id") or f"overlay_{i + 1:03d}",
                    "type": "preserve_visual",
                    "bbox": overlay.get("bbox"),
                    "source": overlay.get("source") or "immutable_overlay",
                }
                for i, overlay in enumerate(
                    input_data["assets"]["immutable_overlays"]
                )
                if isinstance(overlay, dict)
            ],
        }

        # --- extraction (sources brutes) ---
        input_data["extraction"] = {
            "pipeline": extraction_result.get("pipeline") or "legacy",
            "source_priority": extraction_result.get("source_priority")
            or ["native_pdf", "ocr", "layout_ai", "heuristic"],
            "native_pdf_available": bool(extraction_result.get("native_pdf_available")),
            "ocr_used": bool(extraction_result.get("ocr_used")),
            "ocr_engine": extraction_result.get("ocr_engine"),
            "target_dpi": dpi,
            "raw_sources": extraction_result.get("raw_sources") or {},
            "raw_source_details": extraction_result.get("raw_source_details") or {},
            "region_detect_result": region_detect_result,
        }
        input_data["debug"]["page_region_detect"] = region_detect_result

        # --- units ---
        factory_stats: dict = {}
        units = build_units(
            page_structure, page_index=page_index, sx=sx, sy=sy, dpi=dpi,
            language=input_data["document"]["language"]["source_lang"],
            stats=factory_stats,
        )
        input_data["units"] = units

        # --- regions + memberships ---
        regions = build_regions(
            page_structure, page_index=page_index, sx=sx, sy=sy,
            stats=factory_stats,
        )
        attach_region_memberships(units, regions)
        page_unit_id = f"p{page_index + 1:03d}_page"
        region_units = build_region_units(
            regions,
            page_unit_id=page_unit_id,
            page_index=page_index,
            sx=sx,
            sy=sy,
            dpi=dpi,
            language=input_data["document"]["language"]["source_lang"],
            start_index=len(units),
            page_context={
                "page_family": page_structure.get("page_family"),
                "layout_type": page_structure.get("layout_type"),
                "document_type": page_structure.get("document_type"),
                "page_role": page_structure.get("page_role"),
            },
        )
        if region_units:
            page_unit = next((u for u in units if u["unit_id"] == page_unit_id), None)
            if page_unit is not None:
                page_unit["children_ids"].extend(u["unit_id"] for u in region_units)
            units.extend(region_units)
            factory_stats["region_units_created"] = len(region_units)
        else:
            factory_stats["region_units_created"] = 0
        input_data["regions"] = regions
        input_data["extraction"]["normalization"] = factory_stats

        # --- graph + relations ---
        graph = build_graph(units, regions, page_structure,
                            page_node_id=f"p{page_index + 1:03d}_page")
        input_data["graph"] = graph
        input_data["relations"] = build_relations(units, graph, page_structure)

        # --- page_intelligence ---
        input_data["page_intelligence"] = self._build_page_intelligence(page_structure)

        # --- evidence claims + role resolution ---
        input_data["evidence"] = collect_claims(
            units,
            regions,
            page_intelligence=input_data["page_intelligence"],
        )
        evidence_decisions = resolve_all(units)
        input_data["role_resolution"] = resolve_roles(
            units,
            page_intelligence=input_data["page_intelligence"],
            document_context=source_context.get("document_context") or {},
        )
        input_data["logical_structures"] = build_logical_structures(
            units,
            page_intelligence=input_data["page_intelligence"],
        )

        # --- page_role promotion (index/toc/table_page) from dominant content ---
        promoted_role = infer_page_role(
            (input_data.get("role_resolution") or {}).get("role_counts"),
            input_data["logical_structures"],
            current=input_data["page_intelligence"].get("page_role"),
        )
        if promoted_role and promoted_role != input_data["page_intelligence"].get("page_role"):
            input_data["page_intelligence"]["page_role"] = promoted_role
            input_data["page_intelligence"]["page_role_source"] = "inferred_from_content"
            if isinstance(input_data.get("page"), dict):
                input_data["page"]["page_role"] = promoted_role

        # --- style_system / semantic_system ---
        input_data["style_system"] = {
            "schema_version": "style_system.v1",
            "global_styles": page_structure.get("global_styles") or {},
            "page_style_profile": page_structure.get("visual_style_profile")
            or page_structure.get("style_profile") or {},
            "dominant_body_style_id": page_structure.get("dominant_body_style_id"),
            "heading_hierarchy": page_structure.get("heading_hierarchy") or [],
        }
        legacy_semantic_system = self._build_semantic_system(page_structure, units)

        # --- preservation + policies ---
        input_data["preservation"] = compile_preservation(
            units,
            page_intelligence=input_data["page_intelligence"],
        )
        input_data["policies"] = compile_policies(
            units, input_data["page_intelligence"]
        )
        input_data["semantic_system"] = build_semantic_system(
            page_structure,
            units,
            logical_structures=input_data["logical_structures"],
            legacy_semantic_system=legacy_semantic_system,
        )
        input_data["indexes"] = self._build_indexes(units)

        # --- constraints / reconstruction ---
        reconstruction_constraints = compile_page_constraints(
            units, input_data["page_intelligence"]
        )
        input_data["reconstruction_constraints"] = reconstruction_constraints
        # Les contraintes par unité vivent sur units[].constraints (source de
        # vérité) — pas de duplication ici.
        input_data["constraints"] = {
            "schema_version": "constraints.v1",
            "page": reconstruction_constraints["page"],
            "layout": reconstruction_constraints["layout"],
            "unit_constraints_location": "units[].constraints",
        }
        input_data["reconstruction_context"] = {
            "coordinate_unit": "pt",
            "render_dpi": dpi,
            "background_available": bool(input_data["assets"]["background_path"]),
        }

        # --- translation_context ---
        input_data["translation_context"] = self._build_translation_context(
            page_structure, source_context, units
        )

        # --- quality / risks ---
        quality, risks = assess(
            units, regions, page_structure,
            page_intelligence=input_data["page_intelligence"],
        )
        input_data["quality"] = quality
        input_data["risks"] = risks

        # --- compréhension documentaire synthétique ---
        input_data["document_comprehension"] = self._build_document_comprehension(
            units=units,
            regions=regions,
            graph=graph,
            page_structure=page_structure,
        )

        # --- views (multi-vues) ---
        input_data["views"] = self._build_views(units)
        input_data["views"].update(compile_views(
            units,
            semantic_system=input_data["semantic_system"],
            logical_structures=input_data["logical_structures"],
            page_intelligence=input_data["page_intelligence"],
        ))
        input_data["views"]["detected_regions"] = self._build_detected_regions_view(regions)
        input_data["views"]["region_memberships"] = self._build_region_memberships_view(units)
        input_data.setdefault("quality", {}).setdefault("metrics", {}).update(
            input_data["views"].get("metrics") or {}
        )

        # --- provenance + replay ---
        input_data["provenance"] = build_provenance(
            page_structure=page_structure,
            units=units,
            extra_traces=evidence_decisions,
            replay=build_replay(
                source_path=source_context.get("source_path"),
                render_path=input_data["assets"]["source_image_path"],
                config={"default_dpi": self.default_dpi},
                environment_flags=source_context.get("environment_flags") or {},
            ),
        )

        # --- compatibility (vues legacy dérivées, pas la source de vérité) ---
        input_data["compatibility"] = {
            "legacy_page_structure": page_structure,
            "reconstructor_payload_v1": None,
            "translator_payload_v1": None,
        }

        # --- validation du contrat ---
        if validate:
            schema_validation = validate_input_data(input_data)
            functional_validation = validate_functional(input_data)
            input_data["debug"]["validation"] = schema_validation
            input_data["debug"]["functional_validation"] = functional_validation
            input_data["debug"]["audit_status"] = {
                "schema_status": "ok" if schema_validation.get("valid") else "ko",
                "functional_status": functional_validation.get("functional_status"),
                "blocking_reasons": list(functional_validation.get("errors") or []),
                "pageprint_translation_plan_count": len(input_data["views"].get("translation_plan") or []),
                "semantic_segment_count": len(input_data["semantic_system"].get("translation_segments") or []),
            }

        return input_data

    # ------------------------------------------------------------------

    @staticmethod
    def _build_page_intelligence(page_structure: dict) -> dict:
        page_case_v2 = page_structure.get("page_case_v2") or {}
        existing = page_structure.get("page_intelligence") or {}
        intelligence = {
            "schema_version": "page_intelligence.v1",
            "coordinate_unit": "pt",
            "page_role": page_structure.get("page_role"),
            "page_family": page_structure.get("page_family"),
            "page_family_group": page_structure.get("page_family_group"),
            "document_type": page_structure.get("document_type"),
            "layout_type": page_structure.get("layout_type"),
            "page_geometry": page_structure.get("dimensions") or {},
            "style_profile": page_structure.get("style_profile"),
            "page_case": page_structure.get("page_case") or {},
            "page_case_v2": page_case_v2,
            "reading_modes": page_case_v2.get("reading_modes")
            or existing.get("reading_modes") or {},
            "layout_tendencies": page_case_v2.get("layout_tendencies")
            or existing.get("layout_tendencies") or {},
            "translation_sensitivity": page_case_v2.get("translation_sensitivity_signals")
            or existing.get("translation_sensitivity") or {},
            "risk_flags": page_case_v2.get("risk_flags")
            or existing.get("risk_flags") or [],
            "extraction_guidance": existing.get("extraction_guidance") or {},
            "decision_context": existing.get("decision_context") or {},
        }
        return intelligence

    @staticmethod
    def _build_translation_context(page_structure: dict, source_context: dict,
                                   units: list[dict]) -> dict:
        language = source_context.get("language") or {}
        document_units = [
            u for u in units
            if u.get("level") not in AUXILIARY_TOKEN_LEVELS
        ]
        non_translatable = [
            u["unit_id"] for u in document_units
            if (u.get("policy") or {}).get("translatable") is False
        ]
        layout_constrained = [
            u["unit_id"] for u in document_units
            if (u.get("policy") or {}).get("translation_strategy") == "layout_constrained"
            and (u.get("policy") or {}).get("translatable")
        ]
        paragraph_flow = [
            u["unit_id"] for u in document_units
            if (u.get("policy") or {}).get("translation_strategy") == "paragraph_flow"
        ]
        return {
            "source_lang": language.get("source_lang"),
            "target_lang": language.get("target_lang"),
            "document_domain": source_context.get("document_domain"),
            "document_subdomain": source_context.get("document_subdomain"),
            "translation_style": page_structure.get("translation_style"),
            "translation_tone": page_structure.get("translation_tone"),
            "terminology": source_context.get("terminology") or {
                "domain": source_context.get("document_domain"),
                "subdomain": source_context.get("document_subdomain"),
                "locked_terms": [],
                "preferred_terms": [],
                "reserved_terms": [],
            },
            "protected_tokens": [],
            "non_translatable_units": non_translatable,
            "layout_constrained_units": layout_constrained,
            "paragraph_flow_units": paragraph_flow,
            "unit_scope": {
                "primary_levels": sorted(PRIMARY_DOCUMENT_LEVELS),
                "auxiliary_token_levels": sorted(AUXILIARY_TOKEN_LEVELS),
                "fine_tokens_policy": "available_for_alignment_audit_not_translation_units",
            },
        }

    @staticmethod
    def _build_document_comprehension(*, units: list[dict], regions: list[dict],
                                      graph: dict, page_structure: dict) -> dict:
        """Synthèse consommable de ce que PAGEPRINT a compris de la page."""
        unit_level_by_id = {
            unit.get("unit_id"): unit.get("level")
            for unit in units
            if isinstance(unit, dict)
        }
        document_units = [
            unit for unit in units
            if unit.get("level") in PRIMARY_DOCUMENT_LEVELS
        ]
        fine_units = [
            unit for unit in units
            if unit.get("level") in AUXILIARY_TOKEN_LEVELS
        ]

        def count_by(unit_list: list[dict], getter) -> dict[str, int]:
            counts: dict[str, int] = {}
            for item in unit_list:
                key = getter(item) or "unknown"
                counts[key] = counts.get(key, 0) + 1
            return counts

        by_level: dict[str, int] = {}
        by_role: dict[str, int] = {}
        by_object_type: dict[str, int] = {}
        by_semantic_kind: dict[str, int] = {}
        by_source: dict[str, int] = {}
        policy_counts: dict[str, int] = {}
        missing: dict[str, list[str]] = {
            "bbox": [],
            "text_for_textual_unit": [],
            "role": [],
            "object_type": [],
            "region_membership": [],
        }
        object_index: list[dict] = []

        textual_levels = {"block", "line", "phrase", "span"}
        for unit in document_units:
            level = unit.get("level")
            by_level[level] = by_level.get(level, 0) + 1
            understanding = unit.get("understanding") or {}
            extraction = unit.get("extraction") or {}
            policy = unit.get("policy") or {}
            role = understanding.get("role") or "unknown"
            object_type = understanding.get("object_type") or policy.get("unit_type") or "unknown"
            semantic_kind = understanding.get("semantic_kind") or "unknown"
            source = extraction.get("source") or extraction.get("source_kind") or "unknown"
            strategy = policy.get("translation_strategy") or "unknown"
            by_role[role] = by_role.get(role, 0) + 1
            by_object_type[object_type] = by_object_type.get(object_type, 0) + 1
            by_semantic_kind[semantic_kind] = by_semantic_kind.get(semantic_kind, 0) + 1
            by_source[source] = by_source.get(source, 0) + 1
            policy_counts[strategy] = policy_counts.get(strategy, 0) + 1

            if not (unit.get("geometry") or {}).get("bbox"):
                missing["bbox"].append(unit["unit_id"])
            if level in textual_levels and not (unit.get("content") or {}).get("text"):
                missing["text_for_textual_unit"].append(unit["unit_id"])
            if role == "unknown":
                missing["role"].append(unit["unit_id"])
            if object_type == "unknown":
                missing["object_type"].append(unit["unit_id"])
            if level in {"block", "line", "phrase", "span"} and not understanding.get("region_memberships"):
                missing["region_membership"].append(unit["unit_id"])

            if level in {"page", "region", "block", "table", "cell", "formula", "code", "image", "drawing"}:
                object_index.append({
                    "unit_id": unit["unit_id"],
                    "level": level,
                    "role": role,
                    "object_type": object_type,
                    "object_class": understanding.get("object_class"),
                    "bbox": (unit.get("geometry") or {}).get("bbox"),
                    "source": source,
                    "translatable": policy.get("translatable"),
                    "render_policy": policy.get("render_policy"),
                    "confidence": (unit.get("confidence") or {}).get("overall"),
                })

        relation_counts: dict[str, int] = {}
        primary_relation_counts: dict[str, int] = {}
        primary_edges: list[dict] = []
        for edge in graph.get("edges") or []:
            relation = edge.get("relation") or "unknown"
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
            source_level = unit_level_by_id.get(edge.get("source"))
            target_level = unit_level_by_id.get(edge.get("target"))
            if source_level not in AUXILIARY_TOKEN_LEVELS and target_level not in AUXILIARY_TOKEN_LEVELS:
                primary_relation_counts[relation] = primary_relation_counts.get(relation, 0) + 1
                primary_edges.append(edge)

        region_counts: dict[str, int] = {}
        for region in regions:
            region_type = region.get("region_type") or "unknown"
            region_counts[region_type] = region_counts.get(region_type, 0) + 1

        auxiliary_counts = {
            "by_level": count_by(fine_units, lambda u: u.get("level")),
            "by_semantic_kind": count_by(
                fine_units,
                lambda u: (u.get("understanding") or {}).get("semantic_kind"),
            ),
            "total": len(fine_units),
        }

        return {
            "schema_version": "document_comprehension.v1",
            "understanding_focus": "document_structure",
            "unit_scope": {
                "primary_document_levels": sorted(PRIMARY_DOCUMENT_LEVELS),
                "auxiliary_token_levels": sorted(AUXILIARY_TOKEN_LEVELS),
                "rule": "word_char_are_auxiliary_alignment_and_audit_tokens",
            },
            "coverage": {
                "levels_present": sorted(k for k, v in by_level.items() if v),
                "has_page_unit": by_level.get("page", 0) > 0,
                "has_auxiliary_word_units": auxiliary_counts["by_level"].get("word", 0) > 0,
                "has_auxiliary_char_units": auxiliary_counts["by_level"].get("char", 0) > 0,
                "has_word_units": auxiliary_counts["by_level"].get("word", 0) > 0,
                "has_char_units": auxiliary_counts["by_level"].get("char", 0) > 0,
                "has_region_units": any(by_level.get(level, 0) for level in ("region", "table", "cell", "formula", "code")),
                "has_relations": bool(graph.get("edges")),
                "has_primary_relations": bool(primary_edges),
                "has_regions": bool(regions),
            },
            "counts": {
                "by_level": by_level,
                "by_all_level": count_by(units, lambda u: u.get("level")),
                "by_auxiliary_level": auxiliary_counts["by_level"],
                "by_role": by_role,
                "by_object_type": by_object_type,
                "by_semantic_kind": by_semantic_kind,
                "by_source": by_source,
                "by_region_type": region_counts,
                "by_relation": relation_counts,
                "by_primary_relation": primary_relation_counts,
                "by_translation_strategy": policy_counts,
            },
            "auxiliary_tokens": auxiliary_counts,
            "object_index": object_index,
            "region_index": [
                {
                    "region_id": region.get("region_id"),
                    "region_type": region.get("region_type"),
                    "bbox": region.get("bbox"),
                    "member_counts": {
                        key: len(value or [])
                        for key, value in (region.get("members") or {}).items()
                    },
                    "policy": region.get("policy"),
                    "confidence": region.get("confidence"),
                }
                for region in regions
            ],
            "relation_index": graph.get("edges") or [],
            "legacy_understanding_sources": {
                "has_layout_v2": bool(page_structure.get("schema_version") == "layout.v2" or page_structure.get("layout")),
                "has_page_case_v2": bool(page_structure.get("page_case_v2")),
                "has_element_relations": bool(page_structure.get("element_relations")),
                "has_special_regions": bool(page_structure.get("special_regions")),
                "has_semantic_groups": any(
                    bool(block.get("semantic_groups"))
                    for block in (page_structure.get("blocks") or [])
                    if isinstance(block, dict)
                ),
            },
            "primary_relation_index": primary_edges,
            "missing_or_weak_fields": {
                key: value[:200]
                for key, value in missing.items()
                if value
            },
        }

    @staticmethod
    def _build_views(units: list[dict]) -> dict:
        """La même page vue de plusieurs manières — évite que chaque module
        aval reconstruise sa propre vue.

        Les vues sont des index, pas des copies : le détail complet
        (render_contract, evidence, confidence…) vit sur units[].
        Les niveaux word/char restent dans units[] pour audit/alignement,
        mais ils sont exposés dans fine_tokens et exclus de la vue documentaire
        principale.
        debug_units ne liste que les unités notables (confiance faible,
        politique imposée par région, fusion, géométrie dégénérée).
        """
        hierarchical: dict = {}
        children_index = {u["unit_id"]: u for u in units}
        document_units = [
            u for u in units
            if u.get("level") not in AUXILIARY_TOKEN_LEVELS
        ]
        fine_token_units = [
            u for u in units
            if u.get("level") in AUXILIARY_TOKEN_LEVELS
        ]
        document_unit_ids = {u["unit_id"] for u in document_units}

        def node_view(unit: dict, *, include_auxiliary: bool = False) -> dict:
            return {
                "unit_id": unit["unit_id"],
                "level": unit["level"],
                "children": [
                    node_view(children_index[cid], include_auxiliary=include_auxiliary)
                    for cid in unit.get("children_ids") or []
                    if cid in children_index
                    and (include_auxiliary or cid in document_unit_ids)
                ],
            }

        hierarchical["roots"] = [
            node_view(u) for u in document_units if u.get("parent_id") is None
        ]

        def _is_protected_visual_unit(unit: dict) -> bool:
            policy = unit.get("policy") or {}
            constraints = unit.get("constraints") or {}
            understanding = unit.get("understanding") or {}
            return bool(
                policy.get("protected_visual")
                or constraints.get("skip_text_reconstruction")
                or understanding.get("protected_visual")
                or policy.get("render_policy") == "background_only"
                or policy.get("translation_strategy") == "background_only"
                or policy.get("unit_type") in {
                    "protected_visual",
                    "formula",
                    "equation",
                    "code_visible",
                    "symbolic_expression",
                    "chemical_formula",
                }
            )

        def _is_translation_view_unit(unit: dict) -> bool:
            if unit.get("level") not in {"block", "line", "phrase"}:
                return False
            policy = unit.get("policy") or {}
            if policy.get("translatable") is not True:
                return False
            if policy.get("render_policy") == "background_only":
                return False
            if policy.get("translation_strategy") in {"background_only", "exact_preserve", "keep_original"}:
                return False
            if _is_protected_visual_unit(unit):
                return False
            return bool((unit.get("content") or {}).get("text"))

        def _translation_candidate_reason(unit: dict) -> str:
            policy = unit.get("policy") or {}
            constraints = unit.get("constraints") or {}
            if unit.get("level") not in {"block", "line", "phrase"}:
                return "level_not_primary_translation_candidate"
            if not (unit.get("content") or {}).get("text"):
                return "empty_text"
            if _is_protected_visual_unit(unit):
                return "protected_visual"
            if policy.get("translatable") is not True:
                return "policy_non_translatable"
            if constraints.get("skip_translation"):
                return "constraints_skip_translation"
            if policy.get("render_policy") == "background_only":
                return "background_only_render_policy"
            if policy.get("translation_strategy") in {"background_only", "exact_preserve", "keep_original"}:
                return "preserve_translation_strategy"
            return "included"

        translation_units = [
            {
                "unit_id": u["unit_id"],
                "level": u["level"],
                "text": (u.get("content") or {}).get("text"),
                "translation_strategy": (u.get("policy") or {}).get("translation_strategy"),
                "coverage_required": (u.get("policy") or {}).get("coverage_required"),
            }
            for u in document_units if _is_translation_view_unit(u)
        ]
        translation_candidates_debug = [
            {
                "unit_id": u["unit_id"],
                "level": u["level"],
                "text": (u.get("content") or {}).get("text"),
                "candidate": _is_translation_view_unit(u),
                "reason": _translation_candidate_reason(u),
                "translation_strategy": (u.get("policy") or {}).get("translation_strategy"),
                "render_policy": (u.get("policy") or {}).get("render_policy"),
                "translatable": (u.get("policy") or {}).get("translatable"),
            }
            for u in document_units
            if u.get("level") in {"block", "line", "phrase", "span", "formula", "code", "protected_visual"}
        ]
        render_units = [
            {
                "unit_id": u["unit_id"],
                "level": u["level"],
                "bbox": (u.get("geometry") or {}).get("bbox"),
                "render_mode": (u.get("render_contract") or {}).get("mode"),
            }
            for u in document_units
            if u["level"] in {"region", "block", "line", "phrase", "image", "drawing", "table", "cell", "formula", "code", "overlay"}
        ]

        fine_tokens = [
            {
                "unit_id": u["unit_id"],
                "level": u["level"],
                "parent_id": u.get("parent_id"),
                "text": (u.get("content") or {}).get("text"),
                "bbox": (u.get("geometry") or {}).get("bbox"),
                "semantic_kind": (u.get("understanding") or {}).get("semantic_kind"),
            }
            for u in fine_token_units
        ]
        protected_visual_units = [
            {
                "unit_id": u["unit_id"],
                "level": u["level"],
                "type": "protected_visual",
                "bbox": (u.get("geometry") or {}).get("bbox"),
                "source_text_for_audit": (u.get("content") or {}).get("text"),
                "unit_type": (u.get("policy") or {}).get("unit_type"),
                "render_policy": (u.get("policy") or {}).get("render_policy"),
                "translation_strategy": (u.get("policy") or {}).get("translation_strategy"),
                "preserve_original_pixels": bool((u.get("policy") or {}).get("preserve_original_pixels")),
                "skip_translation": bool((u.get("policy") or {}).get("skip_translation")),
                "skip_text_reconstruction": bool((u.get("policy") or {}).get("skip_text_reconstruction")),
                "covered_by_protected_region_id": (u.get("relations") or {}).get("parent_region_id"),
            }
            for u in units
            if _is_protected_visual_unit(u)
        ]

        def is_noteworthy(u: dict) -> str | None:
            confidence = (u.get("confidence") or {}).get("overall")
            if confidence is not None and confidence < 0.7:
                return "low_confidence"
            if str((u.get("policy") or {}).get("policy_source") or "").startswith("region:"):
                return "region_policy_override"
            if (u.get("lifecycle") or {}).get("merged_from"):
                return "merged_unit"
            if (u.get("geometry") or {}).get("degenerate"):
                return "degenerate_geometry"
            return None

        debug_units = []
        for u in document_units:
            reason = is_noteworthy(u)
            if reason:
                debug_units.append({
                    "unit_id": u["unit_id"],
                    "reason": reason,
                    "evidence": u.get("evidence"),
                    "confidence": u.get("confidence"),
                    "lifecycle": u.get("lifecycle"),
                })

        return {
            "hierarchical": hierarchical,
            "full_hierarchical": {
                "roots": [
                    node_view(u, include_auxiliary=True)
                    for u in units if u.get("parent_id") is None
                ],
            },
            "flat_units": [u["unit_id"] for u in units],
            "document_units": [u["unit_id"] for u in document_units],
            "auxiliary_units": [u["unit_id"] for u in fine_token_units],
            "translation_units": translation_units,
            "translation_candidates_debug": translation_candidates_debug,
            "render_units": render_units,
            "protected_visual_units": protected_visual_units,
            "background_preserved_regions": protected_visual_units,
            "fine_tokens": fine_tokens,
            "fine_tokens_note": "word/char sont auxiliaires: audit, ancrage, alignement; pas unités de compréhension documentaire principale",
            "debug_units": debug_units,
            "debug_units_note": "unités notables uniquement ; détail complet sur units[]",
        }

    @staticmethod
    def _build_semantic_system(page_structure: dict, units: list[dict]) -> dict:
        unit_by_id = {
            unit.get("unit_id"): unit
            for unit in units
            if isinstance(unit, dict) and unit.get("unit_id")
        }
        legacy_to_unit_ids: dict[str, list[str]] = {}
        for unit in units:
            legacy_id = unit.get("legacy_id")
            if legacy_id is not None:
                legacy_to_unit_ids.setdefault(str(legacy_id), []).append(unit["unit_id"])

        block_legacy_to_unit_id = {
            str(unit.get("legacy_id")): unit["unit_id"]
            for unit in units
            if unit.get("level") == "block" and unit.get("legacy_id") is not None
        }

        def canonicalize_entry(entry: dict, block: dict, *, semantic_level: str, index: int) -> dict:
            out = dict(entry)
            out.setdefault("unit_id", f"{semantic_level}_{index + 1:04d}")
            source_ids = (
                out.get("source_unit_ids")
                or out.get("unit_ids")
                or out.get("phrase_unit_ids")
                or []
            )
            canonical_source_ids = []
            unresolved_source_ids = []
            for source_id in source_ids:
                source_key = str(source_id)
                if source_key in unit_by_id:
                    canonical_source_ids.append(source_key)
                elif source_key in legacy_to_unit_ids:
                    canonical_source_ids.extend(legacy_to_unit_ids[source_key])
                else:
                    unresolved_source_ids.append(source_key)
            if canonical_source_ids:
                out["source_unit_ids"] = list(dict.fromkeys(canonical_source_ids))
            elif source_ids:
                out["source_unit_ids"] = list(source_ids)
                out["source_unit_resolution"] = {
                    "status": "unresolved",
                    "unresolved_source_ids": unresolved_source_ids,
                }

            structural_context = dict(out.get("structural_context") or {})
            block_id = (
                structural_context.get("block_unit_id")
                or out.get("block_unit_id")
                or block_legacy_to_unit_id.get(str(block.get("id")))
            )
            if block_id:
                structural_context["block_unit_id"] = block_id
            out["structural_context"] = structural_context
            out.setdefault("semantic_level", semantic_level)
            return out

        semantic_phrases = []
        semantic_groups = []
        phrase_index = 0
        group_index = 0
        for block in page_structure.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            for phrase in block.get("semantic_phrases") or []:
                if isinstance(phrase, dict):
                    semantic_phrases.append(canonicalize_entry(
                        phrase,
                        block,
                        semantic_level="semantic_phrase",
                        index=phrase_index,
                    ))
                    phrase_index += 1
            for group in block.get("semantic_groups") or []:
                if isinstance(group, dict):
                    semantic_groups.append(canonicalize_entry(
                        group,
                        block,
                        semantic_level="semantic_group",
                        index=group_index,
                    ))
                    group_index += 1

        return {
            "semantic_phrases": semantic_phrases,
            "semantic_groups": semantic_groups,
        }

    @staticmethod
    def _build_indexes(units: list[dict]) -> dict:
        legacy_id_to_unit_ids: dict[str, list[str]] = {}
        unit_id_to_legacy_id: dict[str, str] = {}
        for unit in units:
            unit_id = unit.get("unit_id")
            legacy_id = unit.get("legacy_id")
            if unit_id and legacy_id is not None:
                legacy_key = str(legacy_id)
                legacy_id_to_unit_ids.setdefault(legacy_key, []).append(unit_id)
                unit_id_to_legacy_id[unit_id] = legacy_key
        return {
            "legacy_id_to_unit_ids": legacy_id_to_unit_ids,
            "unit_id_to_legacy_id": unit_id_to_legacy_id,
        }

    @staticmethod
    def _build_detected_regions_view(regions: list[dict]) -> list[dict]:
        return [
            {
                "region_id": region.get("region_id"),
                "region_type": region.get("region_type"),
                "object_type": region.get("object_type"),
                "object_class": region.get("object_class"),
                "bbox": region.get("bbox"),
                "source": region.get("source"),
                "detection_source": region.get("detection_source") or region.get("source"),
                "confidence": region.get("confidence"),
                "protected_visual": bool(region.get("protected_visual") or (region.get("policy") or {}).get("protected_visual")),
                "translatable": (region.get("policy") or {}).get("translatable"),
                "render_policy": (region.get("policy") or {}).get("render_policy"),
                "translation_strategy": (region.get("policy") or {}).get("translation_strategy"),
                "member_counts": {
                    key: len(value or [])
                    for key, value in (region.get("members") or {}).items()
                },
            }
            for region in regions
        ]

    @staticmethod
    def _build_region_memberships_view(units: list[dict]) -> list[dict]:
        output = []
        for unit in units:
            for membership in (unit.get("understanding") or {}).get("region_memberships") or []:
                output.append({
                    "unit_id": unit.get("unit_id"),
                    "level": unit.get("level"),
                    "region_id": membership.get("region_id"),
                    "region_type": membership.get("region_type"),
                    "overlap_ratio": membership.get("overlap_ratio"),
                    "membership_role": membership.get("membership_role"),
                })
        return output


def build_pageprint_input_data(*, page_structure: dict,
                               source_context: dict | None = None,
                               extraction_result: dict | None = None,
                               assets: dict | None = None,
                               page_image=None,
                               pdf_page=None,
                               page_index: int = 0,
                               validate: bool = True) -> dict:
    """Fonction principale : construit l'INPUT_DATA canonique d'une page."""
    return PagePrintBuilder().build(
        page_structure=page_structure,
        source_context=source_context,
        extraction_result=extraction_result,
        assets=assets,
        page_image=page_image,
        pdf_page=pdf_page,
        page_index=page_index,
        validate=validate,
    )


__all__ = [
    "PagePrintBuilder",
    "build_pageprint_input_data",
    "PAGEPRINT_SCHEMA_VERSION",
]
