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
from .evidence_resolver import resolve_all
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
from .schema import PAGEPRINT_SCHEMA_VERSION, empty_input_data
from .unit_factory import build_units
from .validators import validate_input_data


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
        }

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
        input_data["regions"] = regions
        input_data["extraction"]["normalization"] = factory_stats

        # --- evidence ---
        evidence_decisions = resolve_all(units)

        # --- graph + relations ---
        graph = build_graph(units, regions, page_structure,
                            page_node_id=f"p{page_index + 1:03d}_page")
        input_data["graph"] = graph
        input_data["relations"] = build_relations(units, graph, page_structure)

        # --- page_intelligence ---
        input_data["page_intelligence"] = self._build_page_intelligence(page_structure)

        # --- style_system / semantic_system ---
        input_data["style_system"] = {
            "schema_version": "style_system.v1",
            "global_styles": page_structure.get("global_styles") or {},
            "page_style_profile": page_structure.get("visual_style_profile")
            or page_structure.get("style_profile") or {},
            "dominant_body_style_id": page_structure.get("dominant_body_style_id"),
            "heading_hierarchy": page_structure.get("heading_hierarchy") or [],
        }
        input_data["semantic_system"] = {
            "semantic_groups": [
                group for block in page_structure.get("blocks") or []
                if isinstance(block, dict)
                for group in block.get("semantic_groups") or []
            ],
        }

        # --- policies ---
        input_data["policies"] = compile_policies(
            units, input_data["page_intelligence"]
        )

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

        # --- views (multi-vues) ---
        input_data["views"] = self._build_views(units)

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
            input_data["debug"]["validation"] = validate_input_data(input_data)

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
        non_translatable = [
            u["unit_id"] for u in units
            if (u.get("policy") or {}).get("translatable") is False
        ]
        layout_constrained = [
            u["unit_id"] for u in units
            if (u.get("policy") or {}).get("translation_strategy") == "layout_constrained"
            and (u.get("policy") or {}).get("translatable")
        ]
        paragraph_flow = [
            u["unit_id"] for u in units
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
        }

    @staticmethod
    def _build_views(units: list[dict]) -> dict:
        """La même page vue de plusieurs manières — évite que chaque module
        aval reconstruise sa propre vue.

        Les vues sont des index, pas des copies : le détail complet
        (render_contract, evidence, confidence…) vit sur units[].
        debug_units ne liste que les unités notables (confiance faible,
        politique imposée par région, fusion, géométrie dégénérée).
        """
        hierarchical: dict = {}
        children_index = {u["unit_id"]: u for u in units}

        def node_view(unit: dict) -> dict:
            return {
                "unit_id": unit["unit_id"],
                "level": unit["level"],
                "children": [
                    node_view(children_index[cid])
                    for cid in unit.get("children_ids") or []
                    if cid in children_index
                ],
            }

        hierarchical["roots"] = [
            node_view(u) for u in units if u.get("parent_id") is None
        ]

        translation_units = [
            {
                "unit_id": u["unit_id"],
                "level": u["level"],
                "text": (u.get("content") or {}).get("text"),
                "translation_strategy": (u.get("policy") or {}).get("translation_strategy"),
                "coverage_required": (u.get("policy") or {}).get("coverage_required"),
            }
            for u in units if (u.get("policy") or {}).get("translatable")
        ]
        render_units = [
            {
                "unit_id": u["unit_id"],
                "level": u["level"],
                "bbox": (u.get("geometry") or {}).get("bbox"),
                "render_mode": (u.get("render_contract") or {}).get("mode"),
            }
            for u in units
            if u["level"] in {"block", "image", "drawing", "table", "overlay"}
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
        for u in units:
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
            "flat_units": [u["unit_id"] for u in units],
            "translation_units": translation_units,
            "render_units": render_units,
            "debug_units": debug_units,
            "debug_units_note": "unités notables uniquement ; détail complet sur units[]",
        }


def build_pageprint_input_data(*, page_structure: dict,
                               source_context: dict | None = None,
                               extraction_result: dict | None = None,
                               assets: dict | None = None,
                               page_index: int = 0,
                               validate: bool = True) -> dict:
    """Fonction principale : construit l'INPUT_DATA canonique d'une page."""
    return PagePrintBuilder().build(
        page_structure=page_structure,
        source_context=source_context,
        extraction_result=extraction_result,
        assets=assets,
        page_index=page_index,
        validate=validate,
    )


__all__ = [
    "PagePrintBuilder",
    "build_pageprint_input_data",
    "PAGEPRINT_SCHEMA_VERSION",
]
