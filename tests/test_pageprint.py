"""Tests du contrat PAGEPRINT (pageprint.input.v1).

Vérifie sur une page synthétique legacy (pixels 150 DPI) que :
- le contrat INPUT_DATA_V1_REQUIRED est respecté ;
- les unités respectent UNIT_REQUIRED avec bbox canonique en points ;
- les régions priment sur le texte (formula → exact_preserve/fixed_preserve) ;
- la validation passe ;
- la compatibilité legacy est conservée.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pageprint import (
    INPUT_DATA_V1_REQUIRED,
    PAGEPRINT_SCHEMA_VERSION,
    UNIT_REQUIRED,
    PagePrintBuilder,
    build_pageprint_input_data,
    validate_input_data,
)
from pageprint.normalizer import (
    bbox_pt_to_px,
    bbox_px_to_pt,
    normalize_color,
    normalize_page_geometry,
    normalize_style,
    scale_from_dimensions,
)

# Page A4 rendue à 150 DPI : 1240 × 1754 px ≈ 595.2 × 841.92 pt.
PAGE_DIMENSIONS = {"width": 1240, "height": 1754, "render_dpi": 150}
SCALE = 150.0 / 72.0


def make_page_structure():
    return {
        "page_role": "body",
        "page_family": "body_with_figure",
        "document_type": "scientific_paper",
        "layout_type": "single_column",
        "style_profile": "editorial",
        "dimensions": dict(PAGE_DIMENSIONS),
        "page_case_v2": {
            "reading_modes": {
                "linear_flow": 0.8,
                "anchored_overlay_flow": 0.1,
                "tabular_grid_flow": 0.0,
                "toc_row_flow": 0.0,
            },
            "risk_flags": [{"code": "translation_overflow", "severity": 0.4}],
        },
        "special_regions": [
            {
                "region_type": "formula",
                "bbox": [400, 800, 700, 900],
                "confidence": 0.93,
                "source": "special_region_detector",
            }
        ],
        "blocks": [
            {
                "id": "b1",
                "bbox": [100, 100, 1100, 300],
                "role": "body",
                "source_kind": "native",
                "translation_contract": {
                    "translatable": True,
                    "translation_strategy": "paragraph_flow",
                    "render_policy": "paragraph_flow",
                    "coverage_required": "normal",
                },
                "lines": [
                    {
                        "id": "b1l1",
                        "bbox": [100, 100, 1100, 160],
                        "line_text": "The model learns features from data.",
                        "phrases": [
                            {
                                "id": "b1l1p1",
                                "bbox": [100, 100, 1100, 160],
                                "texte": "The model learns features from data.",
                                "spans": [
                                    {
                                        "id": "b1l1p1s1",
                                        "bbox": [100, 100, 1100, 160],
                                        "text": "The model learns features from data.",
                                        "style": {"font": "Times", "size": 22, "color": [0, 0, 0]},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "id": "b2",
                "bbox": [420, 810, 680, 890],
                "role": "equation_inline",
                "source_kind": "native",
                "lines": [
                    {
                        "id": "b2l1",
                        "bbox": [420, 810, 680, 890],
                        "line_text": "E = mc^2",
                        "phrases": [],
                    }
                ],
            },
        ],
    }


def build():
    return build_pageprint_input_data(
        page_structure=make_page_structure(),
        source_context={
            "document_id": "doc-test",
            "source_path": "/tmp/doc.pdf",
            "file_name": "doc.pdf",
            "file_type": "pdf",
            "page_count": 1,
            "language": {"source_lang": "en", "target_lang": "fr"},
        },
        extraction_result={
            "pipeline": "legacy",
            "native_pdf_available": True,
            "ocr_used": False,
        },
        assets={"source_image_path": None, "background_path": "/tmp/bg.png"},
        page_index=0,
    )


class TestSchemaContract:
    def test_schema_version(self):
        input_data = build()
        assert input_data["schema_version"] == PAGEPRINT_SCHEMA_VERSION == "pageprint.input.v1"

    def test_required_top_level_keys_present(self):
        input_data = build()
        missing = INPUT_DATA_V1_REQUIRED - set(input_data.keys())
        assert not missing

    def test_validation_passes(self):
        input_data = build()
        report = input_data["debug"]["validation"]
        assert report["valid"], report["errors"]

    def test_validate_detects_duplicate_unit_id(self):
        input_data = build()
        input_data["units"].append(dict(input_data["units"][0]))
        report = validate_input_data(input_data)
        assert not report["valid"]
        assert any("duplicate_unit_id" in e for e in report["errors"])

    def test_validate_detects_missing_top_level_key(self):
        input_data = build()
        del input_data["policies"]
        report = validate_input_data(input_data)
        assert not report["valid"]


class TestUnits:
    def test_units_respect_contract(self):
        input_data = build()
        assert input_data["units"]
        for unit in input_data["units"]:
            missing = UNIT_REQUIRED - set(unit.keys())
            assert not missing, f"{unit['unit_id']}: {missing}"

    def test_hierarchy_block_line_phrase_span(self):
        input_data = build()
        levels = {u["level"] for u in input_data["units"]}
        assert {"block", "line", "phrase", "span"} <= levels
        by_id = {u["unit_id"]: u for u in input_data["units"]}
        for unit in input_data["units"]:
            if unit["parent_id"] is not None:
                assert unit["parent_id"] in by_id

    def test_bbox_canonical_in_points(self):
        input_data = build()
        block = next(u for u in input_data["units"] if u["level"] == "block")
        geo = block["geometry"]
        assert geo["bbox_unit"] == "pt"
        # 100 px / (150/72) = 48 pt
        assert math.isclose(geo["bbox"][0], 100 / SCALE, abs_tol=0.01)
        assert geo["bbox_px"] == [100.0, 100.0, 1100.0, 300.0]
        assert geo["bbox_px_dpi"] == 150

    def test_font_size_converted_to_pt(self):
        input_data = build()
        span = next(u for u in input_data["units"] if u["level"] == "span")
        style = span["visual"]["style"]
        # 22 px / (150/72) = 10.56 pt
        assert math.isclose(style["font_size_pt"], 22 / SCALE, abs_tol=0.05)
        assert style["color"] == "#000000"

    def test_reading_order_chained(self):
        input_data = build()
        ordered = sorted(
            input_data["units"],
            key=lambda u: u["geometry"]["reading_order_index"],
        )
        assert ordered[0]["relations"]["next_unit_id"] == ordered[1]["unit_id"]
        assert ordered[1]["relations"]["previous_unit_id"] == ordered[0]["unit_id"]


class TestRegionsAndPolicies:
    def test_formula_region_built(self):
        input_data = build()
        formulas = [r for r in input_data["regions"] if r["region_type"] == "formula"]
        assert len(formulas) == 1
        assert formulas[0]["bbox_unit"] == "pt"
        assert formulas[0]["policy"]["translatable"] is False

    def test_region_stronger_than_text(self):
        """La région formule doit imposer exact_preserve/fixed_preserve au bloc inclus."""
        input_data = build()
        formula_block = next(
            u for u in input_data["units"]
            if u["level"] == "block" and (u["content"]["text"] or "").startswith("E = mc")
        )
        memberships = formula_block["understanding"]["region_memberships"]
        assert any(m["region_type"] == "formula" for m in memberships)
        policy = formula_block["policy"]
        assert policy["translatable"] is False
        assert policy["translation_strategy"] == "exact_preserve"
        assert policy["render_policy"] == "fixed_preserve"
        assert policy["policy_source"] == "region:formula"

    def test_evidence_resolution_rule(self):
        input_data = build()
        formula_block = next(
            u for u in input_data["units"]
            if (u.get("policy") or {}).get("policy_source") == "region:formula"
            and u["level"] == "block"
        )
        evidence = formula_block["evidence"]
        assert evidence["resolved_as"] == "formula"
        assert evidence["resolution_rule"] == "special_region_over_text_when_overlap_gt_0.65"

    def test_prose_block_keeps_legacy_contract(self):
        input_data = build()
        prose = next(
            u for u in input_data["units"]
            if u["level"] == "block" and u["unit_id"].endswith("block_001")
        )
        # Texte agrégé depuis les lignes + contrat legacy conservé.
        assert prose["content"]["text"].startswith("The model learns")
        assert prose["policy"]["translation_strategy"] == "paragraph_flow"

    def test_unit_policies_compiled(self):
        input_data = build()
        policies = input_data["policies"]
        assert "default_policy" in policies
        # Seules les exceptions sont indexées ; la politique complète vit
        # sur chaque unité.
        all_ids = {u["unit_id"] for u in input_data["units"]}
        exceptions = policies["unit_policy_exceptions"]
        assert set(exceptions) <= all_ids
        formula_block = next(
            u for u in input_data["units"]
            if (u.get("policy") or {}).get("policy_source") == "region:formula"
            and u["level"] == "block"
        )
        assert formula_block["unit_id"] in exceptions
        for unit in input_data["units"]:
            assert unit["policy"].get("translation_strategy")


class TestConstraintsAndViews:
    def test_constraints_compiled_per_unit(self):
        input_data = build()
        for unit in input_data["units"]:
            assert "allow_reflow" in unit["constraints"]
            assert "transformation_budget" in unit
            assert "layout_freedom" in unit
            assert "render_contract" in unit

    def test_formula_constraints_all_fixed(self):
        input_data = build()
        formula_block = next(
            u for u in input_data["units"]
            if (u.get("policy") or {}).get("policy_source") == "region:formula"
            and u["level"] == "block"
        )
        assert formula_block["constraints"]["preserve_bbox"] is True
        assert formula_block["constraints"]["allow_reflow"] is False
        assert formula_block["layout_freedom"]["font_size"] == "fixed"

    def test_reconstruction_constraints_present(self):
        input_data = build()
        rc = input_data["reconstruction_constraints"]
        assert rc["page"]["coordinate_unit"] == "pt"
        assert rc["visual_preservation"]["preserve_formulas_as_images"] is True

    def test_views_built(self):
        input_data = build()
        views = input_data["views"]
        assert views["flat_units"]
        assert views["hierarchical"]["roots"]
        assert all(tu["text"] for tu in views["translation_units"])
        formula_ids = {
            u["unit_id"] for u in input_data["units"]
            if (u.get("policy") or {}).get("translatable") is False
        }
        assert formula_ids.isdisjoint({tu["unit_id"] for tu in views["translation_units"]})


class TestQualityProvenanceCompatibility:
    def test_quality_and_risks(self):
        input_data = build()
        assert 0.0 <= input_data["quality"]["overall_confidence"] <= 1.0
        assert input_data["risks"]["risk_flags"]
        for unit in input_data["units"]:
            assert unit["confidence"]["overall"] is not None

    def test_provenance_with_decision_trace(self):
        input_data = build()
        provenance = input_data["provenance"]
        assert provenance["pipeline_version"] == PAGEPRINT_SCHEMA_VERSION
        assert provenance["modules"]["pageprint"] == "enabled"
        assert any(
            t.get("stage") == "policy_compilation"
            for t in provenance["decision_trace"]
        )
        assert "replay" in provenance

    def test_compatibility_keeps_legacy_structure(self):
        input_data = build()
        legacy = input_data["compatibility"]["legacy_page_structure"]
        assert legacy["page_role"] == "body"
        assert legacy["blocks"]

    def test_translation_context_lists_units(self):
        input_data = build()
        ctx = input_data["translation_context"]
        assert ctx["source_lang"] == "en"
        assert ctx["target_lang"] == "fr"
        assert ctx["non_translatable_units"]


class TestNormalizer:
    def test_bbox_roundtrip(self):
        bbox_px = [100.0, 200.0, 300.0, 400.0]
        bbox_pt = bbox_px_to_pt(bbox_px, SCALE, SCALE)
        back = bbox_pt_to_px(bbox_pt, SCALE, SCALE)
        assert all(math.isclose(a, b, abs_tol=0.01) for a, b in zip(bbox_px, back))

    def test_scale_from_legacy_dimensions(self):
        sx, sy = scale_from_dimensions(PAGE_DIMENSIONS)
        assert math.isclose(sx, SCALE, rel_tol=1e-6)
        assert math.isclose(sy, SCALE, rel_tol=1e-6)

    def test_page_geometry_points_first(self):
        geo = normalize_page_geometry(PAGE_DIMENSIONS)
        assert geo["unit"] == "pt"
        assert math.isclose(geo["width"], 1240 / SCALE, abs_tol=0.01)
        assert geo["render_width_px"] == 1240

    def test_normalize_color_variants(self):
        assert normalize_color([255, 0, 0]) == "#ff0000"
        assert normalize_color([1.0, 1.0, 1.0]) == "#ffffff"
        assert normalize_color("#ABC") == "#aabbcc"
        assert normalize_color(0x00FF00) == "#00ff00"

    def test_native_style_size_stays_pt(self):
        style = normalize_style({"size": 12, "font": "Arial"}, sx=SCALE, source="native_pdf")
        assert style["font_size_pt"] == 12.0


class TestSizeOptimizations:
    def test_merge_phrase_spans_same_style(self):
        from pageprint.unit_factory import merge_phrase_spans

        style = {"font": "Times", "size": 10, "color": "#000000"}
        spans = [
            {"id": "s1", "text": "Chapter", "bbox": [10, 10, 50, 22], "style": dict(style)},
            {"id": "s2", "text": " ", "bbox": [50, 10, 54, 22], "style": dict(style)},
            {"id": "s3", "text": "2", "bbox": [54, 10, 60, 22], "style": dict(style)},
            {"id": "s4", "text": "bold", "bbox": [70, 10, 95, 22],
             "style": {**style, "bold": True}},
        ]
        merged = merge_phrase_spans(spans)
        assert len(merged) == 2
        assert merged[0]["text"] == "Chapter 2"
        assert merged[0]["bbox"] == [10, 10, 60, 22]
        assert merged[0]["merged_from"] == ["s1", "s2", "s3"]
        assert merged[1]["text"] == "bold"

    def test_merge_inserts_space_on_gap(self):
        from pageprint.unit_factory import merge_phrase_spans

        style = {"font": "Times", "size": 10}
        spans = [
            {"id": "s1", "text": "Hello", "bbox": [10, 10, 40, 22], "style": dict(style)},
            {"id": "s2", "text": "world", "bbox": [48, 10, 80, 22], "style": dict(style)},
        ]
        merged = merge_phrase_spans(spans)
        assert merged[0]["text"] == "Hello world"

    def test_merge_no_space_when_contiguous(self):
        from pageprint.unit_factory import merge_phrase_spans

        style = {"font": "Times", "size": 10}
        spans = [
            {"id": "s1", "text": "hyphen", "bbox": [10, 10, 40, 22], "style": dict(style)},
            {"id": "s2", "text": "ated", "bbox": [40.2, 10, 60, 22], "style": dict(style)},
        ]
        merged = merge_phrase_spans(spans)
        assert merged[0]["text"] == "hyphenated"

    def test_decorative_strokes_not_units(self):
        """Les filets de 0.5 pt d'épaisseur ne deviennent pas des unités."""
        structure = make_page_structure()
        structure["images"] = [
            # trait de bordure : 30 pt × 0.5 pt (en pixels 150 DPI)
            {"bbox": [100, 500, 100 + 30 * SCALE, 500 + 0.5 * SCALE]},
            # vraie image : 100 × 80 pt
            {"bbox": [100, 600, 100 + 100 * SCALE, 600 + 80 * SCALE]},
        ]
        input_data = PagePrintBuilder().build(page_structure=structure)
        image_units = [u for u in input_data["units"] if u["level"] == "image"]
        assert len(image_units) == 1
        stats = input_data["extraction"]["normalization"]
        assert stats["visuals_filtered_decorative"] == 1
        assert stats["visuals_total"] == 2

    def test_regions_dedup_image_vs_non_text_zone(self):
        """La même bbox en image + non_text_zone ne produit qu'une région."""
        structure = make_page_structure()
        bbox = [100, 600, 100 + 100 * SCALE, 600 + 80 * SCALE]
        structure["images"] = [{"bbox": list(bbox)}]
        structure["non_text_zones"] = [list(bbox)]
        input_data = PagePrintBuilder().build(page_structure=structure)
        visual_regions = [
            r for r in input_data["regions"]
            if r["region_type"] in {"image_region", "non_text_zone"}
        ]
        assert len(visual_regions) == 1
        assert visual_regions[0]["region_type"] == "image_region"
        assert input_data["extraction"]["normalization"]["regions_deduplicated"] == 1

    def test_span_merge_reflected_in_stats(self):
        input_data = build()
        stats = input_data["extraction"]["normalization"]
        assert stats["spans_after_merge"] <= stats["spans_before_merge"]

    def test_no_per_unit_duplication_in_annex_sections(self):
        """units[] est la source de vérité ; les sections annexes n'en
        dupliquent pas le contenu."""
        input_data = build()
        # constraints : plus de copie par unité.
        assert "unit_constraints" not in input_data["constraints"]
        assert input_data["constraints"]["unit_constraints_location"] == "units[].constraints"
        for unit in input_data["units"]:
            assert "allow_reflow" in unit["constraints"]
        # graph : niveaux fins absents (dérivables des unités).
        node_types = {n["type"] for n in input_data["graph"]["nodes"]}
        assert node_types.isdisjoint({"line", "phrase", "span"})
        assert "block" in node_types
        # debug_units : unités notables uniquement, avec raison.
        debug = input_data["views"]["debug_units"]
        assert len(debug) < len(input_data["units"])
        assert all(d.get("reason") for d in debug)
        assert any(d["reason"] == "region_policy_override" for d in debug)
        # render_units : index léger, le contrat complet reste sur l'unité.
        for entry in input_data["views"]["render_units"]:
            assert "render_contract" not in entry
            assert "render_mode" in entry


class TestBuilderAPI:
    def test_builder_class_entry_point(self):
        input_data = PagePrintBuilder().build(page_structure=make_page_structure())
        assert input_data["schema_version"] == PAGEPRINT_SCHEMA_VERSION

    def test_empty_page_structure_is_valid(self):
        input_data = PagePrintBuilder().build(page_structure={})
        report = input_data["debug"]["validation"]
        assert report["valid"], report["errors"]
        assert input_data["units"] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
