from pagetranslate import build_page_translation
from pagetranslate.functional_validator import validate_functional_translation


def test_all_original_units_get_reconstruction_disposition():
    input_data = {
        "schema_version": "pageprint.input.v1",
        "input_id": "all-originals",
        "document": {"language": {"source_lang": "en", "target_lang": "fr"}},
        "page": {},
        "translation_context": {},
        "units": [
            {
                "unit_id": "b1",
                "level": "block",
                "content": {"text": "This paragraph must be translated."},
                "geometry": {"bbox": [0, 0, 200, 40], "reading_order_index": 1},
                "understanding": {"role": "body_paragraph", "object_type": "natural_text"},
                "policy": {"translatable": True, "translation_strategy": "layout_constrained"},
                "visual": {"style": {}},
                "children_ids": [],
            },
            {
                "unit_id": "f1",
                "level": "block",
                "content": {"text": "z = beta0 + beta1 age"},
                "geometry": {"bbox": [0, 60, 200, 80], "reading_order_index": 2},
                "understanding": {"role": "formula_expression", "object_type": "formula"},
                "policy": {"translatable": False, "translation_strategy": "exact_preserve"},
                "visual": {"style": {}},
                "children_ids": [],
            },
        ],
        "views": {},
    }

    result = build_page_translation(input_data, dry_run=True)
    reconstruction_units = result["translated_input_data"]["views"]["reconstruction_units"]
    by_source = {
        sid: unit
        for unit in reconstruction_units
        for sid in unit.get("source_unit_ids") or []
    }

    # Reconstruction units are now strictly for translated text.  Exact-preserve
    # source text must come from PAGEPRINT preservation/exclusion plans, not from
    # PAGETRANSLATE silently re-injecting original text as a TextOp.
    assert set(by_source) == {"b1"}
    assert by_source["b1"]["render_contract"]["mode"] == "translated_text"


def test_original_text_coverage_audit_blocks_missing_text_unit():
    result = build_page_translation({
        "schema_version": "pageprint.input.v1",
        "input_id": "missing-original",
        "document": {"language": {"source_lang": "en", "target_lang": "fr"}},
        "page": {},
        "units": [{
            "unit_id": "b1",
            "level": "block",
            "content": {"text": "This text must not disappear."},
            "geometry": {"bbox": [0, 0, 200, 40], "reading_order_index": 1},
            "understanding": {"role": "body_paragraph", "object_type": "natural_text"},
            "policy": {"translatable": True, "translation_strategy": "layout_constrained"},
            "visual": {"style": {}},
            "children_ids": [],
        }],
        "views": {},
    }, dry_run=True)
    broken = dict(result)
    broken["translated_input_data"] = dict(result["translated_input_data"])
    broken["translated_input_data"]["views"] = {"reconstruction_units": []}

    validation = validate_functional_translation(broken)

    assert validation["functional_status"] == "ko"
    assert validation["metrics"]["original_text_missing_disposition"] == 1
