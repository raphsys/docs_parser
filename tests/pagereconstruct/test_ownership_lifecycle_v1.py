from pageprint import build_pageprint_input_data
from pagetranslate.builder import build_page_translation
from pagereconstruct import compile_page_render_plan
from tests.pagereconstruct._fixtures import translated_input_data
from source_ownership import build_source_ownership


def _formula_page():
    return {
        "page_role": "body",
        "page_family": "body_text",
        "document_type": "technical_book",
        "layout_type": "single_column",
        "dimensions": {"width": 600, "height": 800, "render_dpi": 150},
        "special_regions": [
            {"special_class": "formula", "bbox": [100, 110, 320, 145], "confidence": 0.95}
        ],
        "blocks": [
            {
                "id": "b1",
                "bbox": [50, 50, 550, 95],
                "role": "body",
                "lines": [{"id": "l1", "bbox": [50, 55, 550, 75], "line_text": "This sentence should translate.", "phrases": []}],
            },
            {
                "id": "eq1",
                "bbox": [100, 110, 320, 145],
                "role": "equation_block",
                "lines": [{"id": "eq_l1", "bbox": [100, 110, 320, 145], "line_text": "E = mc^2", "phrases": []}],
            },
        ],
    }


def test_formula_region_has_exclusive_preserved_visual_ownership():
    input_data = build_pageprint_input_data(
        page_structure=_formula_page(),
        source_context={"language": {"source_lang": "en", "target_lang": "fr"}},
    )
    ownership = build_source_ownership(input_data)
    formula_owned = [e for e in ownership.values() if e.get("state") == "preserved_visual"]
    assert formula_owned, "formula source units must be owned by preservation"

    result = build_page_translation(input_data, dry_run=True, allow_fallback=False)
    translated_sids = {
        sid for item in result["translation_units"] for sid in item.get("source_unit_ids") or []
    }
    assert not (translated_sids & {e["source_unit_id"] for e in formula_owned})
    assert any("This sentence should translate" in item["source_text"] for item in result["translation_units"])


def test_formula_preservation_reaches_final_contract_without_text_conflict():
    plan = compile_page_render_plan(translated_input_data()).to_dict()
    states = plan["final_contract"]["source_unit_states"]
    formula_state = [s for s in states if s["source_unit_id"] == "fml1"]
    assert formula_state
    assert formula_state[0]["state"] in {"preserved_visual", "preserved_exact"}
    assert "source_unit_both_preserved_and_translated" not in formula_state[0].get("findings", [])
    assert not any("fml1" in (t.get("source_unit_ids") or []) for t in plan["layers"]["translated_text"])
    assert any("fml1" in (op.get("source_unit_ids") or []) for op in plan["render_ops"] if op.get("op_type") == "preservation")
