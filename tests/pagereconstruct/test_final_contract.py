"""Phase 2/3 — FinalReconstructionContract + LegacyContractBridge."""

from pagereconstruct import compile_page_render_plan
from pagereconstruct.input_adapter import PageReconstructInputAdapter
from pagereconstruct.final_contract import FinalReconstructionContract, LAYER_ORDER
from tests.pagereconstruct._fixtures import translated_input_data


def _build():
    tid = translated_input_data()
    norm = PageReconstructInputAdapter().normalize(tid)
    plan = compile_page_render_plan(tid).to_dict()
    return FinalReconstructionContract.from_pageprint_pagetranslate(norm, plan)


def test_final_contract_accepts_pageprint_pagetranslate_units():
    c = _build()
    assert c.blocks, "blocks should be built from translated_text"
    assert all(b.layout.layout_bbox for b in c.blocks)


def test_final_contract_layer_order_is_fixed():
    c = _build()
    assert c.layer_order == LAYER_ORDER
    assert c.layer_order.index("translated_text") > c.layer_order.index("preserved_underlays")


def test_final_contract_blocks_publication_if_background_not_clean():
    c = _build()  # fixture = source background → leak high
    res = c.validate(mode="publication")
    assert res["valid"] is False
    assert any("clean_background" in e or "leak" in e for e in res["errors"])


def test_final_contract_accepts_legacy_background():
    legacy = {"background_path": "/x/clean.png", "source_image_path": "/x/src.png"}
    c = FinalReconstructionContract.from_legacy_contract(legacy)
    assert c.background.background_mode == "clean_background"
    assert c.background.publication_allowed is True


def test_final_contract_accepts_immutable_overlays():
    legacy = {"background_path": "/x/clean.png",
              "immutable_overlays": [{"bbox": [1, 2, 3, 4], "reason": "logo"}]}
    c = FinalReconstructionContract.from_legacy_contract(legacy)
    assert len(c.preservation.objects) == 1
    assert c.preservation.objects[0].reason == "logo"


def test_final_contract_preserves_old_block_fields():
    legacy = {"background_path": "/x/clean.png",
              "blocks": [{"id": "b1", "bbox": [10, 10, 200, 30], "role": "section_heading",
                          "lines": [{"phrases": [{"spans": [{"style": {"font": "Times", "size": 14, "flags": {"bold": True}}}],
                                                  "texte": "Title"}]}]}]}
    c = FinalReconstructionContract.from_legacy_contract(legacy)
    assert c.blocks and c.blocks[0].role == "section_heading"
    assert c.blocks[0].render.renderer_name == "heading"
    assert c.blocks[0].style.font_size_pt == 14.0


def test_merge_legacy_fills_clean_background():
    c = _build()  # leak high, no clean bg
    legacy = FinalReconstructionContract.from_legacy_contract({"background_path": "/x/clean.png"})
    c.merge_legacy_and_new(legacy)
    assert c.background.background_mode == "clean_background"
    assert c.background.source_text_leak_risk == "low"
