"""Phase 10 — non-régression: pagereconstruct accepte les contrats legacy utiles
(ocr_server/reconstructor: background_path, immutable_overlays, final_blocks,
document_object_contract, render policies, styles)."""

from pagereconstruct.final_contract import FinalReconstructionContract
from pagereconstruct import legacy_contract_bridge as bridge


def _legacy_page(**extra):
    base = {"background_path": "/clean/bg.png", "source_image_path": "/src.png",
            "blocks": [{"id": "b1", "bbox": [10, 10, 200, 30], "role": "body_paragraph",
                        "alignment": "left",
                        "lines": [{"phrases": [{"spans": [{"style": {"font": "Times", "size": 11, "flags": {}}}],
                                                "texte": "Hello world"}]}]}],
            "immutable_overlays": [{"bbox": [300, 10, 360, 26], "reason": "logo"}],
            "non_text_zones": [[100, 400, 300, 600]],
            "mask_master_path": "/mask.png"}
    base.update(extra)
    return base


def test_old_background_contract_still_supported():
    bg = bridge.extract_legacy_background(_legacy_page())
    assert bg.background_mode == "clean_background"
    assert bg.publication_allowed is True


def test_old_immutable_overlays_still_supported():
    ovs = bridge.extract_legacy_immutable_overlays(_legacy_page())
    assert len(ovs) == 1 and ovs[0].reason == "logo"


def test_old_final_blocks_can_be_adapted():
    blocks = bridge.extract_legacy_final_blocks(_legacy_page())
    assert blocks and blocks[0].source_text.strip() == "Hello world"
    assert blocks[0].layout.layout_bbox == [10, 10, 200, 30]


def test_old_document_object_contract_can_be_adapted():
    c = FinalReconstructionContract.from_legacy_contract(_legacy_page())
    # non_text_zones -> ObjectContracts
    assert c.objects, "object contracts from legacy zones"


def test_old_reconstruction_contract_can_be_adapted():
    c = FinalReconstructionContract.from_legacy_contract(_legacy_page())
    assert c.blocks and c.blocks[0].render.renderer_name == "paragraph"


def test_old_style_contract_can_be_adapted():
    styles = bridge.extract_legacy_style_contracts(_legacy_page())
    assert styles["b1"].font_size_pt == 11.0
    assert styles["b1"].source == "extracted"


def test_old_render_policy_can_be_adapted():
    pol = bridge.extract_legacy_render_policies(_legacy_page(blocks=[{"id": "b1", "bbox": [0, 0, 1, 1], "render_policy": "fixed_preserve"}]))
    assert pol["b1"] == "fixed_preserve"


def test_legacy_does_not_reintroduce_source_text_and_merges():
    # un contrat moderne incomplet (sans clean bg) complété par le legacy.
    modern = FinalReconstructionContract()
    modern.merge_legacy_and_new(FinalReconstructionContract.from_legacy_contract(_legacy_page()))
    assert modern.background.background_mode == "clean_background"
