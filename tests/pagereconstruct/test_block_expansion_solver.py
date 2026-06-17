"""Solveur d'expansion + reflux voisins (port legacy)."""

from types import SimpleNamespace

from pagereconstruct.block_expansion_solver import solve_block_expansion


def _block(bid, bbox, text, role="body_paragraph", size=10.0):
    style = SimpleNamespace(font_size_pt=size, bold=False, italic=False,
                            font_class="serif", color="#000000", alignment="left")
    layout = SimpleNamespace(layout_bbox=list(bbox), coverage_bbox=list(bbox),
                             source_bbox=list(bbox), bbox_locked=False)
    return SimpleNamespace(block_id=bid, role=role, style=style, layout=layout,
                           translated_text=text, source_text=text,
                           source_unit_ids=[bid], translation_unit_id=None)


def _contract(blocks, height=800.0, width=500.0):
    return SimpleNamespace(blocks=blocks, objects=[], preservation=None,
                           page=SimpleNamespace(width_pt=width, height_pt=height))


LONG = ("This is a very long paragraph that will not fit into a tiny box and "
        "therefore must force the block to expand downward to contain all of "
        "its text without shrinking the font below the bounded ratio. " * 3)


def test_overflowing_block_expands_downward():
    b1 = _block("b1", [50, 50, 450, 80], LONG)        # boîte courte (30pt) vs texte long
    res = solve_block_expansion(_contract([b1]), enabled=True)
    assert "b1" in res.patches_by_block_id
    p = res.patches_by_block_id["b1"]
    assert p.new_bbox[3] > p.old_bbox[3] + 1          # étendu vers le bas


def test_expansion_pushes_neighbour_below():
    b1 = _block("b1", [50, 50, 450, 80], LONG)
    b2 = _block("b2", [50, 90, 450, 130], "Neighbour paragraph below.")
    res = solve_block_expansion(_contract([b1, b2]), enabled=True)
    assert "b2" in res.patches_by_block_id
    p2 = res.patches_by_block_id["b2"]
    assert p2.new_bbox[1] > p2.old_bbox[1] + 1        # poussé vers le bas (reflux)


def test_overlapping_blocks_are_separated():
    # deux blocs aux boîtes qui se chevauchent verticalement -> packing les sépare
    b1 = _block("b1", [50, 100, 450, 160], "First paragraph short.")
    b2 = _block("b2", [50, 130, 450, 190], "Second paragraph short.")  # overlap 130-160
    res = solve_block_expansion(_contract([b1, b2]), enabled=True)
    boxes = {}
    for bid, base in (("b1", [50, 100, 450, 160]), ("b2", [50, 130, 450, 190])):
        p = res.patches_by_block_id.get(bid)
        boxes[bid] = p.new_bbox if p else base
    # plus de chevauchement vertical entre b1 et b2
    assert boxes["b2"][1] >= boxes["b1"][3] - 0.6


def test_locked_role_not_expanded():
    f = _block("f1", [50, 50, 450, 70], "E=mc^2 long long long " * 5, role="formula_expression")
    res = solve_block_expansion(_contract([f]), enabled=True)
    assert "f1" not in res.patches_by_block_id        # figé : jamais étendu


def test_expansion_capped_by_page_bottom():
    b1 = _block("b1", [50, 760, 450, 790], LONG)      # déjà en bas de page
    res = solve_block_expansion(_contract([b1], height=800.0), enabled=True)
    if "b1" in res.patches_by_block_id:
        assert res.patches_by_block_id["b1"].new_bbox[3] <= 800.0
