"""Dédup hiérarchique parent/enfant : enfants prioritaires."""

from pagetranslate.projection import _dedupe_parent_child_units


def _ti():
    return {"units": [
        {"unit_id": "blk7", "children_ids": ["blk7_l1", "blk7_l2"]},
        {"unit_id": "blk7_l1", "children_ids": []},
        {"unit_id": "blk7_l2", "children_ids": []},
        {"unit_id": "blk9", "children_ids": []},
    ]}


def test_parent_dropped_when_children_rendered():
    output = [
        {"unit_id": "ru_parent", "source_unit_ids": ["blk7"], "translated_text": "parent"},
        {"unit_id": "ru_c1", "source_unit_ids": ["blk7_l1"], "translated_text": "child1"},
        {"unit_id": "ru_c2", "source_unit_ids": ["blk7_l2"], "translated_text": "child2"},
        {"unit_id": "ru_other", "source_unit_ids": ["blk9"], "translated_text": "other"},
    ]
    kept = _dedupe_parent_child_units(output, _ti())
    ids = {u["unit_id"] for u in kept}
    assert "ru_parent" not in ids          # parent supprimé
    assert {"ru_c1", "ru_c2", "ru_other"} <= ids  # enfants + autre gardés


def test_parent_kept_when_no_child_rendered():
    output = [
        {"unit_id": "ru_parent", "source_unit_ids": ["blk7"], "translated_text": "parent"},
        {"unit_id": "ru_other", "source_unit_ids": ["blk9"], "translated_text": "other"},
    ]
    kept = _dedupe_parent_child_units(output, _ti())
    ids = {u["unit_id"] for u in kept}
    assert "ru_parent" in ids               # pas d'enfant rendu -> parent conservé
