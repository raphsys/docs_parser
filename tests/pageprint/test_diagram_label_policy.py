from pageprint.structure_builders.figure_builder import build_figures


def test_diagram_label_policy():
    figures = build_figures([
        {
            "unit_id": "fig1",
            "level": "image",
            "understanding": {"object_type": "diagram"},
            "geometry": {"bbox": [0, 0, 100, 100]},
            "diagram_labels": [{"text": "ReLU"}, {"text": "Hidden layer"}],
        }
    ])
    labels = figures[0]["diagram_labels"]
    assert labels[0]["translation_mode"] == "preserve_text_exactly"
    assert labels[1]["translation_mode"] == "translate"
