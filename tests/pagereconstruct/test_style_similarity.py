from pagereconstruct.style_similarity import similarity


def _style(size=11.0, serif=True, bold=False, color="#000000"):
    return {"font_size_pt": size, "color": color,
            "flags": {"serif": serif, "bold": bold, "italic": False, "monospace": False}}


def test_identical_style_is_perfect():
    s = similarity(_style(), _style())
    assert s["score"] >= 0.99 and s["status"] == "ok"


def test_serif_to_sans_lowers_score():
    s = similarity(_style(serif=True), _style(serif=False))
    assert s["score"] < 0.95


def test_size_mismatch_lowers_score():
    s = similarity(_style(size=11.0), _style(size=6.0))
    assert s["components"]["size"] < 0.7


def test_bold_loss_lowers_score():
    s = similarity(_style(bold=True), _style(bold=False))
    assert s["components"]["bold"] == 0.0
