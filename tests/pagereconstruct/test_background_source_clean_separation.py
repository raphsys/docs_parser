import pytest

from pagereconstruct.background_contract import BackgroundContract
from pagereconstruct.errors import PublicationReadyError
from pagereconstruct.render_ops import BackgroundOp, assert_publication_background_allowed


def test_publication_requires_clean_background():
    contract = BackgroundContract(
        source_image_path="/tmp/source.png",
        background_mode="source_background",
        source_text_leak_risk="high",
    )

    assert "missing_clean_background" in contract.publication_blockers()
    assert "source_background_forbidden" in contract.publication_blockers()
    assert contract.render_path("publication") is None


def test_clean_background_must_be_verified():
    contract = BackgroundContract(
        clean_background_path="/tmp/clean.png",
        source_image_path="/tmp/source.png",
        background_mode="clean_background",
        clean_background_verified=False,
        source_text_leak_risk="low",
    )

    assert "clean_background_not_verified" in contract.publication_blockers()


def test_publication_forbids_source_image_background():
    op = BackgroundOp(path="/tmp/source.png", mode="publication", is_clean=False)

    with pytest.raises(PublicationReadyError, match="source_image_background_forbidden"):
        assert_publication_background_allowed(op, source_image_path="/tmp/source.png")
