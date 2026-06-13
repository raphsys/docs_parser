from translation_engines.placeholder_policy import PLACEHOLDER_STYLES, build_placeholder, placeholder_variants


def test_placeholder_variants_roundtrip():
    for idx, style in enumerate(PLACEHOLDER_STYLES, start=1):
        placeholder = build_placeholder(idx, style)
        assert placeholder
        assert any(variant for variant in placeholder_variants(idx))

