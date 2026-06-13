import json

from translation_engines.profile_store import ProfileStore, load_profile_store


def test_load_profiles_merges_default_and_lang(tmp_path):
    profiles = {
        "default": {"quality_thresholds": {"critical_below": 72, "review_below": 88}, "post_edit": {"generic_cleanup": True}},
        "fr": {"post_edit": {"generic_replacements": [{"pattern": "x", "replace": "y"}]}},
    }
    style_tone = {"fr": {"styles": {}, "tones": {}}}
    p_path = tmp_path / "translation_profiles.json"
    s_path = tmp_path / "style_tone_profiles.json"
    p_path.write_text(json.dumps(profiles), encoding="utf-8")
    s_path.write_text(json.dumps(style_tone), encoding="utf-8")

    store = load_profile_store(str(p_path), str(s_path))
    merged = store.translation_profile("fr")
    assert merged["quality_thresholds"]["critical_below"] == 72
    assert merged["post_edit"]["generic_cleanup"] is True
    assert merged["post_edit"]["generic_replacements"]

    engine_profile = store.engine_profile(target_lang="fr", style="professionnel", tone="neutre", domain="ml")
    assert engine_profile["has_profile"] is True
    assert engine_profile["quality_thresholds"]["review_below"] == 88


def test_missing_files_never_raise():
    store = load_profile_store("/no/such/profiles.json", "/no/such/style.json")
    assert isinstance(store, ProfileStore)
    assert store.is_empty
    # Still returns a usable engine profile without crashing.
    assert store.engine_profile(target_lang="fr")["has_profile"] is False
