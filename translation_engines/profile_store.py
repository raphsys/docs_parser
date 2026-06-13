"""Translation/style profile store.

Loads ``translation_profiles.json`` and ``style_tone_profiles.json`` so the
runtime can pick quality thresholds, post-edit rules, glossary and style/tone
without the engine guessing. Missing files never raise: an empty store is
returned and the caller falls back to defaults.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_PROFILES_PATH = "ai_models/translation/translation_profiles.json"
DEFAULT_STYLE_TONE_PATH = "ai_models/translation/style_tone_profiles.json"


def _read_json(path: str | os.PathLike | None) -> dict:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.is_file():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


class ProfileStore:
    def __init__(self, profiles: dict | None = None, style_tone: dict | None = None):
        self.profiles = profiles or {}
        self.style_tone = style_tone or {}

    @property
    def is_empty(self) -> bool:
        return not self.profiles and not self.style_tone

    def translation_profile(self, target_lang: str | None = None) -> dict:
        """Merge the default profile with the target-language profile."""
        merged: dict = {}
        default = self.profiles.get("default")
        if isinstance(default, dict):
            merged = json.loads(json.dumps(default))
        lang = str(target_lang or "").lower().split("-")[0]
        lang_profile = self.profiles.get(lang)
        if isinstance(lang_profile, dict):
            for key, value in lang_profile.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
        return merged

    def style_tone_profile(self, target_lang: str | None = None) -> dict:
        lang = str(target_lang or "").lower().split("-")[0]
        profile = self.style_tone.get(lang)
        return profile if isinstance(profile, dict) else {}

    def quality_thresholds(self, target_lang: str | None = None) -> dict:
        return self.translation_profile(target_lang).get("quality_thresholds") or {}

    def engine_profile(self, *, target_lang: str | None = None, style: str | None = None, tone: str | None = None, domain: str | None = None) -> dict:
        """Compact profile summary injected into engine context/trace."""
        profile = self.translation_profile(target_lang)
        return {
            "target_lang": target_lang,
            "style": style,
            "tone": tone,
            "domain": domain,
            "quality_thresholds": profile.get("quality_thresholds") or {},
            "post_edit_enabled": bool((profile.get("post_edit") or {}).get("generic_cleanup", False)) or bool(profile.get("post_edit")),
            "has_profile": bool(profile),
            "has_style_tone": bool(self.style_tone_profile(target_lang)),
        }


def load_profile_store(profiles_path: str | os.PathLike | None = None, style_tone_path: str | os.PathLike | None = None) -> ProfileStore:
    profiles_path = profiles_path or os.getenv("TRANSLATION_PROFILES_PATH") or DEFAULT_PROFILES_PATH
    style_tone_path = style_tone_path or os.getenv("TRANSLATION_STYLE_TONE_PATH") or DEFAULT_STYLE_TONE_PATH
    return ProfileStore(profiles=_read_json(profiles_path), style_tone=_read_json(style_tone_path))
