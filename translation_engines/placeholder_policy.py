from __future__ import annotations

import os


PLACEHOLDER_STYLES = ("unicode_bracket", "ascii_xml", "plain_ascii", "at_token")


def choose_placeholder_style(*, engine_name: str | None = None, prefer: str | None = None) -> str:
    env = str(prefer or os.getenv("TRANSLATION_PLACEHOLDER_STYLE") or "").strip().lower()
    if env in PLACEHOLDER_STYLES:
        return env
    if str(engine_name or "").lower() in {"ct2", "ctranslate2", "local", "external"}:
        return "ascii_xml"
    return "unicode_bracket"


def build_placeholder(index: int, style: str) -> str:
    token = f"PT{index:04d}"
    style = style if style in PLACEHOLDER_STYLES else "unicode_bracket"
    if style == "ascii_xml":
        return f'<nt id="{token}"/>'
    if style == "plain_ascii":
        return f"[[{token}]]"
    if style == "at_token":
        return f"@@{token}@@"
    return f"⟦{token}⟧"


def placeholder_variants(index: int) -> list[str]:
    token = f"PT{index:04d}"
    return [
        f"⟦{token}⟧",
        f"<nt id=\"{token}\"/>",
        f"[[{token}]]",
        f"@@{token}@@",
    ]
