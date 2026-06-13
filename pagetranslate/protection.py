"""Protection and restoration of non-translatable inline tokens."""

from __future__ import annotations

import re
from dataclasses import asdict

from translation_engines.placeholder_policy import build_placeholder, choose_placeholder_style

from .schema import TranslationProtection


PATTERNS = [
    ("url", re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)),
    ("email", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("doi", re.compile(r"\b(?:doi:\s*)?10\.\d{4,9}/\S+\b", re.IGNORECASE)),
    ("path", re.compile(r"(?<!\w)(?:[A-Za-z]:\\[^\s]+|/[\w./-]+|\.{1,2}/[^\s]+)")),
    ("reference", re.compile(r"(?<!\w)(?:\[\d+(?:,\s*\d+)*\]|\([A-Z][A-Za-z]+,\s*\d{4}\))")),
    ("number_unit", re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:%|kg|g|mg|cm|mm|m|km|MB|GB|GHz|Hz|ms|s|h|°C|°F|K|px|pt)\b", re.IGNORECASE)),
    ("formula", re.compile(
        r"""
        (?:
            \b[A-Za-z]\w*\s*(?:=|≈|<=|>=|<|>|\+|\*|/|\^)\s*[-\w.()^²³¹₀-₉]+
            |
            [α-ωΑ-Ω∑∫√∞≈≠≤≥±×÷]
            |
            \b\d+(?:[.,]\d+)?\s*(?:/|\*)\s*[A-Za-z]+\b
        )
        """,
        re.VERBOSE,
    )),
    ("number", re.compile(r"(?<![\w./-])\d+(?:[.,]\d+)?(?![\w./-])")),
]


def protect_text(
    text: str,
    explicit_tokens: list[str] | None = None,
    *,
    placeholder_style: str | None = None,
    engine_name: str | None = None,
) -> tuple[str, list[dict]]:
    matches = []
    occupied: list[range] = []
    style = choose_placeholder_style(engine_name=engine_name, prefer=placeholder_style)
    for token in explicit_tokens or []:
        token_text = str(token or "")
        if not token_text:
            continue
        for match in re.finditer(re.escape(token_text), text or ""):
            span = range(match.start(), match.end())
            if any(_overlaps(span, previous) for previous in occupied):
                continue
            occupied.append(span)
            matches.append((match.start(), match.end(), "explicit", match.group(0)))

    for kind, pattern in PATTERNS:
        for match in pattern.finditer(text or ""):
            span = range(match.start(), match.end())
            if any(_overlaps(span, previous) for previous in occupied):
                continue
            occupied.append(span)
            matches.append((match.start(), match.end(), kind, match.group(0)))
    matches.sort(key=lambda item: item[0])

    protected_parts = []
    protections = []
    cursor = 0
    for idx, (start, end, kind, value) in enumerate(matches, start=1):
        placeholder = build_placeholder(idx, style)
        protected_parts.append(text[cursor:start])
        protected_parts.append(placeholder)
        protections.append(asdict(TranslationProtection(
            placeholder=placeholder,
            text=value,
            kind=kind,
            start=start,
            end=end,
        )))
        cursor = end
    protected_parts.append((text or "")[cursor:])
    return "".join(protected_parts), protections


def restore_text(text: str, protections: list[dict]) -> str:
    restored = text or ""
    for item in protections:
        placeholder = item.get("placeholder")
        value = item.get("text")
        if placeholder and value is not None:
            restored = restored.replace(placeholder, value)
            restored = _restore_tolerant_placeholder(restored, placeholder, value)
    return restored


def _placeholder_id(placeholder: str) -> str | None:
    match = re.search(r"PT\s*_*\s*(\d{4})", str(placeholder or ""))
    return match.group(1) if match else None


def _tolerant_variants(token_id: str) -> list[str]:
    """All placeholder spellings a model might emit for a given PT id.

    Tolerant across every style (unicode bracket, ascii xml, plain ascii, at
    token) and common corruptions: spaces, single/double/no quotes, missing
    slash, underscores.
    """
    return [
        rf"⟦\s*PT\s*{token_id}\s*⟧",
        rf"<\s*nt\s+id\s*=\s*[\"']?\s*PT\s*{token_id}\s*[\"']?\s*/?\s*>",
        rf"\[\s*\[\s*\[\s*PT\s*{token_id}\s*\]\s*\]\s*\]",
        rf"\[\s*\[\s*PT\s*{token_id}\s*\]\s*\]",
        rf"@\s*@\s*PT\s*{token_id}\s*@\s*@",
        rf"_*\s*PT\s*_*\s*{token_id}\s*_+",
        rf"\bPT\s*{token_id}\b",
    ]


def _restore_tolerant_placeholder(text: str, placeholder: str, value: str) -> str:
    token_id = _placeholder_id(placeholder)
    if not token_id:
        return text
    restored = text
    for pattern in _tolerant_variants(token_id):
        restored = re.sub(pattern, lambda _m: value, restored)
    return restored


def audit_placeholders(text: str, protections: list[dict]) -> dict:
    """Detect placeholder corruption after a translation round-trip.

    Reports placeholders that were lost (value missing), left unrestored (any
    spelling survives), or duplicated (value occurs more than originally).
    """
    text = text or ""
    missing: list[str] = []
    unrestored: list[str] = []
    duplicated: list[str] = []
    for item in protections:
        token_id = _placeholder_id(item.get("placeholder") or "")
        value = item.get("text")
        if value is None:
            continue
        if token_id and any(re.search(pattern, text) for pattern in _tolerant_variants(token_id)):
            unrestored.append(item.get("placeholder"))
            continue
        occurrences = text.count(str(value)) if value else 0
        if occurrences == 0:
            missing.append(value)
        elif occurrences > 1:
            duplicated.append(value)
    corruption_count = len(missing) + len(unrestored) + len(duplicated)
    return {
        "missing": missing,
        "unrestored": unrestored,
        "duplicated": duplicated,
        "placeholder_corruption_count": corruption_count,
    }


def _overlaps(a: range, b: range) -> bool:
    return a.start < b.stop and b.start < a.stop
