"""Pre-translation text cleanup for PAGEPRINT.

Two problems hurt the translation engine far more than the engine itself:

1. Intra-text hyphenation: ``improve the unsu-\npervised`` inside one segment.
2. Cross-segment hyphenation: one segment ends with ``... the unsu-`` and the
   next starts with ``pervised ...`` (frequent in dense tables / multi-line
   cells). Sent as-is, the model hallucinates on the truncated fragment.

The repairs are conservative: a hyphen is only treated as a line-break artefact
when the continuation starts lowercase and the left stem is not a known lexical
prefix (``state-of-the-art``, ``semi-supervised``, ``VGG-16``, ``F-score`` …).
"""

from __future__ import annotations

import re


# Prefixes that legitimately keep their hyphen (do not de-hyphenate).
LEXICAL_HYPHEN_PREFIXES = {
    "state", "semi", "end", "non", "self", "multi", "pre", "post", "co", "well",
    "high", "low", "real", "long", "short", "open", "deep", "large", "fine",
    "hard", "cross", "sub", "anti", "re", "inter", "intra", "well", "off", "on",
    "in", "out", "up", "down", "over", "under", "mid", "ex", "all",
}

# left stem + "-" + REQUIRED whitespace (line break) + lowercase continuation.
# Requiring whitespace avoids touching lexical compounds like state-of-the-art.
_INTRA_HYPHEN_RE = re.compile(r"([A-Za-zÀ-ÿ]{2,})-(?:[ \t]*\n[ \t]*|[ \t]+)([a-zà-ÿ]{2,})")
# trailing line-break hyphen at end of a fragment
_TRAILING_HYPHEN_RE = re.compile(r"([A-Za-zÀ-ÿ]{2,})-\s*$")


def _should_join(left: str, right: str) -> bool:
    if len(left) < 2 or len(right) < 2:
        return False
    if not right[0].islower():
        return False
    if left.lower() in LEXICAL_HYPHEN_PREFIXES:
        return False
    if any(ch.isdigit() for ch in right):
        return False
    return True


def repair_hyphenation(text: str) -> str:
    """Collapse intra-text line-break hyphenation (``unsu-\\npervised`` -> ``unsupervised``)."""
    if not text or "-" not in text:
        return text

    def _replace(match: re.Match) -> str:
        left, right = match.group(1), match.group(2)
        if _should_join(left, right):
            return left + right
        return match.group(0)

    return _INTRA_HYPHEN_RE.sub(_replace, text)


def ends_with_break_hyphen(text: str) -> str | None:
    """Return the trailing stem if ``text`` ends with a line-break hyphen, else None."""
    if not text:
        return None
    match = _TRAILING_HYPHEN_RE.search(text.rstrip())
    if not match:
        return None
    stem = match.group(1)
    if stem.lower() in LEXICAL_HYPHEN_PREFIXES or len(stem) < 2:
        return None
    return stem


def join_across_break(left: str, right: str) -> str:
    """Join two fragments where ``left`` ends with a break hyphen."""
    left = left.rstrip()
    right = right.lstrip()
    stem_match = _TRAILING_HYPHEN_RE.search(left)
    if not stem_match:
        return f"{left} {right}".strip()
    first_word = re.match(r"[a-zà-ÿ]+", right)
    if first_word and _should_join(stem_match.group(1), first_word.group(0)):
        # drop the hyphen, glue stem + continuation, keep the rest of the right side
        joined = left[: stem_match.start(1)] + stem_match.group(1) + right
        return joined.strip()
    return f"{left} {right}".strip()


def merge_hyphenated_segments(segments: list[dict]) -> list[dict]:
    """Merge consecutive translatable segments split by a line-break hyphen.

    Two segments are merged when the first ends with a break hyphen, both are in
    ``translate`` mode and share a compatible role. ``source_unit_ids`` are
    concatenated so reconstruction keeps every origin.
    """
    if not segments:
        return segments
    merged: list[dict] = []
    for seg in segments:
        if merged:
            prev = merged[-1]
            prev_text = str(prev.get("source_text") or "")
            if (
                ends_with_break_hyphen(prev_text)
                and prev.get("translation_mode", "translate") == "translate"
                and seg.get("translation_mode", "translate") == "translate"
                and _compatible_roles(prev.get("role"), seg.get("role"))
            ):
                joined = join_across_break(prev_text, str(seg.get("source_text") or ""))
                prev["source_text"] = joined
                prev["text"] = joined
                prev["source_unit_ids"] = list(dict.fromkeys(
                    [*(prev.get("source_unit_ids") or []), *(seg.get("source_unit_ids") or [])]
                ))
                prev.setdefault("normalization_applied", [])
                if "dehyphenation_merge" not in prev["normalization_applied"]:
                    prev["normalization_applied"].append("dehyphenation_merge")
                continue
        merged.append(seg)
    return merged


def _compatible_roles(a, b) -> bool:
    a = str(a or "")
    b = str(b or "")
    if a == b:
        return True
    body_like = {"body_paragraph", "table_body_cell", "list_item", "index_subentry"}
    return a in body_like and b in body_like


# --- helpers reused by translation QA -----------------------------------------
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]{2,}")


def looks_like_truncated_fragment(text: str) -> bool:
    """Source fragment unlikely to translate cleanly (ends hyphen / very short)."""
    s = str(text or "").strip()
    if not s:
        return False
    if ends_with_break_hyphen(s):
        return True
    words = _WORD_RE.findall(s)
    if len(words) <= 1 and len(s) < 4:
        return True
    return False


def has_repeated_ngram(text: str, *, n: int = 3, min_repeats: int = 2) -> bool:
    """Detect degenerate repetition like 'les défis et les défis et les défis'."""
    words = str(text or "").lower().split()
    if len(words) < n + 1:
        return False
    counts: dict[tuple, int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= min_repeats:
            return True
    return False
