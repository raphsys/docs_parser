"""Infer the real font class from PDF subset font names (directive Lot 5/A2).

PDF subset names are often mangled ("FrctghDrdrdhXjdpbgTimes-"), and the
extracted ``serif`` flag is frequently wrong (absent -> False). We infer
serif / sans / mono from the family name so Times/Janson/Baskerville no longer
render as sans-serif.
"""

from __future__ import annotations

import re

_SERIF = ("times", "janson", "baskerville", "garamond", "minion", "georgia",
          "serif", "roman", "palatino", "caslon", "bembo", "didot", "century")
_MONO = ("courier", "mono", "consolas", "menlo", "ubuntumono", "inconsolata", "typewriter")
_SANS = ("arial", "helvetica", "franklin", "tradegothic", "calibri", "verdana",
         "tahoma", "segoe", "gothic", "frutiger", "univers", "futura", "sans")


def normalize_font_family(raw: str | None) -> str | None:
    if not raw:
        return None
    name = str(raw)
    # drop subset prefix "ABCDEF+Name"
    if "+" in name:
        name = name.split("+", 1)[1]
    return name.strip().strip("-") or None


def infer_font_class(raw: str | None, flags: dict | None = None) -> str:
    flags = flags or {}
    if flags.get("monospace"):
        return "mono"
    name = (normalize_font_family(raw) or "").lower()
    if any(k in name for k in _MONO):
        return "mono"
    if any(k in name for k in _SERIF):
        return "serif"
    if any(k in name for k in _SANS):
        return "sans"
    # Absence of a serif flag must NOT force sans (directive). Unknown -> serif
    # is a safer default for book pages; flag it via confidence elsewhere.
    if flags.get("serif"):
        return "serif"
    return "unknown"


def apply_font_class(style: dict) -> dict:
    """Set flags.serif/monospace from the inferred class (in place, returns style)."""
    fam = style.get("font_family")
    flags = style.setdefault("flags", {})
    cls = infer_font_class(fam, flags)
    style["font_family_normalized"] = normalize_font_family(fam)
    style["font_class"] = cls
    if cls == "mono":
        flags["monospace"] = True
        flags["serif"] = False
    elif cls == "serif":
        flags["serif"] = True
        flags["monospace"] = False
    elif cls == "sans":
        flags["serif"] = False
        flags["monospace"] = False
    # unknown -> leave flags as-is
    return style
