import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

try:
    import fitz
except Exception:
    fitz = None

FONT_EXTENSIONS = (".ttf", ".otf", ".cff", ".cid")
DEFAULT_FONT_DIRS = (
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
    os.path.join(tempfile.gettempdir(), "docs_parser_embedded_fonts"),
)


class FontResolver:
    """Resolve original font hints to local font files or builtin PDF fonts."""

    def __init__(self, font_dirs: Optional[Tuple[str, ...]] = None):
        self.font_dirs = font_dirs or DEFAULT_FONT_DIRS
        self._by_name: Dict[str, List[str]] = {}
        self._glyph_support_cache: Dict[Tuple[str, str], bool] = {}
        self._discover_local_fonts()

    def _discover_local_fonts(self) -> None:
        for root in self.font_dirs:
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    if not filename.lower().endswith(FONT_EXTENSIONS):
                        continue
                    full_path = os.path.join(dirpath, filename)
                    keys = self._font_keys_from_filename(filename)
                    for key in keys:
                        self._by_name.setdefault(key, []).append(full_path)

    def _font_keys_from_filename(self, filename: str) -> List[str]:
        basename = os.path.splitext(filename)[0]
        keys = {self._normalize_name(basename)}
        for sep in ("-", "_"):
            if sep in basename:
                keys.add(self._normalize_name(basename.split(sep)[0]))
        return [k for k in keys if k]

    def _normalize_name(self, name: str) -> str:
        # Remove PDF subset prefixes like "ABCDEE+Calibri-Bold"
        clean = name.split("+", 1)[-1]
        clean = clean.lower()
        clean = re.sub(r"[^a-z0-9]+", "", clean)
        return clean

    def _strip_style_tokens(self, font_name: str) -> str:
        raw = font_name.split("+", 1)[-1]
        raw = re.sub(
            r"(bold|italic|itali|ital|regular|medium|semibold|demibold|light|black|condensed|cond|narrow|oblique|obl)",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        return self._normalize_name(raw)

    def _normalized_name_variants(self, name: str) -> List[str]:
        key = self._normalize_name(name)
        if not key:
            return []
        variants = {key}
        suffix_map = {
            "itali": "italic",
            "ital": "italic",
            "obl": "oblique",
            "cond": "condensed",
        }
        changed = True
        while changed:
            changed = False
            current = list(variants)
            for value in current:
                for short_suffix, full_suffix in suffix_map.items():
                    if value.endswith(short_suffix) and not value.endswith(full_suffix):
                        candidate = value[: -len(short_suffix)] + full_suffix
                        if candidate not in variants:
                            variants.add(candidate)
                            changed = True
        compact = re.sub(r"([a-z0-9])(?:[a-f0-9]{6,}|[a-z]{2,}\d{2,})$", r"\1", key, flags=re.IGNORECASE)
        if compact and len(compact) >= max(8, len(key) - 12):
            variants.add(compact)
        return [variant for variant in variants if variant]

    def _find_prefix_key_match(self, candidate_keys: List[str], flags: Dict) -> Optional[str]:
        matches = []
        for candidate in candidate_keys:
            if not candidate:
                continue
            for mapped_key, paths in self._by_name.items():
                if not mapped_key or not paths:
                    continue
                if mapped_key.startswith(candidate) or candidate.startswith(mapped_key):
                    matches.append((abs(len(mapped_key) - len(candidate)), mapped_key, paths))
        if not matches:
            return None
        matches.sort(key=lambda item: item[0])
        return self._pick_best_font_file(matches[0][2], flags)

    def _score_font_path(self, path: str, want_bold: bool, want_italic: bool) -> int:
        n = os.path.basename(path).lower()
        score = 0
        has_bold = "bold" in n or "demi" in n or "semibold" in n
        has_italic = "italic" in n or "oblique" in n
        ext = os.path.splitext(n)[1]
        if has_bold == want_bold:
            score += 2
        if has_italic == want_italic:
            score += 2
        if ext in {".ttf", ".otf"}:
            score += 2
        return score

    def _pick_best_font_file(self, paths: List[str], flags: Dict) -> str:
        want_bold = bool(flags.get("bold"))
        want_italic = bool(flags.get("italic"))
        return max(paths, key=lambda p: self._score_font_path(p, want_bold, want_italic))

    def _find_alias_match(self, font_name: str, flags: Dict) -> Optional[str]:
        key = self._normalize_name(font_name)
        aliases = []

        if any(x in key for x in ("arial", "helvetica", "sans")):
            aliases = [
                "liberationsans",
                "dejavusans",
                "nimbussans",
                "freesans",
                "notosans",
            ]
        elif any(x in key for x in ("times", "serif", "garamond", "georgia")):
            aliases = [
                "liberationserif",
                "dejavuserif",
                "nimbusroman",
                "freeserif",
                "notoserif",
            ]
        elif any(x in key for x in ("courier", "mono", "consolas", "menlo")):
            aliases = [
                "liberationmono",
                "dejavusansmono",
                "freemono",
                "notosansmono",
            ]
        elif any(x in key for x in ("calibri", "cambria", "candara")):
            aliases = ["carlito", "caladea", "liberationsans", "dejavusans"]

        for alias in aliases:
            if alias in self._by_name:
                return self._pick_best_font_file(self._by_name[alias], flags)
        return None

    def _builtin_font(self, flags: Dict) -> str:
        serif = bool(flags.get("serif"))
        mono = bool(flags.get("monospace"))
        bold = bool(flags.get("bold"))
        italic = bool(flags.get("italic"))

        if mono:
            if bold and italic:
                return "cobi"
            if bold:
                return "cobo"
            if italic:
                return "coit"
            return "cour"

        if serif:
            if bold and italic:
                return "tibi"
            if bold:
                return "tibo"
            if italic:
                return "tiit"
            return "tiro"

        if bold and italic:
            return "hebi"
        if bold:
            return "hebo"
        if italic:
            return "heit"
        return "helv"

    def _normalize_text_for_support_check(self, text: str) -> str:
        if not text:
            return ""
        chars = []
        for ch in text:
            if ch.isspace():
                continue
            if ord(ch) < 128:
                continue
            chars.append(ch)
        return "".join(sorted(set(chars)))

    def _font_supports_text(self, fontfile: Optional[str], builtin: Optional[str], text: str) -> bool:
        probe = self._normalize_text_for_support_check(text or "")
        if not probe or fitz is None:
            return True
        cache_key = (str(fontfile or builtin or ""), probe)
        cached = self._glyph_support_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            if fontfile:
                font_obj = fitz.Font(fontfile=fontfile)
            elif builtin:
                font_obj = fitz.Font(fontname=builtin)
            else:
                self._glyph_support_cache[cache_key] = False
                return False
            ok = all(bool(font_obj.has_glyph(ord(ch))) for ch in probe)
        except Exception:
            ok = False
        self._glyph_support_cache[cache_key] = ok
        return ok

    def _family_unicode_preferences(self, font_name: str, flags: Dict) -> List[str]:
        key = self._normalize_name(font_name or "")
        mono = bool(flags.get("monospace")) or any(x in key for x in ("courier", "mono", "consolas", "menlo", "ubuntu"))
        serif = bool(flags.get("serif")) or any(
            x in key for x in ("times", "serif", "roman", "baskerville", "garamond", "janson", "georgia")
        )
        if mono:
            return [
                "liberationmono",
                "dejavusansmono",
                "notosansmono",
                "notomono",
                "freemono",
                "nimbusmonops",
            ]
        if serif:
            return [
                "liberationserif",
                "dejavuserif",
                "freeserif",
                "nimbusroman",
                "notoserif",
            ]
        return [
            "liberationsans",
            "dejavusans",
            "freesans",
            "nimbussans",
            "notosans",
        ]

    def _find_unicode_safe_fallback(self, font_name: str, flags: Dict, text: str) -> Optional[str]:
        seen = set()
        for family in self._family_unicode_preferences(font_name, flags):
            for key, paths in self._by_name.items():
                if family not in key or not paths:
                    continue
                path = self._pick_best_font_file(paths, flags)
                if not path or path in seen:
                    continue
                seen.add(path)
                if self._font_supports_text(path, None, text):
                    return path
        return None

    def _is_temp_embedded_subset_path(self, path: Optional[str]) -> bool:
        if not path:
            return False
        try:
            target = os.path.realpath(path)
            cache_root = os.path.realpath(os.path.join(tempfile.gettempdir(), "docs_parser_embedded_fonts"))
            return target.startswith(cache_root + os.sep) or target == cache_root
        except Exception:
            return False

    def resolve(self, style: Dict, text: str = "") -> Dict[str, Optional[str]]:
        font_name = style.get("font", "") or ""
        font_key_normalized = style.get("font_key_normalized", "") or ""
        flags = style.get("flags", {}) or {}
        embedded_font_path = str(style.get("embedded_font_path") or "").strip()

        def _finalize(fontfile: Optional[str], builtin: Optional[str]) -> Dict[str, Optional[str]]:
            if text and fontfile and self._is_temp_embedded_subset_path(fontfile) and not embedded_font_path:
                alias_file = self._find_alias_match(font_name, flags) if font_name else None
                if alias_file and self._font_supports_text(alias_file, None, text):
                    return {"fontfile": alias_file, "builtin": None}
                unicode_file = self._find_unicode_safe_fallback(font_name, flags, text)
                if unicode_file:
                    return {"fontfile": unicode_file, "builtin": None}
            if self._font_supports_text(fontfile, builtin, text):
                return {"fontfile": fontfile, "builtin": builtin}
            alias_file = self._find_alias_match(font_name, flags) if font_name else None
            if alias_file and self._font_supports_text(alias_file, None, text):
                return {"fontfile": alias_file, "builtin": None}
            unicode_file = self._find_unicode_safe_fallback(font_name, flags, text)
            if unicode_file:
                return {"fontfile": unicode_file, "builtin": None}
            builtin_fallback = self._builtin_font(flags)
            return {"fontfile": None, "builtin": builtin_fallback}

        if embedded_font_path and os.path.isfile(embedded_font_path):
            return _finalize(embedded_font_path, None)

        for key in self._normalized_name_variants(font_key_normalized):
            if key in self._by_name:
                return _finalize(self._pick_best_font_file(self._by_name[key], flags), None)
        prefix_key_match = self._find_prefix_key_match(self._normalized_name_variants(font_key_normalized), flags)
        if prefix_key_match:
            return _finalize(prefix_key_match, None)

        if font_name:
            exact_keys = self._normalized_name_variants(font_name)
            for exact_key in exact_keys:
                if exact_key in self._by_name:
                    return _finalize(self._pick_best_font_file(self._by_name[exact_key], flags), None)
            prefix_match = self._find_prefix_key_match(exact_keys, flags)
            if prefix_match:
                return _finalize(prefix_match, None)

            base_key = self._strip_style_tokens(font_name)
            if base_key in self._by_name:
                return _finalize(self._pick_best_font_file(self._by_name[base_key], flags), None)

            alias_file = self._find_alias_match(font_name, flags)
            if alias_file:
                return _finalize(alias_file, None)

        return _finalize(None, self._builtin_font(flags))
