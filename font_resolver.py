import os
import re
from typing import Dict, List, Optional, Tuple


FONT_EXTENSIONS = (".ttf", ".otf")
DEFAULT_FONT_DIRS = (
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
)


class FontResolver:
    """Resolve original font hints to local font files or builtin PDF fonts."""

    def __init__(self, font_dirs: Optional[Tuple[str, ...]] = None):
        self.font_dirs = font_dirs or DEFAULT_FONT_DIRS
        self._by_name: Dict[str, List[str]] = {}
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
            r"(bold|italic|regular|medium|semibold|demibold|light|black|condensed|narrow)",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        return self._normalize_name(raw)

    def _score_font_path(self, path: str, want_bold: bool, want_italic: bool) -> int:
        n = os.path.basename(path).lower()
        score = 0
        has_bold = "bold" in n or "demi" in n or "semibold" in n
        has_italic = "italic" in n or "oblique" in n
        if has_bold == want_bold:
            score += 2
        if has_italic == want_italic:
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

    def resolve(self, style: Dict) -> Dict[str, Optional[str]]:
        font_name = style.get("font", "") or ""
        flags = style.get("flags", {}) or {}

        if font_name:
            exact_key = self._normalize_name(font_name)
            if exact_key in self._by_name:
                return {"fontfile": self._pick_best_font_file(self._by_name[exact_key], flags), "builtin": None}

            base_key = self._strip_style_tokens(font_name)
            if base_key in self._by_name:
                return {"fontfile": self._pick_best_font_file(self._by_name[base_key], flags), "builtin": None}

            alias_file = self._find_alias_match(font_name, flags)
            if alias_file:
                return {"fontfile": alias_file, "builtin": None}

        return {"fontfile": None, "builtin": self._builtin_font(flags)}
