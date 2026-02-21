from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class ComposeOptions:
    enable_hyphenation: bool = False
    max_font_shrink: float = 2.0
    min_tracking: float = 0.0
    max_tracking: float = 0.0
    min_word_spacing: float = 0.0
    max_word_spacing: float = 0.0
    step_pt: float = 0.25
    min_font_pt: float = 7.0


class TextComposer:
    """
    Lightweight paragraph-in-box composer with fit loop.
    """

    def __init__(self):
        try:
            import pyphen  # optional
            self._hyphenators = {"fr": pyphen.Pyphen(lang="fr")}
        except Exception:
            self._hyphenators = {}

    def compose_text_in_box(
        self,
        text: str,
        box_w: float,
        box_h: float,
        base_font_pt: float,
        line_height_factor: float,
        measure_fn: Callable[[str, float], float],
        alignment: str = "left",
        lang: str = "fr",
        options: Optional[ComposeOptions] = None,
    ) -> Dict:
        opts = options or ComposeOptions()
        words = [w for w in (text or "").split() if w]
        if not words:
            return {"font_size": base_font_pt, "line_height": base_font_pt * line_height_factor, "lines": [], "overflow": ""}

        fs = float(base_font_pt)
        while fs >= max(opts.min_font_pt, base_font_pt - opts.max_font_shrink):
            lines = self._wrap_words(words, box_w, fs, measure_fn, bool(opts.enable_hyphenation), lang)
            line_h = max(1.0, fs * line_height_factor)
            max_lines = max(1, int(box_h / line_h))
            if len(lines) <= max_lines:
                return {
                    "font_size": fs,
                    "line_height": line_h,
                    "lines": lines,
                    "overflow": "",
                    "alignment": alignment,
                }
            fs -= max(0.05, opts.step_pt)

        # best effort with truncation
        fs = max(opts.min_font_pt, base_font_pt - opts.max_font_shrink)
        lines = self._wrap_words(words, box_w, fs, measure_fn, bool(opts.enable_hyphenation), lang)
        line_h = max(1.0, fs * line_height_factor)
        max_lines = max(1, int(box_h / line_h))
        kept = lines[:max_lines]
        overflow = " ".join(lines[max_lines:]).strip()
        return {"font_size": fs, "line_height": line_h, "lines": kept, "overflow": overflow, "alignment": alignment}

    def _wrap_words(self, words: List[str], max_w: float, fs: float, measure_fn: Callable[[str, float], float], hyphenate: bool, lang: str) -> List[str]:
        out = []
        cur = ""
        i = 0
        while i < len(words):
            w = words[i]
            if not cur:
                if measure_fn(w, fs) <= max_w:
                    cur = w
                    i += 1
                    continue
                head, tail = self._split_word(w, max_w, fs, measure_fn, hyphenate, lang)
                cur = head
                words[i] = tail if tail else ""
                if not tail:
                    i += 1
                out.append(cur)
                cur = ""
                continue

            cand = f"{cur} {w}"
            if measure_fn(cand, fs) <= max_w:
                cur = cand
                i += 1
            else:
                out.append(cur)
                cur = ""
        if cur:
            out.append(cur)
        return [ln for ln in out if ln]

    def _split_word(self, word: str, max_w: float, fs: float, measure_fn: Callable[[str, float], float], hyphenate: bool, lang: str):
        if hyphenate and lang.startswith("fr") and "fr" in self._hyphenators:
            h = self._hyphenators["fr"]
            parts = h.inserted(word).split("-")
            if len(parts) > 1:
                acc = ""
                used = 0
                for p in parts:
                    cand = (acc + p) if not acc else (acc + p)
                    trial = cand + "-"
                    if measure_fn(trial, fs) <= max_w:
                        acc = cand
                        used += 1
                    else:
                        break
                if used > 0 and used < len(parts):
                    head = "".join(parts[:used]) + "-"
                    tail = "".join(parts[used:])
                    return head, tail

        chunk = ""
        idx = 0
        for ch in word:
            cand = chunk + ch
            if chunk and measure_fn(cand, fs) > max_w:
                break
            chunk = cand
            idx += 1
        if not chunk:
            chunk = word[:1]
            idx = 1
        return chunk, word[idx:]
