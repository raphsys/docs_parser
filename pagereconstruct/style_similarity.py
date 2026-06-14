"""Score how faithfully a resolved style matches the source style (PR-Lot 6).

Weighted over font class, size, bold, italic, colour and alignment. Publication
requires >= 0.95; < 0.85 is a ko.
"""

from __future__ import annotations

from .font_resolver_bridge import infer_font_class

_W = {"font_class": 0.30, "size": 0.25, "bold": 0.15, "italic": 0.10, "color": 0.10, "alignment": 0.10}


def _class(style):
    return style.get("font_class") or infer_font_class(style.get("font_family"), style.get("flags"))


def _color_delta(a, b) -> float:
    def rgb(c):
        s = str(c or "#000000").lstrip("#")
        return tuple(int(s[k:k + 2], 16) for k in (0, 2, 4)) if len(s) == 6 else (0, 0, 0)
    ra, rb = rgb(a), rgb(b)
    return sum(abs(ra[i] - rb[i]) for i in range(3)) / 765.0


def similarity(source_style: dict, resolved_style: dict) -> dict:
    src, res = source_style or {}, resolved_style or {}
    sflags, rflags = src.get("flags") or {}, res.get("flags") or {}
    comp = {}
    sc, rc = _class(src), _class(res)
    comp["font_class"] = 1.0 if (sc == rc or sc in {"unknown", None}) else 0.0
    ss, rs = src.get("font_size_pt"), res.get("font_size_pt")
    if ss and rs:
        comp["size"] = max(0.0, 1.0 - abs(ss - rs) / max(ss, rs))
    else:
        comp["size"] = 0.7
    comp["bold"] = 1.0 if bool(sflags.get("bold")) == bool(rflags.get("bold")) else 0.0
    comp["italic"] = 1.0 if bool(sflags.get("italic")) == bool(rflags.get("italic")) else 0.0
    comp["color"] = max(0.0, 1.0 - _color_delta(src.get("color"), res.get("color")))
    comp["alignment"] = 1.0 if (src.get("alignment") or "left") == (res.get("alignment") or "left") else 0.5
    score = round(sum(_W[k] * comp[k] for k in _W), 3)
    status = "ok" if score >= 0.95 else ("review" if score >= 0.85 else "ko")
    return {"score": score, "status": status, "components": comp}
