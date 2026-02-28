import math
import re
from collections import Counter, defaultdict
from statistics import median


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _hex_to_rgb(color):
    if not isinstance(color, str):
        return None
    c = color.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        return None
    try:
        return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return None


def _rgb_to_hex(rgb):
    r, g, b = [int(_clamp(v, 0, 255)) for v in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def _relative_luminance(rgb):
    def chan(v):
        x = v / 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)


def _contrast_ratio(rgb_a, rgb_b):
    la = _relative_luminance(rgb_a)
    lb = _relative_luminance(rgb_b)
    l1, l2 = (la, lb) if la >= lb else (lb, la)
    return (l1 + 0.05) / (l2 + 0.05)


def _ensure_contrast_on_white(color, min_ratio=4.5):
    rgb = _hex_to_rgb(color) or (0, 0, 0)
    if _contrast_ratio(rgb, (255, 255, 255)) >= min_ratio:
        return _rgb_to_hex(rgb)
    # Keep hue tendency but darken strongly.
    dark = tuple(int(v * 0.42) for v in rgb)
    if _contrast_ratio(dark, (255, 255, 255)) < min_ratio:
        return "#222222"
    return _rgb_to_hex(dark)


def _norm_text(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def _block_text(block):
    if block.get("translated_text"):
        return _norm_text(block.get("translated_text"))
    parts = []
    for line in block.get("lines", []):
        for phrase in line.get("phrases", []):
            t = phrase.get("translated_text") or phrase.get("texte") or ""
            t = _norm_text(t)
            if t:
                parts.append(t)
    return _norm_text(" ".join(parts))


def _iter_spans(block):
    for line in block.get("lines", []):
        for phrase in line.get("phrases", []):
            spans = phrase.get("spans", [])
            if spans:
                for span in spans:
                    yield span
            else:
                yield {"style": phrase.get("style", {})}


def _style_default():
    return {"font": "Arial", "size": 12.0, "color": "#222222", "flags": {"bold": False, "italic": False, "serif": False}}


def _normalize_style(style):
    style = style or {}
    flags = style.get("flags", {}) if isinstance(style.get("flags", {}), dict) else {}
    font = style.get("font") or ("Times-New-Roman" if flags.get("serif") else "Arial")
    size = style.get("size", 12.0)
    try:
        size = float(size)
    except Exception:
        size = 12.0
    size = _clamp(size, 6.0, 72.0)
    color = _ensure_contrast_on_white(style.get("color", "#222222"))
    return {
        "font": str(font),
        "size": float(size),
        "color": color,
        "flags": {
            "bold": bool(flags.get("bold", False)),
            "italic": bool(flags.get("italic", False)),
            "serif": bool(flags.get("serif", False)),
            "underline": bool(flags.get("underline", False)),
            "uppercase": bool(flags.get("uppercase", False)),
        },
    }


def _merge_styles(primary, secondary):
    # Primary has precedence, secondary fills missing details.
    p = _normalize_style(primary)
    s = _normalize_style(secondary)
    flags = dict(s.get("flags", {}))
    flags.update(p.get("flags", {}))
    out = dict(s)
    out.update({k: v for k, v in p.items() if k != "flags"})
    out["flags"] = flags
    return out


def _dominant_style(block):
    styles = []
    for span in _iter_spans(block):
        styles.append(_normalize_style(span.get("style", {})))
    if not styles:
        return _style_default()

    font = Counter(s["font"] for s in styles).most_common(1)[0][0]
    color = Counter(s["color"] for s in styles).most_common(1)[0][0]
    size = float(median(s["size"] for s in styles))
    flags = {
        "bold": Counter(bool(s["flags"].get("bold")) for s in styles).most_common(1)[0][0],
        "italic": Counter(bool(s["flags"].get("italic")) for s in styles).most_common(1)[0][0],
        "serif": Counter(bool(s["flags"].get("serif")) for s in styles).most_common(1)[0][0],
        "underline": Counter(bool(s["flags"].get("underline")) for s in styles).most_common(1)[0][0],
        "uppercase": Counter(bool(s["flags"].get("uppercase")) for s in styles).most_common(1)[0][0],
    }
    return _normalize_style({"font": font, "size": size, "color": color, "flags": flags})


def _semantic_from_block(block, style, page_h):
    role = (block.get("role") or "body").lower()
    txt = _block_text(block)
    words = len(txt.split())
    bbox = block.get("bbox", [0, 0, 0, 0])
    y0 = float(bbox[1]) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else 0.0
    sz = float(style.get("size", 12.0))
    bold = bool(style.get("flags", {}).get("bold"))
    upper = bool(style.get("flags", {}).get("uppercase"))

    if role in {"header", "footer"}:
        return role, None
    if role in {"figure_caption"}:
        return "caption", None
    if role in {"title", "section_heading"}:
        return "heading", None
    if words <= 12 and (bold or upper) and sz >= 11.0 and y0 < page_h * 0.65:
        return "heading", None
    return "body", None


def _quantize_color(color):
    rgb = _hex_to_rgb(color) or (0, 0, 0)
    q = tuple(int(round(v / 32.0) * 32) for v in rgb)
    return _rgb_to_hex(q)


def _cluster_key(semantic_type, style):
    flags = style.get("flags", {})
    size_bucket = int(round(float(style.get("size", 12.0))))
    color_bucket = _quantize_color(style.get("color", "#222222"))
    return (
        semantic_type,
        size_bucket,
        1 if flags.get("bold") else 0,
        1 if flags.get("italic") else 0,
        1 if flags.get("serif") else 0,
        color_bucket,
    )


def _infer_heading_levels(heading_sizes):
    levels = {}
    if not heading_sizes:
        return levels
    unique = sorted(set(int(round(s)) for s in heading_sizes), reverse=True)[:3]
    for idx, size in enumerate(unique):
        levels[size] = idx + 1
    return levels


def _infer_columns(blocks, page_w):
    left, right, crossing = 0, 0, 0
    mid = page_w / 2.0
    for b in blocks:
        bb = b.get("bbox", [0, 0, 0, 0])
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        x0, _, x1, _ = [float(v) for v in bb]
        if x0 < mid < x1:
            crossing += 1
        elif x1 <= mid:
            left += 1
        elif x0 >= mid:
            right += 1
    if left >= 3 and right >= 3 and crossing <= max(2, int(0.25 * (left + right))):
        return {"columns": 2, "gutter_px_estimate": int(page_w * 0.06)}
    return {"columns": 1, "gutter_px_estimate": 0}


def build_page_style_profile(blocks, layout_meta, page_width, page_height):
    annotated = []
    cluster_stats = defaultdict(lambda: {"count": 0, "sizes": [], "colors": [], "fonts": [], "flags": Counter()})
    heading_sizes = []
    all_sizes = []
    all_colors = []
    block_gaps = []

    ordered = sorted(blocks, key=lambda b: ((b.get("bbox") or [0, 0, 0, 0])[1], (b.get("bbox") or [0, 0, 0, 0])[0]))
    prev_bottom = None
    for block in ordered:
        style = _dominant_style(block)
        semantic_type, _ = _semantic_from_block(block, style, page_height)
        if semantic_type == "heading":
            heading_sizes.append(style["size"])
        all_sizes.append(style["size"])
        all_colors.append(style["color"])
        key = _cluster_key(semantic_type, style)

        cs = cluster_stats[key]
        cs["count"] += 1
        cs["sizes"].append(style["size"])
        cs["colors"].append(style["color"])
        cs["fonts"].append(style["font"])
        for fk, fv in style.get("flags", {}).items():
            if fv:
                cs["flags"][fk] += 1

        bb = block.get("bbox", [0, 0, 0, 0])
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            y0, y1 = float(bb[1]), float(bb[3])
            if prev_bottom is not None and y0 >= prev_bottom:
                block_gaps.append(y0 - prev_bottom)
            prev_bottom = y1

        block = dict(block)
        block["_style_work"] = {"style": style, "semantic_type": semantic_type, "cluster_key": key}
        annotated.append(block)

    heading_levels = _infer_heading_levels(heading_sizes)
    # Stable class labels by frequency.
    ranked = sorted(cluster_stats.items(), key=lambda kv: kv[1]["count"], reverse=True)
    class_by_key = {}
    component_styles = {}
    for idx, (key, stats) in enumerate(ranked, start=1):
        semantic_type = key[0]
        style = {
            "font": Counter(stats["fonts"]).most_common(1)[0][0] if stats["fonts"] else "Arial",
            "size": float(median(stats["sizes"])) if stats["sizes"] else 12.0,
            "color": Counter(stats["colors"]).most_common(1)[0][0] if stats["colors"] else "#222222",
            "flags": {name: bool(cnt >= max(1, stats["count"] * 0.4)) for name, cnt in stats["flags"].items()},
        }
        heading_level = None
        if semantic_type == "heading":
            heading_level = heading_levels.get(int(round(style["size"])), 3)
            class_name = f"Heading{heading_level}"
        elif semantic_type == "caption":
            class_name = "Caption"
        elif semantic_type == "header":
            class_name = "Header"
        elif semantic_type == "footer":
            class_name = "Footer"
        else:
            class_name = "Body"
        class_id = f"{class_name}_{idx:02d}"
        class_by_key[key] = class_id
        component_styles[class_id] = {
            "class_name": class_name,
            "semantic_type": semantic_type,
            "heading_level": heading_level,
            "style": _normalize_style(style),
            "occurrences": stats["count"],
        }

    for block in annotated:
        work = block.pop("_style_work")
        style = work["style"]
        semantic_type = work["semantic_type"]
        class_id = class_by_key.get(work["cluster_key"])
        class_style = component_styles.get(class_id, {}).get("style", _style_default())
        confidence = 0.9
        if style.get("font") in {"Arial", "Times-New-Roman"} and not style.get("flags", {}).get("serif"):
            confidence -= 0.1
        if not style.get("color"):
            confidence -= 0.1
        confidence = float(_clamp(confidence, 0.4, 0.95))

        resolved = _merge_styles(class_style, style)
        if semantic_type == "caption":
            resolved["size"] = min(resolved["size"], max(8.0, resolved["size"] - 1.0))
        if semantic_type == "heading":
            resolved["flags"]["bold"] = True

        semantic = {"type": semantic_type}
        if semantic_type == "heading":
            sz = int(round(resolved["size"]))
            semantic["heading_level"] = heading_levels.get(sz, component_styles.get(class_id, {}).get("heading_level", 3))

        block["style_class"] = class_id
        block["style_confidence"] = round(confidence, 3)
        block["semantic"] = semantic
        block["resolved_style"] = resolved

        # Propagate harmonized style to spans in low-confidence/ambiguous cases.
        for line in block.get("lines", []):
            for phrase in line.get("phrases", []):
                spans = phrase.get("spans", [])
                if not spans:
                    continue
                for span in spans:
                    sp_style = _normalize_style(span.get("style", {}))
                    if confidence < 0.6:
                        span["style"] = dict(resolved)
                    else:
                        span["style"] = _merge_styles(sp_style, resolved)

    typography = sorted({int(round(s)) for s in all_sizes if s > 0}, reverse=True)
    color_palette = [c for c, _ in Counter(all_colors).most_common(8)]
    spacing_scale = {
        "block_gap_px": round(float(median(block_gaps)), 2) if block_gaps else 0.0,
        "line_gap_px": 0.0,  # parser-level line spacing varies by extractor; keep neutral when absent.
    }
    profile = {
        "version": "v1_style_profile",
        "palette": {"text_colors": color_palette},
        "typography_scale": typography,
        "spacing_scale": spacing_scale,
        "margins": (layout_meta or {}).get("margins", {}),
        "page_grid": _infer_columns(annotated, float(page_width)),
        "component_styles": component_styles,
        "fallback_rules": {
            "heading_level_policy": "prefer_lower_when_ambiguous",
            "font_fallback": "preserve_serif_category",
            "color_fallback": "ensure_wcag_aa_on_white",
            "layout_fallback": "single_column_readable",
        },
    }
    return annotated, profile
