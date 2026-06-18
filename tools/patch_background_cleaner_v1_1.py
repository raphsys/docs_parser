#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
old = s

s = s.replace(
    "clean background\nmeans page substrate only: all visible source content is removed from the\nbackground, including formulas, figures, diagrams, page numbers, captions and\npreserved-exact text.",
    "clean background means visual substrate: all text and text-like special content is removed, while non-text visual material may remain."
)
s = s.replace(
    "clean background\nmeans page substrate only: all visible source content is removed from the\nbackground, including formulas, figures, diagrams, page numbers, captions and\npreserved-exact text.",
    "clean background means visual substrate: all text and text-like special content is removed, while non-text visual material may remain."
)
s = s.replace("No visible source content is protected from clean background removal.", "Non-text visual content may remain in clean background; text/formula/code cannot.")
s = s.replace("cleanbg is page substrate only.", "cleanbg is visual substrate without text/formula/code.")

if "collect_background_purity_boxes" not in s:
    s = s.replace(
        "from pipelines.background_cover import build_deterministic_text_cover_background",
        "from pipelines.background_cover import build_deterministic_text_cover_background, collect_background_purity_boxes",
        1,
    )

if "def _text_regions_px(" in s and "purity_boxes_pt = collect_background_purity_boxes(input_data)" not in s:
    needle = '''    units = input_data.get("units") or []
    protected_boxes_pt = protected_boxes_pt or []
    by_id = {u.get("unit_id"): u for u in units if isinstance(u, dict) and u.get("unit_id")}
    candidate_boxes: list[list[int]] = []
'''
    repl = '''    units = input_data.get("units") or []
    protected_boxes_pt = protected_boxes_pt or []
    by_id = {u.get("unit_id"): u for u in units if isinstance(u, dict) and u.get("unit_id")}
    candidate_boxes: list[list[int]] = []

    try:
        purity_boxes_pt = collect_background_purity_boxes(input_data)
    except Exception:
        purity_boxes_pt = []
    if purity_boxes_pt:
        for bpt in purity_boxes_pt:
            x0, y0, x1, y1 = (bpt[0] * sx, bpt[1] * sy, bpt[2] * sx, bpt[3] * sy)
            if x1 > x0 and y1 > y0:
                candidate_boxes.append(_pad_px([int(x0), int(y0), int(x1), int(y1)]))
        return _merge_px_boxes(candidate_boxes)
'''
    if needle in s:
        s = s.replace(needle, repl, 1)

if s != old:
    p.write_text(s, encoding="utf-8")
    print(f"corrigé: {p}")
else:
    print(f"aucune modification nécessaire: {p}")
