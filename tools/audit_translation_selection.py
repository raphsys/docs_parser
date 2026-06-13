#!/usr/bin/env python3
"""Audit the QUALITY of PAGEPRINT's translation selection.

For a page's PAGEPRINT output, this tool answers three questions per element:

1. What was extracted (the chosen granularity: phrase > expression > word >
   abbreviation, mutually exclusive)?
2. What type/class is it (role / object_type / semantic_kind)?
3. Translatable or not — and why?

The goal is to judge whether PAGEPRINT's output is clean enough to enter
translation: ideally whole phrases reach the engine, not isolated words,
fragments or abbreviations; and natural text is not wrongly marked
non-translatable.

Input: PAGEPRINT input_data JSON file(s) or a directory of them.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pageprint.semantic_builder import _select_translation_source_units

WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+")
SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)")
ABBREV_RE = re.compile(r"^[A-Za-z]*[A-Z][A-Za-z0-9]*$")  # has an uppercase, no spaces
FUNCTION_WORD_RE = re.compile(r"\b(the|and|or|to|of|in|for|with|that|is|are|was|were|be|by|as|it|we|you)\b", re.IGNORECASE)


def classify_granularity(text: str) -> str:
    """phrase > expression > word > abbreviation (mutually exclusive)."""
    s = str(text or "").strip()
    if not s:
        return "empty"
    tokens = s.split()
    words = WORD_RE.findall(s)
    if len(tokens) >= 2:
        if SENTENCE_END_RE.search(s) or (len(words) >= 5 and FUNCTION_WORD_RE.search(s)):
            return "phrase"
        return "expression"
    # single token
    token = tokens[0]
    core = token.strip(".,;:()[]")
    if len(core) <= 12 and not core.isdigit() and ABBREV_RE.match(core) and core.upper() == core or (
        len(core) <= 8 and any(c.isupper() for c in core) and any(c.isupper() for c in core[1:])
    ):
        return "abbreviation"
    if WORD_RE.fullmatch(core):
        return "word"
    return "abbreviation" if any(c.isupper() for c in core) else "word"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def audit_page(input_data: dict) -> dict:
    views = input_data.get("views") or {}
    plan = views.get("translation_plan") or []
    page_role = (input_data.get("page") or {}).get("page_role") or (input_data.get("page_intelligence") or {}).get("page_role")

    translatable = []
    for entry in plan:
        text = entry.get("source_text") or entry.get("text") or ""
        translatable.append({
            "text": text,
            "granularity": classify_granularity(text),
            "role": entry.get("role"),
            "object_type": entry.get("object_type"),
            "semantic_kind": entry.get("semantic_kind"),
            "level": entry.get("level"),
            "translation_mode": entry.get("translation_mode"),
            "translatable": entry.get("translation_mode") == "translate",
            "reason": entry.get("reason_included") or "in_translation_plan",
            "bbox": entry.get("bbox") or (entry.get("render_target") or {}).get("bbox"),
        })
    plan_texts = {_norm(t["text"]) for t in translatable}

    # Non-translatable: leaf textual units (one granularity per branch) excluded.
    non_translatable = []
    for unit in _select_translation_source_units(input_data.get("units") or []):
        policy = unit.get("policy") or {}
        text = (unit.get("content") or {}).get("text") or ""
        if not text.strip():
            continue
        if policy.get("translatable") is not False:
            continue
        if _norm(text) in plan_texts:
            continue
        non_translatable.append({
            "text": text,
            "granularity": classify_granularity(text),
            "role": (unit.get("understanding") or {}).get("role"),
            "object_type": (unit.get("understanding") or {}).get("object_type"),
            "level": unit.get("level"),
            "translatable": False,
            "reason": policy.get("non_translatable_reason") or policy.get("policy_source") or "non_translatable",
            "bbox": (unit.get("geometry") or {}).get("bbox"),
        })

    return {
        "page_role": page_role,
        "translatable": translatable,
        "non_translatable": non_translatable,
        "structural": _structural(input_data),
        "summary": _summary(translatable, non_translatable),
    }


def _structural(input_data: dict) -> dict:
    """Inventory of non-textual / structural elements from PAGEPRINT."""
    ls = input_data.get("logical_structures") or {}
    regions = input_data.get("regions") or []
    region_types = collections.Counter()
    region_boxes: dict[str, list] = {}
    for r in regions:
        t = r.get("region_type") or r.get("type") or "unknown"
        region_types[t] += 1
        region_boxes.setdefault(t, []).append({"bbox": r.get("bbox"), "confidence": r.get("confidence")})
    vl = input_data.get("visual_layers") or {}
    assets = input_data.get("assets") or {}
    return {
        "formulas": [
            {"text": f.get("text"), "bbox": f.get("bbox"), "preservation_mode": f.get("preservation_mode")}
            for f in ls.get("formula_units") or []
        ],
        "code_blocks": [
            {"text": c.get("text"), "bbox": c.get("bbox")} for c in ls.get("code_blocks") or []
        ],
        "tables": [
            {
                "table_id": t.get("table_id") or t.get("logical_unit_id"),
                "columns": t.get("columns"),
                "rows": len(t.get("rows") or []) if isinstance(t.get("rows"), list) else t.get("rows"),
                "cells": len(t.get("cells") or []),
                "bbox": t.get("bbox"),
            }
            for t in ls.get("tables") or []
        ],
        "figures": [
            {"bbox": f.get("bbox"), "diagram_labels": len(f.get("diagram_labels") or [])}
            for f in ls.get("figures") or []
        ],
        "regions": dict(region_types),
        "region_boxes": region_boxes,
        "background": {
            "has_background_layer": bool(vl.get("background") or assets.get("background")),
            "background_path": assets.get("background_path"),
            "masks": len(vl.get("masks") or []),
            "overlays": len(vl.get("overlays") or []),
            "immutable_overlays": len(assets.get("immutable_overlays") or []),
            "source_image_path": assets.get("source_image_path"),
        },
    }


def _summary(translatable: list[dict], non_translatable: list[dict]) -> dict:
    def by(key_items, attr):
        out = {}
        for it in key_items:
            out[it.get(attr) or "none"] = out.get(it.get(attr) or "none", 0) + 1
        return out

    gran_t = by(translatable, "granularity")
    n_t = len(translatable)
    phrase_ratio = round((gran_t.get("phrase", 0)) / n_t, 3) if n_t else None
    fragment_like = sum(gran_t.get(k, 0) for k in ("word", "abbreviation"))
    return {
        "translatable_count": n_t,
        "non_translatable_count": len(non_translatable),
        "translatable_by_granularity": gran_t,
        "translatable_by_role": by(translatable, "role"),
        "non_translatable_by_granularity": by(non_translatable, "granularity"),
        "non_translatable_by_reason": by(non_translatable, "reason"),
        "phrase_ratio_among_translatable": phrase_ratio,
        "word_or_abbrev_among_translatable": fragment_like,
        "selection_health": _health(phrase_ratio, fragment_like, n_t),
    }


def _health(phrase_ratio, fragment_like, n_t) -> str:
    if not n_t:
        return "empty"
    # Good selection sends mostly phrases/expressions, few bare words/abbrevs.
    if (phrase_ratio or 0) >= 0.5 and fragment_like / n_t <= 0.2:
        return "good"
    if fragment_like / n_t > 0.4:
        return "fragmented"
    return "mixed"


GRANULARITY_ORDER = ["phrase", "expression", "word", "abbreviation", "empty"]
GRANULARITY_LABEL = {"phrase": "Phrases", "expression": "Expressions", "word": "Mots", "abbreviation": "Abréviations", "empty": "Vides"}


def write_markdown(tag: str, audit: dict, out_path: Path) -> None:
    s = audit["summary"]
    st = audit["structural"]
    elements = audit["translatable"] + audit["non_translatable"]
    by_gran: dict[str, list] = {}
    for it in elements:
        by_gran.setdefault(it["granularity"], []).append(it)

    lines = [
        f"# Inventaire des éléments — {tag}", "",
        f"- page_role: `{audit['page_role']}`",
        f"- éléments textuels: **{len(elements)}** (traduisibles {s['translatable_count']}, non {s['non_translatable_count']})",
        f"- granularité (traduisibles): {s['translatable_by_granularity']}",
        f"- **santé sélection: {s['selection_health']}**",
        "",
        "## 1. Éléments textuels (par granularité)",
    ]
    for gran in GRANULARITY_ORDER:
        items = by_gran.get(gran)
        if not items:
            continue
        lines += ["", f"### {GRANULARITY_LABEL[gran]} ({len(items)})", "",
                  "| traduisible | role | object_type | raison | texte |", "|---|---|---|---|---|"]
        for it in items:
            trad = "✅ oui" if it.get("translatable") else "🚫 non"
            lines.append(f"| {trad} | {it.get('role')} | {it.get('object_type')} | {it.get('reason')} | {_md(it['text'])} |")

    lines += ["", "## 2. Éléments non textuels / structurels", ""]

    lines += [f"### Formules / équations ({len(st['formulas'])})"]
    if st["formulas"]:
        lines += ["", "| preservation_mode | bbox | texte |", "|---|---|---|"]
        for f in st["formulas"]:
            lines.append(f"| {f.get('preservation_mode')} | {f.get('bbox')} | {_md(f.get('text'))} |")
    lines.append("")

    lines += [f"### Code ({len(st['code_blocks'])})"]
    if st["code_blocks"]:
        lines += ["", "| bbox | texte |", "|---|---|"]
        for c in st["code_blocks"]:
            lines.append(f"| {c.get('bbox')} | {_md(c.get('text'))} |")
    lines.append("")

    lines += [f"### Tables ({len(st['tables'])})"]
    if st["tables"]:
        lines += ["", "| table_id | colonnes | lignes | cellules | bbox |", "|---|---|---|---|---|"]
        for t in st["tables"]:
            lines.append(f"| {t.get('table_id')} | {_md(str(t.get('columns')))} | {t.get('rows')} | {t.get('cells')} | {t.get('bbox')} |")
    lines.append("")

    lines += [f"### Figures / images ({len(st['figures'])})"]
    if st["figures"]:
        lines += ["", "| bbox | diagram_labels |", "|---|---|"]
        for f in st["figures"]:
            lines.append(f"| {f.get('bbox')} | {f.get('diagram_labels')} |")
    lines.append("")

    lines += ["### Zones / régions détectées", "", "| type | nombre |", "|---|---|"]
    for rtype, count in sorted(st["regions"].items(), key=lambda x: -x[1]):
        lines.append(f"| {rtype} | {count} |")
    lines.append("")

    bg = st["background"]
    lines += ["### Fond / background (trame sans texte)", "",
              f"- couche de fond présente : **{'oui' if bg['has_background_layer'] else 'non'}**",
              f"- background_path : `{bg['background_path']}`",
              f"- masques : {bg['masks']} · overlays : {bg['overlays']} · overlays immuables : {bg['immutable_overlays']}",
              f"- image source : `{bg['source_image_path']}`"]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md(text: str) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ")[:140]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", default=None, help="A single PAGEPRINT input_data JSON.")
    parser.add_argument("--audit-dir", default=None, help="Directory of input_data_*.json files.")
    parser.add_argument("--out", default=None, help="Directory to write per-page markdown + a summary JSON.")
    args = parser.parse_args()

    files = []
    if args.input_data:
        files = [args.input_data]
    elif args.audit_dir:
        files = sorted(glob.glob(str(Path(args.audit_dir) / "input_data_*.json")))
    if not files:
        parser.error("provide --input-data or --audit-dir")

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for path in files:
        tag = Path(path).stem.replace("input_data_", "")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        audit = audit_page(data)
        all_summaries.append({"tag": tag, **audit["summary"], "page_role": audit["page_role"]})
        if out_dir:
            write_markdown(tag, audit, out_dir / f"selection_{tag}.md")
        s = audit["summary"]
        print(f"{tag:34s} role={str(audit['page_role']):11s} translatable={s['translatable_count']:3d} "
              f"phrases={s['translatable_by_granularity'].get('phrase',0):3d} expr={s['translatable_by_granularity'].get('expression',0):3d} "
              f"word={s['translatable_by_granularity'].get('word',0):3d} abbr={s['translatable_by_granularity'].get('abbreviation',0):3d} "
              f"-> {s['selection_health']}")

    if out_dir:
        (out_dir / "selection_summary.json").write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nOutput:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
