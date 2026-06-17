#!/usr/bin/env python3
"""Contrôle de survie du texte — PAGEPRINT → PAGERECONSTRUCT.

Petit script (NA-01→07). Garantit que 100% du texte extrait par PAGEPRINT se
retrouve dans le rendu PAGERECONSTRUCT, traduit OU préservé OU exclu avec une
raison valide. Sinon : KO bloquant (code de sortie != 0).

Il NE fabrique aucune traduction : il consomme strictement la sortie de
PAGETRANSLATE (translated_input_data) et le plan PAGERECONSTRUCT (render_ops).
La preuve est contractuelle (texte → TextOp/PreservationOp), pas d'OCR image
en V1 (cf. NA-07).

Réutilise pagereconstruct.source_text_lifecycle_ledger (pas de voie parallèle).

Usage:
  python tools/audit_text_survival.py TID.json [TID2.json ...] --out results/text_survival
  python tools/audit_text_survival.py --plan plan.json --tid TID.json --out DIR

Entrée principale: translated_input_data (sortie PAGETRANSLATE). Le plan est
recompilé via compile_page_render_plan si --plan n'est pas fourni.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pagereconstruct.source_text_lifecycle_ledger import (
    build_source_text_lifecycle_ledger, VALID_EXCLUSION_REASONS)


# --- colonnes du rapport (1 ligne par texte original PAGEPRINT) ---
CSV_COLUMNS = [
    "page_id", "source_unit_id", "level", "state", "kind",
    "source_text", "rendered_text", "reconstruction_unit_ids",
    "textop_ids", "preservationop_ids", "source_bbox", "status", "findings",
]


def _translated_text_by_source(normalized: dict) -> dict[str, str]:
    """source_unit_id -> translated/projected text (pour la colonne rendered_text)."""
    out: dict[str, str] = {}
    for u in normalized.get("translated_units") or []:
        txt = (u.get("translated_text") or u.get("text") or "").strip()
        if not txt:
            continue
        for sid in u.get("source_unit_ids") or []:
            out.setdefault(str(sid), txt)
    return out


def _preserved_text_by_source(normalized: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in normalized.get("preservation_plan") or []:
        txt = (p.get("text") or "").strip()
        for sid in p.get("source_unit_ids") or []:
            if txt:
                out.setdefault(str(sid), txt)
    return out


def audit_text_survival(translated_input_data: dict, *, page_id: str = "page",
                        mode: str = "debug", plan: dict | None = None) -> dict:
    """Retourne un rapport de survie du texte pour UNE page.

    100% = tous les textes PAGEPRINT ont un état rendu valide (traduit/préservé/
    exclu-raison-valide) ET l'op de rendu correspondante existe."""
    from pagereconstruct.input_adapter import PageReconstructInputAdapter
    normalized = PageReconstructInputAdapter().normalize(translated_input_data)
    if plan is None:
        from pagereconstruct import compile_page_render_plan
        plan = compile_page_render_plan(translated_input_data, reconstruction_mode=mode).to_dict()

    ledger = build_source_text_lifecycle_ledger(plan, normalized)
    tr_by_src = _translated_text_by_source(normalized)
    pr_by_src = _preserved_text_by_source(normalized)

    rows: list[dict] = []
    duplicates: list[str] = []
    no_render_op: list[str] = []
    valid_exclusions = 0
    ok = ko = 0
    for e in ledger:
        state = e.pagetranslate_state
        if state.startswith("excluded:"):
            kind, rendered = "excluded", ""
            valid_exclusions += 1
        elif state == "preserved":
            kind, rendered = "preserved", pr_by_src.get(e.source_unit_id, e.source_text)
        elif state == "translated_or_projected":
            kind, rendered = "translated", tr_by_src.get(e.source_unit_id, "")
        else:
            kind, rendered = "missing", ""
        if "source_text_missing_render_op" in e.findings or \
           "source_text_missing_preservation_op" in e.findings:
            no_render_op.append(e.source_unit_id)
        if len(e.reconstruction_unit_ids) > 1 or len(e.textop_ids) > 1:
            duplicates.append(e.source_unit_id)
        if e.status == "ok":
            ok += 1
        else:
            ko += 1
        rows.append({
            "page_id": page_id, "source_unit_id": e.source_unit_id, "level": e.level,
            "state": state, "kind": kind, "source_text": e.source_text,
            "rendered_text": rendered,
            "reconstruction_unit_ids": ";".join(e.reconstruction_unit_ids),
            "textop_ids": ";".join(e.textop_ids),
            "preservationop_ids": ";".join(e.preservationop_ids),
            "source_bbox": e.source_bbox, "status": e.status,
            "findings": ";".join(e.findings),
        })

    total = len(ledger)
    coverage = round(ok / total, 4) if total else 1.0
    blockers = sorted({f for e in ledger for f in e.findings})
    return {
        "page_id": page_id,
        "status": "ok" if ko == 0 else "ko",
        "publication_ready_text": ko == 0,
        "coverage_ratio": coverage,
        "summary": {
            "total_texts": total, "ok": ok, "ko": ko,
            "valid_exclusions": valid_exclusions,
            "duplicates": sorted(set(duplicates)),
            "texts_without_render_op": sorted(set(no_render_op)),
        },
        "hard_blockers": blockers,
        "rows": rows,
    }


# --- writers ---------------------------------------------------------------

def write_reports(report: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pid = report["page_id"]
    json_path = out_dir / f"text_survival_{pid}.json"
    csv_path = out_dir / f"text_survival_{pid}.csv"
    md_path = out_dir / f"text_survival_{pid}.md"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in report["rows"]:
            w.writerow({k: (json.dumps(r[k], ensure_ascii=False) if k == "source_bbox" else r[k])
                        for k in CSV_COLUMNS})

    s = report["summary"]
    L = [f"# Survie du texte — {pid}",
         f"**Statut** `{report['status']}` · couverture **{report['coverage_ratio']*100:.1f}%** "
         f"({s['ok']}/{s['total_texts']} OK, {s['ko']} KO)", ""]
    if s["valid_exclusions"]:
        L.append(f"- exclusions valides : {s['valid_exclusions']}")
    if s["duplicates"]:
        L.append(f"- doublons (texte rendu >1 fois) : {', '.join(s['duplicates'])}")
    if s["texts_without_render_op"]:
        L.append(f"- textes sans op de rendu : {', '.join(s['texts_without_render_op'])}")
    if report["hard_blockers"]:
        L += ["", "## Blocages", *[f"- `{b}`" for b in report["hard_blockers"]]]
    ko_rows = [r for r in report["rows"] if r["status"] != "ok"]
    if ko_rows:
        L += ["", "## Textes manquants au rendu", "", "| unit | niveau | état | texte source |",
              "|---|---|---|---|"]
        for r in ko_rows[:50]:
            txt = (r["source_text"] or "")[:60].replace("|", "/").replace("\n", " ")
            L.append(f"| {r['source_unit_id']} | {r['level']} | {r['state']} | {txt} |")
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}


def _page_id_from_path(p: Path) -> str:
    name = p.stem
    for pref in ("pagetranslate_", "pagereconstruct_plan_", "tid_", "translated_"):
        if name.startswith(pref):
            name = name[len(pref):]
    return name or "page"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Contrôle de survie du texte PAGEPRINT→PAGERECONSTRUCT")
    ap.add_argument("tids", nargs="*", help="translated_input_data JSON (1 ou plusieurs pages)")
    ap.add_argument("--plan", default=None, help="plan PAGERECONSTRUCT JSON (sinon recompilé)")
    ap.add_argument("--tid", default=None, help="tid JSON (alternative à positionnel, avec --plan)")
    ap.add_argument("--mode", default="debug", choices=["debug", "review", "publication"])
    ap.add_argument("--out", default="results/text_survival", help="dossier de sortie")
    args = ap.parse_args(argv)

    out = Path(args.out)
    inputs: list[tuple[Path, dict | None]] = []
    if args.tid:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else None
        inputs.append((Path(args.tid), plan))
    for t in args.tids:
        inputs.append((Path(t), None))
    if not inputs:
        ap.error("fournir au moins un translated_input_data JSON")

    reports = []
    worst = 0
    for tid_path, plan in inputs:
        tid = json.loads(tid_path.read_text(encoding="utf-8"))
        pid = _page_id_from_path(tid_path)
        rep = audit_text_survival(tid, page_id=pid, mode=args.mode, plan=plan)
        paths = write_reports(rep, out)
        s = rep["summary"]
        print(f"{pid}: {rep['status']} coverage={rep['coverage_ratio']*100:.1f}% "
              f"({s['ok']}/{s['total_texts']} OK, {s['ko']} KO) -> {paths['md']}", flush=True)
        reports.append({"page_id": pid, "status": rep["status"],
                        "coverage_ratio": rep["coverage_ratio"], "summary": s,
                        "files": paths})
        if rep["status"] != "ok":
            worst = 1

    doc = {
        "pages": reports,
        "all_text_rendered": worst == 0,
        "page_count": len(reports),
        "ko_pages": [r["page_id"] for r in reports if r["status"] != "ok"],
    }
    (out / "text_survival_summary.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDocument: all_text_rendered={doc['all_text_rendered']} "
          f"ko_pages={doc['ko_pages']}", flush=True)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
