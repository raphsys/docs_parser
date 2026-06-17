"""Export des rapports publication-ready (JSON + Markdown lisible)."""

from __future__ import annotations

import json
import os


def write_page_report(report, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    d = report.to_dict()
    base = os.path.join(out_dir, f"pubready_{report.page_id}")
    json_path = base + ".json"
    with open(json_path, "w", encoding="utf-8") as h:
        json.dump(d, h, ensure_ascii=False, indent=2)
    return {"json": json_path}


def page_markdown(report) -> str:
    d = report.to_dict()
    L = [f"# Publication-ready — {d['page_id']}",
         f"**Statut**: `{d['status']}` · **ready**: {d['publication_ready']} · **score**: {d['publication_ready_score']}",
         "", "## Scores par étape"]
    for k, v in d["stage_scores"].items():
        L.append(f"- {k}: **{v}**")
    if d["hard_blockers"]:
        L += ["", "## Blocages durs", *[f"- `{b}`" for b in d["hard_blockers"]]]
    # détail granulaire typo
    for st in d["stages"]:
        if st["stage_name"] == "typography" and st["elements"]:
            L += ["", "## Typographie (bloc → dimensions)"]
            for b in st["elements"][:12]:
                dims = " ".join(f"{x['name']}={x['score']}" for x in b["dimensions"])
                L.append(f"- `{b['element_id']}` ({b['role']}) score {b['score']} — {dims}")
    return "\n".join(L) + "\n"


def write_document_report(doc_report, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "pubready_document.json")
    with open(jpath, "w", encoding="utf-8") as h:
        json.dump(doc_report.to_dict(), h, ensure_ascii=False, indent=2)
    md = [f"# Document publication-ready — {doc_report.document_id}",
          f"**Statut** `{doc_report.status}` · ready {doc_report.publication_ready} · score **{round(doc_report.publication_ready_score,3)}** · pages {doc_report.page_count}",
          f"Pages bloquantes: {doc_report.blocking_pages or '—'}", ""]
    for p in doc_report.pages:
        md.append(f"- {p.page_id}: `{p.status}` {round(p.publication_ready_score,3)} {('⛔'+','.join(p.hard_blockers)) if p.hard_blockers else ''}")
    mpath = os.path.join(out_dir, "pubready_document.md")
    open(mpath, "w", encoding="utf-8").write("\n".join(md) + "\n")
    return {"json": jpath, "md": mpath}
