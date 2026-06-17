"""compare_legacy_reconstruction — comparateur ancien/nouveau + score pubready.

Pour une page : produit le rendu MODERNE (pipeline pageprint→pagetranslate→
pagereconstruct), le compare à la SOURCE (diff heatmap), et le NOTE avec pubready
(score explicable par étape). Le rendu LEGACY (reconstructor.DocumentReconstructor)
n'est exécuté que si des données au format ancien (final_blocks) sont fournies —
sinon on se contente du signal moderne vs source, qui est la mesure anti-régression
réellement utilisée.

Usage:
    python tools/compare_legacy_reconstruction.py --pdf <p.pdf> --page N [--out results/legacy_compare]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


def _diff_heatmap(a_path, b_path, out_path):
    try:
        import cv2
        import numpy as np
        a = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            return None
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        d = cv2.absdiff(a, b)
        hm = cv2.applyColorMap(d, cv2.COLORMAP_JET)
        cv2.imwrite(out_path, hm)
        return float(d.mean())
    except Exception:
        return None


def run(pdf: str, page: int, out_root: str) -> dict:
    from tools.run_pipeline_full_demo import process
    from tools.run_pageprint_pagetranslate_audit import make_orchestrator, make_engine
    from pagereconstruct.input_adapter import PageReconstructInputAdapter
    import pubready

    out = os.path.join(out_root, f"{Path(pdf).stem[:20]}_p{page:04d}")
    os.makedirs(out, exist_ok=True)
    orch = make_orchestrator(os.path.join(out, "_render"), enable_ocr=False)
    engine = make_engine("ct2", model="opus_mt_tc_big_en_fr", source_lang="en", target_lang="fr")
    summary = process(orch, engine, Path(pdf), int(page), Path(out), "en", "fr")
    if summary.get("error"):
        return {"error": summary["error"]}

    src = (glob.glob(os.path.join(out, "source_*.png")) or [None])[0]
    rec = (glob.glob(os.path.join(out, "reconstructed_*.png")) or [None])[0]
    plan_f = (glob.glob(os.path.join(out, "pagereconstruct_plan_*.json")) or [None])[0]
    diff_mean = _diff_heatmap(src, rec, os.path.join(out, "diff_modern_source.png")) if (src and rec) else None

    report = {"modern": {"summary": summary, "diff_modern_source_mean": diff_mean}, "legacy": None}
    # score pubready moderne (granulaire)
    if plan_f:
        plan = json.load(open(plan_f))
        # source pageprint complet requis pour la typo: on relit l'input_data live.
        doc = orch.run(pdf, pages=str(page), language={"source_lang": "en", "target_lang": "fr"})
        ok = [p for p in (doc.get("pages") or []) if p.get("status") == "ok"]
        if ok:
            norm = PageReconstructInputAdapter().normalize(
                __import__("pagetranslate").build_page_translation(
                    ok[0]["input_data"], translator=engine, target_lang="fr", source_lang="en",
                    allow_fallback=True)["translated_input_data"])
            pr = pubready.evaluate_page(plan, norm, page_id=summary.get("tag", "page"), mode="review")
            report["pubready"] = pr.to_dict()
    json.dump(report, open(os.path.join(out, "report.json"), "w"), ensure_ascii=False, indent=2)
    return {"ok": True, "out": out, "diff_mean": diff_mean,
            "pubready_score": report.get("pubready", {}).get("publication_ready_score")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--out", default="results/legacy_compare")
    a = ap.parse_args()
    r = run(a.pdf, a.page, a.out)
    print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
