"""API du clone objectif — serveur HTTP mince + atelier d'analyse.

Responsabilités strictement limitées :
    - upload (multi-fichiers) ;
    - routing ;
    - appel du PipelineOrchestrator / pipeline complet ;
    - persistance des résultats sous results/web/<run>/ + manifeste ;
    - retour JSON.

Pas de classification, postprocessing, sémantique, politiques, normalisation,
relations, inpainting, reconstruction ni traduction ici : tout est dans
pipelines/ et pageprint/.

Lancement (port 8002 pour ne pas entrer en conflit avec ocr_server.py:8001) :
    python -m server.api
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pipelines import PipelineOrchestrator

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"                    # racine unique de tous les artefacts
UPLOAD_DIR = RESULTS_ROOT / "uploads"              # PDF téléversés conservés
RESULTS_DIR = RESULTS_ROOT / "pageprint"           # /ocr (INPUT_DATA brut)
WEB_OUT = RESULTS_ROOT / "web"                     # atelier web : 1 dossier par run
STATIC_DIR = Path(__file__).resolve().parent / "static"
WEB_OUT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PAGEPRINT Studio", version="0.3.0")

# Pipeline lourd (orchestrateur + moteur CT2) construit une seule fois.
_ENGINE = {"orch": None, "engine": None}

MAX_PAGES_PER_RUN = 24

# Classement des artefacts par catégorie (préfixe de fichier -> catégorie).
CATEGORIES = [
    ("reconstructed_", ".pdf", "pdf", "PDF reconstruit"),
    ("reconstructed_", ".png", "reconstructed", "Rendu (PNG)"),
    ("source_", ".png", "source", "Source (PNG)"),
    ("pagereconstruct_overlay_", ".png", "overlay", "Zones détectées"),
    ("cleanbg_", ".png", "cleanbg", "Fond propre"),
    ("audit_", ".json", "audit", "Audit qualité"),
    ("pubready_", ".json", "pubready", "Publication-ready"),
    ("pagereconstruct_plan_", ".json", "plan", "Plan reconstruction"),
    ("pageprint_", ".json", "pageprint", "Données PAGEPRINT"),
    ("pagetranslate_", ".json", "pagetranslate", "Traduction"),
]


def _pipeline():
    if _ENGINE["orch"] is None:
        from tools.run_pageprint_pagetranslate_audit import make_orchestrator, make_engine
        _ENGINE["orch"] = make_orchestrator(str(WEB_OUT / "_render"), enable_ocr=False)
        _ENGINE["engine"] = make_engine("ct2", model="opus_mt_tc_big_en_fr",
                                        source_lang="en", target_lang="fr")
    return _ENGINE["orch"], _ENGINE["engine"]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _parse_pages(spec: str | None, n_pages: int) -> list[int]:
    """'1-3,5' -> [1,2,3,5] borné à [1, n_pages] et à MAX_PAGES_PER_RUN."""
    if not spec or not spec.strip():
        return [1]
    out: list[int] = []
    for chunk in spec.replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk:
            a, _, b = chunk.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            out.extend(range(min(lo, hi), max(lo, hi) + 1))
        else:
            try:
                out.append(int(chunk))
            except ValueError:
                continue
    seen, clean = set(), []
    for p in out:
        if 1 <= p <= n_pages and p not in seen:
            seen.add(p)
            clean.append(p)
    return clean[:MAX_PAGES_PER_RUN]


def _page_count(pdf_path: Path) -> int:
    try:
        import fitz
        with fitz.open(str(pdf_path)) as doc:
            return doc.page_count
    except Exception:
        return 1


def _url(run: str, name: str) -> str:
    from urllib.parse import quote
    return f"/results/{quote(run)}/{quote(name)}"


def _categorize(run: str, out: Path, tag: str) -> dict:
    """Retourne {key: {url,label,name}} pour les artefacts d'une page (tag)."""
    files: dict[str, dict] = {}
    for prefix, ext, key, label in CATEGORIES:
        fname = f"{prefix}{tag}{ext}"
        if (out / fname).is_file():
            files[key] = {"url": _url(run, fname), "label": label, "name": fname}
    return files


def _read_audit(out: Path, tag: str) -> dict:
    f = out / f"audit_{tag}.json"
    if not f.is_file():
        return {}
    try:
        a = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "status": a.get("status"),
        "publication_ready": a.get("publication_ready"),
        "publication_ready_score": a.get("publication_ready_score"),
        "scores": a.get("visual_scores") or {},
        "ko_findings": [f for f in (a.get("findings") or []) if f.get("severity") == "ko"][:12],
    }


def _read_manifest(run_dir: Path) -> dict | None:
    f = run_dir / "manifest.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Routes — UI + atelier
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    # no-store : évite que le navigateur serve une vieille UI en cache après mise à jour.
    return FileResponse(str(STATIC_DIR / "index.html"),
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/healthz")
def healthcheck():
    orchestrator = PipelineOrchestrator()
    return {
        "status": "ok",
        "service": "pageprint-pipeline",
        "role": "objective_clone_of_ocr_server",
        **orchestrator.status(),
    }


@app.post("/api/run")
async def run_pipeline(
    files: list[UploadFile] = File(default=[]),
    pages: str = Form(default="1"),
    source_lang: str = Form(default="en"),
    target_lang: str = Form(default="fr"),
    pubready_mode: str = Form(default="review"),
):
    """Téléverse un ou plusieurs PDF, exécute le pipeline complet sur les pages
    demandées, persiste les artefacts sous results/web/<run>/ et renvoie le
    manifeste (pages, scores, fichiers classés par catégorie)."""
    from tools.run_pipeline_full_demo import process

    files = [f for f in files if f is not None and f.filename]
    if not files:
        return JSONResponse(status_code=400, content={"error": "aucun fichier téléversé"})

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    run = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out = WEB_OUT / run
    out.mkdir(parents=True, exist_ok=True)

    orch, engine = _pipeline()
    page_entries: list[dict] = []
    errors: list[dict] = []
    t0 = time.time()

    for uf in files:
        stem = Path(uf.filename).stem
        pdf_path = UPLOAD_DIR / f"web_{uuid.uuid4().hex[:10]}.pdf"
        with open(pdf_path, "wb") as h:
            shutil.copyfileobj(uf.file, h)
        n = _page_count(pdf_path)
        wanted = _parse_pages(pages, n)
        for pg in wanted:
            try:
                summary = process(orch, engine, pdf_path, int(pg), out, source_lang, target_lang,
                                  pubready_mode=pubready_mode, tid_cache=None, reuse_tid=False)
            except Exception as exc:
                errors.append({"document": uf.filename, "page": pg, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if summary.get("error"):
                errors.append({"document": uf.filename, "page": pg, "error": summary["error"]})
                continue
            tag = summary.get("tag")
            audit = _read_audit(out, tag)
            pr = summary.get("pubready") or {}
            page_entries.append({
                "tag": tag,
                "document": uf.filename,
                "page": int(pg),
                "status": summary.get("status"),
                "pubready_score": pr.get("score"),
                "pubready_status": pr.get("status"),
                "publication_ready": pr.get("publication_ready"),
                "hard_blockers": pr.get("hard_blockers") or [],
                "counts": {k: summary.get(k) for k in ("translated_text_count", "protected_region_count",
                                                       "preserved_overlay_count", "preserved_underlay_count",
                                                       "finding_count")},
                "audit": audit,
                "files": _categorize(run, out, tag),
            })

    scores = [p["pubready_score"] for p in page_entries if isinstance(p.get("pubready_score"), (int, float))]
    manifest = {
        "run": run,
        "created": datetime.now(timezone.utc).isoformat(),
        "label": (files[0].filename if len(files) == 1 else f"{len(files)} documents"),
        "params": {"pages": pages, "source_lang": source_lang, "target_lang": target_lang,
                   "pubready_mode": pubready_mode},
        "elapsed_s": round(time.time() - t0, 1),
        "documents": sorted({p["document"] for p in page_entries}),
        "pages": page_entries,
        "errors": errors,
        "summary": {
            "page_count": len(page_entries),
            "doc_count": len({p["document"] for p in page_entries}),
            "avg_pubready": round(sum(scores) / len(scores), 3) if scores else None,
            "ready_count": sum(1 for p in page_entries if p.get("publication_ready")),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not page_entries:
        return JSONResponse(status_code=422, content={"error": "aucune page traitée", "errors": errors})
    return manifest


@app.get("/api/runs")
def list_runs():
    """Bibliothèque : tous les runs persistés, le plus récent d'abord."""
    runs = []
    if WEB_OUT.is_dir():
        for d in sorted(WEB_OUT.iterdir(), reverse=True):
            if not d.is_dir() or d.name.startswith("_"):
                continue
            m = _read_manifest(d)
            if not m:
                continue
            runs.append({
                "run": m.get("run", d.name),
                "label": m.get("label"),
                "created": m.get("created"),
                "documents": m.get("documents", []),
                "summary": m.get("summary", {}),
                "params": m.get("params", {}),
            })
    return {"runs": runs}


@app.get("/api/runs/{run}")
def get_run(run: str):
    m = _read_manifest(WEB_OUT / run)
    if not m:
        return JSONResponse(status_code=404, content={"error": "run introuvable"})
    return m


@app.delete("/api/runs/{run}")
def delete_run(run: str):
    d = WEB_OUT / run
    if not d.is_dir() or (WEB_OUT.resolve() not in d.resolve().parents):
        return JSONResponse(status_code=404, content={"error": "run introuvable"})
    shutil.rmtree(d)
    return {"ok": True, "deleted": run}


@app.get("/api/runs/{run}/zip")
def zip_run(run: str):
    d = WEB_OUT / run
    if not d.is_dir():
        return JSONResponse(status_code=404, content={"error": "run introuvable"})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in d.rglob("*"):
            if f.is_file() and "_render" not in f.parts:
                z.write(f, f.relative_to(d))
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{run}.zip"'})


@app.get("/api/file")
def get_file(run: str, name: str):
    """Renvoie le contenu JSON d'un artefact (audit/plan/traduction…) inline."""
    safe = os.path.basename(name)
    f = WEB_OUT / run / safe
    if not f.is_file():
        return JSONResponse(status_code=404, content={"error": "fichier introuvable"})
    if f.suffix == ".json":
        try:
            return JSONResponse(content=json.loads(f.read_text(encoding="utf-8")))
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})
    return FileResponse(str(f))


@app.post("/ocr")
async def perform_ocr(
    file: UploadFile = File(...),
    pages: str | None = Form(default=None),
    enable_ocr: bool = Form(default=False),
    enable_understanding: bool = Form(default=True),
    source_lang: str | None = Form(default=None),
    target_lang: str | None = Form(default=None),
):
    """Charge un document et retourne INPUT_DATA canonique par page."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(file.filename or "upload.pdf")[1] or ".pdf"
    save_path = UPLOAD_DIR / f"pageprint_{uuid.uuid4().hex[:10]}{ext}"
    with open(save_path, "wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        orchestrator = PipelineOrchestrator(
            enable_ocr=enable_ocr,
            enable_understanding=enable_understanding,
            save_render_dir=str(RESULTS_DIR),
        )
        result = orchestrator.run(
            str(save_path),
            pages=pages,
            language={"source_lang": source_lang, "target_lang": target_lang},
        )
        return JSONResponse(content=_jsonable(result))
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"{type(exc).__name__}: {exc}"},
        )


def _jsonable(value):
    """Conversion défensive : ne laisser passer que du JSON-compatible."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(WEB_OUT)), name="results")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
