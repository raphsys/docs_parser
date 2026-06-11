"""API du clone objectif — serveur HTTP mince.

Responsabilités strictement limitées :
    - upload ;
    - routing ;
    - appel du PipelineOrchestrator ;
    - retour JSON.

Pas de classification, postprocessing, sémantique, politiques, normalisation,
relations, inpainting, reconstruction ni traduction ici : tout est dans
pipelines/ et pageprint/.

Lancement (port 8002 pour ne pas entrer en conflit avec ocr_server.py:8001) :
    python -m server.api
"""

from __future__ import annotations

import os
import shutil
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from pipelines import PipelineOrchestrator

UPLOAD_DIR = "uploads"
RESULTS_DIR = os.path.join("ocr_results", "pageprint")

app = FastAPI(title="PAGEPRINT Pipeline Server", version="0.1.0")


@app.get("/healthz")
def healthcheck():
    orchestrator = PipelineOrchestrator()
    return {
        "status": "ok",
        "service": "pageprint-pipeline",
        "role": "objective_clone_of_ocr_server",
        **orchestrator.status(),
    }


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
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename or "upload.pdf")[1] or ".pdf"
    save_path = os.path.join(UPLOAD_DIR, f"pageprint_{uuid.uuid4().hex[:10]}{ext}")
    with open(save_path, "wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        orchestrator = PipelineOrchestrator(
            enable_ocr=enable_ocr,
            enable_understanding=enable_understanding,
            save_render_dir=RESULTS_DIR,
        )
        result = orchestrator.run(
            save_path,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
