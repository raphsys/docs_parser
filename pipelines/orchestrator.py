"""PipelineOrchestrator — le clone objectif d'ocr_server.py.

Orchestre les unités du pipeline cible, page par page :

    SourceLoader → PageRenderer → RawExtractors → PageUnderstanding → PAGEPRINT

Sortie par page : INPUT_DATA canonique (pageprint.input.v1) + trace de pipeline.
Chaque étape est tracée et fail-safe ; une page en erreur n'arrête pas le
document. ocr_server.py historique n'est ni importé ni modifié.
"""

from __future__ import annotations

import os
import time

from .page_renderer import DEFAULT_TARGET_DPI, PageRenderer
from .page_understanding import PageUnderstanding
from .pageprint_stage import PagePrintStage
from .raw_extractors import RawExtractors
from .source_loader import SourceLoader


def _parse_pages(pages, page_count: int) -> list[int]:
    """pages: None (toutes), liste d'index 0-based, ou chaîne '1,3-5' (1-based)."""
    if pages is None:
        return list(range(page_count))
    if isinstance(pages, str):
        indices: list[int] = []
        for part in pages.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                indices.extend(range(int(start) - 1, int(end)))
            else:
                indices.append(int(part) - 1)
        pages = indices
    return [i for i in pages if 0 <= i < page_count]


class PipelineOrchestrator:
    def __init__(self, *, target_dpi: int = DEFAULT_TARGET_DPI,
                 enable_ocr: bool = False,
                 enable_understanding: bool = True,
                 enable_postprocessors: bool = True,
                 enable_special_regions: bool = True,
                 save_render_dir: str | None = None):
        self.source_loader = SourceLoader()
        self.page_renderer = PageRenderer(target_dpi=target_dpi)
        self.raw_extractors = RawExtractors(enable_ocr=enable_ocr)
        self.page_understanding = PageUnderstanding(
            enable_layout_v2=enable_understanding,
            enable_postprocessors=enable_postprocessors,
            enable_special_regions=enable_special_regions,
        ) if enable_understanding else None
        self.pageprint_stage = PagePrintStage(default_dpi=target_dpi)
        self.save_render_dir = save_render_dir

    def status(self) -> dict:
        """Modules disponibles — consommé par /healthz."""
        availability = {}
        for module in ("fitz", "PIL", "native_pdf_extractor", "structure_extractor",
                       "page_extraction_postprocessors", "special_region_detector",
                       "rapidocr_onnxruntime", "pageprint"):
            try:
                __import__(module)
                availability[module] = True
            except Exception:
                availability[module] = False
        return {
            "pipeline": "objective_clone",
            "stages": ["source_loader", "page_renderer", "raw_extractors",
                       "page_understanding", "pageprint"],
            "pageprint_schema": "pageprint.input.v1",
            "modules": availability,
        }

    def run(self, source_path: str, *, pages=None,
            language: dict | None = None) -> dict:
        """Exécute le pipeline complet. Retourne {document, pages: [...]}."""
        source_context = self.source_loader.load(source_path)
        document = source_context.get("document")
        source_context["language"] = language or {}

        page_count = source_context["page_count"]
        selected = _parse_pages(pages, page_count)

        results = []
        try:
            for page_index in selected:
                results.append(self._run_page(source_context, document, page_index))
        finally:
            if document is not None:
                document.close()

        return {
            "document": {
                "document_id": source_context["document_id"],
                "file_name": source_context["file_name"],
                "file_type": source_context["file_type"],
                "page_count": page_count,
                "pages_processed": [r["page"] for r in results],
                "source_file_hash": source_context["source_file_hash"],
            },
            "pages": results,
        }

    def _run_page(self, source_context: dict, document, page_index: int) -> dict:
        trace: dict = {}
        started = time.perf_counter()
        try:
            # 2. PageRenderer
            t0 = time.perf_counter()
            if document is not None:
                pdf_page = document[page_index]
                render = self.page_renderer.render_pdf_page(pdf_page)
            else:
                pdf_page = None
                render = self.page_renderer.load_image(source_context["source_path"])
            trace["page_renderer"] = {
                "duration_s": round(time.perf_counter() - t0, 3),
                "dimensions": render["dimensions"],
            }

            render_path = self._save_render(render["image"], source_context, page_index)

            # 3. RawExtractors
            t0 = time.perf_counter()
            extracted = self.raw_extractors.extract(
                pdf_page=pdf_page, image=render["image"],
                sx=render["sx"], sy=render["sy"],
                dimensions=render["dimensions"],
            )
            page_structure = extracted["page_structure"]
            extraction_result = extracted["extraction_result"]
            trace["raw_extractors"] = {
                "duration_s": round(time.perf_counter() - t0, 3),
                **extraction_result["trace"],
            }

            # 4. PageUnderstanding
            if self.page_understanding is not None:
                t0 = time.perf_counter()
                page_structure, understanding_trace = self.page_understanding.apply(
                    page_structure, img=render["image"], pdf_page=pdf_page,
                    sx=render["sx"], sy=render["sy"],
                )
                trace["page_understanding"] = {
                    "duration_s": round(time.perf_counter() - t0, 3),
                    **understanding_trace,
                }

            # 5. PAGEPRINT
            t0 = time.perf_counter()
            input_data = self.pageprint_stage.build(
                page_structure=page_structure,
                source_context=source_context,
                extraction_result=extraction_result,
                assets={"source_image_path": render_path},
                page_image=render["image"],
                pdf_page=pdf_page,
                page_index=page_index,
            )

            # 5b. Clean (text-removed) background — the legacy pipeline produced
            # this and PAGERECONSTRUCT needs it; without it every page leaks the
            # source text (source_background mode). Inpaint translatable text out.
            try:
                from .background_cleaner import build_clean_background
                clean_path = build_clean_background(
                    input_data,
                    out_path=os.path.splitext(render_path)[0] + "_clean_bg.png",
                )
                if clean_path:
                    input_data["assets"]["background_path"] = clean_path
                    bg = input_data.setdefault("background", {})
                    if isinstance(bg, dict):
                        bg["path"] = clean_path
                        bg["text_removed"] = True
            except Exception as exc:  # pragma: no cover - never block extraction
                trace.setdefault("background_cleaner", {})["error"] = repr(exc)
            trace["pageprint"] = {
                "duration_s": round(time.perf_counter() - t0, 3),
                "unit_count": len(input_data["units"]),
                "region_count": len(input_data["regions"]),
                "valid": input_data["debug"]["validation"]["valid"],
            }

            return {
                "page": page_index + 1,
                "status": "ok",
                "input_data": input_data,
                "pipeline_trace": trace,
                "duration_s": round(time.perf_counter() - started, 3),
            }
        except Exception as exc:
            return {
                "page": page_index + 1,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "pipeline_trace": trace,
                "duration_s": round(time.perf_counter() - started, 3),
            }

    def _save_render(self, image, source_context: dict, page_index: int) -> str | None:
        if not self.save_render_dir:
            return None
        import os

        os.makedirs(self.save_render_dir, exist_ok=True)
        base = os.path.splitext(source_context["file_name"])[0]
        path = os.path.join(self.save_render_dir, f"src_{base}_p{page_index + 1:03d}.png")
        try:
            image.save(path)
            return path
        except Exception:
            return None
