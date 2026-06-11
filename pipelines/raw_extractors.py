"""RawExtractors — unité 3 du pipeline cible.

Extraction brute multi-sources :
    - NativePDFExtractor (texte natif, images, dessins, zones non textuelles) ;
    - OCRExtractor (RapidOCR + DocumentParser), optionnel et paresseux,
      utilisé en complément quand le natif est pauvre.

Sortie : page_structure initiale (legacy, pixels) + métadonnées d'extraction.
La normalisation en points est faite ensuite par PAGEPRINT.
"""

from __future__ import annotations

OCR_NATIVE_OVERLAP_SKIP = 0.5


def _bbox_overlap_ratio(a, b) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    if area <= 0:
        return 0.0
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    return inter / area


class RawExtractors:
    """native PDF first, OCR when needed."""

    def __init__(self, *, enable_ocr: bool = False):
        self.enable_ocr = enable_ocr
        self._native_extractor = None
        self._ocr_engine = None
        self._ocr_parser = None

    # --- chargements paresseux -----------------------------------------

    def _native(self):
        if self._native_extractor is None:
            from native_pdf_extractor import NativePDFExtractor

            self._native_extractor = NativePDFExtractor()
        return self._native_extractor

    def _ocr(self):
        if self._ocr_engine is None:
            from rapidocr_onnxruntime import RapidOCR
            from structure_extractor import DocumentParser

            self._ocr_engine = RapidOCR()
            self._ocr_parser = DocumentParser()
        return self._ocr_engine, self._ocr_parser

    # --- extraction -----------------------------------------------------

    def extract(self, *, pdf_page=None, image=None, sx: float = 1.0,
                sy: float = 1.0, dimensions: dict | None = None) -> dict:
        trace = {"native": None, "ocr": None}
        blocks: list = []
        non_text_zones: list = []
        images: list = []
        drawings: list = []
        native_available = False

        if pdf_page is not None:
            native = self._native().extract_page(pdf_page, sx=sx, sy=sy)
            blocks = native.get("blocks") or []
            non_text_zones = native.get("non_text_zones") or []
            images = native.get("images") or []
            drawings = native.get("drawings") or []
            native_available = bool(blocks)
            trace["native"] = {"applied": True, "block_count": len(blocks)}

        ocr_used = False
        if self.enable_ocr and image is not None and not native_available:
            ocr_blocks = self._extract_ocr(image, blocks)
            blocks.extend(ocr_blocks)
            ocr_used = bool(ocr_blocks)
            trace["ocr"] = {"applied": ocr_used, "block_count": len(ocr_blocks)}

        page_structure = {
            "blocks": blocks,
            "non_text_zones": non_text_zones,
            "images": images,
            "drawings": drawings,
            "dimensions": dict(dimensions or {}),
        }
        extraction_result = {
            "pipeline": "objective_clone",
            "source_priority": ["native_pdf", "ocr", "layout_ai", "heuristic"],
            "native_pdf_available": native_available,
            "ocr_used": ocr_used,
            "ocr_engine": "RapidOCR" if ocr_used else None,
            # Compteurs uniquement : les listes complètes restent dans
            # page_structure (et compatibility.legacy) — pas de duplication
            # volumineuse dans INPUT_DATA.extraction.
            "raw_sources": {
                "native_blocks": len(blocks) if native_available else 0,
                "ocr_blocks": trace["ocr"]["block_count"] if trace["ocr"] else 0,
                "images": len(images),
                "drawings": len(drawings),
                "non_text_zones": len(non_text_zones),
            },
            "trace": trace,
        }
        return {"page_structure": page_structure, "extraction_result": extraction_result}

    def _extract_ocr(self, image, native_blocks: list) -> list:
        import numpy as np

        engine, parser = self._ocr()
        result, _ = engine(np.array(image))
        raw_ocr = []
        native_bboxes = [
            b.get("bbox") for b in native_blocks
            if isinstance(b, dict) and b.get("bbox")
        ]
        for entry in result or []:
            try:
                box, text, score = entry[0], entry[1], float(entry[2])
            except Exception:
                continue
            bbox = [
                float(min(p[0] for p in box)),
                float(min(p[1] for p in box)),
                float(max(p[0] for p in box)),
                float(max(p[1] for p in box)),
            ]
            # Ignorer l'OCR qui recouvre déjà fortement du texte natif.
            if any(_bbox_overlap_ratio(bbox, nb) > OCR_NATIVE_OVERLAP_SKIP
                   for nb in native_bboxes):
                continue
            raw_ocr.append({"label": text, "bbox": bbox, "score": score})
        if not raw_ocr:
            return []
        return parser.parse(raw_ocr, image) or []
