"""SourceLoader — unité 1 du pipeline cible.

Responsabilités :
    - lit PDF / image / Office ;
    - convertit Office en PDF si besoin (LibreOffice headless) ;
    - calcule hash / source metadata.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid

from pageprint.provenance import file_hash

OFFICE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".odt", ".odp"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class SourceLoadError(RuntimeError):
    pass


def _convert_office_to_pdf(input_path: str, output_dir: str) -> str:
    binary = shutil.which("soffice") or shutil.which("libreoffice")
    if not binary:
        raise SourceLoadError("LibreOffice/soffice introuvable pour convertir le document Office.")
    subprocess.run(
        [binary, "--headless", "--convert-to", "pdf", "--outdir", output_dir, input_path],
        check=True, capture_output=True, timeout=180,
    )
    base = os.path.splitext(os.path.basename(input_path))[0]
    pdf_path = os.path.join(output_dir, base + ".pdf")
    if not os.path.isfile(pdf_path):
        raise SourceLoadError(f"Conversion Office→PDF échouée: {pdf_path} absent.")
    return pdf_path


class SourceLoader:
    """Charge la source et produit un source_context canonique."""

    def load(self, source_path: str, *, work_dir: str | None = None) -> dict:
        if not os.path.isfile(source_path):
            raise SourceLoadError(f"Fichier introuvable: {source_path}")

        ext = os.path.splitext(source_path)[1].lower()
        converted_pdf = None
        if ext in OFFICE_EXTENSIONS:
            out_dir = work_dir or tempfile.mkdtemp(prefix="pageprint_office_")
            converted_pdf = _convert_office_to_pdf(source_path, out_dir)
            file_type = "office"
        elif ext == ".pdf":
            file_type = "pdf"
        elif ext in IMAGE_EXTENSIONS:
            file_type = "image"
        else:
            raise SourceLoadError(f"Extension non supportée: {ext}")

        pdf_path = converted_pdf or (source_path if file_type == "pdf" else None)
        page_count = 1
        document = None
        if pdf_path:
            import fitz  # PyMuPDF — chargé seulement si nécessaire

            document = fitz.open(pdf_path)
            page_count = document.page_count

        return {
            "document_id": f"doc_{uuid.uuid4().hex[:12]}",
            "source_path": source_path,
            "file_name": os.path.basename(source_path),
            "file_type": file_type,
            "pdf_path": pdf_path,
            "converted_pdf": converted_pdf,
            "document": document,           # handle PyMuPDF (None pour image pure)
            "page_count": page_count,
            "source_file_hash": file_hash(source_path),
        }
