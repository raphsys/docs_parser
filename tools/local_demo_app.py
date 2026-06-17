#!/usr/bin/env python3
"""Small local Tkinter application for docs_parser demos.

Non-web interface:
- choose a PDF;
- select pages in a window;
- choose a pipeline level;
- run the selected stage;
- write all outputs into results/ and open them locally.
"""

from __future__ import annotations

import os
import random
import shlex
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


STAGES = [
    ("full", "Complet : PAGEPRINT → PAGETRANSLATE → PAGERECONSTRUCT + audits"),
    ("pageprint", "PAGEPRINT seulement : extraction / bboxes / classification"),
    ("pagetranslate", "PAGETRANSLATE : unités de traduction + translated_input_data"),
    ("pagereconstruct", "PAGERECONSTRUCT : plan + overlay + rendu PNG/PDF"),
    ("view_background", "VIEW_BACKGROUND : source vs clean background"),
    ("audit_translation_selection", "Audit sélection de traduction PAGEPRINT"),
    ("audit_text_survival", "Audit survie texte source → rendu"),
]


class LocalDemoApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("docs_parser — Démo locale pipeline")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.proc: subprocess.Popen[str] | None = None
        self.last_out_dir: Path | None = None
        self.page_count = 0

        self.pdf_var = tk.StringVar()
        self.pages_var = tk.StringVar(value="1")
        self.stage_var = tk.StringVar(value="full")
        self.engine_var = tk.StringVar(value="ct2")
        self.model_var = tk.StringVar(value="opus_mt_tc_big_en_fr")
        self.source_lang_var = tk.StringVar(value="en")
        self.target_lang_var = tk.StringVar(value="fr")
        self.pubready_var = tk.StringVar(value="review")
        self.reconstruct_mode_var = tk.StringVar(value="debug")
        self.enable_ocr_var = tk.BooleanVar(value=False)
        self.reuse_tid_var = tk.BooleanVar(value=False)
        self.fail_fast_var = tk.BooleanVar(value=False)
        self.out_name_var = tk.StringVar(value="")

        self._build_ui()
        self._load_default_pdfs()

    def _build_ui(self) -> None:
        root = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        root.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(root, padding=8)
        right = ttk.Frame(root, padding=8)
        root.add(left, weight=0)
        root.add(right, weight=1)

        # --- PDF selection -------------------------------------------------
        pdf_box = ttk.LabelFrame(left, text="1. Document source", padding=8)
        pdf_box.pack(fill=tk.X)
        ttk.Label(pdf_box, text="PDF").pack(anchor=tk.W)
        self.pdf_combo = ttk.Combobox(pdf_box, textvariable=self.pdf_var, width=64)
        self.pdf_combo.pack(fill=tk.X, pady=(2, 4))
        self.pdf_combo.bind("<<ComboboxSelected>>", lambda _e: self.load_pdf_pages())
        row = ttk.Frame(pdf_box)
        row.pack(fill=tk.X)
        ttk.Button(row, text="Parcourir…", command=self.browse_pdf).pack(side=tk.LEFT)
        ttk.Button(row, text="Charger pages", command=self.load_pdf_pages).pack(side=tk.LEFT, padx=6)

        # --- page selection ------------------------------------------------
        pages_box = ttk.LabelFrame(left, text="2. Pages", padding=8)
        pages_box.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        ttk.Label(pages_box, text="Sélection fenêtre (Ctrl/Shift possible)").pack(anchor=tk.W)
        list_frame = ttk.Frame(pages_box)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 6))
        self.page_list = tk.Listbox(list_frame, selectmode=tk.EXTENDED, height=16, exportselection=False)
        yscroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.page_list.yview)
        self.page_list.configure(yscrollcommand=yscroll.set)
        self.page_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.page_list.bind("<<ListboxSelect>>", lambda _e: self.sync_pages_from_selection())

        ttk.Label(pages_box, text="Pages/ranges à lancer").pack(anchor=tk.W)
        ttk.Entry(pages_box, textvariable=self.pages_var).pack(fill=tk.X, pady=(2, 4))
        row = ttk.Frame(pages_box)
        row.pack(fill=tk.X)
        ttk.Button(row, text="Random 5", command=lambda: self.pick_random(5)).pack(side=tk.LEFT)
        ttk.Button(row, text="Tout", command=self.pick_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Vider", command=self.clear_pages).pack(side=tk.LEFT)

        # --- options -------------------------------------------------------
        opts = ttk.LabelFrame(right, text="3. Options pipeline", padding=8)
        opts.pack(fill=tk.X)

        grid = ttk.Frame(opts)
        grid.pack(fill=tk.X)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)

        ttk.Label(grid, text="Niveau").grid(row=0, column=0, sticky=tk.W, padx=(0, 6), pady=3)
        stage_combo = ttk.Combobox(grid, textvariable=self.stage_var, values=[s[0] for s in STAGES], state="readonly")
        stage_combo.grid(row=0, column=1, sticky=tk.EW, pady=3)
        ttk.Label(grid, text="Moteur").grid(row=0, column=2, sticky=tk.W, padx=(12, 6), pady=3)
        ttk.Combobox(grid, textvariable=self.engine_var, values=["ct2", "mock", "prefix", "rule", "local", "external"], width=16).grid(row=0, column=3, sticky=tk.EW, pady=3)

        ttk.Label(grid, text="Modèle").grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=3)
        ttk.Entry(grid, textvariable=self.model_var).grid(row=1, column=1, sticky=tk.EW, pady=3)
        ttk.Label(grid, text="Pubready").grid(row=1, column=2, sticky=tk.W, padx=(12, 6), pady=3)
        ttk.Combobox(grid, textvariable=self.pubready_var, values=["debug", "review", "publication"], state="readonly", width=16).grid(row=1, column=3, sticky=tk.EW, pady=3)

        ttk.Label(grid, text="Langues").grid(row=2, column=0, sticky=tk.W, padx=(0, 6), pady=3)
        lang_row = ttk.Frame(grid)
        lang_row.grid(row=2, column=1, sticky=tk.EW, pady=3)
        ttk.Entry(lang_row, textvariable=self.source_lang_var, width=8).pack(side=tk.LEFT)
        ttk.Label(lang_row, text=" → ").pack(side=tk.LEFT)
        ttk.Entry(lang_row, textvariable=self.target_lang_var, width=8).pack(side=tk.LEFT)
        ttk.Label(grid, text="Mode reconstruction").grid(row=2, column=2, sticky=tk.W, padx=(12, 6), pady=3)
        ttk.Entry(grid, textvariable=self.reconstruct_mode_var, width=16).grid(row=2, column=3, sticky=tk.EW, pady=3)

        ttk.Label(grid, text="Nom dossier results/").grid(row=3, column=0, sticky=tk.W, padx=(0, 6), pady=3)
        ttk.Entry(grid, textvariable=self.out_name_var).grid(row=3, column=1, sticky=tk.EW, pady=3)
        flags = ttk.Frame(grid)
        flags.grid(row=3, column=2, columnspan=2, sticky=tk.W, padx=(12, 0), pady=3)
        ttk.Checkbutton(flags, text="OCR", variable=self.enable_ocr_var).pack(side=tk.LEFT)
        ttk.Checkbutton(flags, text="Reuse TID", variable=self.reuse_tid_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(flags, text="Fail fast", variable=self.fail_fast_var).pack(side=tk.LEFT)

        desc = ttk.Label(opts, text=self.stage_description(), foreground="#555")
        desc.pack(anchor=tk.W, pady=(6, 0))
        self.stage_desc = desc
        stage_combo.bind("<<ComboboxSelected>>", lambda _e: self.stage_desc.configure(text=self.stage_description()))

        # --- actions -------------------------------------------------------
        actions = ttk.Frame(right)
        actions.pack(fill=tk.X, pady=(8, 8))
        ttk.Button(actions, text="Lancer", command=self.run_pipeline).pack(side=tk.LEFT)
        ttk.Button(actions, text="Stop", command=self.stop_process).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Ouvrir results", command=self.open_results_root).pack(side=tk.LEFT)
        ttk.Button(actions, text="Ouvrir dernier dossier", command=self.open_last_results).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Ouvrir contact_sheet", command=self.open_contact_sheet).pack(side=tk.LEFT)

        # --- log -----------------------------------------------------------
        log_box = ttk.LabelFrame(right, text="Journal", padding=6)
        log_box.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_box, wrap=tk.WORD, height=24)
        log_scroll = ttk.Scrollbar(log_box, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def stage_description(self) -> str:
        stage = self.stage_var.get()
        for key, desc in STAGES:
            if key == stage:
                return desc
        return ""

    def log(self, text: str) -> None:
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self.update_idletasks()

    def _load_default_pdfs(self) -> None:
        pdfs = []
        default_dir = ROOT / "tests" / "doc_pdf"
        if default_dir.is_dir():
            pdfs.extend(str(p) for p in sorted(default_dir.glob("*.pdf")))
        self.pdf_combo.configure(values=pdfs)
        if pdfs and not self.pdf_var.get():
            self.pdf_var.set(pdfs[0])
            self.load_pdf_pages()

    def browse_pdf(self) -> None:
        path = filedialog.askopenfilename(
            title="Choisir un PDF",
            initialdir=str(ROOT / "tests" / "doc_pdf") if (ROOT / "tests" / "doc_pdf").is_dir() else str(ROOT),
            filetypes=[("PDF", "*.pdf"), ("Tous les fichiers", "*.*")],
        )
        if path:
            self.pdf_var.set(path)
            self.load_pdf_pages()

    def load_pdf_pages(self) -> None:
        pdf = Path(self.pdf_var.get()).expanduser()
        self.page_list.delete(0, tk.END)
        self.page_count = 0
        if not pdf.is_file():
            return
        if fitz is None:
            messagebox.showerror("PyMuPDF manquant", "Le module fitz/PyMuPDF est nécessaire pour compter les pages.")
            return
        try:
            with fitz.open(pdf) as doc:
                self.page_count = int(doc.page_count)
        except Exception as exc:
            messagebox.showerror("PDF illisible", str(exc))
            return
        for i in range(1, self.page_count + 1):
            self.page_list.insert(tk.END, f"Page {i}")
        if self.page_count:
            self.page_list.selection_set(0)
            self.pages_var.set("1")
        self.log(f"PDF chargé: {pdf} ({self.page_count} pages)\n")

    def sync_pages_from_selection(self) -> None:
        selected = [i + 1 for i in self.page_list.curselection()]
        if selected:
            self.pages_var.set(self.compress_pages(selected))

    @staticmethod
    def compress_pages(pages: list[int]) -> str:
        if not pages:
            return ""
        pages = sorted(set(pages))
        ranges = []
        start = prev = pages[0]
        for p in pages[1:]:
            if p == prev + 1:
                prev = p
            else:
                ranges.append(f"{start}-{prev}" if start != prev else str(start))
                start = prev = p
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        return ",".join(ranges)

    def pick_random(self, count: int) -> None:
        if self.page_count <= 0:
            self.load_pdf_pages()
        if self.page_count <= 0:
            return
        picks = sorted(random.sample(range(1, self.page_count + 1), min(count, self.page_count)))
        self.page_list.selection_clear(0, tk.END)
        for p in picks:
            self.page_list.selection_set(p - 1)
        self.pages_var.set(self.compress_pages(picks))

    def pick_all(self) -> None:
        if self.page_count <= 0:
            self.load_pdf_pages()
        if self.page_count <= 0:
            return
        self.page_list.selection_set(0, tk.END)
        self.pages_var.set(f"1-{self.page_count}")

    def clear_pages(self) -> None:
        self.page_list.selection_clear(0, tk.END)
        self.pages_var.set("")

    def build_command(self) -> tuple[list[str], Path]:
        pdf = Path(self.pdf_var.get()).expanduser()
        if not pdf.is_file():
            raise ValueError("Choisis un PDF valide.")
        pages = self.pages_var.get().strip()
        if not pages:
            raise ValueError("Choisis au moins une page.")
        out_name = self.out_name_var.get().strip()
        if not out_name:
            out_name = f"local_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir = ROOT / "results" / out_name
        cmd = [
            sys.executable, str(ROOT / "tools" / "local_demo_runner.py"),
            "--pdf", str(pdf),
            "--pages", pages,
            "--stage", self.stage_var.get(),
            "--out", str(out_dir),
            "--engine", self.engine_var.get().strip() or "ct2",
            "--model", self.model_var.get().strip() or "opus_mt_tc_big_en_fr",
            "--source-lang", self.source_lang_var.get().strip() or "en",
            "--target-lang", self.target_lang_var.get().strip() or "fr",
            "--pubready-mode", self.pubready_var.get(),
            "--reconstruction-mode", self.reconstruct_mode_var.get().strip() or "debug",
        ]
        if self.enable_ocr_var.get():
            cmd.append("--enable-ocr")
        if self.reuse_tid_var.get():
            cmd.append("--reuse-tid")
        if self.fail_fast_var.get():
            cmd.append("--fail-fast")
        return cmd, out_dir

    def run_pipeline(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Déjà en cours", "Un traitement est déjà en cours.")
            return
        try:
            cmd, out_dir = self.build_command()
        except Exception as exc:
            messagebox.showerror("Options invalides", str(exc))
            return
        self.last_out_dir = out_dir
        self.log("\n=== Lancement ===\n")
        self.log(" ".join(shlex.quote(c) for c in cmd) + "\n\n")
        t = threading.Thread(target=self._run_subprocess, args=(cmd,), daemon=True)
        t.start()

    def _run_subprocess(self, cmd: list[str]) -> None:
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.after(0, self.log, line)
            code = self.proc.wait()
            self.after(0, self.log, f"\n=== Fin: code {code} ===\n")
            self.after(0, lambda: self.open_contact_sheet(silent=True))
        except Exception as exc:
            self.after(0, self.log, f"\nERREUR PROCESS: {exc}\n")
        finally:
            self.proc = None

    def stop_process(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            if os.name == "posix":
                self.proc.send_signal(signal.SIGINT)
            else:
                self.proc.terminate()
            self.log("\nSignal d'arrêt envoyé.\n")
        except Exception as exc:
            self.log(f"\nImpossible d'arrêter: {exc}\n")

    def _open_path(self, path: Path) -> None:
        if not path.exists():
            messagebox.showwarning("Introuvable", str(path))
            return
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("Ouverture impossible", str(exc))

    def open_results_root(self) -> None:
        path = ROOT / "results"
        path.mkdir(exist_ok=True)
        self._open_path(path)

    def open_last_results(self) -> None:
        if self.last_out_dir:
            self._open_path(self.last_out_dir)
        else:
            self.open_results_root()

    def open_contact_sheet(self, silent: bool = False) -> None:
        if not self.last_out_dir:
            if not silent:
                messagebox.showinfo("Aucun résultat", "Aucun dossier de résultat encore lancé.")
            return
        path = self.last_out_dir / "contact_sheet.jpg"
        if path.is_file():
            self._open_path(path)
        elif not silent:
            messagebox.showwarning("Absent", f"Pas de contact_sheet.jpg dans {self.last_out_dir}")


def main() -> int:
    try:
        app = LocalDemoApp()
        app.mainloop()
        return 0
    except tk.TclError as exc:
        print(f"Erreur Tkinter: {exc}", file=sys.stderr)
        print("Sur Debian, installe au besoin: sudo apt install python3-tk", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
