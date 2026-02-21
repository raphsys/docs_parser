#!/usr/bin/env python3
import argparse
import os
import tarfile
import zipfile
import urllib.request
import urllib.error
import socket
import time
from pathlib import Path


SOURCES = {
    # Existing public font collection (font files, ideal for synthetic generation)
    "google_fonts_repo": "https://github.com/google/fonts/archive/refs/heads/main.zip",
    # Existing public collection of packaged font files
    "fontsource_files_repo": "https://github.com/fontsource/font-files/archive/refs/heads/main.zip",
    # Existing OCR/font-style benchmark assets (rendered glyphs)
    "chars74k_english_fnt": "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz",
}


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[DL] {url}")
    part_path = Path(str(out_path) + ".part")
    resume_from = part_path.stat().st_size if part_path.exists() else 0

    headers = {"User-Agent": "Mozilla/5.0"}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
        print(f"[DL] resume requested from byte {resume_from}")

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        status = getattr(resp, "status", None) or resp.getcode()
        accept_ranges = (resp.headers.get("Accept-Ranges", "") or "").lower()
        content_range = resp.headers.get("Content-Range", "")

        if resume_from > 0:
            if status == 206 or "bytes" in accept_ranges or content_range:
                mode = "ab"
                print("[DL] resume accepted by server")
            else:
                mode = "wb"
                resume_from = 0
                print("[DL] server does not support resume; restarting from 0")
        else:
            mode = "wb"

        with open(part_path, mode) as f:
            total_size_hdr = int(resp.headers.get("Content-Length", "0") or 0)
            total_size = total_size_hdr + resume_from if total_size_hdr > 0 else 0

            if total_size > 0:
                print(f"[DL] total size: {total_size / (1024 * 1024):.1f} MB")
            else:
                print("[DL] total size: unknown (no Content-Length)")

            downloaded = resume_from
            chunk_size = 1024 * 1024  # 1MB
            next_pct = int((downloaded * 100) / total_size) + 5 if total_size > 0 else 5
            next_mb_mark = int(downloaded / (1024 * 1024)) + 5
            start_ts = time.monotonic()
            last_heartbeat_ts = start_ts

            while True:
                try:
                    chunk = resp.read(chunk_size)
                except (TimeoutError, socket.timeout):
                    now = time.monotonic()
                    if now - last_heartbeat_ts >= 10:
                        mb_done = downloaded / (1024 * 1024)
                        print(f"[DL] heartbeat: still downloading... {mb_done:.1f} MB received")
                        last_heartbeat_ts = now
                    continue

                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()

                if total_size > 0:
                    pct = int((downloaded * 100) / total_size)
                    if pct >= next_pct:
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"[DL] progress {min(pct, 100):3d}% ({mb_done:.1f}/{mb_total:.1f} MB)")
                        while next_pct <= pct:
                            next_pct += 5
                        last_heartbeat_ts = now
                else:
                    mb_done = downloaded / (1024 * 1024)
                    if mb_done >= next_mb_mark:
                        print(f"[DL] progress {mb_done:.1f} MB downloaded")
                        next_mb_mark += 5
                        last_heartbeat_ts = now

                if now - last_heartbeat_ts >= 10:
                    mb_done = downloaded / (1024 * 1024)
                    if total_size > 0:
                        mb_total = total_size / (1024 * 1024)
                        print(f"[DL] heartbeat: {mb_done:.1f}/{mb_total:.1f} MB")
                    else:
                        print(f"[DL] heartbeat: {mb_done:.1f} MB downloaded")
                    last_heartbeat_ts = now

            if total_size > 0 and downloaded < total_size:
                print(f"[DL] warning: expected {total_size} bytes, got {downloaded} bytes")
            elapsed = time.monotonic() - start_ts
            print(f"[DL] done in {elapsed:.1f}s")

    os.replace(part_path, out_path)
    print(f"[OK] Downloaded -> {out_path}")


def extract(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()
    print(f"[EXTRACT] {archive_path} -> {dest_dir}")
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.infolist()
            total = max(1, len(members))
            next_mark = 10
            for i, m in enumerate(members, start=1):
                zf.extract(m, dest_dir)
                pct = int((i * 100) / total)
                if pct >= next_mark or i == total:
                    print(f"[EXTRACT] progress {pct:3d}% ({i}/{total} files)")
                    next_mark += 10
    elif name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".tar"):
        with tarfile.open(archive_path, "r:*") as tf:
            members = tf.getmembers()
            total = max(1, len(members))
            next_mark = 10
            for i, m in enumerate(members, start=1):
                tf.extract(m, dest_dir)
                pct = int((i * 100) / total)
                if pct >= next_mark or i == total:
                    print(f"[EXTRACT] progress {pct:3d}% ({i}/{total} files)")
                    next_mark += 10
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")
    print("[OK] Extracted")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download existing font datasets/collections.")
    ap.add_argument("--list", action="store_true", help="List available sources and exit")
    ap.add_argument("--source", choices=sorted(SOURCES.keys()), nargs="+", help="Sources to download")
    ap.add_argument("--output-dir", default="./datasets/font_sources", help="Where to store archives/extracted data")
    ap.add_argument("--extract", action="store_true", help="Extract archives after download")
    ap.add_argument("--keep-archive", action="store_true", help="Keep archive files after extraction")
    args = ap.parse_args()

    if args.list:
        for k, v in SOURCES.items():
            print(f"{k}: {v}")
        return

    if not args.source:
        raise SystemExit("Provide at least one --source or use --list.")

    out_root = Path(args.output_dir)
    archives_dir = out_root / "archives"
    extracted_dir = out_root / "extracted"

    for source_key in args.source:
        url = SOURCES[source_key]
        filename = url.rstrip("/").split("/")[-1]
        archive_path = archives_dir / filename
        source_extract_dir = extracted_dir / source_key

        try:
            if not archive_path.exists():
                download(url, archive_path)
            else:
                print(f"[SKIP] Archive already exists: {archive_path}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"[ERR] Download failed for {source_key}: {exc}")
            continue

        if args.extract:
            try:
                extract(archive_path, source_extract_dir)
                if not args.keep_archive:
                    archive_path.unlink(missing_ok=True)
                    print(f"[CLEAN] Removed archive: {archive_path}")
            except Exception as exc:
                print(f"[ERR] Extraction failed for {source_key}: {exc}")


if __name__ == "__main__":
    main()
