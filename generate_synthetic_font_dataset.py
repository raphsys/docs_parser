#!/usr/bin/env python3
import argparse
import csv
import os
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


DEFAULT_FONT_DIRS = (
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
)

TEXT_SNIPPETS = [
    "Hamburgefonstiv 012345",
    "The quick brown fox jumps",
    "AaBbCcDdEeFfGgHhIiJj",
    "Vision Systems 2026",
    "OCR Layout Translation",
    "Invoice Total Amount",
    "CONFIDENTIAL REPORT",
    "Neural embedding retrieval",
]


def normalize_font_name(font_path: Path) -> str:
    name = font_path.stem
    name = name.split("+", 1)[-1]
    return re.sub(r"[^a-zA-Z0-9]+", "", name).lower()


def discover_fonts(font_dirs: Tuple[str, ...]) -> List[Path]:
    paths: List[Path] = []
    for root in font_dirs:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                low = fn.lower()
                if low.endswith(".ttf") or low.endswith(".otf"):
                    paths.append(Path(dirpath) / fn)
    return sorted(set(paths))


def should_exclude_font(font_path: Path, font_key: str, exclude_rx: re.Pattern | None) -> bool:
    if exclude_rx is None:
        return False
    hay = f"{font_key} {str(font_path).lower()}"
    return exclude_rx.search(hay) is not None


def random_text(rng: random.Random) -> str:
    base = rng.choice(TEXT_SNIPPETS)
    if rng.random() < 0.35:
        base = base.upper()
    if rng.random() < 0.25:
        base += " " + str(rng.randint(1, 9999))
    return base


def add_noise(img_arr: np.ndarray, rng: random.Random, std: float) -> np.ndarray:
    if std <= 0:
        return img_arr
    noise = rng.normalvariate(0, std)
    noise_map = np.random.normal(loc=noise, scale=std, size=img_arr.shape).astype(np.float32)
    out = np.clip(img_arr.astype(np.float32) + noise_map, 0, 255).astype(np.uint8)
    return out


def apply_hard_ocr_effects(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.array(img, dtype=np.float32)

    # Lower contrast / uneven exposure.
    if rng.random() < 0.9:
        contrast = rng.uniform(0.55, 0.95)
        brightness = rng.uniform(-18.0, 20.0)
        arr = (arr - 127.5) * contrast + 127.5 + brightness

    # Scanner-like horizontal lighting drift.
    if rng.random() < 0.5:
        h, w = arr.shape
        grad = np.linspace(rng.uniform(-20, 20), rng.uniform(-20, 20), w, dtype=np.float32)
        arr = arr + grad.reshape(1, w)

    # JPEG-ish quantization + sensor noise.
    if rng.random() < 0.8:
        q = rng.choice([6, 8, 12, 16])
        arr = np.round(arr / q) * q
    if rng.random() < 0.9:
        std = rng.uniform(2.0, 10.0)
        arr = arr + np.random.normal(0.0, std, size=arr.shape).astype(np.float32)

    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")

    # Downscale-upscale to mimic low DPI captures.
    if rng.random() < 0.6:
        w, h = out.size
        down = rng.uniform(0.45, 0.8)
        nw = max(1, int(round(w * down)))
        nh = max(1, int(round(h * down)))
        out = out.resize((nw, nh), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)

    if rng.random() < 0.5:
        out = out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.4)))
    return out


def render_sample(font_path: Path, out_size: int, rng: random.Random, hard_ocr_augment: bool = False) -> Image.Image:
    canvas_w = rng.randint(260, 520)
    canvas_h = rng.randint(90, 180)
    bg = rng.randint(230, 255)
    fg = rng.randint(0, 40)

    img = Image.new("L", (canvas_w, canvas_h), color=bg)
    draw = ImageDraw.Draw(img)

    font_size = rng.randint(22, 68)
    font = ImageFont.truetype(str(font_path), size=font_size)
    text = random_text(rng)
    tx = rng.randint(8, 28)
    ty = rng.randint(8, max(10, canvas_h // 3))
    draw.text((tx, ty), text, fill=fg, font=font)

    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.2)))

    arr = np.array(img)
    arr = add_noise(arr, rng, std=rng.uniform(0.0, 8.0))
    img = Image.fromarray(arr, mode="L")
    if hard_ocr_augment:
        img = apply_hard_ocr_effects(img, rng)
        arr = np.array(img)

    ys, xs = np.where(arr < 245)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        pad = rng.randint(2, 10)
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(arr.shape[1] - 1, x1 + pad)
        y1 = min(arr.shape[0] - 1, y1 + pad)
        img = img.crop((x0, y0, x1 + 1, y1 + 1))

    w, h = img.size
    scale = min(out_size / max(w, 1), out_size / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    rs = img.resize((nw, nh), Image.Resampling.BILINEAR)
    out = Image.new("L", (out_size, out_size), color=255)
    ox = (out_size - nw) // 2
    oy = (out_size - nh) // 2
    out.paste(rs, (ox, oy))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic font dataset for embedding training.")
    ap.add_argument("--output-dir", default="./datasets/font_ai_synth", help="Dataset output directory")
    ap.add_argument("--samples-per-font", type=int, default=300, help="Images per font")
    ap.add_argument("--max-fonts", type=int, default=1200, help="Max number of local fonts to use")
    ap.add_argument("--image-size", type=int, default=128, help="Square output size")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--font-dir", action="append", default=[], help="Additional font directory (repeatable)")
    ap.add_argument(
        "--exclude-pattern",
        default=r"(emoji|symbol|dingbats|math|music|icons|ancient|jsmath|fallback)",
        help="Regex on font key/path to skip non-OCR-relevant fonts",
    )
    ap.add_argument("--strict-balance", type=int, default=1, help="Require exact samples-per-font for every kept class (1/0)")
    ap.add_argument("--max-attempts-multiplier", type=int, default=6, help="Max render attempts = samples-per-font * multiplier")
    ap.add_argument("--hard-ocr-augment", type=int, default=1, help="Enable harder OCR-like degradations (1/0)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    font_dirs = list(DEFAULT_FONT_DIRS) + args.font_dir
    fonts = discover_fonts(tuple(font_dirs))
    if not fonts:
        raise SystemExit("No fonts found in provided directories.")
    fonts = fonts[: args.max_fonts]
    exclude_rx = re.compile(args.exclude_pattern, flags=re.IGNORECASE) if args.exclude_pattern else None
    strict_balance = int(args.strict_balance) == 1
    hard_ocr_augment = int(args.hard_ocr_augment) == 1

    # Keep only one path per normalized key and filter out non OCR-relevant families.
    selected_by_key: Dict[str, Path] = {}
    filtered_count = 0
    for f in fonts:
        key = normalize_font_name(f)
        if not key:
            filtered_count += 1
            continue
        if should_exclude_font(f, key, exclude_rx):
            filtered_count += 1
            continue
        if key not in selected_by_key:
            selected_by_key[key] = f
    selected_fonts: List[Tuple[str, Path]] = sorted(selected_by_key.items(), key=lambda kv: kv[0])
    if not selected_fonts:
        raise SystemExit("No fonts left after filtering. Adjust --exclude-pattern.")
    print(f"[INFO] Source fonts={len(fonts)} | filtered={filtered_count} | selected_classes={len(selected_fonts)}")

    out_root = Path(args.output_dir)
    img_root = out_root / "images"
    if img_root.exists():
        shutil.rmtree(img_root, ignore_errors=True)
    img_root.mkdir(parents=True, exist_ok=True)
    meta_path = out_root / "metadata.csv"
    classes_path = out_root / "classes.csv"

    metadata_rows = []
    kept_classes: List[Tuple[str, str]] = []
    expected_total = len(selected_fonts) * args.samples_per_font
    written_total = 0
    next_mark = 5
    max_attempts = max(1, args.samples_per_font * max(1, args.max_attempts_multiplier))

    for font_idx, (font_key, font_path) in enumerate(selected_fonts, start=1):
        font_dir = img_root / font_key
        font_dir.mkdir(parents=True, exist_ok=True)
        written_for_font = 0
        attempts = 0
        local_rows = []

        while written_for_font < args.samples_per_font and attempts < max_attempts:
            attempts += 1
            try:
                img = render_sample(
                    font_path,
                    args.image_size,
                    rng,
                    hard_ocr_augment=hard_ocr_augment,
                )
            except Exception:
                continue
            filename = f"{written_for_font:05d}.png"
            rel_path = Path("images") / font_key / filename
            img.save(out_root / rel_path)
            split = "val" if rng.random() < args.val_ratio else "train"
            local_rows.append((str(rel_path), font_key, split))
            written_for_font += 1
            written_total += 1

            pct = int((written_total * 100) / max(1, expected_total))
            if pct >= next_mark:
                print(
                    f"[GEN] progress {pct:3d}% "
                    f"({written_total}/{expected_total} samples)"
                )
                next_mark += 5

        keep_class = (written_for_font == args.samples_per_font) if strict_balance else (written_for_font > 0)
        if keep_class:
            kept_classes.append((font_key, str(font_path)))
            metadata_rows.extend(local_rows)
        else:
            shutil.rmtree(font_dir, ignore_errors=True)

        if font_idx % 10 == 0 or font_idx == len(selected_fonts):
            print(
                f"[GEN] font {font_idx}/{len(selected_fonts)} "
                f"({font_key}) -> {written_for_font} samples (attempts={attempts})"
            )

    class_to_id = {font_key: idx for idx, (font_key, _) in enumerate(kept_classes)}

    with open(classes_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["font_key", "class_id", "font_path"])
        for font_key, font_path in kept_classes:
            w.writerow([font_key, class_to_id[font_key], font_path])

    with open(meta_path, "w", newline="", encoding="utf-8") as mf:
        w = csv.writer(mf)
        w.writerow(["image_path", "font_key", "class_id", "split"])
        for rel_path, font_key, split in metadata_rows:
            w.writerow([rel_path, font_key, class_to_id[font_key], split])

    print(f"[OK] Dataset generated at: {out_root}")
    print(f"[OK] Metadata: {meta_path}")
    print(f"[OK] Classes: {classes_path}")
    print(f"[OK] Kept classes: {len(kept_classes)}")
    print(f"[OK] Total samples: {len(metadata_rows)}")


if __name__ == "__main__":
    main()
