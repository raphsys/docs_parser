#!/usr/bin/env python3
import argparse
from itertools import combinations
import fitz


def overlap_ratio(a, b):
    r1 = fitz.Rect(a)
    r2 = fitz.Rect(b)
    inter = (r1 & r2).get_area()
    if inter <= 0:
        return 0.0
    return inter / max(1e-9, min(r1.get_area(), r2.get_area()))


def main():
    ap = argparse.ArgumentParser(description="Evaluate reconstructed PDF layout quality")
    ap.add_argument("pdf", help="Path to reconstructed PDF")
    ap.add_argument("--overlap-threshold", type=float, default=0.25)
    ap.add_argument("--text-image-threshold", type=float, default=0.10)
    ap.add_argument("--keep-fullpage-images", action="store_true", default=False)
    ap.add_argument("--fullpage-area-ratio", type=float, default=0.95)
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    total_words = 0
    total_off_page = 0
    total_word_overlap = 0
    total_text_img_coll = 0

    for pi, page in enumerate(doc, start=1):
        words = page.get_text("words")
        wrects = [fitz.Rect(w[:4]) for w in words if str(w[4]).strip()]
        total_words += len(wrects)

        off_page = 0
        for r in wrects:
            if r.x0 < page.rect.x0 or r.y0 < page.rect.y0 or r.x1 > page.rect.x1 or r.y1 > page.rect.y1:
                off_page += 1
        total_off_page += off_page

        ov = 0
        for i, j in combinations(range(len(wrects)), 2):
            if overlap_ratio(wrects[i], wrects[j]) > args.overlap_threshold:
                ov += 1
        total_word_overlap += ov

        d = page.get_text("dict")
        txt = [fitz.Rect(b["bbox"]) for b in d["blocks"] if b.get("type", 0) == 0 and "bbox" in b]
        imgs = [fitz.Rect(b["bbox"]) for b in d["blocks"] if b.get("type", 0) == 1 and "bbox" in b]
        if not args.keep_fullpage_images:
            page_area = max(1e-9, page.rect.get_area())
            kept_imgs = []
            for im in imgs:
                ratio = im.get_area() / page_area
                if ratio >= args.fullpage_area_ratio:
                    continue
                kept_imgs.append(im)
            imgs = kept_imgs
        coll = 0
        for t in txt:
            if any(overlap_ratio(t, im) > args.text_image_threshold for im in imgs):
                coll += 1
        total_text_img_coll += coll

        print(
            f"page={pi} words={len(wrects)} off_page_words={off_page} "
            f"word_overlaps={ov} text_img_collisions={coll}"
        )

    print("---")
    print(f"total_words={total_words}")
    print(f"off_page_words={total_off_page}")
    print(f"word_overlaps={total_word_overlap}")
    print(f"text_img_collisions={total_text_img_coll}")


if __name__ == "__main__":
    main()
