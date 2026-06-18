#!/usr/bin/env python3
"""Audit selective clean background purity for a demo_studio run.

Contract:
    background may keep non-text visuals;
    background must remove text and formula/code/math content.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from PIL import Image, ImageChops, ImageStat
except Exception:  # pragma: no cover
    Image = None
    ImageChops = None
    ImageStat = None

from pipelines.background_cover import collect_background_purity_boxes


def _page_key(path: Path) -> str:
    name = path.name
    for prefix in ("translated_input_data_", "pageprint_full_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name.endswith(".json"):
        name = name[:-5]
    return name


def _scale(data: dict) -> tuple[float, float]:
    g = ((data.get("page") or {}).get("geometry") or {})
    sx = float(g.get("scale_x_px_per_pt") or 0) or 0.0
    sy = float(g.get("scale_y_px_per_pt") or 0) or 0.0
    if not sx and g.get("render_width_px") and g.get("width"):
        sx = float(g["render_width_px"]) / max(1e-6, float(g["width"]))
    if not sy and g.get("render_height_px") and g.get("height"):
        sy = float(g["render_height_px"]) / max(1e-6, float(g["height"]))
    return sx or 1.0, sy or 1.0


def _find_image(run_dir: Path, prefix: str, key: str) -> Path | None:
    direct = run_dir / f"{prefix}_{key}.png"
    if direct.exists():
        return direct
    key2 = key.replace(" ", "_")
    for m in sorted(run_dir.glob(f"{prefix}_*.png")):
        stem = m.stem
        if key in stem or key2 in stem.replace(" ", "_"):
            return m
    return None


def _mean_absdiff(src: Image.Image, bg: Image.Image, rect: tuple[int, int, int, int]) -> float:
    crop_a = src.crop(rect).convert("RGB")
    crop_b = bg.crop(rect).convert("RGB")
    diff = ImageChops.difference(crop_a, crop_b)
    stat = ImageStat.Stat(diff)
    return float(sum(stat.mean) / 3.0)


def _visual_variance_ratio(src: Image.Image, bg: Image.Image, rect: tuple[int, int, int, int]) -> float:
    src_stat = ImageStat.Stat(src.crop(rect).convert("L"))
    bg_stat = ImageStat.Stat(bg.crop(rect).convert("L"))
    return float(bg_stat.stddev[0]) / max(1.0, float(src_stat.stddev[0]))


def audit_file(path: Path, run_dir: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    key = _page_key(path)
    boxes = collect_background_purity_boxes(data)
    vl = data.get("visual_layers") or {}
    assets = data.get("assets") or {}

    clean_path = (
        vl.get("clean_background_path")
        or assets.get("background_clean_path")
        or str(_find_image(run_dir, "cleanbg", key) or "")
    )
    source_path = (
        assets.get("source_image_path")
        or str(_find_image(run_dir, "source", key) or "")
    )

    clean_exists = bool(clean_path and Path(clean_path).exists())
    source_exists = bool(source_path and Path(source_path).exists())
    strategy = str(vl.get("background_strategy") or assets.get("background_strategy") or "").lower()
    contract = vl.get("background_purity_contract") or {}

    blockers = []
    if boxes and not clean_exists:
        blockers.append("cleanbg_missing")
    if clean_exists and "text_special_purity" not in strategy and not contract.get("no_source_text"):
        blockers.append("background_not_selective_text_special_strategy")

    geom = ((data.get("page") or {}).get("geometry") or {})
    page_area = float(geom.get("width") or 0) * float(geom.get("height") or 0)
    oversized = [b for b in boxes if page_area and ((b[2] - b[0]) * (b[3] - b[1])) / page_area >= 0.85]
    if oversized:
        blockers.append("background_erase_box_covers_page")

    allowed_visuals = []
    raster_visuals = []
    for unit in data.get("units") or []:
        if not isinstance(unit, dict):
            continue
        level = str(unit.get("level") or "").lower()
        role = str(((unit.get("understanding") or {}).get("role") or "")).lower()
        if level not in {"image", "drawing"} and role not in {"image", "figure", "diagram", "chart"}:
            continue
        bbox = (unit.get("geometry") or {}).get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            allowed_visuals.append([float(v) for v in bbox])
            if level == "image":
                raster_visuals.append([float(v) for v in bbox])

    leak_samples = []
    sampled = 0
    suspicious = 0
    destroyed_visuals = 0
    if Image is not None and clean_exists and source_exists:
        try:
            src = Image.open(source_path).convert("RGB")
            bg = Image.open(clean_path).convert("RGB")
            sx, sy = _scale(data)
            w, h = bg.size
            sample_boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:64]
            for b in sample_boxes:
                x0 = max(0, int(round(b[0] * sx)) - 4)
                y0 = max(0, int(round(b[1] * sy)) - 2)
                x1 = min(w, int(round(b[2] * sx)) + 4)
                y1 = min(h, int(round(b[3] * sy)) + 2)
                if x1 <= x0 or y1 <= y0:
                    continue
                sampled += 1
                mad = _mean_absdiff(src, bg, (x0, y0, x1, y1))
                if mad < 2.0:
                    suspicious += 1
                    if len(leak_samples) < 10:
                        leak_samples.append({"bbox": b, "mean_absdiff": round(mad, 3)})
            # Pixel-variance preservation is reliable for raster images. Vector
            # drawings may be sparse lines whose variance legitimately drops
            # when overlaid text is removed; those are counted but not judged by
            # this raster metric.
            for b in raster_visuals[:32]:
                x0 = max(0, int(round(b[0] * sx)))
                y0 = max(0, int(round(b[1] * sy)))
                x1 = min(w, int(round(b[2] * sx)))
                y1 = min(h, int(round(b[3] * sy)))
                if x1 <= x0 or y1 <= y0:
                    continue
                if _visual_variance_ratio(src, bg, (x0, y0, x1, y1)) < 0.12:
                    destroyed_visuals += 1
        except Exception as exc:
            leak_samples.append({"pixel_check_error": str(exc)})

    if sampled >= 8 and suspicious / max(1, sampled) > 0.25:
        blockers.append("background_text_or_special_leak_suspected")
    if destroyed_visuals:
        blockers.append("non_text_visual_destroyed")

    return {
        "page_key": key,
        "status": "ko" if blockers else "ok",
        "hard_blockers": sorted(set(blockers)),
        "expected_text_special_cover_box_count": len(boxes),
        "clean_background_exists": clean_exists,
        "source_image_exists": source_exists,
        "background_strategy": strategy,
        "pixel_sample_count": sampled,
        "pixel_suspicious_count": suspicious,
        "text_leak_count": suspicious,
        "formula_code_leak_count": suspicious,
        "non_text_visual_allowed_count": len(allowed_visuals),
        "non_text_visual_destroyed_count": destroyed_visuals,
        "leak_samples": leak_samples,
        "input_file": str(path),
        "clean_background": clean_path,
        "source_image": source_path,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: audit_background_purity.py results/<demo_studio_run>", file=sys.stderr)
        return 2
    run_dir = Path(argv[1])
    files = sorted(run_dir.glob("translated_input_data_*.json")) or sorted(run_dir.glob("pageprint_full_*.json"))
    reports = [audit_file(p, run_dir) for p in files]
    blockers = sorted({b for r in reports for b in r.get("hard_blockers") or []})
    out = {
        "schema_version": "background_text_special_purity_audit.v1_1",
        "status": "ko" if blockers else "ok",
        "hard_blockers": blockers,
        "report_count": len(reports),
        "ko_report_count": sum(1 for r in reports if r["status"] == "ko"),
        "reports": reports,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
